// Port of src/tree/updater_quantile_hist.cc.
using XGBoost.Collective;

namespace XGBoost.Tree;

/// <summary>Operations the generic <c>UpdateTree</c> driver needs from a hist builder.</summary>
internal interface IHistTreeBuilder<TEntry> where TEntry : class, IExpandEntry
{
    void InitData(DMatrix fmat, RegTree tree, TensorView<GradientPair> gpair);
    TEntry InitRoot(DMatrix fmat, TensorView<GradientPair> gpair, RegTree tree);
    void ApplyTreeSplit(TEntry candidate, RegTree tree);
    void UpdatePosition(DMatrix fmat, RegTree tree, List<TEntry> applied);
    void BuildHistogram(DMatrix fmat, RegTree tree, List<TEntry> validCandidates, TensorView<GradientPair> gpair);
    void EvaluateSplits(DMatrix fmat, RegTree tree, List<TEntry> bestSplits);
    void LeafPartition(RegTree tree, TensorView<GradientPair> gpair, int[] outPosition);
    TEntry MakeEntry(int nidx, int depth);
}

internal static class HistUpdate
{
    public static BatchParam HistBatch(TrainParam param) => new(param.MaxBin, param.SparseThreshold);

    public static void UpdateTree<TEntry>(TensorView<GradientPair> gpair, IHistTreeBuilder<TEntry> updater, DMatrix fmat, TrainParam param,
        HostDeviceVector<int> outPosition, RegTree tree) where TEntry : class, IExpandEntry
    {
        updater.InitData(fmat, tree, gpair);
        var driver = new Driver<TEntry>(param);
        driver.Push(updater.InitRoot(fmat, gpair, tree));
        var expandSet = driver.Pop();
        while (expandSet.Count != 0)
        {
            var validCandidates = new List<TEntry>();
            var applied = new List<TEntry>();
            foreach (var candidate in expandSet)
            {
                updater.ApplyTreeSplit(candidate, tree);
                Check.Gt(tree.LeftChild(candidate.Nid), candidate.Nid);
                applied.Add(candidate);
                if (driver.IsChildValid(candidate)) validCandidates.Add(candidate);
            }
            updater.UpdatePosition(fmat, tree, applied);
            var bestSplits = new List<TEntry>();
            if (validCandidates.Count != 0)
            {
                updater.BuildHistogram(fmat, tree, validCandidates, gpair);
                foreach (var candidate in validCandidates)
                {
                    var childDepth = candidate.Depth + 1;
                    bestSplits.Add(updater.MakeEntry(tree.LeftChild(candidate.Nid), childDepth));
                    bestSplits.Add(updater.MakeEntry(tree.RightChild(candidate.Nid), childDepth));
                }
                updater.EvaluateSplits(fmat, tree, bestSplits);
            }
            driver.Push(bestSplits);
            expandSet = driver.Pop();
        }
        var position = new int[gpair.Shape(0)];
        updater.LeafPartition(tree, gpair, position);
        outPosition.Assign(position, true);
    }

    public static List<CommonRowPartitioner> ResetPartitioners(Context ctx, DMatrix fmat, TrainParam param, List<CommonRowPartitioner> partitioner,
        out int nTotalBins)
    {
        nTotalBins = 0;
        var pageIdx = 0;
        foreach (var page in fmat.GetGradientIndex(ctx, HistBatch(param)))
        {
            if (nTotalBins == 0) nTotalBins = page.Cut.TotalBins;
            else Check.Eq(nTotalBins, page.Cut.TotalBins);
            if (pageIdx < partitioner.Count) partitioner[pageIdx].Reset(ctx, page.Size, page.BaseRowId);
            else partitioner.Add(new CommonRowPartitioner(ctx, page.Size, page.BaseRowId));
            pageIdx++;
        }
        if (partitioner.Count > pageIdx) partitioner.RemoveRange(pageIdx, partitioner.Count - pageIdx);
        return partitioner;
    }
}

/// <summary><c>HistUpdater</c>: scalar-leaf trees.</summary>
internal sealed class HistUpdater(Context ctx, ColumnSampler columnSampler, TrainParam param, HistMakerTrainParam histParam)
    : IHistTreeBuilder<CPUExpandEntry>
{
    private HistEvaluator _evaluator = null!;
    private readonly List<CommonRowPartitioner> _partitioner = [];
    private readonly MultiHistogramBuilder _histogramBuilder = new();

    public CPUExpandEntry MakeEntry(int nidx, int depth) => new(nidx, depth);

    public void InitData(DMatrix fmat, RegTree tree, TensorView<GradientPair> gpair)
    {
        HistUpdate.ResetPartitioners(ctx, fmat, param, _partitioner, out var nTotalBins);
        _histogramBuilder.Reset(ctx, nTotalBins, 1, histParam);
        _evaluator = new HistEvaluator(ctx, param, fmat.Info, columnSampler);
    }

    public void EvaluateSplits(DMatrix fmat, RegTree tree, List<CPUExpandEntry> bestSplits)
    {
        var histograms = _histogramBuilder.Histogram(0);
        var ft = fmat.Info.FeatureTypes.ConstHostSpan;
        foreach (var gmat in fmat.GetGradientIndex(ctx, HistUpdate.HistBatch(param)))
        {
            _evaluator.EvaluateSplits(histograms, gmat.Cut, ft, bestSplits);
            break;
        }
    }

    public void ApplyTreeSplit(CPUExpandEntry candidate, RegTree tree) => _evaluator.ApplyTreeSplit(candidate, tree);

    public CPUExpandEntry InitRoot(DMatrix fmat, TensorView<GradientPair> gpair, RegTree tree)
    {
        var node = new CPUExpandEntry(RegTree.Root, tree.GetDepth(0));
        _histogramBuilder.BuildRootHist(fmat, tree.View(), _partitioner, gpair, node.Nid, HistUpdate.HistBatch(param));
        var gradStat = new GradStats();
        if (fmat.IsDenseMatrix && !Communicator.IsDistributed())
        {
            // For dense data the sum of the first feature's histogram is the node sum.
            GHistIndexMatrix? gmat = null;
            foreach (var p in fmat.GetGradientIndex(ctx, HistUpdate.HistBatch(param)))
            {
                gmat = p;
                break;
            }
            var rowPtr = gmat!.Cut.Ptrs.ConstHostSpan;
            Check.Ge(rowPtr.Length, 2);
            var ibegin = rowPtr[0];
            var iend = rowPtr[1];
            var hist = _histogramBuilder.Histogram(0)[RegTree.Root];
            for (var i = ibegin; i < iend; ++i) gradStat.Add(hist[i].Grad, hist[i].Hess);
        }
        else
        {
            for (long i = 0; i < gpair.Shape(0); ++i)
            {
                var g = gpair[i, 0];
                gradStat.Add(g.Grad, g.Hess);
            }
            Span<double> buf = [gradStat.SumGrad, gradStat.SumHess];
            Communicator.Allreduce(buf, Op.Sum);
            gradStat = new GradStats(buf[0], buf[1]);
        }
        var weight = _evaluator.InitRoot(gradStat);
        tree.Stat(RegTree.Root).SumHess = (float)gradStat.GetHess();
        tree.Stat(RegTree.Root).BaseWeight = weight;
        tree[RegTree.Root].SetLeaf(param.LearningRate * weight);
        var entries = new List<CPUExpandEntry> { node };
        var ft = fmat.Info.FeatureTypes.ConstHostSpan;
        foreach (var gmat in fmat.GetGradientIndex(ctx, HistUpdate.HistBatch(param)))
        {
            _evaluator.EvaluateSplits(_histogramBuilder.Histogram(0), gmat.Cut, ft, entries);
            break;
        }
        return entries[0];
    }

    public void BuildHistogram(DMatrix fmat, RegTree tree, List<CPUExpandEntry> validCandidates, TensorView<GradientPair> gpair)
    {
        var view = tree.View();
        var nodesToBuild = new List<int>();
        var nodesToSub = new List<int>();
        MultiHistogramBuilder.AssignNodes(view, validCandidates, nodesToBuild, nodesToSub);
        _histogramBuilder.BuildHistLeftRight(fmat, view, _partitioner, nodesToBuild, nodesToSub, gpair, HistUpdate.HistBatch(param));
    }

    public void UpdatePosition(DMatrix fmat, RegTree tree, List<CPUExpandEntry> applied)
    {
        var pageId = 0;
        var view = tree.View();
        var nodes = applied.Select(a => a.Nid).ToList();
        var values = applied.Select(a => a.Split.SplitValue).ToList();
        foreach (var page in fmat.GetGradientIndex(ctx, HistUpdate.HistBatch(param)))
        {
            _partitioner[pageId].UpdatePosition(ctx, page, nodes, values, view);
            pageId++;
        }
    }

    public void LeafPartition(RegTree tree, TensorView<GradientPair> gpair, int[] outPosition)
    {
        var view = tree.View();
        foreach (var part in _partitioner) part.LeafPartition(ctx, view, gpair, outPosition);
    }
}

/// <summary><c>MultiTargetHistBuilder</c>: vector-leaf trees.</summary>
internal sealed class MultiTargetHistBuilder(Context ctx, TrainParam param, HistMakerTrainParam histParam, ColumnSampler columnSampler)
    : IHistTreeBuilder<MultiExpandEntry>
{
    private HistMultiEvaluator _evaluator = null!;
    private readonly MultiHistogramBuilder _histogramBuilder = new();
    private readonly List<CommonRowPartitioner> _partitioner = [];

    public MultiExpandEntry MakeEntry(int nidx, int depth) => new(nidx, depth);

    public void UpdatePosition(DMatrix fmat, RegTree tree, List<MultiExpandEntry> applied)
    {
        var pageId = 0;
        var view = tree.View();
        var nodes = applied.Select(a => a.Nid).ToList();
        var values = applied.Select(a => a.Split.SplitValue).ToList();
        foreach (var page in fmat.GetGradientIndex(ctx, HistUpdate.HistBatch(param)))
        {
            _partitioner[pageId].UpdatePosition(ctx, page, nodes, values, view);
            pageId++;
        }
    }

    public void ApplyTreeSplit(MultiExpandEntry candidate, RegTree tree) => _evaluator.ApplyTreeSplit(candidate, tree);

    public void InitData(DMatrix fmat, RegTree tree, TensorView<GradientPair> gpair)
    {
        HistUpdate.ResetPartitioners(ctx, fmat, param, _partitioner, out var nTotalBins);
        var nTargets = (int)gpair.Shape(1);
        _histogramBuilder.Reset(ctx, nTotalBins, nTargets, histParam);
        _evaluator = new HistMultiEvaluator(ctx, fmat.Info, param, (uint)nTargets, columnSampler);
    }

    public MultiExpandEntry InitRoot(DMatrix fmat, TensorView<GradientPair> gpair, RegTree tree)
    {
        var best = new MultiExpandEntry(RegTree.Root, 0);
        var nTargets = (int)gpair.Shape(1);
        var rootSum = Stats.SumGradients(ctx, gpair);
        var buf = new double[nTargets * 2];
        for (var t = 0; t < nTargets; ++t)
        {
            buf[2 * t] = rootSum[t].Grad;
            buf[2 * t + 1] = rootSum[t].Hess;
        }
        Communicator.Allreduce(buf.AsSpan(), Op.Sum);
        for (var t = 0; t < nTargets; ++t) rootSum[t] = new GradientPairPrecise(buf[2 * t], buf[2 * t + 1]);
        _histogramBuilder.BuildRootHist(fmat, tree.View(), _partitioner, gpair, best.Nid, HistUpdate.HistBatch(param));
        var weight = _evaluator.InitRoot(rootSum);
        for (var i = 0; i < weight.Length; ++i) weight[i] *= param.LearningRate;
        var rootSumHess = 0.0f;
        for (var t = 0; t < nTargets; ++t) rootSumHess += (float)rootSum[t].Hess;
        tree.SetRoot(weight, rootSumHess);
        var hists = new List<BoundedHistCollection>();
        for (var t = 0; t < nTargets; ++t) hists.Add(_histogramBuilder.Histogram(t));
        var nodes = new List<MultiExpandEntry> { new(RegTree.Root, 0) };
        var ft = fmat.Info.FeatureTypes.ConstHostSpan;
        foreach (var gmat in fmat.GetGradientIndex(ctx, HistUpdate.HistBatch(param)))
        {
            _evaluator.EvaluateSplits(hists, gmat.Cut, ft, nodes);
            break;
        }
        return nodes[0];
    }

    public void BuildHistogram(DMatrix fmat, RegTree tree, List<MultiExpandEntry> validCandidates, TensorView<GradientPair> gpair)
    {
        var view = tree.View();
        var nodesToBuild = new List<int>();
        var nodesToSub = new List<int>();
        MultiHistogramBuilder.AssignNodes(view, validCandidates, nodesToBuild, nodesToSub);
        _histogramBuilder.BuildHistLeftRight(fmat, view, _partitioner, nodesToBuild, nodesToSub, gpair, HistUpdate.HistBatch(param));
    }

    public void EvaluateSplits(DMatrix fmat, RegTree tree, List<MultiExpandEntry> bestSplits)
    {
        var hists = new List<BoundedHistCollection>();
        for (var t = 0; t < _histogramBuilder.NumTargets; ++t) hists.Add(_histogramBuilder.Histogram(t));
        var ft = fmat.Info.FeatureTypes.ConstHostSpan;
        foreach (var gmat in fmat.GetGradientIndex(ctx, HistUpdate.HistBatch(param)))
        {
            _evaluator.EvaluateSplits(hists, gmat.Cut, ft, bestSplits);
            break;
        }
    }

    public void LeafPartition(RegTree tree, TensorView<GradientPair> gpair, int[] outPosition)
    {
        var view = tree.View();
        foreach (var part in _partitioner) part.LeafPartition(ctx, view, gpair, outPosition);
    }

    /// <summary>Recomputes the leaf weights from the value gradient (reduced-gradient training).</summary>
    public void ExpandTreeLeaf(Tensor<GradientPair> fullGrad, RegTree tree)
    {
        Check.Eq(fullGrad.Shape(1), (long)tree.NumTargets);
        var view = new MultiTargetTreeView(tree);
        var nTargets = (int)tree.NumTargets;
        var valueGpair = fullGrad.HostView();
        var leavesIdx = new List<int>();
        Check.That(_partitioner.Count != 0);
        foreach (var node in _partitioner[0].Partitions.Elems)
            if (node.NodeId >= 0 && view.IsLeaf(node.NodeId)) leavesIdx.Add(node.NodeId);
        var nLeaves = leavesIdx.Count;
        Check.Eq(tree.GetNumLeaves(), nLeaves);
        Check.Gt(nLeaves, 0);
        var nThreads = ctx.Threads();
        var tloc = new GradientPairPrecise[nThreads, nLeaves, nTargets];
        foreach (var part in _partitioner)
        {
            var space = new BlockedSpace2d(nLeaves, leafIdx => part[leavesIdx[(int)leafIdx]].Size, 1024);
            Threading.ParallelFor2d(space, nThreads, (leafIdx, r) =>
            {
                var node = part[leavesIdx[(int)leafIdx]];
                var tidx = Threading.ThreadNum;
                for (var it = node.Begin + r.Begin; it != node.Begin + r.End; ++it)
                {
                    var row = part.Partitions.Data[it];
                    for (var t = 0; t < nTargets; ++t) tloc[tidx, leafIdx, t] += new GradientPairPrecise(valueGpair[row, t]);
                }
            });
        }
        var leafSums = new GradientPairPrecise[nLeaves * nTargets];
        for (var i = 0; i < nThreads; ++i)
            for (var j = 0; j < nLeaves; ++j)
                for (var k = 0; k < nTargets; ++k)
                    leafSums[j * nTargets + k] += tloc[i, j, k];
        var weights = new float[nLeaves * nTargets];
        var eta = param.LearningRate;
        var evaluator = _evaluator.Evaluator;
        Threading.ParallelFor(nLeaves, nThreads, leafIdx =>
        {
            var gradSum = leafSums.AsSpan((int)leafIdx * nTargets, nTargets);
            var weight = weights.AsSpan((int)leafIdx * nTargets, nTargets);
            evaluator.CalcWeight(leavesIdx[(int)leafIdx], param, gradSum, weight);
            for (var t = 0; t < nTargets; ++t) weight[t] *= eta;
        });
        tree.SetLeaves(leavesIdx, weights);
    }
}

/// <summary><c>QuantileHistMaker</c>, registered as <c>grow_quantile_histmaker</c>.</summary>
public sealed class QuantileHistMaker(Context ctx) : TreeUpdater(ctx)
{
    private HistUpdater? _pImpl;
    private MultiTargetHistBuilder? _pMtImpl;
    private readonly ColumnSampler _columnSampler = new();
    private readonly HistMakerTrainParam _histParam = new();

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _histParam.UpdateAllowUnknown(args));

    public override void LoadConfig(Json input) => _histParam.FromJson(input.AsObject["hist_train_param"]);

    public override void SaveConfig(JsonObject output) => output["hist_train_param"] = _histParam.ToJson();

    public override string Name => "grow_quantile_histmaker";

    public override void Update(TrainParam param, GradientContainer inGpair, DMatrix fmat, List<HostDeviceVector<int>> outPosition,
        List<RegTree> trees)
    {
        Check.That(_histParam.GetInitialised());
        if (trees[0].IsMultiTarget) _pMtImpl ??= new MultiTargetHistBuilder(Ctx, param, _histParam, _columnSampler);
        else _pImpl ??= new HistUpdater(Ctx, _columnSampler, param, _histParam);

        var nTargets = trees[0].NumTargets;
        var gpair = inGpair.Grad;
        var hGpair = gpair.HostView();
        var needCopy = trees.Count > 1 || nTargets > 1 || inGpair.HasValueGrad;
        var hSampleOut = hGpair;
        Tensor<GradientPair>? sampleOut = null;
        if (needCopy)
        {
            sampleOut = new Tensor<GradientPair>(hGpair.Shape().ToArray(), Order.F);
            hSampleOut = sampleOut.HostView();
        }
        var sampler = new Sampler(param);
        for (var i = 0; i < trees.Count; ++i)
        {
            if (needCopy)
            {
                for (long r = 0; r < hGpair.Shape(0); ++r)
                    for (long c = 0; c < hGpair.Shape(1); ++c)
                        hSampleOut[r, c] = hGpair[r, c];
            }
            sampler.Sample(Ctx, hSampleOut);
            var hOutPosition = outPosition[i];
            if (trees[i].IsMultiTarget)
            {
                HistUpdate.UpdateTree(hSampleOut, _pMtImpl!, fmat, param, hOutPosition, trees[i]);
                if (inGpair.HasValueGrad)
                {
                    var valueGrad = inGpair.ValueGpair.Clone();
                    sampler.ApplySampling(Ctx, hGpair, valueGrad);
                    _pMtImpl!.ExpandTreeLeaf(valueGrad, trees[i]);
                }
                else
                {
                    trees[i].GetMultiTargetTree().SetLeaves();
                }
            }
            else
            {
                HistUpdate.UpdateTree(hSampleOut, _pImpl!, fmat, param, hOutPosition, trees[i]);
            }
            _histParam.CheckTreesSynchronized(Ctx, trees[i]);
        }
    }
}
