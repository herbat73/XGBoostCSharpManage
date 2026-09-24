// Port of src/tree/updater_approx.cc (grow_histmaker).
using XGBoost.Collective;

namespace XGBoost.Tree;

internal sealed class GlobalApproxBuilder
{
    private readonly TrainParam _param;
    private readonly HistMakerTrainParam _histParam;
    private readonly HistEvaluator _evaluator;
    private readonly MultiHistogramBuilder _histogramBuilder = new();
    private readonly Context _ctx;
    private readonly ObjInfo _task;
    private readonly List<CommonRowPartitioner> _partitioner = [];
    private HistogramCuts _featureValues = new(0);

    public GlobalApproxBuilder(TrainParam param, HistMakerTrainParam histParam, MetaInfo info, Context ctx, ColumnSampler columnSampler,
        ObjInfo task)
    {
        _param = param;
        _histParam = histParam;
        _evaluator = new HistEvaluator(ctx, param, info, columnSampler);
        _ctx = ctx;
        _task = task;
    }

    private BatchParam BatchSpecTask(float[] hess) => new(_param.MaxBin, hess, !_task.ConstHess);
    private BatchParam BatchSpec(float[] hess) => new(_param.MaxBin, hess, false);

    private void InitData(DMatrix fmat, RegTree tree, float[] hess)
    {
        var nTotalBins = 0;
        _partitioner.Clear();
        foreach (var page in fmat.GetGradientIndex(_ctx, BatchSpecTask(hess)))
        {
            if (nTotalBins == 0)
            {
                nTotalBins = page.Cut.TotalBins;
                _featureValues = page.Cut;
            }
            else
            {
                Check.Eq(nTotalBins, page.Cut.TotalBins);
            }
            _partitioner.Add(new CommonRowPartitioner(_ctx, page.Size, page.BaseRowId));
        }
        _histogramBuilder.Reset(_ctx, nTotalBins, (int)tree.NumTargets, _histParam);
    }

    private CPUExpandEntry InitRoot(DMatrix fmat, TensorView<GradientPair> gpair, float[] hess, RegTree tree)
    {
        var best = new CPUExpandEntry(RegTree.Root, 0);
        var rootSum = new GradStats();
        for (long i = 0; i < gpair.Shape(0); ++i) rootSum.Add(gpair[i, 0]);
        Span<double> buf = [rootSum.SumGrad, rootSum.SumHess];
        Communicator.Allreduce(buf, Op.Sum);
        rootSum = new GradStats(buf[0], buf[1]);
        var nodes = new List<CPUExpandEntry> { best };
        _histogramBuilder.BuildRootHist(fmat, tree.View(), _partitioner, gpair, best.Nid, BatchSpec(hess));
        var weight = _evaluator.InitRoot(rootSum);
        tree.Stat(RegTree.Root).SumHess = (float)rootSum.GetHess();
        tree.Stat(RegTree.Root).BaseWeight = weight;
        tree[RegTree.Root].SetLeaf(_param.LearningRate * weight);
        _evaluator.EvaluateSplits(_histogramBuilder.Histogram(0), _featureValues, fmat.Info.FeatureTypes.ConstHostSpan, nodes);
        return nodes[0];
    }

    public void UpdateTree(DMatrix fmat, TensorView<GradientPair> gpair, float[] hess, RegTree tree, HostDeviceVector<int> outPosition)
    {
        Check.That(!tree.IsMultiTarget, "approx" + ErrorMsg.MTNotImplemented);
        InitData(fmat, tree, hess);
        var driver = new Driver<CPUExpandEntry>(_param);
        driver.Push(InitRoot(fmat, gpair, hess, tree));
        var expandSet = driver.Pop();
        while (expandSet.Count != 0)
        {
            var validCandidates = new List<CPUExpandEntry>();
            var applied = new List<CPUExpandEntry>();
            foreach (var candidate in expandSet)
            {
                _evaluator.ApplyTreeSplit(candidate, tree);
                applied.Add(candidate);
                if (driver.IsChildValid(candidate)) validCandidates.Add(candidate);
            }
            var pageId = 0;
            var view = tree.View();
            var appliedNodes = applied.Select(a => a.Nid).ToList();
            var appliedValues = applied.Select(a => a.Split.SplitValue).ToList();
            foreach (var page in fmat.GetGradientIndex(_ctx, BatchSpec(hess)))
            {
                _partitioner[pageId].UpdatePosition(_ctx, page, appliedNodes, appliedValues, view);
                pageId++;
            }
            var bestSplits = new List<CPUExpandEntry>();
            if (validCandidates.Count != 0)
            {
                view = tree.View();
                var nodesToBuild = new List<int>();
                var nodesToSub = new List<int>();
                MultiHistogramBuilder.AssignNodes(view, validCandidates, nodesToBuild, nodesToSub);
                _histogramBuilder.BuildHistLeftRight(fmat, view, _partitioner, nodesToBuild, nodesToSub, gpair, BatchSpec(hess));
                foreach (var candidate in validCandidates)
                {
                    var l = tree[candidate.Nid].LeftChild;
                    var r = tree[candidate.Nid].RightChild;
                    bestSplits.Add(new CPUExpandEntry(l, tree.GetDepth(l)));
                    bestSplits.Add(new CPUExpandEntry(r, tree.GetDepth(r)));
                }
                _evaluator.EvaluateSplits(_histogramBuilder.Histogram(0), _featureValues, fmat.Info.FeatureTypes.ConstHostSpan, bestSplits);
            }
            driver.Push(bestSplits);
            expandSet = driver.Pop();
        }
        var position = new int[hess.Length];
        var finalView = tree.View();
        foreach (var part in _partitioner) part.LeafPartition(_ctx, finalView, hess, position);
        outPosition.Assign(position, true);
    }
}

public sealed class GlobalApproxUpdater(Context ctx, ObjInfo task) : TreeUpdater(ctx)
{
    private readonly ColumnSampler _columnSampler = new();
    private readonly HistMakerTrainParam _histParam = new();

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _histParam.UpdateAllowUnknown(args));

    public override void LoadConfig(Json input) => _histParam.FromJson(input.AsObject["hist_train_param"]);

    public override void SaveConfig(JsonObject output) => output["hist_train_param"] = _histParam.ToJson();

    public override string Name => "grow_histmaker";

    public override void Update(TrainParam param, GradientContainer inGpair, DMatrix m, List<HostDeviceVector<int>> outPosition,
        List<RegTree> trees)
    {
        Check.That(_histParam.GetInitialised());
        var pimpl = new GlobalApproxBuilder(param, _histParam, m.Info, Ctx, _columnSampler, task);
        var gpair = inGpair.FullGradOnly();
        var sampled = new Tensor<GradientPair>([gpair.Size, 1]);
        gpair.Data.ConstHostSpan.CopyTo(sampled.Data.HostSpan);
        new Sampler(param).Sample(Ctx, sampled.HostView());
        var sGpair = sampled.Data.ToArray();
        var hess = Array.ConvertAll(sGpair, g => g.Hess);
        var view = sampled.HostView();
        for (var t = 0; t < trees.Count; ++t)
        {
            pimpl.UpdateTree(m, view, hess, trees[t], outPosition[t]);
            _histParam.CheckTreesSynchronized(Ctx, trees[t]);
        }
    }
}
