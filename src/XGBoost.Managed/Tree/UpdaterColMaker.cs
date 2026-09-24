// Port of src/tree/updater_colmaker.cc (grow_colmaker, the exact tree method).
using XGBoost.Collective;
using XGBoost.Predictors;

namespace XGBoost.Tree;

public sealed class ColMakerTrainParam : XGBoostParameter<ColMakerTrainParam>
{
    public float OptDenseCol;
    public int DefaultDirection;

    protected override void Declare(ParamManager<ColMakerTrainParam> m)
    {
        Field(m, "opt_dense_col", p => p.OptDenseCol, (p, v) => p.OptDenseCol = v).SetRange(0.0f, 1.0f).SetDefault(1.0f)
            .Describe("EXP Param: speed optimization for dense column.");
        CustomField(m, "default_direction", "int", p => p.DefaultDirection, (p, v) => p.DefaultDirection = v, ParseDirection,
                v => v switch { 0 => "learn", 1 => "left", 2 => "right", _ => v.ToString(System.Globalization.CultureInfo.InvariantCulture) })
            .SetDefault(0).Describe("Default direction choice when encountering a missing value");
    }

    private static int ParseDirection(string v) => v switch
    {
        "learn" => 0,
        "left" => 1,
        "right" => 2,
        _ => int.TryParse(v, out var i) && i is >= 0 and <= 2
            ? i
            : throw new XGBoostException($"Invalid Input: '{v}', valid values are: {{'learn', 'left', 'right'}}"),
    };

    public bool NeedForwardSearch(float colDensity, bool indicator) =>
        DefaultDirection == 2 || (DefaultDirection == 0 && colDensity < OptDenseCol && !indicator);

    public bool NeedBackwardSearch() => DefaultDirection != 2;
}

public sealed class ColMaker(Context ctx) : TreeUpdater(ctx)
{
    private readonly ColMakerTrainParam _colmakerParam = new();
    private float[] _columnDensities = [];
    private readonly ColumnSampler _columnSampler = new();

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _colmakerParam.UpdateAllowUnknown(args));

    public override void LoadConfig(Json input) => _colmakerParam.FromJson(input.AsObject["colmaker_train_param"]);

    public override void SaveConfig(JsonObject output) => output["colmaker_train_param"] = _colmakerParam.ToJson();

    public override string Name => "grow_colmaker";

    private void LazyGetColumnDensity(DMatrix dmat)
    {
        if (_columnDensities.Length != 0) return;
        var columnSize = new long[dmat.Info.NumCol];
        foreach (var batch in dmat.GetSortedColumnBatches(Ctx))
        {
            var page = batch.GetView();
            for (var i = 0; i < page.Size; i++) columnSize[i] += page[i].Length;
        }
        _columnDensities = new float[columnSize.Length];
        for (var i = 0; i < _columnDensities.Length; i++)
        {
            var nmiss = dmat.Info.NumRow - columnSize[i];
            _columnDensities[i] = 1.0f - (float)nmiss / dmat.Info.NumRow;
        }
    }

    public override void Update(TrainParam param, GradientContainer inGpair, DMatrix dmat, List<HostDeviceVector<int>> outPosition,
        List<RegTree> trees)
    {
        if (Communicator.IsDistributed())
            Check.Fail("Updater `grow_colmaker` or `exact` tree method doesn't support distributed training.");
        if (!dmat.SingleColBlock)
            Check.Fail("Updater `grow_colmaker` or `exact` tree method doesn't support external memory training.");
        if (dmat.Info.HasCategorical) Check.Fail(ErrorMsg.NoCategorical("Updater `grow_colmaker` or `exact` tree method"));
        if (param.ColsampleBynode - 1.0 != 0.0) Check.Fail("column sample by node is not yet supported by the exact tree method");
        LazyGetColumnDensity(dmat);
        var gpair = inGpair.FullGradOnly();
        Check.Eq(gpair.Shape(1), 1L, ErrorMsg.MTNotImplemented);
        var hGpair = gpair.Data.ToArray();
        for (var i = 0; i < trees.Count; ++i)
        {
            Check.That(!trees[i].IsMultiTarget, "exact" + ErrorMsg.MTNotImplemented);
            // The builder takes a copy of the freshly configured interaction constraints.
            var constraints = new FeatureInteractionConstraintHost();
            constraints.Configure(param, (uint)dmat.Info.NumRow);
            var builder = new Builder(param, _colmakerParam, constraints, Ctx, _columnDensities, _columnSampler);
            builder.Update(hGpair, dmat, trees[i], outPosition[i]);
        }
    }

    private sealed class ThreadEntry
    {
        public GradStats Stats;
        public float LastFvalue;
        public SplitEntry Best = new();
    }

    private sealed class NodeEntry
    {
        public GradStats Stats;
        public float RootGain;
        public float Weight;
        public SplitEntry Best = new();
    }

    private sealed class Builder(TrainParam param, ColMakerTrainParam colmakerParam, FeatureInteractionConstraintHost interactionConstraints,
        Context ctx, float[] columnDensities, ColumnSampler columnSampler)
    {
        private readonly TreeEvaluator _treeEvaluator = new(param, (uint)columnDensities.Length, 1u);
        private int[] _position = [];
        private bool[] _rowIsValid = [];
        private List<ThreadEntry>[] _stemp = [];
        private readonly List<NodeEntry> _snode = [];
        private List<int> _qexpand = [];

        public void Update(GradientPair[] gpair, DMatrix fmat, RegTree tree, HostDeviceVector<int> outPosition)
        {
            InitData(gpair, fmat);
            InitRoot(gpair, fmat, tree);
            var newnodes = new List<int>();
            Check.Gt(param.MaxDepth, 0, "exact tree method doesn't support unlimited depth.");
            for (var depth = 0; depth < param.MaxDepth; ++depth)
            {
                FindSplit(depth, _qexpand, gpair, fmat, tree);
                ResetPosition(_qexpand, fmat, tree);
                UpdateQueueExpand(tree, _qexpand, newnodes);
                InitNewNode(newnodes, gpair, fmat, tree);
                foreach (var nid in _qexpand)
                {
                    if (tree[nid].IsLeaf) continue;
                    var cleft = tree[nid].LeftChild;
                    var cright = tree[nid].RightChild;
                    _treeEvaluator.AddSplit(nid, cleft, cright, _snode[nid].Best.SplitIndex, _snode[cleft].Weight, _snode[cright].Weight);
                    interactionConstraints.Split(nid, _snode[nid].Best.SplitIndex, cleft, cright);
                }
                _qexpand = [.. newnodes];
                if (_qexpand.Count == 0) break;
            }
            foreach (var nid in _qexpand) tree[nid].SetLeaf(_snode[nid].Weight * param.LearningRate);
            for (var nid = 0; nid < tree.NumNodes; ++nid)
            {
                ref var stat = ref tree.Stat(nid);
                stat.LossChg = _snode[nid].Best.LossChg;
                stat.BaseWeight = _snode[nid].Weight;
                stat.SumHess = (float)_snode[nid].Stats.SumHess;
            }
            var hPosition = new int[_position.Length];
            Check.Eq(_rowIsValid.Length, _position.Length);
            for (var i = 0; i < _position.Length; ++i)
                hPosition[i] = SamplePosition.Encode(SamplePosition.Decode(_position[i]), _rowIsValid[i]);
            outPosition.Assign(hPosition, true);
        }

        private void InitData(GradientPair[] gpair, DMatrix fmat)
        {
            _position = new int[gpair.Length];
            Check.Eq(fmat.Info.NumRow, (long)_position.Length);
            _rowIsValid = new bool[_position.Length];
            Array.Fill(_rowIsValid, true);
            for (var ridx = 0; ridx < _position.Length; ++ridx)
            {
                if (gpair[ridx].Hess < 0.0f)
                {
                    _position[ridx] = ~_position[ridx];
                    _rowIsValid[ridx] = false;
                }
            }
            if (param.Subsample < 1.0f)
            {
                Check.That(param.SamplingMethod == SamplingMethod.Uniform,
                    "Only uniform sampling is supported, gradient-based sampling is only support by the `hist` tree method.");
                var rnd = ctx.Rng;
                for (var ridx = 0; ridx < _position.Length; ++ridx)
                {
                    if (!_rowIsValid[ridx]) continue;
                    if (!StdRandom.Bernoulli(rnd, param.Subsample))
                    {
                        _position[ridx] = ~_position[ridx];
                        _rowIsValid[ridx] = false;
                    }
                }
            }
            columnSampler.Init(ctx, fmat.Info.NumCol, fmat.Info.FeatureWeights.ToArray(), param.ColsampleBynode, param.ColsampleBylevel,
                param.ColsampleBytree);
            _stemp = new List<ThreadEntry>[ctx.Threads()];
            for (var i = 0; i < _stemp.Length; ++i) _stemp[i] = [];
            _qexpand = [RegTree.Root];
        }

        private void InitNodeStats(List<int> nodes, GradientPair[] gpair, DMatrix fmat, RegTree tree)
        {
            var nNodes = tree.NumNodes;
            foreach (var s in _stemp)
                while (s.Count < nNodes) s.Add(new ThreadEntry());
            while (_snode.Count < nNodes) _snode.Add(new NodeEntry());
            Threading.ParallelFor(fmat.Info.NumRow, ctx.Threads(), (ridx, tid) =>
            {
                if (_position[ridx] < 0) return;
                _stemp[tid][_position[ridx]].Stats.Add(gpair[ridx]);
            });
            foreach (var nid in nodes)
            {
                var stats = new GradStats();
                foreach (var s in _stemp) stats.Add(s[nid].Stats);
                _snode[nid].Stats = stats;
            }
        }

        private void InitRoot(GradientPair[] gpair, DMatrix fmat, RegTree tree)
        {
            Check.Eq(_qexpand.Count, 1);
            Check.Eq(_qexpand[0], RegTree.Root);
            InitNodeStats(_qexpand, gpair, fmat, tree);
            var root = _snode[RegTree.Root];
            root.Weight = _treeEvaluator.CalcWeight(RegTree.Root, param, root.Stats);
            root.RootGain = _treeEvaluator.CalcGain(RegTree.Root, param, root.Stats);
        }

        private void InitNewNode(List<int> qexpand, GradientPair[] gpair, DMatrix fmat, RegTree tree)
        {
            foreach (var nidx in qexpand) Check.Ne(nidx, RegTree.Root);
            InitNodeStats(qexpand, gpair, fmat, tree);
            foreach (var nidx in qexpand)
            {
                var parentId = tree[nidx].Parent;
                _snode[nidx].Weight = _treeEvaluator.CalcWeight(parentId, param, _snode[nidx].Stats);
                _snode[nidx].RootGain = _treeEvaluator.CalcGain(parentId, param, _snode[nidx].Stats);
            }
        }

        private static void UpdateQueueExpand(RegTree tree, List<int> qexpand, List<int> newnodes)
        {
            newnodes.Clear();
            foreach (var nidx in qexpand)
            {
                if (!tree[nidx].IsLeaf)
                {
                    newnodes.Add(tree[nidx].LeftChild);
                    newnodes.Add(tree[nidx].RightChild);
                }
            }
        }

        private void UpdateEnumeration(int nid, GradientPair gstats, float fvalue, int dStep, uint fid, ref GradStats c, List<ThreadEntry> temp)
        {
            var e = temp[nid];
            if (e.Stats.Empty)
            {
                e.Stats.Add(gstats);
                e.LastFvalue = fvalue;
                return;
            }
            if (fvalue != e.LastFvalue && e.Stats.SumHess >= param.MinChildWeight)
            {
                c.SetSubstract(_snode[nid].Stats, e.Stats);
                if (c.SumHess >= param.MinChildWeight)
                {
                    float lossChg;
                    var proposedSplit = (fvalue + e.LastFvalue) * 0.5f;
                    var splitValue = proposedSplit == fvalue ? e.LastFvalue : proposedSplit;
                    if (dStep == -1)
                    {
                        lossChg = _treeEvaluator.CalcSplitGain(param, nid, fid, c, e.Stats) - _snode[nid].RootGain;
                        e.Best.Update(lossChg, fid, splitValue, dStep == -1, false, c, e.Stats);
                    }
                    else
                    {
                        lossChg = _treeEvaluator.CalcSplitGain(param, nid, fid, e.Stats, c) - _snode[nid].RootGain;
                        e.Best.Update(lossChg, fid, splitValue, dStep == -1, false, e.Stats, c);
                    }
                }
            }
            e.Stats.Add(gstats);
            e.LastFvalue = fvalue;
        }

        private void EnumerateSplit(ReadOnlySpan<Entry> col, int dStep, uint fid, GradientPair[] gpair, List<ThreadEntry> temp)
        {
            foreach (var nid in _qexpand) temp[nid].Stats = new GradStats();
            var c = new GradStats();
            // The C++ code prefetches in buffers of 32 entries, which does not change the visiting order.
            var n = col.Length;
            for (var k = 0; k < n; ++k)
            {
                var entry = dStep > 0 ? col[k] : col[n - 1 - k];
                var nid = _position[entry.Index];
                if (nid < 0 || !interactionConstraints.Query(nid, fid)) continue;
                UpdateEnumeration(nid, gpair[entry.Index], entry.Fvalue, dStep, fid, ref c, temp);
            }
            foreach (var nid in _qexpand)
            {
                var e = temp[nid];
                c.SetSubstract(_snode[nid].Stats, e.Stats);
                if (e.Stats.SumHess >= param.MinChildWeight && c.SumHess >= param.MinChildWeight)
                {
                    var gap = MathF.Abs(e.LastFvalue) + Constants.RtEps;
                    var delta = dStep == +1 ? gap : -gap;
                    if (dStep == -1)
                    {
                        var lossChg = _treeEvaluator.CalcSplitGain(param, nid, fid, c, e.Stats) - _snode[nid].RootGain;
                        e.Best.Update(lossChg, fid, e.LastFvalue + delta, dStep == -1, false, c, e.Stats);
                    }
                    else
                    {
                        var lossChg = _treeEvaluator.CalcSplitGain(param, nid, fid, e.Stats, c) - _snode[nid].RootGain;
                        e.Best.Update(lossChg, fid, e.LastFvalue + delta, dStep == -1, false, e.Stats, c);
                    }
                }
            }
        }

        private void UpdateSolution(SortedCscPage batch, uint[] featSet, GradientPair[] gpair)
        {
            var numFeatures = featSet.Length;
            var batchSize = Math.Max(numFeatures / ctx.Threads() / 32, 1);
            var page = batch.GetView();
            Threading.ParallelFor(numFeatures, ctx.Threads(), Sched.Dyn(batchSize), (i, tid) =>
            {
                var fid = featSet[i];
                var c = page[fid];
                var ind = c.Length != 0 && c[0].Fvalue == c[^1].Fvalue;
                if (colmakerParam.NeedForwardSearch(columnDensities[fid], ind)) EnumerateSplit(c, +1, fid, gpair, _stemp[tid]);
                if (colmakerParam.NeedBackwardSearch()) EnumerateSplit(c, -1, fid, gpair, _stemp[tid]);
            });
        }

        private void FindSplit(int depth, List<int> qexpand, GradientPair[] gpair, DMatrix fmat, RegTree tree)
        {
            var featSet = columnSampler.GetFeatureSet(ctx, depth);
            foreach (var batch in fmat.GetSortedColumnBatches(ctx)) UpdateSolution(batch, featSet, gpair);
            foreach (var nid in qexpand)
            {
                var e = _snode[nid];
                for (var tid = 0; tid < ctx.Threads(); ++tid) e.Best.Update(_stemp[tid][nid].Best);
            }
            foreach (var nid in qexpand)
            {
                var e = _snode[nid];
                if (e.Best.LossChg > Constants.RtEps)
                {
                    var leftLeafWeight = _treeEvaluator.CalcWeight(nid, param, e.Best.LeftSum) * param.LearningRate;
                    var rightLeafWeight = _treeEvaluator.CalcWeight(nid, param, e.Best.RightSum) * param.LearningRate;
                    tree.ExpandNode(nid, e.Best.SplitIndex, e.Best.SplitValue, e.Best.DefaultLeft, e.Weight, leftLeafWeight, rightLeafWeight,
                        e.Best.LossChg, (float)e.Stats.SumHess, (float)e.Best.LeftSum.GetHess(), (float)e.Best.RightSum.GetHess(), 0);
                }
                else
                {
                    tree[nid].SetLeaf(e.Weight * param.LearningRate);
                }
            }
        }

        private void SetEncodePosition(long ridx, int nidx)
        {
            var isInvalid = _position[ridx] < 0;
            _position[ridx] = SamplePosition.Encode(nidx, !isInvalid);
        }

        private void ResetPosition(List<int> qexpand, DMatrix fmat, RegTree tree)
        {
            SetNonDefaultPosition(qexpand, fmat, tree);
            var nodes = tree.GetNodes().ToArray();
            Threading.ParallelFor(fmat.Info.NumRow, ctx.Threads(), ridx =>
            {
                Check.Lt(ridx, (long)_position.Length);
                var nidx = SamplePosition.Decode(_position[ridx]);
                if (nodes[nidx].IsLeaf)
                {
                    if (nodes[nidx].RightChild == -1) _position[ridx] = ~nidx;
                }
                else
                {
                    SetEncodePosition(ridx, nodes[nidx].DefaultLeft ? nodes[nidx].LeftChild : nodes[nidx].RightChild);
                }
            });
        }

        private void SetNonDefaultPosition(List<int> qexpand, DMatrix fmat, RegTree tree)
        {
            var nodes = tree.GetNodes().ToArray();
            var fsplits = new List<uint>();
            foreach (var nid in qexpand)
                if (!nodes[nid].IsLeaf) fsplits.Add(nodes[nid].SplitIndex);
            var unique = fsplits.Distinct().Order().ToArray();
            foreach (var batch in fmat.GetSortedColumnBatches(ctx))
            {
                var page = batch.GetView();
                foreach (var fid in unique)
                {
                    var col = page[fid].ToArray();
                    Threading.ParallelFor(col.Length, ctx.Threads(), j =>
                    {
                        var ridx = col[j].Index;
                        var nidx = SamplePosition.Decode(_position[ridx]);
                        var fvalue = col[j].Fvalue;
                        if (!nodes[nidx].IsLeaf && nodes[nidx].SplitIndex == fid)
                            SetEncodePosition(ridx, fvalue < nodes[nidx].SplitCond ? nodes[nidx].LeftChild : nodes[nidx].RightChild);
                    });
                }
            }
        }
    }
}
