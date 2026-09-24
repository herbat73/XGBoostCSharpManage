// Ports of src/tree/updater_prune.cc, updater_refresh.cc and the updater registrations.
using XGBoost.Predictors;

namespace XGBoost.Tree;

public sealed class TreePruner(Context ctx) : TreeUpdater(ctx)
{
    public override string Name => "prune";
    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override bool CanModifyTree => true;

    public override void Update(TrainParam param, GradientContainer gpair, DMatrix fmat, List<HostDeviceVector<int>> outPosition,
        List<RegTree> trees)
    {
        foreach (var tree in trees) DoPrune(param, tree);
        // Synchronize: a no-op with a single worker.
        for (var i = 0; i < trees.Count; ++i) UpdatePosition(fmat, trees[i], outPosition[i]);
    }

    private static int TryPruneLeaf(TrainParam param, RegTree tree, int nid, int depth, int npruned)
    {
        Check.That(tree[nid].IsLeaf);
        if (tree[nid].IsRoot) return npruned;
        var pid = tree[nid].Parent;
        Check.That(!tree[pid].IsLeaf);
        var s = tree.Stat(pid);
        var left = tree[pid].LeftChild;
        var right = tree[pid].RightChild;
        var balanced = tree[left].IsLeaf && right != RegTree.InvalidNodeId && tree[right].IsLeaf;
        if (balanced && param.NeedPrune(s.LossChg, depth))
        {
            tree.ChangeToLeaf(pid, param.LearningRate * s.BaseWeight);
            return TryPruneLeaf(param, tree, pid, depth - 1, npruned + 2);
        }
        return npruned;
    }

    private void PredictPosition(DMatrix fmat, RegTree tree, HostDeviceVector<int> position)
    {
        var hPosition = new int[fmat.Info.NumRow];
        var feats = new RegTree.FVec();
        feats.Init((int)tree.NumFeatures);
        var view = tree.View();
        var cats = view.GetCategoriesMatrix();
        foreach (var batch in fmat.GetRowBatches())
        {
            var page = batch.GetView();
            for (long i = 0; i < page.Size; ++i)
            {
                feats.Fill(page[i]);
                var nidx = RegTree.Root;
                while (!view.IsLeaf(nidx))
                {
                    var splitIndex = (int)view.SplitIndex(nidx);
                    nidx = PredictFn.GetNextNode(view, nidx, feats.GetFvalue(splitIndex), feats.IsMissing(splitIndex), cats,
                        fmat.Info.HasCategorical);
                }
                hPosition[batch.BaseRowId + i] = nidx;
                feats.Drop();
            }
        }
        position.Assign(hPosition, true);
    }

    private void UpdatePosition(DMatrix fmat, RegTree tree, HostDeviceVector<int> position)
    {
        if (position.Size != fmat.Info.NumRow) PredictPosition(fmat, tree, position);
        var nodes = tree.GetNodes();
        var hPosition = position.HostSpan;
        for (var i = 0; i < hPosition.Length; ++i)
        {
            var encoded = hPosition[i];
            var valid = SamplePosition.IsValid(encoded);
            var nidx = SamplePosition.Decode(encoded);
            while (nodes[nidx].IsDeleted) nidx = nodes[nidx].Parent;
            Check.That(nodes[nidx].IsLeaf);
            hPosition[i] = SamplePosition.Encode(nidx, valid);
        }
    }

    private static void DoPrune(TrainParam param, RegTree tree)
    {
        Check.That(!tree.IsMultiTarget, "Pruning" + ErrorMsg.MTNotImplemented);
        var npruned = 0;
        for (var nid = 0; nid < tree.NumNodes; ++nid)
        {
            if (tree[nid].IsLeaf && !tree[nid].IsDeleted) npruned = TryPruneLeaf(param, tree, nid, tree.GetDepth(nid), npruned);
        }
        Log.Info($"tree pruning end, {tree.NumExtraNodes} extra nodes, {npruned} pruned nodes, max_depth={tree.MaxDepth()}");
    }
}

public sealed class TreeRefresher(Context ctx) : TreeUpdater(ctx)
{
    public override string Name => "refresh";
    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override bool CanModifyTree => true;

    public override void Update(TrainParam param, GradientContainer inGpair, DMatrix fmat, List<HostDeviceVector<int>> outPosition,
        List<RegTree> trees)
    {
        if (trees.Count == 0) return;
        Check.That(!param.HasMonotone(), "Monotonic constraint is not supported by the `refresh` updater.");
        var gpair = inGpair.FullGradOnly();
        Check.Eq(gpair.Shape(1), 1L, ErrorMsg.MTNotImplemented);
        var gpairH = gpair.Data.ToArray();
        var nthread = Ctx.Threads();
        var numNodes = 0;
        foreach (var tree in trees) numNodes += tree.NumNodes;
        var stemp = new GradStats[nthread][];
        var fvecTemp = new RegTree.FVec[nthread];
        for (var tid = 0; tid < nthread; ++tid)
        {
            stemp[tid] = new GradStats[numNodes];
            fvecTemp[tid] = new RegTree.FVec();
            fvecTemp[tid].Init((int)trees[0].NumFeatures);
        }
        Check.Eq(outPosition.Count, trees.Count);
        var info = fmat.Info;
        var hPosition = new int[trees.Count][];
        for (var i = 0; i < trees.Count; ++i) hPosition[i] = new int[info.NumRow];
        var views = trees.Select(t => t.View()).ToArray();
        foreach (var batch in fmat.GetRowBatches())
        {
            var page = batch.GetView();
            Threading.ParallelFor(page.Size, Ctx.Threads(), (i, tid) =>
            {
                var feats = fvecTemp[tid];
                feats.Fill(page[i]);
                var ridx = (int)(batch.BaseRowId + i);
                var offset = 0;
                for (var treeIdx = 0; treeIdx < trees.Count; ++treeIdx)
                {
                    var leaf = AddStats(views[treeIdx], feats, gpairH, ridx, stemp[tid], offset);
                    hPosition[treeIdx][ridx] = SamplePosition.Encode(leaf, true);
                    offset += trees[treeIdx].NumNodes;
                }
                feats.Drop();
            });
        }
        Threading.ParallelFor(numNodes, Ctx.Threads(), nid =>
        {
            for (var tid = 1; tid < nthread; ++tid) stemp[0][nid].Add(stemp[tid][nid]);
        });
        var sumGrad = stemp[0];
        var off = 0;
        for (var t = 0; t < trees.Count; ++t)
        {
            Refresh(param, sumGrad, off, 0, trees[t]);
            off += trees[t].NumNodes;
            outPosition[t].Assign(hPosition[t], true);
        }
    }

    private static int AddStats(ITreeView tree, RegTree.FVec feat, GradientPair[] gpair, int ridx, GradStats[] gstats, int offset)
    {
        var pid = RegTree.Root;
        gstats[offset + pid].Add(gpair[ridx]);
        var cats = tree.GetCategoriesMatrix();
        while (!tree.IsLeaf(pid))
        {
            var splitIndex = (int)tree.SplitIndex(pid);
            pid = PredictFn.GetNextNode(tree, pid, feat.GetFvalue(splitIndex), feat.IsMissing(splitIndex), cats, true);
            gstats[offset + pid].Add(gpair[ridx]);
        }
        return pid;
    }

    private static void Refresh(TrainParam param, GradStats[] gstats, int offset, int nid, RegTree tree)
    {
        var s = gstats[offset + nid];
        tree.Stat(nid).BaseWeight = (float)SplitMath.CalcWeight(param, s.SumGrad, s.SumHess);
        tree.Stat(nid).SumHess = (float)s.SumHess;
        if (tree[nid].IsLeaf)
        {
            if (param.RefreshLeaf) tree[nid].SetLeaf(tree.Stat(nid).BaseWeight * param.LearningRate);
        }
        else
        {
            var l = tree[nid].LeftChild;
            var r = tree[nid].RightChild;
            tree.Stat(nid).LossChg = (float)(SplitMath.CalcGain(param, gstats[offset + l]) + SplitMath.CalcGain(param, gstats[offset + r]) -
                                             SplitMath.CalcGain(param, s));
            Refresh(param, gstats, offset, l, tree);
            Refresh(param, gstats, offset, r, tree);
        }
    }
}

internal static class UpdaterRegistry
{
    public static void Register()
    {
        var r = Registry.TreeUpdaterFactories;
        r["grow_quantile_histmaker"] = (ctx, _) => new QuantileHistMaker(ctx);
        r["grow_histmaker"] = (ctx, task) => new GlobalApproxUpdater(ctx, task);
        r["grow_colmaker"] = (ctx, _) => new ColMaker(ctx);
        r["prune"] = (ctx, _) => new TreePruner(ctx);
        r["refresh"] = (ctx, _) => new TreeRefresher(ctx);
    }
}
