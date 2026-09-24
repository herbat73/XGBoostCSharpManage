// Ports of src/tree/hist/hist_cache.h, histogram.h/.cc.
namespace XGBoost.Tree;

/// <summary><c>BoundedHistCollection</c>: node histograms with a bounded cache.</summary>
public sealed class BoundedHistCollection
{
    private readonly SortedDictionary<int, GradientPairPrecise[]> _nodeMap = [];
    private int _nTotalBins;
    private ulong _maxCachedNodes;
    private bool _hasExceeded;

    public GradientPairPrecise[] this[int nidx] => _nodeMap[nidx];

    public void Reset(int nTotalBins, ulong nCachedNodes)
    {
        _nTotalBins = nTotalBins;
        _maxCachedNodes = nCachedNodes;
        Clear(false);
    }

    public void Clear(bool exceeded)
    {
        _nodeMap.Clear();
        _hasExceeded = exceeded;
    }

    public bool CanHost(IReadOnlyCollection<int> nodesToBuild, IReadOnlyCollection<int> nodesToSub) =>
        (ulong)(nodesToBuild.Count + nodesToSub.Count + _nodeMap.Count) <= _maxCachedNodes;

    public void AllocateHistograms(IEnumerable<int> nodesToBuild, IEnumerable<int> nodesToSub)
    {
        foreach (var nidx in nodesToBuild) _nodeMap[nidx] = new GradientPairPrecise[_nTotalBins];
        foreach (var nidx in nodesToSub) _nodeMap[nidx] = new GradientPairPrecise[_nTotalBins];
    }

    public bool HasExceeded => _hasExceeded;
    public bool HistogramExists(int nidx) => _nodeMap.ContainsKey(nidx);
}

/// <summary><c>HistogramBuilder</c>: histograms for one target.</summary>
public sealed class HistogramBuilder
{
    private readonly BoundedHistCollection _hist = new();
    private readonly ParallelGHistBuilder _buffer = new();
    private int _nThreads = -1;

    public BoundedHistCollection Histogram => _hist;

    public void Reset(Context ctx, int totalBins, HistMakerTrainParam param)
    {
        _nThreads = ctx.Threads();
        _hist.Reset(totalBins, param.MaxCachedHistNodes());
        _buffer.Init(totalBins);
    }

    private void BuildLocalHistograms(bool anyMissing, BlockedSpace2d space, GHistIndexMatrix gidx, List<int> nodesToBuild,
        RowSetCollection rowSet, GradientPair[] gpair, bool readByColumn)
    {
        Threading.ParallelFor2d(space, _nThreads, (nidInSet, r) =>
        {
            var tid = Threading.ThreadNum;
            var nidx = nodesToBuild[(int)nidInSet];
            var elem = rowSet[nidx];
            var start = Math.Min(r.Begin, elem.Size);
            var end = Math.Min(r.End, elem.Size);
            var hist = _buffer.GetInitializedHist(tid, (int)nidInSet);
            if (end - start != 0)
            {
                var rid = rowSet.Data.AsSpan((int)(elem.Begin + start), (int)(end - start));
                HistOps.BuildHist(anyMissing, gpair, rid, gidx, hist, readByColumn);
            }
        });
    }

    /// <summary>Allocates histograms; rearranges nodes when the cache limit is reached.</summary>
    public void AddHistRows(ITreeView tree, List<int> nodesToBuild, List<int> nodesToSub, bool rearrange)
    {
        var canHost = _hist.CanHost(nodesToBuild, nodesToSub);
        var cacheIsValid = canHost && !_hist.HasExceeded;
        if (!canHost) _hist.Clear(true);
        if (!rearrange || cacheIsValid)
        {
            _hist.AllocateHistograms(nodesToBuild, nodesToSub);
            if (rearrange) Check.That(!_hist.HasExceeded);
            return;
        }
        var canSubtract = new List<int>();
        foreach (var v in nodesToSub)
        {
            if (_hist.HistogramExists(tree.Parent(v))) canSubtract.Add(v);
            else nodesToBuild.Add(v);
        }
        nodesToSub.Clear();
        nodesToSub.AddRange(canSubtract);
        _hist.AllocateHistograms(nodesToBuild, nodesToSub);
    }

    public void BuildHist(int pageIdx, BlockedSpace2d space, GHistIndexMatrix gidx, RowSetCollection rowSet, List<int> nodesToBuild,
        GradientPair[] gpair, bool readByColumn)
    {
        if (pageIdx == 0)
        {
            var targetHists = new List<GradientPairPrecise[]>(nodesToBuild.Count);
            foreach (var nidx in nodesToBuild) targetHists.Add(_hist[nidx]);
            _buffer.Reset(_nThreads, nodesToBuild.Count, space, targetHists);
        }
        BuildLocalHistograms(!gidx.IsDense, space, gidx, nodesToBuild, rowSet, gpair, readByColumn);
    }

    public void SyncHistogram(ITreeView tree, List<int> nodesToBuild, List<int> nodesToTrick)
    {
        var nTotalBins = _buffer.TotalBins;
        var space = new BlockedSpace2d(nodesToBuild.Count, _ => nTotalBins, 1024);
        Threading.ParallelFor2d(space, _nThreads, (node, r) => _buffer.ReduceHist((int)node, (int)r.Begin, (int)r.End));
        // Distributed allreduce: identity with a single worker.
        var subspace = nodesToTrick.Count == nodesToBuild.Count ? space : new BlockedSpace2d(nodesToTrick.Count, _ => nTotalBins, 1024);
        Threading.ParallelFor2d(subspace, _nThreads, (nidxInSet, r) =>
        {
            var subtractionNidx = nodesToTrick[(int)nidxInSet];
            var parentId = tree.Parent(subtractionNidx);
            var siblingNidx = tree.IsLeftChild(subtractionNidx) ? tree.RightChild(parentId) : tree.LeftChild(parentId);
            HistOps.SubtractionHist(_hist[subtractionNidx], _hist[parentId], _hist[siblingNidx], (int)r.Begin, (int)r.End);
        });
    }
}

/// <summary><c>MultiHistogramBuilder</c>: one histogram builder per target.</summary>
public sealed class MultiHistogramBuilder
{
    private readonly List<HistogramBuilder> _targetBuilders = [];
    private Context _ctx = null!;
    private readonly CacheManager _cacheManager = CacheManager.Instance;

    private bool ReadByColumn(GHistIndexMatrix gidx, bool forceReadByColumn)
    {
        if (forceReadByColumn) return true;
        var nbins = gidx.Cut.Ptrs[gidx.Cut.Ptrs.Size - 1];
        var histSize = (ulong)(2 * sizeof(double)) * nbins;
        var l3PerThread = (double)_cacheManager.L3Size / _ctx.Threads();
        var usableCacheSize = 0.8 * (_cacheManager.L2Size + l3PerThread);
        var histFitToL2 = usableCacheSize > histSize;
        return !histFitToL2 && gidx.IsDense;
    }

    public static BlockedSpace2d ConstructHistSpace(IReadOnlyList<CommonRowPartitioner> partitioners, List<int> nodesToBuild,
        GHistIndexMatrix gidx, long l1Size, int maxBin, bool readByColumn)
    {
        var partitionSize = new long[nodesToBuild.Count];
        foreach (var partition in partitioners)
        {
            var k = 0;
            foreach (var nidx in nodesToBuild)
            {
                var nRows = partition.Partitions[nidx].Size;
                partitionSize[k] = Math.Max(partitionSize[k], nRows);
                k++;
            }
        }
        const long sizeofGradientPair = 8, sizeofSizeT = 8, sizeofGpp = 16;
        var l1RowFootPrint = sizeofGradientPair + 3 * sizeofSizeT;
        var usableL1Size = 0.8 * l1Size;
        ulong spaceInL1ForRows;
        if (readByColumn)
        {
            var histColSize = (ulong)(2 * sizeofGpp * maxBin);
            var histColFitToL1 = histColSize < usableL1Size;
            spaceInL1ForRows = (ulong)(usableL1Size - (histColFitToL1 ? histColSize : 0));
        }
        else
        {
            ulong nBins = gidx.Cut.Ptrs[gidx.Cut.Ptrs.Size - 1];
            var nColumns = (ulong)(gidx.Cut.Ptrs.Size - 1);
            var anyMissing = !gidx.IsDense;
            var histSize = (ulong)(2 * sizeofGpp) * nBins;
            var offsetsSize = anyMissing ? 0 : nColumns * 4;
            l1RowFootPrint += sizeofGradientPair;
            var idxBinSize = nColumns * 4;
            var histFitToL1 = histSize + offsetsSize + idxBinSize < usableL1Size;
            var occupiedSpace = (histFitToL1 ? histSize : 0) + offsetsSize + idxBinSize;
            spaceInL1ForRows = usableL1Size > occupiedSpace ? (ulong)(usableL1Size - occupiedSpace) : 0;
        }
        var blockSize = (long)(spaceInL1ForRows / (ulong)l1RowFootPrint);
        const long minBlockSize = 64 / sizeofGradientPair;
        blockSize = Math.Max(minBlockSize, blockSize);
        return new BlockedSpace2d(nodesToBuild.Count, i => partitionSize[i], blockSize);
    }

    private static GradientPair[] TargetGpair(TensorView<GradientPair> gpair, long t)
    {
        var n = gpair.Shape(0);
        var res = new GradientPair[n];
        for (long i = 0; i < n; ++i) res[i] = gpair[i, t];
        return res;
    }

    public void BuildRootHist(DMatrix fmat, ITreeView tree, IReadOnlyList<CommonRowPartitioner> partitioners, TensorView<GradientPair> gpair,
        int bestNid, BatchParam param, bool forceReadByColumn = false)
    {
        var nTargets = gpair.Shape(1);
        Check.Eq(fmat.Info.NumRow, gpair.Shape(0));
        Check.Eq((long)_targetBuilders.Count, nTargets);
        var nodes = new List<int> { bestNid };
        var dummySub = new List<int>();
        for (var t = 0; t < nTargets; ++t) _targetBuilders[t].AddHistRows(tree, nodes, dummySub, false);
        Check.That(dummySub.Count == 0);
        var pageIdx = 0;
        var tGpairs = new GradientPair[nTargets][];
        for (var t = 0; t < nTargets; ++t) tGpairs[t] = TargetGpair(gpair, t);
        foreach (var gidx in fmat.GetGradientIndex(_ctx, param))
        {
            var readByColumn = ReadByColumn(gidx, forceReadByColumn);
            var space = ConstructHistSpace(partitioners, nodes, gidx, _cacheManager.L1Size, param.MaxBin, readByColumn);
            for (var t = 0; t < nTargets; ++t)
                _targetBuilders[t].BuildHist(pageIdx, space, gidx, partitioners[pageIdx].Partitions, nodes, tGpairs[t], readByColumn);
            ++pageIdx;
        }
        for (var t = 0; t < nTargets; ++t) _targetBuilders[t].SyncHistogram(tree, nodes, dummySub);
    }

    public void BuildHistLeftRight(DMatrix fmat, ITreeView tree, IReadOnlyList<CommonRowPartitioner> partitioners, List<int> nodesToBuild,
        List<int> nodesToSub, TensorView<GradientPair> gpair, BatchParam param, bool forceReadByColumn = false)
    {
        var nCandidates = nodesToBuild.Count;
        _targetBuilders[0].AddHistRows(tree, nodesToBuild, nodesToSub, true);
        Check.Ge(nodesToBuild.Count, nodesToSub.Count);
        Check.Eq(nodesToSub.Count + nodesToBuild.Count, nCandidates * 2);
        for (var t = 1; t < _targetBuilders.Count; ++t) _targetBuilders[t].AddHistRows(tree, nodesToBuild, nodesToSub, false);
        var nTargets = gpair.Shape(1);
        var tGpairs = new GradientPair[nTargets][];
        for (var t = 0; t < nTargets; ++t) tGpairs[t] = TargetGpair(gpair, t);
        var pageIdx = 0;
        foreach (var page in fmat.GetGradientIndex(_ctx, param))
        {
            var readByColumn = ReadByColumn(page, forceReadByColumn);
            var space = ConstructHistSpace(partitioners, nodesToBuild, page, _cacheManager.L1Size, param.MaxBin, readByColumn);
            for (var t = 0; t < nTargets; ++t)
            {
                Check.Eq(gpair.Shape(0), fmat.Info.NumRow);
                _targetBuilders[t].BuildHist(pageIdx, space, page, partitioners[pageIdx].Partitions, nodesToBuild, tGpairs[t], readByColumn);
            }
            pageIdx++;
        }
        for (var t = 0; t < nTargets; ++t) _targetBuilders[t].SyncHistogram(tree, nodesToBuild, nodesToSub);
    }

    /// <summary><c>AssignNodes</c> for scalar trees: build the child with fewer hessian, subtract the other.</summary>
    public static void AssignNodes(ITreeView tree, IReadOnlyList<CPUExpandEntry> candidates, List<int> nodesToBuild, List<int> nodesToSub)
    {
        nodesToBuild.Clear();
        nodesToSub.Clear();
        foreach (var c in candidates)
        {
            var leftNidx = tree.LeftChild(c.Nid);
            var rightNidx = tree.RightChild(c.Nid);
            var fewerRight = c.Split.RightSum.GetHess() < c.Split.LeftSum.GetHess();
            nodesToBuild.Add(fewerRight ? rightNidx : leftNidx);
            nodesToSub.Add(fewerRight ? leftNidx : rightNidx);
        }
    }

    public static void AssignNodes(ITreeView tree, IReadOnlyList<MultiExpandEntry> candidates, List<int> nodesToBuild, List<int> nodesToSub)
    {
        nodesToBuild.Clear();
        nodesToSub.Clear();
        foreach (var c in candidates)
        {
            var leftNidx = tree.LeftChild(c.Nid);
            var rightNidx = tree.RightChild(c.Nid);
            var leftSum = 0.0;
            foreach (var g in c.Split.LeftSum) leftSum += g.Hess;
            var rightSum = 0.0;
            foreach (var g in c.Split.RightSum) rightSum += g.Hess;
            var fewerRight = rightSum < leftSum;
            nodesToBuild.Add(fewerRight ? rightNidx : leftNidx);
            nodesToSub.Add(fewerRight ? leftNidx : rightNidx);
        }
    }

    public BoundedHistCollection Histogram(int t) => _targetBuilders[t].Histogram;
    public int NumTargets => _targetBuilders.Count;

    public void Reset(Context ctx, int totalBins, int nTargets, HistMakerTrainParam param)
    {
        _ctx = ctx;
        while (_targetBuilders.Count < nTargets) _targetBuilders.Add(new HistogramBuilder());
        if (_targetBuilders.Count > nTargets) _targetBuilders.RemoveRange(nTargets, _targetBuilders.Count - nTargets);
        Check.Ge(nTargets, 1);
        foreach (var v in _targetBuilders) v.Reset(ctx, totalBins, param);
    }
}
