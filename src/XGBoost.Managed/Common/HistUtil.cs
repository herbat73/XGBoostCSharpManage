// Port of src/common/hist_util.h/.cc and src/common/cache_manager.h/.cc.
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using XGBoost.Data;

namespace XGBoost.Common;

/// <summary>Cut points of every feature in CSC form: bin upper bounds (<c>HistogramCuts</c>).</summary>
public sealed class HistogramCuts
{
    private bool _hasCategorical;
    private float _maxCat = -1.0f;

    public HostDeviceVector<float> Values { get; private set; } = new();
    public HostDeviceVector<uint> Ptrs { get; private set; }

    public HistogramCuts(int nFeatures) => Ptrs = new HostDeviceVector<uint>(nFeatures + 1);

    public HistogramCuts(HistogramCuts that)
    {
        Values = new HostDeviceVector<float>(that.Values.ToArray(), true);
        Ptrs = new HostDeviceVector<uint>(that.Ptrs.ToArray(), true);
        _hasCategorical = that._hasCategorical;
        _maxCat = that._maxCat;
    }

    public HistogramCuts(float[] values, uint[] ptrs, bool hasCat, float maxCat)
    {
        Values = new HostDeviceVector<float>(values, true);
        Ptrs = new HostDeviceVector<uint>(ptrs, true);
        _hasCategorical = hasCat;
        _maxCat = maxCat;
    }

    public int FeatureBins(int feature) => (int)(Ptrs[feature + 1] - Ptrs[feature]);
    public int NumFeatures => Ptrs.Size - 1;
    public bool HasCategorical => _hasCategorical;
    public float MaxCategory => _maxCat;

    public void SetCategorical(bool hasCat, float maxCat)
    {
        _hasCategorical = hasCat;
        _maxCat = maxCat;
    }

    public int TotalBins => Values.Size;

    /// <summary>Index of the first cut strictly greater than <paramref name="value"/>, or the last bin.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int SearchBin(float value, int columnId, ReadOnlySpan<uint> ptrs, ReadOnlySpan<float> values)
    {
        var end = (int)ptrs[columnId + 1];
        var beg = (int)ptrs[columnId];
        var idx = beg + StdAlgo.UpperBound(values[beg..end], value);
        if (idx == end) idx--;
        return idx;
    }

    public int SearchBin(float value, int columnId) => SearchBin(value, columnId, Ptrs.ConstHostSpan, Values.ConstHostSpan);

    public int SearchBin(Entry e) => SearchBin(e.Fvalue, (int)e.Index);

    public static int SearchCatBin(float value, int fidx, ReadOnlySpan<uint> ptrs, ReadOnlySpan<float> vals)
    {
        var end = (int)ptrs[fidx + 1];
        var beg = (int)ptrs[fidx];
        var v = (float)Categorical.AsCat(value);
        var binIdx = beg + StdAlgo.LowerBound(vals[beg..end], v);
        if (binIdx == end) binIdx--;
        return binIdx;
    }

    public int SearchCatBin(float value, int fidx) => SearchCatBin(value, fidx, Ptrs.ConstHostSpan, Values.ConstHostSpan);

    public int SearchCatBin(Entry e) => SearchCatBin(e.Fvalue, (int)e.Index);

    public static float NumericBinValue(ReadOnlySpan<uint> ptrs, ReadOnlySpan<float> vals, int fidx, int binIdx)
    {
        var lower = (int)ptrs[fidx];
        if (binIdx == lower) return MathF.BitDecrement(vals[lower]);
        return vals[binIdx - 1];
    }

    public static float NumericBinLowerBound(ReadOnlySpan<uint> ptrs, ReadOnlySpan<float> vals, int fidx, long binIdx)
    {
        var lower = ptrs[fidx];
        if (binIdx == lower) return float.NegativeInfinity;
        return vals[(int)binIdx - 1];
    }

    public void Save(BinaryWriter fo)
    {
        fo.Write(Ptrs.Size);
        foreach (var p in Ptrs.ConstHostSpan) fo.Write(p);
        fo.Write(Values.Size);
        foreach (var v in Values.ConstHostSpan) fo.Write(v);
        fo.Write(_hasCategorical);
        fo.Write(_maxCat);
    }

    public static HistogramCuts Load(BinaryReader fi)
    {
        var np = fi.ReadInt32();
        var ptrs = new uint[np];
        for (var i = 0; i < np; i++) ptrs[i] = fi.ReadUInt32();
        var nv = fi.ReadInt32();
        var vals = new float[nv];
        for (var i = 0; i < nv; i++) vals[i] = fi.ReadSingle();
        var hasCat = fi.ReadBoolean();
        var maxCat = fi.ReadSingle();
        return new HistogramCuts(vals, ptrs, hasCat, maxCat);
    }
}

public enum BinTypeSize : byte
{
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 4,
}

/// <summary>Optionally compressed gradient index (<c>common::Index</c>).</summary>
public sealed class GHistIndex
{
    private byte[] _data = [];
    private int _byteSize;
    private uint[] _binOffset = [];

    public BinTypeSize BinTypeSize { get; private set; } = BinTypeSize.Uint8;

    public GHistIndex() { }

    public GHistIndex(byte[] data, int byteSize, BinTypeSize binSize)
    {
        _data = data;
        _byteSize = byteSize;
        BinTypeSize = binSize;
    }

    public byte[] Raw => _data;
    public int SizeBytes => _byteSize;

    public Span<T> Data<T>() where T : unmanaged => MemoryMarshal.Cast<byte, T>(_data.AsSpan(0, _byteSize));

    public uint this[long i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            uint v = BinTypeSize switch
            {
                BinTypeSize.Uint8 => _data[i],
                BinTypeSize.Uint16 => Unsafe.ReadUnaligned<ushort>(ref _data[i * 2]),
                _ => Unsafe.ReadUnaligned<uint>(ref _data[i * 4]),
            };
            return _binOffset.Length != 0 ? v + _binOffset[i % _binOffset.Length] : v;
        }
    }

    public ReadOnlySpan<uint> Offset => _binOffset;
    public bool HasOffset => _binOffset.Length != 0;
    public long Size => _byteSize / (int)BinTypeSize;

    public void SetBinOffset(ReadOnlySpan<uint> cutPtrs)
    {
        _binOffset = cutPtrs[..^1].ToArray();
    }

    /// <summary>Grows the storage to <paramref name="nBytes"/>, keeping existing content (external memory batches).</summary>
    public void Resize(int nBytes, BinTypeSize t)
    {
        Check.Ge(nBytes, _byteSize);
        if (nBytes > _data.Length) Array.Resize(ref _data, nBytes);
        _byteSize = nBytes;
        BinTypeSize = t;
    }

    public void SetBinOffsetArray(uint[] offset) => _binOffset = offset;
}

/// <summary>Histograms of multiple nodes, <c>HistCollection</c>.</summary>
public sealed class HistCollection
{
    private int _nbins;
    private int _nNodesAdded;
    private readonly List<GradientPairPrecise[]?> _data = [];
    private readonly List<int> _rowPtr = [];

    public int NBins => _nbins;

    public Span<GradientPairPrecise> this[int nid]
    {
        get
        {
            var id = _rowPtr[nid];
            Check.Ne(id, -1);
            return _data[id].AsSpan(0, _nbins);
        }
    }

    public GradientPairPrecise[] Array(int nid) => _data[_rowPtr[nid]]!;

    public bool RowExists(int nid) => nid < _rowPtr.Count && _rowPtr[nid] != -1;

    public void Init(int nTotalBins)
    {
        if (_nbins != nTotalBins)
        {
            _nbins = nTotalBins;
            _data.Clear();
        }
        _rowPtr.Clear();
        _nNodesAdded = 0;
    }

    public void AddHistRow(int nid)
    {
        while (nid >= _rowPtr.Count) _rowPtr.Add(-1);
        Check.Eq(_rowPtr[nid], -1);
        while (_data.Count < nid + 1) _data.Add(null);
        _rowPtr[nid] = _nNodesAdded;
        _nNodesAdded++;
    }

    public void AllocateData(int nid)
    {
        var id = _rowPtr[nid];
        if (_data[id] is null || _data[id]!.Length == 0) _data[id] = new GradientPairPrecise[_nbins];
    }
}

/// <summary>Thread-local histograms reduced into targeted node histograms, <c>ParallelGHistBuilder</c>.</summary>
public sealed class ParallelGHistBuilder
{
    private int _nbins;
    private int _nthreads;
    private int _nodes;
    private readonly HistCollection _histBuffer = new();
    private int[] _histWasUsed = [];
    private bool[] _threadsToNidsMap = [];
    private List<GradientPairPrecise[]> _targetedHists = [];
    private readonly Dictionary<(int, int), int> _tidNidToHist = [];

    public void Init(int nbins)
    {
        if (nbins != _nbins)
        {
            _histBuffer.Init(nbins);
            _nbins = nbins;
        }
    }

    public void Reset(int nthreads, int nodes, BlockedSpace2d space, List<GradientPairPrecise[]> targetedHists)
    {
        _histBuffer.Init(_nbins);
        _tidNidToHist.Clear();
        _targetedHists = targetedHists;
        Check.Eq(nodes, targetedHists.Count);
        _nodes = nodes;
        _nthreads = nthreads;
        MatchThreadsToNodes(space);
        AllocateAdditionalHistograms();
        MatchNodeNidPairToHist();
        _histWasUsed = new int[nthreads * _nodes];
    }

    public Span<GradientPairPrecise> GetInitializedHist(int tid, int nid)
    {
        Check.Lt(nid, _nodes);
        Check.Lt(tid, _nthreads);
        var idx = _tidNidToHist[(tid, nid)];
        if (idx >= 0) _histBuffer.AllocateData(idx);
        var hist = idx == -1 ? _targetedHists[nid].AsSpan(0, _nbins) : _histBuffer[idx];
        if (_histWasUsed[tid * _nodes + nid] == 0)
        {
            hist.Clear();
            _histWasUsed[tid * _nodes + nid] = 1;
        }
        return hist;
    }

    public void ReduceHist(int nid, int begin, int end)
    {
        Check.Gt(end, begin);
        Check.Lt(nid, _nodes);
        var dst = _targetedHists[nid].AsSpan(0, _nbins);
        var isUpdated = false;
        for (var tid = 0; tid < _nthreads; tid++)
        {
            if (_histWasUsed[tid * _nodes + nid] == 0) continue;
            isUpdated = true;
            var idx = _tidNidToHist[(tid, nid)];
            if (idx == -1) continue; // src is dst
            HistOps.IncrementHist(dst, _histBuffer[idx], begin, end);
        }
        if (!isUpdated) dst[begin..end].Clear();
    }

    private void MatchThreadsToNodes(BlockedSpace2d space)
    {
        var spaceSize = space.Size;
        var chunk = spaceSize / _nthreads + (spaceSize % _nthreads != 0 ? 1 : 0);
        _threadsToNidsMap = new bool[_nthreads * _nodes];
        for (var tid = 0; tid < _nthreads; tid++)
        {
            var begin = chunk * tid;
            var end = Math.Min(begin + chunk, spaceSize);
            if (begin < spaceSize)
            {
                var nidBegin = (int)space.GetFirstDimension(begin);
                var nidEnd = (int)space.GetFirstDimension(end - 1);
                for (var nid = nidBegin; nid <= nidEnd; nid++) _threadsToNidsMap[tid * _nodes + nid] = true;
            }
        }
    }

    private void AllocateAdditionalHistograms()
    {
        var additional = 0;
        for (var nid = 0; nid < _nodes; nid++)
        {
            var n = 0;
            for (var tid = 0; tid < _nthreads; tid++)
                if (_threadsToNidsMap[tid * _nodes + nid]) n++;
            additional += Math.Max(0, n - 1);
        }
        for (var i = 0; i < additional; i++) _histBuffer.AddHistRow(i);
    }

    private void MatchNodeNidPairToHist()
    {
        var additional = 0;
        for (var nid = 0; nid < _nodes; nid++)
        {
            var first = true;
            for (var tid = 0; tid < _nthreads; tid++)
            {
                if (!_threadsToNidsMap[tid * _nodes + nid]) continue;
                if (first)
                {
                    _tidNidToHist[(tid, nid)] = -1;
                    first = false;
                }
                else
                {
                    _tidNidToHist[(tid, nid)] = additional++;
                }
            }
        }
    }

    public int TotalBins => _nbins;
}

public static class HistOps
{
    public static void IncrementHist(Span<GradientPairPrecise> dst, ReadOnlySpan<GradientPairPrecise> add, int begin, int end)
    {
        var pdst = MemoryMarshal.Cast<GradientPairPrecise, double>(dst);
        var padd = MemoryMarshal.Cast<GradientPairPrecise, double>(add);
        for (var i = 2 * begin; i < 2 * end; i++) pdst[i] += padd[i];
    }

    public static void CopyHist(Span<GradientPairPrecise> dst, ReadOnlySpan<GradientPairPrecise> src, int begin, int end) =>
        src[begin..end].CopyTo(dst[begin..end]);

    public static void SubtractionHist(Span<GradientPairPrecise> dst, ReadOnlySpan<GradientPairPrecise> src1,
        ReadOnlySpan<GradientPairPrecise> src2, int begin, int end)
    {
        var pdst = MemoryMarshal.Cast<GradientPairPrecise, double>(dst);
        var p1 = MemoryMarshal.Cast<GradientPairPrecise, double>(src1);
        var p2 = MemoryMarshal.Cast<GradientPairPrecise, double>(src2);
        for (var i = 2 * begin; i < 2 * end; i++) pdst[i] = p1[i] - p2[i];
    }

    /// <summary>Binary search for the bin of a feature within a sparse row, <c>BinarySearchBin</c>.</summary>
    public static int BinarySearchBin(long begin, long end, GHistIndex data, uint fidxBegin, uint fidxEnd)
    {
        var previousMiddle = long.MaxValue;
        while (end != begin)
        {
            var middle = begin + (end - begin) / 2;
            if (middle == previousMiddle) break;
            previousMiddle = middle;
            var gidx = data[middle];
            if (gidx >= fidxBegin && gidx < fidxEnd) return (int)gidx;
            if (gidx < fidxBegin) begin = middle;
            else end = middle;
        }
        return -1;
    }

    /// <summary><c>SketchOnDMatrix</c>: builds cuts from the row (or sorted column) pages of <paramref name="m"/>.</summary>
    public static HistogramCuts SketchOnDMatrix(Context ctx, DMatrix m, int maxBins, bool useSorted = false, float[]? hessian = null)
    {
        var info = m.Info;
        var nThreads = ctx.Threads();
        var reduced = new long[info.NumCol];
        foreach (var page in m.GetRowBatches())
        {
            var entries = HostSketchContainer.CalcColumnSize(new SparsePageAdapterBatch(page.GetView()), (int)info.NumCol, nThreads, static _ => true);
            Check.Eq((long)entries.Length, info.NumCol);
            for (var i = 0; i < entries.Length; i++) reduced[i] += entries[i];
        }
        var container = new HostSketchContainer(ctx, maxBins, info.FeatureTypes.ConstHostSpan, reduced, HostSketchContainer.UseGroup(info));
        if (!useSorted)
        {
            foreach (var page in m.GetRowBatches()) container.PushRowPage(page, info, hessian);
        }
        else
        {
            foreach (var page in m.GetSortedColumnBatches(ctx)) container.PushColPage(page, info, hessian ?? []);
        }
        return container.MakeCuts(ctx, info);
    }

    // ---- histogram building kernels ---------------------------------------------------------

    private const int PrefetchOffset = 10;
    private const int CacheLineSize = 64;
    private static readonly int NoPrefetchSizeConst = PrefetchOffset + CacheLineSize / sizeof(ulong);

    /// <summary><c>BuildHist&lt;any_missing&gt;</c>: accumulates gradients of <paramref name="rowIndices"/> into <paramref name="hist"/>.</summary>
    public static void BuildHist(bool anyMissing, ReadOnlySpan<GradientPair> gpair, ReadOnlySpan<long> rowIndices, GHistIndexMatrix gmat,
        Span<GradientPairPrecise> hist, bool readByColumn)
    {
        var firstPage = gmat.BaseRowId == 0;
        if (readByColumn)
        {
            ColsWiseBuildHistKernel(anyMissing, firstPage, gpair, rowIndices, gmat, hist);
            return;
        }
        if (rowIndices.IsEmpty) return;
        var nrows = rowIndices.Length;
        var noPrefetchSize = Math.Min(nrows, NoPrefetchSizeConst);
        var contiguous = rowIndices[nrows - 1] - rowIndices[0] == nrows - 1;
        if (contiguous)
        {
            RowsWiseBuildHistKernel(anyMissing, firstPage, gpair, rowIndices, gmat, hist);
        }
        else
        {
            // Prefetching does not change results; the split into two calls is kept for fidelity since
            // the sparse tile path decision is made per call.
            var span1 = rowIndices[..(nrows - noPrefetchSize)];
            if (!span1.IsEmpty) RowsWiseBuildHistKernel(anyMissing, firstPage, gpair, span1, gmat, hist);
            var span2 = rowIndices[(nrows - noPrefetchSize)..];
            if (!span2.IsEmpty) RowsWiseBuildHistKernel(anyMissing, firstPage, gpair, span2, gmat, hist);
        }
    }

    private static void RowsWiseBuildHistKernel(bool anyMissing, bool firstPage, ReadOnlySpan<GradientPair> gpair,
        ReadOnlySpan<long> rid, GHistIndexMatrix gmat, Span<GradientPairPrecise> hist)
    {
        if (anyMissing && BuildSparseHistByBlocks(firstPage, gpair, rid, gmat, hist)) return;
        switch (gmat.Index.BinTypeSize)
        {
            case BinTypeSize.Uint8:
                RowsWiseKernel<byte>(anyMissing, firstPage, gpair, rid, gmat, hist);
                break;
            case BinTypeSize.Uint16:
                RowsWiseKernel<ushort>(anyMissing, firstPage, gpair, rid, gmat, hist);
                break;
            default:
                RowsWiseKernel<uint>(anyMissing, firstPage, gpair, rid, gmat, hist);
                break;
        }
    }

    private static void RowsWiseKernel<TBin>(bool anyMissing, bool firstPage, ReadOnlySpan<GradientPair> gpair,
        ReadOnlySpan<long> rid, GHistIndexMatrix gmat, Span<GradientPairPrecise> hist) where TBin : unmanaged, IConvertible
    {
        var size = rid.Length;
        var pgpair = MemoryMarshal.Cast<GradientPair, float>(gpair);
        var gradientIndex = gmat.Index.Data<TBin>();
        var rowPtr = gmat.RowPtr;
        var baseRowId = gmat.BaseRowId;
        var offsets = gmat.Index.Offset;
        if (anyMissing) Check.That(offsets.IsEmpty);
        else Check.That(!offsets.IsEmpty);

        long GetRowPtr(long r) => firstPage ? rowPtr[r] : rowPtr[r - baseRowId];
        long GetRid(long r) => firstPage ? r : r - baseRowId;

        Check.Ne(size, 0);
        var nFeatures = GetRowPtr(rid[0] + 1) - GetRowPtr(rid[0]);
        var histData = MemoryMarshal.Cast<GradientPairPrecise, double>(hist);
        for (var i = 0; i < size; i++)
        {
            var icolStart = anyMissing ? GetRowPtr(rid[i]) : GetRid(rid[i]) * nFeatures;
            var icolEnd = anyMissing ? GetRowPtr(rid[i] + 1) : icolStart + nFeatures;
            var rowSize = icolEnd - icolStart;
            var idxGh = 2 * rid[i];
            var g = pgpair[(int)idxGh];
            var h = pgpair[(int)idxGh + 1];
            var local = gradientIndex[(int)icolStart..];
            for (var j = 0; j < rowSize; j++)
            {
                var bin = ToU32(local[j]) + (anyMissing ? 0u : offsets[j]);
                var idxBin = (int)(2 * bin);
                histData[idxBin] += g;
                histData[idxBin + 1] += h;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ToU32<TBin>(TBin v) where TBin : unmanaged
    {
        if (typeof(TBin) == typeof(byte)) return (byte)(object)v;
        if (typeof(TBin) == typeof(ushort)) return (ushort)(object)v;
        return (uint)(object)v;
    }

    private static bool BuildSparseHistByBlocks(bool firstPage, ReadOnlySpan<GradientPair> gpair, ReadOnlySpan<long> rid,
        GHistIndexMatrix gmat, Span<GradientPairPrecise> hist)
    {
        const int colBlockSize = 32;
        var cutPtrs = gmat.Cut.Ptrs.ConstHostSpan;
        var nFeatures = cutPtrs.Length - 1;
        var totalHistBins = cutPtrs[^1];
        var histBytes = 2L * sizeof(double) * totalHistBins;
        var l3PerThread = (double)CacheManager.Instance.L3Size / Math.Max(1, Threading.MaxThreads);
        var usableCache = 0.8 * (CacheManager.Instance.L2Size + l3PerThread);
        if (histBytes <= usableCache) return false;

        var size = rid.Length;
        var rowPtr = gmat.RowPtr;
        var baseRowId = gmat.BaseRowId;
        long GetRowPtr(long r) => firstPage ? rowPtr[r] : rowPtr[r - baseRowId];

        long nnz = 0;
        for (var i = 0; i < size; i++) nnz += GetRowPtr(rid[i] + 1) - GetRowPtr(rid[i]);
        var nBlocks = (nFeatures + colBlockSize - 1) / colBlockSize;
        var tileOverhead = (long)size * nBlocks + totalHistBins;
        if (nnz <= 2 * tileOverhead) return false;

        var maxBlockBins = 0;
        for (var jj = 0; jj < nFeatures; jj += colBlockSize)
        {
            var jjEnd = Math.Min(jj + colBlockSize, nFeatures);
            maxBlockBins = Math.Max(maxBlockBins, (int)(cutPtrs[jjEnd] - cutPtrs[jj]));
        }
        var tlBuf = new double[maxBlockBins * 2];
        var rowStarts = new long[size];
        var rowSizes = new long[size];
        var cursors = new long[size];
        for (var i = 0; i < size; i++)
        {
            rowStarts[i] = GetRowPtr(rid[i]);
            rowSizes[i] = GetRowPtr(rid[i] + 1) - rowStarts[i];
        }
        var pg = MemoryMarshal.Cast<GradientPair, float>(gpair);
        var index = gmat.Index;
        var histData = MemoryMarshal.Cast<GradientPairPrecise, double>(hist);
        for (var cidBegin = 0; cidBegin < nFeatures; cidBegin += colBlockSize)
        {
            var cidEnd = Math.Min(cidBegin + colBlockSize, nFeatures);
            var binLo = cutPtrs[cidBegin];
            var binHi = cutPtrs[cidEnd];
            var blockNBins = (int)(binHi - binLo);
            Array.Clear(tlBuf, 0, blockNBins * 2);
            for (var i = 0; i < size; i++)
            {
                var j = cursors[i];
                var idxGh = (int)(2 * rid[i]);
                var g = pg[idxGh];
                var h = pg[idxGh + 1];
                while (j < rowSizes[i] && index[rowStarts[i] + j] < binLo) j++;
                while (j < rowSizes[i] && index[rowStarts[i] + j] < binHi)
                {
                    var localBin = (int)(2 * (index[rowStarts[i] + j] - binLo));
                    tlBuf[localBin] += g;
                    tlBuf[localBin + 1] += h;
                    j++;
                }
                cursors[i] = j;
            }
            var dst = (int)(2 * binLo);
            for (var j = 0; j < blockNBins * 2; j++) histData[dst + j] += tlBuf[j];
        }
        return true;
    }

    private static void ColsWiseBuildHistKernel(bool anyMissing, bool firstPage, ReadOnlySpan<GradientPair> gpair,
        ReadOnlySpan<long> rid, GHistIndexMatrix gmat, Span<GradientPairPrecise> hist)
    {
        var size = rid.Length;
        var pgh = MemoryMarshal.Cast<GradientPair, float>(gpair);
        var index = gmat.Index;
        var rowPtr = gmat.RowPtr;
        var baseRowId = gmat.BaseRowId;
        var offsets = gmat.Index.Offset;
        long GetRowPtr(long r) => firstPage ? rowPtr[r] : rowPtr[r - baseRowId];
        long GetRid(long r) => firstPage ? r : r - baseRowId;

        var cutPtrs = gmat.Cut.Ptrs.ConstHostSpan;
        var nFeatures = cutPtrs.Length - 1;
        var nColumns = nFeatures;
        var histData = MemoryMarshal.Cast<GradientPairPrecise, double>(hist);
        const int colBlockSize = 32;
        var maxBlockBins = 0;
        for (var jj = 0; jj < nColumns; jj += colBlockSize)
        {
            var jjEnd = Math.Min(jj + colBlockSize, nColumns);
            maxBlockBins = Math.Max(maxBlockBins, (int)(cutPtrs[jjEnd] - cutPtrs[jj]));
        }
        var tl = new double[maxBlockBins * 2];
        // Raw stored (compressed) values: index[] adds the offset back, so read the raw bytes instead.
        var raw = gmat.Index;
        for (var cidBegin = 0; cidBegin < nColumns; cidBegin += colBlockSize)
        {
            var cidEnd = Math.Min(cidBegin + colBlockSize, nColumns);
            var chunkBinBegin = cutPtrs[cidBegin];
            var chunkBinEnd = cutPtrs[cidEnd];
            var chunkNBins = (int)(chunkBinEnd - chunkBinBegin);
            Array.Clear(tl, 0, chunkNBins * 2);
            for (var i = 0; i < size; i++)
            {
                var rowId = rid[i];
                var icolStart = anyMissing ? GetRowPtr(rowId) : GetRid(rowId) * nFeatures;
                var icolEnd = anyMissing ? GetRowPtr(rid[i] + 1) : icolStart + nFeatures;
                var rowSize = icolEnd - icolStart;
                var idxGh = (int)(2 * rowId);
                var g = pgh[idxGh];
                var h = pgh[idxGh + 1];
                for (var cid = cidBegin; cid < cidEnd; cid++)
                {
                    if (cid < rowSize)
                    {
                        // index[] already adds the dense offset.
                        var globalBin = raw[icolStart + cid];
                        var localBin = (int)(2 * (globalBin - chunkBinBegin));
                        tl[localBin] += g;
                        tl[localBin + 1] += h;
                    }
                }
            }
            var dst = (int)(2 * chunkBinBegin);
            for (var j = 0; j < chunkNBins * 2; j++) histData[dst + j] += tl[j];
        }
        _ = offsets;
        _ = index;
    }
}

/// <summary>CPU cache sizes, <c>CacheManager</c>. Defaults on Windows (as with MSVC builds), CPUID elsewhere.</summary>
public sealed class CacheManager
{
    public static readonly CacheManager Instance = new();

    private const long DefaultL1 = 32 * 1024;
    private const long DefaultL2 = 1024 * 1024;
    private const long DefaultL3 = 0;
    private readonly long[] _sizes = [-1, -1, -1, -1];

    private CacheManager()
    {
        if (OperatingSystem.IsWindows() || !System.Runtime.Intrinsics.X86.X86Base.IsSupported)
        {
            _sizes[0] = DefaultL1;
            _sizes[1] = DefaultL2;
            _sizes[2] = DefaultL3;
            return;
        }
        try
        {
            DetectX86();
        }
        catch
        {
            _sizes[0] = DefaultL1;
            _sizes[1] = DefaultL2;
            _sizes[2] = DefaultL3;
        }
    }

    private void DetectX86()
    {
        var (_, ebx0, _, _) = System.Runtime.Intrinsics.X86.X86Base.CpuId(0, 0);
        var isAmd = (uint)ebx0 == 0x68747541;
        var leaf = isAmd ? unchecked((int)0x8000001DU) : 0x4;
        int cacheNum = 0, idx = 0;
        while (idx < 4)
        {
            var (eax, ebx, ecx, _) = System.Runtime.Intrinsics.X86.X86Base.CpuId(leaf, cacheNum++);
            var type = eax & 0x1f;
            if (type == 0) break;
            if (type == 2) continue;
            long sets = (uint)ecx + 1L;
            long lineSize = (ebx & 0x7ff) + 1;
            long partitions = ((ebx & 0x3ff800) >> 11) + 1;
            long ways = (int)(((uint)ebx & 0xffc00000U) >> 22) + 1;
            _sizes[idx++] = ways * partitions * lineSize * sets;
        }
    }

    public long L1Size => _sizes[0] != -1 ? _sizes[0] : DefaultL1;
    public long L2Size => _sizes[1] != -1 ? _sizes[1] : DefaultL2;
    public long L3Size => _sizes[2] != -1 ? _sizes[2] : DefaultL3;
}
