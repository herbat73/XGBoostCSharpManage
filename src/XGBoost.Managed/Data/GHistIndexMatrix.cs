// Port of src/data/gradient_index.h/.cc: the quantised (binned) CSR matrix used by hist/approx.
using XGBoost.Common;

namespace XGBoost.Data;

public sealed class GHistIndexMatrix
{
    /// <summary>Row pointer to rows by element position.</summary>
    public long[] RowPtr = [];

    /// <summary>The histogram index.</summary>
    public GHistIndex Index = new();

    /// <summary>Hit count of each bin, used for constructing the column matrix.</summary>
    public long[] HitCount = [];

    public HistogramCuts Cut = new(0);
    public int MaxNumericBinsPerFeat;
    public long BaseRowId;

    private ColumnMatrix? _columns;
    private long[] _hitCountTloc = [];
    private bool _isDense;

    public int MaxNumBinPerFeat => Math.Max((int)(Cut.MaxCategory + 1), MaxNumericBinsPerFeat);

    public GHistIndexMatrix() => _columns = new ColumnMatrix();

    /// <summary>Constructor for SimpleDMatrix.</summary>
    public GHistIndexMatrix(Context ctx, DMatrix fmat, int maxBinsPerFeat, double sparseThresh, bool sortedSketch, float[] hess)
    {
        MaxNumericBinsPerFeat = maxBinsPerFeat;
        Check.That(fmat.SingleColBlock);
        Cut = HistOps.SketchOnDMatrix(ctx, fmat, maxBinsPerFeat, sortedSketch, hess);
        var nbins = Cut.Ptrs[Cut.Ptrs.Size - 1];
        HitCount = new long[nbins];
        _hitCountTloc = new long[ctx.Threads() * nbins];
        long newSize = 1;
        foreach (var batch in fmat.GetRowBatches()) newSize += batch.Size;
        RowPtr = new long[newSize];
        _isDense = fmat.IsDenseMatrix;
        var ft = fmat.Info.FeatureTypes.ToArray();
        foreach (var batch in fmat.GetRowBatches()) PushBatch(ctx, batch, ft);
        _columns = new ColumnMatrix();
        if (hess.Length == 0 && !double.IsNaN(sparseThresh))
        {
            Check.That(!sortedSketch);
            foreach (var page in fmat.GetRowBatches()) _columns.InitFromSparse(page, this, sparseThresh, ctx.Threads());
        }
    }

    /// <summary>Constructor for QuantileDMatrix: prepares for <see cref="PushAdapterBatch{TBatch}"/>.</summary>
    public GHistIndexMatrix(MetaInfo info, HistogramCuts cuts, int maxBinPerFeat)
    {
        RowPtr = new long[info.NumRow + 1];
        HitCount = new long[cuts.TotalBins];
        Cut = cuts;
        MaxNumericBinsPerFeat = maxBinPerFeat;
        _isDense = info.IsDense;
    }

    /// <summary>Constructor for the external memory QuantileDMatrix.</summary>
    public GHistIndexMatrix(long nSamples, long baseRowId, HistogramCuts cuts, int maxBinPerFeat, bool isDense)
    {
        RowPtr = new long[nSamples + 1];
        HitCount = new long[cuts.TotalBins];
        Cut = cuts;
        MaxNumericBinsPerFeat = maxBinPerFeat;
        BaseRowId = baseRowId;
        _isDense = isDense;
    }

    /// <summary>Constructor for an external memory page.</summary>
    public GHistIndexMatrix(Context ctx, SparsePage batch, FeatureType[] ft, HistogramCuts cuts, int maxBinsPerFeat, bool isDense,
        double sparseThresh)
    {
        Cut = cuts;
        MaxNumericBinsPerFeat = maxBinsPerFeat;
        BaseRowId = batch.BaseRowId;
        _isDense = isDense;
        RowPtr = new long[batch.Size + 1];
        var nbins = Cut.Ptrs[Cut.Ptrs.Size - 1];
        HitCount = new long[nbins];
        var nThreads = ctx.Threads();
        _hitCountTloc = new long[nThreads * nbins];
        PushBatch(ctx, batch, ft);
        _columns = new ColumnMatrix();
        if (!double.IsNaN(sparseThresh)) _columns.InitFromSparse(batch, this, sparseThresh, nThreads);
        if (!_isDense)
        {
            SortRowBins(batch.Size, nThreads, 0);
        }
    }

    private void SortRowBins(long batchSize, int nThreads, long rbegin)
    {
        var raw = Index.Raw;
        Threading.ParallelFor(batchSize, nThreads, i =>
        {
            var begin = (int)RowPtr[rbegin + i];
            var end = (int)RowPtr[rbegin + i + 1];
            var span = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(raw.AsSpan()).Slice(begin, end - begin);
            StdAlgo.Sort(span);
        });
    }

    public bool IsDense => _isDense;
    public void SetDense(bool isDense) => _isDense = isDense;
    public long RowIdx(long gridx) => RowPtr[gridx - BaseRowId];
    public long Size => RowPtr.Length == 0 ? 0 : RowPtr.Length - 1;
    public int Features => Cut.NumFeatures;
    public HistogramCuts Cuts => Cut;

    public ColumnMatrix Transpose()
    {
        Check.That(_columns is not null);
        return _columns!;
    }

    private void PushBatch(Context ctx, SparsePage batch, FeatureType[] ft)
    {
        var page = batch.GetView();
        var counts = new long[page.Size];
        for (long r = 0; r < page.Size; r++) counts[r] = page.Offset[r + 1] - page.Offset[r];
        Numeric.PartialSum(ctx.Threads(), counts, 0, RowPtr, 0);
        PushBatchImpl(ctx, new SparsePageAdapterBatch(page), 0, static _ => true, ft);
    }

    /// <summary>Pushes an adapter batch for the QuantileDMatrix path.</summary>
    public void PushAdapterBatch<TBatch>(Context ctx, long rbegin, long prevSum, TBatch batch, float missing, FeatureType[] ft,
        double sparseThresh, long nSamplesTotal) where TBatch : IAdapterBatch
    {
        var nBinsTotal = Cut.TotalBins;
        _hitCountTloc = new long[ctx.Threads() * nBinsTotal];
        var nThreads = ctx.Threads();
        var isValid = new IsValidFunctor(missing);
        var validCounts = GetRowCounts(batch, isValid, nThreads);
        Numeric.PartialSum(nThreads, validCounts, prevSum, RowPtr, rbegin);
        PushBatchImpl(ctx, batch, rbegin, v => isValid.Valid(v), ft);
        if (rbegin + batch.Size == nSamplesTotal) ResizeColumns(sparseThresh);
    }

    public void PushAdapterBatchColumns<TBatch>(Context ctx, TBatch batch, float missing, long rbegin) where TBatch : IAdapterBatch
    {
        Check.That(_columns is not null);
        _columns!.PushBatch(ctx.Threads(), batch, missing, this, rbegin);
        if (!_isDense)
        {
            var nThreads = ctx.Threads();
            var batchThreads = (int)Math.Max(1, Math.Min(batch.Size, nThreads));
            SortRowBins(batch.Size, batchThreads, rbegin);
        }
    }

    private static long[] GetRowCounts<TBatch>(TBatch batch, IsValidFunctor isValid, int nThreads) where TBatch : IAdapterBatch
    {
        var validCounts = new long[batch.Size];
        Threading.ParallelFor(batch.Size, nThreads, i =>
        {
            var n = batch.LineSize(i);
            long c = 0;
            for (var j = 0; j < n; j++)
                if (isValid.Valid(batch.GetElement(i, j).Value)) c++;
            validCounts[i] = c;
        });
        return validCounts;
    }

    private void PushBatchImpl<TBatch>(Context ctx, TBatch batch, long rbegin, Func<float, bool> isValid, FeatureType[] ft)
        where TBatch : IAdapterBatch
    {
        var nThreads = ctx.Threads();
        var batchThreads = (int)Math.Max(1, Math.Min(batch.Size, nThreads));
        var nBinsTotal = Cut.TotalBins;
        var nIndex = RowPtr[rbegin + batch.Size];
        ResizeIndex(nIndex, _isDense);
        if (_isDense) Index.SetBinOffset(Cut.Ptrs.ConstHostSpan);
        if (_hitCountTloc.Length < (long)nThreads * nBinsTotal) _hitCountTloc = new long[nThreads * nBinsTotal];

        var ptrs = Cut.Ptrs.ToArray();
        var values = Cut.Values.ToArray();
        var valid = true;
        var raw = Index.Raw;
        var binSize = Index.BinTypeSize;
        var offsets = Index.Offset.ToArray();
        var dense = _isDense;
        var tloc = _hitCountTloc;
        Threading.ParallelFor(batch.Size, batchThreads, (i, tid) =>
        {
            var ibegin = RowPtr[rbegin + i];
            var k = 0;
            var n = batch.LineSize(i);
            for (var j = 0; j < n; j++)
            {
                var elem = batch.GetElement(i, j);
                if (!isValid(elem.Value)) continue;
                if (float.IsInfinity(elem.Value)) valid = false;
                var binIdx = Categorical.IsCat(ft, elem.ColumnIdx)
                    ? HistogramCuts.SearchCatBin(elem.Value, (int)elem.ColumnIdx, ptrs, values)
                    : HistogramCuts.SearchBin(elem.Value, (int)elem.ColumnIdx, ptrs, values);
                var pos = ibegin + k;
                if (dense)
                {
                    var compressed = (uint)binIdx - offsets[j];
                    switch (binSize)
                    {
                        case BinTypeSize.Uint8: raw[pos] = (byte)compressed; break;
                        case BinTypeSize.Uint16: System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref raw[pos * 2], (ushort)compressed); break;
                        default: System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref raw[pos * 4], compressed); break;
                    }
                }
                else
                {
                    System.Runtime.CompilerServices.Unsafe.WriteUnaligned(ref raw[pos * 4], (uint)binIdx);
                }
                tloc[tid * nBinsTotal + binIdx]++;
                k++;
            }
        });
        Check.That(valid, ErrorMsg.InfInData);
        GatherHitCount(nThreads, nBinsTotal);
    }

    private void GatherHitCount(int nThreads, int nBinsTotal)
    {
        Check.Eq(HitCount.Length, nBinsTotal);
        var tloc = _hitCountTloc;
        Threading.ParallelFor(nBinsTotal, nThreads, idx =>
        {
            for (var tid = 0; tid < nThreads; tid++)
            {
                HitCount[idx] += tloc[tid * nBinsTotal + idx];
                tloc[tid * nBinsTotal + idx] = 0;
            }
        });
    }

    private void ResizeColumns(double sparseThresh)
    {
        Check.That(!double.IsNaN(sparseThresh));
        _columns = new ColumnMatrix(this, sparseThresh);
    }

    public void ResizeIndex(long nIndex, bool isDense)
    {
        BinTypeSize t;
        if (MaxNumBinPerFeat - 1 <= byte.MaxValue && isDense) t = BinTypeSize.Uint8;
        else if (MaxNumBinPerFeat - 1 > byte.MaxValue && MaxNumBinPerFeat - 1 <= ushort.MaxValue && isDense) t = BinTypeSize.Uint16;
        else t = BinTypeSize.Uint32;
        var nBytes = checked((int)(nIndex * (int)t));
        Index.Resize(nBytes, t);
    }

    public void GetFeatureCounts(long[] counts)
    {
        var nfeature = Cut.Ptrs.Size - 1;
        for (var fid = 0; fid < nfeature; fid++)
        {
            var ibegin = Cut.Ptrs[fid];
            var iend = Cut.Ptrs[fid + 1];
            for (var i = ibegin; i < iend; i++) counts[fid] += HitCount[i];
        }
    }

    public int GetGindex(long ridx, int fidx)
    {
        var begin = RowIdx(ridx);
        if (IsDense) return (int)Index[begin + fidx];
        var end = RowIdx(ridx + 1);
        var fBegin = Cut.Ptrs[fidx];
        var fEnd = Cut.Ptrs[fidx + 1];
        return HistOps.BinarySearchBin(begin, end, Index, fBegin, fEnd);
    }

    public float GetFvalue(long ridx, int fidx, bool isCat) => GetFvalue(Cut.Ptrs.ConstHostSpan, Cut.Values.ConstHostSpan, ridx, fidx, isCat);

    public float GetFvalue(ReadOnlySpan<uint> ptrs, ReadOnlySpan<float> values, long ridx, int fidx, bool isCat)
    {
        if (isCat)
        {
            var gidx = GetGindex(ridx, fidx);
            if (gidx == -1) return float.NaN;
            return values[gidx];
        }
        if (IsDense)
        {
            var begin = RowIdx(ridx);
            var binIdx = Index[begin + fidx];
            return HistogramCuts.NumericBinValue(ptrs, values, fidx, (int)binIdx);
        }
        return GetFvalueImpl(ptrs, values, ridx, fidx);
    }

    private float GetFvalueImpl(ReadOnlySpan<uint> ptrs, ReadOnlySpan<float> values, long ridx, int fidx)
    {
        var columns = _columns!;
        int binIdx;
        if (columns.GetColumnType(fidx) == ColumnType.Dense)
        {
            binIdx = columns.DenseBin(fidx, ridx - BaseRowId, columns.AnyMissing);
        }
        else
        {
            var it = columns.SparseColumn(fidx, 0);
            binIdx = it.Get(ridx - BaseRowId);
        }
        if (binIdx == ColumnMatrix.MissingId) return float.NaN;
        return HistogramCuts.NumericBinValue(ptrs, values, fidx, binIdx);
    }

    /// <summary><c>AssignColumnBinIndex</c>: visits (bin, position, row, feature) for each stored bin.</summary>
    public void AssignColumnBinIndex(Action<uint, long, long, int> assign)
    {
        var batchSize = Size;
        var ptrs = Cut.Ptrs.ConstHostSpan.ToArray();
        long k = 0;
        var dense = IsDense;
        for (long ridx = 0; ridx < batchSize; ridx++)
        {
            var rBeg = RowPtr[ridx];
            var rEnd = RowPtr[ridx + 1];
            var fidx = 0;
            if (dense)
            {
                for (var j = rBeg; j < rEnd; j++)
                {
                    var f = (int)(j - rBeg);
                    assign(Index[k], k, ridx, f);
                    k++;
                }
            }
            else
            {
                // row_index = index.data<BinT>() + row_ptr[base_rowid]
                var baseOff = RowPtr[BaseRowId];
                for (var j = rBeg; j < rEnd; j++)
                {
                    var binIdx = Index[baseOff + k];
                    while (binIdx >= ptrs[fidx + 1]) fidx++;
                    assign(binIdx, k, ridx, fidx);
                    k++;
                }
            }
        }
    }
}

/// <summary>Port of the helpers in src/common/numeric.h.</summary>
public static class Numeric
{
    /// <summary>
    /// <c>PartialSum</c>: <c>out[outOffset + 1 + i] = init + sum(in[0..i])</c> computed in thread blocks.
    /// </summary>
    public static void PartialSum(int nThreads, long[] input, long init, long[] output, long outOffset)
    {
        var n = input.Length;
        var batchThreads = Math.Max(1, Math.Min(n, nThreads));
        var partialSums = new long[batchThreads];
        var blockSize = n / batchThreads;
        Threading.ParallelFor(batchThreads, batchThreads, tidL =>
        {
            var tid = (int)tidL;
            var ibegin = blockSize * tid;
            var iend = tid == batchThreads - 1 ? n : blockSize * (tid + 1);
            long running = 0;
            for (var r = ibegin; r < iend; r++)
            {
                running += input[r];
                output[outOffset + 1 + r] = running;
            }
        });
        partialSums[0] = init;
        for (var i = 1; i < batchThreads; i++) partialSums[i] = partialSums[i - 1] + output[outOffset + i * blockSize];
        Threading.ParallelFor(batchThreads, batchThreads, tidL =>
        {
            var tid = (int)tidL;
            var ibegin = blockSize * tid;
            var iend = tid == batchThreads - 1 ? n : blockSize * (tid + 1);
            for (var i = ibegin; i < iend; i++) output[outOffset + 1 + i] += partialSums[tid];
        });
    }

    /// <summary><c>cpu_impl::Reduce</c> over float values into double with per-thread partial sums.</summary>
    public static double Reduce(Context ctx, ReadOnlySpan<float> values)
    {
        var n = values.Length;
        var nThreads = Math.Min(n, ctx.Threads());
        if (nThreads <= 0) return 0.0;
        var tloc = new double[nThreads];
        var arr = values.ToArray();
        Threading.ParallelFor(n, nThreads, (i, tid) => tloc[tid] += arr[i]);
        var result = 0.0;
        for (var t = 0; t < nThreads; t++) result += tloc[t];
        return result;
    }
}
