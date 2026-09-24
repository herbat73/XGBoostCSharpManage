// Port of the SparsePage family in include/xgboost/data.h and src/data/data.cc, plus
// src/common/group_data.h and src/data/entry.h.
using XGBoost.Common;

namespace XGBoost.Data;

/// <summary>Element of an adapter line (<c>data::COOTuple</c>).</summary>
public readonly record struct CooTuple(long RowIdx, long ColumnIdx, float Value);

/// <summary><c>data::IsValidFunctor</c>.</summary>
public readonly struct IsValidFunctor(float missing)
{
    public readonly float Missing = missing;
    public bool Valid(float value) => !(float.IsNaN(value) || value == Missing);
}

/// <summary>Read-only CSR view, <c>HostSparsePageView</c>.</summary>
public readonly struct HostSparsePageView(long[] offset, int offsetCount, Entry[] data)
{
    public readonly long[] Offset = offset;
    public readonly Entry[] Data = data;
    private readonly int _offsetCount = offsetCount;

    public ReadOnlySpan<Entry> this[long i] => Data.AsSpan((int)Offset[i], (int)(Offset[i + 1] - Offset[i]));

    public long Size => _offsetCount == 0 ? 0 : _offsetCount - 1;
}

/// <summary>CSR batch of rows (or columns for the CSC variants), <c>SparsePage</c>.</summary>
public class SparsePage
{
    public HostDeviceVector<long> Offset { get; set; } = new();
    public HostDeviceVector<Entry> Data { get; set; } = new();
    public long BaseRowId { get; set; }

    public SparsePage() => Clear();

    public HostSparsePageView GetView() => new(Offset.RawArray, Offset.Size, Data.RawArray);

    /// <summary>Number of instances (rows) in the page.</summary>
    public long Size => Offset.Size == 0 ? 0 : Offset.Size - 1;

    public long MemCostBytes => Offset.Size * 8L + Data.Size * 8L;

    public void Clear()
    {
        BaseRowId = 0;
        Offset.Resize(0);
        Offset.Add(0);
        Data.Resize(0);
    }

    public void SetBaseRowId(long rowId) => BaseRowId = rowId;

    public ReadOnlySpan<Entry> this[long i] => GetView()[i];

    public SparsePage GetTranspose(int numColumns, int nThreads)
    {
        var transpose = new SparsePage();
        var builder = new ParallelGroupBuilder<Entry>(transpose.Offset, transpose.Data, 0, isRowMajor: false);
        builder.InitBudget(numColumns, nThreads);
        var batchSize = Size;
        var page = GetView();
        Threading.ParallelFor(batchSize, nThreads, (i, tid) =>
        {
            foreach (var entry in page[i]) builder.AddBudget(entry.Index, tid);
        });
        builder.InitStorage();
        Threading.ParallelFor(batchSize, nThreads, (i, tid) =>
        {
            foreach (var entry in page[i]) builder.Push(entry.Index, new Entry((uint)(BaseRowId + i), entry.Fvalue), tid);
        });
        if (Data.Empty)
        {
            transpose.Offset.Resize(numColumns + 1);
            transpose.Offset.Fill(0);
        }
        Check.Eq(transpose.Offset.Size, numColumns + 1);
        return transpose;
    }

    public bool IsIndicesSorted(int nThreads)
    {
        var offset = Offset.RawArray;
        var data = Data.RawArray;
        nThreads = (int)Math.Max(Math.Min(nThreads, Size), 1);
        var sortedTloc = new long[nThreads];
        Threading.ParallelFor(Size, nThreads, (i, tid) =>
        {
            var beg = (int)offset[i];
            var end = (int)offset[i + 1];
            var sorted = true;
            for (var k = beg + 1; k < end; k++)
            {
                if (Entry.CmpIndex(data[k], data[k - 1]))
                {
                    sorted = false;
                    break;
                }
            }
            if (sorted) sortedTloc[tid]++;
        });
        return sortedTloc.Sum() == Size;
    }

    public void SortIndices(int nThreads)
    {
        var offset = Offset.RawArray;
        var data = Data.RawArray;
        Threading.ParallelFor(Size, nThreads, i =>
        {
            var beg = (int)offset[i];
            var end = (int)offset[i + 1];
            StdAlgo.Sort(data.AsSpan(beg, end - beg), new IndexLess());
        });
    }

    public void Reindex(ulong featureOffset, int nThreads)
    {
        var data = Data.RawArray;
        Threading.ParallelFor(Data.Size, nThreads, i => data[i].Index += (uint)featureOffset);
    }

    public void SortRows(int nThreads)
    {
        var offset = Offset.RawArray;
        var data = Data.RawArray;
        Threading.ParallelFor(Size, nThreads, i =>
        {
            if (offset[i] < offset[i + 1])
            {
                var beg = (int)offset[i];
                StdAlgo.Sort(data.AsSpan(beg, (int)(offset[i + 1] - offset[i])), new ValueLess());
            }
        });
    }

    public void Push(SparsePage batch)
    {
        var top = Offset[Offset.Size - 1];
        Data.Extend(batch.Data.ConstHostSpan);
        var begin = Offset.Size;
        Offset.Resize(begin + (int)batch.Size);
        for (var i = 0; i < batch.Size; i++) Offset[begin + i] = top + batch.Offset[i + 1];
    }

    /// <summary>
    /// Pushes an adapter batch, returning the maximum number of columns seen. Port of the templated
    /// <c>SparsePage::Push</c>.
    /// </summary>
    public long Push<TBatch>(in TBatch batch, float missing, int nthread) where TBatch : IAdapterBatch
    {
        var isRowMajor = batch.IsRowMajor;
        nthread = isRowMajor ? nthread : 1;
        var builderBaseRowOffset = Size;
        var builder = new ParallelGroupBuilder<Entry>(Offset, Data, builderBaseRowOffset, isRowMajor);

        long expectedRows = 0;
        var batchSize = batch.Size;
        if (batchSize > 0)
        {
            var lastLine = batchSize - 1;
            var n = batch.LineSize(lastLine);
            if (n > 0) expectedRows = batch.GetElement(lastLine, n - 1).RowIdx - BaseRowId;
        }
        expectedRows = isRowMajor ? batchSize : expectedRows;
        long maxColumns = 0;
        if (batchSize == 0) return maxColumns;
        var threadSize = batchSize / nthread;

        builder.InitBudget(expectedRows, nthread);
        var maxColumnsVector = new long[nthread];
        var valid = true;
        var b = batch;
        var baseRowId = BaseRowId;
        Threading.ParallelFor(nthread, nthread, (tidL, _) =>
        {
            var tid = (int)tidL;
            var begin = tid * threadSize;
            var end = tid != nthread - 1 ? (tid + 1) * threadSize : batchSize;
            long maxLocal = 0;
            for (var i = begin; i < end; i++)
            {
                var size = b.LineSize(i);
                for (var j = 0; j < size; j++)
                {
                    var element = b.GetElement(i, j);
                    if (!float.IsInfinity(missing) && float.IsInfinity(element.Value)) valid = false;
                    var key = element.RowIdx - baseRowId;
                    Check.Ge(key, builderBaseRowOffset);
                    maxLocal = Math.Max(maxLocal, element.ColumnIdx + 1);
                    if (!float.IsNaN(element.Value) && element.Value != missing) builder.AddBudget(key, tid);
                }
            }
            maxColumnsVector[tid] = maxLocal;
        });
        if (!valid) Check.Fail(ErrorMsg.InfInData);
        foreach (var m in maxColumnsVector) maxColumns = Math.Max(maxColumns, m);

        builder.InitStorage();

        var isValid = new IsValidFunctor(missing);
        Threading.ParallelFor(nthread, nthread, (tidL, _) =>
        {
            var tid = (int)tidL;
            var begin = tid * threadSize;
            var end = tid != nthread - 1 ? (tid + 1) * threadSize : batchSize;
            for (var i = begin; i < end; i++)
            {
                var size = b.LineSize(i);
                for (var j = 0; j < size; j++)
                {
                    var element = b.GetElement(i, j);
                    var key = element.RowIdx - baseRowId;
                    if (isValid.Valid(element.Value)) builder.Push(key, new Entry((uint)element.ColumnIdx, element.Value), tid);
                }
            }
        });
        return maxColumns;
    }

    public void PushCsc(SparsePage batch)
    {
        var selfData = Data;
        var selfOffset = Offset;
        var otherData = batch.Data;
        var otherOffset = batch.Offset;
        if (otherData.Empty)
        {
            Offset = new HostDeviceVector<long>(otherOffset.ToArray(), true);
            return;
        }
        if (!selfData.Empty)
        {
            Check.Eq(selfOffset.Size, otherOffset.Size);
        }
        else
        {
            Data = new HostDeviceVector<Entry>(otherData.ToArray(), true);
            Offset = new HostDeviceVector<long>(otherOffset.ToArray(), true);
            return;
        }
        var offset = new long[otherOffset.Size];
        var data = new Entry[selfData.Size + otherData.Size];
        var nFeatures = otherOffset.Size - 1;
        long beg = 0;
        var ptr = 1;
        for (var i = 0; i < nFeatures; i++)
        {
            var selfBeg = selfOffset[i];
            var selfLength = selfOffset[i + 1] - selfBeg;
            selfData.ConstHostSpan.Slice((int)selfBeg, (int)selfLength).CopyTo(data.AsSpan((int)beg));
            beg += selfLength;
            var otherBeg = otherOffset[i];
            var otherLength = otherOffset[i + 1] - otherBeg;
            otherData.ConstHostSpan.Slice((int)otherBeg, (int)otherLength).CopyTo(data.AsSpan((int)beg));
            beg += otherLength;
            offset[ptr++] = beg;
        }
        Data = new HostDeviceVector<Entry>(data, true);
        Offset = new HostDeviceVector<long>(offset, true);
    }

    private readonly struct IndexLess : IStdLess<Entry>
    {
        public bool Less(Entry a, Entry b) => a.Index < b.Index;
    }

    private readonly struct ValueLess : IStdLess<Entry>
    {
        public bool Less(Entry a, Entry b) => a.Fvalue < b.Fvalue;
    }
}

/// <summary>Column-major page (CSC), <c>CSCPage</c>.</summary>
public sealed class CscPage : SparsePage
{
    public CscPage() { }

    public CscPage(SparsePage page)
    {
        Offset = page.Offset;
        Data = page.Data;
        BaseRowId = page.BaseRowId;
    }
}

/// <summary>Column-major page with each column sorted by value, <c>SortedCSCPage</c>.</summary>
public sealed class SortedCscPage : SparsePage
{
    public SortedCscPage() { }

    public SortedCscPage(SparsePage page)
    {
        Offset = page.Offset;
        Data = page.Data;
        BaseRowId = page.BaseRowId;
    }
}

/// <summary>Port of <c>common::ParallelGroupBuilder</c>.</summary>
public sealed class ParallelGroupBuilder<TValue>(HostDeviceVector<long> rptr, HostDeviceVector<TValue> data, long baseRowOffset, bool isRowMajor)
{
    private List<long>[] _threadRptr = [];
    private long _threadDisplacement;

    public void InitBudget(long maxKey, int nthread)
    {
        _threadRptr = new List<long>[nthread];
        var fullSize = isRowMajor ? maxKey : maxKey - Math.Min(baseRowOffset, maxKey);
        _threadDisplacement = isRowMajor ? fullSize / nthread : 0;
        for (var i = 0; i < nthread - 1; i++)
        {
            var threadSize = isRowMajor ? _threadDisplacement : fullSize;
            _threadRptr[i] = new List<long>(new long[threadSize]);
        }
        var lastThreadSize = isRowMajor ? fullSize - (nthread - 1) * _threadDisplacement : fullSize;
        _threadRptr[nthread - 1] = new List<long>(new long[lastThreadSize]);
    }

    public void AddBudget(long key, int threadId, long nelem = 1)
    {
        var trptr = _threadRptr[threadId];
        var offsetKey = isRowMajor ? key - baseRowOffset - threadId * _threadDisplacement : key - baseRowOffset;
        while (trptr.Count < offsetKey + 1) trptr.Add(0);
        trptr[(int)offsetKey] += nelem;
    }

    public void InitStorage()
    {
        if (isRowMajor)
        {
            long expectedRows = 0;
            foreach (var t in _threadRptr) expectedRows += t.Count;
            var fill = rptr.Size == 0 ? 0 : rptr[rptr.Size - 1];
            rptr.Resize((int)(expectedRows + baseRowOffset + 1), fill);
            long count = 0;
            var offsetIdx = baseRowOffset + 1;
            foreach (var trptr in _threadRptr)
            {
                for (var i = 0; i < trptr.Count; i++)
                {
                    var threadCount = trptr[i];
                    trptr[i] = count + fill;
                    count += threadCount;
                    if (offsetIdx < rptr.Size) rptr[(int)offsetIdx++] += count;
                }
            }
            data.Resize((int)rptr[rptr.Size - 1]);
        }
        else
        {
            var fill = rptr.Size == 0 ? 0 : rptr[rptr.Size - 1];
            foreach (var t in _threadRptr)
                if (rptr.Size <= t.Count + baseRowOffset) rptr.Resize((int)(t.Count + baseRowOffset + 1), fill);
            long count = 0;
            for (var i = baseRowOffset; i + 1 < rptr.Size; i++)
            {
                foreach (var trptr in _threadRptr)
                {
                    if (i < trptr.Count + baseRowOffset)
                    {
                        var threadCount = trptr[(int)(i - baseRowOffset)];
                        trptr[(int)(i - baseRowOffset)] = count + rptr[rptr.Size - 1];
                        count += threadCount;
                    }
                }
                rptr[(int)(i + 1)] += count;
            }
            data.Resize((int)rptr[rptr.Size - 1]);
        }
    }

    public void Push(long key, TValue value, int threadId)
    {
        var offsetKey = isRowMajor ? key - baseRowOffset - threadId * _threadDisplacement : key - baseRowOffset;
        var list = _threadRptr[threadId];
        var rp = list[(int)offsetKey];
        data.RawArray[rp] = value;
        list[(int)offsetKey] = rp + 1;
    }
}

/// <summary>Parameters for constructing histogram index batches, <c>BatchParam</c>.</summary>
public sealed class BatchParam
{
    public int MaxBin;
    public float[]? Hess;
    public bool Regen;
    public bool ForbidRegen;
    public double SparseThresh = double.NaN;
    public bool PrefetchCopy = true;
    public int NPrefetchBatches = 3;

    public BatchParam() { }

    public BatchParam(int maxBin, double sparseThresh)
    {
        MaxBin = maxBin;
        SparseThresh = sparseThresh;
    }

    public BatchParam(int maxBin, float[] hessian, bool regenerate)
    {
        MaxBin = maxBin;
        Hess = hessian;
        Regen = regenerate;
    }

    public bool ParamNotEqual(BatchParam other)
    {
        var cond = MaxBin != other.MaxBin;
        var lNan = double.IsNaN(SparseThresh);
        var rNan = double.IsNaN(other.SparseThresh);
        var stChg = lNan != rNan || (!lNan && !rNan && SparseThresh != other.SparseThresh);
        return cond | stChg;
    }

    public bool Initialized => MaxBin != 0;

    public BatchParam MakeCache()
    {
        var p = (BatchParam)MemberwiseClone();
        p.Regen = false;
        p.ForbidRegen = false;
        return p;
    }
}

/// <summary>Port of src/data/batch_utils.h helpers used by DMatrix implementations.</summary>
public static class BatchUtils
{
    /// <summary>Whether the gradient index needs to be regenerated for <paramref name="param"/>.</summary>
    public static bool RegenGHist(BatchParam old, BatchParam param)
    {
        // Parameter is renewed or caller requests a regen
        if (!param.Initialized) return false;
        return param.Regen || old.ParamNotEqual(param);
    }

    public static void CheckEmpty(BatchParam l, BatchParam r)
    {
        if (!l.Initialized) Check.That(r.Initialized, "Batch parameter is not initialized.");
    }

    /// <summary>Validates the batch parameter from the caller against the one used at construction.</summary>
    public static void CheckParam(BatchParam init, BatchParam param)
    {
        Check.Eq(param.MaxBin, init.MaxBin, ErrorMsg.InconsistentMaxBin);
        Check.That(!param.Regen && (param.Hess is null || param.Hess.Length == 0),
            "Only the `hist` tree method can use the `QuantileDMatrix`.");
    }
}
