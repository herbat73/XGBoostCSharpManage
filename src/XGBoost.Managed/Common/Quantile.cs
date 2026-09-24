// Port of src/common/quantile.h/.cc: weighted quantile summaries (WQSummary), the merge/prune sketch
// (WQuantileSketch) and the per-feature HostSketchContainer that produces HistogramCuts.
using System.Runtime.InteropServices;
using XGBoost.Data;

namespace XGBoost.Common;

/// <summary>Entry of a weighted quantile summary.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct WQEntry(float rmin, float rmax, float wmin, float value)
{
    public float RMin = rmin;
    public float RMax = rmax;
    public float WMin = wmin;
    public float Value = value;

    public readonly float RMinNext() => RMin + WMin;
    public readonly float RMaxPrev() => RMax - WMin;
}

/// <summary>
/// Weighted quantile summary, <c>WQSummary&lt;float, float&gt;</c>. Owns its storage (the C++ version is a view,
/// with <c>WQSummaryContainer</c> adding ownership; both are merged here).
/// </summary>
public sealed class WQSummary
{
    private WQEntry[] _data;
    private int _size;

    public WQSummary() => _data = [];

    public WQSummary(WQEntry[] storage, int size)
    {
        _data = storage;
        _size = size;
    }

    public int Size => _size;
    public bool Empty => _size == 0;
    public int Capacity => _data.Length;
    public ReadOnlySpan<WQEntry> Entries => _data.AsSpan(0, _size);
    public WQEntry[] Storage => _data;

    public void SetSize(int n)
    {
        Check.Le(n, _data.Length);
        _size = n;
    }

    public void Clear() => _size = 0;

    public void Reserve(int size)
    {
        if (size > _data.Length) Array.Resize(ref _data, size);
    }

    public void CopyFrom(WQSummary src)
    {
        if (src._size == 0 && src._data.Length == 0)
        {
            Clear();
            return;
        }
        if (_data.Length == 0)
        {
            Check.Eq(_size, 0);
            Check.Eq(src._size, 0);
            return;
        }
        _size = src._size;
        src._data.AsSpan(0, _size).CopyTo(_data);
    }

    public void SetFromSorted(List<(float Value, float Weight)> queue)
    {
        Clear();
        var wsum = 0f;
        for (var i = 0; i < queue.Count;)
        {
            var j = i + 1;
            var w = queue[i].Weight;
            while (j < queue.Count && queue[j].Value == queue[i].Value)
            {
                w += queue[j].Weight;
                j++;
            }
            _data[_size++] = new WQEntry(wsum, wsum + w, w, queue[i].Value);
            wsum += w;
            i = j;
        }
    }

    /// <summary>Summary of a sorted column, pruned to <paramref name="maxSize"/>.</summary>
    public void SetPruneSorted(ReadOnlySpan<Entry> column, float[] weights, int maxSize)
    {
        Check.Ge(maxSize, 1);
        Check.Ge(_data.Length, maxSize + 1);
        Clear();
        var colSize = column.Length;
        var sumTotal = 0.0;
        long uniqueValues = 0;
        var rmin = 0.0;
        var wmin = 0.0;
        var lastFvalue = 0f;
        double nextGoal = -1.0f;

        for (var i = 0; i < colSize; i++)
        {
            if (i == 0 || column[i - 1].Fvalue != column[i].Fvalue) uniqueValues++;
            sumTotal += weights[column[i].Index];
        }

        if (uniqueValues <= maxSize)
        {
            for (var i = 0; i < colSize; i++)
            {
                var c = column[i];
                if (i == 0)
                {
                    lastFvalue = c.Fvalue;
                    wmin = weights[c.Index];
                    continue;
                }
                if (lastFvalue == c.Fvalue)
                {
                    wmin += weights[c.Index];
                    continue;
                }
                var rmax = rmin + wmin;
                _data[_size] = new WQEntry((float)rmin, (float)rmax, (float)wmin, lastFvalue);
                SetSize(_size + 1);
                rmin = rmax;
                lastFvalue = c.Fvalue;
                wmin = weights[c.Index];
            }
            if (colSize != 0)
            {
                var rmax = rmin + wmin;
                _data[_size] = new WQEntry((float)rmin, (float)rmax, (float)wmin, lastFvalue);
                SetSize(_size + 1);
            }
            return;
        }

        for (var i = 0; i < colSize; i++)
        {
            var c = column[i];
            if (nextGoal == -1.0f)
            {
                nextGoal = 0.0f;
                lastFvalue = c.Fvalue;
                wmin = weights[c.Index];
                continue;
            }
            if (lastFvalue != c.Fvalue)
            {
                var rmax = rmin + wmin;
                var summarySize = _size;
                if (rmax >= nextGoal && summarySize != maxSize)
                {
                    if (summarySize == 0 || lastFvalue > _data[summarySize - 1].Value)
                    {
                        Check.Lt(summarySize, maxSize, $"invalid maximum size max_size={maxSize}, stemp.current_elements={summarySize}");
                        _data[summarySize] = new WQEntry((float)rmin, (float)rmax, (float)wmin, lastFvalue);
                        summarySize++;
                        SetSize(summarySize);
                    }
                    if (summarySize == maxSize) nextGoal = sumTotal * 2.0f + 1e-5f;
                    else nextGoal = (float)(summarySize * sumTotal / maxSize);
                }
                rmin = rmax;
                wmin = weights[c.Index];
                lastFvalue = c.Fvalue;
            }
            else
            {
                wmin += weights[c.Index];
            }
        }

        if (colSize != 0)
        {
            var summarySize = _size;
            var rmax = rmin + wmin;
            if (summarySize == 0 || lastFvalue > _data[summarySize - 1].Value)
            {
                Check.Le(summarySize, maxSize);
                _data[summarySize] = new WQEntry((float)rmin, (float)rmax, (float)wmin, lastFvalue);
                summarySize++;
                SetSize(summarySize);
            }
        }
    }

    /// <summary>Prunes in place to at most <paramref name="maxsize"/> entries.</summary>
    public void SetPrune(int maxsize)
    {
        if (maxsize == 0)
        {
            _size = 0;
            return;
        }
        var srcSize = _size;
        if (srcSize <= maxsize) return;
        var src = _data;
        if (maxsize == 1)
        {
            _size = 1;
            return;
        }
        var begin = src[0].RMax;
        var range = src[srcSize - 1].RMin - src[0].RMax;
        var n = maxsize - 1;
        var tail = src[srcSize - 1];
        var left = src[1];
        var right = src[2];
        var current = 1;
        int i = 1, lastidx = 0;
        for (var k = 1; k < n; k++)
        {
            var dx2 = 2 * (((float)k * range) / n + begin);
            while (i < srcSize - 1 && dx2 >= right.RMax + right.RMin)
            {
                i++;
                left = right;
                if (i < srcSize - 1) right = src[i + 1];
            }
            if (i == srcSize - 1) break;
            if (dx2 < left.RMinNext() + right.RMaxPrev())
            {
                if (i != lastidx)
                {
                    _data[current++] = left;
                    lastidx = i;
                }
            }
            else
            {
                if (i + 1 != lastidx)
                {
                    _data[current++] = right;
                    lastidx = i + 1;
                }
            }
        }
        if (lastidx != srcSize - 1) _data[current++] = tail;
        _size = current;
    }

    /// <summary>Histogram cut values from this summary, including the final sentinel upper bound.</summary>
    public List<float> QueryCutValues(int maxBin)
    {
        if (Empty) return [1e-5f];
        var nEntries = _size;
        var cutValues = new List<float>(Math.Min(nEntries, maxBin) + 1);
        var d = _data;

        int AdvanceToNextDistinct(int cursor, float value)
        {
            while (cursor < nEntries && d[cursor].Value <= value) cursor++;
            return cursor;
        }

        var lastCut = d[0].Value;
        var nextValueCursor = AdvanceToNextDistinct(1, lastCut);
        if (nEntries <= maxBin)
        {
            while (nextValueCursor < nEntries)
            {
                var cpt = d[nextValueCursor].Value;
                cutValues.Add(cpt);
                lastCut = cpt;
                nextValueCursor = AdvanceToNextDistinct(nextValueCursor + 1, lastCut);
            }
        }
        else
        {
            var total = (double)d[nEntries - 1].RMax;
            var queryCursor = 0;
            for (var i = 1; i < maxBin; i++)
            {
                var rank = (double)i * total / maxBin;
                var rank2 = 2.0 * rank;
                while (queryCursor < nEntries - 2 && rank2 >= (double)(d[queryCursor + 1].RMin + d[queryCursor + 1].RMax))
                    queryCursor++;
                var queried = rank2 < (double)(d[queryCursor].RMinNext() + d[queryCursor + 1].RMaxPrev())
                    ? d[queryCursor]
                    : d[queryCursor + 1];
                var cpt = queried.Value;
                if (cpt <= lastCut)
                {
                    nextValueCursor = AdvanceToNextDistinct(nextValueCursor, lastCut);
                    if (nextValueCursor == nEntries) break;
                    cpt = d[nextValueCursor].Value;
                }
                else if (nextValueCursor < nEntries && d[nextValueCursor].Value <= cpt)
                {
                    nextValueCursor = AdvanceToNextDistinct(nextValueCursor + 1, cpt);
                }
                cutValues.Add(cpt);
                lastCut = cpt;
            }
        }
        var last = d[nEntries - 1].Value;
        cutValues.Add(last + (MathF.Abs(last) + 1e-5f));
        return cutValues;
    }

    /// <summary>Merges <paramref name="other"/> into this summary.</summary>
    public void SetCombine(WQSummary other, List<WQEntry>? workspace = null)
    {
        if (other.Empty) return;
        if (_data.Length == 0)
        {
            _size = 0;
            return;
        }
        if (Empty)
        {
            Check.Ge(_data.Length, other._size);
            CopyFrom(other);
            return;
        }
        var mergedSize = _size + other._size;
        Check.Ge(_data.Length, mergedSize);
        var merged = new WQEntry[mergedSize];

        var a = _data;
        var b = other._data;
        int ai = 0, aEnd = _size, bi = 0, bEnd = other._size, dst = 0;
        float aprevRmin = 0, bprevRmin = 0;
        while (ai != aEnd && bi != bEnd)
        {
            if (a[ai].Value == b[bi].Value)
            {
                merged[dst++] = new WQEntry(a[ai].RMin + b[bi].RMin, a[ai].RMax + b[bi].RMax, a[ai].WMin + b[bi].WMin, a[ai].Value);
                aprevRmin = a[ai].RMinNext();
                bprevRmin = b[bi].RMinNext();
                ai++;
                bi++;
            }
            else if (a[ai].Value < b[bi].Value)
            {
                merged[dst++] = new WQEntry(a[ai].RMin + bprevRmin, a[ai].RMax + b[bi].RMaxPrev(), a[ai].WMin, a[ai].Value);
                aprevRmin = a[ai].RMinNext();
                ai++;
            }
            else
            {
                merged[dst++] = new WQEntry(b[bi].RMin + aprevRmin, b[bi].RMax + a[ai].RMaxPrev(), b[bi].WMin, b[bi].Value);
                bprevRmin = b[bi].RMinNext();
                bi++;
            }
        }
        if (ai != aEnd)
        {
            var brmax = b[bEnd - 1].RMax;
            do
            {
                merged[dst++] = new WQEntry(a[ai].RMin + bprevRmin, a[ai].RMax + brmax, a[ai].WMin, a[ai].Value);
                ai++;
            } while (ai != aEnd);
        }
        if (bi != bEnd)
        {
            var armax = a[aEnd - 1].RMax;
            do
            {
                merged[dst++] = new WQEntry(b[bi].RMin + aprevRmin, b[bi].RMax + armax, b[bi].WMin, b[bi].Value);
                bi++;
            } while (bi != bEnd);
        }

        FixError(merged, dst, out var errMingap, out var errMaxgap, out var errWgap);
        const float tol = 10;
        if (errMingap > tol || errMaxgap > tol || errWgap > tol)
            Log.Info($"mingap={Format.Stream(errMingap)}, maxgap={Format.Stream(errMaxgap)}, wgap={Format.Stream(errWgap)}");
        Check.That(dst <= _size + other._size, "bug in combine");
        merged.AsSpan(0, dst).CopyTo(_data);
        _size = dst;
        _ = workspace;
    }

    private static void FixError(WQEntry[] entries, int n, out float errMingap, out float errMaxgap, out float errWgap)
    {
        errMingap = 0;
        errMaxgap = 0;
        errWgap = 0;
        float prevRmin = 0, prevRmax = 0;
        for (var i = 0; i < n; i++)
        {
            if (entries[i].RMin < prevRmin)
            {
                entries[i].RMin = prevRmin;
                errMingap = Math.Max(errMingap, prevRmin - entries[i].RMin);
            }
            else
            {
                prevRmin = entries[i].RMin;
            }
            if (entries[i].RMax < prevRmax)
            {
                entries[i].RMax = prevRmax;
                errMaxgap = Math.Max(errMaxgap, prevRmax - entries[i].RMax);
            }
            var rminNext = entries[i].RMinNext();
            if (entries[i].RMax < rminNext)
            {
                entries[i].RMax = rminNext;
                errWgap = Math.Max(errWgap, entries[i].RMax - rminNext);
            }
            prevRmax = entries[i].RMax;
        }
    }
}

/// <summary>Input buffer merging consecutive duplicates, <c>Queue</c>.</summary>
public sealed class SketchQueue
{
    public readonly List<(float Value, float Weight)> Queue = [];
    public int MaxSize { get; }

    public SketchQueue(int maxSize = 1)
    {
        Check.Ge(maxSize, 1);
        MaxSize = maxSize;
    }

    public int Size => Queue.Count;

    public bool Push(float x, float w)
    {
        if (Queue.Count == 0 || Queue[^1].Value != x)
        {
            if (Queue.Count == MaxSize) return false;
            Queue.Add((x, w));
            return true;
        }
        var last = Queue[^1];
        Queue[^1] = (last.Value, last.Weight + w);
        return true;
    }

    public void PopSummary(WQSummary output)
    {
        output.Reserve(Queue.Count);
        StdAlgo.Sort(CollectionsMarshal.AsSpan(Queue), static (l, r) => l.Value < r.Value);
        output.SetFromSorted(Queue);
        Queue.Clear();
    }
}

/// <summary>Weighted quantile sketch using merge/prune, <c>WQuantileSketch</c>.</summary>
public sealed class WQuantileSketch
{
    public const float Factor = 2.0f;

    private SketchQueue _inqueue = new(1);
    private int _limitSize = 1;
    private readonly List<WQSummary> _level = [];
    private readonly WQSummary _temp = new();
    private long _numElements;

    public WQuantileSketch() { }

    public WQuantileSketch(long maxn, double eps)
    {
        _limitSize = LimitSizeLevel(maxn, eps);
        _inqueue = new SketchQueue(_limitSize * 2);
    }

    public long NumElements => _numElements;

    public static int LimitSizeLevel(long maxn, double eps)
    {
        if (maxn == 0) return 1;
        var internalEps = eps / Factor;
        long nlevel = 1;
        long limitSize;
        while (true)
        {
            limitSize = (long)Math.Ceiling(nlevel / internalEps) + 1;
            limitSize = Math.Min(maxn, limitSize);
            var n = 1L << (int)nlevel;
            if (n * limitSize >= maxn) break;
            nlevel++;
        }
        var nn = 1L << (int)nlevel;
        Check.That(nn * limitSize >= maxn, "invalid init parameter");
        Check.That(nlevel <= Math.Max(1L, (long)(limitSize * internalEps)), "invalid init parameter");
        return (int)limitSize;
    }

    public void Push(float x, float w = 1)
    {
        if (w == 0f) return;
        _numElements++;
        if (!_inqueue.Push(x, w))
        {
            _inqueue.PopSummary(_temp);
            PushSummary(_temp);
            _inqueue.Push(x, w);
        }
    }

    public void PushSorted(ReadOnlySpan<Entry> column, float[] weights, int numRetainedItems)
    {
        Check.Ge(numRetainedItems, 1);
        if (weights.Length == 0)
        {
            _numElements += column.Length;
        }
        else
        {
            foreach (var e in column)
                if (weights[e.Index] != 0f) _numElements++;
        }
        var maxSize = numRetainedItems;
        _temp.Reserve(maxSize + 1);
        _temp.SetPruneSorted(column, weights, maxSize);
        if (!column.IsEmpty) PushSummary(_temp);
    }

    public void PushSummary(WQSummary summary)
    {
        summary.Reserve(_limitSize * 2);
        var l = 0;
        while (true)
        {
            LazyInitLevel(l + 1);
            summary.SetPrune(_limitSize);
            summary.SetCombine(_level[l]);
            _level[l].Clear();
            if (summary.Size <= _limitSize) break;
            l++;
        }
        _level[l].CopyFrom(summary);
    }

    public WQSummary GetSummary(int maxSize)
    {
        _inqueue.PopSummary(_temp);
        PushSummary(_temp);

        var pruneSize = Math.Max(maxSize, _limitSize);
        long observed = 0;
        foreach (var s in _level) observed += s.Size;
        var initialReserve = (int)Math.Min(observed, (long)pruneSize + _limitSize);
        var output = new WQSummary();
        if (initialReserve > 0) output.Reserve(initialReserve);
        foreach (var levelSummary in _level)
        {
            var combineNeeded = output.Size + levelSummary.Size;
            if (combineNeeded > output.Capacity) output.Reserve(combineNeeded);
            output.SetCombine(levelSummary);
            output.SetPrune(pruneSize);
        }
        output.SetPrune(maxSize);
        return output;
    }

    private void LazyInitLevel(int nlevel)
    {
        if (_level.Count >= nlevel) return;
        // The C++ code reallocates all levels (discarding contents) when growing, which is only
        // reached while the levels above the current one are empty.
        var old = _level.ToList();
        _level.Clear();
        for (var l = 0; l < nlevel; l++)
        {
            var s = new WQSummary(new WQEntry[_limitSize], 0);
            if (l < old.Count)
            {
                // Level contents are preserved by the vector::resize of data_ in C++.
                s.CopyFrom(old[l]);
            }
            _level.Add(s);
        }
    }
}

/// <summary>Sketches for every feature, <c>HostSketchContainer</c>.</summary>
public sealed class HostSketchContainer
{
    private readonly WQuantileSketch[] _sketches;
    private readonly SortedSet<float>[] _categories;
    private readonly FeatureType[] _featureTypes;
    private readonly long[] _columnsSize;
    private readonly int _maxBins;
    private readonly bool _useGroupInd;
    private readonly int _nThreads;
    private readonly bool _hasCategorical;

    public HostSketchContainer(Context ctx, int maxBin, ReadOnlySpan<FeatureType> featureTypes, long[] columnsSize, bool useGroup)
    {
        _featureTypes = featureTypes.ToArray();
        _columnsSize = columnsSize;
        _maxBins = maxBin;
        _useGroupInd = useGroup;
        _nThreads = ctx.Threads();
        Check.Ge(maxBin, 2, ErrorMsg.InvalidMaxBin);
        Check.Ne(_columnsSize.Length, 0);
        _sketches = new WQuantileSketch[_columnsSize.Length];
        _categories = new SortedSet<float>[_columnsSize.Length];
        for (var i = 0; i < _categories.Length; i++) _categories[i] = [];
        _hasCategorical = _featureTypes.Any(Categorical.IsCatOp);
        var ft = _featureTypes;
        Threading.ParallelFor(_sketches.Length, _nThreads, Sched.Auto(), i =>
        {
            var columnSize = _columnsSize[i];
            var eps = SketchEpsilon(_maxBins, columnSize);
            _sketches[i] = Categorical.IsCat(ft, i) ? new WQuantileSketch() : new WQuantileSketch(columnSize, eps);
        });
    }

    public static double SketchEpsilon(int maxBins, long numSamples)
    {
        var n = Math.Max(1L, numSamples);
        var nBins = Math.Min(maxBins, n);
        return 1.0 / nBins;
    }

    public static int SketchSummaryBudget(int maxBins, long numSamples) =>
        WQuantileSketch.LimitSizeLevel(numSamples, SketchEpsilon(maxBins, numSamples));

    public static bool UseGroup(MetaInfo info)
    {
        var numGroups = info.GroupPtr.Count == 0 ? 0 : info.GroupPtr.Count - 1;
        return numGroups != 0 && info.Weights.Size != info.NumRow;
    }

    public static float[] UnrollGroupWeights(MetaInfo info)
    {
        var groupWeights = info.Weights;
        if (groupWeights.Empty) return [];
        var groupPtr = info.GroupPtr;
        Check.Ge(groupPtr.Count, 2);
        Check.Eq(groupWeights.Size, groupPtr.Count - 1, ErrorMsg.GroupWeight);
        Check.Eq((long)groupPtr[^1], info.NumRow, ErrorMsg.GroupSize + " the number of rows from the data.");
        var output = new float[info.NumRow];
        var curGroup = 0;
        for (long i = 0; i < info.NumRow; i++)
        {
            while (curGroup + 1 < groupPtr.Count && i >= groupPtr[curGroup + 1]) curGroup++;
            output[i] = groupWeights[curGroup];
        }
        return output;
    }

    private static float[] MergeWeights(MetaInfo info, ReadOnlySpan<float> hessian, bool useGroup, int nThreads)
    {
        Check.Eq((long)hessian.Length, info.NumRow);
        var results = new float[hessian.Length];
        var groupPtr = info.GroupPtr;
        var weights = info.Weights;
        float GetWeight(int i) => weights.Empty ? 1.0f : weights[i];
        if (useGroup)
        {
            Check.Ge(groupPtr.Count, 2);
            Check.Eq((long)groupPtr[^1], (long)hessian.Length);
            var curGroup = 0;
            for (var i = 0; i < hessian.Length; i++)
            {
                while (curGroup + 1 < groupPtr.Count && i >= groupPtr[curGroup + 1]) curGroup++;
                results[i] = hessian[i] * GetWeight(curGroup);
            }
        }
        else
        {
            var h = hessian.ToArray();
            Threading.ParallelFor(h.Length, nThreads, Sched.Auto(), i => results[i] = h[i] * GetWeight((int)i));
        }
        return results;
    }

    private float[] Weights(MetaInfo info, float[] hessian)
    {
        if (hessian.Length == 0) return _useGroupInd ? UnrollGroupWeights(info) : info.Weights.ToArray();
        return MergeWeights(info, hessian, _useGroupInd, _nThreads);
    }

    public void PushRowPage(SparsePage page, MetaInfo info, float[]? hessian = null)
    {
        var nColumns = info.NumCol;
        var isDense = info.NumNonZero == info.NumCol * info.NumRow;
        Check.Eq((long)_sketches.Length, nColumns);
        var weights = Weights(info, hessian ?? []);
        if (weights.Length != 0) Check.Eq((long)weights.Length, info.NumRow);
        var batch = new SparsePageAdapterBatch(page.GetView());
        PushRowPageImpl(batch, page.BaseRowId, new OptionalWeights(weights, weights.Length), page.Data.Size, (int)info.NumCol, isDense,
            static _ => true);
    }

    public void PushAdapterBatch<TBatch>(TBatch batch, long baseRowId, MetaInfo info, float missing) where TBatch : IAdapterBatch
    {
        var hWeights = _useGroupInd ? UnrollGroupWeights(info) : info.Weights.ToArray();
        if (!_useGroupInd && hWeights.Length != 0) Check.Eq((long)hWeights.Length, batch.Size, "Invalid size of sample weight.");
        var isValid = new IsValidFunctor(missing);
        var isDense = info.NumNonZero == info.NumCol * info.NumRow;
        Check.That(_columnsSize.Length != 0);
        PushRowPageImpl(batch, baseRowId, new OptionalWeights(hWeights, hWeights.Length), info.NumNonZero, (int)info.NumCol, isDense,
            v => isValid.Valid(v));
    }

    private void PushRowPageImpl<TBatch>(TBatch batch, long baseRowId, OptionalWeights weights, long nnz, int nFeatures, bool isDense,
        Func<float, bool> isValid) where TBatch : IAdapterBatch
    {
        var threadColumnsPtr = LoadBalance(batch, nnz, nFeatures, _nThreads, isValid);
        var ft = _featureTypes;
        Threading.ParallelFor(_nThreads, _nThreads, tidL =>
        {
            var tid = (int)tidL;
            var begin = threadColumnsPtr[tid];
            var end = threadColumnsPtr[tid + 1];
            if (begin < end && end <= nFeatures)
            {
                for (long ridx = 0; ridx < batch.Size; ridx++)
                {
                    var w = weights[ridx + baseRowId];
                    if (isDense)
                    {
                        for (var ii = begin; ii < end; ii++)
                        {
                            var elem = batch.GetElement(ridx, (int)ii);
                            if (!isValid(elem.Value)) continue;
                            if (Categorical.IsCat(ft, ii)) _categories[ii].Add(elem.Value);
                            else _sketches[ii].Push(elem.Value, w);
                        }
                    }
                    else
                    {
                        var size = batch.LineSize(ridx);
                        for (var i = 0; i < size; i++)
                        {
                            var elem = batch.GetElement(ridx, i);
                            if (isValid(elem.Value) && elem.ColumnIdx >= begin && elem.ColumnIdx < end)
                            {
                                if (Categorical.IsCat(ft, elem.ColumnIdx)) _categories[elem.ColumnIdx].Add(elem.Value);
                                else _sketches[elem.ColumnIdx].Push(elem.Value, w);
                            }
                        }
                    }
                }
            }
        });
    }

    public static long[] CalcColumnSize<TBatch>(TBatch batch, int nColumns, int nThreads, Func<float, bool> isValid)
        where TBatch : IAdapterBatch
    {
        var tloc = new long[nThreads][];
        for (var t = 0; t < nThreads; t++) tloc[t] = new long[nColumns];
        Threading.ParallelFor(batch.Size, nThreads, (i, tid) =>
        {
            var local = tloc[tid];
            var size = batch.LineSize(i);
            for (var j = 0; j < size; j++)
            {
                var elem = batch.GetElement(i, j);
                if (isValid(elem.Value)) local[elem.ColumnIdx]++;
            }
        });
        var entries = tloc[0];
        for (var t = 1; t < nThreads; t++)
            for (var j = 0; j < nColumns; j++) entries[j] += tloc[t][j];
        return entries;
    }

    private static uint[] LoadBalance<TBatch>(TBatch batch, long nnz, int nColumns, int nthreads, Func<float, bool> isValid)
        where TBatch : IAdapterBatch
    {
        var totalEntries = nnz;
        var entriesPerThread = StringUtils.DivRoundUp(totalEntries, nthreads);
        var entriesPerColumns = CalcColumnSize(batch, nColumns, nthreads, isValid);
        var colsPtr = new uint[nthreads + 1];
        long count = 0;
        var currentThread = 1;
        foreach (var col in entriesPerColumns)
        {
            colsPtr[currentThread]++;
            count += col;
            Check.Le(count, totalEntries);
            if (count > entriesPerThread)
            {
                currentThread++;
                count = 0;
                colsPtr[currentThread] = colsPtr[currentThread - 1];
            }
        }
        for (; currentThread < colsPtr.Length - 1; currentThread++) colsPtr[currentThread + 1] = colsPtr[currentThread];
        return colsPtr;
    }

    public void PushColPage(SparsePage page, MetaInfo info, float[] hessian)
    {
        var weights = Weights(info, hessian);
        Check.Eq((long)weights.Length, info.NumRow);
        var view = page.GetView();
        var ft = _featureTypes;
        Threading.ParallelFor(view.Size, _nThreads, fidx =>
        {
            var column = view[fidx];
            if (Categorical.IsCat(ft, fidx))
            {
                foreach (var c in column) _categories[fidx].Add(c.Fvalue);
                return;
            }
            _sketches[fidx].PushSorted(column, weights, _maxBins);
        });
    }

    private WQSummary[] AllReduce(uint[] numericFeatures)
    {
        var nColumns = (long)_sketches.Length;
        Collective.Communicator.AllreduceMax(ref nColumns);
        Check.Eq(nColumns, (long)_sketches.Length, "Number of columns differs across workers");
        var reduced = new WQSummary[_sketches.Length];
        for (var i = 0; i < reduced.Length; i++) reduced[i] = new WQSummary();
        var numElements = new long[_sketches.Length];
        Threading.ParallelFor(numericFeatures.Length, _nThreads, idx =>
        {
            var fidx = numericFeatures[idx];
            numElements[fidx] = _sketches[fidx].NumElements;
            var cutTarget = SketchSummaryBudget(_maxBins, numElements[fidx]);
            reduced[fidx] = _sketches[fidx].GetSummary(cutTarget);
        });
        if (Collective.Communicator.GetWorldSize() == 1 || numericFeatures.Length == 0) return reduced;
        return DistributedSketch.Reduce(numericFeatures, reduced, numElements, _maxBins);
    }

    private SortedSet<float>[] AllreduceCategories(uint[] categoricalFeatures)
    {
        var reduced = new SortedSet<float>[categoricalFeatures.Length];
        for (var i = 0; i < categoricalFeatures.Length; i++) reduced[i] = new SortedSet<float>(_categories[categoricalFeatures[i]]);
        if (categoricalFeatures.Length == 0 || Collective.Communicator.GetWorldSize() == 1) return reduced;
        return DistributedSketch.ReduceCategories(reduced);
    }

    public HistogramCuts MakeCuts(Context ctx, MetaInfo info)
    {
        _ = ctx;
        _ = info;
        var cuts = new HistogramCuts(_sketches.Length);
        var numeric = new List<uint>();
        var categorical = new List<uint>();
        for (uint fidx = 0; fidx < _sketches.Length; fidx++)
        {
            if (Categorical.IsCat(_featureTypes, fidx)) categorical.Add(fidx);
            else numeric.Add(fidx);
        }
        var reducedNumerical = AllReduce([.. numeric]);
        var reducedCategories = AllreduceCategories([.. categorical]);
        var categoricalIndex = new int[_sketches.Length];
        for (var i = 0; i < categorical.Count; i++) categoricalIndex[categorical[i]] = i;

        var cutPtrs = cuts.Ptrs;
        var maxCat = -1f;
        for (var fid = 0; fid < reducedNumerical.Length; fid++)
        {
            var maxNumBins = Math.Min(reducedNumerical[fid].Size, _maxBins);
            if (Categorical.IsCat(_featureTypes, fid)) AddCategories(reducedCategories[categoricalIndex[fid]], ref maxCat, cuts);
            else AddCutPoints(reducedNumerical[fid], maxNumBins, cuts);
            var cutSize = (uint)cuts.Values.Size;
            Check.Gt(cutSize, cutPtrs[fid]);
            cutPtrs[fid + 1] = cutSize;
        }
        cuts.SetCategorical(_hasCategorical, maxCat);
        return cuts;
    }

    private static void AddCutPoints(WQSummary summary, int maxBin, HistogramCuts cuts)
    {
        foreach (var v in summary.QueryCutValues(maxBin)) cuts.Values.Add(v);
    }

    private static void AddCategories(SortedSet<float> categories, ref float maxCat, HistogramCuts cuts)
    {
        if (categories.Any(Categorical.InvalidCat)) Categorical.InvalidCategory();
        var featureMaxCat = categories.Count == 0 ? 0.0f : categories.Max;
        Categorical.CheckMaxCat(featureMaxCat, categories.Count);
        maxCat = Math.Max(maxCat, featureMaxCat);
        for (var i = 0; i <= Categorical.AsCat(featureMaxCat); i++) cuts.Values.Add(i);
    }
}

/// <summary>Adapter over a SparsePage view (<c>SparsePageAdapterBatch</c>).</summary>
public readonly struct SparsePageAdapterBatch(HostSparsePageView page) : IAdapterBatch
{
    public long Size => page.Size;
    public bool IsRowMajor => true;
    public int LineSize(long line) => (int)(page.Offset[line + 1] - page.Offset[line]);

    public CooTuple GetElement(long line, int idx)
    {
        var e = page.Data[page.Offset[line] + idx];
        return new CooTuple(line, e.Index, e.Fvalue);
    }

    public float[]? Labels => null;
    public float[]? Weights => null;
    public ulong[]? Qid => null;
    public float[]? BaseMargin => null;
}

/// <summary>Distributed merging of sketches (AllreduceV with combine/prune per feature).</summary>
internal static class DistributedSketch
{
    public static WQSummary[] Reduce(uint[] numericFeatures, WQSummary[] reduced, long[] numElements, int maxBins)
    {
        // Serialise [num_elements, n_entries, entries...] per numeric feature and fold them in rank order.
        byte[] Serialize(WQSummary[] r, long[] ne)
        {
            using var ms = new MemoryStream();
            using var w = new BinaryWriter(ms);
            foreach (var f in numericFeatures)
            {
                w.Write(ne[f]);
                w.Write(r[f].Size);
                foreach (var e in r[f].Entries)
                {
                    w.Write(e.RMin);
                    w.Write(e.RMax);
                    w.Write(e.WMin);
                    w.Write(e.Value);
                }
            }
            return ms.ToArray();
        }

        (WQSummary[] R, long[] Ne) Parse(byte[] blob)
        {
            var r = new WQSummary[reduced.Length];
            var ne = new long[reduced.Length];
            using var br = new BinaryReader(new MemoryStream(blob));
            foreach (var f in numericFeatures)
            {
                ne[f] = br.ReadInt64();
                var n = br.ReadInt32();
                var entries = new WQEntry[n];
                for (var i = 0; i < n; i++) entries[i] = new WQEntry(br.ReadSingle(), br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                r[f] = new WQSummary(entries, n);
            }
            return (r, ne);
        }

        var all = Collective.Communicator.AllgatherV(Serialize(reduced, numElements));
        var (acc, accNe) = Parse(all[0]);
        for (var k = 1; k < all.Count; k++)
        {
            var (b, bNe) = Parse(all[k]);
            foreach (var f in numericFeatures)
            {
                var num = accNe[f] + bNe[f];
                var cutTarget = HostSketchContainer.SketchSummaryBudget(maxBins, num);
                var tmp = new WQSummary();
                tmp.Reserve(acc[f].Size + b[f].Size);
                tmp.CopyFrom(acc[f]);
                tmp.SetCombine(b[f]);
                tmp.SetPrune(cutTarget);
                acc[f] = tmp;
                accNe[f] = num;
            }
        }
        for (var i = 0; i < reduced.Length; i++) acc[i] ??= reduced[i];
        return acc;
    }

    public static SortedSet<float>[] ReduceCategories(SortedSet<float>[] local)
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, true))
        {
            foreach (var s in local)
            {
                w.Write(s.Count);
                foreach (var v in s) w.Write(v);
            }
        }
        var all = Collective.Communicator.AllgatherV(ms.ToArray());
        var result = new SortedSet<float>[local.Length];
        for (var i = 0; i < local.Length; i++) result[i] = [];
        foreach (var blob in all)
        {
            using var br = new BinaryReader(new MemoryStream(blob));
            for (var i = 0; i < local.Length; i++)
            {
                var n = br.ReadInt32();
                for (var k = 0; k < n; k++) result[i].Add(br.ReadSingle());
            }
        }
        return result;
    }
}
