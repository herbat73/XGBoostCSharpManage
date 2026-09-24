// Ports of the MSVC 14.29 STL algorithms whose results depend on the implementation: std::sort
// (introsort), std::nth_element, std::partial_sort, and the heap functions behind std::priority_queue.
// Reproducing them keeps the ordering of equivalent elements identical to the native library, which
// in turn keeps floating point accumulation order (and therefore results) identical.
// Also includes common/algorithm.h and common/numeric.h helpers.
using System.Runtime.CompilerServices;

namespace XGBoost.Common;

/// <summary>A strict weak ordering, <c>comp(a, b)</c>.</summary>
public interface IStdLess<in T>
{
    bool Less(T a, T b);
}

public readonly struct FuncLess<T>(Func<T, T, bool> f) : IStdLess<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Less(T a, T b) => f(a, b);
}

public static class StdAlgo
{
    private const int IsortMax = 32;

    // ---- std::sort -------------------------------------------------------------------------

    public static void Sort<T>(Span<T> s, Func<T, T, bool> less) => Sort(s, new FuncLess<T>(less));

    public static void Sort<T>(Span<T> s) where T : IComparable<T> => Sort(s, new ComparableLess<T>());

    public static void Sort<T, TLess>(Span<T> s, TLess less) where TLess : IStdLess<T> =>
        SortUnchecked(s, 0, s.Length, s.Length, less);

    private static void SortUnchecked<T, TLess>(Span<T> s, int first, int last, int ideal, TLess pred) where TLess : IStdLess<T>
    {
        for (;;)
        {
            if (last - first <= IsortMax)
            {
                InsertionSort(s, first, last, pred);
                return;
            }
            if (ideal <= 0)
            {
                MakeHeap(s, first, last, pred);
                SortHeap(s, first, last, pred);
                return;
            }
            var (midFirst, midSecond) = PartitionByMedianGuess(s, first, last, pred);
            ideal = (ideal >> 1) + (ideal >> 2);
            if (midFirst - first < last - midSecond)
            {
                SortUnchecked(s, first, midFirst, ideal, pred);
                first = midSecond;
            }
            else
            {
                SortUnchecked(s, midSecond, last, ideal, pred);
                last = midFirst;
            }
        }
    }

    private static void InsertionSort<T, TLess>(Span<T> s, int first, int last, TLess pred) where TLess : IStdLess<T>
    {
        if (first == last) return;
        for (var mid = first + 1; mid != last; mid++)
        {
            var hole = mid;
            var val = s[mid];
            if (pred.Less(val, s[first]))
            {
                // move_backward(first, mid, mid + 1)
                for (var k = mid; k > first; k--) s[k] = s[k - 1];
                s[first] = val;
            }
            else
            {
                for (var prev = hole - 1; pred.Less(val, s[prev]); hole = prev, prev--) s[hole] = s[prev];
                s[hole] = val;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Swap<T>(Span<T> s, int a, int b) => (s[a], s[b]) = (s[b], s[a]);

    private static void Med3<T, TLess>(Span<T> s, int first, int mid, int last, TLess pred) where TLess : IStdLess<T>
    {
        if (pred.Less(s[mid], s[first])) Swap(s, mid, first);
        if (pred.Less(s[last], s[mid]))
        {
            Swap(s, last, mid);
            if (pred.Less(s[mid], s[first])) Swap(s, mid, first);
        }
    }

    private static void GuessMedian<T, TLess>(Span<T> s, int first, int mid, int last, TLess pred) where TLess : IStdLess<T>
    {
        var count = last - first;
        if (40 < count)
        {
            var step = (count + 1) >> 3;
            var twoStep = step << 1;
            Med3(s, first, first + step, first + twoStep, pred);
            Med3(s, mid - step, mid, mid + step, pred);
            Med3(s, last - twoStep, last - step, last, pred);
            Med3(s, first + step, mid, last - step, pred);
        }
        else
        {
            Med3(s, first, mid, last, pred);
        }
    }

    private static (int, int) PartitionByMedianGuess<T, TLess>(Span<T> s, int first, int last, TLess pred) where TLess : IStdLess<T>
    {
        var mid = first + ((last - first) >> 1);
        GuessMedian(s, first, mid, last - 1, pred);
        var pfirst = mid;
        var plast = pfirst + 1;

        while (first < pfirst && !pred.Less(s[pfirst - 1], s[pfirst]) && !pred.Less(s[pfirst], s[pfirst - 1])) pfirst--;
        while (plast < last && !pred.Less(s[plast], s[pfirst]) && !pred.Less(s[pfirst], s[plast])) plast++;

        var gfirst = plast;
        var glast = pfirst;

        for (;;)
        {
            for (; gfirst < last; gfirst++)
            {
                if (pred.Less(s[pfirst], s[gfirst])) continue;
                if (pred.Less(s[gfirst], s[pfirst])) break;
                if (plast != gfirst)
                {
                    Swap(s, plast, gfirst);
                    plast++;
                }
                else
                {
                    plast++;
                }
            }

            for (; first < glast; glast--)
            {
                if (pred.Less(s[glast - 1], s[pfirst])) continue;
                if (pred.Less(s[pfirst], s[glast - 1])) break;
                if (--pfirst != glast - 1) Swap(s, pfirst, glast - 1);
            }

            if (glast == first && gfirst == last) return (pfirst, plast);

            if (glast == first)
            {
                if (plast != gfirst) Swap(s, pfirst, plast);
                plast++;
                Swap(s, pfirst, gfirst);
                pfirst++;
                gfirst++;
            }
            else if (gfirst == last)
            {
                if (--glast != --pfirst) Swap(s, glast, pfirst);
                Swap(s, pfirst, --plast);
            }
            else
            {
                Swap(s, gfirst, --glast);
                gfirst++;
            }
        }
    }

    // ---- heaps -----------------------------------------------------------------------------

    private static void PushHeapByIndex<T, TLess>(Span<T> s, int first, int hole, int top, T val, TLess pred) where TLess : IStdLess<T>
    {
        for (var idx = (hole - 1) >> 1; top < hole && pred.Less(s[first + idx], val); idx = (hole - 1) >> 1)
        {
            s[first + hole] = s[first + idx];
            hole = idx;
        }
        s[first + hole] = val;
    }

    private static void PopHeapHoleByIndex<T, TLess>(Span<T> s, int first, int hole, int bottom, T val, TLess pred) where TLess : IStdLess<T>
    {
        var top = hole;
        var idx = hole;
        var maxNonLeaf = (bottom - 1) >> 1;
        while (idx < maxNonLeaf)
        {
            idx = 2 * idx + 2;
            if (pred.Less(s[first + idx], s[first + idx - 1])) idx--;
            s[first + hole] = s[first + idx];
            hole = idx;
        }
        if (idx == maxNonLeaf && bottom % 2 == 0)
        {
            s[first + hole] = s[first + bottom - 1];
            hole = bottom - 1;
        }
        PushHeapByIndex(s, first, hole, top, val, pred);
    }

    private static void PopHeapHole<T, TLess>(Span<T> s, int first, int last, int dest, T val, TLess pred) where TLess : IStdLess<T>
    {
        s[dest] = s[first];
        PopHeapHoleByIndex(s, first, 0, last - first, val, pred);
    }

    private static void PopHeapUnchecked<T, TLess>(Span<T> s, int first, int last, TLess pred) where TLess : IStdLess<T>
    {
        if (2 <= last - first)
        {
            --last;
            var val = s[last];
            PopHeapHole(s, first, last, last, val, pred);
        }
    }

    private static void MakeHeap<T, TLess>(Span<T> s, int first, int last, TLess pred) where TLess : IStdLess<T>
    {
        var bottom = last - first;
        for (var hole = bottom >> 1; hole > 0;)
        {
            --hole;
            var val = s[first + hole];
            PopHeapHoleByIndex(s, first, hole, bottom, val, pred);
        }
    }

    private static void SortHeap<T, TLess>(Span<T> s, int first, int last, TLess pred) where TLess : IStdLess<T>
    {
        for (; last - first >= 2; --last) PopHeapUnchecked(s, first, last, pred);
    }

    public static void MakeHeap<T, TLess>(Span<T> s, TLess pred) where TLess : IStdLess<T> => MakeHeap(s, 0, s.Length, pred);

    /// <summary><c>std::push_heap</c>: the last element is pushed onto the heap formed by the others.</summary>
    public static void PushHeap<T, TLess>(Span<T> s, TLess pred) where TLess : IStdLess<T>
    {
        var count = s.Length;
        if (2 <= count)
        {
            var val = s[count - 1];
            PushHeapByIndex(s, 0, count - 1, 0, val, pred);
        }
    }

    /// <summary><c>std::pop_heap</c>: moves the top to the last position and re-heaps the rest.</summary>
    public static void PopHeap<T, TLess>(Span<T> s, TLess pred) where TLess : IStdLess<T> => PopHeapUnchecked(s, 0, s.Length, pred);

    // ---- nth_element / partial_sort -------------------------------------------------------

    public static void NthElement<T>(Span<T> s, int nth, Func<T, T, bool> less) => NthElement(s, nth, new FuncLess<T>(less));

    public static void NthElement<T, TLess>(Span<T> s, int nth, TLess pred) where TLess : IStdLess<T>
    {
        var first = 0;
        var last = s.Length;
        if (nth == last) return;
        while (IsortMax < last - first)
        {
            var (mf, ms) = PartitionByMedianGuess(s, first, last, pred);
            if (ms <= nth) first = ms;
            else if (mf <= nth) return;
            else last = mf;
        }
        InsertionSort(s, first, last, pred);
    }

    public static void PartialSort<T>(Span<T> s, int mid, Func<T, T, bool> less) => PartialSort(s, mid, new FuncLess<T>(less));

    public static void PartialSort<T, TLess>(Span<T> s, int mid, TLess pred) where TLess : IStdLess<T>
    {
        if (mid == 0) return;
        MakeHeap(s, 0, mid, pred);
        for (var next = mid; next < s.Length; next++)
        {
            if (pred.Less(s[next], s[0]))
            {
                var val = s[next];
                PopHeapHole(s, 0, mid, next, val, pred);
            }
        }
        SortHeap(s, 0, mid, pred);
    }

    // ---- stable sort ------------------------------------------------------------------------

    /// <summary><c>std::stable_sort</c>. Any stable algorithm produces the same order.</summary>
    public static void StableSort<T>(Span<T> s, Func<T, T, bool> less) => StableSort(s, new FuncLess<T>(less));

    public static void StableSort<T, TLess>(Span<T> s, TLess pred) where TLess : IStdLess<T>
    {
        if (s.Length <= IsortMax)
        {
            InsertionSort(s, 0, s.Length, pred);
            return;
        }
        var buffer = new T[s.Length];
        MergeSort(s, buffer, 0, s.Length, pred);
    }

    private static void MergeSort<T, TLess>(Span<T> s, T[] buf, int lo, int hi, TLess pred) where TLess : IStdLess<T>
    {
        if (hi - lo <= IsortMax)
        {
            InsertionSort(s, lo, hi, pred);
            return;
        }
        var mid = lo + ((hi - lo) >> 1);
        MergeSort(s, buf, lo, mid, pred);
        MergeSort(s, buf, mid, hi, pred);
        if (!pred.Less(s[mid], s[mid - 1])) return;
        s[lo..hi].CopyTo(buf.AsSpan(lo, hi - lo));
        int i = lo, j = mid, k = lo;
        while (i < mid && j < hi)
        {
            // Take from the right only when strictly less: keeps equal elements in order.
            if (pred.Less(buf[j], buf[i])) s[k++] = buf[j++];
            else s[k++] = buf[i++];
        }
        while (i < mid) s[k++] = buf[i++];
        while (j < hi) s[k++] = buf[j++];
    }

    // ---- searching ---------------------------------------------------------------------------

    public static int LowerBound<T>(ReadOnlySpan<T> s, T value) where T : IComparable<T>
    {
        int lo = 0, count = s.Length;
        while (count > 0)
        {
            var step = count >> 1;
            var mid = lo + step;
            if (s[mid].CompareTo(value) < 0)
            {
                lo = mid + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return lo;
    }

    public static int LowerBound<T>(ReadOnlySpan<T> s, T value, Func<T, T, bool> less)
    {
        int lo = 0, count = s.Length;
        while (count > 0)
        {
            var step = count >> 1;
            var mid = lo + step;
            if (less(s[mid], value))
            {
                lo = mid + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return lo;
    }

    public static int UpperBound<T>(ReadOnlySpan<T> s, T value) where T : IComparable<T>
    {
        int lo = 0, count = s.Length;
        while (count > 0)
        {
            var step = count >> 1;
            var mid = lo + step;
            if (value.CompareTo(s[mid]) >= 0)
            {
                lo = mid + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return lo;
    }

    public static int UpperBound<T>(ReadOnlySpan<T> s, T value, Func<T, T, bool> less)
    {
        int lo = 0, count = s.Length;
        while (count > 0)
        {
            var step = count >> 1;
            var mid = lo + step;
            if (!less(value, s[mid]))
            {
                lo = mid + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return lo;
    }

    // ---- common/algorithm.h, common/numeric.h -----------------------------------------------

    /// <summary><c>common::ArgSort</c> (stable).</summary>
    public static int[] ArgSort<T>(ReadOnlySpan<T> values, Func<T, T, bool> comp)
    {
        var result = new int[values.Length];
        for (var i = 0; i < result.Length; i++) result[i] = i;
        var copy = values.ToArray();
        StableSort(result.AsSpan(), (l, r) => comp(copy[l], copy[r]));
        return result;
    }

    /// <summary><c>common::SegmentId</c>: index of the segment in an indptr that contains <paramref name="idx"/>.</summary>
    public static int SegmentId<T>(ReadOnlySpan<T> indptr, T idx) where T : IComparable<T> => UpperBound(indptr, idx) - 1;

    /// <summary><c>common::RunLengthEncode</c>, input must be sorted.</summary>
    public static List<uint> RunLengthEncode<T>(ReadOnlySpan<T> values) where T : IEquatable<T>
    {
        var result = new List<uint> { 0 };
        for (var i = 1; i < values.Length; i++)
            if (!values[i].Equals(values[i - 1])) result.Add((uint)i);
        if (result[^1] != values.Length) result.Add((uint)values.Length);
        return result;
    }

    public static void Iota(Span<int> s, int value = 0)
    {
        for (var i = 0; i < s.Length; i++) s[i] = i + value;
    }

    public static void Iota(Span<long> s, long value = 0)
    {
        for (var i = 0; i < s.Length; i++) s[i] = i + value;
    }

    public static void Iota(Span<uint> s, uint value = 0)
    {
        for (var i = 0; i < s.Length; i++) s[i] = (uint)i + value;
    }

    private readonly struct ComparableLess<T> : IStdLess<T> where T : IComparable<T>
    {
        public bool Less(T a, T b) => a.CompareTo(b) < 0;
    }
}

/// <summary><c>std::priority_queue</c> with MSVC heap semantics (max-heap for <c>less</c>).</summary>
public sealed class StdPriorityQueue<T>(Func<T, T, bool> less)
{
    private readonly List<T> _c = [];
    private readonly FuncLess<T> _less = new(less);

    public int Count => _c.Count;
    public bool Empty => _c.Count == 0;
    public T Top => _c[0];

    public void Push(T value)
    {
        _c.Add(value);
        StdAlgo.PushHeap(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_c), _less);
    }

    public T Pop()
    {
        var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_c);
        StdAlgo.PopHeap(span, _less);
        var v = _c[^1];
        _c.RemoveAt(_c.Count - 1);
        return v;
    }

    public void Clear() => _c.Clear();
}
