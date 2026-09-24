// Port of src/common/threading_utils.h/.cc. OpenMP loops become static partitions over a fixed
// number of workers so that per-thread buffers indexed by thread id work the same way.
using System.Runtime.ExceptionServices;

namespace XGBoost.Common;

/// <summary>Half-open range [Begin, End), <c>common::Range1d</c>.</summary>
public readonly record struct Range1d
{
    public Range1d(long begin, long end)
    {
        Check.Lt(begin, end);
        Begin = begin;
        End = end;
    }

    public long Begin { get; }
    public long End { get; }
    public long Size => End - Begin;
}

/// <summary><c>common::BlockedSpace2d</c>: rows of different lengths split into blocks of at most grain size.</summary>
public sealed class BlockedSpace2d
{
    private readonly List<Range1d> _ranges = [];
    private readonly List<long> _firstDimension = [];

    public BlockedSpace2d(long dim1, Func<long, long> sizeDim2, long grainSize)
    {
        for (long i = 0; i < dim1; i++)
        {
            var size = sizeDim2(i);
            var nBlocks = size / grainSize + (size % grainSize != 0 ? 1 : 0);
            for (long b = 0; b < nBlocks; b++)
            {
                var begin = b * grainSize;
                var end = Math.Min(begin + grainSize, size);
                _firstDimension.Add(i);
                _ranges.Add(new Range1d(begin, end));
            }
        }
    }

    public int Size => _ranges.Count;
    public long GetFirstDimension(int i) => _firstDimension[i];
    public Range1d GetRange(int i) => _ranges[i];
}

public enum SchedKind
{
    Auto,
    Dynamic,
    Static,
    Guided,
}

/// <summary>OpenMP schedule, <c>common::Sched</c>.</summary>
public readonly record struct Sched(SchedKind Kind, long Chunk = 0)
{
    public static Sched Auto() => new(SchedKind.Auto);
    public static Sched Dyn(long n = 0) => new(SchedKind.Dynamic, n);
    public static Sched Static(long n = 0) => new(SchedKind.Static, n);
    public static Sched Guided() => new(SchedKind.Guided);
}

public static class Threading
{
    [ThreadStatic] private static bool _inParallel;
    [ThreadStatic] private static int _threadNum;

    /// <summary>Equivalent of <c>omp_get_thread_num()</c> inside a parallel region; 0 outside.</summary>
    public static int ThreadNum => _threadNum;

    /// <summary><c>omp_in_parallel()</c>.</summary>
    public static bool InParallel => _inParallel;

    public static int MaxThreads => Environment.ProcessorCount;

    /// <summary>Port of <c>OmpGetNumThreads</c>.</summary>
    public static int OmpGetNumThreads(int nThreads)
    {
        if (_inParallel) return 1;
        var max = Math.Max(1, MaxThreads);
        if (GlobalConfig.Current.NThread > 0) max = Math.Min(max, GlobalConfig.Current.NThread);
        if (nThreads <= 0) nThreads = max;
        nThreads = Math.Min(nThreads, max);
        return Math.Max(nThreads, 1);
    }

    /// <summary>Runs <paramref name="body"/> once per worker id in [0, nThreads), like <c>#pragma omp parallel</c>.</summary>
    public static void ParallelRegion(int nThreads, Action<int> body)
    {
        Check.Ge(nThreads, 1);
        if (nThreads == 1 || _inParallel)
        {
            RunAs(0, () => body(0));
            return;
        }
        ExceptionDispatchInfo? error = null;
        var threads = new Thread[nThreads - 1];
        var config = GlobalConfig.Current;
        for (var t = 1; t < nThreads; t++)
        {
            var tid = t;
            threads[t - 1] = new Thread(() =>
            {
                GlobalConfig.SetThreadConfig(config);
                try
                {
                    RunAs(tid, () => body(tid));
                }
                catch (Exception e)
                {
                    Interlocked.CompareExchange(ref error, ExceptionDispatchInfo.Capture(e), null);
                }
            }, 16 * 1024 * 1024) { IsBackground = true };
            threads[t - 1].Start();
        }
        try
        {
            RunAs(0, () => body(0));
        }
        catch (Exception e)
        {
            Interlocked.CompareExchange(ref error, ExceptionDispatchInfo.Capture(e), null);
        }
        foreach (var th in threads) th.Join();
        error?.Throw();
    }

    private static void RunAs(int tid, Action a)
    {
        var (oldIn, oldNum) = (_inParallel, _threadNum);
        _inParallel = true;
        _threadNum = tid;
        try
        {
            a();
        }
        finally
        {
            _inParallel = oldIn;
            _threadNum = oldNum;
        }
    }

    /// <summary>Static partition of [0, size) for worker <paramref name="tid"/> of <paramref name="n"/>.</summary>
    public static (long Begin, long End) StaticRange(long size, int n, int tid)
    {
        var q = size / n;
        var r = size % n;
        var begin = tid * q + Math.Min(tid, r);
        var end = begin + q + (tid < r ? 1 : 0);
        return (begin, end);
    }

    /// <summary><c>common::ParallelFor</c> with the thread id passed to the body.</summary>
    public static void ParallelFor(long size, int nThreads, Sched sched, Action<long, int> fn)
    {
        if (size <= 0) return;
        if (nThreads == 1 || _inParallel)
        {
            if (_inParallel)
            {
                for (long i = 0; i < size; i++) fn(i, _threadNum);
            }
            else
            {
                RunAs(0, () =>
                {
                    for (long i = 0; i < size; i++) fn(i, 0);
                });
            }
            return;
        }
        nThreads = (int)Math.Min(nThreads, size);
        if (sched.Kind is SchedKind.Dynamic or SchedKind.Guided)
        {
            long next = 0;
            var chunk = Math.Max(1, sched.Chunk);
            ParallelRegion(nThreads, tid =>
            {
                while (true)
                {
                    var begin = Interlocked.Add(ref next, chunk) - chunk;
                    if (begin >= size) break;
                    var end = Math.Min(begin + chunk, size);
                    for (var i = begin; i < end; i++) fn(i, tid);
                }
            });
            return;
        }
        if (sched.Kind == SchedKind.Static && sched.Chunk > 0)
        {
            var chunk = sched.Chunk;
            ParallelRegion(nThreads, tid =>
            {
                for (var begin = tid * chunk; begin < size; begin += chunk * nThreads)
                {
                    var end = Math.Min(begin + chunk, size);
                    for (var i = begin; i < end; i++) fn(i, tid);
                }
            });
            return;
        }
        ParallelRegion(nThreads, tid =>
        {
            var (begin, end) = StaticRange(size, nThreads, tid);
            for (var i = begin; i < end; i++) fn(i, tid);
        });
    }

    public static void ParallelFor(long size, int nThreads, Action<long, int> fn) =>
        ParallelFor(size, nThreads, Sched.Static(), fn);

    public static void ParallelFor(long size, int nThreads, Action<long> fn) =>
        ParallelFor(size, nThreads, Sched.Static(), (i, _) => fn(i));

    public static void ParallelFor(long size, int nThreads, Sched sched, Action<long> fn) =>
        ParallelFor(size, nThreads, sched, (i, _) => fn(i));

    /// <summary><c>common::ParallelFor2d</c>: blocks split into contiguous chunks per thread.</summary>
    public static void ParallelFor2d(BlockedSpace2d space, int nThreads, Action<long, Range1d> func)
    {
        var nBlocks = space.Size;
        Check.Ge(nThreads, 1);
        ParallelRegion(nThreads, tid =>
        {
            var chunk = nBlocks / nThreads + (nBlocks % nThreads != 0 ? 1 : 0);
            var begin = chunk * tid;
            var end = Math.Min(begin + chunk, nBlocks);
            for (var i = begin; i < end; i++) func(space.GetFirstDimension(i), space.GetRange(i));
        });
    }

    /// <summary><c>common::ParallelFor1d</c>: blocks of <paramref name="blockSize"/> rows.</summary>
    public static void ParallelFor1d(long size, int nThreads, long blockSize, Action<Range1d> fn)
    {
        var nBlocks = StringUtils.DivRoundUp(size, blockSize);
        ParallelFor(nBlocks, nThreads, blockId =>
        {
            var begin = blockId * blockSize;
            var len = Math.Min(size - begin, blockSize);
            fn(new Range1d(begin, begin + len));
        });
    }

    /// <summary><c>common::ParallelForBlock</c>: one block per thread.</summary>
    public static void ParallelForBlock(long size, int nThreads, Action<Range1d> fn)
    {
        var blk = size / nThreads + (size % nThreads > 0 ? 1 : 0);
        ParallelFor(nThreads, nThreads, tid =>
        {
            var b = tid * blk;
            var e = Math.Min((tid + 1) * blk, size);
            if (e <= b) return;
            fn(new Range1d(b, e));
        });
    }

    public const int DefaultMaxThreads = 128;
}
