// Port of src/common/stats.h/.cc and src/tree/fit_stump.h/.cc.
using XGBoost.Collective;

namespace XGBoost.Common;

public static class Stats
{
    /// <summary>Quantile with linear interpolation (NIST method).</summary>
    public static float Quantile(double alpha, ReadOnlySpan<float> values)
    {
        Check.That(alpha >= 0 && alpha <= 1);
        var n = (double)values.Length;
        if (n == 0) return float.NaN;
        var copy = values.ToArray();
        var sortedIdx = new int[values.Length];
        StdAlgo.Iota(sortedIdx);
        StdAlgo.StableSort(sortedIdx.AsSpan(), (l, r) => copy[l] < copy[r]);
        float Val(int i) => copy[sortedIdx[i]];
        if (alpha <= 1 / (n + 1)) return Val(0);
        if (alpha >= n / (n + 1)) return Val(sortedIdx.Length - 1);
        var x = alpha * (n + 1);
        var k = Math.Floor(x) - 1;
        Check.Ge(k, 0.0);
        var d = x - 1 - k;
        var v0 = Val((int)k);
        var v1 = Val((int)k + 1);
        return (float)(v0 + d * (v1 - v0));
    }

    /// <summary>Weighted quantile with a step function (no interpolation).</summary>
    public static float WeightedQuantile(double alpha, ReadOnlySpan<float> values, Func<int, float> weight)
    {
        var n = values.Length;
        if (n == 0) return float.NaN;
        var copy = values.ToArray();
        var sortedIdx = new int[n];
        StdAlgo.Iota(sortedIdx);
        StdAlgo.StableSort(sortedIdx.AsSpan(), (l, r) => copy[l] < copy[r]);
        var cdf = new float[n];
        cdf[0] = weight(sortedIdx[0]);
        for (var i = 1; i < n; i++) cdf[i] = cdf[i - 1] + weight(sortedIdx[i]);
        var thresh = (float)(cdf[^1] * alpha);
        var idx = StdAlgo.LowerBound<float>(cdf, thresh);
        idx = Math.Min(idx, n - 1);
        return copy[sortedIdx[idx]];
    }

    /// <summary>Median of each column of <paramref name="t"/>, optionally weighted.</summary>
    public static Tensor<float> Median(Tensor<float> t, HostDeviceVector<float> weights)
    {
        var w = new OptionalWeights(weights);
        var view = t.View();
        var cols = t.Shape(1);
        var output = new Tensor<float>([cols]);
        for (long i = 0; i < cols; i++)
        {
            var column = view.Slice(SliceArg.All, i).ToArray();
            float q;
            if (w.Empty)
            {
                q = Quantile(0.5, column);
            }
            else
            {
                Check.Ne(view.Shape(1), 0L);
                q = WeightedQuantile(0.5, column, j => w[j]);
            }
            output[i] = q;
        }
        return output;
    }

    /// <summary>Mean of a vector, with per-thread float partial sums of <c>v(i) / n</c>.</summary>
    public static float Mean(Context ctx, ReadOnlySpan<float> v)
    {
        var n = (float)v.Length;
        var nThreads = ctx.Threads();
        var tloc = new float[nThreads];
        var arr = v.ToArray();
        Threading.ParallelFor(arr.Length, nThreads, (i, tid) => tloc[tid] += arr[i] / n);
        var ret = 0f;
        foreach (var x in tloc) ret += x;
        return ret;
    }

    /// <summary>Mean of each column (<c>SampleMean</c>).</summary>
    public static Tensor<float> SampleMean(Context ctx, Tensor<float> v)
    {
        var output = new Tensor<float>([Math.Max(v.Shape(1), 1)]);
        var hv = v.View();
        Check.That(hv.CContiguous());
        var nSamples = v.Shape(0);
        Span<long> total = [nSamples];
        Communicator.Allreduce(total, Op.Sum);
        nSamples = total[0];
        var nColumns = v.Shape(1);
        var nRows = (double)nSamples;
        var nThreads = ctx.Threads();
        for (long j = 0; j < nColumns; j++)
        {
            var tloc = new double[nThreads];
            var jj = j;
            Threading.ParallelFor(v.Shape(0), nThreads, (i, tid) => tloc[tid] += hv[i, jj] / nRows);
            var mean = 0.0;
            foreach (var x in tloc) mean += x;
            output[j] = (float)mean;
        }
        return output;
    }

    /// <summary>Weighted mean of each column (<c>WeightedSampleMean</c>).</summary>
    public static Tensor<float> WeightedSampleMean(Context ctx, Tensor<float> v, HostDeviceVector<float> w)
    {
        var output = new Tensor<float>([Math.Max(v.Shape(1), 1)]);
        Check.Eq(v.Shape(0), (long)w.Size);
        var hv = v.View();
        var hw = w.ToArray();
        var sumW = 0.0;
        foreach (var x in hw) sumW += x;
        Check.Gt(sumW, 0.0, "weights must contain at least one non-zero value.");
        var nThreads = ctx.Threads();
        for (long j = 0; j < v.Shape(1); j++)
        {
            var tloc = new double[nThreads];
            var jj = j;
            Threading.ParallelFor(v.Shape(0), nThreads, (i, tid) => tloc[tid] += hv[i, jj] / sumW * hw[i]);
            var mean = 0.0;
            foreach (var x in tloc) mean += x;
            output[j] = (float)mean;
        }
        return output;
    }

    // ---- src/tree/fit_stump.h/.cc ------------------------------------------------------------

    public static double CalcUnregularizedWeight(double sumGrad, double sumHess) =>
        -sumGrad / Math.Max(sumHess, (double)Constants.RtEps);

    /// <summary>Sums gradients for each target with per-thread double accumulators.</summary>
    public static GradientPairPrecise[] SumGradients(Context ctx, TensorView<GradientPair> gpair)
    {
        var nTargets = gpair.Shape(1);
        var nThreads = ctx.Threads();
        var tloc = new GradientPairPrecise[nThreads, nTargets];
        Threading.ParallelFor(gpair.Shape(0), nThreads, (i, tid) =>
        {
            for (long t = 0; t < nTargets; t++) tloc[tid, t] += new GradientPairPrecise(gpair[i, t]);
        });
        var sum = new GradientPairPrecise[nTargets];
        for (long j = 0; j < nTargets; j++) sum[j] = tloc[0, j];
        for (var i = 1; i < nThreads; i++)
            for (long j = 0; j < nTargets; j++) sum[j] += tloc[i, j];
        return sum;
    }

    /// <summary><c>FitStump</c>: intercept estimate <c>-sum(grad) / sum(hess)</c> per target.</summary>
    public static Tensor<float> FitStump(Context ctx, Tensor<GradientPair> gpair, uint nTargets)
    {
        var output = new Tensor<float>([nTargets]);
        var view = gpair.View();
        Check.Eq((long)nTargets, view.Shape(1));
        var sum = SumGradients(ctx, view);
        Communicator.Allreduce(sum);
        for (var i = 0; i < sum.Length; i++) output[i] = (float)CalcUnregularizedWeight(sum[i].Grad, sum[i].Hess);
        return output;
    }
}
