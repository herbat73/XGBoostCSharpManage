// Ports of src/tree/hist/hist_param.h/.cc, hist/sampler.h/.cc, hist/expand_entry.h, driver.h,
// constraints.h/.cc and split_evaluator.h.
using XGBoost.Collective;

namespace XGBoost.Tree;

public sealed class HistMakerTrainParam : XGBoostParameter<HistMakerTrainParam>
{
    private const ulong NotSet = ulong.MaxValue;
    public const ulong CpuDefaultNodes = 1UL << 16;

    public ulong MaxCachedHistNode = NotSet;
    public bool DebugSynchronize;

    protected override void Declare(ParamManager<HistMakerTrainParam> m)
    {
        Field(m, "debug_synchronize", p => p.DebugSynchronize, (p, v) => p.DebugSynchronize = v).SetDefault(false)
            .Describe("Check if all distributed tree are identical after tree construction.");
        Field(m, "max_cached_hist_node", p => p.MaxCachedHistNode, (p, v) => p.MaxCachedHistNode = v).SetDefault(NotSet).SetLowerBound(1)
            .Describe("Maximum number of nodes in histogram cache.");
    }

    public ulong MaxCachedHistNodes() => MaxCachedHistNode != NotSet ? MaxCachedHistNode : CpuDefaultNodes;

    public void CheckTreesSynchronized(Context ctx, RegTree localTree)
    {
        if (!DebugSynchronize) return;
        var model = new JsonObject();
        if (Communicator.GetRank() == 0) localTree.SaveModel(model);
        var bytes = Json.DumpBytes(model, true);
        var refTree = new RegTree();
        refTree.LoadModel(Json.Load(bytes, true));
        Check.That(localTree.Equal(refTree));
    }
}

// ---- Sampling ---------------------------------------------------------------------------------

internal static class RandomReplace
{
    public const ulong Base = 16807;
    public const ulong Mod = 1UL << 63;

    public static LinearCongruential Engine(ulong seed) => new(Base, 0, Mod, seed);

    public static ulong SimpleSkip(ulong exponent, ulong initialSeed, ulong @base, ulong mod)
    {
        Check.Le(exponent, mod);
        ulong result = 1;
        while (exponent > 0)
        {
            if (exponent % 2 == 1) result = unchecked(result * @base) % mod;
            @base = unchecked(@base * @base) % mod;
            exponent >>= 1;
        }
        return unchecked(result * initialSeed) % mod;
    }
}

/// <summary><c>cpu_impl::Sampler</c>: row subsampling of the gradient.</summary>
public sealed class Sampler(TrainParam param)
{
    private const float DefaultMvsLambda = 0.1f;
    private readonly SamplingMethod _samplingMethod = param.SamplingMethod;
    private readonly float _subsample = param.Subsample;
    private bool _isSampling;
    private ulong _initialSeed;

    private static void ParallelSampling(Context ctx, long nSamples, ulong initialSeed, Action<long, long, LinearCongruential> fn)
    {
        var nThreads = ctx.Threads();
        var discardSize = nSamples / nThreads;
        Threading.ParallelFor(nThreads, nThreads, tid =>
        {
            var ibegin = tid * discardSize;
            var iend = tid == nThreads - 1 ? nSamples : ibegin + discardSize;
            var displacedSeed = RandomReplace.SimpleSkip((ulong)ibegin, initialSeed, RandomReplace.Base, RandomReplace.Mod);
            fn(ibegin, iend, RandomReplace.Engine(displacedSeed));
        });
    }

    private static float SamplingProbability(float u, float regAbsGrad)
    {
        if (float.IsInfinity(u)) return 0.0f;
        if (MathF.Abs(u) < Constants.RtEps) u = MathF.CopySign(Constants.RtEps, u);
        return regAbsGrad / u;
    }

    private static GradientPair RescaleGrad(float p, GradientPair g) => p >= 1.0f ? g : g * (1.0f / p);

    private static bool IsSampled(float p, float rnd) => p >= 1.0f || (p > 0.0f && rnd <= p);

    internal static float[] CalcRegAbsGrad(Context ctx, TensorView<GradientPair> gpairs, out float[] thresholds)
    {
        var nSamples = gpairs.Shape(0);
        var nTargets = gpairs.Shape(1);
        var regAbsGrad = new float[nSamples];
        Threading.ParallelFor(nSamples, ctx.Threads(), i =>
        {
            var sumSq = 0.0f;
            for (long t = 0; t < nTargets; ++t)
            {
                var g = gpairs[i, t];
                sumSq += XMath.Sqr(g.Grad) + DefaultMvsLambda * XMath.Sqr(g.Hess);
            }
            regAbsGrad[i] = MathF.Sqrt(sumSq);
        });
        thresholds = new float[nSamples + 1];
        regAbsGrad.CopyTo(thresholds, 0);
        thresholds[^1] = float.MaxValue;
        StdAlgo.Sort<float>(thresholds.AsSpan(0, (int)nSamples), static (a, b) => a < b);
        return regAbsGrad;
    }

    internal static float CalculateThreshold(float[] sortedRag, float[] gradCsum, long nSamples, long sampleRows)
    {
        Check.Ge(nSamples, 1L);
        long lowIdx = 0;
        var highIdx = nSamples - 1;
        while (lowIdx <= highIdx)
        {
            var i = lowIdx + (highIdx - lowIdx) / 2;
            var lower = sortedRag[i];
            var upper = sortedRag[i + 1];
            var nAbove = nSamples - i - 1;
            var denom = (float)sampleRows - nAbove;
            if (denom <= 0)
            {
                lowIdx = i + 1;
                continue;
            }
            var u = gradCsum[i] / denom;
            if (u > lower && u <= upper) return u;
            if (u <= lower) highIdx = i - 1;
            else lowIdx = i + 1;
        }
        if (sampleRows == 0) return float.MaxValue;
        return gradCsum[^1] / sampleRows;
    }

    private static float CalcSamplingInfo(Context ctx, TensorView<GradientPair> gpairs, float subsample, out float[] regAbsGrad)
    {
        var nSamples = gpairs.Shape(0);
        var sampleRows = (long)(nSamples * subsample);
        regAbsGrad = CalcRegAbsGrad(ctx, gpairs, out var thresholds);
        var gradCsum = new float[nSamples];
        var acc = 0.0f;
        for (var i = 0; i < nSamples; ++i)
        {
            acc = i == 0 ? thresholds[0] : acc + thresholds[i];
            gradCsum[i] = acc;
        }
        return CalculateThreshold(thresholds, gradCsum, nSamples, sampleRows);
    }

    private static void ForEachGradientSample(Context ctx, long nSamples, float[] regAbsGrad, float threshold, ulong initialSeed,
        Action<long, float, bool> fn)
    {
        ParallelSampling(ctx, nSamples, initialSeed, (ibegin, iend, eng) =>
        {
            for (var i = ibegin; i < iend; ++i)
            {
                var p = SamplingProbability(threshold, regAbsGrad[i]);
                var rnd = StdRandom.UniformFloat(eng, 0.0f, 1.0f);
                fn(i, p, IsSampled(p, rnd));
            }
        });
    }

    private static void UniformSample(Context ctx, TensorView<GradientPair> output, float subsample, ulong initialSeed)
    {
        var nSamples = output.Shape(0);
        var nTargets = output.Shape(1);
        Check.Ge(nTargets, 1L);
        ParallelSampling(ctx, nSamples, initialSeed, (ibegin, iend, eng) =>
        {
            for (var i = ibegin; i < iend; ++i)
            {
                if (!StdRandom.Bernoulli(eng, subsample))
                    for (long t = 0; t < nTargets; ++t) output[i, t] = default;
            }
        });
    }

    private static void ZeroGradientPairs(TensorView<GradientPair> g)
    {
        for (long i = 0; i < g.Size; ++i) g.Flat(i) = default;
    }

    public void Sample(Context ctx, TensorView<GradientPair> output)
    {
        Check.That(output.Contiguous());
        var nSamples = output.Shape(0);
        var sampleRows = (long)(nSamples * _subsample);
        if (sampleRows >= nSamples || nSamples == 0)
        {
            _isSampling = false;
            return;
        }
        _isSampling = true;
        _initialSeed = ctx.Rng.NextUInt();
        switch (_samplingMethod)
        {
            case SamplingMethod.Uniform:
                UniformSample(ctx, output, _subsample, _initialSeed);
                break;
            case SamplingMethod.GradientBased:
            {
                if (sampleRows == 0)
                {
                    ZeroGradientPairs(output);
                    break;
                }
                var threshold = CalcSamplingInfo(ctx, output, _subsample, out var regAbsGrad);
                var nTargets = output.Shape(1);
                ForEachGradientSample(ctx, nSamples, regAbsGrad, threshold, _initialSeed, (i, p, isSampled) =>
                {
                    for (long t = 0; t < nTargets; ++t) output[i, t] = isSampled ? RescaleGrad(p, output[i, t]) : default;
                });
                break;
            }
            default:
                Check.Fail($"Unknown sampling method: {(int)_samplingMethod}");
                break;
        }
    }

    public void ApplySampling(Context ctx, TensorView<GradientPair> splitGpair, Tensor<GradientPair> valueGpair)
    {
        if (!_isSampling) return;
        Check.Eq(splitGpair.Shape(0), valueGpair.Shape(0));
        var hValue = valueGpair.HostView();
        switch (_samplingMethod)
        {
            case SamplingMethod.Uniform:
                UniformSample(ctx, hValue, _subsample, _initialSeed);
                break;
            case SamplingMethod.GradientBased:
            {
                var sampleRows = (long)(splitGpair.Shape(0) * _subsample);
                if (sampleRows == 0)
                {
                    ZeroGradientPairs(hValue);
                    break;
                }
                var threshold = CalcSamplingInfo(ctx, splitGpair, _subsample, out var regAbsGrad);
                var nTargets = hValue.Shape(1);
                ForEachGradientSample(ctx, hValue.Shape(0), regAbsGrad, threshold, _initialSeed, (i, p, isSampled) =>
                {
                    for (long t = 0; t < nTargets; ++t) hValue[i, t] = isSampled ? RescaleGrad(p, hValue[i, t]) : default;
                });
                break;
            }
            default:
                Check.Fail($"Unknown sampling method: {(int)_samplingMethod}");
                break;
        }
    }
}

// ---- Expand entries and driver ----------------------------------------------------------------

public interface IExpandEntry
{
    int Nid { get; }
    int Depth { get; }
    float LossChange { get; }
}

public sealed class CPUExpandEntry : IExpandEntry
{
    public int Nid;
    public int Depth;
    public SplitEntry Split = new();

    public CPUExpandEntry(int nidx, int depth)
    {
        Nid = nidx;
        Depth = depth;
    }

    public CPUExpandEntry(int nidx, int depth, SplitEntry split) : this(nidx, depth) => Split = split;

    int IExpandEntry.Nid => Nid;
    int IExpandEntry.Depth => Depth;
    public float LossChange => Split.LossChg;

    public CPUExpandEntry Clone() => new(Nid, Depth, Split.Clone());
}

public sealed class MultiExpandEntry : IExpandEntry
{
    public int Nid;
    public int Depth;
    public MultiSplitEntry Split = new();

    public MultiExpandEntry(int nidx, int depth)
    {
        Nid = nidx;
        Depth = depth;
    }

    int IExpandEntry.Nid => Nid;
    int IExpandEntry.Depth => Depth;
    public float LossChange => Split.LossChg;

    public MultiExpandEntry Clone()
    {
        var e = new MultiExpandEntry(Nid, Depth);
        e.Split.LossChg = Split.LossChg;
        e.Split.SIndex = Split.SIndex;
        e.Split.SplitValue = Split.SplitValue;
        e.Split.IsCat = Split.IsCat;
        e.Split.CatBits = [.. Split.CatBits];
        e.Split.LeftSum = (GradientPairPrecise[])Split.LeftSum.Clone();
        e.Split.RightSum = (GradientPairPrecise[])Split.RightSum.Clone();
        return e;
    }
}

/// <summary><c>tree::Driver</c>: priority queue of candidate splits.</summary>
public sealed class Driver<T> where T : class, IExpandEntry
{
    private readonly TrainParam _param;
    private int _numLeaves = 1;
    private readonly int _maxNodeBatchSize;
    private readonly StdPriorityQueue<T> _queue;

    private static bool DepthWise(T lhs, T rhs) => lhs.Nid > rhs.Nid;

    private static bool LossGuide(T lhs, T rhs)
    {
        if (lhs.LossChange == rhs.LossChange) return lhs.Nid > rhs.Nid;
        return lhs.LossChange < rhs.LossChange;
    }

    public Driver(TrainParam param, int maxNodeBatchSize = 256)
    {
        _param = param;
        _maxNodeBatchSize = maxNodeBatchSize;
        _queue = new StdPriorityQueue<T>(param.GrowPolicy == TreeGrowPolicy.DepthWise ? DepthWise : LossGuide);
    }

    public static bool IsValidExpandEntry(T entry, TrainParam param, int numLeaves)
    {
        var lossChg = entry.LossChange;
        if (lossChg <= Constants.RtEps) return false;
        if (lossChg < param.MinSplitLoss) return false;
        if (param.MaxDepth > 0 && entry.Depth == param.MaxDepth) return false;
        if (param.MaxLeaves > 0 && numLeaves == param.MaxLeaves) return false;
        return true;
    }

    public void Push(IEnumerable<T> entries)
    {
        foreach (var e in entries)
            if (e.LossChange > Constants.RtEps) _queue.Push(e);
    }

    public void Push(T e) => _queue.Push(e);

    public bool IsEmpty => _queue.Count == 0;

    public bool IsChildValid(T parentEntry)
    {
        if (_param.MaxDepth > 0 && parentEntry.Depth + 1 >= _param.MaxDepth) return false;
        if (_param.MaxLeaves > 0 && _numLeaves >= _param.MaxLeaves) return false;
        return true;
    }

    public List<T> Pop()
    {
        if (_queue.Count == 0) return [];
        if (_param.GrowPolicy == TreeGrowPolicy.LossGuide)
        {
            var e = _queue.Top;
            _queue.Pop();
            if (IsValidExpandEntry(e, _param, _numLeaves))
            {
                _numLeaves++;
                return [e];
            }
            return [];
        }
        var result = new List<T>();
        var top = _queue.Top;
        var level = top.Depth;
        while (top.Depth == level && _queue.Count != 0 && result.Count < _maxNodeBatchSize)
        {
            _queue.Pop();
            if (IsValidExpandEntry(top, _param, _numLeaves))
            {
                _numLeaves++;
                result.Add(top);
            }
            if (_queue.Count != 0) top = _queue.Top;
        }
        return result;
    }
}

// ---- Interaction constraints ------------------------------------------------------------------

/// <summary><c>FeatureInteractionConstraintHost</c>.</summary>
public sealed class FeatureInteractionConstraintHost
{
    private readonly List<HashSet<uint>> _interactionConstraints = [];
    private List<HashSet<uint>> _nodeConstraints = [];
    private List<HashSet<uint>> _splits = [];
    private string _interactionConstraintStr = "";
    private uint _nFeatures;
    private bool _enabled;

    public void Split(int nodeId, uint featureId, int leftId, int rightId)
    {
        if (!_enabled) return;
        var newSize = Math.Max(leftId, rightId) + 1;
        var featureSplits = new HashSet<uint>(_splits[nodeId]) { featureId };
        while (_splits.Count < newSize) _splits.Add([]);
        _splits[leftId] = featureSplits;
        _splits[rightId] = new HashSet<uint>(featureSplits);
        Check.Ne(newSize, 0);
        while (_nodeConstraints.Count < newSize) _nodeConstraints.Add([]);
        foreach (var fid in featureSplits)
        {
            _nodeConstraints[leftId].Add(fid);
            _nodeConstraints[rightId].Add(fid);
        }
        foreach (var constraint in _interactionConstraints)
        {
            var flag = true;
            foreach (var checkvar in featureSplits)
            {
                if (!constraint.Contains(checkvar))
                {
                    flag = false;
                    break;
                }
            }
            if (flag)
            {
                foreach (var k in constraint)
                {
                    _nodeConstraints[leftId].Add(k);
                    _nodeConstraints[rightId].Add(k);
                }
            }
        }
    }

    public bool Query(int nid, uint fid)
    {
        if (!_enabled) return true;
        return _nodeConstraints[nid].Contains(fid);
    }

    public void Configure(TrainParam param, uint nFeatures)
    {
        if (param.InteractionConstraints.Length == 0)
        {
            _enabled = false;
            return;
        }
        _enabled = true;
        _interactionConstraintStr = param.InteractionConstraints;
        _nFeatures = nFeatures;
        Reset();
    }

    public void Reset()
    {
        if (!_enabled) return;
        List<List<uint>> tmp;
        try
        {
            tmp = SplitMath.ParseInteractionConstraint(_interactionConstraintStr);
        }
        catch (XGBoostException e)
        {
            throw new XGBoostException($"Failed to parse feature interaction constraint:\n{_interactionConstraintStr}\nWith error:\n{e.Message}");
        }
        foreach (var e in tmp) _interactionConstraints.Add([.. e]);
        _nodeConstraints = [[]];
        for (uint i = 0; i < _nFeatures; ++i) _nodeConstraints[0].Add(i);
        _splits = [[]];
    }
}

// ---- Split evaluator (monotone constraints) ---------------------------------------------------

/// <summary><c>TreeEvaluator</c> with its <c>SplitEvaluator</c>; the evaluator view reads the live bounds.</summary>
public sealed class TreeEvaluator
{
    private float[] _lowerBounds = [];
    private float[] _upperBounds = [];
    private readonly int[] _monotone;
    private readonly uint _nTargets;
    private readonly bool _hasConstraint;

    public TreeEvaluator(TrainParam p, uint nFeatures, uint nTargets)
    {
        Check.Gt(nTargets, 0u);
        _nTargets = nTargets;
        _hasConstraint = p.HasMonotone();
        if (p.MonotoneConstraints.Length != 0)
            Check.Le((long)p.MonotoneConstraints.Length, (long)nFeatures,
                "The size of monotone constraint should be less or equal to the number of features.");
        _monotone = new int[nFeatures];
        Array.Copy(p.MonotoneConstraints, _monotone, Math.Min(p.MonotoneConstraints.Length, (int)nFeatures));
        if (_hasConstraint)
        {
            _lowerBounds = Filled(256 * (int)nTargets, -float.MaxValue);
            _upperBounds = Filled(256 * (int)nTargets, float.MaxValue);
        }
    }

    private static float[] Filled(int n, float v)
    {
        var a = new float[n];
        Array.Fill(a, v);
        return a;
    }

    private static float[] Grow(float[] a, int n, float v)
    {
        if (a.Length >= n) return a;
        var b = new float[n];
        a.CopyTo(b, 0);
        Array.Fill(b, v, a.Length, n - a.Length);
        return b;
    }

    private void EnsureBounds(int maxNidx, uint nTargets)
    {
        var nNodes = (long)maxNidx * 2 + 1;
        var n = (int)(nNodes * nTargets);
        _lowerBounds = Grow(_lowerBounds, n, -float.MaxValue);
        _upperBounds = Grow(_upperBounds, n, float.MaxValue);
    }

    public bool HasConstraint => _hasConstraint;
    public int Constraint(uint fidx) => _monotone[fidx];

    public float ApplyBounds(int nidx, uint target, float w)
    {
        if (!_hasConstraint) return w;
        var idx = nidx * _nTargets + target;
        if (w < _lowerBounds[idx]) return _lowerBounds[idx];
        if (w > _upperBounds[idx]) return _upperBounds[idx];
        return w;
    }

    // ---- scalar -------------------------------------------------------------------------------

    /// <summary><c>tree::CalcWeight(param, GradStats)</c>: computed in double, returned as float.</summary>
    public static float CalcWeightRaw(TrainParam p, double sumGrad, double sumHess) => (float)SplitMath.CalcWeight(p, sumGrad, sumHess);

    /// <summary><c>tree::CalcGainGivenWeight&lt;double, float&gt;</c>: Sqr/abs of the float weight are float ops.</summary>
    public static double CalcGainGivenWeight(TrainParam p, double sumGrad, double sumHess, float w) =>
        -(2.0 * sumGrad * w + (sumHess + p.RegLambda) * (double)(w * w) + 2.0 * p.RegAlpha * (double)MathF.Abs(w));

    public float CalcWeight(int nidx, uint target, TrainParam param, GradStats stats) =>
        ApplyBounds(nidx, target, CalcWeightRaw(param, stats.SumGrad, stats.SumHess));

    public float CalcWeight(int nidx, TrainParam param, GradStats stats) => CalcWeight(nidx, 0, param, stats);

    public float CalcWeightCat(TrainParam param, GradStats stats) => CalcWeightRaw(param, stats.SumGrad, stats.SumHess);

    public float CalcWeightCat(TrainParam param, GradientPairPrecise stats) => CalcWeightRaw(param, stats.Grad, stats.Hess);

    public float CalcGainGivenWeightS(TrainParam p, GradStats stats, float w)
    {
        if (stats.SumHess <= 0) return 0.0f;
        return (float)CalcGainGivenWeight(p, stats.SumGrad, stats.SumHess, w);
    }

    public float CalcGain(int nidx, TrainParam p, GradStats stats) => CalcGainGivenWeightS(p, stats, CalcWeight(nidx, p, stats));

    public float CalcSplitGain(TrainParam param, int nidx, uint fidx, GradStats left, GradStats right)
    {
        const float negInf = float.NegativeInfinity;
        if (!SplitMath.IsValidSplit(param, left.SumHess, right.SumHess)) return negInf;
        var constraint = _hasConstraint ? _monotone[fidx] : 0;
        var wleft = CalcWeight(nidx, param, left);
        var wright = CalcWeight(nidx, param, right);
        var gain = CalcGainGivenWeightS(param, left, wleft) + CalcGainGivenWeightS(param, right, wright);
        if (constraint == 0) return gain;
        if (constraint > 0) return wleft <= wright ? gain : negInf;
        return wleft >= wright ? gain : negInf;
    }

    // ---- vector ------------------------------------------------------------------------------

    private float CalcPooledWeight(TrainParam param, int nidx, uint target, GradientPairPrecise left, GradientPairPrecise right)
    {
        var alpha = 2.0f * param.RegAlpha;
        var lambda = 2.0f * param.RegLambda;
        var sumGrad = left.Grad + right.Grad;
        var sumHess = left.Hess + right.Hess;
        double pooled;
        if (sumHess <= 0.0)
        {
            pooled = 0.0;
        }
        else
        {
            var dw = -SplitMath.ThresholdL1(sumGrad, alpha) / (sumHess + lambda);
            if (param.MaxDeltaStep != 0.0f && Math.Abs(dw) > param.MaxDeltaStep) dw = Math.CopySign((double)param.MaxDeltaStep, dw);
            pooled = dw;
        }
        return ApplyBounds(nidx, target, (float)pooled);
    }

    public (float, float) CalcSplitWeights(TrainParam param, int nidx, uint fidx, uint target, GradientPairPrecise left,
        GradientPairPrecise right)
    {
        var wleft = CalcWeight(nidx, target, param, new GradStats(left));
        var wright = CalcWeight(nidx, target, param, new GradStats(right));
        if (!_hasConstraint) return (wleft, wright);
        var constraint = _monotone[fidx];
        var ordered = constraint == 0 || (constraint > 0 && wleft <= wright) || (constraint < 0 && wleft >= wright);
        if (ordered) return (wleft, wright);
        var pooled = CalcPooledWeight(param, nidx, target, left, right);
        return (pooled, pooled);
    }

    public double CalcSplitGain(TrainParam param, int nidx, uint fidx, ReadOnlySpan<GradientPairPrecise> left,
        ReadOnlySpan<GradientPairPrecise> right)
    {
        var nTargets = left.Length;
        double leftHess = 0.0, rightHess = 0.0, gain = 0.0;
        for (var t = 0; t < nTargets; ++t)
        {
            var leftT = left[t];
            var rightT = right[t];
            leftHess += leftT.Hess;
            rightHess += rightT.Hess;
            if (!_hasConstraint)
            {
                gain += SplitMath.CalcGain(param, leftT.Grad, leftT.Hess);
                gain += SplitMath.CalcGain(param, rightT.Grad, rightT.Hess);
            }
        }
        var k = (double)nTargets;
        if (!SplitMath.IsValidSplit(param, leftHess / k, rightHess / k)) return double.NegativeInfinity;
        if (!_hasConstraint) return gain;
        for (var t = 0; t < nTargets; ++t)
        {
            var (lw, rw) = CalcSplitWeights(param, nidx, fidx, (uint)t, left[t], right[t]);
            gain += CalcGainGivenWeight(param, left[t].Grad, left[t].Hess, lw);
            gain += CalcGainGivenWeight(param, right[t].Grad, right[t].Hess, rw);
        }
        return gain;
    }

    public void CalcWeight(int nidx, TrainParam param, ReadOnlySpan<GradientPairPrecise> stats, Span<float> output)
    {
        for (var t = 0; t < stats.Length; ++t) output[t] = CalcWeight(nidx, (uint)t, param, new GradStats(stats[t]));
    }

    public void CalcSplitWeights(TrainParam param, int nidx, uint fidx, ReadOnlySpan<GradientPairPrecise> left,
        ReadOnlySpan<GradientPairPrecise> right, Span<float> leftWeight, Span<float> rightWeight)
    {
        for (var t = 0; t < left.Length; ++t)
        {
            var (lw, rw) = CalcSplitWeights(param, nidx, fidx, (uint)t, left[t], right[t]);
            leftWeight[t] = lw;
            rightWeight[t] = rw;
        }
    }

    public void CalcWeightCat(TrainParam param, ReadOnlySpan<GradientPairPrecise> stats, Span<float> output)
    {
        for (var t = 0; t < stats.Length; ++t) output[t] = CalcWeightCat(param, stats[t]);
    }

    public double CalcGainGivenWeight(TrainParam p, ReadOnlySpan<GradientPairPrecise> stats, ReadOnlySpan<float> weight)
    {
        var gain = 0.0;
        for (var t = 0; t < stats.Length; ++t) gain += CalcGainGivenWeight(p, stats[t].Grad, stats[t].Hess, weight[t]);
        return gain;
    }

    public double CalcGain(int nidx, TrainParam p, ReadOnlySpan<GradientPairPrecise> stats)
    {
        var gain = 0.0;
        for (var t = 0; t < stats.Length; ++t)
        {
            var w = CalcWeight(nidx, (uint)t, p, new GradStats(stats[t]));
            gain += CalcGainGivenWeight(p, stats[t].Grad, stats[t].Hess, w);
        }
        return gain;
    }

    // ---- constraint propagation ---------------------------------------------------------------

    public void AddSplit(int nodeId, int leftId, int rightId, uint f, float leftWeight, float rightWeight)
    {
        if (!_hasConstraint) return;
        EnsureBounds(Math.Max(leftId, rightId), 1);
        var lower = _lowerBounds;
        var upper = _upperBounds;
        lower[leftId] = lower[nodeId];
        upper[leftId] = upper[nodeId];
        lower[rightId] = lower[nodeId];
        upper[rightId] = upper[nodeId];
        var c = _monotone[f];
        var mid = (leftWeight + rightWeight) / 2.0f;
        if (c < 0)
        {
            lower[leftId] = mid;
            upper[rightId] = mid;
        }
        else if (c > 0)
        {
            upper[leftId] = mid;
            lower[rightId] = mid;
        }
    }

    public void AddSplit(int nodeId, int leftId, int rightId, uint f, ReadOnlySpan<float> leftWeight, ReadOnlySpan<float> rightWeight)
    {
        if (!_hasConstraint) return;
        Check.Eq((long)leftWeight.Length, (long)_nTargets);
        Check.Eq((long)rightWeight.Length, (long)_nTargets);
        var nTargets = _nTargets;
        EnsureBounds(Math.Max(leftId, rightId), nTargets);
        var lower = _lowerBounds;
        var upper = _upperBounds;
        for (uint t = 0; t < nTargets; ++t)
        {
            var parentIdx = nodeId * nTargets + t;
            var leftIdx = leftId * nTargets + t;
            var rightIdx = rightId * nTargets + t;
            lower[leftIdx] = lower[parentIdx];
            upper[leftIdx] = upper[parentIdx];
            lower[rightIdx] = lower[parentIdx];
            upper[rightIdx] = upper[parentIdx];
            var mid = leftWeight[(int)t] + 0.5f * (rightWeight[(int)t] - leftWeight[(int)t]);
            var constraint = _monotone[f];
            if (constraint < 0)
            {
                lower[leftIdx] = mid;
                upper[rightIdx] = mid;
            }
            else if (constraint > 0)
            {
                upper[leftIdx] = mid;
                lower[rightIdx] = mid;
            }
        }
    }
}
