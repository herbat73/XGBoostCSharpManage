// Port of src/common/ranking_utils.h/.cc.
using System.Globalization;

namespace XGBoost.Common;

public enum PairMethod { TopK = 0, Mean = 1 }

public sealed class LambdaRankParam : XGBoostParameter<LambdaRankParam>, IEquatable<LambdaRankParam>
{
    public const uint NotSet = uint.MaxValue;
    private const uint DefaultK = 32;
    private const uint DefaultSamplePairs = 1;

    public PairMethod LambdarankPairMethod = PairMethod.TopK;
    public ulong LambdarankNumPairPerSample = NotSet;
    public bool LambdarankUnbiased;
    public bool LambdarankNormalization = true;
    public bool LambdarankScoreNormalization = true;
    public double LambdarankBiasNorm = 1.0;
    public bool NdcgExpGain = true;

    protected override void Declare(ParamManager<LambdaRankParam> m)
    {
        EnumField<PairMethod>(m, "lambdarank_pair_method", p => p.LambdarankPairMethod, (p, v) => p.LambdarankPairMethod = v)
            .SetDefault(PairMethod.TopK).AddEnum("mean", PairMethod.Mean).AddEnum("topk", PairMethod.TopK)
            .Describe("Method for constructing pairs.");
        Field(m, "lambdarank_num_pair_per_sample", p => p.LambdarankNumPairPerSample, (p, v) => p.LambdarankNumPairPerSample = v)
            .SetDefault(NotSet).SetLowerBound(1).Describe("Number of pairs for each sample in the list.");
        Field(m, "lambdarank_unbiased", p => p.LambdarankUnbiased, (p, v) => p.LambdarankUnbiased = v).SetDefault(false)
            .Describe("Unbiased lambda mart. Use extended IPW to debias click position");
        Field(m, "lambdarank_normalization", p => p.LambdarankNormalization, (p, v) => p.LambdarankNormalization = v).SetDefault(true)
            .Describe("Whether to normalize the leaf value for lambda rank.");
        Field(m, "lambdarank_score_normalization", p => p.LambdarankScoreNormalization, (p, v) => p.LambdarankScoreNormalization = v)
            .SetDefault(true).Describe("Whether to normalize the delta by prediction score difference.");
        Field(m, "lambdarank_bias_norm", p => p.LambdarankBiasNorm, (p, v) => p.LambdarankBiasNorm = v).SetDefault(1.0).SetLowerBound(0.0)
            .Describe("Lp regularization for unbiased lambdarank.");
        Field(m, "ndcg_exp_gain", p => p.NdcgExpGain, (p, v) => p.NdcgExpGain = v).SetDefault(true)
            .Describe("When set to true, the label gain is 2^rel - 1, otherwise it's rel.");
    }

    public double Regularizer => 1.0 / (1.0 + LambdarankBiasNorm);

    /// <summary>Number of pairs for each sample.</summary>
    public uint NumPair()
    {
        if (LambdarankNumPairPerSample == NotSet)
            return LambdarankPairMethod == PairMethod.Mean ? DefaultSamplePairs : DefaultK;
        return unchecked((uint)LambdarankNumPairPerSample);
    }

    public bool HasTruncation => LambdarankPairMethod == PairMethod.TopK;
    public bool IsMean => LambdarankPairMethod == PairMethod.Mean;
    public uint TopK => HasTruncation ? NumPair() : NotSet;

    public LambdaRankParam Clone() => (LambdaRankParam)MemberwiseClone();

    public bool Equals(LambdaRankParam? that) =>
        that is not null && LambdarankPairMethod == that.LambdarankPairMethod &&
        LambdarankNumPairPerSample == that.LambdarankNumPairPerSample && LambdarankUnbiased == that.LambdarankUnbiased &&
        LambdarankNormalization == that.LambdarankNormalization &&
        LambdarankScoreNormalization == that.LambdarankScoreNormalization &&
        LambdarankBiasNorm == that.LambdarankBiasNorm && NdcgExpGain == that.NdcgExpGain;

    public override bool Equals(object? obj) => obj is LambdaRankParam p && Equals(p);
    public override int GetHashCode() => HashCode.Combine(LambdarankPairMethod, LambdarankNumPairPerSample, LambdarankUnbiased, NdcgExpGain);
}

public static class Ltr
{
    public const int MaxRel = 31;

    public static double CalcDCGGain(uint label) => (double)((1u << (int)label) - 1);

    public static double CalcDCGDiscount(long idx) => 1.0 / Math.Log2(idx + 2.0);

    public static double CalcInvIDCG(double idcg) => idcg == 0.0 ? 0.0 : 1.0 / idcg;

    /// <summary>C++ <c>static_cast&lt;std::uint32_t&gt;(float)</c> as compiled by MSVC for x64 (via int64 truncation).</summary>
    public static uint ToRel(float v) => unchecked((uint)(long)v);

    public static void CheckNDCGLabels(LambdaRankParam p, ReadOnlySpan<float> labels)
    {
        if (p.NdcgExpGain)
        {
            var labelIsInteger = true;
            foreach (var v in labels)
            {
                var l = MathF.Floor(v);
                if (MathF.Abs(l - v) > Constants.RtEps || v < 0.0f)
                {
                    labelIsInteger = false;
                    break;
                }
            }
            Check.That(labelIsInteger, "When using relevance degree as target, label must be either 0 or positive integer.");
            var labelIsValid = true;
            foreach (var v in labels)
            {
                if (ToRel(v) > MaxRel)
                {
                    labelIsValid = false;
                    break;
                }
            }
            Check.That(labelIsValid, $"Relevance degress must be lesser than or equal to {MaxRel} when the exponential NDCG gain function is used. Set `ndcg_exp_gain` to false to use custom DCG gain.");
        }
    }

    public static bool IsBinaryRel(ReadOnlySpan<float> labels)
    {
        foreach (var y in labels)
            if (!(MathF.Abs(y - 1.0f) < Constants.RtEps || MathF.Abs(y - 0.0f) < Constants.RtEps)) return false;
        return true;
    }

    public static void CheckPreLabels(string name, ReadOnlySpan<float> labels) =>
        Check.That(IsBinaryRel(labels), $"{name} can only be used with binary labels.");

    /// <summary><c>ParseMetricName</c>: returns the display name and updates <paramref name="topn"/>/<paramref name="minus"/>.</summary>
    public static string ParseMetricName(string name, string? param, ref uint topn, ref bool minus)
    {
        string outName;
        if (!string.IsNullOrEmpty(param))
        {
            // sscanf("%u[-]?"): leading whitespace, optional sign, digits.
            if (TryScanUnsigned(param, out var value))
            {
                topn = value;
                outName = name + "@" + param;
            }
            else
            {
                outName = name + param;
            }
            if (param[^1] == '-') minus = true;
        }
        else
        {
            outName = name;
        }
        return outName;
    }

    private static bool TryScanUnsigned(string s, out uint value)
    {
        value = 0;
        var i = 0;
        while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
        var neg = false;
        if (i < s.Length && (s[i] == '+' || s[i] == '-'))
        {
            neg = s[i] == '-';
            i++;
        }
        var start = i;
        ulong acc = 0;
        var overflow = false;
        while (i < s.Length && s[i] >= '0' && s[i] <= '9')
        {
            acc = acc * 10 + (ulong)(s[i] - '0');
            if (acc > uint.MaxValue) overflow = true;
            i++;
        }
        if (i == start) return false;
        var v = overflow ? uint.MaxValue : (uint)acc;
        value = neg ? unchecked((uint)-(int)v) : v;
        return true;
    }

    public static string MakeMetricName(string name, uint topn, bool minus)
    {
        var s = topn == LambdaRankParam.NotSet ? name : name + "@" + topn.ToString(CultureInfo.InvariantCulture);
        if (minus) s += "-";
        return s;
    }
}

/// <summary><c>ltr::RankingCache</c>: group layout, prediction ranks and weight normalisation.</summary>
public class RankingCache
{
    private readonly LambdaRankParam _param;
    private readonly uint[] _groupPtr;
    private int[]? _sortedIdxCache;
    protected readonly long MaxGroupSize;
    private readonly double _weightNorm = 1.0;

    public RankingCache(Context ctx, MetaInfo info, LambdaRankParam p)
    {
        _param = p.Clone();
        Check.That(_param.GetInitialised());
        if (info.GroupPtr.Count != 0)
            Check.Eq((long)info.GroupPtr[^1], (long)info.Labels.Size, ErrorMsg.GroupSize + "the size of label.");

        if (info.GroupPtr.Count == 0) _groupPtr = [0, (uint)info.NumRow];
        else _groupPtr = [.. info.GroupPtr];
        for (var i = 1; i < _groupPtr.Length; i++)
        {
            long n = _groupPtr[i] - _groupPtr[i - 1];
            MaxGroupSize = Math.Max(MaxGroupSize, n);
        }
        var sumWeights = 0.0;
        var nGroups = Groups;
        var weight = new OptionalWeights(info.Weights);
        for (var k = 0; k < nGroups; k++) sumWeights += weight[k];
        _weightNorm = nGroups / sumWeights;

        if (!info.Weights.Empty) Check.Eq((long)Groups, (long)info.Weights.Size, ErrorMsg.GroupWeight);
        if (_param.HasTruncation) Check.Ge(_param.NumPair(), 1u);
    }

    public long MaxPositionSize => _param.HasTruncation ? _param.NumPair() : Math.Min(MaxGroupSize, 32L);

    public ReadOnlySpan<uint> DataGroupPtr => _groupPtr;
    public uint[] GroupPtrArray => _groupPtr;
    public LambdaRankParam Param => _param;
    public int Groups => _groupPtr.Length - 1;
    public double WeightNorm => _weightNorm;

    /// <summary>Per-group argsort of the predictions in descending order (group-local indices).</summary>
    public int[] SortedIdx(Context ctx, float[] predt, int n)
    {
        _sortedIdxCache ??= new int[n];
        var rank = _sortedIdxCache;
        Check.Eq(rank.Length, n);
        var gptr = _groupPtr;
        Threading.ParallelFor(Groups, ctx.Threads(), g =>
        {
            var begin = (int)gptr[g];
            var cnt = (int)(gptr[g + 1] - gptr[g]);
            var sorted = StdAlgo.ArgSort<float>(predt.AsSpan(begin, cnt), static (l, r) => l > r);
            sorted.CopyTo(rank, begin);
        });
        return rank;
    }
}

public sealed class NDCGCache : RankingCache
{
    private readonly double[] _discounts;
    private readonly double[] _invIdcg;
    private double[]? _dcg;

    public NDCGCache(Context ctx, MetaInfo info, LambdaRankParam p) : base(ctx, info, p)
    {
        var gptr = GroupPtrArray;
        _discounts = new double[MaxGroupSize];
        for (var i = 0; i < MaxGroupSize; i++) _discounts[i] = Ltr.CalcDCGDiscount(i);
        var nGroups = gptr.Length - 1;
        var labels = info.Labels.HostView().Slice(SliceArg.All, 0).ToArray();
        Ltr.CheckNDCGLabels(Param, labels);
        _invIdcg = new double[nGroups];
        long topk = Param.TopK;
        var expGain = Param.NdcgExpGain;
        var discounts = _discounts;
        var invIdcg = _invIdcg;
        Threading.ParallelFor(nGroups, ctx.Threads(), g =>
        {
            var begin = (int)gptr[g];
            var cnt = (int)(gptr[g + 1] - gptr[g]);
            var gLabels = labels.AsSpan(begin, cnt);
            var sortedIdx = StdAlgo.ArgSort<float>(gLabels, static (l, r) => l > r);
            var idcg = 0.0;
            var n = Math.Min(cnt, topk);
            for (var i = 0; i < n; i++)
            {
                if (expGain) idcg += discounts[i] * Ltr.CalcDCGGain(Ltr.ToRel(gLabels[sortedIdx[i]]));
                else idcg += discounts[i] * gLabels[sortedIdx[i]];
            }
            invIdcg[g] = Ltr.CalcInvIDCG(idcg);
        });
    }

    public double[] InvIDCG => _invIdcg;
    public double[] Discount => _discounts;
    public double[] Dcg => _dcg ??= new double[Groups];
}

public sealed class PreCache : RankingCache
{
    private double[]? _pre;

    public PreCache(Context ctx, MetaInfo info, LambdaRankParam p) : base(ctx, info, p) =>
        Ltr.CheckPreLabels("pre", info.Labels.HostView().Slice(SliceArg.All, 0).ToArray());

    public double[] Pre => _pre ??= new double[Groups];
}

public sealed class MAPCache : RankingCache
{
    private readonly long _nSamples;
    private double[]? _nRel;
    private double[]? _acc;
    private double[]? _map;

    public MAPCache(Context ctx, MetaInfo info, LambdaRankParam p) : base(ctx, info, p)
    {
        _nSamples = info.NumRow;
        Ltr.CheckPreLabels("map", info.Labels.HostView().Slice(SliceArg.All, 0).ToArray());
    }

    public double[] NumRelevant => _nRel ??= new double[_nSamples];
    public double[] Acc => _acc ??= new double[_nSamples];
    public double[] Map => _map ??= new double[Groups];
}
