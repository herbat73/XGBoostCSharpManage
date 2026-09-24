// Port of src/tree/param.h/.cc and src/common/param_array.h/.cc.
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using XGBoost.Common;

namespace XGBoost.Tree;

public enum TreeGrowPolicy
{
    DepthWise = 0,
    LossGuide = 1,
}

public enum SamplingMethod
{
    Uniform = 0,
    GradientBased = 1,
}

/// <summary>Training parameters for regression trees, <c>tree::TrainParam</c>.</summary>
public sealed class TrainParam : XGBoostParameter<TrainParam>
{
    public const double DftSparseThreshold = 0.2;

    public float LearningRate;
    public float MinSplitLoss;
    public int MaxDepth;
    public int MaxLeaves;
    public int MaxBin;
    public TreeGrowPolicy GrowPolicy;
    public uint MaxCatToOnehot = uint.MaxValue;
    public int MaxCatThreshold = 64;
    public float MinChildWeight;
    public float RegLambda;
    public float RegAlpha;
    public float MaxDeltaStep;
    public float Subsample;
    public SamplingMethod SamplingMethod;
    public float ColsampleBynode;
    public float ColsampleBylevel;
    public float ColsampleBytree;
    public bool RefreshLeaf;
    public int[] MonotoneConstraints = [];
    public string InteractionConstraints = "";
    public double SparseThreshold = DftSparseThreshold;

    protected override void Declare(ParamManager<TrainParam> m)
    {
        Field(m, "learning_rate", p => p.LearningRate, (p, v) => p.LearningRate = v).SetLowerBound(0.0f).SetDefault(0.3f)
            .Describe("Learning rate(step size) of update.");
        Field(m, "min_split_loss", p => p.MinSplitLoss, (p, v) => p.MinSplitLoss = v).SetLowerBound(0.0f).SetDefault(0.0f)
            .Describe("Minimum loss reduction required to make a further partition.");
        Field(m, "max_depth", p => p.MaxDepth, (p, v) => p.MaxDepth = v).SetLowerBound(0).SetDefault(6)
            .Describe("Maximum depth of the tree; 0 indicates no limit; a limit is required for depthwise policy");
        Field(m, "max_leaves", p => p.MaxLeaves, (p, v) => p.MaxLeaves = v).SetLowerBound(0).SetDefault(0)
            .Describe("Maximum number of leaves; 0 indicates no limit.");
        Field(m, "max_bin", p => p.MaxBin, (p, v) => p.MaxBin = v).SetLowerBound(2).SetDefault(256)
            .Describe("if using histogram-based algorithm, maximum number of bins per feature");
        EnumField<TreeGrowPolicy>(m, "grow_policy", p => p.GrowPolicy, (p, v) => p.GrowPolicy = v).SetDefault(TreeGrowPolicy.DepthWise)
            .AddEnum("depthwise", TreeGrowPolicy.DepthWise).AddEnum("lossguide", TreeGrowPolicy.LossGuide)
            .Describe("Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change. (cf. LightGBM)");
        Field(m, "max_cat_to_onehot", p => p.MaxCatToOnehot, (p, v) => p.MaxCatToOnehot = v).SetDefault(uint.MaxValue).SetLowerBound(1)
            .Describe("Maximum number of categories to use one-hot encoding based split.");
        Field(m, "max_cat_threshold", p => p.MaxCatThreshold, (p, v) => p.MaxCatThreshold = v).SetDefault(64).SetLowerBound(1)
            .Describe("Maximum number of categories considered for split. Used only by partition-basedsplits.");
        Field(m, "min_child_weight", p => p.MinChildWeight, (p, v) => p.MinChildWeight = v).SetLowerBound(0.0f).SetDefault(1.0f)
            .Describe("Minimum sum of instance weight(hessian) needed in a child.");
        Field(m, "reg_lambda", p => p.RegLambda, (p, v) => p.RegLambda = v).SetLowerBound(0.0f).SetDefault(1.0f)
            .Describe("L2 regularization on leaf weight");
        Field(m, "reg_alpha", p => p.RegAlpha, (p, v) => p.RegAlpha = v).SetLowerBound(0.0f).SetDefault(0.0f)
            .Describe("L1 regularization on leaf weight");
        Field(m, "max_delta_step", p => p.MaxDeltaStep, (p, v) => p.MaxDeltaStep = v).SetLowerBound(0.0f).SetDefault(0.0f)
            .Describe("Maximum delta step we allow each tree's weight estimate to be. If the value is set to 0, it means there is no constraint");
        Field(m, "subsample", p => p.Subsample, (p, v) => p.Subsample = v).SetRange(0.0f, 1.0f).SetDefault(1.0f)
            .Describe("Row subsample ratio of training instance.");
        EnumField<SamplingMethod>(m, "sampling_method", p => p.SamplingMethod, (p, v) => p.SamplingMethod = v).SetDefault(SamplingMethod.Uniform)
            .AddEnum("uniform", SamplingMethod.Uniform).AddEnum("gradient_based", SamplingMethod.GradientBased)
            .Describe("Sampling method. 0: select random training instances uniformly. 1: select random training instances with higher probability when the gradient and hessian are larger. (cf. CatBoost)");
        Field(m, "colsample_bynode", p => p.ColsampleBynode, (p, v) => p.ColsampleBynode = v).SetRange(0.0f, 1.0f).SetDefault(1.0f)
            .Describe("Subsample ratio of columns, resample on each node (split).");
        Field(m, "colsample_bylevel", p => p.ColsampleBylevel, (p, v) => p.ColsampleBylevel = v).SetRange(0.0f, 1.0f).SetDefault(1.0f)
            .Describe("Subsample ratio of columns, resample on each level.");
        Field(m, "colsample_bytree", p => p.ColsampleBytree, (p, v) => p.ColsampleBytree = v).SetRange(0.0f, 1.0f).SetDefault(1.0f)
            .Describe("Subsample ratio of columns, resample on each tree construction.");
        Field(m, "refresh_leaf", p => p.RefreshLeaf, (p, v) => p.RefreshLeaf = v).SetDefault(true)
            .Describe("Whether the refresh updater needs to update leaf values.");
        CustomField(m, "monotone_constraints", "std::vector<int>", p => p.MonotoneConstraints, (p, v) => p.MonotoneConstraints = v,
                v => IntTuple.Parse("monotone_constraints", v), IntTuple.Print)
            .SetDefault([]).Describe("Constraint of variable monotonicity");
        Field(m, "interaction_constraints", p => p.InteractionConstraints, (p, v) => p.InteractionConstraints = v).SetDefault("")
            .Describe("Constraints for interaction representing permitted interactions.The constraints must be specified in the form of a nest list,e.g. [[0, 1], [2, 3, 4]], where each inner list is a group ofindices of features that are allowed to interact with each other.See tutorial for more information");
        Field(m, "sparse_threshold", p => p.SparseThreshold, (p, v) => p.SparseThreshold = v).SetRange(0, 1.0).SetDefault(DftSparseThreshold)
            .Describe("percentage threshold for treating a feature as sparse");
        Alias(m, "reg_lambda", "lambda");
        Alias(m, "reg_alpha", "alpha");
        Alias(m, "min_split_loss", "gamma");
        Alias(m, "learning_rate", "eta");
    }

    public bool NeedPrune(double lossChg, int depth) => lossChg < MinSplitLoss || (MaxDepth != 0 && depth > MaxDepth);

    public int MaxNodes()
    {
        if (MaxDepth == 0 && MaxLeaves == 0) Check.Fail("Max leaves and max depth cannot both be unconstrained.");
        int nNodes;
        if (MaxLeaves > 0)
        {
            nNodes = MaxLeaves * 2 - 1;
        }
        else
        {
            Check.Le(MaxDepth, 30, "max_depth can not be greater than 30 as that might generate 2^31 - 1nodes.");
            nNodes = (1 << MaxDepth) + ((1 << MaxDepth) - 1);
        }
        Check.Gt(nNodes, 0);
        return nNodes;
    }

    public bool HasMonotone() => MonotoneConstraints.Any(v => v != 0);
}

/// <summary>Stream format of <c>std::vector&lt;int&gt;</c> parameters: a python style tuple.</summary>
public static class IntTuple
{
    public static string Print(int[] t)
    {
        var sb = new StringBuilder("(");
        for (var i = 0; i < t.Length; i++)
        {
            if (i != 0) sb.Append(',');
            sb.Append(t[i].ToString(CultureInfo.InvariantCulture));
        }
        if (t.Length == 1) sb.Append(',');
        sb.Append(')');
        return sb.ToString();
    }

    /// <summary>Port of <c>operator&gt;&gt;(istream&amp;, vector&lt;int&gt;&amp;)</c>.</summary>
    public static int[] Parse(string key, string s)
    {
        var p = 0;
        XGBoostException Fail() => new($"Invalid Parameter format for {key} expect std::vector<int> but value='{s}'");
        bool ReadInt(out int v)
        {
            v = 0;
            while (p < s.Length && char.IsWhiteSpace(s[p])) p++;
            var start = p;
            if (p < s.Length && (s[p] == '-' || s[p] == '+')) p++;
            while (p < s.Length && char.IsAsciiDigit(s[p])) p++;
            return p > start && int.TryParse(s.AsSpan(start, p - start), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out v);
        }
        while (true)
        {
            if (p >= s.Length) throw Fail();
            var ch = s[p];
            if (char.IsAsciiDigit(ch))
            {
                if (ReadInt(out var single)) return [single];
                throw Fail();
            }
            p++;
            if (ch == '(') break;
            if (!char.IsWhiteSpace(ch)) throw Fail();
        }
        while (p < s.Length && char.IsWhiteSpace(s[p])) p++;
        if (p < s.Length && s[p] == ')') return [];
        var tmp = new List<int>();
        while (ReadInt(out var idx))
        {
            tmp.Add(idx);
            char ch;
            do
            {
                if (p >= s.Length) throw Fail();
                ch = s[p++];
            } while (char.IsWhiteSpace(ch));
            if (ch == 'L')
            {
                if (p >= s.Length) throw Fail();
                ch = s[p++];
            }
            if (ch == ',')
            {
                while (p < s.Length && char.IsWhiteSpace(s[p])) p++;
                if (p < s.Length && s[p] == ')')
                {
                    p++;
                    break;
                }
            }
            else if (ch == ')')
            {
                break;
            }
            else
            {
                throw Fail();
            }
        }
        return [.. tmp];
    }
}

/// <summary>Float array parameter written as a JSON array, <c>common::ParamArray&lt;float&gt;</c>.</summary>
public static class ParamArray
{
    public static string Print(float[] values)
    {
        var w = Json.Dump(new F32Array((float[])values.Clone()));
        return w;
    }

    public static float[] Parse(string name, string str)
    {
        var s = string.Concat(str.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        if (s.Length == 0) s = str;
        var chars = s.ToCharArray();
        var head = 0;
        while (head < chars.Length && char.IsWhiteSpace(chars[head])) head++;
        if (head < chars.Length && chars[head] == '(') chars[head] = '[';
        var tail = chars.Length - 1;
        while (tail >= 0 && char.IsWhiteSpace(chars[tail])) tail--;
        if (tail >= 0 && chars[tail] == ')') chars[tail] = ']';
        var j = Json.Load(new string(chars));
        if (j is JsonNumber n) return [n.Value];
        if (j is JsonInteger i) return [i.Value];
        var list = new List<float>();
        foreach (var v in j.AsArray)
        {
            if (v is JsonNumber vn) list.Add(vn.Value);
            else if (v is JsonInteger vi) list.Add(vi.Value);
            else Check.Fail($"Invalid type for: `{name}`, expecting one of the: {{Number, Integer}}, got: `{v.TypeStr}`");
        }
        return [.. list];
    }
}

/// <summary>Core gradient statistics in double precision, <c>GradStats</c>.</summary>
public struct GradStats
{
    public double SumGrad;
    public double SumHess;

    public GradStats(double grad, double hess)
    {
        SumGrad = grad;
        SumHess = hess;
    }

    public GradStats(GradientPairPrecise g)
    {
        SumGrad = g.Grad;
        SumHess = g.Hess;
    }

    public GradStats(GradientPair g)
    {
        SumGrad = g.Grad;
        SumHess = g.Hess;
    }

    public readonly double GetGrad() => SumGrad;
    public readonly double GetHess() => SumHess;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(GradientPair p) => Add(p.Grad, p.Hess);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(GradStats b)
    {
        SumGrad += b.SumGrad;
        SumHess += b.SumHess;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(double grad, double hess)
    {
        SumGrad += grad;
        SumHess += hess;
    }

    public void SetSubstract(GradStats a, GradStats b)
    {
        SumGrad = a.SumGrad - b.SumGrad;
        SumHess = a.SumHess - b.SumHess;
    }

    public readonly bool Empty => SumHess == 0.0;

    public override readonly string ToString() => $"{Format.Stream(SumGrad)}/{Format.Stream(SumHess)}";
}

/// <summary>Best split found for a node, <c>SplitEntryContainer&lt;GradStats&gt;</c>.</summary>
public sealed class SplitEntry
{
    public float LossChg;
    public uint SIndex;
    public float SplitValue;
    public List<uint> CatBits = [];
    public bool IsCat;
    public GradStats LeftSum;
    public GradStats RightSum;

    public uint SplitIndex => SIndex & ((1U << 31) - 1U);
    public bool DefaultLeft => (SIndex >> 31) != 0;

    public bool NeedReplace(float newLossChg, uint splitIndex)
    {
        if (float.IsInfinity(newLossChg)) return false;
        if (SplitIndex <= splitIndex) return newLossChg > LossChg;
        return !(LossChg > newLossChg);
    }

    public bool Update(SplitEntry e)
    {
        if (!NeedReplace(e.LossChg, e.SplitIndex)) return false;
        LossChg = e.LossChg;
        SIndex = e.SIndex;
        SplitValue = e.SplitValue;
        IsCat = e.IsCat;
        CatBits = [.. e.CatBits];
        LeftSum = e.LeftSum;
        RightSum = e.RightSum;
        return true;
    }

    public bool Update(float newLossChg, uint splitIndex, float newSplitValue, bool defaultLeft, bool isCat, GradStats leftSum,
        GradStats rightSum)
    {
        if (!NeedReplace(newLossChg, splitIndex)) return false;
        LossChg = newLossChg;
        if (defaultLeft) splitIndex |= 1U << 31;
        SIndex = splitIndex;
        SplitValue = newSplitValue;
        IsCat = isCat;
        LeftSum = leftSum;
        RightSum = rightSum;
        return true;
    }

    public SplitEntry Clone() => new()
    {
        LossChg = LossChg,
        SIndex = SIndex,
        SplitValue = SplitValue,
        CatBits = [.. CatBits],
        IsCat = IsCat,
        LeftSum = LeftSum,
        RightSum = RightSum,
    };
}

/// <summary>Best split for a multi-target node, <c>SplitEntryContainer&lt;std::vector&lt;GradientPairPrecise&gt;&gt;</c>.</summary>
public sealed class MultiSplitEntry
{
    public float LossChg;
    public uint SIndex;
    public float SplitValue;
    public List<uint> CatBits = [];
    public bool IsCat;
    public GradientPairPrecise[] LeftSum = [];
    public GradientPairPrecise[] RightSum = [];

    public uint SplitIndex => SIndex & ((1U << 31) - 1U);
    public bool DefaultLeft => (SIndex >> 31) != 0;

    public bool NeedReplace(float newLossChg, uint splitIndex)
    {
        if (float.IsInfinity(newLossChg)) return false;
        if (SplitIndex <= splitIndex) return newLossChg > LossChg;
        return !(LossChg > newLossChg);
    }

    public bool Update(MultiSplitEntry e)
    {
        if (!NeedReplace(e.LossChg, e.SplitIndex)) return false;
        LossChg = e.LossChg;
        SIndex = e.SIndex;
        SplitValue = e.SplitValue;
        IsCat = e.IsCat;
        CatBits = [.. e.CatBits];
        LeftSum = (GradientPairPrecise[])e.LeftSum.Clone();
        RightSum = (GradientPairPrecise[])e.RightSum.Clone();
        return true;
    }

    public bool Update(float newLossChg, uint splitIndex, float newSplitValue, bool defaultLeft, bool isCat,
        ReadOnlySpan<GradientPairPrecise> leftSum, ReadOnlySpan<GradientPairPrecise> rightSum)
    {
        if (!NeedReplace(newLossChg, splitIndex)) return false;
        LossChg = newLossChg;
        if (defaultLeft) splitIndex |= 1U << 31;
        SIndex = splitIndex;
        SplitValue = newSplitValue;
        IsCat = isCat;
        LeftSum = leftSum.ToArray();
        RightSum = rightSum.ToArray();
        return true;
    }
}

/// <summary>Gain and weight formulas from src/tree/param.h.</summary>
public static class SplitMath
{
    public static bool IsValidSplit(TrainParam p, double leftHess, double rightHess) =>
        leftHess > 0.0 && rightHess > 0.0 && leftHess >= p.MinChildWeight && rightHess >= p.MinChildWeight;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ThresholdL1(double sumGrad, float alpha)
    {
        if (sumGrad > +alpha) return sumGrad - alpha;
        if (sumGrad < -alpha) return sumGrad + alpha;
        return 0.0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ThresholdL1(float sumGrad, float alpha)
    {
        if (sumGrad > +alpha) return sumGrad - alpha;
        if (sumGrad < -alpha) return sumGrad + alpha;
        return 0.0f;
    }

    public static double CalcGainGivenWeight(TrainParam p, double sumGrad, double sumHess, double w) =>
        -(2.0 * sumGrad * w + (sumHess + p.RegLambda) * XMath.Sqr(w) + 2.0 * p.RegAlpha * Math.Abs(w));

    public static float CalcGainGivenWeight(TrainParam p, float sumGrad, float sumHess, float w) =>
        -(2.0f * sumGrad * w + (sumHess + p.RegLambda) * XMath.Sqr(w) + 2.0f * p.RegAlpha * MathF.Abs(w));

    public static double CalcWeight(TrainParam p, double sumGrad, double sumHess)
    {
        if (sumHess <= 0.0) return 0.0;
        var dw = -ThresholdL1(sumGrad, p.RegAlpha) / (sumHess + p.RegLambda);
        if (p.MaxDeltaStep != 0.0f && Math.Abs(dw) > p.MaxDeltaStep) dw = Math.CopySign((double)p.MaxDeltaStep, dw);
        return dw;
    }

    public static float CalcWeight(TrainParam p, float sumGrad, float sumHess)
    {
        if (sumHess <= 0.0) return 0.0f;
        var dw = -ThresholdL1(sumGrad, p.RegAlpha) / (sumHess + p.RegLambda);
        // ::fabs and ::copysign operate in double.
        if (p.MaxDeltaStep != 0.0f && Math.Abs((double)dw) > p.MaxDeltaStep) dw = (float)Math.CopySign((double)p.MaxDeltaStep, dw);
        return dw;
    }

    public static double CalcGain(TrainParam p, double sumGrad, double sumHess)
    {
        if (sumHess <= 0.0) return 0.0;
        if (p.MaxDeltaStep == 0.0f)
        {
            if (p.RegAlpha == 0.0f) return XMath.Sqr(sumGrad) / (sumHess + p.RegLambda);
            return XMath.Sqr(ThresholdL1(sumGrad, p.RegAlpha)) / (sumHess + p.RegLambda);
        }
        var w = CalcWeight(p, sumGrad, sumHess);
        return CalcGainGivenWeight(p, sumGrad, sumHess, w);
    }

    public static float CalcGain(TrainParam p, float sumGrad, float sumHess)
    {
        if (sumHess <= 0.0) return 0.0f;
        if (p.MaxDeltaStep == 0.0f)
        {
            if (p.RegAlpha == 0.0f) return XMath.Sqr(sumGrad) / (sumHess + p.RegLambda);
            return XMath.Sqr(ThresholdL1(sumGrad, p.RegAlpha)) / (sumHess + p.RegLambda);
        }
        var w = CalcWeight(p, sumGrad, sumHess);
        return CalcGainGivenWeight(p, sumGrad, sumHess, w);
    }

    public static double CalcGain(TrainParam p, GradStats s) => CalcGain(p, s.SumGrad, s.SumHess);

    public static float CalcWeight(TrainParam p, GradientPair g) => CalcWeight(p, g.Grad, g.Hess);

    /// <summary><c>ParseInteractionConstraint</c>.</summary>
    public static List<List<uint>> ParseInteractionConstraint(string constraint)
    {
        var all = Json.Load(constraint).AsArray;
        var output = new List<List<uint>>(all.Count);
        foreach (var set in all)
        {
            var o = new List<uint>();
            foreach (var v in set.AsArray)
            {
                if (v is JsonInteger i)
                {
                    o.Add((uint)i.Value);
                }
                else if (v is JsonNumber n)
                {
                    double d = n.Value;
                    Check.Eq(Math.Floor(d), d, "Found floating point number in interaction constraints");
                    o.Add((uint)d);
                }
                else
                {
                    Check.Fail($"Unknown value type for interaction constraint:{v.TypeStr}");
                }
            }
            output.Add(o);
        }
        return output;
    }
}
