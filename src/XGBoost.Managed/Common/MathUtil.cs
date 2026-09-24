// Port of src/common/math.h, src/common/bitfield.h and src/common/categorical.h.
using System.Runtime.CompilerServices;

namespace XGBoost.Common;

public static class XMath
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sqr(float w) => w * w;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sqr(double w) => w * w;

    /// <summary>Single precision sigmoid with the same overflow guard as the C++ code.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x)
    {
        const float eps = 1e-16f;
        x = Math.Min(-x, 88.7f);
        var denom = MathF.Exp(x) + 1.0f + eps;
        return 1.0f / denom;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sigmoid(double x)
    {
        var denom = Math.Exp(-x) + 1.0;
        return 1.0 / denom;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Logit(float x) => -MathF.Log(1.0f / x - 1.0f);

    public static bool CloseTo(double a, double b) => Math.Abs(a - b) < 1e-6;

    /// <summary>In-place softmax over a float span; the sum is accumulated in double.</summary>
    public static void Softmax(Span<float> v)
    {
        var wmax = v[0];
        for (var i = 1; i < v.Length; i++) wmax = FMaxF(v[i], wmax);
        var wsum = 0.0;
        for (var i = 0; i < v.Length; i++)
        {
            v[i] = MathF.Exp(v[i] - wmax);
            wsum += v[i];
        }
        for (var i = 0; i < v.Length; i++) v[i] /= (float)wsum;
    }

    /// <summary>C <c>fmaxf</c>: returns the non-NaN operand if one is NaN.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FMaxF(float a, float b)
    {
        if (float.IsNaN(a)) return b;
        if (float.IsNaN(b)) return a;
        return a > b ? a : b;
    }

    /// <summary>C <c>hypotf</c>: evaluated in double precision, as the UCRT does.</summary>
    public static float HypotF(float x, float y)
    {
        if (float.IsInfinity(x) || float.IsInfinity(y)) return float.PositiveInfinity;
        if (float.IsNaN(x) || float.IsNaN(y)) return float.NaN;
        double dx = x, dy = y;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary><c>std::max(a, b)</c>: <c>(a &lt; b) ? b : a</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T StdMax<T>(T a, T b) where T : System.Numerics.INumber<T> => a < b ? b : a;

    /// <summary><c>std::min(a, b)</c>: <c>(b &lt; a) ? b : a</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T StdMin<T>(T a, T b) where T : System.Numerics.INumber<T> => b < a ? b : a;

    /// <summary><c>std::clamp(v, lo, hi)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T StdClamp<T>(T v, T lo, T hi) where T : System.Numerics.INumber<T> => v < lo ? lo : (hi < v ? hi : v);

    public static double SoftPlus(double x) => x > 0.0 ? x + Log1P(Math.Exp(-x)) : Log1P(Math.Exp(x));

    public static float SoftPlus(float x) => x > 0.0 ? x + Log1PF(MathF.Exp(-x)) : Log1PF(MathF.Exp(x));

    public static double SoftPlusInv(double x)
    {
        x = Math.Max(x, Constants.RtEps);
        return x + Math.Log(-ExpM1(-x));
    }

    public static float SoftPlusInv(float x)
    {
        x = StdMax(x, Constants.RtEps);
        return x + MathF.Log(-ExpM1F(-x));
    }

    /// <summary><c>std::expm1(float)</c> (fdlibm).</summary>
    public static float ExpM1F(float x) => FdLibm.ExpM1F(x);

    /// <summary><c>std::log1p</c> (fdlibm).</summary>
    public static double Log1P(double x) => FdLibm.Log1P(x);

    /// <summary><c>log1pf</c> (fdlibm).</summary>
    public static float Log1PF(float x) => FdLibm.Log1PF(x);

    /// <summary><c>std::expm1</c> (fdlibm).</summary>
    public static double ExpM1(double x) => FdLibm.ExpM1(x);

    /// <summary><c>std::erf</c> (fdlibm).</summary>
    public static double Erf(double x) => FdLibm.Erf(x);

    /// <summary><c>lgammaf</c> (fdlibm).</summary>
    public static float LogGamma(float x) => FdLibm.LGammaF(x);

    public static int FindMaxIndex(ReadOnlySpan<float> v)
    {
        var maxIt = 0;
        for (var i = 0; i < v.Length; i++)
            if (v[i] > v[maxIt]) maxIt = i;
        return maxIt;
    }

    public static float LogSum(float x, float y) =>
        x < y ? y + MathF.Log(MathF.Exp(x - y) + 1.0f) : x + MathF.Log(MathF.Exp(y - x) + 1.0f);

    public static float LogSum(ReadOnlySpan<float> v)
    {
        var mx = v[0];
        foreach (var x in v) mx = Math.Max(mx, x);
        var sum = 0f;
        foreach (var x in v) sum += MathF.Exp(x - mx);
        return mx + MathF.Log(sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool CheckNan(float x) => float.IsNaN(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool CheckNan(double x) => double.IsNaN(x);
}

/// <summary>Bit field with bits numbered from the most significant bit (<c>LBitField32</c>).</summary>
public readonly ref struct LBitField32
{
    private readonly Span<uint> _bits;

    public LBitField32(Span<uint> bits) => _bits = bits;

    public const int ValueSize = 32;

    public static (int IntPos, int BitPos) ToBitPos(long pos) => pos == 0 ? (0, 0) : ((int)(pos / ValueSize), (int)(pos % ValueSize));

    public static int ComputeStorageSize(long size) => (int)StringUtils.DivRoundUp(size, ValueSize);

    public Span<uint> Bits => _bits;
    public int NumValues => _bits.Length;
    public long Capacity => (long)ValueSize * _bits.Length;

    public void Set(long pos)
    {
        var (ip, bp) = ToBitPos(pos);
        _bits[ip] |= 1u << (ValueSize - bp - 1);
    }

    public void Clear(long pos)
    {
        var (ip, bp) = ToBitPos(pos);
        _bits[ip] &= ~(1u << (ValueSize - bp - 1));
    }

    public bool Check(long pos)
    {
        var (ip, bp) = ToBitPos(pos);
        return (_bits[ip] & (1u << (ValueSize - bp - 1))) != 0;
    }

    public static bool Check(ReadOnlySpan<uint> bits, long pos)
    {
        var (ip, bp) = ToBitPos(pos);
        return (bits[ip] & (1u << (ValueSize - bp - 1))) != 0;
    }

    public void Or(ReadOnlySpan<uint> rhs)
    {
        var n = Math.Min(_bits.Length, rhs.Length);
        for (var i = 0; i < n; i++) _bits[i] |= rhs[i];
    }

    public void And(ReadOnlySpan<uint> rhs)
    {
        var n = Math.Min(_bits.Length, rhs.Length);
        for (var i = 0; i < n; i++) _bits[i] &= rhs[i];
    }
}

/// <summary>Bit field with bits numbered from the least significant bit (<c>RBitField8</c>), used by validity masks.</summary>
public readonly ref struct RBitField8
{
    private readonly ReadOnlySpan<byte> _bits;

    public RBitField8(ReadOnlySpan<byte> bits) => _bits = bits;

    public bool Check(long pos)
    {
        var ip = (int)(pos / 8);
        var bp = (int)(pos % 8);
        return (_bits[ip] & (1 << bp)) != 0;
    }

    public int NumValues => _bits.Length;
    public bool IsNull => _bits.IsEmpty;
}

/// <summary>Port of src/common/categorical.h.</summary>
public static class Categorical
{
    public const int OutOfRangeCat = 16777217 - 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int AsCat(float v) => (int)v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int AsCat(double v) => (int)v;

    public static bool IsCat(ReadOnlySpan<FeatureType> ft, long fidx) => !ft.IsEmpty && ft[(int)fidx] == FeatureType.Categorical;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool InvalidCat(float cat) => cat < 0 || cat >= OutOfRangeCat;

    /// <summary>Returns true when the value goes to the right (not in the category set).</summary>
    public static bool Decision(ReadOnlySpan<uint> cats, float cat)
    {
        if (InvalidCat(cat)) return true;
        var (intPos, _) = LBitField32.ToBitPos((long)cat);
        if (intPos >= cats.Length) return true;
        return !LBitField32.Check(cats, AsCat(cat));
    }

    public static void InvalidCategory() =>
        Check.Fail("Invalid categorical value detected.  Categorical value should be non-negative, less than total number of "
            + $"categories in training data and less than {OutOfRangeCat}");

    public static void CheckMaxCat(float maxCat, long nCategories) =>
        Check.Ge(maxCat + 1, (float)nCategories, "Maximum cateogry should not be lesser than the total number of categories.");

    public static bool UseOneHot(uint nCats, uint maxCatToOnehot) => nCats < maxCatToOnehot;

    public static bool IsCatOp(FeatureType ft) => ft == FeatureType.Categorical;

    public static uint TrailingZeroBits(uint value) => value == 0 ? 32u : (uint)System.Numerics.BitOperations.TrailingZeroCount(value);
}
