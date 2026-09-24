// Port of include/xgboost/base.h.
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace XGBoost;

public static class Constants
{
    /// <summary><c>kRtEps</c>: epsilon used to avoid division by zero and for split comparisons.</summary>
    public const float RtEps = 1e-6f;

    /// <summary>Major version implemented by this port.</summary>
    public const int VersionMajor = 3;
    public const int VersionMinor = 5;
    public const int VersionPatch = 0;
}

/// <summary>Gradient statistics in single precision, <c>GradientPair</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GradientPair : IEquatable<GradientPair>
{
    public float Grad;
    public float Hess;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GradientPair(float grad, float hess)
    {
        Grad = grad;
        Hess = hess;
    }

    public readonly float GetGrad() => Grad;
    public readonly float GetHess() => Hess;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(float grad, float hess)
    {
        Grad += grad;
        Hess += hess;
    }

    public static GradientPair operator +(GradientPair a, GradientPair b) => new(a.Grad + b.Grad, a.Hess + b.Hess);
    public static GradientPair operator -(GradientPair a, GradientPair b) => new(a.Grad - b.Grad, a.Hess - b.Hess);
    public static GradientPair operator *(GradientPair a, float m) => new(a.Grad * m, a.Hess * m);
    public static GradientPair operator /(GradientPair a, float d) => new(a.Grad / d, a.Hess / d);
    public static bool operator ==(GradientPair a, GradientPair b) => a.Grad == b.Grad && a.Hess == b.Hess;
    public static bool operator !=(GradientPair a, GradientPair b) => !(a == b);

    public readonly bool Equals(GradientPair other) => this == other;
    public override readonly bool Equals(object? obj) => obj is GradientPair g && Equals(g);
    public override readonly int GetHashCode() => HashCode.Combine(Grad, Hess);
    public override readonly string ToString() => $"{Common.Format.Stream(Grad)}/{Common.Format.Stream(Hess)}";
}

/// <summary>Gradient statistics in double precision, <c>GradientPairPrecise</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GradientPairPrecise : IEquatable<GradientPairPrecise>
{
    public double Grad;
    public double Hess;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GradientPairPrecise(double grad, double hess)
    {
        Grad = grad;
        Hess = hess;
    }

    /// <summary>Conversion from single precision (explicit constructor in C++).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GradientPairPrecise(GradientPair g)
    {
        Grad = g.Grad;
        Hess = g.Hess;
    }

    public readonly double GetGrad() => Grad;
    public readonly double GetHess() => Hess;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(double grad, double hess)
    {
        Grad += grad;
        Hess += hess;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(GradientPairPrecise g)
    {
        Grad += g.Grad;
        Hess += g.Hess;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(GradientPair g)
    {
        Grad += g.Grad;
        Hess += g.Hess;
    }

    public static GradientPairPrecise operator +(GradientPairPrecise a, GradientPairPrecise b) => new(a.Grad + b.Grad, a.Hess + b.Hess);
    public static GradientPairPrecise operator -(GradientPairPrecise a, GradientPairPrecise b) => new(a.Grad - b.Grad, a.Hess - b.Hess);
    public static GradientPairPrecise operator *(GradientPairPrecise a, float m) => new(a.Grad * m, a.Hess * m);
    public static GradientPairPrecise operator /(GradientPairPrecise a, float d) => new(a.Grad / d, a.Hess / d);
    public static bool operator ==(GradientPairPrecise a, GradientPairPrecise b) => a.Grad == b.Grad && a.Hess == b.Hess;
    public static bool operator !=(GradientPairPrecise a, GradientPairPrecise b) => !(a == b);

    /// <summary>Narrowing conversion, <c>GradientPair{GradientPairPrecise}</c>.</summary>
    public readonly GradientPair ToSingle() => new((float)Grad, (float)Hess);

    public readonly bool Equals(GradientPairPrecise other) => this == other;
    public override readonly bool Equals(object? obj) => obj is GradientPairPrecise g && Equals(g);
    public override readonly int GetHashCode() => HashCode.Combine(Grad, Hess);
    public override readonly string ToString() => $"{Common.Format.Stream(Grad)}/{Common.Format.Stream(Hess)}";
}

/// <summary>Quantised gradient, <c>GradientPairInt64</c>.</summary>
public struct GradientPairInt64(long grad, long hess)
{
    public long Grad = grad;
    public long Hess = hess;

    public readonly long GetQuantisedGrad() => Grad;
    public readonly long GetQuantisedHess() => Hess;

    public static GradientPairInt64 operator +(GradientPairInt64 a, GradientPairInt64 b) => new(a.Grad + b.Grad, a.Hess + b.Hess);
    public static GradientPairInt64 operator -(GradientPairInt64 a, GradientPairInt64 b) => new(a.Grad - b.Grad, a.Hess - b.Hess);
    public override readonly string ToString() => Grad.ToString(CultureInfo.InvariantCulture) + "/" + Hess.ToString(CultureInfo.InvariantCulture);
}

/// <summary>Element of a sparse row: feature index and value (<c>Entry</c>).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct Entry(uint index, float fvalue) : IEquatable<Entry>
{
    public uint Index = index;
    public float Fvalue = fvalue;

    public static bool CmpValue(Entry a, Entry b) => a.Fvalue < b.Fvalue;
    public static bool CmpIndex(Entry a, Entry b) => a.Index < b.Index;

    public readonly bool Equals(Entry other) => Index == other.Index && Fvalue == other.Fvalue;
    public override readonly bool Equals(object? obj) => obj is Entry e && Equals(e);
    public override readonly int GetHashCode() => HashCode.Combine(Index, Fvalue);
    public override readonly string ToString() => $"({Index}, {Fvalue})";
}

public enum FeatureType : byte
{
    Numerical = 0,
    Categorical = 1,
}

/// <summary>Type tags used by the binary DMatrix format and typed info getters.</summary>
public enum DataType : byte
{
    Float32 = 1,
    Double = 2,
    UInt32 = 3,
    UInt64 = 4,
    Str = 5,
}
