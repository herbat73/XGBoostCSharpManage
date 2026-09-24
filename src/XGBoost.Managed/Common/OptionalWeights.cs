// Port of src/common/optional_weight.h/.cc.
using System.Runtime.CompilerServices;

namespace XGBoost.Common;

/// <summary>Sample weights that default to a constant when none are given.</summary>
public readonly struct OptionalWeights
{
    private readonly float[]? _weights;
    private readonly int _size;
    public readonly float Dft;

    public OptionalWeights(float[] weights, int size)
    {
        _weights = weights;
        _size = size;
        Dft = 1f;
    }

    public OptionalWeights(HostDeviceVector<float> weights) : this(weights.RawArray, weights.Size) { }

    public OptionalWeights(float dft)
    {
        _weights = null;
        _size = 0;
        Dft = dft;
    }

    public float this[long i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _size == 0 ? Dft : _weights![i];
    }

    public bool Empty => _size == 0;
    public int Size => _size;
    public ReadOnlySpan<float> Data => _weights is null ? default : _weights.AsSpan(0, _size);

    public static OptionalWeights Make(HostDeviceVector<float> weights) => new(weights);

    /// <summary><c>SumOptionalWeights</c>: accumulates in double from 0.0.</summary>
    public double Sum(long nSamples)
    {
        if (Empty) return nSamples * (double)Dft;
        var sum = 0.0;
        foreach (var w in Data) sum += w;
        return sum;
    }
}
