namespace XGBoost.Demos.Common;

/// <summary>
/// Seeded random numbers, standing in for <c>numpy.random.RandomState</c>. The streams differ from
/// NumPy's, so the generated data (and model outputs) do not match the Python demos value for value.
/// </summary>
internal sealed class Rng(int seed)
{
    private readonly Random _random = new(seed);
    private double? _spareNormal;

    /// <summary>Uniform sample in [low, high).</summary>
    public double Uniform(double low = 0, double high = 1) => low + (high - low) * _random.NextDouble();

    /// <summary>Uniform integer in [low, high).</summary>
    public int Integers(int low, int high) => _random.Next(low, high);

    /// <summary>Normal sample (Box-Muller).</summary>
    public double Normal(double mean = 0, double std = 1)
    {
        if (_spareNormal is { } spare)
        {
            _spareNormal = null;
            return mean + std * spare;
        }
        double u, v, s;
        do
        {
            u = 2 * _random.NextDouble() - 1;
            v = 2 * _random.NextDouble() - 1;
            s = u * u + v * v;
        } while (s is >= 1 or 0);
        var scale = Math.Sqrt(-2 * Math.Log(s) / s);
        _spareNormal = v * scale;
        return mean + std * u * scale;
    }

    /// <summary>Log-normal sample with the given underlying normal's parameters.</summary>
    public double LogNormal(double mean = 0, double sigma = 1) => Math.Exp(Normal(mean, sigma));

    /// <summary>Gamma sample (Marsaglia-Tsang).</summary>
    public double Gamma(double shape, double scale)
    {
        if (shape < 1) return Gamma(shape + 1, scale) * Math.Pow(_random.NextDouble(), 1 / shape);
        var d = shape - 1.0 / 3;
        var c = 1 / Math.Sqrt(9 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = Normal();
                v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            var u = _random.NextDouble();
            if (Math.Log(u) < 0.5 * x * x + d - d * v + d * Math.Log(v)) return d * v * scale;
        }
    }

    /// <summary>Array of standard normal samples as float32.</summary>
    public float[] NormalArray(int count)
    {
        var result = new float[count];
        for (var i = 0; i < count; i++) result[i] = (float)Normal();
        return result;
    }

    /// <summary>A random permutation of 0..n-1.</summary>
    public int[] Permutation(int n)
    {
        var p = Enumerable.Range(0, n).ToArray();
        _random.Shuffle(p);
        return p;
    }
}
