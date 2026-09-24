// Port of src/common/random.h/.cc and the pieces of the MSVC STL <random> used by XGBoost.
//
// The generator algorithms (mt19937, linear congruential) are fixed by the C++ standard, but the
// distributions are implementation defined. They are reproduced from MSVC 14.29 (VS 2019), the
// toolchain used for the reference native library, so that sampling matches bit for bit.
using System.Globalization;
using System.Text;

namespace XGBoost.Common;

/// <summary>A uniform random bit generator (<c>UniformRandomBitGenerator</c>).</summary>
public interface IUrng
{
    ulong Min { get; }
    ulong Max { get; }
    ulong Next();
}

/// <summary>
/// <c>std::mt19937</c> as implemented by MSVC: a 2n-word history buffer refilled half at a time.
/// Streaming (<see cref="Save"/>/<see cref="Load"/>) matches MSVC's <c>operator&lt;&lt;</c>/<c>&gt;&gt;</c>.
/// </summary>
public sealed class Mt19937 : IUrng
{
    private const int N = 624;
    private const int M = 397;
    private const uint Px = 0x9908b0df;
    private const uint Hmsk = 0x80000000;
    private const uint Lmsk = 0x7fffffff;
    public const uint DefaultSeed = 5489u;

    private readonly uint[] _ax = new uint[2 * N];
    private int _idx;

    public Mt19937() : this(DefaultSeed) { }

    public Mt19937(uint seed) => Seed(seed);

    public ulong Min => 0;
    public ulong Max => uint.MaxValue;

    public void Seed(uint seed)
    {
        var prev = _ax[0] = seed;
        for (var i = 1; i < N; i++) prev = _ax[i] = unchecked((uint)i + 1812433253u * (prev ^ (prev >> 30)));
        _idx = N;
    }

    /// <summary>Seeds from a wider integer; <c>result_type</c> is 32-bit so the value is truncated.</summary>
    public void Seed(long seed) => Seed(unchecked((uint)seed));

    public void Seed(ulong seed) => Seed(unchecked((uint)seed));

    public uint NextUInt()
    {
        if (_idx == N) RefillUpper();
        else if (2 * N <= _idx) RefillLower();

        var res = _ax[_idx++];
        res ^= res >> 11;
        res ^= (res << 7) & 0x9d2c5680;
        res ^= (res << 15) & 0xefc60000;
        res ^= res >> 18;
        return res;
    }

    public ulong Next() => NextUInt();

    public void Discard(ulong n)
    {
        for (ulong k = 0; k < n; k++) NextUInt();
    }

    private void RefillLower()
    {
        int ix;
        for (ix = 0; ix < N - M; ix++)
        {
            var tmp = (_ax[ix + N] & Hmsk) | (_ax[ix + N + 1] & Lmsk);
            _ax[ix] = (tmp >> 1) ^ ((tmp & 1) != 0 ? Px : 0) ^ _ax[ix + N + M];
        }
        for (; ix < N - 1; ix++)
        {
            var tmp = (_ax[ix + N] & Hmsk) | (_ax[ix + N + 1] & Lmsk);
            _ax[ix] = (tmp >> 1) ^ ((tmp & 1) != 0 ? Px : 0) ^ _ax[ix - N + M];
        }
        var t = (_ax[ix + N] & Hmsk) | (_ax[0] & Lmsk);
        _ax[ix] = (t >> 1) ^ ((t & 1) != 0 ? Px : 0) ^ _ax[M - 1];
        _idx = 0;
    }

    private void RefillUpper()
    {
        for (var ix = N; ix < 2 * N; ix++)
        {
            var tmp = (_ax[ix - N] & Hmsk) | (_ax[ix - N + 1] & Lmsk);
            _ax[ix] = (tmp >> 1) ^ ((tmp & 1) != 0 ? Px : 0) ^ _ax[ix - N + M];
        }
    }

    private int Base(int ix)
    {
        ix += _idx;
        return ix < N ? ix + N : ix - N;
    }

    /// <summary><c>ss &lt;&lt; std::hex &lt;&lt; rng</c> on MSVC: n words, each followed by a space.</summary>
    public string Save()
    {
        var sb = new StringBuilder(N * 9);
        for (var i = 0; i < N; i++) sb.Append(_ax[Base(i)].ToString("x", CultureInfo.InvariantCulture)).Append(' ');
        return sb.ToString();
    }

    /// <summary>
    /// <c>ss &gt;&gt; std::hex &gt;&gt; rng</c>. Accepts MSVC's n words; a libstdc++ state (n words plus the
    /// position) is also accepted, and positioned so the next outputs continue that stream.
    /// </summary>
    public void Load(string state)
    {
        var parts = state.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < N) throw new XGBoostException("Invalid RNG state.");
        for (var k = 0; k < N; k++) _ax[k] = uint.Parse(parts[k], NumberStyles.HexNumber, CultureInfo.InvariantCulture);
        _idx = N;
        if (parts.Length > N)
        {
            // libstdc++ stores its current (already twisted) block and the next index p. Outputs come
            // from the upper half here, and the next refill only reads the upper half.
            var p = int.Parse(parts[N], NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            if (p < N)
            {
                Array.Copy(_ax, 0, _ax, N, N);
                _idx = N + p;
            }
        }
    }
}

/// <summary><c>std::linear_congruential_engine&lt;UInt, a, c, m&gt;</c> with exact modular arithmetic.</summary>
public sealed class LinearCongruential : IUrng
{
    private readonly ulong _a, _c, _m; // m == 0 means 2^64
    private ulong _state;

    public LinearCongruential(ulong a, ulong c, ulong m, ulong seed)
    {
        _a = a;
        _c = c;
        _m = m;
        Seed(seed);
    }

    /// <summary><c>std::minstd_rand</c> (48271, 0, 2^31-1).</summary>
    public static LinearCongruential MinstdRand(uint seed) => new(48271, 0, 2147483647, seed);

    public ulong Min => _c == 0 ? 1ul : 0ul;
    public ulong Max => _m == 0 ? ulong.MaxValue : _m - 1;

    public void Seed(ulong s)
    {
        if (_m != 0) s %= _m;
        if (_c == 0 && s == 0) s = 1;
        _state = s;
    }

    public ulong Next()
    {
        if (_m == 0)
        {
            _state = unchecked(_a * _state + _c);
        }
        else
        {
            var prod = (UInt128)_a * _state + _c;
            _state = (ulong)(prod % _m);
        }
        return _state;
    }

    public ulong State => _state;
}

/// <summary>Distributions and algorithms reproduced from the MSVC 14.29 STL.</summary>
public static class StdRandom
{
    /// <summary><c>generate_canonical&lt;float, -1&gt;</c>.</summary>
    public static float CanonicalFloat(IUrng g)
    {
        var gmin = (float)g.Min;
        var gmax = (float)g.Max;
        var rx = gmax - gmin + 1f;
        var ceil = (int)MathF.Ceiling(24f / MathF.Log2(rx));
        var k = ceil < 1 ? 1 : ceil;
        var ans = 0f;
        var factor = 1f;
        for (var i = 0; i < k; i++)
        {
            ans += ((float)g.Next() - gmin) * factor;
            factor *= rx;
        }
        return ans / factor;
    }

    /// <summary><c>generate_canonical&lt;double, -1&gt;</c>.</summary>
    public static double CanonicalDouble(IUrng g)
    {
        var gmin = (double)g.Min;
        var gmax = (double)g.Max;
        var rx = gmax - gmin + 1.0;
        var ceil = (int)Math.Ceiling(53.0 / Math.Log2(rx));
        var k = ceil < 1 ? 1 : ceil;
        var ans = 0.0;
        var factor = 1.0;
        for (var i = 0; i < k; i++)
        {
            ans += ((double)g.Next() - gmin) * factor;
            factor *= rx;
        }
        return ans / factor;
    }

    /// <summary><c>std::uniform_real_distribution&lt;float&gt;(a, b)</c>.</summary>
    public static float UniformFloat(IUrng g, float a = 0f, float b = 1f) => CanonicalFloat(g) * (b - a) + a;

    /// <summary><c>std::uniform_real_distribution&lt;double&gt;(a, b)</c>.</summary>
    public static double UniformDouble(IUrng g, double a = 0.0, double b = 1.0) => CanonicalDouble(g) * (b - a) + a;

    /// <summary><c>std::bernoulli_distribution(p)</c>.</summary>
    public static bool Bernoulli(IUrng g, double p) => CanonicalDouble(g) < p;

    /// <summary><c>std::discrete_distribution</c> over <paramref name="weights"/> (normalised as MSVC does).</summary>
    public static int Discrete(IUrng g, IReadOnlyList<double> weights)
    {
        var p = weights.Count == 0 ? [1.0] : weights.ToArray();
        var sum = 0.0;
        foreach (var w in p) sum += w;
        if (sum != 1.0)
            for (var i = 0; i < p.Length; i++) p[i] /= sum;
        var cdf = new double[p.Length];
        cdf[0] = p[0];
        for (var i = 1; i < p.Length; i++) cdf[i] = p[i] + cdf[i - 1];
        var px = CanonicalDouble(g);
        // lower_bound over [cdf.begin(), cdf.end() - 1).
        int lo = 0, hi = cdf.Length - 1;
        while (lo < hi)
        {
            var mid = lo + ((hi - lo) >> 1);
            if (cdf[mid] < px) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /// <summary>Port of MSVC <c>_Rng_from_urng</c> for a 64-bit difference type.</summary>
    public sealed class RngFromUrng
    {
        private readonly IUrng _g;
        private readonly int _bits;
        private readonly ulong _bmask;
        private readonly int _udiffBits;

        /// <param name="g">Generator.</param>
        /// <param name="diffBits">Bit width of the (unsigned) difference type: 64 for size_t/ptrdiff_t, 32 for int.</param>
        public RngFromUrng(IUrng g, int diffBits = 64)
        {
            _g = g;
            // _Udiff is the wider of the difference type and the generator result type.
            var resultBits = g.Max > uint.MaxValue ? 64 : 32;
            _udiffBits = Math.Max(diffBits, resultBits);
            _bits = _udiffBits;
            _bmask = _udiffBits == 64 ? ulong.MaxValue : uint.MaxValue;
            for (; g.Max - g.Min < _bmask; _bmask >>= 1) _bits--;
        }

        private ulong Mask(ulong v) => _udiffBits == 64 ? v : v & uint.MaxValue;

        private ulong GetBits()
        {
            while (true)
            {
                var val = Mask(_g.Next() - _g.Min);
                if (val <= _bmask) return val;
            }
        }

        public ulong GetAllBits()
        {
            ulong ret = 0;
            for (var num = 0; num < _udiffBits; num += _bits)
            {
                ret = Mask(ret << (_bits - 1));
                ret = Mask(ret << 1);
                ret |= GetBits();
            }
            return ret;
        }

        /// <summary>Uniform value in [0, index).</summary>
        public ulong Next(ulong index)
        {
            while (true)
            {
                ulong ret = 0;
                ulong mask = 0;
                while (mask < Mask(index - 1))
                {
                    ret = Mask(ret << (_bits - 1));
                    ret = Mask(ret << 1);
                    ret |= GetBits();
                    mask = Mask(mask << (_bits - 1));
                    mask = Mask(mask << 1);
                    mask |= _bmask;
                }
                if (ret / index < mask / index || mask % index == Mask(index - 1)) return ret % index;
            }
        }
    }

    /// <summary><c>std::uniform_int_distribution&lt;size_t&gt;(min, max)</c>.</summary>
    public static ulong UniformInt(IUrng g, ulong min, ulong max)
    {
        var gen = new RngFromUrng(g, 64);
        ulong ret;
        if (max - min == ulong.MaxValue) ret = gen.GetAllBits();
        else ret = gen.Next(max - min + 1);
        return ret + min;
    }

    /// <summary><c>std::uniform_int_distribution&lt;int&gt;(min, max)</c>.</summary>
    public static int UniformInt32(IUrng g, int min, int max)
    {
        var gen = new RngFromUrng(g, 32);
        static uint Adjust(uint u) => u < 0x80000000u ? u + 0x80000000u : u - 0x80000000u;
        var umin = Adjust(unchecked((uint)min));
        var umax = Adjust(unchecked((uint)max));
        uint ret;
        if (umax - umin == uint.MaxValue) ret = (uint)gen.GetAllBits();
        else ret = (uint)gen.Next(umax - umin + 1u);
        return unchecked((int)Adjust(ret + umin));
    }

    /// <summary><c>std::shuffle(first, last, g)</c> with a 64-bit difference type.</summary>
    public static void Shuffle<T>(Span<T> values, IUrng g)
    {
        if (values.Length == 0) return;
        var rng = new RngFromUrng(g, 64);
        for (var target = 1; target < values.Length; target++)
        {
            var off = (int)rng.Next((ulong)target + 1);
            if (off != target) (values[target], values[off]) = (values[off], values[target]);
        }
    }
}

/// <summary>Port of <c>common::ColumnSampler</c>.</summary>
public sealed class ColumnSampler
{
    private uint[] _featureSetTree = [];
    private readonly Dictionary<int, uint[]> _featureSetLevel = [];
    private float[] _featureWeights = [];
    private float _colsampleBylevel = 1f;
    private float _colsampleBytree = 1f;
    private float _colsampleBynode = 1f;

    /// <param name="ctx">Runtime context, supplies the RNG.</param>
    /// <param name="numCol">Number of features.</param>
    /// <param name="featureWeights">Optional per-feature sampling weights.</param>
    /// <param name="colsampleBynode">Node sampling rate.</param>
    /// <param name="colsampleBylevel">Level sampling rate.</param>
    /// <param name="colsampleBytree">Tree sampling rate.</param>
    public void Init(Context ctx, long numCol, float[] featureWeights, float colsampleBynode, float colsampleBylevel,
        float colsampleBytree)
    {
        _featureWeights = featureWeights.ToArray();
        _colsampleBylevel = colsampleBylevel;
        _colsampleBytree = colsampleBytree;
        _colsampleBynode = colsampleBynode;
        Reset();
        _featureSetTree = new uint[numCol];
        for (var i = 0; i < numCol; i++) _featureSetTree[i] = (uint)i;
        _featureSetTree = ColSample(ctx, _featureSetTree, _colsampleBytree);
    }

    public void Reset()
    {
        _featureSetTree = [];
        _featureSetLevel.Clear();
    }

    public uint[] GetFeatureSet(Context ctx, int depth)
    {
        if (_colsampleBylevel == 1f && _colsampleBynode == 1f) return _featureSetTree;
        if (!_featureSetLevel.TryGetValue(depth, out var level))
            _featureSetLevel[depth] = level = ColSample(ctx, _featureSetTree, _colsampleBylevel);
        if (_colsampleBynode == 1f) return level;
        return ColSample(ctx, level, _colsampleBynode);
    }

    private uint[] ColSample(Context ctx, uint[] features, float colsample)
    {
        if (colsample == 1f) return features;
        var n = Math.Max(1, (int)(colsample * features.Length));
        var seed = ctx.Rng.NextUInt();
        var rng = new Mt19937(seed);
        Check.Gt(features.Length, 0);
        uint[] result;
        if (_featureWeights.Length != 0)
        {
            var weight = new float[features.Length];
            for (var i = 0; i < features.Length; i++) weight[i] = _featureWeights[features[i]];
            result = WeightedSamplingWithoutReplacement(ctx, rng, features, weight, n);
        }
        else
        {
            result = features.ToArray();
            StdRandom.Shuffle(result.AsSpan(), rng);
            Array.Resize(ref result, n);
        }
        StdAlgo.Sort(result.AsSpan(), static (a, b) => a < b);
        return result;
    }

    /// <summary>Efraimidis-Spirakis weighted sampling without replacement.</summary>
    public static T[] WeightedSamplingWithoutReplacement<T>(Context ctx, IUrng rng, T[] array, float[] weights, int n)
    {
        Check.Eq(array.Length, weights.Length);
        var keys = new float[weights.Length];
        for (var i = 0; i < array.Length; i++)
        {
            var w = Math.Max(weights[i], Constants.RtEps);
            var u = StdRandom.UniformFloat(rng);
            keys[i] = MathF.Log(u) / w;
        }
        var ind = StdAlgo.ArgSort(keys, static (a, b) => a > b);
        var results = new T[n];
        for (var k = 0; k < n; k++) results[k] = array[ind[k]];
        return results;
    }
}
