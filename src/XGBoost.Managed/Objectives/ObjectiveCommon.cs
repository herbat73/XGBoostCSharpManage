// Port of src/objective/init_estimation.h/.cc, elementwise_objective.h and radix_select.cc.
using XGBoost.Collective;

namespace XGBoost.Objectives;

/// <summary>Objective whose intercept is fitted with one Newton step from a zero prediction.</summary>
public abstract class FitIntercept : ObjFunction
{
    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        if (Task.Task == ObjTask.Regression) ObjectiveUtils.CheckInitInputs(info);
        var nTargets = Targets(info);
        var dummyPredt = new HostDeviceVector<float>((int)(info.NumRow * nTargets), 0.0f);
        var gpair = new Tensor<GradientPair>([info.NumRow, nTargets]);

        var config = new JsonObject();
        SaveConfig(config);
        var newObj = Registry.CreateObjective(config["name"].AsString, Ctx);
        newObj.LoadConfig(config);
        newObj.GetGradient(dummyPredt, info, 0, gpair);
        baseScore.Assign(Stats.FitStump(Ctx, gpair, nTargets));
        PredTransform(baseScore.Data);
    }
}

/// <summary>Objective whose intercept is the (weighted) sample mean of the labels.</summary>
public abstract class FitInterceptGlmLike : FitIntercept
{
    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore) => GlmLikeInit(info, baseScore);

    protected void GlmLikeInit(MetaInfo info, Tensor<float> baseScore)
    {
        if (Task.Task == ObjTask.Regression) ObjectiveUtils.CheckInitInputs(info);
        baseScore.Assign(info.Weights.Empty
            ? Stats.SampleMean(Ctx, info.Labels)
            : Stats.WeightedSampleMean(Ctx, info.Labels, info.Weights));
        Check.Ge(baseScore.Size, 1);
    }
}

public static class ObjectiveUtils
{
    public static void CheckInitInputs(MetaInfo info)
    {
        Check.Eq(info.Labels.Shape(0), info.NumRow, "Invalid shape of labels.");
        if (!info.Weights.Empty)
            Check.Eq((long)info.Weights.Size, info.NumRow, "Number of weights should be equal to number of data points.");
    }

    /// <summary><c>std::max(1, labels.Shape(1))</c>.</summary>
    public static uint LabelTargets(MetaInfo info) => (uint)Math.Max(1L, info.Labels.Shape(1));

    public static void CheckWeights(MetaInfo info)
    {
        if (!info.Weights.Empty)
            Check.Eq((long)info.Weights.Size, info.NumRow, "Number of weights should be equal to the number of data points.");
    }

    /// <summary><c>elementwise::detail::GradientCpu</c>.</summary>
    public static void ElementwiseGradient(Context ctx, HostDeviceVector<float> preds, MetaInfo info, uint nTargets,
        Func<float, float, float, GradientPair> gradient, Tensor<GradientPair> outGpair)
    {
        var predt = Linalg.MakeTensorView(preds, info.NumRow, nTargets);
        var labels = info.Labels.HostView();
        var weights = new OptionalWeights(info.Weights);
        outGpair.Reshape(info.NumRow, nTargets);
        var gpair = outGpair.HostView();
        Linalg.ElementWiseKernel2(ctx, gpair, (i, j) => gpair[i, j] = gradient(predt[i, j], labels[i, j], weights[i]));
    }

    /// <summary><c>elementwise::detail::TransformCpu</c>.</summary>
    public static void ElementwiseTransform(Context ctx, HostDeviceVector<float> preds, Func<float, float> transform)
    {
        var values = preds.RawArray;
        Threading.ParallelFor(preds.Size, ctx.Threads(), i => values[i] = transform(values[i]));
    }

    /// <summary><c>elementwise::detail::ValidationCpu</c>: <c>std::all_of</c> over the labels.</summary>
    public static bool ElementwiseValidate(Tensor<float> values, Func<float, bool> check)
    {
        var view = values.HostView();
        for (long i = 0; i < view.Size; i++)
            if (!check(view.Flat(i))) return false;
        return true;
    }

    public static bool AllOf(Tensor<float> values, Func<float, bool> check) => ElementwiseValidate(values, check);
}

/// <summary><c>RadixSelect</c>: weighted quantiles of each label column through an 8-bit radix search.</summary>
public static class RadixSelect
{
    private const int RadixBits = 8;
    private const int RadixBins = 1 << RadixBits;
    private const int RadixPasses = 32 / RadixBits;

    private static uint ToOrderedKey(float value)
    {
        var bits = BitConverter.SingleToUInt32Bits(value);
        var mask = (bits & 0x80000000U) != 0 ? uint.MaxValue : 0x80000000U;
        return bits ^ mask;
    }

    private static float FromOrderedKey(uint key)
    {
        var mask = (key & 0x80000000U) != 0 ? 0x80000000U : uint.MaxValue;
        return BitConverter.UInt32BitsToSingle(key ^ mask);
    }

    private static int SelectBin(ReadOnlySpan<double> histogram, double rank, float alpha)
    {
        if (alpha == 0.0f)
        {
            for (var bin = 0; bin < histogram.Length; bin++)
                if (histogram[bin] > 0.0) return bin;
        }
        else if (alpha == 1.0f)
        {
            for (var bin = histogram.Length; bin-- > 0;)
                if (histogram[bin] > 0.0) return bin;
        }
        else
        {
            var cumulative = 0.0;
            for (var bin = 0; bin < histogram.Length; bin++)
            {
                cumulative += histogram[bin];
                if (histogram[bin] > 0.0 && cumulative >= rank) return bin;
            }
        }
        for (var bin = histogram.Length; bin-- > 0;)
            if (histogram[bin] > 0.0) return bin;
        return 0;
    }

    public static Tensor<float> Run(Context ctx, Tensor<float> values, HostDeviceVector<float> weights, HostDeviceVector<float> alphas, uint nTargets)
    {
        Check.That(weights.Empty || weights.Size == values.Shape(0));
        Check.That(!alphas.Empty);
        Check.Ne(nTargets, 0u);
        Check.That(values.Shape(1) == 0 || values.Shape(1) * alphas.Size == nTargets, "Invalid number of outputs.");
        foreach (var alpha in alphas.ConstHostSpan)
        {
            Check.Ge(alpha, 0.0f);
            Check.Le(alpha, 1.0f);
        }

        var nAlphas = alphas.Size;
        var nOutputs = (int)nTargets;
        var output = Linalg.Zeros<float>(nOutputs);
        if (nOutputs == 0) return output;

        var hValues = values.HostView();
        var hWeights = new OptionalWeights(weights);
        var hAlphas = alphas.ToArray();
        var nThreads = Math.Max(ctx.Threads(), 1);
        var nRows = values.Shape(0);

        var prefixes = new uint[nOutputs];
        var ranks = new double[nOutputs];
        var histogram = new double[nOutputs * RadixBins];
        var threadHistogram = new double[nThreads * histogram.Length];

        for (var pass = 0; pass < RadixPasses; pass++)
        {
            Array.Clear(threadHistogram);
            var shift = 32 - (pass + 1) * RadixBits;
            var prefixMask = pass == 0 ? 0U : uint.MaxValue << (shift + 8);
            Threading.ParallelFor(nRows * nOutputs, nThreads, (i, thread) =>
            {
                var o = i / nRows;
                var row = i % nRows;
                var column = o / nAlphas;
                var key = ToOrderedKey(hValues[row, column]);
                if ((key & prefixMask) != prefixes[o]) return;
                var bin = (key >> shift) & (RadixBins - 1);
                var offset = (thread * nOutputs + o) * RadixBins + bin;
                threadHistogram[offset] += hWeights[row];
            });
            Array.Clear(histogram);
            for (var thread = 0; thread < nThreads; thread++)
            {
                var offset = thread * histogram.Length;
                for (var i = 0; i < histogram.Length; i++) histogram[i] += threadHistogram[offset + i];
            }
            Communicator.Allreduce(histogram.AsSpan(), Op.Sum);

            for (var o = 0; o < nOutputs; o++)
            {
                var bins = histogram.AsSpan(o * RadixBins, RadixBins);
                var alpha = hAlphas[o % nAlphas];
                if (pass == 0)
                {
                    var total = 0.0;
                    foreach (var b in bins) total += b;
                    if (total == 0.0)
                    {
                        ranks[o] = -1.0;
                        continue;
                    }
                    ranks[o] = alpha * total;
                }
                else if (ranks[o] < 0.0)
                {
                    continue;
                }
                var sel = SelectBin(bins, ranks[o], alpha);
                var weightBefore = 0.0;
                for (var b = 0; b < sel; b++) weightBefore += bins[b];
                ranks[o] = XMath.StdClamp(ranks[o] - weightBefore, 0.0, bins[sel]);
                prefixes[o] |= (uint)sel << shift;
            }
        }

        for (var o = 0; o < nOutputs; o++) output[o] = ranks[o] < 0.0 ? 0.0f : FromOrderedKey(prefixes[o]);
        return output;
    }
}
