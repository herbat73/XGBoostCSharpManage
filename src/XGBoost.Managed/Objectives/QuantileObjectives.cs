// Ports of src/objective/quantile_obj.cc, expectile_obj.cc, multiclass_obj.cc and
// src/common/quantile_loss_utils.h, expectile_loss_utils.h.
using XGBoost.Collective;
using static XGBoost.Objectives.ObjectiveUtils;

namespace XGBoost.Objectives;

public sealed class QuantileLossParam : XGBoostParameter<QuantileLossParam>
{
    public float[] QuantileAlpha = [];

    protected override void Declare(ParamManager<QuantileLossParam> m)
    {
        CustomField(m, "quantile_alpha", "ParamArray<float>", p => p.QuantileAlpha, (p, v) => p.QuantileAlpha = v,
                v => Tree.ParamArray.Parse("quantile_alpha", v), Tree.ParamArray.Print)
            .SetDefault([]).Describe("List of quantiles for quantile loss.");
    }

    public void Validate() => ValidateAlpha(this, QuantileAlpha, "quantile");

    internal static void ValidateAlpha<T>(XGBoostParameter<T> p, float[] array, string name) where T : XGBoostParameter<T>, new()
    {
        Check.That(p.GetInitialised());
        Check.That(array.Length != 0);
        Check.That(array.All(q => q >= 0.0 && q <= 1.0), $"{name} alpha must be in the range [0.0, 1.0].");
        for (var i = 1; i < array.Length; i++)
            Check.That(!(array[i] < array[i - 1]), $"{name} alpha must be sorted in ascending order.");
    }
}

public sealed class ExpectileLossParam : XGBoostParameter<ExpectileLossParam>
{
    public float[] ExpectileAlpha = [];

    protected override void Declare(ParamManager<ExpectileLossParam> m)
    {
        CustomField(m, "expectile_alpha", "ParamArray<float>", p => p.ExpectileAlpha, (p, v) => p.ExpectileAlpha = v,
                v => Tree.ParamArray.Parse("expectile_alpha", v), Tree.ParamArray.Print)
            .SetDefault([]).Describe("List of expectiles for expectile loss.");
    }

    public void Validate() => QuantileLossParam.ValidateAlpha(this, ExpectileAlpha, "expectile");
}

// ---- reg:quantileerror ------------------------------------------------------------------------

public sealed class QuantileRegression : ObjFunction
{
    private const float SmoothingScale = 0.04f;
    private const float MinSurrogateRatio = 3.0e-4f;

    private readonly QuantileLossParam _param = new();
    private readonly HostDeviceVector<float> _alpha = new();

    public override uint Targets(MetaInfo info)
    {
        var alpha = _param.QuantileAlpha;
        Check.Eq(alpha.Length, _alpha.Size, "The objective is not yet configured.");
        var nRows = info.Labels.Shape(0);
        var nColumns = info.Labels.Shape(1);
        Check.That(nColumns == 1 || (nRows == 0 && nColumns == 0), "Multi-target is not yet supported by the quantile loss.");
        Check.That(alpha.Length != 0);
        return (uint)_alpha.Size;
    }

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        if (iter == 0) CheckInitInputs(info);
        Check.Eq(_param.QuantileAlpha.Length, _alpha.Size);
        var nTargets = Targets(info);
        var nAlphas = _alpha.Size;
        Check.Ne(nAlphas, 0);
        Check.Ge((long)nTargets, nAlphas);
        Check.Eq((long)preds.Size, info.NumRow * nTargets);

        var labels = info.Labels.HostView();
        var predt = Linalg.MakeTensorView(preds, info.NumRow, nTargets);
        var weights = new OptionalWeights(info.Weights);
        var alphaH = _alpha.ToArray();

        var scaleStats = MeanAbsoluteError.RootResidualStats(Ctx, info, nTargets, (i, target) =>
            weights[i] * MathF.Sqrt(MathF.Abs(predt[i, target] - labels[i, 0])));
        var scale = new float[nTargets];
        for (var t = 0; t < nTargets; t++)
        {
            if (scaleStats[^1] != 0.0)
            {
                var rootMean = scaleStats[t] / scaleStats[^1];
                scale[t] = (float)(rootMean * rootMean);
            }
        }

        outGpair.Reshape(info.NumRow, nTargets);
        var gpair = outGpair.HostView();
        Linalg.ElementWiseKernel2(Ctx, gpair, (i, j) =>
        {
            var residual = predt[i, j] - labels[i, 0];
            var residualScale = scale[j];
            var weight = weights[i];
            if (!(residualScale > 0.0f) || weight == 0.0f)
            {
                gpair[i, j] = new GradientPair(0.0f, 0.0f);
                return;
            }
            var x = residual / (SmoothingScale * residualScale);
            var tanhX = MathF.Tanh(x);
            var ratio = x == 0.0f ? 1.0f : tanhX / x;
            ratio = XMath.FMaxF(ratio, MinSurrogateRatio);
            var grad = 0.5f * residualScale * (tanhX + 1.0f - 2.0f * alphaH[j]);
            var hess = 0.5f / SmoothingScale * ratio;
            gpair[i, j] = new GradientPair(weight * grad, weight * hess);
        });
    }

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        CheckInitInputs(info);
        var nTargets = Targets(info);
        baseScore.Assign(RadixSelect.Run(Ctx, info.Labels, info.Weights, _alpha, nTargets));
        Check.Eq((long)baseScore.Size, (long)nTargets);
    }

    public override void PredTransform(HostDeviceVector<float> predictions)
    {
        Check.That(!_alpha.Empty);
        Check.Eq(predictions.Size % _alpha.Size, 0);
        var nAlphas = _alpha.Size;
        var values = predictions.RawArray;
        var nRows = predictions.Size / nAlphas;
        Threading.ParallelFor(nRows, Ctx.Threads(), row => StdAlgo.Sort(values.AsSpan((int)row * nAlphas, nAlphas)));
    }

    public override SortedSet<string> Configure(Args args)
    {
        var used = ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));
        _param.Validate();
        _alpha.Assign(_param.QuantileAlpha);
        return used;
    }

    public override ObjInfo Task => new(ObjTask.Regression, false);

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = "reg:quantileerror";
        output["quantile_loss_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input)
    {
        Check.Eq(input["name"].AsString, "reg:quantileerror");
        _param.FromJson(input["quantile_loss_param"]);
        _param.Validate();
        _alpha.Assign(_param.QuantileAlpha);
    }

    public override string DefaultEvalMetric => "quantile";

    public override Json DefaultMetricConfig()
    {
        Check.That(_param.GetInitialised());
        var config = new JsonObject();
        config["name"] = DefaultEvalMetric;
        config["quantile_loss_param"] = _param.ToJson();
        return config;
    }
}

// ---- reg:expectileerror -----------------------------------------------------------------------

public sealed class ExpectileRegression : FitIntercept
{
    private readonly ExpectileLossParam _param = new();
    private readonly HostDeviceVector<float> _alpha = new();

    public override uint Targets(MetaInfo info)
    {
        var alpha = _param.ExpectileAlpha;
        Check.Eq(alpha.Length, _alpha.Size, "The objective is not yet configured.");
        Check.Eq(info.Labels.Shape(1), 1L, "Multi-target is not yet supported by the expectile loss.");
        Check.That(alpha.Length != 0);
        return (uint)_alpha.Size;
    }

    public override SortedSet<string> Configure(Args args)
    {
        var used = ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));
        _param.Validate();
        _alpha.Assign(_param.ExpectileAlpha);
        return used;
    }

    public override ObjInfo Task => new(ObjTask.Regression);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        if (iter == 0) CheckInitInputs(info);
        var nTargets = Targets(info);
        Check.Eq((long)preds.Size, info.NumRow * nTargets);

        var labels = info.Labels.HostView();
        var weights = new OptionalWeights(info.Weights);
        var predt = Linalg.MakeTensorView(preds, info.NumRow, nTargets);
        var alphaH = _alpha.ToArray();
        outGpair.Reshape(info.NumRow, nTargets);
        var gpair = outGpair.HostView();
        Linalg.ElementWiseKernel2(Ctx, gpair, (i, j) =>
        {
            var label = labels[i, 0];
            var sampleWeight = weights[i];
            var pred = predt[i, 0];
            var gradSum = 0.0f;
            var hessSum = 0.0f;
            for (var k = 0; k < alphaH.Length; k++)
            {
                if (k > 0) pred += Constants.RtEps + XMath.SoftPlus(predt[i, k]);
                if (k >= j)
                {
                    var diff = pred - label;
                    var weightScale = diff >= 0.0f ? 1.0f - alphaH[k] : alphaH[k];
                    gradSum += weightScale * diff * sampleWeight;
                    hessSum += weightScale * sampleWeight;
                }
            }
            var scale = j == 0 ? 1.0f : XMath.Sigmoid(predt[i, j]);
            gpair[i, j] = new GradientPair(scale * gradSum, scale * scale * hessSum);
        });
    }

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        var nTargets = Targets(info);
        var labelMean = info.Weights.Empty ? Stats.SampleMean(Ctx, info.Labels) : Stats.WeightedSampleMean(Ctx, info.Labels, info.Weights);
        Check.Eq(labelMean.Size, 1);
        var mean = labelMean[0];
        var labels = info.Labels.HostView();
        var weights = new OptionalWeights(info.Weights);
        var alphaH = _alpha.ToArray();
        var gpair = new Tensor<GradientPair>([info.NumRow, nTargets]);
        var gpairH = gpair.HostView();
        Linalg.ElementWiseKernel2(Ctx, gpairH, (i, j) =>
        {
            var diff = mean - labels[i, 0];
            var weightScale = diff >= 0.0f ? 1.0f - alphaH[j] : alphaH[j];
            gpairH[i, j] = new GradientPair(weightScale * diff * weights[i], weightScale * weights[i]);
        });
        var output = Stats.FitStump(Ctx, gpair, nTargets);
        for (var j = 0; j < nTargets; j++) output[j] += mean;
        for (var j = 1; j < nTargets; j++) output[j] = XMath.StdMax(output[j], output[j - 1]);
        baseScore.Assign(output);
    }

    public override void PredTransform(HostDeviceVector<float> predictions)
    {
        Check.Ne(_alpha.Size, 0);
        Check.Eq(predictions.Size % _alpha.Size, 0);
        var nAlphas = _alpha.Size;
        var nSamples = predictions.Size / nAlphas;
        var predt = Linalg.MakeTensorView(predictions, nSamples, nAlphas);
        Threading.ParallelFor(nSamples, Ctx.Threads(), i =>
        {
            var pred = predt[i, 0];
            for (var j = 1; j < nAlphas; j++)
            {
                pred += Constants.RtEps + XMath.SoftPlus(predt[i, j]);
                predt[i, j] = pred;
            }
        });
    }

    public override void ProbToMargin(Tensor<float> baseScore)
    {
        Check.Eq(baseScore.Size, _alpha.Size);
        for (var j = baseScore.Size - 1; j > 0; --j)
            baseScore[j] = XMath.SoftPlusInv(baseScore[j] - baseScore[j - 1] - Constants.RtEps);
    }

    public override string DefaultEvalMetric => "expectile";

    public override Json DefaultMetricConfig()
    {
        Check.That(_param.GetInitialised());
        var config = new JsonObject();
        config["name"] = DefaultEvalMetric;
        config["expectile_loss_param"] = _param.ToJson();
        return config;
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = "reg:expectileerror";
        output["expectile_loss_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input)
    {
        Check.Eq(input["name"].AsString, "reg:expectileerror");
        if (input.AsObject.TryGetValue("expectile_loss_param", out var p))
        {
            _param.FromJson(p);
            _alpha.Assign(_param.ExpectileAlpha);
        }
    }
}

// ---- multi:softmax / multi:softprob -----------------------------------------------------------

public sealed class SoftmaxMultiClassParam : XGBoostParameter<SoftmaxMultiClassParam>
{
    public int NumClass = 1;

    protected override void Declare(ParamManager<SoftmaxMultiClassParam> m)
    {
        Field(m, "num_class", p => p.NumClass, (p, v) => p.NumClass = v).SetLowerBound(1)
            .Describe("Number of output class in the multi-class classification.");
    }
}

public sealed class SoftmaxMultiClassObj(bool outputProb) : ObjFunction
{
    private readonly SoftmaxMultiClassParam _param = new();

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));

    public override ObjInfo Task => new(ObjTask.Classification);

    private static bool LabelCheck(float value, long nClasses) =>
        value >= 0.0f && value < nClasses && MathF.Floor(value) == value;

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        if (info.Labels.Size == 0) return;
        long nClasses = _param.NumClass;
        Check.Eq((long)preds.Size, nClasses * info.Labels.Size, "SoftmaxMultiClassObj: label size and pred size does not match.");
        Check.Eq(preds.Size / nClasses, info.NumRow);
        Check.Le(info.Labels.Shape(1), 1L, "multi-class-multi-label is not yet supported.");
        if (!info.Weights.Empty)
            Check.Eq((long)info.Weights.Size, info.NumRow, "Number of weights should be equal to number of data points.");
        if (iter == 0)
            Check.That(ElementwiseValidate(info.Labels, v => LabelCheck(v, nClasses)),
                "SoftmaxMultiClassObj: label must be discrete values in the range of [0, num_class).");

        var nSamples = info.NumRow;
        var predt = Linalg.MakeTensorView(preds, nSamples, nClasses);
        var labels = info.Labels.HostView();
        var weights = new OptionalWeights(info.Weights);
        outGpair.Reshape(nSamples, nClasses);
        var gpair = outGpair.HostView();
        Threading.ParallelFor(nSamples, Ctx.Threads(), row =>
        {
            var wmax = 1.17549435E-38f; // std::numeric_limits<float>::min()
            for (long k = 0; k < nClasses; k++) wmax = XMath.FMaxF(predt[row, k], wmax);
            var wsum = 0.0;
            for (long k = 0; k < nClasses; k++) wsum += MathF.Exp(predt[row, k] - wmax);
            var label = labels[row, 0];
            var weight = weights[row];
            for (long k = 0; k < nClasses; k++)
            {
                var probability = MathF.Exp(predt[row, k] - wmax) / (float)wsum;
                var grad = label == k ? probability - 1.0f : probability;
                var hess = XMath.FMaxF(MathF.Abs(grad) * weight, 1e-16f);
                gpair[row, k] = new GradientPair(grad * weight, hess);
            }
        });
    }

    public override void PredTransform(HostDeviceVector<float> predictions) => Transform(predictions, outputProb);

    public override void EvalTransform(HostDeviceVector<float> predictions) => Transform(predictions, true);

    public override string DefaultEvalMetric => "mlogloss";

    private void Transform(HostDeviceVector<float> predictions, bool probability)
    {
        var nClasses = _param.NumClass;
        var values = predictions.RawArray;
        var nSamples = predictions.Size / nClasses;
        if (probability)
        {
            Threading.ParallelFor(nSamples, Ctx.Threads(), row => XMath.Softmax(values.AsSpan((int)row * nClasses, nClasses)));
        }
        else
        {
            var output = new float[nSamples];
            Threading.ParallelFor(nSamples, Ctx.Threads(), row =>
                output[row] = XMath.FindMaxIndex(values.AsSpan((int)row * nClasses, nClasses)));
            predictions.Assign(output, true);
        }
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = outputProb ? "multi:softprob" : "multi:softmax";
        output["softmax_multiclass_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input) => _param.FromJson(input["softmax_multiclass_param"]);

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        long nClasses = _param.NumClass;
        Check.Le(info.Labels.Shape(1), 1L, "multi-class-multi-label is not yet supported.");
        Check.That(ElementwiseValidate(info.Labels, v => LabelCheck(v, nClasses)),
            "SoftmaxMultiClassObj: label must be discrete values in the range of [0, num_class).");

        var output = Linalg.Zeros<float>(nClasses);
        var labels = info.Labels.HostView();
        var weights = new OptionalWeights(info.Weights);
        var intercept = output.HostView();
        Linalg.SmallHistogram(labels, weights, intercept);
        var sumWeight = weights.Sum(info.Labels.Size);
        Communicator.Allreduce(intercept.Values, Op.Sum);
        Span<double> sw = [sumWeight];
        Communicator.Allreduce(sw, Op.Sum);
        sumWeight = sw[0];
        Check.Ge(sumWeight, (double)Constants.RtEps);
        Linalg.VecScaDiv(Ctx, intercept, sumWeight);
        Linalg.LogE(Ctx, intercept, Constants.RtEps);
        var mean = Stats.Mean(Ctx, intercept.Values);
        ElementwiseTransform(Ctx, output.Data, v => v - mean);
        baseScore.Assign(output);
    }
}
