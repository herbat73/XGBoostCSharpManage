// Ports of src/objective/{squared_error,logistic,pseudohuber,squared_log,poisson,gamma,tweedie,
// absolute_error,hinge,normal}_obj.* and the parameter structs they use.
using System.Globalization;
using XGBoost.Collective;
using static XGBoost.Objectives.ObjectiveUtils;

namespace XGBoost.Objectives;

// ---- Parameters -------------------------------------------------------------------------------

public sealed class LogisticParam : XGBoostParameter<LogisticParam>
{
    public float ScalePosWeight;

    protected override void Declare(ParamManager<LogisticParam> m)
    {
        Field(m, "scale_pos_weight", p => p.ScalePosWeight, (p, v) => p.ScalePosWeight = v).SetDefault(1.0f).SetLowerBound(0.0f)
            .Describe("Scale the weight of positive examples by this factor");
    }
}

public sealed class PseudoHuberParam : XGBoostParameter<PseudoHuberParam>
{
    public float HuberSlope = 1.0f;

    protected override void Declare(ParamManager<PseudoHuberParam> m)
    {
        Field(m, "huber_slope", p => p.HuberSlope, (p, v) => p.HuberSlope = v).SetDefault(1.0f)
            .Describe("The delta term in Pseudo-Huber loss.");
    }
}

public sealed class TweedieRegressionParam : XGBoostParameter<TweedieRegressionParam>
{
    public float TweedieVariancePower;

    protected override void Declare(ParamManager<TweedieRegressionParam> m)
    {
        Field(m, "tweedie_variance_power", p => p.TweedieVariancePower, (p, v) => p.TweedieVariancePower = v).SetRange(1.0f, 2.0f).SetDefault(1.5f)
            .Describe("Tweedie variance power.  Must be between in range [1, 2).");
    }
}

// ---- reg:squarederror / reg:linear ------------------------------------------------------------

public sealed class SquaredErrorRegression : FitInterceptGlmLike
{
    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression, true);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        CheckWeights(info);
        ElementwiseGradient(Ctx, preds, info, Targets(info), static (p, y, w) => new GradientPair((p - y) * w, w), outGpair);
    }

    public override string DefaultEvalMetric => "rmse";
    public override void SaveConfig(JsonObject output) => output["name"] = "reg:squarederror";
    public override void LoadConfig(Json input) { }
}

// ---- reg:logistic / binary:logistic / binary:logitraw -----------------------------------------

public enum LogisticKind { Regression, Classification, Raw }

public sealed class LogisticObjective(LogisticKind kind) : FitInterceptGlmLike
{
    private readonly LogisticParam _param = new();

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));

    public override ObjInfo Task => kind == LogisticKind.Classification ? new(ObjTask.Binary) : new(ObjTask.Regression);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    internal static GradientPair Gradient(float predt, float label, float weight, float scalePosWeight)
    {
        var prediction = XMath.Sigmoid(predt);
        if (label == 1.0f) weight *= scalePosWeight;
        var hess = XMath.FMaxF(prediction * (1.0f - prediction), 1e-16f);
        return new GradientPair((prediction - label) * weight, hess * weight);
    }

    internal static float ProbToMarginValue(float value)
    {
        value = XMath.StdMin(XMath.StdMax(value, Constants.RtEps), 1.0f - Constants.RtEps);
        return XMath.Logit(value);
    }

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        if (iter == 0)
        {
            if (!ElementwiseValidate(info.Labels, v => v >= 0.0f && v <= 1.0f))
                Check.Fail("label must be in [0,1] for logistic regression");
            CheckWeights(info);
        }
        var spw = _param.ScalePosWeight;
        ElementwiseGradient(Ctx, preds, info, Targets(info), (p, y, w) => Gradient(p, y, w, spw), outGpair);
    }

    public override void PredTransform(HostDeviceVector<float> ioPreds)
    {
        if (kind != LogisticKind.Raw) ElementwiseTransform(Ctx, ioPreds, XMath.Sigmoid);
    }

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        if (Math.Abs(_param.ScalePosWeight - 1.0f) <= Constants.RtEps)
        {
            GlmLikeInit(info, baseScore);
        }
        else
        {
            CheckInitInputs(info);
            var nTargets = Targets(info);
            var dummyPredt = new HostDeviceVector<float>(info.Labels.Size, 0.0f);
            var gpair = new Tensor<GradientPair>(2);
            var spw = _param.ScalePosWeight;
            ElementwiseGradient(Ctx, dummyPredt, info, nTargets, (_, label, weight) =>
            {
                if (label == 1.0f) weight *= spw;
                return new GradientPair(-label * weight, weight);
            }, gpair);
            baseScore.Assign(Stats.FitStump(Ctx, gpair, nTargets));
        }
        if (kind == LogisticKind.Raw) ElementwiseTransform(Ctx, baseScore.Data, ProbToMarginValue);
    }

    public override void ProbToMargin(Tensor<float> baseScore)
    {
        if (kind == LogisticKind.Raw) return;
        Check.That(AllOf(baseScore, v => v >= 0.0f && v <= 1.0f), "base_score must be in (0,1) for the logistic loss.");
        ElementwiseTransform(Ctx, baseScore.Data, ProbToMarginValue);
    }

    public override string DefaultEvalMetric => kind == LogisticKind.Regression ? "rmse" : "logloss";

    public string Name => kind switch
    {
        LogisticKind.Regression => "reg:logistic",
        LogisticKind.Classification => "binary:logistic",
        _ => "binary:logitraw",
    };

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = Name;
        output["reg_loss_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input)
    {
        if (input.AsObject.TryGetValue("reg_loss_param", out var p)) _param.FromJson(p);
    }
}

// ---- reg:pseudohubererror ---------------------------------------------------------------------

public sealed class PseudoHuberRegression : FitIntercept
{
    private readonly PseudoHuberParam _param = new();

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));

    public override ObjInfo Task => new(ObjTask.Regression);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        var slope = _param.HuberSlope;
        Check.Ne((double)slope, 0.0, "slope for pseudo huber cannot be 0.");
        ElementwiseGradient(Ctx, preds, info, Targets(info), (predt, label, weight) =>
        {
            var z = predt - label;
            var hess = MathF.Abs(slope) / XMath.HypotF(slope, z);
            var grad = z * hess;
            return new GradientPair(grad * weight, hess * weight);
        }, outGpair);
    }

    public override string DefaultEvalMetric => "mphe";

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = "reg:pseudohubererror";
        output["pseudo_huber_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input)
    {
        if (!input.AsObject.TryGetValue("pseudo_huber_param", out var p)) return;
        _param.FromJson(p);
    }

    public override Json DefaultMetricConfig()
    {
        Check.That(_param.GetInitialised());
        var config = new JsonObject();
        config["name"] = DefaultEvalMetric;
        config["pseudo_huber_param"] = _param.ToJson();
        return config;
    }
}

// ---- reg:squaredlogerror ----------------------------------------------------------------------

public sealed class SquaredLogErrorRegression : ObjFunction
{
    private const string LabelErrorMsg = "label must be greater than -1 for rmsle so that log(label + 1) can be valid.";

    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    private static bool CheckLabel(float label) => label > -1.0f;

    private static float FirstOrderGradient(float predt, float label)
    {
        predt = XMath.FMaxF(predt, -1.0f + 1e-6f);
        return (XMath.Log1PF(predt) - XMath.Log1PF(label)) / (predt + 1.0f);
    }

    private static float SecondOrderGradient(float predt, float label)
    {
        predt = XMath.FMaxF(predt, -1.0f + 1e-6f);
        var hess = (-XMath.Log1PF(predt) + XMath.Log1PF(label) + 1.0f) / MathF.Pow(predt + 1.0f, 2.0f);
        return XMath.FMaxF(hess, 1e-6f);
    }

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        CheckInitInputs(info);
        if (!ElementwiseValidate(info.Labels, CheckLabel)) Check.Fail(LabelErrorMsg);
        var transformed = new Tensor<float>(info.Labels.Shape().ToArray());
        transformed.Data.Copy(info.Labels.Data);
        ElementwiseTransform(Ctx, transformed.Data, XMath.Log1PF);
        baseScore.Assign(info.Weights.Empty
            ? Stats.SampleMean(Ctx, transformed)
            : Stats.WeightedSampleMean(Ctx, transformed, info.Weights));
        var max = (double)float.MaxValue;
        for (var i = 0; i < baseScore.Size; i++)
            baseScore[i] = (float)XMath.StdMin(XMath.ExpM1(baseScore[i]), max);
    }

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        if (iter == 0)
        {
            if (!ElementwiseValidate(info.Labels, CheckLabel)) Check.Fail(LabelErrorMsg);
            CheckWeights(info);
        }
        ElementwiseGradient(Ctx, preds, info, Targets(info), static (p, y, w) =>
            new GradientPair(FirstOrderGradient(p, y) * w, SecondOrderGradient(p, y) * w), outGpair);
    }

    public override string DefaultEvalMetric => "rmsle";
    public override void SaveConfig(JsonObject output) => output["name"] = "reg:squaredlogerror";
    public override void LoadConfig(Json input) { }
}

// ---- count:poisson ----------------------------------------------------------------------------

public sealed class PoissonRegression : FitInterceptGlmLike
{
    internal const string LabelErrorMsg = "label must be non-negative for Poisson/Tweedie regression.";

    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        if (iter == 0)
        {
            if (!ElementwiseValidate(info.Labels, v => v >= 0.0f)) Check.Fail(LabelErrorMsg);
            CheckWeights(info);
        }
        ElementwiseGradient(Ctx, preds, info, Targets(info), static (predt, label, weight) =>
        {
            var mu = MathF.Exp(predt);
            var grad = (mu - label) * weight;
            var hess = (2.0f * mu + label) * weight / 3.0f;
            return new GradientPair(grad, hess);
        }, outGpair);
    }

    public override void PredTransform(HostDeviceVector<float> ioPreds) => ElementwiseTransform(Ctx, ioPreds, MathF.Exp);
    public override void ProbToMargin(Tensor<float> baseScore) => ElementwiseTransform(Ctx, baseScore.Data, MathF.Log);
    public override string DefaultEvalMetric => "poisson-nloglik";
    public override void SaveConfig(JsonObject output) => output["name"] = "count:poisson";
    public override void LoadConfig(Json input) { }
}

// ---- reg:gamma --------------------------------------------------------------------------------

public sealed class GammaRegression : FitInterceptGlmLike
{
    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        if (iter == 0)
        {
            if (!ElementwiseValidate(info.Labels, v => v > 0.0f)) Check.Fail("label must be positive for gamma regression.");
            CheckWeights(info);
        }
        ElementwiseGradient(Ctx, preds, info, Targets(info), static (predt, label, weight) =>
        {
            var prediction = MathF.Exp(predt);
            var ratio = label / prediction;
            var grad = 1.0f - ratio;
            var hess = (2.0f * ratio + 1.0f) / 3.0f;
            return new GradientPair(grad * weight, hess * weight);
        }, outGpair);
    }

    public override void PredTransform(HostDeviceVector<float> ioPreds) => ElementwiseTransform(Ctx, ioPreds, MathF.Exp);

    public override void ProbToMargin(Tensor<float> baseScore)
    {
        Check.That(AllOf(baseScore, v => v > 0.0f), "`base_score` must be greater than 0 for gamma regression");
        ElementwiseTransform(Ctx, baseScore.Data, MathF.Log);
    }

    public override string DefaultEvalMetric => "gamma-deviance";
    public override void SaveConfig(JsonObject output) => output["name"] = "reg:gamma";
    public override void LoadConfig(Json input) { }
}

// ---- reg:tweedie ------------------------------------------------------------------------------

public sealed class TweedieRegression : FitInterceptGlmLike
{
    private readonly TweedieRegressionParam _param = new();
    private string _metric = "";

    public override SortedSet<string> Configure(Args args)
    {
        var used = ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));
        // std::ostream default formatting (%g, precision 6).
        _metric = "tweedie-nloglik@" + Format.G(_param.TweedieVariancePower, 6);
        return used;
    }

    public override ObjInfo Task => new(ObjTask.Regression);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        if (iter == 0)
        {
            if (!ElementwiseValidate(info.Labels, v => v >= 0.0f)) Check.Fail(PoissonRegression.LabelErrorMsg);
            CheckWeights(info);
        }
        var rho = _param.TweedieVariancePower;
        ElementwiseGradient(Ctx, preds, info, Targets(info), (predt, label, weight) =>
        {
            var a = label * MathF.Exp((1.0f - rho) * predt);
            var b = MathF.Exp((2.0f - rho) * predt);
            var grad = (b - a) * weight;
            var hess = (rho * a + (3.0f - rho) * b) * weight / 3.0f;
            return new GradientPair(grad, hess);
        }, outGpair);
    }

    public override void PredTransform(HostDeviceVector<float> ioPreds) => ElementwiseTransform(Ctx, ioPreds, MathF.Exp);
    public override void ProbToMargin(Tensor<float> baseScore) => ElementwiseTransform(Ctx, baseScore.Data, MathF.Log);
    public override string DefaultEvalMetric => _metric;

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = "reg:tweedie";
        output["tweedie_regression_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input) => _param.FromJson(input["tweedie_regression_param"]);
}

// ---- reg:absoluteerror ------------------------------------------------------------------------

public sealed class MeanAbsoluteError : ObjFunction
{
    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression, false);
    public override uint Targets(MetaInfo info) => LabelTargets(info);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        var nTargets = Targets(info);
        var labels = info.Labels.HostView();
        var predt = Linalg.MakeTensorView(preds, info.NumRow, nTargets);
        var weights = new OptionalWeights(info.Weights);

        var scaleStats = RootResidualStats(Ctx, info, nTargets, (i, target) =>
            weights[i] * MathF.Sqrt(MathF.Abs(predt[i, target] - labels[i, target])));
        var scale = new float[nTargets];
        for (var t = 0; t < nTargets; t++)
        {
            if (!XMath.CloseTo(scaleStats[^1], 0.0))
            {
                var rootMean = scaleStats[t] / scaleStats[^1];
                scale[t] = (float)(rootMean * rootMean);
            }
        }

        outGpair.Reshape(info.NumRow, nTargets);
        var gpair = outGpair.HostView();
        Linalg.ElementWiseKernel2(Ctx, gpair, (i, j) =>
        {
            var residual = predt[i, j] - labels[i, j];
            var norm = XMath.HypotF(scale[j], residual);
            var curvature = norm > 0.0f ? scale[j] / norm : 1.0f;
            var weight = weights[i];
            gpair[i, j] = new GradientPair(weight * residual * curvature, weight * curvature);
        });
    }

    /// <summary>Per-target sums of <paramref name="residual"/> (reduced in double) plus the total weight.</summary>
    internal static double[] RootResidualStats(Context ctx, MetaInfo info, uint nTargets, Func<long, long, float> residual)
    {
        var weights = new OptionalWeights(info.Weights);
        var rootResidual = new float[info.NumRow];
        var scaleStats = new double[nTargets + 1];
        for (var target = 0; target < nTargets; target++)
        {
            if (info.NumRow != 0)
            {
                var t = target;
                Threading.ParallelFor(info.NumRow, ctx.Threads(), i => rootResidual[i] = residual(i, t));
                scaleStats[target] = Numeric.Reduce(ctx, rootResidual);
            }
        }
        scaleStats[^1] = weights.Sum(info.NumRow);
        Communicator.Allreduce(scaleStats.AsSpan(), Op.Sum);
        return scaleStats;
    }

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        CheckInitInputs(info);
        var nTargets = Targets(info);
        var alpha = new HostDeviceVector<float>([0.5f]);
        baseScore.Assign(RadixSelect.Run(Ctx, info.Labels, info.Weights, alpha, nTargets));
        Check.Eq((long)baseScore.Size, (long)nTargets);
    }

    public override string DefaultEvalMetric => "mae";
    public override void SaveConfig(JsonObject output) => output["name"] = "reg:absoluteerror";
    public override void LoadConfig(Json input) => Check.Eq(input["name"].AsString, "reg:absoluteerror");
}

// ---- binary:hinge -----------------------------------------------------------------------------

public sealed class HingeObj : ObjFunction
{
    private static void CheckHingeLabels(MetaInfo info)
    {
        if (!ElementwiseValidate(info.Labels, l => l == 0.0f || l == 1.0f))
            Check.Fail("label must be either 0 or 1 for hinge loss.");
    }

    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression);

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        CheckInitInputs(info);
        CheckHingeLabels(info);
        baseScore.Assign(info.Weights.Empty
            ? Stats.SampleMean(Ctx, info.Labels)
            : Stats.WeightedSampleMean(Ctx, info.Labels, info.Weights));
        for (var i = 0; i < baseScore.Size; i++)
        {
            var v = baseScore[i];
            baseScore[i] = v > 0.5f ? 1.0f : v < 0.5f ? -1.0f : 0.0f;
        }
    }

    public override uint Targets(MetaInfo info) => LabelTargets(info);

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq((long)info.Labels.Size, (long)preds.Size, "Invalid shape of labels.");
        if (iter == 0) CheckHingeLabels(info);
        ElementwiseGradient(Ctx, preds, info, Targets(info), static (margin, label, weight) =>
        {
            var y = label * 2.0f - 1.0f;
            if (margin * y < 1.0f) return new GradientPair(-y * weight, weight);
            return new GradientPair(0.0f, 1.17549435E-38f);
        }, outGpair);
    }

    public override void PredTransform(HostDeviceVector<float> ioPreds) =>
        ElementwiseTransform(Ctx, ioPreds, static m => m > 0.0f ? 1.0f : 0.0f);

    public override string DefaultEvalMetric => "error";
    public override void SaveConfig(JsonObject output) => output["name"] = "binary:hinge";
    public override void LoadConfig(Json input) { }
}

// ---- reg:normal -------------------------------------------------------------------------------

public sealed class NormalRegression : ObjFunction
{
    private const float NormalMinVariance = 1.1920929E-07f; // FLT_EPSILON

    private static GradientPair FinitePair(double grad, double hess)
    {
        var limit = Math.Sqrt(float.MaxValue) / 2.0;
        var magnitude = XMath.StdMax(Math.Abs(grad), hess);
        var scale = magnitude > limit ? limit / magnitude : 1.0;
        return new GradientPair((float)(grad * scale), (float)(hess * scale));
    }

    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression);

    public override uint Targets(MetaInfo info)
    {
        Check.Le(info.Labels.Shape(1), 1L, "Normal regression requires a single response column.");
        return 2;
    }

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        CheckInitInputs(info);
        Check.Eq(info.Labels.Shape(1), 1L, "Normal regression requires a single response column.");
        Check.Eq((long)preds.Size, info.NumRow * 2, "Normal regression requires two predictions per row: mean and log variance.");
        CheckWeights(info);

        var predt = Linalg.MakeTensorView(preds, info.NumRow, 2);
        var labels = info.Labels.HostView();
        var weights = new OptionalWeights(info.Weights);
        outGpair.Reshape(info.NumRow, 2);
        var gpair = outGpair.HostView();
        var logLimit = Math.Log(float.MaxValue);
        Linalg.ElementWiseKernel2(Ctx, labels, (i, _) =>
        {
            float mean = predt[i, 0], logVariance = predt[i, 1], label = labels[i, 0], weight = weights[i];
            var residual = (double)mean - label;
            var precision = Math.Exp(XMath.StdClamp(-(double)logVariance, -logLimit, logLimit));
            var standardizedResidual = (residual * residual + NormalMinVariance) * precision;
            gpair[i, 0] = FinitePair(weight * residual * precision, weight * precision);
            var gradLogVariance = 0.5f * (1.0f - standardizedResidual);
            var hessLogVariance = (1.0f + 2.0f * standardizedResidual) / 6.0f;
            gpair[i, 1] = FinitePair(weight * gradLogVariance, weight * hessLogVariance);
        });
    }

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        CheckInitInputs(info);
        Check.Eq(info.Labels.Shape(1), 1L, "Normal regression requires a single response column.");
        CheckWeights(info);

        var mean = info.Weights.Empty ? Stats.SampleMean(Ctx, info.Labels) : Stats.WeightedSampleMean(Ctx, info.Labels, info.Weights);
        Check.Eq(mean.Size, 1);
        var labels = info.Labels.HostView();
        var meanValue = mean[0];
        var weights = new OptionalWeights(info.Weights);
        // omp parallel for reduction(+): static partition, per-thread partial sums combined in thread order.
        var nThreads = Ctx.Threads();
        var partSq = new double[nThreads];
        var partW = new double[nThreads];
        Threading.ParallelFor(info.NumRow, nThreads, (i, tid) =>
        {
            var diff = (double)labels[i, 0] - meanValue;
            var weight = (double)weights[i];
            partSq[tid] += weight * diff * diff;
            partW[tid] += weight;
        });
        double sumSq = 0.0, sumW = 0.0;
        for (var t = 0; t < nThreads; t++)
        {
            sumSq += partSq[t];
            sumW += partW[t];
        }
        Span<double> stats = [sumSq, sumW];
        Communicator.Allreduce(stats, Op.Sum);
        Check.Gt(stats[1], 0.0);
        var output = new Tensor<float>([2]);
        output[0] = meanValue;
        output[1] = (float)Math.Log(stats[0] / stats[1] + NormalMinVariance);
        baseScore.Assign(output);
    }

    public override string DefaultEvalMetric => "normal-nloglik";
    public override void SaveConfig(JsonObject output) => output["name"] = "reg:normal";
    public override void LoadConfig(Json input) => Check.Eq(input["name"].AsString, "reg:normal");
}

// ---- survival:cox (regression_obj.cu) ---------------------------------------------------------

public sealed class CoxRegression : ObjFunction
{
    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);
    public override ObjInfo Task => new(ObjTask.Regression);

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        baseScore.Assign(Linalg.Zeros<float>(Targets(info)));
        PredTransform(baseScore.Data);
    }

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        Check.Ne(info.Labels.Size, 0, "label set cannot be empty");
        Check.Eq(preds.Size, info.Labels.Size, "labels are not correctly provided");
        var predsH = preds.RawArray;
        outGpair.Reshape(info.NumRow, Targets(info));
        var gpair = outGpair.HostView();
        var labelOrder = info.LabelAbsSort();
        long ndata = preds.Size;
        var isNullWeight = info.Weights.Size == 0;
        if (!isNullWeight) Check.Eq((long)info.Weights.Size, ndata, "Number of weights should be equal to number of data points.");

        var expPSum = 0.0;
        for (long i = 0; i < ndata; ++i) expPSum += MathF.Exp(predsH[labelOrder[i]]);

        var labels = info.Labels.HostView();
        double rK = 0, sK = 0, lastExpP = 0.0, lastAbsY = 0.0, accumulatedSum = 0;
        for (long i = 0; i < ndata; ++i)
        {
            var ind = labelOrder[i];
            double p = predsH[ind];
            var expP = Math.Exp(p);
            double w = info.GetWeight(ind);
            double y = labels.Flat(ind);
            var absY = Math.Abs(y);
            accumulatedSum += lastExpP;
            if (lastAbsY < absY)
            {
                expPSum -= accumulatedSum;
                accumulatedSum = 0;
            }
            else
            {
                Check.That(lastAbsY <= absY, "CoxRegression: labels must be in sorted order, MetaInfo::LabelArgsort failed!");
            }
            if (y > 0)
            {
                rK += 1.0 / expPSum;
                sK += 1.0 / (expPSum * expPSum);
            }
            var grad = expP * rK - (y > 0 ? 1.0f : 0.0f);
            var hess = expP * rK - expP * expP * sK;
            gpair.Flat(ind) = new GradientPair((float)(grad * w), (float)(hess * w));
            lastAbsY = absY;
            lastExpP = expP;
        }
    }

    public override void PredTransform(HostDeviceVector<float> ioPreds) => ElementwiseTransform(Ctx, ioPreds, MathF.Exp);
    public override void EvalTransform(HostDeviceVector<float> ioPreds) => PredTransform(ioPreds);
    public override void ProbToMargin(Tensor<float> baseScore) => ElementwiseTransform(Ctx, baseScore.Data, MathF.Log);
    public override string DefaultEvalMetric => "cox-nloglik";
    public override void SaveConfig(JsonObject output) => output["name"] = "survival:cox";
    public override void LoadConfig(Json input) { }
}
