// Ports of src/metric/{metric_common,elementwise_metric,alpha_metric}.h and the element-wise,
// quantile, expectile, normal, pseudo-huber, multiclass and survival metrics.
using System.Globalization;
using XGBoost.Collective;
using XGBoost.Objectives;

namespace XGBoost.Metrics;

public readonly record struct PackedReduceResult(double Residue, double Weights);

/// <summary><c>MetricNoCache</c>: evaluation only needs the meta info.</summary>
public abstract class MetricNoCache : Metric
{
    public abstract double Eval(HostDeviceVector<float> preds, MetaInfo info);

    public sealed override double Evaluate(HostDeviceVector<float> preds, DMatrix fmat) => Eval(preds, fmat.Info);

    internal Context Context => Ctx;
    internal void UseContext(Context ctx) => SetContext(ctx);
}

public static class MetricUtils
{
    public static void CheckRowWeights(MetaInfo info)
    {
        if (info.Weights.Empty) return;
        Check.That(info.GroupPtr.Count == 0, "Row-wise metric does not support query group weights.");
        Check.Eq((long)info.Weights.Size, info.NumRow, "Number of weights should be equal to the number of data points.");
    }

    public static (double, double) GlobalSum(double a, double b)
    {
        Span<double> dat = [a, b];
        Communicator.Allreduce(dat, Op.Sum);
        return (dat[0], dat[1]);
    }

    public static double GlobalRatio(double dividend, double divisor)
    {
        (dividend, divisor) = GlobalSum(dividend, divisor);
        return divisor <= 0 ? double.NaN : dividend / divisor;
    }

    /// <summary><c>elementwise::detail::EvalCpu</c>: blocked reduction of <c>eval(label, predt) * weight</c>.</summary>
    public static PackedReduceResult EvalRows(Context ctx, HostDeviceVector<float> preds, MetaInfo info, Func<float, float, float> eval)
    {
        var labels = info.Labels.HostView();
        var predts = preds.RawArray;
        var weights = new OptionalWeights(info.Weights);
        var nThreads = ctx.Threads();
        var scoreTloc = new double[nThreads];
        var weightTloc = new double[nThreads];
        var nCols = labels.Shape(1);
        Threading.ParallelFor1d(labels.Size, nThreads, 2048, block =>
        {
            var sumScore = 0.0;
            var sumWeight = 0.0;
            for (var i = block.Begin; i < block.End; ++i)
            {
                var sampleId = i / nCols;
                var targetId = i - sampleId * nCols;
                var weight = weights[sampleId];
                var residue = eval(labels[sampleId, targetId], predts[i]) * weight;
                sumScore += residue;
                sumWeight += weight;
            }
            var tIdx = Threading.ThreadNum;
            scoreTloc[tIdx] += sumScore;
            weightTloc[tIdx] += sumWeight;
        });
        double residueSum = 0.0, weightsSum = 0.0;
        foreach (var v in scoreTloc) residueSum += v;
        foreach (var v in weightTloc) weightsSum += v;
        return new PackedReduceResult(residueSum, weightsSum);
    }

    /// <summary><c>alpha::detail::EvalCpu</c> over predictions shaped (rows, alphas, targets).</summary>
    public static PackedReduceResult EvalAlpha(Context ctx, HostDeviceVector<float> preds, MetaInfo info, float[] alpha,
        Func<float, float, float, float> eval)
    {
        var labels = info.Labels.HostView();
        var nTargets = labels.Shape(1);
        var nAlpha = alpha.Length;
        var predts = Linalg.MakeTensorView(preds, info.NumRow, nAlpha, nTargets);
        var weights = new OptionalWeights(info.Weights);
        var nThreads = ctx.Threads();
        var scores = new double[nThreads];
        var weightsSum = new double[nThreads];
        Threading.ParallelFor1d(predts.Size, nThreads, 2048, block =>
        {
            double score = 0.0, weightSum = 0.0;
            for (var i = block.Begin; i < block.End; ++i)
            {
                var row = i / (nAlpha * nTargets);
                var rem = i - row * nAlpha * nTargets;
                var alphaIdx = rem / nTargets;
                var target = rem - alphaIdx * nTargets;
                var weight = weights[row];
                var loss = eval(predts[row, alphaIdx, target], labels[row, target], alpha[alphaIdx]);
                score += loss * weight;
                weightSum += weight;
            }
            var tid = Threading.ThreadNum;
            scores[tid] += score;
            weightsSum[tid] += weightSum;
        });
        double s = 0.0, w = 0.0;
        foreach (var v in scores) s += v;
        foreach (var v in weightsSum) w += v;
        return new PackedReduceResult(s, w);
    }

    /// <summary><c>std::ostream &lt;&lt; float</c> with the default precision (6).</summary>
    public static string StreamFloat(double v) => Format.G(v, 6);
}

/// <summary><c>EvalEWiseMetric&lt;EvalFn&gt;</c>.</summary>
public sealed class EvalEWiseMetric(string name, Func<float, float, float> eval, Func<double, double, double> getFinal) : MetricNoCache
{
    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        Check.Eq(preds.Size, info.Labels.Size,
            "label and prediction size not match, hint: use merror or mlogloss for multi-class classification");
        if (info.Labels.Size != 0) Check.Ne(info.Labels.Shape(1), 0L);
        MetricUtils.CheckRowWeights(info);
        var result = MetricUtils.EvalRows(Ctx, preds, info, eval);
        var (a, b) = MetricUtils.GlobalSum(result.Residue, result.Weights);
        return getFinal(a, b);
    }

    public override string Name => name;

    internal static double MeanFinal(double esum, double wsum) => wsum == 0 ? esum : esum / wsum;
    internal static double SqrtFinal(double esum, double wsum) => wsum == 0 ? Math.Sqrt(esum) : Math.Sqrt(esum / wsum);

    public static Metric Rmse() => new EvalEWiseMetric("rmse", static (label, pred) =>
    {
        var diff = label - pred;
        return diff * diff;
    }, SqrtFinal);

    public static Metric Rmsle() => new EvalEWiseMetric("rmsle", static (label, pred) =>
    {
        var diff = XMath.Log1PF(label) - XMath.Log1PF(pred);
        return diff * diff;
    }, SqrtFinal);

    public static Metric Mae() => new EvalEWiseMetric("mae", static (label, pred) => MathF.Abs(label - pred), MeanFinal);

    public static Metric Mape() => new EvalEWiseMetric("mape", static (label, pred) => MathF.Abs((label - pred) / label), MeanFinal);

    public static Metric LogLoss() => new EvalEWiseMetric("logloss", static (y, py) =>
    {
        static float XLogY(float x, float y)
        {
            const float eps = 1e-16f;
            return x - 0.0f == 0.0f ? 0.0f : x * MathF.Log(XMath.StdMax(y, eps));
        }
        var pneg = 1.0f - py;
        return XLogY(-y, py) + XLogY(-(1.0f - y), pneg);
    }, MeanFinal);

    public static Metric Error(string? param)
    {
        float threshold;
        bool hasParam;
        if (param is not null)
        {
            Check.That(Format.TryParseFloatPrefix(param.TrimStart(), out var d, out var consumed) && consumed > 0,
                "unable to parse the threshold value for the error metric");
            threshold = (float)d;
            hasParam = true;
        }
        else
        {
            threshold = 0.5f;
            hasParam = false;
        }
        var name = hasParam && threshold != 0.5f ? "error@" + MetricUtils.StreamFloat(threshold) : "error";
        return new EvalEWiseMetric(name, (label, pred) => pred > threshold ? 1.0f - label : label, MeanFinal);
    }

    public static Metric PoissonNLogLik() => new EvalEWiseMetric("poisson-nloglik", static (y, py) =>
    {
        const float eps = 1e-16f;
        if (py < eps) py = eps;
        return XMath.LogGamma(y + 1.0f) + py - MathF.Log(py) * y;
    }, MeanFinal);

    public static Metric GammaDeviance() => new EvalEWiseMetric("gamma-deviance", static (label, predt) =>
    {
        predt += Constants.RtEps;
        label += Constants.RtEps;
        return MathF.Log(predt / label) + label / predt - 1;
    }, static (esum, wsum) =>
    {
        if (wsum <= 0) wsum = Constants.RtEps;
        return 2 * esum / wsum;
    });

    public static Metric GammaNLogLik() => new EvalEWiseMetric("gamma-nloglik", static (y, py) =>
    {
        py = XMath.StdMax(py, 1e-6f);
        const float psi = 1.0f;
        var theta = (float)(-1.0 / py);
        var a = psi;
        var b = -MathF.Log(-theta);
        var c = 0f;
        return -((y * theta - b) / a + c);
    }, MeanFinal);

    public static Metric TweedieNLogLik(string? param)
    {
        Check.That(param is not null, "tweedie-nloglik must be in format tweedie-nloglik@rho");
        Format.TryParseFloatPrefix(param!.TrimStart(), out var d, out _); // atof
        var rho = (float)d;
        Check.That(rho < 2 && rho >= 1, "tweedie variance power must be in interval [1, 2)");
        return new EvalEWiseMetric("tweedie-nloglik@" + MetricUtils.StreamFloat(rho), (y, p) =>
        {
            var a = y * MathF.Exp((1 - rho) * MathF.Log(p)) / (1 - rho);
            var b = MathF.Exp((2 - rho) * MathF.Log(p)) / (2 - rho);
            return -a + b;
        }, MeanFinal);
    }
}

// ---- mphe -------------------------------------------------------------------------------------

public sealed class PseudoErrorLoss : MetricNoCache
{
    private readonly PseudoHuberParam _param = new();

    public override string Name => "mphe";

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));

    public override void LoadConfig(Json input) => _param.FromJson(input["pseudo_huber_param"]);

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = Name;
        output["pseudo_huber_param"] = _param.ToJson();
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        Check.Eq(info.Labels.Shape(0), info.NumRow);
        MetricUtils.CheckRowWeights(info);
        var slope = _param.HuberSlope;
        Check.Ne((double)slope, 0.0, "slope for pseudo huber cannot be 0.");
        var result = MetricUtils.EvalRows(Ctx, preds, info, (label, pred) =>
        {
            var a = label - pred;
            return XMath.Sqr(slope) * (MathF.Sqrt(1 + XMath.Sqr(a / slope)) - 1);
        });
        var (s, w) = MetricUtils.GlobalSum(result.Residue, result.Weights);
        return w == 0 ? s : s / w;
    }
}

// ---- quantile / expectile ---------------------------------------------------------------------

public sealed class QuantileError : MetricNoCache
{
    private float[] _alpha = [];
    private readonly QuantileLossParam _param = new();

    public override SortedSet<string> Configure(Args args)
    {
        var used = ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));
        _param.Validate();
        _alpha = (float[])_param.QuantileAlpha.Clone();
        return used;
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        Check.That(_alpha.Length != 0);
        Check.Eq(info.Labels.Shape(0), info.NumRow, "Invalid shape of labels.");
        Check.Eq((long)preds.Size, (long)info.Labels.Size * _alpha.Length,
            "Prediction size must equal label size times the number of alpha values.");
        if (info.NumRow == 0)
        {
            var (s0, w0) = MetricUtils.GlobalSum(0.0, 0.0);
            Check.Gt(w0, 0.0);
            return s0 / w0;
        }
        MetricUtils.CheckRowWeights(info);
        Check.Ne(info.Labels.Shape(1), 0L);
        var result = MetricUtils.EvalAlpha(Ctx, preds, info, _alpha, static (pred, label, alpha) =>
        {
            var d = label - pred;
            float sign = d >= 0.0f ? 1.0f : 0.0f;
            return alpha * sign * d - (1.0f - alpha) * (1.0f - sign) * d;
        });
        var (s, w) = MetricUtils.GlobalSum(result.Residue, result.Weights);
        Check.Gt(w, 0.0);
        return s / w;
    }

    public override string Name => "quantile";

    public override void LoadConfig(Json input)
    {
        if (input.AsObject.TryGetValue("quantile_loss_param", out var p))
        {
            _param.FromJson(p);
            Check.Eq(input["name"].AsString, "quantile");
            _param.Validate();
            _alpha = (float[])_param.QuantileAlpha.Clone();
        }
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = Name;
        output["quantile_loss_param"] = _param.ToJson();
    }
}

public sealed class ExpectileError : MetricNoCache
{
    private float[] _alpha = [];
    private readonly ExpectileLossParam _param = new();

    public override SortedSet<string> Configure(Args args)
    {
        var used = ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));
        _param.Validate();
        _alpha = (float[])_param.ExpectileAlpha.Clone();
        return used;
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        Check.That(_alpha.Length != 0);
        Check.Eq(info.Labels.Shape(0), info.NumRow, "Invalid shape of labels.");
        Check.Eq((long)preds.Size, (long)info.Labels.Size * _alpha.Length,
            "Prediction size must equal label size times the number of alpha values.");
        if (info.NumRow == 0)
        {
            var (s0, w0) = MetricUtils.GlobalSum(0.0, 0.0);
            Check.Gt(w0, 0.0);
            return s0 / w0;
        }
        MetricUtils.CheckRowWeights(info);
        Check.Ne(info.Labels.Shape(1), 0L);
        var result = MetricUtils.EvalAlpha(Ctx, preds, info, _alpha, static (pred, label, alpha) =>
        {
            var diff = pred - label;
            var weightScale = diff >= 0.0f ? 1.0f - alpha : alpha;
            return weightScale * diff * diff;
        });
        var (s, w) = MetricUtils.GlobalSum(result.Residue, result.Weights);
        Check.Gt(w, 0.0);
        return s / w;
    }

    public override string Name => "expectile";

    public override void LoadConfig(Json input)
    {
        if (input.AsObject.TryGetValue("expectile_loss_param", out var p))
        {
            _param.FromJson(p);
            Check.Eq(input["name"].AsString, "expectile");
            _param.Validate();
            _alpha = (float[])_param.ExpectileAlpha.Clone();
        }
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = Name;
        output["expectile_loss_param"] = _param.ToJson();
    }
}

// ---- normal-nloglik ---------------------------------------------------------------------------

public sealed class NormalNLogLik : MetricNoCache
{
    public override SortedSet<string> Configure(Args args) => new(StringComparer.Ordinal);

    private static float EvalRow(float label, float mean, float logVariance)
    {
        const float logTwoPi = 1.8378770664093453f;
        var residual = label - mean;
        var standardizedResidual = residual == 0.0f ? 0.0f : MathF.Exp(2.0f * MathF.Log(MathF.Abs(residual)) - logVariance);
        return 0.5f * (logTwoPi + logVariance + standardizedResidual);
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        Check.Eq(info.Labels.Shape(1), 1L, "Normal NLL requires a single response column.");
        Check.Eq((long)preds.Size, info.NumRow * 2, "Normal NLL requires two predictions per row: mean and log variance.");
        MetricUtils.CheckRowWeights(info);

        var labels = info.Labels.HostView();
        var predts = preds.RawArray;
        var weights = new OptionalWeights(info.Weights);
        var nThreads = Ctx.Threads();
        var scoreTloc = new double[nThreads];
        var weightTloc = new double[nThreads];
        var nCols = labels.Shape(1);
        Threading.ParallelFor1d(labels.Size, nThreads, 2048, block =>
        {
            double sumScore = 0.0, sumWeight = 0.0;
            for (var i = block.Begin; i < block.End; ++i)
            {
                var sampleId = i / nCols;
                var targetId = i - sampleId * nCols;
                var weight = weights[sampleId];
                var residue = EvalRow(labels[sampleId, targetId], predts[sampleId * 2], predts[sampleId * 2 + 1]) * weight;
                sumScore += residue;
                sumWeight += weight;
            }
            var tIdx = Threading.ThreadNum;
            scoreTloc[tIdx] += sumScore;
            weightTloc[tIdx] += sumWeight;
        });
        double rs = 0, ws = 0;
        foreach (var v in scoreTloc) rs += v;
        foreach (var v in weightTloc) ws += v;
        var (s, w) = MetricUtils.GlobalSum(rs, ws);
        return w == 0.0 ? s : s / w;
    }

    public override string Name => "normal-nloglik";
}

// ---- merror / mlogloss ------------------------------------------------------------------------

public sealed class EvalMClass(bool logLoss) : MetricNoCache
{
    private static float EvalRowError(int label, ReadOnlySpan<float> pred) => XMath.FindMaxIndex(pred) != label ? 1.0f : 0.0f;

    private static float EvalRowLogLoss(int label, ReadOnlySpan<float> pred)
    {
        const float eps = 1e-16f;
        var k = label;
        return pred[k] > eps ? -MathF.Log(pred[k]) : -MathF.Log(eps);
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        MetricUtils.CheckRowWeights(info);
        if (info.Labels.Size == 0)
        {
            Check.Eq(preds.Size, 0);
        }
        else
        {
            Check.Eq(info.Labels.Shape(1), 1L, "`merror` and `mlogloss` do not support multi-target labels.");
            Check.That(preds.Size % info.Labels.Size == 0, "label and prediction size not match");
        }
        double d0 = 0.0, d1 = 0.0;
        if (info.Labels.Size != 0)
        {
            var nclass = preds.Size / info.Labels.Size;
            Check.Ge(nclass, 1, "mlogloss and merror are only used for multi-class classification, use logloss for binary classification");
            var hLabels = info.Labels.Data.RawArray;
            var hWeights = info.Weights.RawArray;
            var hPreds = preds.RawArray;
            var isNullWeight = info.Weights.Size == 0;
            var nThreads = Ctx.Threads();
            var scoresTloc = new double[nThreads];
            var weightsTloc = new double[nThreads];
            var labelError = 0;
            Threading.ParallelFor(info.Labels.Size, nThreads, (idx, tid) =>
            {
                var weight = isNullWeight ? 1.0f : hWeights[idx];
                var label = (int)hLabels[idx];
                if (label >= 0 && label < nclass)
                {
                    var row = hPreds.AsSpan((int)idx * nclass, nclass);
                    scoresTloc[tid] += (logLoss ? EvalRowLogLoss(label, row) : EvalRowError(label, row)) * weight;
                    weightsTloc[tid] += weight;
                }
                else
                {
                    Volatile.Write(ref labelError, label);
                }
            });
            foreach (var v in scoresTloc) d0 += v;
            foreach (var v in weightsTloc) d1 += v;
            Check.That(labelError >= 0 && labelError < nclass,
                $"MultiClassEvaluation: label must be in [0, num_class), num_class={nclass} but found {labelError} in label");
        }
        var (s, w) = MetricUtils.GlobalSum(d0, d1);
        return s / w;
    }

    public override string Name => logLoss ? "mlogloss" : "merror";
}

// ---- survival metrics -------------------------------------------------------------------------

internal static class SurvivalReduce
{
    public static PackedReduceResult Reduce(Context ctx, MetaInfo info, HostDeviceVector<float> preds, Func<double, double, double, double> evalRow)
    {
        var ndata = info.LabelsLowerBound.Size;
        Check.Eq(ndata, info.LabelsUpperBound.Size);
        var lower = info.LabelsLowerBound.RawArray;
        var upper = info.LabelsUpperBound.RawArray;
        var weights = info.Weights.RawArray;
        var hasWeights = info.Weights.Size != 0;
        var hPreds = preds.RawArray;
        var nThreads = ctx.Threads();
        var scoreTloc = new double[nThreads];
        var weightTloc = new double[nThreads];
        Threading.ParallelFor(ndata, nThreads, (i, tid) =>
        {
            var wt = hasWeights ? weights[i] : 1.0;
            scoreTloc[tid] += evalRow(lower[i], upper[i], hPreds[i]) * wt;
            weightTloc[tid] += wt;
        });
        double s = 0, w = 0;
        foreach (var v in scoreTloc) s += v;
        foreach (var v in weightTloc) w += v;
        return new PackedReduceResult(s, w);
    }

    public static double Finish(MetaInfo info, HostDeviceVector<float> preds, Func<PackedReduceResult> reduce)
    {
        MetricUtils.CheckRowWeights(info);
        Check.Eq(preds.Size, info.LabelsLowerBound.Size);
        Check.Eq(preds.Size, info.LabelsUpperBound.Size);
        var r = reduce();
        var (s, w) = MetricUtils.GlobalSum(r.Residue, r.Weights);
        return EvalEWiseMetric.MeanFinal(s, w);
    }
}

public sealed class IntervalRegressionAccuracy : MetricNoCache
{
    public override SortedSet<string> Configure(Args args)
    {
        Check.That(Ctx is not null);
        return new SortedSet<string>(StringComparer.Ordinal);
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info) =>
        SurvivalReduce.Finish(info, preds, () => SurvivalReduce.Reduce(Ctx, info, preds, static (lo, hi, logPred) =>
        {
            var pred = Math.Exp(logPred);
            return pred >= lo && pred <= hi ? 1.0 : 0.0;
        }));

    public override string Name => "interval-regression-accuracy";
}

public sealed class AFTNLogLikDispatcher : MetricNoCache
{
    private readonly AFTParam _param = new();
    private Func<double, double, double, double>? _loss;
    private readonly AFTParam _innerParam = new();

    public override string Name => "aft-nloglik";

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        Check.That(_loss is not null, "AFT metric must be configured first, with distribution type and scale");
        return SurvivalReduce.Finish(info, preds, () => SurvivalReduce.Reduce(Ctx, info, preds, _loss!));
    }

    public override SortedSet<string> Configure(Args args)
    {
        var used = ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));
        // The inner EvalAFTNLogLik owns its own AFTParam, configured from the same arguments.
        var innerUsed = ParameterUtils.GetUsedParameters(args, _innerParam.UpdateAllowUnknown(args));
        var scale = (double)_innerParam.AftLossDistributionScale;
        _loss = _param.AftLossDistribution switch
        {
            ProbabilityDistributionType.Normal => (lo, hi, p) => AFTLoss<NormalDistribution>.Loss(lo, hi, p, scale),
            ProbabilityDistributionType.Logistic => (lo, hi, p) => AFTLoss<LogisticDistribution>.Loss(lo, hi, p, scale),
            ProbabilityDistributionType.Extreme => (lo, hi, p) => AFTLoss<ExtremeDistribution>.Loss(lo, hi, p, scale),
            _ => Check.Fail<Func<double, double, double, double>>("Unknown probability distribution"),
        };
        used.UnionWith(innerUsed);
        return used;
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = Name;
        output["aft_loss_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input) => _param.FromJson(input["aft_loss_param"]);
}
