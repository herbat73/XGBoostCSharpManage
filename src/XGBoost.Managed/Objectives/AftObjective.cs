// Ports of src/objective/aft_obj.cc, src/common/survival_util.h and probability_distribution.h.
namespace XGBoost.Objectives;

public enum ProbabilityDistributionType { Normal = 0, Logistic = 1, Extreme = 2 }

public enum CensoringType : byte { Uncensored, RightCensored, LeftCensored, IntervalCensored }

public sealed class AFTParam : XGBoostParameter<AFTParam>
{
    public ProbabilityDistributionType AftLossDistribution;
    public float AftLossDistributionScale;

    protected override void Declare(ParamManager<AFTParam> m)
    {
        EnumField<ProbabilityDistributionType>(m, "aft_loss_distribution", p => p.AftLossDistribution, (p, v) => p.AftLossDistribution = v)
            .SetDefault(ProbabilityDistributionType.Normal)
            .AddEnum("normal", ProbabilityDistributionType.Normal)
            .AddEnum("logistic", ProbabilityDistributionType.Logistic)
            .AddEnum("extreme", ProbabilityDistributionType.Extreme)
            .Describe("Choice of distribution for the noise term in Accelerated Failure Time model");
        Field(m, "aft_loss_distribution_scale", p => p.AftLossDistributionScale, (p, v) => p.AftLossDistributionScale = v).SetDefault(1.0f)
            .Describe("Scaling factor used to scale the distribution in Accelerated Failure Time model");
    }
}

public interface IDistribution
{
    static abstract double PDF(double z);
    static abstract double CDF(double z);
    static abstract double GradPDF(double z);
    static abstract double HessPDF(double z);
    static abstract double LimitGradAtInfPred(CensoringType censorType, bool sign, double sigma);
    static abstract double LimitHessAtInfPred(CensoringType censorType, bool sign, double sigma);
}

internal static class Aft
{
    public const double MinGradient = -15.0;
    public const double MaxGradient = 15.0;
    public const double MinHessian = 1e-16;
    public const double MaxHessian = 15.0;
    public const double Eps = 1e-12;
    public const double PI = 3.14159265358979323846;

    public static double Clip(double x, double xMin, double xMax)
    {
        if (x < xMin) return xMin;
        if (x > xMax) return xMax;
        return x;
    }

    /// <summary>C <c>fmax</c>.</summary>
    public static double FMax(double a, double b)
    {
        if (double.IsNaN(a)) return b;
        if (double.IsNaN(b)) return a;
        return a > b ? a : b;
    }
}

public readonly struct NormalDistribution : IDistribution
{
    public static double PDF(double z) => Math.Exp(-z * z / 2.0) / Math.Sqrt(2.0 * Aft.PI);
    public static double CDF(double z) => 0.5 * (1 + XMath.Erf(z / Math.Sqrt(2.0)));
    public static double GradPDF(double z) => -z * PDF(z);
    public static double HessPDF(double z) => (z * z - 1.0) * PDF(z);

    public static double LimitGradAtInfPred(CensoringType censorType, bool sign, double sigma) => censorType switch
    {
        CensoringType.Uncensored => sign ? Aft.MinGradient : Aft.MaxGradient,
        CensoringType.RightCensored => sign ? Aft.MinGradient : 0.0,
        CensoringType.LeftCensored => sign ? 0.0 : Aft.MaxGradient,
        CensoringType.IntervalCensored => sign ? Aft.MinGradient : Aft.MaxGradient,
        _ => double.NaN,
    };

    public static double LimitHessAtInfPred(CensoringType censorType, bool sign, double sigma) => censorType switch
    {
        CensoringType.Uncensored => 1.0 / (sigma * sigma),
        CensoringType.RightCensored => sign ? 1.0 / (sigma * sigma) : Aft.MinHessian,
        CensoringType.LeftCensored => sign ? Aft.MinHessian : 1.0 / (sigma * sigma),
        CensoringType.IntervalCensored => 1.0 / (sigma * sigma),
        _ => double.NaN,
    };
}

public readonly struct LogisticDistribution : IDistribution
{
    public static double PDF(double z)
    {
        var w = Math.Exp(z);
        var sqrtDenominator = 1 + w;
        if (double.IsInfinity(w) || double.IsInfinity(w * w)) return 0.0;
        return w / (sqrtDenominator * sqrtDenominator);
    }

    public static double CDF(double z)
    {
        var w = Math.Exp(z);
        return double.IsInfinity(w) ? 1.0 : w / (1 + w);
    }

    public static double GradPDF(double z)
    {
        var w = Math.Exp(z);
        return double.IsInfinity(w) ? 0.0 : PDF(z) * (1 - w) / (1 + w);
    }

    public static double HessPDF(double z)
    {
        var w = Math.Exp(z);
        if (double.IsInfinity(w) || double.IsInfinity(w * w)) return 0.0;
        return PDF(z) * (w * w - 4 * w + 1) / ((1 + w) * (1 + w));
    }

    public static double LimitGradAtInfPred(CensoringType censorType, bool sign, double sigma) => censorType switch
    {
        CensoringType.Uncensored => sign ? -1.0 / sigma : 1.0 / sigma,
        CensoringType.RightCensored => sign ? -1.0 / sigma : 0.0,
        CensoringType.LeftCensored => sign ? 0.0 : 1.0 / sigma,
        CensoringType.IntervalCensored => sign ? -1.0 / sigma : 1.0 / sigma,
        _ => double.NaN,
    };

    public static double LimitHessAtInfPred(CensoringType censorType, bool sign, double sigma) => Aft.MinHessian;
}

public readonly struct ExtremeDistribution : IDistribution
{
    public static double PDF(double z)
    {
        var w = Math.Exp(z);
        return double.IsInfinity(w) ? 0.0 : w * Math.Exp(-w);
    }

    public static double CDF(double z)
    {
        var w = Math.Exp(z);
        return 1 - Math.Exp(-w);
    }

    public static double GradPDF(double z)
    {
        var w = Math.Exp(z);
        return double.IsInfinity(w) ? 0.0 : (1 - w) * PDF(z);
    }

    public static double HessPDF(double z)
    {
        var w = Math.Exp(z);
        if (double.IsInfinity(w) || double.IsInfinity(w * w)) return 0.0;
        return (w * w - 3 * w + 1) * PDF(z);
    }

    public static double LimitGradAtInfPred(CensoringType censorType, bool sign, double sigma) => censorType switch
    {
        CensoringType.Uncensored => sign ? Aft.MinGradient : 1.0 / sigma,
        CensoringType.RightCensored => sign ? Aft.MinGradient : 0.0,
        CensoringType.LeftCensored => sign ? 0.0 : 1.0 / sigma,
        CensoringType.IntervalCensored => sign ? Aft.MinGradient : 1.0 / sigma,
        _ => double.NaN,
    };

    public static double LimitHessAtInfPred(CensoringType censorType, bool sign, double sigma) => censorType switch
    {
        CensoringType.Uncensored or CensoringType.RightCensored => sign ? Aft.MaxHessian : Aft.MinHessian,
        CensoringType.LeftCensored => Aft.MinHessian,
        CensoringType.IntervalCensored => sign ? Aft.MaxHessian : Aft.MinHessian,
        _ => double.NaN,
    };
}

/// <summary><c>AFTLoss&lt;Distribution&gt;</c>.</summary>
public static class AFTLoss<TDist> where TDist : IDistribution
{
    public static double Loss(double yLower, double yUpper, double yPred, double sigma)
    {
        var logYLower = Math.Log(yLower);
        var logYUpper = Math.Log(yUpper);
        double cost;
        if (yLower == yUpper)
        {
            var z = (logYLower - yPred) / sigma;
            var pdf = TDist.PDF(z);
            cost = -Math.Log(Aft.FMax(pdf / (sigma * yLower), Aft.Eps));
        }
        else
        {
            double cdfU, cdfL;
            if (double.IsInfinity(yUpper))
            {
                cdfU = 1;
            }
            else
            {
                var zU = (logYUpper - yPred) / sigma;
                cdfU = TDist.CDF(zU);
            }
            if (yLower <= 0.0)
            {
                cdfL = 0;
            }
            else
            {
                var zL = (logYLower - yPred) / sigma;
                cdfL = TDist.CDF(zL);
            }
            cost = -Math.Log(Aft.FMax(cdfU - cdfL, Aft.Eps));
        }
        return cost;
    }

    public static double Gradient(double yLower, double yUpper, double yPred, double sigma)
    {
        var logYLower = Math.Log(yLower);
        var logYUpper = Math.Log(yUpper);
        double numerator, denominator;
        CensoringType censorType;
        bool zSign;
        if (yLower == yUpper)
        {
            var z = (logYLower - yPred) / sigma;
            var pdf = TDist.PDF(z);
            var gradPdf = TDist.GradPDF(z);
            censorType = CensoringType.Uncensored;
            numerator = gradPdf;
            denominator = sigma * pdf;
            zSign = z > 0;
        }
        else
        {
            double zU = 0.0, zL = 0.0, pdfU, pdfL, cdfU, cdfL;
            censorType = CensoringType.IntervalCensored;
            if (double.IsInfinity(yUpper))
            {
                pdfU = 0;
                cdfU = 1;
                censorType = CensoringType.RightCensored;
            }
            else
            {
                zU = (logYUpper - yPred) / sigma;
                pdfU = TDist.PDF(zU);
                cdfU = TDist.CDF(zU);
            }
            if (yLower <= 0.0)
            {
                pdfL = 0;
                cdfL = 0;
                censorType = CensoringType.LeftCensored;
            }
            else
            {
                zL = (logYLower - yPred) / sigma;
                pdfL = TDist.PDF(zL);
                cdfL = TDist.CDF(zL);
            }
            zSign = zU > 0 || zL > 0;
            numerator = pdfU - pdfL;
            denominator = sigma * (cdfU - cdfL);
        }
        var gradient = numerator / denominator;
        if (denominator < Aft.Eps && (double.IsNaN(gradient) || double.IsInfinity(gradient)))
            gradient = TDist.LimitGradAtInfPred(censorType, zSign, sigma);
        return Aft.Clip(gradient, Aft.MinGradient, Aft.MaxGradient);
    }

    public static double Hessian(double yLower, double yUpper, double yPred, double sigma)
    {
        var logYLower = Math.Log(yLower);
        var logYUpper = Math.Log(yUpper);
        double numerator, denominator;
        CensoringType censorType;
        bool zSign;
        if (yLower == yUpper)
        {
            var z = (logYLower - yPred) / sigma;
            var pdf = TDist.PDF(z);
            var gradPdf = TDist.GradPDF(z);
            var hessPdf = TDist.HessPDF(z);
            censorType = CensoringType.Uncensored;
            numerator = -(pdf * hessPdf - gradPdf * gradPdf);
            denominator = sigma * sigma * pdf * pdf;
            zSign = z > 0;
        }
        else
        {
            double zU = 0.0, zL = 0.0, gradPdfU, gradPdfL, pdfU, pdfL, cdfU, cdfL;
            censorType = CensoringType.IntervalCensored;
            if (double.IsInfinity(yUpper))
            {
                pdfU = 0;
                cdfU = 1;
                gradPdfU = 0;
                censorType = CensoringType.RightCensored;
            }
            else
            {
                zU = (logYUpper - yPred) / sigma;
                pdfU = TDist.PDF(zU);
                cdfU = TDist.CDF(zU);
                gradPdfU = TDist.GradPDF(zU);
            }
            if (yLower <= 0.0)
            {
                pdfL = 0;
                cdfL = 0;
                gradPdfL = 0;
                censorType = CensoringType.LeftCensored;
            }
            else
            {
                zL = (logYLower - yPred) / sigma;
                pdfL = TDist.PDF(zL);
                cdfL = TDist.CDF(zL);
                gradPdfL = TDist.GradPDF(zL);
            }
            var cdfDiff = cdfU - cdfL;
            var pdfDiff = pdfU - pdfL;
            var gradDiff = gradPdfU - gradPdfL;
            var sqrtDenominator = sigma * cdfDiff;
            zSign = zU > 0 || zL > 0;
            numerator = -(cdfDiff * gradDiff - pdfDiff * pdfDiff);
            denominator = sqrtDenominator * sqrtDenominator;
        }
        var hessian = numerator / denominator;
        if (denominator < Aft.Eps && (double.IsNaN(hessian) || double.IsInfinity(hessian)))
            hessian = TDist.LimitHessAtInfPred(censorType, zSign, sigma);
        return Aft.Clip(hessian, Aft.MinHessian, Aft.MaxHessian);
    }
}

public sealed class AFTObj : ObjFunction
{
    private readonly AFTParam _param = new();

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));

    public override ObjInfo Task => new(ObjTask.Survival);

    private static void GradientImpl<TDist>(HostDeviceVector<float> preds, MetaInfo info, float scale, int nThreads,
        Tensor<GradientPair> outGpair) where TDist : IDistribution
    {
        var predt = preds.RawArray;
        var lower = info.LabelsLowerBound.RawArray;
        var upper = info.LabelsUpperBound.RawArray;
        var weights = info.Weights.RawArray;
        var isNullWeight = info.Weights.Empty;
        var gpair = outGpair.HostView();
        Threading.ParallelFor(preds.Size, nThreads, i =>
        {
            var grad = (float)AFTLoss<TDist>.Gradient(lower[i], upper[i], predt[i], scale);
            var hess = (float)AFTLoss<TDist>.Hessian(lower[i], upper[i], predt[i], scale);
            var weight = isNullWeight ? 1.0f : weights[i];
            gpair[i, 0] = new GradientPair(grad * weight, hess * weight);
        });
    }

    public override void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        var ndata = preds.Size;
        Check.Eq(info.LabelsLowerBound.Size, ndata);
        Check.Eq(info.LabelsUpperBound.Size, ndata);
        if (!info.Weights.Empty)
            Check.Eq(info.Weights.Size, ndata, "Number of weights should be equal to number of data points.");
        outGpair.Reshape(ndata, 1);
        var scale = _param.AftLossDistributionScale;
        switch (_param.AftLossDistribution)
        {
            case ProbabilityDistributionType.Normal:
                GradientImpl<NormalDistribution>(preds, info, scale, Ctx.Threads(), outGpair);
                break;
            case ProbabilityDistributionType.Logistic:
                GradientImpl<LogisticDistribution>(preds, info, scale, Ctx.Threads(), outGpair);
                break;
            case ProbabilityDistributionType.Extreme:
                GradientImpl<ExtremeDistribution>(preds, info, scale, Ctx.Threads(), outGpair);
                break;
            default:
                Check.Fail("Unrecognized distribution");
                break;
        }
    }

    public override void PredTransform(HostDeviceVector<float> predictions) =>
        ObjectiveUtils.ElementwiseTransform(Ctx, predictions, MathF.Exp);

    public override void EvalTransform(HostDeviceVector<float> ioPreds) { }

    public override void ProbToMargin(Tensor<float> baseScore)
    {
        for (var i = 0; i < baseScore.Size; i++) baseScore[i] = MathF.Log(baseScore[i]);
    }

    public override string DefaultEvalMetric => "aft-nloglik";

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = "survival:aft";
        output["aft_loss_param"] = _param.ToJson();
    }

    public override void LoadConfig(Json input) => _param.FromJson(input["aft_loss_param"]);

    public override Json DefaultMetricConfig()
    {
        var config = new JsonObject();
        config["name"] = DefaultEvalMetric;
        config["aft_loss_param"] = _param.ToJson();
        return config;
    }
}
