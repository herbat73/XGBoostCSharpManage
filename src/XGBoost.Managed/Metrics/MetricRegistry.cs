// XGBOOST_REGISTER_METRIC entries of src/metric/*.
namespace XGBoost.Metrics;

internal static class MetricRegistry
{
    public static void Register()
    {
        var r = Registry.MetricFactories;
        r["rmse"] = _ => EvalEWiseMetric.Rmse();
        r["rmsle"] = _ => EvalEWiseMetric.Rmsle();
        r["mae"] = _ => EvalEWiseMetric.Mae();
        r["mape"] = _ => EvalEWiseMetric.Mape();
        r["logloss"] = _ => EvalEWiseMetric.LogLoss();
        r["error"] = EvalEWiseMetric.Error;
        r["poisson-nloglik"] = _ => EvalEWiseMetric.PoissonNLogLik();
        r["gamma-deviance"] = _ => EvalEWiseMetric.GammaDeviance();
        r["gamma-nloglik"] = _ => EvalEWiseMetric.GammaNLogLik();
        r["tweedie-nloglik"] = EvalEWiseMetric.TweedieNLogLik;
        r["mphe"] = _ => new PseudoErrorLoss();
        r["quantile"] = _ => new QuantileError();
        r["expectile"] = _ => new ExpectileError();
        r["normal-nloglik"] = _ => new NormalNLogLik();
        r["merror"] = _ => new EvalMClass(false);
        r["mlogloss"] = _ => new EvalMClass(true);
        r["aft-nloglik"] = _ => new AFTNLogLikDispatcher();
        r["interval-regression-accuracy"] = _ => new IntervalRegressionAccuracy();
        r["ams"] = p => new EvalAMS(p);
        r["cox-nloglik"] = _ => new EvalCox();
        r["pre"] = p => new EvalPrecision("pre", p);
        r["map"] = p => new EvalMAPScore("map", p);
        r["ndcg"] = p => new EvalNDCG("ndcg", p);
        r["auc"] = _ => new EvalROCAUC();
        r["aucpr"] = _ => new EvalPRAUC();
    }
}
