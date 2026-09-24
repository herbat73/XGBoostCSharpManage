// XGBOOST_REGISTER_OBJECTIVE entries of src/objective/*.
namespace XGBoost.Objectives;

internal static class ObjectiveRegistry
{
    public static void Register()
    {
        var r = Registry.ObjectiveFactories;
        r["reg:squarederror"] = () => new SquaredErrorRegression();
        r["reg:linear"] = () =>
        {
            Log.Warning("reg:linear is now deprecated in favor of reg:squarederror.");
            return new SquaredErrorRegression();
        };
        r["reg:logistic"] = () => new LogisticObjective(LogisticKind.Regression);
        r["binary:logistic"] = () => new LogisticObjective(LogisticKind.Classification);
        r["binary:logitraw"] = () =>
        {
            Log.Warning("`binary:logitraw` is now deprecated in favor of `binary:logistic` with `output_margin=true`.");
            return new LogisticObjective(LogisticKind.Raw);
        };
        r["reg:pseudohubererror"] = () => new PseudoHuberRegression();
        r["reg:squaredlogerror"] = () => new SquaredLogErrorRegression();
        r["count:poisson"] = () => new PoissonRegression();
        r["reg:gamma"] = () => new GammaRegression();
        r["reg:tweedie"] = () => new TweedieRegression();
        r["reg:absoluteerror"] = () => new MeanAbsoluteError();
        r["binary:hinge"] = () => new HingeObj();
        r["reg:normal"] = () => new NormalRegression();
        r["reg:quantileerror"] = () => new QuantileRegression();
        r["reg:expectileerror"] = () => new ExpectileRegression();
        r["survival:aft"] = () => new AFTObj();
        r["survival:cox"] = () => new CoxRegression();
        r["multi:softmax"] = () => new SoftmaxMultiClassObj(false);
        r["multi:softprob"] = () => new SoftmaxMultiClassObj(true);
        r["rank:ndcg"] = () => new LambdaRankObj(LambdaRankKind.NDCG);
        r["rank:pairwise"] = () => new LambdaRankObj(LambdaRankKind.Pairwise);
        r["rank:map"] = () => new LambdaRankObj(LambdaRankKind.MAP);
    }
}
