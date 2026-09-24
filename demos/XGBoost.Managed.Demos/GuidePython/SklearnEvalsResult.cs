using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Accessing the eval metrics. Port of <c>demo/guide-python/sklearn_evals_result.py</c> using the native
/// interface, with the evaluation sets named like the scikit-learn wrapper names them.
/// </summary>
internal static class SklearnEvalsResult
{
    public static void Run(string[] args)
    {
        // Labels are already mapped from {-1, 1} to {0, 1}.
        var (x, y) = Datasets.MakeHastie(2000, new Rng(42));

        var train = Enumerable.Range(0, 1600).ToArray();
        var test = Enumerable.Range(1600, 400).ToArray();
        using var dtrain = x.TakeRows(train).ToDMatrix(y.Gather(train));
        using var dtest = x.TakeRows(test).ToDMatrix(y.Gather(test));

        var evalsResult = new EvalsLog();
        using var clf = Training.Train(
            P.Of(("objective", "binary:logistic"), ("eval_metric", "logloss")), dtrain, numBoostRound: 2,
            evals: [(dtrain, "validation_0"), (dtest, "validation_1")], evalsResult: evalsResult);

        EvalsResult.Print(evalsResult, "validation_0");
    }
}
