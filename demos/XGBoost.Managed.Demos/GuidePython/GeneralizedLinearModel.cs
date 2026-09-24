using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>Fitting a generalized linear model. Port of <c>demo/guide-python/generalized_linear_model.py</c>.</summary>
internal static class GeneralizedLinearModel
{
    public static void Run(string[] args)
    {
        // this demo fits a generalized linear model: linear boosters instead of trees
        using var dtrain = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.train"));
        using var dtest = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.test"));

        // change booster to gblinear, so that we are fitting a linear model
        // alpha is the L1 regularizer
        // lambda is the L2 regularizer
        // you can also set lambda_bias which is L2 regularizer on the bias term
        var param = P.Of(
            ("objective", "binary:logistic"),
            ("booster", "gblinear"),
            ("alpha", 0.0001),
            ("lambda", 1));

        // normally, you do not need to set eta (step_size)
        // XGBoost uses a parallel coordinate descent algorithm (shotgun),
        // there could be affection on convergence with parallelization on certain cases
        // setting eta to be smaller value, e.g 0.5 can make the optimization more stable
        // param.Add(KeyValuePair.Create("eta", "1"));

        // the rest of settings are the same
        (DMatrix, string)[] watchlist = [(dtest, "eval"), (dtrain, "train")];
        const int numRound = 4;
        using var bst = XGB.Train(param, dtrain, numRound, watchlist,
            onIteration: (i, results) => Console.WriteLine(Training.FormatLine(i, results)));
        var preds = bst.Predict(dtest).Values;
        var labels = dtest.Label;
        var errors = preds.Where((p, i) => (p > 0.5 ? 1 : 0) != labels[i]).Count();
        Console.WriteLine($"error={(double)errors / preds.Length:F6}");
    }
}
