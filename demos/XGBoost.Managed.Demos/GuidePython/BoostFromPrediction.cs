using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>Demo for boosting from prediction. Port of <c>demo/guide-python/boost_from_prediction.py</c>.</summary>
internal static class BoostFromPrediction
{
    public static void Run(string[] args)
    {
        using var dtrain = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.train"));
        using var dtest = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.test"));
        (DMatrix, string)[] watchlist = [(dtest, "eval"), (dtrain, "train")];

        // advanced: start from a initial base prediction
        Console.WriteLine("start running example to start from a initial prediction");
        // specify parameters via map, definition are same as c++ version
        var param = P.Of(("max_depth", 2), ("eta", 1), ("objective", "binary:logistic"));
        // train xgboost for 1 round
        using var bst = XGB.Train(param, dtrain, 1, watchlist, onIteration: Print);
        // Note: we need the margin value instead of transformed prediction in BaseMargin.
        // PredictionType.Margin always gives you margin values before the logistic transformation.
        var ptrain = bst.Predict(dtrain, PredictionType.Margin).Values;
        var ptest = bst.Predict(dtest, PredictionType.Margin).Values;
        dtrain.BaseMargin = ptrain;
        dtest.BaseMargin = ptest;

        Console.WriteLine("this is result of running from initial prediction");
        using var bst2 = XGB.Train(param, dtrain, 1, watchlist, onIteration: Print);
    }

    private static void Print(int i, IReadOnlyDictionary<string, double> results) =>
        Console.WriteLine(Training.FormatLine(i, results));
}
