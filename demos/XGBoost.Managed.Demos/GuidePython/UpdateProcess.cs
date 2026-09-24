using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Using <c>process_type</c> with <c>prune</c> and <c>refresh</c>. Port of
/// <c>demo/guide-python/update_process.py</c>, using the synthetic regression data the Python demo falls
/// back to when it cannot download the California housing dataset.
/// </summary>
/// <remarks>
/// Modifying existing trees is not a well established use for XGBoost, so feel free to experiment.
/// </remarks>
internal static class UpdateProcess
{
    public static void Run(string[] args)
    {
        const int nRounds = 32;

        var (x, y) = Datasets.MakeRegression(20640, 8, new Rng(1234));
        var half = x.Rows / 2;
        var firstHalf = Enumerable.Range(0, half).ToArray();
        var secondHalf = Enumerable.Range(half, x.Rows - half).ToArray();

        // Train a model first
        var xTrain = x.TakeRows(firstHalf);
        var yTrain = y.Gather(firstHalf);
        using var xy = xTrain.ToDMatrix(yTrain);
        var evalsResult = new EvalsLog();
        using var booster = Training.Train(P.Of(("tree_method", "hist"), ("max_depth", 6)),
            xy, nRounds, evals: [(xy, "Train")], evalsResult: evalsResult, verbose: false);
        var shap = booster.Predict(xy, PredictionType.Contribution).Values;
        Console.WriteLine($"Original model, Train rmse: {Fmt.Num(evalsResult["Train"]["rmse"][^1])}");

        // Refresh the leaf value and tree statistic
        using var xyRefresh = x.TakeRows(secondHalf).ToDMatrix(y.Gather(secondHalf));
        // The model will adapt to other half of the data by changing leaf value (no change in split
        // condition) with refresh_leaf set to True.
        var refreshResult = new EvalsLog();
        using (Training.Train(P.Of(("process_type", "update"), ("updater", "refresh"), ("refresh_leaf", true)),
                   xyRefresh, nRounds, xgbModel: booster, evals: [(xy, "Original"), (xyRefresh, "Train")],
                   evalsResult: refreshResult, verbose: false))
        {
            Console.WriteLine($"Refreshed leaves, Original rmse: {Fmt.Num(refreshResult["Original"]["rmse"][^1])}, " +
                $"Train rmse: {Fmt.Num(refreshResult["Train"]["rmse"][^1])}");
        }

        // Refresh the model without changing the leaf value, but tree statistic including cover and weight
        // are refreshed.
        refreshResult = new EvalsLog();
        using var refreshed = Training.Train(
            P.Of(("process_type", "update"), ("updater", "refresh"), ("refresh_leaf", false)),
            xyRefresh, nRounds, xgbModel: booster, evals: [(xy, "Original"), (xyRefresh, "Train")],
            evalsResult: refreshResult, verbose: false);
        // Without refreshing the leaf value, resulting trees should be the same with original model except
        // for accumulated statistic. The rtol is for floating point error in prediction.
        Check.AllClose(refreshResult["Original"]["rmse"], evalsResult["Train"]["rmse"], rtol: 1e-5,
            what: "rmse of the refreshed model");
        // But SHAP value is changed as cover in tree nodes are changed.
        var refreshedShap = refreshed.Predict(xy, PredictionType.Contribution).Values;
        Check.That(!Check.IsClose(shap, refreshedShap, rtol: 1e-3), "SHAP values should change after refresh");
        Console.WriteLine("Refreshed statistics only: same predictions, different SHAP values.");

        // Prune the trees with smaller max_depth
        using var xyUpdate = xTrain.ToDMatrix(yTrain);

        var pruneResult = new EvalsLog();
        using var pruned = Training.Train(
            P.Of(("process_type", "update"), ("updater", "prune"), ("max_depth", 2)),
            xyUpdate, nRounds, xgbModel: booster, evals: [(xy, "Original"), (xyUpdate, "Train")],
            evalsResult: pruneResult, verbose: false);
        // Have a smaller model, but similar accuracy.
        Check.AllClose(pruneResult["Original"]["rmse"], pruneResult["Train"]["rmse"], atol: 1e-5,
            what: "rmse of the pruned model");
        Console.WriteLine($"Pruned to max_depth=2, Train rmse: {Fmt.Num(pruneResult["Train"]["rmse"][^1])}");
    }
}
