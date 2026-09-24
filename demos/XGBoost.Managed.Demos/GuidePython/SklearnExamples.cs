using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Port of <c>demo/guide-python/sklearn_examples.py</c> using the native interface. scikit-learn's digits,
/// iris and California housing datasets are replaced by synthetic data of the same shape, and
/// <c>KFold</c>, <c>GridSearchCV</c> and the metrics by the helpers in <see cref="ModelSelection"/>.
/// </summary>
internal static class SklearnExamples
{
    public static void Run(string[] args)
    {
        var rng = new Rng(31337);

        Console.WriteLine("Zeros and Ones (digits-like data): binary classification");
        var (x, y) = Datasets.Blobs(360, 64, 2, new Rng(0), spread: 6);
        foreach (var (trainIndex, testIndex) in Datasets.KFold(x.Rows, 2, rng))
        {
            var predictions = FitPredictClasses(P.Of(("objective", "binary:logistic"), ("nthread", 1)),
                x, y, trainIndex, testIndex, classes: 2);
            Console.WriteLine(ModelSelection.ConfusionMatrix(y.Gather(testIndex), predictions, 2));
        }

        Console.WriteLine("Iris-like data: multiclass classification");
        (x, y) = Datasets.Blobs(150, 4, 3, new Rng(1), spread: 2);
        foreach (var (trainIndex, testIndex) in Datasets.KFold(x.Rows, 2, rng))
        {
            var predictions = FitPredictClasses(
                P.Of(("objective", "multi:softprob"), ("num_class", 3), ("nthread", 1)),
                x, y, trainIndex, testIndex, classes: 3);
            Console.WriteLine(ModelSelection.ConfusionMatrix(y.Gather(testIndex), predictions, 3));
        }

        Console.WriteLine("Synthetic housing-like data: regression");
        (x, y) = Datasets.MakeRegression(20640, 8, new Rng(1234), noise: 10);
        var regParams = P.Of(("objective", "reg:squarederror"), ("nthread", 1));
        foreach (var (trainIndex, testIndex) in Datasets.KFold(x.Rows, 2, rng))
        {
            var predictions = ModelSelection.FitPredict(regParams, 100,
                x.TakeRows(trainIndex), y.Gather(trainIndex), x.TakeRows(testIndex));
            Console.WriteLine(ModelSelection.MeanSquaredError(y.Gather(testIndex), predictions));
        }

        Console.WriteLine("Parameter optimization");
        var (bestScore, bestParams) = ModelSelection.GridSearch(x, y, regParams,
            [("max_depth", [2, 4]), ("n_estimators", [50, 100])], cv: 3);
        Console.WriteLine(bestScore);
        Console.WriteLine(bestParams);

        // A full snapshot of a model can be stored and restored, like pickling the sklearn estimator.
        Console.WriteLine("Serializing models");
        using (var dall = x.ToDMatrix(y))
        using (var best = XGB.Train([.. regParams, KeyValuePair.Create("max_depth", "4")], dall, 100))
        {
            var path = DemoPaths.Output("best_calif.snapshot");
            File.WriteAllBytes(path, best.Serialize());
            using var restored = Booster.Deserialize(File.ReadAllBytes(path));
            Console.WriteLine(Check.IsClose(best.InplacePredict(x.Values, x.Rows, x.Cols).Values,
                restored.InplacePredict(x.Values, x.Rows, x.Cols).Values));
        }

        // Early-stopping
        (x, y) = Datasets.Blobs(1797, 64, 10, new Rng(2), spread: 6);
        var (train, test) = Datasets.TrainTestSplit(x.Rows, 0.25, new Rng(0));
        using var dtrain = x.TakeRows(train).ToDMatrix(y.Gather(train));
        using var dtest = x.TakeRows(test).ToDMatrix(y.Gather(test));
        using var clf = Training.Train(
            P.Of(("objective", "multi:softprob"), ("num_class", 10), ("nthread", 1), ("eval_metric", "auc")),
            dtrain, 100, evals: [(dtest, "validation_0")], earlyStoppingRounds: 10);
        Console.WriteLine($"best_iteration: {clf.GetAttribute("best_iteration")}");
    }

    /// <summary>Trains a classifier and returns the predicted class of every test row.</summary>
    private static float[] FitPredictClasses(List<KeyValuePair<string, string>> parameters, DenseData x, float[] y,
        int[] trainIndex, int[] testIndex, int classes)
    {
        using var dtrain = x.TakeRows(trainIndex).ToDMatrix(y.Gather(trainIndex));
        using var booster = XGB.Train(parameters, dtrain, 100);
        var test = x.TakeRows(testIndex);
        var predt = booster.InplacePredict(test.Values, test.Rows, test.Cols).Values;
        if (classes == 2) return predt.Select(p => p > 0.5f ? 1f : 0f).ToArray();
        return Enumerable.Range(0, test.Rows)
            .Select(r => (float)Enumerable.Range(0, classes).MaxBy(c => predt[r * classes + c]))
            .ToArray();
    }
}
