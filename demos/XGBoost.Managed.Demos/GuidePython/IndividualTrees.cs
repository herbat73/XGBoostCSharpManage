using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Prediction using individual trees and model slices. Port of
/// <c>demo/guide-python/individual_trees.py</c>.
/// </summary>
internal static class IndividualTrees
{
    private const int NRounds = 4;
    // Specify the base score, otherwise xgboost will estimate one from the training data.
    private const double BaseScore = 0.5;

    private static readonly List<KeyValuePair<string, string>> Params = P.Of(
        ("max_depth", 2),
        ("eta", 1),
        ("objective", "reg:logistic"),
        ("tree_method", "hist"),
        ("base_score", BaseScore));

    public static void Run(string[] args)
    {
        IndividualTree();
        ModelSlices();
        Console.WriteLine("Accumulated per-tree predictions match the full model.");
    }

    /// <summary>Get prediction from each individual tree and combine them together.</summary>
    private static void IndividualTree()
    {
        var (train, test) = LoadData();
        using var xyTrain = train.ToDMatrix(Math.Max(train.NumCols, test.NumCols));
        using var booster = XGB.Train(Params, xyTrain, NRounds);

        // Use logit to inverse the base score back to raw leaf value (margin)
        var scores = Enumerable.Repeat((float)Logit(BaseScore), test.Rows).ToArray();
        for (var i = 0; i < NRounds; i++)
        {
            // - Use PredictionType.Margin to get raw leaf values
            // - Use iterationBegin/iterationEnd to get prediction for only one tree
            // - Use previous prediction as base margin for the model
            using var xyTest = TestMatrix(test, booster, scores);

            // last round, get the transformed prediction; otherwise raw leaf value for accumulation
            var type = i == NRounds - 1 ? PredictionType.Value : PredictionType.Margin;
            scores = booster.Predict(xyTest, type, iterationBegin: i, iterationEnd: i + 1).Values;
        }

        using var full = TestMatrix(test, booster, null);
        Check.AllClose(scores, booster.Predict(full).Values, what: "per-tree predictions");
    }

    /// <summary>Inference with each individual tree using model slices.</summary>
    private static void ModelSlices()
    {
        var (train, test) = LoadData();
        using var xyTrain = train.ToDMatrix(Math.Max(train.NumCols, test.NumCols));
        using var booster = XGB.Train(Params, xyTrain, NRounds);
        var trees = Enumerable.Range(0, NRounds).Select(t => booster.Slice(t, t + 1)).ToList();

        // Use logit to inverse the base score back to raw leaf value (margin)
        var scores = Enumerable.Repeat((float)Logit(BaseScore), test.Rows).ToArray();
        for (var i = 0; i < trees.Count; i++)
        {
            // Feed previous scores into base margin.
            using var xyTest = TestMatrix(test, booster, scores);

            // last round, get the transformed prediction; otherwise raw leaf value for accumulation
            var type = i == NRounds - 1 ? PredictionType.Value : PredictionType.Margin;
            scores = trees[i].Predict(xyTest, type).Values;
        }

        using var full = TestMatrix(test, booster, null);
        Check.AllClose(scores, booster.Predict(full).Values, what: "sliced-model predictions");
        trees.ForEach(t => t.Dispose());
    }

    private static (SparseData Train, SparseData Test) LoadData() =>
        (Datasets.LoadLibSvm(DemoPaths.Data("agaricus.txt.train")), Datasets.LoadLibSvm(DemoPaths.Data("agaricus.txt.test")));

    private static DMatrix TestMatrix(SparseData test, Booster booster, float[]? baseMargin)
    {
        var d = DMatrix.FromCsr(test.Indptr, test.Indices, test.Values, (int)booster.NumFeatures);
        if (baseMargin is not null) d.BaseMargin = baseMargin;
        return d;
    }

    private static double Logit(double p) => Math.Log(p / (1 - p));
}
