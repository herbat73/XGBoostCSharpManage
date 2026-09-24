using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Using feature weights to change column sampling. Port of <c>demo/guide-python/feature_weights.py</c>;
/// the importance plot becomes a console bar chart.
/// </summary>
internal static class FeatureWeights
{
    public static void Run(string[] args)
    {
        var rng = new Rng(1994);

        const int rows = 4196;
        const int cols = 10;

        var x = new DenseData(rng.NormalArray(rows * cols), rows, cols);
        var y = rng.NormalArray(rows);
        var fw = Enumerable.Range(0, cols).Select(i => (float)i).ToArray();

        using var dtrain = x.ToDMatrix(y);
        dtrain.SetInfo("feature_weights", fw);

        // Perform column sampling for each node split evaluation, the sampling process is weighted by
        // feature weights.
        using var bst = Training.Train(P.Of(("tree_method", "hist"), ("colsample_bynode", 0.2)),
            dtrain, 10, evals: [(dtrain, "d")]);
        var score = bst.GetScore("weight");
        var featureMap = score.Features.Zip(score.Scores).ToDictionary(p => p.First, p => p.Second);

        // feature zero has 0 weight
        Check.That(!featureMap.ContainsKey("f0"), "f0 has zero weight and should never be used");
        Check.That(featureMap.Values.Max() == featureMap["f9"], "f9 has the largest weight and should be used most");

        Console.WriteLine("Feature importance (weight):");
        var max = featureMap.Values.Max();
        foreach (var (feature, value) in featureMap.OrderByDescending(p => p.Value))
            Console.WriteLine($"  {feature,-4} {value,5} {new string('#', (int)Math.Round(50 * value / max))}");
    }
}
