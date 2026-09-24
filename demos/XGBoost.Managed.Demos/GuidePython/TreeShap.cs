using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// SHAP values for feature importance. Port of <c>demo/guide-python/gpu_tree_shap.py</c>.
/// </summary>
/// <remarks>
/// Pass <c>--device cuda</c> to train and explain on the GPU, as the Python demo does; the default is the
/// CPU, where only the first rows are explained because exact TreeSHAP on 500 deep trees is slow. The
/// <c>shap</c> package's plots become a printed explanation of the first prediction and a ranking of mean
/// absolute SHAP values. The data is the synthetic regression the Python demo falls back to.
/// </remarks>
internal static class TreeShap
{
    public static void Run(string[] args)
    {
        var device = new DemoArgs(args).Get("--device", "cpu");
        var (x, y) = Datasets.MakeRegression(20640, 8, new Rng(1234));
        var featureNames = Enumerable.Range(0, 8).Select(i => $"f{i}").ToArray();

        const int numRound = 500;
        var param = P.Of(("eta", 0.05), ("max_depth", 10), ("tree_method", "hist"), ("device", device));

        using var dtrain = x.ToDMatrix(y);
        dtrain.FeatureNames = featureNames;
        using var model = XGB.Train(param, dtrain, numRound);

        // Compute shap values on the same device as the model
        model.SetParam("device", device);
        var rows = device == "cpu" ? 256 : x.Rows;
        using var explained = dtrain.Slice(Enumerable.Range(0, rows).ToArray());
        var shapValues = model.Predict(explained, PredictionType.Contribution);
        // Compute shap interaction values
        var shapInteractionValues = model.Predict(explained, PredictionType.Interaction);
        Console.WriteLine($"Explained {rows} rows: contributions {Fmt.Shape(shapValues.Shape)}, " +
            $"interactions {Fmt.Shape(shapInteractionValues.Shape)}");

        // The last column of the contributions is the bias: the expected value of the model output.
        var cols = featureNames.Length + 1;
        var expectedValue = shapValues.Values[cols - 1];
        var margin = model.Predict(explained, PredictionType.Margin).Values;

        // explain the first prediction (the Python demo draws a force plot)
        Console.WriteLine();
        Console.WriteLine($"Expected value {Fmt.Num(expectedValue)}, first prediction {Fmt.Num(margin[0])}:");
        foreach (var f in Enumerable.Range(0, featureNames.Length).OrderByDescending(f => Math.Abs(shapValues.Values[f])))
            Console.WriteLine($"  {featureNames[f]} = {Fmt.Num(x[0, f]),-10} contributes {Fmt.Num(shapValues.Values[f])}");

        // Show a summary of feature importance
        Console.WriteLine();
        Console.WriteLine("Mean |SHAP value| per feature:");
        var importance = Enumerable.Range(0, featureNames.Length)
            .Select(f => (Feature: featureNames[f],
                Value: Enumerable.Range(0, rows).Average(r => Math.Abs(shapValues.Values[r * cols + f]))))
            .OrderByDescending(p => p.Value)
            .ToList();
        var max = importance[0].Value;
        foreach (var (feature, value) in importance)
            Console.WriteLine($"  {feature} {Fmt.Num(value),-10} {new string('#', (int)Math.Round(40 * value / max))}");

        // Contributions of each row add up to the margin.
        for (var r = 0; r < rows; r++)
        {
            var sum = shapValues.Values.AsSpan(r * cols, cols).ToArray().Sum();
            Check.That(Math.Abs(sum - margin[r]) <= 1e-3 * Math.Max(1, Math.Abs(margin[r])), "SHAP values should sum to the margin");
        }
    }
}
