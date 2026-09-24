using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Prediction intervals with quantile and expectile regression. Port of
/// <c>demo/guide-python/prediction_intervals.py</c>; instead of plotting, the bands are printed on a coarse
/// grid together with their coverage of the test observations.
/// </summary>
/// <remarks>
/// Options: <c>--multi_strategy one_output_per_tree|multi_output_tree</c>, <c>--device cpu|cuda</c>.
/// Quantile crossing can happen due to limitation in the algorithm. Expectiles are asymmetric means, not
/// percentiles, but they can be used to construct tail-sensitive bands around the conditional mean.
/// </remarks>
internal static class PredictionIntervals
{
    private static readonly double[] Alpha = [0.05, 0.5, 0.95];

    /// <summary>The function to predict.</summary>
    private static double F(double x) => x * Math.Sin(x);

    public static void Run(string[] args)
    {
        var cli = new DemoArgs(args);
        var rng = new Rng(1994);

        var (train, test, testData) = MakeMatrices(rng);
        using (train)
        using (test)
        {
            var parameters = BaseParams(cli.Get("--multi_strategy", "one_output_per_tree"), cli.Get("--device", "cpu"));

            var quantileEvals = new EvalsLog();
            using var quantile = TrainIntervalModel(
                [.. parameters, KeyValuePair.Create("objective", "reg:quantileerror")], "quantile_alpha",
                train, test, quantileEvals);
            var expectileEvals = new EvalsLog();
            using var expectile = TrainIntervalModel(
                [.. parameters, KeyValuePair.Create("objective", "reg:expectileerror")], "expectile_alpha",
                train, test, expectileEvals);
            using var mean = SquaredErrorModel(parameters, train, test);

            // Predict on a dense grid.
            var grid = Enumerable.Range(0, 1000).Select(i => (float)(10.0 * i / 999)).ToArray();
            var predQuantile = quantile.InplacePredict(grid, grid.Length, 1);
            var predExpectile = expectile.InplacePredict(grid, grid.Length, 1);
            var predMean = mean.InplacePredict(grid, grid.Length, 1);

            Check.That(predQuantile.Shape.SequenceEqual([grid.Length, Alpha.Length]), "quantile prediction shape");
            Check.That(predExpectile.Shape.SequenceEqual([grid.Length, Alpha.Length]), "expectile prediction shape");

            Console.WriteLine($"Quantile test metric: {quantileEvals["Test"]["quantile"][^1]}");
            Console.WriteLine($"Expectile test metric: {expectileEvals["Test"]["expectile"][^1]}");

            PrintBands(grid, predQuantile.Values, predExpectile.Values, predMean.Values);
            PrintCoverage(quantile, expectile, testData);
        }
    }

    /// <summary>Generate heteroscedastic data with asymmetric noise.</summary>
    private static (DenseData Features, float[] Target) MakeDataset(Rng rng)
    {
        var features = DenseData.Zeros(1000, 1);
        var target = new float[1000];
        for (var i = 0; i < 1000; i++) features[i, 0] = (float)rng.Uniform(0, 10.0);
        for (var i = 0; i < 1000; i++)
        {
            var x = features[i, 0];
            var sigma = 0.5 + x / 10.0;
            var noise = rng.LogNormal(0, sigma) - Math.Exp(sigma * sigma / 2.0);
            target[i] = (float)(F(x) + noise);
        }
        return (features, target);
    }

    /// <summary>Create train/test DMatrices and return held-out data for evaluation.</summary>
    private static (DMatrix Train, DMatrix Test, (DenseData X, float[] Y) TestData) MakeMatrices(Rng rng)
    {
        var (features, target) = MakeDataset(rng);
        var (trainIdx, testIdx) = Datasets.TrainTestSplit(features.Rows, 0.25, rng);
        var testX = features.TakeRows(testIdx);
        var testY = target.Gather(testIdx);
        return (features.TakeRows(trainIdx).ToDMatrix(target.Gather(trainIdx)), testX.ToDMatrix(testY), (testX, testY));
    }

    /// <summary>Parameters shared by the three models in the demo.</summary>
    private static List<KeyValuePair<string, string>> BaseParams(string multiStrategy, string device) => P.Of(
        ("tree_method", "hist"),
        ("learning_rate", 0.04),
        ("max_depth", 5),
        ("multi_strategy", multiStrategy),
        ("device", device));

    /// <summary>Train a multi-output interval model.</summary>
    private static Booster TrainIntervalModel(List<KeyValuePair<string, string>> parameters, string alphaName,
        DMatrix train, DMatrix test, EvalsLog evalsResult)
    {
        parameters.Add(KeyValuePair.Create(alphaName, P.ToParam(Alpha)));
        return Training.Train(parameters, train, 64, earlyStoppingRounds: 4,
            evals: [(train, "Train"), (test, "Test")], evalsResult: evalsResult);
    }

    /// <summary>Train a squared-error model for comparison.</summary>
    private static Booster SquaredErrorModel(List<KeyValuePair<string, string>> parameters, DMatrix train, DMatrix test) =>
        Training.Train([.. parameters, KeyValuePair.Create("objective", "reg:squarederror")], train, 64,
            earlyStoppingRounds: 4, evals: [(train, "Train"), (test, "Test")]);

    private static void PrintBands(float[] grid, float[] quantile, float[] expectile, float[] mean)
    {
        Console.WriteLine();
        Console.WriteLine("    x     f(x) |  q0.05   median   q0.95 |  e0.05   e0.50   e0.95 |  mean");
        for (var i = 0; i < grid.Length; i += 111)
        {
            Console.WriteLine($"{grid[i],5:F2} {F(grid[i]),8:F3} | " +
                $"{quantile[i * 3],6:F2} {quantile[i * 3 + 1],8:F2} {quantile[i * 3 + 2],7:F2} | " +
                $"{expectile[i * 3],6:F2} {expectile[i * 3 + 1],7:F2} {expectile[i * 3 + 2],7:F2} | {mean[i],6:F2}");
        }
    }

    private static void PrintCoverage(Booster quantile, Booster expectile, (DenseData X, float[] Y) test)
    {
        double Coverage(Booster model)
        {
            var predt = model.InplacePredict(test.X.Values, test.X.Rows, test.X.Cols).Values;
            return (double)test.Y.Where((y, i) => predt[i * 3] <= y && y <= predt[i * 3 + 2]).Count() / test.Y.Length;
        }
        Console.WriteLine();
        Console.WriteLine($"Test observations inside the 90% quantile interval: {Coverage(quantile):P1}");
        Console.WriteLine($"Test observations inside the expectile band:        {Coverage(expectile):P1}");
    }
}
