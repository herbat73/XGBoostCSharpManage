using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Multi-output regression. Port of <c>demo/guide-python/multioutput_regression.py</c>; instead of plotting,
/// the final errors and a few predictions are printed.
/// </summary>
/// <remarks>
/// See doc/tutorials/multioutput.rst. The feature is experimental; for the <c>multi_output_tree</c>
/// strategy, many features are missing.
/// </remarks>
internal static class MultioutputRegression
{
    private const int Targets = 2;

    public static void Run(string[] args)
    {
        // Train with builtin RMSE objective
        // - One model per output.
        RmseModel("one_output_per_tree");
        // - One model for all outputs, this is still experimental.
        RmseModel("multi_output_tree");

        // Train with custom objective.
        // - One model per output.
        CustomRmseModel("one_output_per_tree");
        // - One model for all outputs, this is still experimental.
        CustomRmseModel("multi_output_tree");
    }

    /// <summary>Generate a sample dataset that y is a 2 dim circle.</summary>
    private static (DenseData X, float[] Y) GenCircle()
    {
        var rng = new Rng(1994);
        var xs = Enumerable.Range(0, 100).Select(_ => 200 * rng.Uniform() - 100).Order().ToArray();
        var y = new double[xs.Length * Targets];
        for (var r = 0; r < xs.Length; r++)
        {
            y[r * Targets] = Math.PI * Math.Sin(xs[r]);
            y[r * Targets + 1] = Math.PI * Math.Cos(xs[r]);
            if (r % 5 == 0)
                for (var t = 0; t < Targets; t++) y[r * Targets + t] += 0.5 - rng.Uniform();
        }
        var min = y.Min();
        for (var i = 0; i < y.Length; i++) y[i] -= min;
        var max = y.Max();
        var x = new DenseData(xs.Select(v => (float)v).ToArray(), xs.Length, 1);
        return (x, y.Select(v => (float)(v / max)).ToArray());
    }

    private static void Report(string name, DenseData x, float[] y, Booster booster)
    {
        var predt = booster.InplacePredict(x.Values, x.Rows, x.Cols);
        Check.That(predt.Shape.SequenceEqual([x.Rows, Targets]), "prediction should have one column per target");
        Console.WriteLine($"{name}: first rows (target -> prediction)");
        for (var r = 0; r < 3; r++)
        {
            Console.WriteLine($"  ({Fmt.Num(y[r * Targets])}, {Fmt.Num(y[r * Targets + 1])}) -> " +
                $"({Fmt.Num(predt.Values[r * Targets])}, {Fmt.Num(predt.Values[r * Targets + 1])})");
        }
    }

    /// <summary>Draw a circle with 2-dim coordinate as target variables.</summary>
    private static void RmseModel(string strategy)
    {
        var (x, y) = GenCircle();
        using var xy = x.ToDMatrix();
        // A 2-D label: one column per target.
        xy.SetLabel(y, Targets);
        // Train a regressor on it
        var results = new EvalsLog();
        using var reg = Training.Train(
            P.Of(("tree_method", "hist"),
                ("nthread", 2), // This small dataset does not benefit from using many threads.
                ("max_depth", 8),
                ("multi_strategy", strategy),
                ("subsample", 0.6)),
            xy, 128, evals: [(xy, "validation_0")], evalsResult: results, verbose: false);

        Console.WriteLine($"RMSE-{strategy}: rmse {Fmt.Num(results["validation_0"]["rmse"][^1])}");
        Report($"RMSE-{strategy}", x, y, reg);
    }

    /// <summary>Train using a C# implementation of Squared Error.</summary>
    private static void CustomRmseModel(string strategy)
    {
        // Gradient and Hessian of squared error.
        static (float[], float[]) SquaredError(Prediction predt, DMatrix dtrain)
        {
            var y = dtrain.Label; // row-major (rows, targets), same layout as predt
            Check.That(dtrain.NumRows == predt.Rows, "prediction and data row counts differ");
            var grad = predt.Values.Select((p, i) => p - y[i]).ToArray();
            var hess = Enumerable.Repeat(1f, grad.Length).ToArray();
            return (grad, hess);
        }

        static (string, double) Rmse(Prediction predt, DMatrix dtrain)
        {
            var y = dtrain.Label;
            var v = Math.Sqrt(predt.Values.Select((p, i) => Math.Pow(y[i] - p, 2)).Average());
            return ("CustomRMSE", v);
        }

        var (x, y) = GenCircle();
        using var xy = x.ToDMatrix(nthread: 2);
        xy.SetLabel(y, Targets);
        var results = new EvalsLog();
        // Make sure num_target is passed to XGBoost when a custom objective is used. With a builtin
        // objective XGBoost can figure out the number of targets automatically.
        using var booster = Training.Train(
            P.Of(("tree_method", "hist"), ("num_target", Targets), ("nthread", 2), ("multi_strategy", strategy)),
            xy, 128, obj: SquaredError, customMetric: Rmse, evals: [(xy, "Train")], evalsResult: results,
            verbose: false);

        Console.WriteLine($"CustomRMSE-{strategy}: rmse {Fmt.Num(results["Train"]["rmse"][^1])}, " +
            $"CustomRMSE {Fmt.Num(results["Train"]["CustomRMSE"][^1])}");
        Report($"CustomRMSE-{strategy}", x, y, booster);

        Check.AllClose(results["Train"]["rmse"], results["Train"]["CustomRMSE"], rtol: 1e-2, what: "rmse");
    }
}
