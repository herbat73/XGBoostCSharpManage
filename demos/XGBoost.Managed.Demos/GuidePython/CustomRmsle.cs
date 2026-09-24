using System.Diagnostics;
using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Defining a custom regression objective and metric. Port of <c>demo/guide-python/custom_rmsle.py</c>.
/// </summary>
/// <remarks>
/// Implements the Squared Log Error (SLE) objective and RMSLE metric as custom functions, then compares
/// them with the native implementation. SLE reduces the impact of outliers in the training data, so it
/// is also compared with standard squared error. Instead of plotting, the final metrics are printed.
/// </remarks>
internal static class CustomRmsle
{
    // shape of generated data.
    private const int Rows = 4096;
    private const int Cols = 16;

    private const int Outlier = 10000; // mean of generated outliers
    private const int NumberOfOutliers = 64;

    private const double Ratio = 0.7;
    private const int Seed = 1994;

    private const int BoostRound = 20;

    public static void Run(string[] args)
    {
        var (dtrain, dtest) = GenerateData();
        using (dtrain)
        using (dtest)
        {
            var rmseEvals = NativeRmse(dtrain, dtest);
            var rmsleEvals = NativeRmsle(dtrain, dtest);
            var customRmsleEvals = CustomRmsleTrain(dtrain, dtest);

            Console.WriteLine();
            Console.WriteLine($"test-RMSE (squared error):        {Fmt.Num(rmseEvals["dtest"]["rmse"][^1])}");
            Console.WriteLine($"test-RMSLE (native SLE):          {Fmt.Num(rmsleEvals["dtest"]["rmsle"][^1])}");
            Console.WriteLine($"test-CustomRMSLE (C# SLE):        {Fmt.Num(customRmsleEvals["dtest"]["CustomRMSLE"][^1])}");
        }
    }

    /// <summary>Generate data containing outliers.</summary>
    private static (DMatrix Train, DMatrix Test) GenerateData()
    {
        var rng = new Rng(Seed);
        var x = new DenseData(rng.NormalArray(Rows * Cols), Rows, Cols);
        var y = rng.NormalArray(Rows);
        var shift = Math.Abs(y.Min());
        for (var i = 0; i < y.Length; i++) y[i] += shift;

        // Create outliers
        for (var i = 0; i < NumberOfOutliers; i++)
        {
            var ind = rng.Integers(0, y.Length - 1);
            y[ind] += rng.Integers(0, Outlier);
        }

        var trainPortion = (int)(Rows * Ratio);

        // rmsle requires all label be greater than -1.
        Check.That(y.All(v => v > -1.0), "labels must be greater than -1");

        var train = Enumerable.Range(0, trainPortion).ToArray();
        var test = Enumerable.Range(trainPortion, Rows - trainPortion).ToArray();
        return (x.TakeRows(train).ToDMatrix(y.Gather(train)), x.TakeRows(test).ToDMatrix(y.Gather(test)));
    }

    /// <summary>Train using native implementation of Root Mean Squared Loss.</summary>
    private static EvalsLog NativeRmse(DMatrix dtrain, DMatrix dtest)
    {
        Console.WriteLine("Squared Error");
        var squaredError = P.Of(
            ("objective", "reg:squarederror"),
            ("eval_metric", "rmse"),
            ("tree_method", "hist"),
            ("seed", Seed));
        var start = Stopwatch.StartNew();
        var results = new EvalsLog();
        using var _ = Training.Train(squaredError, dtrain, BoostRound,
            evals: [(dtrain, "dtrain"), (dtest, "dtest")], evalsResult: results);
        Console.WriteLine($"Finished Squared Error in: {start.Elapsed.TotalSeconds} \n");
        return results;
    }

    /// <summary>Train using native implementation of Squared Log Error.</summary>
    private static EvalsLog NativeRmsle(DMatrix dtrain, DMatrix dtest)
    {
        Console.WriteLine("Squared Log Error");
        var results = new EvalsLog();
        var squaredLogError = P.Of(
            ("objective", "reg:squaredlogerror"),
            ("eval_metric", "rmsle"),
            ("tree_method", "hist"),
            ("seed", Seed));
        var start = Stopwatch.StartNew();
        using var _ = Training.Train(squaredLogError, dtrain, BoostRound,
            evals: [(dtrain, "dtrain"), (dtest, "dtest")], evalsResult: results);
        Console.WriteLine($"Finished Squared Log Error in: {start.Elapsed.TotalSeconds}");
        return results;
    }

    /// <summary>Train using a C# implementation of Squared Log Error.</summary>
    private static EvalsLog CustomRmsleTrain(DMatrix dtrain, DMatrix dtest)
    {
        // Squared Log Error objective, a simplified version of RMSLE used as objective function:
        // 1/2 [log(pred + 1) - log(label + 1)]^2
        static (float[], float[]) SquaredLog(Prediction predt, DMatrix data)
        {
            var y = data.Label;
            var grad = new float[y.Length];
            var hess = new float[y.Length];
            for (var i = 0; i < y.Length; i++)
            {
                double p = Math.Max(predt.Values[i], -1 + 1e-6f);
                // gradient and hessian of the squared log error
                grad[i] = (float)((Math.Log(1 + p) - Math.Log(1 + y[i])) / (p + 1));
                hess[i] = (float)((-Math.Log(1 + p) + Math.Log(1 + y[i]) + 1) / Math.Pow(p + 1, 2));
            }
            return (grad, hess);
        }

        // Root mean squared log error metric: sqrt(1/N [log(pred + 1) - log(label + 1)]^2)
        static (string, double) Rmsle(Prediction predt, DMatrix data)
        {
            var y = data.Label;
            double sum = 0;
            for (var i = 0; i < y.Length; i++)
            {
                double p = Math.Max(predt.Values[i], -1 + 1e-6f);
                sum += Math.Pow(Math.Log(1 + y[i]) - Math.Log(1 + p), 2);
            }
            return ("CustomRMSLE", Math.Sqrt(sum / y.Length));
        }

        var results = new EvalsLog();
        using var _ = Training.Train(
            P.Of(("tree_method", "hist"), ("seed", Seed), ("disable_default_eval_metric", 1)),
            dtrain, BoostRound, obj: SquaredLog, customMetric: Rmsle,
            evals: [(dtrain, "dtrain"), (dtest, "dtest")], evalsResult: results);
        return results;
    }
}
