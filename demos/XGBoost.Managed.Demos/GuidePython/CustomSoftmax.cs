using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Creating a customized multi-class objective function. Port of
/// <c>demo/guide-python/custom_softmax.py</c>. Instead of plotting, the error histories are printed.
/// </summary>
internal static class CustomSoftmax
{
    private const int Rows = 100;
    private const int Cols = 10;
    private const int Classes = 4; // number of classes

    private const int Rounds = 10; // number of boosting rounds.

    public static void Run(string[] args)
    {
        // Generate some random data for demo.
        var rng = new Rng(1994);
        var x = new DenseData(rng.NormalArray(Rows * Cols), Rows, Cols);
        var y = Enumerable.Range(0, Rows).Select(_ => (float)rng.Integers(0, Classes)).ToArray();
        using var m = x.ToDMatrix(y);

        // Since 3.1, XGBoost can estimate the base_score automatically for built-in multi-class objectives.
        // We specify it explicitly to disable the estimation, for a proper comparison between the custom
        // implementation and the built-in one.
        var intercept = Enumerable.Repeat(1.0 / Classes, Classes).ToArray();

        var customResults = new EvalsLog();
        // Use our custom objective function
        using var boosterCustom = Training.Train(
            P.Of(("num_class", Classes), ("base_score", intercept), ("disable_default_eval_metric", true)),
            m, Rounds, obj: SoftprobObj, customMetric: MError, evalsResult: customResults, evals: [(m, "train")]);

        var predtCustom = Predict(boosterCustom, m);

        var nativeResults = new EvalsLog();
        // Use the same objective function defined in XGBoost.
        using var boosterNative = Training.Train(
            P.Of(("num_class", Classes), ("base_score", intercept), ("objective", "multi:softmax"),
                ("eval_metric", "merror")),
            m, Rounds, evalsResult: nativeResults, evals: [(m, "train")]);
        var predtNative = boosterNative.Predict(m).Values;

        // We are reimplementing the loss function in XGBoost, so it should be the same for normal cases.
        Check.That(predtCustom.SequenceEqual(predtNative), "custom and native predictions differ");
        Check.AllClose(customResults["train"]["CustomMError"], nativeResults["train"]["merror"], what: "merror");

        Console.WriteLine();
        Console.WriteLine($"Custom objective: {Fmt.List(customResults["train"]["CustomMError"])}");
        Console.WriteLine($"multi:softmax:    {Fmt.List(nativeResults["train"]["merror"])}");
    }

    /// <summary>Softmax function with x as input vector.</summary>
    private static double[] Softmax(ReadOnlySpan<float> x)
    {
        var e = new double[x.Length];
        for (var i = 0; i < x.Length; i++) e[i] = Math.Exp(x[i]);
        var sum = e.Sum();
        for (var i = 0; i < e.Length; i++) e[i] /= sum;
        return e;
    }

    /// <summary>
    /// Loss function. Computes the gradient and a diagonal pseudo-Hessian (the absolute residual |p - y|,
    /// neither the true Hessian nor an upper bound on it, which keeps every leaf value bounded).
    /// Reimplements the <c>multi:softprob</c> inside XGBoost.
    /// </summary>
    private static (float[], float[]) SoftprobObj(Prediction predt, DMatrix data)
    {
        var labels = data.Label;
        // Use 1 as weight if we don't have custom weight.
        var weights = data.Weight;
        if (weights.Length == 0) weights = Enumerable.Repeat(1f, Rows).ToArray();

        // The prediction is of shape (rows, classes), each element in a row represents a raw prediction
        // (leaf weight, hasn't gone through softmax yet).
        Check.That(predt.Shape.SequenceEqual([Rows, Classes]), "unexpected prediction shape");

        var grad = new float[Rows * Classes];
        var hess = new float[Rows * Classes];
        const double eps = 1e-6;

        // compute the gradient and pseudo-Hessian. The one in native XGBoost core is more robust to
        // numeric overflow as we don't do anything to mitigate the exp in softmax here.
        for (var r = 0; r < Rows; r++)
        {
            var target = (int)labels[r];
            double weight = weights[r];
            var p = Softmax(predt.Values.AsSpan(r * Classes, Classes));
            for (var c = 0; c < Classes; c++)
            {
                Check.That(target is >= 0 and < Classes, "label out of range");
                var g = c == target ? p[c] - 1.0 : p[c];
                // Absolute-residual pseudo-Hessian, matching the native objective.
                var h = Math.Max(Math.Abs(g) * weight, eps);
                grad[r * Classes + c] = (float)(g * weight);
                hess[r * Classes + c] = (float)h;
            }
        }
        return (grad, hess);
    }

    /// <summary>A customized prediction function that converts raw prediction to target class.</summary>
    private static float[] Predict(Booster booster, DMatrix x)
    {
        // Margin means we want the raw prediction obtained from tree leaf weights.
        var predt = booster.Predict(x, PredictionType.Margin);
        // The class with maximum "probability" (not strictly probability as it hasn't gone through softmax
        // yet, but the argmax is the same).
        return ArgMax(predt);
    }

    private static (string, double) MError(Prediction predt, DMatrix data)
    {
        var y = data.Label;
        // Like the custom objective, predt is the untransformed leaf weight when a custom objective is used.
        Check.That(predt.Shape.SequenceEqual([Rows, Classes]), "unexpected prediction shape");
        var output = ArgMax(predt);
        var errors = y.Where((label, i) => label != output[i]).Count();
        return ("CustomMError", (double)errors / Rows);
    }

    private static float[] ArgMax(Prediction predt)
    {
        var rows = (int)predt.Rows;
        var cols = predt.Values.Length / rows;
        var result = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            var best = 0;
            for (var c = 1; c < cols; c++)
                if (predt.Values[r * cols + c] > predt.Values[r * cols + best]) best = c;
            result[r] = best;
        }
        return result;
    }
}
