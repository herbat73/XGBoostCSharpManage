using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Getting started with categorical data. Port of <c>demo/guide-python/categorical.py</c>. Instead of a
/// pandas DataFrame with category columns, the features are integer codes and the DMatrix is told which
/// columns are categorical through <see cref="DMatrix.FeatureTypes"/>.
/// </summary>
internal static class Categorical
{
    public static void Run(string[] args)
    {
        // Use builtin categorical data support
        var (x, y) = MakeCategorical(100, 10, 4, onehot: false);
        using var xy = x.ToDMatrix(y);
        // "c" marks a feature as categorical.
        xy.FeatureTypes = Enumerable.Repeat("c", x.Cols).ToArray();
        // Use onehot-encoding-based split here for demonstration. For details see the document of
        // max_cat_to_onehot.
        var regResults = new EvalsLog();
        using var reg = Training.Train(P.Of(("tree_method", "hist"), ("max_cat_to_onehot", 5)),
            xy, 100, evals: [(xy, "validation_0")], evalsResult: regResults, verbose: false);

        // Pass in already encoded data
        var (xEnc, yEnc) = MakeCategorical(100, 10, 4, onehot: true);
        using var xyEnc = xEnc.ToDMatrix(yEnc);
        var regEncResults = new EvalsLog();
        using var regEnc = Training.Train(P.Of(("tree_method", "hist")),
            xyEnc, 100, evals: [(xyEnc, "validation_0")], evalsResult: regEncResults, verbose: false);

        // Check that they have same results
        var rmse = regResults["validation_0"]["rmse"];
        var rmseEnc = regEncResults["validation_0"]["rmse"];
        Check.AllClose(rmse, rmseEnc, what: "rmse history");
        Console.WriteLine($"rmse with categorical splits: {Fmt.Num(rmse[^1])}, with one-hot encoded data: {Fmt.Num(rmseEnc[^1])}");

        // SHAP values of the categorical model
        var shap = reg.Predict(xy, PredictionType.Contribution);
        var margin = reg.Predict(xy, PredictionType.Margin).Values;
        var cols = (int)shap.Shape[^1];
        var sums = Enumerable.Range(0, margin.Length)
            .Select(r => shap.Values.AsSpan(r * cols, cols).ToArray().Sum()).ToArray();
        Check.AllClose(sums, margin, rtol: 1e-3, what: "SHAP sums");
        Console.WriteLine("SHAP values sum to the margin.");
    }

    /// <summary>Make some random data for demo.</summary>
    private static (DenseData X, float[] Y) MakeCategorical(int nSamples, int nFeatures, int nCategories, bool onehot)
    {
        var rng = new Rng(1994);
        var columns = Enumerable.Range(0, nFeatures + 1)
            .Select(_ => Enumerable.Range(0, nSamples).Select(_ => rng.Integers(0, nCategories)).ToArray())
            .ToArray();

        // The first column only contributes to the label.
        var label = new float[nSamples];
        for (var r = 0; r < nSamples; r++)
            label[r] = columns.Sum(c => c[r]) + 1;

        var features = columns[1..];
        if (!onehot)
        {
            var x = DenseData.Zeros(nSamples, nFeatures);
            for (var r = 0; r < nSamples; r++)
                for (var c = 0; c < nFeatures; c++) x[r, c] = features[c][r];
            return (x, label);
        }

        // One indicator column per (feature, category), like pd.get_dummies.
        var enc = DenseData.Zeros(nSamples, nFeatures * nCategories);
        for (var r = 0; r < nSamples; r++)
            for (var c = 0; c < nFeatures; c++) enc[r, c * nCategories + features[c][r]] = 1;
        return (enc, label);
    }
}
