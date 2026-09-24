using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Feature engineering pipeline for categorical data. Port of <c>demo/guide-python/cat_pipeline.py</c>.
/// </summary>
/// <remarks>
/// Shows how to keep the categorical encoding consistent across training and inference: an ordinal
/// encoder is fitted on the training data only and reused for every later input, with unseen categories
/// mapped to missing. scikit-learn's <c>OrdinalEncoder</c> becomes the small encoder below.
/// </remarks>
internal static class CatPipeline
{
    // We have three categorical features, while the rest are numerical.
    private static readonly string[] CategoricalFeatures = ["brand_id", "retailer_id", "category_id"];
    private static readonly string[] FeatureNames = [.. CategoricalFeatures, "price", "stock_status", "on_sale"];

    public static void Run(string[] args)
    {
        Pipeline();
        Native();
        Console.WriteLine("Predictions are consistent between separately encoded batches.");
    }

    /// <summary>Maps each categorical column's values to codes 0..n-1 learnt from the training data.</summary>
    private sealed class OrdinalEncoder
    {
        private readonly Dictionary<int, Dictionary<float, int>> _categories = [];

        public OrdinalEncoder Fit(DenseData x, IEnumerable<int> columns)
        {
            foreach (var c in columns)
            {
                var sorted = x.Column(c).Distinct().Order().ToArray();
                _categories[c] = sorted.Select((v, i) => (v, i)).ToDictionary(p => p.v, p => p.i);
            }
            return this;
        }

        /// <summary>Encodes a copy of <paramref name="x"/>; unknown categories become NaN (missing).</summary>
        public DenseData Transform(DenseData x)
        {
            var result = x with { Values = (float[])x.Values.Clone() };
            foreach (var (c, codes) in _categories)
                for (var r = 0; r < x.Rows; r++)
                    result[r, c] = codes.TryGetValue(x[r, c], out var code) ? code : float.NaN;
            return result;
        }
    }

    /// <summary>Generate data for demo.</summary>
    private static (DenseData X, float[] Y) MakeExampleData()
    {
        const int nSamples = 2048;
        var rng = new Rng(1994);
        var x = DenseData.Zeros(nSamples, FeatureNames.Length);
        var y = new float[nSamples];
        for (var r = 0; r < nSamples; r++)
        {
            for (var c = 0; c < CategoricalFeatures.Length; c++) x[r, c] = rng.Integers(32, 96);
            x[r, 3] = rng.Integers(100, 200); // price
            x[r, 4] = rng.Integers(0, 2); // stock_status
            x[r, 5] = rng.Integers(0, 2); // on_sale
            y[r] = (float)rng.Normal();
        }
        return (x, y);
    }

    private static DMatrix ToDMatrix(DenseData x, float[] y, string[] featureTypes)
    {
        var d = x.ToDMatrix(y);
        d.FeatureNames = FeatureNames;
        d.FeatureTypes = featureTypes;
        return d;
    }

    private static readonly int[] CategoricalColumns = [0, 1, 2];

    /// <summary>Using the native XGBoost interface.</summary>
    private static void Native()
    {
        var (x, y) = MakeExampleData();
        var (trainIdx, testIdx) = Datasets.TrainTestSplit(x.Rows, 0.2, new Rng(1994));
        var xTrain = x.TakeRows(trainIdx);
        var xTest = x.TakeRows(testIdx);

        // Create an encoder based on training data.
        var enc = new OrdinalEncoder().Fit(xTrain, CategoricalColumns);

        // Encode the data based on fitted encoder.
        var xTrainEnc = enc.Transform(xTrain);
        var xTestEnc = enc.Transform(xTest);
        // Train XGBoost model using the native interface.
        string[] featureTypes = ["c", "c", "c", "q", "q", "q"];
        using var xyTrain = ToDMatrix(xTrainEnc, y.Gather(trainIdx), featureTypes);
        using var xyTest = ToDMatrix(xTestEnc, y.Gather(testIdx), featureTypes);
        using var booster = XGB.Train([], xyTrain, 10);
        booster.Predict(xyTest);

        // Following shows that data are encoded consistently.

        // We first obtain result from newly encoded data
        var head = enc.Transform(xTrain.Head(16));
        var predt0 = booster.InplacePredict(head.Values, head.Rows, head.Cols).Values;
        // then we obtain result from already encoded data from training.
        var trainHead = xTrainEnc.Head(16);
        var predt1 = booster.InplacePredict(trainHead.Values, trainHead.Rows, trainHead.Cols).Values;

        Check.AllClose(predt0, predt1, what: "predictions");
    }

    /// <summary>Encoder and model used together, like a scikit-learn pipeline.</summary>
    private static void Pipeline()
    {
        var (x, y) = MakeExampleData();
        var (trainIdx, _) = Datasets.TrainTestSplit(x.Rows, 0.2, new Rng(3));
        var xTrain = x.TakeRows(trainIdx);
        var yTrain = y.Gather(trainIdx);

        // all categorical feature names end with "_id"
        var catColumns = FeatureNames.Select((n, i) => (n, i)).Where(p => p.n.EndsWith("_id")).Select(p => p.i).ToArray();
        var featureTypes = FeatureNames.Select(n => CategoricalFeatures.Contains(n) ? "c" : "q").ToArray();

        var enc = new OrdinalEncoder().Fit(xTrain, catColumns);
        using var xy = ToDMatrix(enc.Transform(xTrain), yTrain, featureTypes);
        using var reg = XGB.Train([], xy, 10);

        // check XGBoost is using the feature type correctly.
        Check.That(reg.FeatureTypes.SequenceEqual(featureTypes), "model feature types differ from the data");

        float[] Predict(DenseData data)
        {
            var encoded = enc.Transform(data);
            return reg.InplacePredict(encoded.Values, encoded.Rows, encoded.Cols).Values;
        }

        // We first create a slice of data that doesn't contain all the categories
        var predt0 = Predict(xTrain.Head(16));
        // Then we use the data that contains all the categories
        var predt1 = Predict(xTrain)[..16];

        // The resulting encoding is the same
        Check.AllClose(predt0, predt1, what: "predictions");
    }
}
