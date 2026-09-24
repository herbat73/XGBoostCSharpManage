using System.Globalization;
using XGBoost.Demos.Common;

namespace XGBoost.Demos.Other;

/// <summary>
/// Multi-class classification with softmax and softprob. Port of
/// <c>demo/multiclass_classification/train.py</c>.
/// </summary>
/// <remarks>
/// The Python demo reads the UCI dermatology dataset downloaded by <c>runexp.sh</c>; pass its path with
/// <c>--data dermatology.data</c>. Without it, synthetic data of the same shape (366 rows, 33 features,
/// 6 classes) is used.
/// </remarks>
internal static class MulticlassClassification
{
    private const int NumClass = 6;

    public static void Run(string[] args)
    {
        var (x, y) = Load(new DemoArgs(args).Get("--data"));
        var trainRows = (int)(x.Rows * 0.7);
        var train = Enumerable.Range(0, trainRows).ToArray();
        var test = Enumerable.Range(trainRows, x.Rows - trainRows).ToArray();
        var testY = y.Gather(test);

        using var xgTrain = x.TakeRows(train).ToDMatrix(y.Gather(train));
        using var xgTest = x.TakeRows(test).ToDMatrix(testY);
        // setup parameters for xgboost
        var param = P.Of(
            // use softmax multi-class classification
            ("objective", "multi:softmax"),
            // scale weight of positive examples
            ("eta", 0.1),
            ("max_depth", 6),
            ("nthread", 4),
            ("num_class", NumClass));

        (DMatrix, string)[] watchlist = [(xgTrain, "train"), (xgTest, "test")];
        const int numRound = 5;
        using (var bst = XGB.Train(param, xgTrain, numRound, watchlist, onIteration: Print))
        {
            // get prediction
            var pred = bst.Predict(xgTest).Values;
            var errorRate = (double)pred.Where((p, i) => p != testY[i]).Count() / testY.Length;
            Console.WriteLine($"Test error using softmax = {errorRate}");
        }

        // do the same thing again, but output probabilities
        param[0] = KeyValuePair.Create("objective", "multi:softprob");
        using (var bst = XGB.Train(param, xgTrain, numRound, watchlist, onIteration: Print))
        {
            // the prediction has shape (ndata, nclass)
            var predProb = bst.Predict(xgTest);
            Check.That(predProb.Shape.SequenceEqual([testY.Length, NumClass]), "unexpected prediction shape");
            var predLabel = Enumerable.Range(0, testY.Length)
                .Select(r => Enumerable.Range(0, NumClass).MaxBy(c => predProb.Values[r * NumClass + c]))
                .ToArray();
            var errorRate = (double)predLabel.Where((p, i) => p != testY[i]).Count() / testY.Length;
            Console.WriteLine($"Test error using softprob = {errorRate}");
        }
    }

    private static void Print(int i, IReadOnlyDictionary<string, double> results) =>
        Console.WriteLine(Training.FormatLine(i, results));

    private static (DenseData X, float[] Y) Load(string? path)
    {
        if (path is null)
        {
            Console.WriteLine("No --data given; using synthetic data shaped like the dermatology dataset.");
            return Datasets.Blobs(366, 33, NumClass, new Rng(0), spread: 4);
        }

        // label need to be 0 to num_class -1. As in the Python demo, the features are the first 33 columns,
        // leaving out column 33 (age, which has missing values).
        var lines = File.ReadLines(path).Where(l => l.Length > 0).ToList();
        var x = DenseData.Zeros(lines.Count, 33);
        var y = new float[lines.Count];
        for (var r = 0; r < lines.Count; r++)
        {
            var cells = lines[r].Split(',');
            for (var c = 0; c < 33; c++) x[r, c] = float.Parse(cells[c], CultureInfo.InvariantCulture);
            y[r] = int.Parse(cells[34], CultureInfo.InvariantCulture) - 1;
        }
        return (x, y);
    }
}
