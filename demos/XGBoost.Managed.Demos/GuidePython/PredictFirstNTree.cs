using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Prediction using a number of trees. Port of <c>demo/guide-python/predict_first_ntree.py</c>; the
/// scikit-learn half of the Python demo becomes in-place prediction from a CSR matrix.
/// </summary>
internal static class PredictFirstNTree
{
    public static void Run(string[] args)
    {
        // load data in do training
        using var dtrain = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.train"));
        using var dtest = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.test"));
        var param = P.Of(("max_depth", 2), ("eta", 1), ("objective", "binary:logistic"));
        (DMatrix, string)[] watchlist = [(dtest, "eval"), (dtrain, "train")];
        const int numRound = 3;
        using var bst = XGB.Train(param, dtrain, numRound, watchlist,
            onIteration: (i, results) => Console.WriteLine(Training.FormatLine(i, results)));

        Console.WriteLine("start testing prediction from first n trees");
        // predict using first 1 tree
        var label = dtest.Label;
        var ypred1 = bst.Predict(dtest, iterationEnd: 1).Values;
        // by default, we predict using all the trees
        var ypred2 = bst.Predict(dtest).Values;

        Console.WriteLine($"error of ypred1={Error(ypred1, label):F6}");
        Console.WriteLine($"error of ypred2={Error(ypred2, label):F6}");

        InplaceInterface();
    }

    /// <summary>
    /// Stands in for the Python demo's scikit-learn half: in-memory CSR data, predicting straight from the
    /// CSR arrays without building a DMatrix.
    /// </summary>
    private static void InplaceInterface()
    {
        var train = Datasets.LoadLibSvm(DemoPaths.Data("agaricus.txt.train"));
        var test = Datasets.LoadLibSvm(DemoPaths.Data("agaricus.txt.test"));
        var numCols = Math.Max(train.NumCols, test.NumCols);
        using var dtrain = train.ToDMatrix(numCols);
        using var dtest = test.ToDMatrix(numCols);
        using var clf = XGB.Train(P.Of(("max_depth", 2), ("eta", 1), ("objective", "binary:logistic")), dtrain, 3,
            evals: [(dtest, "validation_0")],
            onIteration: (i, results) => Console.WriteLine(Training.FormatLine(i, results)));

        Console.WriteLine("start testing prediction from first n trees");
        // predict using first 1 tree
        var ypred1 = clf.InplacePredict(test.Indptr, test.Indices, test.Values, numCols, iterationEnd: 1).Values;
        // by default, we predict using all the trees
        var ypred2 = clf.InplacePredict(test.Indptr, test.Indices, test.Values, numCols).Values;

        Console.WriteLine($"error of ypred1={Error(ypred1, test.Labels):F6}");
        Console.WriteLine($"error of ypred2={Error(ypred2, test.Labels):F6}");
    }

    private static double Error(float[] predt, float[] label) =>
        (double)predt.Where((p, i) => (p > 0.5 ? 1 : 0) != label[i]).Count() / label.Length;
}
