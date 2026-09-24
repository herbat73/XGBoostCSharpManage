using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>Obtaining leaf indices. Port of <c>demo/guide-python/predict_leaf_indices.py</c>.</summary>
internal static class PredictLeafIndices
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

        Console.WriteLine("start testing predict the leaf indices");
        // predict using first 2 tree
        var leafIndex = bst.Predict(dtest, PredictionType.Leaf, iterationEnd: 2, strictShape: true);
        // strict shape is (rows, rounds, classes, trees per round); print it as rows x trees.
        Console.WriteLine(Fmt.Shape(leafIndex.Shape));
        Console.WriteLine(Fmt.Matrix(leafIndex.Values, leafIndex.Rows, leafIndex.Values.Length / leafIndex.Rows));
        // predict all trees
        leafIndex = bst.Predict(dtest, PredictionType.Leaf);
        Console.WriteLine(Fmt.Shape(leafIndex.Shape));
    }
}
