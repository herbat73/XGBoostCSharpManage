using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Getting started with XGBoost. Port of <c>demo/guide-python/basic_walkthrough.py</c>.
/// </summary>
internal static class BasicWalkthrough
{
    public static void Run(string[] args)
    {
        // X is a CSR matrix; XGBoost supports many other input types (DMatrix.FromDense, DMatrix.FromFile, ...)
        var x = Datasets.LoadLibSvm(DemoPaths.Data("agaricus.txt.train"));
        var xTest = Datasets.LoadLibSvm(DemoPaths.Data("agaricus.txt.test"));
        var numCols = Math.Max(x.NumCols, xTest.NumCols);
        using var dtrain = x.ToDMatrix(numCols);
        // validation set
        using var dtest = xTest.ToDMatrix(numCols);

        // specify parameters via map, definition are same as c++ version
        var param = P.Of(("max_depth", 2), ("eta", 1), ("objective", "binary:logistic"));

        // specify validations set to watch performance
        (DMatrix, string)[] watchlist = [(dtest, "eval"), (dtrain, "train")];
        // number of boosting rounds
        const int numRound = 2;
        using var bst = XGB.Train(param, dtrain, numRound, watchlist,
            onIteration: (i, results) => Console.WriteLine(Training.FormatLine(i, results)));

        // run prediction
        var preds = bst.Predict(dtest).Values;
        var labels = dtest.Label;
        var errors = preds.Where((p, i) => (p > 0.5 ? 1 : 0) != labels[i]).Count();
        Console.WriteLine($"error={(double)errors / preds.Length:F6}");

        bst.Save(DemoPaths.Output("model-0.json"));
        // dump model
        WriteDump(DemoPaths.Output("dump.raw.txt"), bst.DumpModel());
        // dump model with feature map
        WriteDump(DemoPaths.Output("dump.nice.txt"), bst.DumpModel(featureMap: DemoPaths.Data("featmap.txt")));

        // save dmatrix into binary buffer
        dtest.SaveBinary(DemoPaths.Output("dtest.dmatrix"));
        // save model
        bst.Save(DemoPaths.Output("model-1.json"));
        // load model and data in
        using var bst2 = Booster.Load(DemoPaths.Output("model-1.json"));
        using var dtest2 = DMatrix.FromFile(DemoPaths.Output("dtest.dmatrix"));
        var preds2 = bst2.Predict(dtest2).Values;
        // assert they are the same
        Check.That(preds2.SequenceEqual(preds), "predictions from the reloaded model differ");

        // alternatively, take a full snapshot of the booster (the equivalent of pickling it)
        var snapshot = bst2.Serialize();
        // load model and data in
        using var bst3 = Booster.Deserialize(snapshot);
        var preds3 = bst3.Predict(dtest2).Values;
        // assert they are the same
        Check.That(preds3.SequenceEqual(preds), "predictions from the snapshot differ");

        Console.WriteLine($"Models, dumps and the binary DMatrix were written to {DemoPaths.OutputDir}");
    }

    /// <summary>Writes a dump in the layout of Python's <c>Booster.dump_model</c>.</summary>
    internal static void WriteDump(string path, string[] trees)
    {
        using var writer = new StreamWriter(path);
        for (var i = 0; i < trees.Length; i++)
        {
            writer.WriteLine($"booster[{i}]:");
            writer.Write(trees[i]);
        }
    }
}
