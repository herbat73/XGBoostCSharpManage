using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Parallel parameter optimization. Port of <c>demo/guide-python/sklearn_parallel.py</c>: two fits run
/// at the same time, each with half of the cores, on the synthetic data the Python demo falls back to.
/// </summary>
internal static class SklearnParallel
{
    public static void Run(string[] args)
    {
        Console.WriteLine("Parallel Parameter optimization");
        var (x, y) = Datasets.MakeRegression(20640, 8, new Rng(1234));
        // Make sure the number of threads is balanced.
        var baseParams = P.Of(("tree_method", "hist"), ("nthread", Math.Max(1, Environment.ProcessorCount / 2)));
        var (bestScore, bestParams) = ModelSelection.GridSearch(x, y, baseParams,
            [("max_depth", [2, 4, 6]), ("n_estimators", [50, 100, 200])], cv: 5, maxParallelism: 2);
        Console.WriteLine(bestScore);
        Console.WriteLine(bestParams);
    }
}
