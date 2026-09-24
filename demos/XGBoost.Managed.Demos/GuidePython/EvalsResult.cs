using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>How to access the eval metrics. Port of <c>demo/guide-python/evals_result.py</c>.</summary>
internal static class EvalsResult
{
    public static void Run(string[] args)
    {
        using var dtrain = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.train"));
        using var dtest = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.test"));

        // Parameters are key/value pairs, so a key such as eval_metric can repeat.
        var param = P.Of(
            ("max_depth", 2),
            ("objective", "binary:logistic"),
            ("eval_metric", "logloss"),
            ("eval_metric", "error"));

        const int numRound = 2;
        (DMatrix, string)[] watchlist = [(dtest, "eval"), (dtrain, "train")];

        // XGB.Train reports every round's results to onIteration; collect them into a history.
        var evalsResult = new EvalsLog();
        using var bst = XGB.Train(param, dtrain, numRound, watchlist, onIteration: (i, results) =>
        {
            Console.WriteLine(Training.FormatLine(i, results));
            evalsResult.Append(results);
        });

        Print(evalsResult, "eval");
    }

    internal static void Print(EvalsLog evalsResult, string firstSet)
    {
        Console.WriteLine($"Access logloss metric directly from {firstSet}:");
        Console.WriteLine(Fmt.List(evalsResult[firstSet]["logloss"]));

        Console.WriteLine();
        Console.WriteLine("Access metrics through a loop:");
        foreach (var (name, metrics) in evalsResult)
        {
            Console.WriteLine($"- {name}");
            foreach (var (metric, values) in metrics)
            {
                Console.WriteLine($"   - {metric}");
                Console.WriteLine($"      - {Fmt.List(values)}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("Access complete dictionary:");
        Console.WriteLine(evalsResult);
    }
}
