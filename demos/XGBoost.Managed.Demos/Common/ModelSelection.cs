using System.Collections.Concurrent;
using System.Text;

namespace XGBoost.Demos.Common;

/// <summary>Metrics and grid search standing in for <c>sklearn.metrics</c> and <c>GridSearchCV</c>.</summary>
internal static class ModelSelection
{
    public static double MeanSquaredError(IReadOnlyList<float> actual, IReadOnlyList<float> predicted)
    {
        double sum = 0;
        for (var i = 0; i < actual.Count; i++) sum += Math.Pow(actual[i] - predicted[i], 2);
        return sum / actual.Count;
    }

    /// <summary>Coefficient of determination, the default score of scikit-learn regressors.</summary>
    public static double R2(IReadOnlyList<float> actual, IReadOnlyList<float> predicted)
    {
        var mean = actual.Average(v => (double)v);
        double residual = 0, total = 0;
        for (var i = 0; i < actual.Count; i++)
        {
            residual += Math.Pow(actual[i] - predicted[i], 2);
            total += Math.Pow(actual[i] - mean, 2);
        }
        return 1 - residual / total;
    }

    public static string ConfusionMatrix(IReadOnlyList<float> actual, IReadOnlyList<float> predicted, int classes)
    {
        var counts = new int[classes, classes];
        for (var i = 0; i < actual.Count; i++) counts[(int)actual[i], (int)predicted[i]]++;
        var width = counts.Cast<int>().Max().ToString().Length;
        var sb = new StringBuilder("[");
        for (var r = 0; r < classes; r++)
        {
            sb.Append(r == 0 ? "[" : " [");
            sb.AppendJoin(' ', Enumerable.Range(0, classes).Select(c => counts[r, c].ToString().PadLeft(width)));
            sb.Append(r == classes - 1 ? "]]" : "]\n");
        }
        return sb.ToString();
    }

    /// <summary>Unshuffled K-fold, scikit-learn's default <c>cv</c> for regressors.</summary>
    public static IEnumerable<(int[] Train, int[] Test)> KFoldOrdered(int rows, int folds)
    {
        var start = 0;
        for (var k = 0; k < folds; k++)
        {
            var size = rows / folds + (k < rows % folds ? 1 : 0);
            var test = Enumerable.Range(start, size).ToArray();
            var train = Enumerable.Range(0, start).Concat(Enumerable.Range(start + size, rows - start - size)).ToArray();
            start += size;
            yield return (train, test);
        }
    }

    /// <summary>Trains a regressor and returns its predictions on <paramref name="test"/>.</summary>
    public static float[] FitPredict(IEnumerable<KeyValuePair<string, string>> parameters, int rounds,
        DenseData train, float[] trainLabel, DenseData test)
    {
        using var dtrain = train.ToDMatrix(trainLabel);
        using var booster = XGB.Train(parameters, dtrain, rounds);
        return booster.InplacePredict(test.Values, test.Rows, test.Cols).Values;
    }

    /// <summary>
    /// Exhaustive search over <paramref name="grid"/> scored by mean R² over <paramref name="cv"/> unshuffled
    /// folds, like <c>GridSearchCV</c> with an <c>XGBRegressor</c>. The key <c>n_estimators</c> sets the
    /// number of rounds; every other key is a booster parameter.
    /// </summary>
    /// <param name="maxParallelism">Number of candidate/fold fits run at the same time (<c>n_jobs</c>).</param>
    public static (double BestScore, string BestParams) GridSearch(DenseData x, float[] y,
        IReadOnlyList<KeyValuePair<string, string>> baseParams, IReadOnlyList<(string Key, object[] Values)> grid,
        int cv = 5, int maxParallelism = 1)
    {
        var candidates = new List<List<(string Key, object Value)>> { new() };
        foreach (var (key, values) in grid)
            candidates = [.. candidates.SelectMany(c => values.Select(v => new List<(string, object)>(c) { (key, v) }))];
        var folds = KFoldOrdered(x.Rows, cv).ToList();
        Console.WriteLine($"Fitting {folds.Count} folds for each of {candidates.Count} candidates, " +
            $"totalling {folds.Count * candidates.Count} fits");

        var scores = new ConcurrentDictionary<(int Candidate, int Fold), double>();
        var jobs = from c in Enumerable.Range(0, candidates.Count) from f in Enumerable.Range(0, folds.Count) select (c, f);
        Parallel.ForEach(jobs, new ParallelOptions { MaxDegreeOfParallelism = maxParallelism }, job =>
        {
            var candidate = candidates[job.c];
            var rounds = Convert.ToInt32(candidate.FirstOrDefault(p => p.Key == "n_estimators").Value ?? 100);
            var parameters = baseParams.Concat(candidate.Where(p => p.Key != "n_estimators")
                .Select(p => KeyValuePair.Create(p.Key, P.ToParam(p.Value)))).ToList();
            var (train, test) = folds[job.f];
            var predicted = FitPredict(parameters, rounds, x.TakeRows(train), y.Gather(train), x.TakeRows(test));
            scores[job] = R2(y.Gather(test), predicted);
        });

        var best = Enumerable.Range(0, candidates.Count)
            .Select(c => (Candidate: c, Score: Enumerable.Range(0, folds.Count).Average(f => scores[(c, f)])))
            .MaxBy(s => s.Score);
        var bestParams = "{" + string.Join(", ", candidates[best.Candidate].Select(p => $"'{p.Key}': {p.Value}")) + "}";
        return (best.Score, bestParams);
    }
}
