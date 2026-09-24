using System.Globalization;
using System.Text;

namespace XGBoost.Demos.Common;

/// <summary>Custom objective: gradient and Hessian from the raw (margin) prediction, both shaped like it.</summary>
internal delegate (float[] Grad, float[] Hess) Objective(Prediction predt, DMatrix dtrain);

/// <summary>
/// Custom metric. Receives the margin when a custom objective is used and the transformed prediction
/// otherwise, as in the Python package.
/// </summary>
internal delegate (string Name, double Value) Metric(Prediction predt, DMatrix data);

/// <summary>Called after every round; return true to stop training.</summary>
internal delegate bool TrainingCallback(Booster model, int epoch, EvalsLog log);

/// <summary>Evaluation history, <c>set → metric → value per round</c>, like Python's <c>evals_result</c>.</summary>
internal sealed class EvalsLog : Dictionary<string, Dictionary<string, List<double>>>
{
    /// <summary>Appends one round of <c>"{set}-{metric}" → value</c> results.</summary>
    public void Append(IEnumerable<KeyValuePair<string, double>> results)
    {
        foreach (var (key, value) in results)
        {
            var dash = key.IndexOf('-');
            var set = key[..dash];
            var metric = key[(dash + 1)..];
            if (!TryGetValue(set, out var metrics)) this[set] = metrics = [];
            if (!metrics.TryGetValue(metric, out var history)) metrics[metric] = history = [];
            history.Add(value);
        }
    }

    public override string ToString() =>
        "{" + string.Join(", ", this.Select(s =>
            $"'{s.Key}': {{" + string.Join(", ", s.Value.Select(m => $"'{m.Key}': {Fmt.List(m.Value)}")) + "}")) + "}";
}

/// <summary>
/// The parts of Python's <c>xgboost.train</c> and <c>xgboost.cv</c> that <see cref="XGB.Train"/> does not
/// cover (custom objectives and metrics, evaluation history, callbacks, training continuation and
/// cross validation), built from the <see cref="Booster"/> primitives.
/// </summary>
internal static class Training
{
    /// <summary>Mirrors <c>xgboost.train</c>.</summary>
    /// <param name="parameters">Booster parameters.</param>
    /// <param name="dtrain">Training data.</param>
    /// <param name="numBoostRound">Number of rounds to add.</param>
    /// <param name="evals">Sets evaluated after every round.</param>
    /// <param name="obj">Custom objective.</param>
    /// <param name="customMetric">Custom metric, reported after the built-in ones.</param>
    /// <param name="earlyStoppingRounds">Stop when the last metric on the last set has not improved for this many rounds.</param>
    /// <param name="maximize">Whether larger is better for early stopping; inferred from the metric name when null.</param>
    /// <param name="saveBest">With early stopping, return only the rounds up to the best one.</param>
    /// <param name="evalsResult">Receives the evaluation history.</param>
    /// <param name="verbose">Print the evaluation results of every round.</param>
    /// <param name="xgbModel">Model to continue training from; it is copied, not modified.</param>
    /// <param name="callbacks">Called after every round.</param>
    public static Booster Train(IEnumerable<KeyValuePair<string, string>> parameters, DMatrix dtrain,
        int numBoostRound = 10, IReadOnlyList<(DMatrix Data, string Name)>? evals = null, Objective? obj = null,
        Metric? customMetric = null, int? earlyStoppingRounds = null, bool? maximize = null, bool saveBest = false,
        EvalsLog? evalsResult = null, bool verbose = true, Booster? xgbModel = null,
        IReadOnlyList<TrainingCallback>? callbacks = null)
    {
        var sets = evals?.ToArray() ?? [];
        if (earlyStoppingRounds is not null && sets.Length == 0)
            throw new ArgumentException("Early stopping requires at least one evaluation set.", nameof(evals));

        Booster booster;
        if (xgbModel is null)
        {
            booster = new Booster(parameters, [dtrain, .. sets.Select(s => s.Data)]);
        }
        else
        {
            // A full snapshot keeps the training configuration of the original model.
            booster = Booster.Deserialize(xgbModel.Serialize());
            booster.SetParams(parameters);
        }

        // Early stopping counts rounds of the whole model, so continued training reports absolute rounds.
        var startRound = booster.BoostedRounds;
        var log = evalsResult ?? [];
        var bestIteration = -1;
        var bestScore = double.NaN;

        for (var i = 0; i < numBoostRound; i++)
        {
            Update(booster, dtrain, i, obj);

            var stop = false;
            if (sets.Length > 0)
            {
                var results = Evaluate(booster, sets, i, customMetric, outputMargin: obj is not null);
                log.Append(results);
                if (verbose) Console.WriteLine(FormatLine(i, results));

                if (earlyStoppingRounds is { } patience && results.Count > 0)
                {
                    var (key, score) = results.GetAt(results.Count - 1);
                    var larger = maximize ?? IsMaximizeMetric(key);
                    var epoch = startRound + i;
                    if (bestIteration < 0 || (larger ? score > bestScore : score < bestScore))
                    {
                        bestScore = score;
                        bestIteration = epoch;
                        booster.SetAttribute("best_iteration", bestIteration.ToString(CultureInfo.InvariantCulture));
                        booster.SetAttribute("best_score", bestScore.ToString("R", CultureInfo.InvariantCulture));
                    }
                    else if (epoch - bestIteration >= patience)
                    {
                        stop = true;
                    }
                }
            }

            foreach (var callback in callbacks ?? []) stop |= callback(booster, i, log);
            if (stop) break;
        }

        if (saveBest && bestIteration >= 0)
        {
            var best = booster.Slice(0, bestIteration + 1);
            best.SetAttribute("best_iteration", bestIteration.ToString(CultureInfo.InvariantCulture));
            best.SetAttribute("best_score", bestScore.ToString("R", CultureInfo.InvariantCulture));
            booster.Dispose();
            booster = best;
        }
        return booster;
    }

    /// <summary>One boosting round, with the built-in objective or with <paramref name="obj"/>.</summary>
    public static void Update(Booster booster, DMatrix dtrain, int iteration, Objective? obj)
    {
        if (obj is null)
        {
            booster.Update(dtrain, iteration);
            return;
        }
        var predt = booster.Predict(dtrain, PredictionType.Margin, training: true);
        var (grad, hess) = obj(predt, dtrain);
        var targets = predt.Shape.Length == 2 ? checked((int)predt.Shape[1]) : 1;
        booster.Boost(dtrain, iteration, grad, hess, targets);
    }

    /// <summary>Built-in metrics for every set, followed by the custom metric for every set.</summary>
    public static OrderedDictionary<string, double> Evaluate(Booster booster, (DMatrix Data, string Name)[] sets,
        int iteration, Metric? metric, bool outputMargin)
    {
        var results = booster.Evaluate(iteration, sets);
        if (metric is null) return results;
        foreach (var (data, name) in sets)
        {
            var predt = booster.Predict(data, outputMargin ? PredictionType.Margin : PredictionType.Value);
            var (metricName, value) = metric(predt, data);
            results[$"{name}-{metricName}"] = value;
        }
        return results;
    }

    /// <summary>Mirrors <c>xgboost.cv</c>; returns <c>"{set}-{metric}-mean|std"</c> → value per round.</summary>
    /// <param name="parameters">Booster parameters.</param>
    /// <param name="dtrain">Data to split into folds.</param>
    /// <param name="numBoostRound">Number of rounds.</param>
    /// <param name="nfold">Number of folds.</param>
    /// <param name="metrics">Evaluation metrics, replacing any <c>eval_metric</c> in <paramref name="parameters"/>.</param>
    /// <param name="obj">Custom objective.</param>
    /// <param name="customMetric">Custom metric.</param>
    /// <param name="fpreproc">Adjusts each fold's data and parameters before training.</param>
    /// <param name="earlyStoppingRounds">Stop when the last test metric has not improved for this many rounds.</param>
    /// <param name="maximize">Whether larger is better for early stopping; inferred from the metric name when null.</param>
    /// <param name="seed">Seed for the fold assignment.</param>
    /// <param name="showStdv">Print every round (with or without standard deviation); null prints nothing.</param>
    public static OrderedDictionary<string, List<double>> Cv(IEnumerable<KeyValuePair<string, string>> parameters,
        DMatrix dtrain, int numBoostRound = 10, int nfold = 3, IEnumerable<string>? metrics = null,
        Objective? obj = null, Metric? customMetric = null,
        Func<DMatrix, DMatrix, List<KeyValuePair<string, string>>, List<KeyValuePair<string, string>>>? fpreproc = null,
        int? earlyStoppingRounds = null, bool? maximize = null, int seed = 0, bool? showStdv = null)
    {
        var baseParams = parameters.ToList();
        if (metrics is not null)
        {
            baseParams.RemoveAll(p => p.Key == "eval_metric");
            baseParams.AddRange(metrics.Select(m => KeyValuePair.Create("eval_metric", m)));
        }

        var folds = new List<(Booster Booster, (DMatrix Data, string Name)[] Sets)>();
        var disposables = new List<IDisposable>();
        try
        {
            foreach (var (trainIdx, testIdx) in Datasets.KFold((int)dtrain.NumRows, nfold, new Rng(seed)))
            {
                Array.Sort(trainIdx);
                Array.Sort(testIdx);
                var foldTrain = dtrain.Slice(trainIdx);
                var foldTest = dtrain.Slice(testIdx);
                disposables.Add(foldTrain);
                disposables.Add(foldTest);
                var foldParams = fpreproc?.Invoke(foldTrain, foldTest, [.. baseParams]) ?? baseParams;
                var booster = new Booster(foldParams, foldTrain, foldTest);
                disposables.Add(booster);
                folds.Add((booster, [(foldTrain, "train"), (foldTest, "test")]));
            }

            var table = new OrderedDictionary<string, List<double>>();
            var bestIteration = -1;
            var bestScore = double.NaN;
            for (var i = 0; i < numBoostRound; i++)
            {
                var perFold = folds.Select(f =>
                {
                    Update(f.Booster, f.Sets[0].Data, i, obj);
                    return Evaluate(f.Booster, f.Sets, i, customMetric, outputMargin: obj is not null);
                }).ToList();

                var line = new StringBuilder($"[{i}]");
                foreach (var key in perFold[0].Keys)
                {
                    var values = perFold.Select(r => r[key]).ToArray();
                    var mean = values.Average();
                    var std = Math.Sqrt(values.Select(v => (v - mean) * (v - mean)).Average());
                    Column(table, key + "-mean").Add(mean);
                    Column(table, key + "-std").Add(std);
                    line.Append(CultureInfo.InvariantCulture, $"\t{key}:{mean:F5}");
                    if (showStdv == true) line.Append(CultureInfo.InvariantCulture, $"+{std:F5}");
                }
                if (showStdv is not null) Console.WriteLine(line);

                if (earlyStoppingRounds is not { } patience) continue;
                var lastKey = perFold[0].GetAt(perFold[0].Count - 1).Key;
                var score = table[lastKey + "-mean"][i];
                var larger = maximize ?? IsMaximizeMetric(lastKey);
                if (bestIteration < 0 || (larger ? score > bestScore : score < bestScore))
                {
                    bestScore = score;
                    bestIteration = i;
                }
                else if (i - bestIteration >= patience)
                {
                    // Like Python, keep the rounds up to the best one.
                    foreach (var column in table.Values) column.RemoveRange(bestIteration + 1, column.Count - bestIteration - 1);
                    break;
                }
            }
            return table;
        }
        finally
        {
            foreach (var d in disposables) d.Dispose();
        }
    }

    private static List<double> Column(OrderedDictionary<string, List<double>> table, string name)
    {
        if (!table.TryGetValue(name, out var column)) table[name] = column = [];
        return column;
    }

    /// <summary>Formats results like Python's <c>EvaluationMonitor</c>: <c>[3]\ttrain-rmse:0.12345\ttest-rmse:0.15432</c>.</summary>
    public static string FormatLine(int iteration, IEnumerable<KeyValuePair<string, double>> results) =>
        $"[{iteration}]" + string.Concat(results.Select(r => $"\t{r.Key}:{r.Value.ToString("F5", CultureInfo.InvariantCulture)}"));

    // Same rule as the Python package's EarlyStopping callback.
    private static bool IsMaximizeMetric(string key)
    {
        var metric = key[(key.IndexOf('-') + 1)..];
        string[] maximize = ["auc", "aucpr", "pre", "pre@", "map", "ndcg", "auc@", "aucpr@", "map@", "ndcg@"];
        return maximize.Any(m => metric == m || (m.EndsWith('@') && metric.StartsWith(m, StringComparison.Ordinal)));
    }
}
