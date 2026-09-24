using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Using and defining callback functions. Port of <c>demo/guide-python/callbacks.py</c>. The Python
/// demo's matplotlib callback becomes one that draws the evaluation history in the console.
/// </summary>
internal static class Callbacks
{
    public static void Run(string[] args)
    {
        CheckPointCallback();
        CustomCallback();
    }

    /// <summary>Draws the evaluation history as console bars every <paramref name="period"/> rounds.</summary>
    private sealed class ConsolePlot(int rounds, int period)
    {
        public bool AfterIteration(Booster model, int epoch, EvalsLog evalsLog)
        {
            if ((epoch + 1) % period != 0 && epoch != rounds - 1) return false;
            Console.WriteLine($"round {epoch + 1}/{rounds}");
            foreach (var (data, metrics) in evalsLog)
            {
                foreach (var (metricName, log) in metrics)
                {
                    var key = $"{data}-{metricName}";
                    var max = log.Max();
                    var bar = new string('#', max > 0 ? (int)Math.Round(40 * log[^1] / max) : 0);
                    Console.WriteLine($"  {key,-14} {Fmt.Num(log[^1]),-10} {bar}");
                }
            }
            // false to indicate training should not stop.
            return false;
        }
    }

    /// <summary>
    /// Defining a custom callback that shows the evaluation result during training.
    /// </summary>
    private static void CustomCallback()
    {
        var (x, y) = Datasets.BinaryClassification(new Rng(0));
        var (trainIdx, validIdx) = Datasets.TrainTestSplit(x.Rows, 0.25, new Rng(0));

        using var dTrain = x.TakeRows(trainIdx).ToDMatrix(y.Gather(trainIdx));
        using var dValid = x.TakeRows(validIdx).ToDMatrix(y.Gather(validIdx));

        const int numBoostRound = 100;
        var plotting = new ConsolePlot(numBoostRound, period: 20);

        // Pass it to the callbacks parameter as a list.
        using var _ = Training.Train(
            P.Of(("objective", "binary:logistic"), ("eval_metric", "error"), ("eval_metric", "rmse"),
                ("tree_method", "hist")),
            dTrain, numBoostRound, evals: [(dTrain, "Train"), (dValid, "Valid")], verbose: false,
            callbacks: [plotting.AfterIteration]);
    }

    /// <summary>
    /// Saves the model every <paramref name="interval"/> rounds, like Python's <c>TrainingCheckPoint</c>:
    /// either the model alone (UBJSON) or a full snapshot including the training configuration.
    /// </summary>
    private static TrainingCallback CheckPoint(string directory, string name, int interval, bool asSnapshot)
    {
        var start = -1;
        var counter = 0;
        return (model, epoch, _) =>
        {
            if (start < 0) start = model.BoostedRounds - 1 - epoch;
            if (counter == interval)
            {
                var path = Path.Combine(directory, $"{name}_{epoch + start}.{(asSnapshot ? "snapshot" : "ubj")}");
                counter = 0;
                if (asSnapshot) File.WriteAllBytes(path, model.Serialize());
                else model.Save(path);
            }
            counter++;
            return false;
        };
    }

    /// <summary>
    /// Demo for checkpointing. Custom logic for handling output is usually required and users are encouraged
    /// to write their own callback for checkpointing operations.
    /// </summary>
    private static void CheckPointCallback()
    {
        // Only for demo, set a larger value (like 100) in practice as checkpointing is quite slow.
        const int rounds = 2;
        var tmpdir = Directory.CreateTempSubdirectory("xgboost-checkpoint-");

        void CheckFiles(bool asSnapshot)
        {
            for (var i = rounds; i < 10; i += rounds)
            {
                var path = Path.Combine(tmpdir.FullName, $"model_{i}.{(asSnapshot ? "snapshot" : "ubj")}");
                Check.That(File.Exists(path), $"missing checkpoint {path}");
            }
        }

        try
        {
            var (x, y) = Datasets.BinaryClassification(new Rng(0));
            using var m = x.ToDMatrix(y);
            var param = P.Of(("objective", "binary:logistic"));

            using (Training.Train(param, m, 10, verbose: false,
                       callbacks: [CheckPoint(tmpdir.FullName, "model", rounds, asSnapshot: false)]))
            {
                CheckFiles(false);
            }

            // This version of checkpoint saves everything including parameters and model.
            // See: doc/tutorials/saving_model.rst
            using (Training.Train(param, m, 10, verbose: false,
                       callbacks: [CheckPoint(tmpdir.FullName, "model", rounds, asSnapshot: true)]))
            {
                CheckFiles(true);
            }

            Console.WriteLine("Checkpoints: " + string.Join(", ",
                tmpdir.GetFiles().Select(f => f.Name).Order(StringComparer.Ordinal)));
        }
        finally
        {
            tmpdir.Delete(recursive: true);
        }
    }
}
