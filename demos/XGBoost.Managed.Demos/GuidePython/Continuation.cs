using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Training continuation. Port of <c>demo/guide-python/continuation.py</c>, using the native interface in
/// place of <c>XGBClassifier</c> and a full snapshot (<see cref="Booster.Serialize"/>) in place of pickle.
/// </summary>
internal static class Continuation
{
    private static readonly List<KeyValuePair<string, string>> Params =
        P.Of(("objective", "binary:logistic"), ("eval_metric", "logloss"));

    public static void Run(string[] args)
    {
        var tmpdir = Directory.CreateTempSubdirectory("xgboost-continuation-");
        try
        {
            TrainingContinuationEarlyStop(tmpdir.FullName, useSnapshot: false);
            TrainingContinuationEarlyStop(tmpdir.FullName, useSnapshot: true);

            TrainingContinuation(tmpdir.FullName, useSnapshot: true);
            TrainingContinuation(tmpdir.FullName, useSnapshot: false);
        }
        finally
        {
            tmpdir.Delete(recursive: true);
        }
    }

    /// <summary>Saves and loads back the model; this could be a checkpoint.</summary>
    private static Booster SaveAndLoad(Booster booster, string tmpdir, string name, bool useSnapshot)
    {
        if (useSnapshot)
        {
            var path = Path.Combine(tmpdir, name + ".snapshot");
            File.WriteAllBytes(path, booster.Serialize());
            return Booster.Deserialize(File.ReadAllBytes(path));
        }
        else
        {
            var path = Path.Combine(tmpdir, name + ".json");
            booster.Save(path);
            return Booster.Load(path);
        }
    }

    /// <summary>Basic training continuation.</summary>
    private static void TrainingContinuation(string tmpdir, bool useSnapshot)
    {
        // Train 128 iterations in 1 session
        var (x, y) = Datasets.BinaryClassification(new Rng(0));
        using var xy = x.ToDMatrix(y);
        using (var full = Training.Train(Params, xy, 128, evals: [(xy, "validation_0")], verbose: false))
        {
            Console.WriteLine($"Total boosted rounds: {full.BoostedRounds}");
        }

        // Train 128 iterations in 2 sessions, with the first one runs for 32 iterations and the second one
        // runs for 96 iterations
        using var first = Training.Train(Params, xy, 32, evals: [(xy, "validation_0")], verbose: false);
        Check.That(first.BoostedRounds == 32, "expected 32 rounds");

        using var loaded = SaveAndLoad(first, tmpdir, "model-first-32", useSnapshot);
        using var second = Training.Train(Params, xy, 128 - 32, evals: [(xy, "validation_0")], verbose: false,
            xgbModel: loaded);

        Console.WriteLine($"Total boosted rounds: {second.BoostedRounds}");
        Check.That(second.BoostedRounds == 128, "expected 128 rounds");
    }

    /// <summary>Training continuation with early stopping.</summary>
    private static void TrainingContinuationEarlyStop(string tmpdir, bool useSnapshot)
    {
        const int earlyStoppingRounds = 5;
        const int nEstimators = 512;

        var (x, y) = Datasets.BinaryClassification(new Rng(0));
        using var xy = x.ToDMatrix(y);
        int best;
        using (var clf = Training.Train(Params, xy, nEstimators, evals: [(xy, "validation_0")], verbose: false,
                   earlyStoppingRounds: earlyStoppingRounds, saveBest: true))
        {
            Console.WriteLine($"Total boosted rounds: {clf.BoostedRounds}");
            best = int.Parse(clf.GetAttribute("best_iteration")!);
        }

        // Train 512 iterations in 2 sessions, with the first one runs for 128 iterations and the second one
        // runs until early stop.
        using var first = Training.Train(Params, xy, 128, evals: [(xy, "validation_0")], verbose: false,
            earlyStoppingRounds: earlyStoppingRounds, saveBest: true);
        Check.That(first.BoostedRounds == 128, "expected 128 rounds");

        using var loaded = SaveAndLoad(first, tmpdir, "model-first-128", useSnapshot);
        using var second = Training.Train(Params, xy, nEstimators - 128, evals: [(xy, "validation_0")],
            verbose: false, earlyStoppingRounds: earlyStoppingRounds, saveBest: true, xgbModel: loaded);

        Console.WriteLine($"Total boosted rounds: {second.BoostedRounds}");
        Check.That(int.Parse(second.GetAttribute("best_iteration")!) == best,
            "continued training found a different best iteration");
    }
}
