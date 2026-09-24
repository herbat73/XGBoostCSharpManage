using System.Diagnostics;
using System.Globalization;
using XGBoost;
using XGBoost.Demos.GuidePython;
using XGBoost.Demos.Other;

// C# ports of the Python demos in the repository's demo folder.
//   dotnet run -- <demo> [demo options]   run one demo
//   dotnet run -- all                     run every demo with its default options
//   dotnet run                            list the demos

// Print numbers the same way on every machine (e.g. 0.5, not 0,5).
CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

(string Name, Action<string[]> Run)[] demos =
[
    ("basic_walkthrough", BasicWalkthrough.Run),
    ("boost_from_prediction", BoostFromPrediction.Run),
    ("callbacks", Callbacks.Run),
    ("cat_pipeline", CatPipeline.Run),
    ("categorical", Categorical.Run),
    ("continuation", Continuation.Run),
    ("cross_validation", CrossValidation.Run),
    ("custom_rmsle", CustomRmsle.Run),
    ("custom_softmax", CustomSoftmax.Run),
    ("evals_result", EvalsResult.Run),
    ("feature_weights", FeatureWeights.Run),
    ("gamma_regression", GammaRegression.Run),
    ("generalized_linear_model", GeneralizedLinearModel.Run),
    ("gpu_tree_shap", TreeShap.Run),
    ("individual_trees", IndividualTrees.Run),
    ("learning_to_rank", LearningToRank.Run),
    ("model_parser", ModelParser.Run),
    ("multioutput_regression", MultioutputRegression.Run),
    ("predict_first_ntree", PredictFirstNTree.Run),
    ("predict_leaf_indices", PredictLeafIndices.Run),
    ("prediction_intervals", PredictionIntervals.Run),
    ("sklearn_evals_result", SklearnEvalsResult.Run),
    ("sklearn_examples", SklearnExamples.Run),
    ("sklearn_parallel", SklearnParallel.Run),
    ("update_process", UpdateProcess.Run),
    ("aft_survival", AftSurvival.Run),
    ("multiclass_classification", MulticlassClassification.Run),
];

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine($"XGBoost {XGB.NativeVersion} C# demos. Usage: dotnet run -- <demo>|all [options]");
    Console.WriteLine();
    foreach (var (name,  _) in demos) Console.WriteLine($"  {name, -27}");
    return 0;
}

if (args[0] == "all")
{
    var failed = new List<string>();
    foreach (var (name, run) in demos)
    {
        Console.WriteLine($"==================== {name})");
        var timer = Stopwatch.StartNew();
        try
        {
            run([]);
            Console.WriteLine($"-------------------- {name}: OK in {timer.Elapsed.TotalSeconds:F1}s");
        }
        catch (Exception e)
        {
            failed.Add(name);
            Console.WriteLine($"-------------------- {name}: FAILED\n{e}");
        }
    }
    Console.WriteLine(failed.Count == 0 ? "All demos passed." : $"Failed: {string.Join(", ", failed)}");
    return failed.Count == 0 ? 0 : 1;
}

var demo = demos.FirstOrDefault(d => d.Name == args[0]);
if (demo.Run is null)
{
    Console.Error.WriteLine($"Unknown demo '{args[0]}'. Run without arguments to list the demos.");
    return 2;
}
demo.Run(args[1..]);
return 0;
