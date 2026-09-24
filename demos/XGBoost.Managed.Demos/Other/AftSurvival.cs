using XGBoost.Demos.Common;

namespace XGBoost.Demos.Other;

/// <summary>
/// Survival analysis (regression) with the Accelerated Failure Time (AFT) model. Port of
/// <c>demo/aft_survival/aft_survival_demo.py</c>.
/// </summary>
internal static class AftSurvival
{
    public static void Run(string[] args)
    {
        // The Veterans' Administration Lung Cancer Trial
        // The Statistical Analysis of Failure Time Data by Kalbfleisch J. and Prentice R (1980)
        var (header, df) = Datasets.LoadCsv(DemoPaths.Data("veterans_lung_cancer.csv"));
        Console.WriteLine($"Training data: {df.Rows} rows x {df.Cols} columns");
        Console.WriteLine(string.Join(", ", header));

        // Split features and labels
        var yLowerBound = df.Column(Array.IndexOf(header, "Survival_label_lower_bound"));
        var yUpperBound = df.Column(Array.IndexOf(header, "Survival_label_upper_bound"));
        var featureColumns = Enumerable.Range(0, header.Length)
            .Where(c => header[c] is not ("Survival_label_lower_bound" or "Survival_label_upper_bound"))
            .ToArray();
        var x = df.TakeColumns(featureColumns);

        // Split data into training and validation sets
        var (trainIndex, validIndex) = Datasets.TrainTestSplit(x.Rows, 0.7, new Rng(0));
        using var dtrain = x.TakeRows(trainIndex).ToDMatrix();
        dtrain.LabelLowerBound = yLowerBound.Gather(trainIndex);
        dtrain.LabelUpperBound = yUpperBound.Gather(trainIndex);
        using var dvalid = x.TakeRows(validIndex).ToDMatrix();
        dvalid.LabelLowerBound = yLowerBound.Gather(validIndex);
        dvalid.LabelUpperBound = yUpperBound.Gather(validIndex);

        // Train gradient boosted trees using AFT loss and metric
        var parameters = P.Of(
            ("verbosity", 0),
            ("objective", "survival:aft"),
            ("eval_metric", "aft-nloglik"),
            ("tree_method", "hist"),
            ("learning_rate", 0.05),
            ("aft_loss_distribution", "normal"),
            ("aft_loss_distribution_scale", 1.20),
            ("max_depth", 6),
            ("lambda", 0.01),
            ("alpha", 0.02));
        using var bst = XGB.Train(parameters, dtrain, numBoostRound: 10000,
            evals: [(dtrain, "train"), (dvalid, "valid")], earlyStoppingRounds: 50,
            onIteration: (i, results) => Console.WriteLine(Training.FormatLine(i, results)));
        Console.WriteLine($"best_iteration: {bst.GetAttribute("best_iteration")}, best_score: {bst.GetAttribute("best_score")}");

        // Run prediction on the validation set
        var predicted = bst.Predict(dvalid).Values;
        var lower = dvalid.LabelLowerBound;
        var upper = dvalid.LabelUpperBound;
        PrintTable(Enumerable.Range(0, predicted.Length), lower, upper, predicted);
        // Show only data points with right-censored labels
        Console.WriteLine();
        Console.WriteLine("Right-censored labels:");
        PrintTable(Enumerable.Range(0, predicted.Length).Where(i => float.IsPositiveInfinity(upper[i])),
            lower, upper, predicted);

        // Save trained model
        bst.Save(DemoPaths.Output("aft_model.json"));
        Console.WriteLine($"Model saved to {DemoPaths.Output("aft_model.json")}");
    }

    private static void PrintTable(IEnumerable<int> rows, float[] lower, float[] upper, float[] predicted)
    {
        Console.WriteLine($"{"",4} {"Label (lower bound)",20} {"Label (upper bound)",20} {"Predicted label",16}");
        foreach (var i in rows)
            Console.WriteLine($"{i,4} {lower[i],20} {(float.IsPositiveInfinity(upper[i]) ? "inf" : upper[i]),20} {predicted[i],16:F6}");
    }
}
