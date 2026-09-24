using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>Using cross validation. Port of <c>demo/guide-python/cross_validation.py</c>.</summary>
internal static class CrossValidation
{
    public static void Run(string[] args)
    {
        // load data in do training
        using var dtrain = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.train"));
        var param = P.Of(("max_depth", 2), ("eta", 1), ("objective", "binary:logistic"));
        const int numRound = 2;

        Console.WriteLine("running cross validation");
        // do cross validation, this will print result out as
        // [iteration]  metric_name:mean_value+std_value
        // std_value is standard deviation of the metric
        Training.Cv(param, dtrain, numRound, nfold: 5, metrics: ["error"], seed: 0, showStdv: true);

        Console.WriteLine("running cross validation, disable standard deviation display");
        // do cross validation, this will print result out as
        // [iteration]  metric_name:mean_value
        var res = Training.Cv(param, dtrain, numBoostRound: 10, nfold: 5, metrics: ["error"], seed: 0,
            showStdv: false, earlyStoppingRounds: 3);
        Console.WriteLine(Fmt.Table(res));

        Console.WriteLine("running cross validation, with preprocessing function");
        // The preprocessing function receives each fold's training data, test data and parameters and
        // returns the parameters to use. We can use it to do weight rescale, etc.; as an example, we set
        // scale_pos_weight.
        List<KeyValuePair<string, string>> Fpreproc(DMatrix foldTrain, DMatrix foldTest,
            List<KeyValuePair<string, string>> foldParams)
        {
            var label = foldTrain.Label;
            var ratio = (double)label.Count(l => l == 0) / label.Count(l => l == 1);
            foldParams.Add(KeyValuePair.Create("scale_pos_weight", P.ToParam(ratio)));
            return foldParams;
        }

        // do cross validation, for each fold the data and param will be passed into Fpreproc, and its
        // return value will be used to generate the results of that fold
        res = Training.Cv(param, dtrain, numRound, nfold: 5, metrics: ["auc"], seed: 0, fpreproc: Fpreproc);
        Console.WriteLine(Fmt.Table(res));

        // you can also do cross validation with customized loss function
        Console.WriteLine("running cross validation, with customized loss function");

        static (float[], float[]) LogRegObj(Prediction predt, DMatrix data)
        {
            var labels = data.Label;
            var grad = new float[labels.Length];
            var hess = new float[labels.Length];
            for (var i = 0; i < labels.Length; i++)
            {
                var p = 1.0 / (1.0 + Math.Exp(-predt.Values[i]));
                grad[i] = (float)(p - labels[i]);
                hess[i] = (float)(p * (1.0 - p));
            }
            return (grad, hess);
        }

        // With a custom objective the metric receives the margin; margin > 0 means probability > 0.5.
        static (string, double) EvalError(Prediction predt, DMatrix data)
        {
            var labels = data.Label;
            var errors = labels.Where((l, i) => l != (predt.Values[i] > 0 ? 1 : 0)).Count();
            return ("error", (double)errors / labels.Length);
        }

        var customParam = P.Of(("max_depth", 2), ("eta", 1));
        // train with customized objective
        res = Training.Cv(customParam, dtrain, numRound, nfold: 5, seed: 0, obj: LogRegObj, customMetric: EvalError);
        Console.WriteLine(Fmt.Table(res));
    }
}
