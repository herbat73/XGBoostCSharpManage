using XGBoost;

const string fileFormat = "?format=libsvm";

var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
var outputModelFile = Path.Combine(dir.FullName, "model-0.json");
var dataDir = Path.Combine(dir.FullName, @"..\..\..\..\", "demos", "data");
var pathToTrainData = Path.Combine(dataDir, "agaricus.txt.train");
var train = DMatrix.FromFile($"{pathToTrainData}{fileFormat}");
var pathToTestData = Path.Combine(dataDir, "agaricus.txt.test");
var test = DMatrix.FromFile($"{pathToTestData}{fileFormat}");

(DMatrix, string)[] watchlist = [(test, "eval"), (train, "train")];

var parameters = new Dictionary<string, string>
{
    ["objective"] = "binary:logistic",
    ["max_depth"] = "2",
    ["eta"] = "0.1",
};

using var booster = XGB.Train(parameters, train, numBoostRound: 200,
    evals: watchlist, earlyStoppingRounds: 10,
    onIteration: (i, r) => Console.WriteLine($"{i}: {string.Join(", ", r)}"));

booster.Save(outputModelFile);

Console.WriteLine($"done, saved model to {outputModelFile}");

return 0;