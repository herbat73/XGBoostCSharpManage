namespace XGBoost.Tests;

public class BoosterTests
{
    private const int Cols = 4;

    private static Booster TrainRegression(DMatrix train, int rounds = 20) =>
        XGB.Train(TestData.Params(("objective", "reg:squarederror"), ("max_depth", "3")), train, rounds);

    [Fact]
    public void RegressionLearns()
    {
        var (x, y) = TestData.Regression(500);
        using var train = TestData.ToDMatrix(x, y, Cols);

        var history = new List<double>();
        using var booster = XGB.Train(TestData.Params(("objective", "reg:squarederror"), ("eval_metric", "rmse")),
            train, 30, [(train, "train")], onIteration: (_, r) => history.Add(r["train-rmse"]));

        Assert.Equal(30, booster.BoostedRounds);
        Assert.Equal(Cols, booster.NumFeatures);
        Assert.Equal(30, history.Count);
        Assert.True(history[^1] < history[0] / 5, $"rmse {history[0]} -> {history[^1]}");

        var pred = booster.Predict(train);
        Assert.Equal([500L], pred.Shape);
        var rmse = Math.Sqrt(pred.Values.Zip(y, (p, t) => (p - t) * (p - t)).Average());
        Assert.Equal(history[^1], rmse, 3);
    }

    [Fact]
    public void InplacePredictMatchesDMatrix()
    {
        var (x, y) = TestData.Regression(200);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var booster = TrainRegression(train);

        var viaDMatrix = booster.Predict(train).Values;
        Assert.Equal(viaDMatrix, booster.InplacePredict(x, 200, Cols).Values);

        // Same data as CSR.
        var indptr = Enumerable.Range(0, 201).Select(i => (long)i * Cols).ToArray();
        var indices = Enumerable.Range(0, 200 * Cols).Select(i => (uint)(i % Cols)).ToArray();
        Assert.Equal(viaDMatrix, booster.InplacePredict(indptr, indices, x, Cols).Values);
    }

    [Fact]
    public void BinaryProbabilitiesAndMargin()
    {
        var (x, y) = TestData.Binary(300);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var booster = XGB.Train(TestData.Params(("objective", "binary:logistic")), train, 20);

        var prob = booster.Predict(train).Values;
        var margin = booster.Predict(train, PredictionType.Margin).Values;
        Assert.All(prob, p => Assert.InRange(p, 0f, 1f));
        for (var i = 0; i < prob.Length; i++)
            Assert.Equal(1 / (1 + Math.Exp(-margin[i])), prob[i], 5);

        var accuracy = prob.Zip(y, (p, t) => (p > 0.5 ? 1 : 0) == t).Count(ok => ok) / (double)y.Length;
        Assert.True(accuracy > 0.95, $"accuracy {accuracy}");
    }

    [Fact]
    public void MultiClassSoftprobShape()
    {
        var (x, y) = TestData.MultiClass(300, classes: 3);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var booster = XGB.Train(TestData.Params(("objective", "multi:softprob"), ("num_class", "3")), train, 10);

        var pred = booster.Predict(train);
        Assert.Equal([300L, 3], pred.Shape);
        var p2 = pred.To2D();
        for (var r = 0; r < 300; r++)
            Assert.Equal(1.0, p2[r, 0] + p2[r, 1] + p2[r, 2], 4);
    }

    [Fact]
    public void ContributionsSumToMargin()
    {
        var (x, y) = TestData.Regression(50);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var booster = TrainRegression(train);

        var contrib = booster.Predict(train, PredictionType.Contribution);
        Assert.Equal([50L, Cols + 1], contrib.Shape);
        var margin = booster.Predict(train, PredictionType.Margin).Values;
        var c = contrib.To2D();
        for (var r = 0; r < 50; r++)
        {
            double sum = 0;
            for (var f = 0; f <= Cols; f++) sum += c[r, f];
            Assert.Equal(margin[r], sum, 4);
        }

        var leaves = booster.Predict(train, PredictionType.Leaf);
        Assert.Equal([50L, 20], leaves.Shape);
    }

    [Theory]
    [InlineData(ModelFormat.Json)]
    [InlineData(ModelFormat.Ubj)]
    public void BufferRoundTrip(ModelFormat format)
    {
        var (x, y) = TestData.Regression(100);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var booster = TrainRegression(train);
        booster.SetAttribute("note", "hello");

        var bytes = booster.SaveToBuffer(format);
        if (format == ModelFormat.Json) Assert.Equal((byte)'{', bytes[0]);

        using var loaded = Booster.Load(bytes);
        Assert.Equal(booster.Predict(train).Values, loaded.Predict(train).Values);
        Assert.Equal("hello", loaded.GetAttribute("note"));
    }

    [Fact]
    public void FileAndSnapshotRoundTrip()
    {
        var (x, y) = TestData.Regression(100);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var booster = TrainRegression(train);
        var expected = booster.Predict(train).Values;

        var path = Path.Combine(Path.GetTempPath(), $"xgb-{Guid.NewGuid():N}.json");
        try
        {
            booster.Save(path);
            using var fromFile = Booster.Load(path);
            Assert.Equal(expected, fromFile.Predict(train).Values);
        }
        finally
        {
            File.Delete(path);
        }

        using var restored = Booster.Deserialize(booster.Serialize());
        Assert.Equal(expected, restored.Predict(train).Values);
        Assert.Contains("reg:squarederror", restored.SaveConfig());
    }

    [Fact]
    public void ContinueTrainingFromLoadedModel()
    {
        var (x, y) = TestData.Regression(100);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var booster = TrainRegression(train, rounds: 5);

        using var resumed = Booster.Load(booster.SaveToBuffer());
        for (var i = 0; i < 5; i++) resumed.Update(train, i);
        Assert.Equal(10, resumed.BoostedRounds);
    }

    [Fact]
    public void CustomObjectiveMatchesBuiltIn()
    {
        var (x, y) = TestData.Regression(200);
        using var train = TestData.ToDMatrix(x, y, Cols);
        var p = TestData.Params(("max_depth", "3"), ("base_score", "0.5"));

        using var builtIn = new Booster(p.Append(new("objective", "reg:squarederror")), train);
        using var custom = new Booster(p, train);
        var grad = new float[y.Length];
        var hess = new float[y.Length];
        for (var i = 0; i < 10; i++)
        {
            builtIn.Update(train, i);
            var margin = custom.Predict(train, PredictionType.Margin, training: true).Values;
            for (var r = 0; r < y.Length; r++) { grad[r] = margin[r] - y[r]; hess[r] = 1; }
            custom.Boost(train, i, grad, hess);
        }

        var a = builtIn.Predict(train, PredictionType.Margin).Values;
        var b = custom.Predict(train, PredictionType.Margin).Values;
        for (var r = 0; r < a.Length; r++) Assert.Equal(a[r], b[r], 4);
    }

    [Fact]
    public void EarlyStopping()
    {
        var (x, y) = TestData.Regression(300, seed: 3);
        var (vx, vy) = TestData.Regression(100, seed: 4);
        using var train = TestData.ToDMatrix(x, y, Cols);
        using var valid = TestData.ToDMatrix(vx, vy, Cols);

        // A large learning rate with deep trees overfits quickly, so validation rmse stops improving.
        using var booster = XGB.Train(TestData.Params(("eta", "1"), ("max_depth", "8")), train, 500,
            [(train, "train"), (valid, "valid")], earlyStoppingRounds: 5);

        var best = int.Parse(booster.GetAttribute("best_iteration")!);
        Assert.True(booster.BoostedRounds < 500);
        Assert.Equal(best + 6, booster.BoostedRounds);
    }

    [Fact]
    public void AttributesDumpScoreSlice()
    {
        var (x, y) = TestData.Regression(200);
        using var train = TestData.ToDMatrix(x, y, Cols);
        train.FeatureNames = ["a", "b", "c", "d"];
        using var booster = TrainRegression(train, rounds: 10);

        booster.SetAttribute("k1", "v1");
        Assert.Contains("k1", booster.AttributeNames);
        booster.SetAttribute("k1", null);
        Assert.Null(booster.GetAttribute("k1"));

        Assert.Equal(["a", "b", "c", "d"], booster.FeatureNames);
        Assert.Equal(10, booster.DumpModel().Length);
        Assert.StartsWith("{", booster.DumpModel("json", withStats: true)[0].TrimStart());

        var score = booster.GetScore("gain");
        Assert.Contains("a", score.Features);
        Assert.Equal(score.Features.Length, score.Scores.Length);
        // x0 has the largest coefficient.
        Assert.Equal("a", score.Features[Array.IndexOf(score.Scores, score.Scores.Max())]);

        using var first3 = booster.Slice(0, 3);
        Assert.Equal(3, first3.BoostedRounds);
        Assert.Equal(booster.Predict(train, iterationEnd: 3).Values, first3.Predict(train).Values);
    }

    [Fact]
    public void InvalidParameterThrows()
    {
        var (x, y) = TestData.Regression(20);
        using var train = TestData.ToDMatrix(x, y, Cols);
        // Parameters are applied immediately, so the error surfaces at construction.
        var ex = Assert.Throws<XGBoostException>(() => new Booster(TestData.Params(("objective", "not-an-objective")), train));
        Assert.Contains("not-an-objective", ex.Message);
    }

    [Fact]
    public void EvaluationParserReadsLine()
    {
        var r = EvaluationParser.Parse("[7]\ttrain-rmse:0.25\tvalid-rmse:0.5\tvalid-mae:0.125");
        Assert.Equal(["train-rmse", "valid-rmse", "valid-mae"], r.Keys);
        Assert.Equal(0.125, r["valid-mae"]);
    }
}
