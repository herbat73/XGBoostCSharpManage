using System.Collections.Concurrent;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;

namespace XGBoost.Tests.Parity;

/// <summary>A case trained once in C# and shared by all tests of that case.</summary>
public sealed class TrainedCase : IDisposable
{
    public required Fixture Fixture { get; init; }
    public required DMatrix Train { get; init; }
    public required DMatrix Valid { get; init; }
    public required Booster Booster { get; init; }
    public required Dictionary<string, List<double>> History { get; init; }

    public DMatrix Data(string name) => name == "train" ? Train : Valid;

    public void Dispose()
    {
        Booster.Dispose();
        Train.Dispose();
        Valid.Dispose();
    }
}

/// <summary>Trains each fixture case in C# at most once per test run.</summary>
public sealed class ParityCache : IDisposable
{
    private readonly ConcurrentDictionary<string, Lazy<TrainedCase>> _cases = new();

    public TrainedCase Get(string name) => _cases.GetOrAdd(name, n => new Lazy<TrainedCase>(() => TrainCase(n))).Value;

    private static TrainedCase TrainCase(string name)
    {
        var fixture = Fixture.Load(name);
        var train = fixture.BuildDMatrix("train");
        var valid = fixture.BuildDMatrix("valid");
        var history = new Dictionary<string, List<double>>();

        void Record(IReadOnlyDictionary<string, double> results)
        {
            foreach (var (key, value) in results)
            {
                if (!history.TryGetValue(key, out var list)) history[key] = list = [];
                list.Add(value);
            }
        }

        (DMatrix, string)[] evals = [(train, "train"), (valid, "valid")];
        Booster booster;
        if (fixture.CustomObjective is null)
        {
            booster = XGB.Train(fixture.Parameters, train, fixture.NumBoostRound, evals,
                earlyStoppingRounds: fixture.EarlyStoppingRounds, onIteration: (_, r) => Record(r));
        }
        else
        {
            // Same sequence as Python's xgb.train(obj=...): margin with training=True, gradients, boost, evaluate.
            booster = new Booster(fixture.Parameters, train, train, valid);
            var label = train.Label;
            var grad = new float[label.Length];
            var hess = new float[label.Length];
            for (var i = 0; i < fixture.NumBoostRound; i++)
            {
                var margin = booster.Predict(train, PredictionType.Margin, training: true).Values;
                CustomObjectives.Compute(fixture.CustomObjective, margin, label, grad, hess);
                booster.Boost(train, i, grad, hess);
                Record(booster.Evaluate(i, evals));
            }
        }

        return new TrainedCase { Fixture = fixture, Train = train, Valid = valid, Booster = booster, History = history };
    }

    public void Dispose()
    {
        foreach (var lazy in _cases.Values)
            if (lazy.IsValueCreated) lazy.Value.Dispose();
    }
}

/// <summary>C# ports of the custom objectives in generate_fixtures.py, in float32 like the Python versions.</summary>
internal static class CustomObjectives
{
    public static void Compute(string name, float[] margin, float[] label, float[] grad, float[] hess)
    {
        switch (name)
        {
            case "pseudo_huber":
                for (var i = 0; i < margin.Length; i++)
                {
                    var d = margin[i] - label[i];
                    var scale = 1f + d * d;
                    var sq = MathF.Sqrt(scale);
                    grad[i] = d / sq;
                    hess[i] = 1f / (scale * sq);
                }
                break;
            default:
                throw new NotSupportedException($"Unknown custom objective {name}.");
        }
    }
}

/// <summary>
/// Checks that the C# bindings reproduce the Python package's results. Fixtures come from
/// <c>Parity/generate_fixtures.py</c>, run against the same native library.
/// </summary>
public class PythonParityTests(ParityCache cache) : IClassFixture<ParityCache>
{
    public static TheoryData<string> Cases => new(Fixture.CaseNames);

    [Fact]
    public void FixturesArePresent() =>
        Assert.True(Fixture.CaseNames.Count() >= 10, $"Parity fixtures missing from {Fixture.Root}; run generate_fixtures.py.");

    [Theory]
    [MemberData(nameof(Cases))]
    public void NativeVersionMatchesFixture(string name)
    {
        var version = Fixture.Load(name).XgboostVersion;
        var numeric = version.Split('-')[0];
        Assert.True(Version.Parse(numeric) == XGB.NativeVersion,
            $"Fixture was generated with xgboost {version} but the native library is {XGB.NativeVersion}; regenerate fixtures.");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void TrainingRoundsMatch(string name)
    {
        var c = cache.Get(name);
        var e = c.Fixture.Expected;
        Assert.Equal(e.BoostedRounds, c.Booster.BoostedRounds);
        Assert.Equal(e.NumFeatures, c.Booster.NumFeatures);

        var bestIteration = c.Booster.GetAttribute("best_iteration");
        Assert.Equal(e.BestIteration, bestIteration is null ? null : int.Parse(bestIteration, CultureInfo.InvariantCulture));
        var bestScore = c.Booster.GetAttribute("best_score");
        if (e.BestScore is { } expectedScore)
            Assert.True(Compare.Close(double.Parse(bestScore!, CultureInfo.InvariantCulture), expectedScore),
                $"best_score: expected {expectedScore}, got {bestScore}.");
        else
            Assert.Null(bestScore);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EvaluationHistoryMatches(string name)
    {
        var c = cache.Get(name);
        var expected = c.Fixture.Expected.EvalHistory
            .SelectMany(set => set.Value.Select(m => ($"{set.Key}-{m.Key}", m.Value)))
            .ToDictionary();
        Assert.Equal(expected.Keys.Order(), c.History.Keys.Order());
        foreach (var (key, values) in expected)
            Compare.Values(values, c.History[key], $"{name} {key}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PredictionsFromCSharpTrainedModelMatch(string name)
    {
        var c = cache.Get(name);
        AssertPredictions(c, c.Booster, "C#-trained");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PythonJsonModelPredictsIdentically(string name)
    {
        var c = cache.Get(name);
        using var booster = Booster.Load(c.Fixture.PathOf(c.Fixture.Expected.ModelJson));
        AssertPredictions(c, booster, "Python JSON model");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PythonUbjModelPredictsIdentically(string name)
    {
        var c = cache.Get(name);
        using var booster = Booster.Load(File.ReadAllBytes(c.Fixture.PathOf(c.Fixture.Expected.ModelUbj)));
        AssertPredictions(c, booster, "Python UBJ model");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void SavedModelJsonMatchesPython(string name)
    {
        var c = cache.Get(name);
        var expected = JsonNode.Parse(File.ReadAllText(c.Fixture.PathOf(c.Fixture.Expected.ModelJson)));
        var actual = JsonNode.Parse(Encoding.UTF8.GetString(c.Booster.SaveToBuffer(ModelFormat.Json)));

        // Attributes hold best_score, whose string form depends on the language; checked numerically above.
        Compare.Json(expected, actual, "$", new HashSet<string> { "$.learner.attributes" });
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ModelDumpsMatch(string name)
    {
        var c = cache.Get(name);
        var e = c.Fixture.Expected;
        AssertDump(e.DumpText, c.Booster.DumpModel("text", withStats: true), $"{name} text dump");
        AssertDump(e.DumpJson, c.Booster.DumpModel("json", withStats: true), $"{name} json dump");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FeatureScoresMatch(string name)
    {
        var c = cache.Get(name);
        foreach (var (importance, expected) in c.Fixture.Expected.FeatureScore)
        {
            var score = c.Booster.GetScore(importance);
            var perFeature = score.Features.Length == 0 ? 0 : score.Scores.Length / score.Features.Length;
            var actual = score.Features
                .Select((f, i) => (f, score.Scores.Skip(i * perFeature).Take(perFeature).Select(v => (double)v).ToArray()))
                .ToDictionary();
            Assert.Equal(expected.Keys.Order(), actual.Keys.Order());
            foreach (var (feature, values) in expected)
                Compare.Values(values, actual[feature], $"{name} {importance}[{feature}]");
        }
    }

    private static void AssertDump(string[] expected, string[] actual, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: expected {expected.Length} trees, got {actual.Length}.");
        for (var i = 0; i < expected.Length; i++) Compare.Text(expected[i], actual[i], $"{what} tree {i}");
    }

    private static void AssertPredictions(TrainedCase c, Booster booster, string source)
    {
        foreach (var (predName, spec) in c.Fixture.Expected.Predictions)
        {
            var what = $"{c.Fixture.Name} {predName} ({source})";
            var actual = spec.Inplace ? PredictInplace(c, booster, spec) : booster.Predict(c.Data(spec.Data),
                ParseType(spec.Type), iterationEnd: spec.IterationEnd);
            Assert.True(spec.Shape.SequenceEqual(actual.Shape),
                $"{what}: expected shape [{string.Join(",", spec.Shape)}], got [{string.Join(",", actual.Shape)}].");
            Compare.Values(c.Fixture.Read<float>(spec), actual.Values, what);
        }
    }

    private static Prediction PredictInplace(TrainedCase c, Booster booster, PredictionSpec spec)
    {
        var margin = ParseType(spec.Type) == PredictionType.Margin;
        var data = c.Fixture.Datasets[spec.Data];
        return data.Kind == "dense"
            ? booster.InplacePredict(c.Fixture.Read<float>(data.Data!), data.Rows, data.Cols, margin: margin,
                iterationEnd: spec.IterationEnd)
            : booster.InplacePredict(c.Fixture.Read<long>(data.Indptr!), c.Fixture.Read<uint>(data.Indices!),
                c.Fixture.Read<float>(data.Values!), data.Cols, margin: margin, iterationEnd: spec.IterationEnd);
    }

    private static PredictionType ParseType(string type) => type switch
    {
        "value" => PredictionType.Value,
        "margin" => PredictionType.Margin,
        "contribution" => PredictionType.Contribution,
        "approximate_contribution" => PredictionType.ApproximateContribution,
        "interaction" => PredictionType.Interaction,
        "leaf" => PredictionType.Leaf,
        _ => throw new NotSupportedException(type),
    };
}
