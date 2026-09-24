using System.Runtime.InteropServices;
using System.Text.Json;

namespace XGBoost.Tests.Parity;

// Mirrors case.json written by generate_fixtures.py.

public class ArraySpec
{
    public string File { get; set; } = "";
    public string Dtype { get; set; } = "";
    public long[] Shape { get; set; } = [];
}

public sealed class PredictionSpec : ArraySpec
{
    public string Data { get; set; } = "";
    public string Type { get; set; } = "";
    public int IterationEnd { get; set; }
    public bool Inplace { get; set; }
}

public sealed class DatasetSpec
{
    public int Rows { get; set; }
    public int Cols { get; set; }
    public string Kind { get; set; } = "";
    public ArraySpec? Data { get; set; }
    public ArraySpec? Indptr { get; set; }
    public ArraySpec? Indices { get; set; }
    public ArraySpec? Values { get; set; }
    public Dictionary<string, ArraySpec> Info { get; set; } = [];
    public string[]? FeatureNames { get; set; }
    public string[]? FeatureTypes { get; set; }
}

public sealed class ExpectedResults
{
    public int BoostedRounds { get; set; }
    public int NumFeatures { get; set; }
    public int? BestIteration { get; set; }
    public double? BestScore { get; set; }
    public Dictionary<string, Dictionary<string, double[]>> EvalHistory { get; set; } = [];
    public Dictionary<string, PredictionSpec> Predictions { get; set; } = [];
    public Dictionary<string, Dictionary<string, double[]>> FeatureScore { get; set; } = [];
    public string[] DumpText { get; set; } = [];
    public string[] DumpJson { get; set; } = [];
    public string ModelJson { get; set; } = "";
    public string ModelUbj { get; set; } = "";
}

public sealed class Fixture
{
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public string XgboostVersion { get; set; } = "";
    public string[][] Params { get; set; } = [];
    public int NumBoostRound { get; set; }
    public int? EarlyStoppingRounds { get; set; }
    public string? CustomObjective { get; set; }
    public Dictionary<string, DatasetSpec> Datasets { get; set; } = [];
    public ExpectedResults Expected { get; set; } = new();

    public string Directory { get; private set; } = "";

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    public static string Root => Path.Combine(AppContext.BaseDirectory, "Parity", "Fixtures");

    public static IEnumerable<string> CaseNames =>
        System.IO.Directory.Exists(Root)
            ? System.IO.Directory.GetDirectories(Root).Select(Path.GetFileName).OfType<string>().Order()
            : [];

    public static Fixture Load(string name)
    {
        var dir = Path.Combine(Root, name);
        var fixture = JsonSerializer.Deserialize<Fixture>(File.ReadAllText(Path.Combine(dir, "case.json")), Options)
            ?? throw new InvalidDataException($"Empty fixture {name}.");
        fixture.Directory = dir;
        return fixture;
    }

    public string PathOf(string file) => Path.Combine(Directory, file);

    public IEnumerable<KeyValuePair<string, string>> Parameters =>
        Params.Select(p => new KeyValuePair<string, string>(p[0], p[1]));

    public T[] Read<T>(ArraySpec spec) where T : unmanaged
    {
        var expected = spec.Dtype switch
        {
            "f4" => typeof(float),
            "i8" => typeof(long),
            "u4" => typeof(uint),
            _ => throw new NotSupportedException(spec.Dtype),
        };
        if (expected != typeof(T)) throw new InvalidOperationException($"{spec.File} holds {spec.Dtype}, not {typeof(T)}.");
        return MemoryMarshal.Cast<byte, T>(File.ReadAllBytes(PathOf(spec.File))).ToArray();
    }

    /// <summary>Builds the DMatrix exactly as the Python script did.</summary>
    public DMatrix BuildDMatrix(string datasetName)
    {
        var spec = Datasets[datasetName];
        var d = spec.Kind switch
        {
            "dense" => DMatrix.FromDense(Read<float>(spec.Data!), spec.Rows, spec.Cols),
            "csr" => DMatrix.FromCsr(Read<long>(spec.Indptr!), Read<uint>(spec.Indices!), Read<float>(spec.Values!), spec.Cols),
            _ => throw new NotSupportedException(spec.Kind),
        };

        foreach (var (field, info) in spec.Info)
        {
            switch (field)
            {
                case "qid":
                    d.SetQueryId(Read<uint>(info));
                    break;
                case "label" when info.Shape.Length == 2:
                    d.SetLabel(Read<float>(info), checked((int)info.Shape[1]));
                    break;
                default:
                    d.SetInfo(field, Read<float>(info));
                    break;
            }
        }

        if (spec.FeatureNames is not null) d.FeatureNames = spec.FeatureNames;
        if (spec.FeatureTypes is not null) d.FeatureTypes = spec.FeatureTypes;
        return d;
    }
}
