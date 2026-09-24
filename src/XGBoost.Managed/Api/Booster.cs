// Public booster API, a managed replacement for the C API in src/c_api/c_api.cc with the same surface
// as the csharp-package bindings.
using XGBoost.Data;
using XGBoost.Learning;
using XGBoost.Tree;

namespace XGBoost;

/// <summary>Error raised by the library.</summary>
public class XGBoostException(string message) : Exception(message);

public enum PredictionType
{
    Value = 0,
    Margin = 1,
    Contribution = 2,
    ApproximateContribution = 3,
    Interaction = 4,
    ApproximateInteraction = 5,
    Leaf = 6,
}

public enum ModelFormat
{
    Json,
    Ubj,
}

/// <summary>Prediction values with their shape.</summary>
public sealed record Prediction(float[] Values, long[] Shape)
{
    public long Rows => Shape.Length == 0 ? 0 : Shape[0];

    public float[,] To2D()
    {
        if (Shape.Length is 0 or > 2) throw new InvalidOperationException($"Prediction has {Shape.Length} dimensions.");
        var rows = checked((int)Shape[0]);
        var cols = Shape.Length == 2 ? checked((int)Shape[1]) : 1;
        var result = new float[rows, cols];
        Buffer.BlockCopy(Values, 0, result, 0, Values.Length * sizeof(float));
        return result;
    }
}

/// <summary>Feature importance scores.</summary>
public sealed record FeatureScore(string[] Features, float[] Scores, long[] Shape);

/// <summary>A gradient boosting model.</summary>
public sealed class Booster : IDisposable
{
    internal Learner Learner { get; private set; }

    private Booster(Learner learner) => Learner = learner;

    public Booster(IEnumerable<KeyValuePair<string, string>>? parameters = null, params ReadOnlySpan<DMatrix> cache)
        : this(new Learner(cache.ToArray()))
    {
        if (parameters is not null) SetParams(parameters);
        if (cache.Length > 0)
        {
            var names = cache[0].FeatureNames;
            if (names.Length > 0) FeatureNames = names;
            var types = cache[0].FeatureTypes;
            if (types.Length > 0) FeatureTypes = types;
        }
    }

    private static Json DispatchModelType(ReadOnlySpan<byte> buffer, string ext, bool warn)
    {
        var i = 1;
        while (i < buffer.Length && char.IsWhiteSpace((char)buffer[i])) i++;
        if (i < buffer.Length && buffer[i] == (byte)'"')
        {
            if (warn) Log.Warning($"Unknown file format: `{ext}`. Using JSON (`json`) as a guess.");
            return Json.Load(buffer, false);
        }
        if (i < buffer.Length && char.IsAsciiLetter((char)buffer[i]))
        {
            if (warn) Log.Warning($"Unknown file format: `{ext}`. Using UBJSON (`ubj`) as a guess.");
            return Json.Load(buffer, true);
        }
        throw new XGBoostException($"Invalid model format. Expecting UBJSON (`ubj`) or JSON (`json`), got `{ext}`");
    }

    public static Booster Load(string path)
    {
        var booster = new Booster();
        var buffer = File.ReadAllBytes(path);
        Check.Ge(buffer.Length, 2, ErrorMsg.InvalidModel(path));
        if (buffer.Length >= 4 && buffer.AsSpan(0, 4).SequenceEqual("binf"u8))
            Check.Fail($"Loading old binary model is no longer supported: `{path}`.");
        Check.That(buffer[0] == (byte)'{', ErrorMsg.InvalidModel(path));
        var ext = Path.GetExtension(path).TrimStart('.');
        var input = ext switch
        {
            "json" => Json.Load(buffer, false),
            "ubj" => Json.Load(buffer, true),
            _ => DispatchModelType(buffer, ext, true),
        };
        booster.Learner.LoadModel(input);
        return booster;
    }

    public static Booster Load(ReadOnlySpan<byte> model)
    {
        var booster = new Booster();
        booster.Learner.LoadModel(DispatchModelType(model, "", false));
        return booster;
    }

    public static Booster Deserialize(ReadOnlySpan<byte> snapshot)
    {
        var booster = new Booster();
        booster.Learner.Load(snapshot);
        return booster;
    }

    // ---- Parameters ---------------------------------------------------------------------------

    public void SetParam(string name, string value) => Learner.Configure([new(name, value)]);

    public void SetParams(IEnumerable<KeyValuePair<string, string>> parameters) => Learner.Configure([.. parameters]);

    public string SaveConfig()
    {
        Learner.Configure();
        return Json.Dump(Learner.SaveConfig());
    }

    public void LoadConfig(string config) => Learner.LoadConfig(Json.Load(config));

    // ---- Training -----------------------------------------------------------------------------

    public void Update(DMatrix train, int iteration) => Learner.UpdateOneIter(iteration, train);

    public void Boost(DMatrix train, int iteration, ReadOnlySpan<float> gradient, ReadOnlySpan<float> hessian, int targets = 1)
    {
        if (gradient.Length != hessian.Length) throw new ArgumentException("gradient and hessian must have the same length.");
        if (targets <= 0 || gradient.Length % targets != 0)
            throw new ArgumentException("gradient length must be a multiple of targets.", nameof(targets));
        var rows = gradient.Length / targets;
        Check.Eq((long)rows, train.Info.NumRow, "Mismatched size between the gradient and training data.");
        var gpair = new GradientContainer { Gpair = new Tensor<GradientPair>([rows, targets]) };
        var h = gpair.Gpair.Data.RawArray;
        for (var i = 0; i < gradient.Length; ++i) h[i] = new GradientPair(gradient[i], hessian[i]);
        Learner.BoostOneIter(iteration, train, gpair);
    }

    public string EvaluateRaw(int iteration, params ReadOnlySpan<(DMatrix Data, string Name)> sets)
    {
        var data = new DMatrix[sets.Length];
        var names = new string[sets.Length];
        for (var i = 0; i < sets.Length; i++)
        {
            data[i] = sets[i].Data;
            names[i] = sets[i].Name;
        }
        return Learner.EvalOneIter(iteration, data, names);
    }

    public OrderedDictionary<string, double> Evaluate(int iteration, params ReadOnlySpan<(DMatrix Data, string Name)> sets) =>
        EvaluationParser.Parse(EvaluateRaw(iteration, sets));

    public int BoostedRounds
    {
        get
        {
            Learner.Configure();
            return Learner.BoostedRounds;
        }
    }

    public long NumFeatures
    {
        get
        {
            Learner.Configure();
            return Learner.NumFeature;
        }
    }

    // ---- Prediction ---------------------------------------------------------------------------

    /// <summary><c>CalcPredictShape</c> in src/c_api/c_api_utils.h.</summary>
    private static long[] CalcPredictShape(bool strictShape, PredictionType type, long rows, long cols, long chunksize, long groups,
        long rounds)
    {
        if (type == PredictionType.Margin && rows != 0) Check.Eq(chunksize, groups);
        long[] shape;
        switch (type)
        {
            case PredictionType.Value:
            case PredictionType.Margin:
                shape = chunksize == 1 && !strictShape ? [rows] : [rows, Math.Min(groups, chunksize)];
                break;
            case PredictionType.ApproximateContribution:
            case PredictionType.Contribution:
                shape = groups == 1 && !strictShape ? [rows, cols + 1] : [rows, groups, cols + 1];
                break;
            case PredictionType.ApproximateInteraction:
            case PredictionType.Interaction:
                shape = groups == 1 && !strictShape ? [rows, cols + 1, cols + 1] : [rows, groups, cols + 1, cols + 1];
                break;
            case PredictionType.Leaf:
                if (strictShape)
                {
                    var forest = chunksize / (rounds * groups);
                    forest = Math.Max(1, forest);
                    shape = [rows, rounds, groups, forest];
                }
                else
                {
                    shape = chunksize == 1 ? [rows] : [rows, chunksize];
                }
                break;
            default:
                throw new XGBoostException($"Unknown prediction type:{(int)type}");
        }
        long prod = 1;
        foreach (var s in shape) prod *= s;
        Check.Eq(prod, chunksize * rows);
        return shape;
    }

    public Prediction Predict(DMatrix data, PredictionType type = PredictionType.Value, int iterationBegin = 0, int iterationEnd = 0,
        bool training = false, bool strictShape = false)
    {
        var approximate = type is PredictionType.ApproximateContribution or PredictionType.ApproximateInteraction;
        var contribs = type is PredictionType.Contribution or PredictionType.ApproximateContribution;
        var interactions = type is PredictionType.Interaction or PredictionType.ApproximateInteraction;
        var predictions = new HostDeviceVector<float>();
        Learner.Predict(data, type == PredictionType.Margin, predictions, iterationBegin, iterationEnd, training, type == PredictionType.Leaf,
            contribs, approximate, interactions, strictShape);
        var nRows = data.Info.NumRow;
        var chunksize = nRows == 0 ? 0 : predictions.Size / nRows;
        long nRounds = iterationEnd - iterationBegin;
        nRounds = nRounds == 0 ? Learner.BoostedRounds : nRounds;
        var shape = CalcPredictShape(strictShape, type, nRows, data.Info.NumCol, chunksize, Learner.Groups, nRounds);
        return new Prediction(predictions.ToArray(), shape);
    }

    private Prediction InplaceShape(HostDeviceVector<float> predt, PredictionType type, long nRows, long nCols, bool strictShape)
    {
        var chunksize = nRows == 0 ? 0 : predt.Size / nRows;
        var shape = CalcPredictShape(strictShape, type, nRows, nCols, chunksize, Learner.Groups, Learner.BoostedRounds);
        return new Prediction(predt.ToArray(), shape);
    }

    public Prediction InplacePredict(ReadOnlySpan<float> data, int rows, int cols, float missing = float.NaN, bool margin = false,
        int iterationBegin = 0, int iterationEnd = 0, bool strictShape = false)
    {
        if ((long)rows * cols != data.Length)
            throw new ArgumentException($"Expected {(long)rows * cols} values for a {rows}x{cols} matrix, got {data.Length}.", nameof(data));
        var batch = new DenseAdapterBatch(data.ToArray(), 0, rows, cols);
        var predt = Learner.InplacePredict(batch, rows, cols, margin, missing, iterationBegin, iterationEnd, null);
        return InplaceShape(predt, margin ? PredictionType.Margin : PredictionType.Value, rows, cols, strictShape);
    }

    public Prediction InplacePredict(float[,] data, float missing = float.NaN, bool margin = false, int iterationBegin = 0,
        int iterationEnd = 0, bool strictShape = false)
    {
        var rows = data.GetLength(0);
        var cols = data.GetLength(1);
        var flat = new float[data.Length];
        Buffer.BlockCopy(data, 0, flat, 0, flat.Length * sizeof(float));
        return InplacePredict(flat, rows, cols, missing, margin, iterationBegin, iterationEnd, strictShape);
    }

    public Prediction InplacePredict(ReadOnlySpan<long> indptr, ReadOnlySpan<uint> indices, ReadOnlySpan<float> values, int numCols,
        float missing = float.NaN, bool margin = false, int iterationBegin = 0, int iterationEnd = 0, bool strictShape = false)
    {
        if (indptr.Length == 0) throw new ArgumentException("indptr must not be empty.", nameof(indptr));
        if (indices.Length != values.Length) throw new ArgumentException("indices and values must have the same length.");
        if (indptr[^1] != values.Length) throw new ArgumentException("The last indptr entry must equal the number of values.");
        var batch = new CsrAdapterBatch<long, uint, float>(indptr.ToArray(), indices.ToArray(), values.ToArray(), numCols);
        var rows = indptr.Length - 1;
        var predt = Learner.InplacePredict(batch, rows, numCols, margin, missing, iterationBegin, iterationEnd, null);
        return InplaceShape(predt, margin ? PredictionType.Margin : PredictionType.Value, rows, numCols, strictShape);
    }

    // ---- Serialization ------------------------------------------------------------------------

    public void Save(string path)
    {
        Learner.Configure();
        var ext = Path.GetExtension(path).TrimStart('.');
        byte[] bytes;
        if (ext == "json")
        {
            bytes = Json.DumpBytes(Learner.SaveModel(), false);
        }
        else
        {
            if (ext != "ubj")
                Log.Warning("Saving model in the UBJSON format as default.  You can use a file extension: `json` or `ubj` to choose between formats.");
            bytes = Json.DumpBytes(Learner.SaveModel(), true);
        }
        File.WriteAllBytes(path, bytes);
    }

    public byte[] SaveToBuffer(ModelFormat format = ModelFormat.Ubj)
    {
        Learner.Configure();
        return Json.DumpBytes(Learner.SaveModel(), format == ModelFormat.Ubj);
    }

    public byte[] Serialize()
    {
        Learner.Configure();
        return Learner.Save();
    }

    private FeatureMap GenerateFeatureMap(FeatureMap featureMap, IReadOnlyList<string>? customNames, long nFeatures)
    {
        if (featureMap.Size == 0)
        {
            var featureNames = customNames is { Count: > 0 } ? [.. customNames] : Learner.GetFeatureNames();
            if (customNames is { Count: > 0 }) Check.Eq((long)customNames.Count, nFeatures, "Incorrect number of feature names.");
            if (featureNames.Length != 0) Check.Eq((long)featureNames.Length, nFeatures, "Incorrect number of feature names.");
            var featureTypes = Learner.GetFeatureTypes();
            if (featureTypes.Length != 0) Check.Eq((long)featureTypes.Length, nFeatures, "Incorrect number of feature types.");
            for (var i = 0; i < nFeatures; ++i)
                featureMap.PushBack(i, featureNames.Length == 0 ? $"f{i}" : featureNames[i], featureTypes.Length == 0 ? "q" : featureTypes[i]);
        }
        Check.Eq((long)featureMap.Size, nFeatures);
        return featureMap;
    }

    private static FeatureMap LoadFeatureMap(string uri)
    {
        var feat = new FeatureMap();
        if (uri.Length != 0)
        {
            using var reader = new StreamReader(uri);
            feat.LoadText(reader);
        }
        return feat;
    }

    public string[] DumpModel(string format = "text", bool withStats = false, string featureMap = "")
    {
        var fmap = LoadFeatureMap(featureMap);
        Learner.Configure();
        GenerateFeatureMap(fmap, null, Learner.NumFeature);
        return [.. Learner.DumpModel(fmap, withStats, format)];
    }

    public Booster Slice(int begin, int end, int step = 1)
    {
        var sliced = Learner.Slice(begin, end, step, out var outOfBound);
        if (outOfBound) throw new IndexOutOfRangeException("Slice is out of the range of the boosted rounds.");
        return new Booster(sliced);
    }

    // ---- Attributes and feature info ----------------------------------------------------------

    public string? GetAttribute(string key) => Learner.GetAttr(key);

    public void SetAttribute(string key, string? value)
    {
        if (value is null) Learner.DelAttr(key);
        else Learner.SetAttr(key, value);
    }

    public string[] AttributeNames => Learner.GetAttrNames();

    public string[] FeatureNames { get => Learner.GetFeatureNames(); set => Learner.SetFeatureNames(value); }

    public string[] FeatureTypes { get => Learner.GetFeatureTypes(); set => Learner.SetFeatureTypes(value); }

    public FeatureScore GetScore(string importanceType = "weight")
    {
        var features = new List<uint>();
        var scores = new List<float>();
        Learner.CalcFeatureScore(importanceType, [], features, scores);
        var nFeatures = Learner.NumFeature;
        var fmap = GenerateFeatureMap(new FeatureMap(), null, nFeatures);
        var names = features.Select(f => fmap.Name((int)f)).ToArray();
        Check.Le(features.Count, scores.Count);
        long[] shape;
        if (scores.Count > features.Count)
        {
            Check.Eq(scores.Count % features.Count, 0);
            shape = [nFeatures, scores.Count / features.Count];
        }
        else
        {
            Check.Eq(features.Count, scores.Count);
            shape = [scores.Count];
        }
        return new FeatureScore(names, [.. scores], shape);
    }

    public void Dispose() { }

    private Booster() : this(new Learner([])) { }

}

/// <summary>Training helpers mirroring the Python <c>xgboost.train</c>.</summary>
public static class XGB
{
    public static Version NativeVersion => new(Constants.VersionMajor, Constants.VersionMinor, Constants.VersionPatch);

    public static string BuildInfo => Json.Dump(new JsonObject
    {
        ["USE_OPENMP"] = false,
        ["USE_CUDA"] = false,
        ["USE_NCCL"] = false,
        ["USE_RMM"] = false,
        ["USE_FEDERATED"] = false,
        ["MANAGED"] = true,
    });

    public static string GlobalConfig
    {
        get
        {
            var c = global::XGBoost.GlobalConfig.Current;
            return Json.Dump(new JsonObject
            {
                ["use_cuda_async_pool"] = c.UseCudaAsyncPool,
                ["use_rmm"] = c.UseRmm,
                ["verbosity"] = new JsonInteger(c.Verbosity),
                ["nthread"] = new JsonInteger(c.NThread),
            });
        }
        set
        {
            var args = new List<KeyValuePair<string, string>>();
            foreach (var (k, v) in Json.Load(value).AsObject)
            {
                var s = v switch
                {
                    JsonString js => js.Value,
                    JsonInteger ji => ji.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    JsonBoolean jb => jb.Value ? "1" : "0",
                    JsonNumber jn => Format.ParamFloat(jn.Value),
                    _ => throw new XGBoostException($"Invalid value for global parameter `{k}`."),
                };
                args.Add(new(k, s));
            }
            global::XGBoost.GlobalConfig.Set(args);
        }
    }

    public static Booster Train(IEnumerable<KeyValuePair<string, string>> parameters, DMatrix train, int numBoostRound,
        IReadOnlyList<(DMatrix Data, string Name)>? evals = null, int? earlyStoppingRounds = null, bool? maximize = null,
        Action<int, IReadOnlyDictionary<string, double>>? onIteration = null)
    {
        evals ??= [];
        if (earlyStoppingRounds is not null && evals.Count == 0)
            throw new ArgumentException("Early stopping requires at least one evaluation set.", nameof(evals));
        var cache = new DMatrix[evals.Count + 1];
        cache[0] = train;
        for (var i = 0; i < evals.Count; i++) cache[i + 1] = evals[i].Data;
        var booster = new Booster(parameters, cache);
        var evalArray = evals.ToArray();
        var bestScore = double.NaN;
        var bestIteration = -1;
        for (var i = 0; i < numBoostRound; i++)
        {
            booster.Update(train, i);
            if (evalArray.Length == 0) continue;
            var results = booster.Evaluate(i, evalArray);
            onIteration?.Invoke(i, results);
            if (earlyStoppingRounds is not { } patience || results.Count == 0) continue;
            var (key, score) = results.GetAt(results.Count - 1);
            var larger = maximize ?? IsMaximizeMetric(key);
            if (bestIteration < 0 || (larger ? score > bestScore : score < bestScore))
            {
                bestScore = score;
                bestIteration = i;
                booster.SetAttribute("best_iteration", bestIteration.ToString(System.Globalization.CultureInfo.InvariantCulture));
                booster.SetAttribute("best_score", bestScore.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            }
            else if (i - bestIteration >= patience)
            {
                break;
            }
        }
        return booster;
    }

    private static bool IsMaximizeMetric(string key)
    {
        var metric = key[(key.IndexOf('-') + 1)..];
        string[] maximize = ["auc", "aucpr", "pre", "pre@", "map", "ndcg", "auc@", "aucpr@", "map@", "ndcg@"];
        return maximize.Any(m => metric == m || (m.EndsWith('@') && metric.StartsWith(m, StringComparison.Ordinal)));
    }
}

public static class EvaluationParser
{
    public static OrderedDictionary<string, double> Parse(string line)
    {
        var result = new OrderedDictionary<string, double>();
        foreach (var part in line.Split('\t', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (part.StartsWith('[')) continue;
            var colon = part.LastIndexOf(':');
            if (colon <= 0) continue;
            if (double.TryParse(part.AsSpan(colon + 1), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture,
                    out var v))
                result[part[..colon]] = v;
        }
        return result;
    }
}
