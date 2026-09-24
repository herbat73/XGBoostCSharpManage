// Port of src/learner.cc (LearnerModelStateContainer, LearnerConfiguration, LearnerIO, LearnerImpl).
using System.Globalization;
using System.Text;
using XGBoost.Collective;
using XGBoost.Gbm;
using XGBoost.Tree;

namespace XGBoost.Learning;

public sealed class LearnerTrainParam : XGBoostParameter<LearnerTrainParam>
{
    public float[] BaseScore = [];
    public int NumClass;
    public uint NumTarget = 1;
    public int BoostFromAverage = 1;
    public bool DisableDefaultEvalMetric;
    public string Booster = "gbtree";
    public string Objective = "reg:squarederror";
    public MultiStrategy MultiStrategy = MultiStrategy.OneOutputPerTree;

    protected override void Declare(ParamManager<LearnerTrainParam> m)
    {
        CustomField(m, "base_score", "ParamArray<float>", p => p.BaseScore, (p, v) => p.BaseScore = v,
                v => ParamArray.Parse("base_score", v), ParamArray.Print)
            .SetDefault([]).Describe("Global bias of the model.");
        Field(m, "num_class", p => p.NumClass, (p, v) => p.NumClass = v).SetDefault(0).SetLowerBound(0)
            .Describe("Number of class option for multi-class classifier.  By default equals 0 and corresponds to binary classifier.");
        Field(m, "num_target", p => p.NumTarget, (p, v) => p.NumTarget = v).SetDefault(1u).SetLowerBound(1u)
            .Describe("Number of output targets. Can be set automatically if not specified.");
        CustomField(m, "boost_from_average", "int", p => p.BoostFromAverage, (p, v) => p.BoostFromAverage = v, ParseBoolInt,
                v => v.ToString(CultureInfo.InvariantCulture))
            .SetDefault(1).Describe("Whether we should calculate the base score from training data.");
        Field(m, "disable_default_eval_metric", p => p.DisableDefaultEvalMetric, (p, v) => p.DisableDefaultEvalMetric = v).SetDefault(false)
            .Describe("Flag to disable default metric. Set to >0 to disable");
        Field(m, "booster", p => p.Booster, (p, v) => p.Booster = v).SetDefault("gbtree").Describe("Gradient booster used for training.");
        Field(m, "objective", p => p.Objective, (p, v) => p.Objective = v).SetDefault("reg:squarederror")
            .Describe("Objective function used for obtaining gradient.");
        EnumField<MultiStrategy>(m, "multi_strategy", p => p.MultiStrategy, (p, v) => p.MultiStrategy = v)
            .AddEnum("one_output_per_tree", MultiStrategy.OneOutputPerTree).AddEnum("multi_output_tree", MultiStrategy.MultiOutputTree)
            .SetDefault(MultiStrategy.OneOutputPerTree)
            .Describe("Strategy used for training multi-target models. `multi_output_tree` means building one single tree for all targets.");
    }

    /// <summary>dmlc parses an int field that is initialised from a bool default: accepts true/false too.</summary>
    private static int ParseBoolInt(string v) => v switch
    {
        "true" or "True" => 1,
        "false" or "False" => 0,
        _ => int.TryParse(v.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var i)
            ? i
            : throw new XGBoostException($"Invalid Parameter format for boost_from_average expect int but value='{v}'"),
    };

    public override List<KeyValuePair<string, string>> UpdateAllowUnknown(IEnumerable<KeyValuePair<string, string>> kwargs)
    {
        var list = kwargs.ToList();
        var unknown = base.UpdateAllowUnknown(list);
        if (list.Any(kv => kv.Key == "base_score") && !list.Any(kv => kv.Key == "boost_from_average")) BoostFromAverage = 0;
        return unknown;
    }

    public LearnerTrainParam Clone()
    {
        var c = (LearnerTrainParam)MemberwiseClone();
        c.BaseScore = (float[])BaseScore.Clone();
        return c;
    }
}

/// <summary>The learner: configuration, training, evaluation, prediction and serialization.</summary>
public sealed class Learner
{
    private const string EvalMetricKey = "eval_metric";
    private const int RandSeedMagic = 127;

    private LearnerTrainParam _tparam = new();
    private readonly LearnerModelState _modelState = new();
    private Context _ctx = new();
    private ObjFunction? _obj;
    private GradientBooster? _gbm;
    private readonly List<Metric> _metrics = [];
    private bool _needConfiguration = true;
    private readonly SortedDictionary<string, string> _attributes = new(StringComparer.Ordinal);
    private List<string> _featureNames = [];
    private List<string> _featureTypes = [];
    private readonly List<WeakReference<DMatrix>> _cacheData = [];
    private readonly List<string> _metricNames = [];
    private readonly GradientContainer _gpair = new();
    private readonly object _configLock = new();

    public Learner(IEnumerable<DMatrix?> cache)
    {
        foreach (var d in cache)
            if (d is not null) _cacheData.Add(new WeakReference<DMatrix>(d));
    }

    public Context Ctx => _ctx;
    public LearnerModelState ModelState => _modelState;
    internal GradientBooster Gbm => _gbm!;
    internal ObjFunction Obj => _obj!;

    // ---- model state ---------------------------------------------------------------------------

    private static string CanonicalizeBoosterName(string booster)
    {
        if (booster == "dart")
        {
            Log.WarningOnce("dart-deprecated",
                "`booster=dart` is deprecated. Use the tree booster directly with dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.");
            return "gbtree";
        }
        return booster;
    }

    private static void HandleOldFormat(ref float[] baseScore, uint outputLength)
    {
        if (baseScore.Length == 1 && outputLength > 1) baseScore = Enumerable.Repeat(baseScore[0], (int)outputLength).ToArray();
    }

    private JsonObject ModelStateToJson()
    {
        var init = _modelState.Initialized;
        var nFeatures = init ? _modelState.NumFeature : 0;
        var nClasses = init ? _modelState.NumClass : _tparam.NumClass;
        var nTargets = init ? _modelState.NumTarget : _tparam.NumTarget;
        var boostFromAverage = init ? (_modelState.BoostFromAverage ? 1 : 0) : _tparam.BoostFromAverage;
        var baseScore = init ? _modelState.BaseScoreValue : _tparam.BaseScore;
        return new JsonObject
        {
            ["base_score"] = ParamArray.Print(baseScore),
            ["num_feature"] = nFeatures.ToString(CultureInfo.InvariantCulture),
            ["num_class"] = nClasses.ToString(CultureInfo.InvariantCulture),
            ["num_target"] = nTargets.ToString(CultureInfo.InvariantCulture),
            ["boost_from_average"] = (boostFromAverage != 0 ? 1 : 0).ToString(CultureInfo.InvariantCulture),
        };
    }

    private void ValidateModelState()
    {
        Check.That(_modelState.Initialized, "Model is not yet initialized (not fitted).");
        var baseScore = _modelState.BaseScoreValue;
        Check.Ge(baseScore.Length, 1);
        if (!(baseScore.Length == _modelState.NumClass || baseScore.Length == _modelState.NumTarget))
            ErrorMsg.InvalidIntercept(_modelState.NumClass, _modelState.NumTarget, baseScore.Length);
        Check.That(!baseScore.Any(v => float.IsNaN(v) || float.IsInfinity(v)));
    }

    private void InitModelState(uint nFeatures, int nClasses, uint nTargets, bool boostFromAverage, float[] baseScoreValue)
    {
        var outputLength = Math.Max(Math.Max(nTargets, (uint)nClasses), 1u);
        HandleOldFormat(ref baseScoreValue, outputLength);
        var baseScore = new Tensor<float>((float[])baseScoreValue.Clone(), [baseScoreValue.Length]);
        _obj!.ProbToMargin(baseScore);
        _modelState.Copy(new LearnerModelState(nFeatures, nClasses, nTargets, boostFromAverage, baseScoreValue, baseScore, _obj.Task,
            _tparam.MultiStrategy));
        ValidateModelState();
    }

    private static uint InitNumFeatures(DMatrix train)
    {
        var nFeatures = train.Info.NumCol;
        ErrorMsg.MaxFeatureSize((ulong)nFeatures);
        Check.Ne(nFeatures, 0L, "0 feature is supplied. Are you using the raw Booster interface?");
        return (uint)nFeatures;
    }

    private uint InitNumTargets(DMatrix train)
    {
        Check.That(_obj is not null);
        var localNTargets = _obj!.Targets(train.Info);
        var nTargets = localNTargets;
        if (_modelState.Initialized)
        {
            Check.That(nTargets == 1 || nTargets == _modelState.NumTarget, "Inconsistent number of targets between data and model.");
            nTargets = _modelState.NumTarget;
        }
        else if (_tparam.NumTarget > 1)
        {
            Check.That(nTargets == 1 || nTargets == _tparam.NumTarget,
                $"Inconsistent configuration of the `num_target`.  Configuration result from input data:{nTargets}, configuration from parameters:{_tparam.NumTarget}");
            nTargets = _tparam.NumTarget;
        }
        foreach (var weak in _cacheData)
        {
            if (!weak.TryGetTarget(out var d)) continue;
            var t = _obj.Targets(d.Info);
            Check.That(nTargets == t || t == 1, "Inconsistent labels.");
        }
        if (train.Info.NumRow == 0 && localNTargets != nTargets)
        {
            Check.Eq(train.Info.Labels.Size, 0);
            train.Info.Labels.Reshape(0, nTargets);
        }
        return nTargets;
    }

    private void CheckModelInitialized() => Check.That(_modelState.Initialized, "Model is not yet initialized (not fitted).");

    private void ConfigureModelState(LearnerTrainParam oldTparam, Args args)
    {
        if (_modelState.NeedsInitialization) return;
        bool Has(string key) => args.Any(kv => kv.Key == key);
        var modelInputChanged = Has("base_score") || Has("num_class") || Has("num_target") || Has("boost_from_average");
        var structureChanged = oldTparam.Objective != _tparam.Objective || oldTparam.MultiStrategy != _tparam.MultiStrategy;
        if (!modelInputChanged && !structureChanged) return;
        var baseScore = (float[])_modelState.BaseScoreValue.Clone();
        var nClasses = _modelState.NumClass;
        var nTargets = _modelState.NumTarget;
        var boostFromAverage = _modelState.BoostFromAverage;
        if (Has("base_score")) baseScore = (float[])_tparam.BaseScore.Clone();
        if (Has("num_class")) nClasses = _tparam.NumClass;
        if (Has("num_target")) nTargets = _tparam.NumTarget;
        if (Has("boost_from_average") || Has("base_score")) boostFromAverage = _tparam.BoostFromAverage != 0;
        InitModelState(_modelState.NumFeature, nClasses, nTargets, boostFromAverage, baseScore);
    }

    private enum InterceptInitialization { EstimateIntercept, UseDefaultIntercept }

    private void InitializeModel(DMatrix train, InterceptInitialization mode)
    {
        var nFeatures = InitNumFeatures(train);
        var nTargets = InitNumTargets(train);
        if (_modelState.Initialized) return;
        var outputLength = Math.Max(Math.Max(nTargets, (uint)_tparam.NumClass), 1u);
        float[] baseScore;
        if (_tparam.BoostFromAverage == 0)
        {
            baseScore = (float[])_tparam.BaseScore.Clone();
        }
        else if (mode == InterceptInitialization.EstimateIntercept)
        {
            var info = train.Info;
            info.Validate();
            var estimated = new Tensor<float>([outputLength]);
            _obj!.InitEstimation(info, estimated);
            baseScore = estimated.Data.ToArray();
        }
        else
        {
            baseScore = Enumerable.Repeat(ObjFunction.DefaultBaseScore, (int)outputLength).ToArray();
        }
        InitModelState(nFeatures, _tparam.NumClass, nTargets, _tparam.BoostFromAverage != 0, baseScore);
    }

    private void LoadModelState(Json input)
    {
        var values = input.AsObject;
        var baseScore = ParamArray.Parse("base_score", values["base_score"].AsString);
        var nFeaturesValue = ulong.Parse(values["num_feature"].AsString, CultureInfo.InvariantCulture);
        ErrorMsg.MaxFeatureSize(nFeaturesValue);
        var nFeatures = (uint)nFeaturesValue;
        var nClasses = int.Parse(values["num_class"].AsString, CultureInfo.InvariantCulture);
        Check.Ge(nClasses, 0);
        uint nTargets = 1;
        if (values.TryGetValue("num_target", out var nt))
        {
            var value = ulong.Parse(nt.AsString, CultureInfo.InvariantCulture);
            Check.Le(value, (ulong)uint.MaxValue);
            nTargets = (uint)value;
            Check.Ge(nTargets, 1u);
        }
        var boostFromAverage = true;
        if (values.TryGetValue("boost_from_average", out var bfa)) boostFromAverage = int.Parse(bfa.AsString, CultureInfo.InvariantCulture) != 0;
        _tparam.BaseScore = baseScore;
        _tparam.NumClass = nClasses;
        _tparam.NumTarget = nTargets;
        _tparam.BoostFromAverage = boostFromAverage ? 1 : 0;
        if (nFeatures == 0)
        {
            _modelState.Copy(new LearnerModelState());
            return;
        }
        InitModelState(nFeatures, nClasses, nTargets, boostFromAverage, baseScore);
    }

    // ---- configuration -------------------------------------------------------------------------

    public void Configure(Args? args = null)
    {
        args ??= [];
        var hasArgs = args.Count != 0;
        if (hasArgs) _needConfiguration = true;
        if (!hasArgs && !_needConfiguration) return;
        lock (_configLock)
        {
            var oldTparam = _tparam.Clone();
            var config = new SortedDictionary<string, string>(StringComparer.Ordinal);
            var used = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var (key, value) in args)
            {
                if (key == EvalMetricKey)
                {
                    used.Add(EvalMetricKey);
                    if (!_metricNames.Contains(value)) _metricNames.Add(value);
                }
                else
                {
                    config[key] = value;
                }
            }
            Args configArgs = [.. config];
            used.UnionWith(ParameterUtils.GetUsedParameters(configArgs, _tparam.UpdateAllowUnknown(configArgs)));
            var initialized = _ctx.GetInitialised();
            var oldSeed = _ctx.Seed;
            used.UnionWith(ParameterUtils.GetUsedParameters(configArgs, _ctx.UpdateAllowUnknown(configArgs)));
            used.UnionWith(GlobalConfig.ConfigureLogger(configArgs));
            if (!initialized || _ctx.Seed != oldSeed) _ctx.Rng.Seed(_ctx.Seed);
            used.UnionWith(ConfigureObjective(oldTparam, config, out configArgs));
            _modelState.Task = _obj!.Task;
            used.UnionWith(ConfigureGBM(oldTparam, configArgs));
            ConfigureModelState(oldTparam, args);
            used.UnionWith(ConfigureMetrics(configArgs));
            _needConfiguration = false;
            if (_ctx.ValidateParameters) ValidateParameters(args, used);
        }
    }

    private static void ValidateParameters(Args args, SortedSet<string> used)
    {
        var provided = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var kv in args)
        {
            if (kv.Key.Any(char.IsWhiteSpace)) Check.Fail($"Invalid parameter \"{kv.Key}\" contains whitespace.");
            provided.Add(kv.Key);
        }
        var diff = provided.Where(p => !used.Contains(p)).ToList();
        if (diff.Count != 0)
        {
            var sb = new StringBuilder("\nParameters: { ");
            for (var i = 0; i < diff.Count - 1; ++i) sb.Append('"').Append(diff[i]).Append("\", ");
            sb.Append('"').Append(diff[^1]).Append('"').Append(" } are not used.\n");
            Log.Warning(sb.ToString());
        }
    }

    private SortedSet<string> ConfigureGBM(LearnerTrainParam old, Args args)
    {
        _tparam.Booster = CanonicalizeBoosterName(_tparam.Booster);
        if (_tparam.Booster == "gblinear")
            Log.Warning("`booster=gblinear` is deprecated and support will be removed in a future release.");
        var oldBooster = CanonicalizeBoosterName(old.Booster);
        if (_gbm is null || oldBooster != _tparam.Booster) _gbm = Registry.CreateBooster(_tparam.Booster, _ctx, _modelState);
        return _gbm.Configure(args);
    }

    private SortedSet<string> ConfigureObjective(LearnerTrainParam old, SortedDictionary<string, string> config, out Args args)
    {
        if (config.TryGetValue("num_class", out var nc) && nc != "0" && _tparam.Objective != "multi:softprob")
        {
            config["num_output_group"] = nc;
            if (Format.TryParseIntegerStream(nc, out var n) && n > 1 && !config.ContainsKey("objective")) _tparam.Objective = "multi:softmax";
        }
        if (_obj is null || _tparam.Objective != old.Objective) _obj = Registry.CreateObjective(_tparam.Objective, _ctx);
        var hasNc = config.ContainsKey("num_class");
        config["num_class"] = _tparam.NumClass.ToString(CultureInfo.InvariantCulture);
        args = [.. config];
        var used = _obj.Configure(args);
        if (!hasNc) config.Remove("num_class");
        args = [.. config];
        return used;
    }

    private SortedSet<string> ConfigureMetrics(Args args)
    {
        var used = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var name in _metricNames)
            if (_metrics.All(m => m.Name != name)) _metrics.Add(Registry.CreateMetric(name, _ctx));
        foreach (var m in _metrics) used.UnionWith(m.Configure(args));
        return used;
    }

    private void LoadConfigImpl(Json input)
    {
        Check.That(input is JsonObject);
        var originVersion = VersionInfo.Load(input);
        if (originVersion.Major == VersionInfo.Invalid.Major) Log.Warning("Invalid version string in config");
        if (originVersion != VersionInfo.Self())
        {
            ErrorMsg.WarnOldSerialization();
            return;
        }
        var learnerParameters = input["learner"].AsObject;
        _tparam.FromJson(learnerParameters["learner_train_param"]);
        var gradientBooster = learnerParameters["gradient_booster"];
        var objectiveFn = learnerParameters["objective"];
        if (_obj is null)
        {
            Check.Eq(objectiveFn["name"].AsString, _tparam.Objective);
            _obj = Registry.CreateObjective(_tparam.Objective, _ctx);
        }
        _obj.LoadConfig(objectiveFn);
        _modelState.Task = _obj.Task;
        _tparam.Booster = CanonicalizeBoosterName(gradientBooster["name"].AsString);
        _gbm ??= Registry.CreateBooster(_tparam.Booster, _ctx, _modelState);
        _gbm.LoadConfig(gradientBooster);
        var jMetrics = learnerParameters["metrics"].AsArray;
        _metricNames.Clear();
        _metrics.Clear();
        foreach (var jm in jMetrics)
        {
            var oldSerialization = jm is JsonString;
            string name;
            if (oldSerialization)
            {
                ErrorMsg.WarnOldSerialization();
                name = jm.AsString;
            }
            else
            {
                name = jm["name"].AsString;
            }
            _metricNames.Add(name);
            var m = Registry.CreateMetric(name, _ctx);
            if (!oldSerialization) m.LoadConfig(jm);
            _metrics.Add(m);
        }
        _ctx.LoadJson(learnerParameters["generic_param"]);
        _needConfiguration = true;
    }

    public void LoadConfig(Json input)
    {
        LoadConfigImpl(input);
        Configure();
    }

    public JsonObject SaveConfig()
    {
        Check.That(!_needConfiguration, "Call Configure before saving model.");
        var output = new JsonObject();
        VersionInfo.Save(output);
        var lp = new JsonObject
        {
            ["learner_train_param"] = _tparam.ToJson(),
            ["learner_model_param"] = ModelStateToJson(),
        };
        var gb = new JsonObject();
        _gbm!.SaveConfig(gb);
        lp["gradient_booster"] = gb;
        var obj = new JsonObject();
        _obj!.SaveConfig(obj);
        lp["objective"] = obj;
        var metrics = new JsonArray();
        foreach (var m in _metrics)
        {
            var jm = new JsonObject();
            m.SaveConfig(jm);
            metrics.Add(jm);
        }
        lp["metrics"] = metrics;
        lp["generic_param"] = _ctx.SaveJson();
        output["learner"] = lp;
        return output;
    }

    // ---- attributes & feature info -------------------------------------------------------------

    public uint NumFeature => _modelState.NumFeature;
    public void SetAttr(string key, string value) => _attributes[key] = value;
    public string? GetAttr(string key) => _attributes.TryGetValue(key, out var v) ? v : null;
    public bool DelAttr(string key) => _attributes.Remove(key);
    public string[] GetAttrNames() => [.. _attributes.Keys];
    public void SetFeatureNames(IEnumerable<string> fn) => _featureNames = [.. fn];
    public string[] GetFeatureNames() => [.. _featureNames];
    public void SetFeatureTypes(IEnumerable<string> ft) => _featureTypes = [.. ft];
    public string[] GetFeatureTypes() => [.. _featureTypes];

    public CatContainer Cats()
    {
        CheckModelInitialized();
        return _gbm!.Cats();
    }

    // ---- IO ------------------------------------------------------------------------------------

    private void LoadModelImpl(Json input)
    {
        Check.That(input is JsonObject);
        _modelState.Copy(new LearnerModelState());
        var version = VersionInfo.Load(input);
        if (version.Major == 1 && version.Minor < 6)
            Log.Warning("Found JSON model saved before XGBoost 1.6, please save the model using current version again. The support for old JSON model will be discontinued in XGBoost 3.2");
        var learner = input["learner"].AsObject;
        var objectiveFn = learner["objective"];
        var name = objectiveFn["name"].AsString;
        _tparam.UpdateAllowUnknown([new("objective", name)]);
        _obj = Registry.CreateObjective(name, _ctx);
        _obj.LoadConfig(objectiveFn);
        LoadModelState(learner["learner_model_param"]);
        var gradientBooster = learner["gradient_booster"];
        name = gradientBooster["name"].AsString;
        _tparam.UpdateAllowUnknown([new("booster", name)]);
        _tparam.Booster = CanonicalizeBoosterName(_tparam.Booster);
        _gbm = Registry.CreateBooster(_tparam.Booster, _ctx, _modelState);
        _gbm.LoadModel(gradientBooster);
        _attributes.Clear();
        foreach (var kv in learner["attributes"].AsObject) _attributes[kv.Key] = kv.Value.AsString;
        if (learner.TryGetValue("feature_names", out var fnames)) _featureNames = [.. fnames.AsArray.Select(v => v.AsString)];
        if (learner.TryGetValue("feature_types", out var ftypes)) _featureTypes = [.. ftypes.AsArray.Select(v => v.AsString)];
        _needConfiguration = true;
        _cacheData.Clear();
    }

    public void LoadModel(Json input)
    {
        LoadModelImpl(input);
        Configure();
    }

    private JsonObject SaveModelUnchecked()
    {
        var output = new JsonObject();
        VersionInfo.Save(output);
        var learner = new JsonObject { ["learner_model_param"] = ModelStateToJson() };
        var gb = new JsonObject();
        _gbm!.SaveModel(gb);
        learner["gradient_booster"] = gb;
        var obj = new JsonObject();
        _obj!.SaveConfig(obj);
        learner["objective"] = obj;
        var attrs = new JsonObject();
        foreach (var kv in _attributes) attrs[kv.Key] = kv.Value;
        learner["attributes"] = attrs;
        learner["feature_names"] = new JsonArray(_featureNames.Select(n => (Json)new JsonString(n)));
        learner["feature_types"] = new JsonArray(_featureTypes.Select(n => (Json)new JsonString(n)));
        output["learner"] = learner;
        return output;
    }

    public JsonObject SaveModel()
    {
        Check.That(!_needConfiguration, "Call Configure before saving model.");
        CheckModelInitialized();
        return SaveModelUnchecked();
    }

    /// <summary><c>Learner::Save</c>: a UBJSON snapshot of the model and its configuration.</summary>
    public byte[] Save()
    {
        Check.That(!_needConfiguration, "Call Configure before saving model.");
        var snapshot = new JsonObject { ["Model"] = SaveModelUnchecked(), ["Config"] = SaveConfig() };
        return Json.DumpBytes(snapshot, true);
    }

    public void Load(ReadOnlySpan<byte> buffer)
    {
        const string msg = "Invalid serialization file.";
        Check.That(buffer.Length >= 2 && buffer[0] == (byte)'{', msg);
        Json snapshot;
        if (buffer[1] == (byte)'"')
        {
            snapshot = Json.Load(buffer, false);
            ErrorMsg.WarnOldSerialization();
        }
        else if (char.IsAsciiLetter((char)buffer[1]))
        {
            snapshot = Json.Load(buffer, true);
        }
        else
        {
            snapshot = Check.Fail<Json>(msg);
        }
        LoadModelImpl(snapshot["Model"]);
        LoadConfigImpl(snapshot["Config"]);
        Configure();
    }

    // ---- training --------------------------------------------------------------------------------

    public List<string> DumpModel(FeatureMap fmap, bool withStats, string format)
    {
        Configure();
        CheckModelInitialized();
        return _gbm!.DumpModel(fmap, withStats, format);
    }

    public Learner Slice(int begin, int end, int step, out bool outOfBound)
    {
        Configure();
        CheckModelInitialized();
        Check.Ne(_modelState.NumFeature, 0u);
        Check.Ge(begin, 0);
        var outImpl = new Learner([]);
        outImpl._modelState.Copy(_modelState);
        outImpl._tparam = _tparam.Clone();
        outImpl._ctx = _ctx.Clone();
        var gbm = Registry.CreateBooster(_tparam.Booster, outImpl._ctx, outImpl._modelState);
        _gbm!.Slice(begin, end, step, gbm, out outOfBound);
        outImpl._gbm = gbm;
        var config = SaveConfig();
        foreach (var kv in _attributes) outImpl._attributes[kv.Key] = kv.Value;
        outImpl.SetFeatureNames(_featureNames);
        outImpl.SetFeatureTypes(_featureTypes);
        outImpl.LoadConfig(config);
        outImpl.Configure();
        Check.Eq(outImpl._modelState.NumFeature, _modelState.NumFeature);
        outImpl._attributes.Remove("best_iteration");
        outImpl._attributes.Remove("best_score");
        return outImpl;
    }

    public void Reset()
    {
        Configure();
        if (_modelState.NeedsInitialization)
        {
            foreach (var weak in _cacheData)
            {
                if (weak.TryGetTarget(out var data))
                {
                    InitializeModel(data, InterceptInitialization.EstimateIntercept);
                    break;
                }
            }
        }
        CheckModelInitialized();
        var buf = Save();
        Load(buf);
        Check.That(_cacheData.Count == 0);
        _gpair.Gpair = new Tensor<GradientPair>(2);
    }

    private void SeedPerIteration()
    {
        if (_ctx.SeedPerIteration) _ctx.Rng.Seed(unchecked(_ctx.Seed * RandSeedMagic + BoostedRounds));
    }

    public void UpdateOneIter(int iter, DMatrix train)
    {
        Configure();
        InitializeModel(train, InterceptInitialization.EstimateIntercept);
        SeedPerIteration();
        ValidateDMatrix(train, true);
        var predt = new HostDeviceVector<float>();
        PredictRaw(train, predt, true, 0, 0);
        _gpair.Gpair.Reshape(train.Info.NumRow, _modelState.OutputLength);
        _obj!.GetGradient(predt, train.Info, iter, _gpair.Gpair);
        _gbm!.DoBoost(train, _gpair, _obj);
    }

    public void BoostOneIter(int iter, DMatrix train, GradientContainer inGpair)
    {
        Configure();
        InitializeModel(train, InterceptInitialization.UseDefaultIntercept);
        SeedPerIteration();
        ValidateDMatrix(train, true);
        if (inGpair.HasValueGrad)
            Check.Eq((long)_modelState.OutputLength, inGpair.NumTargets,
                "Value gradient should have the same number of targets as the overall model.");
        else
            Check.Eq((long)_modelState.OutputLength, inGpair.NumSplitTargets,
                "The number of columns in gradient should be equal to the number of targets/classes in the model.");
        _gbm!.DoBoost(train, inGpair, _obj!);
    }

    public string EvalOneIter(int iter, IReadOnlyList<DMatrix> dataSets, IReadOnlyList<string> dataNames)
    {
        Configure();
        CheckModelInitialized();
        var os = new StringBuilder();
        os.Append('[').Append(iter.ToString(CultureInfo.InvariantCulture)).Append(']');
        if (_metrics.Count == 0 && !_tparam.DisableDefaultEvalMetric)
        {
            var m = Registry.CreateMetric(_obj!.DefaultEvalMetric, _ctx);
            var config = _obj.DefaultMetricConfig();
            if (config is not JsonNull) m.LoadConfig(config);
            m.Configure([]);
            _metrics.Add(m);
        }
        for (var i = 0; i < dataSets.Count; ++i)
        {
            var m = dataSets[i];
            ValidateDMatrix(m, false);
            var output = new HostDeviceVector<float>();
            PredictRaw(m, output, false, 0, 0);
            _obj!.EvalTransform(output);
            foreach (var ev in _metrics)
            {
                var v = ev.Evaluate(output, m);
                os.Append('\t').Append(dataNames[i]).Append('-').Append(ev.Name).Append(':').Append(FormatFixed17(v));
            }
        }
        return os.ToString();
    }

    /// <summary><c>std::fixed</c> with <c>precision(17)</c>, including the C++ spellings of non-finite values.</summary>
    private static string FormatFixed17(double v)
    {
        if (double.IsNaN(v)) return double.IsNegative(v) ? "-nan(ind)" : "nan";
        if (double.IsPositiveInfinity(v)) return "inf";
        if (double.IsNegativeInfinity(v)) return "-inf";
        return v.ToString("F17", CultureInfo.InvariantCulture);
    }

    public void Predict(DMatrix data, bool outputMargin, HostDeviceVector<float> outPreds, int layerBegin, int layerEnd, bool training,
        bool predLeaf, bool predContribs, bool approxContribs, bool predInteractions, bool strictShape)
    {
        var multiple = (predLeaf ? 1 : 0) + (predInteractions ? 1 : 0) + (predContribs ? 1 : 0);
        Configure();
        if (training) InitializeModel(data, InterceptInitialization.UseDefaultIntercept);
        CheckModelInitialized();
        Check.Le(multiple, 1, "Perform one kind of prediction at a time.");
        if (predContribs)
        {
            ValidateDMatrix(data, false);
            _gbm!.PredictContribution(data, outPreds, layerBegin, layerEnd, approxContribs);
        }
        else if (predInteractions)
        {
            ValidateDMatrix(data, false);
            _gbm!.PredictInteractionContributions(data, outPreds, layerBegin, layerEnd, approxContribs);
        }
        else if (predLeaf)
        {
            ValidateDMatrix(data, false);
            _gbm!.PredictLeaf(data, outPreds, layerBegin, layerEnd, strictShape);
        }
        else
        {
            PredictRaw(data, outPreds, training, layerBegin, layerEnd);
            if (!outputMargin) _obj!.PredTransform(outPreds);
        }
    }

    public int BoostedRounds
    {
        get
        {
            if (_gbm is null) return 0;
            Check.That(!_needConfiguration);
            return _gbm.BoostedRounds;
        }
    }

    public uint Groups
    {
        get
        {
            Check.That(!_needConfiguration);
            CheckModelInitialized();
            return _modelState.NumOutputGroup;
        }
    }

    /// <summary><c>InplacePredict</c> on a row-major adapter batch.</summary>
    public HostDeviceVector<float> InplacePredict<TBatch>(TBatch batch, long nRows, long nCols, bool margin, float missing, int iterationBegin,
        int iterationEnd, ColumnsView? cats) where TBatch : IAdapterBatch
    {
        Configure();
        CheckModelInitialized();
        var output = new HostDeviceVector<float>();
        var info = new MetaInfo { NumRow = nRows, NumCol = nCols };
        if (_gbm is GBTree tree) tree.InplacePredict(batch, nRows, nCols, info, missing, output, iterationBegin, iterationEnd, cats);
        else Check.Fail("Inplace predict is not supported by the current booster.");
        if (!margin) _obj!.PredTransform(output);
        return output;
    }

    public void CalcFeatureScore(string importanceType, ReadOnlySpan<int> trees, List<uint> features, List<float> scores)
    {
        Configure();
        CheckModelInitialized();
        _gbm!.FeatureScore(importanceType, trees, features, scores);
    }

    private void PredictRaw(DMatrix data, HostDeviceVector<float> outPreds, bool training, int layerBegin, int layerEnd)
    {
        Check.That(_gbm is not null, "Predict must happen after Load or configuration");
        CheckModelInitialized();
        ValidateDMatrix(data, false);
        _gbm!.PredictBatch(data, outPreds, training, layerBegin, layerEnd);
    }

    private void ValidateDMatrix(DMatrix fmat, bool isTraining)
    {
        var info = fmat.Info;
        info.Validate();
        if (isTraining)
            Check.Eq((long)_modelState.NumFeature, info.NumCol, "Number of columns does not match number of features in booster.");
        else
            Check.Ge((long)_modelState.NumFeature, info.NumCol, "Number of columns does not match number of features in booster.");
        if (info.NumRow == 0) ErrorMsg.WarnEmptyDataset();
        if (!info.BaseMargin.Empty) Check.Eq(info.BaseMargin.Shape(1), (long)_modelState.OutputLength);
    }
}
