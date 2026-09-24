// Ports of the plugin interfaces: include/xgboost/objective.h, metric.h, gbm.h, tree_updater.h,
// linear_updater.h, gradient.h, task.h, model.h and the LearnerModelState in learner.h.
using System.Runtime.CompilerServices;
using XGBoost.Common;
using XGBoost.Data;

namespace XGBoost.Core;

public enum ObjTask : byte
{
    Regression = 0,
    Binary = 1,
    Classification = 2,
    Survival = 3,
    Ranking = 4,
    Other = 5,
}

/// <summary>Kind of learning task, <c>ObjInfo</c>.</summary>
public readonly record struct ObjInfo(ObjTask Task, bool ConstHess = false);

/// <summary>Model with JSON persistence, <c>Model</c>.</summary>
public interface IModel
{
    void LoadModel(Json input);
    void SaveModel(JsonObject output);
}

/// <summary>Component with JSON configuration, <c>Configurable</c>.</summary>
public interface IConfigurable
{
    void LoadConfig(Json input);
    void SaveConfig(JsonObject output);
}

/// <summary>Gradient (and optional value gradient), <c>GradientContainer</c>.</summary>
public sealed class GradientContainer
{
    public Tensor<GradientPair> Gpair = new(2);
    public Tensor<GradientPair> ValueGpair = new(2);

    public bool HasValueGrad => !ValueGpair.Empty;
    public long NumSplitTargets => Gpair.Shape(1);
    public long NumTargets => HasValueGrad ? ValueGpair.Shape(1) : Gpair.Shape(1);
    public TensorView<GradientPair> ValueGrad() => HasValueGrad ? ValueGpair.View() : Gpair.View();
    public Tensor<GradientPair> Grad => Gpair;

    public Tensor<GradientPair> FullGradOnly()
    {
        if (HasValueGrad) Check.Fail("Reduced gradient is not yet supported.");
        return Gpair;
    }
}

/// <summary>Objective function, <c>ObjFunction</c>.</summary>
public abstract class ObjFunction : IConfigurable
{
    public const float DefaultBaseScore = 0.5f;

    protected Context Ctx = null!;

    internal void SetContext(Context ctx) => Ctx = ctx;

    public abstract SortedSet<string> Configure(IReadOnlyList<KeyValuePair<string, string>> args);

    public abstract void GetGradient(HostDeviceVector<float> preds, MetaInfo info, int iter, Tensor<GradientPair> outGpair);

    public abstract string DefaultEvalMetric { get; }

    public virtual Json DefaultMetricConfig() => JsonNull.Instance;

    public virtual void PredTransform(HostDeviceVector<float> ioPreds) { }

    public virtual void EvalTransform(HostDeviceVector<float> ioPreds) => PredTransform(ioPreds);

    public virtual void ProbToMargin(Tensor<float> baseScore) { }

    /// <summary>Default intercept: the constant <see cref="DefaultBaseScore"/> for every target.</summary>
    public virtual void InitEstimation(MetaInfo info, Tensor<float> baseScore)
    {
        var nTargets = Targets(info);
        baseScore.Assign(Linalg.Constant(DefaultBaseScore, (long)nTargets));
    }

    public abstract ObjInfo Task { get; }

    public virtual uint Targets(MetaInfo info)
    {
        if (info.Labels.Shape(1) > 1) Check.Fail("multioutput is not supported by the current objective function");
        return 1;
    }

    public Context Context => Ctx;

    public abstract void LoadConfig(Json input);
    public abstract void SaveConfig(JsonObject output);
}

/// <summary>Evaluation metric, <c>Metric</c>.</summary>
public abstract class Metric : IConfigurable
{
    protected Context Ctx = null!;

    internal void SetContext(Context ctx) => Ctx = ctx;

    public virtual SortedSet<string> Configure(IReadOnlyList<KeyValuePair<string, string>> args) => [];

    public virtual void LoadConfig(Json input) { }

    public virtual void SaveConfig(JsonObject output) => output["name"] = new JsonString(Name);

    public abstract double Evaluate(HostDeviceVector<float> preds, DMatrix fmat);

    public abstract string Name { get; }
}

/// <summary>Gradient booster, <c>GradientBooster</c>.</summary>
public abstract class GradientBooster : IModel, IConfigurable
{
    protected readonly Context Ctx;

    protected GradientBooster(Context ctx) => Ctx = ctx;

    public abstract SortedSet<string> Configure(IReadOnlyList<KeyValuePair<string, string>> cfg);

    public virtual void Slice(int begin, int end, int step, GradientBooster output, out bool outOfBound)
    {
        outOfBound = false;
        Check.Fail("Slice is not supported by the current booster.");
    }

    public abstract int BoostedRounds { get; }

    public abstract void DoBoost(DMatrix fmat, GradientContainer inGpair, ObjFunction obj);

    public abstract void PredictBatch(DMatrix dmat, HostDeviceVector<float> outPreds, bool training, int begin, int end);

    public virtual void InplacePredict(DMatrix m, float missing, HostDeviceVector<float> outPreds, int layerBegin, int layerEnd) =>
        Check.Fail("Inplace predict is not supported by the current booster.");

    public abstract void PredictLeaf(DMatrix dmat, HostDeviceVector<float> outPreds, int layerBegin, int layerEnd, bool strictShape);

    public abstract void PredictContribution(DMatrix dmat, HostDeviceVector<float> outContribs, int layerBegin, int layerEnd,
        bool approximate = false);

    public abstract void PredictInteractionContributions(DMatrix dmat, HostDeviceVector<float> outContribs, int layerBegin,
        int layerEnd, bool approximate);

    public abstract List<string> DumpModel(Tree.FeatureMap fmap, bool withStats, string format);

    public abstract void FeatureScore(string importanceType, ReadOnlySpan<int> trees, List<uint> features, List<float> scores);

    public virtual CatContainer Cats() => Check.Fail<CatContainer>("Retrieving categories is not supported by the current booster.");

    public abstract void LoadModel(Json input);
    public abstract void SaveModel(JsonObject output);
    public abstract void LoadConfig(Json input);
    public abstract void SaveConfig(JsonObject output);
}

/// <summary>Tree updater, <c>TreeUpdater</c>.</summary>
public abstract class TreeUpdater : IConfigurable
{
    protected readonly Context Ctx;

    protected TreeUpdater(Context ctx) => Ctx = ctx;

    public abstract SortedSet<string> Configure(IReadOnlyList<KeyValuePair<string, string>> args);

    public virtual bool CanModifyTree => false;

    /// <param name="param">Training parameters.</param>
    /// <param name="gpair">Gradient.</param>
    /// <param name="fmat">Training data.</param>
    /// <param name="outPosition">Leaf position of each row per tree (negative when sampled out).</param>
    /// <param name="outTrees">Trees to build or update.</param>
    public abstract void Update(Tree.TrainParam param, GradientContainer gpair, DMatrix fmat, List<HostDeviceVector<int>> outPosition,
        List<Tree.RegTree> outTrees);

    public abstract string Name { get; }

    public virtual void LoadConfig(Json input) { }
    public virtual void SaveConfig(JsonObject output) { }
}

/// <summary>Linear updater, <c>LinearUpdater</c>.</summary>
public abstract class LinearUpdater : IConfigurable
{
    protected Context Ctx = null!;

    internal void SetContext(Context ctx) => Ctx = ctx;

    public abstract SortedSet<string> Configure(IReadOnlyList<KeyValuePair<string, string>> args);

    public abstract void Update(Tensor<GradientPair> inGpair, DMatrix data, Gbm.GBLinearModel model, double sumInstanceWeight);

    public abstract void LoadConfig(Json input);
    public abstract void SaveConfig(JsonObject output);
}

public enum MultiStrategy
{
    OneOutputPerTree = 0,
    MultiOutputTree = 1,
}

/// <summary>State shared by the learner and gradient booster, <c>LearnerModelState</c>.</summary>
public sealed class LearnerModelState
{
    private Tensor<float> _baseScore = new(1);
    private float[] _baseScoreValue = [];

    public uint NumFeature;
    public int NumClass;
    public uint NumTarget = 1;
    public bool BoostFromAverage = true;
    public uint NumOutputGroup;
    public ObjInfo Task = new(ObjTask.Regression);
    public MultiStrategy MultiStrategy = MultiStrategy.OneOutputPerTree;

    public LearnerModelState() { }

    public LearnerModelState(uint nFeatures, int nClasses, uint nTargets, bool boostFromAverage, float[] baseScoreValue,
        Tensor<float> baseScore, ObjInfo task, MultiStrategy multiStrategy)
    {
        NumFeature = nFeatures;
        NumClass = nClasses;
        NumTarget = nTargets;
        BoostFromAverage = boostFromAverage;
        NumOutputGroup = Math.Max(Math.Max(nTargets, (uint)nClasses), 1u);
        Task = task;
        MultiStrategy = multiStrategy;
        if (NumClass > 1 && NumTarget > 1)
            Check.Fail($"multi-target-multi-class is not yet supported. Output classes:{NumClass}, output targets:{NumTarget}");
        SetBaseScore(baseScoreValue, baseScore);
    }

    public void SetBaseScore(float[] value, Tensor<float> baseScore)
    {
        _baseScoreValue = value;
        _baseScore = baseScore;
    }

    public TensorView<float> BaseScore()
    {
        Check.Ge(_baseScore.Size, 1, "Model is not yet initialized (not fitted).");
        return _baseScore.View();
    }

    public float[] BaseScoreValue => _baseScoreValue;

    public void Copy(LearnerModelState that)
    {
        _baseScore = that._baseScore.Clone();
        _baseScoreValue = (float[])that._baseScoreValue.Clone();
        NumFeature = that.NumFeature;
        NumClass = that.NumClass;
        NumTarget = that.NumTarget;
        BoostFromAverage = that.BoostFromAverage;
        NumOutputGroup = that.NumOutputGroup;
        Task = that.Task;
        MultiStrategy = that.MultiStrategy;
    }

    public bool IsVectorLeaf => MultiStrategy == MultiStrategy.MultiOutputTree;
    public uint OutputLength => NumOutputGroup;
    public uint LeafLength => IsVectorLeaf ? OutputLength : 1;
    public bool NeedsInitialization => NumFeature == 0 || NumOutputGroup == 0 || _baseScore.Size == 0;
    public bool Initialized => !NeedsInitialization;
}

/// <summary>Name based factories, equivalent to the dmlc registries.</summary>
public static class Registry
{
    public static readonly Dictionary<string, Func<ObjFunction>> ObjectiveFactories = new(StringComparer.Ordinal);
    public static readonly Dictionary<string, Func<string?, Metric>> MetricFactories = new(StringComparer.Ordinal);
    public static readonly Dictionary<string, Func<LearnerModelState, Context, GradientBooster>> BoosterFactories = new(StringComparer.Ordinal);
    public static readonly Dictionary<string, Func<Context, ObjInfo, TreeUpdater>> TreeUpdaterFactories = new(StringComparer.Ordinal);
    public static readonly Dictionary<string, Func<LinearUpdater>> LinearUpdaterFactories = new(StringComparer.Ordinal);

    private static int _initialized;

    /// <summary>Registers all built-in components (idempotent).</summary>
    public static void EnsureInitialized()
    {
        if (Interlocked.Exchange(ref _initialized, 1) == 1) return;
        global::XGBoost.Objectives.ObjectiveRegistry.Register();
        global::XGBoost.Metrics.MetricRegistry.Register();
        global::XGBoost.Gbm.BoosterRegistry.Register();
        global::XGBoost.Tree.UpdaterRegistry.Register();
        global::XGBoost.Linear.LinearUpdaterRegistry.Register();
    }

    public static ObjFunction CreateObjective(string name, Context ctx)
    {
        EnsureInitialized();
        if (!ObjectiveFactories.TryGetValue(name, out var f))
        {
            var sb = new System.Text.StringBuilder($"Unknown objective function: `{name}`\nObjective candidate:\n");
            foreach (var k in ObjectiveFactories.Keys.Order(StringComparer.Ordinal)) sb.Append(k).Append('\n');
            Check.Fail(sb.ToString());
        }
        var obj = f!();
        obj.SetContext(ctx);
        return obj;
    }

    /// <summary><c>Metric::Create</c>: names can carry a parameter after '@', e.g. <c>ndcg@5</c>.</summary>
    public static Metric CreateMetric(string name, Context ctx)
    {
        EnsureInitialized();
        var buf = name;
        string? prefix = name;
        string? param = null;
        var pos = buf.IndexOf('@');
        if (pos >= 0)
        {
            prefix = buf[..pos];
            param = buf[(pos + 1)..];
        }
        else if (buf.Length != 0 && buf[^1] == '-')
        {
            prefix = buf[..^1]; // Chop off '-'
            param = "-";
        }
        if (!MetricFactories.TryGetValue(prefix, out var f)) Check.Fail($"Unknown metric function {name}");
        var m = f!(param);
        m.SetContext(ctx);
        return m;
    }

    public static GradientBooster CreateBooster(string name, Context ctx, LearnerModelState state)
    {
        EnsureInitialized();
        if (!BoosterFactories.TryGetValue(name == "dart" ? "gbtree" : name, out var f)) Check.Fail($"Unknown gbm type {name}");
        return f!(state, ctx);
    }

    public static TreeUpdater CreateTreeUpdater(string name, Context ctx, ObjInfo task)
    {
        EnsureInitialized();
        if (!TreeUpdaterFactories.TryGetValue(name, out var f)) Check.Fail($"Unknown tree updater {name}");
        return f!(ctx, task);
    }

    public static LinearUpdater CreateLinearUpdater(string name, Context ctx)
    {
        EnsureInitialized();
        if (!LinearUpdaterFactories.TryGetValue(name, out var f)) Check.Fail($"Unknown linear updater {name}");
        var u = f!();
        u.SetContext(ctx);
        return u;
    }
}

/// <summary>Per-DMatrix cache keyed by object identity, <c>DMatrixCache</c>.</summary>
public sealed class DMatrixCache<T> where T : class, new()
{
    private readonly ConditionalWeakTable<DMatrix, T> _table = new();

    public T CacheItem(DMatrix m) => _table.GetValue(m, _ => new T());

    public T? Entry(DMatrix m) => _table.TryGetValue(m, out var v) ? v : null;

    public void Clear() => _table.Clear();
}
