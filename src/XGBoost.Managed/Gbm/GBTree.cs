// Port of src/gbm/gbtree.h/.cc (tree booster, including the DART dropout that is folded into gbtree).
using XGBoost.Predictors;
using XGBoost.Tree;

namespace XGBoost.Gbm;

public enum TreeMethod { Auto = 0, Approx = 1, Exact = 2, Hist = 3 }

public enum TreeProcessType { Default = 0, Update = 1 }

public enum DartSampleType { Uniform = 0, Weighted = 1 }

public sealed class GBTreeTrainParam : XGBoostParameter<GBTreeTrainParam>
{
    public string UpdaterSeq = "";
    public TreeProcessType ProcessType;
    public TreeMethod TreeMethod;

    protected override void Declare(ParamManager<GBTreeTrainParam> m)
    {
        Field(m, "updater_seq", p => p.UpdaterSeq, (p, v) => p.UpdaterSeq = v).SetDefault("").Describe("Tree updater sequence.");
        EnumField<TreeProcessType>(m, "process_type", p => p.ProcessType, (p, v) => p.ProcessType = v).SetDefault(TreeProcessType.Default)
            .AddEnum("default", TreeProcessType.Default).AddEnum("update", TreeProcessType.Update)
            .Describe("Whether to run the normal boosting process that creates new trees, or to update the trees in an existing model.");
        Alias(m, "updater_seq", "updater");
        EnumField<TreeMethod>(m, "tree_method", p => p.TreeMethod, (p, v) => p.TreeMethod = v).SetDefault(TreeMethod.Auto)
            .AddEnum("auto", TreeMethod.Auto).AddEnum("approx", TreeMethod.Approx).AddEnum("exact", TreeMethod.Exact)
            .AddEnum("hist", TreeMethod.Hist).Describe("Choice of tree construction method.");
    }
}

public enum DartNormalizeType { Tree = 0, Forest = 1 }

public sealed class DartTrainParam : XGBoostParameter<DartTrainParam>
{
    public DartSampleType SampleType;
    public DartNormalizeType NormalizeType;
    public float RateDrop;
    public bool OneDrop;
    public float SkipDrop;

    protected override void Declare(ParamManager<DartTrainParam> m)
    {
        EnumField<DartSampleType>(m, "sample_type", p => p.SampleType, (p, v) => p.SampleType = v).SetDefault(DartSampleType.Uniform)
            .AddEnum("uniform", DartSampleType.Uniform).AddEnum("weighted", DartSampleType.Weighted)
            .Describe("Different types of sampling algorithm.");
        EnumField<DartNormalizeType>(m, "normalize_type", p => p.NormalizeType, (p, v) => p.NormalizeType = v).SetDefault(DartNormalizeType.Tree)
            .AddEnum("tree", DartNormalizeType.Tree).AddEnum("forest", DartNormalizeType.Forest)
            .Describe("Different types of normalization algorithm.");
        Field(m, "rate_drop", p => p.RateDrop, (p, v) => p.RateDrop = v).SetRange(0.0f, 1.0f).SetDefault(0.0f)
            .Describe("Fraction of trees to drop during the dropout.");
        Field(m, "one_drop", p => p.OneDrop, (p, v) => p.OneDrop = v).SetDefault(false)
            .Describe("Whether at least one tree should always be dropped during the dropout.");
        Field(m, "skip_drop", p => p.SkipDrop, (p, v) => p.SkipDrop = v).SetRange(0.0f, 1.0f).SetDefault(0.0f)
            .Describe("Probability of skipping the dropout during a boosting iteration.");
    }

    public bool HasDropout => RateDrop != 0.0f || OneDrop || SkipDrop != 0.0f;

    public DartTrainParam Clone() => (DartTrainParam)MemberwiseClone();
}

public sealed class PredictionCacheEntry
{
    public HostDeviceVector<float> Predictions = new();
    public uint Version;

    public void Update(uint v) => Version += v;
    public void Reset() => Version = 0;
}

public sealed class GBTree : GradientBooster
{
    private readonly GBTreeModel _model;
    private GBTreeTrainParam _tparam = new();
    private DartTrainParam _dparam = new();
    private readonly TrainParam _treeParam = new();
    private bool _specifiedUpdater;
    private readonly List<TreeUpdater> _updaters = [];
    private readonly List<int> _idxDrop = [];
    private readonly DMatrixCache<PredictionCacheEntry> _predictionCache = new();

    public GBTree(LearnerModelState boosterConfig, Context ctx) : base(ctx)
    {
        _model = new GBTreeModel(boosterConfig, ctx);
        // dparam_{} is value-initialised in C++: defaults without Init.
        _dparam.UpdateAllowUnknown([]);
    }

    public GBTreeModel Model => _model;
    public GBTreeTrainParam TrainParam => _tparam;

    public PredictionCacheEntry? PredictionCache(DMatrix m) => _predictionCache.Entry(m);

    private static string MapTreeMethodToUpdaters(TreeMethod treeMethod) => treeMethod switch
    {
        TreeMethod.Auto or TreeMethod.Hist => "grow_quantile_histmaker",
        TreeMethod.Approx => "grow_histmaker",
        TreeMethod.Exact => "grow_colmaker,prune",
        _ => Check.Fail<string>($"Unknown tree_method: `{(int)treeMethod}`."),
    };

    private static bool UpdatersMatched(List<string> updaterSeq, List<TreeUpdater> updaters)
    {
        if (updaterSeq.Count != updaters.Count) return false;
        for (var i = 0; i < updaterSeq.Count; ++i)
            if (updaterSeq[i] != updaters[i].Name) return false;
        return true;
    }

    public override SortedSet<string> Configure(Args cfg)
    {
        var used = ParameterUtils.GetUsedParameters(cfg, _tparam.UpdateAllowUnknown(cfg));
        used.UnionWith(ParameterUtils.GetUsedParameters(cfg, _dparam.UpdateAllowUnknown(cfg)));
        used.UnionWith(ParameterUtils.GetUsedParameters(cfg, _treeParam.UpdateAllowUnknown(cfg)));
        used.UnionWith(_model.Configure(cfg));
        if (_tparam.ProcessType == TreeProcessType.Update) _model.InitTreesToUpdate();
        _specifiedUpdater = cfg.Any(a => a.Key == "updater");
        if (_specifiedUpdater) ErrorMsg.WarnManualUpdater();
        Log.Debug($"Using tree method: {(int)_tparam.TreeMethod}");
        if (!_specifiedUpdater) _tparam.UpdaterSeq = MapTreeMethodToUpdaters(_tparam.TreeMethod);
        var upNames = StringUtils.Split(_tparam.UpdaterSeq, ',');
        if (!UpdatersMatched(upNames, _updaters))
        {
            _updaters.Clear();
            foreach (var name in upNames) _updaters.Add(Registry.CreateTreeUpdater(name, Ctx, _model.LearnerModelState.Task));
        }
        foreach (var up in _updaters) used.UnionWith(up.Configure(cfg));
        return used;
    }

    private static void CopyGradient(Context ctx, Tensor<GradientPair> inGpair, int groupId, Tensor<GradientPair> outGpair)
    {
        outGpair.Reshape(inGpair.Shape(0), 1);
        var hTmp = outGpair.HostView();
        var hIn = inGpair.HostView();
        Threading.ParallelFor(inGpair.Shape(0), ctx.Threads(), i => hTmp[i, 0] = hIn[i, groupId]);
    }

    public override void DoBoost(DMatrix fmat, GradientContainer inGpair, ObjFunction obj)
    {
        var predt = _predictionCache.CacheItem(fmat);
        if (_model.LearnerModelState.IsVectorLeaf)
            Check.That(_tparam.TreeMethod is TreeMethod.Hist or TreeMethod.Auto,
                "Only the hist tree method is supported for building multi-target trees with vector leaf.");
        if (inGpair.HasValueGrad)
        {
            Check.That(_model.LearnerModelState.IsVectorLeaf, "Reduced gradient must be used with vector leaf trees");
            Check.That(!_treeParam.HasMonotone(), "Monotonic constraints are not supported with reduced gradients.");
        }
        var newTrees = new List<List<RegTree>>();
        var nGroups = (int)_model.LearnerModelState.OutputLength;
        if (_model.Cats().Empty && !fmat.Cats.Empty)
        {
            _model.Cats().Copy(fmat.Cats);
            _model.Cats().Sort();
        }
        else
        {
            Check.Eq(_model.Cats().NumCatsTotal, fmat.Cats.NumCatsTotal,
                "A new dataset with different categorical features is used for training an existing model.");
        }
        var predictor = new CpuPredictor(Ctx);
        if (predt.Predictions.Size == 0 && fmat.Info.NumRow != 0)
        {
            Check.Eq(predt.Version, 0u);
            predictor.InitOutPredictions(fmat.Info, predt.Predictions, _model);
        }
        var output = Linalg.MakeTensorView(predt.Predictions, fmat.Info.NumRow, (long)_model.LearnerModelState.OutputLength);
        Check.Ne(nGroups, 0);
        var nodePosition = new List<HostDeviceVector<int>>();

        bool PredictFromNodePositions(List<HostDeviceVector<int>> positions, List<RegTree> trees, TensorView<float> outPreds)
        {
            Check.Eq(positions.Count, trees.Count);
            foreach (var position in positions)
                if (outPreds.Shape(0) != 0 && position.Size != outPreds.Shape(0)) return false;
            predictor.PredictFromLeafIds(positions, trees, outPreds);
            return true;
        }

        if (_model.LearnerModelState.IsVectorLeaf || _model.LearnerModelState.OutputLength == 1u)
        {
            var ret = new List<RegTree>();
            BoostNewTrees(inGpair, fmat, 0, nodePosition, ret);
            if (PredictFromNodePositions(nodePosition, ret, output)) predt.Update(1);
            newTrees.Add(ret);
        }
        else
        {
            Check.Eq(inGpair.Gpair.Size % nGroups, 0, "Must have exactly n_groups * n_samples gpairs.");
            var tmp = new GradientContainer { Gpair = new Tensor<GradientPair>([inGpair.Gpair.Shape(0), 1]) };
            var cacheUpdated = true;
            for (var gid = 0; gid < nGroups; ++gid)
            {
                nodePosition.Clear();
                CopyGradient(Ctx, inGpair.Gpair, gid, tmp.Gpair);
                var ret = new List<RegTree>();
                BoostNewTrees(tmp, fmat, gid, nodePosition, ret);
                var vPredt = output.Slice(SliceArg.All, SliceArg.Range(gid, gid + 1));
                cacheUpdated = PredictFromNodePositions(nodePosition, ret, vPredt) && cacheUpdated;
                newTrees.Add(ret);
            }
            if (cacheUpdated) predt.Update(1);
        }
        CommitModel(newTrees);
    }

    private List<RegTree> InitNewTrees(int bstGroup, List<RegTree> ret)
    {
        ret.Clear();
        for (var i = 0; i < _model.Param.NumParallelTree; ++i)
        {
            if (_tparam.ProcessType == TreeProcessType.Default)
            {
                Check.That(_updaters.Count != 0);
                Check.That(!_updaters[0].CanModifyTree,
                    $"Updater: `{_updaters[0].Name}` can not be used to create new trees. Set `process_type` to `update` if you want to update existing trees.");
                ret.Add(new RegTree(_model.LearnerModelState.LeafLength, _model.LearnerModelState.NumFeature));
            }
            else if (_tparam.ProcessType == TreeProcessType.Update)
            {
                foreach (var up in _updaters)
                    Check.That(up.CanModifyTree,
                        $"Updater: `{up.Name}` can not be used to modify existing trees. Set `process_type` to `default` if you want to build new trees.");
                Check.Lt(_model.Trees.Count, _model.TreesToUpdate.Count,
                    "No more tree left for updating.  For updating existing trees, boosting rounds can not exceed previous training rounds");
                ret.Add(_model.TreesToUpdate[_model.Trees.Count + bstGroup * _model.Param.NumParallelTree + i]);
            }
        }
        return ret;
    }

    private void BoostNewTrees(GradientContainer gpair, DMatrix fmat, int bstGroup, List<HostDeviceVector<int>> outPosition, List<RegTree> ret)
    {
        var newTrees = InitNewTrees(bstGroup, ret);
        var nOut = _model.LearnerModelState.OutputLength * fmat.Info.NumRow;
        const string msg = "Mismatching size between number of rows from input data and size of gradient vector.";
        if (!_model.LearnerModelState.IsVectorLeaf && fmat.Info.NumRow != 0)
            Check.Eq(nOut % gpair.Gpair.Size, 0L, msg);
        else if (_model.LearnerModelState.IsVectorLeaf && !gpair.HasValueGrad)
            Check.Eq((long)gpair.Gpair.Size, nOut, msg);
        outPosition.Clear();
        for (var i = 0; i < newTrees.Count; ++i) outPosition.Add(new HostDeviceVector<int>());
        var lr = _treeParam.LearningRate;
        _treeParam.LearningRate /= newTrees.Count;
        foreach (var up in _updaters) up.Update(_treeParam, gpair, fmat, outPosition, newTrees);
        _treeParam.LearningRate = lr;
    }

    private void CommitModel(List<List<RegTree>> newTrees)
    {
        var nOldTrees = _model.Trees.Count;
        var hasTreeWeights = _model.WeightDrop.Count != 0;
        var dropoutConfigured = _dparam.HasDropout;
        var trackTreeWeights = hasTreeWeights || dropoutConfigured;
        if (trackTreeWeights && _model.WeightDrop.Count < nOldTrees)
            _model.WeightDrop.AddRange(Enumerable.Repeat(1.0f, nOldTrees - _model.WeightDrop.Count));
        var nNewTrees = _model.CommitModel(newTrees);
        if (trackTreeWeights)
        {
            var numDrop = NormalizeTrees(nNewTrees);
            Log.Info($"drop {numDrop} trees, weight = {Format.G(_model.WeightDrop[^1], 6)}");
        }
    }

    public override void LoadConfig(Json input)
    {
        var name = input["name"].AsString;
        Check.That(name is "gbtree" or "dart",
            $"Unknown booster name in model JSON: `{name}`. Only `gbtree` or legacy `dart` boosters are accepted here.");
        var config = name == "dart" ? input["gbtree"] : input;
        _tparam.FromJson(config["gbtree_train_param"]);
        _treeParam.FromJson(config["tree_train_param"]);
        var obj = config.AsObject;
        if (obj.TryGetValue("dart_train_param", out var dp))
        {
            _dparam.FromJson(dp);
        }
        else if (name == "dart")
        {
            _dparam.FromJson(input["dart_train_param"]);
        }
        else
        {
            _dparam = new DartTrainParam();
            _dparam.UpdateAllowUnknown([]);
        }
        _tparam.ProcessType = TreeProcessType.Default;
        var updaterSeq = new List<Json>();
        if (config["updater"] is JsonObject upObj)
        {
            ErrorMsg.WarnOldSerialization();
            foreach (var kv in upObj)
            {
                var c = kv.Value.AsObject;
                c["name"] = kv.Key;
                updaterSeq.Add(c);
            }
        }
        else
        {
            updaterSeq.AddRange(config["updater"].AsArray);
        }
        _updaters.Clear();
        foreach (var c in updaterSeq)
        {
            var upName = c["name"].AsString;
            if (upName == "grow_gpu_hist")
            {
                upName = "grow_quantile_histmaker";
                Log.Warning("Changing updater from `grow_gpu_hist` to `grow_quantile_histmaker`.");
            }
            var up = Registry.CreateTreeUpdater(upName, Ctx, _model.LearnerModelState.Task);
            up.LoadConfig(c);
            _updaters.Add(up);
        }
        _specifiedUpdater = config["specified_updater"].AsBoolean;
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = "gbtree";
        var tp = _tparam.ToJson();
        tp["process_type"] = "default";
        output["gbtree_train_param"] = tp;
        output["tree_train_param"] = _treeParam.ToJson();
        output["dart_train_param"] = _dparam.ToJson();
        output["gbtree_model_param"] = _model.Param.ToJson();
        var jUpdaters = new JsonArray();
        foreach (var up in _updaters)
        {
            var upConfig = new JsonObject { ["name"] = up.Name };
            up.SaveConfig(upConfig);
            jUpdaters.Add(upConfig);
        }
        output["updater"] = jUpdaters;
        output["specified_updater"] = _specifiedUpdater;
    }

    public override void LoadModel(Json input)
    {
        var name = input["name"].AsString;
        Check.That(name is "gbtree" or "dart");
        var model = name == "dart" ? input["gbtree"] : input;
        _model.LoadModel(model["model"]);
        var obj = (name == "dart" ? input : model).AsObject;
        if (obj.TryGetValue("weight_drop", out var wd)) _model.WeightDrop = [.. wd.AsArray.Select(v => v.AsNumber)];
        Check.Le(_model.WeightDrop.Count, _model.Trees.Count);
    }

    public override void SaveModel(JsonObject output)
    {
        output["name"] = "gbtree";
        var m = new JsonObject();
        _model.SaveModel(m);
        output["model"] = m;
    }

    /// <summary><c>DropTrees</c>: DART dropout; returns the per-tree weights with dropped trees zeroed.</summary>
    private float[]? DropTrees(bool isTraining)
    {
        if (!isTraining) return null;
        var dropoutConfigured = _dparam.RateDrop != 0.0f || _dparam.OneDrop || _dparam.SkipDrop != 0.0f;
        if (_model.WeightDrop.Count == 0)
        {
            if (!dropoutConfigured || _model.Trees.Count == 0) return null;
            _model.WeightDrop.AddRange(Enumerable.Repeat(1.0f, _model.Trees.Count));
        }
        _idxDrop.Clear();
        var rnd = Ctx.Rng;
        var skip = false;
        if (_dparam.SkipDrop > 0.0) skip = StdRandom.UniformDouble(rnd) < _dparam.SkipDrop;
        if (skip) return null;
        var wdrop = _model.WeightDrop;
        if (_dparam.SampleType == DartSampleType.Weighted)
        {
            var sumWeight = 0.0f;
            foreach (var elem in wdrop) sumWeight += elem;
            for (var i = 0; i < wdrop.Count; ++i)
            {
                if (StdRandom.UniformDouble(rnd) < _dparam.RateDrop * (float)wdrop.Count * wdrop[i] / sumWeight) _idxDrop.Add(i);
            }
            if (_dparam.OneDrop && _idxDrop.Count == 0 && wdrop.Count != 0)
            {
                var n = wdrop.Count;
                var delta = (double)n / n;
                var weights = new double[n];
                for (var k = 0; k < n; ++k) weights[k] = wdrop[(int)(0.0 + delta * (k + 0.5))];
                _idxDrop.Add(StdRandom.Discrete(rnd, weights));
            }
        }
        else
        {
            for (var i = 0; i < wdrop.Count; ++i)
                if (StdRandom.UniformDouble(rnd) < _dparam.RateDrop) _idxDrop.Add(i);
            if (_dparam.OneDrop && _idxDrop.Count == 0 && wdrop.Count != 0)
                _idxDrop.Add((int)StdRandom.UniformInt(rnd, 0, (ulong)wdrop.Count - 1));
        }
        if (_idxDrop.Count == 0) return null;
        var dropped = wdrop.ToArray();
        foreach (var idx in _idxDrop) dropped[idx] = 0.0f;
        return dropped;
    }

    private int NormalizeTrees(int sizeNewTrees)
    {
        Check.That(_treeParam.GetInitialised());
        var lr = _treeParam.LearningRate;
        var numDrop = _idxDrop.Count;
        if (numDrop == 0)
        {
            for (var i = 0; i < sizeNewTrees; ++i) _model.WeightDrop.Add(1.0f);
        }
        else if (_dparam.NormalizeType == DartNormalizeType.Forest)
        {
            var factor = (float)(1.0 / (1.0 + lr));
            foreach (var i in _idxDrop) _model.WeightDrop[i] *= factor;
            for (var i = 0; i < sizeNewTrees; ++i) _model.WeightDrop.Add(factor);
        }
        else
        {
            // size_t + float is evaluated in float.
            var factor = (float)(1.0 * numDrop / (float)(numDrop + lr));
            foreach (var i in _idxDrop) _model.WeightDrop[i] *= factor;
            for (var i = 0; i < sizeNewTrees; ++i) _model.WeightDrop.Add((float)(1.0 / (float)(numDrop + lr)));
        }
        _idxDrop.Clear();
        return numDrop;
    }

    // ---- layers --------------------------------------------------------------------------------

    public static (int Begin, int End) LayerToTree(GBTreeModel model, int begin, int end)
    {
        Check.That(model.IterationIndptr.Count != 0);
        end = end == 0 ? model.BoostedRounds : end;
        Check.Le(end, model.BoostedRounds, "Out of range for tree layers.");
        var treeBegin = model.IterationIndptr[begin];
        var treeEnd = model.IterationIndptr[end];
        if (model.Trees.Count != 0) Check.Le(treeBegin, treeEnd);
        return (treeBegin, treeEnd);
    }

    private static bool SliceTrees(int begin, int end, int step, GBTreeModel model, Action<int, int> fn)
    {
        end = end == 0 ? model.IterationIndptr.Count : end;
        Check.Ge(step, 1);
        if (step > end - begin) return true;
        if (end > model.BoostedRounds) return true;
        var nLayers = (end - begin) / step;
        var outL = 0;
        for (var l = begin; l < end; l += step)
        {
            var (treeBegin, treeEnd) = LayerToTree(model, l, l + 1);
            if (treeEnd > model.Trees.Count) return true;
            for (var treeIdx = treeBegin; treeIdx < treeEnd; ++treeIdx) fn(treeIdx, outL);
            ++outL;
        }
        Check.Eq(outL, nLayers);
        return false;
    }

    public override void Slice(int begin, int end, int step, GradientBooster output, out bool outOfBound)
    {
        var pGbtree = output as GBTree;
        Check.That(pGbtree is not null);
        var outModel = pGbtree!._model;
        Check.That(_model.LearnerModelState.Initialized);
        end = end == 0 ? _model.BoostedRounds : end;
        Check.Ge(step, 1);
        Check.Ne(end, begin, "Empty slice is not allowed.");
        if (step > end - begin)
        {
            outOfBound = true;
            return;
        }
        var nLayers = (end - begin) / step;
        outModel.IterationIndptr = [.. new int[nLayers + 1]];
        if (_model.TreesToUpdate.Count != 0)
            Check.Eq(_model.TreesToUpdate.Count, _model.Trees.Count,
                $"Not all trees are updated, {_model.TreesToUpdate.Count - _model.Trees.Count} trees remain.  Slice the model before making update if you only want to update a portion of trees.");
        outOfBound = SliceTrees(begin, end, step, _model, (inTreeIdx, outL) =>
        {
            outModel.Trees.Add(_model.Trees[inTreeIdx].Copy());
            outModel.TreeInfo.Add(_model.TreeInfo[inTreeIdx]);
            outModel.IterationIndptr[outL + 1]++;
        });
        for (var i = 1; i < outModel.IterationIndptr.Count; ++i) outModel.IterationIndptr[i] += outModel.IterationIndptr[i - 1];
        Check.Eq(outModel.IterationIndptr[0], 0);
        outModel.Param.NumTrees = outModel.Trees.Count;
        outModel.Param.NumParallelTree = _model.Param.NumParallelTree;
        outModel.Cats().Copy(_model.Cats());
        pGbtree._dparam = _dparam.Clone();
        pGbtree._idxDrop.Clear();
        outModel.WeightDrop.Clear();
        if (_model.WeightDrop.Count != 0)
            SliceTrees(begin, end, step, _model, (inTreeIdx, _) => outModel.WeightDrop.Add(_model.WeightDrop[inTreeIdx]));
    }

    public override int BoostedRounds => _model.BoostedRounds;

    public override void PredictBatch(DMatrix fmat, HostDeviceVector<float> outPreds, bool isTraining, int layerBegin, int layerEnd)
    {
        var cache = _predictionCache.CacheItem(fmat);
        var droppedWeights = DropTrees(isTraining);
        IReadOnlyList<float>? treeWeightsOverride = droppedWeights;
        if (layerEnd == 0) layerEnd = BoostedRounds;
        var cacheVersion = cache.Version;
        var preserveCache = treeWeightsOverride is null && _model.TreeWeights is null && fmat.Info.BaseMargin.Empty && layerBegin == 0;
        var reuseCache = preserveCache && layerEnd >= (int)cacheVersion;
        var initializeOutput = !reuseCache || cacheVersion == 0;
        var predictionBegin = reuseCache ? (int)cacheVersion : layerBegin;
        if (!reuseCache)
        {
            cache.Version = 0;
            cacheVersion = 0;
        }
        if (cache.Predictions.Size == 0 && fmat.Info.NumRow != 0) Check.Eq(cache.Version, 0u);
        var predictor = new CpuPredictor(Ctx);
        if (initializeOutput) predictor.InitOutPredictions(fmat.Info, cache.Predictions, _model);
        var (treeBegin, treeEnd) = LayerToTree(_model, predictionBegin, layerEnd);
        Check.Le(treeEnd, _model.Trees.Count, "Invalid number of trees.");
        if (treeEnd > treeBegin) predictor.PredictBatch(fmat, cache.Predictions, _model, treeBegin, treeEnd, treeWeightsOverride);
        if (!preserveCache) cache.Version = 0;
        else cache.Update((uint)(layerEnd - (int)cacheVersion));
        outPreds.Resize(cache.Predictions.Size);
        outPreds.Copy(cache.Predictions);
    }

    /// <summary>Inplace prediction over an adapter batch.</summary>
    public void InplacePredict<TBatch>(TBatch batch, long nRows, long nCols, MetaInfo info, float missing, HostDeviceVector<float> outPreds,
        int layerBegin, int layerEnd, ColumnsView? cats) where TBatch : IAdapterBatch
    {
        var (treeBegin, treeEnd) = LayerToTree(_model, layerBegin, layerEnd);
        Check.Le(treeEnd, _model.Trees.Count, "Invalid number of trees.");
        new CpuPredictor(Ctx).InplacePredict(batch, nRows, nCols, info, _model, missing, outPreds, treeBegin, treeEnd, cats);
    }

    public override void FeatureScore(string importanceType, ReadOnlySpan<int> treesSpan, List<uint> features, List<float> scores)
    {
        var nFeature = (int)_model.LearnerModelState.NumFeature;
        var splitCounts = new long[nFeature];
        var gainMap = new float[nFeature];
        var trees = treesSpan.IsEmpty ? Enumerable.Range(0, _model.Trees.Count).ToArray() : treesSpan.ToArray();
        var totalNTrees = _model.Trees.Count;

        void AddScore(Action<ITreeView, int, uint> fn)
        {
            foreach (var idx in trees)
            {
                Check.Lt(idx, totalNTrees, "Invalid tree index.");
                var view = _model.Trees[idx].View();
                TreeViews.WalkTree(view, nidx =>
                {
                    if (!view.IsLeaf(nidx))
                    {
                        splitCounts[view.SplitIndex(nidx)]++;
                        fn(view, nidx, view.SplitIndex(nidx));
                    }
                    return true;
                });
            }
        }

        switch (importanceType)
        {
            case "weight":
                AddScore((_, _, split) => gainMap[split] = splitCounts[split]);
                break;
            case "gain" or "total_gain":
                AddScore((tree, nidx, split) => gainMap[split] += tree.LossChg(nidx));
                break;
            case "cover" or "total_cover":
                AddScore((tree, nidx, split) => gainMap[split] += tree.SumHess(nidx));
                break;
            default:
                Check.Fail("Unknown feature importance type, expected one of: {\"weight\", \"total_gain\", \"total_cover\", \"gain\", \"cover\"}, got: " +
                           importanceType);
                break;
        }
        if (importanceType is "gain" or "cover")
            for (var i = 0; i < gainMap.Length; ++i) gainMap[i] /= Math.Max(1.0f, splitCounts[i]);
        features.Clear();
        scores.Clear();
        for (var i = 0; i < splitCounts.Length; ++i)
        {
            if (splitCounts[i] != 0)
            {
                features.Add((uint)i);
                scores.Add(gainMap[i]);
            }
        }
    }

    public override CatContainer Cats() => _model.Cats();

    public override void PredictLeaf(DMatrix fmat, HostDeviceVector<float> outPreds, int layerBegin, int layerEnd, bool strictShape)
    {
        var (treeBegin, treeEnd) = LayerToTree(_model, layerBegin, layerEnd);
        Check.Eq(treeBegin, 0, "Predict leaf supports only iteration end: [0, n_iteration), use model slicing instead.");
        if (strictShape)
            for (var t = treeBegin; t < treeEnd; ++t)
                if (_model.Trees[t].IsMultiTarget)
                    Check.Fail("`strict_shape` with predict leaf is not supported when vector leaf trees are used.");
        new CpuPredictor(Ctx).PredictLeaf(fmat, outPreds, _model, treeEnd);
    }

    public override void PredictContribution(DMatrix fmat, HostDeviceVector<float> outContribs, int layerBegin, int layerEnd,
        bool approximate = false)
    {
        var (treeBegin, treeEnd) = LayerToTree(_model, layerBegin, layerEnd);
        Check.Eq(treeBegin, 0, "Predict contribution supports only iteration end: [0, n_iteration), using model slicing instead.");
        if (approximate) Shap.ApproxFeatureImportance(Ctx, fmat, outContribs, _model, treeEnd, _model.TreeWeights);
        else Shap.ShapValues(Ctx, fmat, outContribs, _model, treeEnd, _model.TreeWeights);
    }

    public override void PredictInteractionContributions(DMatrix fmat, HostDeviceVector<float> outContribs, int layerBegin, int layerEnd,
        bool approximate)
    {
        var (treeBegin, treeEnd) = LayerToTree(_model, layerBegin, layerEnd);
        Check.Eq(treeBegin, 0, "Predict interaction contribution supports only iteration end: [0, n_iteration), using model slicing instead.");
        Shap.ShapInteractionValues(Ctx, fmat, outContribs, _model, treeEnd, _model.TreeWeights, approximate);
    }

    public override List<string> DumpModel(FeatureMap fmap, bool withStats, string format) =>
        _model.DumpModel(fmap, withStats, Ctx.Threads(), format);
}
