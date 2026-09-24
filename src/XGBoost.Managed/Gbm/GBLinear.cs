// Ports of src/gbm/gblinear_model.h/.cc and gblinear.cc.
using System.Text;
using XGBoost.Tree;

namespace XGBoost.Gbm;

public sealed class GBLinearModel(LearnerModelState learnerModelState)
{
    public int NumBoostedRounds;
    public LearnerModelState LearnerModelState { get; } = learnerModelState;
    public float[] WeightArr = [];

    public void LazyInitModel()
    {
        if (WeightArr.Length != 0) return;
        WeightArr = new float[(LearnerModelState.NumFeature + 1) * LearnerModelState.NumOutputGroup];
    }

    public ref float Bias(int gid) => ref WeightArr[LearnerModelState.NumFeature * LearnerModelState.NumOutputGroup + gid];
    public ref float Weight(int fidx, int gid) => ref WeightArr[fidx * LearnerModelState.NumOutputGroup + gid];

    public GBLinearModel Clone() => new(LearnerModelState) { NumBoostedRounds = NumBoostedRounds, WeightArr = (float[])WeightArr.Clone() };

    public void SaveModel(JsonObject output)
    {
        output["weights"] = new F32Array((float[])WeightArr.Clone());
        output["boosted_rounds"] = new JsonInteger(NumBoostedRounds);
    }

    public void LoadModel(Json input)
    {
        var obj = input.AsObject;
        var w = obj["weights"];
        WeightArr = w is F32Array f ? (float[])f.Values.Clone() : [.. w.AsArray.Select(v => v.AsNumber)];
        NumBoostedRounds = obj.TryGetValue("boosted_rounds", out var br) ? (int)br.AsInteger : 0;
    }

    public List<string> DumpModel(string format)
    {
        var ngroup = (int)LearnerModelState.NumOutputGroup;
        var nfeature = (int)LearnerModelState.NumFeature;
        var fo = new StringBuilder();
        static string S(float v) => Format.G(v, 6);
        if (format == "json")
        {
            fo.Append("  { \"bias\": [\n");
            for (var gid = 0; gid < ngroup; ++gid)
            {
                if (gid != 0) fo.Append(",\n");
                fo.Append("      ").Append(S(Bias(gid)));
            }
            fo.Append("\n    ],\n    \"weight\": [\n");
            for (var i = 0; i < nfeature; ++i)
            {
                for (var gid = 0; gid < ngroup; ++gid)
                {
                    if (i != 0 || gid != 0) fo.Append(",\n");
                    fo.Append("      ").Append(S(Weight(i, gid)));
                }
            }
            fo.Append("\n    ]\n  }");
        }
        else if (format == "text")
        {
            fo.Append("bias:\n");
            for (var gid = 0; gid < ngroup; ++gid) fo.Append(S(Bias(gid))).Append('\n');
            fo.Append("weight:\n");
            for (var i = 0; i < nfeature; ++i)
                for (var gid = 0; gid < ngroup; ++gid) fo.Append(S(Weight(i, gid))).Append('\n');
        }
        else
        {
            Check.Fail($"Dump format `{format}` is not supported by the gblinear model.");
        }
        return [fo.ToString()];
    }
}

public sealed class GBLinearTrainParam : XGBoostParameter<GBLinearTrainParam>
{
    public string Updater = "shotgun";
    public float Tolerance;
    public ulong MaxRowPerbatch;

    protected override void Declare(ParamManager<GBLinearTrainParam> m)
    {
        Field(m, "updater", p => p.Updater, (p, v) => p.Updater = v).SetDefault("shotgun")
            .Describe("Update algorithm for linear model. One of shotgun/coord_descent");
        Field(m, "tolerance", p => p.Tolerance, (p, v) => p.Tolerance = v).SetLowerBound(0.0f).SetDefault(0.0f)
            .Describe("Stop if largest weight update is smaller than this number.");
        Field(m, "max_row_perbatch", p => p.MaxRowPerbatch, (p, v) => p.MaxRowPerbatch = v).SetDefault(ulong.MaxValue)
            .Describe("Maximum rows per batch.");
    }
}

public sealed class GBLinear(LearnerModelState learnerModelState, Context ctx) : GradientBooster(ctx)
{
    private GBLinearModel _model = new(learnerModelState);
    private GBLinearModel _previousModel = new(learnerModelState);
    private readonly GBLinearTrainParam _param = new();
    private LinearUpdater? _updater;
    private double _sumInstanceWeight;
    private bool _sumWeightComplete;
    private bool _isConverged;

    private static void LinearCheckLayer(int layerBegin) => Check.Eq(layerBegin, 0, "Linear booster does not support prediction range.");

    public override SortedSet<string> Configure(Args cfg)
    {
        var used = ParameterUtils.GetUsedParameters(cfg, _param.UpdateAllowUnknown(cfg));
        if (_param.Updater == "gpu_coord_descent")
            Check.Fail(ErrorMsg.DeprecatedFunc("gpu_coord_descent", "2.0.0", "device=\"cuda\", updater=\"coord_descent\""));
        var name = _param.Updater;
        Log.Info("Using the updater:" + name);
        _updater = Registry.CreateLinearUpdater(name, Ctx);
        used.UnionWith(_updater.Configure(cfg));
        return used;
    }

    public override int BoostedRounds => _model.NumBoostedRounds;

    public override void SaveModel(JsonObject output)
    {
        output["name"] = "gblinear";
        var m = new JsonObject();
        _model.SaveModel(m);
        output["model"] = m;
    }

    public override void LoadModel(Json input)
    {
        Check.Eq(input["name"].AsString, "gblinear");
        _model.LoadModel(input["model"]);
    }

    public override void LoadConfig(Json input)
    {
        Check.Eq(input["name"].AsString, "gblinear");
        _param.FromJson(input["gblinear_train_param"]);
        _updater = Registry.CreateLinearUpdater(_param.Updater, Ctx);
        _updater.LoadConfig(input["updater"]);
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = "gblinear";
        output["gblinear_train_param"] = _param.ToJson();
        var j = new JsonObject();
        Check.That(_updater is not null);
        _updater!.SaveConfig(j);
        output["updater"] = j;
    }

    public override void DoBoost(DMatrix fmat, GradientContainer inGpair, ObjFunction obj)
    {
        if (inGpair.HasValueGrad) Check.Fail("Multi-target with reduced gradient is not implemented for the current booster.");
        Check.That(!fmat.Info.HasCategorical, ErrorMsg.NoCategorical("`gblinear`"));
        _model.LazyInitModel();
        LazySumWeights(fmat);
        if (!CheckConvergence()) _updater!.Update(inGpair.Grad, fmat, _model, _sumInstanceWeight);
        _model.NumBoostedRounds++;
    }

    public override void PredictBatch(DMatrix fmat, HostDeviceVector<float> outPreds, bool training, int layerBegin, int layerEnd)
    {
        LinearCheckLayer(layerBegin);
        _model.LazyInitModel();
        var baseMargin = fmat.Info.BaseMargin.HostView();
        var ngroup = (int)_model.LearnerModelState.NumOutputGroup;
        outPreds.Resize((int)(fmat.Info.NumRow * ngroup));
        var preds = outPreds.RawArray;
        var baseScore = _model.LearnerModelState.BaseScore();
        var nfeat = _model.LearnerModelState.NumFeature;
        foreach (var page in fmat.GetRowBatches())
        {
            var batch = page.GetView();
            var nsize = batch.Size;
            if (baseMargin.Size != 0) Check.Eq(baseMargin.Size, nsize * ngroup);
            Threading.ParallelFor(nsize, Ctx.Threads(), i =>
            {
                var ridx = page.BaseRowId + i;
                var inst = batch[i];
                for (var gid = 0; gid < ngroup; ++gid)
                {
                    var margin = baseMargin.Size != 0 ? baseMargin[ridx, gid] : baseScore[0];
                    var psum = _model.Bias(gid) + margin;
                    foreach (var ins in inst)
                    {
                        if (ins.Index >= nfeat) continue;
                        psum += ins.Fvalue * _model.Weight((int)ins.Index, gid);
                    }
                    preds[ridx * ngroup + gid] = psum;
                }
            });
        }
    }

    public override void PredictLeaf(DMatrix dmat, HostDeviceVector<float> outPreds, int layerBegin, int layerEnd, bool strictShape) =>
        Check.Fail("gblinear does not support prediction of leaf index");

    public override void PredictContribution(DMatrix fmat, HostDeviceVector<float> outContribs, int layerBegin, int layerEnd,
        bool approximate = false)
    {
        _model.LazyInitModel();
        LinearCheckLayer(layerBegin);
        var baseMargin = fmat.Info.BaseMargin.HostView();
        var ngroup = (int)_model.LearnerModelState.NumOutputGroup;
        var ncolumns = (int)_model.LearnerModelState.NumFeature + 1;
        outContribs.Resize((int)(fmat.Info.NumRow * ncolumns * ngroup));
        outContribs.Fill(0);
        var contribs = outContribs.RawArray;
        var baseScore = _model.LearnerModelState.BaseScore();
        var nfeat = _model.LearnerModelState.NumFeature;
        foreach (var batch in fmat.GetRowBatches())
        {
            var page = batch.GetView();
            Threading.ParallelFor(page.Size, Ctx.Threads(), i =>
            {
                var inst = page[i];
                var rowIdx = batch.BaseRowId + i;
                for (var gid = 0; gid < ngroup; ++gid)
                {
                    var off = (int)((rowIdx * ngroup + gid) * ncolumns);
                    foreach (var ins in inst)
                    {
                        if (ins.Index >= nfeat) continue;
                        contribs[off + ins.Index] = ins.Fvalue * _model.Weight((int)ins.Index, gid);
                    }
                    contribs[off + ncolumns - 1] = _model.Bias(gid) + (baseMargin.Size != 0 ? baseMargin[rowIdx, gid] : baseScore[0]);
                }
            });
        }
    }

    public override void PredictInteractionContributions(DMatrix fmat, HostDeviceVector<float> outContribs, int layerBegin, int layerEnd,
        bool approximate)
    {
        LinearCheckLayer(layerBegin);
        var nelements = (long)_model.LearnerModelState.NumFeature * _model.LearnerModelState.NumFeature;
        outContribs.Resize((int)(fmat.Info.NumRow * nelements * _model.LearnerModelState.NumOutputGroup));
        outContribs.Fill(0);
    }

    public override List<string> DumpModel(FeatureMap fmap, bool withStats, string format) => _model.DumpModel(format);

    public override void FeatureScore(string importanceType, ReadOnlySpan<int> trees, List<uint> features, List<float> scores)
    {
        Check.That(_model.WeightArr.Length != 0, "Model is not initialized");
        Check.That(trees.IsEmpty, "gblinear doesn't support number of trees for feature importance.");
        Check.Eq(importanceType, "weight", "gblinear only has `weight` defined for feature importance.");
        var nfeat = (int)_model.LearnerModelState.NumFeature;
        var nGroups = (int)_model.LearnerModelState.NumOutputGroup;
        features.Clear();
        for (var i = 0; i < nfeat; ++i) features.Add((uint)i);
        scores.Clear();
        for (var i = 0; i < nfeat; ++i)
            for (var g = 0; g < nGroups; ++g) scores.Add(_model.Weight(i, g));
    }

    private bool CheckConvergence()
    {
        if (_param.Tolerance == 0.0f) return false;
        if (_isConverged) return true;
        if (_previousModel.WeightArr.Length != _model.WeightArr.Length)
        {
            _previousModel = _model.Clone();
            return false;
        }
        var largestDw = 0.0f;
        for (var i = 0; i < _model.WeightArr.Length; i++)
            largestDw = XMath.StdMax(largestDw, MathF.Abs(_model.WeightArr[i] - _previousModel.WeightArr[i]));
        _previousModel = _model.Clone();
        _isConverged = largestDw <= _param.Tolerance;
        return _isConverged;
    }

    private void LazySumWeights(DMatrix fmat)
    {
        if (_sumWeightComplete) return;
        var info = fmat.Info;
        for (long i = 0; i < info.NumRow; i++) _sumInstanceWeight += info.GetWeight(i);
        _sumWeightComplete = true;
    }
}

internal static class BoosterRegistry
{
    public static void Register()
    {
        Registry.BoosterFactories["gbtree"] = (state, ctx) => new GBTree(state, ctx);
        Registry.BoosterFactories["gblinear"] = (state, ctx) => new GBLinear(state, ctx);
    }
}
