// Ports of src/linear/param.h, coordinate_common.h, updater_coordinate.cc, updater_shotgun.cc and
// linear_updater.cc.
using XGBoost.Gbm;

namespace XGBoost.Linear;

public enum FeatureSelectorEnum { Cyclic = 0, Shuffle, Thrifty, Greedy, Random }

public sealed class LinearTrainParam : XGBoostParameter<LinearTrainParam>
{
    public float LearningRate;
    public float RegLambda;
    public float RegAlpha;
    public FeatureSelectorEnum FeatureSelector;
    public float RegLambdaDenorm;
    public float RegAlphaDenorm;

    protected override void Declare(ParamManager<LinearTrainParam> m)
    {
        Field(m, "learning_rate", p => p.LearningRate, (p, v) => p.LearningRate = v).SetLowerBound(0.0f).SetDefault(0.5f)
            .Describe("Learning rate of each update.");
        Field(m, "reg_lambda", p => p.RegLambda, (p, v) => p.RegLambda = v).SetLowerBound(0.0f).SetDefault(0.0f)
            .Describe("L2 regularization on weights.");
        Field(m, "reg_alpha", p => p.RegAlpha, (p, v) => p.RegAlpha = v).SetLowerBound(0.0f).SetDefault(0.0f)
            .Describe("L1 regularization on weights.");
        EnumField<FeatureSelectorEnum>(m, "feature_selector", p => p.FeatureSelector, (p, v) => p.FeatureSelector = v)
            .SetDefault(FeatureSelectorEnum.Cyclic).AddEnum("cyclic", FeatureSelectorEnum.Cyclic).AddEnum("shuffle", FeatureSelectorEnum.Shuffle)
            .AddEnum("thrifty", FeatureSelectorEnum.Thrifty).AddEnum("greedy", FeatureSelectorEnum.Greedy)
            .AddEnum("random", FeatureSelectorEnum.Random).Describe("Feature selection or ordering method.");
        Alias(m, "learning_rate", "eta");
        Alias(m, "reg_lambda", "lambda");
        Alias(m, "reg_alpha", "alpha");
    }

    public void DenormalizePenalties(double sumInstanceWeight)
    {
        RegLambdaDenorm = (float)(RegLambda * sumInstanceWeight);
        RegAlphaDenorm = (float)(RegAlpha * sumInstanceWeight);
    }
}

public sealed class CoordinateParam : XGBoostParameter<CoordinateParam>
{
    public int TopK;

    protected override void Declare(ParamManager<CoordinateParam> m)
    {
        Field(m, "top_k", p => p.TopK, (p, v) => p.TopK = v).SetLowerBound(0).SetDefault(0)
            .Describe("The number of top features to select in 'thrifty' feature_selector. The value of zero means using all the features.");
    }
}

public static class CoordinateCommon
{
    public static double CoordinateDelta(double sumGrad, double sumHess, double w, double regAlpha, double regLambda)
    {
        if (sumHess < 1e-5f) return 0.0f;
        var sumGradL2 = sumGrad + regLambda * w;
        var sumHessL2 = sumHess + regLambda;
        var tmp = w - sumGradL2 / sumHessL2;
        if (tmp >= 0) return XMath.StdMax(-(sumGradL2 + regAlpha) / sumHessL2, -w);
        return XMath.StdMin(-(sumGradL2 - regAlpha) / sumHessL2, -w);
    }

    public static double CoordinateDeltaBias(double sumGrad, double sumHess)
    {
        var b = -sumGrad / sumHess;
        if (double.IsNaN(b) || double.IsInfinity(b)) b = 0;
        return b;
    }

    public static (double, double) GetGradientParallel(Context ctx, int groupIdx, int numGroup, int fidx, GradientPair[] gpair, DMatrix fmat)
    {
        var sumGradTloc = new double[ctx.Threads()];
        var sumHessTloc = new double[ctx.Threads()];
        foreach (var batch in fmat.GetColumnBatches(ctx))
        {
            var col = batch.GetView()[fidx].ToArray();
            Threading.ParallelFor(col.Length, ctx.Threads(), (j, tid) =>
            {
                var v = col[j].Fvalue;
                var p = gpair[col[j].Index * numGroup + groupIdx];
                if (p.Hess < 0.0f) return;
                sumGradTloc[tid] += p.Grad * v;
                sumHessTloc[tid] += p.Hess * v * v;
            });
        }
        double sg = 0.0, sh = 0.0;
        foreach (var v in sumGradTloc) sg += v;
        foreach (var v in sumHessTloc) sh += v;
        return (sg, sh);
    }

    public static (double, double) GetBiasGradientParallel(int groupIdx, int numGroup, GradientPair[] gpair, DMatrix fmat, int nThreads)
    {
        var ndata = fmat.Info.NumRow;
        var sumGradTloc = new double[nThreads];
        var sumHessTloc = new double[nThreads];
        Threading.ParallelFor(ndata, nThreads, (i, tid) =>
        {
            var p = gpair[i * numGroup + groupIdx];
            if (p.Hess >= 0.0f)
            {
                sumGradTloc[tid] += p.Grad;
                sumHessTloc[tid] += p.Hess;
            }
        });
        double sg = 0.0, sh = 0.0;
        foreach (var v in sumGradTloc) sg += v;
        foreach (var v in sumHessTloc) sh += v;
        return (sg, sh);
    }

    public static void UpdateResidualParallel(Context ctx, int fidx, int groupIdx, int numGroup, float dw, GradientPair[] gpair, DMatrix fmat)
    {
        if (dw == 0.0f) return;
        foreach (var batch in fmat.GetColumnBatches(ctx))
        {
            var col = batch.GetView()[fidx].ToArray();
            Threading.ParallelFor(col.Length, ctx.Threads(), j =>
            {
                ref var p = ref gpair[col[j].Index * numGroup + groupIdx];
                if (p.Hess < 0.0f) return;
                p += new GradientPair(p.Hess * col[j].Fvalue * dw, 0);
            });
        }
    }

    public static void UpdateBiasResidualParallel(Context ctx, int groupIdx, int numGroup, float dbias, GradientPair[] gpair, DMatrix fmat)
    {
        if (dbias == 0.0f) return;
        Threading.ParallelFor(fmat.Info.NumRow, ctx.Threads(), i =>
        {
            ref var g = ref gpair[i * numGroup + groupIdx];
            if (g.Hess < 0.0f) return;
            g += new GradientPair(g.Hess * dbias, 0);
        });
    }

    internal static (double, double)[] FeatureSums(Context ctx, DMatrix fmat, GradientPair[] gpair, int ngroup, int nfeat, int? onlyGroup)
    {
        var sums = new (double, double)[nfeat * ngroup];
        foreach (var batch in fmat.GetColumnBatches(ctx))
        {
            var page = batch.GetView();
            Threading.ParallelFor(nfeat, ctx.Threads(), i =>
            {
                var col = page[i];
                for (var gid = 0; gid < ngroup; ++gid)
                {
                    if (onlyGroup is { } og && og != gid) continue;
                    var s = sums[gid * nfeat + i];
                    foreach (var e in col)
                    {
                        var v = e.Fvalue;
                        var p = gpair[e.Index * ngroup + gid];
                        if (p.Hess < 0.0f) continue;
                        s.Item1 += p.Grad * v;
                        s.Item2 += p.Hess * v * v;
                    }
                    sums[gid * nfeat + i] = s;
                }
            });
        }
        return sums;
    }
}

public abstract class FeatureSelector
{
    public virtual void Setup(Context ctx, GBLinearModel model, GradientPair[] gpair, DMatrix fmat, float alpha, float lambda, int param) { }

    public abstract int NextFeature(Context ctx, int iteration, GBLinearModel model, int groupIdx, GradientPair[] gpair, DMatrix fmat,
        float alpha, float lambda);

    public static FeatureSelector Create(FeatureSelectorEnum choice) => choice switch
    {
        FeatureSelectorEnum.Cyclic => new CyclicFeatureSelector(),
        FeatureSelectorEnum.Shuffle => new ShuffleFeatureSelector(),
        FeatureSelectorEnum.Thrifty => new ThriftyFeatureSelector(),
        FeatureSelectorEnum.Greedy => new GreedyFeatureSelector(),
        FeatureSelectorEnum.Random => new RandomFeatureSelector(),
        _ => Check.Fail<FeatureSelector>($"unknown coordinate selector: {(int)choice}"),
    };
}

public sealed class CyclicFeatureSelector : FeatureSelector
{
    public override int NextFeature(Context ctx, int iteration, GBLinearModel model, int groupIdx, GradientPair[] gpair, DMatrix fmat,
        float alpha, float lambda) => (int)((uint)iteration % model.LearnerModelState.NumFeature);
}

public sealed class ShuffleFeatureSelector : FeatureSelector
{
    private uint[] _featIndex = [];

    public override void Setup(Context ctx, GBLinearModel model, GradientPair[] gpair, DMatrix fmat, float alpha, float lambda, int param)
    {
        if (_featIndex.Length == 0)
        {
            _featIndex = new uint[model.LearnerModelState.NumFeature];
            StdAlgo.Iota(_featIndex);
        }
        StdRandom.Shuffle(_featIndex.AsSpan(), ctx.Rng);
    }

    public override int NextFeature(Context ctx, int iteration, GBLinearModel model, int groupIdx, GradientPair[] gpair, DMatrix fmat,
        float alpha, float lambda) => (int)_featIndex[(uint)iteration % model.LearnerModelState.NumFeature];
}

public sealed class RandomFeatureSelector : FeatureSelector
{
    public override int NextFeature(Context ctx, int iteration, GBLinearModel model, int groupIdx, GradientPair[] gpair, DMatrix fmat,
        float alpha, float lambda) => (int)(ctx.Rng.NextUInt() % model.LearnerModelState.NumFeature);
}

public sealed class GreedyFeatureSelector : FeatureSelector
{
    private uint _topK;
    private uint[] _counter = [];

    public override void Setup(Context ctx, GBLinearModel model, GradientPair[] gpair, DMatrix fmat, float alpha, float lambda, int param)
    {
        _topK = unchecked((uint)param);
        var ngroup = model.LearnerModelState.NumOutputGroup;
        if (param <= 0) _topK = uint.MaxValue;
        if (_counter.Length == 0) _counter = new uint[ngroup];
        Array.Clear(_counter);
    }

    public override int NextFeature(Context ctx, int iteration, GBLinearModel model, int groupIdx, GradientPair[] gpair, DMatrix fmat,
        float alpha, float lambda)
    {
        var k = _counter[groupIdx]++;
        if (k >= _topK || _counter[groupIdx] == model.LearnerModelState.NumFeature) return -1;
        var ngroup = (int)model.LearnerModelState.NumOutputGroup;
        var nfeat = (int)model.LearnerModelState.NumFeature;
        var sums = CoordinateCommon.FeatureSums(ctx, fmat, gpair, ngroup, nfeat, groupIdx);
        var bestFidx = 0;
        var bestWeightUpdate = 0.0;
        for (var fidx = 0; fidx < nfeat; ++fidx)
        {
            var s = sums[groupIdx * nfeat + fidx];
            var dw = MathF.Abs((float)CoordinateCommon.CoordinateDelta(s.Item1, s.Item2, model.Weight(fidx, groupIdx), alpha, lambda));
            if (dw > bestWeightUpdate)
            {
                bestWeightUpdate = dw;
                bestFidx = fidx;
            }
        }
        return bestFidx;
    }
}

public sealed class ThriftyFeatureSelector : FeatureSelector
{
    private uint _topK;
    private float[] _deltaw = [];
    private long[] _sortedIdx = [];
    private uint[] _counter = [];

    public override void Setup(Context ctx, GBLinearModel model, GradientPair[] gpair, DMatrix fmat, float alpha, float lambda, int param)
    {
        _topK = unchecked((uint)param);
        if (param <= 0) _topK = uint.MaxValue;
        var ngroup = (int)model.LearnerModelState.NumOutputGroup;
        var nfeat = (int)model.LearnerModelState.NumFeature;
        if (_deltaw.Length == 0)
        {
            _deltaw = new float[nfeat * ngroup];
            _sortedIdx = new long[nfeat * ngroup];
            _counter = new uint[ngroup];
        }
        var sums = CoordinateCommon.FeatureSums(ctx, fmat, gpair, ngroup, nfeat, null);
        Array.Clear(_deltaw);
        StdAlgo.Iota(_sortedIdx);
        var pdeltaw = _deltaw;
        for (var gid = 0; gid < ngroup; ++gid)
        {
            for (var i = 0; i < nfeat; ++i)
            {
                var ii = gid * nfeat + i;
                var s = sums[ii];
                _deltaw[ii] = (float)CoordinateCommon.CoordinateDelta(s.Item1, s.Item2, model.Weight(i, gid), alpha, lambda);
            }
            StdAlgo.Sort<long>(_sortedIdx.AsSpan(gid * nfeat, nfeat), (i, j) => MathF.Abs(pdeltaw[i]) > MathF.Abs(pdeltaw[j]));
            _counter[gid] = 0u;
        }
    }

    public override int NextFeature(Context ctx, int iteration, GBLinearModel model, int groupIdx, GradientPair[] gpair, DMatrix fmat,
        float alpha, float lambda)
    {
        var k = _counter[groupIdx]++;
        if (k >= _topK || _counter[groupIdx] == model.LearnerModelState.NumFeature) return -1;
        var grpOffset = (long)groupIdx * model.LearnerModelState.NumFeature;
        return (int)(_sortedIdx[grpOffset + k] - grpOffset);
    }
}

public sealed class CoordinateUpdater : LinearUpdater
{
    private readonly CoordinateParam _cparam = new();
    private readonly LinearTrainParam _tparam = new();
    private FeatureSelector _selector = null!;

    public override SortedSet<string> Configure(Args args)
    {
        var rest = _tparam.UpdateAllowUnknown(args);
        var used = ParameterUtils.GetUsedParameters(args, rest);
        used.UnionWith(ParameterUtils.GetUsedParameters(rest, _cparam.UpdateAllowUnknown(rest)));
        _selector = FeatureSelector.Create(_tparam.FeatureSelector);
        return used;
    }

    public override void LoadConfig(Json input)
    {
        _tparam.FromJson(input.AsObject["linear_train_param"]);
        _cparam.FromJson(input.AsObject["coordinate_param"]);
    }

    public override void SaveConfig(JsonObject output)
    {
        output["linear_train_param"] = _tparam.ToJson();
        output["coordinate_param"] = _cparam.ToJson();
    }

    public override void Update(Tensor<GradientPair> inGpair, DMatrix fmat, GBLinearModel model, double sumInstanceWeight)
    {
        var gpair = inGpair.Data.RawArray;
        _tparam.DenormalizePenalties(sumInstanceWeight);
        var ngroup = (int)model.LearnerModelState.NumOutputGroup;
        for (var g = 0; g < ngroup; ++g)
        {
            var (sg, sh) = CoordinateCommon.GetBiasGradientParallel(g, ngroup, gpair, fmat, Ctx.Threads());
            var dbias = (float)(_tparam.LearningRate * CoordinateCommon.CoordinateDeltaBias(sg, sh));
            model.Bias(g) += dbias;
            CoordinateCommon.UpdateBiasResidualParallel(Ctx, g, ngroup, dbias, gpair, fmat);
        }
        _selector.Setup(Ctx, model, gpair, fmat, _tparam.RegAlphaDenorm, _tparam.RegLambdaDenorm, _cparam.TopK);
        for (var g = 0; g < ngroup; ++g)
        {
            for (var i = 0u; i < model.LearnerModelState.NumFeature; i++)
            {
                var fidx = _selector.NextFeature(Ctx, (int)i, model, g, gpair, fmat, _tparam.RegAlphaDenorm, _tparam.RegLambdaDenorm);
                if (fidx < 0) break;
                UpdateFeature(fidx, g, gpair, fmat, model);
            }
        }
    }

    private void UpdateFeature(int fidx, int groupIdx, GradientPair[] gpair, DMatrix fmat, GBLinearModel model)
    {
        var ngroup = (int)model.LearnerModelState.NumOutputGroup;
        ref var w = ref model.Weight(fidx, groupIdx);
        var (sg, sh) = CoordinateCommon.GetGradientParallel(Ctx, groupIdx, ngroup, fidx, gpair, fmat);
        var dw = (float)(_tparam.LearningRate * CoordinateCommon.CoordinateDelta(sg, sh, w, _tparam.RegAlphaDenorm, _tparam.RegLambdaDenorm));
        w += dw;
        CoordinateCommon.UpdateResidualParallel(Ctx, fidx, groupIdx, ngroup, dw, gpair, fmat);
    }
}

public sealed class ShotgunUpdater : LinearUpdater
{
    private readonly LinearTrainParam _param = new();
    private FeatureSelector _selector = null!;

    public override SortedSet<string> Configure(Args args)
    {
        var used = ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));
        if (_param.FeatureSelector != FeatureSelectorEnum.Cyclic && _param.FeatureSelector != FeatureSelectorEnum.Shuffle)
            Check.Fail("Unsupported feature selector for shotgun updater.\nSupported options are: {cyclic, shuffle}");
        _selector = FeatureSelector.Create(_param.FeatureSelector);
        return used;
    }

    public override void LoadConfig(Json input) => _param.FromJson(input.AsObject["linear_train_param"]);

    public override void SaveConfig(JsonObject output) => output["linear_train_param"] = _param.ToJson();

    public override void Update(Tensor<GradientPair> inGpair, DMatrix fmat, GBLinearModel model, double sumInstanceWeight)
    {
        var gpair = inGpair.Data.RawArray;
        _param.DenormalizePenalties(sumInstanceWeight);
        var ngroup = (int)model.LearnerModelState.NumOutputGroup;
        for (var gid = 0; gid < ngroup; ++gid)
        {
            var (sg, sh) = CoordinateCommon.GetBiasGradientParallel(gid, ngroup, gpair, fmat, Ctx.Threads());
            var dbias = (float)(_param.LearningRate * CoordinateCommon.CoordinateDeltaBias(sg, sh));
            model.Bias(gid) += dbias;
            CoordinateCommon.UpdateBiasResidualParallel(Ctx, gid, ngroup, dbias, gpair, fmat);
        }
        _selector.Setup(Ctx, model, gpair, fmat, _param.RegAlphaDenorm, _param.RegLambdaDenorm, 0);
        foreach (var batch in fmat.GetColumnBatches(Ctx))
        {
            var page = batch.GetView();
            var nfeat = page.Size;
            Threading.ParallelFor(nfeat, Ctx.Threads(), i =>
            {
                var ii = _selector.NextFeature(Ctx, (int)i, model, 0, gpair, fmat, _param.RegAlphaDenorm, _param.RegLambdaDenorm);
                if (ii < 0) return;
                var col = page[ii];
                for (var gid = 0; gid < ngroup; ++gid)
                {
                    double sumGrad = 0.0, sumHess = 0.0;
                    foreach (var c in col)
                    {
                        var p = gpair[c.Index * ngroup + gid];
                        if (p.Hess < 0.0f) continue;
                        var v = c.Fvalue;
                        sumGrad += p.Grad * v;
                        sumHess += p.Hess * v * v;
                    }
                    ref var w = ref model.Weight(ii, gid);
                    var dw = (float)(_param.LearningRate *
                                     CoordinateCommon.CoordinateDelta(sumGrad, sumHess, w, _param.RegAlphaDenorm, _param.RegLambdaDenorm));
                    if (dw == 0.0f) continue;
                    w += dw;
                    foreach (var c in col)
                    {
                        ref var p = ref gpair[c.Index * ngroup + gid];
                        if (p.Hess < 0.0f) continue;
                        p += new GradientPair(p.Hess * c.Fvalue * dw, 0);
                    }
                }
            });
        }
    }
}

internal static class LinearUpdaterRegistry
{
    public static void Register()
    {
        Registry.LinearUpdaterFactories["coord_descent"] = () => new CoordinateUpdater();
        Registry.LinearUpdaterFactories["shotgun"] = () => new ShotgunUpdater();
    }
}
