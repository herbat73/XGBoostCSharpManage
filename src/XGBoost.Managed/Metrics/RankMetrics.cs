// Port of src/metric/rank_metric.cc (CPU).
using System.Runtime.CompilerServices;
using XGBoost.Collective;

namespace XGBoost.Metrics;

/// <summary><c>ams@k</c>.</summary>
public sealed class EvalAMS : MetricNoCache
{
    private readonly string _name;
    private readonly float _ratio;

    public EvalAMS(string? param)
    {
        Check.That(param is not null, "AMS must be in format ams@k");
        Format.TryParseFloatPrefix(param!.TrimStart(), out var d, out _);
        _ratio = (float)d;
        _name = "ams@" + MetricUtils.StreamFloat(_ratio);
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        MetricUtils.CheckRowWeights(info);
        Check.That(!Communicator.IsDistributed(), "metric AMS do not support distributed evaluation");
        var ndata = (uint)info.Labels.Size;
        var rec = new (float First, uint Second)[ndata];
        var hPreds = preds.RawArray;
        Threading.ParallelFor(ndata, Ctx.Threads(), i => rec[i] = (hPreds[i], (uint)i));
        StdAlgo.Sort<(float First, uint Second)>(rec, static (l, r) => l.First > r.First);
        var ntop = (uint)(_ratio * ndata);
        if (ntop == 0) ntop = ndata;
        const double br = 10.0;
        uint thresindex = 0;
        double sTp = 0.0, bFp = 0.0, tams = 0.0;
        var labels = info.Labels.HostView();
        for (uint i = 0; i < unchecked(ndata - 1) && i < ntop; ++i)
        {
            var ridx = rec[i].Second;
            var wt = info.GetWeight(ridx);
            if (labels.Flat(ridx) > 0.5f) sTp += wt;
            else bFp += wt;
            if (rec[i].First != rec[i + 1].First)
            {
                var ams = Math.Sqrt(2 * ((sTp + bFp + br) * Math.Log(1.0 + sTp / (bFp + br)) - sTp));
                if (tams < ams)
                {
                    thresindex = i;
                    tams = ams;
                }
            }
        }
        if (ntop == ndata)
        {
            Log.Info("best-ams-ratio=" + MetricUtils.StreamFloat((float)thresindex / ndata));
            return (float)tams;
        }
        return (float)Math.Sqrt(2 * ((sTp + bFp + br) * Math.Log(1.0 + sTp / (bFp + br)) - sTp));
    }

    public override string Name => _name;
}

/// <summary><c>cox-nloglik</c>.</summary>
public sealed class EvalCox : MetricNoCache
{
    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        Check.That(!Communicator.IsDistributed(), "Cox metric does not support distributed evaluation");
        var ndata = (uint)info.Labels.Size;
        var labelOrder = info.LabelAbsSort();
        var expPSum = 0.0;
        var hPreds = preds.RawArray;
        for (long i = 0; i < ndata; ++i) expPSum += hPreds[i];
        var output = 0.0;
        var accumulatedSum = 0.0;
        uint numEvents = 0;
        var labels = info.Labels.HostView();
        for (uint i = 0; i < ndata; ++i)
        {
            var ind = labelOrder[i];
            var label = labels.Flat(ind);
            if (label > 0)
            {
                output -= Math.Log(hPreds[ind]) - Math.Log(expPSum);
                ++numEvents;
            }
            accumulatedSum += hPreds[ind];
            if (i == ndata - 1 || MathF.Abs(label) < MathF.Abs(labels.Flat(labelOrder[i + 1])))
            {
                expPSum -= accumulatedSum;
                accumulatedSum = 0;
            }
        }
        return output / numEvents;
    }

    public override string Name => "cox-nloglik";
}

/// <summary><c>EvalRankWithCache&lt;Cache&gt;</c>.</summary>
public abstract class EvalRankWithCache<[System.Diagnostics.CodeAnalysis.DynamicallyAccessedMembers(System.Diagnostics.CodeAnalysis.DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)] TCache> : Metric
    where TCache : RankingCache
{
    protected readonly LambdaRankParam Param = new();
    protected readonly bool Minus;
    private readonly string _name;
    private readonly ConditionalWeakTable<DMatrix, TCache> _cache = new();

    protected EvalRankWithCache(string name, string? param)
    {
        var topn = LambdaRankParam.NotSet;
        var minus = false;
        _name = Ltr.ParseMetricName(name, param, ref topn, ref minus);
        Minus = minus;
        if (topn != LambdaRankParam.NotSet)
        {
            Param.UpdateAllowUnknown([
                new("lambdarank_num_pair_per_sample", topn.ToString(System.Globalization.CultureInfo.InvariantCulture)),
                new("lambdarank_pair_method", "topk"),
            ]);
        }
        Param.UpdateAllowUnknown([]);
    }

    protected abstract TCache MakeCache(MetaInfo info);

    public override void LoadConfig(Json input)
    {
        if (input is JsonNull) return;
        if (input.AsObject.TryGetValue("lambdarank_param", out var p)) Param.FromJson(p);
    }

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = Name;
        output["lambdarank_param"] = Param.ToJson();
    }

    public override double Evaluate(HostDeviceVector<float> preds, DMatrix fmat)
    {
        var info = fmat.Info;
        if (!info.Labels.Empty) Check.Eq(info.Labels.Shape(1), 1L, "Ranking metrics do not support multi-target labels.");
        Check.Eq(preds.Size, info.Labels.Size);
        if (!_cache.TryGetValue(fmat, out var pCache))
        {
            pCache = MakeCache(info);
            _cache.AddOrUpdate(fmat, pCache);
        }
        if (!pCache.Param.Equals(Param))
        {
            pCache = MakeCache(info);
            _cache.AddOrUpdate(fmat, pCache);
        }
        Check.That(pCache.Param.Equals(Param));
        return Eval(preds, info, pCache);
    }

    public override string Name => _name;

    public abstract double Eval(HostDeviceVector<float> preds, MetaInfo info, TCache pCache);

    protected static double Finalize(double score, double sw)
    {
        (score, sw) = MetricUtils.GlobalSum(score, sw);
        if (sw > 0.0) score /= sw;
        Check.Le(score, 1.0 + Constants.RtEps, "Invalid output score, might be caused by invalid query group weight.");
        return XMath.StdMin(1.0, score);
    }
}

public sealed class EvalPrecision(string name, string? param) : EvalRankWithCache<PreCache>(name, param)
{
    protected override PreCache MakeCache(MetaInfo info) => new(Ctx, info, Param);

    public override double Eval(HostDeviceVector<float> predt, MetaInfo info, PreCache pCache)
    {
        var nGroups = pCache.Groups;
        if (!info.Weights.Empty) Check.Eq(info.Weights.Size, nGroups, ErrorMsg.GroupWeight);
        var gptr = pCache.GroupPtrArray;
        var hLabel = info.Labels.HostView().Slice(SliceArg.All, 0).ToArray();
        var rankIdx = pCache.SortedIdx(Ctx, predt.RawArray, predt.Size);
        var weight = new OptionalWeights(info.Weights);
        var pre = pCache.Pre;
        long topK = Param.TopK;
        Threading.ParallelFor(nGroups, Ctx.Threads(), g =>
        {
            var begin = (int)gptr[g];
            var cnt = (int)(gptr[g + 1] - gptr[g]);
            var n = Math.Min(topK, cnt);
            var nHits = 0.0;
            for (var i = 0; i < n; ++i) nHits += hLabel[begin + rankIdx[begin + i]] * weight[g];
            pre[g] = nHits / n;
        });
        var sw = 0.0;
        for (var i = 0; i < pre.Length; ++i) sw += weight[i];
        var sum = 0.0;
        foreach (var v in pre) sum += v;
        return Finalize(sum, sw);
    }
}

public sealed class EvalNDCG(string name, string? param) : EvalRankWithCache<NDCGCache>(name, param)
{
    protected override NDCGCache MakeCache(MetaInfo info) => new(Ctx, info, Param);

    public override SortedSet<string> Configure(Args args)
    {
        var used = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var (key, value) in args)
        {
            if (key == "ndcg_exp_gain")
            {
                Param.UpdateAllowUnknown([new(key, value)]);
                used.Add(key);
            }
        }
        return used;
    }

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info, NDCGCache pCache)
    {
        var groupPtr = pCache.GroupPtrArray;
        var nGroups = groupPtr.Length - 1;
        var ndcgGloc = pCache.Dcg;
        Array.Clear(ndcgGloc);
        var hInvIdcg = pCache.InvIDCG;
        var pDiscount = pCache.Discount;
        var hLabel = info.Labels.HostView().Slice(SliceArg.All, 0).ToArray();
        var hPredt = preds.RawArray;
        var weights = new OptionalWeights(info.Weights);
        long topK = Param.TopK;
        var expGain = Param.NdcgExpGain;
        var minus = Minus;
        Threading.ParallelFor(nGroups, Ctx.Threads(), g =>
        {
            var begin = (int)groupPtr[g];
            var cnt = (int)(groupPtr[g + 1] - groupPtr[g]);
            var gLabels = hLabel.AsSpan(begin, cnt);
            var sortedIdx = StdAlgo.ArgSort<float>(hPredt.AsSpan(begin, cnt), static (l, r) => l > r);
            var ndcg = 0.0;
            var invIdcg = hInvIdcg[g];
            if (invIdcg <= 0.0)
            {
                ndcgGloc[g] = minus ? 0.0 : 1.0;
                return;
            }
            var n = Math.Min(sortedIdx.Length, topK);
            if (expGain)
            {
                for (var i = 0; i < n; ++i) ndcg += pDiscount[i] * Ltr.CalcDCGGain(Ltr.ToRel(gLabels[sortedIdx[i]])) * invIdcg;
            }
            else
            {
                for (var i = 0; i < n; ++i) ndcg += pDiscount[i] * gLabels[sortedIdx[i]] * invIdcg;
            }
            ndcgGloc[g] += ndcg * weights[g];
        });
        double sumW;
        if (weights.Empty)
        {
            sumW = nGroups;
        }
        else
        {
            sumW = 0.0;
            foreach (var w in weights.Data) sumW += w;
        }
        var total = 0.0;
        foreach (var v in ndcgGloc) total += v;
        return Finalize(total, sumW);
    }
}

public sealed class EvalMAPScore(string name, string? param) : EvalRankWithCache<MAPCache>(name, param)
{
    protected override MAPCache MakeCache(MetaInfo info) => new(Ctx, info, Param);

    public override double Eval(HostDeviceVector<float> predt, MetaInfo info, MAPCache pCache)
    {
        var gptr = pCache.GroupPtrArray;
        var hLabel = info.Labels.HostView().Slice(SliceArg.All, 0).ToArray();
        var mapGloc = pCache.Map;
        Array.Clear(mapGloc);
        var rankIdx = pCache.SortedIdx(Ctx, predt.RawArray, predt.Size);
        long topK = Param.TopK;
        var minus = Minus;
        Threading.ParallelFor(pCache.Groups, Ctx.Threads(), g =>
        {
            var begin = (int)gptr[g];
            var cnt = (int)(gptr[g + 1] - gptr[g]);
            var n = Math.Min(topK, cnt);
            var nHits = 0.0;
            for (var i = 0; i < n; ++i)
            {
                double p = hLabel[begin + rankIdx[begin + i]];
                nHits += p;
                mapGloc[g] += nHits / (i + 1) * p;
            }
            for (var i = (int)n; i < cnt; ++i) nHits += hLabel[begin + rankIdx[begin + i]];
            if (nHits > 0.0) mapGloc[g] /= XMath.StdMin(nHits, (double)topK);
            else mapGloc[g] = minus ? 0.0 : 1.0;
        });
        var sw = 0.0;
        var weight = new OptionalWeights(info.Weights);
        if (!weight.Empty) Check.Eq(weight.Size, pCache.Groups);
        for (var i = 0; i < mapGloc.Length; ++i)
        {
            mapGloc[i] *= weight[i];
            sw += weight[i];
        }
        var sum = 0.0;
        foreach (var v in mapGloc) sum += v;
        return Finalize(sum, sw);
    }
}
