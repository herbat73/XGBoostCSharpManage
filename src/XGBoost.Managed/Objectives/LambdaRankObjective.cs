// Port of src/objective/lambdarank_obj.h/.cc (CPU implementation).
namespace XGBoost.Objectives;

public enum LambdaRankKind { NDCG, MAP, Pairwise }

public sealed class LambdaRankObj(LambdaRankKind kind) : ObjFunction
{
    private const double Eps64 = 1e-16;

    private MetaInfo? _pInfo;
    private double[] _li = [];
    private double[] _lj = [];
    private double[] _tiPlus = [];
    private double[] _tjMinus = [];
    private double[] _liFull = [];
    private double[] _ljFull = [];
    private readonly LambdaRankParam _param = new();
    private RankingCache? _pCache;

    /// <summary>Signature of the delta metric: (y_high, y_low, rank_high, rank_low, group).</summary>
    private delegate double DeltaFn(float yHigh, float yLow, long rankHigh, long rankLow, int g);

    public static string NameOf(LambdaRankKind kind) => kind switch
    {
        LambdaRankKind.NDCG => "rank:ndcg",
        LambdaRankKind.MAP => "rank:map",
        _ => "rank:pairwise",
    };

    public string Name => NameOf(kind);

    // ---- helpers from lambdarank_obj.h -------------------------------------------------------

    private static double DeltaNDCG(bool exp, float yHigh, float yLow, long rankHigh, long rankLow, double invIdcg, double[] discount)
    {
        var gainHigh = exp ? Ltr.CalcDCGGain(Ltr.ToRel(yHigh)) : yHigh;
        var discountHigh = discount[rankHigh];
        var gainLow = exp ? Ltr.CalcDCGGain(Ltr.ToRel(yLow)) : yLow;
        var discountLow = discount[rankLow];
        var original = gainHigh * discountHigh + gainLow * discountLow;
        var changed = gainLow * discountHigh + gainHigh * discountLow;
        return (original - changed) * invIdcg;
    }

    private static double DeltaMAP(float yHigh, float yLow, long rankHigh, long rankLow, ReadOnlySpan<double> nRel, ReadOnlySpan<double> acc)
    {
        var rH = rankHigh + 1.0;
        var rL = rankLow + 1.0;
        double delta;
        var nTotalRelevances = nRel[^1];
        var m = nRel[(int)rankLow];
        var n = nRel[(int)rankHigh];
        if (yHigh < yLow)
        {
            var a = m / rL - (n + 1.0) / rH;
            var b = acc[(int)rankLow - 1] - acc[(int)rankHigh];
            delta = (a - b) / nTotalRelevances;
        }
        else
        {
            var a = n / rH - m / rL;
            var b = acc[(int)rankLow - 1] - acc[(int)rankHigh];
            delta = (a + b) / nTotalRelevances;
        }
        return delta;
    }

    private static GradientPair LambdaGrad(bool unbiased, bool normByDiff, ReadOnlySpan<float> labels, ReadOnlySpan<float> predts,
        ReadOnlySpan<int> sortedIdx, long rankHigh, long rankLow, Func<float, float, long, long, double> delta,
        double[] tPlus, double[] tMinus, out double cost)
    {
        cost = 0;
        long idxHigh = sortedIdx[(int)rankHigh];
        long idxLow = sortedIdx[(int)rankLow];
        if (labels[(int)idxHigh] == labels[(int)idxLow]) return new GradientPair(0.0f, 0.0f);

        var bestScore = predts[sortedIdx[0]];
        var worstScore = predts[sortedIdx[^1]];
        var yHigh = labels[(int)idxHigh];
        var sHigh = predts[(int)idxHigh];
        var yLow = labels[(int)idxLow];
        var sLow = predts[(int)idxLow];

        double deltaScore = MathF.Abs(sHigh - sLow);
        double sigmoid = XMath.Sigmoid(sHigh - sLow);
        var deltaMetric = Math.Abs(delta(yHigh, yLow, rankHigh, rankLow));
        if (normByDiff && bestScore != worstScore) deltaMetric /= deltaScore + 0.01;
        if (unbiased) cost = Math.Log(1.0 / (1.0 - sigmoid)) * deltaMetric;

        var lambdaIj = (sigmoid - 1.0) * deltaMetric;
        var hessianIj = XMath.StdMax(sigmoid * (1.0 - sigmoid), Eps64) * deltaMetric * 2.0;
        var k = tPlus.Length;
        if (unbiased && idxHigh < k && idxLow < k && tMinus[idxLow] >= Eps64 && tPlus[idxHigh] >= Eps64)
        {
            lambdaIj /= tPlus[idxHigh] * tMinus[idxLow];
            hessianIj /= tPlus[idxHigh] * tMinus[idxLow];
        }
        return new GradientPair((float)lambdaIj, (float)hessianIj);
    }

    private static GradientPair Repulse(GradientPair pg) => new(-pg.Grad, pg.Hess);

    private void MakePairs(uint seed, int g, ReadOnlySpan<float> gLabel, ReadOnlySpan<int> gRank, Action<long, long> op)
    {
        var groupPtr = _pCache!.DataGroupPtr;
        var cnt = groupPtr[g + 1] - groupPtr[g];
        if (_pCache.Param.HasTruncation)
        {
            var n = Math.Min(cnt, _pCache.Param.NumPair());
            for (long i = 0; i < n; ++i)
                for (var j = i + 1; j < cnt; ++j)
                    op(i, j);
        }
        else
        {
            Check.Eq(gRank.Length, gLabel.Length);
            var rnd = LinearCongruential.MinstdRand(unchecked(seed + (uint)g));
            var keys = new float[cnt];
            for (var idx = 0; idx < cnt; idx++) keys[idx] = gLabel[gRank[idx]];
            var ySortedIdx = StdAlgo.ArgSort<float>(keys, static (l, r) => l > r);
            float RevIt(long idx) => keys[ySortedIdx[idx]];
            for (long i = 0; i < cnt;)
            {
                var j = i + 1;
                while (j < cnt && RevIt(i) == RevIt(j)) ++j;
                var nLefts = (ulong)i;
                var nRights = (ulong)(cnt - j);
                if (nLefts + nRights == 0)
                {
                    i = j;
                    continue;
                }
                var nSamples = _pCache.Param.NumPair();
                while (nSamples-- != 0)
                {
                    for (var pairIdx = i; pairIdx < j; ++pairIdx)
                    {
                        var ridx = StdRandom.UniformInt(rnd, 0, nLefts + nRights - 1);
                        if (ridx >= nLefts) ridx = ridx - (ulong)i + (ulong)j;
                        long idx0 = ySortedIdx[pairIdx];
                        long idx1 = ySortedIdx[ridx];
                        op(idx0, idx1);
                    }
                }
                i = j;
            }
        }
    }

    // ---- LambdaRankObj ------------------------------------------------------------------------

    private void UpdatePositionBias()
    {
        var gptr = _pCache!.DataGroupPtr;
        var nGroups = _pCache.Groups;
        var regularizer = _pCache.Param.Regularizer;
        for (var g = 0; g < nGroups; ++g)
        {
            var begin = (int)gptr[g];
            var end = (int)gptr[g + 1];
            long groupSize = end - begin;
            var n = Math.Min(groupSize, _pCache.MaxPositionSize);
            for (var i = 0; i < n; ++i)
            {
                _li[i] += _liFull[begin + i];
                _lj[i] += _ljFull[begin + i];
            }
        }
        for (var i = 0; i < _tiPlus.Length; ++i)
        {
            if (_li[0] >= Eps64) _tiPlus[i] = Math.Pow(_li[i] / _li[0], regularizer);
            if (_lj[0] >= Eps64) _tjMinus[i] = Math.Pow(_lj[i] / _lj[0], regularizer);
        }
        Array.Clear(_liFull);
        Array.Clear(_ljFull);
        Array.Clear(_li);
        Array.Clear(_lj);
    }

    /// <summary>The per-group slice of a position-bias accumulator, or the whole vector when biased.</summary>
    private Span<double> GroupLoss(int g, double[] v)
    {
        var gptr = _pCache!.DataGroupPtr;
        if (_param.LambdarankUnbiased) return v.AsSpan((int)gptr[g], (int)(gptr[g + 1] - gptr[g]));
        return v;
    }

    private void CalcLambdaForGroup(bool unbiased, bool normByDiff, uint seed, ReadOnlySpan<float> gPredt, ReadOnlySpan<float> gLabel,
        float w, ReadOnlySpan<int> gRank, int g, DeltaFn delta, Span<GradientPair> gGpair)
    {
        gGpair.Clear();
        var tiPlus = _tiPlus;
        var tjMinus = _tjMinus;
        var li = GroupLoss(g, _liFull).ToArray();
        var lj = GroupLoss(g, _ljFull).ToArray();
        var sumLambda = 0.0;

        var labels = gLabel.ToArray();
        var predts = gPredt.ToArray();
        var rank = gRank.ToArray();
        var gpair = new GradientPair[gGpair.Length];
        Func<float, float, long, long, double> deltaOp = (yh, yl, rh, rl) => delta(yh, yl, rh, rl, g);

        MakePairs(seed, g, labels, rank, (i, j) =>
        {
            long rankHigh = i, rankLow = j;
            if (labels[rank[rankHigh]] == labels[rank[rankLow]]) return;
            if (labels[rank[rankHigh]] < labels[rank[rankLow]]) (rankHigh, rankLow) = (rankLow, rankHigh);
            var pg = LambdaGrad(unbiased, normByDiff, labels, predts, rank, rankHigh, rankLow, deltaOp, tiPlus, tjMinus, out var cost);
            var ng = Repulse(pg);
            long idxHigh = rank[rankHigh];
            long idxLow = rank[rankLow];
            gpair[idxHigh] += pg;
            gpair[idxLow] += ng;
            if (unbiased)
            {
                var k = tiPlus.Length;
                if (idxHigh < k && idxLow < k)
                {
                    if (tjMinus[idxLow] >= Eps64) li[idxHigh] += cost / tjMinus[idxLow];
                    if (tiPlus[idxHigh] >= Eps64) lj[idxLow] += cost / tiPlus[idxHigh];
                }
            }
            sumLambda += -2.0 * pg.Grad;
        });

        if (unbiased)
        {
            li.CopyTo(GroupLoss(g, _liFull));
            lj.CopyTo(GroupLoss(g, _ljFull));
        }

        if (_param.LambdarankNormalization)
        {
            var norm = 1.0;
            if (_param.IsMean)
            {
                var nPairs = _pCache!.Param.NumPair();
                norm = 1.0 / nPairs;
            }
            else if (sumLambda > 0.0)
            {
                norm = Math.Log2(1.0 + sumLambda) / sumLambda;
            }
            if (norm != 1.0)
                for (var i = 0; i < gpair.Length; i++) gpair[i] *= (float)norm;
        }
        var wNorm = (float)_pCache!.WeightNorm;
        for (var i = 0; i < gpair.Length; i++) gpair[i] = gpair[i] * w * wNorm;
        gpair.CopyTo(gGpair);
    }

    public override void InitEstimation(MetaInfo info, Tensor<float> baseScore) =>
        baseScore.Assign(Linalg.Zeros<float>(Targets(info)));

    public override SortedSet<string> Configure(Args args) =>
        ParameterUtils.GetUsedParameters(args, _param.UpdateAllowUnknown(args));

    public override void SaveConfig(JsonObject output)
    {
        output["name"] = Name;
        output["lambdarank_param"] = _param.ToJson();
        if (_param.LambdarankUnbiased)
        {
            output["ti+"] = new F32Array(Array.ConvertAll(_tiPlus, v => (float)v));
            output["tj-"] = new F32Array(Array.ConvertAll(_tjMinus, v => (float)v));
        }
    }

    private static double[] LoadVector(Json input) => input switch
    {
        F32Array f => Array.ConvertAll(f.Values, v => (double)v),
        F64Array d => (double[])d.Values.Clone(),
        _ => [.. input.AsArray.Select(v => (double)v.AsNumber)],
    };

    public override void LoadConfig(Json input)
    {
        if (input.AsObject.TryGetValue("lambdarank_param", out var p)) _param.FromJson(p);
        if (_param.LambdarankUnbiased)
        {
            _tiPlus = LoadVector(input["ti+"]);
            _tjMinus = LoadVector(input["tj-"]);
        }
    }

    public override ObjInfo Task => new(ObjTask.Ranking);

    public override uint Targets(MetaInfo info)
    {
        Check.Le(info.Labels.Shape(1), 1L, "multi-output for LTR is not yet supported.");
        return 1;
    }

    private string RankEvalMetric(string metric) =>
        Ltr.MakeMetricName(metric, _param.HasTruncation ? _param.NumPair() : LambdaRankParam.NotSet, false);

    public override string DefaultEvalMetric => kind == LambdaRankKind.MAP ? RankEvalMetric("map") : RankEvalMetric("ndcg");

    public override Json DefaultMetricConfig()
    {
        if (kind == LambdaRankKind.MAP) return JsonNull.Instance;
        var config = new JsonObject();
        config["name"] = DefaultEvalMetric;
        config["lambdarank_param"] = _param.ToJson();
        return config;
    }

    public override void GetGradient(HostDeviceVector<float> predt, MetaInfo info, int iter, Tensor<GradientPair> outGpair)
    {
        Check.Eq((long)info.Labels.Size, (long)predt.Size, ErrorMsg.LabelScoreSize);
        if (_pCache is null || !ReferenceEquals(_pInfo, info) || !_pCache.Param.Equals(_param))
        {
            _pCache = kind switch
            {
                LambdaRankKind.NDCG => new NDCGCache(Ctx, info, _param),
                LambdaRankKind.MAP => new MAPCache(Ctx, info, _param),
                _ => new RankingCache(Ctx, info, _param),
            };
            _pInfo = info;
        }
        var nGroups = _pCache.Groups;
        if (!info.Weights.Empty) Check.Eq((long)info.Weights.Size, (long)nGroups, ErrorMsg.GroupWeight);

        if ((_tiPlus.Length == 0 || _liFull.Length == 0) && _param.LambdarankUnbiased)
        {
            Check.Eq(iter, 0);
            var k = (int)_pCache.MaxPositionSize;
            _tiPlus = Enumerable.Repeat(1.0, k).ToArray();
            _tjMinus = Enumerable.Repeat(1.0, k).ToArray();
            _li = new double[k];
            _lj = new double[k];
            _liFull = new double[info.NumRow];
            _ljFull = new double[info.NumRow];
        }
        uint seed = 0;
        if (_param.IsMean) seed = Ctx.Rng.NextUInt();

        switch (kind)
        {
            case LambdaRankKind.NDCG: GetGradientNDCG(seed, predt, info, outGpair); break;
            case LambdaRankKind.MAP: GetGradientMAP(seed, predt, info, outGpair); break;
            default: GetGradientPairwise(seed, predt, info, outGpair); break;
        }
        if (_param.LambdarankUnbiased) UpdatePositionBias();
    }

    private delegate void GroupFn(int g, int begin, int cnt, float w);

    /// <summary>Common per-group driver for all three losses.</summary>
    private void ForEachGroup(HostDeviceVector<float> predt, MetaInfo info, Tensor<GradientPair> outGpair, Sched sched,
        Func<float[], int[]> prepare, DeltaFn delta)
    {
        var gptr = _pCache!.GroupPtrArray;
        var nGroups = _pCache.Groups;
        outGpair.Reshape(info.NumRow, Targets(info));
        var hGpair = outGpair.Data.RawArray;
        var hPredt = predt.ToArray();
        var hLabel = info.Labels.HostView().Slice(SliceArg.All, 0).ToArray();
        var hWeight = new OptionalWeights(info.Weights);
        var rankIdx = prepare(hPredt);
        var unbiased = _param.LambdarankUnbiased;
        var normByDiff = _param.LambdarankScoreNormalization;
        uint seed = _seed;
        Threading.ParallelFor(nGroups, Ctx.Threads(), sched, g =>
        {
            var gi = (int)g;
            var begin = (int)gptr[gi];
            var cnt = (int)(gptr[gi + 1] - gptr[gi]);
            var w = hWeight[gi];
            CalcLambdaForGroup(unbiased, normByDiff, seed, hPredt.AsSpan(begin, cnt), hLabel.AsSpan(begin, cnt), w,
                rankIdx.AsSpan(begin, cnt), gi, delta, hGpair.AsSpan(begin, cnt));
        });
    }

    private uint _seed;

    private void GetGradientNDCG(uint seed, HostDeviceVector<float> predt, MetaInfo info, Tensor<GradientPair> outGpair)
    {
        _seed = seed;
        var cache = (NDCGCache)_pCache!;
        var dct = cache.Discount;
        var invIdcg = cache.InvIDCG;
        var expGain = _param.NdcgExpGain;
        ForEachGroup(predt, info, outGpair, Sched.Guided(), p => cache.SortedIdx(Ctx, p, p.Length),
            (yh, yl, rh, rl, g) => DeltaNDCG(expGain, yh, yl, rh, rl, invIdcg[g], dct));
    }

    private void GetGradientMAP(uint seed, HostDeviceVector<float> predt, MetaInfo info, Tensor<GradientPair> outGpair)
    {
        _seed = seed;
        var cache = (MAPCache)_pCache!;
        Check.Eq(info.Labels.Shape(1), 1L, "multi-target for learning to rank is not yet supported.");
        var gptr = cache.GroupPtrArray;
        var hLabel = info.Labels.HostView().Slice(SliceArg.All, 0).ToArray();
        var nRel = cache.NumRelevant;
        var acc = cache.Acc;
        ForEachGroup(predt, info, outGpair, Sched.Static(), p =>
        {
            var rank = cache.SortedIdx(Ctx, p, p.Length);
            MAPStat(Ctx, hLabel, rank, cache);
            return rank;
        }, (yh, yl, rh, rl, g) =>
        {
            if (rh > rl)
            {
                (rh, rl) = (rl, rh);
                (yh, yl) = (yl, yh);
            }
            var begin = (int)gptr[g];
            var cnt = (int)(gptr[g + 1] - gptr[g]);
            return DeltaMAP(yh, yl, rh, rl, nRel.AsSpan(begin, cnt), acc.AsSpan(begin, cnt));
        });
    }

    private static void MAPStat(Context ctx, float[] label, int[] rankIdx, MAPCache cache)
    {
        var hNRel = cache.NumRelevant;
        var gptr = cache.GroupPtrArray;
        Check.Eq((long)hNRel.Length, (long)gptr[^1]);
        Check.Eq(hNRel.Length, label.Length);
        var hAcc = cache.Acc;
        Threading.ParallelFor(cache.Groups, ctx.Threads(), g =>
        {
            var begin = (int)gptr[g];
            var cnt = (int)(gptr[g + 1] - gptr[g]);
            var gNRel = hNRel.AsSpan(begin, cnt);
            var gRank = rankIdx.AsSpan(begin, cnt);
            var gLabel = label.AsSpan(begin, cnt);
            gNRel[0] = gLabel[gRank[0]];
            for (var k = 1; k < gRank.Length; ++k) gNRel[k] = gNRel[k - 1] + gLabel[gRank[k]];
            var gAcc = hAcc.AsSpan(begin, cnt);
            gAcc[0] = gLabel[gRank[0]] / 1.0;
            for (var k = 1; k < gRank.Length; ++k) gAcc[k] = gAcc[k - 1] + gLabel[gRank[k]] / (double)(k + 1);
        });
    }

    private void GetGradientPairwise(uint seed, HostDeviceVector<float> predt, MetaInfo info, Tensor<GradientPair> outGpair)
    {
        _seed = seed;
        var cache = _pCache!;
        ForEachGroup(predt, info, outGpair, Sched.Static(), p => cache.SortedIdx(Ctx, p, p.Length), static (_, _, _, _, _) => 1.0);
    }
}
