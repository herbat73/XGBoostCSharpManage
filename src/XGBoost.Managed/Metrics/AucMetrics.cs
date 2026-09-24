// Port of src/metric/auc.h/.cc (CPU).
using XGBoost.Collective;

namespace XGBoost.Metrics;

public enum MultiAUCType : byte { MultiClass, MultiLabel }

/// <summary>Shared evaluation driver of <c>EvalAUC&lt;Curve&gt;</c>.</summary>
public abstract class EvalAUCBase : MetricNoCache
{
    protected delegate (double Fp, double Tp, double Auc) BinaryAucFn(Context ctx, float[] predts, float[] labels, OptionalWeights weights);

    protected static double TrapezoidArea(double x0, double x1, double y0, double y1) => Math.Abs(x0 - x1) * (y0 + y1) * 0.5f;

    private static double CalcH(double fpA, double fpB, double tpA, double tpB) => (fpB - fpA) / (tpB - tpA);
    private static double CalcB(double fpA, double h, double tpA, double totalPos) => (fpA - h * tpA) / totalPos;
    private static double CalcA(double h) => h + 1;

    protected static double CalcDeltaPRAUC(double fpPrev, double fp, double tpPrev, double tp, double totalPos)
    {
        var prPrev = tpPrev / totalPos;
        var pr = tp / totalPos;
        double h, a, b;
        if (tp == tpPrev)
        {
            a = 1.0;
            b = 0.0;
        }
        else
        {
            h = CalcH(fpPrev, fp, tpPrev, tp);
            a = CalcA(h);
            b = CalcB(fpPrev, h, tpPrev, totalPos);
        }
        double area;
        if (b != 0.0) area = (pr - prPrev - b / a * (Math.Log(a * pr + b) - Math.Log(a * prPrev + b))) / a;
        else area = (pr - prPrev) / a;
        return area;
    }

    protected static void InvalidGroupAUC() =>
        Log.Info($"Invalid group with less than 3 samples is found on worker {Communicator.GetRank()}.  Calculating AUC value requires at least 2 pairs of samples.");

    protected static (double, double, double) BinaryAUC(float[] predts, float[] labels, OptionalWeights weights, int[] sortedIdx,
        Func<double, double, double, double, double> areaFn)
    {
        Check.Ne(labels.Length, 0);
        Check.Eq(labels.Length, predts.Length);
        var auc = 0.0;
        var label = labels[sortedIdx[0]];
        var w = weights[sortedIdx[0]];
        double fp = (1.0 - label) * w, tp = label * w;
        double tpPrev = 0, fpPrev = 0;
        for (var i = 1; i < sortedIdx.Length; ++i)
        {
            if (predts[sortedIdx[i]] != predts[sortedIdx[i - 1]])
            {
                auc += areaFn(fpPrev, fp, tpPrev, tp);
                tpPrev = tp;
                fpPrev = fp;
            }
            label = labels[sortedIdx[i]];
            var wi = weights[sortedIdx[i]];
            fp += (1.0f - label) * wi;
            tp += label * wi;
        }
        auc += areaFn(fpPrev, fp, tpPrev, tp);
        if (fp <= 0.0f || tp <= 0.0f)
        {
            auc = 0;
            fp = 0;
            tp = 0;
        }
        return (fp, tp, auc);
    }

    protected static (double, double, double) BinaryROCAUC(Context ctx, float[] predts, float[] labels, OptionalWeights weights)
    {
        var sortedIdx = StdAlgo.ArgSort<float>(predts, static (l, r) => l > r);
        return BinaryAUC(predts, labels, weights, sortedIdx, TrapezoidArea);
    }

    protected static (double, double, double) BinaryPRAUC(Context ctx, float[] predts, float[] labels, OptionalWeights weights)
    {
        var sortedIdx = StdAlgo.ArgSort<float>(predts, static (l, r) => l > r);
        double totalPos = 0, totalNeg = 0;
        for (var i = 0; i < labels.Length; ++i)
        {
            var w = weights[i];
            totalPos += w * labels[i];
            totalNeg += w * (1.0f - labels[i]);
        }
        if (totalPos <= 0 || totalNeg <= 0) return (1.0f, 1.0f, float.NaN);
        var (_, _, auc) = BinaryAUC(predts, labels, weights, sortedIdx,
            (fpPrev, fp, tpPrev, tp) => CalcDeltaPRAUC(fpPrev, fp, tpPrev, tp, totalPos));
        return (1.0, 1.0, auc);
    }

    protected static double GroupRankingROC(Context ctx, float[] predts, float[] labels, float w)
    {
        var auc = 0.0;
        var sortedIdx = StdAlgo.ArgSort<float>(labels, static (l, r) => l > r);
        w = XMath.Sqr(w);
        var sumW = 0.0;
        for (var i = 0; i < labels.Length; ++i)
        {
            for (var j = i + 1; j < labels.Length; ++j)
            {
                var predt = predts[sortedIdx[i]] - predts[sortedIdx[j]];
                if (predt > 0) predt = 1.0f;
                else if (predt == 0) predt = 0.5f;
                else predt = 0;
                auc += predt * w;
                sumW += w;
            }
        }
        if (sumW != 0) auc /= sumW;
        Check.Le(auc, 1.0 + Constants.RtEps);
        return auc;
    }

    protected static double MultiAUC(Context ctx, float[] predts, MetaInfo info, long nTargets, int nThreads, MultiAUCType type,
        BinaryAucFn binaryAuc)
    {
        Check.Ne(nTargets, 0L);
        var labels = info.Labels.HostView();
        if (labels.Shape(0) != 0)
        {
            if (type == MultiAUCType.MultiClass) Check.Eq(labels.Shape(1), 1L);
            else Check.Eq(labels.Shape(1), nTargets);
        }
        var nSamples = labels.Shape(0);
        Check.Eq((long)predts.Length, nSamples * nTargets);
        var results = new double[nTargets * 3];
        var weights = new OptionalWeights(info.Weights);
        if (nSamples != 0)
        {
            Threading.ParallelFor(nTargets, nThreads, c =>
            {
                var proba = new float[nSamples];
                var response = new float[nSamples];
                for (long i = 0; i < nSamples; ++i)
                {
                    proba[i] = predts[i * nTargets + c];
                    response[i] = type == MultiAUCType.MultiClass ? (labels.Flat(i) == c ? 1.0f : 0.0f) : labels[i, c];
                }
                var (fp, tp, auc) = binaryAuc(ctx, proba, response, weights);
                results[c * 3 + 1] = tp;
                results[c * 3 + 2] = auc;
                results[c * 3 + 0] = fp * tp;
            });
        }
        Communicator.Allreduce(results.AsSpan(), Op.Sum);
        var aucSum = 0.0;
        var weightSum = 0.0;
        for (long c = 0; c < nTargets; ++c)
        {
            var localArea = results[c * 3];
            var tp = results[c * 3 + 1];
            var auc = results[c * 3 + 2];
            if (localArea > 0 && !double.IsNaN(auc))
            {
                var weight = type == MultiAUCType.MultiClass ? tp : 1.0;
                aucSum += auc / localArea * weight;
                weightSum += weight;
            }
            else
            {
                aucSum = double.NaN;
                break;
            }
        }
        if (weightSum == 0 || double.IsNaN(aucSum)) aucSum = double.NaN;
        else aucSum /= weightSum;
        return aucSum;
    }

    protected (double, uint) RankingAUC(bool isRoc, float[] predts, MetaInfo info, int nThreads)
    {
        Check.Ge(info.GroupPtr.Count, 2);
        var nGroups = (uint)(info.GroupPtr.Count - 1);
        var labels = info.Labels.Data.RawArray;
        var sWeights = info.Weights;
        var invalidGroups = 0;
        var aucTloc = new double[nThreads];
        var gptr = info.GroupPtr;
        Threading.ParallelFor(nGroups, nThreads, (gi, tid) =>
        {
            var g = (int)gi + 1;
            var begin = (int)gptr[g - 1];
            var cnt = (int)(gptr[g] - gptr[g - 1]);
            var w = sWeights.Empty ? 1.0f : sWeights[g - 1];
            var gPredts = predts.AsSpan(begin, cnt).ToArray();
            var gLabels = labels.AsSpan(begin, cnt).ToArray();
            double auc;
            if (isRoc && gLabels.Length < 3)
            {
                Interlocked.Increment(ref invalidGroups);
                auc = 0;
            }
            else
            {
                auc = isRoc ? GroupRankingROC(Ctx, gPredts, gLabels, w) : BinaryPRAUC(Ctx, gPredts, gLabels, new OptionalWeights(w)).Item3;
                if (double.IsNaN(auc))
                {
                    Interlocked.Increment(ref invalidGroups);
                    auc = 0;
                }
            }
            aucTloc[tid] += auc;
        });
        var sumAuc = 0.0;
        foreach (var v in aucTloc) sumAuc += v;
        return (sumAuc, nGroups - (uint)invalidGroups);
    }

    protected abstract (double, uint) EvalRanking(HostDeviceVector<float> predts, MetaInfo info);
    protected abstract double EvalMultiClass(HostDeviceVector<float> predts, MetaInfo info, long nClasses);
    protected abstract double EvalMultiLabel(HostDeviceVector<float> predts, MetaInfo info, long nTargets);
    protected abstract (double, double, double) EvalBinary(HostDeviceVector<float> predts, MetaInfo info);

    public override double Eval(HostDeviceVector<float> preds, MetaInfo info)
    {
        if (info.GroupPtr.Count == 0) MetricUtils.CheckRowWeights(info);
        double auc = 0;
        Span<long> meta = [info.Labels.Size, preds.Size, info.Labels.Shape(1), info.GroupPtr.Count != 0 ? 1 : 0];
        Communicator.Allreduce(meta, Op.Max);
        long nLabels = meta[0], nPredts = meta[1], nTargets = meta[2];
        var isRanking = meta[3] != 0;
        if (nLabels == 0)
        {
            auc = double.NaN;
        }
        else if (isRanking)
        {
            if (nTargets > 1) Check.Fail("AUC and AUCPR do not support multi-output learning-to-rank.");
            Check.Eq(nPredts, nLabels, "Invalid shape of labels and predictions for AUC.");
            uint validGroups = 0;
            if (info.Labels.Size != 0)
            {
                Check.Ge(info.GroupPtr.Count, 2);
                if (!info.Weights.Empty) Check.Eq(info.Weights.Size, info.GroupPtr.Count - 1);
                Check.Eq((long)info.GroupPtr[^1], info.Labels.Shape(0));
                (auc, validGroups) = EvalRanking(preds, info);
            }
            var nGroups = info.GroupPtr.Count == 0 ? 0 : info.GroupPtr.Count - 1;
            if (validGroups != nGroups) InvalidGroupAUC();
            auc = MetricUtils.GlobalRatio(auc, validGroups);
            if (!double.IsNaN(auc))
                Check.Le(auc, 1.0 + Constants.RtEps, $"Total AUC across groups: {auc * validGroups}, valid groups: {validGroups}");
        }
        else if (nTargets > 1)
        {
            if (nPredts > nLabels && nPredts % nLabels == 0) Check.Fail("AUC and AUCPR do not support multi-target-multi-class classification.");
            Check.Eq(nPredts, nLabels, "Invalid shape of labels and predictions for AUC.");
            auc = EvalMultiLabel(preds, info, nTargets);
        }
        else if (nPredts != nLabels)
        {
            Check.Gt(nPredts, nLabels, "Invalid shape of labels and predictions for AUC.");
            Check.Eq(nPredts % nLabels, 0L, "Invalid shape of labels and predictions for AUC.");
            auc = EvalMultiClass(preds, info, nPredts / nLabels);
        }
        else
        {
            double fp = 0, tp = 0;
            if (!(preds.Empty || info.Labels.Size == 0)) (fp, tp, auc) = EvalBinary(preds, info);
            auc = MetricUtils.GlobalRatio(auc, fp * tp);
            if (!double.IsNaN(auc))
            {
                Check.Le(auc, 1.0 + Constants.RtEps);
                auc = XMath.StdMin(auc, 1.0);
            }
        }
        if (double.IsNaN(auc)) Log.Warning("Dataset is empty, or contains only positive or negative samples.");
        return auc;
    }
}

public sealed class EvalROCAUC : EvalAUCBase
{
    protected override (double, uint) EvalRanking(HostDeviceVector<float> predts, MetaInfo info) =>
        RankingAUC(true, predts.ToArray(), info, Ctx.Threads());

    protected override double EvalMultiClass(HostDeviceVector<float> predts, MetaInfo info, long nClasses)
    {
        Check.Ne(nClasses, 0L);
        return MultiAUC(Ctx, predts.ToArray(), info, nClasses, Ctx.Threads(), MultiAUCType.MultiClass, BinaryROCAUC);
    }

    protected override double EvalMultiLabel(HostDeviceVector<float> predts, MetaInfo info, long nTargets) =>
        MultiAUC(Ctx, predts.ToArray(), info, nTargets, Ctx.Threads(), MultiAUCType.MultiLabel, BinaryROCAUC);

    protected override (double, double, double) EvalBinary(HostDeviceVector<float> predts, MetaInfo info) =>
        BinaryROCAUC(Ctx, predts.ToArray(), info.Labels.HostView().Slice(SliceArg.All, 0).ToArray(), new OptionalWeights(info.Weights));

    public override string Name => "auc";
}

public sealed class EvalPRAUC : EvalAUCBase
{
    protected override (double, double, double) EvalBinary(HostDeviceVector<float> predts, MetaInfo info) =>
        BinaryPRAUC(Ctx, predts.ToArray(), info.Labels.HostView().Slice(SliceArg.All, 0).ToArray(), new OptionalWeights(info.Weights));

    protected override double EvalMultiClass(HostDeviceVector<float> predts, MetaInfo info, long nClasses) =>
        MultiAUC(Ctx, predts.ToArray(), info, nClasses, Ctx.Threads(), MultiAUCType.MultiClass, BinaryPRAUC);

    protected override double EvalMultiLabel(HostDeviceVector<float> predts, MetaInfo info, long nTargets) =>
        MultiAUC(Ctx, predts.ToArray(), info, nTargets, Ctx.Threads(), MultiAUCType.MultiLabel, BinaryPRAUC);

    protected override (double, uint) EvalRanking(HostDeviceVector<float> predts, MetaInfo info)
    {
        foreach (var y in info.Labels.Data.ConstHostSpan)
            if (y < 0.0f || y > 1.0f) Check.Fail("PR-AUC supports only binary relevance for learning to rank.");
        return RankingAUC(false, predts.ToArray(), info, Ctx.Threads());
    }

    public override string Name => "aucpr";
}
