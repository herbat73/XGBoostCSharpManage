// Port of src/tree/hist/evaluate_splits.h (CPU HistEvaluator / HistMultiEvaluator).
namespace XGBoost.Tree;

public sealed class HistEvaluator
{
    private struct NodeEntry
    {
        public GradStats Stats;
        public float RootGain;
    }

    private readonly Context _ctx;
    private readonly TrainParam _param;
    private readonly ColumnSampler _columnSampler;
    private readonly TreeEvaluator _treeEvaluator;
    private readonly FeatureInteractionConstraintHost _interactionConstraints = new();
    private readonly List<NodeEntry> _snode = [];

    public HistEvaluator(Context ctx, TrainParam param, MetaInfo info, ColumnSampler sampler)
    {
        _ctx = ctx;
        _param = param;
        _columnSampler = sampler;
        _treeEvaluator = new TreeEvaluator(param, (uint)info.NumCol, 1u);
        _interactionConstraints.Configure(param, (uint)info.NumCol);
        _columnSampler.Init(ctx, info.NumCol, info.FeatureWeights.ToArray(), param.ColsampleBynode, param.ColsampleBylevel,
            param.ColsampleBytree);
    }

    public TreeEvaluator Evaluator => _treeEvaluator;

    private static bool SplitContainsMissingValues(GradStats e, NodeEntry snode) =>
        !(e.GetGrad() == snode.Stats.GetGrad() && e.GetHess() == snode.Stats.GetHess());

    private void EnumerateOneHot(HistogramCuts cut, GradientPairPrecise[] hist, uint fidx, int nidx, SplitEntry pBest)
    {
        var cutPtr = cut.Ptrs.ConstHostSpan;
        var cutVal = cut.Values.ConstHostSpan;
        var ibegin = (int)cutPtr[(int)fidx];
        var iend = (int)cutPtr[(int)fidx + 1];
        var nBins = iend - ibegin;
        var leftSum = new GradStats();
        var rightSum = new GradStats();
        var best = new SplitEntry { IsCat = false };
        var featureSum = new GradientPairPrecise();
        for (var i = 0; i < nBins; ++i) featureSum += hist[cutPtr[(int)fidx] + i];
        var missing = new GradStats();
        var parent = _snode[nidx];
        missing.SetSubstract(parent.Stats, new GradStats(featureSum));
        for (var i = ibegin; i != iend; i += 1)
        {
            var splitPt = cutVal[i];
            rightSum = new GradStats(hist[i]);
            leftSum.SetSubstract(parent.Stats, rightSum);
            var missingLeftChg = (float)((double)_treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parent.RootGain);
            best.Update(missingLeftChg, fidx, splitPt, true, true, leftSum, rightSum);

            rightSum.Add(missing);
            leftSum.SetSubstract(parent.Stats, rightSum);
            var missingRightChg = (float)((double)_treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parent.RootGain);
            best.Update(missingRightChg, fidx, splitPt, false, true, leftSum, rightSum);
        }
        if (best.IsCat)
        {
            var n = LBitField32.ComputeStorageSize(nBins + 1);
            best.CatBits = [.. new uint[n]];
            var bits = best.CatBits.ToArray();
            new LBitField32(bits).Set((long)best.SplitValue);
            best.CatBits = [.. bits];
        }
        pBest.Update(best);
    }

    private void EnumeratePart(int dStep, HistogramCuts cut, int[] sortedIdx, GradientPairPrecise[] hist, uint fidx, int nidx,
        SplitEntry pBest)
    {
        var cutPtr = cut.Ptrs.ConstHostSpan;
        var cutVal = cut.Values.ConstHostSpan;
        var parent = _snode[nidx];
        var fBegin = (int)cutPtr[(int)fidx];
        var fEnd = (int)cutPtr[(int)fidx + 1];
        var nBinsFeature = fEnd - fBegin;
        var nBins = Math.Min(_param.MaxCatThreshold, nBinsFeature);
        var leftSum = new GradStats();
        var rightSum = new GradStats();
        var best = new SplitEntry();
        int itBegin, itEnd;
        if (dStep > 0)
        {
            itBegin = fBegin;
            itEnd = itBegin + nBins - 1;
        }
        else
        {
            itBegin = fEnd - 1;
            itEnd = itBegin - nBins + 1;
        }
        var bestThresh = -1;
        for (var i = itBegin; i != itEnd; i += dStep)
        {
            var j = i - fBegin;
            var g = hist[fBegin + sortedIdx[j]];
            if (dStep == 1)
            {
                rightSum.Add(g.Grad, g.Hess);
                leftSum.SetSubstract(parent.Stats, rightSum);
            }
            else
            {
                leftSum.Add(g.Grad, g.Hess);
                rightSum.SetSubstract(parent.Stats, leftSum);
            }
            var lossChg = (double)_treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parent.RootGain;
            if (best.Update((float)lossChg, fidx, float.NaN, dStep == 1, true, leftSum, rightSum)) bestThresh = i;
        }
        if (bestThresh != -1)
        {
            var n = LBitField32.ComputeStorageSize(nBinsFeature);
            var bits = new uint[n];
            var catBits = new LBitField32(bits);
            var partition = dStep == 1 ? bestThresh - itBegin + 1 : bestThresh - fBegin;
            Check.Gt(partition, 0);
            for (var k = 0; k < partition; ++k)
            {
                var cat = cutVal[sortedIdx[k] + fBegin];
                catBits.Set((long)cat);
            }
            best.CatBits = [.. bits];
        }
        pBest.Update(best);
    }

    private GradStats EnumerateSplit(int dStep, HistogramCuts cut, GradientPairPrecise[] hist, uint fidx, int nidx, SplitEntry pBest)
    {
        var cutPtr = cut.Ptrs.ConstHostSpan;
        var cutVal = cut.Values.ConstHostSpan;
        var parent = _snode[nidx];
        var leftSum = new GradStats();
        var rightSum = new GradStats();
        var best = new SplitEntry();
        int ibegin, iend;
        if (dStep > 0)
        {
            ibegin = (int)cutPtr[(int)fidx];
            iend = (int)cutPtr[(int)fidx + 1];
        }
        else
        {
            ibegin = (int)cutPtr[(int)fidx + 1] - 1;
            iend = (int)cutPtr[(int)fidx] - 1;
        }
        for (var i = ibegin; i != iend; i += dStep)
        {
            leftSum.Add(hist[i].Grad, hist[i].Hess);
            rightSum.SetSubstract(parent.Stats, leftSum);
            float lossChg;
            float splitPt;
            if (dStep > 0)
            {
                lossChg = (float)((double)_treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parent.RootGain);
                splitPt = cutVal[i];
                best.Update(lossChg, fidx, splitPt, dStep == -1, false, leftSum, rightSum);
            }
            else
            {
                lossChg = (float)((double)_treeEvaluator.CalcSplitGain(_param, nidx, fidx, rightSum, leftSum) - parent.RootGain);
                splitPt = HistogramCuts.NumericBinLowerBound(cutPtr, cutVal, (int)fidx, i);
                best.Update(lossChg, fidx, splitPt, dStep == -1, false, rightSum, leftSum);
            }
        }
        pBest.Update(best);
        return leftSum;
    }

    public void EvaluateSplits(BoundedHistCollection hist, HistogramCuts cut, ReadOnlySpan<FeatureType> featureTypesSpan,
        List<CPUExpandEntry> entries)
    {
        var nThreads = _ctx.Threads();
        var featureTypes = featureTypesSpan.ToArray();
        var features = new uint[entries.Count][];
        for (var i = 0; i < entries.Count; ++i) features[i] = _columnSampler.GetFeatureSet(_ctx, entries[i].Depth);
        Check.That(features.Length != 0);
        var grainSize = Math.Max(1, features[0].Length / nThreads);
        var space = new BlockedSpace2d(entries.Count, i => features[i].Length, grainSize);
        var tlocCandidates = new CPUExpandEntry[nThreads * entries.Count];
        for (var i = 0; i < entries.Count; ++i)
            for (var j = 0; j < nThreads; ++j)
                tlocCandidates[i * nThreads + j] = entries[i].Clone();
        var cutPtrs = cut.Ptrs.ToArray();
        Threading.ParallelFor2d(space, nThreads, (nidxInSet, r) =>
        {
            var tidx = Threading.ThreadNum;
            var entry = tlocCandidates[nThreads * nidxInSet + tidx];
            var best = entry.Split;
            var nidx = entry.Nid;
            var histogram = hist[nidx];
            var featuresSet = features[nidxInSet];
            for (var fidxInSet = r.Begin; fidxInSet < r.End; fidxInSet++)
            {
                var fidx = featuresSet[fidxInSet];
                var isCat = Categorical.IsCat(featureTypes, fidx);
                if (!_interactionConstraints.Query(nidx, fidx)) continue;
                if (isCat)
                {
                    var nBins = cutPtrs[fidx + 1] - cutPtrs[fidx];
                    if (Categorical.UseOneHot(nBins, _param.MaxCatToOnehot))
                    {
                        EnumerateOneHot(cut, histogram, fidx, nidx, best);
                    }
                    else
                    {
                        var sortedIdx = new int[nBins];
                        StdAlgo.Iota(sortedIdx);
                        var off = (int)cutPtrs[fidx];
                        StdAlgo.StableSort<int>(sortedIdx, (l, rr) =>
                            _treeEvaluator.CalcWeightCat(_param, histogram[off + l]) < _treeEvaluator.CalcWeightCat(_param, histogram[off + rr]));
                        EnumeratePart(+1, cut, sortedIdx, histogram, fidx, nidx, best);
                        EnumeratePart(-1, cut, sortedIdx, histogram, fidx, nidx, best);
                    }
                }
                else
                {
                    var gradStats = EnumerateSplit(+1, cut, histogram, fidx, nidx, best);
                    if (SplitContainsMissingValues(gradStats, _snode[nidx])) EnumerateSplit(-1, cut, histogram, fidx, nidx, best);
                }
            }
        });
        for (var i = 0; i < entries.Count; ++i)
            for (var t = 0; t < nThreads; ++t)
                entries[i].Split.Update(tlocCandidates[nThreads * i + t].Split);
    }

    public void ApplyTreeSplit(CPUExpandEntry candidate, RegTree tree)
    {
        var parentSum = candidate.Split.LeftSum;
        parentSum.Add(candidate.Split.RightSum);
        var baseWeight = _treeEvaluator.CalcWeight(candidate.Nid, _param, parentSum);
        var leftWeight = _treeEvaluator.CalcWeight(candidate.Nid, _param, candidate.Split.LeftSum);
        var rightWeight = _treeEvaluator.CalcWeight(candidate.Nid, _param, candidate.Split.RightSum);
        if (candidate.Split.IsCat)
        {
            tree.ExpandCategorical(candidate.Nid, candidate.Split.SplitIndex, [.. candidate.Split.CatBits], candidate.Split.DefaultLeft,
                baseWeight, leftWeight * _param.LearningRate, rightWeight * _param.LearningRate, candidate.Split.LossChg,
                (float)parentSum.GetHess(), (float)candidate.Split.LeftSum.GetHess(), (float)candidate.Split.RightSum.GetHess());
        }
        else
        {
            tree.ExpandNode(candidate.Nid, candidate.Split.SplitIndex, candidate.Split.SplitValue, candidate.Split.DefaultLeft,
                baseWeight, leftWeight * _param.LearningRate, rightWeight * _param.LearningRate, candidate.Split.LossChg,
                (float)parentSum.GetHess(), (float)candidate.Split.LeftSum.GetHess(), (float)candidate.Split.RightSum.GetHess());
        }
        var leftChild = tree[candidate.Nid].LeftChild;
        var rightChild = tree[candidate.Nid].RightChild;
        _treeEvaluator.AddSplit(candidate.Nid, leftChild, rightChild, tree[candidate.Nid].SplitIndex, leftWeight, rightWeight);
        while (_snode.Count < tree.Size) _snode.Add(default);
        if (_snode.Count > tree.Size) _snode.RemoveRange(tree.Size, _snode.Count - tree.Size);
        _snode[leftChild] = new NodeEntry
        {
            Stats = candidate.Split.LeftSum,
            RootGain = _treeEvaluator.CalcGain(candidate.Nid, _param, candidate.Split.LeftSum),
        };
        _snode[rightChild] = new NodeEntry
        {
            Stats = candidate.Split.RightSum,
            RootGain = _treeEvaluator.CalcGain(candidate.Nid, _param, candidate.Split.RightSum),
        };
        _interactionConstraints.Split(candidate.Nid, tree[candidate.Nid].SplitIndex, leftChild, rightChild);
    }

    public float InitRoot(GradStats rootSum)
    {
        _snode.Clear();
        var stats = new GradStats(rootSum.GetGrad(), rootSum.GetHess());
        _snode.Add(new NodeEntry { Stats = stats, RootGain = _treeEvaluator.CalcGain(RegTree.Root, _param, stats) });
        return _treeEvaluator.CalcWeight(RegTree.Root, _param, stats);
    }
}

public sealed class HistMultiEvaluator
{
    private readonly List<double> _gain = [];
    private GradientPairPrecise[] _stats = []; // (n_nodes, n_targets)
    private int _statsRows;
    private readonly TrainParam _param;
    private readonly TreeEvaluator _treeEvaluator;
    private readonly FeatureInteractionConstraintHost _interactionConstraints = new();
    private readonly ColumnSampler _columnSampler;
    private readonly Context _ctx;
    private readonly int _nTargets;

    public HistMultiEvaluator(Context ctx, MetaInfo info, TrainParam param, uint nTargets, ColumnSampler sampler)
    {
        _param = param;
        _treeEvaluator = new TreeEvaluator(param, (uint)info.NumCol, nTargets);
        _columnSampler = sampler;
        _ctx = ctx;
        _nTargets = (int)nTargets;
        _interactionConstraints.Configure(param, (uint)info.NumCol);
        _columnSampler.Init(ctx, info.NumCol, info.FeatureWeights.ToArray(), param.ColsampleBynode, param.ColsampleBylevel,
            param.ColsampleBytree);
    }

    public TreeEvaluator Evaluator => _treeEvaluator;

    private Span<GradientPairPrecise> Stats(int nidx) => _stats.AsSpan(nidx * _nTargets, _nTargets);

    private bool EnumerateSplit(int dStep, HistogramCuts cut, uint fidx, GradientPairPrecise[][] hist, GradientPairPrecise[] parentSum,
        double parentGain, int nidx, MultiSplitEntry pBest)
    {
        var cutPtr = cut.Ptrs.ConstHostSpan;
        var cutVal = cut.Values.ConstHostSpan;
        var nTargets = hist.Length;
        var leftSum = new GradientPairPrecise[nTargets];
        var rightSum = new GradientPairPrecise[nTargets];
        int ibegin, iend;
        if (dStep > 0)
        {
            ibegin = (int)cutPtr[(int)fidx];
            iend = (int)cutPtr[(int)fidx + 1];
        }
        else
        {
            ibegin = (int)cutPtr[(int)fidx + 1] - 1;
            iend = (int)cutPtr[(int)fidx] - 1;
        }
        for (var i = ibegin; i != iend; i += dStep)
        {
            for (var t = 0; t < nTargets; ++t)
            {
                leftSum[t] += hist[t][i];
                rightSum[t] = parentSum[t] - leftSum[t];
            }
            if (dStep > 0)
            {
                var splitPt = cutVal[i];
                var lossChg = _treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parentGain;
                pBest.Update((float)lossChg, fidx, splitPt, dStep == -1, false, leftSum, rightSum);
            }
            else
            {
                var splitPt = HistogramCuts.NumericBinLowerBound(cutPtr, cutVal, (int)fidx, i);
                var lossChg = _treeEvaluator.CalcSplitGain(_param, nidx, fidx, rightSum, leftSum) - parentGain;
                pBest.Update((float)lossChg, fidx, splitPt, dStep == -1, false, rightSum, leftSum);
            }
        }
        if (dStep == +1)
        {
            for (var t = 0; t < nTargets; ++t)
                if (leftSum[t] != parentSum[t]) return true;
            return false;
        }
        return false;
    }

    private void EnumerateOneHot(HistogramCuts cut, uint fidx, GradientPairPrecise[][] hist, GradientPairPrecise[] parentSum, double parentGain,
        int nidx, MultiSplitEntry pBest)
    {
        var cutPtr = cut.Ptrs.ConstHostSpan;
        var cutVal = cut.Values.ConstHostSpan;
        var ibegin = (int)cutPtr[(int)fidx];
        var iend = (int)cutPtr[(int)fidx + 1];
        var nBins = iend - ibegin;
        var nTargets = hist.Length;
        var leftSum = new GradientPairPrecise[nTargets];
        var rightSum = new GradientPairPrecise[nTargets];
        var missing = new GradientPairPrecise[nTargets];
        for (var t = 0; t < nTargets; ++t)
        {
            var featureSum = new GradientPairPrecise();
            for (var b = 0; b < nBins; ++b) featureSum += hist[t][ibegin + b];
            missing[t] = parentSum[t] - featureSum;
        }
        var best = new MultiSplitEntry { IsCat = false };
        for (var i = ibegin; i != iend; ++i)
        {
            var splitPt = cutVal[i];
            for (var t = 0; t < nTargets; ++t)
            {
                rightSum[t] = hist[t][i];
                leftSum[t] = parentSum[t] - rightSum[t];
            }
            var missingLeftGain = _treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parentGain;
            best.Update((float)missingLeftGain, fidx, splitPt, true, true, leftSum, rightSum);
            for (var t = 0; t < nTargets; ++t)
            {
                rightSum[t] = hist[t][i] + missing[t];
                leftSum[t] = parentSum[t] - rightSum[t];
            }
            var missingRightGain = _treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parentGain;
            best.Update((float)missingRightGain, fidx, splitPt, false, true, leftSum, rightSum);
        }
        if (best.IsCat)
        {
            var n = LBitField32.ComputeStorageSize(nBins + 1);
            var bits = new uint[n];
            new LBitField32(bits).Set((long)best.SplitValue);
            best.CatBits = [.. bits];
        }
        pBest.Update(best);
    }

    private void EnumeratePart(int dStep, HistogramCuts cut, int[] sortedIdx, GradientPairPrecise[][] hist, uint fidx, int nidx,
        MultiSplitEntry pBest)
    {
        var nTargets = hist.Length;
        var cutPtr = cut.Ptrs.ConstHostSpan;
        var cutVal = cut.Values.ConstHostSpan;
        var parentSum = Stats(nidx).ToArray();
        var parentGain = _gain[nidx];
        var fBegin = (int)cutPtr[(int)fidx];
        var fEnd = (int)cutPtr[(int)fidx + 1];
        var nBinsFeature = fEnd - fBegin;
        var nBins = Math.Min(_param.MaxCatThreshold, nBinsFeature);
        int itBegin, itEnd;
        if (dStep > 0)
        {
            itBegin = fBegin;
            itEnd = itBegin + nBins - 1;
        }
        else
        {
            itBegin = fEnd - 1;
            itEnd = itBegin - nBins + 1;
        }
        var leftSum = new GradientPairPrecise[nTargets];
        var rightSum = new GradientPairPrecise[nTargets];
        var best = new MultiSplitEntry();
        var bestThresh = -1;
        for (var i = itBegin; i != itEnd; i += dStep)
        {
            var j = i - fBegin;
            for (var t = 0; t < nTargets; ++t)
            {
                var g = hist[t][fBegin + sortedIdx[j]];
                if (dStep == 1)
                {
                    rightSum[t] += g;
                    leftSum[t] = parentSum[t] - rightSum[t];
                }
                else
                {
                    leftSum[t] += g;
                    rightSum[t] = parentSum[t] - leftSum[t];
                }
            }
            var lossChg = _treeEvaluator.CalcSplitGain(_param, nidx, fidx, leftSum, rightSum) - parentGain;
            if (best.Update((float)lossChg, fidx, float.NaN, dStep == 1, true, leftSum, rightSum)) bestThresh = i;
        }
        if (bestThresh != -1)
        {
            var n = LBitField32.ComputeStorageSize(nBinsFeature);
            var bits = new uint[n];
            var catBits = new LBitField32(bits);
            var partition = dStep == 1 ? bestThresh - itBegin + 1 : bestThresh - fBegin;
            Check.Gt(partition, 0);
            for (var k = 0; k < partition; ++k) catBits.Set((long)cutVal[sortedIdx[k] + fBegin]);
            best.CatBits = [.. bits];
        }
        pBest.Update(best);
    }

    public void EvaluateSplits(IReadOnlyList<BoundedHistCollection> hist, HistogramCuts cut, ReadOnlySpan<FeatureType> featureTypesSpan,
        List<MultiExpandEntry> entries)
    {
        Check.Eq(_nTargets, hist.Count);
        Check.That(hist.Count != 0);
        var featureTypes = featureTypesSpan.ToArray();
        var features = new uint[entries.Count][];
        for (var i = 0; i < entries.Count; ++i) features[i] = _columnSampler.GetFeatureSet(_ctx, entries[i].Depth);
        Check.That(features.Length != 0);
        var nThreads = _ctx.Threads();
        var grainSize = Math.Max(1, features[0].Length / nThreads);
        var space = new BlockedSpace2d(entries.Count, i => features[i].Length, grainSize);
        var tlocCandidates = new MultiExpandEntry[nThreads * entries.Count];
        for (var i = 0; i < entries.Count; ++i)
            for (var j = 0; j < nThreads; ++j)
                tlocCandidates[i * nThreads + j] = entries[i].Clone();
        var cutPtr = cut.Ptrs.ToArray();
        Threading.ParallelFor2d(space, nThreads, (nidxInSet, r) =>
        {
            var tidx = Threading.ThreadNum;
            var entry = tlocCandidates[nThreads * nidxInSet + tidx];
            var best = entry.Split;
            var parentSum = Stats(entry.Nid).ToArray();
            var hBw = new float[parentSum.Length];
            _treeEvaluator.CalcWeightCat(_param, parentSum, hBw);
            var nodeHist = new GradientPairPrecise[hist.Count][];
            for (var t = 0; t < hist.Count; ++t) nodeHist[t] = hist[t][entry.Nid];
            var featuresSet = features[nidxInSet];
            var nTargets = hist.Count;
            for (var fidxInSet = r.Begin; fidxInSet < r.End; fidxInSet++)
            {
                var fidx = featuresSet[fidxInSet];
                if (!_interactionConstraints.Query(entry.Nid, fidx)) continue;
                var parentGain = _gain[entry.Nid];
                var isCat = Categorical.IsCat(featureTypes, fidx);
                if (!isCat)
                {
                    var missing = EnumerateSplit(+1, cut, fidx, nodeHist, parentSum, parentGain, entry.Nid, best);
                    if (missing) EnumerateSplit(-1, cut, fidx, nodeHist, parentSum, parentGain, entry.Nid, best);
                    continue;
                }
                var nBins = cutPtr[fidx + 1] - cutPtr[fidx];
                if (Categorical.UseOneHot(nBins, _param.MaxCatToOnehot))
                {
                    EnumerateOneHot(cut, fidx, nodeHist, parentSum, parentGain, entry.Nid, best);
                    continue;
                }
                var sortedIdx = new int[nBins];
                StdAlgo.Iota(sortedIdx);
                var hGrads = new GradientPairPrecise[nTargets];
                var childW = new float[nTargets];
                var scores = new double[nBins];
                for (var binIdx = 0; binIdx < nBins; ++binIdx)
                {
                    for (var t = 0; t < nTargets; ++t) hGrads[t] = nodeHist[t][cutPtr[fidx] + binIdx];
                    _treeEvaluator.CalcWeightCat(_param, hGrads, childW);
                    var sc = 0.0;
                    for (var t = 0; t < nTargets; ++t) sc += hBw[t] * childW[t];
                    scores[binIdx] = sc;
                }
                StdAlgo.StableSort<int>(sortedIdx, (l, rr) => scores[l] < scores[rr]);
                EnumeratePart(+1, cut, sortedIdx, nodeHist, fidx, entry.Nid, best);
                EnumeratePart(-1, cut, sortedIdx, nodeHist, fidx, entry.Nid, best);
            }
        });
        for (var i = 0; i < entries.Count; ++i)
            for (var t = 0; t < nThreads; ++t)
                entries[i].Split.Update(tlocCandidates[nThreads * i + t].Split);
    }

    public float[] InitRoot(ReadOnlySpan<GradientPairPrecise> rootSum)
    {
        var nTargets = rootSum.Length;
        _stats = new GradientPairPrecise[nTargets];
        _statsRows = 1;
        _gain.Clear();
        _gain.Add(0);
        var weight = new float[nTargets];
        _treeEvaluator.CalcWeight(RegTree.Root, _param, rootSum, weight);
        _gain[0] = _treeEvaluator.CalcGainGivenWeight(_param, rootSum, weight);
        rootSum.CopyTo(_stats);
        return weight;
    }

    public void ApplyTreeSplit(MultiExpandEntry candidate, RegTree tree)
    {
        var nSplitTargets = candidate.Split.LeftSum.Length;
        var parentSum = Stats(candidate.Nid).ToArray();
        var baseWeight = new float[nSplitTargets];
        _treeEvaluator.CalcWeight(candidate.Nid, _param, parentSum, baseWeight);
        var leftWeight = new float[nSplitTargets];
        var rightWeight = new float[nSplitTargets];
        var leftSum = candidate.Split.LeftSum;
        var rightSum = candidate.Split.RightSum;
        _treeEvaluator.CalcSplitWeights(_param, candidate.Nid, candidate.Split.SplitIndex, leftSum, rightSum, leftWeight, rightWeight);
        var lossChg = candidate.Split.LossChg;
        double leftSumHess = 0.0, rightSumHess = 0.0;
        for (var t = 0; t < nSplitTargets; ++t)
        {
            leftSumHess += leftSum[t].Hess;
            rightSumHess += rightSum[t].Hess;
        }
        uint[] catBits = candidate.Split.IsCat ? [.. candidate.Split.CatBits] : [];
        var batch = new ExpandBatch(_param.LearningRate);
        batch.Push(candidate.Nid, candidate.Split.SplitIndex, candidate.Split.SplitValue, candidate.Split.DefaultLeft, baseWeight,
            leftWeight, rightWeight, lossChg, leftSumHess, rightSumHess, catBits);
        tree.Expand(_ctx, batch);
        Check.That(tree.IsMultiTarget);
        var leftChild = tree.LeftChild(candidate.Nid);
        Check.Gt(leftChild, candidate.Nid);
        var rightChild = tree.RightChild(candidate.Nid);
        Check.Gt(rightChild, candidate.Nid);
        _treeEvaluator.AddSplit(candidate.Nid, leftChild, rightChild, candidate.Split.SplitIndex, leftWeight, rightWeight);
        _interactionConstraints.Split(candidate.Nid, candidate.Split.SplitIndex, leftChild, rightChild);
        var nNodes = tree.Size;
        while (_gain.Count < nNodes) _gain.Add(0);
        if (_gain.Count > nNodes) _gain.RemoveRange(nNodes, _gain.Count - nNodes);
        _gain[leftChild] = _treeEvaluator.CalcGainGivenWeight(_param, leftSum, leftWeight);
        _gain[rightChild] = _treeEvaluator.CalcGainGivenWeight(_param, rightSum, rightWeight);
        if (nNodes >= _statsRows)
        {
            var newRows = nNodes * 2;
            Array.Resize(ref _stats, newRows * _nTargets);
            _statsRows = newRows;
        }
        Check.Eq(_nTargets, nSplitTargets);
        leftSum.CopyTo(Stats(leftChild));
        rightSum.CopyTo(Stats(rightChild));
    }
}
