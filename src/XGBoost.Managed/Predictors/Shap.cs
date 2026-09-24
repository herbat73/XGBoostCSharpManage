// Port of src/predictor/interpretability/shap.cc and quadrature.h (CPU QuadratureTreeSHAP).
using XGBoost.Gbm;
using XGBoost.Tree;

namespace XGBoost.Predictors;

public static class Shap
{
    private const int Points = 8;
    private const float Unseen = -999.0f;
    private const float MinChildWeight = 1e-12f;
    private const double Pi = 3.141592653589793238462643383279502884;

    private sealed class QuadratureRule
    {
        public readonly float[] Nodes = new float[Points];
        public readonly float[] Weights = new float[Points];
    }

    private static readonly Lazy<QuadratureRule> Rule = new(MakeEndpointQuadrature);

    private static float BranchWeight(float cover, float parentCover)
    {
        if (parentCover <= 0.0f) return 0.5f;
        var weight = cover / parentCover;
        if (weight < MinChildWeight) return MinChildWeight;
        return weight;
    }

    private static double LegendrePolynomial(int n, double x)
    {
        var p0 = 1.0;
        if (n == 0) return p0;
        var p1 = x;
        if (n == 1) return p1;
        for (var k = 2; k <= n; ++k)
        {
            var pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
            p0 = p1;
            p1 = pk;
        }
        return p1;
    }

    private static double LegendreDerivative(int n, double x, double pn) => n * (x * pn - LegendrePolynomial(n - 1, x)) / (x * x - 1.0);

    private static QuadratureRule MakeEndpointQuadrature()
    {
        const int n = Points;
        const double convergenceEps = 1e-15;
        var rule = new QuadratureRule();
        for (var i = 0; i < n; ++i)
        {
            var theta = Pi * (i + 0.75) / (n + 0.5);
            var x = Math.Cos(theta);
            for (var iter = 0; iter < 64; ++iter)
            {
                var pn0 = LegendrePolynomial(n, x);
                var dpn0 = LegendreDerivative(n, x, pn0);
                var dx = pn0 / dpn0;
                x -= dx;
                if (Math.Abs(dx) < convergenceEps) break;
            }
            var pn = LegendrePolynomial(n, x);
            var dpn = LegendreDerivative(n, x, pn);
            var w = 2.0 / ((1.0 - x * x) * dpn * dpn);
            var s = 0.5 * (x + 1.0);
            var ws = 0.5 * w;
            var outIdx = n - 1 - i;
            rule.Nodes[outIdx] = (float)(s * s);
            rule.Weights[outIdx] = (float)(2.0 * s * ws);
        }
        return rule;
    }

    private static float LeafValue(ITreeView tree, int nidx, int targetIdx)
    {
        if (tree is ScalarTreeView sc)
        {
            Check.Eq(targetIdx, 0);
            return sc.ScalarLeafValue(nidx);
        }
        var leaf = tree.LeafValue(nidx);
        Check.Lt(targetIdx, leaf.Length);
        return leaf[targetIdx];
    }

    private static double FillRootMeanValue(ITreeView tree, int nidx)
    {
        if (tree.IsLeaf(nidx)) return LeafValue(tree, nidx, 0);
        var left = tree.LeftChild(nidx);
        var right = tree.RightChild(nidx);
        Check.Ge(tree.SumHess(nidx), 0.0f, "QuadratureTreeSHAP is undefined for trees with negative cover at split nodes.");
        Check.Ge(tree.SumHess(left), 0.0f, "QuadratureTreeSHAP is undefined for trees with negative child cover.");
        Check.Ge(tree.SumHess(right), 0.0f, "QuadratureTreeSHAP is undefined for trees with negative child cover.");
        var parentCover = tree.SumHess(nidx);
        var leftMean = FillRootMeanValue(tree, left);
        var rightMean = FillRootMeanValue(tree, right);
        if (parentCover == 0.0f) return 0.5 * (leftMean + rightMean);
        return (leftMean * tree.SumHess(left) + rightMean * tree.SumHess(right)) / parentCover;
    }

    private static void FillRootMeanValues(ITreeView tree, int nidx, double pathWeight, double[] output)
    {
        if (tree.IsLeaf(nidx))
        {
            var leafValue = tree.LeafValue(nidx);
            Check.Eq(leafValue.Length, output.Length);
            for (var t = 0; t < leafValue.Length; ++t) output[t] += pathWeight * leafValue[t];
            return;
        }
        var left = tree.LeftChild(nidx);
        var right = tree.RightChild(nidx);
        Check.Ge(tree.SumHess(nidx), 0.0f, "QuadratureTreeSHAP is undefined for trees with negative cover at split nodes.");
        Check.Ge(tree.SumHess(left), 0.0f, "QuadratureTreeSHAP is undefined for trees with negative child cover.");
        Check.Ge(tree.SumHess(right), 0.0f, "QuadratureTreeSHAP is undefined for trees with negative child cover.");
        var parentCover = tree.SumHess(nidx);
        if (parentCover == 0.0f)
        {
            FillRootMeanValues(tree, left, pathWeight * 0.5, output);
            FillRootMeanValues(tree, right, pathWeight * 0.5, output);
        }
        else
        {
            FillRootMeanValues(tree, left, pathWeight * tree.SumHess(left) / parentCover, output);
            FillRootMeanValues(tree, right, pathWeight * tree.SumHess(right) / parentCover, output);
        }
    }

    // ---- Approximate (Saabas) contributions -------------------------------------------------

    private static float FillNodeMeanValues(ScalarTreeView tree, int nidx, float[] meanValues)
    {
        float result;
        if (tree.IsLeaf(nidx))
        {
            result = tree.ScalarLeafValue(nidx);
        }
        else
        {
            result = FillNodeMeanValues(tree, tree.LeftChild(nidx), meanValues) * tree.Stat(tree.LeftChild(nidx)).SumHess;
            result += FillNodeMeanValues(tree, tree.RightChild(nidx), meanValues) * tree.Stat(tree.RightChild(nidx)).SumHess;
            result /= tree.Stat(nidx).SumHess;
        }
        meanValues[nidx] = result;
        return result;
    }

    private static void CalculateApproxContributions(ScalarTreeView tree, RegTree.FVec feats, float[] meanValues, float[] outContribs)
    {
        Check.Eq(outContribs.Length, feats.Size + 1);
        Check.Gt(meanValues.Length, 0);
        var splitIndex = 0;
        var nodeValue = meanValues[0];
        outContribs[^1] += nodeValue;
        if (tree.IsLeaf(RegTree.Root)) return;
        var nidx = RegTree.Root;
        var cats = tree.GetCategoriesMatrix();
        while (!tree.IsLeaf(nidx))
        {
            splitIndex = (int)tree.SplitIndex(nidx);
            nidx = PredictFn.GetNextNode(tree, nidx, feats.GetFvalue(splitIndex), feats.IsMissing(splitIndex), cats, true);
            var newValue = meanValues[nidx];
            outContribs[splitIndex] += newValue - nodeValue;
            nodeValue = newValue;
        }
        outContribs[splitIndex] += tree.ScalarLeafValue(nidx) - nodeValue;
    }

    // ---- QuadratureTreeSHAP -------------------------------------------------------------------

    private static float ExtractQuadratureDelta(QuadratureRule rule, float[] hVals, float pEnter, float pExit)
    {
        var acc = 0.0f;
        if (pEnter != 1.0f)
        {
            var alphaEnter = pEnter - 1.0f;
            for (var i = 0; i < Points; ++i) acc += alphaEnter * hVals[i] / (1.0f + alphaEnter * rule.Nodes[i]);
        }
        if (pExit != 1.0f)
        {
            var alphaExit = pExit - 1.0f;
            for (var i = 0; i < Points; ++i) acc -= alphaExit * hVals[i] / (1.0f + alphaExit * rule.Nodes[i]);
        }
        return acc;
    }

    private static float ExtractQuadratureInteractionDelta(QuadratureRule rule, float[] hVals, float pEnter, float pExit, float qPartner)
    {
        if (qPartner == 1.0f) return 0.0f;
        var alphaPartner = qPartner - 1.0f;
        var alphaEnter = pEnter - 1.0f;
        var acc = 0.0f;
        if (pExit == 1.0f)
        {
            for (var i = 0; i < Points; ++i)
            {
                var edgeDelta = alphaEnter / (1.0f + alphaEnter * rule.Nodes[i]);
                acc += alphaPartner * hVals[i] * edgeDelta / (1.0f + alphaPartner * rule.Nodes[i]);
            }
        }
        else
        {
            var alphaExit = pExit - 1.0f;
            for (var i = 0; i < Points; ++i)
            {
                var edgeDelta = alphaEnter / (1.0f + alphaEnter * rule.Nodes[i]) - alphaExit / (1.0f + alphaExit * rule.Nodes[i]);
                acc += alphaPartner * hVals[i] * edgeDelta / (1.0f + alphaPartner * rule.Nodes[i]);
            }
        }
        return acc;
    }

    private readonly record struct PathElement(uint SplitIndex, float PChild);

    /// <summary>Either the additive or the interaction contribution formulation.</summary>
    private sealed class Formulation
    {
        public List<PathElement>? Path;
        public float[] Phi = [];          // additive: per-feature contributions; interaction: diagonal
        public float[]? Interactions;     // dense (ncolumns x ncolumns) at InterOffset
        public int InterOffset;
        public int NColumns;
        public float Scale = 1.0f;

        public void Reset() => Path?.Clear();
        public void Push(uint splitIndex, float pChild) => Path?.Add(new PathElement(splitIndex, pChild));
        public void Pop() => Path?.RemoveAt(Path.Count - 1);

        public void HandleReturn(QuadratureRule rule, uint splitIndex, float[] hVals, float pEnter, float pExit)
        {
            if (Path is null)
            {
                Phi[splitIndex] += ExtractQuadratureDelta(rule, hVals, pEnter, pExit);
                return;
            }
            Phi[splitIndex] += Scale * ExtractQuadratureDelta(rule, hVals, pEnter, pExit);
            // ForEachPartner: unique features on the path (newest first), skipping the current split once.
            Check.That(Path.Count != 0);
            var currentSplit = Path[^1].SplitIndex;
            var skippedCurrent = false;
            for (var i = Path.Count; i != 0; --i)
            {
                var idx = i - 1;
                var element = Path[idx];
                var shadowed = false;
                for (var newer = Path.Count; newer > i; --newer)
                {
                    if (Path[newer - 1].SplitIndex == element.SplitIndex)
                    {
                        shadowed = true;
                        break;
                    }
                }
                if (shadowed) continue;
                if (!skippedCurrent && element.SplitIndex == currentSplit)
                {
                    skippedCurrent = true;
                    continue;
                }
                var pairDelta = ExtractQuadratureInteractionDelta(rule, hVals, pEnter, pExit, element.PChild);
                Interactions![InterOffset + (int)splitIndex * NColumns + (int)element.SplitIndex] += Scale * pairDelta;
            }
        }
    }

    private sealed class Runner(ITreeView tree, int targetIdx, RegTree.FVec feat, QuadratureRule rule, float[] pathProb, Formulation formulation)
    {
        private readonly CategoricalSplitMatrix _cats = tree.GetCategoriesMatrix();

        private bool EvaluateGoesLeft(int nidx)
        {
            var splitIndex = (int)tree.SplitIndex(nidx);
            var next = PredictFn.GetNextNode(tree, nidx, feat.GetFvalue(splitIndex), feat.IsMissing(splitIndex), _cats, true);
            return next == tree.LeftChild(nidx);
        }

        private float ChildWeight(int parent, int child)
        {
            var parentCover = tree.SumHess(parent);
            Check.Ge(parentCover, 0.0f);
            Check.Ge(tree.SumHess(child), 0.0f);
            return BranchWeight(tree.SumHess(child), parentCover);
        }

        private void VisitChild(int splitNode, int childNode, float childWeight, bool satisfies, float[] cVals, float wProd, float[] outH)
        {
            var splitIndex = tree.SplitIndex(splitNode);
            var pOld = pathProb[splitIndex];
            float pE;
            if (pOld == Unseen) pE = satisfies ? 1.0f / childWeight : 0.0f;
            else pE = satisfies ? pOld / childWeight : 0.0f;
            var cChild = (float[])cVals.Clone();
            var alphaE = pE - 1.0f;
            for (var i = 0; i < Points; ++i) cChild[i] *= 1.0f + alphaE * rule.Nodes[i];
            if (pOld != Unseen)
            {
                var alphaOld = pOld - 1.0f;
                if (alphaOld != 0.0f)
                    for (var i = 0; i < Points; ++i) cChild[i] /= 1.0f + alphaOld * rule.Nodes[i];
            }
            pathProb[splitIndex] = pE;
            formulation.Push(splitIndex, pE);
            RunNode(childNode, cChild, wProd * childWeight, outH);
            formulation.HandleReturn(rule, splitIndex, outH, pE, pOld == Unseen ? 1.0f : pOld);
            formulation.Pop();
            pathProb[splitIndex] = pOld;
        }

        private void RunNode(int nidx, float[] cVals, float wProd, float[] outH)
        {
            if (tree.IsLeaf(nidx))
            {
                var leafScale = wProd * LeafValue(tree, nidx, targetIdx);
                for (var i = 0; i < Points; ++i) outH[i] = cVals[i] * leafScale * rule.Weights[i];
                return;
            }
            var left = tree.LeftChild(nidx);
            var right = tree.RightChild(nidx);
            var leftWeight = ChildWeight(nidx, left);
            var rightWeight = ChildWeight(nidx, right);
            var goesLeft = EvaluateGoesLeft(nidx);
            var rightH = new float[Points];
            VisitChild(nidx, left, leftWeight, goesLeft, cVals, wProd, outH);
            VisitChild(nidx, right, rightWeight, !goesLeft, cVals, wProd, rightH);
            for (var i = 0; i < Points; ++i) outH[i] += rightH[i];
        }

        public void Run()
        {
            formulation.Reset();
            if (tree.IsLeaf(RegTree.Root)) return;
            var cInit = new float[Points];
            Array.Fill(cInit, 1.0f);
            var hVals = new float[Points];
            RunNode(RegTree.Root, cInit, 1.0f, hVals);
        }
    }

    private readonly record struct TreeEntry(int TreeIdx, int TargetIdx, int GroupIdx, float Weight);

    private sealed class ModelData
    {
        public ITreeView[] Trees = [];
        public List<TreeEntry> Entries = [];
        public List<int>[] EntriesByGroup = [];
        public float[] GroupRootMeanSums = [];
    }

    private static ModelData MakeModelData(GBTreeModel model, int treeEnd, IReadOnlyList<float>? treeWeights)
    {
        var hTreeGroups = model.TreeGroups;
        var nGroups = (int)model.LearnerModelState.NumOutputGroup;
        var output = new ModelData
        {
            Trees = new ITreeView[treeEnd],
            EntriesByGroup = new List<int>[nGroups],
        };
        for (var g = 0; g < nGroups; ++g) output.EntriesByGroup[g] = [];
        for (var i = 0; i < treeEnd; ++i) output.Trees[i] = model.Trees[i].View();
        for (var i = 0; i < treeEnd; ++i)
        {
            var weight = treeWeights is null ? 1.0f : treeWeights[i];
            if (model.Trees[i].IsMultiTarget)
            {
                var nTargets = (int)model.Trees[i].GetMultiTargetTree().NumTargets;
                Check.Eq(nTargets, nGroups);
                for (var t = 0; t < nTargets; ++t)
                {
                    output.EntriesByGroup[t].Add(output.Entries.Count);
                    output.Entries.Add(new TreeEntry(i, t, t, weight));
                }
            }
            else
            {
                var gid = (int)hTreeGroups[i];
                output.EntriesByGroup[gid].Add(output.Entries.Count);
                output.Entries.Add(new TreeEntry(i, 0, gid, weight));
            }
        }
        var groupRootMeanSums = new double[nGroups];
        for (var i = 0; i < treeEnd; ++i)
        {
            var weight = treeWeights is null ? 1.0f : treeWeights[i];
            var tree = output.Trees[i];
            if (model.Trees[i].IsMultiTarget)
            {
                var nTargets = (int)tree.NumTargets;
                Check.Eq(nTargets, nGroups);
                var rootMeans = new double[nTargets];
                FillRootMeanValues(tree, RegTree.Root, 1.0, rootMeans);
                for (var t = 0; t < nTargets; ++t) groupRootMeanSums[t] += rootMeans[t] * weight;
            }
            else
            {
                groupRootMeanSums[hTreeGroups[i]] += FillRootMeanValue(tree, RegTree.Root) * weight;
            }
        }
        output.GroupRootMeanSums = Array.ConvertAll(groupRootMeanSums, v => (float)v);
        return output;
    }

    private static void ValidateTreeWeights(IReadOnlyList<float>? treeWeights, int treeEnd)
    {
        if (treeWeights is null) return;
        Check.Ge(treeWeights.Count, treeEnd);
    }

    private static void ForEachRow(Context ctx, DMatrix fmat, GBTreeModel model, Action<IRowView, long, int> fn)
    {
        var acc = CpuPredictor.MakeAccessor(model, fmat.Cats);
        foreach (var view in RowViews.Batches(ctx, fmat, acc))
        {
            var v = view;
            Threading.ParallelFor(v.Size, ctx.Threads(), (i, tid) => fn(v, i, tid));
        }
    }

    private static RegTree.FVec[] MakeFeats(int nThreads)
    {
        var f = new RegTree.FVec[nThreads];
        for (var i = 0; i < nThreads; ++i) f[i] = new RegTree.FVec();
        return f;
    }

    private static float BaseOf(MetaInfo info, TensorView<float> baseScore, long rowIdx, int gid, int nGroups)
    {
        var baseMargin = info.BaseMargin.HostView();
        if (baseMargin.Size != 0)
        {
            Check.Eq(baseMargin.Shape(1), (long)nGroups);
            return baseMargin[rowIdx, gid];
        }
        return baseScore[gid];
    }

    /// <summary><c>cpu_impl::ShapValues</c> (QuadratureTreeSHAP).</summary>
    public static void ShapValues(Context ctx, DMatrix fmat, HostDeviceVector<float> outContribs, GBTreeModel model, int treeEnd,
        IReadOnlyList<float>? treeWeights)
    {
        var info = fmat.Info;
        treeEnd = PredictFn.GetTreeLimit(model.Trees.Count, treeEnd);
        Check.Ge(treeEnd, 0);
        ValidateTreeWeights(treeWeights, treeEnd);
        var nThreads = ctx.Threads();
        var nGroups = (int)model.LearnerModelState.NumOutputGroup;
        var nFeatures = (int)model.LearnerModelState.NumFeature;
        var ncolumns = nFeatures + 1;
        outContribs.Resize((int)(info.NumRow * ncolumns * nGroups));
        outContribs.Fill(0.0f);
        var contribs = outContribs.RawArray;
        Check.Ne(nGroups, 0);
        var rule = Rule.Value;
        var baseScore = model.LearnerModelState.BaseScore();
        var modelData = MakeModelData(model, treeEnd, treeWeights);
        var featsTloc = MakeFeats(nThreads);
        var contribsTloc = new float[nThreads][];
        var pathProbTloc = new float[nThreads][];
        for (var t = 0; t < nThreads; ++t)
        {
            contribsTloc[t] = new float[ncolumns];
            pathProbTloc[t] = new float[nFeatures];
            Array.Fill(pathProbTloc[t], Unseen);
        }
        ForEachRow(ctx, fmat, model, (view, i, tid) =>
        {
            var feats = featsTloc[tid];
            if (feats.Size == 0) feats.Init(nFeatures);
            var thisTreeContribs = contribsTloc[tid];
            var pathProb = pathProbTloc[tid];
            var rowIdx = view.BaseRowId + i;
            RowViews.Fill(view, i, feats);
            for (var gid = 0; gid < nGroups; ++gid)
            {
                var pOff = (int)((rowIdx * nGroups + gid) * ncolumns);
                foreach (var entryIdx in modelData.EntriesByGroup[gid])
                {
                    var entry = modelData.Entries[entryIdx];
                    Array.Clear(thisTreeContribs);
                    var formulation = new Formulation { Phi = thisTreeContribs };
                    new Runner(modelData.Trees[entry.TreeIdx], entry.TargetIdx, feats, rule, pathProb, formulation).Run();
                    var weight = entry.Weight;
                    for (var ci = 0; ci + 1 < ncolumns; ++ci) contribs[pOff + ci] += thisTreeContribs[ci] * weight;
                }
                contribs[pOff + ncolumns - 1] += modelData.GroupRootMeanSums[gid];
                contribs[pOff + ncolumns - 1] += BaseOf(info, baseScore, rowIdx, gid, nGroups);
            }
            feats.Drop();
        });
    }

    /// <summary><c>cpu_impl::ApproxFeatureImportance</c>.</summary>
    public static void ApproxFeatureImportance(Context ctx, DMatrix fmat, HostDeviceVector<float> outContribs, GBTreeModel model,
        int treeEnd, IReadOnlyList<float>? treeWeights)
    {
        var info = fmat.Info;
        treeEnd = PredictFn.GetTreeLimit(model.Trees.Count, treeEnd);
        Check.Ge(treeEnd, 0);
        ValidateTreeWeights(treeWeights, treeEnd);
        var nThreads = ctx.Threads();
        var nFeatures = (int)model.LearnerModelState.NumFeature;
        var ncolumns = nFeatures + 1;
        var nGroups = (int)model.LearnerModelState.NumOutputGroup;
        outContribs.Resize((int)(info.NumRow * ncolumns * nGroups));
        outContribs.Fill(0.0f);
        var contribs = outContribs.RawArray;
        var meanValues = new float[treeEnd][];
        var views = new ScalarTreeView?[treeEnd];
        var isVectorLeaf = false;
        Threading.ParallelFor(treeEnd, nThreads, i =>
        {
            if (model.Trees[(int)i].IsMultiTarget)
            {
                isVectorLeaf = true;
                return;
            }
            var v = new ScalarTreeView(model.Trees[(int)i]);
            views[i] = v;
            meanValues[i] = new float[v.Size];
            FillNodeMeanValues(v, 0, meanValues[i]);
        });
        if (isVectorLeaf) Check.Fail("Approximate predict contribution " + ErrorMsg.MTNotImplemented);
        Check.Ne(nGroups, 0);
        var baseScore = model.LearnerModelState.BaseScore();
        var hTreeGroups = model.TreeGroups.ToArray();
        var featsTloc = MakeFeats(nThreads);
        var contribsTloc = new float[nThreads][];
        for (var t = 0; t < nThreads; ++t) contribsTloc[t] = new float[ncolumns];
        ForEachRow(ctx, fmat, model, (view, i, tid) =>
        {
            var feats = featsTloc[tid];
            if (feats.Size == 0) feats.Init(nFeatures);
            var thisTreeContribs = contribsTloc[tid];
            var rowIdx = view.BaseRowId + i;
            RowViews.Fill(view, i, feats);
            for (var gid = 0; gid < nGroups; ++gid)
            {
                var pOff = (int)((rowIdx * nGroups + gid) * ncolumns);
                for (var j = 0; j < treeEnd; ++j)
                {
                    if (hTreeGroups[j] != gid) continue;
                    Array.Clear(thisTreeContribs);
                    CalculateApproxContributions(views[j]!, feats, meanValues[j], thisTreeContribs);
                    var w = treeWeights is null ? 1.0f : treeWeights[j];
                    for (var ci = 0; ci < ncolumns; ++ci) contribs[pOff + ci] += thisTreeContribs[ci] * w;
                }
                contribs[pOff + ncolumns - 1] += BaseOf(info, baseScore, rowIdx, gid, nGroups);
            }
            feats.Drop();
        });
    }

    /// <summary><c>cpu_impl::ShapInteractionValues</c>.</summary>
    public static void ShapInteractionValues(Context ctx, DMatrix fmat, HostDeviceVector<float> outContribs, GBTreeModel model,
        int treeEnd, IReadOnlyList<float>? treeWeights, bool approximate)
    {
        if (!approximate)
        {
            QuadratureInteractionValues(ctx, fmat, outContribs, model, treeEnd, treeWeights);
            return;
        }
        Check.That(!model.LearnerModelState.IsVectorLeaf, "Predict interaction contribution" + ErrorMsg.MTNotImplemented);
        var info = fmat.Info;
        var ngroup = (int)model.LearnerModelState.NumOutputGroup;
        var ncolumns = (int)model.LearnerModelState.NumFeature;
        var rowChunk = ngroup * (ncolumns + 1) * (ncolumns + 1);
        var mrowChunk = (ncolumns + 1) * (ncolumns + 1);
        var crowChunk = ngroup * (ncolumns + 1);
        outContribs.Resize((int)(info.NumRow * ngroup * (ncolumns + 1) * (ncolumns + 1)));
        var contribs = outContribs.RawArray;
        var contribsOff = new HostDeviceVector<float>((int)(info.NumRow * ngroup * (ncolumns + 1)));
        var contribsOn = new HostDeviceVector<float>((int)(info.NumRow * ngroup * (ncolumns + 1)));
        var contribsDiag = new HostDeviceVector<float>((int)(info.NumRow * ngroup * (ncolumns + 1)));
        ApproxFeatureImportance(ctx, fmat, contribsDiag, model, treeEnd, treeWeights);
        for (var i = 0; i < ncolumns + 1; ++i)
        {
            ApproxFeatureImportance(ctx, fmat, contribsOff, model, treeEnd, treeWeights);
            ApproxFeatureImportance(ctx, fmat, contribsOn, model, treeEnd, treeWeights);
            var diag = contribsDiag.RawArray;
            var off = contribsOff.RawArray;
            var on = contribsOn.RawArray;
            for (long j = 0; j < info.NumRow; ++j)
            {
                for (var l = 0; l < ngroup; ++l)
                {
                    var oOffset = (int)(j * rowChunk + l * mrowChunk);
                    var cOffset = (int)(j * crowChunk + l * (ncolumns + 1));
                    ref var mii = ref contribs[oOffset + i * (ncolumns + 1) + i];
                    mii = 0;
                    for (var k = 0; k < ncolumns + 1; ++k)
                    {
                        if (k == i)
                        {
                            mii += diag[cOffset + k];
                        }
                        else
                        {
                            ref var mik = ref contribs[oOffset + i * (ncolumns + 1) + k];
                            mik = (on[cOffset + k] - off[cOffset + k]) / 2.0f;
                            mii -= mik;
                        }
                    }
                }
            }
        }
    }

    private static void QuadratureInteractionValues(Context ctx, DMatrix fmat, HostDeviceVector<float> outContribs, GBTreeModel model,
        int treeEnd, IReadOnlyList<float>? treeWeights)
    {
        var info = fmat.Info;
        treeEnd = PredictFn.GetTreeLimit(model.Trees.Count, treeEnd);
        Check.Ge(treeEnd, 0);
        ValidateTreeWeights(treeWeights, treeEnd);
        var nThreads = ctx.Threads();
        var nGroups = (int)model.LearnerModelState.NumOutputGroup;
        var nFeatures = (int)model.LearnerModelState.NumFeature;
        var ncolumns = nFeatures + 1;
        var rowChunk = nGroups * ncolumns * ncolumns;
        var matrixChunk = ncolumns * ncolumns;
        outContribs.Resize((int)(info.NumRow * rowChunk));
        outContribs.Fill(0.0f);
        var contribs = outContribs.RawArray;
        var rule = Rule.Value;
        var baseScore = model.LearnerModelState.BaseScore();
        var modelData = MakeModelData(model, treeEnd, treeWeights);
        var featsTloc = MakeFeats(nThreads);
        var pathTloc = new List<PathElement>[nThreads];
        var pathProbTloc = new float[nThreads][];
        var diagTloc = new float[nThreads][];
        for (var t = 0; t < nThreads; ++t)
        {
            pathTloc[t] = [];
            pathProbTloc[t] = new float[nFeatures];
            Array.Fill(pathProbTloc[t], Unseen);
            diagTloc[t] = new float[ncolumns];
        }
        ForEachRow(ctx, fmat, model, (view, i, tid) =>
        {
            var feats = featsTloc[tid];
            if (feats.Size == 0) feats.Init(nFeatures);
            var path = pathTloc[tid];
            var pathProb = pathProbTloc[tid];
            var diag = diagTloc[tid];
            var rowIdx = view.BaseRowId + i;
            RowViews.Fill(view, i, feats);
            for (var gid = 0; gid < nGroups; ++gid)
            {
                var offset = (int)((rowIdx * nGroups + gid) * matrixChunk);
                Array.Clear(diag);
                foreach (var entryIdx in modelData.EntriesByGroup[gid])
                {
                    var entry = modelData.Entries[entryIdx];
                    var formulation = new Formulation
                    {
                        Path = path, Phi = diag, Interactions = contribs, InterOffset = offset, NColumns = ncolumns, Scale = entry.Weight,
                    };
                    new Runner(modelData.Trees[entry.TreeIdx], entry.TargetIdx, feats, rule, pathProb, formulation).Run();
                }
                diag[ncolumns - 1] += modelData.GroupRootMeanSums[gid];
                diag[ncolumns - 1] += BaseOf(info, baseScore, rowIdx, gid, nGroups);
                for (var r = 0; r < ncolumns; ++r)
                {
                    for (var c = r + 1; c < ncolumns; ++c)
                    {
                        var sym = 0.5f * (contribs[offset + r * ncolumns + c] + contribs[offset + c * ncolumns + r]);
                        contribs[offset + r * ncolumns + c] = sym;
                        contribs[offset + c * ncolumns + r] = sym;
                    }
                }
                for (var r = 0; r < ncolumns; ++r)
                {
                    var value = diag[r];
                    for (var c = 0; c < ncolumns; ++c)
                        if (c != r) value -= contribs[offset + r * ncolumns + c];
                    contribs[offset + r * ncolumns + r] = value;
                }
            }
            feats.Drop();
        });
    }
}
