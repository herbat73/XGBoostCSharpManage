// Port of src/predictor/predictor.cc and cpu_predictor.cc. The array tree layout optimisation of
// the C++ code only changes how a leaf is found, so a plain traversal gives identical results.
using XGBoost.Gbm;
using XGBoost.Tree;

namespace XGBoost.Predictors;

public sealed class CpuPredictor(Context ctx)
{
    private const int BlockOfRowsSize = 64;

    public Context Ctx => ctx;

    private static void ValidateBaseMarginShape(Tensor<float> margin, long nSamples, long nGroups)
    {
        var expected = $"Invalid shape of base_margin. Expected: ({nSamples}, {nGroups})";
        Check.Eq(margin.Shape(0), nSamples, expected);
        Check.Eq(margin.Shape(1), nGroups, expected);
    }

    /// <summary><c>Predictor::InitOutPredictions</c>.</summary>
    public void InitOutPredictions(MetaInfo info, HostDeviceVector<float> outPreds, GBTreeModel model)
    {
        Check.Ne(model.LearnerModelState.NumOutputGroup, 0u);
        var n = (int)(model.LearnerModelState.OutputLength * info.NumRow);
        outPreds.Resize(n);
        var baseMargin = info.BaseMargin.Data;
        if (!baseMargin.Empty)
        {
            ValidateBaseMarginShape(info.BaseMargin, info.NumRow, model.LearnerModelState.OutputLength);
            outPreds.Copy(baseMargin);
            return;
        }
        var baseScore = model.LearnerModelState.BaseScore();
        if (baseScore.Size == 1)
        {
            outPreds.Fill(baseScore[0]);
            return;
        }
        var m = (long)model.LearnerModelState.OutputLength;
        var predt = Linalg.MakeTensorView(outPreds, info.NumRow, m);
        Check.Eq(predt.Size, (long)outPreds.Size);
        Threading.ParallelFor(info.NumRow, ctx.Threads(), i =>
        {
            for (long j = 0; j < m; ++j) predt[i, j] = baseScore[j];
        });
    }

    /// <summary>Tree views for [begin, end), with their output groups.</summary>
    internal sealed class ModelView
    {
        public readonly ITreeView[] Trees;
        public readonly uint[] TreeGroups;
        public readonly int NFeatures;

        public ModelView(GBTreeModel model, int treeBegin, int treeEnd)
        {
            Check.Ge(treeEnd, treeBegin);
            Trees = new ITreeView[treeEnd - treeBegin];
            for (var t = treeBegin; t < treeEnd; ++t) Trees[t - treeBegin] = model.Trees[t].View();
            TreeGroups = model.TreeGroups[treeBegin..treeEnd].ToArray();
            NFeatures = (int)model.LearnerModelState.NumFeature;
        }
    }

    private static bool ShouldUseBlock(DMatrix fmat)
    {
        const double densityThresh = .125;
        var nSamples = fmat.Info.NumRow;
        var total = Math.Max(nSamples * fmat.Info.NumCol, 1L);
        var density = (double)fmat.Info.NumNonZero / total;
        return density > densityThresh;
    }

    /// <summary>Recoding accessor when the data categories differ from the training ones.</summary>
    internal static CatAccessor? MakeAccessor(GBTreeModel model, CatContainer dataCats)
    {
        if (model.Cats().HasCategorical && dataCats.NeedRecode) return model.Cats().MakeAccessor(dataCats.HostView()).Accessor;
        return null;
    }

    private static void PredictByAllTrees(ModelView model, long predictOffset, RegTree.FVec[] fvec, int fOffset, long blockSize,
        TensorView<float> outPredt, OptionalWeights treeWeights)
    {
        var trees = model.Trees;
        for (var treeId = 0; treeId < trees.Length; ++treeId)
        {
            var weight = treeWeights[treeId];
            var tree = trees[treeId];
            var cats = tree.GetCategoriesMatrix();
            if (tree is ScalarTreeView sc)
            {
                var gid = model.TreeGroups[treeId];
                for (var i = 0; i < blockSize; ++i)
                {
                    var leaf = PredictFn.GetLeafIndex(sc, fvec[fOffset + i], cats, RegTree.Root);
                    outPredt[predictOffset + i, gid] += sc.ScalarLeafValue(leaf) * weight;
                }
            }
            else
            {
                for (var i = 0; i < blockSize; ++i)
                {
                    var leaf = PredictFn.GetLeafIndex(tree, fvec[fOffset + i], cats, RegTree.Root);
                    var leafValue = tree.LeafValue(leaf);
                    for (var j = 0; j < leafValue.Length; ++j) outPredt[predictOffset + i, j] += leafValue[j] * weight;
                }
            }
        }
    }

    private sealed class ThreadTmp(int nThreads, int blockOfRowsSize)
    {
        public readonly RegTree.FVec[] FeatVecs = MakeVecs(nThreads * blockOfRowsSize);
        public int Offset(int tid) => tid * blockOfRowsSize;

        private static RegTree.FVec[] MakeVecs(int n)
        {
            var v = new RegTree.FVec[n];
            for (var i = 0; i < n; ++i) v[i] = new RegTree.FVec();
            return v;
        }
    }

    private static void PredictBatchByBlockKernel(IRowView batch, ModelView model, int blockOfRowsSize, int nThreads,
        TensorView<float> outPredt, OptionalWeights treeWeights)
    {
        var tmp = new ThreadTmp(nThreads, blockOfRowsSize);
        var nSamples = batch.Size;
        var nFeatures = model.NFeatures;
        Threading.ParallelFor1d(nSamples, nThreads, blockOfRowsSize, block =>
        {
            var offset = tmp.Offset(Threading.ThreadNum);
            RowViews.FVecFill(batch, block, nFeatures, tmp.FeatVecs, offset);
            PredictByAllTrees(model, block.Begin + batch.BaseRowId, tmp.FeatVecs, offset, block.Size, outPredt, treeWeights);
            RowViews.FVecDrop(tmp.FeatVecs, offset, block.Size);
        });
    }

    private void PredictDMatrix(DMatrix fmat, HostDeviceVector<float> outPreds, GBTreeModel model, int treeBegin, int treeEnd,
        OptionalWeights treeWeights)
    {
        var nThreads = ctx.Threads();
        long nGroups = model.LearnerModelState.OutputLength;
        var nSamples = fmat.Info.NumRow;
        Check.Eq((long)outPreds.Size, nSamples * nGroups);
        var outPredt = Linalg.MakeTensorView(outPreds, nSamples, nGroups);
        var hModel = new ModelView(model, treeBegin, treeEnd);
        var blockSize = ShouldUseBlock(fmat) ? BlockOfRowsSize : 1;
        var acc = MakeAccessor(model, fmat.Cats);
        foreach (var batch in RowViews.Batches(ctx, fmat, acc))
            PredictBatchByBlockKernel(batch, hModel, blockSize, nThreads, outPredt, treeWeights);
    }

    private static OptionalWeights TreeWeightsView(IReadOnlyList<float>? treeWeights, int treeBegin, int treeEnd)
    {
        if (treeWeights is null) return new OptionalWeights(1.0f);
        var w = new float[treeEnd - treeBegin];
        for (var i = treeBegin; i < treeEnd; ++i) w[i - treeBegin] = treeWeights[i];
        return new OptionalWeights(w, w.Length);
    }

    /// <summary><c>CPUPredictor::PredictBatch</c>: adds the predictions of trees [begin, end).</summary>
    public void PredictBatch(DMatrix dmat, HostDeviceVector<float> outPreds, GBTreeModel model, int treeBegin, int treeEnd = 0,
        IReadOnlyList<float>? treeWeightsOverride = null)
    {
        if (treeEnd == 0) treeEnd = model.Trees.Count;
        var treeWeights = treeWeightsOverride ?? model.TreeWeights;
        PredictDMatrix(dmat, outPreds, model, treeBegin, treeEnd, TreeWeightsView(treeWeights, treeBegin, treeEnd));
    }

    /// <summary><c>CPUPredictor::InplacePredict</c> over a row-major adapter batch.</summary>
    public void InplacePredict<TBatch>(TBatch batch, long nRows, long nCols, MetaInfo info, GBTreeModel model, float missing,
        HostDeviceVector<float> outPreds, int treeBegin, int treeEnd, ColumnsView? dataCats) where TBatch : IAdapterBatch
    {
        Check.Eq(nCols, (long)model.LearnerModelState.NumFeature, "Number of columns in data must equal to the trained model.");
        if (treeEnd == 0) treeEnd = model.Trees.Count;
        InitOutPredictions(info, outPreds, model);
        long nGroups = model.LearnerModelState.OutputLength;
        var hModel = new ModelView(model, treeBegin, treeEnd);
        var weights = TreeWeightsView(model.TreeWeights, treeBegin, treeEnd);
        CatAccessor? acc = null;
        if (model.Cats().HasCategorical && dataCats is { Empty: false }) acc = model.Cats().MakeAccessor(dataCats).Accessor;
        var view = new AdapterView<TBatch>(batch, nRows, missing, acc);
        var outPredt = Linalg.MakeTensorView(outPreds, view.Size, nGroups);
        PredictBatchByBlockKernel(view, hModel, BlockOfRowsSize, ctx.Threads(), outPredt, weights);
    }

    /// <summary><c>PredictLeafCPU</c>.</summary>
    public void PredictLeaf(DMatrix fmat, HostDeviceVector<float> outPreds, GBTreeModel model, int ntreeLimit)
    {
        var nThreads = ctx.Threads();
        ntreeLimit = PredictFn.GetTreeLimit(model.Trees.Count, ntreeLimit);
        var info = fmat.Info;
        outPreds.Resize((int)(info.NumRow * ntreeLimit));
        var preds = outPreds.RawArray;
        var nFeatures = (int)model.LearnerModelState.NumFeature;
        var tmp = new ThreadTmp(nThreads, 1);
        var hModel = new ModelView(model, 0, ntreeLimit);
        var acc = MakeAccessor(model, fmat.Cats);
        foreach (var batch in RowViews.Batches(ctx, fmat, acc))
        {
            Threading.ParallelFor1d(batch.Size, nThreads, 1, block =>
            {
                var ridx = batch.BaseRowId + block.Begin;
                var offset = tmp.Offset(Threading.ThreadNum);
                RowViews.FVecFill(batch, block, nFeatures, tmp.FeatVecs, offset);
                for (var j = 0; j < ntreeLimit; ++j)
                {
                    var tree = hModel.Trees[j];
                    var nidx = PredictFn.GetLeafIndex(tree, tmp.FeatVecs[offset], tree.GetCategoriesMatrix(), RegTree.Root);
                    preds[ridx * ntreeLimit + j] = nidx;
                }
                RowViews.FVecDrop(tmp.FeatVecs, offset, block.Size);
            });
        }
    }

    /// <summary><c>PredictFromLeafIds</c>: adds leaf values of known (encoded) leaf positions.</summary>
    public void PredictFromLeafIds(IReadOnlyList<HostDeviceVector<int>> leafIds, IReadOnlyList<RegTree> trees, TensorView<float> outPreds)
    {
        Check.Eq(leafIds.Count, trees.Count);
        for (var treeIdx = 0; treeIdx < trees.Count; ++treeIdx)
        {
            var pTree = trees[treeIdx];
            var hLeafIds = leafIds[treeIdx].RawArray;
            Check.Eq((long)leafIds[treeIdx].Size, outPreds.Shape(0));
            if (!pTree.IsMultiTarget)
            {
                Check.Eq(outPreds.Shape(1), 1L);
                var nodes = pTree.GetNodes().ToArray();
                Threading.ParallelFor(outPreds.Shape(0), ctx.Threads(), rowIdx =>
                {
                    var nidx = SamplePosition.Decode(hLeafIds[rowIdx]);
                    outPreds[rowIdx, 0] += nodes[nidx].LeafValue;
                });
            }
            else
            {
                var tree = new MultiTargetTreeView(pTree);
                var nTargets = tree.NumTargets;
                Check.Eq(outPreds.Shape(1), (long)nTargets);
                Threading.ParallelFor(outPreds.Shape(0), ctx.Threads(), rowIdx =>
                {
                    var nidx = SamplePosition.Decode(hLeafIds[rowIdx]);
                    var weight = tree.LeafValue(nidx);
                    for (var t = 0; t < nTargets; ++t) outPreds[rowIdx, t] += weight[t];
                });
            }
        }
    }
}

/// <summary><c>tree::SamplePosition</c>: sampled-out rows are stored as the bitwise complement.</summary>
public static class SamplePosition
{
    public static int Encode(int nidx, bool isValid) => isValid ? nidx : ~nidx;
    public static int Decode(int nidx) => IsValid(nidx) ? nidx : ~nidx;
    public static bool IsValid(int nidx) => nidx >= 0;
}
