// Port of src/predictor/data_accessor.h and predict_fn.h.
using XGBoost.Tree;

namespace XGBoost.Predictors;

/// <summary>Row source that fills a dense feature vector (<c>DataToFeatVec</c>).</summary>
public interface IRowView
{
    long Size { get; }
    long BaseRowId { get; }

    /// <summary>Writes the valid features of row <paramref name="ridx"/> and returns their count.</summary>
    long DoFill(long ridx, Span<float> output);
}

public static class RowViews
{
    public static void Fill(IRowView view, long ridx, RegTree.FVec feats)
    {
        var nValid = view.DoFill(ridx, feats.Data);
        feats.HasMissing(nValid != feats.Size);
    }

    public static void FVecFill(IRowView view, Range1d block, int nFeatures, RegTree.FVec[] feats, int offset)
    {
        for (var i = 0; i < block.Size; ++i)
        {
            var f = feats[offset + i];
            if (f.Size == 0) f.Init(nFeatures);
            Fill(view, block.Begin + i, f);
        }
    }

    public static void FVecDrop(RegTree.FVec[] feats, int offset, long n)
    {
        for (var i = 0; i < n; ++i) feats[offset + i].Drop();
    }

    /// <summary>Iterates the row batches of a DMatrix as views (SparsePage when present, else gradient index).</summary>
    public static IEnumerable<IRowView> Batches(Context ctx, DMatrix fmat, CatAccessor? acc)
    {
        if (!fmat.SparsePageExists)
        {
            var ft = fmat.Info.FeatureTypes.ToArray();
            foreach (var page in fmat.GetGradientIndex(ctx, new BatchParam()))
                yield return new GHistIndexMatrixView(page, acc, ft);
        }
        else
        {
            foreach (var page in fmat.GetRowBatches())
                yield return new SparsePageView(page.GetView(), page.BaseRowId, acc);
        }
    }
}

public sealed class SparsePageView(HostSparsePageView view, long baseRowId, CatAccessor? acc) : IRowView
{
    public long Size => view.Size;
    public long BaseRowId => baseRowId;

    public long DoFill(long ridx, Span<float> output)
    {
        var row = view[ridx];
        if (acc is { } a)
        {
            foreach (var e in row) output[(int)e.Index] = a.Apply(e.Fvalue, e.Index);
        }
        else
        {
            foreach (var e in row) output[(int)e.Index] = e.Fvalue;
        }
        return row.Length;
    }
}

public sealed class GHistIndexMatrixView : IRowView
{
    private readonly GHistIndexMatrix _page;
    private readonly CatAccessor? _acc;
    private readonly FeatureType[] _ft;
    private readonly uint[] _ptrs;
    private readonly float[] _values;
    private readonly ColumnMatrix _columns;

    public GHistIndexMatrixView(GHistIndexMatrix page, CatAccessor? acc, FeatureType[] ft)
    {
        _page = page;
        _acc = acc;
        _ft = ft;
        _ptrs = page.Cut.Ptrs.ToArray();
        _values = page.Cut.Values.ToArray();
        _columns = page.Transpose();
        BaseRowId = page.BaseRowId;
    }

    public long BaseRowId { get; }
    public long Size => _page.Size;

    private float Acc(float v, int fidx) => _acc is { } a ? a.Apply(v, fidx) : v;

    public long DoFill(long ridx, Span<float> output)
    {
        var gridx = ridx + BaseRowId;
        var nFeatures = _page.Features;
        long nNonMissings = 0;
        if (_page.IsDense)
        {
            var rbeg = _page.RowPtr[ridx];
            for (var fidx = 0; fidx < nFeatures; ++fidx)
            {
                int binIdx;
                float fvalue;
                if (Categorical.IsCat(_ft, fidx))
                {
                    binIdx = _page.GetGindex(gridx, fidx);
                    fvalue = _values[binIdx];
                }
                else
                {
                    binIdx = (int)_page.Index[rbeg + fidx];
                    fvalue = HistogramCuts.NumericBinLowerBound(_ptrs, _values, fidx, binIdx);
                }
                output[fidx] = Acc(fvalue, fidx);
            }
            nNonMissings += nFeatures;
        }
        else
        {
            for (var fidx = 0; fidx < nFeatures; ++fidx)
            {
                var fvalue = float.NaN;
                var isCat = Categorical.IsCat(_ft, fidx);
                if (_columns.GetColumnType(fidx) == ColumnType.Sparse)
                {
                    var binIdx = _page.GetGindex(gridx, fidx);
                    if (binIdx != -1)
                        fvalue = isCat ? _values[binIdx] : HistogramCuts.NumericBinLowerBound(_ptrs, _values, fidx, binIdx);
                }
                else
                {
                    if (isCat)
                    {
                        fvalue = _page.GetFvalue(_ptrs, _values, gridx, fidx, isCat);
                    }
                    else
                    {
                        var binIdx = _page.GetGindex(gridx, fidx);
                        if (binIdx != -1) fvalue = HistogramCuts.NumericBinLowerBound(_ptrs, _values, fidx, binIdx);
                    }
                }
                if (!float.IsNaN(fvalue))
                {
                    output[fidx] = Acc(fvalue, fidx);
                    nNonMissings++;
                }
            }
        }
        return nNonMissings;
    }
}

/// <summary>Row-major adapter batch view for inplace prediction, <c>AdapterView</c>.</summary>
public sealed class AdapterView<TBatch>(TBatch batch, long nRows, float missing, CatAccessor? acc) : IRowView
    where TBatch : IAdapterBatch
{
    public long Size => nRows;
    public long BaseRowId => 0;

    public long DoFill(long ridx, Span<float> output)
    {
        long nNonMissings = 0;
        var n = batch.LineSize(ridx);
        for (var c = 0; c < n; ++c)
        {
            var e = batch.GetElement(ridx, c);
            if (missing != e.Value && !float.IsNaN(e.Value))
            {
                output[(int)e.ColumnIdx] = acc is { } a ? a.Apply(e.Value, e.ColumnIdx) : e.Value;
                nNonMissings++;
            }
        }
        return nNonMissings;
    }
}

public static class PredictFn
{
    public static bool GetDecision(ITreeView tree, int nid, float fvalue, CategoricalSplitMatrix cats, bool hasCategorical)
    {
        if (hasCategorical && Categorical.IsCat(cats.SplitType, nid))
            return Categorical.Decision(cats.NodeCats(nid), fvalue);
        return fvalue < tree.SplitCond(nid);
    }

    public static int GetNextNode(ITreeView tree, int nid, float fvalue, bool isMissing, CategoricalSplitMatrix cats, bool hasCategorical)
    {
        if (isMissing) return tree.DefaultChild(nid);
        return tree.LeftChild(nid) + (GetDecision(tree, nid, fvalue, cats, hasCategorical) ? 0 : 1);
    }

    public static int GetTreeLimit(int nTrees, int ntreeLimit)
    {
        if (ntreeLimit == 0 || ntreeLimit > nTrees) ntreeLimit = nTrees;
        return ntreeLimit;
    }

    public static int GetLeafIndex(ITreeView tree, RegTree.FVec feat, CategoricalSplitMatrix cats, int nidx)
    {
        var hasCat = tree.HasCategoricalSplit;
        var hasMissing = feat.HasMissing();
        while (!tree.IsLeaf(nidx))
        {
            var splitIndex = (int)tree.SplitIndex(nidx);
            var fvalue = feat.GetFvalue(splitIndex);
            nidx = GetNextNode(tree, nidx, fvalue, hasMissing && feat.IsMissing(splitIndex), cats, hasCat);
        }
        return nidx;
    }
}
