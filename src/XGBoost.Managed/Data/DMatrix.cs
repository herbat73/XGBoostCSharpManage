// Port of the DMatrix class in include/xgboost/data.h and the DMatrix entry points of
// src/data/data.cc and src/c_api/c_api.cc. This is both the internal data interface and the public API.
using System.Numerics;
using System.Runtime.CompilerServices;
using XGBoost.Common;
using XGBoost.Data;

namespace XGBoost;

/// <summary>
/// Training or prediction data: a feature matrix plus <see cref="MetaInfo"/> (labels, weights, groups, ...).
/// Equivalent to <c>xgboost::DMatrix</c>.
/// </summary>
public abstract class DMatrix : IDisposable
{
    /// <summary>Meta information about the dataset.</summary>
    public abstract MetaInfo Info { get; }

    /// <summary>Context created from the <c>nthread</c> used at construction.</summary>
    public abstract Context Ctx { get; }

    // ---- batch access (internal interface) ----------------------------------------------------

    public abstract IEnumerable<SparsePage> GetRowBatches();
    public abstract IEnumerable<CscPage> GetColumnBatches(Context ctx);
    public abstract IEnumerable<SortedCscPage> GetSortedColumnBatches(Context ctx);
    public abstract IEnumerable<GHistIndexMatrix> GetGradientIndex(Context ctx, BatchParam param);

    /// <summary>Rows exported for prediction and data access (<c>ExtSparsePage</c>).</summary>
    public virtual IEnumerable<SparsePage> GetExtBatches(Context ctx, BatchParam param) => GetRowBatches();

    public abstract bool GHistIndexExists { get; }
    public abstract bool SparsePageExists { get; }
    public virtual bool EllpackExists => false;

    public bool SingleColBlock => NumBatches == 1;
    public virtual int NumBatches => 1;
    public virtual long[] BatchPtr => [0, Info.NumRow];
    public bool IsDenseMatrix => Info.IsDense;

    /// <summary>Row slice, <c>DMatrix::Slice</c>.</summary>
    public abstract DMatrix SliceRows(ReadOnlySpan<int> ridxs);

    public CatContainer Cats => Info.Cats();

    public virtual void Dispose() => GC.SuppressFinalize(this);

    // ---- construction -----------------------------------------------------------------------

    /// <summary><c>DMatrix::Create(adapter, missing, nthread)</c>.</summary>
    public static DMatrix Create<TBatch>(IAdapter<TBatch> adapter, float missing, int nthread) where TBatch : IAdapterBatch =>
        SimpleDMatrix.FromAdapter(adapter, missing, nthread);

    /// <summary>Creates a DMatrix from a row-major dense matrix (<c>XGDMatrixCreateFromDense</c>).</summary>
    public static DMatrix FromDense(ReadOnlySpan<float> data, int rows, int cols, float missing = float.NaN, int nthread = 0)
    {
        if ((long)rows * cols != data.Length)
            throw new ArgumentException($"Expected {(long)rows * cols} values for a {rows}x{cols} matrix, got {data.Length}.", nameof(data));
        var arr = data.ToArray();
        var batch = new ArrayAdapterBatch<float>(arr, rows, cols, cols, 1);
        return Create(new SingleBatchAdapter<ArrayAdapterBatch<float>>(batch, rows, cols), missing, nthread);
    }

    /// <summary>Creates a DMatrix from a 2-D array.</summary>
    public static DMatrix FromDense(float[,] data, float missing = float.NaN, int nthread = 0)
    {
        var rows = data.GetLength(0);
        var cols = data.GetLength(1);
        var flat = new float[rows * cols];
        Buffer.BlockCopy(data, 0, flat, 0, flat.Length * sizeof(float));
        return FromDense(flat, rows, cols, missing, nthread);
    }

    /// <summary>Creates a DMatrix from a row-major dense matrix of any numeric type.</summary>
    public static DMatrix FromDense<T>(T[] data, int rows, int cols, float missing = float.NaN, int nthread = 0)
        where T : INumberBase<T>
    {
        if ((long)rows * cols != data.Length)
            throw new ArgumentException($"Expected {(long)rows * cols} values for a {rows}x{cols} matrix, got {data.Length}.", nameof(data));
        var batch = new ArrayAdapterBatch<T>(data, rows, cols, cols, 1);
        return Create(new SingleBatchAdapter<ArrayAdapterBatch<T>>(batch, rows, cols), missing, nthread);
    }

    /// <summary><c>XGDMatrixCreateFromMat</c>: raw float matrix with a missing value marker.</summary>
    public static DMatrix FromMat(ReadOnlySpan<float> data, int rows, int cols, float missing, int nthread = 1)
    {
        var batch = new DenseAdapterBatch(data.ToArray(), 0, rows, cols);
        return Create(new SingleBatchAdapter<DenseAdapterBatch>(batch, rows, cols), missing, nthread);
    }

    /// <summary>Creates a DMatrix from CSR arrays (<c>XGDMatrixCreateFromCSR</c>).</summary>
    // Preferred over the generic overload so calls with collection expressions resolve as they do in the bindings.
    [OverloadResolutionPriority(1)]
    public static DMatrix FromCsr(ReadOnlySpan<long> indptr, ReadOnlySpan<uint> indices, ReadOnlySpan<float> values, long numCols,
        float missing = float.NaN, int nthread = 0) =>
        FromCsr<long, uint, float>(indptr.ToArray(), indices.ToArray(), values.ToArray(), numCols, missing, nthread);

    public static DMatrix FromCsr<TPtr, TIdx, TVal>(TPtr[] indptr, TIdx[] indices, TVal[] values, long ncol,
        float missing = float.NaN, int nthread = 0)
        where TPtr : INumberBase<TPtr> where TIdx : INumberBase<TIdx> where TVal : INumberBase<TVal>
    {
        var batch = new CsrAdapterBatch<TPtr, TIdx, TVal>(indptr, indices, values, ncol);
        var rows = indptr.Length == 0 ? 0 : indptr.Length - 1;
        return Create(new SingleBatchAdapter<CsrAdapterBatch<TPtr, TIdx, TVal>>(batch, rows, ncol), missing, nthread);
    }

    /// <summary>Creates a DMatrix from CSC arrays (<c>XGDMatrixCreateFromCSC</c>); <paramref name="nrow"/> 0 means unknown.</summary>
    public static DMatrix FromCsc<TPtr, TIdx, TVal>(TPtr[] indptr, TIdx[] indices, TVal[] values, long nrow,
        float missing = float.NaN, int nthread = 0)
        where TPtr : INumberBase<TPtr> where TIdx : INumberBase<TIdx> where TVal : INumberBase<TVal>
    {
        if (nthread <= 0) nthread = Threading.OmpGetNumThreads(0);
        return Create(new CscAdapter<TPtr, TIdx, TVal>(indptr, indices, values, nrow), missing, nthread);
    }

    /// <summary>
    /// Creates a DMatrix from columns, which may be categorical (<c>XGDMatrixCreateFromColumnar</c>).
    /// The categories are stored with the matrix and recorded in trained models.
    /// </summary>
    public static DMatrix FromColumns(ColumnarColumn[] columns, float missing = float.NaN, int nthread = 0) =>
        SimpleDMatrix.FromColumnar(new ColumnarAdapter(columns), missing, nthread);

    /// <summary>
    /// Loads a DMatrix from a binary DMatrix file, a LIBSVM file or a CSV file
    /// (<c>XGDMatrixCreateFromURI</c>). Append <c>?format=csv</c> or <c>?format=libsvm</c> to choose.
    /// </summary>
    public static DMatrix FromFile(string uri, bool silent = true) => Load(uri, silent);

    public static DMatrix Load(string uri, bool silent = true)
    {
        Check.That(!uri.Contains('#'), "External memory training with text input has been removed.");
        var loaded = SimpleDMatrix.TryLoadBinary(uri, silent);
        if (loaded is not null) return loaded;
        Log.WarningOnce("text-input", "Text file input has been deprecated since 3.1");
        var blocks = TextParser.Parse(uri);
        return Create(new FileAdapter(blocks), float.NaN, new Context().Threads());
    }

    // ---- public accessors -------------------------------------------------------------------

    public long NumRows => Info.NumRow;
    public long NumCols => Info.NumCol;
    public long NumNonMissing => Info.NumNonZero;

    /// <summary>Labels, flattened row-major.</summary>
    public float[] Label { get => GetFloatInfo("label"); set => SetInfo("label", value); }

    public float[] Weight { get => GetFloatInfo("weight"); set => SetInfo("weight", value); }

    public float[] BaseMargin { get => GetFloatInfo("base_margin"); set => SetInfo("base_margin", value); }

    public float[] LabelLowerBound { get => GetFloatInfo("label_lower_bound"); set => SetInfo("label_lower_bound", value); }

    public float[] LabelUpperBound { get => GetFloatInfo("label_upper_bound"); set => SetInfo("label_upper_bound", value); }

    public float[] FeatureWeights { get => GetFloatInfo("feature_weights"); set => SetInfo("feature_weights", value); }

    /// <summary>Sets query group sizes (<c>group</c>).</summary>
    public void SetGroup(ReadOnlySpan<uint> groupSizes) => SetInfo("group", groupSizes);

    /// <summary>Sets query ids, one per row, sorted (<c>qid</c>).</summary>
    public void SetQueryId(ReadOnlySpan<uint> qid) => SetInfo("qid", qid);

    /// <summary>Sets a (n_samples, n_targets) label matrix.</summary>
    public void SetLabel(ReadOnlySpan<float> values, int targets)
    {
        Check.Eq(values.Length % targets, 0, "Size of label must be divisible by the number of targets.");
        Info.SetInfo("label", values, [values.Length / targets, targets]);
    }

    /// <summary>Sets a meta field from typed values; see <see cref="MetaInfo.SetInfo{T}"/>.</summary>
    public virtual void SetInfo<T>(string field, ReadOnlySpan<T> values, long[]? shape = null) where T : unmanaged, INumberBase<T> =>
        Info.SetInfo(field, values, shape);

    public void SetInfo(string field, float[] values) => SetInfo<float>(field, values);

    public float[] GetFloatInfo(string field) => Info.GetFloatInfo(field).Values;

    public uint[] GetUIntInfo(string field) => Info.GetUIntInfo(field);

    public string[] FeatureNames
    {
        get => Info.GetFeatureInfo("feature_name");
        set => Info.SetFeatureInfo("feature_name", value);
    }

    public string[] FeatureTypes
    {
        get => Info.GetFeatureInfo("feature_type");
        set => Info.SetFeatureInfo("feature_type", value);
    }

    /// <summary>Row slice (<c>XGDMatrixSliceDMatrixEx</c>).</summary>
    public DMatrix Slice(ReadOnlySpan<int> rows, bool allowGroups = false)
    {
        if (!allowGroups) Check.Eq(Info.GroupPtr.Count, 0, "slice does not support group structure");
        return SliceRows(rows);
    }

    /// <summary>Saves as a binary DMatrix file (<c>XGDMatrixSaveBinary</c>).</summary>
    public void SaveBinary(string path, bool silent = true)
    {
        _ = silent;
        if (this is SimpleDMatrix s) s.SaveToLocalFile(path);
        else Check.Fail("binary saving only supported by SimpleDMatrix");
    }

    /// <summary>Exports the data as CSR (<c>XGDMatrixGetDataAsCSR</c>).</summary>
    public (long[] Indptr, uint[] Indices, float[] Data) GetDataAsCsr()
    {
        var indptr = new List<long>();
        var indices = new List<uint>();
        var data = new List<float>();
        foreach (var page in GetExtBatches(Ctx, new BatchParam()))
        {
            indptr.AddRange(page.Offset.ConstHostSpan);
            foreach (var e in page.Data.ConstHostSpan)
            {
                indices.Add(e.Index);
                data.Add(e.Fvalue);
            }
        }
        return ([.. indptr], [.. indices], [.. data]);
    }

    /// <summary>Quantile cuts of a QuantileDMatrix or a DMatrix already used for hist training.</summary>
    public (ulong[] Indptr, float[] Data) GetQuantileCut()
    {
        if (!GHistIndexExists)
            Check.Fail("The quantile cut hasn't been generated yet. Unless this is a `QuantileDMatrix`, quantile cut is generated during training.");
        var ctx = Ctx.IsCpu ? Ctx : Ctx.MakeCpu();
        foreach (var page in GetGradientIndex(ctx, new BatchParam()))
        {
            var cut = page.Cuts;
            var ptrs = cut.Ptrs;
            var vals = cut.Values;
            var ft = Info.FeatureTypes.ConstHostSpan;
            var indptr = new ulong[ptrs.Size];
            var data = new List<float>();
            for (var fidx = 0; fidx < Info.NumCol; fidx++)
            {
                indptr[fidx] = (ulong)data.Count;
                if (!Categorical.IsCat(ft, fidx)) data.Add(HistogramCuts.NumericBinLowerBound(ptrs.ConstHostSpan, vals.ConstHostSpan, fidx, ptrs[fidx]));
                for (var i = ptrs[fidx]; i < ptrs[fidx + 1]; i++) data.Add(vals[(int)i]);
            }
            indptr[^1] = (ulong)data.Count;
            return (indptr, [.. data]);
        }
        return ([], []);
    }
}
