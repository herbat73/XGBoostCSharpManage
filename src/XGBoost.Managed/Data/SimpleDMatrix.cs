// Port of src/data/simple_dmatrix.h/.cc: the in-memory DMatrix.
using XGBoost.Common;

namespace XGBoost.Data;

public sealed class SimpleDMatrix : DMatrix
{
    public const int Magic = unchecked((int)0xffffab01);

    private MetaInfo _info = new();
    private readonly SparsePage _sparsePage = new();
    private CscPage? _columnPage;
    private SortedCscPage? _sortedColumnPage;
    private GHistIndexMatrix? _gradientIndex;
    private BatchParam _batchParam = new();
    private Context _fmatCtx = new();

    private SimpleDMatrix() => _fmatCtx.Init([]);

    public override MetaInfo Info => _info;
    public override Context Ctx => _fmatCtx;

    public SparsePage Page => _sparsePage;

    public override DMatrix SliceRows(ReadOnlySpan<int> ridxs)
    {
        var output = new SimpleDMatrix();
        var outPage = output._sparsePage;
        var hRidx = new long[ridxs.Length];
        for (var i = 0; i < ridxs.Length; i++) hRidx[i] = ridxs[i];
        foreach (var page in GetRowBatches())
        {
            var batch = page.GetView();
            long rptr = 0;
            foreach (var ridx in ridxs)
            {
                var inst = batch[ridx];
                rptr += inst.Length;
                outPage.Data.Extend(inst);
                outPage.Offset.Add(rptr);
            }
            output._info = _info.Slice(hRidx, outPage.Offset[outPage.Offset.Size - 1]);
        }
        output._fmatCtx = _fmatCtx.Clone();
        output._info.Cats().Copy(_info.Cats());
        return output;
    }

    public override IEnumerable<SparsePage> GetRowBatches()
    {
        yield return _sparsePage;
    }

    public override IEnumerable<CscPage> GetColumnBatches(Context ctx)
    {
        if (_columnPage is null)
        {
            if (_sparsePage.Size > uint.MaxValue) ErrorMsg.MaxSampleSize(uint.MaxValue);
            _columnPage = new CscPage(_sparsePage.GetTranspose((int)_info.NumCol, ctx.Threads()));
        }
        yield return _columnPage;
    }

    public override IEnumerable<SortedCscPage> GetSortedColumnBatches(Context ctx)
    {
        if (_sortedColumnPage is null)
        {
            if (_sparsePage.Size > uint.MaxValue) ErrorMsg.MaxSampleSize(uint.MaxValue);
            _sortedColumnPage = new SortedCscPage(_sparsePage.GetTranspose((int)_info.NumCol, ctx.Threads()));
            _sortedColumnPage.SortRows(ctx.Threads());
        }
        yield return _sortedColumnPage;
    }

    public override IEnumerable<GHistIndexMatrix> GetGradientIndex(Context ctx, BatchParam param)
    {
        BatchUtils.CheckEmpty(_batchParam, param);
        if (_gradientIndex is not null && param.Initialized && param.ForbidRegen)
        {
            if (BatchUtils.RegenGHist(_batchParam, param)) Check.Eq(_batchParam.MaxBin, param.MaxBin, ErrorMsg.InconsistentMaxBin);
            Check.That(!BatchUtils.RegenGHist(_batchParam, param), "Inconsistent sparse threshold.");
        }
        if (_gradientIndex is null || BatchUtils.RegenGHist(_batchParam, param))
        {
            Log.Debug("Generating new Gradient Index.");
            Check.Ge(param.MaxBin, 2);
            var sortedSketch = param.Regen;
            _gradientIndex = new GHistIndexMatrix(ctx, this, param.MaxBin, param.SparseThresh, sortedSketch, param.Hess ?? []);
            _batchParam = param.MakeCache();
        }
        yield return _gradientIndex;
    }

    public override bool GHistIndexExists => _gradientIndex is not null;
    public override bool SparsePageExists => true;

    /// <summary>Constructs from an adapter, the templated <c>SimpleDMatrix</c> constructor.</summary>
    public static SimpleDMatrix FromAdapter<TBatch>(IAdapter<TBatch> adapter, float missing, int nthread,
        Action<SimpleDMatrix>? setCategories = null) where TBatch : IAdapterBatch
    {
        var m = new SimpleDMatrix();
        var ctx = new Context();
        ctx.Init([new("nthread", nthread.ToString(System.Globalization.CultureInfo.InvariantCulture))]);
        var info = m._info;
        const ulong defaultMax = ulong.MaxValue;
        var lastGroupId = defaultMax;
        uint groupSize = 0;
        long inferredNumColumns = 0;
        long totalBatchSize = 0;
        var labels = new List<float>();
        var hasLabels = false;

        adapter.BeforeFirst();
        while (adapter.Next())
        {
            var batch = adapter.Value;
            var batchMaxColumns = m._sparsePage.Push(batch, missing, ctx.Threads());
            inferredNumColumns = Math.Max(batchMaxColumns, inferredNumColumns);
            totalBatchSize += batch.Size;
            if (batch.Labels is { } bl)
            {
                hasLabels = true;
                labels.AddRange(bl.AsSpan(0, (int)batch.Size));
            }
            if (batch.Weights is { } bw) info.Weights.Extend(bw.AsSpan(0, (int)batch.Size));
            if (batch.BaseMargin is { } bm)
                info.BaseMargin = new Tensor<float>(bm.AsSpan(0, (int)batch.Size).ToArray(), [batch.Size, 1]);
            if (batch.Qid is { } qid)
            {
                for (var i = 0; i < batch.Size; i++)
                {
                    var cur = qid[i];
                    if (lastGroupId == defaultMax || lastGroupId != cur) info.GroupPtr.Add(groupSize);
                    lastGroupId = cur;
                    groupSize++;
                }
            }
        }
        if (hasLabels) info.Labels = new Tensor<float>([.. labels], [labels.Count, 1]);

        if (lastGroupId != defaultMax && groupSize > info.GroupPtr[^1]) info.GroupPtr.Add(groupSize);

        info.NumCol = adapter.NumColumns == AdapterConstants.UnknownSize ? inferredNumColumns : adapter.NumColumns;
        setCategories?.Invoke(m);
        info.SynchronizeNumberOfColumns();

        var offset = m._sparsePage.Offset;
        if (adapter.NumRows == AdapterConstants.UnknownSize)
        {
            if (adapter is FileAdapter)
            {
                info.NumRow = totalBatchSize;
                while (offset.Size - 1 < totalBatchSize) offset.Add(offset[offset.Size - 1]);
            }
            else
            {
                info.NumRow = offset.Size - 1;
            }
        }
        else
        {
            if (offset.Size == 0) offset.Add(0);
            while (offset.Size - 1 < adapter.NumRows) offset.Add(offset[offset.Size - 1]);
            info.NumRow = adapter.NumRows;
        }
        info.NumNonZero = m._sparsePage.Data.Size;

        if (!m._sparsePage.IsIndicesSorted(ctx.Threads())) m._sparsePage.SortIndices(ctx.Threads());
        m._fmatCtx = ctx;
        return m;
    }

    /// <summary>Columnar input: stores the categories (<c>std::is_same_v&lt;AdapterT, ColumnarAdapter&gt;</c> branch).</summary>
    public static SimpleDMatrix FromColumnar(ColumnarAdapter adapter, float missing, int nthread) =>
        FromAdapter(adapter, missing, nthread, m =>
        {
            if (adapter.HasCategorical) m._info.Cats(new CatContainer(adapter.Cats(), false));
        });

    /// <summary>Loads the binary format written by <see cref="SaveToLocalFile"/>, or returns null.</summary>
    public static SimpleDMatrix? TryLoadBinary(string fname, bool silent)
    {
        if (!File.Exists(fname)) return null;
        using var fs = File.OpenRead(fname);
        var head = new byte[4];
        if (fs.Read(head, 0, 4) != 4) return null;
        if (BitConverter.ToInt32(head) != Magic) return null;
        fs.Position = 0;
        using var reader = new DmlcReader(fs);
        var m = new SimpleDMatrix();
        var tmagic = reader.Read<int>();
        Check.Eq(tmagic, Magic, "invalid format, magic number mismatch");
        m._info.LoadBinary(reader);
        m._sparsePage.Offset = new HostDeviceVector<long>(Array.ConvertAll(reader.ReadVector<ulong>(), v => (long)v), true);
        m._sparsePage.Data = new HostDeviceVector<Entry>(reader.ReadVector<Entry>(), true);
        if (!silent)
            Log.Info($"{m._info.NumRow}x{m._info.NumCol} matrix with {m._info.NumNonZero} entries loaded from {fname}");
        return m;
    }

    public void SaveToLocalFile(string fname)
    {
        using var fs = File.Create(fname);
        using var fo = new DmlcWriter(fs);
        fo.Write(Magic);
        _info.SaveBinary(fo);
        fo.WriteVector<ulong>(Array.ConvertAll(_sparsePage.Offset.ToArray(), v => (ulong)v));
        fo.WriteVector<Entry>(_sparsePage.Data.ConstHostSpan);
    }
}
