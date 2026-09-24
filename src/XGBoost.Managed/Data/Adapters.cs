// Port of src/data/adapter.h/.cc. Adapters give uniform "line" access to external data so a single
// DMatrix constructor can consume dense, CSR, CSC, columnar and file input.
using System.Numerics;
using System.Runtime.CompilerServices;
using XGBoost.Common;

namespace XGBoost.Data;

/// <summary>A batch of lines (rows for row-major, columns for column-major input).</summary>
public interface IAdapterBatch
{
    long Size { get; }
    bool IsRowMajor { get; }
    int LineSize(long line);
    CooTuple GetElement(long line, int idx);

    float[]? Labels { get; }
    float[]? Weights { get; }
    ulong[]? Qid { get; }
    float[]? BaseMargin { get; }
}

/// <summary>An adapter: an iterator over batches plus the expected shape.</summary>
public interface IAdapter<TBatch> where TBatch : IAdapterBatch
{
    /// <summary><c>kAdapterUnknownSize</c>.</summary>
    const long UnknownSize = long.MaxValue;

    void BeforeFirst();
    bool Next();
    TBatch Value { get; }
    long NumRows { get; }
    long NumColumns { get; }
}

public static class AdapterConstants
{
    public const long UnknownSize = long.MaxValue;
}

/// <summary>Row-major dense float matrix, <c>DenseAdapterBatch</c>.</summary>
public readonly struct DenseAdapterBatch(float[] values, int offset, long numRows, long numFeatures) : IAdapterBatch
{
    public long Size => numRows;
    public bool IsRowMajor => true;
    public long NumRows => numRows;
    public long NumCols => numFeatures;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int LineSize(long line) => (int)numFeatures;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public CooTuple GetElement(long line, int idx) => new(line, idx, values[offset + line * numFeatures + idx]);

    public float[]? Labels => null;
    public float[]? Weights => null;
    public ulong[]? Qid => null;
    public float[]? BaseMargin => null;
}

/// <summary>Row-major dense array of any numeric type with strides, <c>ArrayAdapterBatch</c>.</summary>
public readonly struct ArrayAdapterBatch<T>(T[] values, long rows, long cols, long rowStride, long colStride) : IAdapterBatch
    where T : INumberBase<T>
{
    public long Size => rows;
    public bool IsRowMajor => true;
    public long NumRows => rows;
    public long NumCols => cols;

    public int LineSize(long line) => (int)cols;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public CooTuple GetElement(long line, int idx) =>
        new(line, idx, float.CreateTruncating(values[line * rowStride + idx * colStride]));

    public float[]? Labels => null;
    public float[]? Weights => null;
    public ulong[]? Qid => null;
    public float[]? BaseMargin => null;
}

/// <summary>CSR input, <c>CSRArrayAdapterBatch</c>. Index and value arrays can be any numeric type.</summary>
public readonly struct CsrAdapterBatch<TPtr, TIdx, TVal>(TPtr[] indptr, TIdx[] indices, TVal[] values, long nFeatures) : IAdapterBatch
    where TPtr : INumberBase<TPtr> where TIdx : INumberBase<TIdx> where TVal : INumberBase<TVal>
{
    public long Size => indptr.Length == 0 ? 0 : indptr.Length - 1;
    public bool IsRowMajor => true;
    public long NumRows => Size;
    public long NumCols => nFeatures;

    public int LineSize(long line) => (int)(long.CreateTruncating(indptr[line + 1]) - long.CreateTruncating(indptr[line]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public CooTuple GetElement(long line, int idx)
    {
        var k = long.CreateTruncating(indptr[line]) + idx;
        return new CooTuple(line, long.CreateTruncating(indices[k]), float.CreateTruncating(values[k]));
    }

    public float[]? Labels => null;
    public float[]? Weights => null;
    public ulong[]? Qid => null;
    public float[]? BaseMargin => null;
}

/// <summary>CSC input, <c>CSCArrayAdapterBatch</c> (column-major).</summary>
public readonly struct CscAdapterBatch<TPtr, TIdx, TVal>(TPtr[] indptr, TIdx[] indices, TVal[] values) : IAdapterBatch
    where TPtr : INumberBase<TPtr> where TIdx : INumberBase<TIdx> where TVal : INumberBase<TVal>
{
    public long Size => indptr.Length == 0 ? 0 : indptr.Length - 1;
    public bool IsRowMajor => false;

    public int LineSize(long line) => (int)(long.CreateTruncating(indptr[line + 1]) - long.CreateTruncating(indptr[line]));

    public CooTuple GetElement(long line, int idx)
    {
        var k = long.CreateTruncating(indptr[line]) + idx;
        return new CooTuple(long.CreateTruncating(indices[k]), line, float.CreateTruncating(values[k]));
    }

    public float[]? Labels => null;
    public float[]? Weights => null;
    public ulong[]? Qid => null;
    public float[]? BaseMargin => null;
}

/// <summary>One column of a columnar (arrow/pandas style) input.</summary>
public sealed class ColumnarColumn
{
    /// <summary>Numeric values or category codes (converted to float on access).</summary>
    public required Func<long, float> Get { get; init; }
    public required long Length { get; init; }
    /// <summary>Optional validity bitmap (arrow layout, LSB first).</summary>
    public byte[]? Valid { get; init; }
    /// <summary>Categories for a categorical column, null for numeric columns.</summary>
    public CatColumn? Categories { get; init; }

    public float Value(long row) => Valid is null || new RBitField8(Valid).Check(row) ? Get(row) : float.NaN;

    public static ColumnarColumn Numeric<T>(T[] values, byte[]? valid = null) where T : INumberBase<T> =>
        new() { Get = i => float.CreateTruncating(values[i]), Length = values.Length, Valid = valid };

    /// <summary>A categorical column given by integer codes into <paramref name="categories"/> (negative for missing).</summary>
    public static ColumnarColumn Categorical<T>(T[] codes, CatColumn categories) where T : IBinaryInteger<T> =>
        new()
        {
            Get = i =>
            {
                var c = long.CreateTruncating(codes[i]);
                return c < 0 ? float.NaN : c;
            },
            Length = codes.Length,
            Categories = categories,
        };
}

/// <summary>Columnar input, <c>EncColumnarAdapterBatchImpl</c> with optional re-coding.</summary>
public readonly struct ColumnarAdapterBatch(ColumnarColumn[] columns, CatAccessor? acc) : IAdapterBatch
{
    public long Size => columns.Length == 0 ? 0 : columns[0].Length;
    public bool IsRowMajor => true;
    public long NumCols => columns.Length;

    public int LineSize(long line) => columns.Length;

    public CooTuple GetElement(long line, int fidx)
    {
        var v = columns[fidx].Value(line);
        if (acc is { } a) v = a.Apply(v, fidx);
        return new CooTuple(line, fidx, v);
    }

    public float[]? Labels => null;
    public float[]? Weights => null;
    public ulong[]? Qid => null;
    public float[]? BaseMargin => null;
}

/// <summary>Rows from a text file, <c>FileAdapterBatch</c> (dmlc::RowBlock).</summary>
public readonly struct FileAdapterBatch(RowBlock block, long rowOffset) : IAdapterBatch
{
    public long Size => block.Size;
    public bool IsRowMajor => true;

    public int LineSize(long line) => (int)(block.Offset[line + 1] - block.Offset[line]);

    public CooTuple GetElement(long line, int idx)
    {
        var k = block.Offset[line] + idx;
        var v = block.Value is null ? 1.0f : block.Value[k];
        return new CooTuple(line + rowOffset, block.Index[k], v);
    }

    public float[]? Labels => block.Label;
    public float[]? Weights => block.Weight;
    public ulong[]? Qid => block.Qid;
    public float[]? BaseMargin => null;
}

/// <summary>Block of parsed text rows (<c>dmlc::RowBlock&lt;uint32_t&gt;</c>).</summary>
public sealed class RowBlock
{
    public long Size;
    public long[] Offset = [0];
    public float[]? Label;
    public float[]? Weight;
    public ulong[]? Qid;
    public uint[] Index = [];
    public float[]? Value;
}

/// <summary>Adapter over a single in-memory batch (<c>SingleBatchDataIter</c>).</summary>
public sealed class SingleBatchAdapter<TBatch>(TBatch batch, long numRows, long numColumns) : IAdapter<TBatch>
    where TBatch : IAdapterBatch
{
    private int _counter;

    public void BeforeFirst() => _counter = 0;

    public bool Next()
    {
        if (_counter == 0)
        {
            _counter++;
            return true;
        }
        return false;
    }

    public TBatch Value => batch;
    public long NumRows => numRows;
    public long NumColumns => numColumns;
}

/// <summary>Adapter over parsed file blocks (<c>FileAdapter</c>).</summary>
public sealed class FileAdapter(IReadOnlyList<RowBlock> blocks) : IAdapter<FileAdapterBatch>
{
    private int _idx = -1;
    private long _rowOffset;
    private FileAdapterBatch _batch;

    public void BeforeFirst()
    {
        _idx = -1;
        _rowOffset = 0;
    }

    public bool Next()
    {
        _idx++;
        if (_idx >= blocks.Count) return false;
        _batch = new FileAdapterBatch(blocks[_idx], _rowOffset);
        _rowOffset += blocks[_idx].Size;
        return true;
    }

    public FileAdapterBatch Value => _batch;
    public long NumRows => AdapterConstants.UnknownSize;
    public long NumColumns => AdapterConstants.UnknownSize;
}

/// <summary>A column-major adapter needs the row count; CSC input can give 0 (unknown).</summary>
public sealed class CscAdapter<TPtr, TIdx, TVal>(TPtr[] indptr, TIdx[] indices, TVal[] values, long numRows)
    : IAdapter<CscAdapterBatch<TPtr, TIdx, TVal>>
    where TPtr : INumberBase<TPtr> where TIdx : INumberBase<TIdx> where TVal : INumberBase<TVal>
{
    private readonly SingleBatchAdapter<CscAdapterBatch<TPtr, TIdx, TVal>> _inner =
        new(new CscAdapterBatch<TPtr, TIdx, TVal>(indptr, indices, values), 0, 0);

    public void BeforeFirst() => _inner.BeforeFirst();
    public bool Next() => _inner.Next();
    public CscAdapterBatch<TPtr, TIdx, TVal> Value => _inner.Value;
    public long NumRows => numRows == 0 ? AdapterConstants.UnknownSize : numRows;
    public long NumColumns => indptr.Length - 1;
}

/// <summary>Columnar adapter with categorical support, <c>ColumnarAdapter</c>.</summary>
public sealed class ColumnarAdapter : IAdapter<ColumnarAdapterBatch>
{
    private readonly ColumnarColumn[] _columns;
    private readonly SingleBatchAdapter<ColumnarAdapterBatch> _inner;

    public ColumnarAdapter(ColumnarColumn[] columns, CatAccessor? acc = null)
    {
        _columns = columns;
        for (var i = 1; i < columns.Length; i++) Check.Eq(columns[i].Length, columns[0].Length);
        _inner = new SingleBatchAdapter<ColumnarAdapterBatch>(new ColumnarAdapterBatch(columns, acc), 0, 0);
    }

    public void BeforeFirst() => _inner.BeforeFirst();
    public bool Next() => _inner.Next();
    public ColumnarAdapterBatch Value => _inner.Value;
    public long NumRows => _columns.Length == 0 ? 0 : _columns[0].Length;
    public long NumColumns => _columns.Length;

    public bool HasCategorical => _columns.Any(c => c.Categories is { IsEmpty: false });

    public IReadOnlyList<ColumnarColumn> Columns => _columns;

    /// <summary>The categories as an encoding scheme (<c>enc::HostColumnsView</c>).</summary>
    public ColumnsView Cats()
    {
        var cols = new List<CatColumn>();
        var segments = new int[_columns.Length + 1];
        for (var i = 0; i < _columns.Length; i++)
        {
            var c = _columns[i].Categories ?? new CatNumColumn(Array.Empty<int>());
            cols.Add(c);
            segments[i + 1] = segments[i] + c.Count;
        }
        return new ColumnsView(cols, segments, segments[^1]);
    }
}
