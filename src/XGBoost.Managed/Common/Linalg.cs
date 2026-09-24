// Port of include/xgboost/linalg.h and src/common/linalg_op.h/.cc.
using System.Runtime.CompilerServices;

namespace XGBoost.Common;

public enum Order : byte
{
    C,
    F,
}

/// <summary>Slice selector: an index (drops the dimension), <see cref="All"/>, or a [begin, end) range.</summary>
public readonly record struct SliceArg(long Begin, long End, bool IsIndex, bool IsAll)
{
    public static readonly SliceArg All = new(0, 0, false, true);
    public static SliceArg Range(long begin, long end) => new(begin, end, false, false);
    public static implicit operator SliceArg(long i) => new(i, i + 1, true, false);
    public static implicit operator SliceArg(int i) => new(i, i + 1, true, false);
}

/// <summary>
/// Strided view over an array, <c>linalg::TensorView</c>. Shape and strides are in elements.
/// Unlike the C++ class the dimension is a runtime value.
/// </summary>
public readonly struct TensorView<T>
{
    private readonly T[] _data;
    private readonly int _offset;
    private readonly int _span; // number of elements reachable from _offset (size of C++ data_)
    private readonly long[] _shape;
    private readonly long[] _stride;

    public TensorView(T[] data, int offset, int span, long[] shape, long[] stride)
    {
        _data = data;
        _offset = offset;
        _span = span;
        _shape = shape;
        _stride = stride;
        Size = span == 0 ? 0 : Linalg.CalcSize(shape);
    }

    /// <summary>A view with shape <paramref name="shape"/> in the given order over <paramref name="data"/>.</summary>
    public TensorView(T[] data, int offset, int span, long[] shape, Order order = Order.C)
        : this(data, offset, span, shape, Linalg.CalcStride(shape, order)) { }

    public TensorView(T[] data, params long[] shape) : this(data, 0, data.Length, shape) { }

    public TensorView(ArraySegment<T> seg, params long[] shape) : this(seg.Array!, seg.Offset, seg.Count, shape) { }

    public long Size { get; }
    public bool Empty => Size == 0;
    public int Dim => _shape.Length;
    public ReadOnlySpan<long> Shape() => _shape;
    public long Shape(int i) => _shape[i];
    public ReadOnlySpan<long> Stride() => _stride;
    public long Stride(int i) => _stride[i];
    public T[] Array => _data;
    public int Offset => _offset;

    /// <summary>The underlying span, <c>Values()</c>. Its length can exceed <see cref="Size"/> for slices.</summary>
    public Span<T> Values => _data is null ? default : _data.AsSpan(_offset, _span);

    public ref T this[long i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _data[_offset + i * _stride[0]];
    }

    public ref T this[long i, long j]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _data[_offset + i * _stride[0] + j * _stride[1]];
    }

    public ref T this[long i, long j, long k]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _data[_offset + i * _stride[0] + j * _stride[1] + k * _stride[2]];
    }

    public ref T this[long i, long j, long k, long l]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _data[_offset + i * _stride[0] + j * _stride[1] + k * _stride[2] + l * _stride[3]];
    }

    /// <summary>Element at a multi-dimensional index (<c>std::apply(t, UnravelIndex(i, shape))</c>).</summary>
    public ref T At(ReadOnlySpan<long> index)
    {
        long off = _offset;
        for (var d = 0; d < index.Length; d++) off += index[d] * _stride[d];
        return ref _data[off];
    }

    /// <summary>Element at the flat, C-order position <paramref name="i"/> of the logical tensor.</summary>
    public ref T Flat(long i)
    {
        if (_shape.Length == 1) return ref this[i];
        long off = _offset;
        for (var d = _shape.Length - 1; d > 0; d--)
        {
            var s = _shape[d];
            var t = i / s;
            off += (i - t * s) * _stride[d];
            i = t;
        }
        off += i * _stride[0];
        return ref _data[off];
    }

    public TensorView<T> Slice(params SliceArg[] slices)
    {
        Check.Le(slices.Length, _shape.Length, "Invalid slice.");
        var newShape = new List<long>();
        var newStride = new List<long>();
        long offset = 0;
        for (var d = 0; d < slices.Length; d++)
        {
            var s = slices[d];
            if (s.IsIndex)
            {
                offset += _stride[d] * s.Begin;
            }
            else if (s.IsAll)
            {
                newShape.Add(_shape[d]);
                newStride.Add(_stride[d]);
            }
            else
            {
                newShape.Add(s.End - s.Begin);
                newStride.Add(_stride[d]);
                offset += _stride[d] * s.Begin;
            }
        }
        // Trailing dimensions not mentioned are dropped as in C++ (slice count must equal dim there).
        for (var d = slices.Length; d < _shape.Length; d++)
        {
            newShape.Add(_shape[d]);
            newStride.Add(_stride[d]);
        }
        var off = _span == 0 ? 0 : (int)offset;
        return new TensorView<T>(_data, _offset + off, _span - off, [.. newShape], [.. newStride]);
    }

    public bool CContiguous()
    {
        var s = Linalg.CalcStride(_shape, Order.C);
        return s.AsSpan().SequenceEqual(_stride);
    }

    public bool FContiguous()
    {
        var s = Linalg.CalcStride(_shape, Order.F);
        return s.AsSpan().SequenceEqual(_stride);
    }

    public bool Contiguous() => _span == Size || CContiguous() || FContiguous();

    /// <summary>Copies the logical elements in C order.</summary>
    public T[] ToArray()
    {
        var r = new T[Size];
        for (long i = 0; i < Size; i++) r[i] = Flat(i);
        return r;
    }
}

/// <summary>Owning tensor, <c>linalg::Tensor</c>.</summary>
public sealed class Tensor<T>
{
    private HostDeviceVector<T> _data = new();
    private long[] _shape;
    private readonly Order _order;

    public Tensor(int dim)
    {
        _shape = new long[dim];
        _order = Order.C;
    }

    public Tensor(long[] shape, Order order = Order.C)
    {
        _shape = (long[])shape.Clone();
        _order = order;
        _data.Resize(checked((int)Linalg.CalcSize(_shape)));
    }

    public Tensor(T[] values, long[] shape, Order order = Order.C)
    {
        _shape = (long[])shape.Clone();
        _order = order;
        _data = new HostDeviceVector<T>(values, takeOwnership: true);
        Check.Eq((long)_data.Size, Linalg.CalcSize(_shape));
    }

    public int Dim => _shape.Length;
    public int Size => _data.Size;
    public bool Empty => Size == 0;
    public ReadOnlySpan<long> Shape() => _shape;
    public long Shape(int i) => _shape[i];
    public HostDeviceVector<T> Data => _data;

    public TensorView<T> View() =>
        new(_data.RawArray, 0, _data.Size, (long[])_shape.Clone(), _order);

    public TensorView<T> HostView() => View();

    public ref T this[long i] => ref View()[i];
    public ref T this[long i, long j] => ref _data.RawArray[i * Linalg.CalcStride(_shape, _order)[0] + j * Linalg.CalcStride(_shape, _order)[1]];

    public void Reshape(params long[] shape)
    {
        Check.Le(shape.Length, _shape.Length, "Invalid shape.");
        for (var i = 0; i < _shape.Length; i++) _shape[i] = i < shape.Length ? shape[i] : 1;
        _data.Resize(checked((int)Linalg.CalcSize(_shape)));
    }

    public void Reshape(ReadOnlySpan<long> shape) => Reshape(shape.ToArray());

    /// <summary><c>ModifyInplace</c>: changes data and shape together.</summary>
    public void ModifyInplace(Action<HostDeviceVector<T>, long[]> fn)
    {
        fn(_data, _shape);
        Check.Eq((long)_data.Size, Linalg.CalcSize(_shape), "Inconsistent size after modification.");
    }

    public TensorView<T> Slice(params SliceArg[] s) => View().Slice(s);

    public Tensor<T> Clone()
    {
        var t = new Tensor<T>(_shape, _order);
        _data.ConstHostSpan.CopyTo(t._data.HostSpan);
        return t;
    }

    public void Assign(Tensor<T> other)
    {
        _shape = (long[])other._shape.Clone();
        _data = new HostDeviceVector<T>(other._data.ToArray(), true);
    }
}

public static class Linalg
{
    public static long CalcSize(ReadOnlySpan<long> shape)
    {
        long size = 1;
        foreach (var d in shape) size *= d;
        return size;
    }

    public static long[] CalcStride(ReadOnlySpan<long> shape, Order order)
    {
        var d = shape.Length;
        var stride = new long[d];
        if (d == 0) return stride;
        if (order == Order.F)
        {
            stride[0] = 1;
            for (var s = 1; s < d; s++) stride[s] = shape[s - 1] * stride[s - 1];
        }
        else
        {
            stride[d - 1] = 1;
            for (var s = d - 2; s >= 0; s--) stride[s] = shape[s + 1] * stride[s + 1];
        }
        return stride;
    }

    /// <summary>numpy-style unravel of a flat index.</summary>
    public static long[] UnravelIndex(long idx, ReadOnlySpan<long> shape)
    {
        var index = new long[shape.Length];
        for (var dim = shape.Length; --dim > 0;)
        {
            var s = shape[dim];
            var t = idx / s;
            index[dim] = idx - t * s;
            idx = t;
        }
        if (shape.Length > 0) index[0] = idx;
        return index;
    }

    public static (long, long) UnravelIndex2(long idx, long rows, long cols)
    {
        _ = rows;
        var t = idx / cols;
        return (t, idx - t * cols);
    }

    public static TensorView<T> MakeVec<T>(T[] data) => new(data, 0, data.Length, [data.Length]);

    public static TensorView<T> MakeVec<T>(T[] data, int offset, int length) => new(data, offset, length, [length]);

    public static TensorView<T> MakeVec<T>(HostDeviceVector<T> data) => new(data.RawArray, 0, data.Size, [data.Size]);

    public static TensorView<T> MakeTensorView<T>(T[] data, params long[] shape) => new(data, 0, data.Length, shape);

    public static TensorView<T> MakeTensorView<T>(HostDeviceVector<T> data, params long[] shape) =>
        new(data.RawArray, 0, data.Size, shape);

    public static TensorView<T> MakeTensorView<T>(Order order, T[] data, params long[] shape) =>
        new(data, 0, data.Length, shape, order);

    public static Tensor<T> Empty<T>(params long[] shape) => new(shape);

    public static Tensor<T> Constant<T>(T v, params long[] shape)
    {
        var t = new Tensor<T>(shape);
        t.Data.Fill(v);
        return t;
    }

    public static Tensor<T> Zeros<T>(params long[] shape) => new(shape);

    /// <summary>Stacks along the first axis.</summary>
    public static void Stack<T>(Tensor<T> l, Tensor<T> r)
    {
        l.ModifyInplace((data, shape) =>
        {
            for (var i = 1; i < shape.Length; i++)
            {
                if (shape[i] == 0) shape[i] = r.Shape(i);
                else Check.Eq(shape[i], r.Shape(i));
            }
            data.Extend(r.Data);
            shape[0] = l.Shape(0) + r.Shape(0);
        });
    }

    /// <summary>Pushes an extra dimension of size one to the end.</summary>
    public static TensorView<T> ExpandDim<T>(TensorView<T> x) =>
        new(x.Array, x.Offset, x.Values.Length, [x.Shape(0), 1], [x.Stride(0), 1]);

    /// <summary><c>cpu_impl::ElementWiseKernel</c>: calls <paramref name="fn"/> with a flat index.</summary>
    public static void ElementWiseKernel<T>(Context ctx, TensorView<T> t, Action<long> fn)
    {
        const long blockSize = 2048;
        Threading.ParallelFor1d(t.Size, ctx.Threads(), blockSize, block =>
        {
            for (var i = block.Begin; i < block.End; i++) fn(i);
        });
    }

    /// <summary>Element-wise kernel over a 2-D view, calls <paramref name="fn"/>(row, column).</summary>
    public static void ElementWiseKernel2<T>(Context ctx, TensorView<T> t, Action<long, long> fn)
    {
        const long blockSize = 2048;
        var nRows = t.Shape(0);
        var nCols = t.Shape(1);
        if (t.CContiguous() && nRows > nCols * 64)
        {
            Threading.ParallelFor1d(nRows, ctx.Threads(), blockSize, block =>
            {
                for (var i = block.Begin; i < block.End; i++)
                    for (long j = 0; j < nCols; j++) fn(i, j);
            });
            return;
        }
        Threading.ParallelFor1d(t.Size, ctx.Threads(), blockSize, block =>
        {
            for (var i = block.Begin; i < block.End; i++)
            {
                var r = i / nCols;
                fn(r, i - r * nCols);
            }
        });
    }

    public static void TransformIdxKernel<T>(Context ctx, TensorView<T> t, Func<long, T, T> fn)
    {
        if (t.Contiguous())
        {
            var arr = t.Array;
            var off = t.Offset;
            Threading.ParallelFor(t.Size, ctx.Threads(), i => arr[off + i] = fn(i, arr[off + i]));
        }
        else
        {
            Threading.ParallelFor(t.Size, ctx.Threads(), i =>
            {
                ref var v = ref t.Flat(i);
                v = fn(i, v);
            });
        }
    }

    public static void TransformKernel<T>(Context ctx, TensorView<T> t, Func<T, T> fn) =>
        TransformIdxKernel(ctx, t, (_, v) => fn(v));

    public static void VecScaMul(Context ctx, TensorView<float> x, double mul) =>
        TransformKernel(ctx, x, v => (float)(v * mul));

    public static void VecScaDiv(Context ctx, TensorView<float> x, double div) => VecScaMul(ctx, x, 1.0 / div);

    public static void LogE(Context ctx, TensorView<float> x, float rtEps = 0f) =>
        TransformKernel(ctx, x, v => MathF.Log(v + rtEps));

    public static void SmallHistogram(TensorView<float> indices, OptionalWeights weights, TensorView<float> bins)
    {
        var n = indices.Size;
        for (long i = 0; i < n; i++)
        {
            var y = indices[i];
            var w = weights[i];
            bins[(long)y] += w;
        }
    }

    public static void SaveVector(Tensor<float> t, JsonObject obj, string key) => obj[key] = new F32Array(t.Data.ToArray());
}
