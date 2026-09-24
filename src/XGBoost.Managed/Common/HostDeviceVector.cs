// Port of include/xgboost/host_device_vector.h and src/common/host_device_vector.cc (host only).
using System.Runtime.CompilerServices;

namespace XGBoost.Common;

/// <summary>
/// A growable array with <c>std::vector</c> semantics. In the C++ library it can also live on a GPU;
/// this port is CPU only, so it is a plain host buffer.
/// </summary>
public sealed class HostDeviceVector<T>
{
    private T[] _data;
    private int _size;

    public HostDeviceVector() => _data = [];

    public HostDeviceVector(int size, T init = default!)
    {
        _data = new T[size];
        if (!EqualityComparer<T>.Default.Equals(init, default!)) Array.Fill(_data, init);
        _size = size;
    }

    public HostDeviceVector(IEnumerable<T> init)
    {
        _data = init.ToArray();
        _size = _data.Length;
    }

    public HostDeviceVector(T[] data, bool takeOwnership)
    {
        _data = takeOwnership ? data : (T[])data.Clone();
        _size = data.Length;
    }

    public int Size => _size;
    public bool Empty => _size == 0;

    public Span<T> HostSpan => _data.AsSpan(0, _size);
    public ReadOnlySpan<T> ConstHostSpan => _data.AsSpan(0, _size);

    /// <summary>The backing array. It may be longer than <see cref="Size"/>.</summary>
    public T[] RawArray => _data;

    public ref T this[int i]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if ((uint)i >= (uint)_size) throw new IndexOutOfRangeException();
            return ref _data[i];
        }
    }

    /// <summary>Exact-size copy of the contents.</summary>
    public T[] ToArray() => _data.AsSpan(0, _size).ToArray();

    public List<T> ToList() => [.. _data.AsSpan(0, _size)];

    /// <summary>Returns the backing array trimmed to <see cref="Size"/> (reallocates only if needed).</summary>
    public T[] HostArray()
    {
        if (_data.Length != _size) Array.Resize(ref _data, _size);
        return _data;
    }

    public void Resize(int n, T init = default!)
    {
        if (n > _data.Length)
        {
            var cap = Math.Max(n, _data.Length * 2);
            if (_data.Length == 0) cap = n;
            Array.Resize(ref _data, cap);
        }
        if (n > _size) _data.AsSpan(_size, n - _size).Fill(init);
        else if (n < _size && RuntimeHelpers.IsReferenceOrContainsReferences<T>()) _data.AsSpan(n, _size - n).Clear();
        _size = n;
    }

    public void Fill(T v) => _data.AsSpan(0, _size).Fill(v);

    public void Clear() => Resize(0);

    public void Add(T v)
    {
        if (_size == _data.Length) Array.Resize(ref _data, Math.Max(4, _data.Length * 2));
        _data[_size++] = v;
    }

    public void Copy(HostDeviceVector<T> other)
    {
        Check.Eq(other.Size, Size, "Input size must be the same as the vector size.");
        other.ConstHostSpan.CopyTo(HostSpan);
    }

    public void Copy(ReadOnlySpan<T> other)
    {
        Check.Eq(other.Length, Size, "Input size must be the same as the vector size.");
        other.CopyTo(HostSpan);
    }

    public void Extend(HostDeviceVector<T> other) => Extend(other.ConstHostSpan);

    public void Extend(ReadOnlySpan<T> other)
    {
        var old = _size;
        Resize(old + other.Length);
        other.CopyTo(_data.AsSpan(old));
    }

    /// <summary>Replaces the content (<c>HostVector() = v</c>).</summary>
    public void Assign(ReadOnlySpan<T> values)
    {
        Resize(0);
        Extend(values);
    }

    public void Assign(T[] values, bool takeOwnership)
    {
        _data = takeOwnership ? values : (T[])values.Clone();
        _size = values.Length;
    }

    public void Swap(HostDeviceVector<T> other)
    {
        (_data, other._data) = (other._data, _data);
        (_size, other._size) = (other._size, _size);
    }
}
