// Port of src/encoder/ordinal.h, src/encoder/types.h and src/data/cat_container.h/.cc.
using System.Text;
using XGBoost.Common;

namespace XGBoost.Data;

/// <summary>Categories of one feature: either strings (arrow layout) or a typed numeric array.</summary>
public abstract class CatColumn
{
    public abstract int Count { get; }
    public bool IsEmpty => Count == 0;
    public abstract CatColumn Clone();
}

/// <summary>String categories as arrow StringArray: <c>offsets</c> and signed bytes.</summary>
public sealed class CatStrColumn(int[] offsets, sbyte[] values) : CatColumn
{
    public int[] Offsets { get; } = offsets;
    public sbyte[] Values { get; } = values;

    public override int Count => Offsets.Length == 0 ? 0 : Offsets.Length - 1;

    public ReadOnlySpan<sbyte> Get(int i) => Values.AsSpan(Offsets[i], Offsets[i + 1] - Offsets[i]);

    public static CatStrColumn FromStrings(IReadOnlyList<string> names)
    {
        var offsets = new int[names.Count + 1];
        var bytes = new List<sbyte>();
        for (var i = 0; i < names.Count; i++)
        {
            foreach (var b in Encoding.UTF8.GetBytes(names[i])) bytes.Add(unchecked((sbyte)b));
            offsets[i + 1] = bytes.Count;
        }
        return new CatStrColumn(offsets, [.. bytes]);
    }

    public string GetString(int i) => Encoding.UTF8.GetString((byte[])(Array)Get(i).ToArray());

    public override CatColumn Clone() => new CatStrColumn((int[])Offsets.Clone(), (sbyte[])Values.Clone());
}

/// <summary>Numeric categories. <see cref="Values"/> is a typed CLR array (byte[], int[], ulong[], ...).</summary>
public sealed class CatNumColumn(Array values) : CatColumn
{
    public Array Values { get; } = values;
    public override int Count => Values.Length;
    public override CatColumn Clone() => new CatNumColumn((Array)Values.Clone());

    /// <summary>Serialisation type id, <c>CatIndexType</c>.</summary>
    public long TypeId => Values switch
    {
        sbyte[] => 9,
        byte[] => 10,
        short[] => 11,
        ushort[] => 12,
        int[] => 13,
        uint[] => 14,
        long[] => 15,
        ulong[] => 16,
        _ => throw new XGBoostException("Invalid type."),
    };

    public int CompareAt(int l, int r) => Values switch
    {
        sbyte[] a => a[l].CompareTo(a[r]),
        byte[] a => a[l].CompareTo(a[r]),
        short[] a => a[l].CompareTo(a[r]),
        ushort[] a => a[l].CompareTo(a[r]),
        int[] a => a[l].CompareTo(a[r]),
        uint[] a => a[l].CompareTo(a[r]),
        long[] a => a[l].CompareTo(a[r]),
        ulong[] a => a[l].CompareTo(a[r]),
        _ => throw new XGBoostException("Invalid type."),
    };

    public string Format(int i) => Convert.ToString(Values.GetValue(i), System.Globalization.CultureInfo.InvariantCulture) ?? "";
}

/// <summary>Encoding scheme of all columns (<c>enc::HostColumnsView</c>).</summary>
public sealed class ColumnsView(IReadOnlyList<CatColumn> columns, int[] featureSegments, int nTotalCats)
{
    public IReadOnlyList<CatColumn> Columns { get; } = columns;
    public int[] FeatureSegments { get; } = featureSegments;
    public int NTotalCats { get; } = nTotalCats;
    public int Size => Columns.Count;
    public bool Empty => Size == 0;
    public bool HasCategorical => NTotalCats != 0;
}

/// <summary>Mapping from new category codes to training codes (<c>enc::MappingView</c>).</summary>
public readonly struct MappingView(int[] offsets, int[] mapping)
{
    public int[] OffsetsArr { get; } = offsets;
    public int[] Mapping { get; } = mapping;
    public bool Empty => OffsetsArr is null || OffsetsArr.Length == 0;

    public ReadOnlySpan<int> this[int fIdx] => Mapping.AsSpan(OffsetsArr[fIdx], OffsetsArr[fIdx + 1] - OffsetsArr[fIdx]);
}

/// <summary>Recodes categorical values, <c>CatAccessor</c>.</summary>
public readonly struct CatAccessor(MappingView enc)
{
    public MappingView Enc { get; } = enc;

    public float Apply(float fvalue, long fIdx)
    {
        if (!Enc.Empty && !Enc[(int)fIdx].IsEmpty)
        {
            var fMapping = Enc[(int)fIdx];
            var catIdx = Categorical.AsCat(fvalue);
            if (catIdx >= 0 && catIdx < fMapping.Length) fvalue = fMapping[catIdx];
        }
        return fvalue;
    }
}

/// <summary>Port of <c>enc::</c> (ordinal re-coder).</summary>
public static class Encoder
{
    private static int CompareBytes(ReadOnlySpan<sbyte> l, ReadOnlySpan<sbyte> r) => l.SequenceCompareTo(r);

    /// <summary><c>SortNames</c> for all features: argsort of the categories inside each segment.</summary>
    public static void SortNames(ColumnsView origEnc, Span<int> sortedIdx)
    {
        if (sortedIdx.Length != origEnc.NTotalCats) Check.Fail("`sorted_idx` should have the same size as `n_total_cats`.");
        for (var f = 0; f < origEnc.Size; f++)
        {
            var beg = origEnc.FeatureSegments[f];
            var fSorted = sortedIdx.Slice(beg, origEnc.FeatureSegments[f + 1] - beg);
            SortNames(origEnc.Columns[f], fSorted);
        }
    }

    public static void SortNames(CatColumn cats, Span<int> sortedIdx)
    {
        if (sortedIdx.Length != cats.Count) Check.Fail("Invalid size of sorted index.");
        for (var i = 0; i < sortedIdx.Length; i++) sortedIdx[i] = i;
        switch (cats)
        {
            case CatStrColumn str:
                StdAlgo.StableSort(sortedIdx, (l, r) => CompareBytes(str.Get(l), str.Get(r)) < 0);
                break;
            case CatNumColumn num:
                StdAlgo.StableSort(sortedIdx, (l, r) => num.CompareAt(l, r) < 0);
                break;
        }
    }

    private static int SearchSorted(CatStrColumn haystack, ReadOnlySpan<int> refSortedIdx, ReadOnlySpan<sbyte> needle)
    {
        int lo = 0, count = haystack.Count;
        while (count > 0)
        {
            var step = count >> 1;
            var mid = lo + step;
            if (CompareBytes(haystack.Get(refSortedIdx[mid]), needle) < 0)
            {
                lo = mid + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        if (lo == haystack.Count) return -1;
        return haystack.Get(refSortedIdx[lo]).SequenceEqual(needle) ? lo : -1;
    }

    private static int SearchSorted(CatNumColumn haystack, ReadOnlySpan<int> refSortedIdx, CatNumColumn needleCol, int needleIdx)
    {
        int Cmp(int hIdx) => CompareAcross(haystack.Values, hIdx, needleCol.Values, needleIdx);
        int lo = 0, count = haystack.Count;
        while (count > 0)
        {
            var step = count >> 1;
            var mid = lo + step;
            if (Cmp(refSortedIdx[mid]) < 0)
            {
                lo = mid + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        if (lo == haystack.Count) return -1;
        return Cmp(refSortedIdx[lo]) == 0 ? lo : -1;
    }

    private static int CompareAcross(Array a, int i, Array b, int j) => (a, b) switch
    {
        (sbyte[] x, sbyte[] y) => x[i].CompareTo(y[j]),
        (byte[] x, byte[] y) => x[i].CompareTo(y[j]),
        (short[] x, short[] y) => x[i].CompareTo(y[j]),
        (ushort[] x, ushort[] y) => x[i].CompareTo(y[j]),
        (int[] x, int[] y) => x[i].CompareTo(y[j]),
        (uint[] x, uint[] y) => x[i].CompareTo(y[j]),
        (long[] x, long[] y) => x[i].CompareTo(y[j]),
        (ulong[] x, ulong[] y) => x[i].CompareTo(y[j]),
        _ => throw new XGBoostException("Mismatched category index types."),
    };

    /// <summary><c>enc::Recode</c>: mapping from the new encoding to the training encoding.</summary>
    public static void Recode(ColumnsView origEnc, ReadOnlySpan<int> sortedIdx, ColumnsView newEnc, Span<int> mapping)
    {
        if (origEnc.Size != newEnc.Size) Check.Fail("New and old encoding should have the same number of columns.");
        if (mapping.Length != newEnc.NTotalCats) Check.Fail("`mapping` should have the same size as `new_enc.n_total_cats`.");
        if (sortedIdx.Length != origEnc.NTotalCats) Check.Fail("`sorted_idx` should have the same size as `orig_enc.n_total_cats`.");
        if (origEnc.FeatureSegments.Length != origEnc.Columns.Count + 1) Check.Fail("Invalid original encoding.");
        if (newEnc.FeatureSegments.Length != newEnc.Columns.Count + 1) Check.Fail("Invalid new encoding.");

        var outIdx = 0;
        for (var f = 0; f < origEnc.Size; f++)
        {
            var l = origEnc.Columns[f];
            var r = newEnc.Columns[f];
            void Report() => Check.Fail($"Invalid new DataFrame input for the: {f}th feature (0-based). "
                + "The data type doesn't match the one used in the training dataset. Both should be either numeric or "
                + "categorical. For a categorical feature, the index type must match between the training and test set.");
            if (l.GetType() != r.GetType() || (l is CatNumColumn ln && r is CatNumColumn rn && ln.Values.GetType() != rn.Values.GetType()))
                Report();
            if (l.IsEmpty != r.IsEmpty) Report();
            if (l.IsEmpty) continue;

            var fBeg = origEnc.FeatureSegments[f];
            var refSorted = sortedIdx.Slice(fBeg, origEnc.FeatureSegments[f + 1] - fBeg);
            var n = r.Count;
            var searched = new int[n];
            for (var j = 0; j < n; j++)
            {
                if (r is CatStrColumn rs)
                {
                    var needle = rs.Get(j);
                    searched[j] = SearchSorted((CatStrColumn)l, refSorted, needle);
                    if (searched[j] == -1) ReportMissing(rs.GetString(j), f);
                }
                else
                {
                    var rnum = (CatNumColumn)r;
                    searched[j] = SearchSorted((CatNumColumn)l, refSorted, rnum, j);
                    if (searched[j] == -1) ReportMissing(rnum.Format(j), f);
                }
            }
            foreach (var i in searched) mapping[outIdx++] = refSorted[i];
        }
    }

    private static void ReportMissing(string name, int fIdx) =>
        Check.Fail($"Found a category not in the training set for the {fIdx}th (0-based) column: `{name}`");
}

/// <summary>Container for user-provided categories (usually from a DataFrame), <c>CatContainer</c>.</summary>
public sealed class CatContainer
{
    private readonly List<CatColumn> _columns = [];
    private int[] _featureSegments = [];
    private int _nTotalCats;
    private int[] _sortedIdx = [];
    private bool _isRef;
    private readonly Lock _mu = new();

    public CatContainer() { }

    public CatContainer(ColumnsView df, bool isRef)
    {
        _isRef = isRef;
        _nTotalCats = df.NTotalCats;
        if (_nTotalCats == 0) return;
        _featureSegments = (int[])df.FeatureSegments.Clone();
        foreach (var col in df.Columns) _columns.Add(col.Clone());
        _sortedIdx = [];
        Check.Eq(_nTotalCats, df.FeatureSegments[^1]);
        Check.Ge(_nTotalCats, 0, "Too many categories.");
    }

    public void Copy(CatContainer that)
    {
        lock (_mu)
        {
            _sortedIdx = (int[])that._sortedIdx.Clone();
            _featureSegments = (int[])that._featureSegments.Clone();
            _nTotalCats = that._nTotalCats;
            _isRef = that._isRef;
            _columns.Clear();
            foreach (var c in that._columns) _columns.Add(c.Clone());
        }
    }

    public void Push(CatColumn column) => _columns.Add(column);

    public bool Empty => _columns.Count == 0;
    public bool NeedRecode => HasCategorical && !_isRef;
    public int NumFeatures => _columns.Count;
    public int NumCatsTotal => _nTotalCats;
    public bool HasCategorical => _nTotalCats != 0;

    public ColumnsView HostView()
    {
        if (_nTotalCats != 0) Check.That(_columns.Count != 0);
        return new ColumnsView(_columns, _featureSegments, _nTotalCats);
    }

    public void Sort()
    {
        var view = HostView();
        _sortedIdx = new int[view.NTotalCats];
        Encoder.SortNames(view, _sortedIdx);
    }

    public ReadOnlySpan<int> RefSortedIndex()
    {
        lock (_mu) return _sortedIdx;
    }

    public void Save(JsonObject output)
    {
        var arr = new JsonArray();
        foreach (var col in _columns)
        {
            var fOut = new JsonObject();
            switch (col)
            {
                case CatStrColumn str:
                    fOut["offsets"] = new I32Array((int[])str.Offsets.Clone());
                    fOut["values"] = new I8Array((sbyte[])str.Values.Clone());
                    break;
                case CatNumColumn num:
                    if (num.Values is ulong[] u && u.Any(v => v > long.MaxValue))
                        Check.Fail("Category index values must not exceed the signed 64-bit range.");
                    fOut["type"] = new JsonInteger(num.TypeId);
                    fOut["values"] = num.Values switch
                    {
                        sbyte[] a => new I8Array((sbyte[])a.Clone()),
                        byte[] a => new U8Array((byte[])a.Clone()),
                        short[] a => new I16Array((short[])a.Clone()),
                        ushort[] a => new U16Array((ushort[])a.Clone()),
                        int[] a => new I32Array((int[])a.Clone()),
                        uint[] a => new U32Array((uint[])a.Clone()),
                        long[] a => new I64Array((long[])a.Clone()),
                        ulong[] a => new U64Array((ulong[])a.Clone()),
                        _ => throw new XGBoostException("Invalid type."),
                    };
                    break;
            }
            arr.Add(fOut);
        }
        output.Map.Clear();
        output["sorted_idx"] = new I32Array((int[])_sortedIdx.Clone());
        output["feature_segments"] = new I32Array((int[])_featureSegments.Clone());
        output["enc"] = arr;
    }

    public void Load(Json input)
    {
        var obj = input.AsObject;
        foreach (var jcol in obj["enc"].AsArray)
        {
            var column = jcol.AsObject;
            if (column.ContainsKey("offsets"))
            {
                var offsets = LoadInts<int>(column["offsets"]);
                var values = LoadInts<sbyte>(column["values"]);
                _columns.Add(new CatStrColumn(offsets, values));
            }
            else
            {
                var type = column["type"].AsInteger;
                var jv = column["values"];
                Array values = type switch
                {
                    9 => LoadInts<sbyte>(jv),
                    10 => LoadInts<byte>(jv),
                    11 => LoadInts<short>(jv),
                    12 => LoadInts<ushort>(jv),
                    13 => LoadInts<int>(jv),
                    14 => LoadInts<uint>(jv),
                    15 => LoadInts<long>(jv),
                    16 => LoadInts<ulong>(jv),
                    7 or 8 => throw new XGBoostException(ErrorMsg.NoFloatCat),
                    _ => throw new XGBoostException("Invalid type."),
                };
                _columns.Add(new CatNumColumn(values));
            }
        }
        _featureSegments = LoadInts<int>(obj["feature_segments"]);
        _nTotalCats = _featureSegments.Length == 0 ? 0 : _featureSegments[^1];
        _sortedIdx = LoadInts<int>(obj["sorted_idx"]);
    }

    // JSON arrays of integers, or UBJSON typed arrays (unsigned stored as same-width signed).
    private static T[] LoadInts<T>(Json j) where T : unmanaged, System.Numerics.IBinaryInteger<T>
    {
        switch (j)
        {
            case JsonArray a:
            {
                var r = new T[a.Count];
                for (var i = 0; i < a.Count; i++) r[i] = T.CreateTruncating(a[i].AsInteger);
                return r;
            }
            case JsonTypedArray<T> t:
                return (T[])t.Values.Clone();
            case I16Array s when typeof(T) == typeof(ushort):
                return (T[])(Array)s.Values.Select(v => unchecked((ushort)v)).ToArray();
            case I32Array s when typeof(T) == typeof(uint):
                return (T[])(Array)s.Values.Select(v => unchecked((uint)v)).ToArray();
            case I64Array s when typeof(T) == typeof(ulong):
                return (T[])(Array)s.Values.Select(v => unchecked((ulong)v)).ToArray();
            case I32Array s:
                return s.Values.Select(v => T.CreateTruncating(v)).ToArray();
            case I8Array s:
                return s.Values.Select(v => T.CreateTruncating(v)).ToArray();
            case U8Array s:
                return s.Values.Select(v => T.CreateTruncating(v)).ToArray();
            case I16Array s:
                return s.Values.Select(v => T.CreateTruncating(v)).ToArray();
            case I64Array s:
                return s.Values.Select(v => T.CreateTruncating(v)).ToArray();
            default:
                throw new XGBoostException($"Invalid category array of type {j.TypeStr}.");
        }
    }

    /// <summary><c>cpu_impl::MakeCatAccessor</c>: accessor recoding <paramref name="newEnc"/> to this encoding.</summary>
    public (CatAccessor Accessor, int[] Mapping) MakeAccessor(ColumnsView newEnc)
    {
        var mapping = new int[newEnc.NTotalCats];
        var origEnc = HostView();
        Encoder.Recode(origEnc, RefSortedIndex(), newEnc, mapping);
        Check.Eq(newEnc.FeatureSegments.Length, origEnc.FeatureSegments.Length);
        return (new CatAccessor(new MappingView(newEnc.FeatureSegments, mapping)), mapping);
    }
}
