// Port of MetaInfo in include/xgboost/data.h, src/data/data.cc and src/data/metainfo.h/.cc.
using System.Numerics;
using XGBoost.Common;

namespace XGBoost.Data;

public enum MetaField : sbyte
{
    Label = 0,
    Weight = 1,
    BaseMargin = 2,
    LabelLowerBound = 3,
    LabelUpperBound = 4,
    FeatureWeights = 5,
    GroupPtr = 6,
    Qid = 7,
}

/// <summary>Meta information about a dataset: labels, weights, groups, feature info, <c>MetaInfo</c>.</summary>
public sealed class MetaInfo
{
    public const ulong NumField = 13;

    public long NumRow;
    public long NumCol;
    public long NumNonZero;

    /// <summary>Labels, shape (n_samples, n_targets).</summary>
    public Tensor<float> Labels = new(2);

    /// <summary>Index of the begin and end of each query group, used for ranking.</summary>
    public List<uint> GroupPtr = [];

    public HostDeviceVector<float> Weights = new();

    /// <summary>Initial margin, shape (n_samples, n_groups).</summary>
    public Tensor<float> BaseMargin = new(2);

    public HostDeviceVector<float> LabelsLowerBound = new();
    public HostDeviceVector<float> LabelsUpperBound = new();

    public List<string> FeatureTypeNames = [];
    public List<string> FeatureNames = [];
    public HostDeviceVector<FeatureType> FeatureTypes = new();
    public HostDeviceVector<float> FeatureWeights = new();

    private long[] _labelOrderCache = [];
    private bool _hasCategorical;
    private CatContainer _cats = new();

    public bool IsDense => NumCol * NumRow == NumNonZero;

    public float GetWeight(long i) => Weights.Size != 0 ? Weights[(int)i] : 1.0f;

    public bool IsRanking => GroupPtr.Count != 0;

    public bool HasCategorical => _hasCategorical;

    public CatContainer Cats() => _cats;

    public void Cats(CatContainer cats)
    {
        _cats = cats;
        Check.Lt(_cats.NumCatsTotal, int.MaxValue);
    }

    public void Clear()
    {
        NumRow = NumCol = NumNonZero = 0;
        Labels = new Tensor<float>(2);
        GroupPtr.Clear();
        Weights.Resize(0);
        BaseMargin = new Tensor<float>(2);
    }

    /// <summary>Argsort of labels by absolute value, used by the Cox loss.</summary>
    public long[] LabelAbsSort()
    {
        if (_labelOrderCache.Length == Labels.Size) return _labelOrderCache;
        var l = Labels.Data.RawArray;
        var order = new long[Labels.Size];
        StdAlgo.Iota(order);
        StdAlgo.StableSort(order.AsSpan(), (i1, i2) => Math.Abs(l[i1]) < Math.Abs(l[i2]));
        _labelOrderCache = order;
        return order;
    }

    // ---- binary IO --------------------------------------------------------------------------

    private static void SaveScalar<T>(DmlcWriter fo, string name, DataType type, T value) where T : unmanaged
    {
        fo.WriteString(name);
        fo.Write((byte)type);
        fo.Write((byte)1);
        fo.Write(value);
    }

    private static void SaveVector<T>(DmlcWriter fo, string name, DataType type, ulong rows, ulong cols, ReadOnlySpan<T> v)
        where T : unmanaged
    {
        fo.WriteString(name);
        fo.Write((byte)type);
        fo.Write((byte)0);
        fo.Write(rows);
        fo.Write(cols);
        fo.WriteVector(v);
    }

    private static void SaveStrings(DmlcWriter fo, string name, List<string> v)
    {
        fo.WriteString(name);
        fo.Write((byte)DataType.Str);
        fo.Write((byte)0);
        fo.Write((ulong)v.Count);
        fo.Write(1ul);
        fo.WriteStrings(v);
    }

    private static void SaveTensor(DmlcWriter fo, string name, Tensor<float> t)
    {
        fo.WriteString(name);
        fo.Write((byte)DataType.Float32);
        fo.Write((byte)0);
        for (var i = 0; i < t.Dim; i++) fo.Write((ulong)t.Shape(i));
        fo.WriteVector(t.Data.ConstHostSpan);
    }

    public void SaveBinary(DmlcWriter fo)
    {
        VersionInfo.Save(fo);
        fo.Write(NumField);
        SaveScalar(fo, "num_row", DataType.UInt64, (ulong)NumRow);
        SaveScalar(fo, "num_col", DataType.UInt64, (ulong)NumCol);
        SaveScalar(fo, "num_nonzero", DataType.UInt64, (ulong)NumNonZero);
        SaveTensor(fo, "labels", Labels);
        SaveVector(fo, "group_ptr", DataType.UInt32, (ulong)GroupPtr.Count, 1, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(GroupPtr));
        SaveVector(fo, "weights", DataType.Float32, (ulong)Weights.Size, 1, Weights.ConstHostSpan);
        SaveTensor(fo, "base_margin", BaseMargin);
        SaveVector(fo, "labels_lower_bound", DataType.Float32, (ulong)LabelsLowerBound.Size, 1, LabelsLowerBound.ConstHostSpan);
        SaveVector(fo, "labels_upper_bound", DataType.Float32, (ulong)LabelsUpperBound.Size, 1, LabelsUpperBound.ConstHostSpan);
        SaveStrings(fo, "feature_names", FeatureNames);
        SaveStrings(fo, "feature_types", FeatureTypeNames);
        SaveVector(fo, "feature_weights", DataType.Float32, (ulong)FeatureWeights.Size, 1, FeatureWeights.ConstHostSpan);
        var jcats = new JsonObject();
        _cats.Save(jcats);
        var values = Json.DumpBytes(jcats, binary: true);
        SaveVector(fo, "cats", DataType.Str, (ulong)values.Length, 1, values.AsSpan());
    }

    private static void ReadHeader(DmlcReader fi, string expected, DataType expectedType, bool expectScalar)
    {
        var invalid = $"MetaInfo: Invalid format for {expected}";
        var name = fi.ReadString();
        Check.That(name == expected, $"{invalid} Expected field: {expected}, got: {name}");
        var type = (DataType)fi.Read<byte>();
        Check.That(type == expectedType, $"{invalid}Expected field of type: {(int)expectedType}, got field type: {(int)type}");
        var isScalar = fi.Read<byte>() != 0;
        if (expectScalar) Check.That(isScalar, $"{invalid}Expected field {expected} to be a scalar; got a vector");
        else Check.That(!isScalar, $"{invalid}Expected field {expected} to be a vector; got a scalar");
    }

    private static T LoadScalar<T>(DmlcReader fi, string name, DataType type) where T : unmanaged
    {
        ReadHeader(fi, name, type, true);
        return fi.Read<T>();
    }

    private static T[] LoadVector<T>(DmlcReader fi, string name, DataType type) where T : unmanaged
    {
        ReadHeader(fi, name, type, false);
        fi.Read<ulong>();
        var cols = fi.Read<ulong>();
        Check.Eq(cols, 1ul, $"MetaInfo: Invalid format for {name}Number of columns is expected to be 1.");
        return fi.ReadVector<T>();
    }

    private static List<string> LoadStrings(DmlcReader fi, string name)
    {
        ReadHeader(fi, name, DataType.Str, false);
        fi.Read<ulong>();
        var cols = fi.Read<ulong>();
        Check.Eq(cols, 1ul);
        return fi.ReadStrings();
    }

    private static Tensor<float> LoadTensor(DmlcReader fi, string name)
    {
        ReadHeader(fi, name, DataType.Float32, false);
        var shape = new long[2];
        for (var i = 0; i < 2; i++) shape[i] = (long)fi.Read<ulong>();
        var data = fi.ReadVector<float>();
        return new Tensor<float>(data, shape);
    }

    public void LoadBinary(DmlcReader fi)
    {
        var version = VersionInfo.Load(fi);
        var msg = $"Binary DMatrix generated by XGBoost: {VersionInfo.String(version)} is no longer supported. "
            + $"Please process and save your data in current version: {VersionInfo.String(VersionInfo.Self())} again.";
        Check.Ge(version.Major, 3, msg);
        Check.Ge(version.Minor, 1, msg);
        var numField = fi.Read<ulong>();
        Check.Ge(numField, NumField, $"MetaInfo: insufficient number of fields (expected at least {NumField} fields, but the binary file only contains {numField}fields.)");
        if (numField > NumField) Log.Warning("MetaInfo: the given binary file contains extra fields which will be ignored.");

        NumRow = (long)LoadScalar<ulong>(fi, "num_row", DataType.UInt64);
        NumCol = (long)LoadScalar<ulong>(fi, "num_col", DataType.UInt64);
        NumNonZero = (long)LoadScalar<ulong>(fi, "num_nonzero", DataType.UInt64);
        Labels = LoadTensor(fi, "labels");
        GroupPtr = [.. LoadVector<uint>(fi, "group_ptr", DataType.UInt32)];
        Weights = new HostDeviceVector<float>(LoadVector<float>(fi, "weights", DataType.Float32), true);
        BaseMargin = LoadTensor(fi, "base_margin");
        LabelsLowerBound = new HostDeviceVector<float>(LoadVector<float>(fi, "labels_lower_bound", DataType.Float32), true);
        LabelsUpperBound = new HostDeviceVector<float>(LoadVector<float>(fi, "labels_upper_bound", DataType.Float32), true);
        FeatureNames = LoadStrings(fi, "feature_names");
        FeatureTypeNames = LoadStrings(fi, "feature_types");
        FeatureWeights = new HostDeviceVector<float>(LoadVector<float>(fi, "feature_weights", DataType.Float32), true);
        _hasCategorical = LoadFeatureType(FeatureTypeNames, FeatureTypes);
        var values = LoadVector<byte>(fi, "cats", DataType.Str);
        var jcats = Json.Load(values, binary: true);
        _cats = new CatContainer();
        _cats.Load(jcats);
    }

    public static bool LoadFeatureType(List<string> typeNames, HostDeviceVector<FeatureType> types)
    {
        types.Resize(0);
        var hasCat = false;
        foreach (var elem in typeNames)
        {
            switch (elem)
            {
                case "int":
                case "float":
                case "i":
                case "q":
                    types.Add(FeatureType.Numerical);
                    break;
                case "c":
                    types.Add(FeatureType.Categorical);
                    hasCat = true;
                    break;
                default:
                    Check.Fail("All feature_types must be one of {int, float, i, q, c}.");
                    break;
            }
        }
        return hasCat;
    }

    // ---- slicing and copying ----------------------------------------------------------------

    private static T[] Gather<T>(ReadOnlySpan<T> input, ReadOnlySpan<long> ridxs, long stride = 1)
    {
        if (input.IsEmpty) return [];
        var size = ridxs.Length;
        var output = new T[size * stride];
        for (var i = 0; i < size; i++)
        {
            var ridx = ridxs[i];
            for (long j = 0; j < stride; j++) output[i * stride + j] = input[(int)(ridx * stride + j)];
        }
        return output;
    }

    public MetaInfo Slice(ReadOnlySpan<long> ridxs, long nnz)
    {
        var output = new MetaInfo
        {
            NumRow = ridxs.Length,
            NumCol = NumCol,
            NumNonZero = nnz,
            FeatureWeights = new HostDeviceVector<float>(FeatureWeights.ToArray(), true),
            FeatureNames = [.. FeatureNames],
            FeatureTypes = new HostDeviceVector<FeatureType>(FeatureTypes.ToArray(), true),
            FeatureTypeNames = [.. FeatureTypeNames],
        };

        if (Labels.Size != NumRow)
        {
            var stride = Labels.View().Stride(0);
            output.Labels = new Tensor<float>(Gather<float>(Labels.Data.ConstHostSpan, ridxs, stride), [ridxs.Length, Labels.Shape(1)]);
        }
        else
        {
            var d = Gather<float>(Labels.Data.ConstHostSpan, ridxs);
            output.Labels = new Tensor<float>(d, [d.Length, 1]);
        }

        output.LabelsUpperBound = new HostDeviceVector<float>(Gather<float>(LabelsUpperBound.ConstHostSpan, ridxs), true);
        output.LabelsLowerBound = new HostDeviceVector<float>(Gather<float>(LabelsLowerBound.ConstHostSpan, ridxs), true);
        if (Weights.Size + 1 == GroupPtr.Count)
        {
            // Assuming all groups are available. (The C++ code assigns the empty output weights to itself.)
            output.Weights = new HostDeviceVector<float>();
        }
        else
        {
            output.Weights = new HostDeviceVector<float>(Gather<float>(Weights.ConstHostSpan, ridxs), true);
        }

        if (BaseMargin.Size != NumRow)
        {
            Check.Eq(BaseMargin.Size % NumRow, 0L, "Incorrect size of base margin vector.");
            var stride = BaseMargin.View().Stride(0);
            output.BaseMargin = new Tensor<float>(Gather<float>(BaseMargin.Data.ConstHostSpan, ridxs, stride),
                [ridxs.Length, BaseMargin.Shape(1)]);
        }
        else
        {
            var d = Gather<float>(BaseMargin.Data.ConstHostSpan, ridxs);
            output.BaseMargin = new Tensor<float>(d, [d.Length, 1]);
        }
        output._hasCategorical = _hasCategorical;
        return output;
    }

    public MetaInfo Copy()
    {
        var output = new MetaInfo();
        output.Extend(this, accumulateRows: true, checkColumn: false);
        return output;
    }

    public void Extend(MetaInfo that, bool accumulateRows, bool checkColumn)
    {
        if (accumulateRows) NumRow += that.NumRow;
        if (NumCol != 0)
        {
            if (checkColumn) Check.Eq(NumCol, that.NumCol, "Number of columns must be consistent across batches.");
            else NumCol = Math.Max(NumCol, that.NumCol);
        }
        NumCol = that.NumCol;

        Linalg.Stack(Labels, that.Labels);
        Weights.Extend(that.Weights);
        LabelsLowerBound.Extend(that.LabelsLowerBound);
        LabelsUpperBound.Extend(that.LabelsUpperBound);
        Linalg.Stack(BaseMargin, that.BaseMargin);

        if (GroupPtr.Count == 0)
        {
            GroupPtr = [.. that.GroupPtr];
        }
        else
        {
            Check.Ne(that.GroupPtr.Count, 0);
            var back = GroupPtr[^1];
            for (var i = 1; i < that.GroupPtr.Count; i++) GroupPtr.Add(that.GroupPtr[i] + back);
        }

        if (that.FeatureNames.Count != 0) FeatureNames = [.. that.FeatureNames];
        if (!FeatureTypes.Empty) CheckFeatureTypes(FeatureTypes, that.FeatureTypes);

        if (that.FeatureTypeNames.Count != 0)
        {
            FeatureTypeNames = [.. that.FeatureTypeNames];
            _hasCategorical = LoadFeatureType(FeatureTypeNames, FeatureTypes);
        }
        else if (!that.FeatureTypes.Empty)
        {
            FeatureTypes = new HostDeviceVector<FeatureType>(that.FeatureTypes.ToArray(), true);
            _hasCategorical = FeatureTypes.ToArray().Any(Categorical.IsCatOp);
        }

        if (!that.FeatureWeights.Empty) FeatureWeights = new HostDeviceVector<float>(that.FeatureWeights.ToArray(), true);
    }

    public static void CheckFeatureTypes(HostDeviceVector<FeatureType> lhs, HostDeviceVector<FeatureType> rhs)
    {
        Check.Eq(lhs.Size, rhs.Size, ErrorMsg.InconsistentFeatureTypes);
        Check.That(lhs.ConstHostSpan.SequenceEqual(rhs.ConstHostSpan), ErrorMsg.InconsistentFeatureTypes);
    }

    /// <summary>Single process: nothing to synchronise.</summary>
    public void SynchronizeNumberOfColumns() => Collective.Communicator.AllreduceMax(ref NumCol);

    public void Validate()
    {
        if (GroupPtr.Count != 0 && Weights.Size != 0)
        {
            Check.Eq(GroupPtr.Count, Weights.Size + 1, ErrorMsg.GroupWeight);
            return;
        }
        if (GroupPtr.Count != 0) Check.Eq((long)GroupPtr[^1], NumRow, ErrorMsg.GroupSize + "the actual number of rows given by data.");
        if (Weights.Size != 0)
        {
            Check.Eq((long)Weights.Size, NumRow, "Size of weights must equal to number of rows.");
            return;
        }
        if (Labels.Size != 0)
        {
            Check.Eq(Labels.Shape(0), NumRow, "Size of labels must equal to number of rows.");
            return;
        }
        if (LabelsLowerBound.Size != 0)
        {
            Check.Eq((long)LabelsLowerBound.Size, NumRow, "Size of label_lower_bound must equal to number of rows.");
            return;
        }
        if (FeatureWeights.Size != 0)
            Check.Eq((long)FeatureWeights.Size, NumCol, "Size of feature_weights must equal to number of columns.");
        if (LabelsUpperBound.Size != 0)
        {
            Check.Eq((long)LabelsUpperBound.Size, NumRow, "Size of label_upper_bound must equal to number of rows.");
            return;
        }
        Check.Le(NumNonZero, NumCol * NumRow);
        if (BaseMargin.Size != 0) Check.Eq(BaseMargin.Size % NumRow, 0L, "Size of base margin must be a multiple of number of rows.");
    }

    // ---- setters ------------------------------------------------------------------------------

    public static MetaField MapMetaField(string key, bool isInput) => key switch
    {
        "label" => MetaField.Label,
        "weight" => MetaField.Weight,
        "base_margin" => MetaField.BaseMargin,
        "label_lower_bound" => MetaField.LabelLowerBound,
        "label_upper_bound" => MetaField.LabelUpperBound,
        "feature_weights" => MetaField.FeatureWeights,
        "group_ptr" when !isInput => MetaField.GroupPtr,
        "group" when isInput => MetaField.GroupPtr,
        "qid" => MetaField.Qid,
        _ => throw new XGBoostException($"Unknown key:{key}"),
    };

    private void ReshapeInfo(Tensor<float> info, string name)
    {
        if (NumRow != 0 && info.Shape(0) != NumRow)
        {
            Check.Eq(info.Size % NumRow, 0L, $"Invalid size for `{name}`:({info.Shape(0)},{info.Shape(1)}). n_samples:{NumRow}");
            var nGroups = info.Size / NumRow;
            info.Reshape(NumRow, nGroups);
        }
    }

    /// <summary>
    /// Sets a meta field from typed values of the given shape (1-D or 2-D, C order), the equivalent of
    /// <c>SetInfo</c> with an array interface.
    /// </summary>
    public void SetInfo<T>(string key, ReadOnlySpan<T> values, long[]? shape = null) where T : unmanaged, INumberBase<T>
    {
        shape ??= [values.Length];
        Check.Eq(Linalg.CalcSize(shape), (long)values.Length, "Shape doesn't match the number of values.");
        switch (MapMetaField(key, true))
        {
            case MetaField.Label:
            {
                Labels = ToTensor(values, shape);
                ReshapeInfo(Labels, "label");
                foreach (var y in Labels.Data.ConstHostSpan)
                    if (float.IsNaN(y) || float.IsInfinity(y)) Check.Fail("Label contains NaN, infinity or a value too large.");
                break;
            }
            case MetaField.Weight:
            {
                Weights = ToVector(values);
                foreach (var w in Weights.ConstHostSpan)
                    if (w < 0 || float.IsInfinity(w) || float.IsNaN(w)) Check.Fail("Weights must be positive values.");
                break;
            }
            case MetaField.BaseMargin:
            {
                BaseMargin = ToTensor(values, shape);
                ReshapeInfo(BaseMargin, "base_margin");
                break;
            }
            case MetaField.LabelLowerBound:
                LabelsLowerBound = ToVector(values);
                break;
            case MetaField.LabelUpperBound:
                LabelsUpperBound = ToVector(values);
                break;
            case MetaField.FeatureWeights:
            {
                FeatureWeights = ToVector(values);
                foreach (var w in FeatureWeights.ConstHostSpan)
                    if (float.IsNaN(w) || float.IsInfinity(w) || w < 0) Check.Fail("Feature weight must be greater than 0.");
                break;
            }
            case MetaField.GroupPtr:
            {
                GroupPtr = new List<uint>(values.Length + 1) { 0 };
                uint sum = 0;
                foreach (var g in values)
                {
                    sum = unchecked(sum + uint.CreateTruncating(g));
                    GroupPtr.Add(sum);
                }
                ValidateQueryGroup(GroupPtr);
                break;
            }
            case MetaField.Qid:
            {
                var qids = new uint[values.Length];
                for (var i = 0; i < values.Length; i++) qids[i] = uint.CreateTruncating(values[i]);
                for (var i = 1; i < qids.Length; i++)
                    if (qids[i] < qids[i - 1]) Check.Fail("`qid` must be sorted in non-decreasing order along with data.");
                GroupPtr = StdAlgo.RunLengthEncode<uint>(qids);
                ValidateQueryGroup(GroupPtr);
                break;
            }
        }
    }

    private static Tensor<float> ToTensor<T>(ReadOnlySpan<T> values, long[] shape) where T : unmanaged, INumberBase<T>
    {
        var data = new float[values.Length];
        for (var i = 0; i < values.Length; i++) data[i] = float.CreateTruncating(values[i]);
        long[] s2 = shape.Length switch
        {
            1 => [shape[0], 1],
            2 => shape,
            _ => throw new XGBoostException("Only 1-D and 2-D meta info are supported."),
        };
        if (values.Length == 0) s2 = shape.Length == 1 ? [0, 1] : shape;
        return new Tensor<float>(data, s2);
    }

    private static HostDeviceVector<float> ToVector<T>(ReadOnlySpan<T> values) where T : unmanaged, INumberBase<T>
    {
        var data = new float[values.Length];
        for (var i = 0; i < values.Length; i++) data[i] = float.CreateTruncating(values[i]);
        return new HostDeviceVector<float>(data, true);
    }

    public static void ValidateQueryGroup(List<uint> groupPtr)
    {
        for (var i = 1; i < groupPtr.Count; i++)
            if (groupPtr[i] < groupPtr[i - 1]) Check.Fail("Invalid group structure.");
    }

    /// <summary><c>GetInfo</c> for float fields, returning a copy of the values and their shape.</summary>
    public (float[] Values, long[] Shape) GetFloatInfo(string key)
    {
        (float[], long[]) Vec(HostDeviceVector<float> v) => (v.ToArray(), [v.Size]);
        (float[], long[]) Mat(Tensor<float> m) => m.Shape(1) <= 1 ? Vec(m.Data) : (m.Data.ToArray(), [m.Shape(0), m.Shape(1)]);
        return MapMetaField(key, false) switch
        {
            MetaField.Label => Mat(Labels),
            MetaField.Weight => Vec(Weights),
            MetaField.BaseMargin => Mat(BaseMargin),
            MetaField.LabelLowerBound => Vec(LabelsLowerBound),
            MetaField.LabelUpperBound => Vec(LabelsUpperBound),
            MetaField.FeatureWeights => Vec(FeatureWeights),
            MetaField.GroupPtr => throw new XGBoostException($"Invalid dtype for the requested field: `{key}`"),
            MetaField.Qid => throw new XGBoostException("Retrieving `qid` is not supported; use `group_ptr` instead."),
            _ => throw new XGBoostException($"Unknown field name: {key}"),
        };
    }

    public uint[] GetUIntInfo(string key) => MapMetaField(key, false) switch
    {
        MetaField.GroupPtr => [.. GroupPtr],
        MetaField.Qid => throw new XGBoostException("Retrieving `qid` is not supported; use `group_ptr` instead."),
        _ => throw new XGBoostException($"Invalid dtype for the requested field: `{key}`"),
    };

    public void SetFeatureInfo(string key, IReadOnlyList<string> info)
    {
        if (info.Count != 0 && NumCol != 0) Check.Eq((long)info.Count, NumCol, $"Length of {key} must be equal to number of columns.");
        switch (key)
        {
            case "feature_type":
                FeatureTypeNames = [.. info];
                _hasCategorical = LoadFeatureType(FeatureTypeNames, FeatureTypes);
                break;
            case "feature_name":
                FeatureNames = [.. info];
                break;
            default:
                Check.Fail($"Unknown feature info name: {key}");
                break;
        }
    }

    public string[] GetFeatureInfo(string field) => field switch
    {
        "feature_type" => [.. FeatureTypeNames],
        "feature_name" => [.. FeatureNames],
        _ => throw new XGBoostException($"Unknown feature info: {field}"),
    };
}
