// Port of dmlc/parameter.h and include/xgboost/parameter.h.
//
// A parameter struct derives from XGBoostParameter<T> and declares its fields once in Declare().
// Field printing and parsing follow the dmlc rules so that saved configurations match the C++ output.
using System.Globalization;
using System.Numerics;

namespace XGBoost.Common;

public abstract class FieldEntry<TParam>
{
    public string Key { get; internal set; } = "";
    public string TypeName { get; set; } = "";
    public string Description { get; internal set; } = "";
    public bool HasDefault { get; protected set; }

    public abstract void SetDefault(TParam head);
    public abstract void Set(TParam head, string value);
    public virtual void Check(TParam head) { }
    public abstract string GetStringValue(TParam head);
    public abstract string DefaultValueString();

    protected XGBoostException InvalidFormat(string value) =>
        new($"Invalid Parameter format for {Key} expect {TypeName} but value='{value}'");
}

/// <summary>Typed entry with default, range and description, mirroring <c>FieldEntryBase</c>/<c>FieldEntryNumeric</c>.</summary>
public class FieldEntry<TParam, TValue>(Func<TParam, TValue> get, Action<TParam, TValue> set) : FieldEntry<TParam>
{
    protected readonly Func<TParam, TValue> Getter = get;
    protected readonly Action<TParam, TValue> Setter = set;
    protected TValue DefaultValue = default!;

    public Func<string, TValue>? Parser { get; init; }
    public Func<TValue, string>? Printer { get; init; }

    private bool _hasBegin, _hasEnd;
    private TValue _begin = default!, _end = default!;

    public FieldEntry<TParam, TValue> SetDefault(TValue value)
    {
        DefaultValue = value;
        HasDefault = true;
        return this;
    }

    public FieldEntry<TParam, TValue> Describe(string description)
    {
        Description = description;
        return this;
    }

    public FieldEntry<TParam, TValue> SetRange(TValue begin, TValue end)
    {
        _begin = begin;
        _end = end;
        _hasBegin = _hasEnd = true;
        return this;
    }

    public FieldEntry<TParam, TValue> SetLowerBound(TValue begin)
    {
        _begin = begin;
        _hasBegin = true;
        return this;
    }

    public override void SetDefault(TParam head)
    {
        if (!HasDefault) throw new XGBoostException($"Required parameter {Key} of {TypeName} is not presented");
        Setter(head, DefaultValue);
    }

    public override void Set(TParam head, string value)
    {
        if (Parser is null) throw new InvalidOperationException($"No parser for {Key}.");
        Setter(head, Parser(value));
    }

    public override string GetStringValue(TParam head) => Print(Getter(head));

    public override string DefaultValueString() => Print(DefaultValue);

    protected virtual string Print(TValue value) => Printer is null ? value?.ToString() ?? "" : Printer(value);

    // C++ comparisons: false whenever NaN is involved.
    private static bool Less(TValue a, TValue b)
    {
        if (a is float fa && (float.IsNaN(fa) || float.IsNaN((float)(object)b!))) return false;
        if (a is double da && (double.IsNaN(da) || double.IsNaN((double)(object)b!))) return false;
        return Comparer<TValue>.Default.Compare(a, b) < 0;
    }

    public override void Check(TParam head)
    {
        if (!_hasBegin && !_hasEnd) return;
        var v = Getter(head);
        var suffix = $"\n{Key}: {Description}";
        if (_hasBegin && _hasEnd)
        {
            if (Less(v, _begin) || Less(_end, v))
                throw new XGBoostException($"value {Print(v)} for Parameter {Key} exceed bound [{Print(_begin)},{Print(_end)}]{suffix}");
        }
        else if (_hasBegin && Less(v, _begin))
        {
            throw new XGBoostException($"value {Print(v)} for Parameter {Key} should be greater equal to {Print(_begin)}{suffix}");
        }
        else if (_hasEnd && Less(_end, v))
        {
            throw new XGBoostException($"value {Print(v)} for Parameter {Key} should be smaller equal to {Print(_end)}{suffix}");
        }
    }
}

/// <summary>Integer field that may be an enum, mirroring <c>FieldEntry&lt;int&gt;</c> with <c>add_enum</c>.</summary>
public sealed class EnumFieldEntry<TParam, TEnum>(Func<TParam, TEnum> get, Action<TParam, TEnum> set)
    : FieldEntry<TParam, TEnum>(get, set) where TEnum : struct, Enum
{
    private readonly SortedDictionary<string, int> _map = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _back = [];

    public EnumFieldEntry<TParam, TEnum> AddEnum(string key, TEnum value)
    {
        var v = Convert.ToInt32(value, CultureInfo.InvariantCulture);
        if (_map.ContainsKey(key) || _back.ContainsKey(v)) throw new XGBoostException($"Enum ({key}: {v} exisit!)");
        _map[key] = v;
        _back[v] = key;
        return this;
    }

    public new EnumFieldEntry<TParam, TEnum> SetDefault(TEnum value)
    {
        base.SetDefault(value);
        return this;
    }

    public new EnumFieldEntry<TParam, TEnum> Describe(string description)
    {
        base.Describe(description);
        return this;
    }

    public override void Set(TParam head, string value)
    {
        if (!_map.TryGetValue(value, out var v))
            throw new XGBoostException($"Invalid Input: '{value}', valid values are: {{{string.Join(", ", _map.Keys.Select(k => $"'{k}'"))}}}");
        Setter(head, (TEnum)Enum.ToObject(typeof(TEnum), v));
    }

    protected override string Print(TEnum value)
    {
        var v = Convert.ToInt32(value, CultureInfo.InvariantCulture);
        return _back.TryGetValue(v, out var s) ? s : throw new XGBoostException("Value not found in enum declared");
    }

    public override string DefaultValueString() => $"'{Print(DefaultValue)}'";
}

/// <summary>Port of <c>ParamManager</c>: ordered entries plus an alias-inclusive key map.</summary>
public sealed class ParamManager<TParam>(string name)
{
    private readonly List<FieldEntry<TParam>> _entries = [];
    private readonly SortedDictionary<string, FieldEntry<TParam>> _map = new(StringComparer.Ordinal);

    public string Name { get; } = name;

    public IReadOnlyList<FieldEntry<TParam>> Entries => _entries;

    public FieldEntry<TParam>? Find(string key) => _map.GetValueOrDefault(key);

    public void AddEntry(string key, FieldEntry<TParam> e)
    {
        if (_map.ContainsKey(key)) throw new XGBoostException($"key {key} has already been registered in {Name}");
        e.Key = key;
        _entries.Add(e);
        _map[key] = e;
    }

    public void AddAlias(string field, string alias)
    {
        if (!_map.TryGetValue(field, out var e)) throw new XGBoostException($"key {field} has not been registered in {Name}");
        if (_map.ContainsKey(alias)) throw new XGBoostException($"Alias {alias} has already been registered in {Name}");
        _map[alias] = e;
    }

    public void RunUpdate(TParam head, IEnumerable<KeyValuePair<string, string>> kwargs,
        List<KeyValuePair<string, string>>? unknown, HashSet<FieldEntry<TParam>>? selected)
    {
        foreach (var kv in kwargs)
        {
            if (Find(kv.Key) is { } e)
            {
                e.Set(head, kv.Value);
                e.Check(head);
                selected?.Add(e);
            }
            else
            {
                unknown?.Add(kv);
            }
        }
    }

    public void RunInit(TParam head, IEnumerable<KeyValuePair<string, string>> kwargs, List<KeyValuePair<string, string>>? unknown)
    {
        var selected = new HashSet<FieldEntry<TParam>>();
        RunUpdate(head, kwargs, unknown, selected);
        foreach (var e in _map.Values)
            if (!selected.Contains(e)) e.SetDefault(head);
    }

    /// <summary>Key/value pairs over the alias-inclusive map, sorted by key (<c>GetDict</c>).</summary>
    public List<KeyValuePair<string, string>> GetDict(TParam head) =>
        [.. _map.Select(kv => new KeyValuePair<string, string>(kv.Key, kv.Value.GetStringValue(head)))];

    public string DocString()
    {
        var sb = new System.Text.StringBuilder();
        foreach (var e in _entries)
        {
            sb.Append(e.Key).Append(" : ").Append(e.TypeName);
            sb.Append(e.HasDefault ? ", optional, default=" + e.DefaultValueString() : ", required").Append('\n');
            if (e.Description.Length != 0) sb.Append("    ").Append(e.Description).Append('\n');
        }
        return sb.ToString();
    }
}

/// <summary>Base class for parameter structs; <c>XGBoostParameter</c> on top of <c>dmlc::Parameter</c>.</summary>
public abstract class XGBoostParameter<T> where T : XGBoostParameter<T>, new()
{
    private static ParamManager<T>? _manager;
    private static readonly Lock ManagerLock = new();

    protected bool Initialised;

    public bool GetInitialised() => Initialised;

    public static ParamManager<T> Manager
    {
        get
        {
            if (_manager is not null) return _manager;
            lock (ManagerLock)
            {
                if (_manager is null)
                {
                    var m = new ParamManager<T>(typeof(T).Name);
                    new T().Declare(m);
                    _manager = m;
                }
            }
            return _manager;
        }
    }

    /// <summary>Declares the fields; called once on a throw-away instance.</summary>
    protected abstract void Declare(ParamManager<T> m);

    private T Self => (T)this;

    public void Init(IEnumerable<KeyValuePair<string, string>> kwargs)
    {
        var unknown = new List<KeyValuePair<string, string>>();
        Manager.RunInit(Self, kwargs, unknown);
        foreach (var kv in unknown)
        {
            // kAllowHidden: names like __xxx__ are skipped silently.
            if (kv.Key.Length > 4 && kv.Key.StartsWith("__", StringComparison.Ordinal) && kv.Key.EndsWith("__", StringComparison.Ordinal))
                continue;
            throw new XGBoostException($"Cannot find argument '{kv.Key}', Possible Arguments:\n----------------\n{Manager.DocString()}");
        }
        Initialised = true;
    }

    public List<KeyValuePair<string, string>> InitAllowUnknown(IEnumerable<KeyValuePair<string, string>> kwargs)
    {
        var unknown = new List<KeyValuePair<string, string>>();
        Manager.RunInit(Self, kwargs, unknown);
        return unknown;
    }

    /// <summary>Initialises with defaults the first time, then only updates the given keys.</summary>
    public virtual List<KeyValuePair<string, string>> UpdateAllowUnknown(IEnumerable<KeyValuePair<string, string>> kwargs)
    {
        if (Initialised)
        {
            var unknown = new List<KeyValuePair<string, string>>();
            Manager.RunUpdate(Self, kwargs, unknown, null);
            return unknown;
        }
        var u = InitAllowUnknown(kwargs);
        Initialised = true;
        return u;
    }

    /// <summary><c>__DICT__</c>: every field and alias with its string value, sorted by name.</summary>
    public SortedDictionary<string, string> Dict()
    {
        var d = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (var kv in Manager.GetDict(Self)) d[kv.Key] = kv.Value;
        return d;
    }

    /// <summary><c>ToJson(param)</c>: an object of string values.</summary>
    public JsonObject ToJson()
    {
        var obj = new JsonObject();
        foreach (var kv in Dict()) obj[kv.Key] = new JsonString(kv.Value);
        return obj;
    }

    /// <summary><c>FromJson(obj, &amp;param)</c>: returns unknown arguments.</summary>
    public List<KeyValuePair<string, string>> FromJson(Json obj)
    {
        var args = obj.AsObject.Select(kv => new KeyValuePair<string, string>(kv.Key, kv.Value.AsString)).ToList();
        return UpdateAllowUnknown(args);
    }

    // ---- Declaration helpers ----------------------------------------------------------------

    protected static FieldEntry<T, float> Field(ParamManager<T> m, string key, Func<T, float> get, Action<T, float> set)
    {
        var e = new FieldEntry<T, float>(get, set) { TypeName = "float", Parser = v => ParseFloat(key, v), Printer = Format.ParamFloat };
        m.AddEntry(key, e);
        return e;
    }

    protected static FieldEntry<T, double> Field(ParamManager<T> m, string key, Func<T, double> get, Action<T, double> set)
    {
        var e = new FieldEntry<T, double>(get, set) { TypeName = "double", Parser = v => ParseDouble(key, v), Printer = Format.ParamDouble };
        m.AddEntry(key, e);
        return e;
    }

    protected static FieldEntry<T, int> Field(ParamManager<T> m, string key, Func<T, int> get, Action<T, int> set)
    {
        var e = new FieldEntry<T, int>(get, set) { TypeName = "int", Parser = v => ParseInt<int>(key, "int", v), Printer = v => Format.I(v) };
        m.AddEntry(key, e);
        return e;
    }

    protected static FieldEntry<T, long> Field(ParamManager<T> m, string key, Func<T, long> get, Action<T, long> set)
    {
        var e = new FieldEntry<T, long>(get, set) { TypeName = "long", Parser = v => ParseInt<long>(key, "long", v), Printer = v => Format.I(v) };
        m.AddEntry(key, e);
        return e;
    }

    protected static FieldEntry<T, uint> Field(ParamManager<T> m, string key, Func<T, uint> get, Action<T, uint> set)
    {
        var e = new FieldEntry<T, uint>(get, set) { TypeName = "unsigned int", Parser = v => ParseUInt<uint>(key, "unsigned int", v), Printer = v => Format.I(v) };
        m.AddEntry(key, e);
        return e;
    }

    protected static FieldEntry<T, ulong> Field(ParamManager<T> m, string key, Func<T, ulong> get, Action<T, ulong> set)
    {
        var e = new FieldEntry<T, ulong>(get, set) { TypeName = "unsigned long", Parser = v => ParseUInt<ulong>(key, "unsigned long", v), Printer = v => Format.I(v) };
        m.AddEntry(key, e);
        return e;
    }

    protected static FieldEntry<T, bool> Field(ParamManager<T> m, string key, Func<T, bool> get, Action<T, bool> set)
    {
        var e = new FieldEntry<T, bool>(get, set) { TypeName = "boolean", Parser = v => ParseBool(key, v), Printer = v => v ? "1" : "0" };
        m.AddEntry(key, e);
        return e;
    }

    protected static FieldEntry<T, string> Field(ParamManager<T> m, string key, Func<T, string> get, Action<T, string> set)
    {
        var e = new StringFieldEntry<T>(get, set);
        m.AddEntry(key, e);
        return e;
    }

    protected static EnumFieldEntry<T, TEnum> EnumField<TEnum>(ParamManager<T> m, string key, Func<T, TEnum> get, Action<T, TEnum> set)
        where TEnum : struct, Enum
    {
        var e = new EnumFieldEntry<T, TEnum>(get, set) { TypeName = "int" };
        m.AddEntry(key, e);
        return e;
    }

    /// <summary>A field with a custom type, parser and printer (e.g. <c>ParamArray</c>).</summary>
    protected static FieldEntry<T, TValue> CustomField<TValue>(ParamManager<T> m, string key, string typeName,
        Func<T, TValue> get, Action<T, TValue> set, Func<string, TValue> parse, Func<TValue, string> print)
    {
        var e = new FieldEntry<T, TValue>(get, set) { TypeName = typeName, Parser = parse, Printer = print };
        m.AddEntry(key, e);
        return e;
    }

    protected static void Alias(ParamManager<T> m, string field, string alias) => m.AddAlias(field, alias);

    public static float ParseFloat(string key, string value)
    {
        if (!Format.TryParseFloatPrefix(value, out var d, out var consumed))
            throw new XGBoostException($"Invalid Parameter format for {key} expect float but value='{value}'");
        if (consumed < value.Length) throw new XGBoostException($"Some trailing characters could not be parsed: '{value[consumed..]}'");
        var f = (float)d;
        if (float.IsInfinity(f) && !double.IsInfinity(d)) throw new XGBoostException($"Out of range value for {key}, value='{value}'");
        return f;
    }

    public static double ParseDouble(string key, string value)
    {
        if (!Format.TryParseFloatPrefix(value, out var d, out var consumed))
            throw new XGBoostException($"Invalid Parameter format for {key} expect double but value='{value}'");
        if (consumed < value.Length) throw new XGBoostException($"Some trailing characters could not be parsed: '{value[consumed..]}'");
        return d;
    }

    private static TInt ParseInt<TInt>(string key, string type, string value) where TInt : IBinaryInteger<TInt>, IMinMaxValue<TInt>
    {
        if (!Format.TryParseIntegerStream(value, out var v))
            throw new XGBoostException($"Invalid Parameter format for {key} expect {type} but value='{value}'");
        // istream sets failbit and clamps on overflow.
        if (v < long.CreateTruncating(TInt.MinValue) || v > long.CreateTruncating(TInt.MaxValue))
            throw new XGBoostException($"Invalid Parameter format for {key} expect {type} but value='{value}'");
        return TInt.CreateTruncating(v);
    }

    private static TInt ParseUInt<TInt>(string key, string type, string value) where TInt : IBinaryInteger<TInt>, IMinMaxValue<TInt>
    {
        if (!Format.TryParseUnsignedStream(value, out var v))
            throw new XGBoostException($"Invalid Parameter format for {key} expect {type} but value='{value}'");
        return TInt.CreateTruncating(v);
    }

    private static bool ParseBool(string key, string value) => value.ToLowerInvariant() switch
    {
        "true" or "1" => true,
        "false" or "0" => false,
        _ => throw new XGBoostException($"Invalid Parameter format for {key} expect boolean but value='{value}'"),
    };
}

internal sealed class StringFieldEntry<TParam> : FieldEntry<TParam, string>
{
    public StringFieldEntry(Func<TParam, string> get, Action<TParam, string> set) : base(get, set)
    {
        TypeName = "string";
    }

    public override void Set(TParam head, string value) => Setter(head, value);
    protected override string Print(string value) => value;
    public override string DefaultValueString() => $"'{DefaultValue}'";
}

public static class ParameterUtils
{
    /// <summary>Supplied parameter names that don't appear in <paramref name="unknown"/>.</summary>
    public static SortedSet<string> GetUsedParameters(IEnumerable<KeyValuePair<string, string>> args,
        IEnumerable<KeyValuePair<string, string>> unknown)
    {
        var used = new SortedSet<string>(args.Select(a => a.Key), StringComparer.Ordinal);
        used.ExceptWith(unknown.Select(u => u.Key));
        return used;
    }
}
