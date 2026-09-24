// Port of include/xgboost/json.h, include/xgboost/json_io.h and src/common/json.cc.
using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace XGBoost.Common;

public enum JsonKind : long
{
    String = 0,
    Number = 1,
    Integer = 2,
    Object = 3,
    Array = 4,
    Boolean = 5,
    Null = 6,
    F32Array = 7,
    F64Array = 8,
    I8Array = 9,
    U8Array = 10,
    I16Array = 11,
    U16Array = 12,
    I32Array = 13,
    U32Array = 14,
    I64Array = 15,
    U64Array = 16,
}

/// <summary>A JSON value. Mirrors <c>xgboost::Json</c>, including the UBJSON typed arrays.</summary>
public abstract class Json
{
    public abstract JsonKind Kind { get; }

    public string TypeStr => Kind.ToString();

    public virtual Json this[string key]
    {
        get => throw new XGBoostException($"Object of type {TypeStr} can not be indexed by string.");
        set => throw new XGBoostException($"Object of type {TypeStr} can not be indexed by string.");
    }

    public virtual Json this[int index]
    {
        get => throw new XGBoostException($"Object of type {TypeStr} can not be indexed by Integer.");
        set => throw new XGBoostException($"Object of type {TypeStr} can not be indexed by Integer.");
    }

    public T Cast<T>() where T : Json =>
        this as T ?? throw new XGBoostException($"Invalid cast, from {TypeStr} to {typeof(T).Name}");

    public bool IsA<T>() where T : Json => this is T;

    // Accessors equivalent to get<T>(json).
    public JsonObject AsObject => Cast<JsonObject>();
    public List<Json> AsArray => Cast<JsonArray>().Values;
    public string AsString => Cast<JsonString>().Value;
    public float AsNumber => Cast<JsonNumber>().Value;
    public long AsInteger => Cast<JsonInteger>().Value;
    public bool AsBoolean => Cast<JsonBoolean>().Value;

    public static implicit operator Json(string value) => new JsonString(value);
    public static implicit operator Json(float value) => new JsonNumber(value);
    public static implicit operator Json(double value) => new JsonNumber((float)value);
    public static implicit operator Json(long value) => new JsonInteger(value);
    public static implicit operator Json(int value) => new JsonInteger(value);
    public static implicit operator Json(uint value) => new JsonInteger(value);
    public static implicit operator Json(ulong value) => new JsonInteger((long)value);
    public static implicit operator Json(bool value) => new JsonBoolean(value);

    public abstract bool ValueEquals(Json other);

    public static Json Load(string text) => new JsonReader(Encoding.UTF8.GetBytes(text)).Load();

    public static Json Load(ReadOnlySpan<byte> data, bool binary)
    {
        if (binary) return new UbjReader(data.ToArray()).Load();
        return new JsonReader(data.ToArray()).Load();
    }

    public static string Dump(Json json)
    {
        var writer = new JsonWriter();
        writer.Save(json);
        return Encoding.UTF8.GetString(writer.ToArray());
    }

    public static byte[] DumpBytes(Json json, bool binary)
    {
        JsonWriter writer = binary ? new UbjWriter() : new JsonWriter();
        writer.Save(json);
        return writer.ToArray();
    }

    public override string ToString() => Dump(this);
}

public sealed class JsonString(string value) : Json
{
    public string Value { get; set; } = value;
    public override JsonKind Kind => JsonKind.String;
    public override bool ValueEquals(Json other) => other is JsonString s && s.Value == Value;
}

public sealed class JsonNumber(float value) : Json
{
    public float Value { get; set; } = value;
    public override JsonKind Kind => JsonKind.Number;

    public override bool ValueEquals(Json other)
    {
        if (other is not JsonNumber n) return false;
        if (float.IsInfinity(Value)) return float.IsInfinity(n.Value);
        if (float.IsNaN(Value)) return float.IsNaN(n.Value);
        return Value - n.Value == 0;
    }
}

public sealed class JsonInteger(long value) : Json
{
    public long Value { get; set; } = value;
    public override JsonKind Kind => JsonKind.Integer;
    public override bool ValueEquals(Json other) => other is JsonInteger i && i.Value == Value;
}

public sealed class JsonBoolean(bool value) : Json
{
    public bool Value { get; set; } = value;
    public override JsonKind Kind => JsonKind.Boolean;
    public override bool ValueEquals(Json other) => other is JsonBoolean b && b.Value == Value;
}

public sealed class JsonNull : Json
{
    public static readonly JsonNull Instance = new();
    public override JsonKind Kind => JsonKind.Null;
    public override bool ValueEquals(Json other) => other is JsonNull;
}

public sealed class JsonArray : Json
{
    public JsonArray() => Values = [];
    public JsonArray(List<Json> values) => Values = values;
    public JsonArray(IEnumerable<Json> values) => Values = [.. values];

    public List<Json> Values { get; }
    public override JsonKind Kind => JsonKind.Array;

    public override Json this[int index]
    {
        get => Values[index];
        set => Values[index] = value;
    }

    public int Count => Values.Count;
    public void Add(Json value) => Values.Add(value);

    public override bool ValueEquals(Json other)
    {
        if (other is not JsonArray a || a.Values.Count != Values.Count) return false;
        for (var i = 0; i < Values.Count; i++)
            if (!Values[i].ValueEquals(a.Values[i])) return false;
        return true;
    }
}

/// <summary>JSON object with keys ordered like <c>std::map&lt;std::string, Json&gt;</c> (byte-wise).</summary>
public sealed class JsonObject : Json, IEnumerable<KeyValuePair<string, Json>>
{
    public SortedDictionary<string, Json> Map { get; } = new(StringComparer.Ordinal);

    public override JsonKind Kind => JsonKind.Object;

    /// <summary>Like <c>obj[key]</c> in C++, inserts a null when the key is missing.</summary>
    public override Json this[string key]
    {
        get
        {
            if (!Map.TryGetValue(key, out var v)) Map[key] = v = JsonNull.Instance;
            return v;
        }
        set => Map[key] = value;
    }

    public bool ContainsKey(string key) => Map.ContainsKey(key);
    public bool TryGetValue(string key, out Json value) => Map.TryGetValue(key, out value!);
    public Json? Get(string key) => Map.TryGetValue(key, out var v) ? v : null;
    public int Count => Map.Count;
    public void Add(string key, Json value) => Map[key] = value;
    public bool Remove(string key) => Map.Remove(key);

    public IEnumerator<KeyValuePair<string, Json>> GetEnumerator() => Map.GetEnumerator();
    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    public override bool ValueEquals(Json other)
    {
        if (other is not JsonObject o || o.Map.Count != Map.Count) return false;
        foreach (var (k, v) in Map)
            if (!o.Map.TryGetValue(k, out var ov) || !v.ValueEquals(ov)) return false;
        return true;
    }
}

public abstract class JsonTypedArray : Json
{
    public abstract int Count { get; }
}

public abstract class JsonTypedArray<T>(T[] values) : JsonTypedArray where T : unmanaged
{
    public T[] Values { get; set; } = values;
    public override int Count => Values.Length;

    public override bool ValueEquals(Json other)
    {
        if (other is not JsonTypedArray<T> a || a.Kind != Kind || a.Values.Length != Values.Length) return false;
        if (typeof(T) == typeof(float))
        {
            var l = (float[])(object)Values;
            var r = (float[])(object)a.Values;
            for (var i = 0; i < l.Length; i++)
            {
                bool eq;
                if (float.IsNaN(l[i])) eq = float.IsNaN(r[i]);
                else if (float.IsInfinity(l[i])) eq = float.IsInfinity(r[i]);
                else eq = r[i] - l[i] == 0;
                if (!eq) return false;
            }
            return true;
        }
        return Values.AsSpan().SequenceEqual(a.Values);
    }
}

public sealed class F32Array(float[] v) : JsonTypedArray<float>(v)
{
    public F32Array(int n) : this(new float[n]) { }
    public override JsonKind Kind => JsonKind.F32Array;
}

public sealed class F64Array(double[] v) : JsonTypedArray<double>(v)
{
    public F64Array(int n) : this(new double[n]) { }
    public override JsonKind Kind => JsonKind.F64Array;
}

public sealed class I8Array(sbyte[] v) : JsonTypedArray<sbyte>(v)
{
    public I8Array(int n) : this(new sbyte[n]) { }
    public override JsonKind Kind => JsonKind.I8Array;
}

public sealed class U8Array(byte[] v) : JsonTypedArray<byte>(v)
{
    public U8Array(int n) : this(new byte[n]) { }
    public override JsonKind Kind => JsonKind.U8Array;
}

public sealed class I16Array(short[] v) : JsonTypedArray<short>(v)
{
    public I16Array(int n) : this(new short[n]) { }
    public override JsonKind Kind => JsonKind.I16Array;
}

public sealed class U16Array(ushort[] v) : JsonTypedArray<ushort>(v)
{
    public U16Array(int n) : this(new ushort[n]) { }
    public override JsonKind Kind => JsonKind.U16Array;
}

public sealed class I32Array(int[] v) : JsonTypedArray<int>(v)
{
    public I32Array(int n) : this(new int[n]) { }
    public override JsonKind Kind => JsonKind.I32Array;
}

public sealed class U32Array(uint[] v) : JsonTypedArray<uint>(v)
{
    public U32Array(int n) : this(new uint[n]) { }
    public override JsonKind Kind => JsonKind.U32Array;
}

public sealed class I64Array(long[] v) : JsonTypedArray<long>(v)
{
    public I64Array(int n) : this(new long[n]) { }
    public override JsonKind Kind => JsonKind.I64Array;
}

public sealed class U64Array(ulong[] v) : JsonTypedArray<ulong>(v)
{
    public U64Array(int n) : this(new ulong[n]) { }
    public override JsonKind Kind => JsonKind.U64Array;
}

/// <summary>Text JSON reader. Accepts <c>NaN</c> and <c>Infinity</c> like the C++ reader.</summary>
public class JsonReader(byte[] raw)
{
    protected readonly byte[] Raw = raw;
    protected int Pos;

    protected int PeekNextChar() => Pos == Raw.Length ? -1 : Raw[Pos];

    protected int GetNextChar() => Pos == Raw.Length ? -1 : Raw[Pos++];

    protected void SkipSpaces()
    {
        while (Pos < Raw.Length)
        {
            var c = Raw[Pos];
            if (c is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t') Pos++;
            else break;
        }
    }

    protected int GetNextNonSpaceChar()
    {
        SkipSpaces();
        return GetNextChar();
    }

    protected int GetConsecutiveChar(char expected)
    {
        var result = GetNextChar();
        if (result != expected) Expect(expected, result);
        return result;
    }

    protected void Expect(char c, int got)
    {
        var msg = $"Expecting: \"{c}\", got: \"";
        msg += got switch { -1 => "EOF\"", 0 => "\\0\"", _ => $"{got} \"" };
        Error(msg);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    protected void Error(string msg)
    {
        msg += $", around character position: {Pos}\n";
        const int extend = 8;
        var beg = Math.Max(0, Pos - extend);
        var end = Math.Min(Raw.Length, Pos + extend);
        var portion = Encoding.UTF8.GetString(Raw, beg, end - beg).Replace("\n", "\\n").Replace("\0", "\\0");
        msg += "    " + portion + "\n    ";
        msg += new string('~', Math.Max(0, Pos - 1 - beg)) + "^" + new string('~', Math.Max(0, end - Pos));
        throw new XGBoostException(msg);
    }

    public virtual Json Load() => Parse();

    protected virtual Json Parse()
    {
        while (true)
        {
            SkipSpaces();
            var c = PeekNextChar();
            if (c == -1) break;
            if (c == '{') return ParseObject();
            if (c == '[') return ParseArray();
            if (c == '-' || (c >= '0' && c <= '9') || c == 'N' || c == 'I') return ParseNumber();
            if (c == '"') return new JsonString(ParseStringLiteral());
            if (c == 't' || c == 'f') return ParseBoolean();
            if (c == 'n') return ParseNull();
            Error("Unknown construct");
        }
        return JsonNull.Instance;
    }

    private string ParseStringLiteral()
    {
        GetConsecutiveChar('"');
        var begin = Pos;
        var pos = begin;
        while (pos < Raw.Length)
        {
            var c = Raw[pos];
            if (c == '"')
            {
                var s = Encoding.UTF8.GetString(Raw, begin, pos - begin);
                Pos = pos + 1;
                return s;
            }
            if (c is (byte)'\\' or (byte)'\r' or (byte)'\n') break;
            pos++;
        }

        var bytes = new List<byte>();
        while (true)
        {
            var ch = GetNextChar();
            if (ch == '\\')
            {
                var next = GetNextChar();
                switch (next)
                {
                    case 'r': bytes.Add((byte)'\r'); break;
                    case 'n': bytes.Add((byte)'\n'); break;
                    case '\\': bytes.Add((byte)'\\'); break;
                    case 't': bytes.Add((byte)'\t'); break;
                    case '"': bytes.Add((byte)'"'); break;
                    case 'u':
                        bytes.Add((byte)'\\');
                        bytes.Add((byte)'u');
                        break;
                    default:
                        Error("Unknown escape");
                        break;
                }
            }
            else
            {
                if (ch == '"') break;
                if (ch != -1) bytes.Add((byte)ch);
            }
            if (ch == -1 || ch == '\r' || ch == '\n') Expect('"', ch);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private Json ParseNull()
    {
        var start = Pos;
        GetNextNonSpaceChar();
        for (var i = 0; i < 3; i++) GetNextChar();
        if (Pos - start < 4 || Encoding.ASCII.GetString(Raw, Pos - 4, 4) != "null") Error("Expecting null value \"null\"");
        return JsonNull.Instance;
    }

    private Json ParseArray()
    {
        var data = new List<Json>();
        GetConsecutiveChar('[');
        while (true)
        {
            if (PeekNextChar() == ']')
            {
                GetConsecutiveChar(']');
                return new JsonArray(data);
            }
            data.Add(Parse());
            var ch = GetNextNonSpaceChar();
            if (ch == ']') break;
            if (ch != ',') Expect(',', ch);
        }
        return new JsonArray(data);
    }

    private Json ParseObject()
    {
        GetConsecutiveChar('{');
        var obj = new JsonObject();
        SkipSpaces();
        var ch = PeekNextChar();
        if (ch == '}')
        {
            GetConsecutiveChar('}');
            return obj;
        }
        while (true)
        {
            SkipSpaces();
            ch = PeekNextChar();
            if (ch == -1) throw new XGBoostException($"cursor_.Pos(): {Pos}, raw_str_.size():{Raw.Length}");
            if (ch != '"') Expect('"', ch);
            var key = ParseStringLiteral();
            ch = GetNextNonSpaceChar();
            if (ch != ':') Expect(':', ch);
            var value = Parse();
            obj.Map[key] = value;
            ch = GetNextNonSpaceChar();
            if (ch == '}') break;
            if (ch != ',') Expect(',', ch);
        }
        return obj;
    }

    private Json ParseNumber()
    {
        var p = Pos;
        var beg = p;
        byte At(int i) => i < Raw.Length ? Raw[i] : (byte)0;

        if (At(p) == 'N')
        {
            GetConsecutiveChar('N');
            GetConsecutiveChar('a');
            GetConsecutiveChar('N');
            return new JsonNumber(float.NaN);
        }

        var negative = false;
        if (At(p) == '-')
        {
            negative = true;
            p++;
        }
        else if (At(p) == '+')
        {
            p++;
        }

        if (At(p) == 'I')
        {
            Pos += p - beg;
            foreach (var c in "Infinity") GetConsecutiveChar(c);
            return new JsonNumber(negative ? float.NegativeInfinity : float.PositiveInfinity);
        }

        var isFloat = false;
        long i = 0;
        if (At(p) == '0') p++;
        while (At(p) >= '0' && At(p) <= '9')
        {
            i = unchecked(i * 10 + (At(p) - '0'));
            p++;
        }
        if (At(p) == '.')
        {
            p++;
            isFloat = true;
            while (At(p) >= '0' && At(p) <= '9') p++;
        }
        if (At(p) == 'E' || At(p) == 'e')
        {
            isFloat = true;
            p++;
            if (At(p) == '-' || At(p) == '+') p++;
            if (At(p) >= '0' && At(p) <= '9')
            {
                p++;
                while (At(p) >= '0' && At(p) <= '9') p++;
            }
            else
            {
                Error("Expecting digit");
            }
        }

        Pos += p - beg;
        if (isFloat)
        {
            var text = Encoding.ASCII.GetString(Raw, beg, p - beg);
            return new JsonNumber(float.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture));
        }
        return new JsonInteger(negative ? -i : i);
    }

    private Json ParseBoolean()
    {
        var ch = GetNextNonSpaceChar();
        if (ch == 't')
        {
            GetConsecutiveChar('r');
            GetConsecutiveChar('u');
            GetConsecutiveChar('e');
            return new JsonBoolean(true);
        }
        GetConsecutiveChar('a');
        GetConsecutiveChar('l');
        GetConsecutiveChar('s');
        GetConsecutiveChar('e');
        return new JsonBoolean(false);
    }
}

/// <summary>Reader for UBJSON (https://ubjson.org/), big-endian.</summary>
public sealed class UbjReader(byte[] raw) : JsonReader(raw)
{
    public override Json Load() => Parse();

    private T ReadPrimitive<T>() where T : unmanaged
    {
        var size = Unsafe.SizeOf<T>();
        if (Pos + size > Raw.Length) throw new XGBoostException("Unexpected end of UBJSON input.");
        var span = Raw.AsSpan(Pos, size);
        Pos += size;
        T v;
        if (size == 1) return Unsafe.ReadUnaligned<T>(ref span[0]);
        Span<byte> tmp = stackalloc byte[size];
        span.CopyTo(tmp);
        tmp.Reverse();
        v = Unsafe.ReadUnaligned<T>(ref tmp[0]);
        return v;
    }

    private string DecodeStr()
    {
        GetConsecutiveChar('L');
        var n = checked((int)ReadPrimitive<long>());
        var s = Encoding.UTF8.GetString(Raw, Pos, n);
        Pos += n;
        return s;
    }

    private T[] ReadTyped<T>(long n) where T : unmanaged
    {
        var result = new T[checked((int)n)];
        for (var i = 0; i < result.Length; i++) result[i] = ReadPrimitive<T>();
        return result;
    }

    private Json ParseArray()
    {
        var marker = PeekNextChar();
        if (marker == '$')
        {
            GetNextChar();
            var type = GetNextChar();
            GetConsecutiveChar('#');
            GetConsecutiveChar('L');
            var n = ReadPrimitive<long>();
            return type switch
            {
                'd' => new F32Array(ReadTyped<float>(n)),
                'D' => new F64Array(ReadTyped<double>(n)),
                'i' => new I8Array(ReadTyped<sbyte>(n)),
                'U' => new U8Array(ReadTyped<byte>(n)),
                'I' => new I16Array(ReadTyped<short>(n)),
                'l' => new I32Array(ReadTyped<int>(n)),
                'L' => new I64Array(ReadTyped<long>(n)),
                _ => throw new XGBoostException($"`{(char)type}` is not supported for typed array."),
            };
        }

        var results = new List<Json>();
        if (marker == '#')
        {
            GetNextChar();
            GetConsecutiveChar('L');
            var n = ReadPrimitive<long>();
            results.Capacity = checked((int)n);
            for (long i = 0; i < n; i++) results.Add(Parse());
        }
        else
        {
            while (marker != ']')
            {
                results.Add(Parse());
                marker = PeekNextChar();
            }
            GetConsecutiveChar(']');
        }
        return new JsonArray(results);
    }

    private Json ParseObject()
    {
        var obj = new JsonObject();
        var marker = PeekNextChar();
        while (marker != '}')
        {
            var key = DecodeStr();
            var value = Parse();
            obj.Map.TryAdd(key, value);
            marker = PeekNextChar();
        }
        GetConsecutiveChar('}');
        return obj;
    }

    protected override Json Parse()
    {
        while (true)
        {
            var c = PeekNextChar();
            if (c == -1) break;
            GetNextChar();
            switch (c)
            {
                case '{': return ParseObject();
                case '[': return ParseArray();
                case 'Z': return JsonNull.Instance;
                case 'T': return new JsonBoolean(true);
                case 'F': return new JsonBoolean(false);
                case 'd': return new JsonNumber(ReadPrimitive<float>());
                case 'D': return new JsonNumber((float)ReadPrimitive<double>());
                case 'S': return new JsonString(DecodeStr());
                case 'i': return new JsonInteger(ReadPrimitive<sbyte>());
                case 'U': return new JsonInteger(ReadPrimitive<byte>());
                case 'I': return new JsonInteger(ReadPrimitive<short>());
                case 'l': return new JsonInteger(ReadPrimitive<int>());
                case 'L': return new JsonInteger(ReadPrimitive<long>());
                case 'C': return new JsonInteger((sbyte)ReadPrimitive<byte>());
                case 'H': throw new XGBoostException("High precision number is not supported.");
                default:
                    Error("Unknown construct");
                    break;
            }
        }
        return JsonNull.Instance;
    }
}

/// <summary>Text JSON writer, byte-compatible with the C++ <c>JsonWriter</c>.</summary>
public class JsonWriter
{
    protected readonly List<byte> Stream = [];

    public byte[] ToArray() => [.. Stream];

    protected void Put(char c) => Stream.Add((byte)c);

    protected void PutAscii(string s)
    {
        foreach (var c in s) Stream.Add((byte)c);
    }

    public virtual void Save(Json json)
    {
        switch (json)
        {
            case JsonObject o: VisitObject(o); break;
            case JsonArray a: VisitArray(a); break;
            case JsonString s: VisitString(s.Value); break;
            case JsonNumber n: VisitNumber(n.Value); break;
            case JsonInteger i: VisitInteger(i.Value); break;
            case JsonBoolean b: VisitBoolean(b.Value); break;
            case JsonNull: VisitNull(); break;
            case JsonTypedArray t: VisitTyped(t); break;
            default: throw new XGBoostException($"Unknown JSON value {json.GetType()}.");
        }
    }

    protected virtual void VisitTyped(JsonTypedArray arr)
    {
        switch (arr)
        {
            case F32Array f:
                WriteArray(f.Values.Length, i => VisitNumber(f.Values[i]));
                break;
            case F64Array:
                throw new XGBoostException("Only UBJSON format can handle f64 array.");
            case I8Array a: WriteArray(a.Count, i => VisitInteger(a.Values[i])); break;
            case U8Array a: WriteArray(a.Count, i => VisitInteger(a.Values[i])); break;
            case I16Array a: WriteArray(a.Count, i => VisitInteger(a.Values[i])); break;
            case U16Array a: WriteArray(a.Count, i => VisitInteger(a.Values[i])); break;
            case I32Array a: WriteArray(a.Count, i => VisitInteger(a.Values[i])); break;
            case U32Array a: WriteArray(a.Count, i => VisitInteger(a.Values[i])); break;
            case I64Array a: WriteArray(a.Count, i => VisitInteger(a.Values[i])); break;
            case U64Array a:
                if (a.Values.Any(v => v > long.MaxValue))
                    throw new XGBoostException("Unsigned integers larger than INT64_MAX are not supported.");
                WriteArray(a.Count, i => VisitInteger((long)a.Values[i]));
                break;
        }
    }

    private void WriteArray(int n, Action<int> visit)
    {
        Put('[');
        for (var i = 0; i < n; i++)
        {
            visit(i);
            if (i != n - 1) Put(',');
        }
        Put(']');
    }

    protected virtual void VisitArray(JsonArray arr) => WriteArray(arr.Values.Count, i => Save(arr.Values[i]));

    protected virtual void VisitObject(JsonObject obj)
    {
        Put('{');
        var i = 0;
        var size = obj.Map.Count;
        foreach (var (k, v) in obj.Map)
        {
            VisitString(k);
            Put(':');
            Save(v);
            if (i != size - 1) Put(',');
            i++;
        }
        Put('}');
    }

    protected virtual void VisitNumber(float value) => PutAscii(CharConv.ToChars(value));

    protected virtual void VisitInteger(long value) => PutAscii(value.ToString(CultureInfo.InvariantCulture));

    protected virtual void VisitNull() => PutAscii("null");

    protected virtual void VisitString(string value)
    {
        Put('"');
        Stream.AddRange(Encoding.UTF8.GetBytes(StringUtils.EscapeU8(value)));
        Put('"');
    }

    protected virtual void VisitBoolean(bool value) => PutAscii(value ? "true" : "false");
}

/// <summary>Writer for UBJSON, byte-compatible with the C++ <c>UBJWriter</c>.</summary>
public sealed class UbjWriter : JsonWriter
{
    private void WritePrimitive<T>(T v) where T : unmanaged
    {
        var size = Unsafe.SizeOf<T>();
        Span<byte> tmp = stackalloc byte[size];
        Unsafe.WriteUnaligned(ref tmp[0], v);
        if (size > 1) tmp.Reverse();
        foreach (var b in tmp) Stream.Add(b);
    }

    private void EncodeStr(string s)
    {
        Put('L');
        var bytes = Encoding.UTF8.GetBytes(s);
        WritePrimitive((long)bytes.Length);
        Stream.AddRange(bytes);
    }

    protected override void VisitArray(JsonArray arr)
    {
        Put('[');
        Put('#');
        Put('L');
        WritePrimitive((long)arr.Values.Count);
        foreach (var v in arr.Values) Save(v);
    }

    private void WriteTyped<T>(char marker, T[] values) where T : unmanaged
    {
        Put('[');
        Put('$');
        Put(marker);
        Put('#');
        Put('L');
        WritePrimitive((long)values.Length);
        foreach (var v in values) WritePrimitive(v);
    }

    protected override void VisitTyped(JsonTypedArray arr)
    {
        switch (arr)
        {
            case F32Array a: WriteTyped('d', a.Values); break;
            case F64Array a: WriteTyped('D', a.Values); break;
            case I8Array a: WriteTyped('i', a.Values); break;
            case U8Array a: WriteTyped('U', a.Values); break;
            case I16Array a: WriteTyped('I', a.Values); break;
            case U16Array a: WriteTyped('I', a.Values); break;
            case I32Array a: WriteTyped('l', a.Values); break;
            case U32Array a: WriteTyped('l', a.Values); break;
            case I64Array a: WriteTyped('L', a.Values); break;
            case U64Array a: WriteTyped('L', a.Values); break;
        }
    }

    protected override void VisitObject(JsonObject obj)
    {
        Put('{');
        foreach (var (k, v) in obj.Map)
        {
            EncodeStr(k);
            Save(v);
        }
        Put('}');
    }

    protected override void VisitNumber(float value)
    {
        Put('d');
        WritePrimitive(value);
    }

    protected override void VisitInteger(long i)
    {
        if (i > sbyte.MinValue && i < sbyte.MaxValue)
        {
            Put('i');
            WritePrimitive((sbyte)i);
        }
        else if (i > short.MinValue && i < short.MaxValue)
        {
            Put('I');
            WritePrimitive((short)i);
        }
        else if (i > int.MinValue && i < int.MaxValue)
        {
            Put('l');
            WritePrimitive((int)i);
        }
        else
        {
            Put('L');
            WritePrimitive(i);
        }
    }

    protected override void VisitNull() => Put('Z');

    protected override void VisitString(string value)
    {
        Put('S');
        EncodeStr(value);
    }

    protected override void VisitBoolean(bool value) => Put(value ? 'T' : 'F');
}

/// <summary>Port of src/common/charconv.cc: shortest round-trip float formatting in Ryu's layout.</summary>
public static class CharConv
{
    /// <summary>Formats like the C++ <c>to_chars(float)</c>: <c>d.dddE[-]x</c>, <c>NaN</c>, <c>Infinity</c>, <c>0E0</c>.</summary>
    public static string ToChars(float f)
    {
        if (float.IsNaN(f)) return "NaN";
        var sign = float.IsNegative(f);
        if (float.IsInfinity(f)) return sign ? "-Infinity" : "Infinity";
        if (f == 0) return sign ? "-0E0" : "0E0";

        // "E8" is not shortest; "R" is. Extract digits and exponent from the shortest form.
        var s = Math.Abs(f).ToString("R", CultureInfo.InvariantCulture);
        var exp = 0;
        var e = s.IndexOfAny(['E', 'e']);
        if (e >= 0)
        {
            exp = int.Parse(s.AsSpan(e + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            s = s[..e];
        }
        var dot = s.IndexOf('.');
        string digits;
        if (dot >= 0)
        {
            digits = s[..dot] + s[(dot + 1)..];
            exp += dot - 1;
        }
        else
        {
            digits = s;
            exp += s.Length - 1;
        }
        // Strip leading zeros (e.g. "0.00123").
        var lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0')
        {
            lead++;
            exp--;
        }
        digits = digits[lead..].TrimEnd('0');
        if (digits.Length == 0) digits = "0";

        var sb = new StringBuilder(16);
        if (sign) sb.Append('-');
        sb.Append(digits[0]);
        if (digits.Length > 1) sb.Append('.').Append(digits, 1, digits.Length - 1);
        sb.Append('E').Append(exp.ToString(CultureInfo.InvariantCulture));
        return sb.ToString();
    }
}
