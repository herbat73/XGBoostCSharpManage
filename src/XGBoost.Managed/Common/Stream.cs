// Binary stream helpers equivalent to dmlc::Stream serialisation (little endian PODs, uint64
// length prefixed vectors and strings). Also src/common/version.h/.cc.
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace XGBoost.Common;

public sealed class DmlcWriter(Stream stream) : IDisposable
{
    private readonly BinaryWriter _w = new(stream, Encoding.UTF8, leaveOpen: true);

    public Stream BaseStream => _w.BaseStream;

    public void Write<T>(T value) where T : unmanaged
    {
        Span<byte> buf = stackalloc byte[Unsafe.SizeOf<T>()];
        MemoryMarshal.Write(buf, in value);
        _w.Write(buf);
    }

    public void WriteRaw(ReadOnlySpan<byte> bytes) => _w.Write(bytes);

    public void WriteVector<T>(ReadOnlySpan<T> values) where T : unmanaged
    {
        Write((ulong)values.Length);
        if (values.Length != 0) _w.Write(MemoryMarshal.AsBytes(values));
    }

    public void WriteString(string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        Write((ulong)bytes.Length);
        _w.Write(bytes);
    }

    public void WriteStrings(IReadOnlyList<string> values)
    {
        Write((ulong)values.Count);
        foreach (var s in values) WriteString(s);
    }

    public void Dispose() => _w.Dispose();
}

public sealed class DmlcReader(Stream stream) : IDisposable
{
    private readonly BinaryReader _r = new(stream, Encoding.UTF8, leaveOpen: true);

    public Stream BaseStream => _r.BaseStream;

    public bool TryRead<T>(out T value) where T : unmanaged
    {
        var size = Unsafe.SizeOf<T>();
        Span<byte> buf = stackalloc byte[size];
        var n = _r.Read(buf);
        if (n != size)
        {
            value = default;
            return false;
        }
        value = MemoryMarshal.Read<T>(buf);
        return true;
    }

    public T Read<T>() where T : unmanaged
    {
        if (!TryRead<T>(out var v)) throw new XGBoostException("Unexpected end of stream.");
        return v;
    }

    public byte[] ReadRaw(int n) => _r.ReadBytes(n);

    public T[] ReadVector<T>() where T : unmanaged
    {
        var n = checked((int)Read<ulong>());
        var result = new T[n];
        if (n != 0)
        {
            var bytes = MemoryMarshal.AsBytes(result.AsSpan());
            var read = _r.Read(bytes);
            if (read != bytes.Length) throw new XGBoostException("Unexpected end of stream.");
        }
        return result;
    }

    public string ReadString()
    {
        var n = checked((int)Read<ulong>());
        return Encoding.UTF8.GetString(_r.ReadBytes(n));
    }

    public List<string> ReadStrings()
    {
        var n = checked((int)Read<ulong>());
        var r = new List<string>(n);
        for (var i = 0; i < n; i++) r.Add(ReadString());
        return r;
    }

    public void Dispose() => _r.Dispose();
}

/// <summary>Port of src/common/version.h/.cc.</summary>
public static class VersionInfo
{
    public static readonly (int Major, int Minor, int Patch) Invalid = (-1, -1, -1);

    public static (int Major, int Minor, int Patch) Self() =>
        (Constants.VersionMajor, Constants.VersionMinor, Constants.VersionPatch);

    public static (int Major, int Minor, int Patch) Load(Json input)
    {
        var obj = input.AsObject;
        if (!obj.ContainsKey("version")) return Invalid;
        try
        {
            var arr = obj["version"].AsArray;
            return ((int)arr[0].AsInteger, (int)arr[1].AsInteger, (int)arr[2].AsInteger);
        }
        catch (XGBoostException)
        {
            throw new XGBoostException($"Invaid version format in loaded JSON object: {input}");
        }
    }

    public static void Save(JsonObject output)
    {
        var (major, minor, patch) = Self();
        output["version"] = new JsonArray([new JsonInteger(major), new JsonInteger(minor), new JsonInteger(patch)]);
    }

    public static (int Major, int Minor, int Patch) Load(DmlcReader fi)
    {
        const string msg = "Incorrect version format found in binary file.  Binary file from XGBoost < 1.0.0 is no longer "
            + "supported. Please generate it again.";
        var read = fi.ReadRaw(8);
        if (Encoding.ASCII.GetString(read) != "version:") Check.Fail(msg);
        return (fi.Read<int>(), fi.Read<int>(), fi.Read<int>());
    }

    public static void Save(DmlcWriter fo)
    {
        var (major, minor, patch) = Self();
        fo.WriteRaw("version:"u8);
        fo.Write(major);
        fo.Write(minor);
        fo.Write(patch);
    }

    public static string String((int Major, int Minor, int Patch) v) => $"{v.Major}.{v.Minor}.{v.Patch}";

    public static bool Same((int, int, int) triplet) => triplet == Self();
}
