// Port of the collective API surface (src/collective/*.h, communicator-inl.h) used by the rest of
// the library. The in-process communicator has a world size of one; reductions are identities.
using XGBoost.Common;

namespace XGBoost.Collective;

public enum Op
{
    Max = 0,
    Min = 1,
    Sum = 2,
    BitwiseAnd = 3,
    BitwiseOr = 4,
    BitwiseXor = 5,
}

/// <summary>Communicator access, <c>collective::GetRank</c>, <c>Allreduce</c> and friends.</summary>
public static class Communicator
{
    private static ICommunicator _impl = new InProcessCommunicator();

    /// <summary>Replaces the active communicator (e.g. a TCP ring set up by <c>CommunicatorContext</c>).</summary>
    public static void Init(ICommunicator comm) => _impl = comm;

    public static void Shutdown()
    {
        _impl.Dispose();
        _impl = new InProcessCommunicator();
    }

    public static ICommunicator Current => _impl;

    public static int GetRank() => _impl.Rank;
    public static int GetWorldSize() => _impl.WorldSize;
    public static bool IsDistributed() => _impl.WorldSize > 1;
    public static bool IsFederated() => false;
    public static string GetProcessorName() => _impl.ProcessorName;

    public static void Print(string message) => Log.Console(message);

    public static void Allreduce(Span<double> data, Op op) => _impl.Allreduce(data, op);
    public static void Allreduce(Span<float> data, Op op) => _impl.Allreduce(data, op);
    public static void Allreduce(Span<long> data, Op op) => _impl.Allreduce(data, op);
    public static void Allreduce(Span<int> data, Op op) => _impl.Allreduce(data, op);
    public static void Allreduce(Span<ulong> data, Op op) => _impl.Allreduce(data, op);
    public static void Allreduce(Span<uint> data, Op op) => _impl.Allreduce(data, op);
    public static void Allreduce(Span<byte> data, Op op) => _impl.Allreduce(data, op);

    public static void AllreduceMax(ref long value)
    {
        Span<long> v = [value];
        Allreduce(v, Op.Max);
        value = v[0];
    }

    public static void Allreduce(Span<GradientPairPrecise> data)
    {
        var d = System.Runtime.InteropServices.MemoryMarshal.Cast<GradientPairPrecise, double>(data);
        Allreduce(d, Op.Sum);
    }

    public static void Broadcast(Span<byte> data, int root) => _impl.Broadcast(data, root);

    public static string BroadcastString(string s, int root)
    {
        if (!IsDistributed()) return s;
        return _impl.BroadcastString(s, root);
    }

    /// <summary>Gathers variable length byte blobs from all workers.</summary>
    public static List<byte[]> AllgatherV(byte[] data) => _impl.AllgatherV(data);

    public static List<string> AllgatherStrings(IReadOnlyList<string> input)
    {
        if (!IsDistributed()) return [.. input];
        var blob = System.Text.Encoding.UTF8.GetBytes(string.Join('\0', input));
        var all = AllgatherV(blob);
        var result = new List<string>();
        foreach (var b in all)
        {
            if (b.Length == 0) continue;
            result.AddRange(System.Text.Encoding.UTF8.GetString(b).Split('\0'));
        }
        return result;
    }
}

/// <summary>A communicator implementation.</summary>
public interface ICommunicator : IDisposable
{
    int Rank { get; }
    int WorldSize { get; }
    string ProcessorName { get; }
    void Allreduce<T>(Span<T> data, Op op) where T : unmanaged, System.Numerics.INumber<T>;
    void Broadcast(Span<byte> data, int root);
    string BroadcastString(string s, int root);
    List<byte[]> AllgatherV(byte[] data);
}

/// <summary>Single worker: all collective operations are identities.</summary>
public sealed class InProcessCommunicator : ICommunicator
{
    public int Rank => 0;
    public int WorldSize => 1;
    public string ProcessorName => Environment.MachineName;
    public void Allreduce<T>(Span<T> data, Op op) where T : unmanaged, System.Numerics.INumber<T> { }
    public void Broadcast(Span<byte> data, int root) { }
    public string BroadcastString(string s, int root) => s;
    public List<byte[]> AllgatherV(byte[] data) => [data];
    public void Dispose() { }
}
