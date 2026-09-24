// Port of include/xgboost/context.h, src/context.cc, include/xgboost/global_config.h and src/global_config.cc.
using System.Text.RegularExpressions;
using XGBoost.Common;

namespace XGBoost;

public enum DeviceType : short
{
    Cpu = 0,
    Cuda = 1,
    SyclDefault = 2,
    SyclCpu = 3,
    SyclGpu = 4,
}

/// <summary>Device and ordinal, <c>DeviceOrd</c>. Only the CPU is usable in this port.</summary>
public readonly record struct DeviceOrd(DeviceType Device, short Ordinal)
{
    public const short CpuOrdinal = -1;
    public const short InvalidOrdinal = -2;

    public static DeviceOrd Cpu() => new(DeviceType.Cpu, CpuOrdinal);
    public static DeviceOrd Cuda(short ordinal) => new(DeviceType.Cuda, ordinal);

    public bool IsCpu => Device == DeviceType.Cpu;
    public bool IsCuda => Device == DeviceType.Cuda;
    public bool IsSycl => Device is DeviceType.SyclDefault or DeviceType.SyclCpu or DeviceType.SyclGpu;

    public string Name() => Device switch
    {
        DeviceType.Cpu => "cpu",
        DeviceType.Cuda => $"cuda:{Ordinal}",
        DeviceType.SyclDefault => $"sycl:{Ordinal}",
        DeviceType.SyclCpu => $"sycl:cpu:{Ordinal}",
        DeviceType.SyclGpu => $"sycl:gpu:{Ordinal}",
        _ => throw new XGBoostException("Unknown device."),
    };

    public override string ToString() => Name();
}

/// <summary>Runtime parameters shared by the booster components, <c>xgboost::Context</c>.</summary>
public sealed partial class Context : XGBoostParameter<Context>
{
    public const long DefaultSeed = 0;

    private string _device = "cpu";
    private DeviceOrd _deviceOrd = DeviceOrd.Cpu();

    public int NThread;
    public long Seed = DefaultSeed;
    public bool SeedPerIteration;
    public bool FailOnInvalidGpuId;
    public bool ValidateParameters;

    /// <summary>The random engine, <c>std::mt19937</c>.</summary>
    public Mt19937 Rng { get; private set; } = new();

    protected override void Declare(ParamManager<Context> m)
    {
        Field(m, "seed", p => p.Seed, (p, v) => p.Seed = v).SetDefault(DefaultSeed).Describe("Random number seed during training.");
        Alias(m, "seed", "random_state");
        Field(m, "seed_per_iteration", p => p.SeedPerIteration, (p, v) => p.SeedPerIteration = v).SetDefault(false)
            .Describe("Seed PRNG determnisticly via iterator number.");
        Field(m, "device", p => p._device, (p, v) => p._device = v).SetDefault("cpu").Describe("Device ordinal.");
        Field(m, "nthread", p => p.NThread, (p, v) => p.NThread = v).SetDefault(0).Describe("Number of threads to use.");
        Alias(m, "nthread", "n_jobs");
        Field(m, "fail_on_invalid_gpu_id", p => p.FailOnInvalidGpuId, (p, v) => p.FailOnInvalidGpuId = v).SetDefault(false)
            .Describe("Fail with error when gpu_id is invalid.");
        Field(m, "validate_parameters", p => p.ValidateParameters, (p, v) => p.ValidateParameters = v).SetDefault(false)
            .Describe("Enable checking whether parameters are used or not.");
    }

    public DeviceOrd Device => _deviceOrd;
    public bool IsCpu => _deviceOrd.IsCpu;
    public bool IsCuda => _deviceOrd.IsCuda;
    public bool IsSycl => _deviceOrd.IsSycl;
    public short Ordinal => _deviceOrd.Ordinal;
    public string DeviceName => _deviceOrd.Name();
    public DeviceOrd DeviceFP64 => _deviceOrd;

    /// <summary>Number of threads to use, <c>Context::Threads</c>.</summary>
    public int Threads() => Threading.OmpGetNumThreads(NThread);

    public void Init(List<KeyValuePair<string, string>> kwargs)
    {
        var unknown = UpdateAllowUnknown(kwargs);
        if (unknown.Count != 0)
            Check.Fail($"[Internal Error] Unknown parameters passed to the Context {{{string.Join(", ", unknown.Select(u => $"\"{u.Key}\""))}}}\n");
    }

    public override List<KeyValuePair<string, string>> UpdateAllowUnknown(IEnumerable<KeyValuePair<string, string>> kwargs)
    {
        var list = kwargs as IList<KeyValuePair<string, string>> ?? kwargs.ToList();
        var args = base.UpdateAllowUnknown(list);
        SetDeviceOrdinal(list);
        return args;
    }

    private void SetDeviceOrdinal(IList<KeyValuePair<string, string>> kwargs)
    {
        if (kwargs.Any(p => p.Key == "gpu_id")) Check.Fail("`gpu_id` has been removed since 3.1. Use `device` instead.");
        var hasDevice = kwargs.Any(p => p.Key == "device");
        var newD = MakeDeviceOrd(_device, FailOnInvalidGpuId);
        if (!hasDevice) Check.Eq(newD.Ordinal, _deviceOrd.Ordinal);
        SetDevice(newD);
        if (IsCpu) Check.Eq(_deviceOrd.Ordinal, DeviceOrd.CpuOrdinal);
    }

    private Context SetDevice(DeviceOrd d)
    {
        _deviceOrd = d;
        _device = d.Name();
        return this;
    }

    [GeneratedRegex("^(gpu(:[0-9]+)?|cuda(:[0-9]+)?|cpu|sycl(:cpu|:gpu)?(:-1|:[0-9]+)?)$")]
    private static partial Regex DevicePattern();

    private static DeviceOrd MakeDeviceOrd(string input, bool failOnInvalid)
    {
        _ = failOnInvalid;
        if (input == "cpu") return DeviceOrd.Cpu();
        const string msg = "Invalid argument for `device`. Expected to be one of the following:\n- cpu\n- cuda\n"
            + "- cuda:<device ordinal>  # e.g. cuda:0\n- gpu\n- gpu:<device ordinal>   # e.g. gpu:0\n";
        if (!DevicePattern().IsMatch(input)) Check.Fail($"{msg}Got: `{input}`.");
        // This port has no GPU support: like the C++ library on a machine without GPUs, fall back to CPU.
        Log.Warning("Device is changed from GPU to CPU as we couldn't find any available GPU on the system.");
        return DeviceOrd.Cpu();
    }

    public Context MakeCpu()
    {
        var ctx = Clone();
        ctx.SetDevice(DeviceOrd.Cpu());
        return ctx;
    }

    /// <summary>Copies parameters and RNG state.</summary>
    public Context Clone()
    {
        var c = new Context
        {
            _device = _device,
            _deviceOrd = _deviceOrd,
            NThread = NThread,
            Seed = Seed,
            SeedPerIteration = SeedPerIteration,
            FailOnInvalidGpuId = FailOnInvalidGpuId,
            ValidateParameters = ValidateParameters,
            Initialised = Initialised,
        };
        c.Rng.Load(Rng.Save());
        return c;
    }

    public JsonObject SaveJson()
    {
        var obj = ToJson();
        obj["rng_state"] = new JsonString(Rng.Save());
        return obj;
    }

    public void LoadJson(Json input)
    {
        var obj = input.AsObject;
        var args = new List<KeyValuePair<string, string>>();
        foreach (var (k, v) in obj)
        {
            if (k == "rng_state") continue;
            args.Add(new(k, v.AsString));
        }
        // FromJson passes every key including rng_state; unknown keys are ignored by UpdateAllowUnknown.
        UpdateAllowUnknown(args);
        if (obj.TryGetValue("rng_state", out var state)) Rng.Load(state.AsString);
    }
}

/// <summary>Process-wide configuration, <c>GlobalConfiguration</c> (thread local in C++).</summary>
public sealed class GlobalConfig : XGBoostParameter<GlobalConfig>
{
    public int Verbosity = 1;
    public bool UseRmm;
    public bool UseCudaAsyncPool;
    public int NThread;

    private static readonly AsyncLocal<GlobalConfig?> Local = new();
    private static readonly GlobalConfig Default = CreateDefault();

    private static GlobalConfig CreateDefault()
    {
        var c = new GlobalConfig();
        c.Init([]);
        return c;
    }

    protected override void Declare(ParamManager<GlobalConfig> m)
    {
        Field(m, "verbosity", p => p.Verbosity, (p, v) => p.Verbosity = v).SetRange(0, 3).SetDefault(1)
            .Describe("Flag to print out detailed breakdown of runtime.");
        Field(m, "use_rmm", p => p.UseRmm, (p, v) => p.UseRmm = v).SetDefault(false)
            .Describe("Whether to use RAPIDS Memory Manager to allocate GPU memory in XGBoost");
        Field(m, "use_cuda_async_pool", p => p.UseCudaAsyncPool, (p, v) => p.UseCudaAsyncPool = v).SetDefault(false)
            .Describe("Whether to use the async memory pool in CUDA.");
    }

    /// <summary>The configuration of the calling thread (flows into async continuations).</summary>
    public static GlobalConfig Current => Local.Value ?? Default;

    internal static void SetThreadConfig(GlobalConfig config) => Local.Value = config;

    /// <summary><c>ConsoleLogger::Configure</c>: updates the thread configuration, returns the used keys.</summary>
    public static SortedSet<string> ConfigureLogger(IReadOnlyList<KeyValuePair<string, string>> args)
    {
        var c = Current.Copy();
        var unknown = c.UpdateAllowUnknown(args);
        Local.Value = c;
        return ParameterUtils.GetUsedParameters(args, unknown);
    }

    /// <summary>Updates the configuration of the calling thread, like <c>XGBSetGlobalConfig</c>.</summary>
    public static void Set(IEnumerable<KeyValuePair<string, string>> kwargs)
    {
        var c = Current.Copy();
        var list = kwargs.ToList();
        var unknown = c.UpdateAllowUnknown(list);
        foreach (var (k, v) in unknown)
        {
            if (k == "nthread")
            {
                c.NThread = int.Parse(v, System.Globalization.CultureInfo.InvariantCulture);
                continue;
            }
            Check.Fail($"Unknown global parameter: {k}");
        }
        Local.Value = c;
    }

    private GlobalConfig Copy() => new()
    {
        Verbosity = Verbosity,
        UseRmm = UseRmm,
        UseCudaAsyncPool = UseCudaAsyncPool,
        NThread = NThread,
        Initialised = Initialised,
    };
}
