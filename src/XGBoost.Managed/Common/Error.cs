// Port of include/xgboost/logging.h, src/logging.cc and src/common/error_msg.h.
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace XGBoost.Common;



/// <summary>Equivalents of the <c>CHECK_*</c> macros.</summary>
public static class Check
{
    [DoesNotReturn, MethodImpl(MethodImplOptions.NoInlining)]
    public static void Fail(string message) => throw new XGBoostException(message);

    [DoesNotReturn, MethodImpl(MethodImplOptions.NoInlining)]
    public static T Fail<T>(string message) => throw new XGBoostException(message);

    public static void That([DoesNotReturnIf(false)] bool condition, string message = "",
        [CallerArgumentExpression(nameof(condition))] string expr = "")
    {
        if (!condition) Fail($"Check failed: {expr}: {message}");
    }

    public static void That([DoesNotReturnIf(false)] bool condition, Func<string> message,
        [CallerArgumentExpression(nameof(condition))] string expr = "")
    {
        if (!condition) Fail($"Check failed: {expr}: {message()}");
    }

    public static void Eq<T>(T a, T b, string message = "",
        [CallerArgumentExpression(nameof(a))] string ea = "", [CallerArgumentExpression(nameof(b))] string eb = "")
    {
        if (!EqualityComparer<T>.Default.Equals(a, b)) Fail($"Check failed: {ea} == {eb} ({a} vs. {b}) {message}");
    }

    public static void Ne<T>(T a, T b, string message = "",
        [CallerArgumentExpression(nameof(a))] string ea = "", [CallerArgumentExpression(nameof(b))] string eb = "")
    {
        if (EqualityComparer<T>.Default.Equals(a, b)) Fail($"Check failed: {ea} != {eb} ({a} vs. {b}) {message}");
    }

    public static void Lt<T>(T a, T b, string message = "",
        [CallerArgumentExpression(nameof(a))] string ea = "", [CallerArgumentExpression(nameof(b))] string eb = "")
        where T : IComparable<T>
    {
        if (a.CompareTo(b) >= 0) Fail($"Check failed: {ea} < {eb} ({a} vs. {b}) {message}");
    }

    public static void Le<T>(T a, T b, string message = "",
        [CallerArgumentExpression(nameof(a))] string ea = "", [CallerArgumentExpression(nameof(b))] string eb = "")
        where T : IComparable<T>
    {
        if (a.CompareTo(b) > 0) Fail($"Check failed: {ea} <= {eb} ({a} vs. {b}) {message}");
    }

    public static void Gt<T>(T a, T b, string message = "",
        [CallerArgumentExpression(nameof(a))] string ea = "", [CallerArgumentExpression(nameof(b))] string eb = "")
        where T : IComparable<T>
    {
        if (a.CompareTo(b) <= 0) Fail($"Check failed: {ea} > {eb} ({a} vs. {b}) {message}");
    }

    public static void Ge<T>(T a, T b, string message = "",
        [CallerArgumentExpression(nameof(a))] string ea = "", [CallerArgumentExpression(nameof(b))] string eb = "")
        where T : IComparable<T>
    {
        if (a.CompareTo(b) < 0) Fail($"Check failed: {ea} >= {eb} ({a} vs. {b}) {message}");
    }
}

public enum LogVerbosity
{
    Silent = 0,
    Warning = 1,
    Info = 2,
    Debug = 3,
    Ignore = 4,
}

/// <summary>Port of the console logger. Messages go to <see cref="Sink"/> (stderr by default).</summary>
public static class Log
{
    /// <summary>Destination of log lines; replace to capture output.</summary>
    public static Action<string> Sink { get; set; } = line => System.Console.Error.WriteLine(line);

    public static LogVerbosity GlobalVerbosity => GlobalConfig.Current.Verbosity switch
    {
        0 => LogVerbosity.Silent,
        1 => LogVerbosity.Warning,
        2 => LogVerbosity.Info,
        3 => LogVerbosity.Debug,
        _ => LogVerbosity.Ignore,
    };

    private static bool ShouldLog(LogVerbosity v) => v <= GlobalVerbosity || GlobalVerbosity == LogVerbosity.Ignore;

    private static string Stamp() => $"[{DateTime.Now:HH:mm:ss}] ";

    public static void Warning(string message)
    {
        if (ShouldLog(LogVerbosity.Warning)) Sink($"{Stamp()}WARNING: {message}");
    }

    public static void Info(string message)
    {
        if (ShouldLog(LogVerbosity.Info)) Sink($"{Stamp()}{message}");
    }

    public static void Debug(string message)
    {
        if (ShouldLog(LogVerbosity.Debug)) Sink($"{Stamp()}DEBUG: {message}");
    }

    /// <summary>Output of <c>LOG(CONSOLE)</c>: printed regardless of verbosity (training messages).</summary>
    public static void Console(string message) => Sink(message);

    private static readonly HashSet<string> Once = [];

    public static void WarningOnce(string key, string message)
    {
        lock (Once)
            if (!Once.Add(key)) return;
        Warning(message);
    }
}

/// <summary>Port of src/common/error_msg.h/.cc.</summary>
public static class ErrorMsg
{
    public const string MTNotImplemented = " support for multi-target tree is not yet implemented.";
    public const string GroupWeight = "Size of weight must equal to the number of query groups when ranking group is used.";
    public const string GroupSize = "Invalid query group structure. The number of rows obtained from group doesn't equal to ";
    public const string LabelScoreSize = "The size of label doesn't match the size of prediction.";
    public const string InfInData = "Input data contains `inf` or a value too large, while `missing` is not set to `inf`";
    public const string NoF128 = "128-bit floating point is not supported on current platform.";
    public const string InconsistentMaxBin = "Inconsistent `max_bin`. `max_bin` should be the same across different QuantileDMatrix, and consistent with the Booster being trained.";
    public const string InvalidMaxBin = "`max_bin` must be equal to or greater than 2.";
    public const string UnknownDevice = "Unknown device type.";
    public const string InplacePredictProxy = "Inplace predict accepts only DMatrixProxy as input.";
    public const string InvalidCudaOrdinal = "Invalid device. `device` is required to be CUDA and there must be at least one GPU available for using GPU.";
    public const string InconsistentFeatureTypes = "Inconsistent feature types between batches.";
    public const string InconsistentCategories = "Inconsistent number of categories between batches.";
    public const string NoFloatCat = "Category index from DataFrame has floating point dtype, consider using strings or integers instead.";
    public const string CacheHostRatioNotImpl = "`cache_host_ratio` is only used by the GPU `ExtMemQuantileDMatrix`.";
    public const string CacheHostRatioInvalid = "`cache_host_ratio` must be in range [0, 1].";
    public const string NoCuda = "XGBoost.Managed is a CPU-only port; CUDA devices are not supported.";

    public const string OldSerialization =
        "If you are loading a serialized model (like pickle in Python, RDS in R) or\n"
        + "configuration generated by an older version of XGBoost, please export the model by calling\n"
        + "`Booster.save_model` from that version first, then load it back in current version. See:\n"
        + "    https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html\n"
        + "for more details about differences between saving model and serializing.\n";

    public static void MaxFeatureSize(ulong nFeatures)
    {
        if (nFeatures > uint.MaxValue)
            Check.Fail($"Unfortunately, XGBoost does not support data matrices with {uint.MaxValue} features or greater");
    }

    public static void MaxSampleSize(ulong n) =>
        Check.Fail($"Sample size too large for the current updater. Maximum number of samples:{n}. Consider using a different updater or tree_method.");

    public static string NoCategorical(string name) => name + " doesn't support categorical features.";

    public static void WarnOldSerialization() => Log.WarningOnce("old-serialization", OldSerialization);

    public static string InvalidModel(string fname) => $"Invalid model format in: `{fname}`.";

    public static string OldBinaryModel(string fname) =>
        $"Failed to load model: `{fname}`. \n"
        + "The binary format has been deprecated in 1.6 and removed in 3.1, use UBJ or JSON\n"
        + "instead. You can port the binary model to UBJ and JSON by re-saving it with XGBoost\n"
        + "3.0. See:\n    https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html\nfor more info.\n";

    public static void WarnManualUpdater() => Log.WarningOnce("manual-updater",
        "You have manually specified the `updater` parameter. The `tree_method` parameter will be ignored. "
        + "Incorrect sequence of updaters will produce undefined behavior. For common uses, we recommend using "
        + "`tree_method` parameter instead.");

    public static void WarnEmptyDataset() => Log.WarningOnce("empty-dataset", "Empty dataset at worker: 0");

    public static string DeprecatedFunc(string old, string since, string replacement) =>
        $"`{old}` is deprecated since {since}, use `{replacement}` instead.";

    public static void InvalidIntercept(int nClasses, uint nTargets, int interceptLen)
    {
        var msg = "Invalid `base_score`, it should match the number of outputs for multi-class/target "
            + $"models. `base_score` len: {interceptLen}";
        if (nClasses > 1) msg += $", `n_classes`: {nClasses}";
        if (nTargets > 1) msg += $", `n_targets`: {nTargets}";
        Check.Fail(msg);
    }

    [DoesNotReturn]
    public static void Unreachable() => Check.Fail("Unreachable");
}
