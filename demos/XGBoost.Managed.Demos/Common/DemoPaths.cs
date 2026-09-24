namespace XGBoost.Demos.Common;

/// <summary>Locates the repository's <c>demo/data</c> directory and a scratch directory for demo outputs.</summary>
internal static class DemoPaths
{
    private static readonly Lazy<string> DataDirectory = new(FindDataDirectory);

    /// <summary>The repository's <c>demo/data</c> directory.</summary>
    public static string DataDir => DataDirectory.Value;

    /// <summary>Path of a file in <c>demo/data</c>.</summary>
    public static string Data(string file) => Path.Combine(DataDir, file);

    /// <summary>URI of a libsvm file in <c>demo/data</c>, for <see cref="DMatrix.FromFile"/>.</summary>
    public static string LibSvm(string file) => Data(file) + "?format=libsvm";

    /// <summary>
    /// Directory the demos write models, dumps and matrices into. The Python demos use the working
    /// directory; here a fixed temporary directory keeps the source tree clean.
    /// </summary>
    public static string OutputDir
    {
        get
        {
            var dir = Path.Combine(Path.GetTempPath(), "xgboost-csharp-demos");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    /// <summary>Path of a file in <see cref="OutputDir"/>.</summary>
    public static string Output(string file) => Path.Combine(OutputDir, file);

    private static string FindDataDirectory()
    {
        foreach (var start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, "demos", "data");
                if (File.Exists(Path.Combine(candidate, "agaricus.txt.train"))) return candidate;
            }
        }
        throw new DirectoryNotFoundException(
            "Could not find demo/data; run the demos from inside the XGBoost repository.");
    }
}
