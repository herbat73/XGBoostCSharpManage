using System.Globalization;
using System.Text;

namespace XGBoost.Demos.Common;

/// <summary>Replacements for the Python demos' <c>assert</c> and <c>np.testing</c> checks.</summary>
internal static class Check
{
    public static void That(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException("Check failed: " + message);
    }

    /// <summary>Like <c>np.testing.assert_allclose</c>: |a - b| &lt;= atol + rtol * |b| element-wise.</summary>
    public static void AllClose(IReadOnlyList<float> actual, IReadOnlyList<float> expected, double rtol = 1e-7,
        double atol = 0, string what = "values") =>
        AllClose(actual.Select(v => (double)v).ToArray(), expected.Select(v => (double)v).ToArray(), rtol, atol, what);

    public static void AllClose(IReadOnlyList<double> actual, IReadOnlyList<double> expected, double rtol = 1e-7,
        double atol = 0, string what = "values")
    {
        That(actual.Count == expected.Count, $"{what}: length {actual.Count} != {expected.Count}");
        for (var i = 0; i < actual.Count; i++)
        {
            var diff = Math.Abs(actual[i] - expected[i]);
            That(diff <= atol + rtol * Math.Abs(expected[i]),
                $"{what}: element {i} differs ({actual[i]} vs {expected[i]})");
        }
    }

    public static bool IsClose(IReadOnlyList<float> a, IReadOnlyList<float> b, double rtol = 1e-5, double atol = 1e-8)
    {
        if (a.Count != b.Count) return false;
        for (var i = 0; i < a.Count; i++)
            if (Math.Abs(a[i] - b[i]) > atol + rtol * Math.Abs(b[i])) return false;
        return true;
    }
}

/// <summary>Python-like formatting of arrays and tables for console output.</summary>
internal static class Fmt
{
    public static string Num(double v) => v.ToString("G6", CultureInfo.InvariantCulture);

    /// <summary>Values at full (round-trip) precision, like Python prints a list of floats.</summary>
    public static string List(IEnumerable<double> values) =>
        "[" + string.Join(", ", values.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]";

    public static string List(IEnumerable<float> values) =>
        "[" + string.Join(", ", values.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]";

    public static string Shape(IEnumerable<long> shape) => "(" + string.Join(", ", shape) + (shape.Count() == 1 ? ",)" : ")");

    /// <summary>A 2-D array, eliding middle rows like NumPy when there are many.</summary>
    public static string Matrix(float[] values, long rows, long cols, int edge = 3)
    {
        var sb = new StringBuilder("[");
        for (long r = 0; r < rows; r++)
        {
            if (rows > 2 * edge && r == edge)
            {
                sb.Append(" ...\n");
                r = rows - edge - 1;
                continue;
            }
            if (r > 0) sb.Append(' ');
            sb.Append(List(values.AsSpan((int)(r * cols), (int)cols).ToArray()));
            sb.Append(r == rows - 1 ? "]" : "\n");
        }
        return sb.ToString();
    }

    /// <summary>Columns of equal length as a table with a round index, like a pandas DataFrame.</summary>
    public static string Table(IEnumerable<KeyValuePair<string, List<double>>> columns)
    {
        var cols = columns.ToList();
        var rows = cols.Count == 0 ? 0 : cols[0].Value.Count;
        var widths = cols.Select(c => Math.Max(c.Key.Length, 10)).ToArray();
        var sb = new StringBuilder("    ");
        for (var c = 0; c < cols.Count; c++) sb.Append("  ").Append(cols[c].Key.PadLeft(widths[c]));
        for (var r = 0; r < rows; r++)
        {
            sb.Append('\n').Append(r.ToString(CultureInfo.InvariantCulture).PadRight(4));
            for (var c = 0; c < cols.Count; c++)
                sb.Append("  ").Append(cols[c].Value[r].ToString("F6", CultureInfo.InvariantCulture).PadLeft(widths[c]));
        }
        return sb.ToString();
    }
}

/// <summary>Minimal <c>--name value</c> / <c>--name=value</c> argument parsing.</summary>
internal sealed class DemoArgs(string[] args)
{
    public string? Get(string name)
    {
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] == name && i + 1 < args.Length) return args[i + 1];
            if (args[i].StartsWith(name + "=", StringComparison.Ordinal)) return args[i][(name.Length + 1)..];
        }
        return null;
    }

    public string Get(string name, string defaultValue) => Get(name) ?? defaultValue;
}

/// <summary>Shorthand for building parameter lists; keys may repeat (e.g. <c>eval_metric</c>).</summary>
internal static class P
{
    public static List<KeyValuePair<string, string>> Of(params (string Key, object Value)[] items) =>
        [.. items.Select(i => KeyValuePair.Create(i.Key, ToParam(i.Value)))];

    /// <summary>Formats a value the way the Python package does (arrays as <c>[a, b]</c>, bools as 1/0).</summary>
    public static string ToParam(object value) => value switch
    {
        string s => s,
        bool b => b ? "1" : "0",
        double[] a => "[" + string.Join(", ", a.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? "",
    };
}
