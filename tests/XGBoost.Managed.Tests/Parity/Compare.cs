using System.Globalization;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace XGBoost.Tests.Parity;

/// <summary>
/// Numeric comparison helpers. Both bindings call the same native library, so results are
/// expected to be bit-identical; the tiny tolerance only absorbs float formatting round-trips.
/// </summary>
internal static partial class Compare
{
    private const double RelTol = 1e-6;
    private const double AbsTol = 1e-7;

    public static bool Close(double actual, double expected) =>
        (double.IsNaN(actual) && double.IsNaN(expected))
        || actual == expected
        || Math.Abs(actual - expected) <= AbsTol + RelTol * Math.Abs(expected);

    public static void Values(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.True(expected.Length == actual.Length, $"{what}: expected {expected.Length} values, got {actual.Length}.");
        for (var i = 0; i < expected.Length; i++)
        {
            if (!Close(actual[i], expected[i]))
                Assert.Fail($"{what}: value [{i}] differs, expected {expected[i]:R}, got {actual[i]:R}.");
        }
    }

    public static void Values(IReadOnlyList<double> expected, IReadOnlyList<double> actual, string what)
    {
        Assert.True(expected.Count == actual.Count, $"{what}: expected {expected.Count} values, got {actual.Count}.");
        for (var i = 0; i < expected.Count; i++)
        {
            if (!Close(actual[i], expected[i]))
                Assert.Fail($"{what}: value [{i}] differs, expected {expected[i]:R}, got {actual[i]:R}.");
        }
    }

    /// <summary>Compares text token by token, treating numbers numerically.</summary>
    public static void Text(string expected, string actual, string what)
    {
        var e = Tokens().Split(expected);
        var a = Tokens().Split(actual);
        Assert.True(e.Length == a.Length, $"{what}: token count differs.\nexpected: {expected}\nactual:   {actual}");
        for (var i = 0; i < e.Length; i++)
        {
            if (e[i] == a[i]) continue;
            if (double.TryParse(e[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var ev)
                && double.TryParse(a[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var av)
                && Close(av, ev))
                continue;
            Assert.Fail($"{what}: '{e[i]}' != '{a[i]}' at token {i}.");
        }
    }

    // Splits around numbers, keeping them as separate tokens.
    [GeneratedRegex(@"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")]
    private static partial Regex Tokens();

    /// <summary>Deep-compares two JSON documents; <paramref name="skip"/> lists paths to ignore.</summary>
    public static void Json(JsonNode? expected, JsonNode? actual, string path, IReadOnlySet<string> skip)
    {
        if (skip.Contains(path)) return;
        switch (expected)
        {
            case null:
                Assert.True(actual is null, $"{path}: expected null.");
                return;
            case JsonObject eo:
                var ao = Assert.IsType<JsonObject>(actual, exactMatch: false);
                var keys = eo.Select(p => p.Key).Where(k => !skip.Contains($"{path}.{k}")).Order().ToArray();
                var actualKeys = ao.Select(p => p.Key).Where(k => !skip.Contains($"{path}.{k}")).Order().ToArray();
                Assert.True(keys.SequenceEqual(actualKeys),
                    $"{path}: keys differ.\nexpected: {string.Join(",", keys)}\nactual:   {string.Join(",", actualKeys)}");
                foreach (var k in keys) Json(eo[k], ao[k], $"{path}.{k}", skip);
                return;
            case JsonArray ea:
                var aa = Assert.IsType<JsonArray>(actual, exactMatch: false);
                Assert.True(ea.Count == aa.Count, $"{path}: expected {ea.Count} elements, got {aa.Count}.");
                for (var i = 0; i < ea.Count; i++) Json(ea[i], aa[i], $"{path}[{i}]", skip);
                return;
            case JsonValue ev:
                var av = Assert.IsType<JsonValue>(actual, exactMatch: false);
                if (ev.TryGetValue<double>(out var ed) && av.TryGetValue<double>(out var ad))
                {
                    if (!Close(ad, ed)) Assert.Fail($"{path}: expected {ed:R}, got {ad:R}.");
                }
                else
                {
                    Assert.True(ev.ToJsonString() == av.ToJsonString(),
                        $"{path}: expected {ev.ToJsonString()}, got {av.ToJsonString()}.");
                }
                return;
        }
    }
}
