// Formatting and parsing helpers that reproduce the C++ standard streams, plus src/common/common.h.
using System.Globalization;
using System.Text;

namespace XGBoost.Common;

public static class Format
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    /// <summary>Equivalent of <c>printf("%.{precision}g", value)</c> as printed by the MSVC runtime.</summary>
    public static string G(double value, int precision = 6)
    {
        if (double.IsNaN(value)) return double.IsNegative(value) ? "-nan(ind)" : "nan";
        if (double.IsPositiveInfinity(value)) return "inf";
        if (double.IsNegativeInfinity(value)) return "-inf";
        if (precision == 0) precision = 1;
        if (value == 0) return double.IsNegative(value) ? "-0" : "0";

        // Round to `precision` significant digits in scientific form to learn the exponent X.
        var sci = value.ToString("E" + (precision - 1), Inv); // d.ddddE+xxx
        var ePos = sci.IndexOf('E');
        var exp = int.Parse(sci.AsSpan(ePos + 1), NumberStyles.AllowLeadingSign, Inv);
        string result;
        if (exp < -4 || exp >= precision)
        {
            var mantissa = sci[..ePos];
            if (mantissa.Contains('.')) mantissa = mantissa.TrimEnd('0').TrimEnd('.');
            var sign = exp < 0 ? '-' : '+';
            var abs = Math.Abs(exp);
            result = mantissa + "e" + sign + (abs < 10 ? "0" + abs : abs.ToString(Inv));
        }
        else
        {
            result = value.ToString("F" + Math.Max(0, precision - 1 - exp), Inv);
            if (result.Contains('.')) result = result.TrimEnd('0').TrimEnd('.');
        }
        return result;
    }

    /// <summary><c>std::ostream &lt;&lt; float</c> with default precision (6).</summary>
    public static string Stream(float value) => G(value);

    /// <summary><c>std::ostream &lt;&lt; double</c> with default precision (6).</summary>
    public static string Stream(double value) => G(value);

    /// <summary>dmlc parameter printing of float: <c>setprecision(max_digits10)</c>.</summary>
    public static string ParamFloat(float value) => G(value, 9);

    /// <summary>dmlc parameter printing of double: <c>setprecision(max_digits10)</c>.</summary>
    public static string ParamDouble(double value) => G(value, 17);

    /// <summary>Equivalent of <c>std::to_string(double)</c> (<c>%f</c>).</summary>
    public static string ToStringF(double value)
    {
        if (double.IsNaN(value)) return "nan";
        if (double.IsInfinity(value)) return value > 0 ? "inf" : "-inf";
        return value.ToString("F6", Inv);
    }

    public static string I(long v) => v.ToString(Inv);

    public static string I(ulong v) => v.ToString(Inv);

    /// <summary>
    /// Parses a float the way <c>dmlc::stof</c> does: optional leading whitespace, <c>inf</c>/<c>nan</c>
    /// case-insensitively, and returns the number of characters consumed.
    /// </summary>
    public static bool TryParseFloatPrefix(string s, out double value, out int consumed)
    {
        value = 0;
        consumed = 0;
        var i = 0;
        while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
        var start = i;
        var neg = false;
        if (i < s.Length && (s[i] == '+' || s[i] == '-'))
        {
            neg = s[i] == '-';
            i++;
        }

        bool Match(string word)
        {
            if (string.Compare(s, i, word, 0, word.Length, StringComparison.OrdinalIgnoreCase) != 0) return false;
            i += word.Length;
            return true;
        }

        if (Match("infinity") || Match("inf"))
        {
            value = neg ? double.NegativeInfinity : double.PositiveInfinity;
            consumed = i;
            return true;
        }
        if (Match("nan"))
        {
            // Optional (n-char-sequence).
            if (i < s.Length && s[i] == '(')
            {
                var close = s.IndexOf(')', i);
                if (close > 0) i = close + 1;
            }
            value = double.NaN;
            consumed = i;
            return true;
        }

        var digitsStart = i;
        var sawDigit = false;
        while (i < s.Length && char.IsAsciiDigit(s[i]))
        {
            i++;
            sawDigit = true;
        }
        if (i < s.Length && s[i] == '.')
        {
            i++;
            while (i < s.Length && char.IsAsciiDigit(s[i]))
            {
                i++;
                sawDigit = true;
            }
        }
        if (!sawDigit)
        {
            i = digitsStart;
            return false;
        }
        if (i < s.Length && (s[i] == 'e' || s[i] == 'E'))
        {
            var save = i;
            i++;
            if (i < s.Length && (s[i] == '+' || s[i] == '-')) i++;
            if (i < s.Length && char.IsAsciiDigit(s[i]))
            {
                while (i < s.Length && char.IsAsciiDigit(s[i])) i++;
            }
            else
            {
                i = save;
            }
        }
        value = double.Parse(s.AsSpan(start, i - start), NumberStyles.Float, Inv);
        consumed = i;
        return true;
    }

    /// <summary>Equivalent of <c>std::istringstream &gt;&gt; integer</c> followed by trailing-space check.</summary>
    public static bool TryParseIntegerStream(string s, out long value)
    {
        var t = s.Trim();
        return long.TryParse(t, NumberStyles.AllowLeadingSign, Inv, out value);
    }

    public static bool TryParseUnsignedStream(string s, out ulong value)
    {
        var t = s.Trim();
        if (t.StartsWith('-'))
        {
            // istream >> unsigned accepts negatives by wrapping.
            if (long.TryParse(t, NumberStyles.AllowLeadingSign, Inv, out var neg))
            {
                value = unchecked((ulong)neg);
                return true;
            }
            value = 0;
            return false;
        }
        return ulong.TryParse(t.TrimStart('+'), NumberStyles.None, Inv, out value);
    }
}

/// <summary>Port of string helpers in src/common/common.h/.cc.</summary>
public static class StringUtils
{
    /// <summary>Splits like <c>std::getline</c> in a loop: a trailing delimiter yields no empty item.</summary>
    public static List<string> Split(string s, char delim)
    {
        var ret = new List<string>();
        var start = 0;
        while (start < s.Length)
        {
            var idx = s.IndexOf(delim, start);
            if (idx < 0)
            {
                ret.Add(s[start..]);
                break;
            }
            ret.Add(s[start..idx]);
            start = idx + 1;
        }
        return ret;
    }

    public static string TrimFirst(string s) => s.TrimStart(' ', '\t', '\n', '\r');

    public static string TrimLast(string s) => s.TrimEnd(' ', '\t', '\n', '\r');

    public static string EscapeU8(string s)
    {
        var buffer = new StringBuilder(s.Length + 2);
        for (var i = 0; i < s.Length; i++)
        {
            var ch = s[i];
            switch (ch)
            {
                case '\\':
                    buffer.Append(i + 1 < s.Length && s[i + 1] == 'u' ? "\\" : "\\\\");
                    break;
                case '"': buffer.Append("\\\""); break;
                case '\b': buffer.Append("\\b"); break;
                case '\f': buffer.Append("\\f"); break;
                case '\n': buffer.Append("\\n"); break;
                case '\r': buffer.Append("\\r"); break;
                case '\t': buffer.Append("\\t"); break;
                default:
                    if (ch <= 0x1f) buffer.Append("\\u").Append(((int)ch).ToString("x4", CultureInfo.InvariantCulture));
                    else buffer.Append(ch);
                    break;
            }
        }
        return buffer.ToString();
    }

    public static string HumanMemUnit(ulong nBytes)
    {
        var n = (double)nBytes;
        foreach (var (power, unit) in new[] { (3, "GB"), (2, "MB"), (1, "KB") })
            if (n >= Math.Pow(1024, power)) return Format.Stream(n / Math.Pow(1024, power)) + unit;
        return Format.I(nBytes) + "B";
    }

    public static long DivRoundUp(long a, long b) => (long)Math.Ceiling((double)a / b);
}
