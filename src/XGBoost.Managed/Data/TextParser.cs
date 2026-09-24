// Port of the text input path: src/data/file_iterator.cc (URI validation) and the dmlc-core LIBSVM
// and CSV parsers (dmlc-core/src/data/libsvm_parser.h, csv_parser.h).
using System.Globalization;
using XGBoost.Common;

namespace XGBoost.Data;

public static class TextParser
{
    /// <summary><c>ValidateFileFormat</c>: returns (path, args) for <c>path?format=...&amp;k=v</c>.</summary>
    public static (string Path, Dictionary<string, string> Args) ValidateFileFormat(string uri)
    {
        var cache = StringUtils.Split(uri, '#');
        Check.Le(cache.Count, 2, "Only one `#` is allowed in file path for cachefile specification");
        var nameArgs = StringUtils.Split(cache[0], '?');
        const string msg = "URI parameter `format` is required for loading text data: filename?format=csv";
        Check.Eq(nameArgs.Count, 2, msg);
        var args = new Dictionary<string, string>();
        var argList = StringUtils.Split(nameArgs[1], '&');
        for (var i = 0; i < argList.Count; i++)
        {
            var eq = argList[i].IndexOf('=');
            Check.That(eq > 0, $"Invalid uri argument format for key in arg {i + 1}");
            args.TryAdd(argList[i][..eq], argList[i][(eq + 1)..]);
        }
        if (!args.ContainsKey("format")) Check.Fail(msg);
        return (Path.GetFullPath(nameArgs[0]), args);
    }

    public static List<RowBlock> Parse(string uri)
    {
        var (path, args) = ValidateFileFormat(uri);
        var text = File.ReadAllText(path);
        var format = args["format"];
        var block = format switch
        {
            "libsvm" or "auto" => ParseLibSvm(text, args.TryGetValue("indexing_mode", out var im) ? int.Parse(im, CultureInfo.InvariantCulture) : 0),
            "csv" => ParseCsv(text,
                args.TryGetValue("label_column", out var lc) ? int.Parse(lc, CultureInfo.InvariantCulture) : -1,
                args.TryGetValue("weight_column", out var wc) ? int.Parse(wc, CultureInfo.InvariantCulture) : -1,
                args.TryGetValue("delimiter", out var d) ? d : ","),
            _ => throw new XGBoostException($"Unknown data type {format}"),
        };
        return [block];
    }

    private static bool IsDigitChar(char c) => char.IsAsciiDigit(c) || c is '+' or '-' or '.' or 'e' or 'E';

    /// <summary>Parses a number with the C <c>strtod</c> rules at <paramref name="p"/>; returns false if none.</summary>
    private static bool ParseNumber(string s, ref int p, int end, out double value)
    {
        var sub = s[p..end];
        if (!Format.TryParseFloatPrefix(sub, out value, out var consumed) || consumed == 0) return false;
        p += consumed;
        return true;
    }

    public static RowBlock ParseLibSvm(string text, int indexingMode)
    {
        var offset = new List<long> { 0 };
        var labels = new List<float>();
        var weights = new List<float>();
        var qids = new List<ulong>();
        var index = new List<uint>();
        var values = new List<float>();
        var minFeatId = uint.MaxValue;
        var first = true;

        var lines = text.Split('\n', '\r');
        foreach (var raw in lines)
        {
            var line = raw;
            var hash = line.IndexOf('#');
            if (hash >= 0) line = line[..hash];
            var p = 0;
            while (p < line.Length && char.IsWhiteSpace(line[p])) p++;
            if (p == line.Length) continue;
            // label[:weight]
            if (!ParseNumber(line, ref p, line.Length, out var label)) continue;
            if (p < line.Length && line[p] == ':')
            {
                p++;
                if (ParseNumber(line, ref p, line.Length, out var w)) weights.Add((float)w);
            }
            if (!first) offset.Add(index.Count);
            first = false;
            labels.Add((float)label);
            while (p < line.Length && line[p] == ' ') p++;
            if (string.CompareOrdinal(line, p, "qid:", 0, 4) == 0)
            {
                p += 4;
                var start = p;
                while (p < line.Length && IsDigitChar(line[p])) p++;
                qids.Add(ulong.Parse(line.AsSpan(start, p - start), CultureInfo.InvariantCulture));
            }
            while (p < line.Length)
            {
                while (p < line.Length && char.IsWhiteSpace(line[p])) p++;
                if (p >= line.Length) break;
                var save = p;
                if (!ParseNumber(line, ref p, line.Length, out var fid))
                {
                    p = save + 1;
                    continue;
                }
                var featureId = (uint)fid;
                index.Add(featureId);
                minFeatId = Math.Min(featureId, minFeatId);
                if (p < line.Length && line[p] == ':')
                {
                    p++;
                    if (ParseNumber(line, ref p, line.Length, out var v)) values.Add((float)v);
                }
            }
        }
        if (labels.Count != 0) offset.Add(index.Count);
        if (indexingMode > 0 || (indexingMode < 0 && index.Count != 0 && minFeatId > 0))
            for (var i = 0; i < index.Count; i++) index[i]--;

        return new RowBlock
        {
            Size = labels.Count,
            Offset = [.. offset],
            Label = [.. labels],
            Weight = weights.Count == 0 ? null : [.. weights],
            Qid = qids.Count == 0 ? null : [.. qids],
            Index = [.. index],
            Value = values.Count == 0 ? null : [.. values],
        };
    }

    public static RowBlock ParseCsv(string text, int labelColumn, int weightColumn, string delimiter)
    {
        var offset = new List<long> { 0 };
        var labels = new List<float>();
        var weights = new List<float>();
        var index = new List<uint>();
        var values = new List<float>();
        if (text.Length > 0 && text[0] == '﻿') text = text[1..];
        foreach (var line in text.Split('\n', '\r'))
        {
            if (line.Length == 0) continue;
            var fields = line.Split(delimiter[0]);
            if (fields.Length == 1 && fields[0].Length > 0 && line.IndexOf(delimiter[0]) < 0 && labelColumn != 0)
                Check.Fail($"Delimiter '{delimiter}' is not found in the line. Expected '{delimiter}' as the delimiter to separate fields.");
            uint idx = 0;
            var weight = float.NaN;
            for (var col = 0; col < fields.Length; col++)
            {
                var f = fields[col];
                var parsed = Format.TryParseFloatPrefix(f, out var d, out var consumed) && consumed > 0;
                var v = parsed ? (float)d : 0f;
                if (col == labelColumn)
                {
                    labels.Add(v);
                }
                else if (col == weightColumn)
                {
                    weight = v;
                }
                else
                {
                    if (parsed)
                    {
                        values.Add(v);
                        index.Add(idx++);
                    }
                    else
                    {
                        idx++;
                    }
                }
            }
            if (!float.IsNaN(weight)) weights.Add(weight);
            offset.Add(index.Count);
        }
        return new RowBlock
        {
            Size = offset.Count - 1,
            Offset = [.. offset],
            Label = labels.Count == 0 ? null : [.. labels],
            Weight = weights.Count == 0 ? null : [.. weights],
            Index = [.. index],
            Value = [.. values],
        };
    }
}
