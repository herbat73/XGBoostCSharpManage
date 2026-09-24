// Port of the dump generators in src/tree/tree_model.cc (text, json, dot) and include/xgboost/feature_map.h.
using System.Text;
using XGBoost.Common;

namespace XGBoost.Tree;

/// <summary>Feature names and types for model dumps, <c>FeatureMap</c>.</summary>
public sealed class FeatureMap
{
    public enum FType
    {
        Indicator = 0,
        Quantitive = 1,
        Integer = 2,
        Float = 3,
        Categorical = 4,
    }

    private readonly List<string> _names = [];
    private readonly List<FType> _types = [];

    public void LoadText(TextReader reader)
    {
        var tokens = reader.ReadToEnd().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i + 2 < tokens.Length; i += 3)
        {
            if (!int.TryParse(tokens[i], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var fid))
                break;
            PushBack(fid, tokens[i + 1], tokens[i + 2]);
        }
    }

    public void PushBack(int fid, string fname, string ftype)
    {
        Check.Eq(fid, _names.Count);
        _names.Add(fname);
        _types.Add(GetType(ftype));
    }

    public void Clear()
    {
        _names.Clear();
        _types.Clear();
    }

    public int Size => _names.Count;

    public string Name(long idx)
    {
        Check.Lt(idx, _names.Count, "FeatureMap feature index exceed bound");
        return _names[(int)idx];
    }

    public FType TypeOf(long idx)
    {
        Check.Lt(idx, _names.Count, "FeatureMap feature index exceed bound");
        return _types[(int)idx];
    }

    private static FType GetType(string tname) => tname switch
    {
        "i" => FType.Indicator,
        "q" => FType.Quantitive,
        "int" => FType.Integer,
        "float" => FType.Float,
        "c" => FType.Categorical,
        _ => throw new XGBoostException("unknown feature type, use i for indicator and q for quantity"),
    };
}

internal static class TreeDump
{
    private const uint NoTruncate = uint.MaxValue;

    public static string ToStr(float value) => Format.G(value, 9);

    public static string ToStr(ReadOnlySpan<float> value, uint truncateLimit = 3)
    {
        if (value.Length == 1) return Format.G(value[0], 9);
        Check.Ge(truncateLimit, 2u);
        var n = Math.Min((uint)(value.Length - 1), truncateLimit - 1);
        var sb = new StringBuilder("[");
        for (var i = 0; i < n; i++) sb.Append(Format.G(value[i], 9)).Append(", ");
        if (value.Length > truncateLimit) sb.Append("..., ");
        sb.Append(Format.G(value[^1], 9)).Append(']');
        return sb.ToString();
    }

    public static string Match(string input, IEnumerable<(string Key, string Value)> replacements)
    {
        // std::map iteration order: keys sorted.
        var result = input;
        foreach (var (k, v) in replacements.OrderBy(r => r.Key, StringComparer.Ordinal))
        {
            var pos = result.IndexOf(k, StringComparison.Ordinal);
            Check.Ne(pos, -1);
            result = string.Concat(result.AsSpan(0, pos), v, result.AsSpan(pos + k.Length));
        }
        return result;
    }

    public static string I(long v) => Format.I(v);

    public static List<int> GetSplitCategories(ITreeView tree, int nidx)
    {
        var bits = tree.NodeCats(nidx);
        var cats = new List<int>();
        for (var i = 0; i < bits.Length * 32; i++)
            if (LBitField32.Check(bits, i)) cats.Add(i);
        return cats;
    }

    public static string PrintCatsAsSet(List<int> cats) => "{" + string.Join(",", cats) + "}";

    public static string GetFeatureName(FeatureMap fmap, uint splitIndex)
    {
        var fname = splitIndex < fmap.Size ? fmap.Name(splitIndex) : "f" + I(splitIndex);
        return StringUtils.EscapeU8(fname);
    }

    public static string Dump(RegTree tree, FeatureMap fmap, bool withStats, string attrs)
    {
        var pos = attrs.IndexOf(':');
        string name, parms = "";
        if (pos >= 0)
        {
            name = attrs[..pos];
            parms = attrs[(pos + 1)..].Replace('\'', '"');
        }
        else
        {
            name = attrs;
        }
        TreeGenerator gen = name switch
        {
            "dot" => new GraphvizGenerator(fmap, parms, withStats),
            "text" => new TextGenerator(fmap, withStats),
            "json" => new JsonGenerator(fmap, withStats),
            _ => throw new XGBoostException($"Unknown Model Builder:{name}"),
        };
        gen.BuildTree(tree.View());
        return gen.Str();
    }

    public static string Tabs(uint n) => new('\t', (int)n);

    private abstract class TreeGenerator(FeatureMap fmap, bool withStats)
    {
        protected readonly FeatureMap Fmap = fmap;
        protected readonly StringBuilder Ss = new();
        protected readonly bool WithStats = withStats;

        protected virtual string Indicator(ITreeView tree, int nid, uint depth) => "";
        protected abstract string Categorical(ITreeView tree, int nid, uint depth);
        protected virtual string Integer(ITreeView tree, int nid, uint depth) => "";
        protected virtual string Quantitive(ITreeView tree, int nid, uint depth) => "";
        protected virtual string NodeStat(ITreeView tree, int nid) => "";
        protected abstract string PlainNode(ITreeView tree, int nid, uint depth);
        protected abstract string LeafNode(ITreeView tree, int nid, uint depth);
        protected abstract string BuildTree(ITreeView tree, int nid, uint depth);

        protected virtual string SplitNode(ITreeView tree, int nid, uint depth)
        {
            var splitIndex = tree.SplitIndex(nid);
            var isCategorical = tree.SplitType(nid) == FeatureType.Categorical;
            if (splitIndex < Fmap.Size)
            {
                switch (Fmap.TypeOf(splitIndex))
                {
                    case FeatureMap.FType.Categorical:
                        Check.That(isCategorical, $"{Fmap.Name(splitIndex)} in feature map is categorical but the tree node is numerical.");
                        return Categorical(tree, nid, depth);
                    case FeatureMap.FType.Indicator:
                        Check.That(!isCategorical, $"{Fmap.Name(splitIndex)} in feature map is numerical but the tree node is categorical.");
                        return Indicator(tree, nid, depth);
                    case FeatureMap.FType.Integer:
                        Check.That(!isCategorical, $"{Fmap.Name(splitIndex)} in feature map is numerical but the tree node is categorical.");
                        return Integer(tree, nid, depth);
                    default:
                        Check.That(!isCategorical, $"{Fmap.Name(splitIndex)} in feature map is numerical but the tree node is categorical.");
                        return Quantitive(tree, nid, depth);
                }
            }
            return isCategorical ? Categorical(tree, nid, depth) : PlainNode(tree, nid, depth);
        }

        public virtual void BuildTree(ITreeView tree) => Ss.Append(BuildTree(tree, 0, 0));

        public string Str() => Ss.ToString();

        protected static int IntegerThreshold(float cond)
        {
            var floored = MathF.Floor(cond);
            return floored == cond ? (int)floored : (int)floored + 1;
        }
    }

    private sealed class TextGenerator(FeatureMap fmap, bool withStats) : TreeGenerator(fmap, withStats)
    {
        protected override string LeafNode(ITreeView tree, int nid, uint depth) =>
            Match("{tabs}{nid}:leaf={leaf}{stats}",
            [
                ("{tabs}", Tabs(depth)), ("{nid}", I(nid)), ("{leaf}", ToStr(tree.LeafValue(nid))),
                ("{stats}", WithStats ? Match(",cover={cover}", [("{cover}", ToStr(tree.SumHess(nid)))]) : ""),
            ]);

        protected override string Indicator(ITreeView tree, int nid, uint depth)
        {
            var nyes = tree.DefaultLeft(nid) ? tree.RightChild(nid) : tree.LeftChild(nid);
            return Match("{nid}:[{fname}] yes={yes},no={no}",
            [
                ("{nid}", I(nid)), ("{fname}", GetFeatureName(Fmap, tree.SplitIndex(nid))), ("{yes}", I(nyes)),
                ("{no}", I(tree.DefaultChild(nid))),
            ]);
        }

        private string SplitNodeImpl(ITreeView tree, int nid, string template, string cond, uint depth) =>
            Match(template,
            [
                ("{tabs}", Tabs(depth)), ("{nid}", I(nid)), ("{fname}", GetFeatureName(Fmap, tree.SplitIndex(nid))), ("{cond}", cond),
                ("{left}", I(tree.LeftChild(nid))), ("{right}", I(tree.RightChild(nid))), ("{missing}", I(tree.DefaultChild(nid))),
            ]);

        private const string NodeTemplate = "{tabs}{nid}:[{fname}<{cond}] yes={left},no={right},missing={missing}";

        protected override string Integer(ITreeView tree, int nid, uint depth) =>
            SplitNodeImpl(tree, nid, NodeTemplate, I(IntegerThreshold(tree.SplitCond(nid))), depth);

        protected override string Quantitive(ITreeView tree, int nid, uint depth) =>
            SplitNodeImpl(tree, nid, NodeTemplate, ToStr(tree.SplitCond(nid)), depth);

        protected override string PlainNode(ITreeView tree, int nid, uint depth) =>
            SplitNodeImpl(tree, nid, NodeTemplate, ToStr(tree.SplitCond(nid)), depth);

        protected override string Categorical(ITreeView tree, int nid, uint depth) =>
            SplitNodeImpl(tree, nid, "{tabs}{nid}:[{fname}:{cond}] yes={right},no={left},missing={missing}",
                PrintCatsAsSet(GetSplitCategories(tree, nid)), depth);

        protected override string NodeStat(ITreeView tree, int nid) =>
            Match(",gain={loss_chg},cover={sum_hess}", [("{loss_chg}", ToStr(tree.LossChg(nid))), ("{sum_hess}", ToStr(tree.SumHess(nid)))]);

        protected override string BuildTree(ITreeView tree, int nid, uint depth)
        {
            if (tree.IsLeaf(nid)) return LeafNode(tree, nid, depth);
            return Match("{parent}{stat}\n{left}\n{right}",
            [
                ("{parent}", SplitNode(tree, nid, depth)), ("{stat}", WithStats ? NodeStat(tree, nid) : ""),
                ("{left}", BuildTree(tree, tree.LeftChild(nid), depth + 1)), ("{right}", BuildTree(tree, tree.RightChild(nid), depth + 1)),
            ]);
        }

        public override void BuildTree(ITreeView tree) => Ss.Append(Match("{nodes}\n", [("{nodes}", BuildTree(tree, 0, 0))]));
    }

    private sealed class JsonGenerator(FeatureMap fmap, bool withStats) : TreeGenerator(fmap, withStats)
    {
        private static string Indent(uint depth)
        {
            var sb = new StringBuilder();
            for (var i = 0; i < depth + 1; i++) sb.Append("  ");
            return sb.ToString();
        }

        protected override string LeafNode(ITreeView tree, int nid, uint depth) =>
            Match("{ \"nodeid\": {nid}, \"leaf\": {leaf} {stat}}",
            [
                ("{nid}", I(nid)), ("{leaf}", ToStr(tree.LeafValue(nid), NoTruncate)),
                ("{stat}", WithStats ? Match(", \"cover\": {sum_hess} ", [("{sum_hess}", ToStr(tree.SumHess(nid)))]) : ""),
            ]);

        protected override string Indicator(ITreeView tree, int nid, uint depth)
        {
            var nyes = tree.DefaultLeft(nid) ? tree.RightChild(nid) : tree.LeftChild(nid);
            return Match(" \"nodeid\": {nid}, \"depth\": {depth}, \"split\": \"{fname}\", \"yes\": {yes}, \"no\": {no}",
            [
                ("{nid}", I(nid)), ("{depth}", I(depth)), ("{fname}", GetFeatureName(Fmap, tree.SplitIndex(nid))), ("{yes}", I(nyes)),
                ("{no}", I(tree.DefaultChild(nid))),
            ]);
        }

        protected override string Categorical(ITreeView tree, int nid, uint depth)
        {
            var cats = GetSplitCategories(tree, nid);
            var catsPtr = "[" + string.Join(", ", cats) + "]";
            return SplitNodeImpl(tree, nid,
                " \"nodeid\": {nid}, \"depth\": {depth}, \"split\": \"{fname}\", \"split_condition\": {cond}, \"yes\": {right}, \"no\": {left}, \"missing\": {missing}",
                catsPtr, depth);
        }

        private string SplitNodeImpl(ITreeView tree, int nid, string template, string cond, uint depth) =>
            Match(template,
            [
                ("{nid}", I(nid)), ("{depth}", I(depth)), ("{fname}", GetFeatureName(Fmap, tree.SplitIndex(nid))), ("{cond}", cond),
                ("{left}", I(tree.LeftChild(nid))), ("{right}", I(tree.RightChild(nid))), ("{missing}", I(tree.DefaultChild(nid))),
            ]);

        private const string NumericTemplate =
            " \"nodeid\": {nid}, \"depth\": {depth}, \"split\": \"{fname}\", \"split_condition\": {cond}, \"yes\": {left}, \"no\": {right}, \"missing\": {missing}";

        protected override string Integer(ITreeView tree, int nid, uint depth) =>
            SplitNodeImpl(tree, nid, NumericTemplate, I(IntegerThreshold(tree.SplitCond(nid))), depth);

        protected override string Quantitive(ITreeView tree, int nid, uint depth) =>
            SplitNodeImpl(tree, nid, NumericTemplate, ToStr(tree.SplitCond(nid)), depth);

        protected override string PlainNode(ITreeView tree, int nid, uint depth) =>
            SplitNodeImpl(tree, nid, NumericTemplate, ToStr(tree.SplitCond(nid)), depth);

        protected override string NodeStat(ITreeView tree, int nid) =>
            Match(", \"gain\": {loss_chg}, \"cover\": {sum_hess}", [("{loss_chg}", ToStr(tree.LossChg(nid))), ("{sum_hess}", ToStr(tree.SumHess(nid)))]);

        protected override string SplitNode(ITreeView tree, int nid, uint depth)
        {
            var properties = base.SplitNode(tree, nid, depth);
            return Match("{{properties} {stat}, \"children\": [{left}, {right}\n{indent}]}",
            [
                ("{properties}", properties), ("{stat}", WithStats ? NodeStat(tree, nid) : ""),
                ("{left}", BuildTree(tree, tree.LeftChild(nid), depth + 1)), ("{right}", BuildTree(tree, tree.RightChild(nid), depth + 1)),
                ("{indent}", Indent(depth)),
            ]);
        }

        protected override string BuildTree(ITreeView tree, int nid, uint depth) =>
            Match("{newline}{indent}{nodes}",
            [
                ("{newline}", depth == 0 ? "" : "\n"), ("{indent}", Indent(depth)),
                ("{nodes}", tree.IsLeaf(nid) ? LeafNode(tree, nid, depth) : SplitNode(tree, nid, depth)),
            ]);
    }

    private sealed class GraphvizGenerator : TreeGenerator
    {
        private readonly string _yesColor = "#0000FF";
        private readonly string _noColor = "#FF0000";
        private readonly string _rankdir = "TB";
        private readonly string _conditionNodeParams = "";
        private readonly string _leafNodeParams = "";
        private readonly string _graphAttrs = "";

        public GraphvizGenerator(FeatureMap fmap, string attrs, bool withStats) : base(fmap, withStats)
        {
            var kwargs = new SortedDictionary<string, SortedDictionary<string, string>>(StringComparer.Ordinal);
            if (attrs.Length != 0)
            {
                try
                {
                    var j = Json.Load(attrs).AsObject;
                    foreach (var (k, v) in j)
                    {
                        var inner = new SortedDictionary<string, string>(StringComparer.Ordinal);
                        foreach (var (ik, iv) in v.AsObject) inner[ik] = iv.AsString;
                        kwargs[k] = inner;
                    }
                }
                catch (XGBoostException e)
                {
                    Check.Fail($"Failed to parse graphviz parameters:\n\t{attrs}\nWith error:\n{e.Message}");
                }
            }
            if (kwargs.Remove("condition_node_params", out var cnp))
                foreach (var (k, v) in cnp) _conditionNodeParams += k + "=\"" + v + "\" ";
            if (kwargs.Remove("leaf_node_params", out var lnp))
                foreach (var (k, v) in lnp) _leafNodeParams += k + "=\"" + v + "\" ";
            if (kwargs.Remove("edge", out var edge))
            {
                if (edge.TryGetValue("yes_color", out var yc)) _yesColor = yc;
                if (edge.TryGetValue("no_color", out var nc)) _noColor = nc;
            }
            kwargs.Remove("graph_attrs", out var extra);
            foreach (var (k, v) in extra ?? []) _graphAttrs += Match("    graph [ {key}=\"{value}\" ]\n", [("{key}", k), ("{value}", v)]);
            if (kwargs.Count != 0)
                Log.Warning("The following parameters for graphviz are not recognized:\n" + string.Concat(kwargs.Keys.Select(k => k + ", ")));
        }

        private string BuildEdge(bool isCategorical, ITreeView tree, int nidx, int child, bool left)
        {
            var isMissing = tree.DefaultChild(nidx) == child;
            var branch = isCategorical
                ? (left ? "no" : "yes") + (isMissing ? ", missing" : "")
                : (left ? "yes" : "no") + (isMissing ? ", missing" : "");
            return Match("    {nid} -> {child} [label=\"{branch}\" color=\"{color}\"]\n",
            [
                ("{nid}", I(nidx)), ("{child}", I(child)), ("{color}", isMissing ? _yesColor : _noColor), ("{branch}", branch),
            ]);
        }

        protected override string PlainNode(ITreeView tree, int nidx, uint depth)
        {
            var splitIndex = tree.SplitIndex(nidx);
            var cond = tree.SplitCond(nidx);
            var hasLess = splitIndex >= Fmap.Size || Fmap.TypeOf(splitIndex) != FeatureMap.FType.Indicator;
            var result = Match("    {nid} [ label=\"{fname}{<}{cond}{stat}\" {params}]\n",
            [
                ("{nid}", I(nidx)), ("{fname}", GetFeatureName(Fmap, splitIndex)), ("{<}", hasLess ? "<" : ""),
                ("{cond}", hasLess ? ToStr(cond) : ""), ("{stat}", WithStats ? NodeStat(tree, nidx) : ""), ("{params}", _conditionNodeParams),
            ]);
            result += BuildEdge(false, tree, nidx, tree.LeftChild(nidx), true);
            result += BuildEdge(false, tree, nidx, tree.RightChild(nidx), false);
            return result;
        }

        protected override string NodeStat(ITreeView tree, int nidx) =>
            Match("\ngain={gain}\ncover={cover}", [("{cover}", ToStr(tree.SumHess(nidx))), ("{gain}", ToStr(tree.LossChg(nidx)))]);

        protected override string Categorical(ITreeView tree, int nidx, uint depth)
        {
            var result = Match("    {nid} [ label=\"{fname}:{cond}{stat}\" {params}]\n",
            [
                ("{nid}", I(nidx)), ("{fname}", GetFeatureName(Fmap, tree.SplitIndex(nidx))),
                ("{cond}", PrintCatsAsSet(GetSplitCategories(tree, nidx))), ("{stat}", WithStats ? NodeStat(tree, nidx) : ""),
                ("{params}", _conditionNodeParams),
            ]);
            result += BuildEdge(true, tree, nidx, tree.LeftChild(nidx), true);
            result += BuildEdge(true, tree, nidx, tree.RightChild(nidx), false);
            return result;
        }

        protected override string LeafNode(ITreeView tree, int nidx, uint depth) =>
            Match("    {nid} [ label=\"leaf={leaf-value}{cover}\" {params}]\n",
            [
                ("{nid}", I(nidx)), ("{leaf-value}", ToStr(tree.LeafValue(nidx))),
                ("{cover}", WithStats ? Match("\ncover={cover}", [("{cover}", ToStr(tree.SumHess(nidx)))]) : ""),
                ("{params}", _leafNodeParams),
            ]);

        protected override string BuildTree(ITreeView tree, int nidx, uint depth)
        {
            if (tree.IsLeaf(nidx)) return LeafNode(tree, nidx, depth);
            var node = tree.SplitType(nidx) == FeatureType.Categorical ? Categorical(tree, nidx, depth) : PlainNode(tree, nidx, depth);
            return Match("{parent}\n{left}\n{right}",
            [
                ("{parent}", node), ("{left}", BuildTree(tree, tree.LeftChild(nidx), depth + 1)),
                ("{right}", BuildTree(tree, tree.RightChild(nidx), depth + 1)),
            ]);
        }

        public override void BuildTree(ITreeView tree) =>
            Ss.Append(Match("digraph {\n    graph [ rankdir={rankdir} ]\n{graph_attrs}\n{nodes}}",
            [
                ("{rankdir}", _rankdir), ("{graph_attrs}", _graphAttrs), ("{nodes}", BuildTree(tree, 0, 0)),
            ]));
    }
}
