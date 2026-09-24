using System.Globalization;
using System.Text.Json;
using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Parsing JSON/UBJSON tree model files. Port of <c>demo/guide-python/model_parser.py</c>.
/// </summary>
/// <remarks>
/// See doc/tutorials/saving_model.rst for details about the model serialization. Pass
/// <c>--model path.json|path.ubj</c>; without it the demo trains a small model on the agaricus data and
/// parses that. A UBJSON model is converted to JSON by loading it and saving it to a JSON buffer, which
/// avoids needing a UBJSON parser.
/// </remarks>
internal static class ModelParser
{
    private enum SplitType
    {
        Numerical = 0,
        Categorical = 1,
    }

    private sealed record Node(
        // properties
        int Left,
        int Right,
        int Parent,
        uint SplitIdx,
        double SplitCond,
        bool DefaultLeft,
        SplitType SplitType,
        int[] Categories,
        // statistic
        double BaseWeight,
        double LossChg,
        double SumHess);

    /// <summary>A tree built by XGBoost.</summary>
    private sealed class Tree(int treeId, IReadOnlyList<Node> nodes)
    {
        public int TreeId => treeId;

        /// <summary>Loss gain of a node.</summary>
        public double LossChange(int nodeId) => nodes[nodeId].LossChg;

        /// <summary>Sum Hessian of a node.</summary>
        public double SumHessian(int nodeId) => nodes[nodeId].SumHess;

        /// <summary>Base weight of a node.</summary>
        public double BaseWeight(int nodeId) => nodes[nodeId].BaseWeight;

        /// <summary>Split feature index of node.</summary>
        public uint SplitIndex(int nodeId) => nodes[nodeId].SplitIdx;

        /// <summary>Split value of a node.</summary>
        public double SplitCondition(int nodeId) => nodes[nodeId].SplitCond;

        /// <summary>Categories in a node.</summary>
        public int[] SplitCategories(int nodeId) => nodes[nodeId].Categories;

        /// <summary>Whether a node has categorical split.</summary>
        public bool IsCategorical(int nodeId) => nodes[nodeId].SplitType == SplitType.Categorical;

        public bool IsNumerical(int nodeId) => !IsCategorical(nodeId);

        /// <summary>Parent ID of a node.</summary>
        public int Parent(int nodeId) => nodes[nodeId].Parent;

        /// <summary>Left child ID of a node.</summary>
        public int LeftChild(int nodeId) => nodes[nodeId].Left;

        /// <summary>Right child ID of a node.</summary>
        public int RightChild(int nodeId) => nodes[nodeId].Right;

        /// <summary>Whether a node is leaf.</summary>
        public bool IsLeaf(int nodeId) => nodes[nodeId].Left == -1;

        /// <summary>Whether a node is deleted.</summary>
        public bool IsDeleted(int nodeId) => SplitIndex(nodeId) == uint.MaxValue;

        public override string ToString()
        {
            var stack = new Stack<int>([0]);
            var lines = new List<string>();
            while (stack.Count > 0)
            {
                var nid = stack.Pop();
                var node = new List<string>
                {
                    $"'node id': {nid}",
                    $"'gain': {Fmt.Num(LossChange(nid))}",
                    $"'cover': {Fmt.Num(SumHessian(nid))}",
                };

                if (!IsLeaf(nid) && !IsDeleted(nid))
                {
                    stack.Push(LeftChild(nid));
                    stack.Push(RightChild(nid));
                    var categories = SplitCategories(nid);
                    if (categories.Length > 0)
                    {
                        Check.That(IsCategorical(nid), "a node with categories must be categorical");
                        node.Add($"'categories': [{string.Join(", ", categories)}]");
                    }
                    else
                    {
                        Check.That(IsNumerical(nid), "a node without categories must be numerical");
                        node.Add($"'condition': {Fmt.Num(SplitCondition(nid))}");
                    }
                }
                if (IsLeaf(nid)) node.Add($"'weight': {Fmt.Num(SplitCondition(nid))}");
                lines.Add("  {" + string.Join(", ", node) + "}");
            }
            return string.Join("\n", lines);
        }
    }

    /// <summary>Gradient boosted tree model.</summary>
    private sealed class Model
    {
        public int NumOutputGroup { get; }
        public int NumFeature { get; }
        public double[] BaseScore { get; }
        public int NumTrees { get; }
        public IReadOnlyList<Tree> Trees { get; }

        /// <summary>Constructs the model from the JSON representation of an XGBoost boosted tree model.</summary>
        public Model(JsonElement model)
        {
            // Basic properties of a model
            var learnerModelShape = model.GetProperty("learner").GetProperty("learner_model_param");
            NumOutputGroup = int.Parse(learnerModelShape.GetProperty("num_class").GetString()!, CultureInfo.InvariantCulture);
            NumFeature = int.Parse(learnerModelShape.GetProperty("num_feature").GetString()!, CultureInfo.InvariantCulture);
            // base_score is itself a JSON array stored as a string, e.g. "[5E-1]".
            BaseScore = JsonSerializer.Deserialize<double[]>(learnerModelShape.GetProperty("base_score").GetString()!)!;

            var gbtreeModel = model.GetProperty("learner").GetProperty("gradient_booster").GetProperty("model");
            var modelShape = gbtreeModel.GetProperty("gbtree_model_param");

            // JSON representation of trees
            var jTrees = gbtreeModel.GetProperty("trees");

            // Load the trees
            NumTrees = int.Parse(modelShape.GetProperty("num_trees").GetString()!, CultureInfo.InvariantCulture);

            var trees = new List<Tree>();
            for (var i = 0; i < NumTrees; i++)
            {
                var tree = jTrees[i];
                var treeId = tree.GetProperty("id").GetInt32();
                Check.That(treeId == i, $"tree id {treeId} != {i}");
                // - properties
                var leftChildren = Ints(tree, "left_children");
                var rightChildren = Ints(tree, "right_children");
                var parents = Ints(tree, "parents");
                var splitConditions = Doubles(tree, "split_conditions");
                var splitIndices = tree.GetProperty("split_indices").EnumerateArray().Select(e => e.GetUInt32()).ToArray();
                var defaultLeft = Ints(tree, "default_left");

                // - categorical features
                var splitTypes = Ints(tree, "split_type");
                // categories for each node are stored in a CSR style storage with segment as the begin ptr
                // and the `categories' as values.
                var catSegments = Ints(tree, "categories_segments");
                var catSizes = Ints(tree, "categories_sizes");
                // node index for categorical nodes
                var catNodes = Ints(tree, "categories_nodes");
                Check.That(catSegments.Length == catSizes.Length && catSizes.Length == catNodes.Length,
                    "inconsistent categorical storage");
                var cats = Ints(tree, "categories");
                Check.That(leftChildren.Length == splitTypes.Length, "inconsistent node count");

                // The storage for categories is only defined for categorical nodes to prevent unnecessary
                // overhead for numerical splits; we track the categorical nodes processed with a counter.
                var catCnt = 0;
                var lastCatNode = catNodes.Length > 0 ? catNodes[catCnt] : -1;
                var nodeCategories = new List<int[]>();
                for (var nodeId = 0; nodeId < leftChildren.Length; nodeId++)
                {
                    if (nodeId == lastCatNode)
                    {
                        var nodeCats = cats[catSegments[catCnt]..(catSegments[catCnt] + catSizes[catCnt])];
                        // categories are unique for each node
                        Check.That(nodeCats.Distinct().Count() == nodeCats.Length, "duplicated categories");
                        catCnt++;
                        // continue to process the rest of the nodes
                        lastCatNode = catCnt == catNodes.Length ? -1 : catNodes[catCnt];
                        Check.That(nodeCats.Length > 0, "empty categorical split");
                        nodeCategories.Add(nodeCats);
                    }
                    else
                    {
                        // append an empty node, it's either a numerical node or a leaf.
                        nodeCategories.Add([]);
                    }
                }

                // - stats
                var baseWeights = Doubles(tree, "base_weights");
                var lossChanges = Doubles(tree, "loss_changes");
                var sumHessian = Doubles(tree, "sum_hessian");

                // Construct a list of nodes that have complete information
                var nodes = Enumerable.Range(0, leftChildren.Length).Select(nodeId => new Node(
                    leftChildren[nodeId],
                    rightChildren[nodeId],
                    parents[nodeId],
                    splitIndices[nodeId],
                    splitConditions[nodeId],
                    defaultLeft[nodeId] == 1,
                    (SplitType)splitTypes[nodeId],
                    nodeCategories[nodeId],
                    baseWeights[nodeId],
                    lossChanges[nodeId],
                    sumHessian[nodeId])).ToList();

                trees.Add(new Tree(treeId, nodes));
            }
            Trees = trees;
        }

        public void PrintModel()
        {
            Console.WriteLine($"num_feature: {NumFeature}, num_class: {NumOutputGroup}, " +
                $"base_score: {Fmt.List(BaseScore)}, num_trees: {NumTrees}");
            for (var i = 0; i < Trees.Count; i++)
            {
                Console.WriteLine($"\ntree_id: {i}");
                Console.WriteLine(Trees[i]);
            }
        }

        private static int[] Ints(JsonElement tree, string name) =>
            tree.GetProperty(name).EnumerateArray().Select(e => e.GetInt32()).ToArray();

        private static double[] Doubles(JsonElement tree, string name) =>
            tree.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
    }

    public static void Run(string[] args)
    {
        var path = new DemoArgs(args).Get("--model") ?? TrainExampleModel();

        byte[] json;
        if (path.EndsWith("json", StringComparison.OrdinalIgnoreCase))
        {
            // use json format
            json = File.ReadAllBytes(path);
        }
        else if (path.EndsWith("ubj", StringComparison.OrdinalIgnoreCase))
        {
            // use ubjson format: let XGBoost convert it to JSON
            using var booster = Booster.Load(path);
            json = booster.SaveToBuffer(ModelFormat.Json);
        }
        else
        {
            throw new ArgumentException("Unexpected file extension. Supported file extension are json and ubj.");
        }

        using var document = JsonDocument.Parse(json);
        var model = new Model(document.RootElement);
        model.PrintModel();
    }

    /// <summary>Trains a small model to parse when no <c>--model</c> is given.</summary>
    private static string TrainExampleModel()
    {
        using var dtrain = DMatrix.FromFile(DemoPaths.LibSvm("agaricus.txt.train"));
        using var booster = XGB.Train(P.Of(("max_depth", 2), ("eta", 1), ("objective", "binary:logistic")), dtrain, 2);
        var path = DemoPaths.Output("model_parser_demo.ubj");
        booster.Save(path);
        Console.WriteLine($"No --model given; parsing a model trained on the agaricus data: {path}");
        return path;
    }
}
