// Port of include/xgboost/tree_model.h and the model parts of src/tree/tree_model.cc.
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using XGBoost.Common;

namespace XGBoost.Tree;

/// <summary>Meta parameters of a tree, <c>TreeParam</c>.</summary>
public struct TreeParam
{
    public int NumNodes = 1;
    public int NumDeleted = 0;
    public uint NumFeature = 0;
    public uint SizeLeafVector = 1;

    public TreeParam() { }

    public void FromJson(Json input)
    {
        var obj = input.AsObject;
        if (obj.TryGetValue("num_deleted", out var nd)) NumDeleted = int.Parse(nd.AsString, CultureInfo.InvariantCulture);
        NumFeature = (uint)ulong.Parse(obj["num_feature"].AsString, CultureInfo.InvariantCulture);
        NumNodes = int.Parse(obj["num_nodes"].AsString, CultureInfo.InvariantCulture);
        SizeLeafVector = (uint)ulong.Parse(obj["size_leaf_vector"].AsString, CultureInfo.InvariantCulture);
    }

    public readonly void ToJson(JsonObject output)
    {
        output["num_deleted"] = Format.I(NumDeleted);
        output["num_feature"] = Format.I(NumFeature);
        output["num_nodes"] = Format.I(NumNodes);
        output["size_leaf_vector"] = Format.I(SizeLeafVector);
    }
}

/// <summary>Node statistics, <c>RTreeNodeStat</c>.</summary>
public struct RTreeNodeStat(float lossChg, float sumHess, float baseWeight)
{
    public float LossChg = lossChg;
    public float SumHess = sumHess;
    public float BaseWeight = baseWeight;
    public int LeafChildCnt = 0;
}

/// <summary>Tree node, <c>RegTree::Node</c>. Leaf value and split condition share storage.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct Node
{
    private int _parent;
    private int _cleft;
    private int _cright;
    private uint _sindex;
    private float _info;

    public const int InvalidNodeId = -1;
    public const uint DeletedNodeMarker = uint.MaxValue;

    public Node()
    {
        _parent = InvalidNodeId;
        _cleft = InvalidNodeId;
        _cright = InvalidNodeId;
        _sindex = 0;
        _info = 0;
    }

    public Node(int cleft, int cright, int parent, uint splitInd, float splitCond, bool defaultLeft)
    {
        _parent = parent;
        _cleft = cleft;
        _cright = cright;
        _sindex = 0;
        _info = 0;
        SetParent(_parent);
        SetSplit(splitInd, splitCond, defaultLeft);
    }

    public readonly int LeftChild
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _cleft;
    }

    public readonly int RightChild
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _cright;
    }

    public readonly int DefaultChild => DefaultLeft ? LeftChild : RightChild;

    public readonly uint SplitIndex
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _sindex & ((1U << 31) - 1U);
    }

    public readonly bool DefaultLeft
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => (_sindex >> 31) != 0;
    }

    public readonly bool IsLeaf
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _cleft == InvalidNodeId;
    }

    public readonly float LeafValue => _info;
    public readonly float SplitCond => _info;
    public readonly int Parent => (int)((uint)_parent & ((1U << 31) - 1));
    public readonly bool IsLeftChild => ((uint)_parent & (1U << 31)) != 0;
    public readonly bool IsDeleted => _sindex == DeletedNodeMarker;
    public readonly bool IsRoot => _parent == InvalidNodeId;
    public readonly int RawParent => _parent;

    public void SetLeftChild(int nid) => _cleft = nid;
    public void SetRightChild(int nid) => _cright = nid;

    public void SetSplit(uint splitIndex, float splitCond, bool defaultLeft = false)
    {
        if (defaultLeft) splitIndex |= 1U << 31;
        _sindex = splitIndex;
        _info = splitCond;
    }

    public void SetLeaf(float value, int right = InvalidNodeId)
    {
        _info = value;
        _cleft = InvalidNodeId;
        _cright = right;
    }

    public void MarkDelete() => _sindex = DeletedNodeMarker;
    public void Reuse() => _sindex = 0;

    public void SetParent(int pidx, bool isLeftChild = true)
    {
        if (isLeftChild) pidx = (int)((uint)pidx | (1U << 31));
        _parent = pidx;
    }
}

/// <summary>Segment of the categorical bit storage owned by a node.</summary>
public struct CatSegment(long beg, long size)
{
    public long Beg = beg;
    public long Size = size;
}

/// <summary>Regression tree, <c>RegTree</c>. Scalar leaves, or vector leaves via <see cref="MultiTargetTree"/>.</summary>
public sealed class RegTree
{
    public const int InvalidNodeId = -1;
    public const int Root = 0;

    internal TreeParam Param;
    private List<Node> _nodes = [];
    private List<int> _deletedNodes = [];
    private List<RTreeNodeStat> _stats = [];
    private List<FeatureType> _splitTypes = [];
    private List<uint> _splitCategories = [];
    private List<CatSegment> _splitCategoriesSegments = [];
    private MultiTargetTree? _mtTree;

    public RegTree()
    {
        Param = new TreeParam();
        for (var i = 0; i < Param.NumNodes; i++)
        {
            var n = new Node();
            n.SetLeaf(0.0f);
            n.SetParent(InvalidNodeId);
            _nodes.Add(n);
            _stats.Add(default);
            _splitTypes.Add(FeatureType.Numerical);
            _splitCategoriesSegments.Add(default);
        }
    }

    public RegTree(uint nTargets, uint nFeatures) : this()
    {
        Param.NumFeature = nFeatures;
        Param.SizeLeafVector = nTargets;
        if (nTargets > 1) _mtTree = new MultiTargetTree(this);
    }

    public ref Node this[int nidx] => ref CollectionsMarshal.AsSpan(_nodes)[nidx];

    public ReadOnlySpan<Node> GetNodes()
    {
        Check.That(!IsMultiTarget);
        return CollectionsMarshal.AsSpan(_nodes);
    }

    public Node[] NodesArray => [.. _nodes];

    public ReadOnlySpan<RTreeNodeStat> GetStats()
    {
        Check.That(!IsMultiTarget);
        return CollectionsMarshal.AsSpan(_stats);
    }

    public ref RTreeNodeStat Stat(int nid) => ref CollectionsMarshal.AsSpan(_stats)[nid];

    public bool HasCategoricalSplit => _splitCategories.Count != 0;
    public bool IsMultiTarget => _mtTree is not null;
    public uint NumTargets => Param.SizeLeafVector;
    public MultiTargetTree GetMultiTargetTree()
    {
        Check.That(IsMultiTarget);
        return _mtTree!;
    }

    public uint NumFeatures => Param.NumFeature;
    public int NumNodes => Param.NumNodes;
    public int NumValidNodes => Param.NumNodes - Param.NumDeleted;
    public int NumExtraNodes => Param.NumNodes - 1 - Param.NumDeleted;

    public ReadOnlySpan<FeatureType> GetSplitTypes() => CollectionsMarshal.AsSpan(_splitTypes);
    public ReadOnlySpan<uint> GetSplitCategories() => CollectionsMarshal.AsSpan(_splitCategories);
    public ReadOnlySpan<CatSegment> GetSplitCategoriesPtr() => CollectionsMarshal.AsSpan(_splitCategoriesSegments);

    public CategoricalSplitMatrix GetCategoriesMatrix() =>
        new([.. _splitTypes], [.. _splitCategories], [.. _splitCategoriesSegments]);

    public int LeftChild(int nidx) => IsMultiTarget ? _mtTree!.LeftChild(nidx) : _nodes[nidx].LeftChild;
    public int RightChild(int nidx) => IsMultiTarget ? _mtTree!.RightChild(nidx) : _nodes[nidx].RightChild;
    public int Size => IsMultiTarget ? _mtTree!.Size : _nodes.Count;

    public void ChangeToLeaf(int nidx, float value)
    {
        Check.That(this[this[nidx].LeftChild].IsLeaf);
        Check.That(this[this[nidx].RightChild].IsLeaf);
        DeleteNode(this[nidx].LeftChild);
        DeleteNode(this[nidx].RightChild);
        this[nidx].SetLeaf(value);
    }

    public void CollapseToLeaf(int nidx, float value)
    {
        if (this[nidx].IsLeaf) return;
        if (!this[this[nidx].LeftChild].IsLeaf) CollapseToLeaf(this[nidx].LeftChild, 0.0f);
        if (!this[this[nidx].RightChild].IsLeaf) CollapseToLeaf(this[nidx].RightChild, 0.0f);
        ChangeToLeaf(nidx, value);
    }

    private int AllocNode()
    {
        if (Param.NumDeleted != 0)
        {
            var nid = _deletedNodes[^1];
            _deletedNodes.RemoveAt(_deletedNodes.Count - 1);
            this[nid].Reuse();
            Param.NumDeleted--;
            return nid;
        }
        var nd = Param.NumNodes++;
        Check.Lt(Param.NumNodes, int.MaxValue, "number of nodes in the tree exceed 2^31");
        _nodes.Add(new Node());
        _stats.Add(default);
        _splitTypes.Add(FeatureType.Numerical);
        _splitCategoriesSegments.Add(default);
        return nd;
    }

    private void DeleteNode(int nid)
    {
        Check.Ge(nid, 1);
        var pid = this[nid].Parent;
        if (nid == this[pid].LeftChild) this[pid].SetLeftChild(InvalidNodeId);
        else this[pid].SetRightChild(InvalidNodeId);
        _deletedNodes.Add(nid);
        this[nid].MarkDelete();
        Param.NumDeleted++;
    }

    public void ExpandNode(int nid, uint splitIndex, float splitValue, bool defaultLeft, float baseWeight, float leftLeafWeight,
        float rightLeafWeight, float lossChange, float sumHess, float leftSum, float rightSum, int leafRightChild = InvalidNodeId)
    {
        Check.That(!IsMultiTarget);
        var pleft = AllocNode();
        var pright = AllocNode();
        ref var node = ref this[nid];
        Check.That(node.IsLeaf);
        node.SetLeftChild(pleft);
        node.SetRightChild(pright);
        this[node.LeftChild].SetParent(nid, true);
        this[node.RightChild].SetParent(nid, false);
        node.SetSplit(splitIndex, splitValue, defaultLeft);
        this[pleft].SetLeaf(leftLeafWeight, leafRightChild);
        this[pright].SetLeaf(rightLeafWeight, leafRightChild);
        Stat(nid) = new RTreeNodeStat(lossChange, sumHess, baseWeight);
        Stat(pleft) = new RTreeNodeStat(0.0f, leftSum, leftLeafWeight);
        Stat(pright) = new RTreeNodeStat(0.0f, rightSum, rightLeafWeight);
        _splitTypes[nid] = FeatureType.Numerical;
    }

    public void ExpandCategorical(int nidx, uint splitIndex, ReadOnlySpan<uint> splitCat, bool defaultLeft, float baseWeight,
        float leftLeafWeight, float rightLeafWeight, float lossChange, float sumHess, float leftSum, float rightSum)
    {
        Check.That(!IsMultiTarget);
        ExpandNode(nidx, splitIndex, TreeIo.DftBadValue, defaultLeft, baseWeight, leftLeafWeight, rightLeafWeight, lossChange,
            sumHess, leftSum, rightSum);
        var origSize = _splitCategories.Count;
        foreach (var c in splitCat) _splitCategories.Add(c);
        _splitTypes[nidx] = FeatureType.Categorical;
        _splitCategoriesSegments[nidx] = new CatSegment(origSize, splitCat.Length);
    }

    /// <summary>Expands multiple leaves of a multi-target tree.</summary>
    public void Expand(Context ctx, ExpandBatch batch)
    {
        Check.That(IsMultiTarget);
        var categoriesBegin = _splitCategories.Count;
        _mtTree!.Expand(ctx, batch);
        var nNodes = _mtTree.Size;
        while (_splitTypes.Count < nNodes) _splitTypes.Add(FeatureType.Numerical);
        while (_splitCategoriesSegments.Count < nNodes) _splitCategoriesSegments.Add(default);
        if (batch.NCatWords != 0)
        {
            foreach (var cats in batch.CatBits)
                if (cats.Length != 0) _splitCategories.AddRange(cats);
        }
        long categoryOffset = categoriesBegin;
        for (var i = 0; i < batch.Size; i++)
        {
            var nidx = batch.Nidxs[i];
            var cats = batch.CatBits[i];
            if (cats.Length == 0)
            {
                _splitTypes[nidx] = FeatureType.Numerical;
                _splitCategoriesSegments[nidx] = default;
            }
            else
            {
                _splitTypes[nidx] = FeatureType.Categorical;
                _splitCategoriesSegments[nidx] = new CatSegment(categoryOffset, cats.Length);
                categoryOffset += cats.Length;
            }
        }
        Param.NumNodes = nNodes;
    }

    public void SetLeaves(List<int> leaves, ReadOnlySpan<float> weights)
    {
        Check.That(IsMultiTarget);
        _mtTree!.SetLeaves(leaves, weights);
    }

    public void SetRoot(ReadOnlySpan<float> weight, float sumHess)
    {
        Check.That(IsMultiTarget);
        _mtTree!.SetRoot(weight, sumHess);
    }

    public int GetNumLeaves()
    {
        var leaves = 0;
        TreeViews.WalkTree(this, (t, nidx) =>
        {
            if (t.IsLeaf(nidx)) leaves++;
            return true;
        });
        return leaves;
    }

    public int GetNumSplitNodes()
    {
        var splits = 0;
        TreeViews.WalkTree(this, (t, nidx) =>
        {
            if (!t.IsLeaf(nidx)) splits++;
            return true;
        });
        return splits;
    }

    public int GetDepth(int nidx)
    {
        if (nidx == 0) return 0;
        return TreeViews.GetDepth(View(), nidx);
    }

    public int MaxDepth() => TreeViews.MaxDepth(View(), Root);

    /// <summary>Host view of either tree type.</summary>
    public ITreeView View() => IsMultiTarget ? new MultiTargetTreeView(this) : new ScalarTreeView(this);

    public RegTree Copy()
    {
        var ptr = new RegTree
        {
            Param = Param,
            _nodes = [.. _nodes],
            _deletedNodes = [.. _deletedNodes],
            _stats = [.. _stats],
            _splitTypes = [.. _splitTypes],
            _splitCategories = [.. _splitCategories],
            _splitCategoriesSegments = [.. _splitCategoriesSegments],
        };
        if (_mtTree is not null) ptr._mtTree = _mtTree.Copy(ptr);
        return ptr;
    }

    public bool Equal(RegTree b)
    {
        if (NumTargets != b.NumTargets) return false;
        if (HasCategoricalSplit != b.HasCategoricalSplit) return false;
        if (NumExtraNodes != b.NumExtraNodes) return false;
        if (Size != b.Size) return false;
        if (HasCategoricalSplit)
        {
            var lp = GetSplitCategoriesPtr();
            var rp = b.GetSplitCategoriesPtr();
            if (lp.Length != rp.Length) return false;
            for (var i = 0; i < lp.Length; i++)
                if (lp[i].Size != rp[i].Size || lp[i].Beg != rp[i].Beg) return false;
            if (!GetSplitTypes().SequenceEqual(b.GetSplitTypes())) return false;
            if (!GetSplitCategories().SequenceEqual(b.GetSplitCategories())) return false;
        }
        static bool FloatEq(float l, float r) => Math.Abs(l - r) < Constants.RtEps;
        var lhs = View();
        var rhs = b.View();
        var equal = false;
        var hasCat = HasCategoricalSplit;
        var nTargets = NumTargets;
        TreeViews.WalkTree(lhs, nidx =>
        {
            bool LeafSame()
            {
                if (lhs is ScalarTreeView) return FloatEq(lhs.LeafValue(nidx)[0], rhs.LeafValue(nidx)[0]);
                var l = lhs.LeafValue(nidx);
                var r = rhs.LeafValue(nidx);
                for (var t = 0; t < nTargets; t++)
                    if (!FloatEq(l[t], r[t])) return false;
                return true;
            }
            var res = lhs.LeftChild(nidx) == rhs.LeftChild(nidx) && lhs.RightChild(nidx) == rhs.RightChild(nidx)
                && lhs.SplitIndex(nidx) == rhs.SplitIndex(nidx) && lhs.DefaultLeft(nidx) == rhs.DefaultLeft(nidx)
                && (!hasCat || lhs.SplitType(nidx) == rhs.SplitType(nidx)) && lhs.IsLeaf(nidx) == rhs.IsLeaf(nidx)
                && (lhs.IsLeaf(nidx) ? LeafSame() : FloatEq(lhs.SplitCond(nidx), rhs.SplitCond(nidx)));
            equal = res;
            return res;
        });
        return equal;
    }

    // ---- IO -------------------------------------------------------------------------------------

    private void LoadCategoricalSplit(JsonObject input)
    {
        var categoriesSegments = TreeIo.Ints(input["categories_segments"]);
        var categoriesSizes = TreeIo.Ints(input["categories_sizes"]);
        var categoriesNodes = TreeIo.Ints(input["categories_nodes"]);
        var categories = TreeIo.Ints(input["categories"]);
        var splitType = TreeIo.Ints(input["split_type"]);
        var nNodes = splitType.Length;
        var cnt = 0;
        var lastCatNode = -1L;
        if (categoriesNodes.Length != 0) lastCatNode = categoriesNodes[cnt];
        _splitTypes = [.. new FeatureType[nNodes]];
        _splitCategoriesSegments = [.. new CatSegment[nNodes]];
        _splitCategories = [];
        for (var nidx = 0; nidx < nNodes; nidx++)
        {
            _splitTypes[nidx] = (FeatureType)splitType[nidx];
            if (nidx == lastCatNode)
            {
                var jBegin = categoriesSegments[cnt];
                var jEnd = categoriesSizes[cnt] + jBegin;
                var maxCat = int.MinValue;
                Check.Gt(jEnd - jBegin, 0L, nidx.ToString(CultureInfo.InvariantCulture));
                for (var j = jBegin; j < jEnd; j++) maxCat = Math.Max(maxCat, (int)categories[j]);
                Check.Ne(int.MinValue, maxCat);
                var nCats = maxCat + 1;
                var size = LBitField32.ComputeStorageSize(nCats);
                var bits = new uint[size];
                var field = new LBitField32(bits);
                for (var j = jBegin; j < jEnd; j++) field.Set((int)categories[j]);
                var begin = _splitCategories.Count;
                _splitCategories.AddRange(bits);
                _splitCategoriesSegments[nidx] = new CatSegment(begin, bits.Length);
                cnt++;
                lastCatNode = cnt == categoriesNodes.Length ? -1 : categoriesNodes[cnt];
            }
            else
            {
                _splitCategoriesSegments[nidx] = new CatSegment(categories.Length, 0);
            }
        }
    }

    private void SaveCategoricalSplit(JsonObject output)
    {
        Check.Eq(_splitTypes.Count, Size);
        Check.Eq(_splitCategoriesSegments.Count, Size);
        var categoriesSegments = new List<long>();
        var categoriesSizes = new List<long>();
        var categories = new List<int>();
        var categoriesNodes = new List<int>();
        var splitType = new byte[_splitTypes.Count];
        var allCats = GetSplitCategories().ToArray();
        for (var i = 0; i < Size; i++)
        {
            splitType[i] = (byte)_splitTypes[i];
            if (_splitTypes[i] != FeatureType.Categorical) continue;
            categoriesNodes.Add(i);
            var begin = categories.Count;
            categoriesSegments.Add(begin);
            var seg = _splitCategoriesSegments[i];
            var bits = allCats.AsSpan((int)seg.Beg, (int)seg.Size);
            for (var c = 0; c < bits.Length * 32; c++)
                if (LBitField32.Check(bits, c)) categories.Add(c);
            var size = categories.Count - begin;
            categoriesSizes.Add(size);
            Check.Ne(size, 0);
        }
        output["split_type"] = new U8Array(splitType);
        output["categories_segments"] = new I64Array([.. categoriesSegments]);
        output["categories_sizes"] = new I64Array([.. categoriesSizes]);
        output["categories_nodes"] = new I32Array([.. categoriesNodes]);
        output["categories"] = new I32Array([.. categories]);
    }

    public void LoadModel(Json input)
    {
        var obj = input.AsObject;
        Param.FromJson(obj["tree_param"]);
        var hasCat = obj.ContainsKey("split_type");
        if (hasCat) LoadCategoricalSplit(obj);
        if (Param.SizeLeafVector > 1)
        {
            _mtTree = new MultiTargetTree(this);
            _mtTree.LoadModel(obj);
            return;
        }
        var nNodes = Param.NumNodes;
        Check.Ne(nNodes, 0);
        var lossChanges = TreeIo.Floats(obj[TreeIo.LossChg]);
        Check.Eq(lossChanges.Length, nNodes);
        var sumHessian = TreeIo.Floats(obj[TreeIo.SumHess]);
        Check.Eq(sumHessian.Length, nNodes);
        var baseWeights = TreeIo.Floats(obj[TreeIo.BaseWeight]);
        Check.Eq(baseWeights.Length, nNodes);
        var lefts = TreeIo.Ints(obj[TreeIo.Left]);
        Check.Eq(lefts.Length, nNodes);
        var rights = TreeIo.Ints(obj[TreeIo.Right]);
        Check.Eq(rights.Length, nNodes);
        var parents = TreeIo.Ints(obj[TreeIo.Parent]);
        Check.Eq(parents.Length, nNodes);
        var indices = TreeIo.Ints(obj[TreeIo.SplitIdx]);
        Check.Eq(indices.Length, nNodes);
        var conds = TreeIo.Floats(obj[TreeIo.SplitCond]);
        Check.Eq(conds.Length, nNodes);
        var defaultLeft = TreeIo.Bools(obj[TreeIo.DftLeft]);
        Check.Eq(defaultLeft.Length, nNodes);

        _stats = new List<RTreeNodeStat>(nNodes);
        _nodes = new List<Node>(nNodes);
        for (var i = 0; i < nNodes; i++)
        {
            _stats.Add(new RTreeNodeStat(lossChanges[i], sumHessian[i], baseWeights[i]));
            _nodes.Add(new Node((int)lefts[i], (int)rights[i], (int)parents[i], (uint)indices[i], conds[i], defaultLeft[i]));
        }
        if (!hasCat)
        {
            _splitCategoriesSegments = [.. new CatSegment[nNodes]];
            _splitTypes = [.. new FeatureType[nNodes]];
        }
        _deletedNodes.Clear();
        for (var i = 1; i < Param.NumNodes; i++)
            if (_nodes[i].IsDeleted) _deletedNodes.Add(i);
        for (var nid = 1; nid < Param.NumNodes; nid++)
        {
            var parent = this[nid].Parent;
            Check.Ne(parent, InvalidNodeId);
            this[nid].SetParent(this[nid].Parent, this[parent].LeftChild == nid);
        }
        Check.Eq(_deletedNodes.Count, Param.NumDeleted);
        Check.Eq(_splitCategoriesSegments.Count, Param.NumNodes);
    }

    public void SaveModel(JsonObject output)
    {
        var tp = new JsonObject();
        Param.ToJson(tp);
        output["tree_param"] = tp;
        SaveCategoricalSplit(output);
        if (IsMultiTarget)
        {
            Check.Gt(Param.SizeLeafVector, 1u);
            _mtTree!.SaveModel(output);
            return;
        }
        Check.Eq(Param.NumNodes, _nodes.Count);
        Check.Eq(Param.NumNodes, _stats.Count);
        var n = Param.NumNodes;
        var lossChanges = new float[n];
        var sumHessian = new float[n];
        var baseWeights = new float[n];
        var lefts = new int[n];
        var rights = new int[n];
        var parents = new int[n];
        var conds = new float[n];
        var defaultLeft = new byte[n];
        var idx = new long[n];
        for (var i = 0; i < n; i++)
        {
            var s = _stats[i];
            lossChanges[i] = s.LossChg;
            sumHessian[i] = s.SumHess;
            baseWeights[i] = s.BaseWeight;
            var node = _nodes[i];
            lefts[i] = node.LeftChild;
            rights[i] = node.RightChild;
            parents[i] = node.Parent;
            idx[i] = node.SplitIndex;
            conds[i] = node.SplitCond;
            defaultLeft[i] = (byte)(node.DefaultLeft ? 1 : 0);
        }
        output[TreeIo.SplitIdx] = Param.NumFeature > int.MaxValue
            ? new I64Array(idx)
            : new I32Array(Array.ConvertAll(idx, v => (int)v));
        output[TreeIo.LossChg] = new F32Array(lossChanges);
        output[TreeIo.SumHess] = new F32Array(sumHessian);
        output[TreeIo.BaseWeight] = new F32Array(baseWeights);
        output[TreeIo.Left] = new I32Array(lefts);
        output[TreeIo.Right] = new I32Array(rights);
        output[TreeIo.Parent] = new I32Array(parents);
        output[TreeIo.SplitCond] = new F32Array(conds);
        output[TreeIo.DftLeft] = new U8Array(defaultLeft);
    }

    /// <summary>Dumps the tree as text, json or dot (<c>RegTree::DumpModel</c>).</summary>
    public string DumpModel(FeatureMap fmap, bool withStats, string format) => TreeDump.Dump(this, fmap, withStats, format);

    /// <summary>Dense feature vector with NaN for missing values, <c>RegTree::FVec</c>.</summary>
    public sealed class FVec
    {
        private float[] _data = [];
        private bool _hasMissing;

        public void Init(int size)
        {
            if (_data.Length != size) _data = new float[size];
            Array.Fill(_data, float.NaN);
            _hasMissing = true;
        }

        public void Fill(ReadOnlySpan<Entry> inst)
        {
            foreach (var e in inst) _data[e.Index] = e.Fvalue;
            _hasMissing = _data.Length != inst.Length;
        }

        public void Drop() => Init(_data.Length);
        public int Size => _data.Length;
        public float GetFvalue(int i) => _data[i];
        public bool IsMissing(int i) => float.IsNaN(_data[i]);
        public bool HasMissing() => _hasMissing;
        public void HasMissing(bool v) => _hasMissing = v;
        public Span<float> Data => _data;
    }
}

/// <summary>Categorical split storage view, <c>RegTree::CategoricalSplitMatrix</c>.</summary>
public sealed class CategoricalSplitMatrix(FeatureType[] splitType, uint[] categories, CatSegment[] nodePtr)
{
    public FeatureType[] SplitType { get; } = splitType;
    public uint[] Categories { get; } = categories;
    public CatSegment[] NodePtr { get; } = nodePtr;

    public ReadOnlySpan<uint> NodeCats(int nidx)
    {
        var seg = NodePtr[nidx];
        return Categories.AsSpan((int)seg.Beg, (int)seg.Size);
    }
}

/// <summary>Field names and helpers from src/tree/io_utils.h.</summary>
public static class TreeIo
{
    public const string LossChg = "loss_changes";
    public const string SumHess = "sum_hessian";
    public const string BaseWeight = "base_weights";
    public const string LeafWeight = "leaf_weights";
    public const string SplitIdx = "split_indices";
    public const string SplitCond = "split_conditions";
    public const string DftLeft = "default_left";
    public const string Parent = "parents";
    public const string Left = "left_children";
    public const string Right = "right_children";

    /// <summary><c>DftBadValue</c>: smallest subnormal float.</summary>
    public static readonly float DftBadValue = float.Epsilon;

    public static float[] Floats(Json j) => j switch
    {
        F32Array a => a.Values,
        JsonArray a => [.. a.Values.Select(v => v.AsNumber)],
        _ => throw new XGBoostException($"Invalid cast, from {j.TypeStr} to F32Array"),
    };

    public static long[] Ints(Json j) => j switch
    {
        I32Array a => Array.ConvertAll(a.Values, v => (long)v),
        I64Array a => a.Values,
        U8Array a => Array.ConvertAll(a.Values, v => (long)v),
        I8Array a => Array.ConvertAll(a.Values, v => (long)v),
        I16Array a => Array.ConvertAll(a.Values, v => (long)v),
        JsonArray a => [.. a.Values.Select(v => v.AsInteger)],
        _ => throw new XGBoostException($"Invalid cast, from {j.TypeStr} to I32Array"),
    };

    public static bool[] Bools(Json j) => j switch
    {
        U8Array a => Array.ConvertAll(a.Values, v => v == 1),
        JsonArray a => [.. a.Values.Select(v => v is JsonBoolean b ? b.Value : v.AsInteger == 1)],
        _ => throw new XGBoostException($"Invalid cast, from {j.TypeStr} to U8Array"),
    };
}

/// <summary>Inputs for expanding multiple leaves in a multi-target tree, <c>ExpandBatch</c>.</summary>
public sealed class ExpandBatch(float eta)
{
    public List<int> Nidxs = [];
    public List<uint> Fidxs = [];
    public List<float> Conds = [];
    public List<byte> DftLefts = [];
    public List<float[]> BaseWeightBatch = [];
    public List<float[]> LeftWeightBatch = [];
    public List<float[]> RightWeightBatch = [];
    public List<float> LossChgs = [];
    public List<double> LeftSums = [];
    public List<double> RightSums = [];
    public List<uint[]> CatBits = [];
    public long NCatWords;
    public float Eta = eta;

    public int Size => Nidxs.Count;

    public void Push(int nidx, uint fidx, float cond, bool dftLeft, float[] baseWeight, float[] leftWeight, float[] rightWeight,
        float lossChg, double leftSum, double rightSum, uint[]? cats = null)
    {
        cats ??= [];
        Nidxs.Add(nidx);
        Fidxs.Add(fidx);
        Conds.Add(cond);
        DftLefts.Add((byte)(dftLeft ? 1 : 0));
        BaseWeightBatch.Add(baseWeight);
        LeftWeightBatch.Add(leftWeight);
        RightWeightBatch.Add(rightWeight);
        LossChgs.Add(lossChg);
        LeftSums.Add(leftSum);
        RightSums.Add(rightSum);
        CatBits.Add(cats);
        NCatWords += cats.Length;
        Check.Eq(leftWeight.Length, baseWeight.Length);
        Check.Eq(rightWeight.Length, baseWeight.Length);
    }
}

/// <summary>Tree with vector leaves, <c>MultiTargetTree</c>.</summary>
public sealed class MultiTargetTree
{
    public const int InvalidNodeId = -1;

    private readonly RegTree _owner;
    internal List<int> Left = [InvalidNodeId];
    internal List<int> Right = [InvalidNodeId];
    internal List<int> ParentArr = [InvalidNodeId];
    internal List<uint> SplitIndex = [0];
    internal List<byte> DefaultLeft = [0];
    internal List<float> SplitConds = [TreeIo.DftBadValue];
    internal List<float> Weights = [];
    internal List<float> LeafWeightsArr = [];
    internal List<float> LossChg = [0f];
    internal List<float> SumHess = [0f];

    public MultiTargetTree(RegTree owner)
    {
        _owner = owner;
        Check.Gt(owner.Param.SizeLeafVector, 1u);
    }

    private ref TreeParam Param => ref _owner.Param;

    public bool IsLeaf(int nidx) => Left[nidx] == InvalidNodeId;
    public int LeftChild(int nidx) => Left[nidx];
    public int RightChild(int nidx) => Right[nidx];
    public uint NumTargets => _owner.Param.SizeLeafVector;

    public uint NumSplitTargets
    {
        get
        {
            var n = Weights.Count / Left.Count;
            Check.Ne(n, 0);
            return (uint)n;
        }
    }

    public int NumLeaves => (int)(LeafWeightsArr.Count / NumTargets);
    public int Size => ParentArr.Count;
    public ReadOnlySpan<float> LeafWeights => CollectionsMarshal.AsSpan(LeafWeightsArr);

    private ReadOnlySpan<float> NodeWeight(int nidx)
    {
        var st = (int)NumSplitTargets;
        return CollectionsMarshal.AsSpan(Weights).Slice(nidx * st, st);
    }

    public ReadOnlySpan<float> LeafValue(int nidx)
    {
        Check.That(IsLeaf(nidx));
        var nTargets = (int)NumTargets;
        var lidx = Right[nidx];
        Check.Ne(lidx, InvalidNodeId);
        return CollectionsMarshal.AsSpan(LeafWeightsArr).Slice(lidx * nTargets, nTargets);
    }

    public MultiTargetTree Copy(RegTree owner) => new(owner)
    {
        Left = [.. Left],
        Right = [.. Right],
        ParentArr = [.. ParentArr],
        SplitIndex = [.. SplitIndex],
        DefaultLeft = [.. DefaultLeft],
        SplitConds = [.. SplitConds],
        Weights = [.. Weights],
        LeafWeightsArr = [.. LeafWeightsArr],
        LossChg = [.. LossChg],
        SumHess = [.. SumHess],
    };

    private static void Resize<T>(List<T> l, int n, T fill)
    {
        if (l.Count > n) l.RemoveRange(n, l.Count - n);
        while (l.Count < n) l.Add(fill);
    }

    public void SetRoot(ReadOnlySpan<float> weight, float sumHess)
    {
        Check.That(!weight.IsEmpty);
        Resize(Weights, weight.Length, TreeIo.DftBadValue);
        Check.Le((uint)weight.Length, NumTargets);
        for (var i = 0; i < weight.Length; i++) Weights[i] = weight[i];
        Resize(SumHess, 1, 0f);
        SumHess[RegTree.Root] = sumHess;
        Resize(LossChg, 1, 0f);
        Check.Eq(Param.NumNodes, 1);
        Check.Eq(NumSplitTargets, (uint)weight.Length);
    }

    public void Expand(Context ctx, ExpandBatch batch)
    {
        _ = ctx;
        var batchSize = batch.Size;
        var nSplitTargets = (int)NumSplitTargets;
        var oldNNodes = Size;
        var nNodes = oldNNodes + batchSize * 2;
        Resize(Left, nNodes, InvalidNodeId);
        Resize(Right, nNodes, InvalidNodeId);
        Resize(ParentArr, nNodes, InvalidNodeId);
        Resize(SplitIndex, nNodes, 0u);
        Resize(SplitConds, nNodes, TreeIo.DftBadValue);
        Resize(DefaultLeft, nNodes, (byte)0);
        for (var i = 0; i < batchSize; i++)
        {
            var nidx = batch.Nidxs[i];
            Left[nidx] = oldNNodes + i * 2;
            Right[nidx] = Left[nidx] + 1;
            ParentArr[Left[nidx]] = nidx;
            ParentArr[Right[nidx]] = nidx;
            SplitIndex[nidx] = batch.Fidxs[i];
            SplitConds[nidx] = batch.CatBits[i].Length == 0 ? batch.Conds[i] : TreeIo.DftBadValue;
            DefaultLeft[nidx] = batch.DftLefts[i];
        }
        Resize(Weights, nNodes * nSplitTargets, 0f);
        for (var i = 0; i < batchSize; i++)
        {
            var nidx = batch.Nidxs[i];
            batch.BaseWeightBatch[i].CopyTo(CollectionsMarshal.AsSpan(Weights)[(nidx * nSplitTargets)..]);
            batch.LeftWeightBatch[i].CopyTo(CollectionsMarshal.AsSpan(Weights)[(Left[nidx] * nSplitTargets)..]);
            batch.RightWeightBatch[i].CopyTo(CollectionsMarshal.AsSpan(Weights)[(Right[nidx] * nSplitTargets)..]);
        }
        var nChild = batchSize * 2 * nSplitTargets;
        var w = CollectionsMarshal.AsSpan(Weights).Slice(oldNNodes * nSplitTargets, nChild);
        for (var k = 0; k < w.Length; k++) w[k] *= batch.Eta;
        Resize(LossChg, nNodes, 0f);
        Resize(SumHess, nNodes, 0f);
        for (var i = 0; i < batchSize; i++)
        {
            var nidx = batch.Nidxs[i];
            LossChg[nidx] = batch.LossChgs[i];
            SumHess[nidx] = (float)(batch.LeftSums[i] + batch.RightSums[i]);
            SumHess[Left[nidx]] = (float)batch.LeftSums[i];
            SumHess[Right[nidx]] = (float)batch.RightSums[i];
        }
    }

    public void SetLeaves(List<int> leaves, ReadOnlySpan<float> weights)
    {
        var isPartial = NumLeaves == 0;
        Check.That(isPartial || leaves.Count == NumLeaves);
        var nTargets = (int)NumTargets;
        var nidxInSet = 0;
        Resize(LeafWeightsArr, leaves.Count * nTargets, 0f);
        var hw = CollectionsMarshal.AsSpan(LeafWeightsArr);
        foreach (var nidx in leaves)
        {
            Check.That(IsLeaf(nidx));
            weights.Slice(nidxInSet * nTargets, nTargets).CopyTo(hw.Slice(nidxInSet * nTargets, nTargets));
            if (isPartial) Check.Eq(Right[nidx], InvalidNodeId);
            Right[nidx] = nidxInSet;
            nidxInSet++;
        }
    }

    public void SetLeaves()
    {
        Check.Eq(NumLeaves, 0);
        var nTargets = (int)NumTargets;
        Check.Eq((uint)nTargets, NumSplitTargets);
        var nNodes = Param.NumNodes;
        var nidxInSet = 0;
        Check.That(LeafWeightsArr.Count == 0);
        for (var nidx = 0; nidx < nNodes; nidx++)
        {
            if (!IsLeaf(nidx)) continue;
            LeafWeightsArr.AddRange(NodeWeight(nidx));
            Check.Eq(Right[nidx], InvalidNodeId);
            Right[nidx] = nidxInSet;
            nidxInSet++;
        }
    }

    public void LoadModel(JsonObject input)
    {
        Weights = [.. TreeIo.Floats(input[TreeIo.BaseWeight])];
        LeafWeightsArr = [.. TreeIo.Floats(input[TreeIo.LeafWeight])];
        SplitConds = [.. TreeIo.Floats(input[TreeIo.SplitCond])];
        Left = [.. TreeIo.Ints(input[TreeIo.Left]).Select(v => (int)v)];
        Right = [.. TreeIo.Ints(input[TreeIo.Right]).Select(v => (int)v)];
        ParentArr = [.. TreeIo.Ints(input[TreeIo.Parent]).Select(v => (int)v)];
        SplitIndex = [.. TreeIo.Ints(input[TreeIo.SplitIdx]).Select(v => (uint)v)];
        DefaultLeft = [.. TreeIo.Bools(input[TreeIo.DftLeft]).Select(v => (byte)(v ? 1 : 0))];
        LossChg = [.. TreeIo.Floats(input[TreeIo.LossChg])];
        SumHess = [.. TreeIo.Floats(input[TreeIo.SumHess])];
    }

    public void SaveModel(JsonObject output)
    {
        var nNodes = Param.NumNodes;
        var nLeaves = NumLeaves;
        Check.Ge(nLeaves, 1);
        var nst = (int)NumSplitTargets;
        var nt = (int)NumTargets;
        var lefts = new int[nNodes];
        var rights = new int[nNodes];
        var parents = new int[nNodes];
        var conds = new float[nNodes];
        var dl = new byte[nNodes];
        var weights = new float[Weights.Count];
        var lossChg = new float[nNodes];
        var sumHess = new float[nNodes];
        var leafWeights = new float[nLeaves * nt];
        var idx = new long[nNodes];
        for (var nidx = 0; nidx < nNodes; nidx++)
        {
            lefts[nidx] = Left[nidx];
            rights[nidx] = Right[nidx];
            parents[nidx] = ParentArr[nidx];
            idx[nidx] = SplitIndex[nidx];
            conds[nidx] = SplitConds[nidx];
            dl[nidx] = DefaultLeft[nidx];
            lossChg[nidx] = LossChg[nidx];
            sumHess[nidx] = SumHess[nidx];
            NodeWeight(nidx).CopyTo(weights.AsSpan(nidx * nst, nst));
            if (IsLeaf(nidx)) LeafValue(nidx).CopyTo(leafWeights.AsSpan(Right[nidx] * nt, nt));
        }
        output[TreeIo.SplitIdx] = Param.NumFeature > int.MaxValue ? new I64Array(idx) : new I32Array(Array.ConvertAll(idx, v => (int)v));
        output[TreeIo.BaseWeight] = new F32Array(weights);
        output[TreeIo.LeafWeight] = new F32Array(leafWeights);
        output[TreeIo.Left] = new I32Array(lefts);
        output[TreeIo.Right] = new I32Array(rights);
        output[TreeIo.Parent] = new I32Array(parents);
        output[TreeIo.SplitCond] = new F32Array(conds);
        output[TreeIo.DftLeft] = new U8Array(dl);
        output[TreeIo.LossChg] = new F32Array(lossChg);
        output[TreeIo.SumHess] = new F32Array(sumHess);
    }
}
