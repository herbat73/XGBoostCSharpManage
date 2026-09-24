// Port of src/tree/tree_view.h/.cc: uniform read access to scalar and multi-target trees.
namespace XGBoost.Tree;

/// <summary>Read-only access to a tree, independent of the leaf type.</summary>
public interface ITreeView
{
    int Size { get; }
    uint NumTargets { get; }
    bool IsLeaf(int nidx);
    int Parent(int nidx);
    int LeftChild(int nidx);
    int RightChild(int nidx);
    uint SplitIndex(int nidx);
    float SplitCond(int nidx);
    bool DefaultLeft(int nidx);
    bool IsLeftChild(int nidx);
    bool IsRoot(int nidx);
    float SumHess(int nidx);
    float LossChg(int nidx);
    FeatureType SplitType(int nidx);
    ReadOnlySpan<uint> NodeCats(int nidx);
    bool HasCategoricalSplit { get; }
    CategoricalSplitMatrix GetCategoriesMatrix();

    /// <summary>Leaf value(s): a single value for scalar trees.</summary>
    ReadOnlySpan<float> LeafValue(int nidx);

    int DefaultChild(int nidx) => DefaultLeft(nidx) ? LeftChild(nidx) : RightChild(nidx);
}

public sealed class ScalarTreeView : ITreeView
{
    public readonly Node[] Nodes;
    public readonly RTreeNodeStat[] Stats;
    private readonly CategoricalSplitMatrix _cats;
    private readonly float[] _leafBuf = new float[1];

    public ScalarTreeView(RegTree tree)
    {
        Common.Check.That(!tree.IsMultiTarget);
        Nodes = tree.GetNodes().ToArray();
        Stats = tree.GetStats().ToArray();
        _cats = tree.GetCategoriesMatrix();
        Size = tree.NumNodes;
    }

    public int Size { get; }
    public uint NumTargets => 1;
    public bool IsLeaf(int nidx) => Nodes[nidx].IsLeaf;
    public int Parent(int nidx) => Nodes[nidx].Parent;
    public int LeftChild(int nidx) => Nodes[nidx].LeftChild;
    public int RightChild(int nidx) => Nodes[nidx].RightChild;
    public uint SplitIndex(int nidx) => Nodes[nidx].SplitIndex;
    public float SplitCond(int nidx) => Nodes[nidx].SplitCond;
    public bool DefaultLeft(int nidx) => Nodes[nidx].DefaultLeft;
    public bool IsLeftChild(int nidx) => Nodes[nidx].IsLeftChild;
    public bool IsRoot(int nidx) => Nodes[nidx].IsRoot;
    public bool IsDeleted(int nidx) => Nodes[nidx].IsDeleted;
    public float SumHess(int nidx) => Stats[nidx].SumHess;
    public float LossChg(int nidx) => Stats[nidx].LossChg;
    public ref readonly RTreeNodeStat Stat(int nidx) => ref Stats[nidx];
    public FeatureType SplitType(int nidx) => _cats.SplitType[nidx];
    public ReadOnlySpan<uint> NodeCats(int nidx) => _cats.NodeCats(nidx);
    public bool HasCategoricalSplit => _cats.Categories.Length != 0;
    public CategoricalSplitMatrix GetCategoriesMatrix() => _cats;
    public float ScalarLeafValue(int nidx) => Nodes[nidx].LeafValue;

    public ReadOnlySpan<float> LeafValue(int nidx)
    {
        _leafBuf[0] = Nodes[nidx].LeafValue;
        return _leafBuf;
    }

    public int DefaultChild(int nidx) => Nodes[nidx].DefaultChild;
}

public sealed class MultiTargetTreeView : ITreeView
{
    public readonly int[] Left;
    public readonly int[] Right;
    public readonly int[] ParentArr;
    public readonly uint[] SplitIndexArr;
    public readonly byte[] DefaultLeftArr;
    public readonly float[] SplitConds;
    public readonly float[] LeafWeights;
    public readonly float[] LossChgArr;
    public readonly float[] SumHessArr;
    private readonly int _nTargets;
    private readonly CategoricalSplitMatrix _cats;

    public MultiTargetTreeView(RegTree tree)
    {
        var mt = tree.GetMultiTargetTree();
        Left = [.. mt.Left];
        Right = [.. mt.Right];
        ParentArr = [.. mt.ParentArr];
        SplitIndexArr = [.. mt.SplitIndex];
        DefaultLeftArr = [.. mt.DefaultLeft];
        SplitConds = [.. mt.SplitConds];
        LeafWeights = mt.LeafWeights.ToArray();
        LossChgArr = [.. mt.LossChg];
        SumHessArr = [.. mt.SumHess];
        _nTargets = (int)mt.NumTargets;
        _cats = tree.GetCategoriesMatrix();
        Size = tree.NumNodes;
    }

    public int Size { get; }
    public uint NumTargets => (uint)_nTargets;
    public bool IsLeaf(int nidx) => Left[nidx] == MultiTargetTree.InvalidNodeId;
    public int Parent(int nidx) => ParentArr[nidx];
    public int LeftChild(int nidx) => Left[nidx];
    public int RightChild(int nidx) => Right[nidx];
    public bool IsLeftChild(int nidx) => nidx == LeftChild(Parent(nidx));
    public uint SplitIndex(int nidx) => SplitIndexArr[nidx];
    public float SplitCond(int nidx) => SplitConds[nidx];
    public bool DefaultLeft(int nidx) => DefaultLeftArr[nidx] != 0;
    public bool IsRoot(int nidx) => nidx == RegTree.Root;
    public float SumHess(int nidx) => SumHessArr[nidx];
    public float LossChg(int nidx) => LossChgArr[nidx];
    public FeatureType SplitType(int nidx) => _cats.SplitType[nidx];
    public ReadOnlySpan<uint> NodeCats(int nidx) => _cats.NodeCats(nidx);
    public bool HasCategoricalSplit => _cats.Categories.Length != 0;
    public CategoricalSplitMatrix GetCategoriesMatrix() => _cats;
    public ReadOnlySpan<float> LeafValue(int nidx) => LeafWeights.AsSpan(Right[nidx] * _nTargets, _nTargets);
}

public static class TreeViews
{
    /// <summary>Depth-first walk with an explicit stack (right child popped first), stops when <paramref name="fn"/> returns false.</summary>
    public static void WalkTree(ITreeView view, Func<int, bool> fn)
    {
        var nodes = new Stack<int>();
        nodes.Push(RegTree.Root);
        while (nodes.Count != 0)
        {
            var nidx = nodes.Pop();
            if (!fn(nidx)) return;
            if (!view.IsLeaf(nidx))
            {
                nodes.Push(view.LeftChild(nidx));
                nodes.Push(view.RightChild(nidx));
            }
        }
    }

    public static void WalkTree(RegTree tree, Func<ITreeView, int, bool> fn)
    {
        var view = tree.View();
        WalkTree(view, nidx => fn(view, nidx));
    }

    public static int GetDepth(ITreeView view, int nidx)
    {
        var depth = 0;
        while (!view.IsRoot(nidx))
        {
            depth++;
            nidx = view.Parent(nidx);
        }
        return depth;
    }

    public static int MaxDepth(ITreeView view, int nidx)
    {
        if (view.IsLeaf(nidx)) return 0;
        return Math.Max(MaxDepth(view, view.LeftChild(nidx)) + 1, MaxDepth(view, view.RightChild(nidx)) + 1);
    }
}
