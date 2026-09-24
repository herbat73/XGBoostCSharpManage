// Ports of src/common/row_set.h, src/common/partition_builder.h and
// src/tree/common_row_partitioner.h. The block-wise partition of the C++ code merges every block's
// left rows before every block's right rows, which is exactly a stable partition of the node.
using XGBoost.Predictors;

namespace XGBoost.Tree;

/// <summary><c>RowSetCollection</c>: row indices grouped by tree node.</summary>
public sealed class RowSetCollection
{
    /// <summary>A node's rows as a range of <see cref="Data"/>; <see cref="Begin"/> is -1 for "null".</summary>
    public struct Elem(long begin, long end, int nodeId)
    {
        public long Begin = begin;
        public long End = end;
        public int NodeId = nodeId;
        public readonly long Size => Begin < 0 ? 0 : End - Begin;
        public readonly bool IsNull => Begin < 0;
    }

    private readonly List<Elem> _elems = [];
    public long[] Data = [];

    public int Size => _elems.Count;
    public Elem this[int nodeId] => _elems[nodeId];
    public IReadOnlyList<Elem> Elems => _elems;

    public ReadOnlySpan<long> Rows(int nodeId)
    {
        var e = _elems[nodeId];
        return e.IsNull ? default : Data.AsSpan((int)e.Begin, (int)e.Size);
    }

    public void Clear() => _elems.Clear();

    public void Init()
    {
        Check.That(_elems.Count == 0);
        if (Data.Length == 0)
        {
            _elems.Add(new Elem(-1, -1, 0));
            return;
        }
        _elems.Add(new Elem(0, Data.Length, 0));
    }

    public void AddSplit(int nodeId, int leftNodeId, int rightNodeId, long nLeft, long nRight)
    {
        var e = _elems[nodeId];
        long begin = -1, end = -1;
        if (e.IsNull)
        {
            Check.Eq(nLeft, 0L);
            Check.Eq(nRight, 0L);
        }
        else
        {
            begin = e.Begin;
            end = e.End;
        }
        Check.Eq(nLeft + nRight, e.Size);
        while (_elems.Count <= Math.Max(leftNodeId, rightNodeId)) _elems.Add(new Elem(-1, -1, -1));
        _elems[leftNodeId] = begin < 0 ? new Elem(-1, -1, leftNodeId) : new Elem(begin, begin + nLeft, leftNodeId);
        _elems[rightNodeId] = begin < 0 ? new Elem(-1, -1, rightNodeId) : new Elem(begin + nLeft, end, rightNodeId);
        _elems[nodeId] = new Elem(-1, -1, -1);
    }
}

/// <summary>Split description the partitioner needs from an expand entry.</summary>
public interface IPartitionEntry
{
    int Nid { get; }
    float SplitValue { get; }
}

public sealed class CommonRowPartitioner
{
    private const int PartitionBlockSize = 2048;

    public long BaseRowId;
    private readonly RowSetCollection _rowSet = new();
    private readonly List<(long Left, long Right)> _sizes = [];

    public CommonRowPartitioner(Context ctx, long numRow, long baseRowId) => Reset(ctx, numRow, baseRowId);

    public void Reset(Context ctx, long numRow, long baseRowId)
    {
        BaseRowId = baseRowId;
        var rows = new long[numRow];
        Threading.ParallelForBlock(numRow, ctx.Threads(), blk =>
        {
            for (var i = blk.Begin; i < blk.End; ++i) rows[i] = i + baseRowId;
        });
        _rowSet.Data = rows;
        _rowSet.Clear();
        _rowSet.Init();
    }

    public RowSetCollection Partitions => _rowSet;
    public int Size => _rowSet.Size;
    public RowSetCollection.Elem this[int nidx] => _rowSet[nidx];

    public static void FindSplitConditions(IReadOnlyList<int> nodes, ITreeView tree, GHistIndexMatrix gmat, int[] splitConditions)
    {
        var ptrs = gmat.Cut.Ptrs.ConstHostSpan;
        var vals = gmat.Cut.Values.ConstHostSpan;
        for (var i = 0; i < nodes.Count; ++i)
        {
            var nidx = nodes[i];
            var fidx = (int)tree.SplitIndex(nidx);
            var splitPt = tree.SplitCond(nidx);
            var lowerBound = ptrs[fidx];
            var upperBound = ptrs[fidx + 1];
            var splitCond = -1;
            Check.Lt(upperBound, (uint)int.MaxValue);
            for (var bound = lowerBound; bound < upperBound; ++bound)
                if (splitPt == vals[(int)bound]) splitCond = (int)bound;
            splitConditions[i] = splitCond;
        }
    }

    /// <summary><c>UpdatePosition</c>: partitions the rows of every applied node into its children.</summary>
    /// <param name="splitValues">Split values of the candidates (used by the approx predicate).</param>
    public void UpdatePosition(Context ctx, GHistIndexMatrix gmat, IReadOnlyList<int> nodes, IReadOnlyList<float> splitValues, ITreeView tree)
    {
        var columnMatrix = gmat.Transpose();
        var nNodes = nodes.Count;
        int[]? splitConditions = null;
        if (columnMatrix.IsInitialized)
        {
            splitConditions = new int[nNodes];
            FindSplitConditions(nodes, tree, gmat, splitConditions);
        }
        Check.Eq(BaseRowId, gmat.BaseRowId);
        var anyMissing = columnMatrix.IsInitialized && columnMatrix.AnyMissing;
        var anyCat = gmat.Cut.HasCategorical;
        var cutValues = gmat.Cut.Values.ToArray();
        var data = _rowSet.Data;
        _sizes.Clear();
        var sizes = new (long, long)[nNodes];
        Threading.ParallelFor(nNodes, ctx.Threads(), Sched.Dyn(), ni =>
        {
            var nodeInSet = (int)ni;
            var nid = nodes[nodeInSet];
            var elem = _rowSet[nid];
            if (elem.IsNull || elem.Size == 0)
            {
                sizes[nodeInSet] = (0, 0);
                return;
            }
            var fid = (int)tree.SplitIndex(nid);
            var defaultLeft = tree.DefaultLeft(nid);
            var isCat = tree.SplitType(nid) == FeatureType.Categorical;
            var nodeCats = tree.NodeCats(nid).ToArray();
            var rows = data.AsSpan((int)elem.Begin, (int)elem.Size);
            var left = new long[rows.Length];
            var right = new long[rows.Length];
            long nLeft = 0, nRight = 0;

            if (!columnMatrix.IsInitialized)
            {
                var splitValue = splitValues[nodeInSet];
                foreach (var rid in rows)
                {
                    var gidx = gmat.GetGindex(rid, fid);
                    var goLeft = defaultLeft;
                    if (gidx > -1) goLeft = isCat ? Categorical.Decision(nodeCats, cutValues[gidx]) : cutValues[gidx] <= splitValue;
                    if (goLeft) left[nLeft++] = rid;
                    else right[nRight++] = rid;
                }
            }
            else
            {
                var splitCond = splitConditions![nodeInSet];
                bool Pred(long ridx, int binId)
                {
                    if (anyCat && isCat)
                    {
                        var gidx = gmat.GetGindex(ridx, fid);
                        var goLeft = defaultLeft;
                        if (gidx > -1) goLeft = Categorical.Decision(nodeCats, cutValues[gidx]);
                        return goLeft;
                    }
                    return binId <= splitCond;
                }

                // Rows of one partition task are visited in increasing order, which the sparse
                // column cursor relies on; the C++ code creates one cursor per task of 2048 rows.
                SparseColumnIter? sparse = null;
                var isDense = columnMatrix.GetColumnType(fid) == ColumnType.Dense;
                for (var r = 0; r < rows.Length; ++r)
                {
                    var rid = rows[r];
                    int binId;
                    if (isDense)
                    {
                        binId = columnMatrix.DenseBin(fid, rid - gmat.BaseRowId, anyMissing);
                    }
                    else
                    {
                        Check.That(anyMissing);
                        if (r % PartitionBlockSize == 0) sparse = columnMatrix.SparseColumn(fid, rid - gmat.BaseRowId);
                        binId = sparse!.Get(rid - gmat.BaseRowId);
                    }
                    bool goLeft;
                    if (anyMissing && binId == ColumnMatrix.MissingId) goLeft = defaultLeft;
                    else goLeft = Pred(rid, binId);
                    if (goLeft) left[nLeft++] = rid;
                    else right[nRight++] = rid;
                }
            }
            left.AsSpan(0, (int)nLeft).CopyTo(rows);
            right.AsSpan(0, (int)nRight).CopyTo(rows[(int)nLeft..]);
            sizes[nodeInSet] = (nLeft, nRight);
        });
        _sizes.AddRange(sizes);
        for (var i = 0; i < nNodes; ++i)
        {
            var nidx = nodes[i];
            Check.Eq(tree.LeftChild(nidx) + 1, tree.RightChild(nidx));
            _rowSet.AddSplit(nidx, tree.LeftChild(nidx), tree.RightChild(nidx), _sizes[i].Left, _sizes[i].Right);
        }
    }

    /// <summary><c>LeafPartition</c>: writes the (sample-encoded) leaf of every row.</summary>
    public void LeafPartition(Context ctx, ITreeView tree, Func<long, bool> invalidp, int[] position)
    {
        var rowSet = _rowSet;
        var data = rowSet.Data;
        Threading.ParallelFor(rowSet.Size, ctx.Threads(), i =>
        {
            var node = rowSet[(int)i];
            if (node.NodeId < 0) return;
            Check.That(tree.IsLeaf(node.NodeId));
            if (!node.IsNull)
            {
                for (var idx = node.Begin; idx != node.End; ++idx)
                {
                    var r = data[idx];
                    position[r] = SamplePosition.Encode(node.NodeId, !invalidp(r));
                }
            }
        });
    }

    public void LeafPartition(Context ctx, ITreeView tree, TensorView<GradientPair> gpair, int[] position)
    {
        if (gpair.Shape(1) > 1)
        {
            var nT = gpair.Shape(1);
            LeafPartition(ctx, tree, idx =>
            {
                for (long t = 0; t < nT; ++t)
                    if (!(gpair[idx, t].Hess - .0f == .0f)) return false;
                return true;
            }, position);
        }
        else
        {
            LeafPartition(ctx, tree, idx => gpair[idx, 0].Hess - .0f == .0f, position);
        }
    }

    public void LeafPartition(Context ctx, ITreeView tree, ReadOnlySpan<float> hess, int[] position)
    {
        var h = hess.ToArray();
        LeafPartition(ctx, tree, idx => h[idx] - .0f == .0f, position);
    }
}
