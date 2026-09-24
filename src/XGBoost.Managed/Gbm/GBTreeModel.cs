// Port of src/gbm/gbtree_model.h/.cc.
using XGBoost.Tree;

namespace XGBoost.Gbm;

public sealed class GBTreeModelParam : XGBoostParameter<GBTreeModelParam>
{
    public int NumTrees;
    public int NumParallelTree = 1;

    protected override void Declare(ParamManager<GBTreeModelParam> m)
    {
        Field(m, "num_trees", p => p.NumTrees, (p, v) => p.NumTrees = v).SetLowerBound(0).SetDefault(0)
            .Describe("Number of trees for the entire booster model.");
        Field(m, "num_parallel_tree", p => p.NumParallelTree, (p, v) => p.NumParallelTree = v).SetDefault(1).SetLowerBound(1)
            .Describe("Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.");
    }
}

public sealed class GBTreeModel(LearnerModelState learnerModelState, Context ctx)
{
    public LearnerModelState LearnerModelState { get; } = learnerModelState;
    public GBTreeModelParam Param = new();
    public List<RegTree> Trees = [];
    public List<RegTree> TreesToUpdate = [];
    /// <summary>Group index for trees.</summary>
    public List<uint> TreeInfo = [];
    /// <summary>Number of trees accumulated for each iteration.</summary>
    public List<int> IterationIndptr = [0];
    /// <summary>Per-tree weights. An empty list represents unit weights.</summary>
    public List<float> WeightDrop = [];
    private CatContainer _cats = new();

    public Context Ctx => ctx;

    public SortedSet<string> Configure(Args cfg)
    {
        if (Trees.Count == 0) return ParameterUtils.GetUsedParameters(cfg, Param.UpdateAllowUnknown(cfg));
        return new SortedSet<string>(StringComparer.Ordinal);
    }

    public List<float>? TreeWeights => WeightDrop.Count == 0 ? null : WeightDrop;

    public CatContainer Cats() => _cats;
    public void Cats(CatContainer cats) => _cats = cats;

    public ReadOnlySpan<uint> TreeGroups => System.Runtime.InteropServices.CollectionsMarshal.AsSpan(TreeInfo);

    public int BoostedRounds
    {
        get
        {
            if (Trees.Count == 0) Check.Eq(IterationIndptr.Count, 1);
            return IterationIndptr.Count - 1;
        }
    }

    public List<string> DumpModel(FeatureMap fmap, bool withStats, int nThreads, string format)
    {
        var dump = new string[Trees.Count];
        Threading.ParallelFor(Trees.Count, nThreads, i => dump[i] = Trees[(int)i].DumpModel(fmap, withStats, format));
        return [.. dump];
    }

    private void MakeIndptr()
    {
        if (TreeInfo.Count == 0) return;
        var nGroups = (int)TreeInfo.Max() + 1;
        var layerTrees = Param.NumParallelTree * nGroups;
        Check.Ne(layerTrees, 0);
        var n = Param.NumTrees / layerTrees + 1;
        IterationIndptr = [.. new int[n]];
        IterationIndptr[0] = 0;
        for (var i = 1; i < n; ++i) IterationIndptr[i] = nGroups * Param.NumParallelTree;
        for (var i = 1; i < n; ++i) IterationIndptr[i] += IterationIndptr[i - 1];
    }

    private void Validate()
    {
        Check.Eq(Trees.Count, Param.NumTrees);
        Check.Eq(TreeInfo.Count, Param.NumTrees);
        Check.Eq(IterationIndptr[^1], Param.NumTrees);
        Check.Le(WeightDrop.Count, Trees.Count);
    }

    public void SaveModel(JsonObject output)
    {
        Check.Eq(Param.NumTrees, Trees.Count);
        Check.Le(WeightDrop.Count, Trees.Count);
        output["gbtree_model_param"] = Param.ToJson();
        var treesJson = new Json[Trees.Count];
        Threading.ParallelFor(Trees.Count, ctx.Threads(), t =>
        {
            var jtree = new JsonObject();
            Trees[(int)t].SaveModel(jtree);
            jtree["id"] = new JsonInteger(t);
            treesJson[t] = jtree;
        });
        output["trees"] = new JsonArray(treesJson);
        output["tree_info"] = new JsonArray(TreeInfo.Select(v => (Json)new JsonInteger(v)));
        output["iteration_indptr"] = new JsonArray(IterationIndptr.Select(v => (Json)new JsonInteger(v)));
        if (WeightDrop.Count != 0) output["weight_drop"] = new JsonArray(WeightDrop.Select(v => (Json)new JsonNumber(v)));
        var cats = new JsonObject();
        _cats.Save(cats);
        output["cats"] = cats;
    }

    public void LoadModel(Json input)
    {
        Param.FromJson(input["gbtree_model_param"]);
        Trees.Clear();
        TreesToUpdate.Clear();
        var jmodel = input.AsObject;
        var treesJson = jmodel["trees"].AsArray;
        Check.Eq(treesJson.Count, Param.NumTrees);
        var trees = new RegTree[Param.NumTrees];
        var treeInfoJson = jmodel["tree_info"].AsArray;
        Check.Eq(treeInfoJson.Count, Param.NumTrees);
        Threading.ParallelFor(Param.NumTrees, ctx.Threads(), t =>
        {
            var treeId = treesJson[(int)t]["id"].AsInteger;
            Check.Eq(treeId, t);
            var tree = new RegTree();
            tree.LoadModel(treesJson[(int)t]);
            trees[treeId] = tree;
        });
        Trees = [.. trees];
        TreeInfo = [.. treeInfoJson.Select(v => (uint)v.AsInteger)];
        IterationIndptr = [];
        if (jmodel.TryGetValue("iteration_indptr", out var indptr)) IterationIndptr = [.. indptr.AsArray.Select(v => (int)v.AsInteger)];
        else MakeIndptr();
        WeightDrop = [];
        if (jmodel.TryGetValue("weight_drop", out var wd)) WeightDrop = [.. wd.AsArray.Select(v => v.AsNumber)];
        var pCats = new CatContainer();
        if (jmodel.TryGetValue("cats", out var jcats)) pCats.Load(jcats);
        _cats = pCats;
        Validate();
    }

    /// <summary>Adds trees to the model, returns the number of new trees.</summary>
    public int CommitModel(List<List<RegTree>> newTrees)
    {
        Check.That(IterationIndptr.Count != 0);
        Check.Eq(IterationIndptr[^1], Param.NumTrees);
        var nNewTrees = 0;
        if (LearnerModelState.IsVectorLeaf)
        {
            nNewTrees += newTrees[0].Count;
            CommitModelGroup(newTrees[0], 0);
        }
        else
        {
            for (uint gidx = 0; gidx < LearnerModelState.OutputLength; ++gidx)
            {
                nNewTrees += newTrees[(int)gidx].Count;
                CommitModelGroup(newTrees[(int)gidx], gidx);
            }
        }
        IterationIndptr.Add(nNewTrees + IterationIndptr[^1]);
        Validate();
        return nNewTrees;
    }

    public void CommitModelGroup(List<RegTree> newTrees, uint groupIdx)
    {
        foreach (var t in newTrees)
        {
            Trees.Add(t);
            TreeInfo.Add(groupIdx);
        }
        Param.NumTrees += newTrees.Count;
    }

    /// <summary>Moves existing trees into the update queue.</summary>
    public void InitTreesToUpdate()
    {
        if (TreesToUpdate.Count == 0)
        {
            TreesToUpdate.AddRange(Trees);
            Trees.Clear();
            Param.NumTrees = 0;
            TreeInfo.Clear();
            IterationIndptr.Clear();
            IterationIndptr.Add(0);
        }
    }
}
