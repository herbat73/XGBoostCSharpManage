using XGBoost.Demos.Common;

namespace XGBoost.Demos.GuidePython;

/// <summary>
/// Getting started with learning to rank. Port of the relevance-degree part of
/// <c>demo/guide-python/learning_to_rank.py</c>.
/// </summary>
/// <remarks>
/// Pass <c>--data &lt;root of MSLR-WEB10K&gt;</c> to train on Fold1 of the MSLR_10k_letor dataset, as the
/// Python demo does. Without it a synthetic dataset with the same layout is used. The Python demo's second
/// part (position debiasing on simulated click data) relies on helpers from <c>xgboost.testing</c> and is
/// not ported.
/// </remarks>
internal static class LearningToRank
{
    public static void Run(string[] args)
    {
        var dataRoot = new DemoArgs(args).Get("--data");
        SparseData train, test;
        if (dataRoot is null)
        {
            Console.WriteLine("No --data given; using a synthetic ranking dataset.");
            train = Synthetic(queries: 300, new Rng(0));
            test = Synthetic(queries: 100, new Rng(1));
        }
        else
        {
            // Use only the Fold1 for demo:
            // Train,      Valid, Test
            // {S1,S2,S3}, S4,    S5
            var foldPath = Path.Combine(Environment.ExpandEnvironmentVariables(dataRoot), "Fold1");
            train = Datasets.LoadLibSvm(Path.Combine(foldPath, "train.txt"));
            test = Datasets.LoadLibSvm(Path.Combine(foldPath, "test.txt"));
        }

        // Sort data according to query index
        var numCols = Math.Max(train.NumCols, test.NumCols);
        using var dtrain = SortedByQuery(train, numCols);
        using var dtest = SortedByQuery(test, numCols);

        var parameters = P.Of(
            ("objective", "rank:ndcg"),
            ("tree_method", "hist"),
            ("lambdarank_pair_method", "topk"),
            ("lambdarank_num_pair_per_sample", 13),
            ("eval_metric", "ndcg@1"),
            ("eval_metric", "ndcg@8"));
        using var ranker = Training.Train(parameters, dtrain, 100, evals: [(dtest, "validation_0")]);

        // Scores are only comparable within a query; show the ranking of the first test query.
        var scores = ranker.Predict(dtest).Values;
        var groupPtr = dtest.GetUIntInfo("group_ptr");
        var labels = dtest.Label;
        var firstQuery = Enumerable.Range((int)groupPtr[0], (int)(groupPtr[1] - groupPtr[0]))
            .OrderByDescending(i => scores[i]).Take(10);
        Console.WriteLine("Relevance of the top 10 documents of the first test query: " +
            string.Join(" ", firstQuery.Select(i => labels[i])));
    }

    /// <summary>Builds a DMatrix whose rows are sorted by query id, as XGBoost requires.</summary>
    private static DMatrix SortedByQuery(SparseData data, int numCols)
    {
        var order = Enumerable.Range(0, data.Rows).OrderBy(i => data.Qid[i]).ToArray(); // stable
        var indptr = new long[data.Rows + 1];
        var indices = new List<uint>(data.Indices.Length);
        var values = new List<float>(data.Values.Length);
        for (var r = 0; r < order.Length; r++)
        {
            var (begin, end) = ((int)data.Indptr[order[r]], (int)data.Indptr[order[r] + 1]);
            indices.AddRange(data.Indices.AsSpan(begin, end - begin));
            values.AddRange(data.Values.AsSpan(begin, end - begin));
            indptr[r + 1] = values.Count;
        }
        var d = DMatrix.FromCsr(indptr, [.. indices], [.. values], numCols);
        d.Label = data.Labels.Gather(order);
        d.SetQueryId(data.Qid.Gather(order));
        return d;
    }

    /// <summary>
    /// Queries with 20 documents each and relevance degrees 0-4 driven by a few of the 20 features. The
    /// queries are shuffled, like the unsorted qids in the MSLR files.
    /// </summary>
    private static SparseData Synthetic(int queries, Rng rng)
    {
        const int docs = 20;
        const int cols = 20;
        var weights = new[] { 1.0, -0.8, 0.6, 0.5, -0.3 };
        var qids = rng.Permutation(queries);
        var indptr = new List<long> { 0 };
        var indices = new List<uint>();
        var values = new List<float>();
        var labels = new List<float>();
        var qid = new List<uint>();
        foreach (var q in qids)
        {
            for (var d = 0; d < docs; d++)
            {
                double score = 0;
                for (var c = 0; c < cols; c++)
                {
                    var v = (float)rng.Normal();
                    indices.Add((uint)c);
                    values.Add(v);
                    if (c < weights.Length) score += weights[c] * v;
                }
                labels.Add((float)Math.Clamp(Math.Round(score + rng.Normal(0, 0.5) + 1.5), 0, 4));
                qid.Add((uint)q);
                indptr.Add(values.Count);
            }
        }
        return new SparseData([.. indptr], [.. indices], [.. values], [.. labels], [.. qid], cols);
    }
}
