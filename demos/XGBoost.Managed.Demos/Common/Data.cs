using System.Globalization;

namespace XGBoost.Demos.Common;

/// <summary>A row-major dense matrix, the demos' stand-in for a 2-D NumPy array.</summary>
internal sealed record DenseData(float[] Values, int Rows, int Cols)
{
    public static DenseData Zeros(int rows, int cols) => new(new float[rows * cols], rows, cols);

    public float this[int row, int col]
    {
        get => Values[row * Cols + col];
        set => Values[row * Cols + col] = value;
    }

    public Span<float> Row(int row) => Values.AsSpan(row * Cols, Cols);

    /// <summary>Copies the given rows into a new matrix (NumPy <c>X[idx]</c>).</summary>
    public DenseData TakeRows(IReadOnlyList<int> rows)
    {
        var result = Zeros(rows.Count, Cols);
        for (var i = 0; i < rows.Count; i++) Row(rows[i]).CopyTo(result.Row(i));
        return result;
    }

    /// <summary>The first <paramref name="n"/> rows.</summary>
    public DenseData Head(int n) => new(Values[..(n * Cols)], n, Cols);

    /// <summary>Copies the given columns into a new matrix (NumPy <c>X[:, idx]</c>).</summary>
    public DenseData TakeColumns(IReadOnlyList<int> cols)
    {
        var result = Zeros(Rows, cols.Count);
        for (var r = 0; r < Rows; r++)
            for (var c = 0; c < cols.Count; c++) result[r, c] = this[r, cols[c]];
        return result;
    }

    public float[] Column(int col)
    {
        var result = new float[Rows];
        for (var r = 0; r < Rows; r++) result[r] = this[r, col];
        return result;
    }

    public DMatrix ToDMatrix(float[]? label = null, float missing = float.NaN, int nthread = 0)
    {
        var d = DMatrix.FromDense(Values, Rows, Cols, missing, nthread);
        if (label is not null) d.Label = label;
        return d;
    }
}

/// <summary>A CSR matrix with labels (and query ids, when the file has them), as read from a libsvm file.</summary>
internal sealed record SparseData(long[] Indptr, uint[] Indices, float[] Values, float[] Labels, uint[] Qid, int NumCols)
{
    public int Rows => Labels.Length;

    /// <param name="numCols">Number of columns, when it must be wider than this file (e.g. a test set).</param>
    public DMatrix ToDMatrix(int? numCols = null)
    {
        var d = DMatrix.FromCsr(Indptr, Indices, Values, numCols ?? NumCols);
        d.Label = Labels;
        return d;
    }
}

internal static class ArrayExtensions
{
    /// <summary>NumPy-style fancy indexing: <c>a[idx]</c>.</summary>
    public static T[] Gather<T>(this T[] source, IReadOnlyList<int> indices)
    {
        var result = new T[indices.Count];
        for (var i = 0; i < indices.Count; i++) result[i] = source[indices[i]];
        return result;
    }
}

/// <summary>Data loading, splitting and synthetic datasets used in place of scikit-learn's.</summary>
internal static class Datasets
{
    /// <summary>Shuffled train/test split, like <c>sklearn.model_selection.train_test_split</c>.</summary>
    public static (int[] Train, int[] Test) TrainTestSplit(int rows, double testSize, Rng rng)
    {
        var nTest = (int)Math.Ceiling(testSize * rows);
        var perm = rng.Permutation(rows);
        return (perm[nTest..], perm[..nTest]);
    }

    /// <summary>Shuffled K-fold splits, like <c>sklearn.model_selection.KFold(shuffle=True)</c>.</summary>
    public static IEnumerable<(int[] Train, int[] Test)> KFold(int rows, int folds, Rng rng)
    {
        var perm = rng.Permutation(rows);
        var start = 0;
        for (var k = 0; k < folds; k++)
        {
            var size = rows / folds + (k < rows % folds ? 1 : 0);
            var test = perm[start..(start + size)];
            var train = perm[..start].Concat(perm[(start + size)..]).ToArray();
            start += size;
            yield return (train, test);
        }
    }

    /// <summary>
    /// Random linear regression problem, like <c>sklearn.datasets.make_regression</c>: standard normal
    /// features, and targets that are a linear combination of <paramref name="informative"/> of them.
    /// </summary>
    /// <returns>Features and row-major targets of shape [rows, targets].</returns>
    public static (DenseData X, float[] Y) MakeRegression(int rows, int cols, Rng rng, int informative = 10,
        int targets = 1, double noise = 0)
    {
        informative = Math.Min(informative, cols);
        var x = new DenseData(rng.NormalArray(rows * cols), rows, cols);
        var coef = new double[informative * targets];
        for (var i = 0; i < coef.Length; i++) coef[i] = 100 * rng.Uniform();

        var y = new float[rows * targets];
        for (var r = 0; r < rows; r++)
        {
            for (var t = 0; t < targets; t++)
            {
                double sum = 0;
                for (var c = 0; c < informative; c++) sum += x[r, c] * coef[c * targets + t];
                y[r * targets + t] = (float)(sum + (noise > 0 ? rng.Normal(0, noise) : 0));
            }
        }
        return (x, y);
    }

    /// <summary>
    /// Binary problem from Hastie et al. (2009), like <c>sklearn.datasets.make_hastie_10_2</c>, with the
    /// labels already mapped from {-1, 1} to {0, 1}.
    /// </summary>
    public static (DenseData X, float[] Y) MakeHastie(int rows, Rng rng)
    {
        const int cols = 10;
        var x = new DenseData(rng.NormalArray(rows * cols), rows, cols);
        var y = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            double sum = 0;
            foreach (var v in x.Row(r)) sum += v * v;
            y[r] = sum > 9.34 ? 1 : 0;
        }
        return (x, y);
    }

    /// <summary>
    /// Noisy binary classification problem, used where the Python demos load scikit-learn's breast
    /// cancer dataset (569 rows, 30 features).
    /// </summary>
    public static (DenseData X, float[] Y) BinaryClassification(Rng rng, int rows = 569, int cols = 30)
    {
        var x = new DenseData(rng.NormalArray(rows * cols), rows, cols);
        var w = new double[cols];
        for (var c = 0; c < Math.Min(cols, 8); c++) w[c] = rng.Normal();
        var y = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            double logit = 0;
            for (var c = 0; c < cols; c++) logit += w[c] * x[r, c];
            logit += 0.5 * x[r, 0] * x[r, 1];
            y[r] = logit + rng.Normal(0, 0.5) > 0 ? 1 : 0;
        }
        return (x, y);
    }

    /// <summary>
    /// Gaussian clusters, one per class, used where the Python demos load scikit-learn's iris or digits
    /// datasets.
    /// </summary>
    public static (DenseData X, float[] Y) Blobs(int rows, int cols, int classes, Rng rng, double spread = 2.5)
    {
        var centers = new double[classes * cols];
        for (var i = 0; i < centers.Length; i++) centers[i] = rng.Uniform(-5, 5);
        var x = DenseData.Zeros(rows, cols);
        var y = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            var k = r % classes;
            y[r] = k;
            for (var c = 0; c < cols; c++) x[r, c] = (float)rng.Normal(centers[k * cols + c], spread);
        }
        return (x, y);
    }

    /// <summary>
    /// Reads a libsvm text file into CSR form, like <c>sklearn.datasets.load_svmlight_file</c>: when no
    /// column index is 0 the file is taken to be one-based and indices are shifted down by one. (XGBoost's
    /// own parser, <c>DMatrix.FromFile("...?format=libsvm")</c>, keeps them as written.)
    /// </summary>
    public static SparseData LoadLibSvm(string path)
    {
        var indptr = new List<long> { 0 };
        var indices = new List<uint>();
        var values = new List<float>();
        var labels = new List<float>();
        var qid = new List<uint>();
        var numCols = 0;
        foreach (var line in File.ReadLines(path))
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) continue;
            labels.Add(float.Parse(parts[0], CultureInfo.InvariantCulture));
            foreach (var part in parts.AsSpan(1))
            {
                var colon = part.IndexOf(':');
                if (part.StartsWith("qid:", StringComparison.Ordinal))
                {
                    qid.Add(uint.Parse(part.AsSpan(colon + 1), CultureInfo.InvariantCulture));
                    continue;
                }
                var index = uint.Parse(part.AsSpan(0, colon), CultureInfo.InvariantCulture);
                indices.Add(index);
                values.Add(float.Parse(part.AsSpan(colon + 1), CultureInfo.InvariantCulture));
                numCols = Math.Max(numCols, (int)index + 1);
            }
            indptr.Add(values.Count);
        }
        if (indices.Count > 0 && !indices.Contains(0u))
        {
            for (var i = 0; i < indices.Count; i++) indices[i]--;
            numCols--;
        }
        return new SparseData([.. indptr], [.. indices], [.. values], [.. labels], [.. qid], numCols);
    }

    /// <summary>
    /// Reads a numeric CSV file. <c>inf</c>/<c>-inf</c> are parsed as infinities and empty cells or
    /// <c>?</c> as NaN.
    /// </summary>
    public static (string[] Header, DenseData Data) LoadCsv(string path, bool hasHeader = true)
    {
        var lines = File.ReadLines(path).Where(l => l.Length > 0).ToList();
        var header = hasHeader ? lines[0].Split(',') : [];
        var body = hasHeader ? lines.Skip(1).ToList() : lines;
        var cols = body[0].Split(',').Length;
        var data = DenseData.Zeros(body.Count, cols);
        for (var r = 0; r < body.Count; r++)
        {
            var cells = body[r].Split(',');
            for (var c = 0; c < cols; c++) data[r, c] = ParseCell(cells[c].Trim());
        }
        return (header, data);
    }

    private static float ParseCell(string cell) => cell.ToLowerInvariant() switch
    {
        "inf" or "+inf" or "infinity" => float.PositiveInfinity,
        "-inf" or "-infinity" => float.NegativeInfinity,
        "" or "?" or "nan" => float.NaN,
        _ => float.Parse(cell, CultureInfo.InvariantCulture),
    };
}
