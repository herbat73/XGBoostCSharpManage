namespace XGBoost.Tests;

/// <summary>Deterministic synthetic datasets.</summary>
internal static class TestData
{
    /// <summary>y = 3*x0 - 2*x1 + 0.5*x2 + noise.</summary>
    public static (float[] X, float[] Y) Regression(int rows, int cols = 4, int seed = 0)
    {
        var rng = new Random(seed);
        var x = new float[rows * cols];
        var y = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++) x[r * cols + c] = (float)rng.NextDouble();
            y[r] = 3 * x[r * cols] - 2 * x[r * cols + 1] + 0.5f * x[r * cols + 2] + (float)(rng.NextDouble() * 0.05);
        }
        return (x, y);
    }

    /// <summary>Label is 1 when x0 + x1 &gt; 1.</summary>
    public static (float[] X, float[] Y) Binary(int rows, int cols = 4, int seed = 1)
    {
        var (x, _) = Regression(rows, cols, seed);
        var y = new float[rows];
        for (var r = 0; r < rows; r++) y[r] = x[r * cols] + x[r * cols + 1] > 1 ? 1 : 0;
        return (x, y);
    }

    /// <summary>Label is the index of the largest of the first <paramref name="classes"/> columns.</summary>
    public static (float[] X, float[] Y) MultiClass(int rows, int classes, int cols = 4, int seed = 2)
    {
        var (x, _) = Regression(rows, cols, seed);
        var y = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            var best = 0;
            for (var c = 1; c < classes; c++) if (x[r * cols + c] > x[r * cols + best]) best = c;
            y[r] = best;
        }
        return (x, y);
    }

    public static DMatrix ToDMatrix(float[] x, float[] y, int cols)
    {
        var d = DMatrix.FromDense(x, y.Length, cols);
        d.Label = y;
        return d;
    }

    public static Dictionary<string, string> Params(params (string Key, string Value)[] items)
    {
        var p = new Dictionary<string, string> { ["seed"] = "0", ["nthread"] = "1", ["verbosity"] = "0" };
        foreach (var (k, v) in items) p[k] = v;
        return p;
    }
}
