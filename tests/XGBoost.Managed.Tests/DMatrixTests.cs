namespace XGBoost.Tests;

public class DMatrixTests
{
    [Fact]
    public void NativeLibraryLoads()
    {
        Assert.True(XGB.NativeVersion.Major >= 3);
        Assert.Contains("{", XGB.BuildInfo);
        Assert.Contains("verbosity", XGB.GlobalConfig);
    }

    [Fact]
    public void DenseShapeAndMissing()
    {
        float[,] x = { { 1, float.NaN, 3 }, { 4, 5, 6 } };
        using var d = DMatrix.FromDense(x);
        Assert.Equal(2, d.NumRows);
        Assert.Equal(3, d.NumCols);
        Assert.Equal(5, d.NumNonMissing);
    }

    [Fact]
    public void CustomMissingValue()
    {
        using var d = DMatrix.FromDense([1, -1, 3, -1], 2, 2, missing: -1);
        Assert.Equal(2, d.NumNonMissing);
    }

    [Fact]
    public void DenseRejectsWrongLength() =>
        Assert.Throws<ArgumentException>(() => DMatrix.FromDense([1, 2, 3], 2, 2));

    [Fact]
    public void Csr()
    {
        // [[1, 0, 2], [0, 0, 3]]
        using var d = DMatrix.FromCsr([0, 2, 3], [0, 2, 2], [1, 2, 3], numCols: 3);
        Assert.Equal(2, d.NumRows);
        Assert.Equal(3, d.NumCols);
        Assert.Equal(3, d.NumNonMissing);
    }

    [Fact]
    public void MetaInfoRoundTrip()
    {
        using var d = DMatrix.FromDense(new float[6], 3, 2);
        d.Label = [1, 2, 3];
        d.Weight = [0.5f, 1, 2];
        Assert.Equal([1f, 2, 3], d.Label);
        Assert.Equal([0.5f, 1, 2], d.Weight);

        d.SetGroup([1u, 2]);
        Assert.Equal([0u, 1, 3], d.GetUIntInfo("group_ptr"));
    }

    [Fact]
    public void FeatureNamesRoundTrip()
    {
        using var d = DMatrix.FromDense(new float[4], 2, 2);
        Assert.Empty(d.FeatureNames);
        d.FeatureNames = ["age", "income"];
        d.FeatureTypes = ["q", "q"];
        Assert.Equal(["age", "income"], d.FeatureNames);
        Assert.Equal(["q", "q"], d.FeatureTypes);
    }

    [Fact]
    public void SliceAndBinaryRoundTrip()
    {
        var (x, y) = TestData.Regression(10);
        using var d = TestData.ToDMatrix(x, y, 4);
        using var s = d.Slice([1, 3, 5]);
        Assert.Equal(3, s.NumRows);
        Assert.Equal([y[1], y[3], y[5]], s.Label);

        var path = Path.Combine(Path.GetTempPath(), $"xgb-{Guid.NewGuid():N}.buffer");
        try
        {
            s.SaveBinary(path);
            using var loaded = DMatrix.FromFile(path);
            Assert.Equal(3, loaded.NumRows);
            Assert.Equal(s.Label, loaded.Label);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LoadsCsv()
    {
        var path = Path.Combine(Path.GetTempPath(), $"xgb-{Guid.NewGuid():N}.csv");
        File.WriteAllText(path, "1,0.5,2\n0,1.5,3\n");
        try
        {
            using var d = DMatrix.FromFile(path + "?format=csv&label_column=0");
            Assert.Equal(2, d.NumRows);
            Assert.Equal(2, d.NumCols);
            Assert.Equal([1f, 0], d.Label);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // [Fact]
    // public void NativeErrorsBecomeExceptions()
    // {
    //     var ex = Assert.Throws<XgBoostException>(() => DMatrix.FromFile("does-not-exist.buffer"));
    //     Assert.False(string.IsNullOrWhiteSpace(ex.Message));
    // }
}
