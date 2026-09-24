// Port of src/common/column_matrix.h/.cc: column-major view of the gradient index used by the
// hist row partitioner.
using System.Runtime.CompilerServices;
using XGBoost.Data;

namespace XGBoost.Common;

public enum ColumnType : byte
{
    Dense = 0,
    Sparse = 1,
}

/// <summary>Iterator over a sparse column with a moving cursor, <c>SparseColumnIter</c>.</summary>
public sealed class SparseColumnIter
{
    private readonly ColumnMatrix _m;
    private readonly long _featureOffset;
    private readonly long _columnSize;
    private readonly int _indexBase;
    private long _idx;

    public SparseColumnIter(ColumnMatrix m, int fidx, long firstRowIdx)
    {
        _m = m;
        _featureOffset = m.FeatureOffsets[fidx];
        _columnSize = m.FeatureOffsets[fidx + 1] - _featureOffset;
        _indexBase = (int)m.IndexBase[fidx];
        // lower_bound on the (sorted) row indices.
        long lo = 0, count = _columnSize;
        while (count > 0)
        {
            var step = count >> 1;
            var mid = lo + step;
            if (m.RowInd[_featureOffset + mid] < firstRowIdx)
            {
                lo = mid + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        _idx = lo;
    }

    public long Size => _columnSize;
    public long GetRowIdx(long idx) => _m.RowInd[_featureOffset + idx];
    public int GetGlobalBinIdx(long idx) => _indexBase + (int)_m.GetStored(_featureOffset + idx);

    /// <summary><c>operator[]</c>: bin of row <paramref name="rid"/> (rows must be visited in increasing order).</summary>
    public int Get(long rid)
    {
        if (!(_idx < _columnSize)) return ColumnMatrix.MissingId;
        while (_idx < _columnSize && GetRowIdx(_idx) < rid) _idx++;
        if (_idx < _columnSize && GetRowIdx(_idx) == rid) return GetGlobalBinIdx(_idx);
        return ColumnMatrix.MissingId;
    }
}

/// <summary>Column-major gradient index with dense and sparse columns, <c>ColumnMatrix</c>.</summary>
public sealed class ColumnMatrix
{
    public const int MissingId = -1;

    private byte[] _index = [];
    private ColumnType[] _type = [];
    internal long[] RowInd = [];
    internal long[] FeatureOffsets = [];
    private long[] _numNonzeros = [];
    internal uint[] IndexBase = [];
    private uint[] _missing = [];
    private BinTypeSize _binsTypeSize;
    private bool _anyMissing;

    public ColumnMatrix() { }

    public ColumnMatrix(GHistIndexMatrix gmat, double sparseThreshold) => InitStorage(gmat, sparseThreshold);

    public int GetNumFeature => _type.Length;
    public bool IsInitialized => _type.Length != 0;
    public BinTypeSize GetTypeSize => _binsTypeSize;
    public ColumnType GetColumnType(int fidx) => _type[fidx];
    public bool AnyMissing => _anyMissing;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal uint GetStored(long pos) => _binsTypeSize switch
    {
        BinTypeSize.Uint8 => _index[pos],
        BinTypeSize.Uint16 => Unsafe.ReadUnaligned<ushort>(ref _index[pos * 2]),
        _ => Unsafe.ReadUnaligned<uint>(ref _index[pos * 4]),
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SetStored(long pos, uint v)
    {
        switch (_binsTypeSize)
        {
            case BinTypeSize.Uint8: _index[pos] = (byte)v; break;
            case BinTypeSize.Uint16: Unsafe.WriteUnaligned(ref _index[pos * 2], (ushort)v); break;
            default: Unsafe.WriteUnaligned(ref _index[pos * 4], v); break;
        }
    }

    public bool IsMissing(long pos) => LBitField32.Check(_missing, pos);

    private void SetValid(long pos) => new LBitField32(_missing).Clear(pos);

    /// <summary>Bin of row <paramref name="ridx"/> in dense column <paramref name="fidx"/> (<c>DenseColumnIter::operator[]</c>).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int DenseBin(int fidx, long ridx, bool anyMissing)
    {
        var featureOffset = FeatureOffsets[fidx];
        if (anyMissing && IsMissing(featureOffset + ridx)) return MissingId;
        return (int)IndexBase[fidx] + (int)GetStored(featureOffset + ridx);
    }

    public SparseColumnIter SparseColumn(int fidx, long firstRowIdx) => new(this, fidx, firstRowIdx);

    private void GrowMissing(long nElements, bool init)
    {
        var mSize = LBitField32.ComputeStorageSize(nElements);
        Check.Ge(mSize, _missing.Length);
        if (mSize == _missing.Length) return;
        var old = _missing.Length;
        Array.Resize(ref _missing, mSize);
        if (init) _missing.AsSpan(old).Fill(uint.MaxValue);
    }

    private void InitStorage(GHistIndexMatrix gmat, double sparseThreshold)
    {
        var nfeature = gmat.Features;
        var nrow = gmat.Size;
        _type = new ColumnType[nfeature];
        var featureCounts = new long[nfeature];
        gmat.GetFeatureCounts(featureCounts);
        var allDense = true;
        for (var fid = 0; fid < nfeature; fid++)
        {
            if (featureCounts[fid] < sparseThreshold * nrow)
            {
                _type[fid] = ColumnType.Sparse;
                allDense = false;
            }
            else
            {
                _type[fid] = ColumnType.Dense;
            }
        }
        FeatureOffsets = new long[nfeature + 1];
        long accum = 0;
        for (var fid = 1; fid < nfeature + 1; fid++)
        {
            accum += _type[fid - 1] == ColumnType.Dense ? nrow : featureCounts[fid - 1];
            FeatureOffsets[fid] = accum;
        }
        SetTypeSize(gmat.MaxNumBinPerFeat);
        _index = new byte[FeatureOffsets[^1] * (int)_binsTypeSize];
        if (!allDense) RowInd = new long[FeatureOffsets[nfeature]];
        IndexBase = gmat.Cut.Ptrs.ToArray();
        _anyMissing = !gmat.IsDense;
        _missing = [];
    }

    private void SetTypeSize(int maxBinPerFeat)
    {
        if (maxBinPerFeat - 1 <= byte.MaxValue) _binsTypeSize = BinTypeSize.Uint8;
        else if (maxBinPerFeat - 1 <= ushort.MaxValue) _binsTypeSize = BinTypeSize.Uint16;
        else _binsTypeSize = BinTypeSize.Uint32;
    }

    public void InitFromSparse(SparsePage page, GHistIndexMatrix gmat, double sparseThreshold, int nThreads)
    {
        var batch = new SparsePageAdapterBatch(page.GetView());
        InitStorage(gmat, sparseThreshold);
        PushBatch(nThreads, batch, float.NaN, gmat, 0);
    }

    public void InitFromGHist(Context ctx, GHistIndexMatrix gmat)
    {
        if (!_anyMissing) SetIndexNoMissing(gmat.BaseRowId, gmat.Index, gmat.Size, gmat.Features, ctx.Threads());
        else SetIndexMixedColumns(gmat);
    }

    public void PushBatch<TBatch>(int nThreads, TBatch batch, float missing, GHistIndexMatrix gmat, long baseRowId) where TBatch : IAdapterBatch
    {
        if (!_anyMissing) SetIndexNoMissing(baseRowId, gmat.Index, batch.Size, gmat.Features, nThreads);
        else SetIndexMixedColumns(baseRowId, batch, gmat, missing);
    }

    private void SetBinSparse(uint binId, long rid, int fid)
    {
        if (_type[fid] == ColumnType.Dense)
        {
            SetStored(FeatureOffsets[fid] + rid, binId - IndexBase[fid]);
            SetValid(FeatureOffsets[fid] + rid);
        }
        else
        {
            SetStored(FeatureOffsets[fid] + _numNonzeros[fid], binId - IndexBase[fid]);
            RowInd[FeatureOffsets[fid] + _numNonzeros[fid]] = rid;
            _numNonzeros[fid]++;
        }
    }

    private void SetIndexNoMissing(long baseRowId, GHistIndex rowIndex, long nSamples, int nFeatures, int nThreads)
    {
        GrowMissing(FeatureOffsets[nFeatures], false);
        // The row index is compressed: read the stored (feature-local) values.
        var raw = rowIndex.Raw;
        var rowType = rowIndex.BinTypeSize;
        uint RawAt(long i) => rowType switch
        {
            BinTypeSize.Uint8 => raw[i],
            BinTypeSize.Uint16 => Unsafe.ReadUnaligned<ushort>(ref raw[i * 2]),
            _ => Unsafe.ReadUnaligned<uint>(ref raw[i * 4]),
        };
        Threading.ParallelFor(nSamples, nThreads, r =>
        {
            var rid = r + baseRowId;
            var ibegin = rid * nFeatures;
            var iend = (rid + 1) * nFeatures;
            for (long i = ibegin, j = 0; i < iend; i++, j++)
            {
                var idx = FeatureOffsets[j];
                SetStored(idx + rid, RawAt(i));
            }
        });
    }

    private void SetIndexMixedColumns<TBatch>(long baseRowId, TBatch batch, GHistIndexMatrix gmat, float missing) where TBatch : IAdapterBatch
    {
        var nFeatures = gmat.Features;
        GrowMissing(FeatureOffsets[nFeatures], true);
        var rowBase = gmat.RowPtr[baseRowId];
        if (_numNonzeros.Length == 0) _numNonzeros = new long[nFeatures];
        else Check.Eq(_numNonzeros.Length, nFeatures);
        var isValid = new IsValidFunctor(missing);
        long k = 0;
        for (long rid = 0; rid < batch.Size; rid++)
        {
            var n = batch.LineSize(rid);
            for (var i = 0; i < n; i++)
            {
                var coo = batch.GetElement(rid, i);
                if (!isValid.Valid(coo.Value)) continue;
                var binId = gmat.Index[rowBase + k];
                SetBinSparse(binId, rid + baseRowId, (int)coo.ColumnIdx);
                k++;
            }
        }
    }

    private void SetIndexMixedColumns(GHistIndexMatrix gmat)
    {
        var nFeatures = gmat.Features;
        _missing = new uint[LBitField32.ComputeStorageSize(FeatureOffsets[nFeatures])];
        _missing.AsSpan().Fill(uint.MaxValue);
        _numNonzeros = new long[nFeatures];
        Check.That(_anyMissing);
        gmat.AssignColumnBinIndex((binIdx, _, ridx, fidx) => SetBinSparse(binIdx, ridx, fidx));
    }

    public void Write(BinaryWriter fo)
    {
        fo.Write(_index.Length);
        fo.Write(_index);
        fo.Write(_type.Length);
        foreach (var t in _type) fo.Write((byte)t);
        fo.Write(RowInd.Length);
        foreach (var v in RowInd) fo.Write(v);
        fo.Write(FeatureOffsets.Length);
        foreach (var v in FeatureOffsets) fo.Write(v);
        fo.Write(_missing.Length);
        foreach (var v in _missing) fo.Write(v);
        fo.Write((byte)_binsTypeSize);
        fo.Write(_anyMissing);
    }

    public void Read(BinaryReader fi, uint[] indexBase)
    {
        _index = fi.ReadBytes(fi.ReadInt32());
        _type = new ColumnType[fi.ReadInt32()];
        for (var i = 0; i < _type.Length; i++) _type[i] = (ColumnType)fi.ReadByte();
        RowInd = new long[fi.ReadInt32()];
        for (var i = 0; i < RowInd.Length; i++) RowInd[i] = fi.ReadInt64();
        FeatureOffsets = new long[fi.ReadInt32()];
        for (var i = 0; i < FeatureOffsets.Length; i++) FeatureOffsets[i] = fi.ReadInt64();
        _missing = new uint[fi.ReadInt32()];
        for (var i = 0; i < _missing.Length; i++) _missing[i] = fi.ReadUInt32();
        IndexBase = indexBase;
        _binsTypeSize = (BinTypeSize)fi.ReadByte();
        _anyMissing = fi.ReadBoolean();
    }
}
