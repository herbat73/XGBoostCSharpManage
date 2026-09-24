[![](https://img.shields.io/badge/License-APACHE2-green.svg 'Apache 2 License')](https://opensource.org/license/apache-2.0)
[![Build](https://github.com/herbat73/XGBoostCSharpManage/actions/workflows/dotnet_build_and_test.yml/badge.svg)](https://github.com/herbat73/XGBoostCSharpManage/actions/workflows/dotnet_build_and_test.yml)

The C# port for the [XGBoost](https://github.com/dmlc/xgboost) C API (`include/xgboost/c_api.h`), targeting .NET 10.

It is fully managed .NET 10 (net10.0) code with no P/Invoke or native library of any kind. 

How it's tested: tests/XGBoost.Managed.Tests runs the existing csharp-package tests and their Python-generated fixtures unchanged, against the managed engine instead of xgboost.dll. All 13 Python parity cases match. They cover hist, approx, exact, DART, gblinear, categorical, multi-output, softprob, quantile, rank:ndcg, survival:aft, custom objective and early stopping. For each case the tests check predictions, SHAP contributions, leaf indices, eval metrics, the saved JSON/UBJ model and the text dumps. These tests always run with one thread (nthread=1). With more threads, histogram sums are added in a different order, so results can differ in the last bits. That is also true of the native library.

What's ported:
- Core: objectives, metrics, the CPU predictor and SHAP.
- Tree methods: the hist, approx and exact tree builders, plus prune and refresh.
- Boosters: gbtree (with DART), gblinear and the linear updaters.
- Learner: the full learner, including config, save/load and slicing.
- Public API: Booster, XGB.Train and the prediction types, with the same names as the bindings.
- Maths: erf, log1p, expm1 and lgamma are managed ports of FreeBSD's implementations. The random numbers, sorting and shuffling reproduce Microsoft's C++ standard library behaviour, which the reference build used.

Not ported:
- CUDA, NCCL and SYCL code, which can't become managed code.
- Distributed training: the port runs as a single process, so the cross-worker communication steps do nothing.
- External-memory data and QuantileDMatrix (the iterator/proxy data sources).
- There is no second test project comparing the managed engine against the native DLL directly; the Python parity tests do that job for now.

Getting DART to match needed one non-obvious fix. In C++, num_drop + lr adds a size_t to a float, so the result is a float, not a double.
That rounding changed the tree weights in the last bit, and the port now reproduces it.

## Example

```csharp
using XGBoost;

const string fileFormat = "?format=libsvm";

var outputModelFile = "model-0.json";
var pathToTrainData = "agaricus.txt.train";
var train = DMatrix.FromFile($"{pathToTrainData}{fileFormat}");
var pathToTestData = "agaricus.txt.test";
var test = DMatrix.FromFile($"{pathToTestData}{fileFormat}");

(DMatrix, string)[] watchlist = [(test, "eval"), (train, "train")];

var parameters = new Dictionary<string, string>
{
    ["objective"] = "binary:logistic",
    ["max_depth"] = "2",
    ["eta"] = "0.1",
};

using var booster = XGB.Train(parameters, train, numBoostRound: 200,
    evals: watchlist, earlyStoppingRounds: 10,
    onIteration: (i, r) => Console.WriteLine($"{i}: {string.Join(", ", r)}"));

booster.Save(outputModelFile);
```

More examples: [demos](demos/README.md) has C# ports of the Python demos in `demo/`
(`cd demos/XGBoost.Managed.Demos && dotnet run -- all`).

