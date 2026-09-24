"""Generate reference fixtures for the C# <-> Python parity tests.

Each case trains a model with the Python package and records the inputs, the exact
parameter strings sent to the native library, and every output the C# tests compare:
evaluation history, predictions, feature scores, model dumps and saved models.

Run from the repository root against the in-tree package and native library, so both
bindings exercise the same ``lib/xgboost.dll``::

    PYTHONPATH=python-package python csharp-package/tests/XGBoost.Tests/Parity/generate_fixtures.py

Re-run it whenever the native library changes in a way that alters results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse
import xgboost as xgb

OUT = Path(__file__).resolve().parent / "Fixtures"

N_TRAIN, N_VALID, N_COLS = 300, 100, 6


# ---- Data ---------------------------------------------------------------------------


def features(rng: np.random.Generator, n: int, missing: float = 0.0) -> np.ndarray:
    x = rng.random((n, N_COLS), dtype=np.float32)
    if missing > 0:
        x[rng.random((n, N_COLS)) < missing] = np.nan
    return x


def linear_target(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    x0 = np.nan_to_num(x, nan=0.5)
    y = 3 * x0[:, 0] - 2 * x0[:, 1] + x0[:, 2] * x0[:, 3] + 0.1 * rng.standard_normal(len(x))
    return y.astype(np.float32)


# ---- Fixture writing ----------------------------------------------------------------


class Writer:
    """Writes little-endian binary arrays and a JSON manifest for one case."""

    def __init__(self, name: str) -> None:
        self.dir = OUT / name
        self.dir.mkdir(parents=True)

    def array(self, name: str, arr: np.ndarray, dtype: str) -> dict[str, Any]:
        arr = np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<"))
        fname = f"{name}.bin"
        arr.tofile(self.dir / fname)
        return {"file": fname, "dtype": dtype, "shape": list(arr.shape)}

    def json(self, name: str, obj: Any) -> None:
        with open(self.dir / name, "w", encoding="utf-8", newline="\n") as fd:
            json.dump(obj, fd, indent=1, allow_nan=False)
            fd.write("\n")


class Dataset:
    """A named DMatrix plus the recipe to rebuild it in C#."""

    def __init__(
        self,
        name: str,
        x: np.ndarray | scipy.sparse.csr_matrix,
        *,
        label: np.ndarray | None = None,
        weight: np.ndarray | None = None,
        base_margin: np.ndarray | None = None,
        qid: np.ndarray | None = None,
        label_lower_bound: np.ndarray | None = None,
        label_upper_bound: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
    ) -> None:
        self.name = name
        self.x = x
        self.info = {
            "label": label,
            "weight": weight,
            "base_margin": base_margin,
            "qid": qid,
            "label_lower_bound": label_lower_bound,
            "label_upper_bound": label_upper_bound,
        }
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.dmatrix = xgb.DMatrix(
            x,
            label=label,
            weight=weight,
            base_margin=base_margin,
            qid=qid,
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=feature_types is not None and "c" in feature_types,
            missing=np.nan,
        )
        if label_lower_bound is not None:
            self.dmatrix.set_float_info("label_lower_bound", label_lower_bound)
        if label_upper_bound is not None:
            self.dmatrix.set_float_info("label_upper_bound", label_upper_bound)

    @property
    def dense(self) -> bool:
        return not scipy.sparse.issparse(self.x)

    def describe(self, w: Writer) -> dict[str, Any]:
        rows, cols = self.x.shape
        desc: dict[str, Any] = {"rows": rows, "cols": cols}
        if self.dense:
            desc["kind"] = "dense"
            desc["data"] = w.array(f"{self.name}_x", self.x, "f4")
        else:
            csr = self.x
            desc["kind"] = "csr"
            desc["indptr"] = w.array(f"{self.name}_indptr", csr.indptr, "i8")
            desc["indices"] = w.array(f"{self.name}_indices", csr.indices, "u4")
            desc["values"] = w.array(f"{self.name}_values", csr.data, "f4")
        info = {}
        for field, value in self.info.items():
            if value is not None:
                info[field] = w.array(f"{self.name}_{field}", value, "u4" if field == "qid" else "f4")
        desc["info"] = info
        desc["feature_names"] = self.feature_names
        desc["feature_types"] = self.feature_types
        return desc


# ---- Custom objectives (float32 throughout so C# can reproduce them bit for bit) ----


def pseudo_huber(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
    y = dtrain.get_label()
    d = (predt - y).astype(np.float32)
    scale = np.float32(1) + d * d
    sq = np.sqrt(scale)
    return d / sq, np.float32(1) / (scale * sq)


OBJECTIVES: dict[str, Callable] = {"pseudo_huber": pseudo_huber}


# ---- Case runner --------------------------------------------------------------------


def run_case(
    name: str,
    description: str,
    params: dict[str, Any] | list[tuple[str, Any]],
    train: Dataset,
    valid: Dataset,
    num_boost_round: int,
    *,
    early_stopping_rounds: int | None = None,
    custom_objective: str | None = None,
    extra_predictions: bool = False,
    tree_model: bool = True,
    inplace: bool = True,
) -> None:
    w = Writer(name)

    # Build the parameter list exactly as Booster.__init__ sends it to XGBoosterSetParams.
    processed = xgb.core._configure_metrics(dict(params) if isinstance(params, dict) else list(params))
    items = list(processed.items()) if isinstance(processed, dict) else list(processed)
    sent = xgb.Booster._prepare_parameters(items + [("validate_parameters", True)])

    evals_result: dict[str, dict[str, list[float]]] = {}
    bst = xgb.train(
        params,
        train.dmatrix,
        num_boost_round=num_boost_round,
        evals=[(train.dmatrix, "train"), (valid.dmatrix, "valid")],
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        obj=OBJECTIVES[custom_objective] if custom_objective else None,
        verbose_eval=False,
    )

    predictions: dict[str, Any] = {}

    def add(pred_name: str, data: Dataset, kind: str, arr: np.ndarray, **opts: Any) -> None:
        predictions[pred_name] = {"data": data.name, "type": kind, **opts, **w.array(f"pred_{pred_name}", arr, "f4")}

    for data in (train, valid):
        d = data.dmatrix
        add(f"{data.name}_value", data, "value", bst.predict(d))
        add(f"{data.name}_margin", data, "margin", bst.predict(d, output_margin=True))
    add("valid_contrib", valid, "contribution", bst.predict(valid.dmatrix, pred_contribs=True))
    if tree_model:
        add("valid_leaf", valid, "leaf", bst.predict(valid.dmatrix, pred_leaf=True))
        k = max(1, bst.num_boosted_rounds() // 2)
        add("valid_value_half", valid, "value", bst.predict(valid.dmatrix, iteration_range=(0, k)), iteration_end=k)
    if extra_predictions:
        add("valid_approx_contrib", valid, "approximate_contribution",
            bst.predict(valid.dmatrix, pred_contribs=True, approx_contribs=True))
        add("valid_interaction", valid, "interaction", bst.predict(valid.dmatrix, pred_interactions=True))
    if inplace and valid.dense:
        add("valid_inplace_value", valid, "value", bst.inplace_predict(valid.x), inplace=True)
        add("valid_inplace_margin", valid, "margin", bst.inplace_predict(valid.x, predict_type="margin"), inplace=True)
    if inplace and not valid.dense:
        add("valid_inplace_value", valid, "value", bst.inplace_predict(valid.x), inplace=True)

    feature_score = {}
    for importance in ("weight", "gain", "cover", "total_gain", "total_cover") if tree_model else ("weight",):
        scores = bst.get_score(importance_type=importance)
        feature_score[importance] = {k: np.asarray(v, dtype=np.float32).ravel().tolist() for k, v in scores.items()}

    bst.save_model(w.dir / "model.json")
    bst.save_model(w.dir / "model.ubj")

    attrs = bst.attributes()
    manifest = {
        "name": name,
        "description": description,
        "xgboost_version": xgb.__version__,
        "params": [list(p) for p in sent],
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "custom_objective": custom_objective,
        "datasets": {"train": train.describe(w), "valid": valid.describe(w)},
        "expected": {
            "boosted_rounds": bst.num_boosted_rounds(),
            "num_features": bst.num_features(),
            "best_iteration": int(attrs["best_iteration"]) if "best_iteration" in attrs else None,
            "best_score": float(attrs["best_score"]) if "best_score" in attrs else None,
            "eval_history": evals_result,
            "predictions": predictions,
            "feature_score": feature_score,
            "dump_text": bst.get_dump(dump_format="text", with_stats=True),
            "dump_json": bst.get_dump(dump_format="json", with_stats=True),
            "model_json": "model.json",
            "model_ubj": "model.ubj",
        },
    }
    w.json("case.json", manifest)
    print(f"{name:24s} rounds={bst.num_boosted_rounds():4d} predictions={len(predictions)}")


# ---- Cases --------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for old in OUT.iterdir():
        shutil.rmtree(old)
    names = [f"f_{c}" for c in "abcdef"]
    base = {"seed": 0, "nthread": 1}

    rng = np.random.default_rng(2024)
    x, xv = features(rng, N_TRAIN, missing=0.1), features(rng, N_VALID, missing=0.1)
    run_case(
        "reg_squarederror_hist",
        "Regression, hist, missing values, feature names, two metrics.",
        {**base, "objective": "reg:squarederror", "tree_method": "hist", "max_depth": 4, "eta": 0.3,
         "eval_metric": ["rmse", "mae"]},
        Dataset("train", x, label=linear_target(rng, x), feature_names=names),
        Dataset("valid", xv, label=linear_target(rng, xv), feature_names=names),
        20, extra_predictions=True,
    )

    rng = np.random.default_rng(1)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    lab = lambda a: ((a[:, 0] + a[:, 1] + 0.3 * rng.standard_normal(len(a))) > 1).astype(np.float32)  # noqa: E731
    run_case(
        "binary_logistic_early_stop",
        "Binary classification with weights and early stopping on valid auc.",
        {**base, "objective": "binary:logistic", "max_depth": 6, "eta": 0.5, "eval_metric": ["logloss", "auc"]},
        Dataset("train", x, label=lab(x), weight=rng.uniform(0.5, 2, N_TRAIN).astype(np.float32)),
        Dataset("valid", xv, label=lab(xv)),
        200, early_stopping_rounds=5,
    )

    rng = np.random.default_rng(2)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    run_case(
        "multi_softprob",
        "Three-class softprob.",
        {**base, "objective": "multi:softprob", "num_class": 3, "max_depth": 3, "eval_metric": ["mlogloss", "merror"]},
        Dataset("train", x, label=np.argmax(x[:, :3], axis=1).astype(np.float32)),
        Dataset("valid", xv, label=np.argmax(xv[:, :3], axis=1).astype(np.float32)),
        10,
    )

    rng = np.random.default_rng(3)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    rel = lambda a: np.clip(np.floor(a[:, 0] * 3 + a[:, 1]), 0, 3).astype(np.float32)  # noqa: E731
    run_case(
        "rank_ndcg",
        "Learning to rank with query ids.",
        {**base, "objective": "rank:ndcg", "max_depth": 3, "eval_metric": "ndcg@5"},
        Dataset("train", x, label=rel(x), qid=np.repeat(np.arange(N_TRAIN // 10), 10).astype(np.uint32)),
        Dataset("valid", xv, label=rel(xv), qid=np.repeat(np.arange(N_VALID // 10), 10).astype(np.uint32)),
        10,
    )

    rng = np.random.default_rng(4)
    xs = scipy.sparse.random(N_TRAIN, N_COLS, density=0.4, format="csr", dtype=np.float32, rng=rng)
    xvs = scipy.sparse.random(N_VALID, N_COLS, density=0.4, format="csr", dtype=np.float32, rng=rng)
    slab = lambda a: (np.asarray(a.sum(axis=1)).ravel() > 0.8).astype(np.float32)  # noqa: E731
    run_case(
        "csr_approx",
        "Sparse CSR input with the approx tree method.",
        {**base, "objective": "binary:logistic", "tree_method": "approx", "max_depth": 4},
        Dataset("train", xs, label=slab(xs)),
        Dataset("valid", xvs, label=slab(xvs)),
        15,
    )

    rng = np.random.default_rng(5)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    run_case(
        "exact_base_margin",
        "Exact tree method with a per-row base margin.",
        {**base, "objective": "reg:squarederror", "tree_method": "exact", "max_depth": 3},
        Dataset("train", x, label=linear_target(rng, x), base_margin=np.full(N_TRAIN, 0.25, np.float32)),
        Dataset("valid", xv, label=linear_target(rng, xv), base_margin=np.full(N_VALID, 0.25, np.float32)),
        10, inplace=False,
    )

    rng = np.random.default_rng(6)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    run_case(
        "dart",
        "DART booster with dropout.",
        {**base, "booster": "dart", "objective": "reg:squarederror", "rate_drop": 0.2, "skip_drop": 0.0,
         "max_depth": 3},
        Dataset("train", x, label=linear_target(rng, x)),
        Dataset("valid", xv, label=linear_target(rng, xv)),
        15,
    )

    rng = np.random.default_rng(7)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    run_case(
        "gblinear",
        "Linear booster with coordinate descent.",
        {**base, "booster": "gblinear", "objective": "reg:squarederror", "updater": "coord_descent",
         "feature_selector": "cyclic", "lambda": 0.1},
        Dataset("train", x, label=linear_target(rng, x), feature_names=names),
        Dataset("valid", xv, label=linear_target(rng, xv), feature_names=names),
        20, tree_model=False, inplace=False,
    )

    rng = np.random.default_rng(8)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    x[:, 0], xv[:, 0] = rng.integers(0, 5, N_TRAIN), rng.integers(0, 5, N_VALID)
    catlab = lambda a: (np.isin(a[:, 0], [1, 3]) * 2.0 + a[:, 1]).astype(np.float32)  # noqa: E731
    types = ["c"] + ["q"] * (N_COLS - 1)
    run_case(
        "categorical",
        "One categorical feature with partition-based splits.",
        {**base, "objective": "reg:squarederror", "tree_method": "hist", "max_cat_to_onehot": 1, "max_depth": 3},
        Dataset("train", x, label=catlab(x), feature_types=types),
        Dataset("valid", xv, label=catlab(xv), feature_types=types),
        10, inplace=False,
    )

    rng = np.random.default_rng(9)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)

    def bounds(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.exp(a[:, 0] * 2 - a[:, 1]).astype(np.float32)
        censored = rng.random(len(a)) < 0.3
        return t, np.where(censored, np.float32(np.inf), t).astype(np.float32)

    lo, hi = bounds(x)
    vlo, vhi = bounds(xv)
    run_case(
        "survival_aft",
        "Accelerated failure time with right-censored labels.",
        {**base, "objective": "survival:aft", "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
         "max_depth": 3, "eval_metric": "aft-nloglik"},
        Dataset("train", x, label_lower_bound=lo, label_upper_bound=hi),
        Dataset("valid", xv, label_lower_bound=vlo, label_upper_bound=vhi),
        10,
    )

    rng = np.random.default_rng(10)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    two = lambda a: np.stack([linear_target(rng, a), a[:, 4] * 2 - a[:, 5]], axis=1).astype(np.float32)  # noqa: E731
    run_case(
        "multi_output",
        "Two regression targets, one tree per target.",
        {**base, "objective": "reg:squarederror", "tree_method": "hist", "multi_strategy": "one_output_per_tree",
         "max_depth": 3},
        Dataset("train", x, label=two(x)),
        Dataset("valid", xv, label=two(xv)),
        10,
    )

    rng = np.random.default_rng(11)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    run_case(
        "quantile",
        "Quantile regression for three quantiles at once.",
        {**base, "objective": "reg:quantileerror", "quantile_alpha": np.array([0.1, 0.5, 0.9]), "max_depth": 3},
        Dataset("train", x, label=linear_target(rng, x)),
        Dataset("valid", xv, label=linear_target(rng, xv)),
        10,
    )

    rng = np.random.default_rng(12)
    x, xv = features(rng, N_TRAIN), features(rng, N_VALID)
    run_case(
        "custom_objective",
        "Custom pseudo-Huber objective computed in float32.",
        {**base, "max_depth": 3, "eval_metric": "rmse"},
        Dataset("train", x, label=linear_target(rng, x)),
        Dataset("valid", xv, label=linear_target(rng, xv)),
        10, custom_objective="pseudo_huber",
    )


if __name__ == "__main__":
    main()
