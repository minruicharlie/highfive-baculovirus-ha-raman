from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))

import raman_preprocessing as raman_prep


INDEPENDENT_TEST_BATCH = 5
SCANS_PER_OFFLINE_POINT = 5
TARGETS = ["VCD", "Via", "Dim", "HA2", "HA4", "Glc", "Gln", "NH4"]
DISPLAY_TARGET = {"HA2": "eHA", "HA4": "tHA"}
MANUSCRIPT_TARGET_LABELS = {"Glc": "Glc (g/L)", "Gln": "Gln (mM)", "NH4": "NH4 (mM)"}
PLSR_COMPONENTS = {"VCD": 8, "Via": 5, "Dim": 8, "HA2": 3, "HA4": 8, "Glc": 3, "Gln": 8, "NH4": 8}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def finite_array(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def target_mask(df: pd.DataFrame, target: str) -> np.ndarray:
    y = finite_array(df[target])
    mask = np.isfinite(y)
    if target in {"HA2", "HA4"}:
        mask &= y > 0
    return mask


def infer_spectral_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    spectral = cfg.get("spectral", {})
    lo = float(spectral.get("min_wavenumber", 500))
    hi = float(spectral.get("max_wavenumber", 1800))
    stride = max(1, int(spectral.get("feature_stride", 1)))
    pairs = []
    for col in df.columns:
        try:
            wave = float(str(col))
        except ValueError:
            continue
        if lo <= wave <= hi:
            pairs.append((wave, col))
    if not pairs:
        raise ValueError("No spectral columns found in the configured Raman window.")
    pairs.sort(key=lambda item: item[0])
    return [col for _, col in pairs[::stride]]


def chemical_features(raw: np.ndarray, cfg: dict) -> np.ndarray:
    return raman_prep.chemical_feature_matrix(raw, cfg, derivative_order=1)


def clip_target(target: str, pred: np.ndarray, y_train: np.ndarray | None = None) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if target == "Via":
        return np.clip(pred, 0.0, 100.0)
    if target == "Dim":
        return np.clip(pred, 14.5, 22.0)
    if target in {"HA2", "HA4"}:
        hi = float(np.nanmax(y_train)) if y_train is not None and np.isfinite(y_train).any() else 14.0
        return np.clip(pred, 0.0, max(hi, 1.0))
    return np.clip(pred, 0.0, None)


def fit_predict_plsr(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, n_components: int, target: str) -> np.ndarray:
    y_train = y[train_idx]
    n_comp = max(1, min(int(n_components), len(y_train) - 1, X.shape[1]))
    model = make_pipeline(SimpleImputer(strategy="median"), PLSRegression(n_components=n_comp, scale=True))
    model.fit(X[train_idx], y_train)
    pred = np.asarray(model.predict(X[test_idx])).ravel()
    return clip_target(target, pred, y_train)


def point_ids(groups: np.ndarray, target: str) -> np.ndarray:
    ids = np.empty(len(groups), dtype=object)
    for batch in sorted(pd.unique(groups)):
        idx = np.where(groups == batch)[0]
        for offset, row in enumerate(idx):
            point_no = offset // SCANS_PER_OFFLINE_POINT
            ids[row] = f"{DISPLAY_TARGET.get(target, target)}_B{int(float(batch))}_P{point_no:03d}"
    return ids


def aggregate_points(indices: np.ndarray, groups: np.ndarray, times: np.ndarray, y: np.ndarray, pred: np.ndarray, target: str) -> pd.DataFrame:
    pids = point_ids(groups[indices], target)
    frame = pd.DataFrame(
        {
            "point_id": pids,
            "Batch": groups[indices],
            "time": times[indices],
            "reference": y[indices],
            "prediction": pred,
        }
    )
    out = (
        frame.groupby("point_id", as_index=False)
        .agg(
            Batch=("Batch", "first"),
            time=("time", "mean"),
            reference=("reference", "mean"),
            prediction=("prediction", "mean"),
            prediction_sd=("prediction", "std"),
            n_scans=("prediction", "size"),
        )
        .sort_values(["Batch", "time"])
        .fillna({"prediction_sd": 0.0})
        .reset_index(drop=True)
    )
    out["target"] = DISPLAY_TARGET.get(target, target)
    out["candidate"] = f"BasePLSR_full_d1_n{PLSR_COMPONENTS[target]}"
    return out


def point_kfold_indices(indices: np.ndarray, groups: np.ndarray, target: str, n_splits: int = 5):
    local_ids = point_ids(groups[indices], target)
    unique_ids = np.array(sorted(pd.unique(local_ids)))
    splitter = KFold(n_splits=min(n_splits, len(unique_ids)), shuffle=True, random_state=42)
    for train_pos, valid_pos in splitter.split(unique_ids):
        train_set = set(unique_ids[train_pos])
        valid_set = set(unique_ids[valid_pos])
        train_idx = indices[np.array([pid in train_set for pid in local_ids])]
        valid_idx = indices[np.array([pid in valid_set for pid in local_ids])]
        yield train_idx, valid_idx


def compute_plsr_cv_points(label: pd.DataFrame, X_label: np.ndarray, target: str) -> pd.DataFrame:
    y = finite_array(label[target])
    groups = finite_array(label["Batch"])
    times = finite_array(label["time"])
    pool = np.where(target_mask(label, target) & (groups != INDEPENDENT_TEST_BATCH))[0]
    oof = np.full(len(pool), np.nan)
    pos = {idx: i for i, idx in enumerate(pool)}
    for train_idx, valid_idx in point_kfold_indices(pool, groups, target):
        pred = fit_predict_plsr(X_label, y, train_idx, valid_idx, PLSR_COMPONENTS[target], target)
        for idx, val in zip(valid_idx, pred):
            oof[pos[idx]] = val
    valid = np.isfinite(oof)
    return aggregate_points(pool[valid], groups, times, y, oof[valid], target)


def compute_plsr_train_points(label: pd.DataFrame, X_label: np.ndarray, target: str) -> pd.DataFrame:
    y = finite_array(label[target])
    groups = finite_array(label["Batch"])
    times = finite_array(label["time"])
    train_idx = np.where(target_mask(label, target) & (groups != INDEPENDENT_TEST_BATCH))[0]
    pred = fit_predict_plsr(X_label, y, train_idx, train_idx, PLSR_COMPONENTS[target], target)
    return aggregate_points(train_idx, groups, times, y, pred, target)


def compute_plsr_independent_points(label: pd.DataFrame, X_label: np.ndarray, target: str) -> pd.DataFrame:
    y = finite_array(label[target])
    groups = finite_array(label["Batch"])
    times = finite_array(label["time"])
    train_idx = np.where(target_mask(label, target) & (groups != INDEPENDENT_TEST_BATCH))[0]
    test_idx = np.where(target_mask(label, target) & (groups == INDEPENDENT_TEST_BATCH))[0]
    pred = fit_predict_plsr(X_label, y, train_idx, test_idx, PLSR_COMPONENTS[target], target)
    return aggregate_points(test_idx, groups, times, y, pred, target)


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def summarize_points(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for target, sub in df.groupby("target", observed=True):
        ref = finite_array(sub["reference"])
        pred = finite_array(sub["prediction"])
        mask = np.isfinite(ref) & np.isfinite(pred)
        ref = ref[mask]
        pred = pred[mask]
        rows.append(
            {
                "target": target,
                f"{prefix}_n": int(mask.sum()),
                f"{prefix}_rmse": float(np.sqrt(np.mean((pred - ref) ** 2))) if mask.any() else np.nan,
                f"{prefix}_mae": float(np.mean(np.abs(pred - ref))) if mask.any() else np.nan,
                f"{prefix}_bias": float(np.mean(pred - ref)) if mask.any() else np.nan,
                f"{prefix}_r2": safe_r2(ref, pred),
            }
        )
    return pd.DataFrame(rows)


def build_metric_table(train: pd.DataFrame, cv: pd.DataFrame, independent: pd.DataFrame) -> pd.DataFrame:
    out = summarize_points(train, "train").merge(summarize_points(cv, "cv"), on="target").merge(summarize_points(independent, "independent"), on="target")
    order = [DISPLAY_TARGET.get(t, t) for t in TARGETS]
    out["target"] = pd.Categorical(out["target"], order, ordered=True)
    out = out.sort_values("target").reset_index(drop=True)
    out["candidate"] = out["target"].map({DISPLAY_TARGET.get(t, t): f"BasePLSR_full_d1_n{PLSR_COMPONENTS[t]}" for t in TARGETS})
    return out


def table1_for_manuscript(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics[["target", "train_n", "train_rmse", "train_r2", "cv_n", "cv_rmse", "cv_r2"]].copy()
    out = out.rename(
        columns={
            "target": "Indicator",
            "train_n": "Training n",
            "train_rmse": "Training RMSE",
            "train_r2": "Training R²",
            "cv_n": "CV n",
            "cv_rmse": "CV RMSE",
            "cv_r2": "CV R²",
        }
    )
    vcd = out["Indicator"].astype(str).eq("VCD")
    out.loc[vcd, "Training RMSE"] = pd.to_numeric(out.loc[vcd, "Training RMSE"], errors="coerce") / 1e6
    out.loc[vcd, "CV RMSE"] = pd.to_numeric(out.loc[vcd, "CV RMSE"], errors="coerce") / 1e6
    out["Indicator"] = out["Indicator"].astype(str).replace(MANUSCRIPT_TARGET_LABELS)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run common PLSR baselines for all process variables.")
    parser.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "config.yaml")
    parser.add_argument("--label-csv", type=Path, default=ROOT / "data" / "inputs" / "LabelData_time.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "model_outputs" / "plsr_baselines")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = pd.read_csv(args.label_csv)
    cols = infer_spectral_columns(label, cfg)
    raw = label[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X_label = chemical_features(raw, cfg)

    train = pd.concat([compute_plsr_train_points(label, X_label, target) for target in TARGETS], ignore_index=True)
    cv = pd.concat([compute_plsr_cv_points(label, X_label, target) for target in TARGETS], ignore_index=True)
    independent = pd.concat([compute_plsr_independent_points(label, X_label, target) for target in TARGETS], ignore_index=True)
    metrics = build_metric_table(train, cv, independent)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_dir / "plsr_train_point_predictions.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(args.out_dir / "plsr_cv_point_predictions.csv", index=False, encoding="utf-8-sig")
    independent.to_csv(args.out_dir / "plsr_independent_point_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.round(6).to_csv(args.out_dir / "plsr_all_metrics.csv", index=False, encoding="utf-8-sig")
    table1_for_manuscript(metrics).round(4).to_csv(ROOT / "results" / "tables" / "table1_plsr_train_cv.csv", index=False, encoding="utf-8-sig")
    print(table1_for_manuscript(metrics).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
