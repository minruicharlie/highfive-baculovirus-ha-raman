from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT_TEST_BATCH = 5
SCANS_PER_OFFLINE_POINT = 5


@dataclass(frozen=True)
class ViaCandidate:
    step: str
    name: str
    response_layer: str
    k_best: int
    c_value: float = 10.0
    epsilon: float = 0.05
    use_process: bool = False
    use_raman: bool = True


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "n": int(len(y_true)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": safe_r2(y_true, y_pred),
    }


def logit_percent(y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(y, dtype=float) / 100.0, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def inverse_logit_percent(z: np.ndarray) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def transform_y(y: np.ndarray, response_layer: str) -> np.ndarray:
    if response_layer == "bounded_logit":
        return logit_percent(y)
    if response_layer == "identity":
        return np.asarray(y, dtype=float)
    raise ValueError(response_layer)


def inverse_y(z: np.ndarray, response_layer: str) -> np.ndarray:
    if response_layer == "bounded_logit":
        return inverse_logit_percent(z)
    if response_layer == "identity":
        return np.asarray(z, dtype=float)
    raise ValueError(response_layer)


def infer_spectral_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    spectral = cfg.get("spectral", {})
    lo = float(spectral.get("min_wavenumber", 500))
    hi = float(spectral.get("max_wavenumber", 1800))
    stride = max(1, int(spectral.get("feature_stride", 1)))
    pairs: list[tuple[float, str]] = []
    for col in df.columns:
        try:
            wave = float(str(col))
        except ValueError:
            continue
        if lo <= wave <= hi:
            pairs.append((wave, col))
    if not pairs:
        raise ValueError("No numeric Raman spectral columns were found.")
    pairs.sort(key=lambda item: item[0])
    return [col for _, col in pairs[::stride]]


def chemical_feature_matrix(raw: np.ndarray, cfg: dict) -> np.ndarray:
    spectral = cfg.get("spectral", {})
    raw = np.asarray(raw, dtype=float)
    x_axis = np.linspace(-1.0, 1.0, raw.shape[1])
    vander = np.vander(x_axis, N=int(spectral.get("baseline_degree", 3)) + 1, increasing=False)
    baseline_pinv = np.linalg.pinv(vander)
    corrected = raw - (raw @ baseline_pinv.T) @ vander.T
    window = min(int(spectral.get("savgol_window", 15)), corrected.shape[1] - (1 - corrected.shape[1] % 2))
    if window % 2 == 0:
        window -= 1
    window = max(5, window)
    poly = min(int(spectral.get("savgol_polyorder", 3)), window - 2)
    try:
        chem = savgol_filter(corrected, window_length=window, polyorder=poly, deriv=1, axis=1, mode="interp")
    except Exception:
        chem = np.gradient(corrected, axis=1)
    mean = np.mean(chem, axis=1, keepdims=True)
    std = np.std(chem, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (chem - mean) / std


def process_time_features(df: pd.DataFrame, operation: pd.DataFrame) -> np.ndarray:
    batches = pd.to_numeric(df["Batch"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(df["time"], errors="coerce").to_numpy(float)
    op = operation.set_index("Batch")
    inoc = np.asarray([float(op.loc[int(b), "inoculation_time_h"]) for b in batches], dtype=float)
    signed_since = times - inoc
    post = (signed_since >= 0).astype(float)
    before = np.clip(-signed_since, 0, None)
    since = np.clip(signed_since, 0, None)
    scale_t = 260.0
    return np.column_stack(
        [
            inoc / scale_t,
            times / scale_t,
            post,
            before / scale_t,
            np.log1p(before) / np.log1p(scale_t),
            since / scale_t,
            np.sqrt(since) / np.sqrt(scale_t),
            np.log1p(since) / np.log1p(scale_t),
        ]
    )


def point_ids(groups: np.ndarray, times: np.ndarray, y: np.ndarray) -> np.ndarray:
    ids = np.empty(len(y), dtype=object)
    for batch in sorted(pd.unique(groups)):
        idx = np.where(groups == batch)[0]
        # LabelData_time.csv is expanded so that each offline sampling point
        # contributes the five time-nearest Raman spectra as consecutive rows.
        # Use that explicit expansion structure instead of re-inferring points
        # from Raman timestamps, which can split one offline point when the five
        # nearest spectra span a sparse acquisition interval.
        for offset, row in enumerate(idx):
            point_no = offset // SCANS_PER_OFFLINE_POINT
            ids[row] = f"Via_B{int(float(batch))}_P{point_no:03d}"
    return ids


def point_kfold_indices(indices: np.ndarray, groups: np.ndarray, times: np.ndarray, y: np.ndarray, n_splits: int = 5):
    local_ids = point_ids(groups[indices], times[indices], y[indices])
    unique_ids = np.array(sorted(pd.unique(local_ids)))
    splitter = KFold(n_splits=min(n_splits, len(unique_ids)), shuffle=True, random_state=42)
    for train_pos, valid_pos in splitter.split(unique_ids):
        train_set = set(unique_ids[train_pos])
        valid_set = set(unique_ids[valid_pos])
        train_idx = indices[np.array([pid in train_set for pid in local_ids])]
        valid_idx = indices[np.array([pid in valid_set for pid in local_ids])]
        yield train_idx, valid_idx


def aggregate_points(indices: np.ndarray, groups: np.ndarray, times: np.ndarray, y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    pids = point_ids(groups[indices], times[indices], y[indices])
    frame = pd.DataFrame(
        {
            "point_id": pids,
            "Batch": groups[indices],
            "time": times[indices],
            "reference": y[indices],
            "prediction": pred,
        }
    )
    return (
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


def fit_predict_via(
    candidate: ViaCandidate,
    X: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    y_train = y[train_idx]
    y_trans = transform_y(y_train, candidate.response_layer)
    y_mean = float(np.mean(y_trans))
    y_std = max(float(np.std(y_trans)), 1e-8)
    y_scaled = (y_trans - y_mean) / y_std

    train_blocks = []
    test_blocks = []
    if candidate.use_raman:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X[train_idx]))
        X_test = scaler.transform(imputer.transform(X[test_idx]))
        selector = SelectKBest(score_func=f_regression, k=min(candidate.k_best, X_train.shape[1]))
        train_blocks.append(selector.fit_transform(X_train, y_scaled))
        test_blocks.append(selector.transform(X_test))

    if candidate.use_process:
        process_scaler = StandardScaler()
        train_blocks.append(process_scaler.fit_transform(P[train_idx]))
        test_blocks.append(process_scaler.transform(P[test_idx]))
    if not train_blocks:
        raise ValueError(f"{candidate.name} has no feature blocks enabled")
    X_train = np.hstack(train_blocks)
    X_test = np.hstack(test_blocks)

    model = SVR(C=candidate.c_value, gamma="scale", epsilon=candidate.epsilon)
    model.fit(X_train, y_scaled)
    pred_trans = model.predict(X_test) * y_std + y_mean
    return np.clip(inverse_y(pred_trans, candidate.response_layer), 0.0, 100.0)


def feature_label(candidate: ViaCandidate) -> str:
    if candidate.use_raman and candidate.use_process:
        return "Raman SelectSVR + process-time"
    if candidate.use_raman:
        return "Raman SelectSVR"
    if candidate.use_process:
        return "process-time only"
    return "none"


def via_candidates() -> list[ViaCandidate]:
    return [
        ViaCandidate("V1 identity Raman SelectSVR", "Via_SelectSVR_k80_identity_none_C10_eps0.05", "identity", 80),
        ViaCandidate("V2 bounded-logit Raman SelectSVR", "Via_SelectSVR_k80_bounded_logit_none_C10_eps0.05", "bounded_logit", 80),
        ViaCandidate(
            "V3 identity + process-time SelectSVR",
            "Via_SelectSVR_k80_identity_none_inoc_time_C10_eps0.05",
            "identity",
            80,
            use_process=True,
        ),
        ViaCandidate(
            "VT bounded-logit process-time-only SelectSVR",
            "Via_SelectSVR_bounded_logit_process_time_only_C10_eps0.05",
            "bounded_logit",
            1,
            use_process=True,
            use_raman=False,
        ),
        ViaCandidate(
            "V4 bounded-logit + process-time SelectSVR",
            "Via_SelectSVR_k80_bounded_logit_none_inoc_time_C10_eps0.05",
            "bounded_logit",
            80,
            use_process=True,
        ),
    ]


def evaluate_candidate(candidate: ViaCandidate, X: np.ndarray, P: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    idx_all = np.where(np.isfinite(y) & np.isfinite(groups) & np.isfinite(times))[0]
    train_pool = idx_all[groups[idx_all] != INDEPENDENT_TEST_BATCH]
    test_idx = idx_all[groups[idx_all] == INDEPENDENT_TEST_BATCH]

    oof = np.full(len(train_pool), np.nan)
    pos = {idx: i for i, idx in enumerate(train_pool)}
    for train_idx, valid_idx in point_kfold_indices(train_pool, groups, times, y):
        pred = fit_predict_via(candidate, X, P, y, train_idx, valid_idx)
        for idx, value in zip(valid_idx, pred):
            oof[pos[idx]] = value

    valid = np.isfinite(oof)
    cv_points = aggregate_points(train_pool[valid], groups, times, y, oof[valid])
    test_pred = fit_predict_via(candidate, X, P, y, train_pool, test_idx)
    independent_test_points = aggregate_points(test_idx, groups, times, y, test_pred)
    cv = metric_dict(cv_points["reference"], cv_points["prediction"])
    independent_test = metric_dict(independent_test_points["reference"], independent_test_points["prediction"])
    row = {
        "endpoint": "Via",
        "step": candidate.step,
        "model": "SelectSVR",
        "response_layer": candidate.response_layer,
        "features": feature_label(candidate),
        "candidate": candidate.name,
        "weighting": "none",
        "cv_n": cv["n"],
        "cv_rmse": cv["rmse"],
        "cv_mae": cv["mae"],
        "cv_bias": cv["bias"],
        "cv_r2": cv["r2"],
        "independent_test_n": independent_test["n"],
        "independent_test_rmse": independent_test["rmse"],
        "independent_test_mae": independent_test["mae"],
        "independent_test_bias": independent_test["bias"],
        "independent_test_r2": independent_test["r2"],
    }
    cv_points["candidate"] = candidate.name
    independent_test_points["candidate"] = candidate.name
    return row, cv_points, independent_test_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final no-weight Via bounded-logit process-time SelectSVR ablation.")
    parser.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "config.yaml")
    parser.add_argument("--label-csv", type=Path, default=ROOT / "data" / "inputs" / "LabelData_time.csv")
    parser.add_argument("--operation-csv", type=Path, default=ROOT / "data" / "inputs" / "Batch_operation_times.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "model_outputs" / "via_bounded_logit_process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = pd.read_csv(args.label_csv)
    operation = pd.read_csv(args.operation_csv)
    cols = infer_spectral_columns(label, cfg)
    X = chemical_feature_matrix(label[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float), cfg)
    P = process_time_features(label, operation)
    y = pd.to_numeric(label["Via"], errors="coerce").to_numpy(float)
    groups = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(label["time"], errors="coerce").to_numpy(float)

    rows = []
    cv_frames = []
    final_independent_test = pd.DataFrame()
    candidates = via_candidates()
    final_name = candidates[-1].name
    for candidate in candidates:
        row, cv_points, independent_test_points = evaluate_candidate(candidate, X, P, y, groups, times)
        rows.append(row)
        cv_frames.append(cv_points)
        if candidate.name == final_name:
            final_independent_test = independent_test_points.copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    public_summary = summary.copy()
    independent_test_cols = ["independent_test_n", "independent_test_rmse", "independent_test_mae", "independent_test_bias", "independent_test_r2"]
    public_summary.loc[~public_summary["step"].astype(str).str.startswith("V4 "), independent_test_cols] = np.nan
    public_summary.round(6).to_csv(args.out_dir / "via_no_weight_selectsvr_ablation_recomputed.csv", index=False, encoding="utf-8-sig")
    pd.concat(cv_frames, ignore_index=True).to_csv(args.out_dir / "via_no_weight_cv_point_predictions.csv", index=False, encoding="utf-8-sig")
    final_independent_test.to_csv(args.out_dir / "via_no_weight_independent_test_point_predictions.csv", index=False, encoding="utf-8-sig")
    print(public_summary[["candidate", "cv_rmse", "cv_r2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
