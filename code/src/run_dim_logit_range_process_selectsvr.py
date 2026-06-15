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
DIM_SPECTRAL_MIN = 500.0
DIM_SPECTRAL_MAX = 1800.0
DIM_LOWER = 15.0
DIM_UPPER = 20.5


@dataclass(frozen=True)
class DimCandidate:
    step: str
    name: str
    response_layer: str
    features: str
    k_best: int = 360
    c_value: float = 10.0
    epsilon: float = 0.01
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


def infer_spectral_columns(df: pd.DataFrame, cfg: dict, lo: float = DIM_SPECTRAL_MIN, hi: float = DIM_SPECTRAL_MAX) -> list[str]:
    stride = max(1, int(cfg.get("spectral", {}).get("feature_stride", 1)))
    pairs: list[tuple[float, str]] = []
    for col in df.columns:
        try:
            wave = float(str(col))
        except ValueError:
            continue
        if lo <= wave <= hi:
            pairs.append((wave, col))
    if not pairs:
        raise ValueError("No numeric Raman spectral columns were found in the Dim window.")
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


def load_dim_inputs(label: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = infer_spectral_columns(label, cfg)
    raw = label[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = chemical_feature_matrix(raw, cfg)
    y = pd.to_numeric(label["Dim"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(label["time"], errors="coerce").to_numpy(float)
    groups = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
    label_idx = np.where(np.isfinite(y) & np.isfinite(times) & np.isfinite(groups))[0]
    return X, y, times, groups, label_idx


def point_ids(groups: np.ndarray, times: np.ndarray, y: np.ndarray) -> np.ndarray:
    ids = np.empty(len(y), dtype=object)
    for batch in sorted(pd.unique(groups)):
        idx = np.where(groups == batch)[0]
        for offset, row in enumerate(idx):
            point_no = offset // SCANS_PER_OFFLINE_POINT
            ids[row] = f"Dim_B{int(float(batch))}_P{point_no:03d}"
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


def process_time_features(indices: np.ndarray, groups: np.ndarray, times: np.ndarray, inoculation_by_batch: dict[int, float]) -> np.ndarray:
    signed_since = np.asarray([times[i] - inoculation_by_batch[int(float(groups[i]))] for i in indices], dtype=float)
    post = (signed_since >= 0).astype(float)
    before = np.clip(-signed_since, 0, None)
    since = np.clip(signed_since, 0, None)
    return np.column_stack(
        [
            post,
            before / 100.0,
            np.log1p(before),
            since / 100.0,
            np.sqrt(since) / 12.0,
            np.log1p(since),
            np.exp(-0.5 * ((since - 48.0) / 24.0) ** 2) * post,
            np.exp(-0.5 * ((since - 72.0) / 30.0) ** 2) * post,
            (since >= 96.0).astype(float) * post,
            (since >= 120.0).astype(float) * post,
            np.clip(since - 120.0, 0, None) / 100.0,
            np.clip(since - 160.0, 0, None) / 100.0,
        ]
    )


def logit_range_y(y: np.ndarray) -> np.ndarray:
    p = np.clip((np.asarray(y, dtype=float) - DIM_LOWER) / (DIM_UPPER - DIM_LOWER), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def inverse_logit_range_y(z: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
    return np.clip(DIM_LOWER + p * (DIM_UPPER - DIM_LOWER), DIM_LOWER, DIM_UPPER)


def transform_dim_y(y: np.ndarray, response_layer: str) -> np.ndarray:
    if response_layer == "absolute":
        return np.asarray(y, dtype=float)
    if response_layer == "logit_range":
        return logit_range_y(y)
    raise ValueError(response_layer)


def inverse_dim_y(z: np.ndarray, response_layer: str) -> np.ndarray:
    if response_layer == "absolute":
        return np.asarray(z, dtype=float)
    if response_layer == "logit_range":
        return inverse_logit_range_y(z)
    raise ValueError(response_layer)


def fit_predict_dim(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    times: np.ndarray,
    inoculation_by_batch: dict[int, float],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    k_best: int = 360,
    c_value: float = 10.0,
    epsilon: float = 0.01,
    response_layer: str = "logit_range",
    use_process: bool = True,
    use_raman: bool = True,
) -> np.ndarray:
    y_train = y[train_idx]
    y_trans = transform_dim_y(y_train, response_layer)
    y_mean = float(np.mean(y_trans))
    y_std = max(float(np.std(y_trans)), 1e-8)
    y_scaled = (y_trans - y_mean) / y_std

    train_blocks = []
    test_blocks = []
    if use_raman:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X[train_idx]))
        X_test = scaler.transform(imputer.transform(X[test_idx]))
        selector = SelectKBest(score_func=f_regression, k=min(k_best, X_train.shape[1]))
        train_blocks.append(selector.fit_transform(X_train, y_scaled))
        test_blocks.append(selector.transform(X_test))

    if use_process:
        process_scaler = StandardScaler()
        P_train = process_scaler.fit_transform(process_time_features(train_idx, groups, times, inoculation_by_batch))
        P_test = process_scaler.transform(process_time_features(test_idx, groups, times, inoculation_by_batch))
        train_blocks.append(P_train)
        test_blocks.append(P_test)
    if not train_blocks:
        raise ValueError("No feature blocks enabled for Dim candidate")
    X_train = np.hstack(train_blocks)
    X_test = np.hstack(test_blocks)

    model = SVR(C=c_value, gamma="scale", epsilon=epsilon)
    model.fit(X_train, y_scaled)
    pred_trans = model.predict(X_test) * y_std + y_mean
    return inverse_dim_y(pred_trans, response_layer)


def dim_candidates() -> list[DimCandidate]:
    return [
        DimCandidate("D1 absolute Raman SelectSVR", "Dim_D1_absolute_Raman_SelectSVR_k360_C10_eps0.01", "absolute", "Raman SelectSVR"),
        DimCandidate("D2 logit-range Raman SelectSVR", "Dim_D2_logit_range_Raman_SelectSVR_k360_C10_eps0.01", "logit_range", "Raman SelectSVR"),
        DimCandidate("D3 absolute + process-time SelectSVR", "Dim_D3_absolute_process_time_SelectSVR_k360_C10_eps0.01", "absolute", "Raman SelectSVR + process-time", use_process=True),
        DimCandidate(
            "DT logit-range process-time-only SelectSVR",
            "Dim_DT_logit_range_process_time_only_SelectSVR_C10_eps0.01",
            "logit_range",
            "process-time only",
            k_best=1,
            use_process=True,
            use_raman=False,
        ),
        DimCandidate("D4 logit-range + process-time SelectSVR", "Dim_D4_logit_range_process_time_SelectSVR_k360_C10_eps0.01", "logit_range", "Raman SelectSVR + process-time", use_process=True),
    ]


def evaluate_candidate(
    candidate: DimCandidate,
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    groups: np.ndarray,
    train_pool: np.ndarray,
    independent_idx: np.ndarray,
    inoculation_by_batch: dict[int, float],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    oof = np.full(len(train_pool), np.nan)
    pos = {idx: i for i, idx in enumerate(train_pool)}
    for train_idx, valid_idx in point_kfold_indices(train_pool, groups, times, y):
        pred = fit_predict_dim(
            X,
            y,
            groups,
            times,
            inoculation_by_batch,
            train_idx,
            valid_idx,
            k_best=candidate.k_best,
            c_value=candidate.c_value,
            epsilon=candidate.epsilon,
            response_layer=candidate.response_layer,
            use_process=candidate.use_process,
            use_raman=candidate.use_raman,
        )
        for idx, value in zip(valid_idx, pred):
            oof[pos[idx]] = value

    valid = np.isfinite(oof)
    cv_points = aggregate_points(train_pool[valid], groups, times, y, oof[valid])
    independent_pred = fit_predict_dim(
        X,
        y,
        groups,
        times,
        inoculation_by_batch,
        train_pool,
        independent_idx,
        k_best=candidate.k_best,
        c_value=candidate.c_value,
        epsilon=candidate.epsilon,
        response_layer=candidate.response_layer,
        use_process=candidate.use_process,
        use_raman=candidate.use_raman,
    )
    independent_points = aggregate_points(independent_idx, groups, times, y, independent_pred)
    cv = metric_dict(cv_points["reference"], cv_points["prediction"])
    independent = metric_dict(independent_points["reference"], independent_points["prediction"])
    row = {
        "target": "Dim",
        "step": candidate.step,
        "model": "SelectSVR",
        "response_layer": candidate.response_layer,
        "features": candidate.features,
        "candidate": candidate.name,
        "process_time": candidate.use_process,
        "cv_n": cv["n"],
        "cv_rmse": cv["rmse"],
        "cv_mae": cv["mae"],
        "cv_bias": cv["bias"],
        "cv_r2": cv["r2"],
        "independent_n": independent["n"],
        "independent_rmse": independent["rmse"],
        "independent_mae": independent["mae"],
        "independent_bias": independent["bias"],
        "independent_r2": independent["r2"],
    }
    for frame in (cv_points, independent_points):
        frame["target"] = "Dim"
        frame["candidate"] = candidate.name
        frame["step"] = candidate.step
    return row, cv_points, independent_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dim logit-range process-time SelectSVR ablation.")
    parser.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "config.yaml")
    parser.add_argument("--label-csv", type=Path, default=ROOT / "data" / "inputs" / "LabelData_time.csv")
    parser.add_argument("--operation-csv", type=Path, default=ROOT / "data" / "inputs" / "Batch_operation_times.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "model_outputs" / "dim_logit_range_process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = pd.read_csv(args.label_csv)
    operations = pd.read_csv(args.operation_csv)
    inoculation_by_batch = dict(zip(operations["Batch"].astype(int), pd.to_numeric(operations["inoculation_time_h"], errors="coerce")))

    X, y, times, groups, label_idx = load_dim_inputs(label, cfg)
    train_pool = label_idx[groups[label_idx] != INDEPENDENT_TEST_BATCH]
    independent_idx = label_idx[groups[label_idx] == INDEPENDENT_TEST_BATCH]

    rows: list[dict[str, object]] = []
    cv_frames: list[pd.DataFrame] = []
    independent_frames: list[pd.DataFrame] = []
    final_cv = pd.DataFrame()
    final_independent = pd.DataFrame()
    final_name = dim_candidates()[-1].name
    for candidate in dim_candidates():
        row, cv_points, independent_points = evaluate_candidate(
            candidate,
            X,
            y,
            times,
            groups,
            train_pool,
            independent_idx,
            inoculation_by_batch,
        )
        rows.append(row)
        cv_frames.append(cv_points)
        independent_frames.append(independent_points)
        if candidate.name == final_name:
            final_cv = cv_points.copy()
            final_independent = independent_points.copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ablation = pd.DataFrame(rows)
    public_ablation = ablation.copy()
    independent_cols = ["independent_n", "independent_rmse", "independent_mae", "independent_bias", "independent_r2"]
    public_ablation.loc[~public_ablation["step"].astype(str).str.startswith("D4 "), independent_cols] = np.nan
    public_ablation.round(6).to_csv(args.out_dir / "dim_logit_range_ablation.csv", index=False, encoding="utf-8-sig")
    pd.concat(cv_frames, ignore_index=True).to_csv(args.out_dir / "dim_logit_range_ablation_cv_point_predictions.csv", index=False, encoding="utf-8-sig")
    ablation_independent_path = args.out_dir / "dim_logit_range_ablation_independent_point_predictions.csv"
    if ablation_independent_path.exists():
        ablation_independent_path.unlink()
    final_cv.to_csv(args.out_dir / "dim_logit_range_final_cv_point_predictions.csv", index=False, encoding="utf-8-sig")
    final_independent.to_csv(args.out_dir / "dim_logit_range_final_independent_point_predictions.csv", index=False, encoding="utf-8-sig")
    print(public_ablation[["step", "cv_n", "cv_rmse", "cv_r2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
