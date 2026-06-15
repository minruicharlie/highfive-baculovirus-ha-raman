from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT_TEST_BATCH = 5
SCANS_PER_OFFLINE_POINT = 5
TARGET_COLUMNS = {"eHA": "HA2", "tHA": "HA4"}
FIRST_N = 15
PLS_GRID = [1, 3, 5, 8]
COMMON_PLSR_COMPONENTS = {"eHA": 3, "tHA": 8}
PLS_PROB_GRID = [6, 8, 10]
LOGISTIC_C_GRID = [0.3, 1.0, 3.0]
PROCESS_WEIGHT_GRID = [0.25, 0.5, 1.0]
TOLERANCE_STEP = 1.0
LAMBDA_MSE = 0.05


@dataclass(frozen=True)
class HACandidate:
    target: str
    step: str
    module: str
    column: str
    first_n: int = FIRST_N
    sg_window: int = 15
    n_components: int = 8
    process_weight: float = 0.0
    logistic_c: float = 1.0
    pooling: str = "step_mean"
    use_raman: bool = True

    @property
    def name(self) -> str:
        c = f"{self.logistic_c:g}".replace(".", "p")
        pw = f"{self.process_weight:g}".replace(".", "p")
        mode = "_processonly" if not self.use_raman else ""
        return f"{self.target}_{self.step}_first{self.first_n}_pls{self.n_components}_ptime{pw}_C{c}_pool{self.pooling}{mode}"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        raise ValueError("No spectral columns found in the configured Raman window.")
    pairs.sort(key=lambda item: item[0])
    return [col for _, col in pairs[::stride]]


def locked_delta(raw: np.ndarray, batches: np.ndarray, times: np.ndarray, first_n: int) -> tuple[np.ndarray, np.ndarray]:
    delta = np.empty_like(raw, dtype=float)
    usable = np.ones(raw.shape[0], dtype=bool)
    for batch in sorted(pd.unique(batches[np.isfinite(batches)])):
        idx = np.where(batches == batch)[0]
        idx = idx[np.argsort(times[idx], kind="mergesort")]
        base_idx = idx[: min(int(first_n), len(idx))]
        baseline = np.nanmedian(raw[base_idx], axis=0)
        delta[idx] = raw[idx] - baseline
        usable[base_idx] = False
    return delta, usable


def chemical_feature_matrix(raw: np.ndarray, cfg: dict, sg_window: int, derivative_order: int) -> np.ndarray:
    spectral = cfg.get("spectral", {})
    raw = np.asarray(raw, dtype=float)
    x_axis = np.linspace(-1.0, 1.0, raw.shape[1])
    vander = np.vander(x_axis, N=int(spectral.get("baseline_degree", 3)) + 1, increasing=False)
    baseline_pinv = np.linalg.pinv(vander)
    corrected = raw - (raw @ baseline_pinv.T) @ vander.T
    window = min(int(sg_window), corrected.shape[1] - (1 - corrected.shape[1] % 2))
    if window % 2 == 0:
        window -= 1
    window = max(5, window)
    poly = min(int(spectral.get("savgol_polyorder", 3)), window - 2)
    try:
        chem = savgol_filter(corrected, window_length=window, polyorder=poly, deriv=derivative_order, axis=1, mode="interp")
    except Exception:
        chem = np.gradient(corrected, axis=1) if derivative_order else corrected
    mean = np.mean(chem, axis=1, keepdims=True)
    std = np.std(chem, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (chem - mean) / std


def process_time_features(label: pd.DataFrame, operations: pd.DataFrame) -> np.ndarray:
    batches = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(label["time"], errors="coerce").to_numpy(float)
    op = operations.set_index("Batch")
    inoc = np.asarray([float(op.loc[int(b), "inoculation_time_h"]) for b in batches], dtype=float)
    signed_since = times - inoc
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
        ]
    )


def point_ids(groups: np.ndarray, target: str) -> np.ndarray:
    ids = np.empty(len(groups), dtype=object)
    for batch in sorted(pd.unique(groups[np.isfinite(groups)])):
        idx = np.where(groups == batch)[0]
        for offset, row in enumerate(idx):
            point_no = offset // SCANS_PER_OFFLINE_POINT
            ids[row] = f"{target}_B{int(float(batch))}_P{point_no:03d}"
    return ids


def kfold_by_point(indices: np.ndarray, pids: np.ndarray, n_splits: int = 5):
    unique_ids = np.array(sorted(pd.unique(pids[indices])))
    splitter = KFold(n_splits=min(n_splits, len(unique_ids)), shuffle=True, random_state=42)
    for train_pos, valid_pos in splitter.split(unique_ids):
        train_set = set(unique_ids[train_pos])
        valid_set = set(unique_ids[valid_pos])
        train_idx = indices[np.array([pid in train_set for pid in pids[indices]])]
        valid_idx = indices[np.array([pid in valid_set for pid in pids[indices]])]
        yield train_idx, valid_idx


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def tolerance_loss_values(y_true: np.ndarray, y_pred_step: np.ndarray) -> np.ndarray:
    err = np.asarray(y_pred_step, dtype=float) - np.asarray(y_true, dtype=float)
    excess = np.maximum(np.abs(err) - TOLERANCE_STEP, 0.0)
    return excess**2 + LAMBDA_MSE * err**2


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    pred_step = np.rint(y_pred)
    err = y_pred - y_true
    low_cut = 5.0 if target == "eHA" else 7.0
    low = y_true <= low_cut
    return {
        "n": int(len(y_true)),
        "rmse": rmse(y_true, y_pred) if len(y_true) else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else np.nan,
        "bias": float(np.mean(err)) if len(y_true) else np.nan,
        "r2": safe_r2(y_true, y_pred) if len(y_true) else np.nan,
        "spearman_r": float(spearmanr(y_true, y_pred).statistic) if len(y_true) >= 3 else np.nan,
        "rounded_within_1_step_pct": float(np.mean(np.abs(pred_step - y_true) <= 1.0) * 100.0) if len(y_true) else np.nan,
        "rounded_over_1_step_pct": float(np.mean(np.abs(pred_step - y_true) > 1.0) * 100.0) if len(y_true) else np.nan,
        "tolerance_loss": float(np.mean(tolerance_loss_values(y_true, pred_step))) if len(y_true) else np.nan,
        "step_mse": float(np.mean((pred_step - y_true) ** 2)) if len(y_true) else np.nan,
        "step_mae": float(np.mean(np.abs(pred_step - y_true))) if len(y_true) else np.nan,
        "low_n": int(np.sum(low)),
        "low_rmse": rmse(y_true[low], y_pred[low]) if np.any(low) else np.nan,
    }


def balance_weights(y: np.ndarray) -> np.ndarray:
    rounded = np.rint(np.asarray(y, dtype=float))
    values, counts = np.unique(rounded, return_counts=True)
    count_map = dict(zip(values, counts))
    weights = np.asarray([len(y) / (len(values) * count_map[v]) for v in rounded], dtype=float)
    weights[~np.isfinite(weights)] = 1.0
    weights = np.clip(weights, 0.1, 30.0)
    return weights / np.mean(weights)


def class_range(y: np.ndarray) -> np.ndarray:
    return np.arange(int(np.floor(np.min(y))), int(np.ceil(np.max(y))) + 1, dtype=float)


def cumulative_to_pmf(lo: int, hi: int, cumulative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.arange(lo, hi + 1, dtype=float)
    if cumulative.shape[1] == 0:
        return classes[:1], np.ones((cumulative.shape[0], 1), dtype=float)
    p_first = 1.0 - cumulative[:, 0]
    middle = cumulative[:, :-1] - cumulative[:, 1:] if cumulative.shape[1] > 1 else np.empty((len(cumulative), 0))
    p_last = cumulative[:, -1]
    probs = np.column_stack([p_first, middle, p_last])
    probs = np.clip(probs, 0.0, None)
    denom = probs.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return classes, probs / denom


def pls_feature_space(
    cand: HACandidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    P_train: np.ndarray,
    X_test: np.ndarray,
    P_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not cand.use_raman:
        if cand.process_weight <= 0:
            raise ValueError(f"{cand.name} needs process-time features when Raman is disabled")
        p_scaler = StandardScaler()
        return (
            p_scaler.fit_transform(P_train) * float(cand.process_weight),
            p_scaler.transform(P_test) * float(cand.process_weight),
        )

    y_weights = balance_weights(y_train)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    y_mean = float(np.average(y_train, weights=y_weights))
    y_std = max(float(np.sqrt(np.average((y_train - y_mean) ** 2, weights=y_weights))), 1e-8)
    y_scaled = (y_train - y_mean) / y_std
    n = max(1, min(int(cand.n_components), Xtr.shape[0] - 1, Xtr.shape[1]))
    pls = PLSRegression(n_components=n, scale=False)
    pls.fit(Xtr, y_scaled)
    Ttr = pls.transform(Xtr)
    Tte = pls.transform(Xte)
    score_scaler = StandardScaler()
    Ftr = score_scaler.fit_transform(Ttr)
    Fte = score_scaler.transform(Tte)
    if cand.process_weight > 0:
        p_scaler = StandardScaler()
        Ptr = p_scaler.fit_transform(P_train) * float(cand.process_weight)
        Pte = p_scaler.transform(P_test) * float(cand.process_weight)
        Ftr = np.hstack([Ftr, Ptr])
        Fte = np.hstack([Fte, Pte])
    return Ftr, Fte


def ordinal_probabilities(cand: HACandidate, F_train: np.ndarray, y_train: np.ndarray, F_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = balance_weights(y_train)
    lo = int(np.floor(np.min(y_train)))
    hi = int(np.ceil(np.max(y_train)))
    probs = []
    for threshold in range(lo + 1, hi + 1):
        z = (y_train >= threshold).astype(int)
        if len(np.unique(z)) < 2:
            probs.append(np.full(F_test.shape[0], float(z[0])))
            continue
        clf = LogisticRegression(C=float(cand.logistic_c), class_weight="balanced", max_iter=2000)
        clf.fit(F_train, z, sample_weight=weights)
        probs.append(clf.predict_proba(F_test)[:, 1])
    if not probs:
        return np.asarray([float(lo)]), np.ones((F_test.shape[0], 1), dtype=float)
    cumulative = np.vstack(probs).T
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    return cumulative_to_pmf(lo, hi, cumulative)


def tolerance_decode(classes: np.ndarray, probs: np.ndarray) -> np.ndarray:
    classes = np.asarray(classes, dtype=float)
    probs = np.asarray(probs, dtype=float)
    loss_matrix = np.zeros((len(classes), len(classes)), dtype=float)
    for j, pred_step in enumerate(classes):
        loss_matrix[j, :] = tolerance_loss_values(classes, np.full(len(classes), pred_step))
    expected_loss = probs @ loss_matrix.T
    expected_value = probs @ classes
    best = expected_loss.min(axis=1)
    pred = np.empty(len(probs), dtype=float)
    for i in range(len(probs)):
        tied = np.where(np.isclose(expected_loss[i], best[i]))[0]
        pred[i] = classes[tied[np.argmin(np.abs(classes[tied] - expected_value[i]))]]
    return pred


def predict_probabilities(
    cand: HACandidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    P_train: np.ndarray,
    X_test: np.ndarray,
    P_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    F_train, F_test = pls_feature_space(cand, X_train, y_train, P_train, X_test, P_test)
    return ordinal_probabilities(cand, F_train, y_train, F_test)


def aggregate_probability_predictions(
    cand: HACandidate,
    positions: np.ndarray,
    data: dict,
    classes: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "point_id": data["pids"][positions],
            "Batch": data["groups"][positions],
            "time": data["times"][positions],
            "reference": data["y"][positions],
        }
    )
    if cand.pooling == "prob_mean":
        rows = []
        for pid, idx in frame.groupby("point_id").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            prob = probs[idx].mean(axis=0, keepdims=True)
            pred = tolerance_decode(classes, prob)[0]
            rows.append(
                {
                    "point_id": pid,
                    "Batch": frame.loc[idx, "Batch"].iloc[0],
                    "time": frame.loc[idx, "time"].mean(),
                    "reference": frame.loc[idx, "reference"].mean(),
                    "prediction": pred,
                    "prediction_sd": 0.0,
                    "n_scans": len(idx),
                }
            )
        points = pd.DataFrame(rows).sort_values(["Batch", "time"]).reset_index(drop=True)
    elif cand.pooling == "step_mean":
        frame["scan_prediction"] = tolerance_decode(classes, probs)
        points = (
            frame.groupby("point_id", as_index=False)
            .agg(
                Batch=("Batch", "first"),
                time=("time", "mean"),
                reference=("reference", "mean"),
                prediction=("scan_prediction", "mean"),
                prediction_sd=("scan_prediction", "std"),
                n_scans=("scan_prediction", "size"),
            )
            .sort_values(["Batch", "time"])
            .fillna({"prediction_sd": 0.0})
            .reset_index(drop=True)
        )
    else:
        raise ValueError(cand.pooling)
    points["target"] = cand.target
    points["step"] = cand.step
    points["module"] = cand.module
    points["candidate"] = cand.name
    return points


def make_probability_data(cand: HACandidate, X_all: np.ndarray, P_all: np.ndarray, usable: np.ndarray, label: pd.DataFrame) -> dict:
    y_raw = pd.to_numeric(label[cand.column], errors="coerce").to_numpy(float)
    groups_all = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
    times_all = pd.to_numeric(label["time"], errors="coerce").to_numpy(float)
    base_mask = np.isfinite(y_raw) & (y_raw > 0) & np.isfinite(groups_all) & np.isfinite(times_all)
    base_idx = np.where(base_mask)[0]
    base_pids = point_ids(groups_all[base_idx], cand.target)
    keep = usable[base_idx]
    idx = base_idx[keep]
    return {
        "X": X_all[idx],
        "P": P_all[idx],
        "y": y_raw[idx],
        "groups": groups_all[idx],
        "times": times_all[idx],
        "pids": base_pids[keep],
    }


def evaluate_probability_candidate(cand: HACandidate, X_all: np.ndarray, P_all: np.ndarray, usable: np.ndarray, label: pd.DataFrame):
    data = make_probability_data(cand, X_all, P_all, usable, label)
    X = data["X"]
    P = data["P"]
    y = data["y"]
    groups = data["groups"]
    pids = data["pids"]
    all_idx = np.arange(len(y))
    train_pool = all_idx[groups != INDEPENDENT_TEST_BATCH]
    independent_idx = all_idx[groups == INDEPENDENT_TEST_BATCH]

    cv_frames = []
    for train_idx, valid_idx in kfold_by_point(train_pool, pids):
        classes, probs = predict_probabilities(cand, X[train_idx], y[train_idx], P[train_idx], X[valid_idx], P[valid_idx])
        cv_frames.append(aggregate_probability_predictions(cand, valid_idx, data, classes, probs))
    cv_points = pd.concat(cv_frames, ignore_index=True)

    classes, probs = predict_probabilities(cand, X[train_pool], y[train_pool], P[train_pool], X[independent_idx], P[independent_idx])
    independent_points = aggregate_probability_predictions(cand, independent_idx, data, classes, probs)
    cv = metrics_dict(cv_points["reference"], cv_points["prediction"], cand.target)
    independent = metrics_dict(independent_points["reference"], independent_points["prediction"], cand.target)
    row = {
        "target": cand.target,
        "step": cand.step,
        "module": cand.module,
        "model_head": "ordinal-threshold probability",
        "candidate": cand.name,
        "first_n": cand.first_n,
        "n_components": cand.n_components,
        "process_weight": cand.process_weight,
        "logistic_c": cand.logistic_c,
        "pooling": cand.pooling,
        **{f"cv_{k}": v for k, v in cv.items()},
        **{f"independent_{k}": v for k, v in independent.items()},
    }
    return row, cv_points, independent_points


def baseline_d1_feature_matrix(raw: np.ndarray, cfg: dict, sg_window: int = 15) -> np.ndarray:
    return chemical_feature_matrix(raw, cfg, sg_window=sg_window, derivative_order=1)


def make_plsr_data(target: str, column: str, X_all: np.ndarray, label: pd.DataFrame) -> dict:
    y_raw = pd.to_numeric(label[column], errors="coerce").to_numpy(float)
    groups_all = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
    times_all = pd.to_numeric(label["time"], errors="coerce").to_numpy(float)
    mask = np.isfinite(y_raw) & (y_raw > 0) & np.isfinite(groups_all) & np.isfinite(times_all)
    X = X_all[mask]
    y = y_raw[mask]
    groups = groups_all[mask]
    times = times_all[mask]
    pids = point_ids(groups, target)
    return {"X": X, "y": y, "groups": groups, "times": times, "pids": pids}


def fit_predict_plsr(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_components: int) -> np.ndarray:
    n = max(1, min(int(n_components), len(y_train) - 1, X_train.shape[1]))
    model = make_pipeline(SimpleImputer(strategy="median"), PLSRegression(n_components=n, scale=True))
    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_test)).ravel()
    hi = float(np.nanmax(y_train)) if np.isfinite(y_train).any() else 14.0
    return np.clip(pred, 0.0, max(hi, 1.0))


def aggregate_plsr_predictions(target: str, positions: np.ndarray, data: dict, pred: np.ndarray, candidate: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "point_id": data["pids"][positions],
            "Batch": data["groups"][positions],
            "time": data["times"][positions],
            "reference": data["y"][positions],
            "scan_prediction": np.asarray(pred, dtype=float),
        }
    )
    points = (
        frame.groupby("point_id", as_index=False)
        .agg(
            Batch=("Batch", "first"),
            time=("time", "mean"),
            reference=("reference", "mean"),
            prediction=("scan_prediction", "mean"),
            prediction_sd=("scan_prediction", "std"),
            n_scans=("scan_prediction", "size"),
        )
        .sort_values(["Batch", "time"])
        .fillna({"prediction_sd": 0.0})
        .reset_index(drop=True)
    )
    points["target"] = target
    points["step"] = "B0"
    points["module"] = "Model baseline"
    points["candidate"] = candidate
    return points


def evaluate_plsr_candidate(target: str, column: str, X_all: np.ndarray, label: pd.DataFrame, n_components: int):
    data = make_plsr_data(target, column, X_all, label)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    pids = data["pids"]
    all_idx = np.arange(len(y))
    train_pool = all_idx[groups != INDEPENDENT_TEST_BATCH]
    independent_idx = all_idx[groups == INDEPENDENT_TEST_BATCH]
    candidate = f"{target}_B0_PLSR_full_d1_n{int(n_components)}"

    cv_frames = []
    for train_idx, valid_idx in kfold_by_point(train_pool, pids):
        pred = fit_predict_plsr(X[train_idx], y[train_idx], X[valid_idx], n_components)
        cv_frames.append(aggregate_plsr_predictions(target, valid_idx, data, pred, candidate))
    cv_points = pd.concat(cv_frames, ignore_index=True)

    independent_pred = fit_predict_plsr(X[train_pool], y[train_pool], X[independent_idx], n_components)
    independent_points = aggregate_plsr_predictions(target, independent_idx, data, independent_pred, candidate)
    cv = metrics_dict(cv_points["reference"], cv_points["prediction"], target)
    independent = metrics_dict(independent_points["reference"], independent_points["prediction"], target)
    row = {
        "target": target,
        "step": "B0",
        "module": "Model baseline",
        "model_head": "PLSR",
        "candidate": candidate,
        "n_components": int(n_components),
        **{f"cv_{k}": v for k, v in cv.items()},
        **{f"independent_{k}": v for k, v in independent.items()},
    }
    return row, cv_points, independent_points


def probability_candidates(target: str, column: str, step: str, module: str) -> list[HACandidate]:
    if step == "T0":
        return [
            HACandidate(
                target,
                step,
                module,
                column,
                n_components=0,
                process_weight=1.0,
                logistic_c=c,
                pooling="prob_mean",
                use_raman=False,
            )
            for c in LOGISTIC_C_GRID
        ]
    if step == "P0":
        return [
            HACandidate(target, step, module, column, n_components=n, process_weight=0.0, logistic_c=c, pooling="step_mean")
            for n in PLS_PROB_GRID
            for c in LOGISTIC_C_GRID
        ]
    if step == "P1":
        return [
            HACandidate(target, step, module, column, n_components=n, process_weight=w, logistic_c=c, pooling="step_mean")
            for n in PLS_PROB_GRID
            for w in PROCESS_WEIGHT_GRID
            for c in LOGISTIC_C_GRID
        ]
    if step == "H1":
        return [
            HACandidate(target, step, module, column, n_components=n, process_weight=0.0, logistic_c=c, pooling="prob_mean")
            for n in PLS_PROB_GRID
            for c in LOGISTIC_C_GRID
        ]
    if step == "H2":
        return [
            HACandidate(target, step, module, column, n_components=n, process_weight=w, logistic_c=c, pooling="prob_mean")
            for n in PLS_PROB_GRID
            for w in PROCESS_WEIGHT_GRID
            for c in LOGISTIC_C_GRID
        ]
    raise ValueError(step)


def selection_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["cv_tolerance_loss", "cv_rounded_within_1_step_pct", "cv_rmse"], ascending=[True, False, True])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HA ordinal probability ablation with tolerance-aware model selection.")
    parser.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "config.yaml")
    parser.add_argument("--label-csv", type=Path, default=ROOT / "data" / "inputs" / "LabelData_time.csv")
    parser.add_argument("--operation-csv", type=Path, default=ROOT / "data" / "inputs" / "Batch_operation_times.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "model_outputs" / "ha_ordinal_tolerance")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label = pd.read_csv(args.label_csv)
    operations = pd.read_csv(args.operation_csv)
    cols = infer_spectral_columns(label, cfg)
    raw = label[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    P_all = process_time_features(label, operations)
    raw_d1 = baseline_d1_feature_matrix(raw, cfg, sg_window=15)
    batches = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(label["time"], errors="coerce").to_numpy(float)

    selected_rows = []
    all_rows = []
    cv_frames = []
    independent_frames = []

    stages = [
        ("T0", "process-time-only + HA-aware readout"),
        ("P0", "Probability baseline"),
        ("P1", "P0 + process-time descriptors"),
        ("H1", "P0 + HA-aware readout"),
        ("H2", "P0 + process-time + HA-aware readout"),
    ]

    for target, column in TARGET_COLUMNS.items():
        row, cv_points, independent_points = evaluate_plsr_candidate(
            target, column, raw_d1, label, COMMON_PLSR_COMPONENTS[target]
        )
        selected_rows.append(row)
        all_rows.append(pd.DataFrame([row]))
        cv_frames.append(cv_points)
        independent_frames.append(independent_points)

        stage_candidates = [cand for step, module in stages for cand in probability_candidates(target, column, step, module)]
        cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for cand in stage_candidates:
            key = (cand.first_n, cand.sg_window)
            if key not in cache:
                delta, usable = locked_delta(raw, batches, times, cand.first_n)
                cache[key] = (chemical_feature_matrix(delta, cfg, cand.sg_window, derivative_order=0), usable)

        for step, module in stages:
            rows = []
            by_candidate_cv = {}
            by_candidate_independent = {}
            for cand in probability_candidates(target, column, step, module):
                X_all, usable = cache[(cand.first_n, cand.sg_window)]
                row, cv_points, independent_points = evaluate_probability_candidate(cand, X_all, P_all, usable, label)
                rows.append(row)
                by_candidate_cv[row["candidate"]] = cv_points
                by_candidate_independent[row["candidate"]] = independent_points
            stage_table = pd.DataFrame(rows)
            best = selection_sort(stage_table).iloc[0]
            selected_rows.append(best.to_dict())
            all_rows.append(stage_table)
            cv_frames.append(by_candidate_cv[best["candidate"]])
            independent_frames.append(by_candidate_independent[best["candidate"]])

    metrics = pd.DataFrame(selected_rows)
    order_target = pd.CategoricalDtype(["eHA", "tHA"], ordered=True)
    order_step = pd.CategoricalDtype(["B0", "T0", "P0", "P1", "H1", "H2"], ordered=True)
    metrics["target"] = metrics["target"].astype(order_target)
    metrics["step"] = metrics["step"].astype(order_step)
    metrics = metrics.sort_values(["target", "step"]).reset_index(drop=True)
    all_candidates = pd.concat(all_rows, ignore_index=True)
    all_candidates["target"] = all_candidates["target"].astype(order_target)
    all_candidates["step"] = all_candidates["step"].astype(order_step)
    all_candidates = all_candidates.sort_values(["target", "step", "cv_tolerance_loss"]).reset_index(drop=True)
    cv_predictions = pd.concat(cv_frames, ignore_index=True)
    independent_predictions = pd.concat(independent_frames, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    independent_cols = [c for c in metrics.columns if c.startswith("independent_")]
    final_or_baseline = metrics["step"].astype(str).isin(["B0", "H2"])
    public_metrics = metrics.copy()
    public_metrics.loc[~final_or_baseline, independent_cols] = np.nan
    all_candidate_independent_cols = [c for c in all_candidates.columns if c.startswith("independent_")]
    public_all_candidates = all_candidates.drop(columns=all_candidate_independent_cols)
    public_independent_predictions = independent_predictions[independent_predictions["step"].astype(str).isin(["B0", "H2"])].copy()

    public_metrics.to_csv(args.out_dir / "ha_ordinal_tolerance_ablation.csv", index=False, encoding="utf-8-sig")
    public_all_candidates.to_csv(args.out_dir / "ha_ordinal_tolerance_all_candidates.csv", index=False, encoding="utf-8-sig")
    cv_predictions.to_csv(args.out_dir / "ha_ordinal_tolerance_cv_point_predictions.csv", index=False, encoding="utf-8-sig")
    public_independent_predictions.to_csv(args.out_dir / "ha_ordinal_tolerance_independent_point_predictions.csv", index=False, encoding="utf-8-sig")
    show_cols = [
        "target",
        "step",
        "module",
        "cv_n",
        "cv_tolerance_loss",
        "cv_rmse",
        "cv_rounded_within_1_step_pct",
        "independent_n",
        "independent_tolerance_loss",
        "independent_rmse",
        "independent_rounded_within_1_step_pct",
    ]
    print(public_metrics[show_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
