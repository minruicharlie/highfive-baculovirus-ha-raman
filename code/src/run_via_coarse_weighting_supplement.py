from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import run_via_bounded_logit_process_selectsvr as via_mod


ROOT = Path(__file__).resolve().parents[2]


def coarse_inverse_frequency_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    bins = np.asarray([0.0, 100.0 / 3.0, 200.0 / 3.0, 100.0], dtype=float)
    ids = np.digitize(y, bins[1:-1], right=False)
    counts = pd.Series(ids).value_counts().to_dict()
    n_present = max(len(counts), 1)
    weights = np.asarray([len(y) / (n_present * counts.get(i, 1)) for i in ids], dtype=float)
    weights[~np.isfinite(weights)] = 1.0
    weights = np.clip(weights, 0.25, 4.0)
    return weights / max(float(weights.mean()), 1e-12)


def fit_predict_weighted_v4(
    X: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    cand = via_mod.via_candidates()[-1]
    y_train = y[train_idx]
    y_trans = via_mod.transform_y(y_train, cand.response_layer)
    y_mean = float(np.mean(y_trans))
    y_std = max(float(np.std(y_trans)), 1e-8)
    y_scaled = (y_trans - y_mean) / y_std
    weights = coarse_inverse_frequency_weights(y_train)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X[train_idx]))
    X_test = scaler.transform(imputer.transform(X[test_idx]))
    selector = SelectKBest(score_func=f_regression, k=min(cand.k_best, X_train.shape[1]))
    X_train = selector.fit_transform(X_train, y_scaled)
    X_test = selector.transform(X_test)

    process_scaler = StandardScaler()
    X_train = np.hstack([X_train, process_scaler.fit_transform(P[train_idx])])
    X_test = np.hstack([X_test, process_scaler.transform(P[test_idx])])

    model = SVR(C=cand.c_value, gamma="scale", epsilon=cand.epsilon)
    model.fit(X_train, y_scaled, sample_weight=weights)
    pred_trans = model.predict(X_test) * y_std + y_mean
    return np.clip(via_mod.inverse_y(pred_trans, cand.response_layer), 0.0, 100.0)


def evaluate_weighted(X: np.ndarray, P: np.ndarray, y: np.ndarray, groups: np.ndarray, times: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    idx_all = np.where(np.isfinite(y) & np.isfinite(groups) & np.isfinite(times))[0]
    train_pool = idx_all[groups[idx_all] != via_mod.INDEPENDENT_TEST_BATCH]
    test_idx = idx_all[groups[idx_all] == via_mod.INDEPENDENT_TEST_BATCH]
    oof = np.full(len(train_pool), np.nan)
    pos = {idx: i for i, idx in enumerate(train_pool)}
    for train_idx, valid_idx in via_mod.point_kfold_indices(train_pool, groups, times, y):
        pred = fit_predict_weighted_v4(X, P, y, train_idx, valid_idx)
        for idx, value in zip(valid_idx, pred):
            oof[pos[idx]] = value
    valid = np.isfinite(oof)
    cv_points = via_mod.aggregate_points(train_pool[valid], groups, times, y, oof[valid])
    independent_pred = fit_predict_weighted_v4(X, P, y, train_pool, test_idx)
    independent_points = via_mod.aggregate_points(test_idx, groups, times, y, independent_pred)
    cv = via_mod.metric_dict(cv_points["reference"], cv_points["prediction"])
    independent = via_mod.metric_dict(independent_points["reference"], independent_points["prediction"])
    row = {
        "Target": "Via",
        "Route": "V5 coarse inverse-frequency weighted V4",
        "Weighting": "three broad Via strata inverse-frequency weights",
        "CV n": cv["n"],
        "CV RMSE": cv["rmse"],
        "CV R2": cv["r2"],
        "CV MAE": cv["mae"],
        "CV bias": cv["bias"],
    }
    cv_points["candidate"] = "V5_coarse_inverse_frequency_weighted_V4"
    independent_points["candidate"] = "V5_coarse_inverse_frequency_weighted_V4"
    return row, cv_points, independent_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retained Via model with coarse distribution weighting for Supplementary Table S5.")
    parser.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "config.yaml")
    parser.add_argument("--label-csv", type=Path, default=ROOT / "data" / "inputs" / "LabelData_time.csv")
    parser.add_argument("--operation-csv", type=Path, default=ROOT / "data" / "inputs" / "Batch_operation_times.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "model_outputs" / "via_coarse_weighting_supplement")
    parser.add_argument("--table-dir", type=Path, default=ROOT / "results" / "tables")
    args = parser.parse_args()

    cfg = via_mod.load_config(args.config)
    label = pd.read_csv(args.label_csv)
    operation = pd.read_csv(args.operation_csv)
    cols = via_mod.infer_spectral_columns(label, cfg)
    X = via_mod.chemical_feature_matrix(label[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float), cfg)
    P = via_mod.process_time_features(label, operation)
    y = pd.to_numeric(label["Via"], errors="coerce").to_numpy(float)
    groups = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(label["time"], errors="coerce").to_numpy(float)

    row, cv_points, independent_points = evaluate_weighted(X, P, y, groups, times)
    v4 = pd.read_csv(ROOT / "results" / "model_outputs" / "via_bounded_logit_process" / "via_no_weight_selectsvr_ablation_recomputed.csv")
    v4 = v4[v4["step"].astype(str).str.startswith("V4")].iloc[0]
    base = {
        "Target": "Via",
        "Route": "V4 bounded-logit + process-time SelectSVR",
        "Weighting": "none",
        "CV n": v4["cv_n"],
        "CV RMSE": v4["cv_rmse"],
        "CV R2": v4["cv_r2"],
        "CV MAE": v4["cv_mae"],
        "CV bias": v4["cv_bias"],
    }
    table = pd.DataFrame([base, row])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.table_dir / "supplement_table_s5_via_weighting_sensitivity.csv", index=False, encoding="utf-8-sig")
    cv_points.to_csv(args.out_dir / "via_coarse_weighted_cv_point_predictions.csv", index=False, encoding="utf-8-sig")
    independent_path = args.out_dir / "via_coarse_weighted_independent_point_predictions.csv"
    if independent_path.exists():
        independent_path.unlink()
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
