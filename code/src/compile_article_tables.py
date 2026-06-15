from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[2]
TARGET_ORDER = ["VCD", "Via", "Dim", "eHA", "tHA", "Glc", "Gln", "NH4"]
PLSR_COMPONENTS = {"VCD": 8, "Via": 5, "Dim": 8, "eHA": 3, "tHA": 8, "Glc": 3, "Gln": 8, "NH4": 8}
MAIN_TABLE_TARGET_LABELS = {"Glc": "Glc (g/L)", "Gln": "Gln (mM)", "NH4": "NH4 (mM)"}


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def point_metrics(frame: pd.DataFrame, *, vcd_scale: bool = False) -> dict[str, float | int]:
    ref = pd.to_numeric(frame["reference"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(float)
    mask = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[mask]
    pred = pred[mask]
    scale = 1e6 if vcd_scale else 1.0
    err = (pred - ref) / scale
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))) if mask.any() else np.nan,
        "r2": safe_r2(ref, pred),
        "mae": float(np.mean(np.abs(err))) if mask.any() else np.nan,
        "bias": float(np.mean(err)) if mask.any() else np.nan,
    }


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: round(float(v), digits) if pd.notna(v) else v)
    return out


def load_plsr_metrics(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "results" / "model_outputs" / "plsr_baselines" / "plsr_all_metrics.csv")


def plsr_row(metrics: pd.DataFrame, target: str) -> pd.Series:
    return metrics[metrics["target"].astype(str).eq(target)].iloc[0]


def table2_ha(root: Path) -> pd.DataFrame:
    ha = pd.read_csv(root / "results" / "model_outputs" / "ha_ordinal_tolerance" / "ha_ordinal_tolerance_ablation.csv")
    out = ha[
        [
            "target",
            "step",
            "module",
            "cv_n",
            "cv_tolerance_loss",
            "cv_rmse",
            "cv_mae",
            "cv_bias",
            "cv_rounded_within_1_step_pct",
        ]
    ].copy()
    return out.rename(
        columns={
            "target": "Target",
            "step": "Route",
            "module": "Module",
            "cv_n": "CV n",
            "cv_tolerance_loss": "CV tolerance loss",
            "cv_rmse": "CV RMSE",
            "cv_mae": "CV MAE",
            "cv_bias": "CV bias",
            "cv_rounded_within_1_step_pct": "CV rounded ±1 (%)",
        }
    )


def table3_independent(root: Path, plsr_metrics: pd.DataFrame) -> pd.DataFrame:
    via = pd.read_csv(root / "results" / "model_outputs" / "via_bounded_logit_process" / "via_no_weight_selectsvr_ablation_recomputed.csv")
    dim = pd.read_csv(root / "results" / "model_outputs" / "dim_logit_range_process" / "dim_logit_range_ablation.csv")
    ha = pd.read_csv(root / "results" / "model_outputs" / "ha_ordinal_tolerance" / "ha_ordinal_tolerance_ablation.csv")
    rows = []

    def add_plsr(target: str, final_same: bool = True) -> None:
        row = plsr_row(plsr_metrics, target)
        metric = {
            "Target": target,
            "Route": "PLSR baseline",
            "Test n": int(row["independent_n"]),
            "RMSE": float(row["independent_rmse"]) / (1e6 if target == "VCD" else 1.0),
            "R²": float(row["independent_r2"]),
            "MAE": float(row["independent_mae"]) / (1e6 if target == "VCD" else 1.0),
            "Bias": float(row["independent_bias"]) / (1e6 if target == "VCD" else 1.0),
            "Tolerance loss": np.nan,
            "Rounded ±1 (%)": np.nan,
        }
        rows.append(metric)
        if final_same:
            final = metric.copy()
            final["Route"] = "Final model"
            rows.append(final)

    add_plsr("VCD", final_same=True)

    via_plsr = plsr_row(plsr_metrics, "Via")
    rows.append(
        {
            "Target": "Via",
            "Route": "PLSR baseline",
            "Test n": int(via_plsr["independent_n"]),
            "RMSE": float(via_plsr["independent_rmse"]),
            "R²": float(via_plsr["independent_r2"]),
            "MAE": float(via_plsr["independent_mae"]),
            "Bias": float(via_plsr["independent_bias"]),
            "Tolerance loss": np.nan,
            "Rounded ±1 (%)": np.nan,
        }
    )
    via_final = via[via["step"].astype(str).str.startswith("V4 ")].iloc[0]
    rows.append(
        {
            "Target": "Via",
            "Route": "Optimized/final",
            "Test n": int(via_final["independent_test_n"]),
            "RMSE": float(via_final["independent_test_rmse"]),
            "R²": float(via_final["independent_test_r2"]),
            "MAE": float(via_final["independent_test_mae"]),
            "Bias": float(via_final["independent_test_bias"]),
            "Tolerance loss": np.nan,
            "Rounded ±1 (%)": np.nan,
        }
    )

    dim_plsr = plsr_row(plsr_metrics, "Dim")
    rows.append(
        {
            "Target": "Dim",
            "Route": "PLSR baseline",
            "Test n": int(dim_plsr["independent_n"]),
            "RMSE": float(dim_plsr["independent_rmse"]),
            "R²": float(dim_plsr["independent_r2"]),
            "MAE": float(dim_plsr["independent_mae"]),
            "Bias": float(dim_plsr["independent_bias"]),
            "Tolerance loss": np.nan,
            "Rounded ±1 (%)": np.nan,
        }
    )
    dim_final = dim[dim["step"].astype(str).str.startswith("D4 ")].iloc[0]
    rows.append(
        {
            "Target": "Dim",
            "Route": "Optimized/final",
            "Test n": int(dim_final["independent_n"]),
            "RMSE": float(dim_final["independent_rmse"]),
            "R²": float(dim_final["independent_r2"]),
            "MAE": float(dim_final["independent_mae"]),
            "Bias": float(dim_final["independent_bias"]),
            "Tolerance loss": np.nan,
            "Rounded ±1 (%)": np.nan,
        }
    )

    for target in ["Glc", "Gln", "NH4"]:
        add_plsr(target, final_same=True)

    for target in ["eHA", "tHA"]:
        for route in ["B0", "H2"]:
            src = ha[(ha["target"].eq(target)) & (ha["step"].eq(route))].iloc[0]
            rows.append(
                {
                    "Target": target,
                    "Route": "PLSR baseline" if route == "B0" else "Optimized/final",
                    "Test n": int(src["independent_n"]),
                    "RMSE": float(src["independent_rmse"]),
                    "R²": float(src["independent_r2"]),
                    "MAE": float(src["independent_mae"]),
                    "Bias": float(src["independent_bias"]),
                    "Tolerance loss": float(src["independent_tolerance_loss"]),
                    "Rounded ±1 (%)": float(src["independent_rounded_within_1_step_pct"]),
                }
            )

    out = pd.DataFrame(rows)
    out["Target"] = pd.Categorical(out["Target"], TARGET_ORDER, ordered=True)
    out = out.sort_values(["Target", "Route"]).reset_index(drop=True)
    out["Target"] = out["Target"].astype(str).replace(MAIN_TABLE_TARGET_LABELS)
    return out


def supplement_s1_counts(root: Path) -> pd.DataFrame:
    label = pd.read_csv(root / "data" / "inputs" / "LabelData_time.csv", usecols=lambda c: c in {"Batch", "VCD", "Via", "Dim", "Glc", "Gln", "NH4", "HA2", "HA4"})
    online = pd.read_csv(root / "data" / "inputs" / "UnlabelRamanData.csv", usecols=lambda c: c in {"Batch"})
    rows = []
    for batch in sorted(pd.to_numeric(online["Batch"], errors="coerce").dropna().astype(int).unique()):
        sub = label[pd.to_numeric(label["Batch"], errors="coerce").eq(batch)]
        rows.append(
            {
                "Batch": "Independent test" if batch == 5 else f"Batch{batch}",
                "Role": "Independent test" if batch == 5 else "Calibration",
                "Full online Raman spectra": int(pd.to_numeric(online["Batch"], errors="coerce").eq(batch).sum()),
                "Matched spectra rows (VCD/Via/Dim)": int(sub["VCD"].notna().sum()),
                "Matched spectra rows (Glc/Gln/NH4)": int(sub["Glc"].notna().sum()),
                "Matched spectra rows (eHA positive)": int((pd.to_numeric(sub["HA2"], errors="coerce") > 0).sum()),
                "Matched spectra rows (tHA positive)": int((pd.to_numeric(sub["HA4"], errors="coerce") > 0).sum()),
                "Offline points (VCD/Via/Dim)": int(sub["VCD"].notna().sum() / 5),
                "Offline points (Glc/Gln/NH4)": int(sub["Glc"].notna().sum() / 5),
                "Positive offline points (eHA)": int((pd.to_numeric(sub["HA2"], errors="coerce") > 0).sum() / 5),
                "Positive offline points (tHA)": int((pd.to_numeric(sub["HA4"], errors="coerce") > 0).sum() / 5),
            }
        )
    out = pd.DataFrame(rows)
    total = out.drop(columns=["Batch", "Role"]).sum(numeric_only=True)
    rows_total = {"Batch": "Total", "Role": "All batches", **{k: int(v) for k, v in total.items()}}
    return pd.concat([out, pd.DataFrame([rows_total])], ignore_index=True)


def supplement_s2_via_dim(root: Path, plsr_metrics: pd.DataFrame) -> pd.DataFrame:
    via = pd.read_csv(root / "results" / "model_outputs" / "via_bounded_logit_process" / "via_no_weight_selectsvr_ablation_recomputed.csv")
    dim = pd.read_csv(root / "results" / "model_outputs" / "dim_logit_range_process" / "dim_logit_range_ablation.csv")
    rows = []
    via0 = plsr_row(plsr_metrics, "Via")
    rows.append(
        {
            "Target": "Via",
            "Step": "V0 PLSR common baseline",
            "Module": "Full Raman PLSR",
            "CV n": int(via0["cv_n"]),
            "CV RMSE": float(via0["cv_rmse"]),
            "CV R²": float(via0["cv_r2"]),
        }
    )
    for _, row in via.iterrows():
        rows.append(
            {
                "Target": "Via",
                "Step": row["step"],
                "Module": row["features"],
                "CV n": int(row["cv_n"]),
                "CV RMSE": float(row["cv_rmse"]),
                "CV R²": float(row["cv_r2"]),
            }
        )
    dim0 = plsr_row(plsr_metrics, "Dim")
    rows.append(
        {
            "Target": "Dim",
            "Step": "D0 PLSR common baseline",
            "Module": "Full Raman PLSR",
            "CV n": int(dim0["cv_n"]),
            "CV RMSE": float(dim0["cv_rmse"]),
            "CV R²": float(dim0["cv_r2"]),
        }
    )
    for _, row in dim.iterrows():
        rows.append(
            {
                "Target": "Dim",
                "Step": row["step"],
                "Module": row["features"],
                "CV n": int(row["cv_n"]),
                "CV RMSE": float(row["cv_rmse"]),
                "CV R²": float(row["cv_r2"]),
            }
        )
    return pd.DataFrame(rows)


def supplement_s3_model_settings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["VCD", "500-1800 cm^-1", "baseline + SG d1 + standardization", "identity", "full Raman PLSR", "PLSR", "n_components in [1, 3, 5, 8]; selected n=8", "No", "CV RMSE"],
            ["Glc", "500-1800 cm^-1", "baseline + SG d1 + standardization", "identity", "full Raman PLSR", "PLSR", "n_components in [1, 3, 5, 8]; selected n=3", "No", "CV RMSE"],
            ["Gln", "500-1800 cm^-1", "baseline + SG d1 + standardization", "identity", "full Raman PLSR", "PLSR", "n_components in [1, 3, 5, 8]; selected n=8", "No", "CV RMSE"],
            ["NH4", "500-1800 cm^-1", "baseline + SG d1 + standardization", "identity", "full Raman PLSR", "PLSR", "n_components in [1, 3, 5, 8]; selected n=8", "No", "CV RMSE"],
            ["Via", "500-1800 cm^-1", "baseline + SG d1 + standardization", "bounded logit", "SelectKBest k=80", "SVR", "C=10, gamma=scale, epsilon=0.05", "Yes", "CV RMSE"],
            ["Dim", "500-1800 cm^-1", "baseline + SG d1 + standardization", "logit-range, 15.0-20.5 um", "SelectKBest k=360", "SVR", "C=10, gamma=scale, epsilon=0.01", "Yes", "CV RMSE"],
            ["eHA/tHA", "500-1800 cm^-1", "first15 delta + baseline + SG d0 + standardization", "log2 dilution step", "PLS latent scores", "ordinal-threshold probability", "firstN=15; PLS n in [6, 8, 10]; C in [0.3, 1.0, 3.0]; process weight in [0.25, 0.5, 1.0] where applicable", "Route-dependent", "tolerance-aware loss"],
        ],
        columns=["Target", "Raman window", "Spectral preprocessing", "Response scale", "Feature representation", "Model head", "Hyperparameters", "Process-time descriptors", "Selection metric"],
    )


def supplement_s4_plsr_parameters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Target": target,
                "PLSR candidate": f"BasePLSR_full_d1_n{PLSR_COMPONENTS[target]}",
                "Raman window": "500-1800 cm^-1",
                "Preprocessing": "baseline + SG d1 + standardization; median imputation in model pipeline",
                "Response scale": "log2 dilution step; positive labels only" if target in {"eHA", "tHA"} else "identity",
                "PLSR components": PLSR_COMPONENTS[target],
                "Selection criterion": "Calibration-set 5-fold CV RMSE",
            }
            for target in TARGET_ORDER
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile clean manuscript and supplement tables.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    plsr = load_plsr_metrics(root)
    table1 = pd.read_csv(tables / "table1_plsr_train_cv.csv")
    table2 = table2_ha(root)
    table3 = table3_independent(root, plsr)
    s1 = supplement_s1_counts(root)
    s2 = supplement_s2_via_dim(root, plsr)
    s3 = supplement_s3_model_settings()
    s4 = supplement_s4_plsr_parameters()

    outputs = {
        "table1_plsr_train_cv.csv": table1,
        "table2_ha_cv_ablation.csv": table2,
        "table3_independent_test_errors.csv": table3,
        "supplement_table_s1_dataset_counts.csv": s1,
        "supplement_table_s2_model_settings.csv": s3,
        "supplement_table_s3_plsr_parameters.csv": s4,
        "supplement_table_s4_via_dim_ablation.csv": s2,
    }
    for name, frame in outputs.items():
        round_numeric(frame).to_csv(tables / name, index=False, encoding="utf-8-sig")

    print("Table 1")
    print(round_numeric(table1).to_string(index=False))
    print("\nTable 2")
    print(round_numeric(table2).to_string(index=False))
    print("\nTable 3")
    print(round_numeric(table3).to_string(index=False))


if __name__ == "__main__":
    main()
