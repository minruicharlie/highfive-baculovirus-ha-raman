from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LABEL_CSV = ROOT / "data" / "inputs" / "LabelData_time.csv"
OPS_CSV = ROOT / "data" / "inputs" / "Batch_operation_times.csv"
PLSR_DIR = ROOT / "results" / "model_outputs" / "plsr_baselines"
OUT = ROOT / "results" / "figures" / "source_data"

INDEPENDENT_TEST_BATCH = 5
SCANS_PER_OFFLINE_POINT = 5
TARGETS = ["VCD", "Via", "Dim", "HA2", "HA4", "Glc", "Gln", "NH4"]
DISPLAY = {"HA2": "eHA", "HA4": "tHA"}
FEATURE_ORDER = [
    "bounded/skewed",
    "infection-event linked",
    "log2/exponential readout",
    "low-resolution/discrete",
    "narrow dynamic span",
]
REFERENCE_REGION_ORDER = ["low", "middle", "high"]
STAGE_ORDER = ["pre", "0-48 h", "48-96 h", ">96 h"]


def display_target(target: str) -> str:
    return DISPLAY.get(str(target), str(target))


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def valid_target_mask(df: pd.DataFrame, target: str) -> np.ndarray:
    y = finite(df[target])
    mask = np.isfinite(y)
    if target in {"HA2", "HA4"}:
        mask &= y > 0
    return mask


def aggregate_offline_points(label: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    batch_values = finite(label["Batch"])
    for target in TARGETS:
        y_all = finite(label[target])
        mask = valid_target_mask(label, target) & (batch_values != INDEPENDENT_TEST_BATCH)
        sub = label.loc[mask, ["Batch", "time", target]].copy()
        sub["Batch"] = pd.to_numeric(sub["Batch"], errors="coerce")
        sub["time"] = pd.to_numeric(sub["time"], errors="coerce")
        sub[target] = pd.to_numeric(sub[target], errors="coerce")
        for batch, group in sub.groupby("Batch", observed=True):
            group = group.sort_values("time", kind="mergesort").reset_index(drop=True)
            point_ids = np.arange(len(group)) // SCANS_PER_OFFLINE_POINT
            agg = group.groupby(point_ids).agg({"time": "mean", target: "mean"})
            for point_no, row in agg.iterrows():
                rows.append(
                    {
                        "target": target,
                        "target_display": display_target(target),
                        "Batch": int(batch),
                        "point_no": int(point_no),
                        "time": float(row["time"]),
                        "reference": float(row[target]),
                    }
                )
    points = pd.DataFrame(rows)
    points["target_display"] = points["target"].map(display_target)
    return points


def add_normalized_scales(points: pd.DataFrame) -> pd.DataFrame:
    out = points.copy()
    out["reference_zero_max"] = np.nan
    out["reference_norm"] = np.nan
    for target, sub in out.groupby("target", observed=True):
        idx = sub.index
        ref = pd.to_numeric(sub["reference"], errors="coerce")
        ref_max = max(float(ref.max()), 1e-12)
        ref_range = max(float(ref.max() - ref.min()), 1e-12)
        out.loc[idx, "reference_zero_max"] = ref / ref_max
        out.loc[idx, "reference_norm"] = (ref - ref.min()) / ref_range
    return out


def event_effects(points: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    inoc = dict(
        zip(
            pd.to_numeric(ops["Batch"], errors="coerce"),
            pd.to_numeric(ops["inoculation_time_h"], errors="coerce"),
        )
    )
    directions = {"VCD": 1, "Via": -1, "Dim": 1, "HA2": 1, "HA4": 1, "Glc": -1, "Gln": -1, "NH4": 1}
    rows = []
    for target, sub_target in points.groupby("target", observed=True):
        ref_all = pd.to_numeric(sub_target["reference"], errors="coerce")
        ref_range = float(ref_all.max() - ref_all.min())
        if not np.isfinite(ref_range) or ref_range <= 0:
            continue
        for batch, sub in sub_target.groupby("Batch", observed=True):
            tau = float(inoc.get(float(batch), np.nan))
            if not np.isfinite(tau):
                continue
            time = pd.to_numeric(sub["time"], errors="coerce")
            ref = pd.to_numeric(sub["reference"], errors="coerce")
            valid = np.isfinite(time) & np.isfinite(ref)
            time = time[valid]
            ref = ref[valid]
            if len(ref) < 2:
                continue
            baseline_idx = (time - tau).abs().idxmin()
            baseline = float(ref.loc[baseline_idx])
            post = ref[(time >= tau) & (time <= tau + 168)]
            if post.empty:
                continue
            extreme = float(post.max() if directions[target] > 0 else post.min())
            rows.append(
                {
                    "target": target,
                    "target_display": display_target(target),
                    "Batch": int(batch),
                    "event_effect": (extreme - baseline) / ref_range * 100.0,
                }
            )
    return pd.DataFrame(rows)


def structure_matrix(points: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    max_effect = max(float(effects.groupby("target")["event_effect"].median().abs().max()), 1e-12)
    rows = []
    for target, sub in points.groupby("target", observed=True):
        ref = pd.to_numeric(sub["reference"], errors="coerce").dropna()
        ref_range = float(ref.max() - ref.min())
        median = max(abs(float(ref.median())), 1e-12)
        norm = ref / max(float(ref.max()), 1e-12)
        skew = abs(float(norm.skew())) if len(norm) > 2 and np.isfinite(norm.skew()) else 0.0
        bounded_skewed = min(skew / 1.8, 1.0)
        if target == "Via":
            bounded_skewed = max(bounded_skewed, 0.95)
        narrow_span = 1.0 - min((ref_range / median) / 1.0, 1.0)
        if target == "Dim":
            narrow_span = max(narrow_span, 0.95)
        discrete = 1.0 - min(ref.nunique() / max(len(ref), 1), 1.0)
        if target in {"HA2", "HA4"}:
            discrete = max(discrete, 0.9)
        log2_scale = 1.0 if target in {"HA2", "HA4"} else 0.0
        event = effects[effects["target"].eq(target)]["event_effect"].median()
        event_linked = min(abs(float(event)) / max_effect, 1.0) if pd.notna(event) else np.nan
        rows.extend(
            [
                {"feature": "bounded/skewed", "target": target, "target_display": display_target(target), "score": bounded_skewed},
                {"feature": "infection-event linked", "target": target, "target_display": display_target(target), "score": event_linked},
                {"feature": "log2/exponential readout", "target": target, "target_display": display_target(target), "score": log2_scale},
                {"feature": "low-resolution/discrete", "target": target, "target_display": display_target(target), "score": discrete},
                {"feature": "narrow dynamic span", "target": target, "target_display": display_target(target), "score": narrow_span},
            ]
        )
    return pd.DataFrame(rows)


def load_plsr_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(PLSR_DIR / "plsr_all_metrics.csv")
    metrics["target"] = metrics["target"].astype(str)
    return metrics


def build_cv_diagnostics(ops: pd.DataFrame) -> pd.DataFrame:
    inoc = dict(
        zip(
            pd.to_numeric(ops["Batch"], errors="coerce"),
            pd.to_numeric(ops["inoculation_time_h"], errors="coerce"),
        )
    )
    cv = pd.read_csv(PLSR_DIR / "plsr_cv_point_predictions.csv")
    for col in ["Batch", "time", "reference", "prediction"]:
        cv[col] = pd.to_numeric(cv[col], errors="coerce")
    cv = cv[np.isfinite(cv["reference"]) & np.isfinite(cv["prediction"])].copy()
    cv["target"] = cv["target"].astype(str)
    cv["residual"] = cv["prediction"] - cv["reference"]
    cv["absolute_error"] = np.abs(cv["residual"])
    cv["absolute_percentage_error"] = cv["absolute_error"] / cv["reference"].abs() * 100.0
    cv["time_after_inoculation"] = [row.time - inoc.get(float(row.Batch), np.nan) for row in cv.itertuples()]
    return cv


def risk_metrics(metrics: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        target = str(row["target"])
        sub = cv[cv["target"].eq(target)].copy()
        ref = pd.to_numeric(sub["reference"], errors="coerce")
        pred = pd.to_numeric(sub["prediction"], errors="coerce")
        valid = np.isfinite(ref) & np.isfinite(pred) & (np.abs(ref) > 0)
        mape = float(np.nanmean(np.abs((pred[valid] - ref[valid]) / ref[valid])) * 100.0) if valid.any() else np.nan
        rows.append(
            {
                "target": target,
                "target_display": target,
                "train_cv_gap": float(row["train_r2"]) - float(row["cv_r2"]),
                "cv_mape_percent": mape,
                "cv_r2": float(row["cv_r2"]),
            }
        )
    return pd.DataFrame(rows)


def stage_label(value: float) -> str | float:
    if not np.isfinite(value):
        return np.nan
    if value < 0:
        return "pre"
    if value < 48:
        return "0-48 h"
    if value < 96:
        return "48-96 h"
    return ">96 h"


def reference_region_summary(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, sub in cv.groupby("target", observed=True):
        ref = pd.to_numeric(sub["reference"], errors="coerce")
        pred = pd.to_numeric(sub["prediction"], errors="coerce")
        ref_max = max(float(ref.max()), 1e-12)
        ref_range = max(float(ref.max() - ref.min()), 1e-12)
        ref_scaled = ref / ref_max
        residual = pred - ref
        signed_range_pct = residual / ref_range * 100.0
        for label, lo, hi in [("low", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("high", 2 / 3, 1.0000001)]:
            mask = (ref_scaled >= lo) & (ref_scaled < hi) & np.isfinite(signed_range_pct)
            rows.append(
                {
                    "process_variable": target,
                    "reference_region": label,
                    "n": int(mask.sum()),
                    "mean_signed_residual_range_pct": float(signed_range_pct[mask].mean()) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def stage_residual_summary(cv: pd.DataFrame) -> pd.DataFrame:
    work = cv.copy()
    work["inoculation_aligned_stage"] = work["time_after_inoculation"].map(stage_label)
    rows = []
    for target, sub_target in work.groupby("target", observed=True):
        ref = pd.to_numeric(sub_target["reference"], errors="coerce")
        ref_range = max(float(ref.max() - ref.min()), 1e-12)
        signed_range_pct = (pd.to_numeric(sub_target["prediction"], errors="coerce") - ref) / ref_range * 100.0
        for stage in STAGE_ORDER:
            mask = sub_target["inoculation_aligned_stage"].eq(stage) & np.isfinite(signed_range_pct)
            rows.append(
                {
                    "process_variable": target,
                    "inoculation_aligned_stage": stage,
                    "n": int(mask.sum()),
                    "mean_signed_residual_range_pct": float(signed_range_pct[mask].mean()) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    label = pd.read_csv(LABEL_CSV)
    ops = pd.read_csv(OPS_CSV)
    points = add_normalized_scales(aggregate_offline_points(label))
    effects = event_effects(points, ops)
    structure = structure_matrix(points, effects)
    metrics = load_plsr_metrics()
    cv = build_cv_diagnostics(ops)
    risk = risk_metrics(metrics, cv)
    by_region = reference_region_summary(cv)
    by_stage = stage_residual_summary(cv)

    points.to_csv(OUT / "development_offline_points_source.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUT / "event_effect_source.csv", index=False, encoding="utf-8-sig")
    structure.to_csv(OUT / "process_variable_data_structure_scores.csv", index=False, encoding="utf-8-sig")
    metric_source_cols = [
        "target",
        "train_n",
        "train_rmse",
        "train_mae",
        "train_bias",
        "train_r2",
        "cv_n",
        "cv_rmse",
        "cv_mae",
        "cv_bias",
        "cv_r2",
        "candidate",
    ]
    metrics.loc[:, [c for c in metric_source_cols if c in metrics.columns]].to_csv(
        OUT / "figure1_plsr_metrics_source.csv", index=False, encoding="utf-8-sig"
    )
    cv.to_csv(OUT / "plsr_cv_point_diagnostics_source.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(OUT / "plsr_percentage_error_source.csv", index=False, encoding="utf-8-sig")
    by_region.to_csv(OUT / "plsr_compact_binned_diagnostic_source.csv", index=False, encoding="utf-8-sig")
    by_stage.to_csv(OUT / "plsr_inoculation_stage_residual_source.csv", index=False, encoding="utf-8-sig")

    counts = {
        "offline_points": len(points),
        "cv_points": len(cv),
        "structure_rows": len(structure),
        "event_rows": len(effects),
    }
    print(pd.Series(counts).to_string())


if __name__ == "__main__":
    main()
