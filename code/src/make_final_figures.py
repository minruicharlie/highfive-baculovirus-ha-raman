from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import make_figure1
import make_figure1_source_data
import run_dim_logit_range_process_selectsvr as dim_mod
import run_ha_ordinal_tolerance_ablation as ha_mod
import run_plsr_baselines as plsr_mod
import run_via_bounded_logit_process_selectsvr as via_mod


LABEL_CSV = ROOT / "data" / "inputs" / "LabelData_time.csv"
ONLINE_CSV = ROOT / "data" / "inputs" / "UnlabelRamanData.csv"
OPS_CSV = ROOT / "data" / "inputs" / "Batch_operation_times.csv"
CONFIG = ROOT / "code" / "configs" / "config.yaml"
OUT = ROOT / "results" / "figures"
SRC_OUT = OUT / "source_data"

INDEPENDENT_TEST_BATCH = 5
SCANS_PER_OFFLINE_POINT = 5
TARGETS = ["VCD", "Via", "Dim", "eHA", "tHA", "Glc", "Gln", "NH4"]
INTERNAL_TARGET = {"eHA": "HA2", "tHA": "HA4"}
DISPLAY_TARGET = {"HA2": "eHA", "HA4": "tHA"}
PLSR_COMPONENTS = {"VCD": 8, "Via": 5, "Dim": 8, "HA2": 3, "HA4": 8, "Glc": 3, "Gln": 8, "NH4": 8}

GREY = "#8C8C8C"
RED = "#C9543D"
BLUE = "#28577A"
VIA_GREEN = "#77A88D"
DIM_ORANGE = "#D99A4E"
EHA_PURPLE = "#8B73B3"
THA_ROSE = "#B86F7A"
BLACK = "#1F1F1F"
INOC_RED = "#D77A7A"
TIME_GOLD = "#C7A43A"
RAMAN_BLUE = "#4F80A8"
COMBINED_RED = "#B84E3E"
READOUT_TEAL = "#5B9A8E"


def display_target(target: str) -> str:
    return DISPLAY_TARGET.get(str(target), str(target))


def internal_target(target: str) -> str:
    return INTERNAL_TARGET.get(str(target), str(target))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=17, fontweight="bold")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.20)


def ablation_bar_color(step: str, features: str = "") -> str:
    text = f"{step} {features}".lower()
    if "plsr" in text or step.startswith("B0"):
        return GREY
    if "time-only" in text or "process-time only" in text:
        return TIME_GOLD
    if step.startswith(("V4", "D4", "H2")) or "final" in text:
        return COMBINED_RED
    if "h1" in text or "ha-aware" in text:
        return READOUT_TEAL
    if "process-time" in text or "+time" in text or step.startswith(("V3", "V4", "D3", "D4", "P1", "H2")):
        return COMBINED_RED
    return RAMAN_BLUE


def via_ablation_label(step: str) -> str:
    if step.startswith("V0"):
        return "V0\nPLSR"
    if step.startswith("V1"):
        return "V1\nRaman"
    if step.startswith("V2"):
        return "V2\nlogit\nRaman"
    if step.startswith("V3"):
        return "V3\nRaman+\ntime"
    if step.startswith("VT"):
        return "VT\ntime\nonly"
    if step.startswith("V4"):
        return "V4\nfinal"
    return step.replace(" ", "\n")


def dim_ablation_label(step: str) -> str:
    if step.startswith("D0"):
        return "D0\nPLSR"
    if step.startswith("D1"):
        return "D1\nRaman"
    if step.startswith("D2"):
        return "D2\nlogit\nRaman"
    if step.startswith("D3"):
        return "D3\nRaman+\ntime"
    if step.startswith("DT"):
        return "DT\ntime\nonly"
    if step.startswith("D4"):
        return "D4\nfinal"
    return step.replace(" ", "\n")


def load_config() -> dict:
    with CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(float)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))) if mask.any() else np.nan


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def metric_row(label: str, points: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "model": label,
        "n": int(len(points)),
        "rmse": rmse(points["reference"], points["prediction"]),
        "r2": safe_r2(points["reference"], points["prediction"]),
    }


def point_id_series(groups: np.ndarray, target: str) -> np.ndarray:
    ids = np.empty(len(groups), dtype=object)
    for batch in sorted(pd.unique(groups[np.isfinite(groups)])):
        idx = np.where(groups == batch)[0]
        for offset, row in enumerate(idx):
            ids[row] = f"{display_target(target)}_B{int(float(batch))}_P{offset // SCANS_PER_OFFLINE_POINT:03d}"
    return ids


def aggregate_label_points(label: pd.DataFrame, target: str, calibration_only: bool = True, positive_ha: bool = True) -> pd.DataFrame:
    column = internal_target(target)
    y = finite(label[column])
    groups = finite(label["Batch"])
    times = finite(label["time"])
    mask = np.isfinite(y) & np.isfinite(groups) & np.isfinite(times)
    if calibration_only:
        mask &= groups != INDEPENDENT_TEST_BATCH
    else:
        mask &= groups == INDEPENDENT_TEST_BATCH
    if column in {"HA2", "HA4"} and positive_ha:
        mask &= y > 0
    idx = np.where(mask)[0]
    frame = pd.DataFrame(
        {
            "point_id": point_id_series(groups[idx], column),
            "Batch": groups[idx],
            "time": times[idx],
            "reference": y[idx],
        }
    )
    out = (
        frame.groupby("point_id", as_index=False)
        .agg(Batch=("Batch", "first"), time=("time", "mean"), reference=("reference", "mean"), n_scans=("reference", "size"))
        .sort_values(["Batch", "time"])
        .reset_index(drop=True)
    )
    out["target"] = display_target(column)
    return out


def add_time_after_inoc(points: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    inoc = dict(zip(pd.to_numeric(ops["Batch"], errors="coerce"), pd.to_numeric(ops["inoculation_time_h"], errors="coerce")))
    out = points.copy()
    out["time_after_inoculation"] = [row.time - inoc.get(float(row.Batch), np.nan) for row in out.itertuples()]
    return out


def stage_labels(values: pd.Series | np.ndarray) -> pd.Series:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=object)
    out[x < 0] = "pre"
    out[(x >= 0) & (x < 48)] = "0-48 h"
    out[(x >= 48) & (x < 96)] = "48-96 h"
    out[x >= 96] = ">96 h"
    return out


def region_labels_zero_max(values: pd.Series | np.ndarray) -> pd.Series:
    y = pd.to_numeric(pd.Series(values), errors="coerce")
    scaled = y / max(float(y.max()), 1e-12)
    out = pd.Series(np.nan, index=y.index, dtype=object)
    out[scaled < 1 / 3] = "low"
    out[(scaled >= 1 / 3) & (scaled < 2 / 3)] = "middle"
    out[scaled >= 2 / 3] = "high"
    return out


def grouped_rmse(points: pd.DataFrame, group_col: str, labels: list[str], model_label: str) -> pd.DataFrame:
    rows = []
    for label in labels:
        sub = points[points[group_col].astype(str).eq(label)]
        rows.append({"group": label, "model": model_label, "n": len(sub), "rmse": rmse(sub["reference"], sub["prediction"])})
    return pd.DataFrame(rows)


def best_candidate(summary: pd.DataFrame, step_prefix: str, candidate_col: str = "candidate") -> str:
    sub = summary[summary["step"].astype(str).str.startswith(step_prefix)]
    if sub.empty:
        raise ValueError(f"No candidate found for {step_prefix}.")
    return str(sub.iloc[0][candidate_col])


def plsr_cv(target: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "model_outputs" / "plsr_baselines" / "plsr_cv_point_predictions.csv")
    return df[df["target"].astype(str).eq(target)].copy()


def plsr_independent(target: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "model_outputs" / "plsr_baselines" / "plsr_independent_point_predictions.csv")
    return df[df["target"].astype(str).eq(target)].copy()


def make_figure2(label: pd.DataFrame, ops: pd.DataFrame) -> Path:
    via_summary = pd.read_csv(ROOT / "results" / "model_outputs" / "via_bounded_logit_process" / "via_no_weight_selectsvr_ablation_recomputed.csv")
    via_cv_all = pd.read_csv(ROOT / "results" / "model_outputs" / "via_bounded_logit_process" / "via_no_weight_cv_point_predictions.csv")
    final_candidate = best_candidate(via_summary, "V4")
    v4_cv = via_cv_all[via_cv_all["candidate"].astype(str).eq(final_candidate)].copy()
    base_cv = plsr_cv("Via")
    via_points = aggregate_label_points(label, "Via", calibration_only=True)
    via_points = add_time_after_inoc(via_points, ops)

    fig, axes = plt.subplots(2, 3, figsize=(18.8, 10.0))
    ax = axes[0, 0]
    y = pd.to_numeric(label.loc[pd.to_numeric(label["Batch"], errors="coerce").ne(INDEPENDENT_TEST_BATCH), "Via"], errors="coerce").dropna()
    y_scaled = y / max(float(y.max()), 1e-12)
    ax.axvspan(0, 1 / 3, color="#F5D9D7", alpha=0.50, label="low")
    ax.axvspan(1 / 3, 2 / 3, color="#F6E9D6", alpha=0.45, label="middle")
    ax.axvspan(2 / 3, 1.0, color="#DCEBDF", alpha=0.55, label="high")
    ax.hist(y_scaled, bins=np.linspace(0, 1, 26), color=VIA_GREEN, alpha=0.88)
    ax.set_xlabel("Viability reference / maximum")
    ax.set_ylabel("Matched spectra")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "A")
    style_axis(ax)

    ax = axes[0, 1]
    for _, sub in via_points.groupby("Batch", observed=True):
        sub = sub.sort_values("time_after_inoculation")
        ax.plot(sub["time_after_inoculation"], sub["reference"], color=VIA_GREEN, alpha=0.30, linewidth=1.0)
    bins = np.arange(-120, 193, 24)
    centers = (bins[:-1] + bins[1:]) / 2
    med = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = via_points[(via_points["time_after_inoculation"] >= lo) & (via_points["time_after_inoculation"] < hi)]
        med.append(sub["reference"].median() if len(sub) else np.nan)
    ax.plot(centers, med, color=BLACK, linewidth=2.2, marker="o", markersize=3.5)
    ax.axvline(0, color=INOC_RED, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time after inoculation (h)")
    ax.set_ylabel("Viability reference (%)")
    panel_label(ax, "B")
    style_axis(ax)

    ax = axes[0, 2]
    plsr_metrics = pd.read_csv(ROOT / "results" / "model_outputs" / "plsr_baselines" / "plsr_all_metrics.csv")
    base = plsr_metrics[plsr_metrics["target"].eq("Via")].iloc[0]
    rows = [{"step": "V0 PLSR common baseline", "label": "V0\nPLSR", "rmse": float(base["cv_rmse"]), "r2": float(base["cv_r2"]), "color": GREY}]
    for _, row in via_summary.iterrows():
        step = str(row["step"])
        features = str(row.get("features", ""))
        rows.append(
            {
                "step": step,
                "label": via_ablation_label(step),
                "rmse": float(row["cv_rmse"]),
                "r2": float(row["cv_r2"]),
                "color": ablation_bar_color(step, features),
            }
        )
    ab = pd.DataFrame(rows)
    x = np.arange(len(ab))
    ax.bar(x, ab["rmse"], color=ab["color"], width=0.68)
    for xi, val in zip(x, ab["rmse"]):
        ax.text(xi, val + max(ab["rmse"]) * 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(x, ab["r2"], color=BLUE, marker="D", linewidth=1.5, markersize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ab["label"], rotation=0, ha="center")
    ax.set_ylabel("5-fold CV RMSE (%)")
    ax2.set_ylabel("CV $R^2$")
    panel_label(ax, "C")
    style_axis(ax)

    ax = axes[1, 0]
    ax.scatter(base_cv["reference"], base_cv["prediction"], s=32, color=GREY, alpha=0.65, label="PLSR baseline")
    ax.scatter(v4_cv["reference"], v4_cv["prediction"], s=32, color=RED, alpha=0.75, label="Retained Via model")
    lo = min(base_cv["reference"].min(), base_cv["prediction"].min(), v4_cv["prediction"].min(), 0)
    hi = max(base_cv["reference"].max(), base_cv["prediction"].max(), v4_cv["prediction"].max(), 100)
    ax.plot([lo, hi], [lo, hi], color=BLACK, linewidth=0.9)
    ax.text(
        0.98,
        0.08,
        f"PLSR RMSEcv={rmse(base_cv['reference'], base_cv['prediction']):.2f}, $R^2_{{cv}}$={safe_r2(base_cv['reference'], base_cv['prediction']):.3f}\n"
        f"Final RMSEcv={rmse(v4_cv['reference'], v4_cv['prediction']):.2f}, $R^2_{{cv}}$={safe_r2(v4_cv['reference'], v4_cv['prediction']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.25"},
    )
    ax.set_xlabel("Offline viability reference (%)")
    ax.set_ylabel("CV prediction (%)")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "D")
    style_axis(ax)

    compare = []
    for model_label, pts in [("PLSR", base_cv.copy()), ("Retained Via model", v4_cv.copy())]:
        pts["region"] = region_labels_zero_max(pts["reference"]).to_numpy()
        compare.append(grouped_rmse(pts, "region", ["low", "middle", "high"], model_label))
    reg = pd.concat(compare, ignore_index=True)
    ax = axes[1, 1]
    grouped_bars(ax, reg, ["low", "middle", "high"], "group", "rmse", "model", ["PLSR", "Retained Via model"])
    ax.set_xticklabels(["low", "middle", "high"])
    ax.set_ylabel("5-fold CV RMSE (%)")
    panel_label(ax, "E")
    style_axis(ax)

    compare = []
    for model_label, pts in [("PLSR", add_time_after_inoc(base_cv.copy(), ops)), ("Retained Via model", add_time_after_inoc(v4_cv.copy(), ops))]:
        pts["stage"] = stage_labels(pts["time_after_inoculation"]).to_numpy()
        compare.append(grouped_rmse(pts, "stage", ["pre", "0-48 h", "48-96 h", ">96 h"], model_label))
    st = pd.concat(compare, ignore_index=True)
    ax = axes[1, 2]
    grouped_bars(ax, st, ["pre", "0-48 h", "48-96 h", ">96 h"], "group", "rmse", "model", ["PLSR", "Retained Via model"])
    ax.set_ylabel("5-fold CV RMSE (%)")
    panel_label(ax, "F")
    style_axis(ax)

    SRC_OUT.mkdir(parents=True, exist_ok=True)
    via_ablation_source_cols = [
        "endpoint",
        "step",
        "model",
        "response_layer",
        "features",
        "candidate",
        "weighting",
        "cv_n",
        "cv_rmse",
        "cv_mae",
        "cv_bias",
        "cv_r2",
    ]
    via_summary.loc[:, [c for c in via_ablation_source_cols if c in via_summary.columns]].to_csv(
        SRC_OUT / "figure2_via_ablation_source.csv", index=False, encoding="utf-8-sig"
    )
    v4_cv.to_csv(SRC_OUT / "figure2_via_final_cv_predictions_source.csv", index=False, encoding="utf-8-sig")
    pd.concat([reg.assign(panel="E"), st.assign(panel="F")], ignore_index=True).to_csv(
        SRC_OUT / "figure2_via_grouped_rmse_source.csv", index=False, encoding="utf-8-sig"
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.955, bottom=0.10, wspace=0.40, hspace=0.58)
    out = OUT / "Figure2_Via_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def grouped_bars(ax: plt.Axes, data: pd.DataFrame, order: list[str], x_col: str, y_col: str, hue_col: str, hue_order: list[str]) -> None:
    width = 0.36
    x = np.arange(len(order))
    colors = [GREY, RED, BLUE]
    for j, hue in enumerate(hue_order):
        vals = []
        for group in order:
            sub = data[(data[x_col].astype(str).eq(group)) & (data[hue_col].astype(str).eq(hue))]
            vals.append(float(sub[y_col].iloc[0]) if len(sub) else np.nan)
        offset = (j - (len(hue_order) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=colors[j], alpha=0.95, label=hue)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.legend(frameon=False, loc="upper left")


def logit_range_dim(y: np.ndarray) -> np.ndarray:
    p = np.clip((np.asarray(y, dtype=float) - dim_mod.DIM_LOWER) / (dim_mod.DIM_UPPER - dim_mod.DIM_LOWER), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def make_figure3(label: pd.DataFrame, ops: pd.DataFrame) -> Path:
    dim_summary = pd.read_csv(ROOT / "results" / "model_outputs" / "dim_logit_range_process" / "dim_logit_range_ablation.csv")
    final_candidate = best_candidate(dim_summary, "D4")
    dim_cv_all = pd.read_csv(ROOT / "results" / "model_outputs" / "dim_logit_range_process" / "dim_logit_range_ablation_cv_point_predictions.csv")
    d4_cv = dim_cv_all[dim_cv_all["candidate"].astype(str).eq(final_candidate)].copy()
    base_cv = plsr_cv("Dim")
    dim_points = aggregate_label_points(label, "Dim", calibration_only=True)
    dim_points = add_time_after_inoc(dim_points, ops)
    q33 = float(dim_points["reference"].quantile(0.33))
    q75 = float(dim_points["reference"].quantile(0.75))

    fig, axes = plt.subplots(2, 3, figsize=(18.8, 10.0))
    ax = axes[0, 0]
    ax.hist(dim_points["reference"], bins=22, color=DIM_ORANGE, alpha=0.92)
    ax.axvline(dim_mod.DIM_LOWER, color="#666666", linestyle="--", linewidth=0.9)
    ax.text(dim_mod.DIM_LOWER + 0.04, ax.get_ylim()[1] * 0.86, "15 um base", fontsize=11)
    ax.set_xlabel("Cell diameter reference (um)")
    ax.set_ylabel("Offline sampling points")
    inset = ax.inset_axes([0.52, 0.52, 0.43, 0.40])
    inset.hist(logit_range_dim(dim_points["reference"].to_numpy(float)), bins=14, color=VIA_GREEN, alpha=0.85)
    inset.set_xlabel("logit-range", fontsize=9)
    inset.set_ylabel("points", fontsize=9)
    inset.tick_params(labelsize=8)
    panel_label(ax, "A")
    style_axis(ax)

    ax = axes[0, 1]
    for _, sub in dim_points.groupby("Batch", observed=True):
        sub = sub.sort_values("time_after_inoculation")
        ax.plot(sub["time_after_inoculation"], sub["reference"], color=VIA_GREEN, alpha=0.28, linewidth=1.0, marker="o", markersize=2.5)
    bins = np.arange(-120, 193, 24)
    centers = (bins[:-1] + bins[1:]) / 2
    med = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = dim_points[(dim_points["time_after_inoculation"] >= lo) & (dim_points["time_after_inoculation"] < hi)]
        med.append(sub["reference"].median() if len(sub) else np.nan)
    ax.plot(centers, med, color=BLACK, linewidth=2.2, marker="o", markersize=3.5)
    ax.axvline(0, color=INOC_RED, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time after inoculation (h)")
    ax.set_ylabel("Cell diameter reference (um)")
    panel_label(ax, "B")
    style_axis(ax)

    ax = axes[0, 2]
    plsr_metrics = pd.read_csv(ROOT / "results" / "model_outputs" / "plsr_baselines" / "plsr_all_metrics.csv")
    base = plsr_metrics[plsr_metrics["target"].eq("Dim")].iloc[0]
    rows = [{"step": "D0 PLSR common baseline", "label": "D0\nPLSR", "rmse": float(base["cv_rmse"]), "r2": float(base["cv_r2"]), "color": GREY}]
    for _, row in dim_summary.iterrows():
        step = str(row["step"])
        features = str(row.get("features", ""))
        rows.append(
            {
                "step": step,
                "label": dim_ablation_label(step),
                "rmse": float(row["cv_rmse"]),
                "r2": float(row["cv_r2"]),
                "color": ablation_bar_color(step, features),
            }
        )
    ab = pd.DataFrame(rows)
    x = np.arange(len(ab))
    ax.bar(x, ab["rmse"], color=ab["color"], width=0.68)
    for xi, val in zip(x, ab["rmse"]):
        ax.text(xi, val + max(ab["rmse"]) * 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(x, ab["r2"], color=BLUE, marker="D", linewidth=1.5, markersize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ab["label"], rotation=0, ha="center")
    ax.set_ylabel("5-fold CV RMSE (um)")
    ax2.set_ylabel("CV $R^2$")
    panel_label(ax, "C")
    style_axis(ax)

    ax = axes[1, 0]
    ax.scatter(base_cv["reference"], base_cv["prediction"], s=32, color=GREY, alpha=0.65, label="PLSR baseline")
    ax.scatter(d4_cv["reference"], d4_cv["prediction"], s=32, color=RED, alpha=0.75, label="Retained Dim model")
    lo = min(base_cv["reference"].min(), base_cv["prediction"].min(), d4_cv["prediction"].min())
    hi = max(base_cv["reference"].max(), base_cv["prediction"].max(), d4_cv["prediction"].max())
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=BLACK, linewidth=0.9)
    ax.text(
        0.98,
        0.08,
        f"PLSR RMSEcv={rmse(base_cv['reference'], base_cv['prediction']):.3f}, $R^2_{{cv}}$={safe_r2(base_cv['reference'], base_cv['prediction']):.3f}\n"
        f"Final RMSEcv={rmse(d4_cv['reference'], d4_cv['prediction']):.3f}, $R^2_{{cv}}$={safe_r2(d4_cv['reference'], d4_cv['prediction']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.25"},
    )
    ax.set_xlabel("Offline Dim reference (um)")
    ax.set_ylabel("CV prediction (um)")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "D")
    style_axis(ax)

    compare = []
    for model_label, pts in [("PLSR", add_time_after_inoc(base_cv.copy(), ops)), ("Retained Dim model", add_time_after_inoc(d4_cv.copy(), ops))]:
        pts["stage"] = stage_labels(pts["time_after_inoculation"]).to_numpy()
        compare.append(grouped_rmse(pts, "stage", ["pre", "0-48 h", "48-96 h", ">96 h"], model_label))
    st = pd.concat(compare, ignore_index=True)
    ax = axes[1, 1]
    grouped_bars(ax, st, ["pre", "0-48 h", "48-96 h", ">96 h"], "group", "rmse", "model", ["PLSR", "Retained Dim model"])
    ax.set_ylabel("5-fold CV RMSE (um)")
    panel_label(ax, "E")
    style_axis(ax)

    compare = []
    for model_label, pts in [("PLSR", base_cv.copy()), ("Retained Dim model", d4_cv.copy())]:
        ref = pd.to_numeric(pts["reference"], errors="coerce")
        pts["band"] = np.select([ref < q33, (ref >= q33) & (ref < q75), ref >= q75], ["low", "middle", "high-tail"], default="missing")
        compare.append(grouped_rmse(pts, "band", ["low", "middle", "high-tail"], model_label))
    bands = pd.concat(compare, ignore_index=True)
    ax = axes[1, 2]
    grouped_bars(ax, bands, ["low", "middle", "high-tail"], "group", "rmse", "model", ["PLSR", "Retained Dim model"])
    ax.set_xticklabels([f"low\n<{q33:.2f}", f"middle\n{q33:.2f}-{q75:.2f}", f"high-tail\n>={q75:.2f}"])
    ax.set_xlabel("Dim reference band (um)")
    ax.set_ylabel("5-fold CV RMSE (um)")
    panel_label(ax, "F")
    style_axis(ax)

    dim_ablation_source_cols = [
        "target",
        "step",
        "model",
        "response_layer",
        "features",
        "candidate",
        "process_time",
        "cv_n",
        "cv_rmse",
        "cv_mae",
        "cv_bias",
        "cv_r2",
    ]
    dim_summary.loc[:, [c for c in dim_ablation_source_cols if c in dim_summary.columns]].to_csv(
        SRC_OUT / "figure3_dim_ablation_source.csv", index=False, encoding="utf-8-sig"
    )
    d4_cv.to_csv(SRC_OUT / "figure3_dim_final_cv_predictions_source.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"q33": q33, "q75": q75}]).to_csv(SRC_OUT / "figure3_dim_reference_band_cutoffs.csv", index=False, encoding="utf-8-sig")
    pd.concat([st.assign(panel="E"), bands.assign(panel="F")], ignore_index=True).to_csv(
        SRC_OUT / "figure3_dim_grouped_rmse_source.csv", index=False, encoding="utf-8-sig"
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.955, bottom=0.10, wspace=0.40, hspace=0.58)
    out = OUT / "Figure3_Dim_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ha_cv_scatter_panel(ax: plt.Axes, ha_metrics: pd.DataFrame, ha_cv: pd.DataFrame, target: str, color: str, label_text: str) -> None:
    b0 = ha_cv[(ha_cv["target"].astype(str).eq(target)) & (ha_cv["step"].astype(str).eq("B0"))]
    h2 = ha_cv[(ha_cv["target"].astype(str).eq(target)) & (ha_cv["step"].astype(str).eq("H2"))]
    ax.scatter(b0["reference"], b0["prediction"], s=34, color=GREY, alpha=0.62, label="PLSR")
    ax.scatter(h2["reference"], h2["prediction"], s=34, color=color, alpha=0.82, label="Final H2")
    lo = min(b0["reference"].min(), h2["reference"].min(), b0["prediction"].min(), h2["prediction"].min()) - 0.6
    hi = max(b0["reference"].max(), h2["reference"].max(), b0["prediction"].max(), h2["prediction"].max()) + 0.6
    grid = np.linspace(lo, hi, 100)
    ax.fill_between(grid, grid - 1, grid + 1, color="#F0CFC8", alpha=0.35, label="+/-1 step")
    ax.plot([lo, hi], [lo, hi], color=BLACK, linewidth=0.9)
    base_row = ha_metrics[(ha_metrics["target"].astype(str).eq(target)) & (ha_metrics["step"].astype(str).eq("B0"))].iloc[0]
    h2_row = ha_metrics[(ha_metrics["target"].astype(str).eq(target)) & (ha_metrics["step"].astype(str).eq("H2"))].iloc[0]
    ax.text(
        0.97,
        0.08,
        f"PLSR loss={float(base_row['cv_tolerance_loss']):.3f}, +/-1={float(base_row['cv_rounded_within_1_step_pct']):.1f}%\n"
        f"H2 loss={float(h2_row['cv_tolerance_loss']):.3f}, +/-1={float(h2_row['cv_rounded_within_1_step_pct']):.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.25"},
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Offline reference ({target}, log2 step)")
    ax.set_ylabel(f"CV prediction ({target}, log2 step)")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, label_text)
    style_axis(ax)


def make_figure4(label: pd.DataFrame) -> Path:
    ha_metrics = pd.read_csv(ROOT / "results" / "model_outputs" / "ha_ordinal_tolerance" / "ha_ordinal_tolerance_ablation.csv")
    ha_cv = pd.read_csv(ROOT / "results" / "model_outputs" / "ha_ordinal_tolerance" / "ha_ordinal_tolerance_cv_point_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 5.7))
    ax = axes[0]
    cal = label[pd.to_numeric(label["Batch"], errors="coerce").ne(INDEPENDENT_TEST_BATCH)]
    distribution_rows = []
    for target, color, offset in [("HA2", EHA_PURPLE, -0.18), ("HA4", THA_ROSE, 0.18)]:
        y = pd.to_numeric(cal[target], errors="coerce")
        y = y[np.isfinite(y) & (y > 0)]
        counts = y.round().astype(int).value_counts().sort_index()
        ax.bar(counts.index.to_numpy(float) + offset, counts.values, width=0.34, color=color, alpha=0.86, label=display_target(target))
        for step, count in counts.items():
            distribution_rows.append({"target": display_target(target), "log2_dilution_step": int(step), "matched_spectra": int(count)})
    ax.set_xlabel("log2 dilution step")
    ax.set_ylabel("Matched spectra")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "A")
    style_axis(ax)

    plot_ha_cv_scatter_panel(axes[1], ha_metrics, ha_cv, "eHA", EHA_PURPLE, "B")
    plot_ha_cv_scatter_panel(axes[2], ha_metrics, ha_cv, "tHA", THA_ROSE, "C")

    SRC_OUT.mkdir(parents=True, exist_ok=True)
    stale = SRC_OUT / "figure4_ha_ablation_source.csv"
    if stale.exists():
        stale.unlink()
    pd.DataFrame(distribution_rows).to_csv(SRC_OUT / "figure4_ha_label_distribution_source.csv", index=False, encoding="utf-8-sig")
    ha_cv[ha_cv["step"].astype(str).isin(["B0", "H2"])].to_csv(
        SRC_OUT / "figure4_ha_cv_predictions_source.csv", index=False, encoding="utf-8-sig"
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.16, wspace=0.34)
    out = OUT / "Figure4_HA_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def online_frame(online: pd.DataFrame, target: str, model_label: str, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": target,
            "model_label": model_label,
            "Batch": pd.to_numeric(online["Batch"], errors="coerce").to_numpy(float),
            "time": pd.to_numeric(online["time"], errors="coerce").to_numpy(float),
            "prediction": np.asarray(pred, dtype=float),
        }
    )


def plsr_online_predictions(label: pd.DataFrame, online: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = plsr_mod.infer_spectral_columns(label, cfg)
    raw_label = label[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_online = online[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X_label = plsr_mod.chemical_features(raw_label, cfg)
    X_online = plsr_mod.chemical_features(raw_online, cfg)
    X = np.vstack([X_label, X_online])
    rows = []
    for target in ["VCD", "Via", "Dim", "HA2", "HA4", "Glc", "Gln", "NH4"]:
        y_label = pd.to_numeric(label[target], errors="coerce").to_numpy(float)
        groups = pd.to_numeric(label["Batch"], errors="coerce").to_numpy(float)
        mask = np.isfinite(y_label) & (groups != INDEPENDENT_TEST_BATCH)
        if target in {"HA2", "HA4"}:
            mask &= y_label > 0
        y_full = np.concatenate([y_label, np.full(len(online), np.nan)])
        pred = plsr_mod.fit_predict_plsr(
            X,
            y_full,
            np.where(mask)[0],
            np.arange(len(label), len(label) + len(online)),
            PLSR_COMPONENTS[target],
            target,
        )
        rows.append(online_frame(online, display_target(target), "PLSR baseline", pred))
        if target not in {"Via", "Dim", "HA2", "HA4"}:
            rows.append(online_frame(online, display_target(target), "Optimized/final", pred))
    return pd.concat(rows, ignore_index=True)


def via_online_predictions(label: pd.DataFrame, online: pd.DataFrame, ops: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = via_mod.infer_spectral_columns(label, cfg)
    combined = pd.concat([label, online], ignore_index=True, sort=False)
    raw = combined[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = via_mod.chemical_feature_matrix(raw, cfg)
    P = via_mod.process_time_features(combined, ops)
    y = pd.to_numeric(combined["Via"], errors="coerce").to_numpy(float)
    groups = pd.to_numeric(combined["Batch"], errors="coerce").to_numpy(float)
    train_idx = np.where(np.isfinite(y) & (groups != INDEPENDENT_TEST_BATCH))[0]
    test_idx = np.arange(len(label), len(combined))
    candidates = via_mod.via_candidates()
    pred_v4 = via_mod.fit_predict_via(candidates[-1], X, P, y, train_idx, test_idx)
    return online_frame(online, "Via", "Optimized/final", pred_v4)


def dim_online_predictions(label: pd.DataFrame, online: pd.DataFrame, ops: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = dim_mod.infer_spectral_columns(label, cfg)
    combined = pd.concat([label, online], ignore_index=True, sort=False)
    raw = combined[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = dim_mod.chemical_feature_matrix(raw, cfg)
    y = pd.to_numeric(combined["Dim"], errors="coerce").to_numpy(float)
    groups = pd.to_numeric(combined["Batch"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(combined["time"], errors="coerce").to_numpy(float)
    inoc = dict(zip(ops["Batch"].astype(int), pd.to_numeric(ops["inoculation_time_h"], errors="coerce")))
    train_idx = np.where(np.isfinite(y) & (groups != INDEPENDENT_TEST_BATCH))[0]
    test_idx = np.arange(len(label), len(combined))
    candidates = dim_mod.dim_candidates()
    d4 = candidates[-1]
    pred_d4 = dim_mod.fit_predict_dim(
        X,
        y,
        groups,
        times,
        inoc,
        train_idx,
        test_idx,
        k_best=d4.k_best,
        c_value=d4.c_value,
        epsilon=d4.epsilon,
        response_layer=d4.response_layer,
        use_process=d4.use_process,
    )
    return online_frame(online, "Dim", "Optimized/final", pred_d4)


def parse_h2_candidate(row: pd.Series) -> ha_mod.HACandidate:
    return ha_mod.HACandidate(
        target=str(row["target"]),
        step=str(row["step"]),
        module=str(row["module"]),
        column=ha_mod.TARGET_COLUMNS[str(row["target"])],
        first_n=int(row.get("first_n", ha_mod.FIRST_N)),
        sg_window=15,
        n_components=int(row["n_components"]),
        process_weight=float(row["process_weight"]),
        logistic_c=float(row["logistic_c"]),
        pooling=str(row["pooling"]),
    )


def ha_online_predictions(label: pd.DataFrame, online: pd.DataFrame, ops: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    metrics = pd.read_csv(ROOT / "results" / "model_outputs" / "ha_ordinal_tolerance" / "ha_ordinal_tolerance_ablation.csv")
    cols = ha_mod.infer_spectral_columns(label, cfg)
    combined = pd.concat([label, online], ignore_index=True, sort=False)
    raw = combined[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    batches = pd.to_numeric(combined["Batch"], errors="coerce").to_numpy(float)
    times = pd.to_numeric(combined["time"], errors="coerce").to_numpy(float)
    P_all = ha_mod.process_time_features(combined, ops)
    online_idx = np.arange(len(label), len(combined))
    inoc = dict(zip(pd.to_numeric(ops["Batch"], errors="coerce"), pd.to_numeric(ops["inoculation_time_h"], errors="coerce")))
    post_online = np.array([times[i] >= inoc.get(float(batches[i]), np.nan) for i in online_idx], dtype=bool)
    rows = []

    raw_d1 = ha_mod.baseline_d1_feature_matrix(raw, cfg, sg_window=15)
    for target in ["eHA", "tHA"]:
        column = ha_mod.TARGET_COLUMNS[target]
        y = pd.to_numeric(combined[column], errors="coerce").to_numpy(float)
        train = np.where(np.isfinite(y) & (y > 0) & (batches != INDEPENDENT_TEST_BATCH))[0]
        b0 = metrics[(metrics["target"].astype(str).eq(target)) & (metrics["step"].astype(str).eq("B0"))].iloc[0]
        pred_plsr = ha_mod.fit_predict_plsr(raw_d1[train], y[train], raw_d1[online_idx], int(b0["n_components"]))
        rows.append(online_frame(online, target, "PLSR baseline", pred_plsr))

        h2_row = metrics[(metrics["target"].astype(str).eq(target)) & (metrics["step"].astype(str).eq("H2"))].iloc[0]
        cand = parse_h2_candidate(h2_row)
        delta, usable = ha_mod.locked_delta(raw, batches, times, cand.first_n)
        X_all = ha_mod.chemical_feature_matrix(delta, cfg, cand.sg_window, derivative_order=0)
        train = np.where(np.isfinite(y) & (y > 0) & (batches != INDEPENDENT_TEST_BATCH) & usable)[0]
        test = online_idx[usable[online_idx]]
        classes, probs = ha_mod.predict_probabilities(cand, X_all[train], y[train], P_all[train], X_all[test], P_all[test])
        pred = np.full(len(online), np.nan)
        pred[test - len(label)] = ha_mod.tolerance_decode(classes, probs)
        rows.append(online_frame(online, target, "Optimized/final", pred))

    out = pd.concat(rows, ignore_index=True)
    keep = []
    for row in out.itertuples():
        if row.target in {"eHA", "tHA"}:
            pos = int(row.Index % len(online))
            keep.append(bool(post_online[pos]))
        else:
            keep.append(True)
    return out.loc[keep].reset_index(drop=True)


def causal_smooth(values: pd.Series, window: int = 7) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rolling(window=window, min_periods=1).mean()


def scale_for_plot(target: str, values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if target == "VCD":
        return arr / 1e6
    return arr


def make_figure5(label: pd.DataFrame, online_all: pd.DataFrame, ops: pd.DataFrame, cfg: dict) -> Path:
    online = online_all[pd.to_numeric(online_all["Batch"], errors="coerce").eq(INDEPENDENT_TEST_BATCH)].sort_values("time").reset_index(drop=True)
    pred = pd.concat(
        [
            plsr_online_predictions(label, online, cfg),
            via_online_predictions(label, online, ops, cfg),
            dim_online_predictions(label, online, ops, cfg),
            ha_online_predictions(label, online, ops, cfg),
        ],
        ignore_index=True,
    )
    pred = pred.drop_duplicates(["target", "model_label", "time"], keep="last").sort_values(["target", "model_label", "time"])
    refs = pd.concat([aggregate_label_points(label, t, calibration_only=False) for t in TARGETS], ignore_index=True)
    inoc_time = float(ops[pd.to_numeric(ops["Batch"], errors="coerce").eq(INDEPENDENT_TEST_BATCH)]["inoculation_time_h"].iloc[0])
    pred = pred[(~pred["target"].isin(["eHA", "tHA"])) | (pd.to_numeric(pred["time"], errors="coerce") >= inoc_time)].copy()
    refs = refs[(~refs["target"].isin(["eHA", "tHA"])) | (pd.to_numeric(refs["time"], errors="coerce") >= inoc_time)].copy()

    fig, axes = plt.subplots(4, 2, figsize=(15.0, 19.2))
    axes = axes.ravel()
    ylabels = {
        "VCD": "VCD (10^6 cells/mL)",
        "Via": "Viability (%)",
        "Dim": "Cell diameter (um)",
        "eHA": "eHA titer (log2 step)",
        "tHA": "tHA titer (log2 step)",
        "Glc": "Glucose",
        "Gln": "Glutamine",
        "NH4": "Ammonia",
    }
    for ax, target in zip(axes, TARGETS):
        sub_ref = refs[refs["target"].astype(str).eq(target)].copy()
        if target in {"eHA", "tHA"}:
            sub_ref = sub_ref[sub_ref["time"] >= inoc_time]
        ax.plot(sub_ref["time"], scale_for_plot(target, sub_ref["reference"]), color=BLACK, linestyle=":", marker="o", markersize=3.5, linewidth=1.0, label="Offline")
        for model_label, color, linestyle, linewidth in [
            ("PLSR baseline", GREY, "--", 1.05),
            ("Optimized/final", RED, "-", 1.35),
        ]:
            sub = pred[(pred["target"].astype(str).eq(target)) & (pred["model_label"].astype(str).eq(model_label))].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("time")
            y = causal_smooth(pd.Series(scale_for_plot(target, sub["prediction"])))
            ax.plot(sub["time"], y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95, label=model_label)
        ax.axvline(inoc_time, color=INOC_RED, linestyle=":", linewidth=1.0)
        if target in {"eHA", "tHA"}:
            ax.set_xlim(inoc_time, online["time"].max())
        ax.set_title(f"{target}")
        ax.set_xlabel("Culture time (h)")
        ax.set_ylabel(ylabels[target])
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    extra_handles, extra_labels = [], []
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        extra_handles.extend(h)
        extra_labels.extend(l)
    seen = {}
    for h, l in zip(handles + extra_handles, labels + extra_labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False)
    pred.to_csv(SRC_OUT / "figure5_independent_test_online_predictions_source.csv", index=False, encoding="utf-8-sig")
    refs.to_csv(SRC_OUT / "figure5_independent_test_offline_reference_source.csv", index=False, encoding="utf-8-sig")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.965, bottom=0.075, wspace=0.32, hspace=0.56)
    out = OUT / "Figure5_independent_test_final.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC_OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    label = pd.read_csv(LABEL_CSV)
    online = pd.read_csv(ONLINE_CSV)
    ops = pd.read_csv(OPS_CSV)
    cfg = load_config()

    make_figure1_source_data.main()
    fig1 = make_figure1.main()
    paths = [
        OUT / "Figure1_process_variable_PLSR_diagnostics.png",
        make_figure2(label, ops),
        make_figure3(label, ops),
        make_figure4(label),
        make_figure5(label, online, ops, cfg),
    ]
    pd.DataFrame({"figure": [p.name for p in paths], "path": [p.relative_to(ROOT).as_posix() for p in paths]}).to_csv(
        OUT / "figure_index.csv", index=False, encoding="utf-8-sig"
    )
    for path in paths:
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
