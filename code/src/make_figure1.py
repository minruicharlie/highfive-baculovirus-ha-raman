from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = ROOT / "results" / "figures" / "source_data"
OUT = ROOT / "results" / "figures"

TARGET_ORDER = ["VCD", "Via", "Dim", "eHA", "tHA", "Glc", "Gln", "NH4"]
FEATURE_ORDER = [
    "bounded/skewed",
    "infection-event linked",
    "log2/exponential readout",
    "low-resolution/discrete",
    "narrow dynamic span",
]
FEATURE_LABELS = [
    "bounded/\nskewed",
    "infection-event\nlinked",
    "log2/exponential\nreadout",
    "low-resolution/\ndiscrete",
    "narrow dynamic\nspan",
]
REFERENCE_REGION_ORDER = ["low", "middle", "high"]
STAGE_ORDER = ["pre", "0-48 h", "48-96 h", ">96 h"]
SPECIAL = {"Via", "Dim", "eHA", "tHA"}
COLORS = {
    "VCD": "#2F6F7E",
    "Via": "#77A88D",
    "Dim": "#D99A4E",
    "eHA": "#8B73B3",
    "tHA": "#B86F7A",
    "Glc": "#6B8E23",
    "Gln": "#7D6A9F",
    "NH4": "#B0762E",
}


def target_color(target: str) -> str:
    return COLORS.get(str(target), "#8C8C8C")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14)


def text_color(value: float, vmax: float) -> str:
    if not np.isfinite(value):
        return "#2B2B2B"
    return "white" if abs(value) > 0.55 * vmax else "#2B2B2B"


def draw_data_characteristics(ax: plt.Axes, fig: plt.Figure) -> None:
    structure = pd.read_csv(SRC / "process_variable_data_structure_scores.csv")
    mat = (
        structure.pivot(index="feature", columns="target_display", values="score")
        .reindex(index=FEATURE_ORDER, columns=TARGET_ORDER)
        .to_numpy(float)
    )
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(TARGET_ORDER)))
    ax.set_xticklabels(TARGET_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(FEATURE_ORDER)))
    ax.set_yticklabels(FEATURE_LABELS)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.025)
    cbar.ax.set_title("Score", fontsize=8, pad=6)
    add_panel_label(ax, "A")


def draw_reference_distribution(ax: plt.Axes) -> None:
    points = pd.read_csv(SRC / "development_offline_points_source.csv")
    rng = np.random.default_rng(7)
    for i, target in enumerate(TARGET_ORDER):
        sub = points[points["target_display"].astype(str).eq(target)].copy()
        y = pd.to_numeric(sub["reference_zero_max"], errors="coerce").dropna().to_numpy(float)
        color = target_color(target)
        ax.boxplot(
            [y],
            positions=[i],
            widths=[0.54],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.22, "edgecolor": color},
            medianprops={"color": "#222222", "linewidth": 1.0},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        ax.scatter(np.full(len(y), i) + rng.uniform(-0.17, 0.17, len(y)), y, s=11, color=color, alpha=0.58, linewidth=0)
    ax.set_xticks(np.arange(len(TARGET_ORDER)))
    ax.set_xticklabels(TARGET_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Reference value / process-variable maximum")
    ax.grid(axis="y", alpha=0.18)
    add_panel_label(ax, "B")


def draw_event_response(ax: plt.Axes) -> None:
    effects = pd.read_csv(SRC / "event_effect_source.csv")
    med = effects.groupby("target_display")["event_effect"].median().reindex(TARGET_ORDER)
    q1 = effects.groupby("target_display")["event_effect"].quantile(0.25).reindex(TARGET_ORDER)
    q3 = effects.groupby("target_display")["event_effect"].quantile(0.75).reindex(TARGET_ORDER)
    x = np.arange(len(TARGET_ORDER))
    colors = [target_color(t) if t in SPECIAL else "#8C8C8C" for t in TARGET_ORDER]
    ax.bar(x, med.to_numpy(float), color=colors, width=0.68)
    ax.errorbar(
        x,
        med,
        yerr=[np.nan_to_num(med - q1), np.nan_to_num(q3 - med)],
        fmt="none",
        ecolor="#444444",
        elinewidth=0.9,
        capsize=2.5,
    )
    rng = np.random.default_rng(11)
    for i, target in enumerate(TARGET_ORDER):
        vals = pd.to_numeric(effects[effects["target_display"].astype(str).eq(target)]["event_effect"], errors="coerce").dropna().to_numpy(float)
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.13, 0.13, len(vals)), vals, s=13, facecolor="white", edgecolor="#666666", linewidth=0.55, zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Event effect (% observed range)")
    ax.grid(axis="y", alpha=0.18)
    add_panel_label(ax, "C")


def draw_plsr_diagnostic(ax: plt.Axes) -> None:
    risk = pd.read_csv(SRC / "plsr_percentage_error_source.csv")
    label_offsets = {
        "Gln": (-0.018, 1.1),
        "Glc": (0.009, -1.4),
        "Via": (0.009, -0.4),
        "VCD": (0.009, 1.2),
        "Dim": (0.009, 0.8),
        "NH4": (0.009, 0.9),
        "eHA": (0.009, 0.9),
        "tHA": (0.009, 0.9),
    }
    for _, row in risk.iterrows():
        target = str(row["target_display"])
        color = target_color(target) if target in SPECIAL else "#8C8C8C"
        ax.scatter(
            float(row["train_cv_gap"]),
            float(row["cv_mape_percent"]),
            s=88 if target in SPECIAL else 60,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        dx, dy = label_offsets.get(target, (0.008, 0.8))
        ax.text(float(row["train_cv_gap"]) + dx, float(row["cv_mape_percent"]) + dy, target, fontsize=9)
    ax.set_xlabel("PLSR training $R^2$ - CV $R^2$")
    ax.set_ylabel("CV mean absolute percentage error (%)")
    ax.grid(alpha=0.18)
    add_panel_label(ax, "D")


def draw_reference_residual(ax: plt.Axes, fig: plt.Figure) -> None:
    metrics = pd.read_csv(SRC / "plsr_compact_binned_diagnostic_source.csv")
    heat = metrics.pivot(index="process_variable", columns="reference_region", values="mean_signed_residual_range_pct").reindex(
        index=TARGET_ORDER, columns=REFERENCE_REGION_ORDER
    )
    values = heat.to_numpy(float)
    vmax = max(float(np.nanmax(np.abs(values))), 1.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_xticks(np.arange(len(REFERENCE_REGION_ORDER)))
    ax.set_xticklabels(REFERENCE_REGION_ORDER)
    ax.set_yticks(np.arange(len(TARGET_ORDER)))
    ax.set_yticklabels(TARGET_ORDER)
    ax.set_xlabel("Reference-value region")
    for i, target in enumerate(TARGET_ORDER):
        for j, region in enumerate(REFERENCE_REGION_ORDER):
            value = heat.loc[target, region]
            if pd.notna(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8.5, color=text_color(float(value), vmax))
    cbar = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.025)
    cbar.ax.set_title("Residual\n(% range)", fontsize=8, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "E")


def draw_stage_residual(ax: plt.Axes, fig: plt.Figure) -> None:
    metrics = pd.read_csv(SRC / "plsr_inoculation_stage_residual_source.csv")
    heat = metrics.pivot(index="process_variable", columns="inoculation_aligned_stage", values="mean_signed_residual_range_pct").reindex(
        index=TARGET_ORDER, columns=STAGE_ORDER
    )
    nmat = metrics.pivot(index="process_variable", columns="inoculation_aligned_stage", values="n").reindex(
        index=TARGET_ORDER, columns=STAGE_ORDER
    )
    values = heat.to_numpy(float)
    vmax = max(float(np.nanmax(np.abs(values))), 1.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_xticks(np.arange(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_yticks(np.arange(len(TARGET_ORDER)))
    ax.set_yticklabels(TARGET_ORDER)
    ax.set_xlabel("Inoculation-aligned stage")
    for i, target in enumerate(TARGET_ORDER):
        for j, stage in enumerate(STAGE_ORDER):
            value = heat.loc[target, stage]
            n = nmat.loc[target, stage]
            if pd.notna(value):
                ax.text(j, i, f"{value:+.1f}\nn={int(n)}", ha="center", va="center", fontsize=7.8, color=text_color(float(value), vmax))
    cbar = fig.colorbar(im, ax=ax, fraction=0.041, pad=0.025)
    cbar.ax.set_title("Residual\n(% range)", fontsize=8, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_panel_label(ax, "F")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.3))
    draw_data_characteristics(axes[0, 0], fig)
    draw_reference_distribution(axes[0, 1])
    draw_event_response(axes[0, 2])
    draw_plsr_diagnostic(axes[1, 0])
    draw_reference_residual(axes[1, 1], fig)
    draw_stage_residual(axes[1, 2], fig)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.10, wspace=0.38, hspace=0.52)
    out = OUT / "Figure1_process_variable_PLSR_diagnostics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(out.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
