from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


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
        raise ValueError("No numeric Raman spectral columns were found in the configured window.")
    pairs.sort(key=lambda item: item[0])
    return [col for _, col in pairs[::stride]]


def chemical_feature_matrix(raw: np.ndarray, cfg: dict, derivative_order: int = 1, sg_window: int | None = None) -> np.ndarray:
    spectral = cfg.get("spectral", {})
    raw = np.asarray(raw, dtype=float)
    x_axis = np.linspace(-1.0, 1.0, raw.shape[1])
    vander = np.vander(x_axis, N=int(spectral.get("baseline_degree", 3)) + 1, increasing=False)
    baseline_pinv = np.linalg.pinv(vander)
    corrected = raw - (raw @ baseline_pinv.T) @ vander.T

    window = int(sg_window or spectral.get("savgol_window", 15))
    window = min(window, corrected.shape[1] - (1 - corrected.shape[1] % 2))
    if window % 2 == 0:
        window -= 1
    window = max(5, window)
    poly = min(int(spectral.get("savgol_polyorder", 3)), window - 2)
    try:
        chem = savgol_filter(
            corrected,
            window_length=window,
            polyorder=poly,
            deriv=int(derivative_order),
            axis=1,
            mode="interp",
        )
    except Exception:
        chem = np.gradient(corrected, axis=1) if derivative_order else corrected

    mean = np.mean(chem, axis=1, keepdims=True)
    std = np.std(chem, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (chem - mean) / std
