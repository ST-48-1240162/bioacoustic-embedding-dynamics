"""Change-point detection on activity and embedding-trajectory timelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt


def _pelt_times(binned: pd.DataFrame, column: str, pen: float) -> list[float]:
    y = binned[column].to_numpy(dtype=np.float64).reshape(-1, 1)
    if len(y) < 4:
        return []
    algo = rpt.Pelt(model="rbf").fit(y)
    bkpts = algo.predict(pen=pen)
    times = binned["t_center_s"].to_numpy()
    out: list[float] = []
    for b in bkpts[:-1]:
        idx = min(int(b) - 1, len(times) - 1)
        out.append(float(times[idx]))
    return out


def detection_rate_changepoints(binned: pd.DataFrame, column: str = "detection_rate", pen: float = 3.0) -> list[float]:
    return _pelt_times(binned, column, pen)


def trajectory_changepoints(binned: pd.DataFrame, column: str = "pca_x", pen: float = 2.5) -> list[float]:
    """Regime shifts in reduced embedding centroid (PC1)."""
    return _pelt_times(binned, column, pen)
