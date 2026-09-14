"""Feature scaling helpers for high-dimensional embedding matrices."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler
