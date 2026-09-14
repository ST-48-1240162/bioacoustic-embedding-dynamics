"""Gaussian HMM on binned timelines (embedding centroids or activity stats)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

EMBEDDING_COLS = ("pca_x", "pca_y")
ACTIVITY_COLS = ("detection_rate", "species_richness", "mean_confidence")


def _feature_matrix(binned: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    present = [c for c in cols if c in binned.columns]
    if not present:
        raise ValueError(f"none of {cols} in binned columns")
    X = binned[present].to_numpy(dtype=np.float64)
    keep = X.std(axis=0) > 1e-12
    if not keep.any():
        return np.zeros((len(X), 1), dtype=np.float64)
    X = X[:, keep]
    return StandardScaler().fit_transform(X)


def fit_regime_hmm(
    binned: pd.DataFrame,
    n_states: int = 3,
    *,
    feature_set: str = "embedding",
    random_state: int = 42,
) -> tuple[GaussianHMM | None, np.ndarray]:
    """Fit a diagonal Gaussian HMM.

    feature_set:
      embedding: binned PC1/PC2 centroids (the trajectory).
      activity: detection rate, species richness, mean confidence.
    """
    if feature_set not in {"embedding", "activity"}:
        raise ValueError(f"unknown feature_set: {feature_set}")
    cols = EMBEDDING_COLS if feature_set == "embedding" else ACTIVITY_COLS

    n = len(binned)
    if n == 0:
        return None, np.array([], dtype=np.int32)
    if n < 2:
        return None, np.zeros(n, dtype=np.int32)

    X = _feature_matrix(binned, cols)
    n_states = min(int(n_states), max(2, n // 2), n)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        random_state=random_state,
        n_iter=200,
    )
    try:
        model.fit(X)
        states = np.asarray(model.predict(X), dtype=np.int32)
    except ValueError:
        return None, np.zeros(n, dtype=np.int32)
    return model, states


def n_switches(states: np.ndarray) -> int:
    s = np.asarray(states)
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))
