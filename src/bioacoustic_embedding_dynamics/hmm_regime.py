"""Gaussian HMM on PCA-reduced binned timelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA


def fit_regime_hmm(binned: pd.DataFrame, n_states: int = 3, n_pca: int = 2) -> tuple[GaussianHMM, np.ndarray, PCA]:
    cols = ["detection_rate", "species_richness", "mean_confidence"]
    X = binned[cols].to_numpy(dtype=np.float64)
    n_pca = min(n_pca, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    Xp = pca.fit_transform(X)
    n_states = min(n_states, max(2, len(Xp) // 2))
    model = GaussianHMM(n_components=n_states, covariance_type="diag", random_state=42, n_iter=200)
    model.fit(Xp)
    states = model.predict(Xp)
    return model, states, pca
