"""PCA and UMAP dimensionality reduction."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


def run_pca(X: np.ndarray, n_components: int = 2, *, random_state: int = 42) -> tuple[np.ndarray, PCA]:
    n = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z, pca


def run_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    *,
    random_state: int = 42,
) -> np.ndarray:
    if umap is None:
        raise ImportError("umap-learn is required")
    if X.shape[0] < 4:
        Z, _ = run_pca(X, n_components=min(2, X.shape[0], X.shape[1]), random_state=random_state)
        if Z.shape[1] == 1:
            Z = np.column_stack([Z[:, 0], np.zeros(len(Z), dtype=Z.dtype)])
        return Z
    n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)
