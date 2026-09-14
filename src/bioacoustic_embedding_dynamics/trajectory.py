"""Embedding-space trajectories over time (centroid paths, velocity, geometry)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrajectorySummary:
    n_bins: int
    path_length: float
    mean_step_velocity: float
    max_step_velocity: float
    turning_angle_mean_deg: float


def permute_start_times(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Keep embeddings/species on each row; reassign start times from the same set."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    starts = out["start_s"].to_numpy(dtype=np.float64)
    durs = (out["end_s"].to_numpy(dtype=np.float64) - starts)
    new_start = rng.permutation(starts)
    out["start_s"] = new_start
    out["end_s"] = new_start + durs
    return out


def _confidence_weights(mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    w = np.clip(confidence[mask].astype(np.float64), 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        n = int(mask.sum())
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    return w / total


def bin_embedding_centroids(
    df: pd.DataFrame,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    bin_s: float = 60.0,
    weight_by_confidence: bool = True,
) -> pd.DataFrame:
    """Per-bin embedding centroid (confidence-weighted) and reduced coords (e.g. PCA)."""
    t0 = df["start_s"].min()
    t1 = df["start_s"].max()
    edges = np.arange(t0, t1 + bin_s, bin_s)
    if len(edges) < 3:
        edges = np.linspace(t0, t1, num=4)
    bin_idx = np.digitize(df["start_s"].to_numpy(), edges[1:-1], right=False)
    conf = df["confidence"].to_numpy(dtype=np.float64)
    starts = df["start_s"].to_numpy(dtype=np.float64)

    rows: list[dict] = []
    for b in sorted(np.unique(bin_idx)):
        mask = bin_idx == b
        if not mask.any():
            continue
        if weight_by_confidence:
            w = _confidence_weights(mask, conf)
            t_center = float(np.dot(starts[mask], w))
            mean_conf = float(np.dot(conf[mask], w))
            pca_x = float(np.dot(Z[mask, 0], w))
            pca_y = float(np.dot(Z[mask, 1], w)) if Z.shape[1] > 1 else 0.0
            emb = {f"emb_{i}": float(np.dot(X[mask, i], w)) for i in range(min(8, X.shape[1]))}
        else:
            t_center = float(starts[mask].mean())
            mean_conf = float(conf[mask].mean())
            pca_x = float(Z[mask, 0].mean())
            pca_y = float(Z[mask, 1].mean()) if Z.shape[1] > 1 else 0.0
            emb = {f"emb_{i}": float(X[mask, i].mean()) for i in range(min(8, X.shape[1]))}
        rows.append(
            {
                "bin_idx": int(b),
                "t_center_s": t_center,
                "detection_count": int(mask.sum()),
                "species_richness": int(pd.unique(df["species"].to_numpy()[mask]).size),
                "mean_confidence": mean_conf,
                "weight_sum": float(conf[mask].sum()) if weight_by_confidence else float(mask.sum()),
                **emb,
                "pca_x": pca_x,
                "pca_y": pca_y,
            }
        )
    out = pd.DataFrame(rows)
    out["detection_rate"] = out["detection_count"] / bin_s
    return out


def trajectory_metrics(centroids: pd.DataFrame, *, x_col: str = "pca_x", y_col: str = "pca_y") -> TrajectorySummary:
    pts = centroids[[x_col, y_col]].to_numpy(dtype=np.float64)
    if len(pts) < 2:
        return TrajectorySummary(0, 0.0, 0.0, 0.0, 0.0)

    steps = np.diff(pts, axis=0)
    dists = np.linalg.norm(steps, axis=1)
    path_length = float(dists.sum())
    mean_vel = float(dists.mean())
    max_vel = float(dists.max())

    angles: list[float] = []
    for i in range(1, len(steps)):
        v0, v1 = steps[i - 1], steps[i]
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        cos = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos))))

    return TrajectorySummary(
        n_bins=len(centroids),
        path_length=round(path_length, 4),
        mean_step_velocity=round(mean_vel, 4),
        max_step_velocity=round(max_vel, 4),
        turning_angle_mean_deg=round(float(np.mean(angles)) if angles else 0.0, 2),
    )


def embedding_geometry(X: np.ndarray, labels: pd.Series, *, max_pairs: int = 2000) -> dict[str, float]:
    """Intra- vs inter-species cosine distance in embedding space."""
    cats = labels.astype("category")
    codes = cats.cat.codes.to_numpy()
    n = len(codes)
    if n < 4:
        return {"intra_cosine_dist_mean": 0.0, "inter_cosine_dist_mean": 0.0, "separation_ratio": 0.0}

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n, max_pairs), replace=False)

    intra: list[float] = []
    inter: list[float] = []
    for i in idx:
        for j in idx:
            if i >= j:
                continue
            d = 1.0 - float(np.dot(Xn[i], Xn[j]))
            if codes[i] == codes[j]:
                intra.append(d)
            else:
                inter.append(d)

    intra_m = float(np.mean(intra)) if intra else 0.0
    inter_m = float(np.mean(inter)) if inter else 0.0
    ratio = round(inter_m / max(intra_m, 1e-6), 3)
    return {
        "intra_cosine_dist_mean": round(intra_m, 4),
        "inter_cosine_dist_mean": round(inter_m, 4),
        "separation_ratio": ratio,
    }
