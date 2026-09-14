"""End-to-end analysis of high-dimensional bioacoustic embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .changepoint import detection_rate_changepoints, trajectory_changepoints
from .embeddings import embedding_matrix
from .experiment import ExperimentConfig, set_global_seed, write_run_metadata
from .features import scale_features
from .hmm_regime import fit_regime_hmm
from .io import load_detections
from .plots import (
    save_hmm_states,
    save_scatter_2d,
    save_timeline_changepoints,
    save_trajectory_path,
)
from .reduce import run_pca, run_umap
from .trajectory import bin_embedding_centroids, embedding_geometry, trajectory_metrics


@dataclass
class AnalysisSummary:
    n_detections: int
    embed_dim: int
    synthesized_embeddings: bool
    duration_min: float
    pca_explained_variance: list[float]
    embedding_geometry: dict[str, float]
    trajectory: dict[str, float]
    changepoints_min: list[float]
    trajectory_changepoints_min: list[float]
    hmm_n_states: int


def run_analysis(
    manifest_path: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    embed_dim: int = 128,
    bin_s: float = 60.0,
    changepoint_pen: float = 3.0,
    hmm_states: int = 3,
    umap_neighbors: int = 15,
) -> AnalysisSummary:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        manifest=str(manifest_path),
        out_dir=str(out_dir),
        seed=seed,
        embed_dim=embed_dim,
        bin_s=bin_s,
        changepoint_pen=changepoint_pen,
        hmm_states=hmm_states,
        umap_neighbors=umap_neighbors,
    )

    df = load_detections(manifest_path)
    X, synthesized = embedding_matrix(df, dim=embed_dim, seed=seed)
    write_run_metadata(out_dir, config, synthesized=synthesized)

    Xs, _scaler = scale_features(X)
    Z_pca, pca = run_pca(Xs, n_components=2)
    Z_umap = run_umap(Xs, n_neighbors=umap_neighbors)

    save_scatter_2d(
        Z_pca,
        df["species"],
        out_dir / "pca_species.png",
        title="PCA of acoustic embeddings (species color)",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)" if len(pca.explained_variance_ratio_) > 1 else "PC2",
    )
    save_scatter_2d(
        Z_umap,
        df["species"],
        out_dir / "umap_species.png",
        title="UMAP of acoustic embeddings (species color)",
        xlabel="UMAP-1",
        ylabel="UMAP-2",
    )

    geom = embedding_geometry(X, df["species"])
    centroids = bin_embedding_centroids(df, X, Z_pca, bin_s=bin_s)
    centroids.to_csv(out_dir / "embedding_trajectory.csv", index=False)

    traj = trajectory_metrics(centroids)
    save_trajectory_path(centroids, out_dir / "trajectory_pca.png")

    cps = detection_rate_changepoints(centroids, pen=changepoint_pen)
    save_timeline_changepoints(centroids, cps, out_dir / "changepoints.png")

    traj_cps = trajectory_changepoints(centroids, pen=changepoint_pen)
    save_timeline_changepoints(
        centroids,
        traj_cps,
        out_dir / "trajectory_changepoints.png",
        column="pca_x",
        title="Embedding trajectory (PC1) and change-points",
        ylabel="PC1 centroid",
    )

    _hmm, states, _pca_bin = fit_regime_hmm(centroids, n_states=hmm_states)
    save_hmm_states(centroids, states, out_dir / "hmm_regimes.png")
    centroids.assign(hmm_state=states).to_csv(out_dir / "binned_with_hmm.csv", index=False)

    duration_min = float((df["start_s"].max() - df["start_s"].min()) / 60.0)
    summary = AnalysisSummary(
        n_detections=len(df),
        embed_dim=int(X.shape[1]),
        synthesized_embeddings=synthesized,
        duration_min=round(duration_min, 1),
        pca_explained_variance=[round(float(v), 4) for v in pca.explained_variance_ratio_[:2]],
        embedding_geometry=geom,
        trajectory={
            "path_length": traj.path_length,
            "mean_step_velocity": traj.mean_step_velocity,
            "max_step_velocity": traj.max_step_velocity,
            "turning_angle_mean_deg": traj.turning_angle_mean_deg,
        },
        changepoints_min=[round(float(t) / 60.0, 2) for t in cps],
        trajectory_changepoints_min=[round(float(t) / 60.0, 2) for t in traj_cps],
        hmm_n_states=int(len(np.unique(states))),
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    return summary
