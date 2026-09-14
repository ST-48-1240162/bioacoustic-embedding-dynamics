"""End-to-end analysis of high-dimensional bioacoustic embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .changepoint import detection_rate_changepoints, trajectory_changepoints
from .embeddings import embedding_matrix
from .experiment import ExperimentConfig, set_global_seed, write_run_metadata
from .features import scale_features
from .hmm_regime import fit_regime_hmm, n_switches
from .io import load_detections
from .plots import (
    save_hmm_compare,
    save_scatter_2d,
    save_shuffle_null,
    save_timeline_changepoints,
    save_trajectory_path,
)
from .reduce import run_pca, run_umap
from .trajectory import (
    bin_embedding_centroids,
    embedding_geometry,
    occupied_bins,
    permute_start_times,
    trajectory_metrics,
)


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
    shuffle_trajectory_changepoints_min: list[float]
    hmm_n_states: int
    hmm_activity_n_states: int
    hmm_n_switches: int
    shuffle_hmm_n_switches: int
    n_bins: int
    n_occupied_bins: int
    weighted_centroids: bool


def _minutes(times: list[float]) -> list[float]:
    return [round(float(t) / 60.0, 2) for t in times]


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
    Z_pca, pca = run_pca(Xs, n_components=2, random_state=seed)
    Z_umap = run_umap(Xs, n_neighbors=umap_neighbors, random_state=seed)

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

    geom = embedding_geometry(X, df["species"], seed=seed)
    centroids = bin_embedding_centroids(df, X, Z_pca, bin_s=bin_s, weight_by_confidence=True)
    centroids.to_csv(out_dir / "embedding_trajectory.csv", index=False)
    occ = occupied_bins(centroids)

    traj = trajectory_metrics(occ)

    cps = detection_rate_changepoints(centroids, pen=changepoint_pen)
    save_timeline_changepoints(centroids, cps, out_dir / "changepoints.png")

    traj_cps = trajectory_changepoints(occ, pen=changepoint_pen)
    save_timeline_changepoints(
        occ,
        traj_cps,
        out_dir / "trajectory_changepoints.png",
        column="pca_x",
        title="Embedding trajectory (PC1) and change-points",
        ylabel="PC1 centroid",
    )

    _hmm_emb, embed_states = fit_regime_hmm(
        occ, n_states=hmm_states, feature_set="embedding", random_state=seed
    )
    _hmm_act, activity_states = fit_regime_hmm(
        centroids, n_states=hmm_states, feature_set="activity", random_state=seed
    )
    save_hmm_compare(occ, embed_states, centroids, activity_states, out_dir / "hmm_regimes.png")
    save_trajectory_path(occ, out_dir / "trajectory_pca.png", states=embed_states)

    hmm_state = np.full(len(centroids), np.nan)
    occ_mask = centroids["pca_x"].notna().to_numpy()
    hmm_state[occ_mask] = embed_states
    centroids.assign(hmm_state=hmm_state, hmm_activity_state=activity_states).to_csv(
        out_dir / "binned_with_hmm.csv", index=False
    )

    df_shuf = permute_start_times(df, seed=seed)
    centroids_shuf = bin_embedding_centroids(
        df_shuf, X, Z_pca, bin_s=bin_s, weight_by_confidence=True
    )
    occ_shuf = occupied_bins(centroids_shuf)
    traj_cps_shuf = trajectory_changepoints(occ_shuf, pen=changepoint_pen)
    _hmm_shuf, embed_states_shuf = fit_regime_hmm(
        occ_shuf, n_states=hmm_states, feature_set="embedding", random_state=seed
    )
    save_shuffle_null(
        occ,
        occ_shuf,
        traj_cps,
        traj_cps_shuf,
        embed_states,
        embed_states_shuf,
        out_dir / "shuffle_null.png",
    )

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
        changepoints_min=_minutes(cps),
        trajectory_changepoints_min=_minutes(traj_cps),
        shuffle_trajectory_changepoints_min=_minutes(traj_cps_shuf),
        hmm_n_states=int(len(np.unique(embed_states))),
        hmm_activity_n_states=int(len(np.unique(activity_states))),
        hmm_n_switches=n_switches(embed_states),
        shuffle_hmm_n_switches=n_switches(embed_states_shuf),
        n_bins=int(len(centroids)),
        n_occupied_bins=int(len(occ)),
        weighted_centroids=True,
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    return summary
