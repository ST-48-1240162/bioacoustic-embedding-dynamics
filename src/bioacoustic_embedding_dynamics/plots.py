"""Matplotlib figures for embedding geometry and temporal dynamics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_timeline_changepoints(
    binned: pd.DataFrame,
    changepoints: list[float],
    out: Path,
    *,
    column: str = "detection_rate",
    title: str = "Bioacoustic activity + change-points",
    ylabel: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(binned["t_center_s"] / 60.0, binned[column], marker="o", ms=3, lw=1)
    for cp in changepoints:
        ax.axvline(cp / 60.0, color="crimson", ls="--", lw=1, alpha=0.8)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(ylabel or column.replace("_", " "))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_scatter_2d(
    Z: np.ndarray,
    labels: pd.Series,
    out: Path,
    *,
    title: str,
    xlabel: str = "dim 1",
    ylabel: str = "dim 2",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = labels.astype("category")
    for cat in cats.cat.categories:
        mask = cats == cat
        ax.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.65, label=str(cat)[:28])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(cats.cat.categories) <= 12:
        ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_trajectory_path(
    centroids: pd.DataFrame,
    out: Path,
    *,
    states: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = centroids["pca_x"].to_numpy()
    y = centroids["pca_y"].to_numpy()
    ax.plot(x, y, color="0.75", lw=1, alpha=0.8, zorder=1)
    for i in range(len(x) - 1):
        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6, alpha=0.5),
        )
    if states is None:
        t = centroids["t_center_s"].to_numpy() / 60.0
        sc = ax.scatter(x, y, c=t, cmap="viridis", s=28, zorder=2)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Time (min)")
        ax.set_title("Embedding-space trajectory (binned centroids)")
    else:
        states = np.asarray(states)
        if len(states) != len(x):
            raise ValueError("states length must match centroids")
        for s in np.unique(states):
            mask = states == s
            ax.scatter(x[mask], y[mask], s=36, zorder=2, label=f"HMM {int(s)}")
        ax.legend(fontsize=8, loc="best", title="state")
        ax.set_title("Embedding-space trajectory (color = HMM state)")
    ax.set_xlabel("PC1 centroid")
    ax.set_ylabel("PC2 centroid")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_hmm_states(binned: pd.DataFrame, states: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    t = binned["t_center_s"].to_numpy() / 60.0
    ax.step(t, states, where="mid", lw=1.5)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("HMM state")
    ax.set_title("Gaussian HMM regimes on binned embedding dynamics")
    ax.set_yticks(sorted(np.unique(states)))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_hmm_compare(
    embed_binned: pd.DataFrame,
    embed_states: np.ndarray,
    activity_binned: pd.DataFrame,
    activity_states: np.ndarray,
    out: Path,
) -> None:
    embed_states = np.asarray(embed_states)
    activity_states = np.asarray(activity_states)
    if len(embed_states) != len(embed_binned):
        raise ValueError("embed_states length must match embed_binned")
    if len(activity_states) != len(activity_binned):
        raise ValueError("activity_states length must match activity_binned")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    t_emb = embed_binned["t_center_s"].to_numpy() / 60.0
    t_act = activity_binned["t_center_s"].to_numpy() / 60.0
    axes[0].step(t_emb, embed_states, where="mid", lw=1.5)
    axes[0].set_ylabel("state")
    axes[0].set_title("HMM on embedding centroids (PC1/PC2)")
    axes[0].set_yticks(sorted(np.unique(embed_states)))
    axes[1].step(t_act, activity_states, where="mid", lw=1.5)
    axes[1].set_ylabel("state")
    axes[1].set_title("HMM on activity (rate, richness, confidence)")
    axes[1].set_yticks(sorted(np.unique(activity_states)))
    axes[1].set_xlabel("Time (minutes)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_shuffle_null(
    original: pd.DataFrame,
    shuffled: pd.DataFrame,
    orig_cps: list[float],
    shuf_cps: list[float],
    orig_states: np.ndarray,
    shuf_states: np.ndarray,
    out: Path,
) -> None:
    """Time-shuffle control: embedding PC1 change-points and HMM should collapse."""
    if len(orig_states) != len(original) or len(shuf_states) != len(shuffled):
        raise ValueError("HMM state length must match the corresponding binned frame")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex="col")
    panels = (
        (axes[0, 0], original, orig_cps, "PC1 (original)", "pca_x"),
        (axes[0, 1], shuffled, shuf_cps, "PC1 (time shuffled)", "pca_x"),
        (axes[1, 0], original, None, "HMM on centroids (original)", orig_states),
        (axes[1, 1], shuffled, None, "HMM on centroids (time shuffled)", shuf_states),
    )
    for ax, frame, cps, title, extra in panels:
        t = frame["t_center_s"].to_numpy() / 60.0
        if isinstance(extra, str):
            ax.plot(t, frame[extra], marker="o", ms=3, lw=1)
            for cp in cps or []:
                ax.axvline(cp / 60.0, color="crimson", ls="--", lw=1, alpha=0.8)
            ax.set_ylabel("PC1 centroid")
        else:
            ax.step(t, extra, where="mid", lw=1.5)
            ax.set_ylabel("state")
            ax.set_yticks(sorted(np.unique(extra)))
        ax.set_title(title)
    axes[1, 0].set_xlabel("Time (minutes)")
    axes[1, 1].set_xlabel("Time (minutes)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
