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


def save_trajectory_path(centroids: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = centroids["pca_x"].to_numpy()
    y = centroids["pca_y"].to_numpy()
    t = centroids["t_center_s"].to_numpy() / 60.0
    ax.plot(x, y, color="steelblue", lw=1, alpha=0.5, zorder=1)
    sc = ax.scatter(x, y, c=t, cmap="viridis", s=28, zorder=2)
    for i in range(len(x) - 1):
        ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="->", color="gray", lw=0.6, alpha=0.5))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Time (min)")
    ax.set_xlabel("PC1 centroid")
    ax.set_ylabel("PC2 centroid")
    ax.set_title("Embedding-space trajectory (binned centroids)")
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
