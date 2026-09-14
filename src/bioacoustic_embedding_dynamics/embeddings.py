"""High-dimensional acoustic embedding vectors (load, validate, synthesize)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SyntheticEmbedder(nn.Module):
    """Demo PyTorch mapper from species prototype and temporal drift to an embedding."""

    def __init__(self, species: Iterable[str], dim: int = 128, seed: int = 42) -> None:
        super().__init__()
        uniq = sorted(set(species))
        rng = np.random.default_rng(seed)
        protos = rng.standard_normal((len(uniq), dim)).astype(np.float32)
        protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
        self.register_buffer("prototypes", torch.from_numpy(protos))
        self.species_index = {s: i for i, s in enumerate(uniq)}
        self.drift = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, species: list[str], t_norm: np.ndarray, regime_id: np.ndarray) -> np.ndarray:
        idx = torch.tensor([self.species_index.get(s, 0) for s in species], dtype=torch.long)
        base = self.prototypes[idx]
        drift = self.drift.unsqueeze(0).expand(len(species), -1)
        regime = torch.from_numpy(regime_id.astype(np.float32)).unsqueeze(1)
        t = torch.from_numpy(t_norm.astype(np.float32)).unsqueeze(1)
        noise = torch.randn(len(species), base.shape[1]) * 0.08
        out = base + 0.35 * regime * drift + 0.12 * t * drift + noise
        out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        return out.detach().cpu().numpy()


def regime_ids(start_s: np.ndarray, duration_s: float) -> np.ndarray:
    """Split detections into three sequential activity regimes (demo)."""
    t_norm = (start_s - start_s.min()) / max(float(start_s.max() - start_s.min()), 1.0)
    ids = np.zeros(len(start_s), dtype=np.int32)
    ids[t_norm >= 1.0 / 3.0] = 1
    ids[t_norm >= 2.0 / 3.0] = 2
    return ids


def parse_embedding(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        arr = np.asarray(raw, dtype=np.float32)
    elif isinstance(raw, dict) and "values" in raw:
        arr = np.asarray(raw["values"], dtype=np.float32)
    else:
        return None
    if arr.ndim != 1 or arr.size < 4:
        return None
    return arr


def embedding_matrix(
    df: pd.DataFrame,
    *,
    dim: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, bool]:
    """Stack per-detection embeddings; synthesize with PyTorch if column absent."""
    if "embedding" in df.columns:
        vecs = [parse_embedding(v) for v in df["embedding"]]
        ok = [v is not None for v in vecs]
        if all(ok):
            X = np.stack(vecs).astype(np.float32)
            return X, False
        if any(ok):
            raise ValueError("manifest mixes rows with and without usable embeddings")

    t0, t1 = df["start_s"].min(), df["start_s"].max()
    duration = float(t1 - t0) if t1 > t0 else 1.0
    t_norm = (df["start_s"].to_numpy() - t0) / max(duration, 1.0)
    rids = regime_ids(df["start_s"].to_numpy(), duration)

    torch.manual_seed(seed)
    model = SyntheticEmbedder(df["species"].tolist(), dim=dim, seed=seed)
    drift_vec = torch.randn(dim)
    drift_vec = drift_vec / (drift_vec.norm() + 1e-8)
    model.drift.copy_(drift_vec)

    X = model(df["species"].tolist(), t_norm, rids)
    return X.astype(np.float32), True
