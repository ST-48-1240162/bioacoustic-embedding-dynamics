"""Synthetic demo manifest with regime shifts and 128-d embeddings."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .embeddings import SyntheticEmbedder, regime_ids

# Sentinel / soundscape species pools (demo labels only)
REGIME_A = ["Pseudopipra pipra", "Trogon rufus", "Ramphocaenus melanurus", "Myrmotherula axillaris"]
REGIME_B = ["Pseudopipra pipra", "Trogon rufus", "Ramphocaenus melanurus", "Myrmotherula axillaris", "Pteroglossus torquatus", "Catharus ustulatus"]
REGIME_C = ["Pteroglossus torquatus", "Catharus ustulatus", "Momotus momota", "Piaya cayana"]


def write_sample_manifest(
    path: Path,
    *,
    duration_s: float = 3 * 3600.0,
    seed: int = 42,
    embed_dim: int = 128,
) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t = 0.0
    while t < duration_s:
        frac = t / duration_s
        if frac < 1.0 / 3.0:
            pool, rate = REGIME_A, 0.38
        elif frac < 2.0 / 3.0:
            pool, rate = REGIME_B, 0.92
        else:
            pool, rate = REGIME_C, 0.48
        if rng.random() < rate * 0.14:
            start = t
            end = min(t + 3.0, duration_s)
            rows.append(
                {
                    "start_s": round(start, 2),
                    "end_s": round(end, 2),
                    "species": rng.choice(pool),
                    "confidence": round(rng.uniform(0.58, 0.97), 3),
                    "site": "costa-rica-sentinel-demo",
                }
            )
        t += rng.uniform(7.0, 22.0)

    import pandas as pd
    import torch

    df = pd.DataFrame(rows)
    t_norm = (df["start_s"].to_numpy() - df["start_s"].min()) / max(df["start_s"].max() - df["start_s"].min(), 1.0)
    rids = regime_ids(df["start_s"].to_numpy(), duration_s)

    torch.manual_seed(seed)
    model = SyntheticEmbedder(df["species"].tolist(), dim=embed_dim, seed=seed)
    drift = torch.randn(embed_dim)
    drift = drift / (drift.norm() + 1e-8)
    model.drift.copy_(drift)
    X = model(df["species"].tolist(), t_norm, rids)

    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row["embedding"] = np.round(X[i], 5).tolist()
            f.write(json.dumps(row) + "\n")
