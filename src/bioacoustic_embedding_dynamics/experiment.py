"""Reproducible experiment config and run metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class ExperimentConfig:
    manifest: str
    out_dir: str
    seed: int = 42
    embed_dim: int = 128
    bin_s: float = 60.0
    changepoint_pen: float = 3.0
    hmm_states: int = 3
    umap_neighbors: int = 15


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def write_run_metadata(out_dir: Path, config: ExperimentConfig, *, synthesized: bool) -> None:
    meta = {
        **asdict(config),
        "synthesized_embeddings": synthesized,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "pytorch+sklearn",
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
