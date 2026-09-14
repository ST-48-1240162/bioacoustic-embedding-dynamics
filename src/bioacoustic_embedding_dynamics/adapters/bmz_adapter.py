"""Convert bioacoustics-model-zoo BirdNET output to a JSONL manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .manifest_io import write_manifest_jsonl


def _species_label(raw: str) -> str:
    """BirdNET labels are often ScientificName_CommonName."""
    if "_" in raw:
        return raw.split("_", 1)[0]
    return raw


def _logits_to_confidence(logits: np.ndarray) -> float:
    """BirdNET BMZ predict() returns logits, not probabilities."""
    x = np.asarray(logits, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = x - x.max()
    p = np.exp(x)
    return float(p.max() / p.sum())


def bmz_birdnet_to_manifest(
    audio_files: Sequence[str | Path],
    out_path: Path | str,
    *,
    batch_size: int = 32,
    min_confidence: float = 0.0,
) -> Path:
    """Write JSONL compatible with this package."""
    import bioacoustics_model_zoo as bmz

    paths = [str(Path(p).resolve()) for p in audio_files]
    if not paths:
        raise ValueError("audio_files is empty")

    model = bmz.BirdNET()
    scores = model.predict(paths, batch_size=batch_size)
    embeds = model.embed(paths, batch_size=batch_size)

    if not scores.index.equals(embeds.index):
        embeds = embeds.reindex(scores.index)

    rows: list[dict] = []
    for idx, score_row in scores.iterrows():
        conf = _logits_to_confidence(score_row.to_numpy())
        if conf < min_confidence:
            continue
        species = _species_label(str(score_row.idxmax()))
        vec = np.asarray(embeds.loc[idx].to_numpy(), dtype=np.float32).reshape(-1)

        if len(idx) == 3:
            file_path, start_t, end_t = idx
        else:
            file_path, start_t, end_t = idx[0], float(idx[1]), float(idx[2])

        rows.append(
            {
                "start_s": round(float(start_t), 3),
                "end_s": round(float(end_t), 3),
                "species": species,
                "confidence": round(conf, 4),
                "site": str(Path(file_path).name),
                "embedding": np.round(vec, 6).tolist(),
            }
        )

    out = Path(out_path)
    n = write_manifest_jsonl(rows, out)
    if n == 0:
        raise ValueError("No rows passed min_confidence; lower min_confidence or check audio paths")
    return out
