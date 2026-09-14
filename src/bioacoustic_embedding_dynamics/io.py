"""Load bioacoustic detection manifests (JSONL with optional embedding vectors)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .embeddings import parse_embedding

_START_KEYS = ("start_s", "start_sec", "start", "offset_s", "window_start_s")
_END_KEYS = ("end_s", "end_sec", "end", "window_end_s")
_SPECIES_KEYS = ("species", "scientific_name", "label", "common_name")
_CONF_KEYS = ("confidence", "score", "prob")
_SITE_KEYS = ("site", "location", "recorder", "source_file", "audio_path")


def _first(row: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def load_detections(manifest_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        start = _first(raw, _START_KEYS, 0.0)
        end = _first(raw, _END_KEYS, float(start) + 3.0)
        row = {
            "start_s": float(start),
            "end_s": float(end),
            "duration_s": float(end) - float(start),
            "species": str(_first(raw, _SPECIES_KEYS, "Unknown")),
            "confidence": float(_first(raw, _CONF_KEYS, 0.5)),
            "site": str(_first(raw, _SITE_KEYS, "site")),
        }
        emb = parse_embedding(raw.get("embedding"))
        if emb is not None:
            row["embedding"] = emb.tolist()
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {manifest_path}")
    df = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)
    df["species"] = df["species"].astype(str)
    return df
