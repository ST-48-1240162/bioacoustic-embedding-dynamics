"""Convert a bacpipe BirdNET loader to a JSONL manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest_io import write_manifest_jsonl


def _index_to_label(label2index: dict) -> dict[int, str]:
    return {int(v): str(k) for k, v in label2index.items()}


def _species_label(raw: str) -> str:
    if "_" in raw:
        return raw.split("_", 1)[0]
    return raw


def _resolve_window_hop_s(meta: dict, model_name: str) -> tuple[float, float]:
    sr = float(meta.get("sample_rate (Hz)") or meta.get("sample_rate") or 48000)
    seg = meta.get("segment_length (samples)") or meta.get("segment_length")
    if seg is None:
        window_s = 3.0 if "birdnet" in model_name.lower() else 5.0
    else:
        window_s = float(seg) / sr
    hop_s = float(meta.get("hop_length (samples)", seg) or seg or window_s * sr) / sr
    if hop_s <= 0:
        hop_s = window_s
    return window_s, hop_s


def _strip_model_suffix(stem: str) -> str:
    for suffix in ("_birdnet", "_birdnet_v3", "_perch_bird"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _match_embeddings_for_audio(
    audio_path: str,
    embeds_dict: dict,
) -> tuple[str, np.ndarray] | None:
    audio_name = Path(audio_path).name
    audio_stem = Path(audio_path).stem
    for key, arr in embeds_dict.items():
        key_str = str(key)
        key_stem = _strip_model_suffix(Path(key_str).stem)
        if (
            key_str.endswith(audio_name)
            or audio_name in key_str
            or key_stem == audio_stem
            or audio_stem in Path(key_str).name
        ):
            return key_str, np.asarray(arr, dtype=np.float32)
    return None


def bacpipe_loader_to_manifest(
    loader: dict,
    model_name: str,
    out_path: Path | str,
    *,
    min_confidence: float = 0.0,
    window_s: float | None = None,
    hop_s: float | None = None,
) -> Path:
    """Convert bacpipe `run_pipeline_for_models` loader to JSONL manifest."""
    if model_name not in loader:
        raise KeyError(f"model {model_name!r} not in loader keys: {list(loader.keys())}")

    model_loader = loader[model_name]
    meta = model_loader.metadata_dict
    embeds_dict = model_loader.embeddings()

    try:
        predictions, label2index = model_loader.predictions()
    except Exception:
        predictions, label2index = {}, {}

    idx2label = _index_to_label(label2index)
    default_window, default_hop = _resolve_window_hop_s(meta, model_name)
    window_s = window_s or default_window
    hop_s = hop_s or default_hop

    files_block = meta.get("files") or {}
    audio_files = files_block.get("audio_files") or list(predictions.keys()) or list(embeds_dict.keys())

    rows: list[dict] = []
    for audio_path in audio_files:
        matched = _match_embeddings_for_audio(str(audio_path), embeds_dict)
        if matched is None:
            continue
        _key, emb = matched
        if emb.ndim != 2:
            continue

        pred = predictions.get(audio_path) if predictions else None
        if pred is None:
            pred = predictions.get(str(audio_path)) if predictions else None
        if pred is not None:
            pred = np.asarray(pred)

        site = Path(str(audio_path)).name
        for j in range(emb.shape[0]):
            start_s = j * hop_s
            end_s = start_s + window_s
            species = "Unknown"
            confidence = 0.5
            if pred is not None and j < pred.shape[0]:
                if pred.ndim == 2 and pred.shape[1] > 1:
                    top = int(np.argmax(pred[j]))
                    confidence = float(np.max(pred[j]))
                    species = _species_label(idx2label.get(top, f"class_{top}"))
                else:
                    confidence = float(pred[j]) if pred.ndim == 1 else float(pred[j, 0])

            if confidence < min_confidence:
                continue

            rows.append(
                {
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "species": species,
                    "confidence": round(confidence, 4),
                    "site": site,
                    "embedding": np.round(emb[j], 6).tolist(),
                }
            )

    out = Path(out_path)
    n = write_manifest_jsonl(rows, out)
    if n == 0:
        raise ValueError("No rows written; check bacpipe loader outputs and audio paths")
    return out
