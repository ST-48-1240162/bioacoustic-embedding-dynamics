#!/usr/bin/env python3
from pathlib import Path

from bioacoustic_embedding_dynamics.sample_data import write_sample_manifest

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "sample_detections.jsonl"
    write_sample_manifest(out)
    print(f"Wrote {out}")
