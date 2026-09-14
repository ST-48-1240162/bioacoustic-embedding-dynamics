#!/usr/bin/env python3
"""Colab tests for the demo pipeline and Route B (BMZ). About 5-10 minutes."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/content/bioacoustic-embedding-dynamics")
TGZ = Path("/content/bioacoustic-embedding-dynamics.tgz")
PYTHON = sys.executable

REQUIRED_FIGURES = [
    "pca_species.png",
    "umap_species.png",
    "trajectory_pca.png",
    "changepoints.png",
    "trajectory_changepoints.png",
    "hmm_regimes.png",
    "shuffle_null.png",
]


def run(cmd: list[str], *, timeout: int = 3600, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, timeout=timeout, cwd=str(cwd) if cwd else None)


def extract_project() -> None:
    if TGZ.is_file():
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TGZ, "r:gz") as tf:
            tf.extractall(ROOT)
        print(f"Extracted to {ROOT}")
    elif not (ROOT / "pyproject.toml").is_file():
        raise SystemExit(f"Missing {TGZ} and {ROOT}")


def install_base() -> None:
    run([PYTHON, "-m", "pip", "install", "-q", "-r", str(ROOT / "docs/colab-requirements.txt")])
    run([PYTHON, "-m", "pip", "install", "-q", "-e", str(ROOT)])


def assert_report(out_dir: Path, label: str) -> dict:
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"[{label}] missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for name in REQUIRED_FIGURES:
        if not (out_dir / name).is_file():
            raise SystemExit(f"[{label}] missing figure {name}")
    print(
        f"[{label}] n={summary.get('n_detections')} dim={summary.get('embed_dim')} synth={summary.get('synthesized_embeddings')}",
        flush=True,
    )
    return summary


def test_demo() -> None:
    print("\n=== TEST 1: demo manifest ===", flush=True)
    run(
        [PYTHON, "-m", "bioacoustic_embedding_dynamics.cli", "--make-sample", "--out", "reports/demo", "--seed", "42"],
        cwd=ROOT,
    )
    assert_report(ROOT / "reports/demo", "demo")


def _ensure_sample_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    import numpy as np
    import soundfile as sf

    sr = 48000
    duration = 120.0  # ~40 BirdNET 3s windows
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 880 * t) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    sf.write(path, y.astype(np.float32), sr)
    print(f"Wrote synthetic wav to {path}", flush=True)


def test_bmz() -> None:
    print("\n=== TEST 2: Route B BMZ BirdNET ===", flush=True)
    run([PYTHON, "-m", "pip", "install", "-q", "soundfile"], timeout=600)
    run([PYTHON, "-m", "pip", "install", "-q", "bioacoustics-model-zoo[birdnet]"], timeout=1800)

    wav = ROOT / "data/test_audio/sample.wav"
    if wav.is_file() and wav.stat().st_size > 0:
        wav.unlink()  # regenerate 120s clip if a short file exists from prior run
    _ensure_sample_wav(wav)

    code = f"""
from pathlib import Path
from bioacoustic_embedding_dynamics.adapters import bmz_birdnet_to_manifest

manifest = Path("data/bmz_birdnet.jsonl")
bmz_birdnet_to_manifest([r"{wav}"], manifest, batch_size=8, min_confidence=0.0)
print("manifest lines:", sum(1 for _ in manifest.open()))
"""
    run([PYTHON, "-c", code], cwd=ROOT, timeout=1800)
    run(
        [PYTHON, "-m", "bioacoustic_embedding_dynamics.cli", "--manifest", "data/bmz_birdnet.jsonl", "--out", "reports/bmz", "--seed", "42"],
        cwd=ROOT,
    )
    summary = assert_report(ROOT / "reports/bmz", "bmz")
    if summary.get("synthesized_embeddings"):
        raise SystemExit("[bmz] expected real BirdNET embeddings")
    if summary.get("embed_dim") != 1024:
        raise SystemExit(f"[bmz] expected embed_dim 1024, got {summary.get('embed_dim')}")


def main() -> None:
    extract_project()
    install_base()
    test_demo()
    test_bmz()
    print("\n=== FAST TESTS PASSED (demo + BMZ) ===", flush=True)
    print("Route A bacpipe: run docs/Route_A_Bacpipe_Colab.ipynb separately (GPU, ~30-60 min).", flush=True)

    out_tgz = Path("/content/bioacoustic-embedding-dynamics-fast-reports.tgz")
    run(["tar", "czf", str(out_tgz), "-C", str(ROOT / "reports"), "."])
    print(f"Reports tarball: {out_tgz}", flush=True)


if __name__ == "__main__":
    main()
