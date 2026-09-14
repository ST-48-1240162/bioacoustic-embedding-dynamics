#!/usr/bin/env python3
"""Colab VM: full integration tests. Use --skip-bacpipe for fast path (demo + BMZ only)."""

from __future__ import annotations

import argparse
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
        p = out_dir / name
        if not p.is_file():
            raise SystemExit(f"[{label}] missing figure {p}")
    print(f"[{label}] n_detections={summary.get('n_detections')} embed_dim={summary.get('embed_dim')}")
    return summary


def test_demo() -> None:
    print("\n=== TEST 1: demo manifest ===", flush=True)
    run(
        [PYTHON, "-m", "bioacoustic_embedding_dynamics.cli", "--make-sample", "--out", str(ROOT / "reports/demo"), "--seed", "42"],
        cwd=ROOT,
    )
    assert_report(ROOT / "reports/demo", "demo")


def test_bacpipe() -> None:
    print("\n=== TEST 2: Route A bacpipe (GPU or CPU) ===", flush=True)
    run(
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            "--ignore-requires-python",
            "--no-deps",
            "bacpipe",
        ],
        timeout=300,
    )

    code = """
import sys
from pathlib import Path
sys.path.insert(0, "/content/bioacoustic-embedding-dynamics/src")

import librosa
if not hasattr(librosa, "get_duration"):
    from librosa.core.audio import get_duration as _gd
    librosa.get_duration = _gd

import bacpipe
from bioacoustic_embedding_dynamics.adapters import bacpipe_loader_to_manifest

MODEL = "birdnet"
audio_dir = Path(bacpipe.__file__).parent / "tests" / "test_data"
bacpipe.config.models = [MODEL]
bacpipe.config.dashboard = False
bacpipe.config.audio_dir = str(audio_dir)
bacpipe.settings.device = "cpu"
print("audio_dir:", audio_dir)
bacpipe.ensure_models_exist(model_names=[MODEL])
loader = bacpipe.generate_embeddings(
    model_name=MODEL,
    audio_dir=str(audio_dir),
    check_if_already_processed=True,
)
manifest = Path("data/bacpipe_birdnet.jsonl")
bacpipe_loader_to_manifest({MODEL: loader}, MODEL, manifest, min_confidence=0.0)
print("manifest lines:", sum(1 for _ in manifest.open()))
"""
    run([PYTHON, "-c", code], cwd=ROOT, timeout=3600)
    run(
        [
            PYTHON,
            "-m",
            "bioacoustic_embedding_dynamics.cli",
            "--manifest",
            str(ROOT / "data/bacpipe_birdnet.jsonl"),
            "--out",
            str(ROOT / "reports/bacpipe"),
            "--seed",
            "42",
        ],
        cwd=ROOT,
    )
    summary = assert_report(ROOT / "reports/bacpipe", "bacpipe")
    if summary.get("synthesized_embeddings"):
        raise SystemExit("[bacpipe] expected real embeddings, got synthesized")
    if summary.get("embed_dim", 0) < 256:
        raise SystemExit(f"[bacpipe] unexpected embed_dim: {summary.get('embed_dim')}")


def _ensure_sample_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    # 5 s mono 48 kHz. BirdNET BMZ resamples or pads as needed.
    import numpy as np
    import soundfile as sf

    sr = 48000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 880 * t) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    sf.write(path, y.astype(np.float32), sr)
    print(f"Wrote synthetic wav to {path}")


def test_bmz() -> None:
    print("\n=== TEST 3: Route B BMZ ===", flush=True)
    run([PYTHON, "-m", "pip", "install", "-q", "soundfile"])
    run([PYTHON, "-m", "pip", "install", "-q", "bioacoustics-model-zoo[birdnet]"], timeout=3600)

    wav = ROOT / "data/test_audio/sample.wav"
    _ensure_sample_wav(wav)

    code = f"""
from pathlib import Path
from bioacoustic_embedding_dynamics.adapters import bmz_birdnet_to_manifest

manifest = Path("data/bmz_birdnet.jsonl")
bmz_birdnet_to_manifest([r"{wav}"], manifest, batch_size=8, min_confidence=0.0)
print("manifest lines:", sum(1 for _ in manifest.open()))
"""
    run([PYTHON, "-c", code], cwd=ROOT, timeout=3600)
    run(
        [
            PYTHON,
            "-m",
            "bioacoustic_embedding_dynamics.cli",
            "--manifest",
            str(ROOT / "data/bmz_birdnet.jsonl"),
            "--out",
            str(ROOT / "reports/bmz"),
            "--seed",
            "42",
        ],
        cwd=ROOT,
    )
    summary = assert_report(ROOT / "reports/bmz", "bmz")
    if summary.get("synthesized_embeddings"):
        raise SystemExit("[bmz] expected real embeddings, got synthesized")
    if summary.get("embed_dim") != 1024:
        raise SystemExit(f"[bmz] expected embed_dim 1024, got {summary.get('embed_dim')}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-bacpipe", action="store_true", help="skip Route A (use colab_run_fast_tests.py)")
    p.add_argument("--bacpipe-only", action="store_true", help="only Route A bacpipe")
    args = p.parse_args()

    extract_project()
    install_base()
    if not args.bacpipe_only:
        test_demo()
    if not args.skip_bacpipe:
        test_bacpipe()
    if not args.bacpipe_only:
        test_bmz()
    label = "ALL TESTS PASSED" if not args.skip_bacpipe else "FAST TESTS PASSED (demo + BMZ)"
    print(f"\n=== {label} ===", flush=True)

    out_tgz = Path("/content/bioacoustic-embedding-dynamics-all-reports.tgz")
    run(["tar", "czf", str(out_tgz), "-C", str(ROOT / "reports"), "."])
    print(f"Reports tarball: {out_tgz}")


if __name__ == "__main__":
    main()
