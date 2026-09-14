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


def _real_birdnet_manifest(manifest: Path, wav: Path) -> None:
    run([PYTHON, "-m", "pip", "install", "-q", "soundfile"], timeout=600)
    run([PYTHON, "-m", "pip", "install", "-q", "bioacoustics-model-zoo[birdnet]"], timeout=1800)
    code = f"""
from pathlib import Path
from bioacoustic_embedding_dynamics.adapters import bmz_birdnet_to_manifest

manifest = Path({str(manifest)!r})
bmz_birdnet_to_manifest([Path({str(wav)!r})], manifest, batch_size=8, min_confidence=0.0)
print("manifest lines:", sum(1 for _ in manifest.open()))
print("wav:", {str(wav)!r})
"""
    run([PYTHON, "-c", code], cwd=ROOT, timeout=1800)


def _analysis(manifest: Path, out: Path) -> None:
    run(
        [
            PYTHON,
            "-m",
            "bioacoustic_embedding_dynamics.cli",
            "--manifest",
            str(manifest.relative_to(ROOT)),
            "--out",
            str(out.relative_to(ROOT)),
            "--seed",
            "42",
            "--bin-s",
            "15",
        ],
        cwd=ROOT,
    )


def test_demo() -> None:
    print("\n=== TEST 1: demo — BMZ BirdNET on bacpipe test wav ===", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from colab_wav_source import bacpipe_test_wav, install_bacpipe_for_test_wav

    install_bacpipe_for_test_wav()
    wav = bacpipe_test_wav()
    print("test wav:", wav, flush=True)
    manifest = ROOT / "data/demo_birdnet.jsonl"
    _real_birdnet_manifest(manifest, wav)
    _analysis(manifest, ROOT / "reports/demo")
    summary = assert_report(ROOT / "reports/demo", "demo")
    if summary.get("synthesized_embeddings"):
        raise SystemExit("[demo] expected real BirdNET embeddings")
    if int(summary.get("embed_dim") or 0) < 256:
        raise SystemExit(f"[demo] bad embed_dim: {summary.get('embed_dim')}")


def test_bmz() -> None:
    print("\n=== TEST 2: Route B — same short real wav ===", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from colab_wav_source import bacpipe_test_wav, install_bacpipe_for_test_wav

    install_bacpipe_for_test_wav()
    wav = bacpipe_test_wav()
    manifest = ROOT / "data/bmz_birdnet.jsonl"
    _real_birdnet_manifest(manifest, wav)
    _analysis(manifest, ROOT / "reports/bmz")
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
