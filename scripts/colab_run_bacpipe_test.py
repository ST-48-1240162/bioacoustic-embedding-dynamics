#!/usr/bin/env python3
"""Colab GPU test: bacpipe BirdNET embeddings, then this package.

bacpipe 1.3.x requires Python < 3.13 and pins TF/JAX/CUDA/numpy 1.26.
Colab is Python 3.13 with TF 2.20 and Torch already present. Install
bacpipe --no-deps and reuse the runtime stack.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tarfile
import threading
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

# Do not let pip pull these. Colab already has compatible builds.
SKIP_PIP = {
    "tensorflow",
    "tensorflow-hub",
    "tensorboard",
    "torch",
    "torchaudio",
    "torchvision",
    "jax",
    "jaxlib",
    "flax",
    "numpy",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
    "triton",
}


def run(cmd: list[str], *, timeout: int = 3600, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    stop = threading.Event()

    def heartbeat() -> None:
        tick = 0
        while not stop.wait(25):
            tick += 25
            print(f"[still running {tick}s]", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        subprocess.check_call(cmd, timeout=timeout, cwd=str(cwd) if cwd else None)
    finally:
        stop.set()
        thread.join(timeout=1)


def extract_project() -> None:
    if TGZ.is_file():
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TGZ, "r:gz") as tf:
            tf.extractall(ROOT)
        print(f"Extracted to {ROOT}")
    elif not (ROOT / "pyproject.toml").is_file():
        raise SystemExit(f"Missing {TGZ} and {ROOT}")


def verify_gpu() -> None:
    code = """
import torch
ok = torch.cuda.is_available()
print('cuda_available:', ok)
if ok:
    print('gpu:', torch.cuda.get_device_name(0))
if not ok:
    raise SystemExit('T4 GPU not visible. Use: colab new -s bel-gpu --gpu T4')
"""
    run([PYTHON, "-c", code], timeout=120)


def _missing_module(stderr: str) -> str | None:
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
    if m:
        return m.group(1)
    m = re.search(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]", stderr)
    if m:
        return m.group(2)
    return None


def install_bacpipe() -> None:
    print("\n=== install bacpipe (no-deps, Colab 3.13) ===", flush=True)
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
    # Analysis extras. Do not pin or replace numpy or torch.
    run(
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            "-q",
            "umap-learn>=0.5.6,<0.6",
            "ruptures>=1.1.9,<2",
            "hmmlearn>=0.3.3,<0.4",
        ],
        timeout=600,
    )
    run([PYTHON, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT)], timeout=180)

    probe = r"""
import librosa
if not hasattr(librosa, "get_duration"):
    from librosa.core.audio import get_duration as _gd
    librosa.get_duration = _gd
import bacpipe
print("bacpipe import OK", getattr(bacpipe, "__version__", "?"))
"""
    for _ in range(12):
        r = subprocess.run([PYTHON, "-c", probe], capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            print(r.stdout.strip(), flush=True)
            return
        missing = _missing_module(r.stderr)
        print(r.stderr[-1500:], flush=True)
        if not missing or missing.split(".")[0] in SKIP_PIP:
            raise SystemExit(f"bacpipe import failed (missing={missing})")
        pkg = missing.split(".")[0]
        print(f"installing missing runtime dep: {pkg}", flush=True)
        run([PYTHON, "-m", "pip", "install", "-q", pkg], timeout=600)
    raise SystemExit("bacpipe import still failing after dep loop")


def assert_report(out_dir: Path, label: str) -> dict:
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for name in REQUIRED_FIGURES:
        if not (out_dir / name).is_file():
            raise SystemExit(f"[{label}] missing {name}")
    print(
        f"[{label}] n={summary.get('n_detections')} dim={summary.get('embed_dim')} synth={summary.get('synthesized_embeddings')}",
        flush=True,
    )
    return summary


def test_bacpipe() -> None:
    print("\n=== Route A: bacpipe BirdNET (cuda) ===", flush=True)
    code = r"""
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
DEVICE = "cuda"
audio_dir = Path(bacpipe.__file__).parent / "tests" / "test_data"
bacpipe.config.models = [MODEL]
bacpipe.config.dashboard = False
bacpipe.config.audio_dir = str(audio_dir)
bacpipe.settings.device = DEVICE
print("audio_dir:", audio_dir, flush=True)
print("device:", DEVICE, flush=True)
bacpipe.ensure_models_exist(model_names=[MODEL])
loader = bacpipe.generate_embeddings(
    model_name=MODEL,
    audio_dir=str(audio_dir),
    check_if_already_processed=True,
)
print("embedding_size:", loader.metadata_dict.get("embedding_size"), flush=True)
manifest = Path("data/bacpipe_birdnet.jsonl")
bacpipe_loader_to_manifest({MODEL: loader}, MODEL, manifest, min_confidence=0.0)
print("manifest lines:", sum(1 for _ in manifest.open()), flush=True)
"""
    run([PYTHON, "-c", code], cwd=ROOT, timeout=5400)
    import os

    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("+", PYTHON, "-m bioacoustic_embedding_dynamics.cli ...", flush=True)
    subprocess.check_call(
        [
            PYTHON,
            "-m",
            "bioacoustic_embedding_dynamics.cli",
            "--manifest",
            "data/bacpipe_birdnet.jsonl",
            "--out",
            "reports/bacpipe",
            "--seed",
            "42",
        ],
        cwd=str(ROOT),
        env=env,
        timeout=1800,
    )
    summary = assert_report(ROOT / "reports/bacpipe", "bacpipe")
    if summary.get("synthesized_embeddings"):
        raise SystemExit("[bacpipe] expected real embeddings")
    if int(summary.get("embed_dim") or 0) < 256:
        raise SystemExit(f"[bacpipe] bad embed_dim: {summary.get('embed_dim')}")


def main() -> None:
    extract_project()
    verify_gpu()
    install_bacpipe()
    test_bacpipe()
    print("\n=== BACPIPE TEST PASSED ===", flush=True)
    out_tgz = Path("/content/bioacoustic-embedding-dynamics-bacpipe-reports.tgz")
    run(["tar", "czf", str(out_tgz), "-C", str(ROOT / "reports/bacpipe"), "."])
    print(f"Reports tarball: {out_tgz}", flush=True)


if __name__ == "__main__":
    main()
