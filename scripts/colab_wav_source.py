"""Locate bacpipe bundled test wav on Colab (real bird audio, ~1 min)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable

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


def _missing_module(stderr: str) -> str | None:
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
    if m:
        return m.group(1)
    m = re.search(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]", stderr)
    if m:
        return m.group(2)
    return None


def install_bacpipe_for_test_wav() -> None:
    subprocess.check_call(
        [PYTHON, "-m", "pip", "install", "--ignore-requires-python", "--no-deps", "-q", "bacpipe"],
        timeout=300,
    )
    probe = r"""
import librosa
if not hasattr(librosa, "get_duration"):
    from librosa.core.audio import get_duration as _gd
    librosa.get_duration = _gd
import bacpipe
print("bacpipe import OK")
"""
    for _ in range(12):
        r = subprocess.run([PYTHON, "-c", probe], capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            return
        missing = _missing_module(r.stderr)
        if not missing or missing.split(".")[0] in SKIP_PIP:
            raise SystemExit(f"bacpipe import failed (missing={missing})")
        subprocess.check_call([PYTHON, "-m", "pip", "install", "-q", missing.split(".")[0]], timeout=600)
    raise SystemExit("bacpipe import still failing after dep loop")


def bacpipe_test_wav() -> Path:
    code = r"""
from pathlib import Path
import bacpipe
root = Path(bacpipe.__file__).parent / "tests" / "test_data"
wavs = sorted(root.rglob("*.wav"))
if not wavs:
    raise SystemExit("no bacpipe test wav")
print(wavs[0])
"""
    r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit(r.stderr or r.stdout)
    return Path(r.stdout.strip())
