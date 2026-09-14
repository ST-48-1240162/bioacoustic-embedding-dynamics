#!/usr/bin/env python3
"""Unpack the project tarball on a Colab VM and run the demo CLI."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/content/bioacoustic-embedding-dynamics")
TGZ = Path("/content/bioacoustic-embedding-dynamics.tgz")


def main() -> None:
    if TGZ.is_file():
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TGZ, "r:gz") as tf:
            tf.extractall(ROOT)
        print(f"Extracted to {ROOT}")
    elif not (ROOT / "pyproject.toml").is_file():
        raise SystemExit(f"Missing {TGZ} and {ROOT}")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "docs/colab-requirements.txt")]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)])

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for p in reports.iterdir():
        if p.name != ".gitkeep" and p.is_file():
            p.unlink()

    subprocess.check_call(
        [sys.executable, "-m", "bioacoustic_embedding_dynamics.cli", "--make-sample", "--out", str(reports)]
    )

    summary = reports / "summary.json"
    print(summary.read_text(encoding="utf-8"))

    out_tgz = Path("/content/bioacoustic-embedding-dynamics-reports.tgz")
    subprocess.check_call(["tar", "czf", str(out_tgz), "-C", str(reports), "."])
    print(f"Reports tarball: {out_tgz}")


if __name__ == "__main__":
    main()
