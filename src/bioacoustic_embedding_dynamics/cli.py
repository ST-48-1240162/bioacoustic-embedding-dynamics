"""CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_analysis
from .sample_data import write_sample_manifest


def main() -> None:
    p = argparse.ArgumentParser(
        prog="bioacoustic-embedding-dynamics",
        description="Analyse high-dimensional bioacoustic embeddings (PCA, UMAP, trajectory, change-point, Gaussian HMM).",
    )
    p.add_argument("--manifest", type=Path, help="detections.jsonl (optional embedding vectors per row)")
    p.add_argument("--out", type=Path, default=Path("reports"), help="output directory for figures and CSV")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    p.add_argument("--embed-dim", type=int, default=128, help="embedding dimension (when synthesizing)")
    p.add_argument("--bin-s", type=float, default=60.0, help="timeline bin width (seconds)")
    p.add_argument("--changepoint-pen", type=float, default=3.0, help="ruptures PELT penalty")
    p.add_argument("--hmm-states", type=int, default=3, help="Gaussian HMM states")
    p.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--make-sample", action="store_true", help="write demo manifest to data/sample_detections.jsonl")
    args = p.parse_args()

    if args.make_sample:
        path = Path("data/sample_detections.jsonl")
        write_sample_manifest(path, seed=args.seed, embed_dim=args.embed_dim)
        print(f"Wrote {path}")
        if args.manifest is None:
            args.manifest = path

    if args.manifest is None:
        raise SystemExit("Pass --manifest path/to/detections.jsonl (or --make-sample for synthetic JSONL only).")
    manifest = args.manifest
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    summary = run_analysis(
        manifest,
        args.out,
        seed=args.seed,
        embed_dim=args.embed_dim,
        bin_s=args.bin_s,
        changepoint_pen=args.changepoint_pen,
        hmm_states=args.hmm_states,
        umap_neighbors=args.umap_neighbors,
    )
    print(summary)


if __name__ == "__main__":
    main()
