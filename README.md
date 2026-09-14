# bioacoustic-embedding-dynamics

Takes a JSONL of bioacoustic detections (optional embedding vectors) and runs PCA, UMAP, a binned centroid trajectory, change-point detection, and a Gaussian HMM.

If a row has no `embedding`, a small PyTorch mapper synthesizes one for the demo. Real runs should pass BirdNET vectors from bacpipe (Route A) or bioacoustics-model-zoo (Route B).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Demo_Colab.ipynb)

**Notebook (demo):** [docs/Demo_Colab.ipynb](docs/Demo_Colab.ipynb)
**Route A (bacpipe, T4):** [docs/Route_A_Bacpipe_Colab.ipynb](docs/Route_A_Bacpipe_Colab.ipynb)
**Route B (BMZ):** [docs/Route_B_BMZ_Colab.ipynb](docs/Route_B_BMZ_Colab.ipynb)
**Walkthrough:** [docs/COLAB.md](docs/COLAB.md)

## What runs where

| Step | Where |
|------|-------|
| Demo manifest (synthetic 128-d) | Colab CPU, about 2-3 min |
| Route B: BMZ BirdNET 1024-d | Colab CPU, about 5-10 min |
| Route A: bacpipe BirdNET 1024-d | Colab T4, first run downloads weights |

## Install

Colab: run the notebook install cell. Pins are in [`docs/colab-requirements.txt`](docs/colab-requirements.txt).

```sh
python -m pip install -e .
```

Route A also needs bacpipe. On Colab (Python 3.13) install it with `--ignore-requires-python --no-deps` and keep Colab's TensorFlow / Torch. See the Route A notebook.

## Run

```sh
python -m bioacoustic_embedding_dynamics.cli --make-sample --out reports
python -m bioacoustic_embedding_dynamics.cli --manifest data/detections.jsonl --out reports --seed 42
```

`python -m bioacoustic_embedding_dynamics.cli --help` lists the rest (`--bin-s`, `--hmm-states`, ...).

## Manifest

One JSON object per line:

```json
{
  "start_s": 120.5,
  "end_s": 123.5,
  "species": "Pseudopipra pipra",
  "confidence": 0.91,
  "site": "costa-rica-site-a",
  "embedding": [0.012, -0.034]
}
```

Aliases: `scientific_name`, `score`, `window_start_s`. If `embedding` is missing, vectors are synthesized. Treat that as demo data, not field output.

## Outputs (`reports/`)

| File | What it is |
|------|------------|
| `pca_species.png`, `umap_species.png` | Embedding geometry by species |
| `trajectory_pca.png` | Binned centroid path, colored by embedding HMM state |
| `changepoints.png` | Activity-rate change-points |
| `trajectory_changepoints.png` | Trajectory (PC1) change-points |
| `hmm_regimes.png` | Embedding HMM vs activity HMM |
| `shuffle_null.png` | Time-shuffle control (PC1 change-points and embedding HMM) |
| `embedding_trajectory.csv`, `binned_with_hmm.csv` | Binned tables (confidence-weighted centroids) |
| `summary.json`, `run_config.json` | Run metadata |

## License

GPL-3.0-or-later
