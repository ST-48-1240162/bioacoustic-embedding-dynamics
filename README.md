NAME
    bioacoustic-embedding-dynamics - analyse high-dimensional bioacoustic embeddings

SYNOPSIS
    python -m bioacoustic_embedding_dynamics.cli [--make-sample] [--manifest FILE]
        [--out DIR] [--seed N]

DESCRIPTION
    Read a JSONL detection manifest (one object per line). Each row is a
    detection with start time, end time, species, confidence, site, and an
    optional embedding vector.

    The program then:

    1. loads or synthesizes embedding vectors
    2. runs PCA and UMAP
    3. bins a centroid trajectory in reduced space
    4. detects change-points on activity rate and on the trajectory
    5. fits a Gaussian HMM on the binned timeline

    Demo embeddings are a small PyTorch mapper (species prototype plus
    temporal drift). Production runs should pass real embeddings from
    BirdNET (bacpipe or bioacoustics-model-zoo).

OPTIONS
    --manifest FILE
        Path to detections.jsonl. If omitted, data/sample_detections.jsonl
        is used (created if missing).

    --out DIR
        Output directory (default: reports).

    --seed N
        Random seed (default: 42).

    --make-sample
        Write a demo manifest to data/sample_detections.jsonl.

    See python -m bioacoustic_embedding_dynamics.cli --help for the rest.

MANIFEST
    Required-ish keys (aliases accepted):

        start_s, end_s, species, confidence, site, embedding

    Alternative keys: scientific_name, score, window_start_s.

    If embedding is omitted, vectors are synthesized. Treat that as demo
    data, not field output.

COLAB
    docs/Demo_Colab.ipynb
        Synthetic manifest, CPU is enough.

    docs/Route_A_Bacpipe_Colab.ipynb
        Real 1024-d BirdNET embeddings via bacpipe (T4 GPU).

    docs/Route_B_BMZ_Colab.ipynb
        Real 1024-d BirdNET embeddings via bioacoustics-model-zoo.

    See docs/COLAB.md.

FILES
    reports/pca_species.png, reports/umap_species.png
        Embedding geometry by species.

    reports/trajectory_pca.png
        Binned centroid path in PC space.

    reports/changepoints.png
        Activity-rate change-points.

    reports/trajectory_changepoints.png
        Trajectory (PC1) change-points.

    reports/hmm_regimes.png
        HMM state over time.

    reports/embedding_trajectory.csv, reports/binned_with_hmm.csv
        Binned tables.

    reports/summary.json, reports/run_config.json
        Run metadata.

EXAMPLE
    python -m bioacoustic_embedding_dynamics.cli --make-sample --out reports
    python -m bioacoustic_embedding_dynamics.cli --manifest data/detections.jsonl --out reports --seed 42

LICENSE
    GPL-3.0-or-later
