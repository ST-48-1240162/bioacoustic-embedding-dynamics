NAME
    Colab usage for bioacoustic-embedding-dynamics

SYNOPSIS
    colab new -s SESSION
    colab exec -s SESSION --timeout 3600 -f scripts/colab_run_fast_tests.py

NOTEBOOKS
    Demo_Colab.ipynb
        Demo manifest. CPU is enough (about 2-3 minutes).

    Route_A_Bacpipe_Colab.ipynb
        1024-d BirdNET via bacpipe. Use a T4 GPU.

    Route_B_BMZ_Colab.ipynb
        1024-d BirdNET via bioacoustics-model-zoo.

INSTALL (notebook)
    import sys
    !{sys.executable} -m pip install -q -r docs/colab-requirements.txt
    !{sys.executable} -m pip install -q -e .

    Route A: install bacpipe with --ignore-requires-python --no-deps.
    bacpipe requires Python < 3.13; Colab is 3.13. Reuse Colab TF and Torch.

RUN DEMO
    !python -m bioacoustic_embedding_dynamics.cli --make-sample --out reports --seed 42

OWN MANIFEST
    Upload detections.jsonl, then:

    MANIFEST = "/content/detections.jsonl"
    !python -m bioacoustic_embedding_dynamics.cli --manifest {MANIFEST} --out reports --seed 42

GPU
    colab new -s bel-gpu --gpu T4
    colab sessions
    colab stop -s bel-gpu

    Use T4 for Route A. Demo and Route B run on CPU.

TESTS
    Fast (demo + Route B), about 5-10 minutes:

        colab exec -s SESSION --timeout 3600 -f scripts/colab_run_fast_tests.py

    Route A on T4:

        colab new -s bel-gpu --gpu T4
        colab exec -s bel-gpu --timeout 7200 -f scripts/colab_run_bacpipe_test.py

    Skip bacpipe in the full suite:

        colab exec -s SESSION --timeout 7200 -f scripts/colab_run_all_tests.py -- --skip-bacpipe
