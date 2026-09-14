# Colab

CPU is enough for the demo (~2-3 min) and for Route B. Use a T4 for Route A (bacpipe).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ST-48-1240162/bioacoustic-embedding-dynamics/blob/main/docs/Demo_Colab.ipynb)

| Notebook | When |
|----------|------|
| [Demo_Colab.ipynb](./Demo_Colab.ipynb) | Demo manifest, or you already have JSONL |
| [Route_A_Bacpipe_Colab.ipynb](./Route_A_Bacpipe_Colab.ipynb) | 1024-d BirdNET via bacpipe |
| [Route_B_BMZ_Colab.ipynb](./Route_B_BMZ_Colab.ipynb) | 1024-d BirdNET via bioacoustics-model-zoo |

## Install (notebook)

```python
import sys
!{sys.executable} -m pip install -q -r docs/colab-requirements.txt
!{sys.executable} -m pip install -q -e .
```

Route A: bacpipe requires Python < 3.13. Colab is 3.13, so:

```python
!{sys.executable} -m pip install --ignore-requires-python --no-deps bacpipe
```

Reuse Colab's TensorFlow and Torch. Do not let pip pull bacpipe's full pin set.

## Demo

```python
!python -m bioacoustic_embedding_dynamics.cli --make-sample --out reports --seed 42
```

## Your own manifest

Upload `detections.jsonl`, then:

```python
MANIFEST = "/content/detections.jsonl"
!python -m bioacoustic_embedding_dynamics.cli --manifest {MANIFEST} --out reports --seed 42
```

## GPU (T4) via CLI

```bash
colab new -s bel-gpu --gpu T4
colab sessions
colab stop -s bel-gpu
```

## Automated tests

```bash
# demo + Route B, about 5-10 min
colab exec -s SESSION --timeout 3600 -f scripts/colab_run_fast_tests.py

# Route A on T4
colab new -s bel-gpu --gpu T4
colab exec -s bel-gpu --timeout 7200 -f scripts/colab_run_bacpipe_test.py

# skip bacpipe in the full suite
colab exec -s SESSION --timeout 7200 -f scripts/colab_run_all_tests.py -- --skip-bacpipe
```
