# MDE Transformers

Status: research reconstruction in progress.

This repository is being reconstructed as a reproducible research implementation for
monocular depth estimation using transformer-based hierarchical representations and
coarse-to-fine depth refinement. The current branch establishes the research-software
foundation; it does not yet contain the proposed depth architecture or reproduced benchmark
results.

## Development setup

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the local quality checks with:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Datasets, checkpoints, and generated experiment outputs are intentionally not stored in this
repository.

## NYU Depth V2

Download the official NYU Depth V2 labeled dataset and standard split metadata manually from the
[NYU Depth V2 project page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html). Keep both files
outside this repository, for example:

```text
/workspace/datasets/nyu_depth_v2/
├── nyu_depth_v2_labeled.mat
└── splits.mat
```

`NYUDepthV2` reads the standard labeled split of 795 training and 654 test images. It does not use
a separate raw-data evaluation protocol. The default `depths` source is the official in-painted
labeled depth in meters; `rawDepths` can be selected explicitly. Dataset validity is finite
positive depth from `validity_source="target"` by default. Set
`validity_source="rawDepths"` to retain filled target values while supervising only pixels observed
by the raw sensor. Benchmark range masking remains separate. No hidden subset is applied: use
`max_samples=N` explicitly for deterministic debugging.

Verify local files without downloading or modifying them:

```bash
python scripts/verify_nyu.py \
  --mat-path /workspace/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat \
  --split-path /workspace/datasets/nyu_depth_v2/splits.mat
```

## Roadmap

1. Define dataset protocols, geometry-aware preprocessing, depth validity rules, and standard
   metrics for KITTI and NYU Depth v2, followed later by UAVStereo.
2. Implement configurable hierarchical transformer encoders and a coarse-to-fine residual
   depth decoder with a stable multi-stage prediction interface.
3. Add reproducible training, validation, checkpointing, and metric-scale inference workflows.
4. Establish documented baselines and controlled experiments before investigating additional
   research ideas or edge-device deployment.
