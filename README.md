# MDE Transformers

MDE Transformers is a reproducible research codebase for supervised monocular metric-depth
estimation. It combines a hierarchical Swin Transformer encoder with a four-stage coarse-to-fine
residual depth decoder and provides typed data contracts, synchronized geometry, masked losses,
NYU Depth V2 loading, deterministic training, checkpoint/resume, and isolated final evaluation.

The current implementation establishes a controlled NYU supervised baseline. It does not claim
state-of-the-art performance or exact reproduction of historical thesis results.

## Implemented architecture

- torchvision Swin-Tiny and Swin-Base encoders with NCHW feature maps at strides 4, 8, 16, and 32;
- configurable 1x1 feature projections and two-convolution residual refinement stages;
- coarse-to-fine predictions `D1` through `D4`, followed by a separate full-resolution resize;
- positive physical-depth outputs using differentiable Softplus parameterization;
- explicit ImageNet RGB normalization outside the model;
- masked L1, Huber, BerHu, SILog training loss, regularization losses, and multi-scale supervision;
- deterministic NYU train/dev partitioning, AdamW, warmup-cosine scheduling, and strict resume;
- raw metric and standard NYU Eigen evaluation protocol representations.

Generic torchvision ImageNet initialization is supported. It is not SimCLR initialization and is
not presented as the thesis's target-domain contrastive pretraining.

## Repository structure

```text
src/mde_transformers/
├── data/       # samples, batching, geometry, transforms, NYU loader, splits
├── engine/     # training/evaluation steps, epochs, scheduler, checkpoints, manifests
├── losses/     # masked depth losses, regularizers, multi-scale supervision
├── metrics/    # depth metrics, crop masks, explicit NYU protocols
├── models/     # Swin adapter, residual decoder, complete depth model
└── utils/      # reproducibility utilities
scripts/
├── train_nyu.py
├── evaluate_nyu.py
├── overfit_nyu.py
└── verify_nyu.py
tests/          # CPU-only unit and integration tests; no dataset required
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The first run with `--encoder-pretrained` may retrieve official torchvision weights if they are not
already cached. Datasets and checkpoints are never downloaded by repository code.

## NYU Depth V2 setup

Obtain `nyu_depth_v2_labeled.mat` and `splits.mat` manually from the
[official NYU Depth V2 project](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and keep them
outside the repository. A portable shell setup is:

```bash
export NYU_ROOT="${HOME}/datasets/nyu_depth_v2"
export RUN_DIR="${HOME}/experiments/mde-transformers/nyu-raw-swin-t-baseline"
```

The loader uses the official labeled split of 795 training and 654 test samples. The default
`depths` target is dense filled metric depth in metres. `validity_source="target"` uses finite,
positive target pixels; `validity_source="rawDepths"` can instead restrict supervision to raw
sensor observations without replacing the selected target values.

Verify local files without modifying them:

```bash
python scripts/verify_nyu.py \
  --mat-path "${NYU_ROOT}/nyu_depth_v2_labeled.mat" \
  --split-path "${NYU_ROOT}/splits.mat"
```

## Evaluation protocols

Depth-range masking is applied by the evaluator, not the dataset loader. Crop coordinates are
defined in native 480x640 NYU coordinates and are never applied literally after resize.

| Protocol | Range | Alignment | Evaluation crop | Status |
| --- | --- | --- | --- | --- |
| Raw metric baseline | `[0.1, 10.0]` m | none | none | Primary supervised baseline |
| Standard NYU Eigen option | `[0.1, 10.0]` m | explicit | rows `[45, 471)`, columns `[41, 601)` | Literature-standard crop |
| Historical thesis protocol | `[0.1, 10.0]` m | median | fixed indoor crop | Exact pixel coordinates unresolved |

The thesis text available for reconstruction states that its NYU protocol uses median alignment
and a fixed indoor crop, including a fixed crop during training, but does not provide exact crop
coordinates. The implemented `nyu_eigen` option must therefore not be described as the thesis
crop. Crop and median alignment remain independent explicit choices.

## Reproduced supervised baseline

The baseline uses the official 795-sample training split only during training. A deterministic
seed-2025 partition supplies 636 optimization samples and 159 train-derived dev samples. The
reserved 654-sample official test split is evaluated only after checkpoint selection.

| Setting | Value |
| --- | --- |
| Encoder | Swin-Tiny, generic torchvision ImageNet initialization, fully trainable |
| Decoder channels | `(128, 128, 128, 128)` |
| Input resolution | `448 x 576` |
| Depth / validity source | `depths` / `target` |
| Training / evaluation crop | none / none |
| Evaluation alignment and range | none; `[0.1, 10.0]` m |
| Optimizer | AdamW, learning rate `2e-4`, weight decay `0.01` |
| Schedule | 1 warmup epoch, cosine decay, 10 total epochs |
| Batch size / seed | 1 / 2025 |
| Multi-scale stage weights | `(0.125, 0.25, 0.5, 1.0)` |
| Model selection | lowest train-derived dev loss |

### Training command

```bash
python scripts/train_nyu.py \
  --mat-path "${NYU_ROOT}/nyu_depth_v2_labeled.mat" \
  --split-path "${NYU_ROOT}/splits.mat" \
  --output-dir "${RUN_DIR}" \
  --encoder tiny \
  --encoder-pretrained \
  --encoder-policy trainable \
  --decoder-channels 128 \
  --minimum-depth 1e-6 \
  --height 448 \
  --width 576 \
  --validation-fraction 0.2 \
  --seed 2025 \
  --epochs 10 \
  --batch-size 1 \
  --num-workers 0 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --warmup-epochs 1 \
  --warmup-start-factor 0.1 \
  --stage-weights 0.125 0.25 0.5 1.0 \
  --device cuda \
  --depth-source depths \
  --validity-source target \
  --training-crop none \
  --evaluation-crop none \
  --alignment none
```

The run directory contains `run.json`, `last.pt`, and `best.pt`. Full checkpoints restore the
model, optimizer, scheduler, counters, best-dev tracking, and RNG state. Checkpoint reconstruction
never redownloads initialization weights.

### Final evaluation command

```bash
python scripts/evaluate_nyu.py \
  --mat-path "${NYU_ROOT}/nyu_depth_v2_labeled.mat" \
  --split-path "${NYU_ROOT}/splits.mat" \
  --checkpoint "${RUN_DIR}/best.pt" \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --alignment none \
  --crop none
```

Training never invokes the official-test evaluator. The evaluation command strictly loads the
selected checkpoint and does not update parameters or best-checkpoint state.

### Results

The best checkpoint was selected at epoch 8 using train-derived dev loss `1.002118`. Loss is the
weighted four-stage masked-L1 objective; metrics use the full-resolution prediction with equal
weight per image.

| Split | Samples | Loss | AbsRel | SqRel | RMSE | RMSElog | SILog | delta1 | delta2 | delta3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train-derived dev, epoch 8 | 159 | 1.002118 | 0.210025 | 0.196068 | 0.694541 | 0.245187 | 0.202033 | 0.670986 | 0.905154 | 0.976199 |
| Reserved official test | 654 | 0.968177 | 0.211928 | 0.183749 | 0.665873 | 0.245990 | 0.204930 | 0.669592 | 0.907496 | 0.974197 |

These are newly reproduced raw metric-scale baseline results from this reconstructed codebase.
They are not historical thesis results and must not be compared as if crop, alignment,
initialization, and training protocols were identical.

## Reproducibility and quality checks

Python, NumPy, CPU/CUDA seeds and optional deterministic algorithms are centralized in the
reproducibility utility. DataLoader train order is derived from the base seed and epoch so resumed
runs receive the same epoch permutation as uninterrupted runs.

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

The test suite is CPU-only and requires no NYU files, pretrained weights, network, or GPU.

## Current limitations and future work

- The implemented real-data pipeline currently targets NYU Depth V2; KITTI and UAVStereo loaders
  and protocols remain future work.
- SimCLR target-domain initialization has not been implemented. Generic ImageNet initialization
  must not be interpreted as SimCLR.
- The exact historical thesis indoor-crop coordinates remain unresolved.
- AMP, distributed training, uncertainty, adaptive exits, distillation, and edge deployment are
  outside the current supervised baseline.
- Software-release and publication rights are still under review; no license is currently granted.

Future experiments should preserve the explicit protocol boundary and establish controlled
ablations before adding contrastive pretraining or paper-specific methods.
