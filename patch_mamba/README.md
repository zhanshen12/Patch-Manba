# Patch-Mamba Training Module

The **Patch-Mamba Training Module** is a modular training framework for patch-based maritime trajectory forecasting.

This repository reorganizes a monolithic training script into clear functional units for open-source release. The codebase is intended to be readable by external users, reproducible for reviewers, and easier to maintain for future research iterations.

## Scope

The module supports two training paradigms:

1. **Pseudo-recursive training**
   - The model is trained on prebuilt single-step samples.
   - Input samples may already contain pseudo-recursive feedback points generated during preprocessing.
   - The training loop itself remains single-step.

2. **Recursive rollout training**
   - The model is trained over multi-step rollout trajectories.
   - Runtime inputs are rebuilt step by step.
   - Predicted points are fed back into the next step during training and evaluation.

The module also supports two model variants:

1. **Single-view Patch-Mamba**
   - One temporal patch layout per sample.

2. **Multi-window hybrid Patch-Mamba**
   - Several temporal layouts are encoded independently.
   - Branch features are fused by a learned gating mechanism.

## Package layout

```text
patch_mamba/
├── README.md
├── patch_mamba.py
├── run_train_single.py
├── run_train_recursive.py
├── run_multiscale_compare.py
├── pyproject.toml
└── patch_mamba_training/
    ├── __init__.py
    ├── utils.py
    ├── metrics.py
    ├── dataset_io.py
    ├── datasets.py
    ├── runtime_batches.py
    ├── modules.py
    ├── encoders.py
    ├── models.py
    ├── config_builders.py
    ├── train.py
    ├── presets.py
    └── experiments.py
```

## Public entrypoints

The primary public training entrypoint is:

```python
from patch_mamba_training import train_patch_mamba_model
```

For preset-based multi-run experiments:

```python
from patch_mamba_training import run_multiscale_comparison_experiment
```

A compatibility wrapper named `patch_mamba.py` is also provided so that external scripts can continue importing the training interface using the historical filename.

## Dependency on the dataset builder

This training module expects a companion dataset builder module that provides the following functions and constants:

- `EPS`
- `restore_pred_lonlat`
- `pack_samples_to_batch`
- `load_saved_samples_from_csv`
- `load_rollout_dataset_from_csv`
- `build_output_csv_path`

By default, the code tries to import them from `patch_dataset`.

