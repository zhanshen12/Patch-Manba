# Patch-Mamba Training Module

## Overview

This module provides a modular implementation of a patch-based maritime trajectory forecasting pipeline built on top of TTCN-style patch aggregation, time-aware patch attention, and Mamba sequence modeling.

The module supports two training modes:

- **Pseudo-recursive training** for single-step supervision on prebuilt samples.
- **Recursive training** for true rollout-based supervision on sequential forecast targets.

It also supports two model variants:

- **Single-view Patch-Mamba**, which uses one temporal patch configuration.
- **Multi-window hybrid Patch-Mamba**, which fuses several temporal views into one prediction head.

## Directory Structure

```text
patch_mamba/
  ├── __init__.py
  ├── api.py
  ├── config_builders.py
  ├── data_builders.py
  ├── datasets.py
  ├── layers.py
  ├── models.py
  ├── presets.py
  ├── runtime.py
  ├── training.py
  └── utils.py
```

## Module Responsibilities

### `utils.py`
General-purpose helpers for reproducibility, serialization, coordinate restoration, metric computation, path construction, and sample-key generation.

### `datasets.py`
Dataset classes and dataset-splitting logic for single-view training, multi-view training, and recursive rollout training.

### `layers.py`
Low-level neural network blocks, including sinusoidal positional encoding, time-biased patch attention, and the residual Mamba block.

### `models.py`
Encoder and prediction models, including the single-view Patch-Mamba model and the multi-window fusion model.

### `runtime.py`
Runtime utilities for recursive rollout. This module rebuilds model inputs step by step from observed points and generated predictions.

### `training.py`
Training and evaluation loops for pseudo-recursive and recursive modes.

### `data_builders.py`
DataLoader builders for prebuilt pseudo-recursive CSV files and rollout CSV files.

### `config_builders.py`
Configuration builders for single-view and multi-window models.

### `api.py`
Main public training interface. The core entrypoint is `train_patch_mamba_model(...)`.

### `presets.py`
Preset experiment definitions, comparison plotting, and multi-run experiment orchestration.

### `patch_mamba.py`
Compatibility wrapper that preserves the original import entrypoint while redirecting to the modular package.

## Main Entry Points

### Train a model

```python
from patch_mamba_modular import train_patch_mamba_model
```

### Run a preset comparison experiment

```python
from patch_mamba_modular import run_multiscale_comparison_experiment
```

## Design Goal

The purpose of this modularization is to keep the original training logic and public interface intact while making the codebase easier to read, maintain, reuse, and document in an open-source setting.
