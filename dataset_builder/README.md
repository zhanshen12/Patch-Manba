# Patch Dataset Dual-Mode Module

## Overview

`patch_dataset_dual_mode` is a standalone dataset-building module for patch-based maritime trajectory forecasting. It converts raw CSV trajectory cells into model-ready datasets while preserving the original raw-first processing logic of the source script.

This module supports two dataset construction modes:

- **Pseudo-recursive mode**: builds flat single-step samples for the original training pipeline.
- **Recursive rollout mode**: builds sequence-style rollout samples for true recursive training or evaluation.

The refactor only reorganizes the original script into a clear module structure. It does not introduce new modeling logic, new data assumptions, or extra training features.

## Module Scope

This module is responsible for:

- reading raw trajectory cells from a CSV file,
- parsing and sorting point sequences,
- generating fixed-step future points in raw geographic space,
- encoding points into the 10-dimensional feature representation,
- building flat or rollout datasets,
- packing variable-length samples into padded batch arrays,
- saving and reloading prebuilt datasets as CSV files,
- exporting multi-window datasets for a single source.

This module is **not** responsible for model definition, loss computation, training loops, or plotting.

## Directory Structure

```text
dataset_builder/
├── __init__.py
├── __main__.py
├── batching.py
├── builders.py
├── constants.py
├── datasets.py
├── export.py
├── io_utils.py
├── restore.py
├── serialization.py
├── trajectory.py
└── utils.py
```

## Public Entry Points

The main public entry points are:

- `build_patch_forecast_dataset_from_raw_tracks(...)`
- `build_patch_forecast_dataset_from_csv_single_source(...)`
- `build_patch_forecast_dataset_from_raw_tracks_pseudo(...)`
- `build_patch_rollout_dataset_from_raw_tracks(...)`
- `build_and_save_source_multiscale(...)`

## Data Flow

1. Read a source column from CSV.
2. Parse each trajectory cell into a sorted raw trajectory array.
3. Compute per-track restore information.
4. Generate future target points in raw longitude/latitude space.
5. Slice input windows into temporal patches.
6. Encode valid points into the fixed 10-dimensional feature format.
7. Build either flat samples or rollout samples.
8. Pack samples into padded batch structures or export them to CSV.

## Usage Example

```python
from patch_dataset_dual_mode import build_patch_forecast_dataset_from_csv_single_source

samples, batch_data, dataset = build_patch_forecast_dataset_from_csv_single_source(
    csv_path="data.csv",
    source_name="AIS",
    input_patch_num=12,
    patch_minutes=15,
    strict=False,
    pad_value=0.0,
    future_step_minutes=5,
    sample_stride_minutes=5,
    min_total_input_points=1,
    max_future_steps=12,
    training_mode="pseudo_recursive",
)
```

## Design Notes

- The recursive and pseudo-recursive builders remain separate internally, but a unified interface is provided.
- Serialization logic is isolated from dataset construction logic.
- PyTorch dataset wrappers are separated from NumPy batch packing utilities.
- The package can be executed directly with `python -m patch_dataset_dual_mode` to reproduce the original default export behavior.
