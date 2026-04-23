"""
Dataset loading utilities for prebuilt pseudo-recursive and rollout CSV files.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .utils import normalize_source_name

try:
    from patch_dataset import (
        pack_samples_to_batch,
        load_saved_samples_from_csv,
        load_rollout_dataset_from_csv,
        build_output_csv_path,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "The training module requires the companion dataset builder module "
        "`patch_dataset.py` with prebuilt-data loading helpers."
    ) from exc


def build_default_prebuilt_csv_path(
    prebuilt_dir: str = "prebuilt_source_csv",
    source_name: str = "AIS",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    training_mode: str = "pseudo_recursive",
) -> str:
    """
    Build the expected prebuilt CSV path from a standard naming convention.
    """
    return build_output_csv_path(
        output_dir=prebuilt_dir,
        source_name=source_name,
        input_patch_num=input_patch_num,
        patch_minutes=patch_minutes,
        future_step_minutes=future_step_minutes,
        training_mode=training_mode,
    )


def estimate_max_patch_len(patch_index_np: np.ndarray, input_patch_num: int) -> int:
    """
    Estimate the largest number of points found inside any patch slot.
    """
    if patch_index_np.size == 0:
        return 1
    max_len = 1
    for row in patch_index_np:
        for patch_id in range(1, input_patch_num + 1):
            count = int(np.sum(row == patch_id))
            if count > max_len:
                max_len = count
    return max_len


def build_sample_key(sample: Dict[str, Any]) -> tuple:
    """
    Build a stable alignment key for multi-window sample matching.
    """
    return (
        normalize_source_name(sample["source_name"]),
        int(sample["track_id"]),
        str(sample["sample_type"]),
        int(sample["recursive_step"]),
        round(float(sample["window_start_ts"]), 6),
        round(float(sample["future_time_ts"]), 6),
    )


def build_runtime_shape_stub_batch_data(input_patch_num: int, pred_dim: int = 5, in_dim: int = 6) -> Dict[str, np.ndarray]:
    """
    Build a minimal shape-only batch-data dictionary for runtime model config.
    """
    input_patch_num = int(input_patch_num)
    pred_dim = int(pred_dim)
    in_dim = int(in_dim)
    return {
        "model_input": np.zeros((1, 1, in_dim), dtype=np.float32),
        "patch_mask": np.zeros((1, input_patch_num), dtype=np.float32),
        "model_label": np.zeros((1, pred_dim), dtype=np.float32),
    }


def require_existing_file(path: str, label: str) -> str:
    """
    Ensure that an expected input file exists.
    """
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


__all__ = [
    "pack_samples_to_batch",
    "load_saved_samples_from_csv",
    "load_rollout_dataset_from_csv",
    "build_output_csv_path",
    "build_default_prebuilt_csv_path",
    "estimate_max_patch_len",
    "build_sample_key",
    "build_runtime_shape_stub_batch_data",
    "require_existing_file",
]
