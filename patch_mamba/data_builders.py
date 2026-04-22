"""Dataloader construction utilities for modular Patch-Mamba training."""

from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np
from torch.utils.data import DataLoader

from patch_dataset import (
    load_rollout_dataset_from_csv,
    load_saved_samples_from_csv,
    pack_samples_to_batch,
)

from .datasets import MultiViewPatchForecastTrainDataset, PatchForecastTrainDataset, split_dataset
from .runtime import estimate_runtime_patch_len_from_rollout_samples
from .utils import (
    build_default_prebuilt_csv_path,
    build_sample_key,
    build_track_group_keys_from_samples,
    estimate_max_patch_len,
)



def build_singleview_dataloaders_from_prebuilt(
    source_name=None,
    prebuilt_dir: str = "prebuilt_source_csv",
    prebuilt_csv_path: str | None = None,
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    pad_value: float = 0.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    device=None,
) -> Dict[str, object]:
    """Build single-view dataloaders from a prebuilt pseudo-recursive CSV.

    Returns
    -------
    dict
        Metadata bundle containing the raw samples, packed batch data, dataset
        object, dataloaders, sample counts, and automatically estimated patch
        length.
    """
    if prebuilt_csv_path is None:
        prebuilt_csv_path = build_default_prebuilt_csv_path(
            prebuilt_dir=prebuilt_dir,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            training_mode="pseudo_recursive",
        )

    if not os.path.exists(prebuilt_csv_path):
        raise FileNotFoundError(f"Prebuilt CSV not found: {prebuilt_csv_path}")

    samples = load_saved_samples_from_csv(prebuilt_csv_path)
    batch_data = pack_samples_to_batch(samples, pad_value=pad_value)
    auto_patch_len = max(estimate_max_patch_len(batch_data["patch_index"], input_patch_num=input_patch_num), 1)

    normal_cnt = sum(sample.get("sample_type", "") == "normal" for sample in samples)
    recursive_cnt = sum(sample.get("sample_type", "") == "recursive" for sample in samples)

    dataset = PatchForecastTrainDataset(batch_data)
    group_keys = build_track_group_keys_from_samples(samples)
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        group_keys=group_keys,
    )
    use_pin_memory = device is not None and str(device).startswith("cuda")

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)

    return {
        "samples": samples,
        "batch_data": batch_data,
        "dataset": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "normal_cnt": normal_cnt,
        "recursive_cnt": recursive_cnt,
        "auto_patch_len": auto_patch_len,
        "sample_count": len(samples),
        "source_name": source_name,
        "prebuilt_csv_path": prebuilt_csv_path,
    }



def build_multiview_dataloaders_from_prebuilt(
    source_name=None,
    prebuilt_dir: str = "prebuilt_source_csv",
    view_settings: Sequence[Dict[str, object]] | None = None,
    future_step_minutes: int = 5,
    pad_value: float = 0.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    device=None,
) -> Dict[str, object]:
    """Build aligned multi-view dataloaders from several prebuilt CSV branches."""
    if view_settings is None or len(view_settings) == 0:
        raise ValueError("view_settings must not be empty.")

    branch_batch_data = {}
    branch_auto_patch_len = {}
    branch_csv_paths = {}
    branch_key_to_sample = {}

    for view in view_settings:
        view_name = str(view["name"])
        input_patch_num = int(view["input_patch_num"])
        patch_minutes = int(view["patch_minutes"])
        prebuilt_csv_path = view.get("prebuilt_csv_path")
        if prebuilt_csv_path is None:
            prebuilt_csv_path = build_default_prebuilt_csv_path(
                prebuilt_dir=prebuilt_dir,
                source_name=source_name,
                input_patch_num=input_patch_num,
                patch_minutes=patch_minutes,
                future_step_minutes=future_step_minutes,
                training_mode="pseudo_recursive",
            )
        if not os.path.exists(prebuilt_csv_path):
            raise FileNotFoundError(f"Prebuilt CSV for multi-view branch {view_name!r} not found: {prebuilt_csv_path}")

        samples = load_saved_samples_from_csv(prebuilt_csv_path)
        key_to_sample = {build_sample_key(sample): sample for sample in samples}
        branch_key_to_sample[view_name] = key_to_sample
        branch_csv_paths[view_name] = prebuilt_csv_path

    common_keys = None
    for key_to_sample in branch_key_to_sample.values():
        key_set = set(key_to_sample.keys())
        common_keys = key_set if common_keys is None else (common_keys & key_set)
    common_keys = sorted(list(common_keys)) if common_keys is not None else []
    if len(common_keys) == 0:
        raise ValueError("No aligned common samples were found across the multi-view branches.")

    for view in view_settings:
        view_name = str(view["name"])
        aligned_samples = [branch_key_to_sample[view_name][key] for key in common_keys]
        batch_data = pack_samples_to_batch(aligned_samples, pad_value=pad_value)
        auto_patch_len = max(estimate_max_patch_len(batch_data["patch_index"], input_patch_num=int(view["input_patch_num"])), 1)
        branch_batch_data[view_name] = batch_data
        branch_auto_patch_len[view_name] = auto_patch_len

    base_name = str(view_settings[0]["name"])
    base_batch_data = branch_batch_data[base_name]
    for view_name, batch_data in branch_batch_data.items():
        if not np.allclose(batch_data["model_label"], base_batch_data["model_label"], atol=1e-6):
            raise ValueError(f"Branch {view_name!r} has model_label values inconsistent with the base branch.")
        if not np.allclose(batch_data["restore_info"], base_batch_data["restore_info"], atol=1e-6):
            raise ValueError(f"Branch {view_name!r} has restore_info values inconsistent with the base branch.")

    common_samples = [branch_key_to_sample[base_name][key] for key in common_keys]
    normal_cnt = sum(sample.get("sample_type", "") == "normal" for sample in common_samples)
    recursive_cnt = sum(sample.get("sample_type", "") == "recursive" for sample in common_samples)

    dataset = MultiViewPatchForecastTrainDataset(branch_batch_data, branch_auto_patch_len)
    group_keys = build_track_group_keys_from_samples(common_samples)
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        group_keys=group_keys,
    )
    use_pin_memory = device is not None and str(device).startswith("cuda")

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)

    return {
        "samples": common_samples,
        "dataset": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "normal_cnt": normal_cnt,
        "recursive_cnt": recursive_cnt,
        "sample_count": len(common_samples),
        "source_name": source_name,
        "branch_batch_data": branch_batch_data,
        "branch_auto_patch_len": branch_auto_patch_len,
        "branch_csv_paths": branch_csv_paths,
    }



def build_recursive_dataloaders_from_rollout_csv(
    rollout_csv_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    device=None,
) -> Dict[str, object]:
    """Build recursive rollout dataloaders from a rollout CSV file."""
    if rollout_csv_path is None or not os.path.exists(rollout_csv_path):
        raise FileNotFoundError(f"Rollout CSV not found: {rollout_csv_path}")

    samples, batch_data, dataset = load_rollout_dataset_from_csv(rollout_csv_path)
    group_keys = build_track_group_keys_from_samples(samples)
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        group_keys=group_keys,
    )
    use_pin_memory = device is not None and str(device).startswith("cuda")

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=use_pin_memory)

    return {
        "samples": samples,
        "batch_data": batch_data,
        "dataset": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "normal_cnt": len(samples),
        "recursive_cnt": len(samples),
        "sample_count": len(samples),
        "rollout_csv_path": rollout_csv_path,
    }



def estimate_recursive_patch_len_report(
    samples,
    input_patch_num: int,
    patch_minutes: int,
    future_step_minutes: int,
) -> int:
    """Thin wrapper that exposes runtime patch-length estimation to the API layer."""
    return max(
        estimate_runtime_patch_len_from_rollout_samples(
            samples=samples,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
        ),
        1,
    )
