"""
Training interfaces for Patch-Mamba models.

This module contains:
- prebuilt-data loader builders,
- pseudo-recursive and recursive train/eval loops,
- the main public entrypoint `train_patch_mamba_model`.

The goal is to preserve the original training behavior while presenting it
in a clearer, externally readable form for reviewers and open-source users.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config_builders import make_model_cfg, make_multiwindow_model_cfg
from .dataset_io import (
    build_default_prebuilt_csv_path,
    build_runtime_shape_stub_batch_data,
    estimate_max_patch_len,
    build_sample_key,
    load_rollout_dataset_from_csv,
    load_saved_samples_from_csv,
    pack_samples_to_batch,
    require_existing_file,
)
from .datasets import (
    MultiViewPatchForecastTrainDataset,
    PatchForecastTrainDataset,
    RolloutTrainDataset,
    build_track_group_keys_from_samples,
    split_dataset,
)
from .metrics import trajectory_metrics
from .models import PatchTTCN_Mamba_TrajPred, PatchTTCN_MultiWindowFusion_TrajPred
from .runtime_batches import estimate_runtime_patch_len_from_rollout_samples, rollout_forward
from .utils import dumps_json_pretty, ensure_output_dir, move_batch_to_device, set_seed, to_serializable


def build_singleview_dataloaders_from_prebuilt(
    source_name: str | None = None,
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
) -> dict:
    """
    Build dataloaders for single-view pseudo-recursive training.

    This function loads one prebuilt pseudo-recursive CSV, reconstructs
    batch tensors, estimates patch length, performs group-aware splitting,
    and returns ready-to-use dataloaders plus summary metadata.
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

    require_existing_file(prebuilt_csv_path, "prebuilt pseudo-recursive CSV")
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
    source_name: str | None = None,
    prebuilt_dir: str = "prebuilt_source_csv",
    view_settings: Sequence[dict] | None = None,
    future_step_minutes: int = 5,
    pad_value: float = 0.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    device=None,
) -> dict:
    """
    Build dataloaders for aligned multi-window pseudo-recursive training.

    Samples are loaded independently for each window configuration and then
    aligned using a stable sample key. Only samples that exist in every
    requested branch are kept.
    """
    if view_settings is None or len(view_settings) == 0:
        raise ValueError("`view_settings` must not be empty.")

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
        require_existing_file(prebuilt_csv_path, f"prebuilt CSV for branch `{view_name}`")

        samples = load_saved_samples_from_csv(prebuilt_csv_path)
        branch_key_to_sample[view_name] = {build_sample_key(sample): sample for sample in samples}
        branch_csv_paths[view_name] = prebuilt_csv_path

    common_keys = None
    for _, key_to_sample in branch_key_to_sample.items():
        key_set = set(key_to_sample.keys())
        common_keys = key_set if common_keys is None else (common_keys & key_set)

    common_keys = sorted(list(common_keys)) if common_keys is not None else []
    if len(common_keys) == 0:
        raise ValueError("No aligned multi-window samples were found across the requested branches.")

    for view in view_settings:
        view_name = str(view["name"])
        aligned_samples = [branch_key_to_sample[view_name][key] for key in common_keys]
        batch_data = pack_samples_to_batch(aligned_samples, pad_value=pad_value)
        auto_patch_len = max(
            estimate_max_patch_len(batch_data["patch_index"], input_patch_num=int(view["input_patch_num"])),
            1,
        )
        branch_batch_data[view_name] = batch_data
        branch_auto_patch_len[view_name] = auto_patch_len

    base_name = str(view_settings[0]["name"])
    base_batch_data = branch_batch_data[base_name]
    for view_name, batch_data in branch_batch_data.items():
        if not np.allclose(batch_data["model_label"], base_batch_data["model_label"], atol=1e-6):
            raise ValueError(f"Branch `{view_name}` does not match the reference branch labels.")
        if not np.allclose(batch_data["restore_info"], base_batch_data["restore_info"], atol=1e-6):
            raise ValueError(f"Branch `{view_name}` does not match the reference branch restore_info values.")

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
) -> dict:
    """
    Build dataloaders for recursive rollout training.

    The input rollout CSV is expected to be produced by the companion
    dataset builder. Group-aware splitting is still applied at track level.
    """
    require_existing_file(rollout_csv_path, "rollout CSV")
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


def train_one_epoch_pseudo(model, loader, optimizer, device, grad_clip: float = 1.0) -> float:
    """
    Train one epoch in pseudo-recursive mode.
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        loss, _ = model.compute_loss(batch)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch["model_label"].size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def train_one_epoch_recursive(
    model,
    loader,
    optimizer,
    device,
    grad_clip: float = 1.0,
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings: Sequence[dict] | None = None,
    pad_value: float = 0.0,
) -> float:
    """
    Train one epoch in recursive rollout mode.
    """
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        loss, _ = rollout_forward(
            model=model,
            batch=batch,
            model_variant=model_variant,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            multiwindow_view_settings=multiwindow_view_settings,
            pad_value=pad_value,
        )
        if loss is None:
            continue

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch["future_model_labels"].size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip: float = 1.0,
    training_mode: str = "pseudo_recursive",
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings: Sequence[dict] | None = None,
    pad_value: float = 0.0,
) -> float:
    """
    Unified epoch-level training dispatcher.
    """
    if training_mode == "pseudo_recursive":
        return train_one_epoch_pseudo(model, loader, optimizer, device, grad_clip=grad_clip)
    if training_mode == "recursive":
        return train_one_epoch_recursive(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            model_variant=model_variant,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            multiwindow_view_settings=multiwindow_view_settings,
            pad_value=pad_value,
        )
    raise ValueError(f"Unsupported training_mode={training_mode}")


@torch.no_grad()
def evaluate_pseudo(model, loader, device) -> dict:
    """
    Evaluate a model in pseudo-recursive mode.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_batches = []
    tgt_batches = []
    restore_batches = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss, pred = model.compute_loss(batch)

        batch_size = batch["model_label"].size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        pred_batches.append(pred.detach().cpu())
        tgt_batches.append(batch["model_label"].unsqueeze(1).detach().cpu())
        restore_batches.append(batch["restore_info"].detach().cpu())

    avg_loss = total_loss / max(total_count, 1)
    pred_all = torch.cat(pred_batches, dim=0) if pred_batches else torch.empty((0, 1, 5))
    tgt_all = torch.cat(tgt_batches, dim=0) if tgt_batches else torch.empty((0, 1, 5))
    restore_all = torch.cat(restore_batches, dim=0) if restore_batches else torch.empty((0, 4))

    metrics = trajectory_metrics(pred_all, tgt_all, restore_all) if len(pred_all) > 0 else {"mse": None, "fde": None, "dtw": None}
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate_recursive(
    model,
    loader,
    device,
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings: Sequence[dict] | None = None,
    pad_value: float = 0.0,
) -> dict:
    """
    Evaluate a model in recursive rollout mode.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_batches = []
    tgt_batches = []
    restore_batches = []
    mask_batches = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss, pred_steps = rollout_forward(
            model=model,
            batch=batch,
            model_variant=model_variant,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            multiwindow_view_settings=multiwindow_view_settings,
            pad_value=pad_value,
        )

        batch_size = batch["future_model_labels"].size(0)
        if loss is not None:
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        pred_batches.append(pred_steps.detach().cpu())
        tgt_batches.append(batch["future_model_labels"].detach().cpu())
        restore_batches.append(batch["restore_info"].detach().cpu())
        mask_batches.append(batch["rollout_mask"].detach().cpu())

    avg_loss = total_loss / max(total_count, 1)
    if not pred_batches:
        return {"loss": avg_loss, "mse": None, "fde": None, "dtw": None}

    pred_all = torch.cat(pred_batches, dim=0)
    tgt_all = torch.cat(tgt_batches, dim=0)
    restore_all = torch.cat(restore_batches, dim=0)
    mask_all = torch.cat(mask_batches, dim=0)
    metrics = trajectory_metrics(pred_all, tgt_all, restore_all, valid_mask=mask_all)
    metrics["loss"] = avg_loss
    return metrics


def evaluate(
    model,
    loader,
    device,
    training_mode: str = "pseudo_recursive",
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings: Sequence[dict] | None = None,
    pad_value: float = 0.0,
) -> dict:
    """
    Unified evaluation dispatcher.
    """
    if training_mode == "pseudo_recursive":
        return evaluate_pseudo(model, loader, device)
    if training_mode == "recursive":
        return evaluate_recursive(
            model=model,
            loader=loader,
            device=device,
            model_variant=model_variant,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            multiwindow_view_settings=multiwindow_view_settings,
            pad_value=pad_value,
        )
    raise ValueError(f"Unsupported training_mode={training_mode}")


def train_patch_mamba_model(
    csv_path: str = "data2.csv",
    columns: tuple = ("bd", "radar", "AIS"),
    source_name: str | None = None,
    save_dir: str = "patch_manba_output",
    log_filename: str = "train_log.txt",
    model_filename: str = "best_patch_ttcn_mamba_trajpred.pt",
    csv_result_filename: str = "final_result.csv",
    json_result_filename: str = "final_result.json",
    seed: int = 42,
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    pad_value: float = 0.0,
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_tmax: int | None = None,
    grad_clip: float = 1.0,
    hid_dim: int = 128,
    mamba_layers: int = 2,
    gnn_layers: int = 2,
    nhead: int = 4,
    tau_seconds: float = 300.0,
    model_variant: str = "single",
    size_name: str | None = None,
    multiwindow_view_settings: Sequence[dict] | None = None,
    branch_proj_dim: int = 128,
    fusion_hidden: int | None = None,
    prebuilt_dir: str = "prebuilt_source_csv",
    prebuilt_csv_path: str | None = None,
    training_mode: str = "pseudo_recursive",
    rollout_csv_path: str | None = None,
    **model_kwargs,
) -> dict:
    """
    Main public training entrypoint.

    This function preserves the original high-level interface while exposing
    the training logic in a more modular and review-friendly form.
    """
    set_seed(seed)
    save_dir = ensure_output_dir(save_dir)
    log_path = os.path.join(save_dir, log_filename)
    best_model_path = os.path.join(save_dir, model_filename)
    csv_result_path = os.path.join(save_dir, csv_result_filename)
    json_result_path = os.path.join(save_dir, json_result_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if training_mode == "pseudo_recursive":
        if model_variant == "single":
            data_info = build_singleview_dataloaders_from_prebuilt(
                source_name=source_name,
                prebuilt_dir=prebuilt_dir,
                prebuilt_csv_path=prebuilt_csv_path,
                input_patch_num=input_patch_num,
                patch_minutes=patch_minutes,
                future_step_minutes=future_step_minutes,
                pad_value=pad_value,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                device=device,
            )
            batch_data = data_info["batch_data"]
            single_model_kwargs = dict(model_kwargs)
            for key in ["gnn_layers", "nhead", "tau_seconds", "delta_minutes", "head_hidden", "mamba_d_state", "mamba_d_conv", "mamba_expand", "dropout"]:
                single_model_kwargs.pop(key, None)

            cfg = make_model_cfg(
                batch_data=batch_data,
                auto_patch_len=data_info["auto_patch_len"],
                device=device,
                hid_dim=hid_dim,
                mamba_layers=mamba_layers,
                gnn_layers=gnn_layers,
                nhead=nhead,
                tau_seconds=tau_seconds,
                delta_minutes=float(model_kwargs.get("delta_minutes", patch_minutes)),
                **single_model_kwargs,
            )
            model = PatchTTCN_Mamba_TrajPred(cfg).to(device)
            auto_patch_len_report = int(data_info["auto_patch_len"])
            branch_auto_patch_len_report = None
            prebuilt_info_report = {"single": data_info["prebuilt_csv_path"]}

        elif model_variant == "multiwindow_hybrid":
            if multiwindow_view_settings is None or len(multiwindow_view_settings) == 0:
                raise ValueError("`multiwindow_view_settings` must not be empty for multi-window training.")

            data_info = build_multiview_dataloaders_from_prebuilt(
                source_name=source_name,
                prebuilt_dir=prebuilt_dir,
                view_settings=multiwindow_view_settings,
                future_step_minutes=future_step_minutes,
                pad_value=pad_value,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                device=device,
            )

            multiwindow_model_kwargs = dict(model_kwargs)
            for key in ["branch_proj_dim", "fusion_hidden", "dropout", "tau_seconds", "gnn_layers", "nhead", "mamba_layers", "hid_dim", "mamba_d_state", "mamba_d_conv", "mamba_expand"]:
                multiwindow_model_kwargs.pop(key, None)

            cfg = make_multiwindow_model_cfg(
                branch_batch_data=data_info["branch_batch_data"],
                device=device,
                view_settings=multiwindow_view_settings,
                branch_auto_patch_len=data_info["branch_auto_patch_len"],
                branch_proj_dim=branch_proj_dim,
                fusion_hidden=(fusion_hidden if fusion_hidden is not None else max(256, branch_proj_dim * len(multiwindow_view_settings))),
                dropout=float(model_kwargs.get("dropout", 0.1)),
                tau_seconds=tau_seconds,
                gnn_layers=gnn_layers,
                nhead=nhead,
                mamba_layers=mamba_layers,
                hid_dim=hid_dim,
                **multiwindow_model_kwargs,
            )
            model = PatchTTCN_MultiWindowFusion_TrajPred(cfg).to(device)
            auto_patch_len_report = None
            branch_auto_patch_len_report = copy.deepcopy(data_info["branch_auto_patch_len"])
            prebuilt_info_report = copy.deepcopy(data_info["branch_csv_paths"])
        else:
            raise ValueError(f"Unsupported model_variant={model_variant}")

    elif training_mode == "recursive":
        data_info = build_recursive_dataloaders_from_rollout_csv(
            rollout_csv_path=rollout_csv_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            device=device,
        )

        if model_variant == "single":
            rollout_patch_len = max(
                estimate_runtime_patch_len_from_rollout_samples(
                    samples=data_info["samples"],
                    input_patch_num=input_patch_num,
                    patch_minutes=patch_minutes,
                    future_step_minutes=future_step_minutes,
                ),
                1,
            )
            pred_dim = int(data_info["batch_data"]["future_model_labels"].shape[-1])
            cfg_batch_stub = build_runtime_shape_stub_batch_data(
                input_patch_num=input_patch_num,
                pred_dim=pred_dim,
                in_dim=6,
            )
            single_model_kwargs = dict(model_kwargs)
            for key in ["gnn_layers", "nhead", "tau_seconds", "delta_minutes", "head_hidden", "mamba_d_state", "mamba_d_conv", "mamba_expand", "dropout"]:
                single_model_kwargs.pop(key, None)

            cfg = make_model_cfg(
                batch_data=cfg_batch_stub,
                auto_patch_len=rollout_patch_len,
                device=device,
                hid_dim=hid_dim,
                mamba_layers=mamba_layers,
                gnn_layers=gnn_layers,
                nhead=nhead,
                tau_seconds=tau_seconds,
                delta_minutes=float(model_kwargs.get("delta_minutes", patch_minutes)),
                **single_model_kwargs,
            )
            model = PatchTTCN_Mamba_TrajPred(cfg).to(device)
            auto_patch_len_report = int(rollout_patch_len)
            branch_auto_patch_len_report = None
            prebuilt_info_report = {"rollout": rollout_csv_path}

        elif model_variant == "multiwindow_hybrid":
            if multiwindow_view_settings is None or len(multiwindow_view_settings) == 0:
                raise ValueError("`multiwindow_view_settings` must not be empty for multi-window training.")

            branch_batch_data = {}
            branch_auto_patch_len = {}
            pred_dim = int(data_info["batch_data"]["future_model_labels"].shape[-1])

            for view in multiwindow_view_settings:
                name = str(view["name"])
                branch_batch_data[name] = build_runtime_shape_stub_batch_data(
                    input_patch_num=int(view["input_patch_num"]),
                    pred_dim=pred_dim,
                    in_dim=6,
                )
                rollout_patch_len = max(
                    estimate_runtime_patch_len_from_rollout_samples(
                        samples=data_info["samples"],
                        input_patch_num=int(view["input_patch_num"]),
                        patch_minutes=int(view["patch_minutes"]),
                        future_step_minutes=future_step_minutes,
                    ),
                    1,
                )
                branch_auto_patch_len[name] = rollout_patch_len

            multiwindow_model_kwargs = dict(model_kwargs)
            for key in ["branch_proj_dim", "fusion_hidden", "dropout", "tau_seconds", "gnn_layers", "nhead", "mamba_layers", "hid_dim", "mamba_d_state", "mamba_d_conv", "mamba_expand"]:
                multiwindow_model_kwargs.pop(key, None)

            cfg = make_multiwindow_model_cfg(
                branch_batch_data=branch_batch_data,
                device=device,
                view_settings=multiwindow_view_settings,
                branch_auto_patch_len=branch_auto_patch_len,
                branch_proj_dim=branch_proj_dim,
                fusion_hidden=(fusion_hidden if fusion_hidden is not None else max(256, branch_proj_dim * len(multiwindow_view_settings))),
                dropout=float(model_kwargs.get("dropout", 0.1)),
                tau_seconds=tau_seconds,
                gnn_layers=gnn_layers,
                nhead=nhead,
                mamba_layers=mamba_layers,
                hid_dim=hid_dim,
                **multiwindow_model_kwargs,
            )
            model = PatchTTCN_MultiWindowFusion_TrajPred(cfg).to(device)
            auto_patch_len_report = None
            branch_auto_patch_len_report = copy.deepcopy(branch_auto_patch_len)
            prebuilt_info_report = {"rollout": rollout_csv_path}
        else:
            raise ValueError(f"Unsupported model_variant={model_variant}")
    else:
        raise ValueError(f"Unsupported training_mode={training_mode}")

    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    test_loader = data_info["test_loader"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(scheduler_tmax if scheduler_tmax is not None else epochs)
    )

    run_cfg = {
        "csv_path": csv_path,
        "columns": list(columns),
        "source_name": source_name,
        "seed": seed,
        "model_variant": model_variant,
        "size_name": size_name,
        "training_mode": training_mode,
        "dataset_mode": "prebuilt_pseudo_recursive_csv" if training_mode == "pseudo_recursive" else "rollout_recursive_csv",
        "input_patch_num": input_patch_num,
        "patch_minutes": patch_minutes,
        "future_step_minutes": future_step_minutes,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "pad_value": pad_value,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "hid_dim": hid_dim,
        "mamba_layers": mamba_layers,
        "gnn_layers": gnn_layers,
        "nhead": nhead,
        "tau_seconds": tau_seconds,
        "multiwindow_view_settings": copy.deepcopy(multiwindow_view_settings),
        "branch_proj_dim": branch_proj_dim,
        "fusion_hidden": fusion_hidden,
        "prebuilt_dir": prebuilt_dir,
        "prebuilt_csv_path": prebuilt_csv_path,
        "rollout_csv_path": rollout_csv_path,
        "prebuilt_info_report": prebuilt_info_report,
        "model_cfg": copy.deepcopy(cfg),
    }

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Patch-Mamba training started\n")
        f.write(dumps_json_pretty(run_cfg) + "\n")
        f.write(f"sample_count = {data_info['sample_count']}\n")
        f.write(f"normal_cnt = {data_info['normal_cnt']}\n")
        f.write(f"recursive_cnt = {data_info['recursive_cnt']}\n")
        if auto_patch_len_report is not None:
            f.write(f"auto_patch_len = {auto_patch_len_report}\n")
        if branch_auto_patch_len_report is not None:
            f.write("branch_auto_patch_len = " + json.dumps(branch_auto_patch_len_report, ensure_ascii=False) + "\n")
        f.write(f"device = {device}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            training_mode=training_mode,
            model_variant=model_variant,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            multiwindow_view_settings=multiwindow_view_settings,
            pad_value=pad_value,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            training_mode=training_mode,
            model_variant=model_variant,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            multiwindow_view_settings=multiwindow_view_settings,
            pad_value=pad_value,
        )
        scheduler.step()
        epoch_time = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "lr_now": lr_now,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_fde": val_metrics["fde"],
                "val_dtw": val_metrics["dtw"],
                "epoch_time_sec": epoch_time,
            }
        )

        msg = (
            f"[{training_mode}|{model_variant}|Epoch {epoch:03d}] "
            f"lr={lr_now:.8f} | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_MSE={val_metrics['mse']:.6f}"
        )
        if val_metrics["fde"] is not None:
            msg += f" | val_FDE={val_metrics['fde']:.6f}"
        if val_metrics["dtw"] is not None:
            msg += f" | val_DTW={val_metrics['dtw']:.6f}"
        msg += f" | time={epoch_time:.2f}s"

        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": best_state,
                    "cfg": cfg,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "run_cfg": run_cfg,
                },
                best_model_path,
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"  -> Saved best checkpoint to: {best_model_path}\n")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        training_mode=training_mode,
        model_variant=model_variant,
        input_patch_num=input_patch_num,
        patch_minutes=patch_minutes,
        future_step_minutes=future_step_minutes,
        multiwindow_view_settings=multiwindow_view_settings,
        pad_value=pad_value,
    )
    val_best_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        training_mode=training_mode,
        model_variant=model_variant,
        input_patch_num=input_patch_num,
        patch_minutes=patch_minutes,
        future_step_minutes=future_step_minutes,
        multiwindow_view_settings=multiwindow_view_settings,
        pad_value=pad_value,
    )

    result = {
        "source_name": source_name,
        "size_name": size_name,
        "model_variant": model_variant,
        "training_mode": training_mode,
        "sample_count": data_info["sample_count"],
        "normal_cnt": data_info["normal_cnt"],
        "recursive_cnt": data_info["recursive_cnt"],
        "auto_patch_len": auto_patch_len_report,
        "branch_auto_patch_len": branch_auto_patch_len_report,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "val_loss": val_best_metrics["loss"],
        "val_mse": val_best_metrics["mse"],
        "val_fde": val_best_metrics["fde"],
        "val_dtw": val_best_metrics["dtw"],
        "test_loss": test_metrics["loss"],
        "test_mse": test_metrics["mse"],
        "test_fde": test_metrics["fde"],
        "test_dtw": test_metrics["dtw"],
        "lr": lr,
        "hid_dim": hid_dim,
        "mamba_layers": mamba_layers,
        "input_patch_num": input_patch_num,
        "patch_minutes": patch_minutes,
        "save_dir": save_dir,
        "best_model_path": best_model_path,
        "log_path": log_path,
        "prebuilt_info_report": prebuilt_info_report,
    }

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "train_history.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(csv_result_path, index=False, encoding="utf-8-sig")
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"run_cfg": run_cfg, "result": result}), f, ensure_ascii=False, indent=2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n[Test]\n")
        for key, value in result.items():
            f.write(f"{key} = {value}\n")

    return result
