"""Public training interface for the modular Patch-Mamba package."""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, Sequence

import pandas as pd
import torch

from patch_dataset import load_saved_samples_from_csv, pack_samples_to_batch

from .config_builders import make_model_cfg, make_multiwindow_model_cfg
from .data_builders import (
    build_multiview_dataloaders_from_prebuilt,
    build_recursive_dataloaders_from_rollout_csv,
    build_singleview_dataloaders_from_prebuilt,
    estimate_recursive_patch_len_report,
)
from .layers import HAS_MAMBA
from .models import PatchTTCN_Mamba_TrajPred, PatchTTCN_MultiWindowFusion_TrajPred
from .training import evaluate, train_one_epoch
from .utils import (
    build_default_prebuilt_csv_path,
    ensure_output_dir,
    estimate_max_patch_len,
    set_seed,
    to_serializable,
)



def train_patch_mamba_model(
    csv_path: str = "data2.csv",
    columns: Sequence[str] = ("bd", "radar", "AIS"),
    source_name=None,
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
    grad_clip: float | None = 1.0,
    hid_dim: int = 128,
    mamba_layers: int = 2,
    gnn_layers: int = 2,
    nhead: int = 4,
    tau_seconds: float = 300.0,
    model_variant: str = "single",
    size_name: str | None = None,
    multiwindow_view_settings: Sequence[Dict[str, object]] | None = None,
    branch_proj_dim: int = 128,
    fusion_hidden: int | None = None,
    prebuilt_dir: str = "prebuilt_source_csv",
    prebuilt_csv_path: str | None = None,
    training_mode: str = "pseudo_recursive",
    rollout_csv_path: str | None = None,
    **model_kwargs,
) -> Dict[str, object]:
    """Train a Patch-Mamba trajectory model and persist the training artifacts.

    This function intentionally preserves the behavior and interface of the
    original monolithic script while serving as the main public API of the
    modularized package. It supports two orthogonal modes:

    * ``training_mode='pseudo_recursive'`` or ``'recursive'``
    * ``model_variant='single'`` or ``'multiwindow_hybrid'``

    Parameters
    ----------
    csv_path, columns:
        Preserved for interface compatibility with the original script. The
        modularized implementation still stores them in the run configuration,
        even though training now relies on prebuilt CSV inputs.
    source_name:
        Data-source identifier used by the prebuilt dataset naming convention.
    save_dir:
        Output directory for logs, checkpoints, histories, and summary files.
    log_filename, model_filename, csv_result_filename, json_result_filename:
        Output file names written under ``save_dir``.
    seed:
        Global random seed.
    input_patch_num, patch_minutes, future_step_minutes:
        Core temporal hyperparameters of the history window and forecast step.
    train_ratio, val_ratio:
        Dataset split ratios.
    pad_value:
        Sequence padding value used by prebuilt and runtime batch builders.
    train_batch_size, eval_batch_size, num_workers:
        Dataloader hyperparameters.
    epochs, lr, weight_decay, scheduler_tmax, grad_clip:
        Optimizer and training loop settings.
    hid_dim, mamba_layers, gnn_layers, nhead, tau_seconds:
        Main model hyperparameters.
    model_variant:
        ``"single"`` for a single-view model or ``"multiwindow_hybrid"`` for
        the fused multi-window model.
    size_name:
        Optional experiment alias stored in result summaries.
    multiwindow_view_settings:
        Per-branch configuration list for the hybrid model.
    branch_proj_dim, fusion_hidden:
        Fusion-head hyperparameters for the hybrid model.
    prebuilt_dir, prebuilt_csv_path:
        Location of pseudo-recursive prebuilt sample CSV files.
    training_mode:
        ``"pseudo_recursive"`` or ``"recursive"``.
    rollout_csv_path:
        Rollout CSV path required for true recursive training.
    **model_kwargs:
        Forwarded fine-grained model hyperparameters.

    Returns
    -------
    dict
        Final experiment summary containing validation and test metrics, output
        paths, and key metadata.
    """
    set_seed(seed)
    save_dir = ensure_output_dir(save_dir)
    log_path = os.path.join(save_dir, log_filename)
    best_model_path = os.path.join(save_dir, model_filename)
    csv_result_path = os.path.join(save_dir, csv_result_filename)
    json_result_path = os.path.join(save_dir, json_result_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not HAS_MAMBA:
        raise ImportError("The current environment does not provide mamba_ssm. Please install mamba-ssm first.")

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
            for duplicate_key in [
                "gnn_layers",
                "nhead",
                "tau_seconds",
                "delta_minutes",
                "head_hidden",
                "mamba_d_state",
                "mamba_d_conv",
                "mamba_expand",
                "dropout",
            ]:
                single_model_kwargs.pop(duplicate_key, None)
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
                raise ValueError("multiwindow_view_settings must not be empty for the hybrid model.")
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
            for duplicate_key in [
                "branch_proj_dim",
                "fusion_hidden",
                "dropout",
                "tau_seconds",
                "gnn_layers",
                "nhead",
                "mamba_layers",
                "hid_dim",
                "mamba_d_state",
                "mamba_d_conv",
                "mamba_expand",
            ]:
                multiwindow_model_kwargs.pop(duplicate_key, None)
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
            raise ValueError(f"Unsupported model_variant={model_variant!r}.")

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
            pseudo_reference_csv = prebuilt_csv_path
            if pseudo_reference_csv is None:
                pseudo_reference_csv = build_default_prebuilt_csv_path(
                    prebuilt_dir=prebuilt_dir,
                    source_name=source_name,
                    input_patch_num=input_patch_num,
                    patch_minutes=patch_minutes,
                    future_step_minutes=future_step_minutes,
                    training_mode="pseudo_recursive",
                )
            if not os.path.exists(pseudo_reference_csv):
                raise FileNotFoundError(
                    "Recursive training requires a pseudo-recursive CSV with the same window configuration "
                    f"to infer static model shapes: {pseudo_reference_csv}"
                )
            ref_samples = load_saved_samples_from_csv(pseudo_reference_csv)
            ref_batch_data = pack_samples_to_batch(ref_samples, pad_value=pad_value)
            ref_patch_len = max(estimate_max_patch_len(ref_batch_data["patch_index"], input_patch_num=input_patch_num), 1)
            rollout_patch_len = estimate_recursive_patch_len_report(
                samples=data_info["samples"],
                input_patch_num=input_patch_num,
                patch_minutes=patch_minutes,
                future_step_minutes=future_step_minutes,
            )
            auto_patch_len = max(ref_patch_len, rollout_patch_len)
            single_model_kwargs = dict(model_kwargs)
            for duplicate_key in [
                "gnn_layers",
                "nhead",
                "tau_seconds",
                "delta_minutes",
                "head_hidden",
                "mamba_d_state",
                "mamba_d_conv",
                "mamba_expand",
                "dropout",
            ]:
                single_model_kwargs.pop(duplicate_key, None)
            cfg = make_model_cfg(
                batch_data=ref_batch_data,
                auto_patch_len=auto_patch_len,
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
            auto_patch_len_report = int(auto_patch_len)
            branch_auto_patch_len_report = None
            prebuilt_info_report = {"rollout": rollout_csv_path, "shape_ref": pseudo_reference_csv}

        elif model_variant == "multiwindow_hybrid":
            if multiwindow_view_settings is None or len(multiwindow_view_settings) == 0:
                raise ValueError("multiwindow_view_settings must not be empty for recursive hybrid training.")
            branch_batch_data = {}
            branch_auto_patch_len = {}
            branch_csv_paths = {}
            for view in multiwindow_view_settings:
                name = str(view["name"])
                ref_csv = view.get("prebuilt_csv_path")
                if ref_csv is None:
                    ref_csv = build_default_prebuilt_csv_path(
                        prebuilt_dir=prebuilt_dir,
                        source_name=source_name,
                        input_patch_num=int(view["input_patch_num"]),
                        patch_minutes=int(view["patch_minutes"]),
                        future_step_minutes=future_step_minutes,
                        training_mode="pseudo_recursive",
                    )
                if not os.path.exists(ref_csv):
                    raise FileNotFoundError(f"Reference CSV for recursive hybrid branch {name!r} not found: {ref_csv}")
                ref_samples = load_saved_samples_from_csv(ref_csv)
                ref_batch = pack_samples_to_batch(ref_samples, pad_value=pad_value)
                branch_batch_data[name] = ref_batch
                ref_patch_len = max(estimate_max_patch_len(ref_batch["patch_index"], input_patch_num=int(view["input_patch_num"])), 1)
                rollout_patch_len = estimate_recursive_patch_len_report(
                    samples=data_info["samples"],
                    input_patch_num=int(view["input_patch_num"]),
                    patch_minutes=int(view["patch_minutes"]),
                    future_step_minutes=future_step_minutes,
                )
                branch_auto_patch_len[name] = max(ref_patch_len, rollout_patch_len)
                branch_csv_paths[name] = ref_csv
            multiwindow_model_kwargs = dict(model_kwargs)
            for duplicate_key in [
                "branch_proj_dim",
                "fusion_hidden",
                "dropout",
                "tau_seconds",
                "gnn_layers",
                "nhead",
                "mamba_layers",
                "hid_dim",
                "mamba_d_state",
                "mamba_d_conv",
                "mamba_expand",
            ]:
                multiwindow_model_kwargs.pop(duplicate_key, None)
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
            prebuilt_info_report = {"rollout": rollout_csv_path, "shape_ref": branch_csv_paths}
        else:
            raise ValueError(f"Unsupported model_variant={model_variant!r}.")
    else:
        raise ValueError(f"Unsupported training_mode={training_mode!r}.")

    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    test_loader = data_info["test_loader"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_tmax if scheduler_tmax is not None else epochs),
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
        f.write("Training started (dual mode: pseudo-recursive / recursive)\n")
        f.write(json.dumps(to_serializable(run_cfg), ensure_ascii=False, indent=2) + "\n")
        f.write(f"sample_count = {data_info['sample_count']}\n")
        f.write(f"normal_cnt = {data_info['normal_cnt']}\n")
        f.write(f"recursive_cnt = {data_info['recursive_cnt']}\n")
        if auto_patch_len_report is not None:
            f.write(f"auto_patch_len = {auto_patch_len_report}\n")
        if branch_auto_patch_len_report is not None:
            f.write("branch_auto_patch_len = " + json.dumps(branch_auto_patch_len_report, ensure_ascii=False) + "\n")
        f.write(f"device = {device}\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
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
        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        history_row = {
            "epoch": epoch,
            "lr_now": lr_now,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_fde": val_metrics["fde"],
            "val_dtw": val_metrics["dtw"],
            "epoch_time_sec": epoch_time,
        }
        history.append(history_row)

        message = (
            f"[{training_mode}|{model_variant}|Epoch {epoch:03d}] lr={lr_now:.8f} | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_MSE={val_metrics['mse']:.6f}"
        )
        if val_metrics["fde"] is not None:
            message += f" | val_FDE={val_metrics['fde']:.6f}"
        if val_metrics["dtw"] is not None:
            message += f" | val_DTW={val_metrics['dtw']:.6f}"
        message += f" | time={epoch_time:.2f}s"

        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

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
                f.write(f"  -> Saved best model to: {best_model_path}\n")

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

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, "train_history.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(csv_result_path, index=False, encoding="utf-8-sig")
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable({"run_cfg": run_cfg, "result": result}), f, ensure_ascii=False, indent=2)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n[Test]\n")
        for key, value in result.items():
            f.write(f"{key} = {value}\n")

    return result
