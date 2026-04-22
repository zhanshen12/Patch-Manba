"""Training and evaluation loops for the modular Patch-Mamba package."""

from __future__ import annotations

import torch
import torch.nn as nn

from .runtime import rollout_forward
from .utils import move_batch_to_device, trajectory_metrics



def train_one_epoch_pseudo(
    model,
    loader,
    optimizer,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> float:
    """Train for one epoch in pseudo-recursive single-step mode.

    Parameters
    ----------
    model:
        Model exposing ``compute_loss``.
    loader:
        Training dataloader.
    optimizer:
        Optimizer instance.
    device:
        Execution device.
    grad_clip:
        Optional gradient clipping threshold. ``None`` disables clipping.

    Returns
    -------
    float
        Average training loss over all samples in the epoch.
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
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch["model_label"].size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)



def train_one_epoch_recursive(
    model,
    loader,
    optimizer,
    device: torch.device,
    grad_clip: float | None = 1.0,
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings=None,
    pad_value: float = 0.0,
) -> float:
    """Train for one epoch in true recursive rollout mode."""
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
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch["future_model_labels"].size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)



def train_one_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    grad_clip: float | None = 1.0,
    training_mode: str = "pseudo_recursive",
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings=None,
    pad_value: float = 0.0,
) -> float:
    """Dispatch one training epoch according to the configured training mode."""
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
    raise ValueError(f"Unsupported training_mode={training_mode!r}.")


@torch.no_grad()
def evaluate_pseudo(model, loader, device: torch.device) -> dict:
    """Evaluate the model in pseudo-recursive single-step mode."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds = []
    tgts = []
    restores = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss, pred = model.compute_loss(batch)
        batch_size = batch["model_label"].size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        preds.append(pred.detach().cpu())
        tgts.append(batch["model_label"].unsqueeze(1).detach().cpu())
        restores.append(batch["restore_info"].detach().cpu())

    avg_loss = total_loss / max(total_count, 1)
    preds = torch.cat(preds, dim=0) if preds else torch.empty((0, 1, 5))
    tgts = torch.cat(tgts, dim=0) if tgts else torch.empty((0, 1, 5))
    restores = torch.cat(restores, dim=0) if restores else torch.empty((0, 4))
    metrics = trajectory_metrics(preds, tgts, restores) if len(preds) > 0 else {"mse": None, "fde": None, "dtw": None}
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate_recursive(
    model,
    loader,
    device: torch.device,
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings=None,
    pad_value: float = 0.0,
) -> dict:
    """Evaluate the model in true recursive rollout mode."""
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
    device: torch.device,
    training_mode: str = "pseudo_recursive",
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings=None,
    pad_value: float = 0.0,
) -> dict:
    """Dispatch evaluation according to the configured training mode."""
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
    raise ValueError(f"Unsupported training_mode={training_mode!r}.")
