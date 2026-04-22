"""Runtime batch construction helpers for recursive rollout training."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch

from patch_dataset import EPS

from .utils import restore_pred_lonlat_torch



def encode_raw_points_to_model_input_torch(
    points5: torch.Tensor,
    sample_start_ts: torch.Tensor | float,
    restore_info: torch.Tensor,
) -> torch.Tensor:
    """Encode raw trajectory points into the 6D model input representation.

    Parameters
    ----------
    points5:
        Tensor of raw points shaped ``(num_points, 5)`` containing
        ``[lon, lat, sog, cog, timestamp]``.
    sample_start_ts:
        History-window start timestamp used to derive relative time in minutes.
    restore_info:
        Restoration statistics containing ``[lon_min, lon_max, lat_min, lat_max]``.

    Returns
    -------
    torch.Tensor
        Encoded tensor shaped ``(num_points, 6)`` with normalized longitude,
        normalized latitude, scaled speed, heading sine/cosine, and relative
        time in minutes.
    """
    if points5.numel() == 0:
        return torch.empty((0, 6), dtype=restore_info.dtype, device=restore_info.device)

    points5 = points5.float()
    restore_info = restore_info.float()
    sample_start_ts = torch.as_tensor(sample_start_ts, dtype=points5.dtype, device=points5.device)

    lon = points5[:, 0]
    lat = points5[:, 1]
    sog = points5[:, 2]
    cog = points5[:, 3]
    ts = points5[:, 4]

    lon_min = restore_info[0]
    lon_max = restore_info[1]
    lat_min = restore_info[2]
    lat_max = restore_info[3]

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    lon_norm = torch.where(torch.abs(lon_range) < 1e-8, torch.zeros_like(lon), (lon - lon_min) / lon_range)
    lat_norm = torch.where(torch.abs(lat_range) < 1e-8, torch.zeros_like(lat), (lat - lat_min) / lat_range)

    sog_div10 = sog / 10.0
    cog_rad = cog * (math.pi / 180.0)
    cog_sin = torch.sin(cog_rad)
    cog_cos = torch.cos(cog_rad)
    rel_time_min = (ts - sample_start_ts) / 60.0

    return torch.stack([lon_norm, lat_norm, sog_div10, cog_sin, cog_cos, rel_time_min], dim=-1)



def collect_input_patches_as_model_input_torch(
    points6: torch.Tensor,
    window_start_ts: torch.Tensor | float,
    input_patch_num: int,
    patch_minutes: int,
    restore_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect windowed raw points and convert them into patch-annotated model input.

    Parameters
    ----------
    points6:
        Raw point tensor shaped ``(num_points, 6)`` or ``(6,)``.
    window_start_ts:
        Timestamp of the first patch boundary.
    input_patch_num:
        Number of patches in the history window.
    patch_minutes:
        Width of each patch in minutes.
    restore_info:
        Restoration statistics for coordinate normalization.

    Returns
    -------
    tuple
        ``(model_input, patch_index, patch_mask)`` where ``model_input`` is the
        concatenated 6D feature sequence, ``patch_index`` stores one-based patch
        IDs, and ``patch_mask`` marks which patches contain at least one point.
    """
    device = restore_info.device
    dtype = restore_info.dtype
    patch_sec = int(patch_minutes * 60)
    patch_mask = torch.zeros((input_patch_num,), dtype=dtype, device=device)

    if points6 is None or points6.numel() == 0:
        return (
            torch.empty((0, 6), dtype=dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            patch_mask,
        )

    if points6.dim() == 1:
        points6 = points6.unsqueeze(0)
    points6 = points6.float()

    feature_list: List[torch.Tensor] = []
    patch_id_list: List[torch.Tensor] = []
    window_start_ts = torch.as_tensor(window_start_ts, dtype=points6.dtype, device=points6.device)

    for patch_idx in range(input_patch_num):
        left = window_start_ts + patch_idx * patch_sec
        right = left + patch_sec
        if patch_idx < input_patch_num - 1:
            mask = (points6[:, 4] >= left - EPS) & (points6[:, 4] < right - EPS)
        else:
            mask = (points6[:, 4] >= left - EPS) & (points6[:, 4] <= right + EPS)

        patch_points = points6[mask]
        if patch_points.size(0) <= 0:
            continue

        patch_mask[patch_idx] = 1.0
        feature_list.append(
            encode_raw_points_to_model_input_torch(
                points5=patch_points[:, :5],
                sample_start_ts=window_start_ts,
                restore_info=restore_info,
            )
        )
        patch_id_list.append(torch.full((patch_points.size(0),), patch_idx + 1, dtype=torch.long, device=device))

    if not feature_list:
        return (
            torch.empty((0, 6), dtype=dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            patch_mask,
        )

    model_input = torch.cat(feature_list, dim=0)
    patch_index = torch.cat(patch_id_list, dim=0)
    return model_input, patch_index, patch_mask



def build_runtime_singleview_batch_from_tensor_points(
    observed_points6: torch.Tensor,
    observed_points6_mask: torch.Tensor,
    generated_points6: torch.Tensor | None,
    generated_points6_mask: torch.Tensor | None,
    current_cut_times: torch.Tensor,
    restore_info_batch: torch.Tensor,
    input_patch_num: int,
    patch_minutes: int,
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Build one runtime batch for the single-view recursive model.

    This function re-materializes the history window at each recursive step by
    combining observed points with already generated predictions, cropping the
    combined trajectory to the current history range, and then patchifying it.

    Returns
    -------
    dict
        Batch dictionary matching the format expected by the single-view model.
    """
    history_sec = int(input_patch_num * patch_minutes * 60)
    batch_size = observed_points6.size(0)
    device = observed_points6.device
    dtype = observed_points6.dtype

    per_sample_model_input = []
    per_sample_patch_index = []
    per_sample_patch_mask = []
    max_seq_len = 0

    for batch_idx in range(batch_size):
        observed = observed_points6[batch_idx][observed_points6_mask[batch_idx] > 0.5]
        if (
            generated_points6 is not None
            and generated_points6_mask is not None
            and generated_points6.size(1) > 0
        ):
            generated = generated_points6[batch_idx][generated_points6_mask[batch_idx] > 0.5]
            points = torch.cat([observed, generated], dim=0)
        else:
            points = observed

        cut_time_ts = current_cut_times[batch_idx]
        window_start = cut_time_ts - history_sec
        window_end = cut_time_ts

        if points.numel() > 0:
            window_mask = (points[:, 4] >= window_start - EPS) & (points[:, 4] <= window_end + EPS)
            window_points = points[window_mask]
        else:
            window_points = points

        model_input_i, patch_index_i, patch_mask_i = collect_input_patches_as_model_input_torch(
            points6=window_points,
            window_start_ts=window_start,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            restore_info=restore_info_batch[batch_idx],
        )
        per_sample_model_input.append(model_input_i)
        per_sample_patch_index.append(patch_index_i)
        per_sample_patch_mask.append(patch_mask_i)
        max_seq_len = max(max_seq_len, int(model_input_i.size(0)))

    max_seq_len = max(max_seq_len, 1)
    model_input = torch.full((batch_size, max_seq_len, 6), float(pad_value), dtype=dtype, device=device)
    sequence_mask = torch.zeros((batch_size, max_seq_len), dtype=dtype, device=device)
    patch_index = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    patch_mask = (
        torch.stack(per_sample_patch_mask, dim=0)
        if per_sample_patch_mask
        else torch.zeros((batch_size, input_patch_num), dtype=dtype, device=device)
    )

    for batch_idx in range(batch_size):
        seq_len = per_sample_model_input[batch_idx].size(0)
        if seq_len <= 0:
            continue
        model_input[batch_idx, :seq_len] = per_sample_model_input[batch_idx]
        sequence_mask[batch_idx, :seq_len] = 1.0
        patch_index[batch_idx, :seq_len] = per_sample_patch_index[batch_idx]

    return {
        "model_input": model_input,
        "sequence_mask": sequence_mask,
        "patch_index": patch_index,
        "patch_mask": patch_mask,
        "restore_info": restore_info_batch,
    }



def build_runtime_multiview_batch_from_tensor_points(
    observed_points6: torch.Tensor,
    observed_points6_mask: torch.Tensor,
    generated_points6: torch.Tensor | None,
    generated_points6_mask: torch.Tensor | None,
    current_cut_times: torch.Tensor,
    restore_info_batch: torch.Tensor,
    view_settings: Sequence[Dict[str, object]],
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Build runtime input branches for the multi-window fusion model."""
    output = {"restore_info": restore_info_batch}
    for view in view_settings:
        name = str(view["name"])
        single_batch = build_runtime_singleview_batch_from_tensor_points(
            observed_points6=observed_points6,
            observed_points6_mask=observed_points6_mask,
            generated_points6=generated_points6,
            generated_points6_mask=generated_points6_mask,
            current_cut_times=current_cut_times,
            restore_info_batch=restore_info_batch,
            input_patch_num=int(view["input_patch_num"]),
            patch_minutes=int(view["patch_minutes"]),
            pad_value=pad_value,
        )
        output[f"{name}__model_input"] = single_batch["model_input"]
        output[f"{name}__sequence_mask"] = single_batch["sequence_mask"]
        output[f"{name}__patch_index"] = single_batch["patch_index"]
        output[f"{name}__patch_mask"] = single_batch["patch_mask"]
    return output



def pred5_to_raw_point6_torch(
    pred5: torch.Tensor,
    restore_info: torch.Tensor,
    target_ts: torch.Tensor,
) -> torch.Tensor:
    """Convert batched normalized model outputs into raw 6D point records."""
    pred5 = pred5.float()
    restore_info = restore_info.float()
    target_ts = target_ts.to(dtype=pred5.dtype, device=pred5.device)

    lonlat_raw = restore_pred_lonlat_torch(pred5[:, :2], restore_info)
    sog_raw = pred5[:, 2] * 10.0
    cog_raw = torch.remainder(torch.rad2deg(torch.atan2(pred5[:, 3], pred5[:, 4])) + 360.0, 360.0)
    interp_flag = torch.ones_like(sog_raw)

    return torch.stack(
        [
            lonlat_raw[:, 0],
            lonlat_raw[:, 1],
            sog_raw,
            cog_raw,
            target_ts,
            interp_flag,
        ],
        dim=1,
    )



def rollout_forward(
    model,
    batch: Dict[str, torch.Tensor],
    model_variant: str = "single",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    multiwindow_view_settings: Sequence[Dict[str, object]] | None = None,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Run recursive rollout inference for one mini-batch.

    Parameters
    ----------
    model:
        Model instance exposing ``forward_step``.
    batch:
        Rollout batch from ``RolloutTrainDataset``.
    model_variant:
        Either ``"single"`` or ``"multiwindow_hybrid"``.
    input_patch_num, patch_minutes, future_step_minutes:
        Parameters describing the history and forecast step size.
    multiwindow_view_settings:
        View definitions for the hybrid model.
    pad_value:
        Padding value used when constructing runtime sequences.

    Returns
    -------
    tuple
        ``(loss, pred_steps)`` where ``pred_steps`` has shape
        ``(batch, rollout_steps, pred_dim)``. ``loss`` may be ``None`` when no
        valid rollout step exists in the current batch.
    """
    rollout_mask = batch["rollout_mask"].float()
    future_model_labels = batch["future_model_labels"]
    future_points6 = batch["future_points6"]
    restore_info = batch["restore_info"]
    current_cut_times = batch["cut_time_ts"].to(dtype=future_points6.dtype)
    observed_points6 = batch["observed_points6"]
    observed_points6_mask = batch["observed_points6_mask"].float()

    _, rollout_steps, _ = future_model_labels.shape
    future_step_sec = int(future_step_minutes * 60)

    generated_points_list = []
    generated_mask_list = []
    pred_steps_list = []
    step_losses = []

    for step_idx in range(rollout_steps):
        alive_mask = rollout_mask[:, step_idx] > 0.5

        if generated_points_list:
            generated_points6 = torch.stack(generated_points_list, dim=1)
            generated_points6_mask = torch.stack(generated_mask_list, dim=1)
        else:
            generated_points6 = None
            generated_points6_mask = None

        if model_variant == "single":
            runtime_batch = build_runtime_singleview_batch_from_tensor_points(
                observed_points6=observed_points6,
                observed_points6_mask=observed_points6_mask,
                generated_points6=generated_points6,
                generated_points6_mask=generated_points6_mask,
                current_cut_times=current_cut_times,
                restore_info_batch=restore_info,
                input_patch_num=input_patch_num,
                patch_minutes=patch_minutes,
                pad_value=pad_value,
            )
        else:
            runtime_batch = build_runtime_multiview_batch_from_tensor_points(
                observed_points6=observed_points6,
                observed_points6_mask=observed_points6_mask,
                generated_points6=generated_points6,
                generated_points6_mask=generated_points6_mask,
                current_cut_times=current_cut_times,
                restore_info_batch=restore_info,
                view_settings=multiwindow_view_settings,
                pad_value=pad_value,
            )

        pred_step = model.forward_step(runtime_batch)
        pred_steps_list.append(pred_step)

        if torch.any(alive_mask):
            target_step = future_model_labels[:, step_idx, :]
            step_losses.append(torch.nn.functional.smooth_l1_loss(pred_step[alive_mask], target_step[alive_mask]))

        pred_point6 = pred5_to_raw_point6_torch(
            pred5=pred_step,
            restore_info=restore_info,
            target_ts=future_points6[:, step_idx, 4],
        )
        generated_points_list.append(pred_point6)
        generated_mask_list.append(alive_mask.float())
        current_cut_times = current_cut_times + future_step_sec

    if not pred_steps_list:
        pred_steps = torch.zeros_like(future_model_labels)
    else:
        pred_steps = torch.stack(pred_steps_list, dim=1)

    loss = torch.stack(step_losses).mean() if step_losses else None
    return loss, pred_steps



def estimate_runtime_patch_len_from_rollout_samples(
    samples: Sequence[Dict[str, object]],
    input_patch_num: int,
    patch_minutes: int,
    future_step_minutes: int,
) -> int:
    """Estimate the largest patch occupancy encountered during recursive rollout.

    This value is used to provision the patch-major tensor size for models that
    are trained recursively but still need a static patch length during model
    construction.
    """
    history_sec = int(input_patch_num * patch_minutes * 60)
    patch_sec = int(patch_minutes * 60)
    future_step_sec = int(future_step_minutes * 60)
    max_len = 1

    for sample in samples:
        observed_points6 = np.asarray(sample["observed_points6"], dtype=np.float32)
        future_points6 = np.asarray(sample["future_points6"], dtype=np.float32)
        cut_time_ts = float(sample["cut_time_ts"])

        for step_idx in range(len(future_points6)):
            current_cut = cut_time_ts + step_idx * future_step_sec
            window_start = current_cut - history_sec
            if step_idx > 0:
                mixed_points = np.concatenate([observed_points6, future_points6[:step_idx]], axis=0)
            else:
                mixed_points = observed_points6

            if mixed_points.size == 0:
                continue

            window_mask = (mixed_points[:, 4] >= window_start - EPS) & (mixed_points[:, 4] <= current_cut + EPS)
            window_points = mixed_points[window_mask]
            if window_points.size == 0:
                continue

            for patch_idx in range(input_patch_num):
                left = window_start + patch_idx * patch_sec
                right = left + patch_sec
                if patch_idx < input_patch_num - 1:
                    patch_mask = (window_points[:, 4] >= left - EPS) & (window_points[:, 4] < right - EPS)
                else:
                    patch_mask = (window_points[:, 4] >= left - EPS) & (window_points[:, 4] <= right + EPS)
                max_len = max(max_len, int(np.sum(patch_mask)))

    return max_len
