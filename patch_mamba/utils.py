"""Utility helpers for the modular Patch-Mamba training package.

This module collects small, reusable helpers that are shared by multiple
submodules. The functions here intentionally remain lightweight and side-effect
free whenever possible so that the rest of the codebase can import them without
introducing circular dependencies.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from patch_dataset import build_output_csv_path, restore_pred_lonlat as restore_pred_lonlat_np


def set_seed(seed: int = 42) -> None:
    """Seed the major random number generators used by the training pipeline.

    The original single-file script uses Python's ``random`` module, NumPy, and
    PyTorch in the same execution path. Reproducibility therefore requires that
    all three libraries are seeded consistently. When CUDA is available, the
    function also seeds the per-device and all-device CUDA generators.

    Notes
    -----
    This implementation preserves the behavior of the source script, including
    enabling ``torch.backends.cudnn.benchmark``. That setting may improve
    throughput, but depending on input variability it can also reduce strict
    determinism. The goal here is interface preservation rather than changing
    runtime semantics.

    Parameters
    ----------
    seed:
        Integer random seed shared across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



def ensure_output_dir(output_dir: str = "patch_mamba_output") -> str:
    """Create an output directory when it does not already exist.

    Parameters
    ----------
    output_dir:
        Target directory path used for checkpoints, logs, and result files.

    Returns
    -------
    str
        The same directory path that was passed in, which makes the helper
        convenient to use inline during path construction.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor inside a batch dictionary to the requested device.

    The dataloaders in this project return dictionaries instead of tuples. This
    helper keeps that convention and applies ``Tensor.to(...)`` only to tensor
    values while leaving non-tensor metadata unchanged.

    Parameters
    ----------
    batch:
        Mini-batch dictionary returned by a dataset or dataloader.
    device:
        The destination device, typically ``cpu`` or ``cuda``.

    Returns
    -------
    dict
        A new dictionary with the same keys as ``batch`` and tensors moved to
        ``device`` using non-blocking transfer where supported.
    """
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }



def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute a mean across the sequence dimension while ignoring padded steps.

    Parameters
    ----------
    x:
        Tensor of shape ``(batch, length, dim)``.
    mask:
        Binary or real-valued mask of shape ``(batch, length)`` where positive
        entries indicate valid positions.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch, dim)`` containing the masked average.
    """
    weights = mask.unsqueeze(-1).float()
    weighted_sum = (x * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return weighted_sum / denom



def gather_last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather the last valid token from each sequence according to ``mask``.

    This helper is used by the encoder to combine two complementary summaries:
    the final valid patch token and the masked mean over all valid patch tokens.

    Parameters
    ----------
    x:
        Tensor of shape ``(batch, length, dim)``.
    mask:
        Validity mask of shape ``(batch, length)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch, dim)`` containing the final valid token for
        every sequence in the batch.
    """
    batch_size, _, dim = x.shape
    lengths = mask.sum(dim=1).long().clamp(min=1)
    gather_index = (lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, dim)
    return x.gather(dim=1, index=gather_index).squeeze(1)



def restore_pred_lonlat_torch(
    pred_norm_xy: torch.Tensor,
    restore_info: torch.Tensor,
) -> torch.Tensor:
    """Restore normalized longitude/latitude predictions back to raw coordinates.

    Parameters
    ----------
    pred_norm_xy:
        Tensor containing normalized longitude and latitude. Supported shapes are
        ``(2,)`` and ``(batch, 2)``.
    restore_info:
        Tensor containing ``[lon_min, lon_max, lat_min, lat_max]`` per sample.
        Supported shapes are ``(4,)`` and ``(batch, 4)``.

    Returns
    -------
    torch.Tensor
        Raw-coordinate tensor with shape ``(2,)`` or ``(batch, 2)``, matching
        the input rank of ``pred_norm_xy``.
    """
    squeeze_back = False
    if pred_norm_xy.dim() == 1:
        pred_norm_xy = pred_norm_xy.unsqueeze(0)
        squeeze_back = True
    if restore_info.dim() == 1:
        restore_info = restore_info.unsqueeze(0)

    pred_norm_xy = pred_norm_xy.float()
    restore_info = restore_info.float()

    lon_norm = pred_norm_xy[:, 0]
    lat_norm = pred_norm_xy[:, 1]
    lon_min = restore_info[:, 0]
    lon_max = restore_info[:, 1]
    lat_min = restore_info[:, 2]
    lat_max = restore_info[:, 3]

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    lon_raw = torch.where(torch.abs(lon_range) < 1e-8, lon_min, lon_norm * lon_range + lon_min)
    lat_raw = torch.where(torch.abs(lat_range) < 1e-8, lat_min, lat_norm * lat_range + lat_min)

    output = torch.stack([lon_raw, lat_raw], dim=1)
    if squeeze_back:
        output = output[0]
    return output



def dtw_distance_np(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute the Dynamic Time Warping distance between two 2D trajectories.

    Parameters
    ----------
    seq1, seq2:
        Trajectory arrays convertible to shape ``(time, 2)``.

    Returns
    -------
    float
        DTW distance accumulated with Euclidean point-wise cost.
    """
    seq1 = np.asarray(seq1, dtype=np.float32).reshape(-1, 2)
    seq2 = np.asarray(seq2, dtype=np.float32).reshape(-1, 2)
    n, m = len(seq1), len(seq2)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n, m])



def trajectory_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    restore_info: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> Dict[str, float | None]:
    """Evaluate trajectory predictions with normalized MSE and raw-space metrics.

    The source implementation supports two modes:

    * ``(batch, dim)`` tensors for single-step prediction.
    * ``(batch, time, dim)`` tensors for recursive multi-step prediction.

    In both cases, MSE is computed in normalized model space. FDE and DTW are
    computed after de-normalizing the longitude/latitude channels with the
    provided restoration statistics.

    Parameters
    ----------
    pred:
        Predicted model outputs.
    target:
        Ground-truth model outputs with the same shape as ``pred``.
    restore_info:
        Per-sample restoration statistics containing longitude and latitude
        ranges.
    valid_mask:
        Optional mask for multi-step evaluation. When omitted for 3D tensors,
        every time step is treated as valid.

    Returns
    -------
    dict
        Dictionary containing ``mse``, ``fde``, and ``dtw``. Some values may be
        ``None`` when the feature dimension is insufficient to derive spatial
        metrics.
    """
    with torch.no_grad():
        if pred.shape != target.shape:
            raise ValueError(f"pred.shape={pred.shape} does not match target.shape={target.shape}.")

        if pred.ndim == 2:
            mse = ((pred - target) ** 2).mean().item()
            if pred.size(-1) < 2:
                return {"mse": mse, "fde": None, "dtw": None}
            pred_xy_raw = restore_pred_lonlat_torch(pred[:, :2], restore_info)
            tgt_xy_raw = restore_pred_lonlat_torch(target[:, :2], restore_info)
            fde = torch.norm(pred_xy_raw - tgt_xy_raw, dim=-1).mean().item()
            return {"mse": mse, "fde": fde, "dtw": fde}

        if pred.ndim != 3:
            raise ValueError(f"Only 2D and 3D prediction tensors are supported, got rank {pred.ndim}.")

        batch_size, time_len, dim = pred.shape
        if valid_mask is None:
            valid_mask = torch.ones((batch_size, time_len), dtype=torch.float32, device=pred.device)

        valid_mask = valid_mask.float()
        valid_mask3 = valid_mask.unsqueeze(-1)
        valid_value_count = (valid_mask3.sum() * dim).clamp(min=1.0)
        mse = (((pred - target) ** 2) * valid_mask3).sum().item() / float(valid_value_count.item())

        if dim < 2:
            return {"mse": mse, "fde": None, "dtw": None}

        pred_xy_norm = pred[..., :2].reshape(batch_size * time_len, 2)
        tgt_xy_norm = target[..., :2].reshape(batch_size * time_len, 2)
        restore_expand = restore_info.unsqueeze(1).expand(batch_size, time_len, 4).reshape(batch_size * time_len, 4)

        pred_xy_raw = restore_pred_lonlat_torch(pred_xy_norm, restore_expand).view(batch_size, time_len, 2)
        tgt_xy_raw = restore_pred_lonlat_torch(tgt_xy_norm, restore_expand).view(batch_size, time_len, 2)

        pred_xy_raw_np = pred_xy_raw.detach().cpu().numpy()
        tgt_xy_raw_np = tgt_xy_raw.detach().cpu().numpy()
        valid_mask_np = valid_mask.detach().cpu().numpy()

        fde_values = []
        dtw_values = []
        for batch_idx in range(batch_size):
            valid_len = int(np.sum(valid_mask_np[batch_idx] > 0.5))
            if valid_len <= 0:
                continue
            pred_seq = pred_xy_raw_np[batch_idx, :valid_len]
            tgt_seq = tgt_xy_raw_np[batch_idx, :valid_len]
            fde_values.append(float(np.linalg.norm(pred_seq[-1] - tgt_seq[-1])))
            dtw_values.append(dtw_distance_np(pred_seq, tgt_seq))

        fde = float(np.mean(fde_values)) if fde_values else None
        dtw = float(np.mean(dtw_values)) if dtw_values else None
        return {"mse": mse, "fde": fde, "dtw": dtw}



def estimate_max_patch_len(patch_index_np: np.ndarray, input_patch_num: int) -> int:
    """Estimate the maximum number of points assigned to any patch.

    Parameters
    ----------
    patch_index_np:
        NumPy array of patch identifiers produced by the dataset builder.
    input_patch_num:
        Number of patches in the input history window.

    Returns
    -------
    int
        Maximum count of points that appear in any single patch across the full
        batch. A minimum value of ``1`` is always returned.
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



def to_serializable(obj: Any) -> Any:
    """Recursively convert tensors and NumPy scalars into JSON-safe objects.

    Parameters
    ----------
    obj:
        Arbitrary Python object that may contain tensors, NumPy scalars, lists,
        tuples, or nested dictionaries.

    Returns
    -------
    Any
        A structure composed only of JSON-friendly primitive types, lists, and
        dictionaries whenever conversion is possible.
    """
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj



def normalize_source_name(source_name: Any) -> str:
    """Normalize source identifiers into a compact canonical representation.

    The training scripts may refer to the same data source with English names,
    Chinese names, or minor spelling variants. This helper keeps those aliases
    aligned so that sample grouping and cross-view matching behave consistently.

    Parameters
    ----------
    source_name:
        Raw source identifier.

    Returns
    -------
    str
        Canonical lowercase source name.
    """
    source_name_str = str(source_name).strip().lower()
    mapping = {
        "ais": "ais",
        "radar": "radar",
        "rader": "radar",
        "bd": "bd",
        "beidou": "bd",
        "北斗": "bd",
        "雷达": "radar",
    }
    return mapping.get(source_name_str, source_name_str)



def build_default_prebuilt_csv_path(
    prebuilt_dir: str = "prebuilt_source_csv",
    source_name: str = "AIS",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    future_step_minutes: int = 5,
    training_mode: str = "pseudo_recursive",
) -> str:
    """Build the default path to a prebuilt dataset CSV.

    Parameters
    ----------
    prebuilt_dir:
        Directory storing prebuilt sample CSV files.
    source_name:
        Trajectory source name used in the filename convention.
    input_patch_num:
        Number of historical patches.
    patch_minutes:
        Length of each patch in minutes.
    future_step_minutes:
        Forecast horizon of one prediction step in minutes.
    training_mode:
        Dataset type identifier forwarded to the filename builder.

    Returns
    -------
    str
        Default CSV path following the naming convention defined in
        ``patch_dataset.build_output_csv_path``.
    """
    return build_output_csv_path(
        output_dir=prebuilt_dir,
        source_name=source_name,
        input_patch_num=input_patch_num,
        patch_minutes=patch_minutes,
        future_step_minutes=future_step_minutes,
        training_mode=training_mode,
    )



def build_sample_key(sample: Dict[str, Any]) -> Tuple[Any, ...]:
    """Build a stable key used to align samples across multiple views.

    Parameters
    ----------
    sample:
        One prebuilt sample dictionary loaded from CSV.

    Returns
    -------
    tuple
        Immutable key composed of source identity, trajectory identity, sample
        type, recursive step, window start time, and future target time.
    """
    return (
        normalize_source_name(sample["source_name"]),
        int(sample["track_id"]),
        str(sample["sample_type"]),
        int(sample["recursive_step"]),
        round(float(sample["window_start_ts"]), 6),
        round(float(sample["future_time_ts"]), 6),
    )



def build_track_group_keys_from_samples(samples: Iterable[Dict[str, Any]]) -> list[tuple[str, int]]:
    """Build per-sample grouping keys so splits can respect trajectory identity.

    Parameters
    ----------
    samples:
        Iterable of sample dictionaries loaded from prebuilt or rollout CSV
        files.

    Returns
    -------
    list of tuple
        ``(source_name, track_id)`` pairs used to keep samples from the same
        trajectory together during train/validation/test splitting.
    """
    group_keys = []
    for index, sample in enumerate(samples):
        source_name = normalize_source_name(sample.get("source_name", "unknown"))
        track_id = int(sample.get("track_id", index))
        group_keys.append((source_name, track_id))
    return group_keys



def pred5_to_raw_point6(pred5: np.ndarray, restore_info: np.ndarray, target_ts: float) -> np.ndarray:
    """Convert one normalized 5D model prediction into a raw 6D point record.

    Parameters
    ----------
    pred5:
        Model output in the normalized 5D representation
        ``[lon_norm, lat_norm, sog/10, sin(cog), cos(cog)]``.
    restore_info:
        Restoration statistics used to recover raw longitude and latitude.
    target_ts:
        Timestamp assigned to the generated point.

    Returns
    -------
    numpy.ndarray
        Raw 6D point ``[lon, lat, sog, cog, timestamp, interp_flag]``.
    """
    pred5 = np.asarray(pred5, dtype=np.float32)
    restore_info = np.asarray(restore_info, dtype=np.float32)

    lon_raw, lat_raw = restore_pred_lonlat_np(pred5[:2], restore_info)
    sog_raw = float(pred5[2]) * 10.0
    cog_raw = (math.degrees(math.atan2(float(pred5[3]), float(pred5[4]))) + 360.0) % 360.0
    return np.array([lon_raw, lat_raw, sog_raw, cog_raw, float(target_ts), 1.0], dtype=np.float32)
