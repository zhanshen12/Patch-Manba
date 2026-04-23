"""
Metric and restoration utilities for trajectory prediction.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch

try:
    from patch_dataset import restore_pred_lonlat as restore_pred_lonlat_np
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "The training module requires `patch_dataset.restore_pred_lonlat`. "
        "Please make sure the companion dataset builder is available."
    ) from exc


def restore_pred_lonlat_torch(pred_norm_xy: torch.Tensor, restore_info: torch.Tensor) -> torch.Tensor:
    """
    Restore normalized longitude/latitude predictions to raw geographic space.
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
    out = torch.stack([lon_raw, lat_raw], dim=1)
    return out[0] if squeeze_back else out


def dtw_distance_np(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance for two 2D trajectories.
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
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute MSE, FDE, and DTW for single-step or multi-step prediction outputs.
    """
    with torch.no_grad():
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred.shape={pred.shape}, target.shape={target.shape}")

        if pred.ndim == 2:
            mse = ((pred - target) ** 2).mean().item()
            if pred.size(-1) < 2:
                return {"mse": mse, "fde": None, "dtw": None}
            pred_xy_raw = restore_pred_lonlat_torch(pred[:, :2], restore_info)
            tgt_xy_raw = restore_pred_lonlat_torch(target[:, :2], restore_info)
            fde = torch.norm(pred_xy_raw - tgt_xy_raw, dim=-1).mean().item()
            return {"mse": mse, "fde": fde, "dtw": fde}

        if pred.ndim != 3:
            raise ValueError(f"Only 2D or 3D prediction tensors are supported; received ndim={pred.ndim}")

        bsz, tlen, dim = pred.shape
        if valid_mask is None:
            valid_mask = torch.ones((bsz, tlen), dtype=torch.float32, device=pred.device)

        valid_mask = valid_mask.float()
        valid_mask3 = valid_mask.unsqueeze(-1)
        valid_value_count = (valid_mask3.sum() * dim).clamp(min=1.0)
        mse = (((pred - target) ** 2) * valid_mask3).sum().item() / float(valid_value_count.item())

        if dim < 2:
            return {"mse": mse, "fde": None, "dtw": None}

        pred_xy_norm = pred[..., :2].reshape(bsz * tlen, 2)
        tgt_xy_norm = target[..., :2].reshape(bsz * tlen, 2)
        restore_expand = restore_info.unsqueeze(1).expand(bsz, tlen, 4).reshape(bsz * tlen, 4)

        pred_xy_raw = restore_pred_lonlat_torch(pred_xy_norm, restore_expand).view(bsz, tlen, 2)
        tgt_xy_raw = restore_pred_lonlat_torch(tgt_xy_norm, restore_expand).view(bsz, tlen, 2)

        pred_xy_raw_np = pred_xy_raw.detach().cpu().numpy()
        tgt_xy_raw_np = tgt_xy_raw.detach().cpu().numpy()
        valid_mask_np = valid_mask.detach().cpu().numpy()

        fde_list = []
        dtw_list = []
        for i in range(bsz):
            valid_len = int(np.sum(valid_mask_np[i] > 0.5))
            if valid_len <= 0:
                continue
            pred_seq = pred_xy_raw_np[i, :valid_len]
            tgt_seq = tgt_xy_raw_np[i, :valid_len]
            fde_list.append(float(np.linalg.norm(pred_seq[-1] - tgt_seq[-1])))
            dtw_list.append(dtw_distance_np(pred_seq, tgt_seq))

        fde = float(np.mean(fde_list)) if fde_list else None
        dtw = float(np.mean(dtw_list)) if dtw_list else None
        return {"mse": mse, "fde": fde, "dtw": dtw}
