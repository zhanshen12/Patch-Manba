"""
General utility functions used across the training framework.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Global random seed used to make dataset splitting, weight
        initialization, and other stochastic operations reproducible.

    Notes
    -----
    This function also initializes CUDA-related random states when a CUDA
    device is available. The code keeps ``cudnn.benchmark=True`` to match
    the behavior of the original training script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_output_dir(output_dir: str = "patch_mamba_output") -> str:
    """
    Create an output directory if it does not already exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move tensor-valued items in a batch dictionary onto the target device.
    """
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a mask-aware mean across the patch dimension.
    """
    weight = mask.unsqueeze(-1).float()
    summed = (x * weight).sum(dim=1)
    denom = weight.sum(dim=1).clamp(min=1.0)
    return summed / denom


def gather_last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Gather the final valid patch feature for each sample.
    """
    bsz, _, dim = x.shape
    lengths = mask.sum(dim=1).long().clamp(min=1)
    index = (lengths - 1).view(bsz, 1, 1).expand(bsz, 1, dim)
    return x.gather(dim=1, index=index).squeeze(1)


def to_serializable(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable Python types.
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


def dumps_json_pretty(obj: Any) -> str:
    """
    Serialize an object into a human-readable JSON string.
    """
    return json.dumps(to_serializable(obj), ensure_ascii=False, indent=2)


def normalize_source_name(source_name: str) -> str:
    """
    Normalize source aliases into a canonical short identifier.
    """
    s = str(source_name).strip().lower()
    mapping = {
        "ais": "ais",
        "radar": "radar",
        "rader": "radar",
        "bd": "bd",
        "beidou": "bd",
        "北斗": "bd",
        "雷达": "radar",
    }
    return mapping.get(s, s)
