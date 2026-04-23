"""
Configuration builders for Patch-Mamba models.
"""

from __future__ import annotations

from typing import Sequence


def make_model_cfg(
    batch_data: dict,
    auto_patch_len: int,
    device,
    hid_dim: int = 128,
    mamba_layers: int = 2,
    **kwargs,
) -> dict:
    """
    Build a configuration dictionary for a single-view Patch-Mamba model.
    """
    head_hidden = int(kwargs.get("head_hidden", max(256, 2 * int(hid_dim))))
    return {
        "device": str(device),
        "in_dim": batch_data["model_input"].shape[-1],
        "hid_dim": int(hid_dim),
        "npatch": batch_data["patch_mask"].shape[1],
        "patch_len": int(auto_patch_len),
        "gnn_layers": int(kwargs.get("gnn_layers", 2)),
        "nhead": int(kwargs.get("nhead", 4)),
        "tau_seconds": float(kwargs.get("tau_seconds", 300.0)),
        "delta_minutes": float(kwargs.get("delta_minutes", 5.0)),
        "mamba_layers": int(mamba_layers),
        "mamba_d_state": int(kwargs.get("mamba_d_state", 16)),
        "mamba_d_conv": int(kwargs.get("mamba_d_conv", 4)),
        "mamba_expand": int(kwargs.get("mamba_expand", 2)),
        "dropout": float(kwargs.get("dropout", 0.1)),
        "head_hidden": head_hidden,
        "pred_len": 1,
        "pred_dim": batch_data["model_label"].shape[-1],
    }


def make_multiwindow_model_cfg(
    branch_batch_data: dict,
    device,
    view_settings: Sequence[dict],
    branch_auto_patch_len: dict,
    **kwargs,
) -> dict:
    """
    Build a configuration dictionary for the multi-window hybrid model.
    """
    branch_cfgs = []
    for view in view_settings:
        name = str(view["name"])
        batch_data = branch_batch_data[name]
        branch_cfgs.append(
            {
                "device": str(device),
                "branch_name": name,
                "in_dim": batch_data["model_input"].shape[-1],
                "hid_dim": int(view.get("hid_dim", kwargs.get("hid_dim", 128))),
                "npatch": int(view["input_patch_num"]),
                "patch_len": int(branch_auto_patch_len[name]),
                "gnn_layers": int(view.get("gnn_layers", kwargs.get("gnn_layers", 2))),
                "nhead": int(view.get("nhead", kwargs.get("nhead", 4))),
                "tau_seconds": float(view.get("tau_seconds", kwargs.get("tau_seconds", 300.0))),
                "delta_minutes": float(view.get("delta_minutes", view["patch_minutes"])),
                "mamba_layers": int(view.get("mamba_layers", kwargs.get("mamba_layers", 2))),
                "mamba_d_state": int(view.get("mamba_d_state", kwargs.get("mamba_d_state", 16))),
                "mamba_d_conv": int(view.get("mamba_d_conv", kwargs.get("mamba_d_conv", 4))),
                "mamba_expand": int(view.get("mamba_expand", kwargs.get("mamba_expand", 2))),
                "dropout": float(view.get("dropout", kwargs.get("dropout", 0.1))),
            }
        )

    base_name = str(view_settings[0]["name"])
    pred_dim = branch_batch_data[base_name]["model_label"].shape[-1]
    return {
        "device": str(device),
        "pred_len": 1,
        "pred_dim": pred_dim,
        "branch_cfgs": branch_cfgs,
        "branch_proj_dim": int(kwargs.get("branch_proj_dim", 128)),
        "fusion_hidden": int(kwargs.get("fusion_hidden", max(256, int(kwargs.get("branch_proj_dim", 128)) * len(branch_cfgs)))),
        "dropout": float(kwargs.get("dropout", 0.1)),
    }
