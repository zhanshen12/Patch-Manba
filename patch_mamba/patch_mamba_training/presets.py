"""
Preset experiment definitions.
"""

from __future__ import annotations


def build_default_multiscale_size_presets_180_60() -> list[dict]:
    """
    Return default multi-scale experiment presets for the 180/60 setting.
    """
    return [
        {
            "name": "win15_12x15",
            "model_variant": "single",
            "input_patch_num": 12,
            "patch_minutes": 15,
            "hid_dim": 128,
            "mamba_layers": 2,
            "gnn_layers": 2,
            "nhead": 4,
            "delta_minutes": 15.0,
        },
        {
            "name": "win10_18x10",
            "model_variant": "single",
            "input_patch_num": 18,
            "patch_minutes": 10,
            "hid_dim": 144,
            "mamba_layers": 2,
            "gnn_layers": 2,
            "nhead": 4,
            "delta_minutes": 10.0,
        },
        {
            "name": "win20_9x20",
            "model_variant": "single",
            "input_patch_num": 9,
            "patch_minutes": 20,
            "hid_dim": 128,
            "mamba_layers": 2,
            "gnn_layers": 2,
            "nhead": 4,
            "delta_minutes": 20.0,
        },
        {
            "name": "win30_6x30",
            "model_variant": "single",
            "input_patch_num": 6,
            "patch_minutes": 30,
            "hid_dim": 112,
            "mamba_layers": 1,
            "gnn_layers": 1,
            "nhead": 4,
            "delta_minutes": 30.0,
        },
        {
            "name": "mixed_window_15_10_20_30",
            "model_variant": "multiwindow_hybrid",
            "branch_proj_dim": 128,
            "fusion_hidden": 512,
            "multiwindow_view_settings": [
                {"name": "view_12x15", "input_patch_num": 12, "patch_minutes": 15, "hid_dim": 128, "mamba_layers": 2, "gnn_layers": 2, "nhead": 4, "delta_minutes": 15.0},
                {"name": "view_18x10", "input_patch_num": 18, "patch_minutes": 10, "hid_dim": 144, "mamba_layers": 2, "gnn_layers": 2, "nhead": 4, "delta_minutes": 10.0},
                {"name": "view_9x20", "input_patch_num": 9, "patch_minutes": 20, "hid_dim": 128, "mamba_layers": 2, "gnn_layers": 2, "nhead": 4, "delta_minutes": 20.0},
                {"name": "view_6x30", "input_patch_num": 6, "patch_minutes": 30, "hid_dim": 112, "mamba_layers": 1, "gnn_layers": 1, "nhead": 4, "delta_minutes": 30.0},
            ],
        },
    ]
