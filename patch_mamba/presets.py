"""Experiment presets and comparison runners for modular Patch-Mamba training."""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, Sequence

import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency path
    HAS_MPL = False
    plt = None

from .api import train_patch_mamba_model
from .utils import ensure_output_dir, set_seed, to_serializable



def build_default_multiscale_size_presets_180_60() -> list[Dict[str, object]]:
    """Return the default window-size presets defined in the source script.

    The presets represent one family of experiments built around a 180-minute
    history and a 60-minute evaluation context, including both single-window and
    mixed-window configurations.
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



def plot_multiscale_compare(result_df: pd.DataFrame, save_dir: str):
    """Plot bar charts comparing test metrics across window-size presets.

    Parameters
    ----------
    result_df:
        Summary dataframe produced by ``run_multiscale_comparison_experiment``.
    save_dir:
        Output directory where the comparison plots will be saved.

    Returns
    -------
    dict or None
        Mapping from metric name to plot path. Returns ``None`` when matplotlib
        is unavailable or the dataframe is empty.
    """
    if not HAS_MPL or len(result_df) == 0:
        return None

    save_dir = ensure_output_dir(save_dir)
    name_order = result_df["size_name"].tolist()
    plots = {}
    for metric_name in ["test_mse", "test_fde", "test_dtw"]:
        if metric_name not in result_df.columns:
            continue
        values = result_df[metric_name].astype(float).tolist()
        plt.figure(figsize=(9.2, 4.8))
        plt.bar(name_order, values)
        plt.xlabel("Size Preset")
        plt.ylabel(metric_name.upper())
        plt.title(f"Window-size Comparison - {metric_name.upper()}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"multiscale_compare_{metric_name}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        plots[metric_name] = plot_path
    return plots



def run_multiscale_comparison_experiment(
    source_name=None,
    output_root: str = "patch_mamba_window_fusion_compare_180_60",
    size_presets: Sequence[Dict[str, object]] | None = None,
    metric_rank_key: str = "test_mse",
    common_train_kwargs: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Run a full multi-preset comparison experiment and save summary artifacts.

    Parameters
    ----------
    source_name:
        Trajectory source identifier forwarded to the training API.
    output_root:
        Root directory containing one subdirectory per preset.
    size_presets:
        List of preset dictionaries. When omitted, the default preset family is
        used.
    metric_rank_key:
        Column used to sort the final summary table.
    common_train_kwargs:
        Keyword arguments shared by all preset runs.

    Returns
    -------
    dict
        Summary artifact bundle containing CSV, JSON, optional XLSX, optional
        plot paths, and the raw per-preset result rows.
    """
    output_root = ensure_output_dir(output_root)
    if size_presets is None:
        size_presets = build_default_multiscale_size_presets_180_60()
    if common_train_kwargs is None:
        common_train_kwargs = {}

    rows = []
    for preset in size_presets:
        preset = copy.deepcopy(preset)
        size_name = preset.pop("name")
        save_dir = os.path.join(output_root, size_name)
        run_kwargs = dict(common_train_kwargs)
        run_kwargs.update(preset)

        result = train_patch_mamba_model(
            source_name=source_name,
            save_dir=save_dir,
            log_filename=f"train_log_{size_name}.txt",
            model_filename=f"best_{size_name}.pt",
            csv_result_filename=f"result_{size_name}.csv",
            json_result_filename=f"result_{size_name}.json",
            size_name=size_name,
            **run_kwargs,
        )

        row = {"size_name": size_name}
        row.update({key: to_serializable(value) for key, value in preset.items()})
        row.update(result)
        rows.append(row)

    df = pd.DataFrame(rows)
    if metric_rank_key in df.columns:
        df = df.sort_values(by=metric_rank_key, ascending=True).reset_index(drop=True)

    summary_csv = os.path.join(output_root, "multiscale_compare_summary.csv")
    summary_xlsx = os.path.join(output_root, "multiscale_compare_summary.xlsx")
    summary_json = os.path.join(output_root, "multiscale_compare_summary.json")
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(summary_xlsx, index=False)
    except Exception:
        summary_xlsx = None

    plots = plot_multiscale_compare(df, output_root)
    output = {
        "summary_csv": summary_csv,
        "summary_xlsx": summary_xlsx,
        "summary_json": summary_json,
        "plots": plots,
        "rows": rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(to_serializable(output), f, ensure_ascii=False, indent=2)
    return output


if __name__ == "__main__":
    """Keep the original script behavior for direct execution."""
    set_seed(42)
    print("This module is intended to be imported and used by your training scripts.")
