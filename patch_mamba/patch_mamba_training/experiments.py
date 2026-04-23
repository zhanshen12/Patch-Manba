"""
Experiment runners and result summarization utilities.
"""

from __future__ import annotations

import copy
import json
import os

import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None

from .presets import build_default_multiscale_size_presets_180_60
from .train import train_patch_mamba_model
from .utils import ensure_output_dir, to_serializable


def plot_multiscale_compare(result_df: pd.DataFrame, save_dir: str):
    """
    Plot bar charts for multi-scale comparison results.
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
    source_name: str | None = None,
    output_root: str = "patch_mamba_window_fusion_compare_180_60",
    size_presets: list[dict] | None = None,
    metric_rank_key: str = "test_mse",
    common_train_kwargs: dict | None = None,
) -> dict:
    """
    Run a preset-based multi-scale comparison experiment.
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
    out = {
        "summary_csv": summary_csv,
        "summary_xlsx": summary_xlsx,
        "summary_json": summary_json,
        "plots": plots,
        "rows": rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(to_serializable(out), f, ensure_ascii=False, indent=2)
    return out
