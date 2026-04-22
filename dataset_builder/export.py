"""Prebuilt multi-window export helpers for a single trajectory source."""

from __future__ import annotations

import os

import pandas as pd

from .builders import build_patch_forecast_dataset_from_csv_single_source
from .serialization import save_rollout_samples_to_csv, save_samples_to_csv


def build_output_csv_path(
    output_dir: str,
    source_name: str,
    input_patch_num: int,
    patch_minutes: int,
    future_step_minutes: int,
    training_mode: str = "pseudo_recursive",
) -> str:
    """
    Build the standardized output CSV path for one prebuilt dataset export.

    Parameters
    ----------
    output_dir:
        Directory where the exported CSV should be written.
    source_name:
        Source name such as ``AIS``, ``radar``, or ``bd``.
    input_patch_num:
        Number of input patches.
    patch_minutes:
        Duration of each patch in minutes.
    future_step_minutes:
        Forecast step duration in minutes.
    training_mode:
        Either ``pseudo_recursive`` or ``recursive``.

    Returns
    -------
    str
        Standardized output CSV path.
    """
    source_alias = {"AIS": "ais", "radar": "radar", "bd": "bd"}
    source_key = source_alias.get(source_name, str(source_name).lower())
    suffix = "pseudo" if training_mode == "pseudo_recursive" else "recursive"
    return os.path.join(output_dir, f"{source_key}_{input_patch_num}batch_{patch_minutes}min_{future_step_minutes}min_{suffix}.csv")


def default_window_configs():
    """
    Return the module's default multi-window configuration list.

    Returns
    -------
    list[dict]
        Default window configuration dictionaries.
    """
    return [
        {"name": "win15_12x15", "input_patch_num": 12, "patch_minutes": 15},
        {"name": "win10_18x10", "input_patch_num": 18, "patch_minutes": 10},
        {"name": "win20_9x20", "input_patch_num": 9, "patch_minutes": 20},
        {"name": "win30_6x30", "input_patch_num": 6, "patch_minutes": 30},
        {"name": "win25_7x25", "input_patch_num": 7, "patch_minutes": 25},
    ]


def build_and_save_source_multiscale(
    csv_path: str = "data.csv",
    output_dir: str = "prebuilt_source_csv",
    source_name: str = "AIS",
    window_configs=None,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = 12,
    training_mode: str = "pseudo_recursive",
) -> pd.DataFrame:
    """
    Build and export prebuilt datasets for one source across multiple windows.

    Parameters
    ----------
    csv_path:
        Input CSV path.
    output_dir:
        Directory used for exported CSV files and summary CSV.
    source_name:
        Source column to process.
    window_configs:
        List of window configuration dictionaries. If ``None``, the default
        configurations are used.
    strict, pad_value, future_step_minutes, sample_stride_minutes,
    min_total_input_points, max_future_steps, training_mode:
        Passed through to the dataset builder without semantic changes.

    Returns
    -------
    pandas.DataFrame
        Summary table describing all exported datasets.
    """
    if window_configs is None:
        window_configs = default_window_configs()

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for cfg in window_configs:
        name = str(cfg["name"])
        input_patch_num = int(cfg["input_patch_num"])
        patch_minutes = int(cfg["patch_minutes"])

        print("\n" + "=" * 100)
        print(
            f"Processing source={source_name} | config={name} | "
            f"{input_patch_num}x{patch_minutes} | mode={training_mode}"
        )
        print("=" * 100)

        samples, _, _ = build_patch_forecast_dataset_from_csv_single_source(
            csv_path=csv_path,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
            training_mode=training_mode,
        )

        output_csv = build_output_csv_path(
            output_dir=output_dir,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            training_mode=training_mode,
        )
        if training_mode == "pseudo_recursive":
            save_samples_to_csv(samples, output_csv)
            normal_cnt = sum(sample["sample_type"] == "normal" for sample in samples)
            recursive_cnt = sum(sample["sample_type"] == "recursive" for sample in samples)
        else:
            save_rollout_samples_to_csv(samples, output_csv)
            normal_cnt = len(samples)
            recursive_cnt = len(samples)

        summary_rows.append(
            {
                "source_name": source_name,
                "config_name": name,
                "training_mode": training_mode,
                "input_patch_num": input_patch_num,
                "patch_minutes": patch_minutes,
                "history_minutes": input_patch_num * patch_minutes,
                "future_step_minutes": future_step_minutes,
                "max_future_steps": max_future_steps,
                "output_csv": output_csv,
                "sample_count": len(samples),
                "normal_count": normal_cnt,
                "recursive_count": recursive_cnt,
            }
        )

        print(f"Finished: output={output_csv} | samples={len(samples)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, f"summary_{str(source_name).lower()}_{training_mode}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\nSummary saved to: {summary_csv}")
    return summary_df
