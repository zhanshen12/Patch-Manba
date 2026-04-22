"""CSV serialization helpers for flat and rollout datasets."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .batching import pack_rollout_samples_to_batch, pack_samples_to_batch
from .datasets import PatchForecastDataset, PatchForecastRolloutDataset
from .utils import json_to_ndarray, ndarray_to_json


def samples_to_dataframe(samples) -> pd.DataFrame:
    """
    Convert flat pseudo-recursive samples into an exportable DataFrame.

    Parameters
    ----------
    samples:
        List of flat samples.

    Returns
    -------
    pandas.DataFrame
        Tabular representation suitable for CSV export.
    """
    rows = []
    for sample in samples:
        rows.append(
            {
                "source_name": sample["source_name"],
                "track_id": sample["track_id"],
                "sample_type": sample["sample_type"],
                "recursive_step": sample["recursive_step"],
                "window_start_ts": sample["window_start_ts"],
                "window_end_ts": sample["window_end_ts"],
                "future_time_ts": sample["future_time_ts"],
                "future_interp_flag": sample["future_interp_flag"],
                "feedback_point_count": sample["feedback_point_count"],
                "input_point_count": sample["input_point_count"],
                "data_sequence_json": ndarray_to_json(np.round(sample["data_sequence"], 5)),
                "patch_index_json": ndarray_to_json(sample["patch_index"].astype(np.int64)),
                "patch_mask_json": ndarray_to_json(np.round(sample["patch_mask"], 5)),
                "label_json": ndarray_to_json(np.round(sample["label"], 5)),
                "restore_info_json": ndarray_to_json(np.round(sample["restore_info"], 6)),
            }
        )
    return pd.DataFrame(rows)


def save_samples_to_csv(samples, output_csv: str) -> None:
    """
    Save flat pseudo-recursive samples to a CSV file.

    Parameters
    ----------
    samples:
        List of flat samples.
    output_csv:
        Output CSV path.
    """
    folder = os.path.dirname(output_csv)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df = samples_to_dataframe(samples)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved flat samples to: {output_csv}, sample_count={len(df)}")


def load_saved_samples_from_csv(saved_csv: str):
    """
    Load flat pseudo-recursive samples from a previously exported CSV file.

    Parameters
    ----------
    saved_csv:
        Path to the saved sample CSV.

    Returns
    -------
    list[dict]
        Reconstructed sample list.
    """
    df = pd.read_csv(saved_csv, encoding="utf-8-sig")
    samples = []
    for _, row in df.iterrows():
        data_sequence = json_to_ndarray(row["data_sequence_json"], dtype=np.float32)
        patch_index = json_to_ndarray(row["patch_index_json"], dtype=np.int64)
        patch_mask = json_to_ndarray(row["patch_mask_json"], dtype=np.float32)
        label = json_to_ndarray(row["label_json"], dtype=np.float32)
        restore_info = json_to_ndarray(row["restore_info_json"], dtype=np.float32)

        samples.append(
            {
                "source_name": str(row["source_name"]),
                "track_id": int(row["track_id"]),
                "sample_type": str(row["sample_type"]),
                "recursive_step": int(row["recursive_step"]),
                "window_start_ts": float(row["window_start_ts"]),
                "window_end_ts": float(row["window_end_ts"]),
                "future_time_ts": float(row["future_time_ts"]),
                "future_interp_flag": float(row["future_interp_flag"]),
                "feedback_point_count": int(row["feedback_point_count"]),
                "input_point_count": int(row["input_point_count"]),
                "data_sequence": data_sequence.astype(np.float32),
                "patch_index": patch_index.astype(np.int64),
                "patch_mask": patch_mask.astype(np.float32),
                "label": label.astype(np.float32),
                "restore_info": restore_info.astype(np.float32),
            }
        )
    return samples


def load_saved_dataset_from_csv(saved_csv: str, pad_value: float = 0.0):
    """
    Load flat samples from CSV and rebuild the corresponding PyTorch dataset.

    Parameters
    ----------
    saved_csv:
        Path to the saved flat sample CSV.
    pad_value:
        Padding value used while reconstructing the padded batch arrays.

    Returns
    -------
    tuple
        ``(samples, batch_data, dataset)``.
    """
    samples = load_saved_samples_from_csv(saved_csv)
    batch_data = pack_samples_to_batch(samples, pad_value=pad_value)
    dataset = PatchForecastDataset(batch_data)
    return samples, batch_data, dataset


def rollout_samples_to_dataframe(samples) -> pd.DataFrame:
    """
    Convert rollout samples into an exportable DataFrame.

    Parameters
    ----------
    samples:
        List of rollout samples.

    Returns
    -------
    pandas.DataFrame
        Tabular representation suitable for CSV export.
    """
    rows = []
    for sample in samples:
        rows.append(
            {
                "source_name": sample["source_name"],
                "track_id": sample["track_id"],
                "cut_time_ts": sample["cut_time_ts"],
                "observed_points6_json": ndarray_to_json(np.round(sample["observed_points6"], 6)),
                "future_points6_json": ndarray_to_json(np.round(sample["future_points6"], 6)),
                "future_labels_json": ndarray_to_json(np.round(sample["future_labels"], 5)),
                "future_model_labels_json": ndarray_to_json(np.round(sample["future_model_labels"], 5)),
                "restore_info_json": ndarray_to_json(np.round(sample["restore_info"], 6)),
            }
        )
    return pd.DataFrame(rows)


def save_rollout_samples_to_csv(samples, output_csv: str) -> None:
    """
    Save rollout samples to a CSV file.

    Parameters
    ----------
    samples:
        List of rollout samples.
    output_csv:
        Output CSV path.
    """
    folder = os.path.dirname(output_csv)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df = rollout_samples_to_dataframe(samples)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved rollout samples to: {output_csv}, sample_count={len(df)}")


def load_rollout_samples_from_csv(saved_csv: str):
    """
    Load rollout samples from a previously exported CSV file.

    Parameters
    ----------
    saved_csv:
        Path to the saved rollout CSV.

    Returns
    -------
    list[dict]
        Reconstructed rollout sample list.
    """
    df = pd.read_csv(saved_csv, encoding="utf-8-sig")
    samples = []
    for _, row in df.iterrows():
        observed_points6 = json_to_ndarray(row["observed_points6_json"], dtype=np.float32)
        future_points6 = json_to_ndarray(row["future_points6_json"], dtype=np.float32)
        future_labels = json_to_ndarray(row["future_labels_json"], dtype=np.float32)
        future_model_labels = json_to_ndarray(row["future_model_labels_json"], dtype=np.float32)
        restore_info = json_to_ndarray(row["restore_info_json"], dtype=np.float32)

        samples.append(
            {
                "source_name": str(row["source_name"]),
                "track_id": int(row["track_id"]),
                "cut_time_ts": float(row["cut_time_ts"]),
                "observed_points6": observed_points6.astype(np.float32),
                "future_points6": future_points6.astype(np.float32),
                "future_labels": future_labels.astype(np.float32),
                "future_model_labels": future_model_labels.astype(np.float32),
                "restore_info": restore_info.astype(np.float32),
            }
        )
    return samples


def load_rollout_dataset_from_csv(saved_csv: str):
    """
    Load rollout samples from CSV and rebuild the corresponding PyTorch dataset.

    Parameters
    ----------
    saved_csv:
        Path to the saved rollout sample CSV.

    Returns
    -------
    tuple
        ``(samples, batch_data, dataset)``.
    """
    samples = load_rollout_samples_from_csv(saved_csv)
    batch_data = pack_rollout_samples_to_batch(samples)
    dataset = PatchForecastRolloutDataset(batch_data)
    return samples, batch_data, dataset
