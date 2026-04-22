"""Batch-packing utilities for flat and rollout training samples."""

from __future__ import annotations

import numpy as np


def pack_samples_to_batch(samples, pad_value: float = 0.0):
    """
    Pack flat pseudo-recursive samples into padded batch arrays.

    Parameters
    ----------
    samples:
        List of flat samples produced by the pseudo-recursive builder.
    pad_value:
        Value used to pad variable-length point sequences.

    Returns
    -------
    dict
        Batch dictionary containing padded NumPy arrays ready to be wrapped by
        :class:`PatchForecastDataset`.
    """
    if len(samples) == 0:
        return {
            "data_sequence": np.empty((0, 0, 10), dtype=np.float32),
            "model_input": np.empty((0, 0, 6), dtype=np.float32),
            "sequence_mask": np.empty((0, 0), dtype=np.float32),
            "patch_index": np.empty((0, 0), dtype=np.int64),
            "patch_mask": np.empty((0, 0), dtype=np.float32),
            "label": np.empty((0, 10), dtype=np.float32),
            "model_label": np.empty((0, 5), dtype=np.float32),
            "restore_info": np.empty((0, 4), dtype=np.float32),
            "track_id": np.empty((0,), dtype=np.int64),
            "source_name": np.empty((0,), dtype=object),
        }

    batch_size = len(samples)
    seq_len_max = max(sample["data_sequence"].shape[0] for sample in samples)
    patch_count = len(samples[0]["patch_mask"])

    data_sequence = np.full((batch_size, seq_len_max, 10), pad_value, dtype=np.float32)
    model_input = np.full((batch_size, seq_len_max, 6), pad_value, dtype=np.float32)
    sequence_mask = np.zeros((batch_size, seq_len_max), dtype=np.float32)
    patch_index = np.zeros((batch_size, seq_len_max), dtype=np.int64)
    patch_mask = np.zeros((batch_size, patch_count), dtype=np.float32)
    label = np.zeros((batch_size, 10), dtype=np.float32)
    model_label = np.zeros((batch_size, 5), dtype=np.float32)
    restore_info = np.zeros((batch_size, 4), dtype=np.float32)
    track_id = np.zeros((batch_size,), dtype=np.int64)
    source_name = np.empty((batch_size,), dtype=object)

    for i, sample in enumerate(samples):
        seq_len = sample["data_sequence"].shape[0]
        if seq_len > 0:
            data_sequence[i, :seq_len] = sample["data_sequence"]
            model_input[i, :seq_len] = sample["data_sequence"][:, :6]
            sequence_mask[i, :seq_len] = 1.0
            patch_index[i, :seq_len] = sample["patch_index"]

        patch_mask[i] = sample["patch_mask"]
        label[i] = sample["label"]
        model_label[i] = sample["label"][:5]
        restore_info[i] = sample["restore_info"]
        track_id[i] = int(sample.get("track_id", i))
        source_name[i] = str(sample.get("source_name", "unknown"))

    return {
        "data_sequence": data_sequence,
        "model_input": model_input,
        "sequence_mask": sequence_mask,
        "patch_index": patch_index,
        "patch_mask": patch_mask,
        "label": label,
        "model_label": model_label,
        "restore_info": restore_info,
        "track_id": track_id,
        "source_name": source_name,
    }


def pack_rollout_samples_to_batch(samples):
    """
    Pack rollout samples into padded batch arrays for true recursive training.

    Parameters
    ----------
    samples:
        List of rollout samples produced by the recursive builder.

    Returns
    -------
    dict
        Batch dictionary containing padded rollout arrays ready to be wrapped
        by :class:`PatchForecastRolloutDataset`.
    """
    if len(samples) == 0:
        return {
            "observed_points6": np.empty((0, 0, 6), dtype=np.float32),
            "observed_points6_mask": np.empty((0, 0), dtype=np.float32),
            "future_points6": np.empty((0, 0, 6), dtype=np.float32),
            "future_labels": np.empty((0, 0, 10), dtype=np.float32),
            "future_model_labels": np.empty((0, 0, 5), dtype=np.float32),
            "rollout_mask": np.empty((0, 0), dtype=np.float32),
            "restore_info": np.empty((0, 4), dtype=np.float32),
            "cut_time_ts": np.empty((0,), dtype=np.float64),
            "source_name": np.empty((0,), dtype=object),
            "track_id": np.empty((0,), dtype=np.int64),
        }

    batch_size = len(samples)
    observed_len_max = max(sample["observed_points6"].shape[0] for sample in samples)
    future_len_max = max(sample["future_points6"].shape[0] for sample in samples)

    observed_points6 = np.zeros((batch_size, observed_len_max, 6), dtype=np.float32)
    observed_points6_mask = np.zeros((batch_size, observed_len_max), dtype=np.float32)
    future_points6 = np.zeros((batch_size, future_len_max, 6), dtype=np.float32)
    future_labels = np.zeros((batch_size, future_len_max, 10), dtype=np.float32)
    future_model_labels = np.zeros((batch_size, future_len_max, 5), dtype=np.float32)
    rollout_mask = np.zeros((batch_size, future_len_max), dtype=np.float32)
    restore_info = np.zeros((batch_size, 4), dtype=np.float32)
    cut_time_ts = np.zeros((batch_size,), dtype=np.float64)
    source_name = np.empty((batch_size,), dtype=object)
    track_id = np.zeros((batch_size,), dtype=np.int64)

    for i, sample in enumerate(samples):
        observed_len = sample["observed_points6"].shape[0]
        future_len = sample["future_points6"].shape[0]
        if observed_len > 0:
            observed_points6[i, :observed_len] = sample["observed_points6"]
            observed_points6_mask[i, :observed_len] = 1.0
        if future_len > 0:
            future_points6[i, :future_len] = sample["future_points6"]
            future_labels[i, :future_len] = sample["future_labels"]
            future_model_labels[i, :future_len] = sample["future_model_labels"]
            rollout_mask[i, :future_len] = 1.0
        restore_info[i] = sample["restore_info"]
        cut_time_ts[i] = sample["cut_time_ts"]
        source_name[i] = sample["source_name"]
        track_id[i] = sample["track_id"]

    return {
        "observed_points6": observed_points6,
        "observed_points6_mask": observed_points6_mask,
        "future_points6": future_points6,
        "future_labels": future_labels,
        "future_model_labels": future_model_labels,
        "rollout_mask": rollout_mask,
        "restore_info": restore_info,
        "cut_time_ts": cut_time_ts,
        "source_name": source_name,
        "track_id": track_id,
    }
