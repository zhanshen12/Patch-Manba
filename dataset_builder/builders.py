"""Core dataset-building entry points for pseudo-recursive and recursive modes."""

from __future__ import annotations

import numpy as np

from .batching import pack_rollout_samples_to_batch, pack_samples_to_batch
from .datasets import PatchForecastDataset, PatchForecastRolloutDataset
from .io_utils import load_tracks_from_csv_raw_single_source
from .restore import get_track_restore_info
from .trajectory import (
    append_interp_flag,
    build_recursive_mixed_points,
    collect_input_patches_as_feat10,
    encode_raw_point_to_feat10,
    generate_future_fixed_points_from_raw,
)
from .constants import EPS


def build_patch_forecast_dataset_from_raw_tracks_pseudo(
    tracks_raw,
    source_name: str = "unknown",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
):
    """
    Build flat samples for the original pseudo-recursive training mode.

    In this mode, each future step becomes an independent flat sample. Earlier
    generated future points are fed back into later windows, but the labels
    themselves are always produced from real trajectory information through the
    raw-first future-point generator.

    Parameters
    ----------
    tracks_raw:
        List of raw trajectories, each with columns
        ``[lon, lat, sog, cog, timestamp]``.
    source_name:
        Source identifier stored in the exported samples.
    input_patch_num:
        Number of patches in the input history window.
    patch_minutes:
        Duration of each patch in minutes.
    strict:
        If ``True``, every patch in the input window must contain at least one
        point.
    pad_value:
        Padding value used when packing variable-length sequences.
    future_step_minutes:
        Forecast step size in minutes.
    sample_stride_minutes:
        Sliding stride for the base history window.
    min_total_input_points:
        Minimum number of encoded points required to keep a sample.
    max_future_steps:
        Optional cap on the number of future steps retained per base window.

    Returns
    -------
    tuple
        ``(samples, batch_data, dataset)``.
    """
    samples = []

    patch_sec = int(patch_minutes * 60)
    input_horizon_sec = input_patch_num * patch_sec
    future_step_sec = int(future_step_minutes * 60)
    stride_sec = int(sample_stride_minutes * 60)

    for track_id, raw_arr in enumerate(tracks_raw):
        if raw_arr is None or len(raw_arr) <= 1:
            continue

        raw_arr = np.asarray(raw_arr, dtype=np.float32)
        raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]

        track_start = raw_arr[0, 4]
        track_end = raw_arr[-1, 4]

        latest_base_ws = track_end - input_horizon_sec - future_step_sec
        if latest_base_ws < track_start:
            continue

        restore_info = get_track_restore_info(raw_arr)
        real_points6 = append_interp_flag(raw_arr)
        base_window_starts = np.arange(track_start, latest_base_ws + EPS, stride_sec, dtype=np.float64)

        for base_ws in base_window_starts:
            base_we = base_ws + input_horizon_sec
            observed_points6 = real_points6[real_points6[:, 4] <= base_we + EPS]

            future_points = generate_future_fixed_points_from_raw(
                raw_arr=raw_arr,
                cut_time_ts=base_we,
                future_step_minutes=future_step_minutes,
                future_end_time_ts=track_end,
            )
            if len(future_points) == 0:
                continue

            if max_future_steps is not None:
                future_points = future_points[: int(max_future_steps)]
                if len(future_points) == 0:
                    continue

            for step_idx in range(len(future_points)):
                cur_ws = base_ws + step_idx * future_step_sec
                cur_we = cur_ws + input_horizon_sec
                cur_label_point = future_points[step_idx]
                prev_generated = future_points[:step_idx] if step_idx > 0 else np.empty((0, 6), dtype=np.float32)

                mixed_points = build_recursive_mixed_points(
                    observed_points6=observed_points6,
                    generated_points6=prev_generated,
                )
                window_mask = (mixed_points[:, 4] >= cur_ws - EPS) & (mixed_points[:, 4] <= cur_we + EPS)
                window_points = mixed_points[window_mask]

                data_sequence, patch_index, patch_mask, point_count = collect_input_patches_as_feat10(
                    points_arr=window_points,
                    window_start_ts=cur_ws,
                    input_patch_num=input_patch_num,
                    patch_minutes=patch_minutes,
                    restore_info=restore_info,
                )

                if point_count < min_total_input_points:
                    continue
                if strict and np.sum(patch_mask) < input_patch_num:
                    continue

                label = encode_raw_point_to_feat10(
                    raw_point=cur_label_point[:5],
                    sample_start_ts=cur_ws,
                    restore_info=restore_info,
                )

                samples.append(
                    {
                        "source_name": str(source_name),
                        "data_sequence": data_sequence.astype(np.float32),
                        "patch_index": patch_index.astype(np.int64),
                        "patch_mask": patch_mask.astype(np.float32),
                        "label": label.astype(np.float32),
                        "restore_info": restore_info.astype(np.float32),
                        "track_id": int(track_id),
                        "sample_type": "normal" if step_idx == 0 else "recursive",
                        "recursive_step": int(step_idx),
                        "window_start_ts": float(cur_ws),
                        "window_end_ts": float(cur_we),
                        "future_time_ts": float(cur_label_point[4]),
                        "future_interp_flag": float(cur_label_point[5]),
                        "feedback_point_count": int(len(prev_generated)),
                        "input_point_count": int(point_count),
                    }
                )

    batch_data = pack_samples_to_batch(samples, pad_value=pad_value)
    dataset = PatchForecastDataset(batch_data)
    return samples, batch_data, dataset


def build_patch_rollout_dataset_from_raw_tracks(
    tracks_raw,
    source_name: str = "unknown",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
):
    """
    Build rollout samples for true recursive training.

    In this mode, one sample contains the observed history and the full future
    rollout target sequence. The model is expected to perform recursive training
    or evaluation externally using these structured targets.

    Parameters
    ----------
    tracks_raw:
        List of raw trajectories, each with columns
        ``[lon, lat, sog, cog, timestamp]``.
    source_name:
        Source identifier stored in the exported samples.
    input_patch_num:
        Number of patches in the history window.
    patch_minutes:
        Duration of each patch in minutes.
    strict:
        If ``True``, the initial history window must have at least one point in
        every patch.
    pad_value:
        Compatibility argument preserved from the original function signature.
        It is intentionally unused in rollout construction but kept unchanged so
        external callers do not need to modify their code.
    future_step_minutes:
        Forecast step size in minutes.
    sample_stride_minutes:
        Sliding stride for successive cut times.
    min_total_input_points:
        Minimum number of encoded points required in the initial history window.
    max_future_steps:
        Optional cap on rollout length.

    Returns
    -------
    tuple
        ``(samples, batch_data, dataset)``.
    """
    _ = pad_value  # kept for interface compatibility
    samples = []

    history_sec = int(input_patch_num * patch_minutes * 60)
    future_step_sec = int(future_step_minutes * 60)
    stride_sec = int(sample_stride_minutes * 60)

    for track_id, raw_arr in enumerate(tracks_raw):
        if raw_arr is None or len(raw_arr) <= 1:
            continue

        raw_arr = np.asarray(raw_arr, dtype=np.float32)
        raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]
        real_points6 = append_interp_flag(raw_arr)
        restore_info = get_track_restore_info(raw_arr)

        track_start = raw_arr[0, 4]
        track_end = raw_arr[-1, 4]

        earliest_cut = track_start + history_sec
        latest_cut = track_end - future_step_sec
        if latest_cut < earliest_cut:
            continue

        cut_times = np.arange(earliest_cut, latest_cut + EPS, stride_sec, dtype=np.float64)

        for cut_time_ts in cut_times:
            observed_points6 = real_points6[real_points6[:, 4] <= cut_time_ts + EPS]

            future_points = generate_future_fixed_points_from_raw(
                raw_arr=raw_arr,
                cut_time_ts=cut_time_ts,
                future_step_minutes=future_step_minutes,
                future_end_time_ts=track_end,
            )
            if len(future_points) == 0:
                continue

            if max_future_steps is not None:
                future_points = future_points[: int(max_future_steps)]
                if len(future_points) == 0:
                    continue

            init_ws = cut_time_ts - history_sec
            init_we = cut_time_ts
            init_mask = (observed_points6[:, 4] >= init_ws - EPS) & (observed_points6[:, 4] <= init_we + EPS)
            init_window_points = observed_points6[init_mask]

            _, _, init_patch_mask, point_count = collect_input_patches_as_feat10(
                points_arr=init_window_points,
                window_start_ts=init_ws,
                input_patch_num=input_patch_num,
                patch_minutes=patch_minutes,
                restore_info=restore_info,
            )

            if point_count < min_total_input_points:
                continue
            if strict and np.sum(init_patch_mask) < input_patch_num:
                continue

            future_labels = []
            future_model_labels = []
            for step_idx in range(len(future_points)):
                current_cut = cut_time_ts + step_idx * future_step_sec
                current_ws = current_cut - history_sec
                label = encode_raw_point_to_feat10(
                    raw_point=future_points[step_idx][:5],
                    sample_start_ts=current_ws,
                    restore_info=restore_info,
                )
                future_labels.append(label.astype(np.float32))
                future_model_labels.append(label[:5].astype(np.float32))

            samples.append(
                {
                    "source_name": str(source_name),
                    "track_id": int(track_id),
                    "cut_time_ts": float(cut_time_ts),
                    "observed_points6": observed_points6.astype(np.float32),
                    "future_points6": future_points.astype(np.float32),
                    "future_labels": np.asarray(future_labels, dtype=np.float32),
                    "future_model_labels": np.asarray(future_model_labels, dtype=np.float32),
                    "restore_info": restore_info.astype(np.float32),
                }
            )

    batch_data = pack_rollout_samples_to_batch(samples)
    dataset = PatchForecastRolloutDataset(batch_data)
    return samples, batch_data, dataset


def build_patch_forecast_dataset_from_raw_tracks(
    tracks_raw,
    source_name: str = "unknown",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
    training_mode: str = "pseudo_recursive",
):
    """
    Unified raw-track entry point for both training modes.

    Parameters are intentionally kept identical to the original script so that
    external training code can keep the same calling convention.

    Returns
    -------
    tuple
        ``(samples, batch_data, dataset)`` for the selected mode.

    Raises
    ------
    ValueError
        If ``training_mode`` is unsupported.
    """
    if training_mode == "pseudo_recursive":
        return build_patch_forecast_dataset_from_raw_tracks_pseudo(
            tracks_raw=tracks_raw,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
        )
    if training_mode == "recursive":
        return build_patch_rollout_dataset_from_raw_tracks(
            tracks_raw=tracks_raw,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
        )
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def build_patch_forecast_dataset_from_csv_single_source(
    csv_path: str = "data.csv",
    source_name: str = "AIS",
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    strict: bool = False,
    pad_value: float = 0.0,
    future_step_minutes: int = 5,
    sample_stride_minutes: int = 5,
    min_total_input_points: int = 1,
    max_future_steps: int | None = None,
    training_mode: str = "pseudo_recursive",
):
    """
    Unified CSV entry point for building datasets from one source column.

    Parameters
    ----------
    csv_path:
        Path to the source CSV file.
    source_name:
        Source column to load.
    input_patch_num, patch_minutes, strict, pad_value, future_step_minutes,
    sample_stride_minutes, min_total_input_points, max_future_steps,
    training_mode:
        Same semantics as the raw-track builder.

    Returns
    -------
    tuple
        ``(samples, batch_data, dataset)`` for the selected mode.
    """
    tracks_raw = load_tracks_from_csv_raw_single_source(csv_path, source_name=source_name)
    return build_patch_forecast_dataset_from_raw_tracks(
        tracks_raw=tracks_raw,
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
