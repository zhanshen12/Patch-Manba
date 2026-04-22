"""Trajectory projection and feature-building helpers."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .constants import EARTH_RADIUS_M, EPS, KNOT_TO_MPS


def project_point_by_sog_cog(
    lon_deg: float,
    lat_deg: float,
    sog_knots: float,
    cog_deg: float,
    delta_seconds: float,
) -> Tuple[float, float]:
    """
    Project a geographic point forward using SOG/COG motion.

    The implementation uses spherical forward navigation: the heading is given
    by COG, the traveled distance is derived from SOG and time, and the next
    point is computed on a sphere with mean Earth radius.

    Parameters
    ----------
    lon_deg:
        Starting longitude in degrees.
    lat_deg:
        Starting latitude in degrees.
    sog_knots:
        Speed over ground in knots.
    cog_deg:
        Course over ground in degrees.
    delta_seconds:
        Forward time interval in seconds.

    Returns
    -------
    tuple[float, float]
        Projected longitude and latitude in degrees.
    """
    if delta_seconds <= 0:
        return float(lon_deg), float(lat_deg)

    speed_mps = max(float(sog_knots), 0.0) * KNOT_TO_MPS
    distance_m = speed_mps * float(delta_seconds)

    brng = math.radians(cog_deg % 360.0)
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    ang = distance_m / EARTH_RADIUS_M

    sin_lat2 = math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brng)
    sin_lat2 = min(1.0, max(-1.0, sin_lat2))
    lat2 = math.asin(sin_lat2)

    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return float(math.degrees(lon2)), float(math.degrees(lat2))


def encode_raw_point_to_feat10(raw_point: np.ndarray, sample_start_ts: float, restore_info: np.ndarray) -> np.ndarray:
    """
    Encode one raw trajectory point into the module's 10-dimensional feature vector.

    The feature layout is preserved exactly from the original script:

    ``[lon_norm, lat_norm, sog/10, sin(cog), cos(cog), relative_time_min,
    lon_min, lon_max, lat_min, lat_max]``.

    Parameters
    ----------
    raw_point:
        Raw point with at least five elements ``[lon, lat, sog, cog, ts]``.
    sample_start_ts:
        Timestamp used as the zero point for relative time encoding.
    restore_info:
        Per-sample min-max information
        ``[lon_min, lon_max, lat_min, lat_max]``.

    Returns
    -------
    numpy.ndarray
        Feature vector of shape ``(10,)`` with dtype ``float32``.
    """
    lon, lat, sog, cog, ts = raw_point[:5]
    lon_min, lon_max, lat_min, lat_max = restore_info

    lon_norm = 0.0 if abs(lon_max - lon_min) < 1e-8 else (lon - lon_min) / (lon_max - lon_min)
    lat_norm = 0.0 if abs(lat_max - lat_min) < 1e-8 else (lat - lat_min) / (lat_max - lat_min)

    sog_div10 = sog / 10.0
    cog_rad = np.deg2rad(cog)
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)
    rel_time_min = (ts - sample_start_ts) / 60.0

    feat = np.array(
        [lon_norm, lat_norm, sog_div10, cog_sin, cog_cos, rel_time_min, lon_min, lon_max, lat_min, lat_max],
        dtype=np.float32,
    )
    return np.round(feat, 5).astype(np.float32)


def append_interp_flag(raw_arr: np.ndarray) -> np.ndarray:
    """
    Append an interpolation flag column to a raw trajectory array.

    Real observed points receive a zero flag. Generated future points may use
    the same six-column format with a non-zero flag, which allows downstream
    logic to preserve whether a point was directly observed or synthesized.

    Parameters
    ----------
    raw_arr:
        Raw trajectory array with shape ``(N, 5)``.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(N, 6)`` where the last column is the interpolation
        flag initialized to zero.
    """
    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    return np.concatenate([raw_arr, np.zeros((len(raw_arr), 1), dtype=np.float32)], axis=1)


def sort_points6(points_arr: np.ndarray | None) -> np.ndarray:
    """
    Sort six-column point arrays by timestamp.

    Parameters
    ----------
    points_arr:
        Array with columns ``[lon, lat, sog, cog, ts, interp_flag]``.

    Returns
    -------
    numpy.ndarray
        Time-sorted array with dtype ``float32``. An empty array is returned if
        the input is empty or ``None``.
    """
    if points_arr is None or len(points_arr) == 0:
        return np.empty((0, 6), dtype=np.float32)
    points_arr = np.asarray(points_arr, dtype=np.float32)
    return points_arr[np.argsort(points_arr[:, 4])]


def build_recursive_mixed_points(observed_points6: np.ndarray, generated_points6: np.ndarray) -> np.ndarray:
    """
    Merge observed and generated points into one time-sorted sequence.

    This helper is used in pseudo-recursive sample construction where previous
    generated future points are fed back into later input windows.

    Parameters
    ----------
    observed_points6:
        Observed six-column points.
    generated_points6:
        Previously generated six-column points.

    Returns
    -------
    numpy.ndarray
        Concatenated and time-sorted six-column array.
    """
    observed_points6 = sort_points6(observed_points6)
    generated_points6 = sort_points6(generated_points6)

    if len(observed_points6) == 0 and len(generated_points6) == 0:
        return np.empty((0, 6), dtype=np.float32)
    if len(observed_points6) == 0:
        return generated_points6.astype(np.float32)
    if len(generated_points6) == 0:
        return observed_points6.astype(np.float32)

    mixed = np.concatenate([observed_points6, generated_points6], axis=0)
    return sort_points6(mixed).astype(np.float32)


def generate_future_fixed_points_from_raw(
    raw_arr: np.ndarray,
    cut_time_ts: float,
    future_step_minutes: int = 5,
    future_end_time_ts: float | None = None,
) -> np.ndarray:
    """
    Generate fixed-step future points from the raw trajectory.

    This function preserves the original pseudo-recursive label generation
    strategy: each future point is generated independently from the latest real
    point available at or before the target time. The labels are therefore not
    recursively generated from previous predictions.

    Parameters
    ----------
    raw_arr:
        Raw trajectory array with columns ``[lon, lat, sog, cog, timestamp]``.
    cut_time_ts:
        Current cut time separating input history from future targets.
    future_step_minutes:
        Temporal spacing between generated future points.
    future_end_time_ts:
        Optional final future timestamp. If omitted, the last timestamp of the
        trajectory is used.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(T, 6)`` containing
        ``[lon, lat, sog, cog, ts, interp_flag]``.
    """
    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]

    if future_end_time_ts is None:
        future_end_time_ts = raw_arr[-1, 4]

    step_sec = int(future_step_minutes * 60)
    if future_end_time_ts - cut_time_ts < step_sec - EPS:
        return np.empty((0, 6), dtype=np.float32)

    target_times = np.arange(cut_time_ts + step_sec, future_end_time_ts + EPS, step_sec, dtype=np.float64)
    out = []

    for tgt_ts in target_times:
        src_idx = np.searchsorted(raw_arr[:, 4], tgt_ts, side="right") - 1
        if src_idx < 0:
            continue

        base_lon, base_lat, base_sog, base_cog, base_ts = raw_arr[src_idx]
        if abs(base_ts - tgt_ts) < EPS:
            lon_t, lat_t = base_lon, base_lat
            sog_t, cog_t = base_sog, base_cog
            interp_flag = 0.0
        else:
            lon_t, lat_t = project_point_by_sog_cog(base_lon, base_lat, base_sog, base_cog, tgt_ts - base_ts)
            sog_t, cog_t = base_sog, base_cog
            interp_flag = 1.0

        out.append([lon_t, lat_t, sog_t, cog_t, tgt_ts, interp_flag])

    if len(out) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def collect_input_patches_as_feat10(
    points_arr: np.ndarray,
    window_start_ts: float,
    input_patch_num: int = 12,
    patch_minutes: int = 15,
    restore_info: np.ndarray | None = None,
):
    """
    Collect all points that fall inside each input patch and encode them.

    Parameters
    ----------
    points_arr:
        Input points in six-column format ``[lon, lat, sog, cog, ts, flag]``.
    window_start_ts:
        Start timestamp of the entire input window.
    input_patch_num:
        Number of temporal patches inside the input window.
    patch_minutes:
        Duration of each patch in minutes.
    restore_info:
        Min-max coordinate bounds used for feature encoding.

    Returns
    -------
    tuple
        ``(data_sequence, patch_index, patch_mask, point_count)`` where

        - ``data_sequence`` has shape ``(L, 10)``;
        - ``patch_index`` has shape ``(L,)`` and stores 1-based patch ids;
        - ``patch_mask`` has shape ``(input_patch_num,)``;
        - ``point_count`` is the number of encoded points.
    """
    patch_sec = int(patch_minutes * 60)
    all_feats = []
    all_patch_ids = []
    patch_mask = np.zeros((input_patch_num,), dtype=np.float32)

    if points_arr is None or len(points_arr) == 0:
        return np.empty((0, 10), dtype=np.float32), np.empty((0,), dtype=np.int64), patch_mask, 0

    points_arr = np.asarray(points_arr, dtype=np.float32)
    points_arr = points_arr[np.argsort(points_arr[:, 4])]

    for patch_id in range(input_patch_num):
        left = window_start_ts + patch_id * patch_sec
        right = left + patch_sec
        if patch_id < input_patch_num - 1:
            mask = (points_arr[:, 4] >= left - EPS) & (points_arr[:, 4] < right - EPS)
        else:
            mask = (points_arr[:, 4] >= left - EPS) & (points_arr[:, 4] <= right + EPS)

        patch_points = points_arr[mask]
        if len(patch_points) > 0:
            patch_mask[patch_id] = 1.0
            for point in patch_points:
                feat10 = encode_raw_point_to_feat10(
                    raw_point=point[:5],
                    sample_start_ts=window_start_ts,
                    restore_info=restore_info,
                )
                all_feats.append(feat10)
                all_patch_ids.append(patch_id + 1)

    if len(all_feats) == 0:
        return np.empty((0, 10), dtype=np.float32), np.empty((0,), dtype=np.int64), patch_mask, 0

    data_sequence = np.stack(all_feats, axis=0).astype(np.float32)
    patch_index = np.asarray(all_patch_ids, dtype=np.int64)
    return data_sequence, patch_index, patch_mask, len(data_sequence)
