"""Normalization inversion and coordinate restoration helpers."""

from __future__ import annotations

import numpy as np


def inverse_minmax(norm_value, x_min, x_max):
    """
    Convert min-max normalized values back to raw values.

    The function supports both scalar bounds and element-wise array bounds.
    When the bound range is degenerate, the lower bound is returned directly to
    avoid numerical instability.

    Parameters
    ----------
    norm_value:
        Normalized scalar or array in the min-max domain.
    x_min:
        Minimum raw value or array of minima.
    x_max:
        Maximum raw value or array of maxima.

    Returns
    -------
    numpy.ndarray
        Restored raw values rounded to six decimal places.
    """
    norm_value = np.asarray(norm_value, dtype=np.float32)

    if np.isscalar(x_min) and np.isscalar(x_max):
        if abs(x_max - x_min) < 1e-8:
            raw = np.full_like(norm_value, fill_value=x_min, dtype=np.float32)
        else:
            raw = norm_value * (x_max - x_min) + x_min
        return np.round(raw, 6).astype(np.float32)

    x_min = np.asarray(x_min, dtype=np.float32)
    x_max = np.asarray(x_max, dtype=np.float32)
    diff = x_max - x_min

    raw = np.where(np.abs(diff) < 1e-8, x_min, norm_value * diff + x_min)
    return np.round(raw, 6).astype(np.float32)


def restore_pred_lonlat(pred_norm_xy, restore_info):
    """
    Restore normalized longitude/latitude predictions to raw coordinates.

    Parameters
    ----------
    pred_norm_xy:
        Either a 1D array ``[lon_norm, lat_norm]`` or a 2D array with shape
        ``(N, 2)``.
    restore_info:
        Either a 1D array ``[lon_min, lon_max, lat_min, lat_max]`` or a 2D
        array with shape ``(N, 4)``.

    Returns
    -------
    numpy.ndarray
        Restored raw longitude/latitude values with dtype ``float32``.
    """
    pred_norm_xy = np.asarray(pred_norm_xy, dtype=np.float32)
    restore_info = np.asarray(restore_info, dtype=np.float32)

    if pred_norm_xy.ndim == 1:
        lon_norm = pred_norm_xy[0]
        lat_norm = pred_norm_xy[1]
        lon_min, lon_max, lat_min, lat_max = restore_info
        lon_raw = inverse_minmax(lon_norm, lon_min, lon_max)
        lat_raw = inverse_minmax(lat_norm, lat_min, lat_max)
        return np.array([lon_raw, lat_raw], dtype=np.float32)

    lon_norm = pred_norm_xy[:, 0]
    lat_norm = pred_norm_xy[:, 1]
    lon_min = restore_info[:, 0]
    lon_max = restore_info[:, 1]
    lat_min = restore_info[:, 2]
    lat_max = restore_info[:, 3]

    lon_raw = inverse_minmax(lon_norm, lon_min, lon_max)
    lat_raw = inverse_minmax(lat_norm, lat_min, lat_max)
    return np.stack([lon_raw, lat_raw], axis=1).astype(np.float32)


def get_track_restore_info(raw_arr: np.ndarray) -> np.ndarray:
    """
    Compute per-track longitude/latitude min-max bounds.

    These four values are stored alongside each sample so model outputs can be
    restored from normalized coordinates back to raw geographic coordinates.

    Parameters
    ----------
    raw_arr:
        Trajectory array with columns ``[lon, lat, sog, cog, timestamp]``.

    Returns
    -------
    numpy.ndarray
        Array ``[lon_min, lon_max, lat_min, lat_max]`` with dtype ``float32``.
    """
    lon = raw_arr[:, 0]
    lat = raw_arr[:, 1]
    return np.array([np.min(lon), np.max(lon), np.min(lat), np.max(lat)], dtype=np.float32)
