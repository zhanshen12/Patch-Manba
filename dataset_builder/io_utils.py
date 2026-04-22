"""CSV reading and raw trajectory parsing utilities."""

from __future__ import annotations

import ast
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import clean_time_string


def parse_track_cell_raw(cell) -> Optional[np.ndarray]:
    """
    Parse a single trajectory cell into a time-sorted numeric array.

    The expected cell format is a Python-like list where each point contains at
    least ``[lon, lat, sog, cog, time_string]``. Invalid points are skipped,
    and trajectories with fewer than two valid points are discarded.

    Parameters
    ----------
    cell:
        Raw cell content from a CSV column. It may be a string representation of
        a list or an already-materialized Python list.

    Returns
    -------
    numpy.ndarray or None
        A ``float64`` array of shape ``(N, 5)`` with columns
        ``[lon, lat, sog, cog, timestamp]`` sorted by time, or ``None`` if the
        cell cannot produce a valid trajectory.
    """
    if pd.isna(cell):
        return None

    if isinstance(cell, str):
        cell = cell.strip()
        if cell == "":
            return None
        cell = cell.replace('""', '"')

    try:
        traj = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except Exception as exc:
        print(f"Trajectory cell parsing failed: {exc}")
        return None

    if not isinstance(traj, (list, tuple)) or len(traj) == 0:
        return None

    parsed = []
    for point in traj:
        if not isinstance(point, (list, tuple)) or len(point) < 5:
            continue
        try:
            lon = float(point[0])
            lat = float(point[1])
            sog = float(point[2])
            cog = float(point[3])
            time_str = clean_time_string(point[4])
            ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp()
            parsed.append([lon, lat, sog, cog, ts])
        except Exception as exc:
            print(f"Single trajectory point parsing failed, point={point}, error={exc}")
            continue

    if len(parsed) <= 1:
        return None

    arr = np.asarray(parsed, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 4])]
    return arr


def read_csv_auto_encoding(csv_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Read a CSV file using a sequence of common encodings.

    The original script supports datasets saved in different encodings,
    especially GBK-encoded CSV files. This helper preserves that behavior by
    trying several encodings until one succeeds.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        The loaded DataFrame and the encoding that successfully decoded it.

    Raises
    ------
    ValueError
        If the CSV file cannot be decoded by any supported encoding.
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    df = None
    used_encoding = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            used_encoding = enc
            print(f"CSV loaded successfully with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            print(f"Failed to read CSV with encoding {enc}: {exc}")

    if df is None or used_encoding is None:
        raise ValueError("Failed to read the CSV file. Please verify the path, encoding, and file content.")

    df.columns = [str(col).strip() for col in df.columns]
    return df, used_encoding


def resolve_source_column(df: pd.DataFrame, source_name: str) -> str:
    """
    Resolve the actual column name for a requested trajectory source.

    The helper keeps the original compatibility behavior, including the special
    handling of a potential ``rader`` typo in some datasets.

    Parameters
    ----------
    df:
        Input DataFrame.
    source_name:
        Requested source column name such as ``AIS``, ``radar``, or ``bd``.

    Returns
    -------
    str
        The actual column name in the DataFrame.

    Raises
    ------
    ValueError
        If the requested source column is not available.
    """
    source_name = str(source_name).strip()
    actual_columns = list(df.columns)

    if source_name in actual_columns:
        return source_name
    if source_name == "radar" and "rader" in actual_columns:
        return "rader"

    raise ValueError(f"Column {source_name} does not exist. Current columns: {actual_columns}")


def load_tracks_from_csv_raw_single_source(csv_path: str, source_name: str) -> List[np.ndarray]:
    """
    Load all valid raw trajectories for a single source column from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the source CSV file.
    source_name:
        Source column to extract.

    Returns
    -------
    list[numpy.ndarray]
        List of parsed raw trajectories, each stored as ``float32`` and sorted
        by timestamp.
    """
    df, _ = read_csv_auto_encoding(csv_path)
    col = resolve_source_column(df, source_name)

    all_tracks = []
    for _, row in df.iterrows():
        raw_arr = parse_track_cell_raw(row[col])
        if raw_arr is not None and len(raw_arr) > 1:
            all_tracks.append(raw_arr.astype(np.float32))
    return all_tracks
