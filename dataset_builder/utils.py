"""General utility helpers for string cleanup and JSON-array conversion."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def clean_time_string(time_str: Any) -> str:
    """
    Normalize a raw timestamp string extracted from a trajectory cell.

    The original CSV cells may contain inconsistent quoting such as doubled
    quotation marks or surrounding single/double quotes. This helper converts
    the value to a plain string and removes the extra wrappers so that
    ``datetime.strptime`` can parse it reliably.

    Parameters
    ----------
    time_str:
        Raw timestamp object, typically a string but accepted as any value that
        can be converted to ``str``.

    Returns
    -------
    str
        A cleaned timestamp string with redundant quotes removed.
    """
    s = str(time_str).strip()
    s = s.replace('""', '"')
    s = s.strip('"').strip("'").strip()
    return s


def ndarray_to_json(arr: Any) -> str:
    """
    Serialize a NumPy-compatible array into a JSON string.

    This function is used when exporting prebuilt samples to CSV files. The
    array is first converted into a NumPy array and then written as a nested
    Python list so that the stored CSV remains text-based and easy to inspect.

    Parameters
    ----------
    arr:
        Any object that can be converted to a NumPy array.

    Returns
    -------
    str
        A JSON string representation of the input array.
    """
    arr = np.asarray(arr)
    return json.dumps(arr.tolist(), ensure_ascii=False)


def json_to_ndarray(s: str, dtype: Any = np.float32) -> np.ndarray:
    """
    Deserialize a JSON string into a NumPy array.

    Parameters
    ----------
    s:
        JSON string produced by :func:`ndarray_to_json`.
    dtype:
        Target NumPy dtype for the reconstructed array.

    Returns
    -------
    numpy.ndarray
        Reconstructed array cast to ``dtype``.
    """
    return np.asarray(json.loads(s), dtype=dtype)
