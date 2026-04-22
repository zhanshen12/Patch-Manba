"""Open-source style modular dataset builder for patch-based trajectory forecasting.

This package is a modular refactor of the original monolithic script. The
public API intentionally preserves the original high-level function names so
existing training code can migrate with minimal changes.
"""

from .batching import pack_rollout_samples_to_batch, pack_samples_to_batch
from .builders import (
    build_patch_forecast_dataset_from_csv_single_source,
    build_patch_forecast_dataset_from_raw_tracks,
    build_patch_forecast_dataset_from_raw_tracks_pseudo,
    build_patch_rollout_dataset_from_raw_tracks,
)
from .constants import EARTH_RADIUS_M, EPS, KNOT_TO_MPS
from .datasets import PatchForecastDataset, PatchForecastRolloutDataset
from .export import build_and_save_source_multiscale, build_output_csv_path, default_window_configs
from .io_utils import (
    load_tracks_from_csv_raw_single_source,
    parse_track_cell_raw,
    read_csv_auto_encoding,
    resolve_source_column,
)
from .restore import get_track_restore_info, inverse_minmax, restore_pred_lonlat
from .serialization import (
    load_rollout_dataset_from_csv,
    load_rollout_samples_from_csv,
    load_saved_dataset_from_csv,
    load_saved_samples_from_csv,
    rollout_samples_to_dataframe,
    samples_to_dataframe,
    save_rollout_samples_to_csv,
    save_samples_to_csv,
)
from .trajectory import (
    append_interp_flag,
    build_recursive_mixed_points,
    collect_input_patches_as_feat10,
    encode_raw_point_to_feat10,
    generate_future_fixed_points_from_raw,
    project_point_by_sog_cog,
    sort_points6,
)
from .utils import clean_time_string, json_to_ndarray, ndarray_to_json

__all__ = [
    "EARTH_RADIUS_M",
    "KNOT_TO_MPS",
    "EPS",
    "clean_time_string",
    "inverse_minmax",
    "restore_pred_lonlat",
    "get_track_restore_info",
    "parse_track_cell_raw",
    "read_csv_auto_encoding",
    "resolve_source_column",
    "load_tracks_from_csv_raw_single_source",
    "project_point_by_sog_cog",
    "encode_raw_point_to_feat10",
    "append_interp_flag",
    "sort_points6",
    "build_recursive_mixed_points",
    "generate_future_fixed_points_from_raw",
    "collect_input_patches_as_feat10",
    "pack_samples_to_batch",
    "pack_rollout_samples_to_batch",
    "PatchForecastDataset",
    "PatchForecastRolloutDataset",
    "build_patch_forecast_dataset_from_raw_tracks_pseudo",
    "build_patch_rollout_dataset_from_raw_tracks",
    "build_patch_forecast_dataset_from_raw_tracks",
    "build_patch_forecast_dataset_from_csv_single_source",
    "ndarray_to_json",
    "json_to_ndarray",
    "samples_to_dataframe",
    "save_samples_to_csv",
    "load_saved_samples_from_csv",
    "load_saved_dataset_from_csv",
    "rollout_samples_to_dataframe",
    "save_rollout_samples_to_csv",
    "load_rollout_samples_from_csv",
    "load_rollout_dataset_from_csv",
    "build_output_csv_path",
    "default_window_configs",
    "build_and_save_source_multiscale",
]
