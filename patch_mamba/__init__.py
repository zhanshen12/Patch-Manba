"""Modular Patch-Mamba training package.

This package is a block-structured reorganization of the original single-file
training script. Public training interfaces and experiment runners are re-
exported here for convenient external use.
"""

from .api import train_patch_mamba_model
from .config_builders import make_model_cfg, make_multiwindow_model_cfg
from .data_builders import (
    build_multiview_dataloaders_from_prebuilt,
    build_recursive_dataloaders_from_rollout_csv,
    build_singleview_dataloaders_from_prebuilt,
)
from .datasets import (
    MultiViewPatchForecastTrainDataset,
    PatchForecastTrainDataset,
    RolloutTrainDataset,
    SubsetByIndices,
    split_dataset,
)
from .layers import HAS_MAMBA, PatchGraphAttention, PositionalEncoding, ResidualMambaBlock
from .models import PatchTTCN_Mamba_Encoder, PatchTTCN_Mamba_TrajPred, PatchTTCN_MultiWindowFusion_TrajPred
from .presets import (
    build_default_multiscale_size_presets_180_60,
    plot_multiscale_compare,
    run_multiscale_comparison_experiment,
)
from .runtime import (
    build_runtime_multiview_batch_from_tensor_points,
    build_runtime_singleview_batch_from_tensor_points,
    collect_input_patches_as_model_input_torch,
    encode_raw_points_to_model_input_torch,
    estimate_runtime_patch_len_from_rollout_samples,
    pred5_to_raw_point6_torch,
    rollout_forward,
)
from .training import evaluate, evaluate_pseudo, evaluate_recursive, train_one_epoch, train_one_epoch_pseudo, train_one_epoch_recursive
from .utils import (
    build_default_prebuilt_csv_path,
    build_sample_key,
    build_track_group_keys_from_samples,
    dtw_distance_np,
    ensure_output_dir,
    estimate_max_patch_len,
    gather_last_valid,
    masked_mean,
    move_batch_to_device,
    normalize_source_name,
    pred5_to_raw_point6,
    restore_pred_lonlat_torch,
    set_seed,
    to_serializable,
    trajectory_metrics,
)

__all__ = [
    "HAS_MAMBA",
    "MultiViewPatchForecastTrainDataset",
    "PatchForecastTrainDataset",
    "PatchGraphAttention",
    "PatchTTCN_Mamba_Encoder",
    "PatchTTCN_Mamba_TrajPred",
    "PatchTTCN_MultiWindowFusion_TrajPred",
    "PositionalEncoding",
    "ResidualMambaBlock",
    "RolloutTrainDataset",
    "SubsetByIndices",
    "build_default_multiscale_size_presets_180_60",
    "build_default_prebuilt_csv_path",
    "build_multiview_dataloaders_from_prebuilt",
    "build_recursive_dataloaders_from_rollout_csv",
    "build_runtime_multiview_batch_from_tensor_points",
    "build_runtime_singleview_batch_from_tensor_points",
    "build_sample_key",
    "build_singleview_dataloaders_from_prebuilt",
    "build_track_group_keys_from_samples",
    "collect_input_patches_as_model_input_torch",
    "dtw_distance_np",
    "encode_raw_points_to_model_input_torch",
    "ensure_output_dir",
    "estimate_max_patch_len",
    "estimate_runtime_patch_len_from_rollout_samples",
    "evaluate",
    "evaluate_pseudo",
    "evaluate_recursive",
    "gather_last_valid",
    "make_model_cfg",
    "make_multiwindow_model_cfg",
    "masked_mean",
    "move_batch_to_device",
    "normalize_source_name",
    "plot_multiscale_compare",
    "pred5_to_raw_point6",
    "pred5_to_raw_point6_torch",
    "restore_pred_lonlat_torch",
    "rollout_forward",
    "run_multiscale_comparison_experiment",
    "set_seed",
    "split_dataset",
    "to_serializable",
    "train_one_epoch",
    "train_one_epoch_pseudo",
    "train_one_epoch_recursive",
    "train_patch_mamba_model",
    "trajectory_metrics",
]
