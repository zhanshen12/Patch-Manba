"""
Compatibility wrapper for legacy external imports.

Historically, external scripts imported the training entrypoints from a
single file named `patch_mamba.py`. This wrapper preserves that import path
while delegating all public functionality to the modular package.
"""

from patch_mamba_training import (
    build_default_multiscale_size_presets_180_60,
    run_multiscale_comparison_experiment,
    train_patch_mamba_model,
)

__all__ = [
    "train_patch_mamba_model",
    "run_multiscale_comparison_experiment",
    "build_default_multiscale_size_presets_180_60",
]
