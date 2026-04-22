# Patch-Mamba Training Module Description

The Patch-Mamba Training Module is a modular training component for patch-based maritime trajectory forecasting.

It organizes the full training workflow into clear functional units, including data loading, dataset management, neural network layers, model definitions, recursive rollout construction, training and evaluation loops, configuration builders, and experiment presets.

The module is designed for two forecasting workflows:

1. **Pseudo-recursive training**, where the model is trained with prebuilt single-step samples.
2. **Recursive training**, where the model is trained through multi-step rollout behavior.

The module also supports two model structures:

1. **Single-view Patch-Mamba**, which uses one temporal patch layout.
2. **Multi-window hybrid Patch-Mamba**, which combines several temporal patch layouts through feature fusion.

This modular structure improves readability, maintainability, and reuse, while preserving the original training interface and experimental behavior.

The primary public entrypoint is:

```python
train_patch_mamba_model(...)
```

For multi-run preset experiments, the module also provides:

```python
run_multiscale_comparison_experiment(...)
```

A compatibility wrapper named `patch_mamba.py` is included so that existing external scripts can continue to import the module using the original entry filename.
