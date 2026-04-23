"""
Example script for preset-based multi-scale comparison.
"""

from patch_mamba_training import run_multiscale_comparison_experiment


if __name__ == "__main__":
    result = run_multiscale_comparison_experiment(
        source_name="AIS",
        output_root="patch_mamba_window_fusion_compare_output_180_60",
        common_train_kwargs={
            "training_mode": "pseudo_recursive",
            "future_step_minutes": 5,
            "epochs": 50,
            "train_batch_size": 512,
            "eval_batch_size": 512,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "tau_seconds": 300.0,
            "prebuilt_dir": "prebuilt_source_csv",
        },
    )
    print(result)
