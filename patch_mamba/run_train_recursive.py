"""
Example script for recursive rollout training.
"""

from patch_mamba_training import train_patch_mamba_model


if __name__ == "__main__":
    result = train_patch_mamba_model(
        source_name="AIS",
        training_mode="recursive",
        model_variant="single",
        input_patch_num=12,
        patch_minutes=15,
        future_step_minutes=5,
        rollout_csv_path="prebuilt_rollout_csv_4h/ais_12batch_15min_5min_recursive.csv",
        save_dir="patch_mamba_output_recursive_12x15",
    )
    print(result)
