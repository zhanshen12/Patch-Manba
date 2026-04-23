"""
Example script for single-view pseudo-recursive training.
"""

from patch_mamba_training import train_patch_mamba_model


if __name__ == "__main__":
    result = train_patch_mamba_model(
        source_name="AIS",
        training_mode="pseudo_recursive",
        model_variant="single",
        input_patch_num=12,
        patch_minutes=15,
        prebuilt_dir="prebuilt_source_csv",
        save_dir="patch_mamba_output_single_12x15",
    )
    print(result)
