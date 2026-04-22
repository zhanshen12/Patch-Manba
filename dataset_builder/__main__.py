"""Package entry point mirroring the original script's runnable behavior."""

from __future__ import annotations

import numpy as np
import torch

from .export import build_and_save_source_multiscale, default_window_configs


def main() -> None:
    """Run the same default export example that existed in the original script."""
    np.random.seed(42)
    torch.manual_seed(42)

    summary_df = build_and_save_source_multiscale(
        csv_path="data41.csv",
        output_dir="prebuilt_source_csv",
        source_name="AIS",
        window_configs=default_window_configs(),
        strict=False,
        pad_value=0.0,
        future_step_minutes=5,
        sample_stride_minutes=5,
        min_total_input_points=1,
        max_future_steps=12,
        training_mode="pseudo_recursive",
    )

    print("\nSummary results:")
    print(summary_df)


if __name__ == "__main__":
    main()
