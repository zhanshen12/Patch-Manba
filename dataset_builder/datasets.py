"""PyTorch dataset wrappers for flat and rollout sample batches."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchForecastDataset(Dataset):
    """
    Dataset wrapper for flat pseudo-recursive forecasting samples.

    The constructor expects the padded batch dictionary produced by
    :func:`pack_samples_to_batch`. Each item returns both the full feature
    sequence and the truncated model-input sequence, together with masks,
    labels, restoration information, and metadata.
    """

    def __init__(self, batch_data):
        """
        Initialize the dataset from a padded batch dictionary.

        Parameters
        ----------
        batch_data:
            Output dictionary produced by ``pack_samples_to_batch``.
        """
        self.data_sequence = torch.tensor(batch_data["data_sequence"], dtype=torch.float32)
        self.model_input = torch.tensor(batch_data["model_input"], dtype=torch.float32)
        self.sequence_mask = torch.tensor(batch_data["sequence_mask"], dtype=torch.float32)
        self.patch_index = torch.tensor(batch_data["patch_index"], dtype=torch.long)
        self.patch_mask = torch.tensor(batch_data["patch_mask"], dtype=torch.float32)
        self.label = torch.tensor(batch_data["label"], dtype=torch.float32)
        self.model_label = torch.tensor(batch_data["model_label"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.track_id = torch.tensor(
            batch_data.get("track_id", np.zeros((len(self.data_sequence),), dtype=np.int64)),
            dtype=torch.long,
        )
        self.source_name = list(batch_data.get("source_name", ["unknown"] * len(self.data_sequence)))

    def __len__(self) -> int:
        """Return the number of samples stored in the dataset."""
        return len(self.data_sequence)

    def __getitem__(self, idx: int):
        """Return one flat forecasting sample by index."""
        return {
            "data_sequence": self.data_sequence[idx],
            "model_input": self.model_input[idx],
            "sequence_mask": self.sequence_mask[idx],
            "patch_index": self.patch_index[idx],
            "patch_mask": self.patch_mask[idx],
            "label": self.label[idx],
            "model_label": self.model_label[idx],
            "restore_info": self.restore_info[idx],
            "track_id": self.track_id[idx],
            "source_name": self.source_name[idx],
        }


class PatchForecastRolloutDataset(Dataset):
    """
    Dataset wrapper for true recursive rollout samples.

    Each item contains the observed history, the full future rollout target
    sequence, masks, restoration information, and source metadata.
    """

    def __init__(self, batch_data):
        """
        Initialize the rollout dataset from a padded batch dictionary.

        Parameters
        ----------
        batch_data:
            Output dictionary produced by ``pack_rollout_samples_to_batch``.
        """
        self.observed_points6 = torch.tensor(batch_data["observed_points6"], dtype=torch.float32)
        self.observed_points6_mask = torch.tensor(batch_data["observed_points6_mask"], dtype=torch.float32)
        self.future_points6 = torch.tensor(batch_data["future_points6"], dtype=torch.float32)
        self.future_labels = torch.tensor(batch_data["future_labels"], dtype=torch.float32)
        self.future_model_labels = torch.tensor(batch_data["future_model_labels"], dtype=torch.float32)
        self.rollout_mask = torch.tensor(batch_data["rollout_mask"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.cut_time_ts = torch.tensor(batch_data["cut_time_ts"], dtype=torch.float64)
        self.source_name = list(batch_data["source_name"])
        self.track_id = torch.tensor(batch_data["track_id"], dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of rollout samples stored in the dataset."""
        return len(self.future_model_labels)

    def __getitem__(self, idx: int):
        """Return one rollout sample by index."""
        return {
            "observed_points6": self.observed_points6[idx],
            "observed_points6_mask": self.observed_points6_mask[idx],
            "future_points6": self.future_points6[idx],
            "future_labels": self.future_labels[idx],
            "future_model_labels": self.future_model_labels[idx],
            "rollout_mask": self.rollout_mask[idx],
            "restore_info": self.restore_info[idx],
            "cut_time_ts": self.cut_time_ts[idx],
            "track_id": self.track_id[idx],
            "source_name": self.source_name[idx],
        }
