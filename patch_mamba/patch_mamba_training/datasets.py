"""
Dataset classes and split helpers for Patch-Mamba training.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset


def build_track_group_keys_from_samples(samples: Sequence[dict]) -> List[Tuple[str, int]]:
    """
    Build group keys used for track-level train/validation/test splitting.
    """
    group_keys = []
    for i, sample in enumerate(samples):
        source_name = str(sample.get("source_name", "unknown")).strip().lower()
        track_id = int(sample.get("track_id", i))
        group_keys.append((source_name, track_id))
    return group_keys


class PatchForecastTrainDataset(Dataset):
    """
    Dataset for flat pseudo-recursive single-step training samples.
    """

    def __init__(self, batch_data: Dict[str, np.ndarray]) -> None:
        self.model_input = torch.tensor(batch_data["model_input"], dtype=torch.float32)
        self.sequence_mask = torch.tensor(batch_data["sequence_mask"], dtype=torch.float32)
        self.patch_index = torch.tensor(batch_data["patch_index"], dtype=torch.long)
        self.patch_mask = torch.tensor(batch_data["patch_mask"], dtype=torch.float32)
        self.model_label = torch.tensor(batch_data["model_label"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)

    def __len__(self) -> int:
        return self.model_input.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "model_input": self.model_input[idx],
            "sequence_mask": self.sequence_mask[idx],
            "patch_index": self.patch_index[idx],
            "patch_mask": self.patch_mask[idx],
            "model_label": self.model_label[idx],
            "restore_info": self.restore_info[idx],
        }


class MultiViewPatchForecastTrainDataset(Dataset):
    """
    Dataset for aligned multi-window pseudo-recursive training samples.
    """

    def __init__(self, branch_batch_data: Dict[str, dict], branch_auto_patch_len: Dict[str, int]) -> None:
        branch_names = list(branch_batch_data.keys())
        if not branch_names:
            raise ValueError("`branch_batch_data` must not be empty.")

        self.branch_names = branch_names
        base_name = branch_names[0]
        base = branch_batch_data[base_name]
        self.model_label = torch.tensor(base["model_label"], dtype=torch.float32)
        self.restore_info = torch.tensor(base["restore_info"], dtype=torch.float32)

        self.branch_model_input = {}
        self.branch_sequence_mask = {}
        self.branch_patch_index = {}
        self.branch_patch_mask = {}
        self.branch_auto_patch_len = {key: int(value) for key, value in branch_auto_patch_len.items()}

        for name, batch_data in branch_batch_data.items():
            self.branch_model_input[name] = torch.tensor(batch_data["model_input"], dtype=torch.float32)
            self.branch_sequence_mask[name] = torch.tensor(batch_data["sequence_mask"], dtype=torch.float32)
            self.branch_patch_index[name] = torch.tensor(batch_data["patch_index"], dtype=torch.long)
            self.branch_patch_mask[name] = torch.tensor(batch_data["patch_mask"], dtype=torch.float32)

        self.length = self.model_label.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {
            "model_label": self.model_label[idx],
            "restore_info": self.restore_info[idx],
        }
        for name in self.branch_names:
            out[f"{name}__model_input"] = self.branch_model_input[name][idx]
            out[f"{name}__sequence_mask"] = self.branch_sequence_mask[name][idx]
            out[f"{name}__patch_index"] = self.branch_patch_index[name][idx]
            out[f"{name}__patch_mask"] = self.branch_patch_mask[name][idx]
        return out


class RolloutTrainDataset(Dataset):
    """
    Dataset for recursive rollout training.
    """

    def __init__(self, batch_data: Dict[str, np.ndarray]) -> None:
        self.observed_points6 = torch.tensor(batch_data["observed_points6"], dtype=torch.float32)
        self.observed_points6_mask = torch.tensor(batch_data["observed_points6_mask"], dtype=torch.float32)
        self.future_points6 = torch.tensor(batch_data["future_points6"], dtype=torch.float32)
        self.future_labels = torch.tensor(batch_data["future_labels"], dtype=torch.float32)
        self.future_model_labels = torch.tensor(batch_data["future_model_labels"], dtype=torch.float32)
        self.rollout_mask = torch.tensor(batch_data["rollout_mask"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.cut_time_ts = torch.tensor(batch_data["cut_time_ts"], dtype=torch.float64)

    def __len__(self) -> int:
        return self.future_model_labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "observed_points6": self.observed_points6[idx],
            "observed_points6_mask": self.observed_points6_mask[idx],
            "future_points6": self.future_points6[idx],
            "future_labels": self.future_labels[idx],
            "future_model_labels": self.future_model_labels[idx],
            "rollout_mask": self.rollout_mask[idx],
            "restore_info": self.restore_info[idx],
            "cut_time_ts": self.cut_time_ts[idx],
        }


class SubsetByIndices(Dataset):
    """
    Simple dataset wrapper that exposes a fixed list of sample indices.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    group_keys: Optional[Sequence[Tuple[str, int]]] = None,
):
    """
    Split a dataset into train, validation, and test subsets.
    """
    n_total = len(dataset)
    if n_total == 0:
        empty = SubsetByIndices(dataset, [])
        return empty, empty, empty

    if group_keys is None:
        idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_ids = idx[:n_train]
        val_ids = idx[n_train:n_train + n_val]
        test_ids = idx[n_train + n_val:]
    else:
        if len(group_keys) != n_total:
            raise ValueError(f"group_keys length {len(group_keys)} does not match dataset length {n_total}")

        group_to_indices = defaultdict(list)
        for idx, group_key in enumerate(group_keys):
            group_to_indices[group_key].append(idx)

        groups = list(group_to_indices.keys())
        rng = random.Random(seed)
        rng.shuffle(groups)
        n_groups = len(groups)

        if n_groups == 1:
            train_groups, val_groups, test_groups = groups, [], []
        else:
            n_train_groups = int(n_groups * train_ratio)
            n_val_groups = int(n_groups * val_ratio)

            if n_groups >= 3:
                n_train_groups = max(1, n_train_groups)
                n_val_groups = max(1, n_val_groups)
                if n_train_groups + n_val_groups >= n_groups:
                    n_val_groups = max(1, n_groups - n_train_groups - 1)
                    if n_train_groups + n_val_groups >= n_groups:
                        n_train_groups = max(1, n_groups - n_val_groups - 1)
            else:
                n_train_groups = 1
                n_val_groups = 0

            train_groups = groups[:n_train_groups]
            val_groups = groups[n_train_groups:n_train_groups + n_val_groups]
            test_groups = groups[n_train_groups + n_val_groups:]

            if not test_groups:
                if len(val_groups) > 1:
                    test_groups = [val_groups[-1]]
                    val_groups = val_groups[:-1]
                elif len(train_groups) > 1:
                    test_groups = [train_groups[-1]]
                    train_groups = train_groups[:-1]

        train_ids = [i for g in train_groups for i in group_to_indices[g]]
        val_ids = [i for g in val_groups for i in group_to_indices[g]]
        test_ids = [i for g in test_groups for i in group_to_indices[g]]

    return (
        SubsetByIndices(dataset, train_ids),
        SubsetByIndices(dataset, val_ids),
        SubsetByIndices(dataset, test_ids),
    )
