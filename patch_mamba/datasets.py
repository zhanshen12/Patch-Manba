"""Dataset containers and splitting utilities for modular Patch-Mamba training."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, Sequence

import torch
from torch.utils.data import Dataset


class PatchForecastTrainDataset(Dataset):
    """Single-view dataset for one-step pseudo-recursive training.

    The incoming ``batch_data`` dictionary is expected to already be padded and
    stacked by the upstream dataset builder. This class simply converts each
    array-like field into a tensor and returns dictionary-style samples to match
    the rest of the project.
    """

    def __init__(self, batch_data: Dict[str, object]) -> None:
        """Create the dataset from a packed batch dictionary.

        Parameters
        ----------
        batch_data:
            Dictionary produced by ``pack_samples_to_batch`` containing model
            inputs, masks, labels, and restoration statistics.
        """
        self.model_input = torch.tensor(batch_data["model_input"], dtype=torch.float32)
        self.sequence_mask = torch.tensor(batch_data["sequence_mask"], dtype=torch.float32)
        self.patch_index = torch.tensor(batch_data["patch_index"], dtype=torch.long)
        self.patch_mask = torch.tensor(batch_data["patch_mask"], dtype=torch.float32)
        self.model_label = torch.tensor(batch_data["model_label"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples stored in the dataset."""
        return self.model_input.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one training sample as a dictionary of tensors.

        Parameters
        ----------
        idx:
            Integer sample index.

        Returns
        -------
        dict
            Sample containing padded sequence features, patch annotations,
            single-step target, and restoration statistics.
        """
        return {
            "model_input": self.model_input[idx],
            "sequence_mask": self.sequence_mask[idx],
            "patch_index": self.patch_index[idx],
            "patch_mask": self.patch_mask[idx],
            "model_label": self.model_label[idx],
            "restore_info": self.restore_info[idx],
        }


class MultiViewPatchForecastTrainDataset(Dataset):
    """Aligned multi-view dataset for hybrid multi-window fusion training.

    Each sample contains one shared label plus multiple branch-specific input
    streams. Branch tensors are stored under flattened keys such as
    ``view_12x15__model_input`` so that the batch format remains a plain Python
    dictionary compatible with the existing training code.
    """

    def __init__(self, branch_batch_data: Dict[str, dict], branch_auto_patch_len: Dict[str, int]) -> None:
        """Construct a multi-view dataset from aligned branch batches.

        Parameters
        ----------
        branch_batch_data:
            Mapping from branch name to packed batch data.
        branch_auto_patch_len:
            Mapping from branch name to automatically estimated patch length.
            The value is stored for bookkeeping even though it is not directly
            consumed during item retrieval.
        """
        branch_names = list(branch_batch_data.keys())
        if not branch_names:
            raise ValueError("branch_batch_data must not be empty.")

        self.branch_names = branch_names
        base_name = branch_names[0]
        base_batch = branch_batch_data[base_name]
        self.model_label = torch.tensor(base_batch["model_label"], dtype=torch.float32)
        self.restore_info = torch.tensor(base_batch["restore_info"], dtype=torch.float32)

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
        """Return the total number of aligned multi-view samples."""
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one aligned sample with shared targets and branch-specific inputs.

        Parameters
        ----------
        idx:
            Integer sample index.

        Returns
        -------
        dict
            Dictionary containing the shared target fields plus one input branch
            per view.
        """
        output = {
            "model_label": self.model_label[idx],
            "restore_info": self.restore_info[idx],
        }
        for name in self.branch_names:
            output[f"{name}__model_input"] = self.branch_model_input[name][idx]
            output[f"{name}__sequence_mask"] = self.branch_sequence_mask[name][idx]
            output[f"{name}__patch_index"] = self.branch_patch_index[name][idx]
            output[f"{name}__patch_mask"] = self.branch_patch_mask[name][idx]
        return output


class RolloutTrainDataset(Dataset):
    """Dataset for true recursive rollout training and evaluation.

    Each item contains the observed trajectory prefix, the full future trajectory,
    model-space future labels, validity masks for variable rollout lengths, and
    restoration statistics required for metric computation.
    """

    def __init__(self, batch_data: Dict[str, object]) -> None:
        """Create a rollout dataset from a packed batch dictionary.

        Parameters
        ----------
        batch_data:
            Dictionary returned by ``load_rollout_dataset_from_csv``.
        """
        self.observed_points6 = torch.tensor(batch_data["observed_points6"], dtype=torch.float32)
        self.observed_points6_mask = torch.tensor(batch_data["observed_points6_mask"], dtype=torch.float32)
        self.future_points6 = torch.tensor(batch_data["future_points6"], dtype=torch.float32)
        self.future_labels = torch.tensor(batch_data["future_labels"], dtype=torch.float32)
        self.future_model_labels = torch.tensor(batch_data["future_model_labels"], dtype=torch.float32)
        self.rollout_mask = torch.tensor(batch_data["rollout_mask"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.cut_time_ts = torch.tensor(batch_data["cut_time_ts"], dtype=torch.float64)

    def __len__(self) -> int:
        """Return the number of rollout samples."""
        return self.future_model_labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one rollout training sample.

        Parameters
        ----------
        idx:
            Integer sample index.

        Returns
        -------
        dict
            Dictionary containing observed history, rollout targets, masks, and
            restoration metadata.
        """
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
    """Minimal dataset wrapper that exposes a subset by explicit indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        """Store the parent dataset and the selected indices.

        Parameters
        ----------
        dataset:
            Original dataset.
        indices:
            Sequence of integer indices to expose.
        """
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        """Return the size of the subset."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Return the item from the parent dataset addressed by the subset index."""
        return self.dataset[self.indices[idx]]



def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    group_keys: Sequence[object] | None = None,
):
    """Split a dataset into train, validation, and test subsets.

    The function supports two strategies:

    * Plain random sample splitting when ``group_keys`` is ``None``.
    * Group-aware splitting when related samples should stay together, for
      example samples originating from the same trajectory.

    Parameters
    ----------
    dataset:
        Input dataset to split.
    train_ratio:
        Fraction of groups or samples assigned to the training subset.
    val_ratio:
        Fraction of groups or samples assigned to the validation subset.
    seed:
        Random seed used for the split procedure.
    group_keys:
        Optional per-sample group key sequence. When provided, all samples with
        the same key are kept in the same subset.

    Returns
    -------
    tuple
        ``(train_set, val_set, test_set)`` as ``SubsetByIndices`` objects.
    """
    n_total = len(dataset)
    if n_total == 0:
        empty = SubsetByIndices(dataset, [])
        return empty, empty, empty

    if group_keys is None:
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_ids = indices[:n_train]
        val_ids = indices[n_train:n_train + n_val]
        test_ids = indices[n_train + n_val:]
    else:
        if len(group_keys) != n_total:
            raise ValueError(f"group_keys length {len(group_keys)} does not match dataset length {n_total}.")

        group_to_indices = defaultdict(list)
        for idx, group_key in enumerate(group_keys):
            group_to_indices[group_key].append(idx)

        group_list = list(group_to_indices.keys())
        rng = random.Random(seed)
        rng.shuffle(group_list)
        n_groups = len(group_list)

        if n_groups == 1:
            train_groups = group_list
            val_groups = []
            test_groups = []
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

            train_groups = group_list[:n_train_groups]
            val_groups = group_list[n_train_groups:n_train_groups + n_val_groups]
            test_groups = group_list[n_train_groups + n_val_groups:]

            if len(test_groups) == 0:
                if len(val_groups) > 1:
                    test_groups = [val_groups[-1]]
                    val_groups = val_groups[:-1]
                elif len(train_groups) > 1:
                    test_groups = [train_groups[-1]]
                    train_groups = train_groups[:-1]

        train_ids = [i for group in train_groups for i in group_to_indices[group]]
        val_ids = [i for group in val_groups for i in group_to_indices[group]]
        test_ids = [i for group in test_groups for i in group_to_indices[group]]

    return (
        SubsetByIndices(dataset, train_ids),
        SubsetByIndices(dataset, val_ids),
        SubsetByIndices(dataset, test_ids),
    )
