"""Encoders and predictor models for the modular Patch-Mamba package."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PatchGraphAttention, PositionalEncoding, ResidualMambaBlock
from .utils import gather_last_valid, masked_mean


class PatchTTCN_Mamba_Encoder(nn.Module):
    """Patch-level trajectory encoder using TTCN aggregation, graph attention, and Mamba.

    The encoder follows the original script closely:

    1. Group raw point features into temporal patches.
    2. Aggregate the variable number of points inside each patch with a TTCN-like
       learned filter mechanism.
    3. Refine patch tokens with time-biased graph attention.
    4. Encode the patch sequence with stacked Mamba blocks.
    5. Return a concatenation of the last valid patch token and the masked mean.
    """

    def __init__(self, cfg: Dict[str, object]) -> None:
        """Initialize the encoder from a configuration dictionary.

        Parameters
        ----------
        cfg:
            Model configuration dictionary. Expected keys include ``in_dim``,
            ``hid_dim``, ``npatch``, ``patch_len``, and the hyperparameters for
            the graph-attention and Mamba stacks.
        """
        super().__init__()
        self.device = cfg["device"]
        self.in_dim = int(cfg["in_dim"])
        self.hid_dim = int(cfg["hid_dim"])
        self.npatch = int(cfg["npatch"])
        self.patch_len = int(cfg["patch_len"])
        self.gnn_layers = int(cfg.get("gnn_layers", 2))
        self.nhead = int(cfg.get("nhead", 4))
        self.tau_seconds = float(cfg.get("tau_seconds", 300.0))
        self.delta_minutes = float(cfg.get("delta_minutes", 5.0))
        self.mamba_layers = int(cfg.get("mamba_layers", 2))
        self.mamba_d_state = int(cfg.get("mamba_d_state", 16))
        self.mamba_d_conv = int(cfg.get("mamba_d_conv", 4))
        self.mamba_expand = int(cfg.get("mamba_expand", 2))
        self.dropout = float(cfg.get("dropout", 0.1))

        self.point_dim = self.in_dim
        self.ttcn_dim = self.hid_dim - 1
        if self.ttcn_dim <= 0:
            raise AssertionError("hid_dim must be at least 2 so one channel can encode patch existence.")

        self.Filter_Generators = nn.Sequential(
            nn.Linear(self.point_dim, self.ttcn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.point_dim * self.ttcn_dim),
        )
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))

        self.patch_gattn = nn.ModuleList([
            PatchGraphAttention(
                d_model=self.hid_dim,
                nhead=self.nhead,
                tau_seconds=self.tau_seconds,
                delta_minutes=self.delta_minutes,
                dropout=self.dropout,
            )
            for _ in range(self.gnn_layers)
        ])
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(self.hid_dim) for _ in range(self.gnn_layers)])
        self.pos_encoding = PositionalEncoding(self.hid_dim, max_len=max(1024, self.npatch + 8))
        self.mamba_stack = nn.ModuleList([
            ResidualMambaBlock(
                d_model=self.hid_dim,
                d_state=self.mamba_d_state,
                d_conv=self.mamba_d_conv,
                expand=self.mamba_expand,
                dropout=self.dropout,
            )
            for _ in range(self.mamba_layers)
        ])

    def patchify_by_patch_id(
        self,
        model_input: torch.Tensor,
        patch_index: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack a padded flat sequence into a dense patch-major tensor layout.

        Parameters
        ----------
        model_input:
            Input tensor of shape ``(batch, seq_len, in_dim)``.
        patch_index:
            Patch identifier tensor of shape ``(batch, seq_len)`` where positive
            values indicate the patch assignment of each point.
        patch_mask:
            Patch validity mask of shape ``(batch, npatch)``.

        Returns
        -------
        tuple
            ``(X_patch, mask_pt, patch_mask_float)`` where ``X_patch`` has shape
            ``(batch, npatch, patch_len, in_dim)`` and ``mask_pt`` indicates
            valid points inside each patch.
        """
        batch_size, _, dim = model_input.shape
        patch_num = patch_mask.shape[1]
        patch_len = self.patch_len
        device = model_input.device

        X_patch = torch.zeros((batch_size, patch_num, patch_len, dim), device=device, dtype=model_input.dtype)
        mask_pt = torch.zeros((batch_size, patch_num, patch_len, 1), device=device, dtype=torch.float32)

        for batch_idx in range(batch_size):
            write_ptr = torch.zeros((patch_num,), dtype=torch.long, device=device)
            for time_idx in range(model_input.shape[1]):
                patch_id = int(patch_index[batch_idx, time_idx].item())
                if patch_id <= 0:
                    continue
                patch_pos = patch_id - 1
                if patch_pos < 0 or patch_pos >= patch_num:
                    continue
                slot = int(write_ptr[patch_pos].item())
                if slot >= patch_len:
                    continue
                X_patch[batch_idx, patch_pos, slot, :] = model_input[batch_idx, time_idx, :]
                mask_pt[batch_idx, patch_pos, slot, 0] = 1.0
                write_ptr[patch_pos] += 1

        return X_patch, mask_pt, patch_mask.float()

    def ttcn(self, X_int: torch.Tensor, mask_X: torch.Tensor) -> torch.Tensor:
        """Aggregate the points inside each patch into one patch feature vector.

        Parameters
        ----------
        X_int:
            Input tensor of shape ``(N, patch_len, point_dim)`` where ``N`` is
            the number of patch instances across the batch.
        mask_X:
            Point-validity mask of shape ``(N, patch_len, 1)``.

        Returns
        -------
        torch.Tensor
            Aggregated patch features of shape ``(N, ttcn_dim)``.
        """
        n_patch_instances, patch_len, _ = X_int.shape
        learned_filter = self.Filter_Generators(X_int)
        learned_filter = learned_filter.view(n_patch_instances, patch_len, self.ttcn_dim, self.point_dim)

        mask_expand = mask_X.unsqueeze(-1)
        filter_mask = learned_filter * mask_expand + (1.0 - mask_expand) * (-1e8)
        filter_weights = torch.softmax(filter_mask, dim=1)

        x_broadcast = X_int.unsqueeze(2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(x_broadcast * filter_weights, dim=1), dim=-1)
        return torch.relu(ttcn_out + self.T_bias)

    def forward_features_from_tensors(
        self,
        model_input: torch.Tensor,
        patch_index: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch represented directly by tensors rather than a batch dict.

        Parameters
        ----------
        model_input:
            Input tensor of shape ``(batch, seq_len, in_dim)``.
        patch_index:
            Patch identifier tensor of shape ``(batch, seq_len)``.
        patch_mask:
            Patch-validity mask of shape ``(batch, npatch)``.

        Returns
        -------
        torch.Tensor
            Encoded feature tensor of shape ``(batch, 2 * hid_dim)``.
        """
        batch_size, _, dim = model_input.shape
        if dim != self.in_dim:
            raise AssertionError(f"Expected input dimension {self.in_dim}, got {dim}.")
        if patch_mask.shape[1] != self.npatch:
            raise AssertionError(f"Expected {self.npatch} patches, got {patch_mask.shape[1]}.")

        X_patch, mask_pt, patch_mask = self.patchify_by_patch_id(model_input, patch_index, patch_mask)
        X_int = X_patch.view(batch_size * self.npatch, self.patch_len, self.point_dim)
        mask_pt2 = mask_pt.view(batch_size * self.npatch, self.patch_len, 1)

        patch_feat = self.ttcn(X_int, mask_pt2)
        patch_exists = (mask_pt2.sum(dim=1) > 0).float()
        patch_tokens = torch.cat([patch_feat, patch_exists], dim=-1)

        x = patch_tokens.view(batch_size, self.npatch, self.hid_dim)
        x = x * patch_mask.unsqueeze(-1)

        for norm_layer, gnn_layer in zip(self.gnn_norms, self.patch_gattn):
            x = x + gnn_layer(norm_layer(x), patch_mask)
            x = x * patch_mask.unsqueeze(-1)

        x = self.pos_encoding(x)
        x = x * patch_mask.unsqueeze(-1)

        for block in self.mamba_stack:
            x = block(x, patch_mask)

        x_mean = masked_mean(x, patch_mask)
        x_last = gather_last_valid(x, patch_mask)
        return torch.cat([x_last, x_mean], dim=-1)

    def forward_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a standard batch dictionary into trajectory features."""
        return self.forward_features_from_tensors(
            batch["model_input"],
            batch["patch_index"],
            batch["patch_mask"],
        )


class PatchTTCN_Mamba_TrajPred(nn.Module):
    """Single-view Patch-Mamba predictor for one-step trajectory forecasting."""

    def __init__(self, cfg: Dict[str, object]) -> None:
        """Build the encoder and prediction head from a configuration dictionary."""
        super().__init__()
        self.pred_len = int(cfg["pred_len"])
        self.pred_dim = int(cfg["pred_dim"])
        self.hid_dim = int(cfg["hid_dim"])
        self.dropout = float(cfg.get("dropout", 0.1))
        self.head_hidden = int(cfg.get("head_hidden", 2 * self.hid_dim))

        self.encoder = PatchTTCN_Mamba_Encoder(cfg)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(2 * self.hid_dim),
            nn.Linear(2 * self.hid_dim, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden, self.pred_len * self.pred_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the model and return predictions with shape ``(batch, pred_len, pred_dim)``."""
        feat = self.encoder.forward_features(batch)
        pred = self.pred_head(feat)
        return pred.view(batch["model_input"].size(0), self.pred_len, self.pred_dim)

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convenience wrapper that returns only the first prediction step."""
        return self.forward(batch)[:, 0, :]

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Smooth L1 loss against the single-step target in ``batch``."""
        pred = self.forward(batch)
        target = batch["model_label"].unsqueeze(1)
        return F.smooth_l1_loss(pred, target), pred


class PatchTTCN_MultiWindowFusion_TrajPred(nn.Module):
    """Hybrid predictor that fuses several window-specific Patch-Mamba branches."""

    def __init__(self, cfg: Dict[str, object]) -> None:
        """Initialize the branch encoders, fusion gate, and prediction head.

        Parameters
        ----------
        cfg:
            Configuration dictionary containing branch definitions and fusion
            hyperparameters.
        """
        super().__init__()
        self.pred_len = int(cfg["pred_len"])
        self.pred_dim = int(cfg["pred_dim"])
        self.dropout = float(cfg.get("dropout", 0.1))
        self.branch_cfgs = copy.deepcopy(cfg["branch_cfgs"])
        self.branch_proj_dim = int(cfg.get("branch_proj_dim", 128))
        self.fusion_hidden = int(cfg.get("fusion_hidden", max(256, self.branch_proj_dim * len(self.branch_cfgs))))
        if len(self.branch_cfgs) <= 0:
            raise ValueError("branch_cfgs must not be empty.")

        self.branch_names = [str(branch_cfg["branch_name"]) for branch_cfg in self.branch_cfgs]
        self.branches = nn.ModuleDict()
        self.branch_projectors = nn.ModuleDict()
        for branch_cfg in self.branch_cfgs:
            name = str(branch_cfg["branch_name"])
            hid_dim = int(branch_cfg["hid_dim"])
            self.branches[name] = PatchTTCN_Mamba_Encoder(branch_cfg)
            self.branch_projectors[name] = nn.Sequential(
                nn.LayerNorm(2 * hid_dim),
                nn.Linear(2 * hid_dim, self.branch_proj_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )

        fusion_in_dim = self.branch_proj_dim * len(self.branch_cfgs)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, len(self.branch_cfgs)),
        )
        self.pred_head = nn.Sequential(
            nn.LayerNorm(fusion_in_dim + self.branch_proj_dim),
            nn.Linear(fusion_in_dim + self.branch_proj_dim, self.fusion_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_hidden, self.pred_len * self.pred_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run all branches, fuse their features, and predict one trajectory step."""
        projected_feats = []
        for name in self.branch_names:
            model_input = batch[f"{name}__model_input"]
            patch_index = batch[f"{name}__patch_index"]
            patch_mask = batch[f"{name}__patch_mask"]
            feat = self.branches[name].forward_features_from_tensors(model_input, patch_index, patch_mask)
            projected_feats.append(self.branch_projectors[name](feat))

        fusion_cat = torch.cat(projected_feats, dim=-1)
        stacked = torch.stack(projected_feats, dim=1)
        gate_logits = self.gate_net(fusion_cat)
        gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        weighted_feat = (stacked * gate).sum(dim=1)
        fusion_feat = torch.cat([fusion_cat, weighted_feat], dim=-1)

        pred = self.pred_head(fusion_feat)
        return pred.view(stacked.size(0), self.pred_len, self.pred_dim)

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return only the first prediction step for recursive rollout usage."""
        return self.forward(batch)[:, 0, :]

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Smooth L1 loss against the shared multi-view target."""
        pred = self.forward(batch)
        target = batch["model_label"].unsqueeze(1)
        return F.smooth_l1_loss(pred, target), pred
