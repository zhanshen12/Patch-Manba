"""
Feature encoder definitions for Patch-Mamba models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import PatchGraphAttention, PositionalEncoding, ResidualMambaBlock
from .utils import gather_last_valid, masked_mean


class PatchTTCN_Mamba_Encoder(nn.Module):
    """
    Patch-based encoder composed of TTCN-style aggregation, graph attention,
    positional encoding, and stacked Mamba blocks.
    """

    def __init__(self, cfg: dict) -> None:
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
            raise ValueError("`hid_dim` must be at least 2.")

        self.Filter_Generators = nn.Sequential(
            nn.Linear(self.point_dim, self.ttcn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.point_dim * self.ttcn_dim),
        )
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))

        self.patch_gattn = nn.ModuleList(
            [
                PatchGraphAttention(
                    d_model=self.hid_dim,
                    nhead=self.nhead,
                    tau_seconds=self.tau_seconds,
                    delta_minutes=self.delta_minutes,
                    dropout=self.dropout,
                )
                for _ in range(self.gnn_layers)
            ]
        )
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(self.hid_dim) for _ in range(self.gnn_layers)])
        self.pos_encoding = PositionalEncoding(self.hid_dim, max_len=max(1024, self.npatch + 8))
        self.mamba_stack = nn.ModuleList(
            [
                ResidualMambaBlock(
                    d_model=self.hid_dim,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand,
                    dropout=self.dropout,
                )
                for _ in range(self.mamba_layers)
            ]
        )

    def patchify_by_patch_id(
        self,
        model_input: torch.Tensor,
        patch_index: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rearrange flat point sequences into fixed-size patch tensors.
        """
        bsz, seq_len, feat_dim = model_input.shape
        patch_count = patch_mask.shape[1]
        patch_len = self.patch_len
        device = model_input.device

        X_patch = torch.zeros((bsz, patch_count, patch_len, feat_dim), device=device, dtype=model_input.dtype)
        mask_pt = torch.zeros((bsz, patch_count, patch_len, 1), device=device, dtype=torch.float32)

        for b in range(bsz):
            write_ptr = torch.zeros((patch_count,), dtype=torch.long, device=device)
            for t in range(seq_len):
                patch_id = int(patch_index[b, t].item())
                if patch_id <= 0:
                    continue
                p = patch_id - 1
                if p < 0 or p >= patch_count:
                    continue
                slot = int(write_ptr[p].item())
                if slot >= patch_len:
                    continue
                X_patch[b, p, slot, :] = model_input[b, t, :]
                mask_pt[b, p, slot, 0] = 1.0
                write_ptr[p] += 1

        return X_patch, mask_pt, patch_mask.float()

    def ttcn(self, X_int: torch.Tensor, mask_X: torch.Tensor) -> torch.Tensor:
        """
        Aggregate variable-length point sequences inside each patch.
        """
        n_patch_instances, patch_len, _ = X_int.shape
        filters = self.Filter_Generators(X_int)
        filters = filters.view(n_patch_instances, patch_len, self.ttcn_dim, self.point_dim)

        mask_expand = mask_X.unsqueeze(-1)
        filters_masked = filters * mask_expand + (1.0 - mask_expand) * (-1e8)
        filters_norm = torch.softmax(filters_masked, dim=1)

        X_broad = X_int.unsqueeze(2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_broad * filters_norm, dim=1), dim=-1)
        return torch.relu(ttcn_out + self.T_bias)

    def forward_features_from_tensors(
        self,
        model_input: torch.Tensor,
        patch_index: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one batch into a fixed-size fused representation.
        """
        bsz, _, feat_dim = model_input.shape
        if feat_dim != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, but received {feat_dim}.")
        if patch_mask.shape[1] != self.npatch:
            raise ValueError(f"Expected npatch={self.npatch}, but received {patch_mask.shape[1]}.")

        X_patch, mask_pt, patch_mask = self.patchify_by_patch_id(model_input, patch_index, patch_mask)
        X_int = X_patch.view(bsz * self.npatch, self.patch_len, self.point_dim)
        mask_pt2 = mask_pt.view(bsz * self.npatch, self.patch_len, 1)

        patch_feat = self.ttcn(X_int, mask_pt2)
        patch_exists = (mask_pt2.sum(dim=1) > 0).float()
        patch_tok = torch.cat([patch_feat, patch_exists], dim=-1)

        x = patch_tok.view(bsz, self.npatch, self.hid_dim)
        x = x * patch_mask.unsqueeze(-1)

        for i, gnn in enumerate(self.patch_gattn):
            x = x + gnn(self.gnn_norms[i](x), patch_mask)
            x = x * patch_mask.unsqueeze(-1)

        x = self.pos_encoding(x)
        x = x * patch_mask.unsqueeze(-1)

        for block in self.mamba_stack:
            x = block(x, patch_mask)

        x_mean = masked_mean(x, patch_mask)
        x_last = gather_last_valid(x, patch_mask)
        return torch.cat([x_last, x_mean], dim=-1)

    def forward_features(self, batch: dict) -> torch.Tensor:
        """
        Convenience wrapper that reads tensors from a standard batch dictionary.
        """
        return self.forward_features_from_tensors(
            batch["model_input"],
            batch["patch_index"],
            batch["patch_mask"],
        )
