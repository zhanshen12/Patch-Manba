"""
Top-level Patch-Mamba model definitions.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import PatchTTCN_Mamba_Encoder


class PatchTTCN_Mamba_TrajPred(nn.Module):
    """
    Single-view Patch-Mamba predictor.
    """

    def __init__(self, cfg: dict) -> None:
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

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Run full forward prediction for one batch.
        """
        feat = self.encoder.forward_features(batch)
        pred = self.pred_head(feat)
        return pred.view(batch["model_input"].size(0), self.pred_len, self.pred_dim)

    def forward_step(self, batch: dict) -> torch.Tensor:
        """
        Return only the first prediction step.
        """
        return self.forward(batch)[:, 0, :]

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute single-step Smooth L1 training loss.
        """
        pred = self.forward(batch)
        target = batch["model_label"].unsqueeze(1)
        return F.smooth_l1_loss(pred, target), pred


class PatchTTCN_MultiWindowFusion_TrajPred(nn.Module):
    """
    Multi-window hybrid Patch-Mamba predictor.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.pred_len = int(cfg["pred_len"])
        self.pred_dim = int(cfg["pred_dim"])
        self.dropout = float(cfg.get("dropout", 0.1))
        self.branch_cfgs = copy.deepcopy(cfg["branch_cfgs"])
        self.branch_proj_dim = int(cfg.get("branch_proj_dim", 128))
        self.fusion_hidden = int(cfg.get("fusion_hidden", max(256, self.branch_proj_dim * len(self.branch_cfgs))))
        if len(self.branch_cfgs) <= 0:
            raise ValueError("`branch_cfgs` must not be empty.")

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

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Run multi-branch forward prediction.
        """
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

    def forward_step(self, batch: dict) -> torch.Tensor:
        """
        Return the first prediction step for recursive rollout usage.
        """
        return self.forward(batch)[:, 0, :]

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute single-step Smooth L1 loss for the fusion model.
        """
        pred = self.forward(batch)
        target = batch["model_label"].unsqueeze(1)
        return F.smooth_l1_loss(pred, target), pred
