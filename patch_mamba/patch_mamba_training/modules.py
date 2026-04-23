"""
Neural network building blocks for Patch-Mamba models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to an input sequence.
        """
        return x + self.pe[:, :x.size(1), :]


class PatchGraphAttention(nn.Module):
    """
    Patch-level self-attention with an exponential time-decay bias.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        tau_seconds: float = 300.0,
        delta_minutes: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.tau = float(tau_seconds)
        self.delta = float(delta_minutes) * 60.0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _time_bias(self, patch_count: int, device: torch.device) -> torch.Tensor:
        """
        Build a log-space temporal decay bias matrix.
        """
        idx = torch.arange(patch_count, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().float() * self.delta
        weight = torch.exp(-dist / max(self.tau, 1e-6))
        return torch.log(weight + 1e-12)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply time-biased self-attention over patch tokens.
        """
        bsz, patch_count, d_model = x.shape
        q = self.q_proj(x).view(bsz, patch_count, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, patch_count, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, patch_count, self.nhead, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn + self._time_bias(patch_count, x.device)[None, None, :, :]

        if patch_mask is not None:
            key_pad = patch_mask <= 0
            attn = attn.masked_fill(key_pad[:, None, None, :], -1e9)

        weight = torch.softmax(attn, dim=-1)
        weight = self.dropout(weight)

        out = torch.matmul(weight, v)
        out = out.transpose(1, 2).contiguous().view(bsz, patch_count, d_model)
        out = self.out_proj(out)

        if patch_mask is not None:
            out = out * patch_mask.unsqueeze(-1).float()
        return out


class ResidualMambaBlock(nn.Module):
    """
    Residual Mamba block with a feed-forward sublayer.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is required but not installed.")

        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply a residual Mamba update followed by a residual feed-forward update.
        """
        y = self.mamba(self.norm1(x))
        if patch_mask is not None:
            y = y * patch_mask.unsqueeze(-1)
        x = x + self.drop1(y)

        z = self.ffn(self.norm2(x))
        if patch_mask is not None:
            z = z * patch_mask.unsqueeze(-1)
        x = x + self.drop2(z)

        if patch_mask is not None:
            x = x * patch_mask.unsqueeze(-1)
        return x
