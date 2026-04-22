"""Core neural network layers used by the modular Patch-Mamba models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:  # pragma: no cover - optional dependency path
    HAS_MAMBA = False
    Mamba = None


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for patch-token sequences."""

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        """Pre-compute sinusoidal encodings up to ``max_len`` positions.

        Parameters
        ----------
        d_model:
            Token embedding dimension.
        max_len:
            Maximum supported sequence length.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input token tensor.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, length, d_model)``.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x`` after positional encoding.
        """
        return x + self.pe[:, :x.size(1), :]


class PatchGraphAttention(nn.Module):
    """Multi-head attention with an exponential time-decay bias across patches.

    The layer augments standard scaled dot-product attention with a deterministic
    log-space bias derived from patch distance in time. Nearby patches therefore
    receive a stronger prior connection than distant patches.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        tau_seconds: float = 300.0,
        delta_minutes: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize attention projections and time-bias settings.

        Parameters
        ----------
        d_model:
            Token feature dimension.
        nhead:
            Number of attention heads.
        tau_seconds:
            Time-decay constant controlling how quickly attention prior weakens.
        delta_minutes:
            Duration represented by one patch step.
        dropout:
            Dropout rate applied to the attention matrix.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise AssertionError("d_model must be divisible by nhead.")

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

    def _time_bias(self, pnum: int, device: torch.device) -> torch.Tensor:
        """Build the pairwise log-decay bias matrix for ``pnum`` patches."""
        indices = torch.arange(pnum, device=device)
        distance = (indices[:, None] - indices[None, :]).abs().float() * self.delta
        weights = torch.exp(-distance / max(self.tau, 1e-6))
        return torch.log(weights + 1e-12)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply time-aware attention to patch tokens.

        Parameters
        ----------
        x:
            Patch-token tensor of shape ``(batch, patches, d_model)``.
        patch_mask:
            Optional validity mask of shape ``(batch, patches)``.

        Returns
        -------
        torch.Tensor
            Refined patch-token tensor with the same shape as ``x``.
        """
        batch_size, patch_num, d_model = x.shape
        q = self.q_proj(x).view(batch_size, patch_num, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, patch_num, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, patch_num, self.nhead, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn + self._time_bias(patch_num, x.device)[None, None, :, :]

        if patch_mask is not None:
            key_padding = patch_mask <= 0
            attn = attn.masked_fill(key_padding[:, None, None, :], -1e9)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, patch_num, d_model)
        output = self.out_proj(output)

        if patch_mask is not None:
            output = output * patch_mask.unsqueeze(-1).float()
        return output


class ResidualMambaBlock(nn.Module):
    """Residual Mamba block followed by a feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Construct one Mamba-based sequence block.

        Parameters
        ----------
        d_model:
            Token embedding dimension.
        d_state:
            Mamba internal state dimension.
        d_conv:
            Mamba local convolution width.
        expand:
            Mamba expansion ratio.
        dropout:
            Dropout rate applied after the Mamba path and feed-forward path.
        """
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is required to instantiate ResidualMambaBlock.")

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
        """Apply the residual Mamba block to a batch of patch tokens.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, patches, d_model)``.
        patch_mask:
            Optional validity mask used to zero out padded patch positions.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as the input.
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
