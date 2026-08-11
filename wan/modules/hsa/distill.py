"""Attention self-distillation signal for training the HSA-specific parameters.

The HSA output approximates full attention over the same KV window. When collection is enabled,
every :class:`~worldattention.hsa.attention.HSAAttention` forward records
``MSE(hsa_out, dense_out.detach())`` into a process-level registry; a training loop pops and sums
those per-layer terms and backpropagates them into the fusion gate and the low-rank projections.

This is a per-layer, per-token, every-step signal, which trains the HSA parameters far more
directly than an end-to-end distribution-matching loss on the final output. Collection is off by
default and costs nothing when disabled.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

__all__ = ["enable_distill", "distill_enabled", "pop_distill_losses", "record_distill_loss",
           "dense_attention"]

_ENABLED = False
_LOSSES: List[torch.Tensor] = []


def enable_distill(flag: bool = True) -> None:
    """Turn per-layer HSA-to-dense collection on or off, clearing the registry when turning off."""
    global _ENABLED
    _ENABLED = bool(flag)
    if not _ENABLED:
        _LOSSES.clear()


def distill_enabled() -> bool:
    return _ENABLED


def pop_distill_losses() -> List[torch.Tensor]:
    """Return the losses recorded since the last pop and clear the registry."""
    global _LOSSES
    losses = _LOSSES
    _LOSSES = []
    return losses


def record_distill_loss(loss: torch.Tensor) -> None:
    _LOSSES.append(loss)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full attention over ``[B, L, H, D]`` tensors: the target HSA is distilled towards.

    The cuDNN attention backend is excluded because it has no execution plan for the long-KV,
    head-dim-128 shapes this runs on; the flash and memory-efficient backends do.
    """
    query, key, value = (t.transpose(1, 2) for t in (q, k, v))
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(query, key, value)
    except ImportError:
        out = F.scaled_dot_product_attention(query, key, value)
    return out.transpose(1, 2)


def record_against_dense(hsa_out: torch.Tensor, q, k, v) -> None:
    """Record the MSE between an HSA output and dense attention over the same window."""
    with torch.no_grad():
        teacher = dense_attention(q, k, v).to(hsa_out.dtype)
    record_distill_loss(F.mse_loss(hsa_out.float(), teacher.float().detach()))
