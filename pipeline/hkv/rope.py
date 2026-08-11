"""Temporal rotary-embedding re-rotation for reused KV pages.

A KV cache stores keys that already carry the rotary encoding of the frame they were generated at.
HKV needs two operations on top of that:

* a retrieved page is replayed at a new cache position, so its baked-in temporal rotation must be
  shifted by a constant frame delta;
* to score a page's keys against a pre-rotation query, the per-token temporal rotation must be
  removed to recover the content-space key.

Rotary embedding for a video latent is a per-position complex multiply split into (temporal,
height, width) bands. Moving a token along the frame axis only touches the temporal band, so
neither operation needs to reconstruct the spatial bands.
"""
from __future__ import annotations

import torch

__all__ = ["temporal_complex_dims", "shift_temporal_rope", "derope_temporal"]


def temporal_complex_dims(head_dim: int) -> int:
    """Number of complex dimensions in the temporal band of a ``head_dim`` rotary embedding."""
    half = head_dim // 2
    return half - 2 * (half // 3)


def temporal_band(freqs: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Slice the temporal band out of a ``[max_positions, head_dim // 2]`` frequency table."""
    return freqs[:, : temporal_complex_dims(head_dim)]


def _apply_factor(k: torch.Tensor, factor: torch.Tensor, num_temporal: int) -> torch.Tensor:
    shape = k.shape
    head_dim = shape[-1]
    complex_k = torch.view_as_complex(
        k.to(torch.float32).contiguous().reshape(*shape[:-1], head_dim // 2, 2)
    ).clone()
    complex_k[..., :num_temporal] = complex_k[..., :num_temporal] * factor.to(torch.complex64)
    return torch.view_as_real(complex_k).reshape(shape).to(k.dtype)


def _factor(temporal_freqs: torch.Tensor, new_ids: torch.Tensor, orig_ids: torch.Tensor) -> torch.Tensor:
    max_positions = temporal_freqs.shape[0]
    new = new_ids.clamp(0, max_positions - 1)
    orig = orig_ids.clamp(0, max_positions - 1)
    return temporal_freqs[new] * temporal_freqs[orig].conj()


def shift_temporal_rope(
    k: torch.Tensor,
    temporal_freqs: torch.Tensor,
    new_frame0: int,
    orig_frame0: int,
) -> torch.Tensor:
    """Move a page of keys from ``orig_frame0`` to ``new_frame0``.

    All tokens of a page share one frame delta, so a single rotation factor is exact for the page.
    ``k`` is ``[..., tokens, H, head_dim]``.
    """
    num_temporal = temporal_complex_dims(k.shape[-1])
    factor = _factor(
        temporal_freqs,
        torch.as_tensor(int(new_frame0), device=temporal_freqs.device),
        torch.as_tensor(int(orig_frame0), device=temporal_freqs.device),
    )
    return _apply_factor(k, factor, num_temporal)


def derope_temporal(
    k: torch.Tensor,
    temporal_freqs: torch.Tensor,
    frame_ids: torch.Tensor,
) -> torch.Tensor:
    """Remove each token's own temporal rotation, recovering content-space keys.

    ``k`` is ``[B, tokens, H, head_dim]`` and ``frame_ids`` is the absolute frame of each token.
    """
    num_temporal = temporal_complex_dims(k.shape[-1])
    factor = _factor(temporal_freqs, torch.zeros_like(frame_ids), frame_ids)
    factor = factor.view(*([1] * (k.dim() - 3)), frame_ids.shape[0], 1, num_temporal)
    return _apply_factor(k, factor, num_temporal)
