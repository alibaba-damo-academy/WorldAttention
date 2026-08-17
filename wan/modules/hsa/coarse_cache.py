"""Ever-growing compressed-KV history for the linear branch (the global memory)."""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["GrowingCoarseCache"]


class GrowingCoarseCache:
    """Per-layer growing buffer of compressed K/V rows, ``[B, rows, H, D]`` each."""

    __slots__ = ("enabled", "_k", "_v", "_rows")

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None
        self._rows = 0

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def capacity(self) -> int:
        return 0 if self._k is None else self._k.shape[1]

    def reserve(self, bsz: int, rows: int, heads: int, head_dim: int,
                device, dtype=torch.bfloat16) -> None:
        """Preallocate capacity for ``rows`` without changing the stored contents."""
        if self.capacity >= rows:
            return
        k = torch.empty(bsz, rows, heads, head_dim, device=device, dtype=dtype)
        v = torch.empty_like(k)
        if self._rows:
            k[:, : self._rows] = self._k[:, : self._rows]
            v[:, : self._rows] = self._v[:, : self._rows]
        self._k, self._v = k, v

    def append(self, k_rows: torch.Tensor, v_rows: torch.Tensor) -> None:
        """Append finalized coarse rows ``[B, r, H, D]``; they are never rewritten."""
        if k_rows.shape != v_rows.shape:
            raise ValueError(f"k/v row shapes differ: {tuple(k_rows.shape)} vs {tuple(v_rows.shape)}")
        needed = self._rows + k_rows.shape[1]
        if needed > self.capacity:
            bsz, _, heads, head_dim = k_rows.shape
            self.reserve(bsz, max(needed, 2 * max(self.capacity, 1)), heads, head_dim,
                         k_rows.device, k_rows.dtype)
        self._k[:, self._rows : needed] = k_rows
        self._v[:, self._rows : needed] = v_rows
        self._rows = needed

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Views of the stored rows, ``[B, rows, H, D]`` each. No copy."""
        return self._k[:, : self._rows], self._v[:, : self._rows]

    def clear(self) -> None:
        """Forget the history but keep the allocation."""
        self._rows = 0

    def memory_bytes(self) -> int:
        return 0 if self._k is None else self._k.numel() * self._k.element_size() * 2
