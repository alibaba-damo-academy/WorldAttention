"""Reuse of the pooled key blocks that do not change between denoising steps."""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["PooledKeyCache"]


class PooledKeyCache:
    """Stores the pooled rows of a stable leading span of the key tensor."""

    __slots__ = ("enabled", "_rows", "_shape", "hits", "misses")

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._rows: Optional[torch.Tensor] = None
        self._shape = None
        self.hits = 0
        self.misses = 0

    def invalidate(self) -> None:
        """Drop the entry. Call whenever the stable span's contents may have changed."""
        self._rows = None
        self._shape = None

    @staticmethod
    def stable_blocks(stable_tokens: int, block_size: int) -> int:
        """Whole pooled rows covered by ``stable_tokens``."""
        return max(0, int(stable_tokens) // int(block_size))

    def get(self, rows: int, block_size: int) -> Optional[torch.Tensor]:
        """The cached pooled rows for this prefix, or ``None``."""
        if not self.enabled or self._rows is None or rows <= 0:
            return None
        if self._shape != (rows, block_size):
            self.invalidate()
            return None
        self.hits += 1
        return self._rows

    def put(self, rows: int, block_size: int, pooled_rows: torch.Tensor) -> torch.Tensor:
        if self.enabled and rows > 0:
            self._rows = pooled_rows
            self._shape = (rows, block_size)
        self.misses += 1
        return pooled_rows

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
