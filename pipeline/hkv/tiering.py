"""Multi-tier residency for KV chunk blobs.

Chunks live in CPU DRAM (L2) up to a capacity, and least-recently-used chunks spill to NVMe (L3)
when NVMe is enabled. NVMe keeps a full backup of anything it has ever held, so a chunk can always
be restored and re-spilling an unchanged chunk costs nothing. With NVMe disabled the manager is a
bounded CPU cache and performs no disk I/O.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

__all__ = ["HKVTierManager"]


class HKVTierManager:
    """Track per-chunk KV blobs across CPU DRAM and NVMe with LRU migration.

    A blob is an opaque dict of CPU tensors, e.g. ``{"k": [...], "v": [...]}``.

    Args:
        cpu_max_chunks: CPU-resident chunk capacity.
        nvme_enabled: spill over-capacity chunks to disk instead of dropping them.
        nvme_dir: directory for spilled chunks; required when ``nvme_enabled``.
    """

    def __init__(
        self,
        cpu_max_chunks: int,
        nvme_enabled: bool = False,
        nvme_dir: Optional[str] = None,
    ):
        self.cpu_max_chunks = max(int(cpu_max_chunks), 1)
        self.nvme_enabled = bool(nvme_enabled)
        self.nvme_dir = nvme_dir
        if self.nvme_enabled:
            if not nvme_dir:
                raise ValueError("nvme_dir is required when nvme_enabled=True")
            os.makedirs(nvme_dir, exist_ok=True)

        self._cpu: Dict[int, dict] = {}
        self._nvme: Dict[int, str] = {}
        self._backup: Dict[int, str] = {}
        self._lru: List[int] = []
        self.n_spills = 0
        self.n_loads = 0
        self.n_drops = 0

    def _touch(self, chunk_id: int) -> None:
        if chunk_id in self._lru:
            self._lru.remove(chunk_id)
        self._lru.append(chunk_id)

    def _path(self, chunk_id: int) -> str:
        return os.path.join(self.nvme_dir, f"hkv_chunk_{chunk_id}.pt")

    def put(self, chunk_id: int, blob: dict) -> None:
        """Register a freshly produced chunk as CPU-resident."""
        self._cpu[chunk_id] = blob
        self._nvme.pop(chunk_id, None)
        self._touch(chunk_id)
        self._evict_if_needed()

    def get(self, chunk_id: int) -> Optional[dict]:
        """Return a chunk blob, reloading it from NVMe if it was spilled."""
        if chunk_id in self._cpu:
            self._touch(chunk_id)
            return self._cpu[chunk_id]
        if chunk_id in self._nvme:
            blob = torch.load(self._nvme[chunk_id], map_location="cpu", weights_only=False)
            self.n_loads += 1
            self._cpu[chunk_id] = blob
            self._nvme.pop(chunk_id, None)
            self._touch(chunk_id)
            self._evict_if_needed()
            return blob
        return None

    def location(self, chunk_id: int) -> str:
        """Current tier of a chunk: ``"cpu"``, ``"nvme"`` or ``"absent"``."""
        if chunk_id in self._cpu:
            return "cpu"
        if chunk_id in self._nvme:
            return "nvme"
        return "absent"

    def _evict_if_needed(self) -> None:
        while len(self._cpu) > self.cpu_max_chunks:
            victim = next((cid for cid in self._lru if cid in self._cpu), None)
            if victim is None:
                break
            blob = self._cpu.pop(victim)
            if self.nvme_enabled:
                path = self._backup.get(victim) or self._path(victim)
                if victim not in self._backup:
                    torch.save(blob, path)
                    self._backup[victim] = path
                self._nvme[victim] = path
                self.n_spills += 1
            else:
                if victim in self._lru:
                    self._lru.remove(victim)
                self.n_drops += 1

    def stats(self) -> dict:
        return {
            "cpu_resident": len(self._cpu),
            "nvme_resident": len(self._nvme),
            "backups": len(self._backup),
            "n_spills": self.n_spills,
            "n_loads": self.n_loads,
            "n_drops": self.n_drops,
        }

    def cleanup(self) -> None:
        """Delete spilled files and clear all bookkeeping."""
        for path in set(list(self._backup.values()) + list(self._nvme.values())):
            try:
                os.remove(path)
            except OSError:
                pass
        self._cpu.clear()
        self._nvme.clear()
        self._backup.clear()
        self._lru.clear()
