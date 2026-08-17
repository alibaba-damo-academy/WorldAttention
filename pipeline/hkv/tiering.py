"""Page-granular residency across GPU, CPU DRAM and NVMe."""
from __future__ import annotations

import os
from typing import Dict, Hashable, List, Optional, Tuple

import torch

__all__ = ["HKVTierManager"]


class HKVTierManager:
    """Track pages across GPU slots, CPU DRAM and NVMe with LRU migration."""

    def __init__(
        self,
        *,
        gpu_max_pages: int = 0,
        cpu_max_pages: int = 512,
        nvme_enabled: bool = False,
        nvme_dir: Optional[str] = None,
    ):
        self.gpu_max_pages = max(int(gpu_max_pages), 0)
        self.cpu_max_pages = max(int(cpu_max_pages), 1)
        self.nvme_enabled = bool(nvme_enabled)
        self.nvme_dir = nvme_dir
        if self.nvme_enabled:
            if not nvme_dir:
                raise ValueError("nvme_dir is required when nvme_enabled=True")
            os.makedirs(nvme_dir, exist_ok=True)

        self._page_shape: Optional[Tuple[int, ...]] = None
        self._gpu_pool: Optional[torch.Tensor] = None
        self._gpu_free: List[int] = []
        self._gpu: Dict[Hashable, int] = {}
        self._gpu_lru: List[Hashable] = []

        self._cpu: Dict[Hashable, torch.Tensor] = {}
        self._cpu_free: List[torch.Tensor] = []
        self._cpu_lru: List[Hashable] = []
        self._cpu_allocated = 0

        self._nvme: Dict[Hashable, str] = {}
        self._backup: Dict[Hashable, str] = {}

        self._xfer_stream: Optional[torch.cuda.Stream] = None
        self._pending: List[Tuple[Hashable, int, torch.cuda.Event]] = []
        self._cpu_events: Dict[Hashable, torch.cuda.Event] = {}
        self._pinned_pages = 0
        self.max_pinned_pages = 64
        self.n_pin_fallbacks = 0
        self.prefill_host_pages = 0
        self.staging_slots = 0
        self._staging: List[int] = []

        self.n_gpu_hits = 0
        self.n_gpu_misses = 0
        self.n_promotions = 0
        self.n_demotions = 0
        self.n_spills = 0
        self.n_loads = 0
        self.n_drops = 0

    def configure(
        self,
        *,
        layers: int,
        batch: int,
        page_tokens: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Fix the page geometry and build the GPU pool."""
        shape = (int(layers), 2, int(batch), int(page_tokens), int(heads), int(head_dim))
        if self._page_shape == shape and (
            self._gpu_pool is None or self._gpu_pool.dtype == dtype
        ):
            return

        self.clear()
        self._page_shape = shape
        self._gpu_pool = None
        self._gpu_free = []
        self._cpu_free = []
        self._cpu_allocated = 0
        if self.gpu_max_pages > 0 and device.type == "cuda":
            self._gpu_pool = torch.zeros(
                (self.gpu_max_pages, *shape), dtype=dtype, device=device
            )
            self._gpu_free = list(range(self.gpu_max_pages))
            reserve = min(self.staging_slots, max(self.gpu_max_pages - 1, 0))
            self._staging = [self._gpu_free.pop() for _ in range(reserve)]
        self._dtype = dtype

        for _ in range(min(self.prefill_host_pages, self.cpu_max_pages)):
            self._cpu_free.append(self._new_host_page())
            self._cpu_allocated += 1

    @property
    def page_bytes(self) -> int:
        if self._page_shape is None:
            return 0
        numel = 1
        for dim in self._page_shape:
            numel *= dim
        return numel * torch.empty((), dtype=self._dtype).element_size()

    def staging_view(self, index: int) -> Optional[torch.Tensor]:
        """A reserved staging slot, or None when none were reserved."""
        if not self._staging or self._gpu_pool is None:
            return None
        return self._gpu_pool[self._staging[index % len(self._staging)]]

    @property
    def n_staging(self) -> int:
        return len(self._staging)

    def allocate(self, key: Hashable) -> torch.Tensor:
        """Reserve storage for a new page and return the tensor to write it into."""
        if self._page_shape is None:
            raise RuntimeError("configure() must be called before pages are allocated")
        self.discard(key)

        slot = self._claim_gpu_slot(evict=True)
        if slot is not None:
            self._gpu[key] = slot
            self._touch(self._gpu_lru, key)
            self._evict_ahead(exclude=key)
            return self._gpu_pool[slot]

        tensor = self._claim_cpu_tensor()
        self._cpu[key] = tensor
        self._touch(self._cpu_lru, key)
        return tensor

    def _evict_ahead(self, *, exclude: Hashable) -> None:
        """Demote the least-recently-used page asynchronously when no free slot remains."""
        if self._gpu_pool is None or self._gpu_free or self._pending:
            return
        victim = next((k for k in self._gpu_lru if k in self._gpu and k != exclude), None)
        if victim is not None:
            self._demote(victim)

    def get(self, key: Hashable) -> Optional[torch.Tensor]:
        """Return the page where it currently lives, or None if it is gone."""
        if key in self._gpu:
            self.n_gpu_hits += 1
            self._touch(self._gpu_lru, key)
            return self._gpu_pool[self._gpu[key]]

        tensor = self._cpu.get(key)
        if tensor is not None:
            self._await_host_page(key)
        if tensor is None and key in self._nvme:
            tensor = self._load_from_nvme(key)
        if tensor is None:
            return None

        self.n_gpu_misses += 1
        slot = self._claim_gpu_slot(evict=False)
        if slot is None:
            self._touch(self._cpu_lru, key)
            return tensor

        self._gpu_pool[slot].copy_(tensor)
        self._release_cpu_tensor(key)
        self._gpu[key] = slot
        self._touch(self._gpu_lru, key)
        self.n_promotions += 1
        return self._gpu_pool[slot]

    def location(self, key: Hashable) -> str:
        """Current tier of a page: ``"gpu"``, ``"cpu"``, ``"nvme"`` or ``"absent"``."""
        if key in self._gpu:
            return "gpu"
        if key in self._cpu:
            return "cpu"
        if key in self._nvme:
            return "nvme"
        return "absent"

    def discard(self, key: Hashable) -> None:
        """Forget a page, returning whatever storage it held to the free lists."""
        self._reclaim_pending()
        slot = self._gpu.pop(key, None)
        if slot is not None:
            self._gpu_free.append(slot)
            self._untouch(self._gpu_lru, key)
        self._release_cpu_tensor(key)
        self._nvme.pop(key, None)

    @staticmethod
    def _touch(lru: List[Hashable], key: Hashable) -> None:
        if key in lru:
            lru.remove(key)
        lru.append(key)

    @staticmethod
    def _untouch(lru: List[Hashable], key: Hashable) -> None:
        if key in lru:
            lru.remove(key)

    def _claim_gpu_slot(self, *, evict: bool) -> Optional[int]:
        if self._gpu_pool is None:
            return None
        self._reclaim_pending()
        if self._gpu_free:
            return self._gpu_free.pop()
        if not evict:
            return None
        if self._pending:
            self._reclaim_pending(block=True)
            if self._gpu_free:
                return self._gpu_free.pop()
        victim = next((k for k in self._gpu_lru if k in self._gpu), None)
        if victim is None:
            return None
        self._demote(victim)
        self._reclaim_pending(block=True)
        return self._gpu_free.pop() if self._gpu_free else None

    def _demote(self, key: Hashable) -> None:
        """Move a page from its GPU slot down to CPU DRAM, without waiting for the copy."""
        slot = self._gpu.pop(key)
        self._untouch(self._gpu_lru, key)
        tensor = self._claim_cpu_tensor()

        if self._xfer_stream is None:
            self._xfer_stream = torch.cuda.Stream(device=self._gpu_pool.device)

        self._xfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._xfer_stream):
            tensor.copy_(self._gpu_pool[slot], non_blocking=tensor.is_pinned())
            event = torch.cuda.Event()
            event.record(self._xfer_stream)

        self._cpu[key] = tensor
        self._cpu_events[key] = event
        self._touch(self._cpu_lru, key)
        self._pending.append((key, slot, event))
        self.n_demotions += 1

    def _reclaim_pending(self, *, block: bool = False) -> None:
        """Return slots whose demotion copy has finished. With ``block``, wait for the oldest."""
        while self._pending:
            key, slot, event = self._pending[0]
            if event.query():
                self._pending.pop(0)
                self._gpu_free.append(slot)
                continue
            if not block:
                return
            event.synchronize()

    def _await_host_page(self, key: Hashable) -> None:
        """Block until a demoted page is actually readable on the host."""
        event = self._cpu_events.pop(key, None)
        if event is not None:
            event.synchronize()

    def _new_host_page(self) -> torch.Tensor:
        """A host page, pinned when the budget allows."""
        if self._pinned_pages < self.max_pinned_pages:
            try:
                page = torch.empty(self._page_shape, dtype=self._dtype, device="cpu",
                                   pin_memory=True)
                self._pinned_pages += 1
                return page
            except RuntimeError:
                self.n_pin_fallbacks += 1
        else:
            self.n_pin_fallbacks += 1
        return torch.empty(self._page_shape, dtype=self._dtype, device="cpu")

    def _claim_cpu_tensor(self) -> torch.Tensor:
        if self._cpu_free:
            return self._cpu_free.pop()
        if self._cpu_allocated < self.cpu_max_pages:
            self._cpu_allocated += 1
            return self._new_host_page()
        victim = next((k for k in self._cpu_lru if k in self._cpu), None)
        if victim is None:
            self._cpu_allocated += 1
            return self._new_host_page()
        self._spill_or_drop(victim)
        return self._cpu_free.pop()

    def _spill_or_drop(self, key: Hashable) -> None:
        self._await_host_page(key)
        tensor = self._cpu.pop(key)
        self._untouch(self._cpu_lru, key)
        if self.nvme_enabled:
            path = self._backup.get(key) or self._path(key)
            if key not in self._backup:
                torch.save(tensor, path)
                self._backup[key] = path
            self._nvme[key] = path
            self.n_spills += 1
        else:
            self.n_drops += 1
        self._cpu_free.append(tensor)

    def _release_cpu_tensor(self, key: Hashable) -> None:
        self._await_host_page(key)
        tensor = self._cpu.pop(key, None)
        if tensor is not None:
            self._cpu_free.append(tensor)
            self._untouch(self._cpu_lru, key)

    def _load_from_nvme(self, key: Hashable) -> torch.Tensor:
        tensor = self._claim_cpu_tensor()
        tensor.copy_(torch.load(self._nvme[key], map_location="cpu", weights_only=False))
        self.n_loads += 1
        self._cpu[key] = tensor
        self._nvme.pop(key, None)
        self._touch(self._cpu_lru, key)
        return tensor

    def _path(self, key: Hashable) -> str:
        name = "_".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        return os.path.join(self.nvme_dir, f"hkv_page_{name}.pt")

    def drain(self) -> None:
        """Wait for every in-flight demotion. Call before tearing down or reading stats."""
        self._reclaim_pending(block=True)
        for key in list(self._cpu_events):
            self._await_host_page(key)

    def clear(self) -> None:
        self.drain()
        """Drop every page but keep the pools, so a new rollout reuses the same storage."""
        for slot in list(self._gpu.values()):
            self._gpu_free.append(slot)
        self._gpu.clear()
        self._gpu_lru.clear()
        for tensor in list(self._cpu.values()):
            self._cpu_free.append(tensor)
        self._cpu.clear()
        self._cpu_lru.clear()
        self._nvme.clear()

    def stats(self) -> dict:
        return {
            "gpu": len(self._gpu),
            "gpu_budget": self.gpu_max_pages,
            "cpu": len(self._cpu),
            "nvme": len(self._nvme),
            "hits": self.n_gpu_hits,
            "misses": self.n_gpu_misses,
            "promotions": self.n_promotions,
            "demotions": self.n_demotions,
            "pinned_pages": self._pinned_pages,
            "pin_fallbacks": self.n_pin_fallbacks,
            "in_flight": len(self._pending),
            "staging_slots": len(self._staging),
            "spills": self.n_spills,
            "loads": self.n_loads,
            "drops": self.n_drops,
        }

    def cleanup(self) -> None:
        self.drain()
        """Delete spilled files and release every pool."""
        for path in set(list(self._backup.values()) + list(self._nvme.values())):
            try:
                os.remove(path)
            except OSError:
                pass
        self.clear()
        self._backup.clear()
        self._gpu_pool = None
        self._gpu_free = []
        self._cpu_free = []
        self._cpu_allocated = 0
        self._page_shape = None
