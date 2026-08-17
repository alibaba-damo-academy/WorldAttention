"""The per-block HKV cycle, shared by inference and training."""
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .cache import HKVCache
from .rope import temporal_band

__all__ = ["HKVBlockCycle", "hkv_config_summary"]


def _cfg(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def hkv_config_summary(cfg) -> dict:
    """The geometry a ``hier_kv`` config asks for, for logging and for config validation."""
    return {
        "enabled": bool(_cfg(cfg, "enabled", False)),
        "page_size_frames": int(_cfg(cfg, "page_size_frames", 3)),
        "topk_pages": int(_cfg(cfg, "topk_pages", 7)),
        "pin_last_page": bool(_cfg(cfg, "pin_last_page", True)),
        "resident_bonus": float(_cfg(cfg, "resident_bonus", 0.0)),
        "index_layers": _cfg(cfg, "index_layers", "all"),
        "recache_pinned_on_switch": bool(_cfg(cfg, "recache_pinned_on_switch", True)),
    }


class HKVBlockCycle:
    """Mixin: the HKV per-block cycle over a host pipeline's online cache."""

    def hkv_configure(self, cfg, *, model, num_output_frames: int = 0) -> bool:
        """Build the bank from a ``hier_kv`` config. Returns whether HKV is enabled."""
        self.hkv: Optional[HKVCache] = None
        self._hkv_model = model
        self._hkv_qbar: Optional[torch.Tensor] = None
        self._hkv_qbar_prev: Optional[torch.Tensor] = None
        self._hkv_installed: List[Tuple[int, int]] = []
        self._hkv_sink_frames = 0
        self.hkv_region_frames = 0
        self.hkv_region_tokens = 0
        self.hkv_recache_pinned_on_switch = False
        self._hkv_stats = {"blocks": 0, "installed_pages": 0, "page0_selected": 0,
                           "newest_selected": 0, "age_hist": {}, "recaches": 0}
        if not bool(_cfg(cfg, "enabled", False)):
            return False

        page_size_frames = int(_cfg(cfg, "page_size_frames", 3))
        topk_pages = int(_cfg(cfg, "topk_pages", 7))
        pin_last_page = bool(_cfg(cfg, "pin_last_page", True))
        if page_size_frames != self.num_frame_per_block:
            raise ValueError(
                f"hier_kv.page_size_frames={page_size_frames} must equal num_frame_per_block="
                f"{self.num_frame_per_block}: a page is one block, so archiving happens at block "
                "boundaries and the cache layout is expressed in blocks")

        nvme = _cfg(cfg, "nvme", None)
        nvme_enabled = bool(_cfg(nvme, "enabled", False))
        nvme_dir = str(_cfg(nvme, "path", "") or "")

        gpu_budget = _cfg(cfg, "gpu_max_pages", 0)
        if isinstance(gpu_budget, str) and gpu_budget.lower() == "auto":
            gpu_budget = -(-int(num_output_frames) // page_size_frames) if num_output_frames else 0
        gpu_budget = int(gpu_budget)
        total_pages = -(-int(num_output_frames or 0) // page_size_frames)
        host_prefill = max(0, total_pages - max(gpu_budget, 0))

        self.hkv = HKVCache(
            frame_seq_length=self.frame_seq_length,
            page_size_frames=page_size_frames,
            topk_pages=topk_pages,
            stage1_topk_chunks=int(_cfg(cfg, "stage1_topk_chunks", 10_000)),
            max_pages_per_chunk=int(_cfg(cfg, "max_pages_per_chunk", 0)),
            gpu_max_pages=gpu_budget,
            cpu_max_pages=int(_cfg(cfg, "cpu_max_pages", 512)),
            host_prefill_pages=host_prefill,
            nvme_enabled=nvme_enabled,
            nvme_dir=nvme_dir or None,
            rerope=bool(_cfg(cfg, "rerope_retrieved", True)),
            index_layers=_cfg(cfg, "index_layers", "all"),
            pin_last_page=pin_last_page,
            resident_bonus=float(_cfg(cfg, "resident_bonus", 0.0)),
        )
        self.hkv_recache_pinned_on_switch = bool(_cfg(cfg, "recache_pinned_on_switch", True))
        self._hkv_stage_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() and bool(_cfg(cfg, "stage_async", True))
            else None)
        self._hkv_stage_event = None
        region_pages = topk_pages + (1 if pin_last_page else 0)
        self.hkv_region_frames = region_pages * page_size_frames
        self.hkv_region_tokens = self.hkv_region_frames * self.frame_seq_length

        required_local = self.num_frame_per_block + self.hkv_region_frames
        if self.local_attn_size in (None, -1, 0):
            self.local_attn_size = required_local
        elif int(self.local_attn_size) != required_local:
            raise ValueError(
                f"local_attn_size {self.local_attn_size} does not match the paging geometry: "
                f"num_frame_per_block + (topk_pages + pinned) * page_size_frames = "
                f"{self.num_frame_per_block} + {region_pages} * {page_size_frames} = {required_local}")
        model.local_attn_size = required_local
        for block in model.blocks:
            block.self_attn.local_attn_size = required_local
        return True

    @property
    def hkv_enabled(self) -> bool:
        return getattr(self, "hkv", None) is not None

    def hkv_reset(self) -> None:
        """Start a new rollout: empty bank, no query, no region."""
        if self.hkv is not None:
            self.hkv.reset()
        self._hkv_qbar = None
        self._hkv_qbar_prev = None
        self._hkv_installed = []
        self._hkv_stats = {"blocks": 0, "installed_pages": 0, "page0_selected": 0,
                           "newest_selected": 0, "age_hist": {}, "recaches": 0}
        self._hkv_stage_event = None
        self.hkv_set_sink(0)

    def _hkv_is_main(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def hkv_temporal_freqs(self, kv_cache) -> torch.Tensor:
        head_dim = int(kv_cache[0]["k"].shape[-1])
        return temporal_band(self._hkv_model.freqs, head_dim).to(kv_cache[0]["k"].device)

    def hkv_hsa_modules(self, kv_cache) -> Optional[list]:
        """Per-layer HSA modules when the cache carries a compressed tier, else None."""
        if "k_coarse" not in kv_cache[0]:
            return None
        return [block.self_attn.hsa_attention for block in self._hkv_model.blocks]

    def hkv_hsa_attention_modules(self) -> list:
        modules = []
        for block in self._hkv_model.blocks:
            attn = block.self_attn
            if getattr(attn, "use_hsa_kv_cache", False) and getattr(attn, "hsa_attention", None) is not None:
                modules.append(attn.hsa_attention)
        return modules

    def hkv_set_sink(self, num_frames: int) -> None:
        """Mark off the first ``num_frames`` frames of the cache as the non-rolling region."""
        self._hkv_sink_frames = int(num_frames)
        model = getattr(self, "_hkv_model", None)
        if model is None:
            return
        for block in model.blocks:
            block.self_attn.sink_size = int(num_frames)

    @property
    def hkv_sink_frames(self) -> int:
        return int(getattr(self, "_hkv_sink_frames", 0))

    @contextlib.contextmanager
    def hkv_capture_qbar(self):
        """Capture the per-layer mean pre-rotation query of the forward run inside the block."""
        if self.hkv is None:
            yield
            return
        layers = self.hkv.indexed_layers(len(self._hkv_model.blocks))
        sums = {}
        handles = []

        def make_hook(layer):
            def hook(_module, _inputs, output):
                sums[layer] = output.detach().float().mean(dim=(0, 1))
            return hook

        for layer in layers:
            handles.append(self._hkv_model.blocks[layer].self_attn.norm_q.register_forward_hook(make_hook(layer)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
        if len(sums) == len(layers):
            self._hkv_qbar_prev = self._hkv_qbar
            self._hkv_qbar = torch.stack([sums[layer] for layer in layers], dim=0)

    def hkv_select(self, conditional_dict, *, staging: bool) -> List[Tuple[int, int]]:
        """Pages for the next block."""
        return self.hkv.retrieve(
            prompt_embeds=conditional_dict["prompt_embeds"],
            query_vec=self._hkv_qbar if staging else self._hkv_qbar_prev,
            previous=self._hkv_installed,
            include_pinned=not staging,
        )

    def hkv_stage(self, kv_cache, conditional_dict, next_start_frame: int) -> Optional[dict]:
        """Pick and assemble the next block's scored pages without touching the online cache."""
        if self.hkv is None:
            return None
        pinned_frames = self.num_frame_per_block if self.hkv.pin_last_page else 0
        pages = self.hkv_select(conditional_dict, staging=True) if len(self.hkv) > 0 else []
        if not pages:
            return {"staged": None, "pages": [], "pinned_frames": pinned_frames}
        stream = self._hkv_stage_stream
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            staged = self.hkv.prepare_region(
                kv_cache, pages,
                region_tokens=self.hkv_region_tokens - pinned_frames * self.frame_seq_length,
                current_start_frame=next_start_frame - pinned_frames,
                temporal_freqs=self.hkv_temporal_freqs(kv_cache),
                hsa_modules=self.hkv_hsa_modules(kv_cache),
            )
            if stream is not None:
                self._hkv_stage_event = torch.cuda.Event()
                self._hkv_stage_event.record(stream)
        return {"staged": staged, "pages": pages, "pinned_frames": pinned_frames}

    def hkv_before_block(
        self,
        kv_cache,
        conditional_dict,
        block_start_frame: int,
        *,
        staged: Optional[dict] = None,
        switching: bool = False,
        last_latent: Optional[torch.Tensor] = None,
        generator=None,
        crossattn_cache=None,
        context_noise: int = 0,
    ) -> int:
        """Install the region for the block about to be generated."""
        if self.hkv is None:
            return 0
        if len(self.hkv) == 0:
            self.hkv_set_sink(0)
            self._hkv_installed = []
            self._hkv_after_install(kv_cache)
            return 0

        with torch.no_grad():
            if staged is not None:
                pages = list(staged["pages"])
                installed = 0
                if staged["staged"] is not None:
                    if self._hkv_stage_event is not None:
                        torch.cuda.current_stream().wait_event(self._hkv_stage_event)
                        self._hkv_stage_event = None
                    installed = self.hkv.commit_region(
                        kv_cache, staged["staged"], block_start_frame - staged["pinned_frames"],
                        self.hkv_hsa_modules(kv_cache))
                if staged["pinned_frames"] > 0:
                    last = self.hkv.last_page()
                    installed += self.hkv.append_pages(
                        kv_cache, [last],
                        offset_tokens=installed,
                        current_start_frame=block_start_frame,
                        temporal_freqs=self.hkv_temporal_freqs(kv_cache),
                        hsa_modules=self.hkv_hsa_modules(kv_cache),
                    )
                    pages.append(last)
                    self.hkv.last_pinned = [last]
            else:
                pages = self.hkv_select(conditional_dict, staging=False)
                installed = self.hkv.install_region(
                    kv_cache, pages,
                    region_tokens=self.hkv_region_tokens,
                    current_start_frame=block_start_frame,
                    temporal_freqs=self.hkv_temporal_freqs(kv_cache),
                    hsa_modules=self.hkv_hsa_modules(kv_cache),
                ) if pages else 0
        installed_frames = installed // self.frame_seq_length
        self._hkv_installed = list(pages)

        recached = False
        if (switching and self.hkv_recache_pinned_on_switch and self.hkv.pin_last_page
                and last_latent is not None and generator is not None
                and installed_frames >= self.num_frame_per_block
                and self.hkv.last_page() in pages):
            self._hkv_recache_pinned(
                generator, kv_cache, crossattn_cache, last_latent, conditional_dict,
                block_start_frame, installed_frames, context_noise)
            recached = True

        self.hkv_set_sink(installed_frames)
        self._hkv_after_install(kv_cache)
        self._hkv_record(pages, block_start_frame, installed_frames, recached)
        return installed_frames

    def _hkv_recache_pinned(self, generator, kv_cache, crossattn_cache, last_latent,
                            conditional_dict, block_start_frame, installed_frames, context_noise):
        """Recompute the pinned page's keys and values under the current prompt, in place."""
        block_frames = self.num_frame_per_block
        self.hkv_set_sink(installed_frames - block_frames)
        timestep = torch.full(
            (last_latent.shape[0], last_latent.shape[1]), int(context_noise),
            device=last_latent.device, dtype=torch.int64)
        with torch.no_grad():
            generator(
                noisy_image_or_video=last_latent,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=(block_start_frame - block_frames) * self.frame_seq_length,
            )
        self._hkv_stats["recaches"] += 1

    def _hkv_after_install(self, kv_cache) -> None:
        """The block's KV context is final: reset the HSA per-block state to match it."""
        stable_tokens = self.hkv_sink_frames * self.frame_seq_length
        for module in self.hkv_hsa_attention_modules():
            module.stable_kv_tokens = stable_tokens
            module.begin_decode_block()

    def hkv_after_block(self, kv_cache, conditional_dict, block_start_frame: int) -> None:
        """Archive the block that just finished as one page, before the next block claims its span."""
        if self.hkv is None:
            return
        sink_tokens = self.hkv_sink_frames * self.frame_seq_length
        self.hkv.store_span(
            kv_cache,
            prompt_embeds=conditional_dict["prompt_embeds"],
            token_start=sink_tokens,
            token_end=int(kv_cache[0]["local_end_index"].item()),
            start_frame=block_start_frame,
            temporal_freqs=self.hkv_temporal_freqs(kv_cache),
        )

    def _hkv_record(self, pages, block_start_frame, installed_frames, recached) -> None:
        stats = self._hkv_stats
        stats["blocks"] += 1
        stats["installed_pages"] += len(pages)
        newest = self.hkv.last_page()
        if any(page == (0, 0) for page in pages):
            stats["page0_selected"] += 1
        if newest is not None and newest in pages:
            stats["newest_selected"] += 1
        for page in pages:
            first_frame, _ = self.hkv.page_frames(page)
            age_blocks = (block_start_frame - first_frame) // max(self.num_frame_per_block, 1)
            stats["age_hist"][age_blocks] = stats["age_hist"].get(age_blocks, 0) + 1
        if self._hkv_is_main():
            spans = ",".join(f"{a}-{b}" for a, b in (self.hkv.page_frames(p) for p in pages))
            print(f"[HKV] frame {block_start_frame}: installed {installed_frames} frame(s) from "
                  f"{len(self.hkv)} banked page(s) | frames [{spans}] | pinned {self.hkv.last_pinned}"
                  f"{' | recached pinned page' if recached else ''} | tier={self.hkv.tier.stats()}")

    def hkv_stats_summary(self) -> dict:
        """What retrieval actually chose over the rollout: recency and sink emergence."""
        stats = dict(self._hkv_stats)
        blocks = max(stats["blocks"], 1)
        hist = stats.pop("age_hist")
        total = max(sum(hist.values()), 1)
        stats["page0_fraction"] = stats["page0_selected"] / blocks
        stats["newest_fraction"] = stats["newest_selected"] / blocks
        stats["age_blocks_mean"] = sum(age * n for age, n in hist.items()) / total
        stats["age_hist"] = dict(sorted(hist.items()))
        return stats
