"""Hierarchical KV Cache (HKV)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .retrieval import (
    page_key_index_layers, prompt_index, score_pages_layers, select_top_chunks, select_topk_pages,
)
from .rope import shift_temporal_rope
from .tiering import HKVTierManager

__all__ = ["HKVCache"]


class HKVCache:
    """KV bank with two-stage page retrieval and multi-tier residency."""

    def __init__(
        self,
        *,
        frame_seq_length: int,
        page_size_frames: int = 8,
        topk_pages: int = 4,
        stage1_topk_chunks: int = 1,
        max_pages_per_chunk: int = 0,
        gpu_max_pages: int = 0,
        cpu_max_pages: int = 512,
        host_prefill_pages: int = 0,
        nvme_enabled: bool = False,
        nvme_dir: Optional[str] = None,
        rerope: bool = True,
        index_layers="all",
        pin_last_page: bool = True,
        resident_bonus: float = 0.0,
    ):
        self.frame_seq_length = int(frame_seq_length)
        self.page_size_frames = int(page_size_frames)
        self.topk_pages = int(topk_pages)
        self.stage1_topk_chunks = int(stage1_topk_chunks)
        self.max_pages_per_chunk = int(max_pages_per_chunk)
        self.rerope = bool(rerope)
        self.index_layers = index_layers
        self.pin_last_page = bool(pin_last_page)
        self.resident_bonus = float(resident_bonus)
        self.last_scores: dict = {}
        self.last_pinned: List[Tuple[int, int]] = []
        self.host_prefill_pages = int(host_prefill_pages)
        self.tier = HKVTierManager(
            gpu_max_pages=gpu_max_pages,
            cpu_max_pages=cpu_max_pages,
            nvme_enabled=nvme_enabled,
            nvme_dir=nvme_dir,
        )
        self.reset()

    def reset(self) -> None:
        """Drop the bank and start a fresh rollout, keeping the tier pools allocated."""
        self.chunks: List[dict] = []
        self.prompt_bank: List[torch.Tensor] = []
        self.last_scores = {}
        self.last_pinned = []
        self.tier.clear()

    def indexed_layers(self, num_layers: int) -> List[int]:
        """Resolve ``index_layers`` against a cache with ``num_layers`` layers."""
        spec = self.index_layers
        if isinstance(spec, str):
            if spec.lower() == "all":
                return list(range(num_layers))
            if spec.lower() == "first":
                return [0]
            raise ValueError(f"index_layers must be 'all', 'first' or a list of ids, got {spec!r}")
        layers = [int(layer) for layer in spec]
        if not layers or any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError(f"index_layers {layers} out of range for {num_layers} cache layers")
        return layers

    def last_page(self) -> Optional[Tuple[int, int]]:
        """The most recently archived page, or None for an empty bank."""
        if not self.chunks:
            return None
        return (len(self.chunks) - 1, len(self.chunks[-1]["page_spans"]) - 1)

    def page_frames(self, page: Tuple[int, int]) -> Tuple[int, int]:
        """``(first_frame, end_frame)`` of a page in video time."""
        chunk = self.chunks[page[0]]
        start, end = chunk["page_spans"][page[1]]
        return (chunk["start_frame"] + start // self.frame_seq_length,
                chunk["start_frame"] + -(-end // self.frame_seq_length))

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def page_tokens(self) -> int:
        return self.page_size_frames * self.frame_seq_length

    def store(
        self,
        kv_cache: Sequence[dict],
        *,
        prompt_embeds: torch.Tensor,
        current_start_frame: int,
        temporal_freqs: torch.Tensor,
    ) -> int:
        """Write the populated part of the online cache into the bank as a new chunk."""
        valid_tokens = int(kv_cache[0]["local_end_index"].item())
        if valid_tokens <= 0:
            return -1
        return self.store_span(
            kv_cache,
            prompt_embeds=prompt_embeds,
            token_start=0,
            token_end=valid_tokens,
            start_frame=int(current_start_frame - valid_tokens // self.frame_seq_length),
            temporal_freqs=temporal_freqs,
        )

    def store_span(
        self,
        kv_cache: Sequence[dict],
        *,
        prompt_embeds: torch.Tensor,
        token_start: int,
        token_end: int,
        start_frame: int,
        temporal_freqs: torch.Tensor,
    ) -> int:
        """Archive ``cache[token_start:token_end]`` as a new chunk starting at ``start_frame``."""
        span_tokens = int(token_end) - int(token_start)
        if span_tokens <= 0:
            return -1

        chunk_id = len(self.chunks)
        valid_tokens = span_tokens
        page_spans = [
            (start, min(start + self.page_tokens, valid_tokens))
            for start in range(0, valid_tokens, self.page_tokens)
        ]
        start_frame = int(start_frame)

        self._configure_tier(kv_cache)
        with torch.no_grad():
            for page_id, (start, end) in enumerate(page_spans):
                page = self.tier.allocate((chunk_id, page_id))
                take = end - start
                for layer, block in enumerate(kv_cache):
                    page[layer, 0, :, :take].copy_(
                        block["k"][:, token_start + start:token_start + end].detach()
                    )
                    page[layer, 1, :, :take].copy_(
                        block["v"][:, token_start + start:token_start + end].detach()
                    )

            index_device = self.prompt_bank[0].device if self.prompt_bank else kv_cache[0]["k"].device
            page_index = page_key_index_layers(
                kv_cache, token_start, token_end, page_spans,
                frame_seq_length=self.frame_seq_length,
                start_frame=start_frame,
                temporal_freqs=temporal_freqs,
                layers=self.indexed_layers(len(kv_cache)),
                device=index_device,
            )
            prompt_vec = prompt_index(prompt_embeds).detach().to(index_device)

        self.chunks.append({
            "chunk_id": chunk_id,
            "valid_tokens": valid_tokens,
            "page_spans": page_spans,
            "start_frame": start_frame,
            "page_index": page_index,
        })
        self.prompt_bank.append(prompt_vec)
        return chunk_id

    def _configure_tier(self, kv_cache: Sequence[dict]) -> None:
        """Hand the tier manager the page geometry so it can build its pools."""
        sample = kv_cache[0]["k"]
        self.tier.prefill_host_pages = max(self.tier.prefill_host_pages, self.host_prefill_pages)
        self.tier.configure(
            layers=len(kv_cache),
            batch=int(sample.shape[0]),
            page_tokens=self.page_tokens,
            heads=int(sample.shape[2]),
            head_dim=int(sample.shape[3]),
            dtype=sample.dtype,
            device=sample.device,
        )

    def _chronological(self, pages: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return sorted(
            pages,
            key=lambda pair: (
                self.chunks[pair[0]]["start_frame"],
                self.chunks[pair[0]]["page_spans"][pair[1]][0],
            ),
        )

    def retrieve(
        self,
        *,
        prompt_embeds: torch.Tensor,
        query_vec: Optional[torch.Tensor],
        previous: Optional[Sequence[Tuple[int, int]]] = None,
        include_pinned: bool = True,
    ) -> List[Tuple[int, int]]:
        """Select the pages to install for the block about to be generated."""
        self.last_scores, self.last_pinned = {}, []
        if not self.chunks:
            return []

        pinned: List[Tuple[int, int]] = []
        if self.pin_last_page:
            last = self.last_page()
            if last is not None and include_pinned:
                pinned.append(last)
        self.last_pinned = list(pinned)
        budget = self.topk_pages
        if budget <= 0 or query_vec is None:
            return self._chronological(pinned)

        bank = torch.stack(self.prompt_bank, dim=0)
        candidates = select_top_chunks(
            prompt_index(prompt_embeds).detach(), bank, self.stage1_topk_chunks
        )

        pool: List[Tuple[int, int, torch.Tensor]] = []
        for chunk_id in candidates:
            page_index = self.chunks[chunk_id].get("page_index")
            if page_index is None or page_index.numel() == 0:
                continue
            for page_id in range(page_index.shape[0]):
                if (chunk_id, page_id) in pinned:
                    continue
                pool.append((chunk_id, page_id, page_index[page_id]))
        if not pool:
            return self._chronological(pinned)

        query = query_vec
        if query.dim() == 1 and query.shape[0] != pool[0][2].shape[-1]:
            raise ValueError(
                f"query dim {query.shape[0]} does not match page index dim {pool[0][2].shape[-1]}"
            )
        if query.dim() == 2 and query.shape[-1] != pool[0][2].shape[-1]:
            raise ValueError(
                f"query dim {query.shape[-1]} does not match page index dim {pool[0][2].shape[-1]}"
            )

        bonus = None
        if self.resident_bonus > 0.0 and previous:
            resident = set(tuple(p) for p in previous)
            index = torch.stack([c[2].float() for c in pool], dim=0)
            scores = score_pages_layers(query.float(), index)
            spread = float(scores.max() - scores.min()) if scores.numel() > 1 else 0.0
            bonus = torch.tensor(
                [self.resident_bonus * spread if (c, p) in resident else 0.0 for c, p, _ in pool],
                dtype=torch.float32,
            )

        selected = select_topk_pages(
            query, pool, budget, max_per_chunk=self.max_pages_per_chunk or None, bonus=bonus
        )
        self.last_scores = {(c, p): float(score) for c, p, score in selected}
        return self._chronological(pinned + [(c, p) for c, p, _ in selected])

    def install_region(
        self,
        kv_cache: Sequence[dict],
        selected_pairs: Sequence[Tuple[int, int]],
        *,
        region_tokens: int,
        current_start_frame: int,
        temporal_freqs: torch.Tensor,
        hsa_modules: Optional[Sequence] = None,
    ) -> int:
        """Write the selected pages into ``cache[0:region_tokens]``, leaving the rest untouched."""
        region_tokens = int(region_tokens)
        if region_tokens <= 0:
            return 0

        entries, pages = self._resolve_pages(selected_pairs)
        device = kv_cache[0]["k"].device
        dtype = kv_cache[0]["k"].dtype

        budget = min(region_tokens, int(kv_cache[0]["k"].shape[1]))
        installed: List[Tuple[Tuple[int, int], int, int]] = []
        for key, start, end in entries:
            if budget <= 0:
                break
            take = min(end - start, budget)
            installed.append((key, start, take))
            budget -= take
        total_tokens = sum(take for _, _, take in installed)
        if total_tokens <= 0:
            return 0

        frame_moves = self._plan_rerope(installed, current_start_frame) if self.rerope else None

        has_coarse = "k_coarse" in kv_cache[0] and "v_coarse" in kv_cache[0]
        if has_coarse and hsa_modules is None:
            raise ValueError(
                "the online cache carries a compressed tier, so hsa_modules must be supplied so it "
                "can be rebuilt for the installed pages"
            )

        staged = self._stage_region(
            installed, pages, frame_moves, temporal_freqs, device, dtype,
            layers=len(kv_cache), total_tokens=total_tokens,
        )
        self.commit_region(kv_cache, staged, current_start_frame, hsa_modules)
        return total_tokens

    def prepare_region(
        self,
        kv_cache: Sequence[dict],
        selected_pairs: Sequence[Tuple[int, int]],
        *,
        region_tokens: int,
        current_start_frame: int,
        temporal_freqs: torch.Tensor,
        hsa_modules: Optional[Sequence] = None,
    ) -> Optional[dict]:
        """Assemble the region's contents without touching the cache."""
        region_tokens = int(region_tokens)
        if region_tokens <= 0:
            return None

        entries, pages = self._resolve_pages(selected_pairs)
        device = kv_cache[0]["k"].device
        dtype = kv_cache[0]["k"].dtype

        budget = min(region_tokens, int(kv_cache[0]["k"].shape[1]))
        installed: List[Tuple[Tuple[int, int], int, int]] = []
        for key, start, end in entries:
            if budget <= 0:
                break
            take = min(end - start, budget)
            installed.append((key, start, take))
            budget -= take
        total_tokens = sum(take for _, _, take in installed)
        if total_tokens <= 0:
            return None

        frame_moves = self._plan_rerope(installed, current_start_frame) if self.rerope else None
        staged = self._stage_region(
            installed, pages, frame_moves, temporal_freqs, device, dtype,
            layers=len(kv_cache), total_tokens=total_tokens,
        )
        staged["current_start_frame"] = int(current_start_frame)

        if hsa_modules is not None and "k_coarse" in kv_cache[0]:
            buf = staged["buf"]
            coarse = []
            for block_idx, hsa in enumerate(hsa_modules):
                loaded_k = buf[block_idx, 0, :, :total_tokens]
                loaded_v = buf[block_idx, 1, :, :total_tokens]
                c0, c1 = hsa.token_range_to_coarse(0, int(loaded_k.shape[1]))
                with torch.no_grad():
                    k_coarse, v_coarse = hsa.compress_kv_cache(loaded_k, loaded_v)
                coarse.append((k_coarse, v_coarse, c0, c1))
            staged["coarse"] = coarse
        return staged

    def _staging_buffer(self, layers: int, total_tokens: int, device, dtype):
        """Scratch shaped ``[layers, 2, B, total_tokens, H, D]`` for one assembled region."""
        need_pages = -(-total_tokens // self.page_tokens)
        if self.tier.n_staging >= need_pages:
            views = [self.tier.staging_view(i) for i in range(need_pages)]
            if all(v is not None for v in views):
                return torch.cat([v[:, :, :, :, :, :] for v in views], dim=3)[
                    :, :, :, :total_tokens
                ]

        key = (layers, total_tokens, device, dtype)
        if getattr(self, "_stage_key", None) != key:
            sample = self.tier._page_shape
            batch, heads, head_dim = sample[2], sample[4], sample[5]
            self._stage_buf = torch.empty(
                (layers, 2, batch, total_tokens, heads, head_dim), dtype=dtype, device=device
            )
            self._stage_key = key
        return self._stage_buf

    def _stage_region(self, installed, pages, frame_moves, temporal_freqs, device, dtype,
                      *, layers: int, total_tokens: int) -> dict:
        """Gather, re-rope and write the selected pages into staging scratch, one page at a time across every layer."""
        buf = self._staging_buffer(layers, total_tokens, device, dtype)
        freqs = temporal_freqs.to(device)
        offset = 0
        for i, (key, _, take) in enumerate(installed):
            page = pages[key][:layers, :, :, :take].to(device=device)
            keys = page[:, 0]
            if frame_moves is not None:
                orig_frame, new_frame = frame_moves[i]
                if new_frame != orig_frame:
                    keys = shift_temporal_rope(keys, freqs, new_frame, orig_frame)
            buf[:, 0, :, offset:offset + take] = keys.to(dtype=dtype)
            buf[:, 1, :, offset:offset + take] = page[:, 1].to(dtype=dtype)
            offset += take
        return {"buf": buf, "total_tokens": total_tokens}

    def commit_region(
        self,
        kv_cache: Sequence[dict],
        staged: dict,
        current_start_frame: Optional[int] = None,
        hsa_modules: Optional[Sequence] = None,
    ) -> int:
        """Land a staged region into the cache. Contiguous copies and index bookkeeping only."""
        if not staged:
            return 0
        total_tokens = int(staged["total_tokens"])
        if current_start_frame is None:
            current_start_frame = int(staged["current_start_frame"])
        has_coarse = "k_coarse" in kv_cache[0] and "v_coarse" in kv_cache[0]
        if has_coarse and hsa_modules is None:
            raise ValueError(
                "the online cache carries a compressed tier, so hsa_modules must be supplied so it "
                "can be rebuilt for the installed pages"
            )

        buf = staged["buf"]
        coarse = staged.get("coarse")
        if coarse is not None and buf.is_cuda:
            current = torch.cuda.current_stream()
            for ck, cv, _, _ in coarse:
                ck.record_stream(current)
                cv.record_stream(current)
        for block_idx, block in enumerate(kv_cache):
            block["k"][:, :total_tokens] = buf[block_idx, 0, :, :total_tokens]
            block["v"][:, :total_tokens] = buf[block_idx, 1, :, :total_tokens]

            if has_coarse:
                if coarse is not None:
                    ck, cv, c0, c1 = coarse[block_idx]
                    block["k_coarse"][:, c0:c1] = ck.to(
                        dtype=block["k_coarse"].dtype, device=block["k_coarse"].device)
                    block["v_coarse"][:, c0:c1] = cv.to(
                        dtype=block["v_coarse"].dtype, device=block["v_coarse"].device)
                else:
                    self._write_coarse(
                        block, hsa_modules[block_idx],
                        buf[block_idx, 0, :, :total_tokens], buf[block_idx, 1, :, :total_tokens],
                    )

            block["global_end_index"].fill_(current_start_frame * self.frame_seq_length)
            block["local_end_index"].fill_(total_tokens)

        return total_tokens

    def append_pages(
        self,
        kv_cache: Sequence[dict],
        pages: Sequence[Tuple[int, int]],
        *,
        offset_tokens: int,
        current_start_frame: int,
        temporal_freqs: torch.Tensor,
        hsa_modules: Optional[Sequence] = None,
    ) -> int:
        """Write ``pages`` at cache offset ``offset_tokens``, ending right before the block."""
        entries, page_tensors = self._resolve_pages(pages)
        if not entries:
            return 0
        device = kv_cache[0]["k"].device
        dtype = kv_cache[0]["k"].dtype
        offset_tokens = int(offset_tokens)
        budget = int(kv_cache[0]["k"].shape[1]) - offset_tokens
        installed: List[Tuple[Tuple[int, int], int, int]] = []
        for key, start, end in entries:
            if budget <= 0:
                break
            take = min(end - start, budget)
            installed.append((key, start, take))
            budget -= take
        total = sum(take for _, _, take in installed)
        if total <= 0:
            return 0

        has_coarse = "k_coarse" in kv_cache[0] and "v_coarse" in kv_cache[0]
        if has_coarse and hsa_modules is None:
            raise ValueError(
                "the online cache carries a compressed tier, so hsa_modules must be supplied so it "
                "can be rebuilt for the appended pages"
            )
        frame_moves = self._plan_rerope(installed, current_start_frame) if self.rerope else None

        with torch.no_grad():
            freqs = temporal_freqs.to(device)
            layers = len(kv_cache)
            cursor = offset_tokens
            for i, (key, _, take) in enumerate(installed):
                page = page_tensors[key][:layers, :, :, :take].to(device=device)
                keys = page[:, 0]
                if frame_moves is not None:
                    orig_frame, new_frame = frame_moves[i]
                    if new_frame != orig_frame:
                        keys = shift_temporal_rope(keys, freqs, new_frame, orig_frame)
                keys = keys.to(dtype=dtype)
                values = page[:, 1].to(dtype=dtype)
                for layer, block in enumerate(kv_cache):
                    block["k"][:, cursor:cursor + take] = keys[layer]
                    block["v"][:, cursor:cursor + take] = values[layer]
                cursor += take
            for layer, block in enumerate(kv_cache):
                if has_coarse:
                    self._write_coarse(
                        block, hsa_modules[layer],
                        block["k"][:, offset_tokens:offset_tokens + total],
                        block["v"][:, offset_tokens:offset_tokens + total],
                        token_offset=offset_tokens,
                    )
                block["global_end_index"].fill_(int(current_start_frame) * self.frame_seq_length)
                block["local_end_index"].fill_(offset_tokens + total)
        return total

    def _resolve_pages(self, selected_pairs):
        """Look up the selected pages, dropping any the bank can no longer produce."""
        entries, pages = [], {}
        for chunk_id, page_id in selected_pairs:
            if not (0 <= chunk_id < len(self.chunks)):
                continue
            spans = self.chunks[chunk_id]["page_spans"]
            if not (0 <= page_id < len(spans)):
                continue
            key = (chunk_id, page_id)
            if key not in pages:
                page = self.tier.get(key)
                if page is None:
                    continue
                pages[key] = page
            start, end = spans[page_id]
            entries.append((key, start, end))
        return entries, pages

    def _plan_rerope(self, installed, current_start_frame):
        """Frame each installed page moves from and to."""
        moves = []
        base = current_start_frame - sum(take for _, _, take in installed) // self.frame_seq_length
        offset = 0
        for (chunk_id, _), start, take in installed:
            orig_frame = self.chunks[chunk_id]["start_frame"] + start // self.frame_seq_length
            moves.append((orig_frame, base + offset // self.frame_seq_length))
            offset += take
        return moves

    @staticmethod
    def _write_coarse(block: dict, hsa, loaded_k: torch.Tensor, loaded_v: torch.Tensor,
                      token_offset: int = 0) -> None:
        """Rebuild the compressed tier for pages installed at cache offset ``token_offset``."""
        coarse_start, coarse_end = hsa.token_range_to_coarse(
            int(token_offset), int(token_offset) + int(loaded_k.shape[1]))
        with torch.no_grad():
            k_coarse, v_coarse = hsa.compress_kv_cache(loaded_k, loaded_v)
        block["k_coarse"][:, coarse_start:coarse_end] = k_coarse.to(
            dtype=block["k_coarse"].dtype, device=block["k_coarse"].device
        )
        block["v_coarse"][:, coarse_start:coarse_end] = v_coarse.to(
            dtype=block["v_coarse"].dtype, device=block["v_coarse"].device
        )
