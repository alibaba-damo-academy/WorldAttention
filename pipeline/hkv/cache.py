"""Hierarchical KV Cache (HKV).

HKV keeps the history of an autoregressive video rollout outside GPU memory and pages the relevant
part back in on demand. A finished segment is written into a KV bank as a *chunk*, each chunk is
split into fixed-size *pages* of ``page_size_frames`` consecutive frames, and every page carries a
mean-key index. When the prompt changes, two-stage retrieval picks the ``K_p`` pages most relevant
to the new prompt and installs exactly those pages into the online cache, so GPU residency is
bounded by the page budget instead of growing with video length.

Online cache contract
---------------------
The online cache is a list with one entry per transformer block, each a dict holding:

``k``, ``v``
    ``[B, capacity, H, D]`` token caches.
``global_end_index``, ``local_end_index``
    scalar tensors tracking how far generation has advanced and how much of the cache is populated.
``k_coarse``, ``v_coarse`` *(optional)*
    the compressed cache read by the HSA linear branch, sized by
    :meth:`~worldattention.hsa.attention.HSAAttention.coarse_cache_size`.

When the compressed cache is present, :meth:`HKVCache.apply` rebuilds it from the pages it installs.
The two HSA branches read the same KV that way; leaving it untouched would have the linear branch
serve the pre-switch window while the sparse branch serves the retrieved pages.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .retrieval import page_key_index, prompt_index, select_top_chunks, select_topk_pages
from .rope import shift_temporal_rope
from .tiering import HKVTierManager

__all__ = ["HKVCache"]


class HKVCache:
    """KV bank with two-stage page retrieval and multi-tier residency.

    Args:
        frame_seq_length: tokens per latent frame.
        page_size_frames: frames per page.
        topk_pages: page budget ``K_p`` installed per retrieval.
        stage1_topk_chunks: how many prompt-similar chunks feed Stage 2.
        cpu_max_chunks: CPU-resident chunk capacity.
        nvme_enabled, nvme_dir: enable and locate the NVMe tier.
        rerope: re-rotate retrieved keys to the cache slots they are replayed at.
    """

    def __init__(
        self,
        *,
        frame_seq_length: int,
        page_size_frames: int = 8,
        topk_pages: int = 4,
        stage1_topk_chunks: int = 1,
        cpu_max_chunks: int = 64,
        nvme_enabled: bool = False,
        nvme_dir: Optional[str] = None,
        rerope: bool = True,
    ):
        self.frame_seq_length = int(frame_seq_length)
        self.page_size_frames = int(page_size_frames)
        self.topk_pages = int(topk_pages)
        self.stage1_topk_chunks = int(stage1_topk_chunks)
        self.rerope = bool(rerope)
        self._tier_kwargs = dict(
            cpu_max_chunks=cpu_max_chunks, nvme_enabled=nvme_enabled, nvme_dir=nvme_dir
        )
        self.reset()

    def reset(self) -> None:
        """Drop the bank and start a fresh rollout."""
        self.chunks: List[dict] = []
        self.prompt_bank: List[torch.Tensor] = []
        self.tier = HKVTierManager(**self._tier_kwargs)

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
        """Write the populated part of the online cache into the bank as a new chunk.

        Returns the new chunk id, or -1 when the cache holds nothing to store.
        """
        valid_tokens = int(kv_cache[0]["local_end_index"].item())
        if valid_tokens <= 0:
            return -1

        chunk_id = len(self.chunks)
        blob = {"k": [], "v": []}
        for block in kv_cache:
            blob["k"].append(block["k"][:, :valid_tokens].detach().to("cpu", copy=True))
            blob["v"].append(block["v"][:, :valid_tokens].detach().to("cpu", copy=True))

        page_spans = [
            (start, min(start + self.page_tokens, valid_tokens))
            for start in range(0, valid_tokens, self.page_tokens)
        ]
        start_frame = int(current_start_frame - valid_tokens // self.frame_seq_length)
        index_device = self.prompt_bank[0].device if self.prompt_bank else kv_cache[0]["k"].device

        page_index = page_key_index(
            blob["k"][0].to(index_device),
            page_spans,
            frame_seq_length=self.frame_seq_length,
            start_frame=start_frame,
            temporal_freqs=temporal_freqs.to(index_device),
        )

        self.chunks.append({
            "chunk_id": chunk_id,
            "valid_tokens": valid_tokens,
            "page_spans": page_spans,
            "start_frame": start_frame,
            "page_index": page_index,
        })
        self.prompt_bank.append(prompt_index(prompt_embeds).detach().to(index_device))
        self.tier.put(chunk_id, blob)
        return chunk_id

    def retrieve(
        self,
        *,
        prompt_embeds: torch.Tensor,
        query_vec: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """Select the pages to install for the incoming prompt.

        Returns ``(chunk_id, page_id)`` pairs in chronological order, so the installed pages keep
        their original temporal ordering in the cache.
        """
        if not self.chunks:
            return []

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
                pool.append((chunk_id, page_id, page_index[page_id]))
        if not pool:
            return []

        query = query_vec.reshape(-1)
        if query.shape[0] != pool[0][2].reshape(-1).shape[0]:
            raise ValueError(
                f"query dim {query.shape[0]} does not match page index dim "
                f"{pool[0][2].reshape(-1).shape[0]}"
            )

        selected = select_topk_pages(query, pool, self.topk_pages)
        return sorted(
            [(chunk_id, page_id) for chunk_id, page_id, _ in selected],
            key=lambda pair: (
                self.chunks[pair[0]]["start_frame"],
                self.chunks[pair[0]]["page_spans"][pair[1]][0],
            ),
        )

    def apply(
        self,
        kv_cache: Sequence[dict],
        selected_pairs: Sequence[Tuple[int, int]],
        *,
        current_start_frame: int,
        temporal_freqs: torch.Tensor,
        hsa_modules: Optional[Sequence] = None,
    ) -> int:
        """Install the selected pages into the online cache, replacing its contents.

        ``hsa_modules`` is the per-block sequence of :class:`HSAAttention` modules; it is required
        whenever the cache carries a compressed tier, because each block owns its own projection.
        Returns the number of tokens installed.
        """
        entries, blobs = self._resolve_pages(selected_pairs)
        if not entries:
            return 0

        capacity = int(kv_cache[0]["k"].shape[1])
        device = kv_cache[0]["k"].device
        dtype = kv_cache[0]["k"].dtype

        budget = capacity
        installed: List[Tuple[int, int, int]] = []
        for chunk_id, start, end in entries:
            if budget <= 0:
                break
            take = min(end - start, budget)
            installed.append((chunk_id, start, start + take))
            budget -= take
        total_tokens = sum(end - start for _, start, end in installed)

        frame_moves = self._plan_rerope(installed, current_start_frame) if self.rerope else None

        has_coarse = "k_coarse" in kv_cache[0] and "v_coarse" in kv_cache[0]
        if has_coarse and hsa_modules is None:
            raise ValueError(
                "the online cache carries a compressed tier, so hsa_modules must be supplied so it "
                "can be rebuilt for the installed pages"
            )

        for block_idx, block in enumerate(kv_cache):
            block["k"].zero_()
            block["v"].zero_()
            if has_coarse:
                block["k_coarse"].zero_()
                block["v_coarse"].zero_()

            if total_tokens > 0:
                keys = []
                for i, (chunk_id, start, end) in enumerate(installed):
                    segment = blobs[chunk_id]["k"][block_idx][:, start:end].to(device=device)
                    if frame_moves is not None:
                        orig_frame, new_frame = frame_moves[i]
                        segment = shift_temporal_rope(
                            segment, temporal_freqs.to(device), new_frame, orig_frame
                        )
                    keys.append(segment)
                values = [
                    blobs[chunk_id]["v"][block_idx][:, start:end].to(device=device)
                    for chunk_id, start, end in installed
                ]
                loaded_k = torch.cat(keys, dim=1).to(dtype=dtype)
                loaded_v = torch.cat(values, dim=1).to(dtype=dtype)
                block["k"][:, : loaded_k.shape[1]] = loaded_k
                block["v"][:, : loaded_v.shape[1]] = loaded_v

                if has_coarse:
                    self._write_coarse(block, hsa_modules[block_idx], loaded_k, loaded_v)

            block["global_end_index"].fill_(current_start_frame * self.frame_seq_length)
            block["local_end_index"].fill_(total_tokens)

        return total_tokens

    def _resolve_pages(self, selected_pairs):
        entries, blobs = [], {}
        for chunk_id, page_id in selected_pairs:
            if not (0 <= chunk_id < len(self.chunks)):
                continue
            spans = self.chunks[chunk_id]["page_spans"]
            if not (0 <= page_id < len(spans)):
                continue
            if chunk_id not in blobs:
                blob = self.tier.get(chunk_id)
                if blob is None:
                    continue
                blobs[chunk_id] = blob
            start, end = spans[page_id]
            entries.append((chunk_id, start, end))
        return entries, blobs

    def _plan_rerope(self, installed, current_start_frame):
        """Frame each installed page moves from and to.

        Installed pages occupy consecutive frames ending at ``current_start_frame``, matching the
        cache's end-index convention, so the incoming query sees them at correct relative positions.
        """
        moves = []
        base = current_start_frame - sum(e - s for _, s, e in installed) // self.frame_seq_length
        offset = 0
        for chunk_id, start, end in installed:
            orig_frame = self.chunks[chunk_id]["start_frame"] + start // self.frame_seq_length
            moves.append((orig_frame, base + offset // self.frame_seq_length))
            offset += end - start
        return moves

    @staticmethod
    def _write_coarse(block: dict, hsa, loaded_k: torch.Tensor, loaded_v: torch.Tensor) -> None:
        """Rebuild the compressed tier for pages installed at cache offset 0."""
        coarse_start, coarse_end = hsa.token_range_to_coarse(0, int(loaded_k.shape[1]))
        with torch.no_grad():
            k_coarse, v_coarse = hsa.compress_kv_cache(loaded_k, loaded_v)
        block["k_coarse"][:, coarse_start:coarse_end] = k_coarse.to(
            dtype=block["k_coarse"].dtype, device=block["k_coarse"].device
        )
        block["v_coarse"][:, coarse_start:coarse_end] = v_coarse.to(
            dtype=block["v_coarse"].dtype, device=block["v_coarse"].device
        )
