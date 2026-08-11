"""Two-stage retrieval over the hierarchical KV bank.

Stage 1 narrows the bank to the chunks whose prompt embedding is most similar to the current
prompt, by cosine similarity. Stage 2 pools the pages of those candidate chunks and takes the
global top-``K_p`` by the affinity

    alpha = (Qbar . Kbar) / sqrt(d)

between the current chunk's mean query and each page's mean key. Pooling across candidates before
the top-K matters: selecting within a single chunk degenerates whenever the chunk holds no more
pages than the budget.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .rope import derope_temporal

__all__ = [
    "prompt_index",
    "query_index",
    "page_key_index",
    "score_pages",
    "select_top_chunks",
    "select_topk_pages",
]


def prompt_index(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Mean-pool a prompt embedding into a unit vector for Stage-1 cosine similarity.

    ``prompt_embeds`` is ``[B, tokens, dim]``; returns ``[dim]``.
    """
    pooled = prompt_embeds.float().mean(dim=1).mean(dim=0)
    return F.normalize(pooled, dim=0)


def query_index(q_tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool pre-rotation attention queries into the Stage-2 query vector ``Qbar``.

    ``q_tokens`` is ``[B, tokens, H * D]`` taken before the rotary embedding is applied, so it lives
    in the same content space as :func:`page_key_index`. Returns ``[H * D]``.
    """
    return q_tokens.float().mean(dim=1).mean(dim=0)


def page_key_index(
    keys: torch.Tensor,
    page_spans: Sequence[Tuple[int, int]],
    *,
    frame_seq_length: int,
    start_frame: int,
    temporal_freqs: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build the per-page mean key ``Kbar`` in content space.

    ``keys`` is ``[B, tokens, H, D]`` as stored in the cache, i.e. already rotated. Each token's
    temporal rotation is removed before pooling so the index matches a pre-rotation query.
    Returns ``[num_pages, H * D]``, or None when there are no pages.
    """
    if not page_spans:
        return None
    device = keys.device
    num_tokens = keys.shape[1]
    frame_ids = start_frame + (torch.arange(num_tokens, device=device) // frame_seq_length)
    content_keys = derope_temporal(keys, temporal_freqs, frame_ids).flatten(2).float()
    return torch.stack(
        [content_keys[:, start:end].mean(dim=1).mean(dim=0) for start, end in page_spans], dim=0
    )


def score_pages(
    query_vec: torch.Tensor,
    page_index: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Affinity ``alpha_j = (Qbar . Kbar_j) / sqrt(d)`` for every page in ``page_index``."""
    if page_index.numel() == 0:
        return page_index.new_zeros((0,))
    query = query_vec.to(device=page_index.device, dtype=page_index.dtype).reshape(-1)
    scores = page_index @ query
    if scale is None:
        scale = float(page_index.shape[-1]) ** 0.5
    if scale and scale > 0:
        scores = scores / scale
    return scores


def select_top_chunks(query_vec: torch.Tensor, chunk_index: torch.Tensor, topk: int) -> List[int]:
    """Stage 1: chunk ids whose prompt index is most cosine-similar to ``query_vec``.

    ``chunk_index`` is ``[num_chunks, dim]`` of unit vectors; ``query_vec`` is a unit vector.
    """
    if chunk_index.numel() == 0 or topk <= 0:
        return []
    query = query_vec.to(device=chunk_index.device, dtype=chunk_index.dtype).reshape(-1)
    scores = chunk_index @ query
    k = min(int(topk), scores.shape[0])
    return [int(i) for i in torch.topk(scores, k=k, largest=True, sorted=True).indices.tolist()]


def select_topk_pages(
    query_vec: torch.Tensor,
    candidates: Sequence[Tuple[int, int, torch.Tensor]],
    topk: int,
    scale: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    """Stage 2: global top-``K_p`` pages across all candidate chunks.

    ``candidates`` holds ``(chunk_id, page_id, page_index_vector)`` triples. Returns
    ``(chunk_id, page_id, score)`` highest-first, at most ``topk`` entries.
    """
    if topk <= 0 or len(candidates) == 0:
        return []
    index = torch.stack([c[2].reshape(-1).float() for c in candidates], dim=0)
    scores = score_pages(query_vec.float(), index, scale=scale)
    k = min(int(topk), scores.shape[0])
    top = torch.topk(scores, k=k, largest=True, sorted=True)
    out = []
    for rank in range(k):
        chunk_id, page_id, _ = candidates[int(top.indices[rank].item())]
        out.append((int(chunk_id), int(page_id), float(top.values[rank].item())))
    return out
