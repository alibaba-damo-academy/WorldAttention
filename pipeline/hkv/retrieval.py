"""Two-stage retrieval over the hierarchical KV bank."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .rope import derope_temporal

__all__ = [
    "prompt_index",
    "query_index",
    "page_key_index",
    "page_key_index_layers",
    "score_pages",
    "score_pages_layers",
    "select_top_chunks",
    "select_topk_pages",
]


def prompt_index(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Mean-pool a prompt embedding into a unit vector for Stage-1 cosine similarity."""
    pooled = prompt_embeds.float().mean(dim=1).mean(dim=0)
    return F.normalize(pooled, dim=0)


def query_index(q_tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool pre-rotation attention queries into the Stage-2 query vector ``Qbar``."""
    return q_tokens.float().mean(dim=1).mean(dim=0)


def page_key_index(
    keys: torch.Tensor,
    page_spans: Sequence[Tuple[int, int]],
    *,
    frame_seq_length: int,
    start_frame: int,
    temporal_freqs: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build the per-page mean key ``Kbar`` in content space."""
    if not page_spans:
        return None
    device = keys.device
    num_tokens = keys.shape[1]
    frame_ids = start_frame + (torch.arange(num_tokens, device=device) // frame_seq_length)
    content_keys = derope_temporal(keys, temporal_freqs, frame_ids).flatten(2).float()
    return torch.stack(
        [content_keys[:, start:end].mean(dim=1).mean(dim=0) for start, end in page_spans], dim=0
    )


def page_key_index_layers(
    kv_cache: Sequence[dict],
    token_start: int,
    token_end: int,
    page_spans: Sequence[Tuple[int, int]],
    *,
    frame_seq_length: int,
    start_frame: int,
    temporal_freqs: torch.Tensor,
    layers: Sequence[int],
    device=None,
) -> Optional[torch.Tensor]:
    """Per-page mean keys for several cache layers: ``[num_pages, len(layers), H * D]``."""
    if not page_spans or not layers:
        return None
    per_layer = []
    for layer in layers:
        keys = kv_cache[layer]["k"][:, token_start:token_end].detach()
        if device is not None:
            keys = keys.to(device)
        per_layer.append(page_key_index(
            keys, page_spans, frame_seq_length=frame_seq_length, start_frame=start_frame,
            temporal_freqs=temporal_freqs.to(keys.device),
        ))
    return torch.stack(per_layer, dim=1)


def score_pages_layers(
    query_vec: torch.Tensor,
    page_index: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Combined affinity of every page, over one or more indexed layers."""
    if page_index.numel() == 0:
        return page_index.new_zeros((0,))
    index = page_index.float()
    if index.dim() == 2:
        index = index.unsqueeze(1)
    query = query_vec.to(device=index.device, dtype=index.dtype)
    if query.dim() == 1:
        query = query.unsqueeze(0)
    num_layers = index.shape[1]
    if query.shape[0] != num_layers:
        if query.shape[0] == 1:
            index = index[:, :1]
        elif num_layers == 1:
            query = query[:1]
        else:
            raise ValueError(
                f"query has {query.shape[0]} layers but the page index has {num_layers}")
    if scale is None:
        scale = float(index.shape[-1]) ** 0.5
    logits = torch.einsum("pld,ld->pl", index, query)
    if scale and scale > 0:
        logits = logits / scale
    return torch.softmax(logits, dim=0).mean(dim=1)


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
    """Stage 1: chunk ids whose prompt index is most cosine-similar to ``query_vec``."""
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
    max_per_chunk: Optional[int] = None,
    bonus: Optional[torch.Tensor] = None,
) -> List[Tuple[int, int, float]]:
    """Stage 2: top-``K_p`` pages across all candidate chunks."""
    if topk <= 0 or len(candidates) == 0:
        return []
    vectors = [c[2].float() for c in candidates]
    if vectors[0].dim() == 1:
        index = torch.stack([v.reshape(-1) for v in vectors], dim=0)
    else:
        index = torch.stack(vectors, dim=0)
    scores = score_pages_layers(query_vec.float(), index, scale=scale)
    if bonus is not None:
        scores = scores + bonus.to(device=scores.device, dtype=scores.dtype).reshape(-1)

    cap = int(max_per_chunk) if max_per_chunk else None
    taken: dict = {}
    out: List[Tuple[int, int, float]] = []
    for position in torch.argsort(scores, descending=True).tolist():
        if len(out) >= int(topk):
            break
        chunk_id, page_id, _ = candidates[position]
        if cap is not None and taken.get(chunk_id, 0) >= cap:
            continue
        taken[chunk_id] = taken.get(chunk_id, 0) + 1
        out.append((int(chunk_id), int(page_id), float(scores[position].item())))
    return out
