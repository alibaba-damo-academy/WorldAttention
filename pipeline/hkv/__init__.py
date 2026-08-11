"""Hierarchical KV Cache: paged KV bank, two-stage retrieval, multi-tier residency."""

from .cache import HKVCache
from .retrieval import (
    page_key_index,
    prompt_index,
    query_index,
    score_pages,
    select_top_chunks,
    select_topk_pages,
)
from .rope import derope_temporal, shift_temporal_rope, temporal_complex_dims
from .tiering import HKVTierManager

__all__ = [
    "HKVCache",
    "HKVTierManager",
    "prompt_index",
    "query_index",
    "page_key_index",
    "score_pages",
    "select_top_chunks",
    "select_topk_pages",
    "shift_temporal_rope",
    "derope_temporal",
    "temporal_complex_dims",
]
