"""Hybrid Sparse Attention: linear branch, head-adaptive block-sparse branch, gated fusion."""

from .attention import (
    HSAAttention,
    compress_kv,
    hsa_attention,
    hsa_parameter_names,
    require_trained_hsa,
)
from .routing import BlockRouting, block_attention_map, block_mean_pool, build_block_routing

__all__ = [
    "HSAAttention",
    "hsa_attention",
    "compress_kv",
    "hsa_parameter_names",
    "require_trained_hsa",
    "BlockRouting",
    "build_block_routing",
    "block_attention_map",
    "block_mean_pool",
]
