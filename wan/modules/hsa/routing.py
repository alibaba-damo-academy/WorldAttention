"""Head-adaptive block routing for the HSA sparse branch.

Blocks are selected per (query block, head) by cumulative attention mass rather than by a fixed
Top-K. The coverage threshold is head-specific and derived from the Gini coefficient of the head's
block-attention distribution, so heads with peaked attention keep more mass and heads with diffuse
attention keep less.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "BlockRouting",
    "build_block_routing",
    "block_attention_map",
    "block_mean_pool",
    "block_counts",
    "pad_to_len",
    "variable_block_sizes",
]


class BlockRouting:
    """Result of routing: which key blocks each query block attends to.

    Attributes:
        index: ``[B, H, Nq, Nkv]`` int32, key-block ids ordered by descending block-attention
            weight. Only the first ``num[b, h, i]`` entries of row ``(b, h, i)`` are selected.
        num: ``[B, H, Nq]`` int32, number of selected blocks per query block.
        mask: ``[B, H, Nq, Nkv]`` bool selection mask, or None when not requested.
        thresholds: ``[H]`` the per-head coverage threshold that produced this routing.
    """

    __slots__ = ("index", "num", "mask", "thresholds")

    def __init__(self, index, num, mask, thresholds):
        self.index = index
        self.num = num
        self.mask = mask
        self.thresholds = thresholds


def block_mean_pool(x: torch.Tensor, logical_len: int, num_blocks: int, block_size: int) -> torch.Tensor:
    """Mean-pool ``[B, H, num_blocks * block_size, D]`` into ``[B, H, num_blocks, D]``.

    ``logical_len`` is the number of real tokens; a trailing partial block is averaged over its
    real tokens only, so zero padding does not bias the pooled key.
    """
    bsz, heads, _, head_dim = x.shape
    blocks = x.view(bsz, heads, num_blocks, block_size, head_dim)
    if logical_len % block_size == 0:
        return blocks.mean(dim=3)

    sizes = torch.full((num_blocks,), block_size, device=x.device, dtype=torch.float32)
    sizes[-1] = logical_len - (num_blocks - 1) * block_size
    token_ids = torch.arange(block_size, device=x.device).view(1, 1, 1, block_size, 1)
    valid = token_ids < sizes.to(torch.int64).view(1, 1, num_blocks, 1, 1)
    pooled = (blocks * valid.to(x.dtype)).sum(dim=3)
    return pooled / sizes.view(1, 1, num_blocks, 1)


def block_attention_map(q_pooled: torch.Tensor, k_pooled: torch.Tensor) -> torch.Tensor:
    """Softmax attention over pooled block representatives: ``[B, H, Nq, Nkv]``."""
    scale = q_pooled.shape[-1] ** -0.5
    logits = torch.matmul(q_pooled.float(), k_pooled.float().transpose(-2, -1)) * scale
    return torch.softmax(logits, dim=-1)


def build_block_routing(
    block_attn: torch.Tensor,
    tau_min: float,
    tau_max: float,
    *,
    return_mask: bool = False,
) -> BlockRouting:
    """Select, per (query block, head), the minimal block set covering mass ``tau_h``.

    With ``p`` the block-attention weights of query block ``i`` in head ``h``, sorted descending by
    a permutation ``sigma``, the head's Gini coefficient is

        G = E_i[ (1 / N) * sum_m (N - 2m + 1) * p[i, sigma(m)] ]

    which is 0 for a uniform distribution and ``(N - 1) / N`` for a one-hot one. It maps to the
    coverage threshold

        tau_h = tau_min + G * (tau_max - tau_min)

    and the selected set is ``{sigma(1), ..., sigma(K)}`` with

        K = min { k : sum_{m <= k} p[i, sigma(m)] >= tau_h }.
    """
    _, heads, _, kv_blocks = block_attn.shape
    block_attn = block_attn.float()

    sorted_probs, sorted_index = torch.sort(block_attn, dim=-1, descending=True)

    ranks = torch.arange(1, kv_blocks + 1, device=block_attn.device, dtype=sorted_probs.dtype)
    gini = torch.sum((kv_blocks + 1 - 2 * ranks) * sorted_probs, dim=-1) / kv_blocks
    head_gini = gini.mean(dim=(0, 2))

    thresholds = tau_min + head_gini * (tau_max - tau_min)

    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep block m while the mass strictly before it is still short of the threshold; this is
    # exactly the minimal k whose inclusive cumulative sum reaches tau_h.
    keep_sorted = (cumulative - sorted_probs) < thresholds.view(1, heads, 1, 1)

    index = sorted_index.to(torch.int32).contiguous()
    num = torch.sum(keep_sorted, dim=-1, dtype=torch.int32).contiguous()

    mask = None
    if return_mask:
        mask = torch.zeros_like(block_attn, dtype=torch.bool)
        mask.scatter_(-1, sorted_index, keep_sorted)

    return BlockRouting(index=index, num=num, mask=mask, thresholds=thresholds)


def variable_block_sizes(kv_len: int, kv_len_padded: int, block_size: int, device) -> torch.Tensor:
    """Real token count of each key block; the trailing block may be partial."""
    kv_blocks = kv_len_padded // block_size
    sizes = torch.full((kv_blocks,), block_size, dtype=torch.int32, device=device)
    sizes[-1] = kv_len - (kv_blocks - 1) * block_size
    return sizes


def pad_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Zero-pad ``[B, L, H, D]`` along the token axis up to ``target_len``."""
    pad_len = target_len - x.shape[1]
    if pad_len <= 0:
        return x
    pad = torch.zeros((x.shape[0], pad_len, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def block_counts(q_len: int, kv_len: int, block_size: int) -> tuple[int, int, int, int]:
    """Padded lengths and block counts for a (q_len, kv_len) attention problem."""
    q_len_padded = math.ceil(q_len / block_size) * block_size
    kv_len_padded = math.ceil(kv_len / block_size) * block_size
    return q_len_padded, kv_len_padded, q_len_padded // block_size, kv_len_padded // block_size
