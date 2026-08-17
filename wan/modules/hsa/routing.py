"""Head-adaptive block routing for the HSA sparse branch."""
from __future__ import annotations

import math

import torch

__all__ = [
    "BlockRouting",
    "build_block_routing",
    "build_block_routing_from_logits",
    "block_attention_logits",
    "pair_kv_candidates",
    "expand_pair_selection",
    "block_attention_map",
    "block_mean_pool",
    "block_counts",
    "pad_to_len",
    "variable_block_sizes",
]


class BlockRouting:
    """Result of routing: which key blocks each query block attends to."""

    __slots__ = ("index", "num", "mask", "thresholds")

    def __init__(self, index, num, mask, thresholds):
        self.index = index
        self.num = num
        self.mask = mask
        self.thresholds = thresholds


def block_mean_pool(x: torch.Tensor, logical_len: int, num_blocks: int, block_size: int) -> torch.Tensor:
    """Mean-pool ``[B, H, num_blocks * block_size, D]`` into ``[B, H, num_blocks, D]``."""
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
    return torch.softmax(block_attention_logits(q_pooled, k_pooled), dim=-1)


def block_attention_logits(q_pooled: torch.Tensor, k_pooled: torch.Tensor) -> torch.Tensor:
    """Pre-softmax pooled block logits in fp32: ``[B, H, Nq, Nkv]``."""
    scale = q_pooled.shape[-1] ** -0.5
    return torch.matmul(q_pooled.float(), k_pooled.float().transpose(-2, -1)) * scale


def pair_kv_candidates(logits: torch.Tensor, block_size: int, kv_granularity: str,
                       inter_tokens: int) -> tuple[torch.Tensor, int]:
    """Merge adjacent 64-block logits into 128-token pair candidates (paper method §6)."""
    kv_blocks = logits.shape[-1]
    if kv_granularity == "all64":
        return logits, 0
    if kv_granularity == "all128":
        n_inter64 = kv_blocks
    elif kv_granularity == "mixed":
        n_inter64 = max(0, min(int(inter_tokens) // block_size, kv_blocks))
    else:
        raise ValueError(f"unknown kv_granularity {kv_granularity!r}; "
                         "expected 'mixed', 'all64' or 'all128'")
    n_pairs = n_inter64 // 2
    if n_pairs == 0:
        return logits, 0
    head = logits[..., : 2 * n_pairs]
    pair_logits = torch.logsumexp(head.view(*head.shape[:-1], n_pairs, 2), dim=-1)
    return torch.cat([pair_logits, logits[..., 2 * n_pairs :]], dim=-1), n_pairs


def expand_pair_selection(mask_c: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Candidate-selection mask back to a 64-block mask (each pair opens both blocks)."""
    if n_pairs == 0:
        return mask_c
    return torch.cat([mask_c[..., :n_pairs].repeat_interleave(2, dim=-1),
                      mask_c[..., n_pairs:]], dim=-1)


def build_block_routing_from_logits(
    logits: torch.Tensor,
    tau_min: float,
    tau_max: float,
    *,
    force_density: float | None = None,
    kv_granularity: str = "all64",
    inter_tokens: int = 0,
    block_size: int = 64,
    budget_floor: float | None = None,
    budget_cap: float | None = None,
    return_mask: bool = False,
) -> BlockRouting:
    """Routing over (possibly pair-merged) KV candidates, expanded back to 64-block ids."""
    _, heads, _, kv_blocks = logits.shape
    cand, n_pairs = pair_kv_candidates(logits, block_size, kv_granularity, inter_tokens)
    probs = torch.softmax(cand.float(), dim=-1)
    sorted_probs, sorted_index = torch.sort(probs, dim=-1, descending=True)
    n_cand = probs.shape[-1]

    if force_density is not None:
        if not 0.0 < force_density <= 1.0:
            raise ValueError(f"force_density must be in (0, 1], got {force_density}")
        keep = max(1, math.ceil(force_density * n_cand))
        num_c = torch.full(probs.shape[:-1], keep, dtype=torch.int32, device=logits.device)
        thresholds = torch.full((heads,), float(force_density), device=logits.device)
    else:
        ranks = torch.arange(1, n_cand + 1, device=logits.device, dtype=sorted_probs.dtype)
        gini = torch.sum((n_cand + 1 - 2 * ranks) * sorted_probs, dim=-1) / n_cand
        thresholds = tau_min + gini.mean(dim=(0, 2)) * (tau_max - tau_min)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_sorted = (cumulative - sorted_probs) < thresholds.view(1, heads, 1, 1)
        num_c = torch.sum(keep_sorted, dim=-1, dtype=torch.int32)

    if budget_floor is not None or budget_cap is not None:
        lo = 1 if budget_floor is None else max(1, math.ceil(budget_floor * n_cand))
        hi = n_cand if budget_cap is None else max(1, math.ceil(budget_cap * n_cand))
        num_c = num_c.clamp(min=lo, max=hi)

    ranks_c = torch.arange(n_cand, device=logits.device).view(1, 1, 1, -1)
    mask_c = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        -1, sorted_index, ranks_c < num_c.long().unsqueeze(-1))
    mask64 = expand_pair_selection(mask_c, n_pairs)
    num = mask64.sum(-1, dtype=torch.int32).contiguous()
    index = torch.argsort(mask64.to(torch.int8), dim=-1, descending=True, stable=True)
    mask = mask64[..., :kv_blocks].contiguous() if return_mask else None
    return BlockRouting(index=index.to(torch.int32).contiguous(), num=num, mask=mask,
                        thresholds=thresholds)


def build_block_routing(
    block_attn: torch.Tensor,
    tau_min: float,
    tau_max: float,
    *,
    return_mask: bool = False,
    force_density: float | None = None,
) -> BlockRouting:
    """Select, per (query block, head), the minimal block set covering mass ``tau_h``."""
    _, heads, _, kv_blocks = block_attn.shape
    block_attn = block_attn.float()

    sorted_probs, sorted_index = torch.sort(block_attn, dim=-1, descending=True)

    if force_density is not None:
        if not 0.0 < force_density <= 1.0:
            raise ValueError(f"force_density must be in (0, 1], got {force_density}")
        keep = max(1, math.ceil(force_density * kv_blocks))
        ranks = torch.arange(kv_blocks, device=block_attn.device)
        keep_sorted = (ranks < keep).view(1, 1, 1, kv_blocks).expand_as(sorted_probs)
        index = sorted_index.to(torch.int32).contiguous()
        num = torch.full(
            block_attn.shape[:-1], keep, dtype=torch.int32, device=block_attn.device
        ).contiguous()
        mask = None
        if return_mask:
            mask = torch.zeros_like(block_attn, dtype=torch.bool)
            mask.scatter_(-1, sorted_index, keep_sorted)
        thresholds = torch.full((heads,), float(force_density), device=block_attn.device)
        return BlockRouting(index=index, num=num, mask=mask, thresholds=thresholds)

    ranks = torch.arange(1, kv_blocks + 1, device=block_attn.device, dtype=sorted_probs.dtype)
    gini = torch.sum((kv_blocks + 1 - 2 * ranks) * sorted_probs, dim=-1) / kv_blocks
    head_gini = gini.mean(dim=(0, 2))

    thresholds = tau_min + head_gini * (tau_max - tau_min)

    cumulative = torch.cumsum(sorted_probs, dim=-1)
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
