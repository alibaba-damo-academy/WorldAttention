"""Pure-PyTorch block-sparse attention: the reference semantics of the HSA sparse branch.

Same masking rules as the Triton kernel (each query block attends only to its selected key blocks,
with a trailing partial block masked to its real token count), written with plain tensor ops so
autograd supplies the backward and so the routing logic can be validated without a GPU.
"""
from __future__ import annotations

import math

import torch

__all__ = ["block_sparse_attention_reference"]


def block_sparse_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    block_sizes: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Attention over selected blocks for ``[B, H, N, D]`` inputs padded to ``block_size``.

    Gathers the selected key/value blocks per query block, so peak memory scales with the widest
    selection actually made rather than with the total number of key blocks.
    """
    bsz, heads, q_tokens, head_dim = q.shape
    num_q_blocks = q_tokens // block_size
    num_kv_blocks = k.shape[2] // block_size

    # Only the first q2k_num entries of each row are selected; narrowing the gather width to the
    # widest row keeps the materialized selection proportional to the real sparsity.
    width = max(int(q2k_num.max().item()), 1)
    index = q2k_index[..., :width].long().clamp_(0, max(num_kv_blocks - 1, 0))

    compute_dtype = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32
    q_blocks = q.to(compute_dtype).view(bsz, heads, num_q_blocks, block_size, head_dim)
    k_flat = k.to(compute_dtype).reshape(bsz, heads, num_kv_blocks, block_size * head_dim)
    v_flat = v.to(compute_dtype).reshape(bsz, heads, num_kv_blocks, block_size * head_dim)

    gather_index = index.unsqueeze(-1).expand(bsz, heads, num_q_blocks, width, block_size * head_dim)
    k_sel = torch.gather(
        k_flat.unsqueeze(2).expand(bsz, heads, num_q_blocks, num_kv_blocks, block_size * head_dim),
        3, gather_index,
    ).view(bsz, heads, num_q_blocks, width * block_size, head_dim)
    v_sel = torch.gather(
        v_flat.unsqueeze(2).expand(bsz, heads, num_q_blocks, num_kv_blocks, block_size * head_dim),
        3, gather_index,
    ).view(bsz, heads, num_q_blocks, width * block_size, head_dim)

    scores = torch.matmul(q_blocks, k_sel.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))

    slots = torch.arange(width, device=q.device).view(1, 1, 1, width)
    slot_valid = slots < q2k_num.long().unsqueeze(-1)
    tokens = torch.arange(block_size, device=q.device).view(1, 1, 1, 1, block_size)
    token_valid = tokens < block_sizes.long()[index].unsqueeze(-1)
    valid = (slot_valid.unsqueeze(-1) & token_valid).view(bsz, heads, num_q_blocks, 1, width * block_size)

    scores = scores.masked_fill(~valid, float("-inf"))
    attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.matmul(attn, v_sel).view(bsz, heads, q_tokens, head_dim).to(q.dtype)
