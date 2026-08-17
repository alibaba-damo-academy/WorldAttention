"""Hybrid Sparse Attention (HSA)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import distill
from .coarse_cache import GrowingCoarseCache
from .pooled_cache import PooledKeyCache
from .block_sparse import block_sparse_attention
from .routing import (
    block_attention_logits,
    block_attention_map,
    block_counts,
    block_mean_pool,
    build_block_routing,
    build_block_routing_from_logits,
    pad_to_len,
    variable_block_sizes,
)

__all__ = [
    "HSAAttention",
    "hsa_attention",
    "compress_kv",
    "compress_kv_batched",
    "hsa_parameter_names",
    "require_trained_hsa",
]

_HSA_ATTR = "hsa_attention"


def compress_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    segment_len: int,
    k_proj_mat: torch.Tensor,
    v_proj_mat: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``[B, H, L, D]`` keys and values onto the low-rank sequence basis."""
    if k_proj_mat.shape != v_proj_mat.shape:
        raise ValueError(
            "k_proj_mat and v_proj_mat must have the same shape, got "
            f"{tuple(k_proj_mat.shape)} and {tuple(v_proj_mat.shape)}"
        )

    kv_len = k.shape[2]
    if segment_len <= 0:
        segment_len = kv_len
    max_segment_len = k_proj_mat.shape[1]

    compressed_k, compressed_v = [], []
    for start in range(0, kv_len, segment_len):
        end = min(start + segment_len, kv_len)
        seg_len = end - start
        if seg_len > max_segment_len:
            raise ValueError(
                f"segment length {seg_len} exceeds projection length {max_segment_len}; "
                "increase proj_segment_len"
            )
        rank = min(k_proj_mat.shape[0], math.ceil(seg_len / block_size))

        seg_k = k[:, :, start:end]
        seg_v = v[:, :, start:end]
        proj_k = k_proj_mat[:rank, :seg_len]
        proj_v = v_proj_mat[:rank, :seg_len]
        if not torch.is_grad_enabled():
            proj_k, proj_v = proj_k.detach(), proj_v.detach()
        proj_k = proj_k.to(device=seg_k.device, dtype=torch.float32)
        proj_v = proj_v.to(device=seg_v.device, dtype=torch.float32)

        pooled_k = torch.matmul(seg_k.float().transpose(-1, -2), proj_k.transpose(0, 1))
        pooled_v = torch.matmul(seg_v.float().transpose(-1, -2), proj_v.transpose(0, 1))
        compressed_k.append(pooled_k.transpose(-1, -2).to(seg_k.dtype))
        compressed_v.append(pooled_v.transpose(-1, -2).to(seg_v.dtype))

    return torch.cat(compressed_k, dim=2), torch.cat(compressed_v, dim=2)


def compress_kv_batched(k, v, segment_len, k_proj_mat, v_proj_mat, block_size, detach=True):
    """Low-rank projection of ``[B, L, H, D]`` keys and values as one batched matmul."""
    if detach:
        k_proj_mat, v_proj_mat = k_proj_mat.detach(), v_proj_mat.detach()
    bsz, length, heads, head_dim = k.shape
    full_segments = length // segment_len
    tail = length - full_segments * segment_len
    outs = []
    for x, proj in ((k, k_proj_mat), (v, v_proj_mat)):
        parts = []
        if full_segments:
            xf = x[:, : full_segments * segment_len].reshape(bsz, full_segments, segment_len, heads * head_dim)
            parts.append(torch.matmul(proj.to(x.dtype), xf).reshape(bsz, -1, heads, head_dim))
        if tail:
            rank_t = min(proj.shape[0], math.ceil(tail / block_size))
            xt = x[:, full_segments * segment_len :].reshape(bsz, 1, tail, heads * head_dim)
            parts.append(torch.matmul(proj[:rank_t, :tail].to(x.dtype), xt).reshape(bsz, rank_t, heads, head_dim))
        outs.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=1))
    return outs[0], outs[1]


def _linear_branch(q: torch.Tensor, k_coarse: torch.Tensor, v_coarse: torch.Tensor) -> torch.Tensor:
    """Attention of full-resolution queries against the compressed keys and values."""
    scale = q.shape[-1] ** -0.5
    logits = torch.matmul(q.float(), k_coarse.float().transpose(-2, -1)) * scale
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights.to(v_coarse.dtype), v_coarse)


def hsa_parameter_names(model: nn.Module) -> list[str]:
    """State-dict keys of every HSA parameter in ``model``, found by module type."""
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, HSAAttention):
            prefix = f"{module_name}." if module_name else ""
            names.extend(prefix + name for name, _ in module.named_parameters())
    return names


def require_trained_hsa(missing_keys, checkpoint_path: str = "") -> None:
    """Raise if a checkpoint load left any HSA parameter unfilled."""
    missing_hsa = [key for key in missing_keys if _HSA_ATTR in key]
    if not missing_hsa:
        return
    where = f" in {checkpoint_path}" if checkpoint_path else ""
    raise RuntimeError(
        f"{len(missing_hsa)} HSA parameters are absent{where}, so this checkpoint has not been "
        f"through HSA training. Load a checkpoint from the HSA stage, or set attn_backend=flash to "
        f"run the dense baseline. First missing keys: {missing_hsa[:6]}"
    )


def hsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    k_coarse: torch.Tensor | None = None,
    v_coarse: torch.Tensor | None = None,
    k_proj_mat: torch.Tensor,
    v_proj_mat: torch.Tensor,
    gate_lin: nn.Module | None = None,
    block_size: int = 64,
    tau_min: float = 0.35,
    tau_max: float = 1.0,
    proj_segment_len: int = 1560,
    backend: str = "auto",
    sparse_only: bool = False,
    force_density: float | None = None,
    pooled_cache: "PooledKeyCache | None" = None,
    stable_kv_tokens: int = 0,
    kv_granularity: str = "mixed",
    current_chunk_tokens: int = 0,
    routing_cache: dict | None = None,
    budget_floor: float | None = None,
    budget_cap: float | None = None,
) -> torch.Tensor:
    """Run HSA over ``[B, L, H, D]`` queries, keys and values."""
    if (k_coarse is None) != (v_coarse is None):
        raise ValueError("k_coarse and v_coarse must be provided together")

    bsz, q_len, heads, head_dim = q.shape
    kv_len = k.shape[1]
    if q_len <= 0 or kv_len <= 0:
        raise ValueError("q_len and kv_len must be positive")

    out_dtype = q.dtype
    q_t = q.transpose(1, 2).contiguous()

    if sparse_only:
        k_coarse_t = v_coarse_t = None
    elif k_coarse is None:
        k_coarse_t, v_coarse_t = compress_kv(
            k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
            segment_len=proj_segment_len,
            k_proj_mat=k_proj_mat, v_proj_mat=v_proj_mat, block_size=block_size,
        )
    else:
        if k_coarse.shape[0] != bsz or k_coarse.shape[2:] != (heads, head_dim):
            raise ValueError(
                f"expected k_coarse shaped [B, Lc, {heads}, {head_dim}], got {tuple(k_coarse.shape)}"
            )
        if v_coarse.shape != k_coarse.shape:
            raise ValueError("v_coarse must have the same shape as k_coarse")
        k_coarse_t = k_coarse.transpose(1, 2).contiguous()
        v_coarse_t = v_coarse.transpose(1, 2).contiguous()

    if sparse_only:
        output_linear = None
    else:
        output_linear = _linear_branch(q_t, k_coarse_t, v_coarse_t)

    q_len_pad, kv_len_pad, q_blocks, kv_blocks = block_counts(q_len, kv_len, block_size)
    q_t_pad = pad_to_len(q, q_len_pad).transpose(1, 2).contiguous()
    k_t_pad = pad_to_len(k, kv_len_pad).transpose(1, 2).contiguous()
    v_t_pad = pad_to_len(v, kv_len_pad).transpose(1, 2).contiguous()

    differentiable = torch.is_grad_enabled()

    routing_key = (q_len, kv_len, block_size, kv_granularity, current_chunk_tokens,
                   force_density, tau_min, tau_max)
    reuse = (routing_cache is not None and not differentiable
             and routing_cache.get("key") == routing_key)
    if reuse:
        routing, block_sizes = routing_cache["routing"], routing_cache["block_sizes"]
    else:
      stable_rows = 0 if pooled_cache is None else pooled_cache.stable_blocks(
          stable_kv_tokens, block_size)
      stable_rows = min(stable_rows, kv_blocks)
      cached_rows = None if stable_rows <= 0 else pooled_cache.get(stable_rows, block_size)

      if cached_rows is None:
          k_pooled = block_mean_pool(k_t_pad, kv_len, kv_blocks, block_size)
          if stable_rows > 0:
              pooled_cache.put(stable_rows, block_size, k_pooled[:, :, :stable_rows].clone())
      else:
          tail_start = stable_rows * block_size
          tail_pooled = block_mean_pool(
              k_t_pad[:, :, tail_start:], kv_len - tail_start, kv_blocks - stable_rows, block_size
          )
          k_pooled = torch.cat([cached_rows, tail_pooled], dim=2)

      q_pooled = block_mean_pool(q_t_pad, q_len, q_blocks, block_size)
      inter_tokens = kv_len - current_chunk_tokens if current_chunk_tokens > 0 else 0
      paired = kv_granularity != "all64" and not (kv_granularity == "mixed" and inter_tokens <= 0)
      if paired:
          routing = build_block_routing_from_logits(
              block_attention_logits(q_pooled, k_pooled),
              tau_min, tau_max, force_density=force_density,
              kv_granularity=kv_granularity, inter_tokens=inter_tokens, block_size=block_size,
              budget_floor=budget_floor, budget_cap=budget_cap,
              return_mask=False,
          )
      else:
          routing = build_block_routing(
              block_attention_map(q_pooled, k_pooled),
              tau_min=tau_min,
              tau_max=tau_max,
              return_mask=False,
              force_density=force_density,
          )
          if (budget_floor is not None or budget_cap is not None) and routing.mask is None:
              lo = 1 if budget_floor is None else max(1, math.ceil(budget_floor * kv_blocks))
              hi = kv_blocks if budget_cap is None else max(1, math.ceil(budget_cap * kv_blocks))
              routing.num.clamp_(min=lo, max=hi)
      block_sizes = variable_block_sizes(kv_len, kv_len_pad, block_size, q.device)
      if routing_cache is not None and not differentiable:
          routing_cache["key"] = routing_key
          routing_cache["routing"] = routing
          routing_cache["block_sizes"] = block_sizes

    q_sparse = q_t_pad.to(torch.bfloat16)
    k_sparse = k_t_pad.to(torch.bfloat16)
    v_sparse = v_t_pad.to(torch.bfloat16)

    output_sparse = block_sparse_attention(
        q_sparse, k_sparse, v_sparse, routing.index, routing.num, block_sizes, block_size,
    )

    sparse_blhd = output_sparse.transpose(1, 2)[:, :q_len]

    if sparse_only:
        output = sparse_blhd.to(out_dtype)
    else:
      if gate_lin is not None:
          gate_dtype = gate_lin.weight.dtype
          gate_input = q if q.dtype == gate_dtype else q.to(gate_dtype)
          gate = torch.sigmoid(gate_lin(gate_input)).to(output_linear.dtype)
      else:
          gate = torch.sigmoid(q.to(output_linear.dtype))

      output = torch.addcmul(sparse_blhd.to(output_linear.dtype),
                             output_linear.transpose(1, 2), gate)

    if distill.distill_enabled() and torch.is_grad_enabled():
        distill.record_against_dense(output, q, k, v)

    return output.to(out_dtype)


class HSAAttention(nn.Module):
    """Hybrid Sparse Attention over a KV window."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        block_size: int = 64,
        tau_min: float = 0.35,
        tau_max: float = 1.0,
        proj_segment_len: int = 1560,
        backend: str = "auto",
        kv_granularity: str = "mixed",
    ):
        super().__init__()
        if block_size != 64:
            raise ValueError("the sparse-branch kernels are specialized for block_size=64")
        if proj_segment_len <= 0:
            raise ValueError(f"proj_segment_len must be positive, got {proj_segment_len}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.proj_segment_len = proj_segment_len
        self.proj_rank = max(1, proj_segment_len // block_size)
        self.backend = backend
        self.sparse_only = False
        self.force_density = None
        self.pooled_cache = PooledKeyCache()
        self.stable_kv_tokens = 0
        self.kv_granularity = kv_granularity
        self.current_chunk_tokens = 0
        self.coarse_history = GrowingCoarseCache()
        self.routing_reuse = False
        self._routing_cache: dict = {}
        self.budget_floor: float | None = None
        self.budget_cap: float | None = None

        self.k_proj_mat = nn.Parameter(
            torch.zeros((self.proj_rank, proj_segment_len), dtype=torch.bfloat16)
        )
        self.v_proj_mat = nn.Parameter(
            torch.zeros((self.proj_rank, proj_segment_len), dtype=torch.bfloat16)
        )
        self.gate_lin = nn.Linear(head_dim, head_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Return every HSA parameter to its zero placeholder."""
        with torch.no_grad():
            self.k_proj_mat.zero_()
            self.v_proj_mat.zero_()
            self.gate_lin.weight.zero_()
            self.gate_lin.bias.zero_()

    def coarse_cache_size(self, kv_cache_size: int) -> int:
        """Number of compressed slots needed to mirror a token cache of ``kv_cache_size``."""
        if kv_cache_size <= 0:
            return 0
        return math.ceil(kv_cache_size / self.proj_segment_len) * self.proj_rank

    def token_range_to_coarse(self, start_index: int, end_index: int) -> tuple[int, int]:
        """Map a token range onto compressed-cache slots. Both bounds must be segment-aligned."""
        if start_index % self.proj_segment_len != 0 or end_index % self.proj_segment_len != 0:
            raise ValueError(
                "coarse-cache access requires segment-aligned token indices, got "
                f"start={start_index}, end={end_index}, proj_segment_len={self.proj_segment_len}"
            )
        return (
            (start_index // self.proj_segment_len) * self.proj_rank,
            (end_index // self.proj_segment_len) * self.proj_rank,
        )

    def compress_kv_cache(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress ``[B, L, H, D]`` keys and values for the compressed cache."""
        if k.shape != v.shape:
            raise ValueError(f"k and v must match, got {tuple(k.shape)} and {tuple(v.shape)}")
        if k.ndim != 4:
            raise ValueError(f"expected [B, L, H, D] tensors, got ndim={k.ndim}")
        k_coarse, v_coarse = compress_kv_batched(
            k, v, self.proj_segment_len, self.k_proj_mat, self.v_proj_mat, self.block_size,
            detach=not torch.is_grad_enabled(),
        )
        return k_coarse.contiguous(), v_coarse.contiguous()

    def finalize_chunk(self, k_chunk: torch.Tensor, v_chunk: torch.Tensor) -> None:
        """Compress a finalized chunk's ``[B, L, H, D]`` K/V into the global coarse history."""
        kc, vc = compress_kv_batched(k_chunk, v_chunk, self.proj_segment_len,
                                     self.k_proj_mat, self.v_proj_mat, self.block_size)
        self.coarse_history.append(kc, vc)

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_coarse: torch.Tensor | None = None,
        v_coarse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (k_coarse is None and self.coarse_history.enabled and self.coarse_history.rows > 0
                and 0 < self.current_chunk_tokens <= k.shape[1]):
            kc_hist, vc_hist = self.coarse_history.view()
            kc_cur, vc_cur = compress_kv_batched(
                k[:, -self.current_chunk_tokens :], v[:, -self.current_chunk_tokens :],
                self.proj_segment_len, self.k_proj_mat, self.v_proj_mat, self.block_size)
            k_coarse = torch.cat([kc_hist, kc_cur], dim=1)
            v_coarse = torch.cat([vc_hist, vc_cur], dim=1)
        return hsa_attention(
            q, k, v,
            k_coarse=k_coarse, v_coarse=v_coarse,
            k_proj_mat=self.k_proj_mat, v_proj_mat=self.v_proj_mat,
            gate_lin=self.gate_lin,
            block_size=self.block_size,
            tau_min=self.tau_min, tau_max=self.tau_max,
            proj_segment_len=self.proj_segment_len,
            backend=self.backend,
            sparse_only=self.sparse_only,
            force_density=self.force_density,
            pooled_cache=self.pooled_cache,
            stable_kv_tokens=self.stable_kv_tokens,
            kv_granularity=self.kv_granularity,
            current_chunk_tokens=self.current_chunk_tokens,
            routing_cache=self._routing_cache if self.routing_reuse else None,
            budget_floor=self.budget_floor,
            budget_cap=self.budget_cap,
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._forward(*args, **kwargs)

    def begin_decode_block(self) -> None:
        """Tell the module the KV window is about to change, so pooled keys must be recomputed."""
        self.pooled_cache.invalidate()
        self._routing_cache.clear()
