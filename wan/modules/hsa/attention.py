"""Hybrid Sparse Attention (HSA).

HSA replaces full attention over a KV window with two branches and a learned fusion:

* a **linear branch** that projects keys and values onto a low-rank basis along the sequence axis
  (Linformer-style) and attends against the compressed representation, giving a dense but low-rank
  view of the whole window;
* a **sparse branch** that attends at full token resolution but only inside blocks selected by
  head-adaptive cumulative-mass routing (see :mod:`worldattention.hsa.routing`);
* a **gated fusion** that mixes the two per token and per head.

The projection is applied per segment of ``proj_segment_len`` tokens and the segment outputs are
concatenated, so a growing KV cache is compressed incrementally and the compressed cache can be
maintained alongside the token cache. One projection pair is shared across heads.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import distill
from .block_sparse_torch import block_sparse_attention_reference
from .routing import (
    block_attention_map,
    block_counts,
    block_mean_pool,
    build_block_routing,
    pad_to_len,
    variable_block_sizes,
)

__all__ = [
    "HSAAttention",
    "hsa_attention",
    "compress_kv",
    "resolve_backend",
    "hsa_parameter_names",
    "require_trained_hsa",
]

_BACKENDS = ("auto", "triton", "torch")
_HSA_ATTR = "hsa_attention"


def resolve_backend(backend: str) -> str:
    """Resolve ``"auto"`` to a sparse-branch backend available on the current device.

    Triton 3.1 does not implement the shared-memory encoding this block-sparse kernel needs on SM90,
    so ``"auto"`` selects the reference implementation there and the Triton kernels elsewhere.
    """
    backend = str(backend or "auto").lower()
    if backend not in _BACKENDS:
        raise ValueError(f"unknown HSA backend {backend!r}; expected one of {_BACKENDS}")
    if backend != "auto":
        return backend
    if not torch.cuda.is_available():
        return "torch"
    major, minor = torch.cuda.get_device_capability(0)
    return "torch" if (major, minor) == (9, 0) else "triton"


def compress_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    segment_len: int,
    k_proj_mat: torch.Tensor,
    v_proj_mat: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``[B, H, L, D]`` keys and values onto the low-rank sequence basis.

    Each segment of at most ``segment_len`` tokens is reduced to ``rank`` rows by
    ``Ktilde = E_K^T K``, and the per-segment results are concatenated along the token axis.
    """
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
        proj_k = k_proj_mat[:rank, :seg_len].to(device=seg_k.device, dtype=torch.float32)
        proj_v = v_proj_mat[:rank, :seg_len].to(device=seg_v.device, dtype=torch.float32)

        pooled_k = torch.matmul(seg_k.float().transpose(-1, -2), proj_k.transpose(0, 1))
        pooled_v = torch.matmul(seg_v.float().transpose(-1, -2), proj_v.transpose(0, 1))
        compressed_k.append(pooled_k.transpose(-1, -2).to(seg_k.dtype))
        compressed_v.append(pooled_v.transpose(-1, -2).to(seg_v.dtype))

    return torch.cat(compressed_k, dim=2), torch.cat(compressed_v, dim=2)


def _linear_branch(q: torch.Tensor, k_coarse: torch.Tensor, v_coarse: torch.Tensor) -> torch.Tensor:
    """Attention of full-resolution queries against the compressed keys and values."""
    scale = q.shape[-1] ** -0.5
    logits = torch.matmul(q.float(), k_coarse.float().transpose(-2, -1)) * scale
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights.to(v_coarse.dtype), v_coarse)


def hsa_parameter_names(model: nn.Module) -> list[str]:
    """State-dict keys of every HSA parameter in ``model``, found by module type.

    Works both on a whole model and on a single :class:`HSAAttention`, unlike matching on the
    attribute name.
    """
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, HSAAttention):
            prefix = f"{module_name}." if module_name else ""
            names.extend(prefix + name for name, _ in module.named_parameters())
    return names


def require_trained_hsa(missing_keys, checkpoint_path: str = "") -> None:
    """Raise if a checkpoint load left any HSA parameter unfilled.

    HSA parameters are trained; there is no meaningful default for them. Running with unfilled
    projections and an unfilled fusion gate produces output that looks plausible but is not what the
    architecture computes, so a missing HSA parameter is an error rather than a warning.
    """
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
) -> torch.Tensor:
    """Run HSA over ``[B, L, H, D]`` queries, keys and values.

    ``k_coarse`` / ``v_coarse`` supply an already-compressed view of the same KV range, which lets a
    decoding loop maintain the compressed cache incrementally instead of recompressing the window at
    every step. When omitted, the compression is computed here.
    """
    if (k_coarse is None) != (v_coarse is None):
        raise ValueError("k_coarse and v_coarse must be provided together")

    bsz, q_len, heads, head_dim = q.shape
    kv_len = k.shape[1]
    if q_len <= 0 or kv_len <= 0:
        raise ValueError("q_len and kv_len must be positive")

    out_dtype = q.dtype
    q_t = q.transpose(1, 2).contiguous()

    if k_coarse is None:
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

    output_linear = _linear_branch(q_t, k_coarse_t, v_coarse_t)

    q_len_pad, kv_len_pad, q_blocks, kv_blocks = block_counts(q_len, kv_len, block_size)
    q_t_pad = pad_to_len(q, q_len_pad).transpose(1, 2).contiguous()
    k_t_pad = pad_to_len(k, kv_len_pad).transpose(1, 2).contiguous()
    v_t_pad = pad_to_len(v, kv_len_pad).transpose(1, 2).contiguous()

    resolved = resolve_backend(backend)
    differentiable = torch.is_grad_enabled()

    routing = build_block_routing(
        block_attention_map(
            block_mean_pool(q_t_pad, q_len, q_blocks, block_size),
            block_mean_pool(k_t_pad, kv_len, kv_blocks, block_size),
        ),
        tau_min=tau_min,
        tau_max=tau_max,
        # The differentiable Triton path consumes a selection mask; every other path reads the
        # descending-order index directly.
        return_mask=differentiable and resolved == "triton",
    )
    block_sizes = variable_block_sizes(kv_len, kv_len_pad, block_size, q.device)

    q_sparse = q_t_pad.to(torch.bfloat16)
    k_sparse = k_t_pad.to(torch.bfloat16)
    v_sparse = v_t_pad.to(torch.bfloat16)

    if resolved == "triton":
        from .block_sparse_triton import block_sparse_attention

        output_sparse = block_sparse_attention(
            q_sparse, k_sparse, v_sparse,
            block_sizes=block_sizes,
            q2k_index=routing.index, q2k_num=routing.num,
            # Routing is a discrete selection, so the mask enters the kernel detached and no
            # gradient flows through the block choice itself.
            block_map=routing.mask.detach() if routing.mask is not None else None,
        )
    else:
        output_sparse = block_sparse_attention_reference(
            q_sparse, k_sparse, v_sparse,
            routing.index, routing.num, block_sizes, block_size,
        )

    if gate_lin is not None:
        gate_dtype = gate_lin.weight.dtype
        gate_input = q if q.dtype == gate_dtype else q.to(gate_dtype)
        gate = torch.sigmoid(gate_lin(gate_input)).to(output_linear.dtype)
    else:
        gate = torch.sigmoid(q.to(output_linear.dtype))

    output_sparse = output_sparse.transpose(1, 2)[:, :q_len].to(output_linear.dtype)
    output = torch.addcmul(output_sparse, output_linear.transpose(1, 2), gate)

    if distill.distill_enabled() and torch.is_grad_enabled():
        distill.record_against_dense(output, q, k, v)

    return output.to(out_dtype)


class HSAAttention(nn.Module):
    """Hybrid Sparse Attention over a KV window.

    Owns the branch-specific parameters: the low-rank sequence projections ``k_proj_mat`` and
    ``v_proj_mat`` for the linear branch, and the fusion gate ``gate_lin``. They are allocated as
    zeros and carry no default initialization — they are trained parameters, so they must come from
    a checkpoint. :func:`require_trained_hsa` turns a checkpoint that lacks them into an error, and
    ``worldattention.trainer.hsa_stage.init_hsa_parameters`` provides the initialization used when a
    training run introduces HSA on top of a dense base model.

    Args:
        num_heads: attention heads.
        head_dim: per-head channel count.
        block_size: sparse-branch block size; the Triton kernels are specialized for 64.
        tau_min, tau_max: bounds of the head-adaptive coverage threshold.
        proj_segment_len: tokens per projection segment. For a video DiT this is the token count of
            one latent frame, so the compressed cache grows one segment at a time.
        backend: ``"auto"``, ``"triton"`` or ``"torch"``.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        block_size: int = 64,
        tau_min: float = 0.35,
        tau_max: float = 1.0,
        proj_segment_len: int = 1560,
        backend: str = "auto",
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
        self.proj_rank = math.ceil(proj_segment_len / block_size)
        self.backend = backend

        # Zero placeholders: a trained checkpoint fills these in.
        self.k_proj_mat = nn.Parameter(
            torch.zeros((self.proj_rank, proj_segment_len), dtype=torch.bfloat16)
        )
        self.v_proj_mat = nn.Parameter(
            torch.zeros((self.proj_rank, proj_segment_len), dtype=torch.bfloat16)
        )
        self.gate_lin = nn.Linear(head_dim, head_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Return every HSA parameter to its zero placeholder.

        There is no default initialization to fall back on: these are trained parameters. The host
        model calls this after its own generic weight initialization, which would otherwise leave the
        fusion gate holding whatever the generic scheme produced.
        """
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
        k_coarse, v_coarse = compress_kv(
            k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
            segment_len=self.proj_segment_len,
            k_proj_mat=self.k_proj_mat, v_proj_mat=self.v_proj_mat,
            block_size=self.block_size,
        )
        return k_coarse.transpose(1, 2).contiguous(), v_coarse.transpose(1, 2).contiguous()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_coarse: torch.Tensor | None = None,
        v_coarse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hsa_attention(
            q, k, v,
            k_coarse=k_coarse, v_coarse=v_coarse,
            k_proj_mat=self.k_proj_mat, v_proj_mat=self.v_proj_mat,
            gate_lin=self.gate_lin,
            block_size=self.block_size,
            tau_min=self.tau_min, tau_max=self.tau_max,
            proj_segment_len=self.proj_segment_len,
            backend=self.backend,
        )
