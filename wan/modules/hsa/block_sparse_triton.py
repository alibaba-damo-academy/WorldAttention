"""Triton block-sparse attention kernels for the HSA sparse branch.

Forward and backward operate on a compressed selection: for each query block, ``q2k_index`` lists
the key blocks to visit and ``q2k_num`` how many of them are valid. Skipped blocks are never
loaded, so block-level sparsity turns directly into skipped tensor-core work.

Everything here is Triton JIT, so there is no build step.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

__all__ = ["block_sparse_attention", "map_to_index"]


@triton.jit
def _map_to_index_kernel(
    map_ptr, index_ptr, index_num_ptr,
    map_bs_stride, map_h_stride, map_q_stride, map_kv_stride,
    index_bs_stride, index_h_stride, index_q_stride, index_kv_stride,
    index_num_bs_stride, index_num_h_stride, index_num_q_stride,
    num_kv_blocks,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_base = index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    map_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    num = 0
    for i in tl.range(num_kv_blocks):
        if tl.load(map_base + i * map_kv_stride):
            tl.store(index_base + num * index_kv_stride, i)
            num += 1

    tl.store(
        index_num_ptr + b * index_num_bs_stride + h * index_num_h_stride + q * index_num_q_stride,
        num,
    )


def map_to_index(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a boolean ``[B, H, Nq, Nkv]`` selection mask to (index, count) form."""
    bsz, heads, num_q_blocks, _ = block_map.shape
    index = torch.full(block_map.shape, -1, dtype=torch.int32, device=block_map.device)
    index_num = torch.empty((bsz, heads, num_q_blocks), dtype=torch.int32, device=block_map.device)

    _map_to_index_kernel[(bsz, heads, num_q_blocks)](
        block_map, index, index_num,
        block_map.stride(0), block_map.stride(1), block_map.stride(2), block_map.stride(3),
        index.stride(0), index.stride(1), index.stride(2), index.stride(3),
        index_num.stride(0), index_num.stride(1), index_num.stride(2),
        num_kv_blocks=block_map.shape[-1],
    )
    return index, index_num


_FWD_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=s, num_warps=w)
    for s in (3, 4, 7)
    for w in (4, 8)
]


@triton.autotune(_FWD_CONFIGS, key=["q_ctx", "kv_ctx", "HEAD_DIM"])
@triton.jit
def _fwd_kernel(
    q, k, v, sm_scale,
    q2k_index, q2k_num, max_kv_blks, block_sizes,
    m, out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    z, h, q_ctx, kv_ctx,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch = off_hz // h
    head = off_hz % h
    meta_base = (batch * h + head) * (q_ctx // BLOCK_M) + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    q_off = batch.to(tl.int64) * stride_qz + head.to(tl.int64) * stride_qh
    k_off = batch.to(tl.int64) * stride_kz + head.to(tl.int64) * stride_kh
    v_off = batch.to(tl.int64) * stride_vz + head.to(tl.int64) * stride_vh
    o_off = batch.to(tl.int64) * stride_oz + head.to(tl.int64) * stride_oh

    q_ptr = tl.make_block_ptr(
        base=q + q_off, shape=(q_ctx, HEAD_DIM), strides=(stride_qm, stride_qk),
        offsets=(q_blk * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    k_base = tl.make_block_ptr(
        base=k + k_off, shape=(HEAD_DIM, kv_ctx), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    v_base = tl.make_block_ptr(
        base=v + v_off, shape=(kv_ctx, HEAD_DIM), strides=(stride_vk, stride_vn),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    o_ptr = tl.make_block_ptr(
        base=out + o_off, shape=(q_ctx, HEAD_DIM), strides=(stride_om, stride_on),
        offsets=(q_blk * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q_block = tl.load(q_ptr)

    for i in tl.range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(block_sizes + kv_idx)
        k_ptr = tl.advance(k_base, (0, kv_idx * BLOCK_N))
        v_ptr = tl.advance(v_base, (kv_idx * BLOCK_N, 0))

        qk = tl.dot(q_block, tl.load(k_ptr))
        mask = tl.arange(0, BLOCK_N) < tl.minimum(block_size, BLOCK_N)
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), tl.load(v_ptr), acc)
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(m + off_hz * q_ctx + offs_m, m_i)
    tl.store(o_ptr, acc.to(out.type.element_ty))


@triton.jit
def _bwd_preprocess(o, do, delta, z, h, n_ctx, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o_tile = tl.load(o + off_hz * HEAD_DIM * n_ctx + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do_tile = tl.load(
        do + off_hz * HEAD_DIM * n_ctx + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    tl.store(delta + off_hz * n_ctx + off_m, tl.sum(o_tile * do_tile, axis=1))


@triton.jit
def _bwd_dkdv(
    dk, dv, q, k, v, sm_scale, do, m, d,
    k2q_index, k2q_num, max_q_blks, block_sizes,
    stride_tok, stride_d, h, n_ctx,
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_n, start_m,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_k = tl.arange(0, HEAD_DIM)
    q_t_ptrs = q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = do + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    step_m = BLOCK_M1

    kv_blk = tl.program_id(0)
    off_hz = tl.program_id(2)
    b = off_hz // h
    head_idx = off_hz % h
    meta_base = (b * h + head_idx) * (n_ctx // BLOCK_N1) + kv_blk

    q_blocks = tl.load(k2q_num + meta_base)
    q_ptr = k2q_index + meta_base * max_q_blks
    block_size = tl.load(block_sizes + kv_blk)

    for blk_idx in range(q_blocks * 2):
        offset = (tl.load(q_ptr + blk_idx // 2).to(tl.int32) * 2 + blk_idx % 2) * step_m
        q_t = tl.load(q_t_ptrs + offset * stride_tok)
        offs_m_blk = start_m + offset + tl.arange(0, BLOCK_M1)
        m_tile = tl.load(m + offs_m_blk)

        p_t = tl.math.exp2(tl.dot(k, q_t) - m_tile[None, :])
        p_t = tl.where((tl.arange(0, BLOCK_N1) < block_size)[:, None], p_t, 0.0)

        do_tile = tl.load(do_ptrs + offset * stride_tok)
        dv += tl.dot(p_t.to(tl.bfloat16), do_tile)

        d_i = tl.load(d + offs_m_blk)
        dp_t = tl.dot(v, tl.trans(do_tile)).to(tl.float32)
        ds_t = p_t * (dp_t - d_i[None, :])
        dk += tl.dot(ds_t.to(tl.bfloat16), tl.trans(q_t))

    return dk, dv


@triton.jit
def _bwd_dq(
    dq, q, k, v, do, m, d,
    q2k_index, q2k_num, max_kv_blks, block_sizes,
    stride_tok, stride_d, h, n_ctx,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_m, start_n,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    k_t_ptrs = k + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    v_t_ptrs = v + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    d_i = tl.load(d + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    step_n = BLOCK_N2

    q_blk = tl.program_id(0)
    off_hz = tl.program_id(2)
    b = off_hz // h
    head_idx = off_hz % h
    meta_base = (b * h + head_idx) * (n_ctx // BLOCK_M2) + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    # Each key block spans two BLOCK_N2 halves; the padding mask comes from the block's own real
    # size, indexed by its key-block id rather than by its position in the selection list.
    for blk_idx in range(kv_blocks * 2):
        kv_idx = tl.load(kv_ptr + blk_idx // 2).to(tl.int32)
        offset = (kv_idx * 2 + blk_idx % 2) * step_n * stride_tok
        block_size = tl.load(block_sizes + kv_idx) - (blk_idx % 2) * step_n

        k_t = tl.load(k_t_ptrs + offset)
        v_t = tl.load(v_t_ptrs + offset)
        p = tl.math.exp2(tl.dot(q, k_t) - m)
        p = tl.where((tl.arange(0, BLOCK_N2) < block_size.to(tl.int32))[None, :], p, 0.0)
        dp = tl.dot(do, v_t).to(tl.float32)
        dq += tl.dot((p * (dp - d_i[:, None])).to(tl.bfloat16), tl.trans(k_t))

    return dq


@triton.jit
def _bwd_kernel(
    q, k, v, sm_scale, do, dq, dk, dv, m, d,
    q2k_index, q2k_num, max_kv_blks,
    k2q_index, k2q_num, max_q_blks,
    block_sizes,
    stride_z, stride_h, stride_tok, stride_d, h, n_ctx,
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    LN2 = 0.6931471824645996

    bhid = tl.program_id(2)
    off_chz = (bhid * n_ctx).to(tl.int64)
    adj = (stride_h * (bhid % h) + stride_z * (bhid // h)).to(tl.int64)
    pid = tl.program_id(0)

    q += adj
    k += adj
    v += adj
    do += adj
    dq += adj
    dk += adj
    dv += adj
    m += off_chz
    d += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    k_tile = tl.load(k + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v_tile = tl.load(v + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    dk_acc, dv_acc = _bwd_dkdv(
        tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32),
        tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32),
        q, k_tile, v_tile, sm_scale, do, m, d,
        k2q_index, k2q_num, max_q_blks, block_sizes,
        stride_tok, stride_d, h, n_ctx,
        BLOCK_M1, BLOCK_N1, HEAD_DIM, start_n, 0,
    )
    tl.store(dv + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, dv_acc)
    tl.store(dk + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, dk_acc * sm_scale)

    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q_tile = tl.load(q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    do_tile = tl.load(do + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m_tile = tl.load(m + offs_m)[:, None]

    dq_acc = _bwd_dq(
        tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32),
        q_tile, k, v, do_tile, m_tile, d,
        q2k_index, q2k_num, max_kv_blks, block_sizes,
        stride_tok, stride_d, h, n_ctx,
        BLOCK_M2, BLOCK_N2, HEAD_DIM, start_m, 0,
    )
    tl.store(dq + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, dq_acc * LN2)


def _forward(q, k, v, q2k_index, q2k_num, block_sizes):
    bsz, heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    out = torch.empty_like(q)
    m = torch.empty((bsz, heads, q_len), dtype=torch.float32, device=q.device)

    _fwd_kernel[lambda _: (triton.cdiv(q_len, 64), bsz * heads, 1)](
        q, k, v, 1.0 / math.sqrt(head_dim),
        q2k_index, q2k_num, q2k_index.shape[-1], block_sizes,
        m, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        bsz, heads, q_len, kv_len,
        HEAD_DIM=head_dim,
    )
    return out, m


def _backward(do, q, k, v, o, m, q2k_index, q2k_num, k2q_index, k2q_num, block_sizes):
    do = do.contiguous()
    bsz, heads, n_ctx, head_dim = q.shape
    sm_scale = 1.0 / math.sqrt(head_dim)

    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
    PRE_BLOCK = 64
    if n_ctx % PRE_BLOCK != 0:
        raise ValueError(f"token count must be a multiple of {PRE_BLOCK}, got {n_ctx}")

    delta = torch.empty_like(m)
    _bwd_preprocess[(n_ctx // PRE_BLOCK, bsz * heads)](
        o, do, delta, bsz, heads, n_ctx, BLOCK_M=PRE_BLOCK, HEAD_DIM=head_dim,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _bwd_kernel[(n_ctx // BLOCK_N1, 1, bsz * heads)](
        q, k * (sm_scale * 1.4426950408889634), v, sm_scale, do, dq, dk, dv, m, delta,
        q2k_index, q2k_num, q2k_index.shape[-1],
        k2q_index, k2q_num, k2q_index.shape[-1],
        block_sizes,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), heads, n_ctx,
        BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1, BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
        HEAD_DIM=head_dim,
    )
    return dq, dk, dv


class _BlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_map, block_sizes):
        q2k_index, q2k_num = map_to_index(block_map)
        out, m = _forward(q, k, v, q2k_index, q2k_num, block_sizes)
        ctx.save_for_backward(q, k, v, out, m, block_map, block_sizes, q2k_index, q2k_num)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, m, block_map, block_sizes, q2k_index, q2k_num = ctx.saved_tensors
        k2q_index, k2q_num = map_to_index(block_map.transpose(-1, -2).contiguous())
        dq, dk, dv = _backward(
            grad_out, q, k, v, out, m, q2k_index, q2k_num, k2q_index, k2q_num, block_sizes,
        )
        return dq, dk, dv, None, None


def block_sparse_attention(q, k, v, *, block_sizes, q2k_index=None, q2k_num=None, block_map=None):
    """Block-sparse attention over ``[B, H, N, D]`` tensors padded to the block size.

    Pass ``block_map`` for a differentiable call, or the precomputed ``(q2k_index, q2k_num)`` pair
    for inference. Returns ``[B, H, N, D]``.
    """
    if block_map is not None:
        return _BlockSparseAttention.apply(q, k, v, block_map, block_sizes)
    if q2k_index is None or q2k_num is None:
        raise ValueError("provide either block_map or both q2k_index and q2k_num")
    out, _ = _forward(q, k, v, q2k_index, q2k_num, block_sizes)
    return out
