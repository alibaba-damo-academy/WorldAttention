"""HSA module wiring: branch composition, compressed cache, gradients, distillation. CPU only."""
import sys

import torch

from wan.modules.hsa import (
    HSAAttention,
    block_sparse_attention_reference,
    compress_kv,
    hsa_parameter_names,
    require_trained_hsa,
)
from wan.modules.hsa.attention import _linear_branch, resolve_backend
from trainer.hsa_stage import init_hsa_parameters
from wan.modules.hsa.routing import (
    block_attention_map,
    block_counts,
    block_mean_pool,
    build_block_routing,
    pad_to_len,
    variable_block_sizes,
)
from wan.modules.hsa import distill

HEADS, HEAD_DIM = 3, 16
SEGMENT = 50
Q_LEN, KV_LEN = 100, 200


def _module(gate_bias=None, initialized=True):
    torch.manual_seed(0)
    hsa = HSAAttention(
        num_heads=HEADS, head_dim=HEAD_DIM, block_size=64,
        tau_min=0.35, tau_max=1.0, proj_segment_len=SEGMENT, backend="torch",
    )
    if initialized:
        init_hsa_parameters(hsa)
    if gate_bias is not None:
        with torch.no_grad():
            hsa.gate_lin.bias.fill_(gate_bias)
    return hsa


def _qkv(dtype=torch.float32):
    torch.manual_seed(1)
    q = torch.randn(1, Q_LEN, HEADS, HEAD_DIM, dtype=dtype)
    k = torch.randn(1, KV_LEN, HEADS, HEAD_DIM, dtype=dtype)
    v = torch.randn(1, KV_LEN, HEADS, HEAD_DIM, dtype=dtype)
    return q, k, v


def _branches(hsa, q, k, v):
    """Recompute both branches from the public building blocks."""
    k_coarse, v_coarse = compress_kv(
        k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(),
        segment_len=hsa.proj_segment_len,
        k_proj_mat=hsa.k_proj_mat, v_proj_mat=hsa.v_proj_mat, block_size=hsa.block_size,
    )
    linear = _linear_branch(q.transpose(1, 2).contiguous(), k_coarse, v_coarse).transpose(1, 2)

    block = hsa.block_size
    q_pad_len, kv_pad_len, q_blocks, kv_blocks = block_counts(Q_LEN, KV_LEN, block)
    q_t = pad_to_len(q, q_pad_len).transpose(1, 2).contiguous()
    k_t = pad_to_len(k, kv_pad_len).transpose(1, 2).contiguous()
    v_t = pad_to_len(v, kv_pad_len).transpose(1, 2).contiguous()
    routing = build_block_routing(
        block_attention_map(
            block_mean_pool(q_t, Q_LEN, q_blocks, block),
            block_mean_pool(k_t, KV_LEN, kv_blocks, block),
        ),
        tau_min=hsa.tau_min, tau_max=hsa.tau_max,
    )
    sparse = block_sparse_attention_reference(
        q_t.to(torch.bfloat16), k_t.to(torch.bfloat16), v_t.to(torch.bfloat16),
        routing.index, routing.num,
        variable_block_sizes(KV_LEN, kv_pad_len, block, q.device), block,
    )
    sparse = sparse.transpose(1, 2)[:, :Q_LEN].to(linear.dtype)
    return linear, sparse


def test_forward_preserves_shape_and_dtype():
    hsa = _module()
    q, k, v = _qkv()
    out = hsa(q, k, v)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()


def test_output_is_the_gated_sum_of_the_two_branches():
    q, k, v = _qkv()
    for gate_bias in (-12.0, 0.0, 8.0):
        hsa = _module(gate_bias=gate_bias)
        with torch.no_grad():
            out = hsa(q, k, v)
            linear, sparse = _branches(hsa, q, k, v)
            gate = torch.sigmoid(hsa.gate_lin(q.to(hsa.gate_lin.weight.dtype))).to(linear.dtype)
        assert torch.allclose(out, sparse + linear * gate, atol=2e-2), f"gate_bias={gate_bias}"


def test_parameters_are_unset_until_loaded_or_initialized():
    """The module fabricates nothing: every HSA parameter starts at zero."""
    hsa = _module(initialized=False)
    assert torch.count_nonzero(hsa.k_proj_mat) == 0
    assert torch.count_nonzero(hsa.v_proj_mat) == 0
    assert torch.count_nonzero(hsa.gate_lin.weight) == 0
    assert torch.count_nonzero(hsa.gate_lin.bias) == 0
    hsa.reset_parameters()   # idempotent: still the zero placeholder
    assert torch.count_nonzero(hsa.k_proj_mat) == 0


def test_a_checkpoint_without_hsa_weights_is_rejected():
    hsa = _module(initialized=False)
    keys = hsa_parameter_names(hsa)
    assert len(keys) == 4

    require_trained_hsa([], "ckpt.pt")                    # nothing missing -> fine
    require_trained_hsa(["blocks.0.self_attn.q.weight"])  # unrelated key -> fine
    try:
        require_trained_hsa(["blocks.0.self_attn.hsa_attention.k_proj_mat"], "ckpt.pt")
    except RuntimeError as err:
        assert "HSA training" in str(err) and "ckpt.pt" in str(err)
    else:
        raise AssertionError("expected a missing HSA parameter to be rejected")


def test_precomputed_compressed_cache_matches_inline_compression():
    hsa = _module(gate_bias=0.0)
    q, k, v = _qkv()
    with torch.no_grad():
        inline = hsa(q, k, v)
        k_coarse, v_coarse = hsa.compress_kv_cache(k, v)
        cached = hsa(q, k, v, k_coarse=k_coarse, v_coarse=v_coarse)
    assert torch.allclose(inline, cached, atol=1e-5)


def test_compress_kv_applies_the_projection():
    hsa = _module()
    _, k, v = _qkv()
    k_coarse, _ = hsa.compress_kv_cache(k, v)

    segments = KV_LEN // SEGMENT
    assert k_coarse.shape == (1, segments * hsa.proj_rank, HEADS, HEAD_DIM)

    expected = torch.matmul(
        hsa.k_proj_mat[: hsa.proj_rank, :SEGMENT].float(),
        k[0, :SEGMENT, 0].float(),
    )
    assert torch.allclose(k_coarse[0, : hsa.proj_rank, 0].float(), expected, atol=2e-2)


def test_training_initialization_gives_block_pooling():
    """init_hsa_parameters makes each projection row average one block of tokens."""
    hsa = _module(initialized=False)
    assert init_hsa_parameters(hsa) == 1
    row = hsa.k_proj_mat[0]
    assert int((row != 0).sum()) == min(hsa.block_size, SEGMENT)
    assert abs(row[0].item() - 1.0 / min(hsa.block_size, SEGMENT)) < 1e-3
    assert torch.count_nonzero(hsa.gate_lin.bias) == 0


def test_coarse_cache_sizing_and_alignment():
    hsa = _module()
    assert hsa.coarse_cache_size(0) == 0
    assert hsa.coarse_cache_size(SEGMENT) == hsa.proj_rank
    assert hsa.coarse_cache_size(4 * SEGMENT) == 4 * hsa.proj_rank
    assert hsa.token_range_to_coarse(0, 2 * SEGMENT) == (0, 2 * hsa.proj_rank)
    assert hsa.token_range_to_coarse(SEGMENT, 3 * SEGMENT) == (hsa.proj_rank, 3 * hsa.proj_rank)
    try:
        hsa.token_range_to_coarse(0, SEGMENT + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected unaligned token range to be rejected")


def test_gradients_reach_the_hsa_parameters():
    hsa = _module(gate_bias=0.0)
    q, k, v = _qkv()
    hsa(q, k, v).float().pow(2).mean().backward()
    for name in ("k_proj_mat", "v_proj_mat"):
        grad = getattr(hsa, name).grad
        assert grad is not None and grad.abs().sum().item() > 0, name
    assert hsa.gate_lin.weight.grad is not None
    assert hsa.gate_lin.weight.grad.abs().sum().item() > 0
    assert hsa.gate_lin.bias.grad.abs().sum().item() > 0


def test_a_saturated_gate_starves_the_gate_gradient():
    """Why the training initialization puts the gate bias at zero rather than far from it."""
    grads = {}
    for gate_bias in (-12.0, 0.0):
        hsa = _module(gate_bias=gate_bias)
        q, k, v = _qkv()
        hsa(q, k, v).float().pow(2).mean().backward()
        grads[gate_bias] = hsa.gate_lin.bias.grad.abs().sum().item()
    assert grads[0.0] > grads[-12.0] * 100


def test_distillation_records_one_loss_per_forward():
    hsa = _module(gate_bias=0.0)
    q, k, v = _qkv()

    distill.enable_distill(True)
    try:
        hsa(q, k, v)
        hsa(q, k, v)
        losses = distill.pop_distill_losses()
    finally:
        distill.enable_distill(False)

    assert len(losses) == 2
    assert all(loss.requires_grad for loss in losses)
    assert distill.pop_distill_losses() == []


def test_distillation_is_silent_when_disabled():
    hsa = _module()
    q, k, v = _qkv()
    hsa(q, k, v)
    assert distill.pop_distill_losses() == []


def test_backend_resolution():
    assert resolve_backend("torch") == "torch"
    assert resolve_backend("triton") == "triton"
    assert resolve_backend("auto") in ("torch", "triton")
    try:
        resolve_backend("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected an unknown backend to be rejected")


def test_block_size_is_validated():
    try:
        HSAAttention(num_heads=1, head_dim=8, block_size=32)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a non-64 block size to be rejected")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(dict(globals()).items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print("PASS", name)
        except Exception as err:  # noqa: BLE001
            failures += 1
            print("FAIL", name, "->", err)
    print("\nRESULT:", "ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
