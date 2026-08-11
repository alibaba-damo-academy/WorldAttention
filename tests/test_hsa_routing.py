"""Head-adaptive routing and the block-sparse reference kernel. Runs on CPU."""
import math
import sys

import torch

from wan.modules.hsa import block_sparse_attention_reference, build_block_routing
from wan.modules.hsa.routing import block_attention_map, block_mean_pool, variable_block_sizes


def _reference_routing(block_attn, tau_min, tau_max):
    """Independent transcription of the paper's Gini / threshold / minimal-k rules."""
    bsz, heads, num_q, num_kv = block_attn.shape
    probs, order = torch.sort(block_attn, dim=-1, descending=True)

    gini = torch.zeros(bsz, heads, num_q)
    for b in range(bsz):
        for h in range(heads):
            for i in range(num_q):
                total = sum(
                    (num_kv - 2 * (m + 1) + 1) * probs[b, h, i, m].item() for m in range(num_kv)
                )
                gini[b, h, i] = total / num_kv
    thresholds = tau_min + gini.mean(dim=(0, 2)) * (tau_max - tau_min)

    counts = torch.zeros(bsz, heads, num_q, dtype=torch.int32)
    for b in range(bsz):
        for h in range(heads):
            for i in range(num_q):
                acc, k = 0.0, 0
                while k < num_kv:
                    acc += probs[b, h, i, k].item()
                    k += 1
                    if acc >= thresholds[h].item():
                        break
                counts[b, h, i] = k
    return thresholds, counts, order


def test_gini_threshold_and_selection_match_the_definition():
    torch.manual_seed(0)
    block_attn = torch.softmax(torch.randn(2, 4, 5, 16) * 2.0, dim=-1)
    tau_min, tau_max = 0.35, 1.0

    routing = build_block_routing(block_attn, tau_min, tau_max, return_mask=True)
    thresholds, counts, order = _reference_routing(block_attn, tau_min, tau_max)

    assert torch.allclose(routing.thresholds, thresholds, atol=1e-5)
    assert torch.equal(routing.num, counts)
    assert torch.equal(routing.index, order.to(torch.int32))

    expected_mask = torch.zeros_like(routing.mask)
    for b in range(block_attn.shape[0]):
        for h in range(block_attn.shape[1]):
            for i in range(block_attn.shape[2]):
                expected_mask[b, h, i, order[b, h, i, : counts[b, h, i]]] = True
    assert torch.equal(routing.mask, expected_mask)


def test_gini_spans_uniform_to_one_hot():
    num_kv = 16
    uniform = torch.full((1, 1, 1, num_kv), 1.0 / num_kv)
    one_hot = torch.zeros(1, 1, 1, num_kv)
    one_hot[..., 3] = 1.0

    tau_uniform = build_block_routing(uniform, 0.0, 1.0).thresholds.item()
    tau_one_hot = build_block_routing(one_hot, 0.0, 1.0).thresholds.item()

    assert abs(tau_uniform) < 1e-6
    assert abs(tau_one_hot - (num_kv - 1) / num_kv) < 1e-6
    # A peaked head keeps more cumulative mass than a diffuse one.
    assert tau_one_hot > tau_uniform


def test_thresholds_are_per_head():
    num_kv = 16
    block_attn = torch.stack([
        torch.full((1, 1, num_kv), 1.0 / num_kv),
        torch.nn.functional.one_hot(torch.tensor([[0]]), num_kv).float(),
    ], dim=1)
    thresholds = build_block_routing(block_attn, 0.35, 1.0).thresholds
    assert thresholds.shape == (2,)
    assert thresholds[1] > thresholds[0] + 0.5


def test_block_mean_pool_ignores_padding_in_a_partial_block():
    block_size, num_blocks = 4, 3
    logical_len = 10
    x = torch.zeros(1, 1, num_blocks * block_size, 2)
    x[0, 0, :logical_len] = 1.0

    pooled = block_mean_pool(x, logical_len, num_blocks, block_size)
    # The trailing block holds 2 real tokens; averaging over them must give 1.0, not 0.5.
    assert torch.allclose(pooled[0, 0, 2], torch.ones(2))


def test_reference_kernel_matches_dense_attention_on_full_selection():
    torch.manual_seed(0)
    bsz, heads, tokens, head_dim, block_size = 1, 2, 128, 16, 64
    num_blocks = tokens // block_size
    q, k, v = (torch.randn(bsz, heads, tokens, head_dim, dtype=torch.float32) for _ in range(3))

    index = torch.arange(num_blocks, dtype=torch.int32).view(1, 1, 1, num_blocks)
    index = index.expand(bsz, heads, num_blocks, num_blocks).contiguous()
    num = torch.full((bsz, heads, num_blocks), num_blocks, dtype=torch.int32)
    sizes = variable_block_sizes(tokens, tokens, block_size, q.device)

    got = block_sparse_attention_reference(q, k, v, index, num, sizes, block_size)
    want = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, want, atol=1e-5)


def test_reference_kernel_skips_unselected_blocks():
    torch.manual_seed(0)
    bsz, heads, tokens, head_dim, block_size = 1, 1, 128, 16, 64
    num_blocks = tokens // block_size
    q, k, v = (torch.randn(bsz, heads, tokens, head_dim) for _ in range(3))

    # Every query block attends to key block 0 only.
    index = torch.zeros(bsz, heads, num_blocks, num_blocks, dtype=torch.int32)
    num = torch.ones((bsz, heads, num_blocks), dtype=torch.int32)
    sizes = variable_block_sizes(tokens, tokens, block_size, q.device)

    got = block_sparse_attention_reference(q, k, v, index, num, sizes, block_size)
    want = torch.nn.functional.scaled_dot_product_attention(
        q, k[:, :, :block_size], v[:, :, :block_size]
    )
    assert torch.allclose(got, want, atol=1e-5)


def test_reference_kernel_masks_a_partial_trailing_block():
    torch.manual_seed(0)
    bsz, heads, head_dim, block_size = 1, 1, 16, 64
    kv_len, kv_len_padded = 96, 128
    num_blocks = kv_len_padded // block_size
    q = torch.randn(bsz, heads, block_size, head_dim)
    k = torch.randn(bsz, heads, kv_len_padded, head_dim)
    v = torch.randn(bsz, heads, kv_len_padded, head_dim)
    k[:, :, kv_len:] = 0
    v[:, :, kv_len:] = 0

    index = torch.tensor([0, 1], dtype=torch.int32).view(1, 1, 1, 2)
    num = torch.full((bsz, heads, 1), 2, dtype=torch.int32)
    sizes = variable_block_sizes(kv_len, kv_len_padded, block_size, q.device)
    assert sizes.tolist() == [block_size, kv_len - block_size]

    got = block_sparse_attention_reference(q, k, v, index, num, sizes, block_size)
    want = torch.nn.functional.scaled_dot_product_attention(
        q, k[:, :, :kv_len], v[:, :, :kv_len]
    )
    assert torch.allclose(got, want, atol=1e-5)


def test_reference_kernel_ignores_slots_past_the_selection_count():
    """Only the first q2k_num entries of a row may influence its output."""
    torch.manual_seed(0)
    bsz, heads, tokens, head_dim, block_size = 1, 2, 256, 16, 64
    num_blocks = tokens // block_size
    q, k, v = (torch.randn(bsz, heads, tokens, head_dim) for _ in range(3))

    index = torch.arange(num_blocks, dtype=torch.int32)
    index = index.view(1, 1, 1, num_blocks).expand(bsz, heads, num_blocks, num_blocks).contiguous()
    num = torch.randint(1, num_blocks + 1, (bsz, heads, num_blocks), dtype=torch.int32)
    sizes = variable_block_sizes(tokens, tokens, block_size, q.device)

    baseline = block_sparse_attention_reference(q, k, v, index, num, sizes, block_size)

    scrambled = index.clone()
    for b in range(bsz):
        for h in range(heads):
            for i in range(num_blocks):
                kept = int(num[b, h, i])
                scrambled[b, h, i, kept:] = torch.randint(
                    0, num_blocks, (num_blocks - kept,), dtype=torch.int32
                )
    assert not torch.equal(scrambled, index)
    perturbed = block_sparse_attention_reference(q, k, v, scrambled, num, sizes, block_size)
    assert torch.allclose(baseline, perturbed, atol=1e-6)


def test_reference_kernel_returns_zero_for_a_query_block_with_no_selection():
    torch.manual_seed(0)
    bsz, heads, tokens, head_dim, block_size = 1, 1, 128, 16, 64
    num_blocks = tokens // block_size
    q, k, v = (torch.randn(bsz, heads, tokens, head_dim) for _ in range(3))

    index = torch.zeros(bsz, heads, num_blocks, num_blocks, dtype=torch.int32)
    num = torch.tensor([[[0, 1]]], dtype=torch.int32)
    sizes = variable_block_sizes(tokens, tokens, block_size, q.device)

    out = block_sparse_attention_reference(q, k, v, index, num, sizes, block_size)
    assert torch.count_nonzero(out[:, :, :block_size]) == 0
    assert torch.count_nonzero(out[:, :, block_size:]) > 0


def test_end_to_end_routing_from_pooled_maps():
    torch.manual_seed(0)
    bsz, heads, tokens, head_dim, block_size = 1, 3, 256, 16, 64
    num_blocks = tokens // block_size
    q = torch.randn(bsz, heads, tokens, head_dim)
    k = torch.randn(bsz, heads, tokens, head_dim)

    block_attn = block_attention_map(
        block_mean_pool(q, tokens, num_blocks, block_size),
        block_mean_pool(k, tokens, num_blocks, block_size),
    )
    assert torch.allclose(block_attn.sum(-1), torch.ones(bsz, heads, num_blocks), atol=1e-5)

    routing = build_block_routing(block_attn, 0.35, 1.0, return_mask=True)
    assert routing.num.min().item() >= 1
    assert routing.num.max().item() <= num_blocks
    assert torch.equal(routing.mask.sum(-1).to(torch.int32), routing.num)


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
