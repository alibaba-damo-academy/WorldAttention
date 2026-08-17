"""KV-side routing granularity (paper method §6): pairing, expansion, routing. CPU only."""
import math

import torch

from wan.modules.hsa.routing import (
    block_attention_logits,
    build_block_routing,
    build_block_routing_from_logits,
    expand_pair_selection,
    pair_kv_candidates,
)


def _sel_set(index, num, b, h, r):
    return set(index[b, h, r, : num[b, h, r]].tolist())


def test_pair_logits_are_logsumexp():
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 3, 8)
    cand, n_pairs = pair_kv_candidates(logits, 64, "all128", 0)
    assert n_pairs == 4 and cand.shape[-1] == 4
    ref = torch.logsumexp(logits.view(1, 2, 3, 4, 2), dim=-1)
    assert torch.allclose(cand, ref)


def test_all128_odd_block_stays_fine():
    logits = torch.randn(1, 1, 2, 7)
    cand, n_pairs = pair_kv_candidates(logits, 64, "all128", 0)
    assert n_pairs == 3 and cand.shape[-1] == 4
    assert torch.allclose(cand[..., -1], logits[..., -1])


def test_mixed_boundary_straddler_stays_fine():
    logits = torch.randn(1, 1, 2, 10)
    cand, n_pairs = pair_kv_candidates(logits, 64, "mixed", inter_tokens=5 * 64 + 32)
    assert n_pairs == 2 and cand.shape[-1] == 2 + 6
    assert torch.allclose(cand[..., 2:], logits[..., 4:])


def test_mixed_without_history_is_all64():
    logits = torch.randn(1, 1, 2, 10)
    cand, n_pairs = pair_kv_candidates(logits, 64, "mixed", inter_tokens=0)
    assert n_pairs == 0 and torch.equal(cand, logits)


def test_expand_opens_both_blocks_of_a_pair():
    mask_c = torch.tensor([True, False, True]).view(1, 1, 1, 3)
    mask64 = expand_pair_selection(mask_c, n_pairs=2)
    assert mask64.squeeze().tolist() == [True, True, False, False, True]


def test_routing_from_logits_all64_matches_reference_selection():
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 4, 16)
    r_new = build_block_routing_from_logits(logits, 0.1, 0.95)
    r_old = build_block_routing(torch.softmax(logits, dim=-1), 0.1, 0.95)
    assert torch.equal(r_new.num, r_old.num)
    for h in range(3):
        for i in range(4):
            assert _sel_set(r_new.index, r_new.num, 0, h, i) == \
                   _sel_set(r_old.index, r_old.num, 0, h, i)


def test_routing_from_logits_mixed_expands_pairs():
    torch.manual_seed(2)
    logits = torch.randn(1, 2, 3, 12)
    r = build_block_routing_from_logits(logits, 0.1, 0.95, kv_granularity="mixed",
                                        inter_tokens=8 * 64)
    for h in range(2):
        for i in range(3):
            sel = _sel_set(r.index, r.num, 0, h, i)
            for blk in sel:
                if blk < 8:
                    assert (blk ^ 1) in sel


def test_routing_from_logits_force_density():
    logits = torch.randn(1, 1, 2, 10)
    r = build_block_routing_from_logits(logits, 0.1, 0.95, force_density=0.30,
                                        kv_granularity="all128")
    assert torch.all(r.num == 4)


def test_paired_routing_returns_a_mask_for_the_differentiable_path():
    """Training consumes a mask; it must describe exactly the paired selection."""
    torch.manual_seed(3)
    logits = torch.randn(1, 2, 3, 12)
    r = build_block_routing_from_logits(logits, 0.1, 0.95, kv_granularity="mixed",
                                        inter_tokens=8 * 64, return_mask=True)
    assert r.mask is not None and r.mask.shape == logits.shape
    for h in range(2):
        for i in range(3):
            from_mask = set(r.mask[0, h, i].nonzero().flatten().tolist())
            from_index = _sel_set(r.index, r.num, 0, h, i)
            assert from_mask == from_index, "mask and index must agree"
            for blk in from_mask:
                if blk < 8:
                    assert (blk ^ 1) in from_mask, "paired region stays paired in the mask"


def test_paired_routing_mask_absent_by_default():
    logits = torch.randn(1, 1, 2, 10)
    r = build_block_routing_from_logits(logits, 0.1, 0.95, kv_granularity="all128")
    assert r.mask is None
