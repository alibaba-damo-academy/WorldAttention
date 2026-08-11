"""HKVCache end to end: chunk store, two-stage retrieval, page install, compressed-cache rebuild.

Runs on CPU against a synthetic online cache, so it needs no model weights.
"""
import shutil
import sys
import tempfile

import torch

from pipeline.hkv import HKVCache
from pipeline.hkv.rope import shift_temporal_rope, temporal_band
from wan.modules.hsa import HSAAttention

FRAME_TOKENS = 4
PAGE_FRAMES = 2
PAGE_TOKENS = PAGE_FRAMES * FRAME_TOKENS
CAPACITY = 16          # 4 frames = 2 pages
BLOCKS = 2
HEADS, HEAD_DIM = 1, 24
PROMPT_DIM = 8
STALE = -999.0


def _freqs():
    pairs = HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, pairs, dtype=torch.float64) / pairs))
    table = torch.polar(
        torch.ones(128, pairs, dtype=torch.float64),
        torch.outer(torch.arange(128, dtype=torch.float64), inv_freq),
    )
    return temporal_band(table, HEAD_DIM)


FREQS = _freqs()


def _hsa_modules():
    modules = []
    for block in range(BLOCKS):
        hsa = HSAAttention(
            num_heads=HEADS, head_dim=HEAD_DIM, block_size=64,
            proj_segment_len=FRAME_TOKENS, backend="torch",
        )
        # A distinct projection per layer, so a cross-layer mix-up is detectable.
        with torch.no_grad():
            hsa.k_proj_mat.fill_(0.0)
            hsa.k_proj_mat[0, :] = (block + 1) / FRAME_TOKENS
            hsa.v_proj_mat.fill_(0.0)
            hsa.v_proj_mat[0, :] = (block + 1) / FRAME_TOKENS
        modules.append(hsa)
    return modules


HSA_MODULES = _hsa_modules()
COARSE_SLOTS = HSA_MODULES[0].coarse_cache_size(CAPACITY)


def _make_cache(with_coarse=True, stale=False):
    cache = []
    for _ in range(BLOCKS):
        block = {
            "k": torch.zeros(1, CAPACITY, HEADS, HEAD_DIM),
            "v": torch.zeros(1, CAPACITY, HEADS, HEAD_DIM),
            "global_end_index": torch.zeros([], dtype=torch.long),
            "local_end_index": torch.zeros([], dtype=torch.long),
        }
        if with_coarse:
            fill = STALE if stale else 0.0
            block["k_coarse"] = torch.full((1, COARSE_SLOTS, HEADS, HEAD_DIM), fill)
            block["v_coarse"] = torch.full((1, COARSE_SLOTS, HEADS, HEAD_DIM), fill)
        cache.append(block)
    return cache


def _fill(cache, valid_tokens, tag):
    for block_idx, block in enumerate(cache):
        for token in range(valid_tokens):
            block["k"][0, token, 0, :] = tag * 1000 + block_idx * 100 + token
            block["v"][0, token, 0, :] = -(tag * 1000 + block_idx * 100 + token)
        block["local_end_index"].fill_(valid_tokens)


def _prompt(seed):
    torch.manual_seed(seed)
    return torch.randn(1, 3, PROMPT_DIM)


def _bank(topk_pages=2, stage1=2, rerope=False, **kwargs):
    return HKVCache(
        frame_seq_length=FRAME_TOKENS, page_size_frames=PAGE_FRAMES,
        topk_pages=topk_pages, stage1_topk_chunks=stage1, rerope=rerope, **kwargs,
    )


def _seed_chunks(bank, count=2):
    """Store `count` chunks, each a full window with a recognizable tag."""
    for tag in range(1, count + 1):
        cache = _make_cache()
        _fill(cache, CAPACITY, tag)
        bank.store(
            cache, prompt_embeds=_prompt(tag),
            current_start_frame=tag * 10, temporal_freqs=FREQS,
        )
    return bank


def test_store_splits_a_window_into_pages():
    bank = _bank()
    cache = _make_cache()
    _fill(cache, CAPACITY, tag=1)
    chunk_id = bank.store(cache, prompt_embeds=_prompt(1), current_start_frame=10, temporal_freqs=FREQS)

    assert chunk_id == 0 and len(bank) == 1
    chunk = bank.chunks[0]
    assert chunk["valid_tokens"] == CAPACITY
    assert chunk["page_spans"] == [(0, PAGE_TOKENS), (PAGE_TOKENS, CAPACITY)]
    assert chunk["start_frame"] == 10 - CAPACITY // FRAME_TOKENS
    assert chunk["page_index"].shape == (2, HEADS * HEAD_DIM)
    assert bank.tier.location(0) == "cpu"


def test_store_handles_a_partial_final_page():
    bank = _bank()
    cache = _make_cache()
    _fill(cache, 12, tag=1)          # 3 frames -> one full page plus a half page
    bank.store(cache, prompt_embeds=_prompt(1), current_start_frame=10, temporal_freqs=FREQS)
    assert bank.chunks[0]["page_spans"] == [(0, 8), (8, 12)]


def test_store_skips_an_empty_cache():
    bank = _bank()
    assert bank.store(_make_cache(), prompt_embeds=_prompt(1), current_start_frame=0,
                      temporal_freqs=FREQS) == -1
    assert len(bank) == 0


def test_reset_clears_the_bank():
    bank = _seed_chunks(_bank())
    assert len(bank) == 2
    bank.reset()
    assert len(bank) == 0 and bank.tier.location(0) == "absent"


def test_retrieve_returns_nothing_from_an_empty_bank():
    bank = _bank()
    assert bank.retrieve(prompt_embeds=_prompt(1), query_vec=torch.randn(HEADS * HEAD_DIM)) == []


def test_retrieve_picks_the_page_matching_the_query():
    bank = _seed_chunks(_bank(topk_pages=1))
    target = (1, 1)
    query = bank.chunks[target[0]]["page_index"][target[1]].clone()
    assert bank.retrieve(prompt_embeds=_prompt(2), query_vec=query) == [target]


def test_retrieve_pools_pages_across_chunks():
    bank = _seed_chunks(_bank(topk_pages=4, stage1=2), count=2)
    query = torch.randn(HEADS * HEAD_DIM)
    selected = bank.retrieve(prompt_embeds=_prompt(1), query_vec=query)
    assert len(selected) == 4
    assert len({chunk_id for chunk_id, _ in selected}) == 2


def test_retrieve_orders_pages_chronologically():
    bank = _seed_chunks(_bank(topk_pages=4, stage1=2), count=2)
    selected = bank.retrieve(prompt_embeds=_prompt(1), query_vec=torch.randn(HEADS * HEAD_DIM))
    keys = [
        (bank.chunks[cid]["start_frame"], bank.chunks[cid]["page_spans"][pid][0])
        for cid, pid in selected
    ]
    assert keys == sorted(keys)


def test_retrieve_rejects_a_mismatched_query_dim():
    bank = _seed_chunks(_bank())
    try:
        bank.retrieve(prompt_embeds=_prompt(1), query_vec=torch.randn(5))
    except ValueError:
        pass
    else:
        raise AssertionError("expected a mismatched query dimension to be rejected")


def test_apply_installs_the_selected_pages_verbatim():
    bank = _seed_chunks(_bank(topk_pages=1))
    selected = [(1, 1)]
    cache = _make_cache()
    installed = bank.apply(
        cache, selected, current_start_frame=30, temporal_freqs=FREQS, hsa_modules=HSA_MODULES,
    )

    assert installed == PAGE_TOKENS
    start, end = bank.chunks[1]["page_spans"][1]
    for block_idx, block in enumerate(cache):
        for offset, token in enumerate(range(start, end)):
            expected = 2 * 1000 + block_idx * 100 + token
            assert torch.allclose(block["k"][0, offset, 0], torch.full((HEAD_DIM,), float(expected)))
        assert (block["k"][0, PAGE_TOKENS:] == 0).all()
        assert int(block["local_end_index"].item()) == PAGE_TOKENS
        assert int(block["global_end_index"].item()) == 30 * FRAME_TOKENS


def test_apply_reropes_keys_to_their_new_positions():
    bank = _seed_chunks(_bank(topk_pages=1, rerope=True))
    selected = [(1, 1)]
    cache = _make_cache()
    bank.apply(cache, selected, current_start_frame=30, temporal_freqs=FREQS,
               hsa_modules=HSA_MODULES)

    chunk = bank.chunks[1]
    start, end = chunk["page_spans"][1]
    source = bank.tier.get(1)["k"][0][:, start:end]
    orig_frame = chunk["start_frame"] + start // FRAME_TOKENS
    new_frame = 30 - (end - start) // FRAME_TOKENS
    expected = shift_temporal_rope(source, FREQS, new_frame, orig_frame)

    assert torch.allclose(cache[0]["k"][:, : end - start], expected, atol=1e-3)
    # Values carry no rotary encoding, so they are installed unchanged.
    assert torch.allclose(cache[0]["v"][:, : end - start], bank.tier.get(1)["v"][0][:, start:end])


def test_apply_truncates_to_the_cache_capacity():
    bank = _seed_chunks(_bank(topk_pages=4, stage1=2), count=2)
    selected = bank.retrieve(prompt_embeds=_prompt(1), query_vec=torch.randn(HEADS * HEAD_DIM))
    assert len(selected) * PAGE_TOKENS > CAPACITY

    cache = _make_cache()
    installed = bank.apply(cache, selected, current_start_frame=30, temporal_freqs=FREQS,
                           hsa_modules=HSA_MODULES)
    assert installed == CAPACITY
    assert int(cache[0]["local_end_index"].item()) == CAPACITY


def test_apply_with_nothing_selected_is_a_no_op():
    bank = _seed_chunks(_bank())
    cache = _make_cache()
    _fill(cache, CAPACITY, tag=7)
    assert bank.apply(cache, [], current_start_frame=30, temporal_freqs=FREQS,
                      hsa_modules=HSA_MODULES) == 0
    assert cache[0]["k"].abs().sum() > 0


def test_apply_rebuilds_the_compressed_cache_from_the_installed_pages():
    bank = _seed_chunks(_bank(topk_pages=2, stage1=2))
    selected = bank.retrieve(prompt_embeds=_prompt(1), query_vec=torch.randn(HEADS * HEAD_DIM))
    cache = _make_cache(stale=True)
    installed = bank.apply(cache, selected, current_start_frame=30, temporal_freqs=FREQS,
                           hsa_modules=HSA_MODULES)

    for block_idx, block in enumerate(cache):
        assert not (block["k_coarse"] == STALE).any()
        assert not (block["v_coarse"] == STALE).any()
        assert block["k_coarse"].abs().sum() > 0

        loaded_k = block["k"][:, :installed]
        loaded_v = block["v"][:, :installed]
        expected_k, expected_v = HSA_MODULES[block_idx].compress_kv_cache(loaded_k, loaded_v)
        width = expected_k.shape[1]
        assert torch.allclose(block["k_coarse"][:, :width], expected_k, atol=1e-4)
        assert torch.allclose(block["v_coarse"][:, :width], expected_v, atol=1e-4)
        assert (block["k_coarse"][:, width:] == 0).all()

        other = HSA_MODULES[(block_idx + 1) % BLOCKS]
        other_k, _ = other.compress_kv_cache(loaded_k, loaded_v)
        assert not torch.allclose(block["k_coarse"][:, :width], other_k, atol=1e-4)


def test_apply_requires_hsa_modules_when_a_compressed_tier_is_present():
    bank = _seed_chunks(_bank(topk_pages=1))
    cache = _make_cache(stale=True)
    try:
        bank.apply(cache, [(0, 0)], current_start_frame=30, temporal_freqs=FREQS)
    except ValueError:
        pass
    else:
        raise AssertionError("expected the missing hsa_modules to be rejected")


def test_apply_works_without_a_compressed_tier():
    bank = _seed_chunks(_bank(topk_pages=1))
    cache = _make_cache(with_coarse=False)
    installed = bank.apply(cache, [(0, 0)], current_start_frame=30, temporal_freqs=FREQS)
    assert installed == PAGE_TOKENS
    assert all("k_coarse" not in block for block in cache)


def test_pages_survive_an_nvme_round_trip():
    tmp = tempfile.mkdtemp()
    try:
        bank = _seed_chunks(
            _bank(topk_pages=1, nvme_enabled=True, nvme_dir=tmp, cpu_max_chunks=1), count=2
        )
        assert bank.tier.location(0) == "nvme"

        cache = _make_cache()
        installed = bank.apply(cache, [(0, 0)], current_start_frame=30, temporal_freqs=FREQS,
                               hsa_modules=HSA_MODULES)
        assert installed == PAGE_TOKENS
        assert bank.tier.stats()["n_loads"] >= 1
        assert torch.allclose(cache[0]["k"][0, 0, 0], torch.full((HEAD_DIM,), 1000.0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unresolvable_pages_are_skipped():
    bank = _seed_chunks(_bank(topk_pages=1))
    cache = _make_cache()
    installed = bank.apply(
        cache, [(99, 0), (0, 99), (0, 0)],
        current_start_frame=30, temporal_freqs=FREQS, hsa_modules=HSA_MODULES,
    )
    assert installed == PAGE_TOKENS


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
