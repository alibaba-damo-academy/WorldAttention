"""The per-block HKV cycle: archive the finished block, install pages into the region."""
import sys

import torch

from pipeline.hkv import HKVCache
from pipeline.hkv.rope import shift_temporal_rope, temporal_band
from wan.modules.hsa import HSAAttention

FRAME_TOKENS = 4
PAGE_FRAMES = 2
PAGE_TOKENS = PAGE_FRAMES * FRAME_TOKENS
TOPK_PAGES = 2
REGION_FRAMES = TOPK_PAGES * PAGE_FRAMES
REGION_TOKENS = REGION_FRAMES * FRAME_TOKENS
BLOCK_FRAMES = 4
BLOCK_TOKENS = BLOCK_FRAMES * FRAME_TOKENS
CAPACITY = REGION_TOKENS + BLOCK_TOKENS
BLOCKS = 2
HEADS, HEAD_DIM = 1, 24
PROMPT_DIM = 8


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
        with torch.no_grad():
            hsa.k_proj_mat.fill_(0.0)
            hsa.k_proj_mat[0, :] = (block + 1) / FRAME_TOKENS
            hsa.v_proj_mat.fill_(0.0)
            hsa.v_proj_mat[0, :] = (block + 1) / FRAME_TOKENS
        modules.append(hsa)
    return modules

HSA_MODULES = _hsa_modules()
COARSE_SLOTS = HSA_MODULES[0].coarse_cache_size(CAPACITY)


def _make_cache(with_coarse=True):
    cache = []
    for _ in range(BLOCKS):
        block = {
            "k": torch.zeros(1, CAPACITY, HEADS, HEAD_DIM),
            "v": torch.zeros(1, CAPACITY, HEADS, HEAD_DIM),
            "global_end_index": torch.zeros([], dtype=torch.long),
            "local_end_index": torch.zeros([], dtype=torch.long),
        }
        if with_coarse:
            block["k_coarse"] = torch.zeros(1, COARSE_SLOTS, HEADS, HEAD_DIM)
            block["v_coarse"] = torch.zeros(1, COARSE_SLOTS, HEADS, HEAD_DIM)
        cache.append(block)
    return cache


def _write_block(cache, token_start, token_end, tag, global_end):
    """Stand in for the forward that writes a generated block into its cache span."""
    for block_idx, block in enumerate(cache):
        for token in range(token_start, token_end):
            block["k"][0, token, 0, :] = tag * 1000 + block_idx * 100 + token
            block["v"][0, token, 0, :] = -(tag * 1000 + block_idx * 100 + token)
        block["local_end_index"].fill_(token_end)
        block["global_end_index"].fill_(global_end)


def _prompt(seed):
    torch.manual_seed(seed)
    return torch.randn(1, 3, PROMPT_DIM)


def _bank(rerope=True, topk_pages=TOPK_PAGES, stage1=4):
    return HKVCache(
        frame_seq_length=FRAME_TOKENS, page_size_frames=PAGE_FRAMES,
        topk_pages=topk_pages, stage1_topk_chunks=stage1, rerope=rerope, pin_last_page=False,
    )


def _cycle(bank, cache, num_blocks, region_frames=REGION_FRAMES):
    """Run the pipeline's per-block cycle: install, write the block, archive it."""
    sink_frames = 0
    sink_sizes = []
    for block_index in range(num_blocks):
        start_frame = block_index * BLOCK_FRAMES
        if len(bank) > 0:
            pages = bank.retrieve(
                prompt_embeds=_prompt(block_index),
                query_vec=bank.chunks[-1]["page_index"][-1].clone(),
            )
            installed = bank.install_region(
                cache, pages,
                region_tokens=region_frames * FRAME_TOKENS,
                current_start_frame=start_frame,
                temporal_freqs=FREQS,
                hsa_modules=HSA_MODULES,
            )
            sink_frames = installed // FRAME_TOKENS
        sink_sizes.append(sink_frames)

        sink_tokens = sink_frames * FRAME_TOKENS
        _write_block(
            cache, sink_tokens, sink_tokens + BLOCK_TOKENS,
            tag=block_index + 1, global_end=(start_frame + BLOCK_FRAMES) * FRAME_TOKENS,
        )
        bank.store_span(
            cache, prompt_embeds=_prompt(block_index),
            token_start=sink_tokens, token_end=sink_tokens + BLOCK_TOKENS,
            start_frame=start_frame, temporal_freqs=FREQS,
        )
    return sink_sizes


def test_store_span_archives_exactly_the_block():
    bank, cache = _bank(), _make_cache()
    _write_block(cache, REGION_TOKENS, CAPACITY, tag=7, global_end=CAPACITY)

    chunk_id = bank.store_span(
        cache, prompt_embeds=_prompt(0),
        token_start=REGION_TOKENS, token_end=CAPACITY,
        start_frame=REGION_FRAMES, temporal_freqs=FREQS,
    )

    assert chunk_id == 0
    chunk = bank.chunks[0]
    assert chunk["valid_tokens"] == BLOCK_TOKENS
    assert chunk["start_frame"] == REGION_FRAMES
    assert chunk["page_spans"] == [(0, PAGE_TOKENS), (PAGE_TOKENS, BLOCK_TOKENS)]
    for page_id, (start, end) in enumerate(chunk["page_spans"]):
        page = bank.tier.get((0, page_id))
        take = end - start
        for block_idx, block in enumerate(cache):
            source = slice(REGION_TOKENS + start, REGION_TOKENS + end)
            assert torch.equal(page[block_idx, 0, :, :take], block["k"][:, source])
            assert torch.equal(page[block_idx, 1, :, :take], block["v"][:, source])


def test_store_span_skips_an_empty_span():
    bank = _bank()
    assert bank.store_span(
        _make_cache(), prompt_embeds=_prompt(0),
        token_start=REGION_TOKENS, token_end=REGION_TOKENS,
        start_frame=0, temporal_freqs=FREQS,
    ) == -1
    assert len(bank) == 0


def test_install_region_leaves_the_block_span_untouched():
    bank, cache = _bank(), _make_cache()
    _cycle(bank, cache, num_blocks=1)
    before = [block["k"][:, REGION_TOKENS:].clone() for block in cache]

    installed = bank.install_region(
        cache, bank.retrieve(prompt_embeds=_prompt(0), query_vec=bank.chunks[0]["page_index"][0]),
        region_tokens=REGION_TOKENS, current_start_frame=BLOCK_FRAMES,
        temporal_freqs=FREQS, hsa_modules=HSA_MODULES,
    )

    assert installed == REGION_TOKENS
    for block_idx, block in enumerate(cache):
        assert torch.equal(block["k"][:, REGION_TOKENS:], before[block_idx])


def test_install_region_parks_the_indices_at_the_end_of_the_region():
    """The block must land right after the pages, whether or not they filled the region."""
    bank, cache = _bank(), _make_cache()
    _cycle(bank, cache, num_blocks=1)

    installed = bank.install_region(
        cache, [(0, 0)],
        region_tokens=REGION_TOKENS, current_start_frame=BLOCK_FRAMES,
        temporal_freqs=FREQS, hsa_modules=HSA_MODULES,
    )

    assert installed == PAGE_TOKENS
    for block in cache:
        assert int(block["local_end_index"].item()) == PAGE_TOKENS
        assert int(block["global_end_index"].item()) == BLOCK_FRAMES * FRAME_TOKENS


def test_install_region_reropes_pages_to_the_slots_before_the_block():
    bank, cache = _bank(rerope=True), _make_cache()
    _cycle(bank, cache, num_blocks=1)
    banked = [bank.tier.get((0, page_id)) for page_id in range(len(bank.chunks[0]["page_spans"]))]
    chunk_start = bank.chunks[0]["start_frame"]

    current_start_frame = 12
    bank.install_region(
        cache, [(0, 0), (0, 1)],
        region_tokens=REGION_TOKENS, current_start_frame=current_start_frame,
        temporal_freqs=FREQS, hsa_modules=HSA_MODULES,
    )

    base = current_start_frame - REGION_FRAMES
    for block_idx, block in enumerate(cache):
        for page_id, (start, end) in enumerate(bank.chunks[0]["page_spans"]):
            take = end - start
            expected = shift_temporal_rope(
                banked[page_id][block_idx, 0, :, :take], FREQS,
                base + page_id * PAGE_FRAMES, chunk_start + start // FRAME_TOKENS,
            )
            assert torch.allclose(block["k"][:, start:end], expected, atol=1e-5)
            assert torch.equal(block["v"][:, start:end], banked[page_id][block_idx, 1, :, :take])


def test_install_region_reports_a_short_install():
    """One page is all the bank has, so the caller shrinks the sink instead of padding with zeros."""
    bank, cache = _bank(topk_pages=1), _make_cache()
    _cycle(bank, cache, num_blocks=1)

    installed = bank.install_region(
        cache, [(0, 0)],
        region_tokens=REGION_TOKENS, current_start_frame=BLOCK_FRAMES,
        temporal_freqs=FREQS, hsa_modules=HSA_MODULES,
    )
    assert installed == PAGE_TOKENS


def test_install_region_rebuilds_the_compressed_tier():
    bank, cache = _bank(), _make_cache()
    _cycle(bank, cache, num_blocks=1)
    for block in cache:
        block["k_coarse"].zero_()
        block["v_coarse"].zero_()

    bank.install_region(
        cache, [(0, 0), (0, 1)],
        region_tokens=REGION_TOKENS, current_start_frame=BLOCK_FRAMES,
        temporal_freqs=FREQS, hsa_modules=HSA_MODULES,
    )

    for block_idx, block in enumerate(cache):
        coarse_start, coarse_end = HSA_MODULES[block_idx].token_range_to_coarse(0, REGION_TOKENS)
        expected, _ = HSA_MODULES[block_idx].compress_kv_cache(
            block["k"][:, :REGION_TOKENS], block["v"][:, :REGION_TOKENS]
        )
        assert torch.allclose(block["k_coarse"][:, coarse_start:coarse_end], expected, atol=1e-5)


def test_install_region_needs_hsa_modules_for_a_compressed_cache():
    bank, cache = _bank(), _make_cache()
    _cycle(bank, cache, num_blocks=1)
    try:
        bank.install_region(
            cache, [(0, 0)], region_tokens=REGION_TOKENS, current_start_frame=BLOCK_FRAMES,
            temporal_freqs=FREQS, hsa_modules=None,
        )
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a compressed cache without hsa_modules")


def test_every_block_reaches_the_bank():
    """No frame is lost: with the window gone, the bank is the only thing holding history."""
    bank, cache = _bank(), _make_cache()
    sink_sizes = _cycle(bank, cache, num_blocks=4)

    assert len(bank) == 4
    assert sink_sizes == [0, REGION_FRAMES, REGION_FRAMES, REGION_FRAMES]
    assert [chunk["start_frame"] for chunk in bank.chunks] == [0, 4, 8, 12]
    for chunk in bank.chunks:
        assert chunk["valid_tokens"] == BLOCK_TOKENS
    assert sum(chunk["valid_tokens"] for chunk in bank.chunks) == 4 * BLOCK_TOKENS


def test_first_block_runs_with_no_region():
    """Nothing is banked yet, so the first block attends only to itself."""
    bank, cache = _bank(), _make_cache()
    assert _cycle(bank, cache, num_blocks=1) == [0]
    assert int(cache[0]["local_end_index"].item()) == BLOCK_TOKENS

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(dict(globals()).items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print("PASS", name)
        except Exception as err:
            failures += 1
            print("FAIL", name, "->", err)
    print("\nRESULT:", "ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
