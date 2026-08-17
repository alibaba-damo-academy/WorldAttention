"""The retrieval policy on a page-per-block bank: pinned newest page, multi-layer scoring, resident bonus, and the shared per-block cycle that inference and training both run."""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from pipeline.hkv import HKVBlockCycle, HKVCache, hkv_config_summary, score_pages_layers
from pipeline.hkv.rope import temporal_band

FRAME_TOKENS = 4
BLOCK_FRAMES = 2
BLOCK_TOKENS = BLOCK_FRAMES * FRAME_TOKENS
TOPK = 3
REGION_PAGES = TOPK + 1
REGION_FRAMES = REGION_PAGES * BLOCK_FRAMES
REGION_TOKENS = REGION_FRAMES * FRAME_TOKENS
CAPACITY = REGION_TOKENS + BLOCK_TOKENS
LAYERS = 3
HEADS, HEAD_DIM = 1, 8
DIM = HEADS * HEAD_DIM
PROMPT_DIM = 6


def _freqs():
    pairs = HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, pairs, dtype=torch.float64) / pairs))
    table = torch.polar(torch.ones(256, pairs, dtype=torch.float64),
                        torch.outer(torch.arange(256, dtype=torch.float64), inv_freq))
    return table

FREQS_TABLE = _freqs()
FREQS = temporal_band(FREQS_TABLE, HEAD_DIM)


def _make_cache():
    return [{
        "k": torch.zeros(1, CAPACITY, HEADS, HEAD_DIM),
        "v": torch.zeros(1, CAPACITY, HEADS, HEAD_DIM),
        "global_end_index": torch.zeros([], dtype=torch.long),
        "local_end_index": torch.zeros([], dtype=torch.long),
    } for _ in range(LAYERS)]


def _prompt(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, PROMPT_DIM, generator=g)


def _bank(**kw):
    args = dict(frame_seq_length=FRAME_TOKENS, page_size_frames=BLOCK_FRAMES, topk_pages=TOPK,
                stage1_topk_chunks=1000, rerope=False, pin_last_page=True, resident_bonus=0.0)
    args.update(kw)
    return HKVCache(**args)


def _write_block(cache, token_start, tag, seed, global_end):
    """A block's keys: a per-layer, per-token signature so any mix-up is visible."""
    g = torch.Generator().manual_seed(seed)
    for layer, block in enumerate(cache):
        block["k"][0, token_start:token_start + BLOCK_TOKENS, 0, :] = (
            torch.randn(BLOCK_TOKENS, HEAD_DIM, generator=g) + tag + layer)
        block["v"][0, token_start:token_start + BLOCK_TOKENS, 0, :] = -(tag + layer)
        block["local_end_index"].fill_(token_start + BLOCK_TOKENS)
        block["global_end_index"].fill_(global_end)


def _archive_blocks(bank, count):
    """Archive `count` blocks straight from a synthetic cache (no region installed)."""
    for i in range(count):
        cache = _make_cache()
        _write_block(cache, 0, tag=10 * (i + 1), seed=i, global_end=(i + 1) * BLOCK_TOKENS)
        bank.store_span(cache, prompt_embeds=_prompt(i), token_start=0, token_end=BLOCK_TOKENS,
                        start_frame=i * BLOCK_FRAMES, temporal_freqs=FREQS)
    return bank


def test_a_page_is_one_block_and_indexes_every_layer():
    bank = _archive_blocks(_bank(), 2)
    assert len(bank) == 2
    for chunk in bank.chunks:
        assert chunk["page_spans"] == [(0, BLOCK_TOKENS)]
        assert chunk["page_index"].shape == (1, LAYERS, DIM)
    assert bank.last_page() == (1, 0)
    assert bank.page_frames((1, 0)) == (BLOCK_FRAMES, 2 * BLOCK_FRAMES)


def test_index_layers_can_be_restricted():
    bank = _archive_blocks(_bank(index_layers="first"), 1)
    assert bank.chunks[0]["page_index"].shape == (1, 1, DIM)
    bank = _archive_blocks(_bank(index_layers=[0, 2]), 1)
    assert bank.chunks[0]["page_index"].shape == (1, 2, DIM)


def test_newest_page_is_always_installed():
    bank = _archive_blocks(_bank(), 6)
    query = torch.randn(LAYERS, DIM)
    for _ in range(5):
        pages = bank.retrieve(prompt_embeds=_prompt(0), query_vec=query)
        assert bank.last_page() in pages
        assert len(pages) == TOPK + 1
        assert bank.last_pinned == [bank.last_page()]
        assert pages[-1] == bank.last_page()


def test_without_a_query_only_the_newest_page_is_installed():
    bank = _archive_blocks(_bank(), 4)
    assert bank.retrieve(prompt_embeds=_prompt(0), query_vec=None) == [bank.last_page()]


def test_pinning_can_be_switched_off():
    bank = _archive_blocks(_bank(pin_last_page=False), 6)
    query = bank.chunks[0]["page_index"][0].clone()
    pages = bank.retrieve(prompt_embeds=_prompt(0), query_vec=query)
    assert len(pages) == TOPK and pages[0] == (0, 0)
    assert bank.last_pinned == []


def test_scored_slots_pick_the_pages_whose_keys_match_the_query():
    bank = _archive_blocks(_bank(), 6)
    target = (2, 0)
    query = bank.chunks[target[0]]["page_index"][0].clone()
    pages = bank.retrieve(prompt_embeds=_prompt(0), query_vec=query)
    assert target in pages and bank.last_page() in pages


def test_multilayer_scoring_averages_the_layers():
    torch.manual_seed(0)
    index = torch.randn(5, LAYERS, DIM)
    query = torch.randn(LAYERS, DIM)
    scores = score_pages_layers(query, index)
    expected = torch.stack([
        torch.softmax(index[:, l] @ query[l] / math.sqrt(DIM), dim=0) for l in range(LAYERS)
    ], dim=1).mean(dim=1)
    assert torch.allclose(scores, expected, atol=1e-6)
    assert abs(float(scores.sum()) - 1.0) < 1e-5
    single = score_pages_layers(query[0], index)
    assert torch.allclose(single, torch.softmax(index[:, 0] @ query[0] / math.sqrt(DIM), dim=0), atol=1e-6)


def test_a_layer_that_disagrees_can_change_the_ranking():
    """Layer 0 alone would pick page A; layers 1-2 both prefer page B; the average picks B."""
    index = torch.zeros(2, LAYERS, DIM)
    query = torch.zeros(LAYERS, DIM)
    query[:, 0] = 1.0
    index[0, 0, 0] = 3.0
    index[1, 1, 0] = 3.0
    index[1, 2, 0] = 3.0
    bank = _bank(pin_last_page=False, topk_pages=1)
    for i in range(2):
        cache = _make_cache()
        bank.store_span(cache, prompt_embeds=_prompt(i), token_start=0, token_end=BLOCK_TOKENS,
                        start_frame=i * BLOCK_FRAMES, temporal_freqs=FREQS)
        bank.chunks[i]["page_index"] = index[i:i + 1]
    assert bank.retrieve(prompt_embeds=_prompt(0), query_vec=query) == [(1, 0)]
    assert bank.retrieve(prompt_embeds=_prompt(0), query_vec=query[0]) == [(0, 0)]


def test_resident_bonus_keeps_an_installed_page_over_a_marginal_challenger():
    bank = _archive_blocks(_bank(pin_last_page=False, topk_pages=1), 3)
    base = torch.randn(LAYERS, DIM)
    bank.chunks[0]["page_index"] = (1.01 * base).unsqueeze(0)
    bank.chunks[1]["page_index"] = (1.02 * base).unsqueeze(0)
    bank.chunks[2]["page_index"] = (-base).unsqueeze(0)
    plain = bank.retrieve(prompt_embeds=_prompt(0), query_vec=base, previous=[(0, 0)])
    assert plain == [(1, 0)]
    bank.resident_bonus = 0.2
    sticky = bank.retrieve(prompt_embeds=_prompt(0), query_vec=base, previous=[(0, 0)])
    assert sticky == [(0, 0)]
    bank.chunks[1]["page_index"] = (10 * base).unsqueeze(0)
    assert bank.retrieve(prompt_embeds=_prompt(0), query_vec=base, previous=[(0, 0)]) == [(1, 0)]


def test_config_summary_reads_the_deployed_geometry():
    cfg = SimpleNamespace(enabled=True, page_size_frames=3, topk_pages=7, pin_last_page=True,
                          resident_bonus=0.0, index_layers="all", recache_pinned_on_switch=True)
    summary = hkv_config_summary(cfg)
    assert summary["page_size_frames"] == 3 and summary["topk_pages"] == 7
    assert summary["pin_last_page"] and summary["recache_pinned_on_switch"]


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = nn.Identity()
        self.sink_size = 0
        self.local_attn_size = -1
        self.use_hsa_kv_cache = False


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(LAYERS)])
        self.freqs = FREQS_TABLE
        self.local_attn_size = -1


class _Host(HKVBlockCycle):
    def __init__(self, cfg, local_attn_size=-1):
        self.frame_seq_length = FRAME_TOKENS
        self.num_frame_per_block = BLOCK_FRAMES
        self.local_attn_size = local_attn_size
        self.model = _Model()
        self.enabled = self.hkv_configure(cfg, model=self.model, num_output_frames=40)


def _cfg(**kw):
    base = dict(enabled=True, page_size_frames=BLOCK_FRAMES, topk_pages=TOPK, pin_last_page=True,
                resident_bonus=0.0, index_layers="all", rerope_retrieved=False,
                recache_pinned_on_switch=True, gpu_max_pages=0, cpu_max_pages=64)
    base.update(kw)
    return SimpleNamespace(**base)


def _replay(host, seed):
    """Stand in for the clean replay: a forward through every layer's norm_q under the hooks."""
    g = torch.Generator().manual_seed(seed)
    with host.hkv_capture_qbar():
        for layer in range(LAYERS):
            host.model.blocks[layer].self_attn.norm_q(torch.randn(1, BLOCK_TOKENS, DIM, generator=g) + layer)


def _run_blocks(host, cache, num_blocks, *, staged_path=False):
    """The per-block cycle exactly as a pipeline drives it; returns the pages installed per block."""
    installed = []
    staged = None
    for i in range(num_blocks):
        start = i * BLOCK_FRAMES
        frames = host.hkv_before_block(cache, {"prompt_embeds": _prompt(i)}, start, staged=staged)
        staged = None
        installed.append(list(host._hkv_installed))
        assert frames == host.hkv_sink_frames
        _write_block(cache, frames * FRAME_TOKENS, tag=10 * (i + 1), seed=i,
                     global_end=(start + BLOCK_FRAMES) * FRAME_TOKENS)
        if staged_path:
            staged = host.hkv_stage(cache, {"prompt_embeds": _prompt(i + 1)}, start + BLOCK_FRAMES)
        _replay(host, seed=100 + i)
        host.hkv_after_block(cache, {"prompt_embeds": _prompt(i)}, start)
    return installed


def test_configure_derives_the_window_from_the_geometry():
    host = _Host(_cfg())
    assert host.enabled
    assert host.local_attn_size == BLOCK_FRAMES + REGION_FRAMES
    assert host.model.local_attn_size == BLOCK_FRAMES + REGION_FRAMES
    assert all(b.self_attn.local_attn_size == BLOCK_FRAMES + REGION_FRAMES for b in host.model.blocks)


def test_configure_rejects_a_page_that_is_not_a_block():
    try:
        _Host(_cfg(page_size_frames=BLOCK_FRAMES + 1))
    except ValueError as err:
        assert "num_frame_per_block" in str(err)
    else:
        raise AssertionError("a page that is not one block must be rejected")


def test_configure_rejects_a_window_that_disagrees_with_the_geometry():
    try:
        _Host(_cfg(), local_attn_size=BLOCK_FRAMES + REGION_FRAMES + 1)
    except ValueError:
        return
    raise AssertionError("expected the mismatched window to be rejected")


def test_disabled_config_leaves_the_host_alone():
    host = _Host(_cfg(enabled=False))
    assert not host.enabled and host.hkv is None
    cache = _make_cache()
    assert host.hkv_before_block(cache, {"prompt_embeds": _prompt(0)}, 0) == 0


def test_cycle_pins_the_newest_page_and_fills_the_region_as_the_bank_grows():
    host = _Host(_cfg())
    cache = _make_cache()
    installed = _run_blocks(host, cache, 5)
    assert installed[0] == []
    assert [len(pages) for pages in installed] == [0, 1, 2, 3, 4]
    for i in range(1, 5):
        assert (i - 1, 0) in installed[i]
        assert installed[i][-1] == (i - 1, 0)
    assert len(host.hkv) == 5
    summary = host.hkv_stats_summary()
    assert summary["newest_fraction"] == 1.0
    assert summary["blocks"] == 4


def test_replay_hooks_capture_one_query_per_indexed_layer():
    host = _Host(_cfg())
    assert host._hkv_qbar is None
    _replay(host, seed=1)
    assert host._hkv_qbar.shape == (LAYERS, DIM)
    host = _Host(_cfg(index_layers="first"))
    _replay(host, seed=1)
    assert host._hkv_qbar.shape == (1, DIM)


def test_the_region_holds_the_archived_pages_verbatim():
    host = _Host(_cfg())
    cache = _make_cache()
    _run_blocks(host, cache, 5)
    frames = host.hkv_before_block(cache, {"prompt_embeds": _prompt(9)}, 5 * BLOCK_FRAMES)
    assert frames == REGION_FRAMES
    offset = 0
    for page in host._hkv_installed:
        banked = host.hkv.tier.get(page)
        for layer, block in enumerate(cache):
            assert torch.equal(block["k"][:, offset:offset + BLOCK_TOKENS], banked[layer, 0, :, :BLOCK_TOKENS])
            assert torch.equal(block["v"][:, offset:offset + BLOCK_TOKENS], banked[layer, 1, :, :BLOCK_TOKENS])
        offset += BLOCK_TOKENS
    assert int(cache[0]["local_end_index"]) == REGION_TOKENS


def test_staged_and_fresh_installs_build_the_same_region():
    """Inference stages the region during the replay; training installs it fresh. Same result."""
    fresh_host, staged_host = _Host(_cfg()), _Host(_cfg())
    fresh_cache, staged_cache = _make_cache(), _make_cache()
    fresh = _run_blocks(fresh_host, fresh_cache, 8, staged_path=False)
    staged = _run_blocks(staged_host, staged_cache, 8, staged_path=True)
    assert fresh == staged
    assert all(len(pages) == REGION_PAGES for pages in fresh[REGION_PAGES:])
    for layer in range(LAYERS):
        assert torch.equal(fresh_cache[layer]["k"], staged_cache[layer]["k"])
        assert torch.equal(fresh_cache[layer]["v"], staged_cache[layer]["v"])
    assert torch.equal(fresh_cache[0]["local_end_index"], staged_cache[0]["local_end_index"])
    assert torch.equal(fresh_cache[0]["global_end_index"], staged_cache[0]["global_end_index"])


def test_staging_scores_with_the_query_from_two_replays_ago():
    """Both paths use the same query: what the staged path can have when it runs."""
    host = _Host(_cfg())
    cache = _make_cache()
    _run_blocks(host, cache, 4)
    assert host._hkv_qbar is not None and host._hkv_qbar_prev is not None
    assert not torch.equal(host._hkv_qbar, host._hkv_qbar_prev)
    fresh = host.hkv_select({"prompt_embeds": _prompt(0)}, staging=False)
    assert host.hkv.last_page() in fresh and host.hkv.last_pinned == [host.hkv.last_page()]
    assert len(fresh) == TOPK + 1
    staged = host.hkv_select({"prompt_embeds": _prompt(0)}, staging=True)
    assert len(staged) == TOPK and host.hkv.last_pinned == []


def test_reset_empties_the_bank_and_the_region():
    host = _Host(_cfg())
    cache = _make_cache()
    _run_blocks(host, cache, 3)
    host.hkv_reset()
    assert len(host.hkv) == 0 and host.hkv_sink_frames == 0 and host._hkv_qbar is None
    assert host.hkv_before_block(cache, {"prompt_embeds": _prompt(0)}, 0) == 0


def test_switch_recomputes_the_pinned_page_in_place():
    """On a prompt switch the pinned page is re-encoded by the generator into its own slot."""
    host = _Host(_cfg())
    cache = _make_cache()
    _run_blocks(host, cache, 5)
    calls = []

    def generator(*, noisy_image_or_video, conditional_dict, timestep, kv_cache, crossattn_cache, current_start):
        calls.append(dict(frames=noisy_image_or_video.shape[1], start=current_start,
                          sink=host.hkv_sink_frames, prompt=conditional_dict["tag"]))

    latent = torch.zeros(1, BLOCK_FRAMES, 4, 2, 2)
    frames = host.hkv_before_block(
        cache, {"prompt_embeds": _prompt(7), "tag": "new"}, 5 * BLOCK_FRAMES,
        switching=True, last_latent=latent, generator=generator, crossattn_cache=[], context_noise=0)
    assert frames == REGION_FRAMES
    assert len(calls) == 1
    call = calls[0]
    assert call["prompt"] == "new" and call["frames"] == BLOCK_FRAMES
    assert call["start"] == (5 * BLOCK_FRAMES - BLOCK_FRAMES) * FRAME_TOKENS
    assert call["sink"] == REGION_FRAMES - BLOCK_FRAMES
    assert host.hkv_sink_frames == REGION_FRAMES
    assert host.hkv_stats_summary()["recaches"] == 1


def test_switch_without_recache_leaves_the_generator_alone():
    host = _Host(_cfg(recache_pinned_on_switch=False))
    cache = _make_cache()
    _run_blocks(host, cache, 3)
    calls = []
    host.hkv_before_block(
        cache, {"prompt_embeds": _prompt(7)}, 3 * BLOCK_FRAMES, switching=True,
        last_latent=torch.zeros(1, BLOCK_FRAMES, 4, 2, 2),
        generator=lambda **kw: calls.append(kw), crossattn_cache=[], context_noise=0)
    assert calls == []
