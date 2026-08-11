"""Two-stage retrieval scoring and index construction. CPU only."""
import math
import sys

import torch

from pipeline.hkv import (
    page_key_index,
    prompt_index,
    query_index,
    score_pages,
    select_top_chunks,
    select_topk_pages,
)
from pipeline.hkv.rope import temporal_band

DIM = 32


def test_prompt_index_is_a_unit_vector():
    torch.manual_seed(0)
    index = prompt_index(torch.randn(2, 7, DIM))
    assert index.shape == (DIM,)
    assert abs(index.norm().item() - 1.0) < 1e-5


def test_prompt_index_is_invariant_to_token_order():
    torch.manual_seed(0)
    embeds = torch.randn(1, 7, DIM)
    shuffled = embeds[:, torch.randperm(7)]
    assert torch.allclose(prompt_index(embeds), prompt_index(shuffled), atol=1e-6)


def test_query_index_mean_pools_tokens():
    torch.manual_seed(0)
    q_tokens = torch.randn(2, 5, DIM)
    assert torch.allclose(query_index(q_tokens), q_tokens.float().mean(dim=1).mean(dim=0), atol=1e-6)


def test_stage1_ranks_chunks_by_cosine_similarity():
    torch.manual_seed(0)
    bank = torch.nn.functional.normalize(torch.randn(6, DIM), dim=1)
    query = bank[4].clone()
    assert select_top_chunks(query, bank, topk=1) == [4]

    ranked = select_top_chunks(query, bank, topk=6)
    scores = (bank @ query).tolist()
    assert ranked == sorted(range(6), key=lambda i: -scores[i])


def test_stage1_handles_degenerate_inputs():
    bank = torch.nn.functional.normalize(torch.randn(3, DIM), dim=1)
    assert select_top_chunks(bank[0], bank, topk=0) == []
    assert select_top_chunks(bank[0], torch.zeros(0, DIM), topk=2) == []
    assert len(select_top_chunks(bank[0], bank, topk=99)) == 3


def test_page_affinity_uses_the_sqrt_d_scale():
    torch.manual_seed(0)
    query = torch.randn(DIM)
    pages = torch.randn(5, DIM)
    expected = pages @ query / math.sqrt(DIM)
    assert torch.allclose(score_pages(query, pages), expected, atol=1e-5)


def test_stage2_pools_pages_across_candidate_chunks():
    torch.manual_seed(0)
    query = torch.randn(DIM)
    candidates = [(chunk, page, torch.randn(DIM)) for chunk in range(3) for page in range(4)]

    selected = select_topk_pages(query, candidates, topk=4)
    assert len(selected) == 4

    scores = torch.stack([c[2] for c in candidates]) @ query / math.sqrt(DIM)
    expected = torch.topk(scores, 4).indices.tolist()
    got = [candidates.index(next(c for c in candidates if c[0] == cid and c[1] == pid))
           for cid, pid, _ in selected]
    assert got == expected
    # Selection is not confined to one chunk.
    assert len({cid for cid, _, _ in selected}) > 1
    assert [s for _, _, s in selected] == sorted([s for _, _, s in selected], reverse=True)


def test_stage2_picks_the_page_whose_index_is_the_query():
    torch.manual_seed(0)
    candidates = [(chunk, page, torch.randn(DIM)) for chunk in range(3) for page in range(4)]
    target = candidates[9]
    selected = select_topk_pages(target[2].clone(), candidates, topk=1)
    assert selected[0][0] == target[0] and selected[0][1] == target[1]


def test_stage2_handles_degenerate_inputs():
    torch.manual_seed(0)
    query = torch.randn(DIM)
    assert select_topk_pages(query, [], topk=4) == []
    assert select_topk_pages(query, [(0, 0, torch.randn(DIM))], topk=0) == []
    assert len(select_topk_pages(query, [(0, 0, torch.randn(DIM))], topk=4)) == 1
    assert score_pages(query, torch.zeros(0, DIM)).numel() == 0


def test_page_key_index_is_built_in_content_space():
    """Keys stored at different absolute frames yield the same page index after de-rotation."""
    torch.manual_seed(0)
    head_dim, heads = 24, 2
    frame_tokens, frames = 6, 4
    tokens = frame_tokens * frames
    pairs = head_dim // 2
    num_positions = 64

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, pairs, dtype=torch.float64) / pairs))
    freqs = torch.polar(
        torch.ones(num_positions, pairs, dtype=torch.float64),
        torch.outer(torch.arange(num_positions, dtype=torch.float64), inv_freq),
    )
    band = temporal_band(freqs, head_dim)
    num_temporal = band.shape[1]

    content = torch.randn(1, tokens, heads, head_dim)
    page_spans = [(0, tokens // 2), (tokens // 2, tokens)]

    indices = []
    for start_frame in (0, 9):
        frame_ids = start_frame + torch.arange(tokens) // frame_tokens
        rotated = torch.view_as_complex(
            content.to(torch.float64).reshape(1, tokens, heads, pairs, 2)
        ).clone()
        factor = band[frame_ids].view(1, tokens, 1, num_temporal)
        rotated[..., :num_temporal] = rotated[..., :num_temporal] * factor
        stored = torch.view_as_real(rotated).reshape(content.shape).to(content.dtype)

        indices.append(page_key_index(
            stored, page_spans,
            frame_seq_length=frame_tokens, start_frame=start_frame, temporal_freqs=band,
        ))

    assert indices[0].shape == (2, heads * head_dim)
    assert torch.allclose(indices[0], indices[1], atol=1e-3)


def test_page_key_index_returns_none_without_pages():
    assert page_key_index(
        torch.randn(1, 8, 1, 8), [],
        frame_seq_length=4, start_frame=0, temporal_freqs=torch.ones(4, 2, dtype=torch.complex64),
    ) is None


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
