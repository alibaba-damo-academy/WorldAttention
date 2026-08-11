"""Temporal re-rotation helpers, checked against a full rotary-embedding reference. CPU only."""
import sys

import torch

from pipeline.hkv.rope import (
    derope_temporal,
    shift_temporal_rope,
    temporal_band,
    temporal_complex_dims,
)

HEAD_DIM = 24
HEADS = 2
FRAMES, LAT_H, LAT_W = 4, 2, 3
FRAME_TOKENS = LAT_H * LAT_W
MAX_POSITIONS = 64


def _freq_table(dim_pairs: int) -> torch.Tensor:
    """``exp(i * p * theta_j)`` table, the layout the cache's rotary embedding is built from."""
    positions = torch.arange(MAX_POSITIONS, dtype=torch.float64)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim_pairs, dtype=torch.float64) / dim_pairs))
    return torch.polar(torch.ones(MAX_POSITIONS, dim_pairs, dtype=torch.float64),
                       torch.outer(positions, inv_freq))


def _reference_rope(x: torch.Tensor, freqs: torch.Tensor, start_frame: int,
                    temporal_at_zero: bool = False) -> torch.Tensor:
    """Apply the (temporal, height, width) rotary bands to ``[tokens, H, D]``."""
    tokens, heads, head_dim = x.shape
    pairs = head_dim // 2
    bands = freqs.split([pairs - 2 * (pairs // 3), pairs // 3, pairs // 3], dim=1)

    temporal_positions = (
        torch.zeros(FRAMES, dtype=torch.long) if temporal_at_zero
        else torch.arange(start_frame, start_frame + FRAMES)
    )
    per_position = torch.cat([
        bands[0][temporal_positions].view(FRAMES, 1, 1, -1).expand(FRAMES, LAT_H, LAT_W, -1),
        bands[1][:LAT_H].view(1, LAT_H, 1, -1).expand(FRAMES, LAT_H, LAT_W, -1),
        bands[2][:LAT_W].view(1, 1, LAT_W, -1).expand(FRAMES, LAT_H, LAT_W, -1),
    ], dim=-1).reshape(tokens, 1, -1)

    complex_x = torch.view_as_complex(x.to(torch.float64).reshape(tokens, heads, pairs, 2))
    return torch.view_as_real(complex_x * per_position).flatten(2).to(x.dtype)


def _setup():
    torch.manual_seed(0)
    tokens = FRAMES * FRAME_TOKENS
    x = torch.randn(tokens, HEADS, HEAD_DIM, dtype=torch.float32)
    freqs = _freq_table(HEAD_DIM // 2)
    return x, freqs, temporal_band(freqs, HEAD_DIM)


def test_temporal_band_width_matches_the_rope_split():
    pairs = HEAD_DIM // 2
    assert temporal_complex_dims(HEAD_DIM) == pairs - 2 * (pairs // 3)
    _, freqs, band = _setup()
    assert band.shape == (MAX_POSITIONS, temporal_complex_dims(HEAD_DIM))
    assert torch.equal(band, freqs[:, : temporal_complex_dims(HEAD_DIM)])


def test_shifting_a_page_equals_encoding_it_at_the_new_position():
    x, freqs, band = _setup()
    orig_frame, new_frame = 3, 11

    encoded = _reference_rope(x, freqs, start_frame=orig_frame)
    shifted = shift_temporal_rope(encoded.unsqueeze(0), band, new_frame, orig_frame).squeeze(0)
    expected = _reference_rope(x, freqs, start_frame=new_frame)

    assert torch.allclose(shifted, expected, atol=1e-4)


def test_shifting_by_zero_is_a_no_op():
    x, freqs, band = _setup()
    encoded = _reference_rope(x, freqs, start_frame=5)
    same = shift_temporal_rope(encoded.unsqueeze(0), band, 5, 5).squeeze(0)
    assert torch.allclose(same, encoded, atol=1e-5)


def test_shifting_is_invertible():
    x, freqs, band = _setup()
    encoded = _reference_rope(x, freqs, start_frame=2)
    there = shift_temporal_rope(encoded.unsqueeze(0), band, 9, 2)
    back = shift_temporal_rope(there, band, 2, 9).squeeze(0)
    assert torch.allclose(back, encoded, atol=1e-4)


def test_deroping_removes_only_the_temporal_band():
    x, freqs, band = _setup()
    start_frame = 6
    encoded = _reference_rope(x, freqs, start_frame=start_frame)

    tokens = FRAMES * FRAME_TOKENS
    frame_ids = start_frame + torch.arange(tokens) // FRAME_TOKENS
    deroped = derope_temporal(encoded.unsqueeze(0), band, frame_ids).squeeze(0)

    spatial_only = _reference_rope(x, freqs, start_frame=0, temporal_at_zero=True)
    assert torch.allclose(deroped, spatial_only, atol=1e-4)


def test_deroping_is_position_independent():
    """Keys from different frames derope to the same content vector."""
    x, freqs, band = _setup()
    tokens = FRAMES * FRAME_TOKENS
    results = []
    for start_frame in (0, 7, 21):
        encoded = _reference_rope(x, freqs, start_frame=start_frame)
        frame_ids = start_frame + torch.arange(tokens) // FRAME_TOKENS
        results.append(derope_temporal(encoded.unsqueeze(0), band, frame_ids).squeeze(0))
    for other in results[1:]:
        assert torch.allclose(results[0], other, atol=1e-4)


def test_positions_beyond_the_table_are_clamped():
    x, _, band = _setup()
    out = shift_temporal_rope(x.unsqueeze(0), band, MAX_POSITIONS + 100, 0)
    assert torch.isfinite(out).all()


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
