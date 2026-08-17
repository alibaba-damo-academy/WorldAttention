"""Growing coarse history: buffer mechanics and the global-memory linear branch. CPU only."""
import torch
import torch.nn as nn

from wan.modules.hsa.attention import HSAAttention, hsa_attention
from wan.modules.hsa.coarse_cache import GrowingCoarseCache

HEADS, DIM, SEG = 2, 16, 32


def test_append_view_and_geometric_growth():
    c = GrowingCoarseCache(enabled=True)
    r1 = torch.randn(1, 3, HEADS, DIM)
    c.append(r1, r1 * 2)
    assert c.rows == 3
    r2 = torch.randn(1, 5, HEADS, DIM)
    c.append(r2, r2 * 2)
    k, v = c.view()
    assert k.shape == (1, 8, HEADS, DIM)
    assert torch.equal(k[:, :3], r1) and torch.equal(k[:, 3:], r2)
    assert torch.equal(v[:, 3:], r2 * 2)


def test_reserve_keeps_contents_and_avoids_regrowth():
    c = GrowingCoarseCache(enabled=True)
    c.reserve(1, 100, HEADS, DIM, torch.device("cpu"), torch.float32)
    assert c.capacity == 100 and c.rows == 0
    r = torch.randn(1, 4, HEADS, DIM)
    c.append(r, r)
    buf_before = c._k.data_ptr()
    c.append(torch.randn(1, 90, HEADS, DIM), torch.randn(1, 90, HEADS, DIM))
    assert c._k.data_ptr() == buf_before, "reserved buffer must not reallocate"
    assert torch.equal(c.view()[0][:, :4], r)


def test_clear_keeps_allocation():
    c = GrowingCoarseCache(enabled=True)
    c.append(torch.randn(1, 4, HEADS, DIM), torch.randn(1, 4, HEADS, DIM))
    cap = c.capacity
    c.clear()
    assert c.rows == 0 and c.capacity == cap


def _module():
    torch.manual_seed(0)
    hsa = HSAAttention(num_heads=HEADS, head_dim=DIM, block_size=64, tau_min=0.35,
                       tau_max=1.0, proj_segment_len=SEG, backend="torch")
    with torch.no_grad():
        hsa.k_proj_mat.copy_(torch.randn_like(hsa.k_proj_mat) / SEG**0.5)
        hsa.v_proj_mat.copy_(torch.randn_like(hsa.v_proj_mat) / SEG**0.5)
        hsa.gate_lin.weight.copy_(torch.randn_like(hsa.gate_lin.weight) * 0.1)
    return hsa
