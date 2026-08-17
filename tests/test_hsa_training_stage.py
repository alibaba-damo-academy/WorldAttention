"""HSA training plumbing: runtime configuration, trainable sets, and a real gradient step. CPU."""
import types

import torch
import torch.nn as nn

from trainer.hsa_stage import (
    collect_distill_losses,
    configure_hsa_runtime,
    configure_hsa_trainable,
    init_hsa_parameters,
    reduce_distill_losses,
)
from wan.modules.hsa import HSAAttention, distill

SEG = 64


class _FakeAttn(nn.Module):
    """The HSA-facing surface of the model's self-attention, without the rest of the block."""

    def __init__(self):
        super().__init__()
        self.hsa_attention = HSAAttention(num_heads=2, head_dim=128, block_size=64,
                                          proj_segment_len=SEG, backend="torch")
        self.use_hsa_kv_cache = False
        self.kv_cache_attn_backend = "flash"

    def set_kv_cache_attn_backend(self, backend, *, hsa_backend="auto"):
        self.hsa_attention.backend = hsa_backend
        self.kv_cache_attn_backend = backend


def _generator(layers=3):
    gen = nn.Module()
    gen.blocks = nn.ModuleList([_FakeAttn() for _ in range(layers)])
    gen.backbone = nn.Linear(8, 8)
    return gen


def _config(**overrides):
    base = dict(enable_hsa=True, hsa_backend="torch", hsa_kv_granularity="mixed",
                hsa_tau_min=0.10, hsa_tau_max=0.95, hsa_budget_floor=0.08, hsa_budget_cap=0.13,
                hsa_routing_reuse=False, num_frame_per_block=3, frame_seq_length=SEG)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_runtime_options_reach_every_module():
    gen = _generator()
    applied = configure_hsa_runtime(gen, _config())
    assert applied["modules"] == 3
    for block in gen.blocks:
        hsa = block.hsa_attention
        assert hsa.kv_granularity == "mixed"
        assert hsa.budget_floor == 0.08 and hsa.budget_cap == 0.13
        assert hsa.tau_min == 0.10 and hsa.tau_max == 0.95
        assert hsa.current_chunk_tokens == 3 * SEG
        assert hsa.routing_reuse is False


def test_runtime_defaults_leave_budgets_unbounded():
    gen = _generator()
    cfg = types.SimpleNamespace(num_frame_per_block=3, frame_seq_length=SEG)
    configure_hsa_runtime(gen, cfg)
    hsa = gen.blocks[0].hsa_attention
    assert hsa.budget_floor is None and hsa.budget_cap is None
    assert hsa.kv_granularity == "mixed"


def test_warmup_trains_only_hsa_parameters():
    gen = _generator()
    init_hsa_parameters(gen)
    stats = configure_hsa_trainable(gen, hsa_only=True)
    assert stats["hsa_trainable"] > 0 and stats["frozen"] > 0
    assert stats["gate_biases_reset"] == 3
    trainable = {n for n, p in gen.named_parameters() if p.requires_grad}
    assert trainable and all("hsa_attention" in n for n in trainable)
    assert not gen.backbone.weight.requires_grad


def test_tune_unfreezes_hsa_on_top_of_a_frozen_base():
    gen = _generator()
    init_hsa_parameters(gen)
    for p in gen.parameters():
        p.requires_grad_(False)
    configure_hsa_trainable(gen, hsa_only=False)
    assert all(p.requires_grad for n, p in gen.named_parameters() if "hsa_attention" in n)
    assert not gen.backbone.weight.requires_grad
