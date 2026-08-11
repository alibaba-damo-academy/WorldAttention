"""HSA training-stage helpers: trainable sets, backend selection, distillation collection."""
import sys

import torch
import torch.nn as nn

from wan.modules.hsa import HSAAttention
from wan.modules.hsa import distill
from trainer.hsa_stage import (
    collect_distill_losses,
    configure_hsa_trainable,
    hsa_named_parameters,
    init_hsa_parameters,
    reduce_distill_losses,
    set_hsa_backend,
)


def _trained_gate(model, bias=-12.0):
    """Stand in for a checkpoint whose fusion gate sits far from zero."""
    with torch.no_grad():
        for block in model.blocks:
            block.hsa_attention.gate_lin.bias.fill_(bias)
    return model

HEADS, HEAD_DIM, SEGMENT = 2, 16, 64


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HEADS * HEAD_DIM, HEADS * HEAD_DIM)
        self.hsa_attention = HSAAttention(
            num_heads=HEADS, head_dim=HEAD_DIM, proj_segment_len=SEGMENT, backend="torch",
        )

    def forward(self, x):
        tokens = self.q(x).view(x.shape[0], x.shape[1], HEADS, HEAD_DIM)
        return self.hsa_attention(tokens, tokens, tokens).flatten(2)


class _Model(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList(_Attention() for _ in range(depth))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _inputs():
    torch.manual_seed(0)
    return torch.randn(1, SEGMENT, HEADS * HEAD_DIM)


def test_hsa_named_parameters_finds_only_hsa_tensors():
    model = _Model()
    names = [name for name, _ in hsa_named_parameters(model)]
    assert len(names) == 2 * 4      # two projections plus gate weight and bias, per block
    assert all("hsa_attention" in name for name in names)
    assert not any(name.endswith("q.weight") for name in names)


def test_warmup_freezes_everything_except_hsa():
    model = _Model()
    stats = configure_hsa_trainable(model, hsa_only=True)

    for name, param in model.named_parameters():
        assert param.requires_grad == ("hsa_attention" in name), name
    assert stats["hsa_trainable"] > 0
    assert stats["frozen"] > 0
    assert stats["gate_biases_reset"] == 2


def test_warmup_desaturates_the_fusion_gate():
    model = _trained_gate(_Model())
    assert all(
        block.hsa_attention.gate_lin.bias.abs().max().item() > 1.0 for block in model.blocks
    )
    configure_hsa_trainable(model, hsa_only=True)
    for block in model.blocks:
        assert torch.equal(
            block.hsa_attention.gate_lin.bias, torch.zeros_like(block.hsa_attention.gate_lin.bias)
        )


def test_tune_unfreezes_hsa_on_top_of_a_frozen_base():
    model = _trained_gate(_Model())
    model.requires_grad_(False)

    stats = configure_hsa_trainable(model, hsa_only=False)

    for name, param in model.named_parameters():
        assert param.requires_grad == ("hsa_attention" in name), name
    assert stats["frozen"] == 0
    assert stats["gate_biases_reset"] == 0
    # A tune keeps the trained gate rather than resetting it.
    assert all(
        block.hsa_attention.gate_lin.bias.abs().max().item() > 1.0 for block in model.blocks
    )


def test_tune_leaves_an_existing_trainable_set_alone():
    model = _Model()
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith("q.weight"))

    configure_hsa_trainable(model, hsa_only=False)

    assert model.blocks[0].q.weight.requires_grad
    assert not model.blocks[0].q.bias.requires_grad
    assert model.blocks[0].hsa_attention.k_proj_mat.requires_grad


def test_gate_desaturation_can_be_requested_independently():
    model = _Model()
    stats = configure_hsa_trainable(model, hsa_only=False, desaturate_gate=True)
    assert stats["gate_biases_reset"] == 2
    assert torch.equal(
        model.blocks[0].hsa_attention.gate_lin.bias,
        torch.zeros_like(model.blocks[0].hsa_attention.gate_lin.bias),
    )


def test_initialization_reaches_every_module():
    model = _Model(depth=3)
    assert init_hsa_parameters(model) == 3
    for block in model.blocks:
        assert torch.count_nonzero(block.hsa_attention.k_proj_mat) > 0
        assert torch.count_nonzero(block.hsa_attention.gate_lin.bias) == 0


def test_set_hsa_backend_reaches_every_module():
    model = _Model(depth=3)
    assert set_hsa_backend(model, "torch") == 3
    assert all(block.hsa_attention.backend == "torch" for block in model.blocks)


def test_distillation_collects_one_loss_per_layer():
    model = _Model(depth=2)
    init_hsa_parameters(model)
    configure_hsa_trainable(model, hsa_only=True)

    with collect_distill_losses() as losses:
        model(_inputs())

    assert len(losses) == 2
    assert not distill.distill_enabled()

    loss = reduce_distill_losses(losses)
    assert loss.requires_grad
    loss.backward()
    assert model.blocks[0].hsa_attention.k_proj_mat.grad is not None
    assert model.blocks[0].hsa_attention.gate_lin.bias.grad.abs().sum().item() > 0
    assert model.blocks[0].q.weight.grad is None


def test_collection_stops_on_an_exception():
    model = _Model()

    class _Boom(RuntimeError):
        pass

    try:
        with collect_distill_losses():
            model(_inputs())
            raise _Boom
    except _Boom:
        pass

    assert not distill.distill_enabled()
    assert distill.pop_distill_losses() == []


def test_reducing_an_empty_collection_gives_zero():
    loss = reduce_distill_losses([])
    assert loss.item() == 0.0
    assert not loss.requires_grad


def test_no_collection_outside_the_context():
    model = _Model()
    model(_inputs())
    assert distill.pop_distill_losses() == []


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
