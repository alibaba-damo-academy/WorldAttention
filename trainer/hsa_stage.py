"""Training helpers for the HSA parameters.

HSA introduces three parameter groups per attention layer: the two low-rank sequence projections and
the fusion gate. They are trained in two steps.

**Warmup.** The base model is frozen and only the HSA parameters train, against the per-layer
attention self-distillation signal from :mod:`worldattention.hsa.distill`. This is a dense,
every-step target, so the gate opens and the projections move away from their pooling
initialization quickly and cheaply.

**Tune.** HSA stays active for the final distillation of the generator, so the model is distilled
under the attention it will use at inference and there is no dense-to-sparse train/test mismatch.
The HSA parameters remain trainable on top of whatever adapter the host trainer uses, and the
self-distillation term is kept as a small auxiliary regularizer.

Both steps need HSA enabled on the rollout path, since HSA only replaces attention where a KV cache
is in use.
"""
from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn

from wan.modules.hsa import HSAAttention
from wan.modules.hsa import distill

__all__ = [
    "hsa_named_parameters",
    "init_hsa_parameters",
    "configure_hsa_trainable",
    "set_hsa_backend",
    "collect_distill_losses",
    "reduce_distill_losses",
]

_HSA_ATTR = "hsa_attention"


def init_hsa_parameters(model: nn.Module) -> int:
    """Initialize the HSA parameters of a model that is introducing HSA for the first time.

    Only training needs this: a dense base checkpoint has no HSA parameters, and the module
    allocates them as zeros, which is a degenerate starting point. A zero key projection makes the
    compressed keys zero, so the linear branch outputs zero and no gradient reaches ``k_proj_mat``
    at all. Each projection row is therefore initialized to average one block of input tokens, which
    reproduces the pooled representation the sparse branch routes on and leaves the projection free
    to move away from it. The fusion gate starts at zero weight and zero bias, i.e. an even mix of
    the two branches with the sigmoid in its steepest region.

    Returns the number of HSA modules initialized.
    """
    modules = [m for m in model.modules() if isinstance(m, HSAAttention)]
    if not modules:
        return 0

    # After FSDP wrapping the projections are flat shards rather than [rank, seq_len], so gather the
    # full parameters before writing to them.
    sharded = any(p.dim() != 2 for m in modules for p in (m.k_proj_mat, m.v_proj_mat))
    if sharded:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        gather = FSDP.summon_full_params(model, writeback=True)
    else:
        gather = contextlib.nullcontext()

    with gather:
        for module in modules:
            with torch.no_grad():
                for projection in (module.k_proj_mat, module.v_proj_mat):
                    projection.zero_()
                    rank, seq_len = projection.shape
                    for row in range(rank):
                        start = row * module.block_size
                        end = min((row + 1) * module.block_size, seq_len)
                        if end > start:
                            projection[row, start:end] = 1.0 / float(end - start)
                module.gate_lin.weight.zero_()
                module.gate_lin.bias.zero_()
    return len(modules)


def hsa_named_parameters(model: nn.Module, attr_name: str = _HSA_ATTR) -> Iterator[Tuple[str, nn.Parameter]]:
    """Yield the ``(name, parameter)`` pairs that belong to HSA submodules."""
    for name, param in model.named_parameters():
        if attr_name in name:
            yield name, param


def configure_hsa_trainable(
    model: nn.Module,
    *,
    hsa_only: bool = False,
    desaturate_gate: Optional[bool] = None,
    attr_name: str = _HSA_ATTR,
) -> dict:
    """Set ``requires_grad`` for an HSA training stage.

    With ``hsa_only`` the whole model is frozen except the HSA parameters, which is the warmup
    configuration. Otherwise the existing trainable set is left alone and the HSA parameters are
    additionally unfrozen, which is what a tune on top of a frozen base plus adapters needs: an
    adapter library freezes every non-adapter parameter, which would otherwise leave the gate and
    the projections frozen for the whole stage.

    ``desaturate_gate`` resets the fusion gate bias to zero. The bias is initialized deeply negative
    so an untrained gate keeps the sparse branch dominant, but that also puts the sigmoid in a
    region where its derivative is ~1e-5 and the gate cannot move under gradient descent. Resetting
    it to zero puts the gate at 0.5 with a healthy gradient. Defaults to the value of ``hsa_only``,
    so a warmup desaturates and a resumed or already-trained stage does not.

    Returns counts of trainable HSA parameters, frozen parameters and reset gate biases.
    """
    if desaturate_gate is None:
        desaturate_gate = hsa_only

    n_hsa, n_frozen, n_gate = 0, 0, 0
    for name, param in model.named_parameters():
        is_hsa = attr_name in name
        if hsa_only:
            param.requires_grad_(is_hsa)
            if is_hsa:
                n_hsa += param.numel()
            else:
                n_frozen += param.numel()
        elif is_hsa and not param.requires_grad:
            param.requires_grad_(True)
            n_hsa += param.numel()

        if desaturate_gate and name.endswith("gate_lin.bias"):
            with torch.no_grad():
                param.zero_()
            n_gate += 1

    return {"hsa_trainable": n_hsa, "frozen": n_frozen, "gate_biases_reset": n_gate}


def set_hsa_backend(model: nn.Module, backend: str = "auto") -> int:
    """Point every HSA module in ``model`` at a sparse-branch backend. Returns how many were set."""
    count = 0
    for module in model.modules():
        if isinstance(module, HSAAttention):
            module.backend = backend
            count += 1
    return count


@contextmanager
def collect_distill_losses():
    """Collect per-layer HSA-to-dense losses for the forward passes inside the block.

    The collected terms are yielded as a list once the block exits::

        with collect_distill_losses() as losses:
            chunk = rollout_one_chunk(requires_grad=True)
        aux = reduce_distill_losses(losses)
    """
    collected: list = []
    distill.enable_distill(True)
    try:
        yield collected
    finally:
        collected.extend(distill.pop_distill_losses())
        distill.enable_distill(False)


def reduce_distill_losses(losses, device=None) -> torch.Tensor:
    """Mean of the collected per-layer losses, or a zero scalar when nothing was collected."""
    if not losses:
        return torch.zeros([], device=device, dtype=torch.float32)
    return torch.stack([loss.float() for loss in losses]).mean()
