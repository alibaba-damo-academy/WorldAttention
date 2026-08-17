"""Training helpers for the HSA parameters."""
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
    "configure_hsa_runtime",
    "collect_distill_losses",
    "reduce_distill_losses",
]

_HSA_ATTR = "hsa_attention"


def init_hsa_parameters(model: nn.Module) -> int:
    """Initialize the HSA parameters of a model that is introducing HSA for the first time."""
    modules = [m for m in model.modules() if isinstance(m, HSAAttention)]
    if not modules:
        return 0

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
    """Set ``requires_grad`` for an HSA training stage."""
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


def configure_hsa_runtime(model: nn.Module, config) -> dict:
    """Apply the routing options a stage runs with to every HSA module."""
    backend = str(getattr(config, "hsa_backend", "auto"))
    kv_granularity = str(getattr(config, "hsa_kv_granularity", "mixed")).lower()
    tau_min = getattr(config, "hsa_tau_min", None)
    tau_max = getattr(config, "hsa_tau_max", None)
    budget_floor = getattr(config, "hsa_budget_floor", None)
    budget_cap = getattr(config, "hsa_budget_cap", None)
    frame_seq_length = int(getattr(config, "frame_seq_length", 1560))
    chunk_frames = int(getattr(config, "num_frame_per_block", 0) or 0)

    applied = 0
    for module in model.modules():
        if not isinstance(module, HSAAttention):
            continue
        module.backend = backend
        module.kv_granularity = kv_granularity
        module.current_chunk_tokens = chunk_frames * frame_seq_length
        if tau_min is not None:
            module.tau_min = float(tau_min)
        if tau_max is not None:
            module.tau_max = float(tau_max)
        module.budget_floor = None if budget_floor is None else float(budget_floor)
        module.budget_cap = None if budget_cap is None else float(budget_cap)
        module.routing_reuse = bool(getattr(config, "hsa_routing_reuse", False))
        applied += 1

    return {
        "modules": applied,
        "backend": backend,
        "kv_granularity": kv_granularity,
        "current_chunk_tokens": chunk_frames * frame_seq_length,
        "tau": (tau_min, tau_max),
        "budget": (budget_floor, budget_cap),
        "routing_reuse": bool(getattr(config, "hsa_routing_reuse", False)),
    }


@contextmanager


def collect_distill_losses():
    """Collect per-layer HSA-to-dense losses for the forward passes inside the block."""
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
