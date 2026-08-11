# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Streaming prompt-switch rollout with HSA as the active attention.

HSA replaces attention only where a KV cache is in use, so the rollout that feeds the distillation
objective has to run with the HSA backend installed. Enabling it here means the final distillation
sees the same attention the model will use at inference, rather than distilling under dense
attention and swapping in sparse attention afterwards.
"""
import torch.distributed as dist

from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline
from wan.modules.hsa import resolve_backend


class HSAStreamingSwitchTrainingPipeline(StreamingSwitchTrainingPipeline):
    """Streaming switch training pipeline with the HSA KV-cache backend enabled.

    Args:
        enable_hsa: install HSA on the rollout path.
        hsa_backend: sparse-branch backend, ``"auto"`` / ``"triton"`` / ``"torch"``. The gradient
            path resolves its own backend inside the attention module; this attribute governs the
            no-grad rollout steps.
    """

    def __init__(self, *args, enable_hsa: bool = True, hsa_backend: str = "auto", **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_hsa = bool(enable_hsa)
        self.hsa_backend = str(hsa_backend or "auto").lower()
        if self.enable_hsa:
            self._enable_hsa_kv_cache_backend()

    def _resolve_generator_model(self):
        """Unwrap the underlying model from sharding and adapter wrappers."""
        module = self.generator
        if hasattr(module, "module"):
            module = module.module
        if hasattr(module, "_fsdp_wrapped_module"):
            module = module._fsdp_wrapped_module
        if hasattr(module, "model"):
            return module.model
        return getattr(self.generator, "model", None)

    def _enable_hsa_kv_cache_backend(self):
        generator_model = self._resolve_generator_model()
        is_main = not dist.is_initialized() or dist.get_rank() == 0

        if generator_model is None:
            if is_main:
                print("[HSA-Train] Could not resolve the generator model; HSA backend not enabled.")
            return

        enabled_count = 0
        for module in generator_model.modules():
            if not hasattr(module, "use_hsa_kv_cache"):
                continue
            module.use_hsa_kv_cache = True
            module.set_kv_cache_attn_backend("hsa", hsa_backend=self.hsa_backend)
            enabled_count += 1

        if is_main:
            print(
                f"[HSA-Train] Enabled the HSA KV-cache backend on {enabled_count} attention modules "
                f"(sparse branch: {self.hsa_backend} -> {resolve_backend(self.hsa_backend)})."
            )
