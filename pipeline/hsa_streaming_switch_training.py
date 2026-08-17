# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Streaming prompt-switch rollout with HSA as the active attention."""
import torch.distributed as dist

from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline


class HSAStreamingSwitchTrainingPipeline(StreamingSwitchTrainingPipeline):
    """Streaming switch training pipeline with the HSA KV-cache backend enabled."""

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
                f"(sparse branch: {self.hsa_backend})."
            )
