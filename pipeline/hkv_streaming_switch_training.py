# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Streaming prompt-switch rollout with HSA active and the hierarchical KV cache as its memory."""
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from pipeline.hkv import HKVBlockCycle
from pipeline.hsa_streaming_switch_training import HSAStreamingSwitchTrainingPipeline


class HKVStreamingSwitchTrainingPipeline(HKVBlockCycle, HSAStreamingSwitchTrainingPipeline):
    """HSA streaming switch rollout whose KV context is built by HKV, exactly as at inference."""

    def __init__(self, *args, hier_kv=None, num_output_frames: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        model = self._resolve_generator_model()
        self.hkv_configured = False
        if model is not None:
            self.hkv_configured = self.hkv_configure(
                hier_kv, model=model, num_output_frames=int(num_output_frames or 0))
        if self.hkv_configured:
            self.kv_cache_size = int(self.local_attn_size) * self.frame_seq_length
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[HKV-Train] page = block = {self.num_frame_per_block} frames; region = "
                      f"{self.hkv.topk_pages} retrieved pages + pinned newest = {self.hkv_region_frames} frames "
                      f"(newest pinned: {self.hkv.pin_last_page}, "
                      f"resident bonus {self.hkv.resident_bonus}, index layers {self.hkv.index_layers}); "
                      f"cache = {self.local_attn_size} frames; recache pinned page on switch: "
                      f"{self.hkv_recache_pinned_on_switch}")

    def clear_kv_cache(self):
        super().clear_kv_cache()
        if self.hkv_configured:
            self.hkv_reset()

    def _reset_crossattn_cache(self):
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        switch_frame_index: Optional[int] = None,
        switch_conditional_dict: Optional[dict] = None,
        switch_recache_frames: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        if not self.hkv_configured:
            return super().generate_chunk_with_cache(
                noise=noise, conditional_dict=conditional_dict,
                current_start_frame=current_start_frame, requires_grad=requires_grad,
                switch_frame_index=switch_frame_index,
                switch_conditional_dict=switch_conditional_dict,
                switch_recache_frames=switch_recache_frames,
                return_sim_step=return_sim_step,
            )

        batch_size, chunk_frames = noise.shape[:2]
        block_frames = self.num_frame_per_block
        assert chunk_frames % block_frames == 0
        num_blocks = chunk_frames // block_frames
        output = torch.zeros_like(noise)

        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_blocks, num_denoising_steps, device=noise.device)

        has_switch = switch_conditional_dict is not None and switch_frame_index is not None
        if not requires_grad:
            start_gradient_frame_index = chunk_frames
        elif has_switch:
            start_gradient_frame_index = switch_frame_index
        else:
            start_gradient_frame_index = 0

        self.generator.model.local_attn_size = int(self.local_attn_size)
        self._set_all_modules_max_attention_size(int(self.local_attn_size))

        cond_in_use = conditional_dict
        using_second = False
        local_start = 0
        for block_index in range(num_blocks):
            abs_start = current_start_frame + local_start

            switching = False
            if has_switch and not using_second and local_start >= switch_frame_index:
                cond_in_use = switch_conditional_dict
                using_second = True
                switching = True
                self._reset_crossattn_cache()

            last_latent = None
            if switching:
                if local_start >= block_frames:
                    last_latent = output[:, local_start - block_frames:local_start].detach()
                elif switch_recache_frames is not None and switch_recache_frames.shape[1] >= block_frames:
                    last_latent = switch_recache_frames[:, -block_frames:].detach()

            self.hkv_before_block(
                self.kv_cache1, cond_in_use, abs_start,
                switching=switching, last_latent=last_latent,
                generator=self.generator, crossattn_cache=self.crossattn_cache,
                context_noise=int(self.context_noise),
            )

            noisy_input = noise[:, local_start:local_start + block_frames]
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (
                    step_idx == exit_flags[0] if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )
                timestep = torch.ones(
                    [batch_size, block_frames], device=noise.device, dtype=torch.int64
                ) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=abs_start * self.frame_seq_length,
                        )
                        if step_idx < num_denoising_steps - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * block_frames], device=noise.device, dtype=torch.long
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    enable_grad = local_start >= start_gradient_frame_index
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=abs_start * self.frame_seq_length,
                        )
                    break

            output[:, local_start:local_start + block_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.detach().flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad(), self.hkv_capture_qbar():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=abs_start * self.frame_seq_length,
                )
            self.hkv_after_block(self.kv_cache1, cond_in_use, abs_start)

            local_start += block_frames

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == num_denoising_steps - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1
        return output, denoised_timestep_from, denoised_timestep_to
