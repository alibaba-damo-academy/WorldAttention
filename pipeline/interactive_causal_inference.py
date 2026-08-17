# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
"""Interactive long-video generation whose only memory is a hierarchical KV cache."""
from typing import List, Optional

import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pipeline.causal_inference import CausalInferencePipeline
from pipeline.hkv import HKVBlockCycle
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class InteractiveCausalInferencePipeline(HKVBlockCycle, CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        self.recache_after_switch = bool(getattr(args, "recache_after_switch", False))

        cfg = getattr(args, "hier_kv", None)
        self.hier_kv_enabled = self.hkv_configure(
            cfg, model=self.generator.model,
            num_output_frames=int(getattr(args, "num_output_frames", 0)),
        )
        if self.hier_kv_enabled and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[HKV] page = block = {self.num_frame_per_block} frames; region = "
                  f"{self.hkv.topk_pages} retrieved pages + pinned newest = {self.hkv_region_frames} frames "
                  f"(newest pinned: {self.hkv.pin_last_page}, resident bonus {self.hkv.resident_bonus}, "
                  f"index layers {self.hkv.index_layers}); cache = {self.local_attn_size} frames; "
                  f"recache pinned page on switch: {self.hkv_recache_pinned_on_switch}")

    def _reset_crossattn_cache(self):
        for cache in self.crossattn_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["is_init"] = False

    def _recache_after_switch(self, output, current_start_frame, conditional_dict):
        """Baseline switch handling: replay recent frames through the model to rebuild the cache."""
        if not self.global_sink:
            for cache in self.kv_cache1:
                cache["k"].zero_()
                cache["v"].zero_()
                if "k_coarse" in cache:
                    cache["k_coarse"].zero_()
                    cache["v_coarse"].zero_()

        self._reset_crossattn_cache()
        if current_start_frame == 0 or not self.recache_after_switch:
            return

        num_recache_frames = (
            current_start_frame if self.local_attn_size == -1
            else min(self.local_attn_size, current_start_frame)
        )
        recache_start_frame = current_start_frame - num_recache_frames
        frames = output[:, recache_start_frame:current_start_frame]
        if frames.device.type == "cpu":
            frames = frames.to(next(self.generator.parameters()).device)

        self.generator.model.block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=frames.device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )
        context_timestep = torch.ones(
            [frames.shape[0], num_recache_frames], device=frames.device, dtype=torch.int64
        ) * self.args.context_noise

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        self._reset_crossattn_cache()

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video, switching prompts at the given frame indices."""
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "switch_frame_indices must have one entry fewer than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        self._denoise_seconds = 0.0
        cond_list = [self.text_encoder(text_prompts=prompts) for prompts in text_prompts_list]

        if low_memory:
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=get_cuda_free_memory_gb(gpu) + 5,
            )

        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        if self.hkv is not None:
            self.hkv_reset()

        kv_cache_size = (
            int(self.local_attn_size) * self.frame_seq_length if self.local_attn_size not in (None, -1)
            else num_output_frames * self.frame_seq_length
        )
        self._initialize_kv_cache(
            batch_size, dtype=noise.dtype, device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device,
        )
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[interactive] {num_blocks} blocks, kv_cache_size={kv_cache_size} tokens, "
                f"hierarchical KV cache {'on' if self.hkv is not None else 'off'}"
            )

        current_start_frame = 0
        segment_idx = 0
        next_switch_pos = switch_frame_indices[0] if switch_frame_indices else None
        previous_latent = None
        staged_region = None

        for _ in range(num_blocks):
            current_num_frames = self.num_frame_per_block

            switching = False
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                switching = True
                self._reset_crossattn_cache()
                if self.hkv is None:
                    self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])

                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices) else None
                )
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[interactive] segment {segment_idx} at frame {current_start_frame}")

            conditional_dict = cond_list[segment_idx]

            if self.hkv is not None:
                self.hkv_before_block(
                    self.kv_cache1, conditional_dict, current_start_frame,
                    staged=staged_region, switching=switching, last_latent=previous_latent,
                    generator=self.generator, crossattn_cache=self.crossattn_cache,
                    context_noise=int(self.args.context_noise),
                )
                staged_region = None
            else:
                for hsa_module in getattr(self, "_hsa_attention_modules", []):
                    hsa_module.stable_kv_tokens = 0
                    hsa_module.begin_decode_block()

            next_start_frame = current_start_frame + current_num_frames
            if self.hkv is not None and len(self.hkv) > 0 and next_start_frame < num_output_frames:
                next_segment = segment_idx
                if next_switch_pos is not None and next_start_frame >= next_switch_pos:
                    next_segment = min(segment_idx + 1, len(cond_list) - 1)
                staged_region = self.hkv_stage(self.kv_cache1, cond_list[next_segment], next_start_frame)

            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            torch.cuda.current_stream().synchronize()
            _denoise_t0 = time.perf_counter()
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.int64
                ) * current_timestep

                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])

            output[:, current_start_frame:current_start_frame + current_num_frames] = (
                denoised_pred.to(output.device)
            )

            with self.hkv_capture_qbar():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=torch.ones_like(timestep) * self.args.context_noise,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            torch.cuda.current_stream().synchronize()
            self._denoise_seconds += time.perf_counter() - _denoise_t0

            if self.hkv is not None:
                self.hkv_after_block(self.kv_cache1, conditional_dict, current_start_frame)

            previous_latent = denoised_pred
            current_start_frame += current_num_frames

        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pixel_frames = 4 * num_output_frames - 3
            print(f"[timing] denoise-only: {self._denoise_seconds:.2f}s for {num_output_frames} "
                  f"latents ({pixel_frames} pixel frames) = "
                  f"{pixel_frames / max(self._denoise_seconds, 1e-9):.1f} FPS")
            if self.hkv is not None:
                print(f"[HKV] retrieval summary: {self.hkv_stats_summary()}")
        if return_latents:
            return video, output
        return video
