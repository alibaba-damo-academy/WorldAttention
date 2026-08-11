# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
"""Interactive long-video generation with prompt switching backed by a hierarchical KV cache.

Generation proceeds chunk by chunk. When the driving prompt changes, the window that has just been
generated is archived into the KV bank as a chunk, and two-stage retrieval installs the ``K_p`` pages
of history most relevant to the incoming prompt. That replaces the baseline behaviour of re-running
the model over recent frames to rebuild the cache, which costs a full forward pass per switch and
still only ever sees the sliding window.

Set ``hier_kv.enabled: false`` to fall back to that re-cache baseline.
"""
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pipeline.causal_inference import CausalInferencePipeline
from pipeline.hkv import HKVCache, query_index
from pipeline.hkv.rope import temporal_band
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
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

        cfg = getattr(args, "hier_kv", None)
        self.hier_kv_enabled = bool(getattr(cfg, "enabled", False)) if cfg is not None else False
        if self.hier_kv_enabled:
            nvme = getattr(cfg, "nvme", None)
            nvme_enabled = bool(getattr(nvme, "enabled", False)) if nvme is not None else False
            nvme_dir = str(getattr(nvme, "path", "") or "") if nvme is not None else ""
            self.hkv = HKVCache(
                frame_seq_length=self.frame_seq_length,
                page_size_frames=int(getattr(cfg, "page_size_frames", 8)),
                topk_pages=int(getattr(cfg, "topk_pages", 4)),
                stage1_topk_chunks=int(getattr(cfg, "stage1_topk_chunks", 1)),
                cpu_max_chunks=int(getattr(cfg, "cpu_max_chunks", 64)),
                nvme_enabled=nvme_enabled,
                nvme_dir=nvme_dir or None,
                rerope=bool(getattr(cfg, "rerope_retrieved", True)),
            )
        else:
            self.hkv = None

    # ------------------------------------------------------------------ model-coupled accessors
    def _temporal_freqs(self) -> torch.Tensor:
        """Temporal rotary band of the model's frequency table, on the KV cache device."""
        head_dim = int(self.kv_cache1[0]["k"].shape[-1])
        return temporal_band(self.generator.model.freqs, head_dim).to(self.kv_cache1[0]["k"].device)

    def _query_index(self, noisy_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """Stage-2 query vector for the incoming chunk.

        Taken from the first block's attention query before the rotary embedding is applied, which
        is the space the page key index lives in. Returns None when the chunk is empty.
        """
        if noisy_chunk is None or noisy_chunk.numel() == 0:
            return None
        model = self.generator.model
        attn = model.blocks[0].self_attn
        with torch.no_grad():
            pooled = []
            for sample in noisy_chunk.permute(0, 2, 1, 3, 4):
                tokens = model.patch_embedding(sample.unsqueeze(0)).flatten(2).transpose(1, 2)
                pooled.append(query_index(attn.norm_q(attn.q(tokens))))
            return torch.stack(pooled, dim=0).mean(dim=0)

    def _hsa_modules(self) -> Optional[list]:
        """Per-block HSA modules, or None when the cache carries no compressed tier."""
        if "k_coarse" not in self.kv_cache1[0]:
            return None
        return [block.self_attn.hsa_attention for block in self.generator.model.blocks]

    def _reset_crossattn_cache(self):
        for cache in self.crossattn_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["is_init"] = False

    # ------------------------------------------------------------------ prompt switching
    def _switch_via_hkv(self, conditional_dict, next_conditional_dict, noisy_chunk,
                        current_start_frame: int) -> bool:
        """Archive the current window, then install the pages the new prompt needs.

        Returns False when the bank had nothing to install, so the caller can fall back to the
        re-cache baseline.
        """
        temporal_freqs = self._temporal_freqs()
        self.hkv.store(
            self.kv_cache1,
            prompt_embeds=conditional_dict["prompt_embeds"],
            current_start_frame=current_start_frame,
            temporal_freqs=temporal_freqs,
        )
        self._reset_crossattn_cache()

        pages = self.hkv.retrieve(
            prompt_embeds=next_conditional_dict["prompt_embeds"],
            query_vec=self._query_index(noisy_chunk),
        )
        if not pages:
            return False

        installed = self.hkv.apply(
            self.kv_cache1,
            pages,
            current_start_frame=current_start_frame,
            temporal_freqs=temporal_freqs,
            hsa_modules=self._hsa_modules(),
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[HKV] installed {len(pages)} page(s) "
                f"({installed // self.frame_seq_length} frames) from {len(self.hkv)} banked chunk(s)"
            )
        return installed > 0

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
        if current_start_frame == 0:
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

    # ------------------------------------------------------------------ generation
    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video, switching prompts at the given frame indices.

        Args:
            noise: ``(B, T, C, H, W)`` latent noise.
            text_prompts_list: one prompt list per segment, aligned with the batch.
            switch_frame_indices: frame index at which each segment after the first takes over;
                length is ``len(text_prompts_list) - 1``.
            return_latents: also return the latent tensor.
            low_memory: keep the output on CPU and swap the text encoder in on demand.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "switch_frame_indices must have one entry fewer than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

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
            self.hkv.reset()

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_cache_size = (
            local_attn_cfg * self.frame_seq_length if local_attn_cfg != -1
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

        for _ in range(num_blocks):
            current_num_frames = self.num_frame_per_block

            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                previous_cond = cond_list[segment_idx]
                segment_idx += 1
                next_cond = cond_list[segment_idx]
                noisy_chunk = noise[:, current_start_frame:current_start_frame + current_num_frames]

                switched = False
                if self.hkv is not None:
                    switched = self._switch_via_hkv(
                        previous_cond, next_cond, noisy_chunk, current_start_frame,
                    )
                if not switched:
                    self._recache_after_switch(output, current_start_frame, next_cond)

                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices) else None
                )
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[interactive] segment {segment_idx} at frame {current_start_frame}")

            conditional_dict = cond_list[segment_idx]
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

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

            # Replay the chunk at the context timestep so the cache holds clean keys and values.
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=torch.ones_like(timestep) * self.args.context_noise,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += current_num_frames

        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
