# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from pipeline.streaming_training import StreamingTrainingPipeline
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist


class StreamingSwitchTrainingPipeline(StreamingTrainingPipeline):
    """Training pipeline supporting mid-video prompt switching."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.global_sink = getattr(args, "global_sink", False)

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
        """Chunk generation method tailored for sequential training with prompt switching."""

        
        
        if switch_conditional_dict is None or switch_frame_index is None:
            return super().generate_chunk_with_cache(
                noise=noise,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                requires_grad=requires_grad,
                return_sim_step=return_sim_step,
            )
        
        
        batch_size, chunk_frames, num_channels, height, width = noise.shape
        assert chunk_frames % self.num_frame_per_block == 0
        num_blocks = chunk_frames // self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks

        output = torch.zeros_like(noise)
        
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        
        if not requires_grad:
            start_gradient_frame_index = chunk_frames
        else:
            start_gradient_frame_index = switch_frame_index
        
        
        local_start_frame = 0
        self.generator.model.local_attn_size = int(self.local_attn_size)
        self._set_all_modules_max_attention_size(int(self.local_attn_size))

        using_second = False
        cond_in_use = conditional_dict
        for block_index, current_num_frames in enumerate(all_num_frames):
            if (not using_second) and (local_start_frame >= switch_frame_index):
                
                    
                self._recache_after_switch(output[:, :local_start_frame, ...], current_start_frame+local_start_frame, switch_conditional_dict, local_start_frame, switch_recache_frames)
                
                
                cond_in_use = switch_conditional_dict
                using_second = True
                
            
            
            
            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]
            
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep
                
                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                        
                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    enable_grad = local_start_frame >= start_gradient_frame_index
                    
                    
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                    break
            
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred
            
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                )
            
            local_start_frame += current_num_frames
        
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
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

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict, local_start_frame=None, switch_recache_frames=None):
        if not self.global_sink:
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
            
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        
        if current_start_frame == 0:
            return

        if switch_recache_frames is not None:
            frames_to_recache = torch.cat([switch_recache_frames, output], dim=1)[:, -21:, ...]
            num_recache_frames = frames_to_recache.shape[1]
        else:
            if local_start_frame is not None:
                num_recache_frames = min(local_start_frame, 21)
                frames_to_recache = output[:, -num_recache_frames:]
            else:
                num_recache_frames = min(current_start_frame, 21)
                frames_to_recache = output[:, -num_recache_frames:]
            
        batch_size, num_recache_frames, c, h, w = frames_to_recache.shape
        
        
        device = frames_to_recache.device
        
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=21
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.context_noise
        
        self.generator.model.block_mask = block_mask
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=(current_start_frame - num_recache_frames) * self.frame_seq_length,
            )

        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
