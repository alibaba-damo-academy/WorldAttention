# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import time
import torch
import torch.distributed as dist
from typing import Tuple, Dict, Any, Optional, List
from einops import rearrange

from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline


class StreamingTrainingModel:
    """A model wrapper specifically for streaming/serialized training."""
    
    def __init__(self, base_model, config):
        """Initialize the streaming training model."""
        self.base_model = base_model
        self.config = config
        self.device = base_model.device
        self.dtype = base_model.dtype
        self.image_or_video_shape = getattr(config, 'image_or_video_shape', None)
        
        self.chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.max_length = getattr(config, "streaming_max_length", 57)
        self.possible_max_length = getattr(config, "streaming_possible_max_length", None)
        self.min_new_frame = getattr(config, "streaming_min_new_frame", 18)

        self.generator = base_model.generator
        self.fake_score = base_model.fake_score
        self.scheduler = base_model.scheduler
        self.denoising_loss_func = base_model.denoising_loss_func
        
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(base_model.inference_pipeline, 'frame_seq_length', 1560)
        
        self.inference_pipeline = base_model.inference_pipeline
        if self.inference_pipeline is None:
            base_model._initialize_inference_pipeline()
            self.inference_pipeline = base_model.inference_pipeline
        
        self.reset_state()  
        

    def _process_first_frame_encoding(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply special encoding to the first frame, following the logic in _run_generator."""
        total_frames = frames.shape[1]
        
        if total_frames <= 1:
            return frames
        
        process_frames = min(21, total_frames)
        
            
        with torch.no_grad():
            frames_to_decode = frames[:, :-(process_frames - 1), ...]
            pixels = self.base_model.vae.decode_to_pixel(frames_to_decode)
            
            last_frame_pixel = pixels[:, -1:, ...].to(self.dtype)
            last_frame_pixel = rearrange(last_frame_pixel, "b t c h w -> b c t h w")
            
            image_latent = self.base_model.vae.encode_to_latent(last_frame_pixel).to(self.dtype)
            
        remaining_frames = frames[:, -(process_frames - 1):, ...]
        processed_frames = torch.cat([image_latent, remaining_frames], dim=1)
                
            
        return processed_frames

    def reset_state(self):
        """Reset streaming training state"""
            
        self.state = {
            "current_length": 0,
            "conditional_info": None,
            "has_switched": False,
            "previous_frames": None,
            "temp_max_length": None,
        }

        self.inference_pipeline.clear_kv_cache()
        
    
    def _should_switch_prompt(self, chunk_start_frame: int, chunk_size: int) -> bool:
        """Determine whether to switch prompt (DMDSwitch only)"""
        from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline
        if not isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            return False
        
        if self.state.get("has_switched", False):
            return False
        
        switch_info = self.state["conditional_info"].get("switch_info", {})
        switch_frame_index = switch_info.get("switch_frame_index")
        
        if switch_frame_index is None:
            return False
        
        chunk_end_frame = chunk_start_frame + chunk_size
        should_switch = chunk_start_frame <= switch_frame_index < chunk_end_frame
        
            
        return should_switch

    def _get_current_conditional_dict(self, chunk_start_frame: int) -> dict:
        """Get the conditional_dict to use for the current chunk"""
        cond_info = self.state["conditional_info"]
        
        switch_info = cond_info.get("switch_info", {})
        if switch_info:
            switch_frame_index = switch_info.get("switch_frame_index")
            if switch_frame_index is not None:
                if self.state.get("has_switched", False) or chunk_start_frame >= switch_frame_index:
                    return switch_info.get("switch_conditional_dict", cond_info["conditional_dict"])
        
        return cond_info["conditional_dict"]
    
    def _generate_chunk(
        self, 
        noise_chunk: torch.Tensor,
        chunk_start_frame: int,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """Generate a single chunk."""
        
            
        current_conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        
        kwargs = {
            "noise": noise_chunk,
            "conditional_dict": current_conditional_dict,
            "current_start_frame": chunk_start_frame,
            "requires_grad": requires_grad,
            "return_sim_step": False,
        }
        
        if isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            switch_info = self.state["conditional_info"].get("switch_info", {})
            if switch_info and self._should_switch_prompt(chunk_start_frame, noise_chunk.shape[1]):
                if (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[StreamingTrain-Model] Switching prompt at frame {switch_info['switch_frame_index']}")
                
                relative_switch_index = max(0, switch_info["switch_frame_index"] - chunk_start_frame)
                kwargs["switch_frame_index"] = relative_switch_index
                kwargs["switch_conditional_dict"] = switch_info["switch_conditional_dict"]
                
                if self.state["previous_frames"] is not None:
                    kwargs["switch_recache_frames"] = self.state["previous_frames"]
                    
                
                
                self.state["has_switched"] = True
        
        
            
        output, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.generate_chunk_with_cache(**kwargs)

        return output, denoised_timestep_from, denoised_timestep_to
    
    def setup_sequence(
        self,
        conditional_dict: Dict,
        unconditional_dict: Dict,
        initial_latent: Optional[torch.Tensor] = None,
        switch_conditional_dict: Optional[Dict] = None,
        switch_frame_index: Optional[int] = None,
        temp_max_length: Optional[int] = None,
    ):
        """Set up a new sequence"""
            

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.image_or_video_shape[0]
        if self.inference_pipeline.kv_cache1 is None:
            self.inference_pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )

        if self.inference_pipeline.crossattn_cache is None:
            self.inference_pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )
        
        self.reset_state()
        self.state["temp_max_length"] = temp_max_length
        
            
            
        
        if initial_latent is not None:
            self.state["current_length"] = initial_latent.shape[1]
        else:
            self.state["current_length"] = 0
        
        self.state["conditional_info"] = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
        }
        
        if switch_conditional_dict is not None and switch_frame_index is not None:
            self.state["conditional_info"]["switch_info"] = {
                "switch_conditional_dict": switch_conditional_dict,
                "switch_frame_index": switch_frame_index,
            }
        
        if initial_latent is not None:
            
            timestep = torch.zeros([batch_size, initial_latent.shape[1]], device=self.device, dtype=torch.int64)
            with torch.no_grad():
                self.inference_pipeline.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.inference_pipeline.kv_cache1,
                    crossattn_cache=self.inference_pipeline.crossattn_cache,
                    current_start=0
                )
            
        else:
            pass

    def can_generate_more(self) -> bool:
        """Check whether more chunks can be generated"""
        current_length = self.state["current_length"]
        temp_max_length = self.state.get("temp_max_length")
        can_generate = current_length < temp_max_length and (current_length + self.min_new_frame) <= temp_max_length
        
            
        return can_generate
    def generate_next_chunk(self, requires_grad: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate the next chunk, supporting overlap to ensure temporal continuity."""
            
            
        if not self.can_generate_more():
            raise ValueError("Cannot generate more chunks")
        
        current_length = self.state["current_length"]
        batch_size = self.image_or_video_shape[0]
        
        
        previous_frames = self.state.get("previous_frames")
        if previous_frames is not None:
            max_new_frames = min(self.state["temp_max_length"] - current_length + 1, self.chunk_size)
            possible_new_frames = list(range(self.min_new_frame, max_new_frames, 3))
            
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(possible_new_frames) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(possible_new_frames) - 1)
            
            new_frames_to_generate = possible_new_frames[selected_idx]

            overlap_frames = self.chunk_size - new_frames_to_generate
            if overlap_frames > 0 and overlap_frames <= previous_frames.shape[1]:
                overlap_frames_to_use = overlap_frames
            else:
                overlap_frames_to_use = 0
                new_frames_to_generate = self.chunk_size
            
        else:
            overlap_frames_to_use = 0
            new_frames_to_generate = self.chunk_size
        
        
        noise_chunk = torch.randn(
            [batch_size, new_frames_to_generate, *self.image_or_video_shape[2:]],
            device=self.device,
            dtype=self.dtype
        )
        
        generated_new_frames, denoised_timestep_from, denoised_timestep_to = self._generate_chunk(
            noise_chunk=noise_chunk,
            chunk_start_frame=current_length,
            requires_grad=requires_grad,
        )

        if previous_frames is not None:
            full_chunk = torch.cat([previous_frames, generated_new_frames], dim=1)
        else:
            full_chunk = generated_new_frames
        
        frames_to_save = full_chunk.detach().clone()[:, -self.chunk_size:, ...]
        

        if previous_frames is not None:
            full_chunk = self._process_first_frame_encoding(full_chunk)
        
        if previous_frames is not None:
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            gradient_mask[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate, ...] = True
        else:
            gradient_mask = torch.ones_like(full_chunk, dtype=torch.bool)

        self.state["current_length"] += new_frames_to_generate
        

        self.state["previous_frames"] = frames_to_save

        info = {
            "denoised_timestep_from": denoised_timestep_from,
            "denoised_timestep_to": denoised_timestep_to,
            "chunk_start_frame": current_length,
            "chunk_frames": full_chunk.shape[1],
            "new_frames_generated": new_frames_to_generate,
            "current_length": self.state["current_length"],
            "gradient_mask": gradient_mask,
            "overlap_frames_used": overlap_frames_to_use,
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[StreamingTrain-Model] current_training_chunk: ({self.state['current_length'] - new_frames_to_generate} -> {self.state['current_length']})/{self.state['temp_max_length']}")
        return full_chunk, info
    
    def compute_generator_loss(self,
        chunk: torch.Tensor,
        chunk_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the generator loss."""
        _t_loss_start = time.time()

        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        unconditional_dict = self.state["conditional_info"]["unconditional_dict"]
        
        gradient_mask = chunk_info.get("gradient_mask", None)
        
        
        dmd_loss, dmd_log_dict = self.base_model.compute_distribution_matching_loss(
            image_or_video=chunk,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=chunk_info["denoised_timestep_from"],
            denoised_timestep_to=chunk_info["denoised_timestep_to"]
        )
        
        
        dmd_log_dict.update({
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", chunk.shape[1]),
        })
        
        return dmd_loss, dmd_log_dict
    
    def _clear_cache_gradients(self):
        """Clear possible gradient references in KV cache and cross-attention cache."""
            
        if hasattr(self.inference_pipeline, 'kv_cache1') and self.inference_pipeline.kv_cache1 is not None:
            for cache_block in self.inference_pipeline.kv_cache1:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()
        
        if hasattr(self.inference_pipeline, 'crossattn_cache') and self.inference_pipeline.crossattn_cache is not None:
            for cache_block in self.inference_pipeline.crossattn_cache:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()
        

    def compute_critic_loss(self, chunk: torch.Tensor, chunk_info: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute critic loss."""
        _t_loss_start = time.time()
            
            
        if chunk.requires_grad:
            chunk = chunk.detach()

        self._clear_cache_gradients()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            
        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        
        gradient_mask = chunk_info.get("gradient_mask", None)
        
        batch_size, num_frame = chunk.shape[:2]
        
        
        denoised_timestep_from = chunk_info.get("denoised_timestep_from", None)
        denoised_timestep_to = chunk_info.get("denoised_timestep_to", None)
        
        min_timestep = denoised_timestep_to if (getattr(self.base_model, 'ts_schedule', False) and denoised_timestep_to is not None) else getattr(self.base_model, 'min_score_timestep')
        max_timestep = denoised_timestep_from if (getattr(self.base_model, 'ts_schedule_max', False) and denoised_timestep_from is not None) else getattr(self.base_model, 'num_train_timestep')
        
        critic_timestep = self.base_model._get_timestep(
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            num_frame_per_block=getattr(self.base_model, 'num_frame_per_block', 3),
            uniform_timestep=True
        ).to(self.device)
        
        if getattr(self.base_model, 'timestep_shift') > 1:
            timestep_shift = self.base_model.timestep_shift
            critic_timestep = timestep_shift * \
                (critic_timestep / 1000) / (1 + (timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        
        critic_timestep = critic_timestep.clamp(self.base_model.min_step, self.base_model.max_step)
        
        critic_noise = torch.randn_like(chunk)
        
        noisy_chunk = self.scheduler.add_noise(
            chunk.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        
        
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_chunk,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )
        

        denoising_loss_type = getattr(self.base_model.args, 'denoising_loss_type', 'mse')
        if denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))

        gradient_mask_flat = gradient_mask.flatten(0, 1) if gradient_mask is not None else None
        denoising_loss = self.denoising_loss_func(
            x=chunk.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat
        )
        
        

        del conditional_dict, critic_noise, noisy_chunk, pred_fake_image
        if 'flow_pred' in locals():
            del flow_pred
        if 'pred_fake_noise' in locals():
            del pred_fake_noise

        critic_log_dict = {
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", num_frame),
        }

        return denoising_loss, critic_log_dict
    
    def get_sequence_length(self) -> int:
        """Get current sequence length"""
        return self.state.get("current_length", 0) 
