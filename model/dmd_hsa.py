# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from model.dmd_switch import DMDSwitch
from pipeline.hsa_streaming_switch_training import HSAStreamingSwitchTrainingPipeline


class DMDHSA(DMDSwitch):
    """Prompt-switch DMD stage whose rollout runs with HSA as the active attention."""

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = HSAStreamingSwitchTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            context_noise=self.args.context_noise,
            local_attn_size=getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1),
            slice_last_frames=getattr(self.args, "slice_last_frames", 21),
            global_sink=getattr(self.args, "global_sink", False),
            enable_hsa=getattr(self.args, "enable_hsa", True),
            hsa_backend=getattr(self.args, "hsa_backend", "auto"),
        )
