"""
Multi-shot validation pipeline for SceneGen LTX-Video training.

This module implements sequential video generation for validation:
1. Start with SOS token conditioning for the first shot
2. Generate each shot using the previous shot as conditioning (without noise removal)
3. Sequentially build up the multi-shot video generation
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
from torch.amp import autocast
from diffusers.utils import export_to_video
from copy import deepcopy

from ltxv_trainer import logger
from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
from ltxv_trainer.SG_datasets import SOSTokenLatents


class MultiShotValidationPipeline:
    """Pipeline for generating multi-shot videos during validation."""
    
    def __init__(
        self,
        pipeline: SGMultiShotPipeline,
        device: torch.device,
        accelerator,
        sos_token_generator: Optional[SOSTokenLatents] = None,
    ):
        self.pipeline = pipeline
        self.device = device 
        self.accelerator = accelerator
        self.sos_token_generator = sos_token_generator or SOSTokenLatents(d_model=128)
        
    def generate_multi_shot_sequence(
        self,
        prompt: str,  # Single prompt for multi-shot video
        output_dir: Path,
        global_step: int,
        video_dims: tuple[int, int, int],  # (width, height, frames)
        num_shots: int = 3,
        inference_steps: int = 300,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> List[Path]:
        """
        Generate a multi-shot video sequence using a single prompt.
        
        Args:
            prompt: Single text prompt for the entire multi-shot sequence
            num_shots: Number of shots to generate
            output_dir: Directory to save generated videos
            global_step: Current training step (for filename)
            video_dims: (width, height, frames) for each shot
            inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            negative_prompt: Negative prompt for all shots
            seed: Random seed
            
        Returns:
            List of paths to generated video files
        """

        
        output_dir.mkdir(exist_ok=True, parents=True)
        video_paths = []
        width, height, frames = video_dims
        
        # Storage for previous shot conditioning
        prev_latent = None
        
        for shot_idx in range(num_shots):            
            logger.info(f"Generating shot {shot_idx + 1}/{num_shots} with prompt: {prompt[:50]}...")
            
            # Set up generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(seed + shot_idx)
            
            # Prepare pipeline inputs
            pipeline_inputs = {
                "prompt": prompt,  # Same prompt for all shots
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": frames,
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_reference_comparison": True,
            }
            
            # Handle conditioning for multi-shot generation
            if shot_idx == 0:
                # First shot: use SOS token conditioning
                pipeline_inputs["use_sos_conditioning"] = True
                logger.info("Using SOS token conditioning for first shot")
            else:
                # Subsequent shots: use previous shot as conditioning
                pipeline_inputs["prev_latent"] = prev_latent
                pipeline_inputs["use_prev_conditioning"] = True
                logger.info(f"Using previous shot conditioning for shot {shot_idx + 1}")
            
            # Generate the video using SGMultiShotPipeline
            with autocast(self.device.type, dtype=torch.bfloat16):
                result = self.pipeline(**pipeline_inputs)
                videos = result.frames
                
                # Store latent representation for next shot conditioning
                if len(videos) > 0:
                    prev_latent = self.pipeline.encode_video_to_latent(videos[0])
            
            # Save the generated video
            if videos:
                video_path = output_dir / f"multishot_step_{global_step:06d}_shot_{shot_idx:02d}.mp4"
                export_to_video(videos[0], str(video_path), fps=24)
                video_paths.append(video_path)
                logger.info(f"Saved shot {shot_idx + 1} to {video_path.name}")
        
        return video_paths
    


def create_multi_shot_validation_pipeline(
    scheduler,
    vae,
    text_encoder, 
    tokenizer,
    transformer,
    device: torch.device,
    accelerator,
    d_model: int = 128
) -> MultiShotValidationPipeline:
    """Create a multi-shot validation pipeline."""
    
    # Create SOS token generator
    sos_generator = SOSTokenLatents(d_model=d_model)
    
    # Create the SGMultiShotPipeline with multi-shot capabilities
    multishot_pipeline = SGMultiShotPipeline(
        scheduler=deepcopy(scheduler),
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder), 
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
        sos_token_generator=sos_generator
    )
    multishot_pipeline.set_progress_bar_config(disable=True)
    
    # Create and return the multi-shot validation pipeline
    return MultiShotValidationPipeline(
        pipeline=multishot_pipeline,
        device=device,
        accelerator=accelerator,
        sos_token_generator=sos_generator
    )