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
            logger.info(f"Calling pipeline with inputs: {list(pipeline_inputs.keys())}")
            try:
                with autocast(self.device.type, dtype=torch.bfloat16):
                    logger.info("Pipeline generation starting...")
                    result = self.pipeline(**pipeline_inputs)
                    logger.info("Pipeline generation completed")
                    videos = result.frames
                    logger.info(f"Generated videos count: {len(videos) if videos else 0}")
                    
                    # Store latent representation for next shot conditioning
                    if len(videos) > 0:
                        logger.info("Encoding video to latent for next shot...")
                        prev_latent = self.pipeline.encode_video_to_latent(videos[0])
                    else:
                        logger.warning("No videos generated!")
                        
            except Exception as e:
                logger.error(f"Pipeline generation failed for shot {shot_idx + 1}: {e}")
                raise
            
            # Save the generated video
            if videos:
                # Debug: Check video dimensions
                video = videos[0]
                if isinstance(video, list) and len(video) > 0:
                    logger.info(f"🎥 Multi-shot validation video frames: {len(video)}, first frame size: {video[0].size}")
                elif hasattr(video, 'shape'):
                    pass
                
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
    
    # Create SOS token generator and move to device
    sos_generator = SOSTokenLatents(d_model=d_model).to(device)
    
    # Models are already unwrapped from trainer.py, use them directly
    # No need to unwrap again since trainer already calls accelerator.unwrap_model()
    unwrapped_vae = vae
    unwrapped_text_encoder = text_encoder
    unwrapped_transformer = transformer
    
    # Force all model parameters to the correct device (skip 8-bit models)
    def safe_to_device(model, device, model_name):
        """Safely move model to device, handling 8-bit quantized models."""
        try:
            # Check if model is 8-bit quantized
            is_8bit = hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit
            if not is_8bit:
                model = model.to(device)
            else:
                logger.info(f"{model_name} is 8-bit quantized, skipping device move")
            return model
        except Exception as e:
            logger.warning(f"Could not move {model_name} to device: {e}")
            return model
    
    unwrapped_vae = safe_to_device(unwrapped_vae, device, "VAE")
    unwrapped_text_encoder = safe_to_device(unwrapped_text_encoder, device, "TextEncoder") 
    unwrapped_transformer = safe_to_device(unwrapped_transformer, device, "Transformer")
    
    # WARNING: DO NOT directly modify model parameters during validation!
    # The previous code was causing model collapse by modifying param.data and buffer.data
    # Models should already be on the correct device from safe_to_device() calls above
    # If device placement is still needed, it should be done through proper model.to(device) calls
    
    # Log device status for debugging without modifying parameters
    for model, name in [(unwrapped_vae, "VAE"), (unwrapped_text_encoder, "TextEncoder"), (unwrapped_transformer, "Transformer")]:
        is_8bit = hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit
        if is_8bit:
            logger.info(f"{name} is 8-bit quantized, skipping device check")
            continue
            
        # Only check and log, do not modify
        param_devices = set()
        buffer_devices = set()
        for param_name, param in model.named_parameters():
            param_devices.add(str(param.device))
        for buffer_name, buffer in model.named_buffers():
            buffer_devices.add(str(buffer.device))
            
        logger.info(f"{name} parameter devices: {param_devices}")
        logger.info(f"{name} buffer devices: {buffer_devices}")

    # Create a fresh scheduler copy and ensure device placement
    scheduler_copy = deepcopy(scheduler)
    if hasattr(scheduler_copy, 'to'):
        scheduler_copy = scheduler_copy.to(device)
    
    # Create the SGMultiShotPipeline with multi-shot capabilities
    multishot_pipeline = SGMultiShotPipeline(
        scheduler=scheduler_copy,
        vae=unwrapped_vae,
        text_encoder=unwrapped_text_encoder, 
        tokenizer=tokenizer,
        transformer=unwrapped_transformer,
        sos_token_generator=sos_generator
    )
    
    # Ensure pipeline components are on the correct device
    multishot_pipeline.set_progress_bar_config(disable=True)
    multishot_pipeline = multishot_pipeline.to(device)
    
    # Force all pipeline submodules to the correct device (skip 8-bit models)
    if hasattr(multishot_pipeline, 'vae') and multishot_pipeline.vae is not None:
        multishot_pipeline.vae = safe_to_device(multishot_pipeline.vae, device, "Pipeline VAE")
    if hasattr(multishot_pipeline, 'text_encoder') and multishot_pipeline.text_encoder is not None:
        multishot_pipeline.text_encoder = safe_to_device(multishot_pipeline.text_encoder, device, "Pipeline TextEncoder")
    if hasattr(multishot_pipeline, 'transformer') and multishot_pipeline.transformer is not None:
        multishot_pipeline.transformer = safe_to_device(multishot_pipeline.transformer, device, "Pipeline Transformer")
    
    # Set pipeline execution device (if possible)
    try:
        multishot_pipeline._execution_device = device
    except (AttributeError, TypeError):
        # If we can't set _execution_device directly, it should be handled by the pipeline's to() method
        logger.info("Could not set _execution_device directly, relying on pipeline.to(device)")
        pass

    
    # Create and return the multi-shot validation pipeline
    return MultiShotValidationPipeline(
        pipeline=multishot_pipeline,
        device=device,
        accelerator=accelerator,
        sos_token_generator=sos_generator
    )