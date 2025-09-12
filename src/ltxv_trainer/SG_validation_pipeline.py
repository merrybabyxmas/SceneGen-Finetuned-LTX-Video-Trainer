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
                    
                    # Save intermediate steps if available
                    if hasattr(result, 'intermediate_steps') and result.intermediate_steps:
                        logger.info(f"Saving {len(result.intermediate_steps)} intermediate steps...")
                        try:
                            self._save_intermediate_steps(result.intermediate_steps, output_dir, global_step, shot_idx)
                        except Exception as intermediate_error:
                            logger.error(f"Intermediate steps saving failed completely: {intermediate_error}")
                            logger.info("Continuing without intermediate steps...")
                    
                    # Save prev frames if available
                    if hasattr(result, 'prev_frames') and result.prev_frames:
                        logger.info("Saving prev frames...")
                        logger.info(f"Prev frames type: {type(result.prev_frames)}")
                        if isinstance(result.prev_frames, list) and len(result.prev_frames) > 0:
                            logger.info(f"First prev frame type: {type(result.prev_frames[0])}")
                        
                        try:
                            # Handle nested list structure
                            prev_frames_to_save = result.prev_frames
                            if isinstance(prev_frames_to_save, list) and len(prev_frames_to_save) > 0:
                                # If it's a nested list (list of lists), flatten it
                                if isinstance(prev_frames_to_save[0], list):
                                    logger.info("Flattening nested list structure for prev frames")
                                    prev_frames_to_save = prev_frames_to_save[0]  # Take the first (and likely only) list
                                    logger.info(f"Flattened prev frames length: {len(prev_frames_to_save)}")
                            
                            prev_video_path = output_dir / f"multishot_step_{global_step:06d}_shot_{shot_idx:02d}_prev.mp4"
                            export_to_video(prev_frames_to_save, str(prev_video_path), fps=24)
                            logger.info(f"Saved prev frames to {prev_video_path.name}")
                        except Exception as prev_save_error:
                            logger.error(f"Failed to save prev frames: {prev_save_error}")
                            logger.error(f"Prev frames shape/type: {result.prev_frames.shape if hasattr(result.prev_frames, 'shape') else type(result.prev_frames)}")
                            # Try alternative save approach
                            if isinstance(result.prev_frames, list) and len(result.prev_frames) > 0:
                                try:
                                    logger.info("Trying alternative save approach...")
                                    # Try to access the first video if it's a batch
                                    alt_prev_frames = result.prev_frames[0] if isinstance(result.prev_frames[0], list) else result.prev_frames
                                    prev_video_path = output_dir / f"multishot_step_{global_step:06d}_shot_{shot_idx:02d}_prev.mp4"
                                    export_to_video(alt_prev_frames, str(prev_video_path), fps=24)
                                    logger.info(f"Alternative save successful: {prev_video_path.name}")
                                except Exception as alt_error:
                                    logger.error(f"Alternative save also failed: {alt_error}")
                    
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
                logger.info(f"Videos type: {type(videos)}, length: {len(videos) if isinstance(videos, list) else 'not a list'}")
                
                video = videos[0] if isinstance(videos, list) else videos
                logger.info(f"Selected video type: {type(video)}")
                
                if isinstance(video, list) and len(video) > 0:
                    logger.info(f"🎥 Multi-shot validation video frames: {len(video)}, first frame size: {video[0].size}")
                    frame_count = len(video)
                elif hasattr(video, 'shape'):
                    logger.info(f"🎥 Multi-shot validation video tensor shape: {video.shape}")
                    frame_count = video.shape[0] if len(video.shape) >= 4 else "unknown"
                else:
                    logger.info(f"🎥 Multi-shot validation video type: {type(video)}")
                    frame_count = "unknown"
                
                # The discrepancy might be here - videos structure
                logger.info(f"🔍 Detailed videos analysis:")
                logger.info(f"   videos type: {type(videos)}")
                if isinstance(videos, list):
                    logger.info(f"   videos length: {len(videos)}")
                    if len(videos) > 0:
                        logger.info(f"   first video type: {type(videos[0])}")
                        if isinstance(videos[0], list):
                            logger.info(f"   first video frames: {len(videos[0])}")
                
                video_path = output_dir / f"multishot_step_{global_step:06d}_shot_{shot_idx:02d}.mp4"
                export_to_video(videos[0] if isinstance(videos, list) else videos, str(video_path), fps=24)
                video_paths.append(video_path)
                logger.info(f"Saved shot {shot_idx + 1} to {video_path.name} (expected {frame_count} frames)")
        
        return video_paths
    
    def _save_intermediate_steps(self, intermediate_steps: List[Dict], output_dir: Path, global_step: int, shot_idx: int):
        """Save intermediate denoising steps as videos."""
        try:
            from diffusers.utils import export_to_video
            
            for step_data in intermediate_steps:
                step_num = step_data['step']
                timestep = step_data['timestep']
                combined_latents = step_data['combined_latents']
                prev_seq_len = step_data['prev_seq_len']
                curr_seq_len = step_data['curr_seq_len']
                
                # Extract current part from intermediate step
                current_part = combined_latents[:, prev_seq_len:prev_seq_len + curr_seq_len]
                
                # Use the same dimensions as the main generation (from config)
                # video_dims in config is [width, height, frames]
                width = 768
                height = 448
                frames = 17
                vae_temporal_downsample = getattr(self.pipeline.vae, 'temporal_downsample_factor', 7)
                vae_spatial_downsample = getattr(self.pipeline.vae, 'spatial_downsample_factor', 32)
                
                latent_frames = frames // vae_temporal_downsample + 1
                latent_height = height // vae_spatial_downsample
                latent_width = width // vae_spatial_downsample
                
                # Unpack and decode intermediate latents with error handling
                try:
                    logger.info(f"Unpacking latents for step {step_num}: {current_part.shape}")
                    current_part = self.pipeline._unpack_latents(
                        current_part,
                        latent_frames,
                        latent_height,
                        latent_width,
                        self.pipeline.transformer_spatial_patch_size,
                        self.pipeline.transformer_temporal_patch_size,
                    )
                    logger.info(f"Unpacking successful, new shape: {current_part.shape}")
                except Exception as unpack_error:
                    logger.error(f"Unpack failed for step {step_num}: {unpack_error}")
                    continue  # Skip this intermediate step
                
                # Use exact same denormalization as main pipeline
                try:
                    if (hasattr(self.pipeline.vae, 'latents_mean') and 
                        self.pipeline.vae.latents_mean is not None and
                        hasattr(self.pipeline.vae, 'latents_std') and
                        self.pipeline.vae.latents_std is not None):
                        
                        latents_mean = self.pipeline.vae.latents_mean.view(1, -1, 1, 1, 1).to(current_part.device, current_part.dtype)
                        latents_std = self.pipeline.vae.latents_std.view(1, -1, 1, 1, 1).to(current_part.device, current_part.dtype)
                        scaling_factor = getattr(self.pipeline.vae.config, 'scaling_factor', 0.18215)
                        current_part = current_part * latents_std / scaling_factor + latents_mean
                        logger.info(f"Applied full denormalization for intermediate step {step_num}")
                    else:
                        scaling_factor = getattr(self.pipeline.vae.config, 'scaling_factor', 0.18215)
                        current_part = current_part / scaling_factor
                        logger.info(f"Applied simple scaling factor {scaling_factor} for intermediate step {step_num}")
                except Exception as denorm_error:
                    logger.warning(f"Denormalization failed, using raw latents: {denorm_error}")
                
                # Use exact same decode method as main pipeline
                with torch.no_grad():
                    try:
                        logger.info(f"Decoding intermediate step {step_num} with current_part shape: {current_part.shape}")
                        
                        # Match main pipeline: ensure correct dtype
                        # We need a reference dtype - use bfloat16 as default for intermediate steps
                        target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                        current_part = current_part.to(target_dtype)
                        
                        # Handle timestep conditioning exactly like main pipeline
                        timestep = None
                        if hasattr(self.pipeline.vae.config, 'timestep_conditioning') and self.pipeline.vae.config.timestep_conditioning:
                            timestep = torch.tensor([0.0], device=current_part.device, dtype=current_part.dtype)
                        
                        video_frames = self.pipeline.vae.decode(current_part, timestep, return_dict=False)[0]
                        logger.info(f"VAE decode successful, output shape: {video_frames.shape}")
                        
                        video_frames = self.pipeline.video_processor.postprocess_video(video_frames, output_type='pil')
                        logger.info(f"Video processor successful, output type: {type(video_frames)}")
                        
                    except Exception as decode_error:
                        logger.error(f"Intermediate step {step_num} decode/postprocess failed: {decode_error}")
                        continue  # Skip this intermediate step and continue with others
                
                # Save intermediate step video
                try:
                    # Handle timestep formatting safely
                    if timestep is not None:
                        if hasattr(timestep, 'item'):
                            timestep_val = timestep.item()
                        else:
                            timestep_val = float(timestep)
                    else:
                        timestep_val = step_data['timestep']  # Use original timestep from step_data
                    
                    intermediate_path = output_dir / f"multishot_step_{global_step:06d}_shot_{shot_idx:02d}_denoise_{step_num:02d}_t{timestep_val:.3f}.mp4"
                    
                    # Handle nested list structure like prev frames
                    video_to_save = video_frames
                    if isinstance(video_frames, list) and len(video_frames) > 0:
                        if isinstance(video_frames[0], list):
                            logger.info(f"Flattening nested video structure for intermediate step {step_num}")
                            video_to_save = video_frames[0]  # Take the first (and likely only) list
                    
                    export_to_video(video_to_save, str(intermediate_path), fps=24)
                    logger.info(f"Saved intermediate step {step_num} at timestep {timestep_val:.3f} to {intermediate_path.name}")
                except Exception as save_error:
                    logger.error(f"Failed to save intermediate step {step_num}: {save_error}")
                    logger.error(f"Video frames type: {type(video_frames)}")
                    if hasattr(video_frames, 'shape'):
                        logger.error(f"Video frames shape: {video_frames.shape}")
                    elif isinstance(video_frames, list):
                        logger.error(f"Video frames list length: {len(video_frames)}")
                        if len(video_frames) > 0:
                            logger.error(f"First video frames type: {type(video_frames[0])}")
                            if isinstance(video_frames[0], list):
                                logger.error(f"First video frames length: {len(video_frames[0])}")
                
        except Exception as e:
            logger.error(f"Failed to save intermediate steps: {e}")


def create_multi_shot_validation_pipeline(
    scheduler,
    vae,
    text_encoder, 
    tokenizer,
    transformer,
    device: torch.device,
    accelerator,
    d_model: int = 128,
    conditioning_mode: str = "none"
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
            _ = param_name  # Suppress unused variable warning
            param_devices.add(str(param.device))
        for buffer_name, buffer in model.named_buffers():
            _ = buffer_name  # Suppress unused variable warning
            buffer_devices.add(str(buffer.device))
            
        logger.info(f"{name} parameter devices: {param_devices}")
        logger.info(f"{name} buffer devices: {buffer_devices}")

    # Create a fresh scheduler copy and ensure device placement
    scheduler_copy = deepcopy(scheduler)
    if hasattr(scheduler_copy, 'to'):
        scheduler_copy = scheduler_copy.to(device)
    
    # Create the appropriate pipeline based on conditioning mode
    if conditioning_mode == "cross_attention":
        from ltxv_trainer.SG_enhanced_cross_attention_pipeline import SGEnhancedCrossAttentionPipeline
        multishot_pipeline = SGEnhancedCrossAttentionPipeline(
            scheduler=scheduler_copy,
            vae=unwrapped_vae,
            text_encoder=unwrapped_text_encoder, 
            tokenizer=tokenizer,
            transformer=unwrapped_transformer,
            sos_token_generator=sos_generator
        )
    else:
        # Default to concatenation-based multi-shot pipeline
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