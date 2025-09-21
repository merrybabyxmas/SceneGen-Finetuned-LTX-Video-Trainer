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
import json
from datetime import datetime

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
        # Ensure device is a torch.device object, not a string
        self.device = torch.device(device) if isinstance(device, str) else device
        self.accelerator = accelerator
        # No longer using SOS token generator - using Gaussian noise instead
        self.sos_token_generator = None
        # Store validation prompt information for JSON export
        self.validation_info = {}
        
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
        scenario: Optional[str] = None,
        validation_type: Optional[str] = None,
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
            scenario: Optional scenario string for token-based preprocessing
            validation_type: Type of validation ("t2v_validation" or "v2v_validation")

        Returns:
            List of paths to generated video files
        """


        output_dir.mkdir(exist_ok=True, parents=True)

        # Create scenario subfolder for organized storage
        scenario_dir = output_dir / "scenario"
        scenario_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        width, height, frames = video_dims

        # Collect validation information for JSON export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_data = {
            "prompt": prompt,
            "timestamp": timestamp,
            "global_step": global_step,
            "video_dims": video_dims,
            "num_shots": num_shots,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "scenario_tokens": scenario,
            "validation_type": validation_type,
            "generation_type": "multishot_validation"
        }
        
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
                "return_latents": True,  # Return latents for next shot conditioning
                "save_full_sequence": True,  # Save full transformer output including reference tokens
                "validation_type": validation_type,  # Pass validation type for proper filename generation
                "step": global_step,  # Pass step number for filename
            }

            # Add scenario to pipeline inputs if provided
            if scenario is not None:
                pipeline_inputs["scenario"] = scenario
                logger.info(f"Using scenario for shot {shot_idx + 1}: '{scenario}'")

            # Handle conditioning for multi-shot generation
            if shot_idx == 0:
                # First shot: use SOS token conditioning
                latent_num_frames = (frames - 1) // 8 + 1  # VAE temporal compression ratio
                latent_height = height // 32  # VAE spatial compression ratio
                latent_width = width // 32

                # Generate Gaussian noise SOS latents
                seq_len = latent_num_frames * latent_height * latent_width
                d_model = 128  # Standard latent dimension
                sos_latents = torch.randn(1, seq_len, d_model, device=self.device, dtype=torch.float32)
                pipeline_inputs["reference_latents"] = sos_latents
                logger.info(f"Using Gaussian noise SOS conditioning for first shot (shape: {sos_latents.shape})")
            else:
                # Subsequent shots: use previous shot latents as conditioning
                pipeline_inputs["reference_latents"] = prev_latent
                logger.info(f"Using previous shot conditioning for shot {shot_idx + 1} (latent shape: {prev_latent.shape if prev_latent is not None else 'None'})")
            # Generate the video using SGMultiShotPipeline
            logger.info(f"Calling pipeline with inputs: {list(pipeline_inputs.keys())}")
            try:
                with autocast(self.device.type, dtype=torch.bfloat16):
                    logger.info("Pipeline generation starting...")
                    result = self.pipeline(**pipeline_inputs)
                    logger.info("Pipeline generation completed")

                    # Handle both dict and object results
                    if isinstance(result, dict):
                        logger.info(f"Result keys: {list(result.keys())}")
                        videos = result.get('frames') or result.get('videos') or result.get('video')
                        if videos is None:
                            # Try to get the first available video-like value
                            for key in result.keys():
                                if 'video' in key.lower() or 'frame' in key.lower():
                                    videos = result[key]
                                    logger.info(f"Using key '{key}' for videos")
                                    break
                    else:
                        videos = result.frames if hasattr(result, 'frames') else result

                    logger.info(f"Generated videos count: {len(videos) if videos and hasattr(videos, '__len__') else 'unknown'}")
                    logger.info(f"Videos type: {type(videos)}")
                    
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
                            
                            # Use the same prefix logic for prev frames
                            if validation_type == "t2v_validation":
                                prefix_prev = "t2v"
                            elif validation_type == "v2v_validation":
                                prefix_prev = "v2v"
                            else:
                                prefix_prev = "v2v" if scenario else "t2v"
                            prev_video_path = scenario_dir / f"{prefix_prev}_shot{shot_idx + 1}_prev_step_{global_step:06d}.mp4"
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
                                    # Use the same prefix logic for prev frames
                                    if validation_type == "t2v_validation":
                                        prefix_prev = "t2v"
                                    elif validation_type == "v2v_validation":
                                        prefix_prev = "v2v"
                                    else:
                                        prefix_prev = "v2v" if scenario else "t2v"
                                    prev_video_path = scenario_dir / f"{prefix_prev}_shot{shot_idx + 1}_prev_step_{global_step:06d}.mp4"
                                    export_to_video(alt_prev_frames, str(prev_video_path), fps=24)
                                    logger.info(f"Alternative save successful: {prev_video_path.name}")
                                except Exception as alt_error:
                                    logger.error(f"Alternative save also failed: {alt_error}")
                    
                    # Store latent representation for next shot conditioning
                    if hasattr(result, 'latents') and result.latents is not None:
                        prev_latent = result.latents.clone()
                        logger.info(f"Stored latents for next shot: {prev_latent.shape}")
                    elif isinstance(result, dict) and 'latents' in result:
                        prev_latent = result['latents'].clone() if hasattr(result['latents'], 'clone') else result['latents']
                        logger.info(f"Stored latents from dict for next shot: {prev_latent.shape}")
                    elif len(videos) > 0:
                        logger.warning("No latents returned from pipeline, encoding video to latent for next shot...")
                        # Fallback: encode video if latents not available
                        prev_latent = self.pipeline.encode_video_to_latent(videos[0])
                    else:
                        logger.warning("No videos or latents generated!")
                        prev_latent = None
                        
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
                
                # Create filename with new naming convention based on validation type
                # Determine prefix based on validation type
                if validation_type == "t2v_validation":
                    prefix = "t2v"
                elif validation_type == "v2v_validation":
                    prefix = "v2v"
                else:
                    # Fallback: use v2v for scenario-based validation
                    prefix = "v2v" if scenario else "t2v"

                if shot_idx == 0:
                    filename = f"{prefix}_shot{shot_idx + 1}_step_{global_step:06d}.mp4"
                else:
                    filename = f"{prefix}_shot{shot_idx}_transition_shot{shot_idx + 1}_step_{global_step:06d}.mp4"

                video_path = scenario_dir / filename
                export_to_video(videos[0] if isinstance(videos, list) else videos, str(video_path), fps=24)
                video_paths.append(video_path)
                logger.info(f"Saved shot {shot_idx + 1} to {video_path.name} (expected {frame_count} frames)")

        # Store validation information with video paths
        validation_data["video_paths"] = [str(path) for path in video_paths]
        validation_key = f"validation_step_{global_step:06d}_{timestamp}"
        self.validation_info[validation_key] = validation_data

        # Save validation information to JSON file
        validation_info_path = scenario_dir / "validation_info.json"
        try:
            with open(validation_info_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_info, f, indent=2, ensure_ascii=False)
            logger.info(f"📝 Validation information saved: {validation_info_path}")
        except Exception as e:
            logger.error(f"Failed to save validation info: {e}")

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
                    
                    # Save intermediate steps to scenario folder with new naming
                    scenario_dir = output_dir / "scenario"
                    scenario_dir.mkdir(exist_ok=True, parents=True)

                    # Use consistent prefix logic for intermediate steps
                    # Note: We need to get validation_type from the calling context
                    # For now, we'll use a default prefix until we can pass validation_type properly
                    prefix_inter = "v2v"  # Default for intermediate steps
                    intermediate_path = scenario_dir / f"{prefix_inter}_shot{shot_idx + 1}_denoise_{step_num:02d}_t{timestep_val:.3f}_step_{global_step:06d}.mp4"
                    
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
    conditioning_mode: str = "none",
    use_tokens: bool = True,
    hidden_dim: int = 3072
) -> MultiShotValidationPipeline:
    """Create a multi-shot validation pipeline."""
    
    # No longer need SOS token generator - using Gaussian noise instead
    sos_generator = None
    
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
            sos_token_generator=None,
            use_tokens=use_tokens,
            hidden_dim=hidden_dim
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