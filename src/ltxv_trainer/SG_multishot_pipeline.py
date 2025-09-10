"""
Multi-shot Pipeline for LTX-Video with prev_latent conditioning support.

This pipeline extends the standard LTXConditionPipeline to support:
1. Previous latent conditioning (prev_latent)
2. Conditioning mask for controlling which parts to denoise
3. SOS token generation for the first shot
4. Sequential multi-shot video generation

The pipeline follows the same logic as SG_training_strategy.py for consistency.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.video_processor import VideoProcessor

from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline, LTXVideoCondition
from ltxv_trainer.SG_datasets import SOSTokenLatents
from ltxv_trainer import logger


class SGMultiShotPipeline(LTXConditionPipeline):
    """
    Enhanced LTX Pipeline with multi-shot capabilities.
    
    Supports previous latent conditioning and conditioning masks for sequential video generation.
    """
    
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        sos_token_generator: Optional[SOSTokenLatents] = None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        # SOS token generator for first shot
        self.sos_token_generator = sos_token_generator or SOSTokenLatents(d_model=128)
        
        # Initialize video processor if not already done by parent
        if not hasattr(self, 'video_processor') or self.video_processor is None:
            vae_scale_factor = getattr(self.vae, 'spatial_compression_ratio', 32)
            self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)
        
    @property 
    def device(self):
        """Get the device of the pipeline."""
        # Simple fallback to cuda:0 to avoid infinite logging
        return torch.device('cuda:0')
        
    @torch.no_grad()
    def __call__(
        self,
        # Standard LTX parameters
        conditions: Union[LTXVideoCondition, List[LTXVideoCondition]] = None,
        image=None,
        video=None,
        frame_index: Union[int, List[int]] = 0,
        strength: Union[float, List[float]] = 1.0,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 25,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        image_cond_noise_scale: float = 0.15,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        reference_video: Optional[torch.Tensor] = None,
        output_reference_comparison: bool = False,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        # Multi-shot specific parameters
        prev_latent: Optional[torch.Tensor] = None,
        use_prev_conditioning: bool = False,
        use_sos_conditioning: bool = False,
        **kwargs
    ):
        """
        Enhanced __call__ method with multi-shot support.
        
        Additional Args:
            prev_latent: Previous shot latent for conditioning [seq_len, channels]
            use_prev_conditioning: Whether to use previous latent conditioning
            use_sos_conditioning: Whether to use SOS token conditioning (for first shot)
        """
        
        # If multi-shot conditioning is requested, handle it before calling parent
        if use_prev_conditioning or use_sos_conditioning:
            return self._generate_with_multishot_conditioning(
                prev_latent=prev_latent,
                use_sos_conditioning=use_sos_conditioning,
                conditions=conditions,
                image=image,
                video=video,
                frame_index=frame_index,
                strength=strength,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                image_cond_noise_scale=image_cond_noise_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                reference_video=reference_video,
                output_reference_comparison=output_reference_comparison,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                **kwargs
            )
        
        # Standard generation without multi-shot conditioning
        return super().__call__(
            conditions=conditions,
            image=image,
            video=video,
            frame_index=frame_index,
            strength=strength,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            image_cond_noise_scale=image_cond_noise_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            reference_video=reference_video,
            output_reference_comparison=output_reference_comparison,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            **kwargs
        )
    
    def _generate_with_multishot_conditioning(
        self,
        prev_latent: Optional[torch.Tensor],
        use_sos_conditioning: bool,
        **generation_kwargs
    ):
        """
        Generate video with multi-shot conditioning logic.
        
        This method implements the same logic as SG_training_strategy.py:
        1. Create or use prev_latent (SOS token for first shot)
        2. Concat prev_latent + current latents
        3. Create conditioning mask (prev=True, curr=partially True)
        4. Run denoising only on current part
        """
        # Extract generation parameters
        height = generation_kwargs.get('height', 512)
        width = generation_kwargs.get('width', 704)
        num_frames = generation_kwargs.get('num_frames', 161)
        
        logger.info(f"Multi-shot validation generation: {height}x{width}x{num_frames}")
        logger.info("Getting pipeline device...")
        
        # Use a more direct approach to get device
        try:
            if hasattr(self.transformer, 'parameters'):
                device = next(iter(self.transformer.parameters())).device
                logger.info(f"Got device from transformer: {device}")
            else:
                device = torch.device('cuda:0')
                logger.info(f"Using default device: {device}")
        except Exception as e:
            logger.info(f"Device detection failed, using cuda:0: {e}")
            device = torch.device('cuda:0')
        
        # Calculate expected latent dimensions (consistent with training strategy)
        # VAE downsampling factors should match the actual model configuration
        vae_temporal_downsample = getattr(self.vae, 'temporal_downsample_factor', 7)  # Usually 7 for LTX
        vae_spatial_downsample = getattr(self.vae, 'spatial_downsample_factor', 32)   # Usually 32 for LTX
        
        latent_frames = num_frames // vae_temporal_downsample + 1
        latent_height = height // vae_spatial_downsample
        latent_width = width // vae_spatial_downsample
        curr_seq_len = latent_frames * latent_height * latent_width
        
        logger.info(f"Calculated latent dimensions: frames={latent_frames}, h={latent_height}, w={latent_width}, seq_len={curr_seq_len}")
        
        # 1. Prepare prev_latent
        if use_sos_conditioning or prev_latent is None:
            # Generate SOS token conditioning
            logger.info("Generating SOS token conditioning for multi-shot")
            logger.info(f"SOS generator device: {getattr(self.sos_token_generator, 'device', 'unknown')}")
            # Ensure SOS token generator is on correct device
            if hasattr(self.sos_token_generator, 'to'):
                self.sos_token_generator = self.sos_token_generator.to(device)
                logger.info(f"Moved SOS generator to {device}")
            prev_latent = self.sos_token_generator(curr_seq_len, device=device)
            
        # Ensure prev_latent is on correct device and has batch dimension
        if prev_latent.dim() == 2:  # [seq_len, channels]
            prev_latent = prev_latent.unsqueeze(0)  # [1, seq_len, channels]
            logger.info(f"Added batch dimension to prev_latent: {prev_latent.shape}")
        prev_latent = prev_latent.to(device)
        
        prev_seq_len = prev_latent.shape[1]
        total_seq_len = prev_seq_len + curr_seq_len
        
        logger.info(f"Multi-shot conditioning: prev_seq={prev_seq_len}, curr_seq={curr_seq_len}, total={total_seq_len}")
        
        # 2. Create conditioning mask - simplified approach
        # prev part: all True (keep previous shot unchanged)
        # curr part: all False (allow full denoising for current shot)
        batch_size = 1  # Single video generation
        conditioning_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.bool, device=device)
        
        # Mark prev part as fully conditioned (will be preserved during denoising)
        conditioning_mask[:, :prev_seq_len] = True
        
        # Current part remains False to allow normal denoising
        # This ensures the current shot is generated normally while prev shot provides context
        
        logger.info(f"Conditioning strategy: prev={prev_seq_len} tokens fixed, curr={curr_seq_len} tokens denoising")
        
        # 3. Prepare latents for generation
        # We'll modify the standard generation to use our custom conditioning
        
        # Create modified generation arguments
        modified_kwargs = generation_kwargs.copy()
        modified_kwargs.update({
            'custom_prev_latent': prev_latent,
            'custom_conditioning_mask': conditioning_mask,
            'custom_curr_seq_len': curr_seq_len,
        })
        
        # Implement actual multi-shot denoising with prev conditioning
        logger.info("Running multi-shot generation with prev latent conditioning")
        
        # Create a custom generation call that includes prev conditioning
        return self._generate_with_prev_conditioning(
            prev_latent=prev_latent,
            conditioning_mask=conditioning_mask,
            curr_seq_len=curr_seq_len,
            **{k: v for k, v in modified_kwargs.items() 
               if k not in ['custom_prev_latent', 'custom_conditioning_mask', 'custom_curr_seq_len']}
        )
    
    def _generate_with_prev_conditioning(
        self,
        prev_latent: torch.Tensor,
        conditioning_mask: torch.Tensor,
        curr_seq_len: int,
        **kwargs
    ):
        """
        Generate video with previous latent conditioning.
        
        This method modifies the standard generation pipeline to:
        1. Initialize latents with prev_latent + current noise
        2. Apply conditioning mask during denoising
        3. Only denoise the current part while keeping prev part fixed
        """
        from diffusers.utils.torch_utils import randn_tensor
        
        # Extract parameters
        height = kwargs.get('height', 512)
        width = kwargs.get('width', 704) 
        num_frames = kwargs.get('num_frames', 161)
        num_inference_steps = kwargs.get('num_inference_steps', 50)
        guidance_scale = kwargs.get('guidance_scale', 3.0)
        generator = kwargs.get('generator', None)
        prompt = kwargs.get('prompt', "")
        negative_prompt = kwargs.get('negative_prompt', "")
        device = self.device
        
        logger.info("Starting prompt encoding...")
        # Prepare prompt embeddings
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                device=device,
                num_videos_per_prompt=1,
                max_sequence_length=kwargs.get('max_sequence_length', 256)
            )
        )
        logger.info(f"Using CFG: {guidance_scale > 1.0}, guidance_scale: {guidance_scale}")
        
        # Calculate latent dimensions (consistent with earlier calculations)
        vae_temporal_downsample = getattr(self.vae, 'temporal_downsample_factor', 7)
        vae_spatial_downsample = getattr(self.vae, 'spatial_downsample_factor', 32)
        
        latent_frames = num_frames // vae_temporal_downsample
        latent_height = height // vae_spatial_downsample
        latent_width = width // vae_spatial_downsample
        latent_shape = (1, latent_frames * latent_height * latent_width, self.transformer.config.in_channels)
        logger.info(f"  F : {num_frames}, H : {height}, W : {width}")
        # Initialize current latents with noise
        curr_latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        
        # Combine prev + current latents
        try:
            combined_latents, prev_seq_len, _ = self._concat_prev_curr(prev_latent, curr_latents)
        except Exception as e:
            logger.error(f"Failed to concat prev and curr latents: {e}")
            logger.error(f"Latent dimension mismatch error")
            raise
        
        # Set up scheduler
        logger.info(f"Setting up scheduler for {num_inference_steps} steps")
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        logger.info(f"Timesteps prepared: {len(timesteps)} steps, first: {timesteps[0]}, last: {timesteps[-1]}")
        
        # Denoising loop
        logger.info("Starting denoising loop...")
        for step_idx, t in enumerate(timesteps):
            if step_idx % 50 == 0 or step_idx < 5:  # Log every 50 steps and first 5 steps
                logger.info(f"Denoising step {step_idx + 1}/{len(timesteps)}, t={t.item():.4f}")
            
            # Create timestep tensor
            timestep = t.expand(combined_latents.shape[0])
            if guidance_scale > 1.0:
                # Duplicate latents for CFG
                latent_model_input = torch.cat([combined_latents] * 2)
                timestep = torch.cat([timestep] * 2)
                # Prepare prompt embeddings for CFG
                encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds])
                encoder_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            else:
                latent_model_input = combined_latents
                encoder_hidden_states = prompt_embeds
                encoder_attention_mask = prompt_attention_mask
                
             
            # Create video coordinates for transformer
            # For combined latents (prev + curr), we need proper coordinate handling
            total_seq_len = combined_latents.shape[1]
            
            # Generate coordinates that match the actual combined sequence length
            video_coords = self._prepare_video_ids(
                latent_model_input.shape[0],
                latent_frames,
                latent_height,
                latent_width,
                patch_size=self.transformer_spatial_patch_size,
                patch_size_t=self.transformer_temporal_patch_size,
                device=device,
            )
            
            # If combined sequence is longer than base coordinates, extend them properly
            if total_seq_len > video_coords.shape[-1]:
                # Create additional coordinates for the prev sequence part
                additional_coords = video_coords.clone()  # Same pattern for consistency
                video_coords = torch.cat([additional_coords, video_coords], dim=-1)
            
            # Ensure coordinates match the exact sequence length
            if video_coords.shape[-1] != total_seq_len:
                # Trim or extend to match exact length
                if video_coords.shape[-1] > total_seq_len:
                    video_coords = video_coords[..., :total_seq_len]
                else:
                    # Pad with zeros if needed (shouldn't happen in normal cases)
                    padding = total_seq_len - video_coords.shape[-1]
                    pad_coords = torch.zeros(*video_coords.shape[:-1], padding, device=device, dtype=video_coords.dtype)
                    video_coords = torch.cat([video_coords, pad_coords], dim=-1)
            
            video_coords = self._scale_video_ids(
                video_coords,
                scale_factor=self.vae_spatial_compression_ratio,
                scale_factor_t=self.vae_temporal_compression_ratio,
                frame_index=0,
                device=device,
            ).float()
            
            # Predict noise
            if step_idx == 0:  # Log detailed info for first step
                pass
                
            try:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep.unsqueeze(-1).float(),
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    video_coords=video_coords,
                    return_dict=False,
                )[0]
                
                if step_idx == 0:
                    pass
                    
            except Exception as e:
                logger.error(f"Transformer forward failed at step {step_idx}: {e}")
                raise
            
            # Apply CFG
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            combined_latents = self.scheduler.step(noise_pred, t, combined_latents, generator=generator)[0]
            
            # Apply conditioning: keep prev part unchanged
            combined_latents = self._apply_conditioning_mask(
                combined_latents, prev_latent, conditioning_mask, prev_seq_len
            )
        
        logger.info("Denoising completed. Starting decoding...")
        
        # Extract only the current part for decoding
        current_part = combined_latents[:, prev_seq_len:prev_seq_len + curr_seq_len]
        
        # First unpack latents to proper VAE format
        logger.info("Unpacking latents...")
        current_part = self._unpack_latents(
            current_part,
            latent_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        
        # Denormalize latents using VAE normalization parameters
        logger.info("Denormalizing latents...")
        current_part = self._denormalize_latents(
            current_part, 
            self.vae.latents_mean, 
            self.vae.latents_std,
            self.vae.config.scaling_factor
        )
        
        # Decode to video with proper timestep handling
        logger.info("Starting VAE decoding...")
        
        # Ensure correct dtype for VAE
        current_part = current_part.to(prompt_embeds.dtype)
        
        try:
            # Handle timestep conditioning if needed
            timestep = None
            if hasattr(self.vae.config, 'timestep_conditioning') and self.vae.config.timestep_conditioning:
                # Use decode_timestep if available, otherwise default to 0.0
                decode_timestep = kwargs.get('decode_timestep', 0.0)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep]
                timestep = torch.tensor(decode_timestep, device=device, dtype=current_part.dtype)
                
                # Apply decode noise scale if specified
                decode_noise_scale = kwargs.get('decode_noise_scale')
                if decode_noise_scale is not None:
                    if not isinstance(decode_noise_scale, list):
                        decode_noise_scale = [decode_noise_scale]
                    decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=current_part.dtype)[:, None, None, None, None]
                    noise = torch.randn_like(current_part)
                    current_part = (1 - decode_noise_scale) * current_part + decode_noise_scale * noise
            
            logger.info(f"VAE decode with timestep: {timestep}")
            video_frames = self.vae.decode(current_part, timestep, return_dict=False)[0]
            
            # Post-process video
            logger.info("Post-processing video...")
            output_type = kwargs.get('output_type', 'pil')
            video_frames = self.video_processor.postprocess_video(video_frames, output_type=output_type)
            logger.info(f"Post-processed video frames: {len(video_frames) if isinstance(video_frames, list) else 'single tensor'}")
            
        except Exception as e:
            logger.exception("VAE decoding failed") 
            logger.error(f"VAE decoding failed: {e}")
            raise
        
        # Return in the same format as parent pipeline
        from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
        return LTXPipelineOutput(frames=video_frames)
    
    def _apply_conditioning_mask(
        self,
        combined_latents: torch.Tensor,
        prev_latent: torch.Tensor, 
        conditioning_mask: torch.Tensor,
        prev_seq_len: int
    ) -> torch.Tensor:
        """Apply conditioning mask to keep prev part unchanged."""
        # Ensure prev_latent has the same batch dimension
        if prev_latent.shape[0] != combined_latents.shape[0]:
            prev_latent = prev_latent.expand(combined_latents.shape[0], -1, -1)
        
        # Apply mask: where mask is True, use original prev_latent
        # Note: conditioning_mask is passed but we use prev_seq_len for simplicity
        # The prev part should always be preserved
        combined_latents[:, :prev_seq_len] = prev_latent
        
        return combined_latents
    
    def _concat_prev_curr(
        self, 
        prev_latent: torch.Tensor, 
        curr_latent: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """
        Concat prev and current latents along sequence dimension.
        
        Args:
            prev_latent: [B, Pseq, D] 
            curr_latent: [B, Cseq, D]
            
        Returns:
            concat_latent: [B, Pseq+Cseq, D]
            prev_seq_len: Pseq
            curr_seq_len: Cseq
        """
        from ltxv_trainer import logger
        
        if prev_latent is None:
            return curr_latent, 0, curr_latent.shape[1]
        
        
        # Ensure both tensors have the same number of dimensions
        if prev_latent.dim() != curr_latent.dim():
            logger.error(f"Dimension mismatch: prev_latent.dim()={prev_latent.dim()}, curr_latent.dim()={curr_latent.dim()}")
            # Try to fix dimension mismatch
            if prev_latent.dim() == 2 and curr_latent.dim() == 3:
                prev_latent = prev_latent.unsqueeze(0)
            elif prev_latent.dim() == 3 and curr_latent.dim() == 2:
                curr_latent = curr_latent.unsqueeze(0)
        
        # Ensure batch dimensions match
        if prev_latent.shape[0] != curr_latent.shape[0]:
            logger.warning(f"Batch dimension mismatch detected")
            # Expand the smaller batch to match
            if prev_latent.shape[0] == 1 and curr_latent.shape[0] > 1:
                prev_latent = prev_latent.expand(curr_latent.shape[0], -1, -1)
            elif curr_latent.shape[0] == 1 and prev_latent.shape[0] > 1:
                curr_latent = curr_latent.expand(prev_latent.shape[0], -1, -1)
        
        # Ensure channel dimensions match
        if prev_latent.shape[-1] != curr_latent.shape[-1]:
            logger.error(f"Channel dimension mismatch detected")
            raise ValueError(f"Channel dimensions must match: {prev_latent.shape[-1]} != {curr_latent.shape[-1]}")
        
        prev_seq_len = prev_latent.shape[1]
        curr_seq_len = curr_latent.shape[1]
        
        concat_latent = torch.cat([prev_latent, curr_latent], dim=1)
        
        return concat_latent, prev_seq_len, curr_seq_len
    
    def encode_video_to_latent(self, video) -> torch.Tensor:
        """
        Encode video to latent space for use as prev_latent conditioning.
        
        Args:
            video: Video tensor, PIL images, or video frames
            
        Returns:
            Latent tensor in sequence format [seq_len, channels] for conditioning
        """
        with torch.no_grad():
            # Handle different input types (PIL images, etc.)
            if isinstance(video, list):
                # Convert PIL images to tensor
                import numpy as np
                from PIL import Image
                frames = []
                for frame in video:
                    if isinstance(frame, Image.Image):
                        frame_array = np.array(frame).transpose(2, 0, 1)  # HWC -> CHW
                        frames.append(torch.from_numpy(frame_array))
                video = torch.stack(frames, dim=0)  # [frames, channels, height, width]
            
            # Ensure video has batch dimension and correct format for VAE: [B, C, F, H, W]
            if video.dim() == 4:  # [frames, channels, height, width]
                video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, frames, channels, height, width] -> [1, channels, frames, height, width]
            
            # Move to device and normalize
            device = self.device
            video = video.to(device, dtype=torch.float32)
            
            # Normalize to [-1, 1] based on input range
            if video.max() > 1.0:  # PIL images are typically 0-255
                video = video / 255.0  # [0, 255] -> [0, 1]
                video = video * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            elif video.min() >= 0.0 and video.max() <= 1.0:  # Already in [0, 1]
                video = video * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            
            # Encode using VAE
            from ltxv_trainer.ltxv_pipeline import retrieve_latents
            latent_dist = self.vae.encode(video)
            latent = retrieve_latents(latent_dist)
            
            # Normalize latents like in the pipeline
            latent = self._normalize_latents(
                latent, self.vae.latents_mean, self.vae.latents_std
            )
                
            # Convert to sequence format [seq_len, channels]
            batch, latent_channels, latent_frames, latent_height, latent_width = latent.shape
            seq_len = latent_frames * latent_height * latent_width
            latent = latent.view(batch, latent_channels, seq_len).permute(0, 2, 1).squeeze(0)
            
            return latent