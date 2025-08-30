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
from torch import Tensor
from copy import deepcopy

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
        **kwargs
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            **kwargs
        )
        # SOS token generator for first shot
        self.sos_token_generator = sos_token_generator or SOSTokenLatents(d_model=128)
        
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
        device = self.device
        
        # Calculate expected latent dimensions
        latent_frames = num_frames // 4  # VAE temporal downsampling
        latent_height = height // 8      # VAE spatial downsampling  
        latent_width = width // 8
        curr_seq_len = latent_frames * latent_height * latent_width
        
        # 1. Prepare prev_latent
        if use_sos_conditioning or prev_latent is None:
            # Generate SOS token conditioning
            logger.info("Generating SOS token conditioning for multi-shot")
            prev_latent = self.sos_token_generator(curr_seq_len, device=device)
            
        # Ensure prev_latent is on correct device and has batch dimension
        if prev_latent.dim() == 2:  # [seq_len, channels]
            prev_latent = prev_latent.unsqueeze(0)  # [1, seq_len, channels]
        prev_latent = prev_latent.to(device)
        
        prev_seq_len = prev_latent.shape[1]
        total_seq_len = prev_seq_len + curr_seq_len
        
        logger.info(f"Multi-shot conditioning: prev_seq={prev_seq_len}, curr_seq={curr_seq_len}, total={total_seq_len}")
        
        # 2. Create conditioning mask
        # prev part: all True (fully conditioned)
        # curr part: first frame True (partial conditioning), rest False  
        batch_size = 1  # Single video generation
        conditioning_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.bool, device=device)
        
        # Mark prev part as fully conditioned
        conditioning_mask[:, :prev_seq_len] = True
        
        # Mark first frame of curr part as conditioned (optional)
        first_frame_tokens = latent_height * latent_width
        curr_start = prev_seq_len
        curr_first_frame_end = min(curr_start + first_frame_tokens, total_seq_len)
        # conditioning_mask[:, curr_start:curr_first_frame_end] = True  # Optional first frame conditioning
        
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
        
        # Prepare prompt embeddings
        prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                device=device,
                num_videos_per_prompt=1,
                max_sequence_length=kwargs.get('max_sequence_length', 256)
            )
        )
        
        # Calculate latent dimensions (use standard LTX-Video scaling factors)
        vae_scale_factor_temporal = getattr(self.vae, 'temporal_scale_factor', 8)
        vae_scale_factor_spatial = getattr(self.vae, 'spatial_scale_factor', 8)
        
        latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        latent_shape = (1, latent_frames * latent_height * latent_width, self.transformer.config.in_channels)
        
        # Initialize current latents with noise
        curr_latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        
        # Combine prev + current latents
        combined_latents, prev_seq_len, _ = self._concat_prev_curr(prev_latent, curr_latents)
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for i, t in enumerate(timesteps):
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
            
            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=latent_frames,
                height=latent_height,
                width=latent_width,
                return_dict=False,
            )[0]
            
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
        
        # Extract only the current part for decoding
        current_part = combined_latents[:, prev_seq_len:prev_seq_len + curr_seq_len]
        
        # Reshape for VAE decoding
        current_part = current_part.view(1, self.transformer.config.in_channels, latent_frames, latent_height, latent_width)
        
        # Decode to video
        video_frames = self.vae.decode(current_part, return_dict=False)[0]
        video_frames = self.video_processor.postprocess_video(video_frames, output_type='pil')
        
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
        if prev_latent is None:
            return curr_latent, 0, curr_latent.shape[1]
        
        prev_seq_len = prev_latent.shape[1]
        curr_seq_len = curr_latent.shape[1]
        
        concat_latent = torch.cat([prev_latent, curr_latent], dim=1)
        return concat_latent, prev_seq_len, curr_seq_len
    
    def encode_video_to_latent(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space for use as prev_latent conditioning.
        
        Args:
            video: Video tensor [frames, channels, height, width] or [batch, frames, channels, height, width]
            
        Returns:
            Latent tensor in sequence format [seq_len, channels] for conditioning
        """
        with torch.no_grad():
            # Ensure video has batch dimension
            if video.dim() == 4:  # [frames, channels, height, width]
                video = video.unsqueeze(0)  # [1, frames, channels, height, width]
            
            # Move to device and normalize
            video = video.to(self.device, dtype=torch.float32)
            
            # Normalize to [-1, 1] if needed
            if video.min() >= 0.0 and video.max() <= 1.0:
                video = video * 2.0 - 1.0
            
            # Encode using VAE
            latent_dist = self.vae.encode(video)
            
            if hasattr(latent_dist, 'sample'):
                latent = latent_dist.sample()
            elif hasattr(latent_dist, 'latent_dist'):
                latent = latent_dist.latent_dist.sample()  
            else:
                latent = latent_dist
                
            # Convert to sequence format [seq_len, channels]
            batch, latent_channels, latent_frames, latent_height, latent_width = latent.shape
            seq_len = latent_frames * latent_height * latent_width
            latent = latent.view(batch, latent_channels, seq_len).permute(0, 2, 1).squeeze(0)
            
            return latent