# ruff: noqa

# Copyright 2024 Lightricks and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime

import PIL.Image
import numpy as np
import torch
from ltxv_trainer.ltxv_utils import decode_video
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers.models.transformers import LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import T5EncoderModel, T5TokenizerFast
from torchvision.transforms.functional import center_crop, resize

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
        >>> from diffusers.utils import export_to_video, load_video, load_image

        >>> pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> # Load input image and video
        >>> video = load_video(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
        ... )
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
        ... )

        >>> # Create conditioning objects
        >>> condition1 = LTXVideoCondition(
        ...     image=image,
        ...     frame_index=0,
        ... )
        >>> condition2 = LTXVideoCondition(
        ...     video=video,
        ...     frame_index=80,
        ... )

        >>> prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
        >>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        >>> # Generate video
        >>> generator = torch.Generator("cuda").manual_seed(0)
        >>> # Text-only conditioning is also supported without the need to pass `conditions`
        >>> video = pipe(
        ...     conditions=[condition1, condition2],
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     width=768,
        ...     height=512,
        ...     num_frames=161,
        ...     num_inference_steps=40,
        ...     generator=generator,
        ... ).frames[0]

        >>> export_to_video(video, "output.mp4", fps=24)
        ```
"""

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

@dataclass
class LTXVideoCondition:
    """
    Defines a single frame-conditioning item for LTX Video - a single frame or a sequence of frames.

    Attributes:
        image (`PIL.Image.Image`):
            The image to condition the video on.
        video (`List[PIL.Image.Image]`):
            The video to condition the video on.
        frame_index (`int`):
            The frame index at which the image or video will conditionally effect the video generation.
        strength (`float`, defaults to `1.0`):
            The strength of the conditioning effect. A value of `1.0` means the conditioning effect is fully applied.
    """

    image: Optional[PIL.Image.Image] = None
    video: Optional[List[PIL.Image.Image]] = None
    frame_index: int = 0
    strength: float = 1.0


# from LTX-Video/ltx_video/schedulers/rf.py
def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps




from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor

from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline, LTXVideoCondition
from ltxv_trainer.SG_datasets import SOSTokenLatents
from ltxv_trainer import logger
from ltxv_trainer.token_utils import (
    VideoTokenEmbeddings,
    preprocess_multishot_with_tokens,
    create_token_aware_conditioning_mask,
    extract_token_positions
)


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
        use_tokens: bool = True,
        hidden_dim: int = 128,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )

        # Initialize debug hooks using object.__setattr__ to bypass diffusers restrictions
        object.__setattr__(self, '_debug_hooks_enabled', False)
        object.__setattr__(self, '_debug_latents', {})

        # Initialize token embeddings for multi-shot video generation
        self.use_tokens = use_tokens
        if self.use_tokens:
            self.token_embeddings = VideoTokenEmbeddings(hidden_dim=hidden_dim)
            logger.info(f"SGMultiShotPipeline: Initialized video token embeddings with hidden_dim={hidden_dim}")
        else:
            self.token_embeddings = None
            logger.info("SGMultiShotPipeline: Token embeddings disabled")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        # Initialize prompt info storage for JSON export
        self.prompt_info = {}

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if self.transformer is not None else 1
        )
        # No longer using SOS token generator - using Gaussian noise instead
        self.sos_token_generator = None
        
        # Initialize video processor if not already done by parent
        if not hasattr(self, 'video_processor') or self.video_processor is None:
            vae_scale_factor = getattr(self.vae, 'spatial_compression_ratio', 32)
            self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)

    def enable_debug_hooks(self):
        """Enable debug hooks to capture input/output latents."""
        object.__setattr__(self, '_debug_hooks_enabled', True)
        object.__setattr__(self, '_debug_latents', {})
        # Also enable unified debug capture
        self.enable_debug_capture()

    def disable_debug_hooks(self):
        """Disable debug hooks."""
        object.__setattr__(self, '_debug_hooks_enabled', False)
        object.__setattr__(self, '_debug_latents', {})
        # Also disable unified debug capture
        self.disable_debug_capture()

    def get_debug_latents(self) -> dict:
        """Get captured debug latents."""
        return getattr(self, '_debug_latents', {}).copy()

    def _capture_debug_latents(self, key: str, latents: torch.Tensor):
        """Capture latents for debugging if hooks are enabled."""
        if getattr(self, '_debug_hooks_enabled', False) and latents is not None:
            debug_latents = getattr(self, '_debug_latents', {})
            debug_latents[key] = latents.detach().cpu()
            object.__setattr__(self, '_debug_latents', debug_latents)
        
    @property 
    def device(self):
        """Get the device of the pipeline."""
        # Use override device if set
        if hasattr(self, '_device_override') and self._device_override is not None:
            return self._device_override
        
        # Try to get device from transformer (most reliable)
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'device'):
            return self.transformer.device
            
        # Fallback to cuda:0
        return torch.device('cuda:0')

    def check_inputs(
        self,
        prompt=None,
        conditions=None,
        image=None,
        video=None,
        frame_index=None,
        strength=None,
        height=None,
        width=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        scenario=None,  # Add support for scenario
        **kwargs,
    ):
        """
        Override parent check_inputs to handle reference_latents parameter.
        """
        # Manually build parent_kwargs to avoid locals() issues
        parent_kwargs = {}

        # Add only the parameters that parent class expects
        if prompt is not None:
            parent_kwargs['prompt'] = prompt

        # Provide default values for required parameters that might be None
        parent_kwargs['conditions'] = conditions if conditions is not None else []
        parent_kwargs['image'] = image if image is not None else None
        parent_kwargs['video'] = video if video is not None else None

        # These are required positional arguments for the parent class
        parent_kwargs['frame_index'] = frame_index if frame_index is not None else 0
        parent_kwargs['strength'] = strength if strength is not None else 1.0
        if height is not None:
            parent_kwargs['height'] = height
        if width is not None:
            parent_kwargs['width'] = width
        if callback_on_step_end_tensor_inputs is not None:
            parent_kwargs['callback_on_step_end_tensor_inputs'] = callback_on_step_end_tensor_inputs
        if prompt_embeds is not None:
            parent_kwargs['prompt_embeds'] = prompt_embeds
        if negative_prompt_embeds is not None:
            parent_kwargs['negative_prompt_embeds'] = negative_prompt_embeds
        if prompt_attention_mask is not None:
            parent_kwargs['prompt_attention_mask'] = prompt_attention_mask
        if negative_prompt_attention_mask is not None:
            parent_kwargs['negative_prompt_attention_mask'] = negative_prompt_attention_mask

        # Add other kwargs (excluding scenario)
        for k, v in kwargs.items():
            if k not in ['scenario']:
                parent_kwargs[k] = v

        # Call parent class check_inputs
        super().check_inputs(**parent_kwargs)

        # Additional validation for scenario if needed
        if scenario is not None:
            if not isinstance(scenario, str):
                raise TypeError("scenario must be a string")
            if not scenario.strip():
                raise ValueError("scenario cannot be empty")

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            if isinstance(prompt_embeds, dict):
                batch_size = prompt_embeds["prompt_embeds"].shape[0]
                prompt_attention_mask = prompt_embeds.get("prompt_attention_mask", prompt_attention_mask)
                prompt_embeds = prompt_embeds["prompt_embeds"]
            elif isinstance(prompt_embeds, torch.Tensor):
                batch_size = prompt_embeds.shape[0]


        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @staticmethod
    def _prepare_video_ids(
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, num_frames, patch_size_t, device=device),
            torch.arange(0, height, patch_size, device=device),
            torch.arange(0, width, patch_size, device=device),
            indexing="ij",
        )
        latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)

        return latent_coords

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents


    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Safety check: ensure latents has proper shape
        if latents.numel() == 0:
            raise ValueError("Cannot denormalize empty latents tensor")

        if latents.dim() != 5:
            raise ValueError(f"Expected latents to be 5D [B, C, F, H, W], got shape {latents.shape}")

        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)

        # Handle channel dimension mismatch due to patching
        original_channels = latents_mean.size(1)
        latent_channels = latents.size(1)

        if latent_channels != original_channels:
            # Check if latent channels are a multiple of original channels (due to patching)
            if latent_channels % original_channels == 0:
                # Expand mean and std to match latent channels by repeating
                repeat_factor = latent_channels // original_channels
                expanded_mean = latents_mean.repeat(1, repeat_factor, 1, 1, 1)
                expanded_std = latents_std.repeat(1, repeat_factor, 1, 1, 1)
                latents_mean = expanded_mean
                latents_std = expanded_std
            else:
                raise ValueError(f"Channel dimension mismatch: latents has {latent_channels} channels, "
                               f"but mean/std have {original_channels} channels and are not compatible")

        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def trim_conditioning_sequence(self, start_frame: int, sequence_num_frames: int, target_num_frames: int):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.
        Returns:
            int: updated sequence length
        """ 
        scale_factor = self.vae_temporal_compression_ratio
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames


    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents. This helps with motion continuity, especially
        when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        if conditioning_mask is None:
            # If no conditioning mask, don't add noise
            return latents
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents


    def prepare_latents(
        self,
        conditions: Optional[List[torch.Tensor]] = None,
        condition_strength: Optional[List[float]] = None,
        condition_frame_index: Optional[List[int]] = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        num_prefix_latent_frames: int = 2,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if len(conditions) > 0:
            condition_latent_frames_mask = torch.zeros(
                (batch_size, num_latent_frames), device=device, dtype=torch.float32
            )

            extra_conditioning_latents = []
            extra_conditioning_video_ids = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0
            for data, strength, frame_index in zip(conditions, condition_strength, condition_frame_index, strict=False):
                condition_latents = retrieve_latents(self.vae.encode(data), generator=generator)
                # Apply standard VAE scaling without custom normalization
                condition_latents = condition_latents * self.vae.config.scaling_factor
                condition_latents = condition_latents.to(device, dtype=dtype)

                num_data_frames = data.size(2)
                num_cond_frames = condition_latents.size(2)

                if frame_index == 0:
                    latents[:, :, :num_cond_frames] = torch.lerp(
                        latents[:, :, :num_cond_frames], condition_latents, strength
                    )
                    condition_latent_frames_mask[:, :num_cond_frames] = strength

                else:
                    if num_data_frames > 1:
                        if num_cond_frames < num_prefix_latent_frames:
                            raise ValueError(
                                f"Number of latent frames must be at least {num_prefix_latent_frames} but got {num_data_frames}."
                            )

                        if num_cond_frames > num_prefix_latent_frames:
                            start_frame = frame_index // self.vae_temporal_compression_ratio + num_prefix_latent_frames
                            end_frame = start_frame + num_cond_frames - num_prefix_latent_frames
                            latents[:, :, start_frame:end_frame] = torch.lerp(
                                latents[:, :, start_frame:end_frame],
                                condition_latents[:, :, num_prefix_latent_frames:],
                                strength,
                            )
                            condition_latent_frames_mask[:, start_frame:end_frame] = strength
                            condition_latents = condition_latents[:, :, :num_prefix_latent_frames]

                    noise = randn_tensor(condition_latents.shape, generator=generator, device=device, dtype=dtype)
                    condition_latents = torch.lerp(noise, condition_latents, strength)

                    condition_video_ids = self._prepare_video_ids(
                        batch_size,
                        condition_latents.size(2),
                        latent_height,
                        latent_width,
                        patch_size=self.transformer_spatial_patch_size,
                        patch_size_t=self.transformer_temporal_patch_size,
                        device=device,
                    )
                    condition_video_ids = self._scale_video_ids(
                        condition_video_ids,
                        scale_factor=self.vae_spatial_compression_ratio,
                        scale_factor_t=self.vae_temporal_compression_ratio,
                        frame_index=frame_index,
                        device=device,
                    )
                    condition_latents = self._pack_latents(
                        condition_latents,
                        self.transformer_spatial_patch_size,
                        self.transformer_temporal_patch_size,
                    )
                    condition_conditioning_mask = torch.full(
                        condition_latents.shape[:2], strength, device=device, dtype=dtype
                    )

                    extra_conditioning_latents.append(condition_latents)
                    extra_conditioning_video_ids.append(condition_video_ids)
                    extra_conditioning_mask.append(condition_conditioning_mask)
                    extra_conditioning_num_latents += condition_latents.size(1)

        video_ids = self._prepare_video_ids(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=device,
        )
        if len(conditions) > 0:
            conditioning_mask = condition_latent_frames_mask.gather(1, video_ids[:, 0])
        else:
            conditioning_mask, extra_conditioning_num_latents = None, 0
        video_ids = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_spatial_compression_ratio,
            scale_factor_t=self.vae_temporal_compression_ratio,
            frame_index=0,
            device=device,
        )
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        if len(conditions) > 0 and len(extra_conditioning_latents) > 0:
            latents = torch.cat([*extra_conditioning_latents, latents], dim=1)
            video_ids = torch.cat([*extra_conditioning_video_ids, video_ids], dim=2)
            conditioning_mask = torch.cat([*extra_conditioning_mask, conditioning_mask], dim=1)

        return latents, conditioning_mask, video_ids, extra_conditioning_num_latents

    def prepare_latents_with_scenario(
        self,
        scenario: Optional[str] = None,
        shot_latents: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepare latents with scenario-based reference handling similar to training strategy.
        This method integrates reference latent preparation into the scenario processing logic.

        Args:
            scenario: Scenario string for multi-shot video generation
            shot_latents: Dictionary containing reference shot latents
            batch_size: Batch size
            num_channels_latents: Number of latent channels
            height: Video height
            width: Video width
            num_frames: Number of frames
            generator: Random generator
            device: Device
            dtype: Data type

        Returns:
            Tuple of (latents, conditioning_mask, video_coords, extra_conditioning_num_latents)
        """
        # Calculate latent dimensions
        num_latent_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        # Generate initial latents (current shot)
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        current_latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # If no scenario provided, return standard latents
        if scenario is None or not self.use_tokens:
            # Pack latents and return standard format
            packed_latents = self._pack_latents(
                current_latents,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size
            )

            # No conditioning mask
            conditioning_mask = torch.zeros(
                batch_size, packed_latents.shape[1], dtype=torch.bool, device=device
            )

            # Prepare video coordinates
            video_coords = self._prepare_video_ids(
                batch_size, num_latent_frames, latent_height, latent_width,
                self.transformer_spatial_patch_size, self.transformer_temporal_patch_size, device
            )
            video_coords = self._scale_video_ids(
                video_coords, self.vae_spatial_compression_ratio,
                self.vae_temporal_compression_ratio, 0, device
            )

            return packed_latents, conditioning_mask, video_coords, 0

        # Scenario-based processing with reference handling
        logger.info(f"🎭 Preparing latents with scenario: '{scenario}'")

        # Prepare shot latents dictionary
        scenario_shot_latents = {}

        # Convert current latents to shot format [F, H, W, C]
        current_shot = current_latents[0].permute(1, 2, 3, 0)  # [C, F, H, W] -> [F, H, W, C]

        # # Determine shot names based on available shot_latents
        # if shot_latents:
        #     # Use provided shot latents - copy all available shots
        #     scenario_shot_latents.update(shot_latents)
        #     logger.info(f"🎭 Using provided shot latents: {list(shot_latents.keys())}")

        #     # Add current shot as the last shot in sequence
        #     shot_names = sorted([name for name in shot_latents.keys() if name.startswith('shot')])
        #     if shot_names:
        #         # Extract shot numbers and find the next one
        #         shot_numbers = [int(name.replace('shot', '')) for name in shot_names if name.replace('shot', '').isdigit()]
        #         if shot_numbers:
        #             current_shot_name = f"shot{shot_numbers[-1]}"
        #         else:
        #             current_shot_name = "shot2"  # Default if parsing fails
        #     else:
        #         current_shot_name = "shot2"  # Default
        # else:
        #     # No reference provided - generate two-shot sequence with SOS
        #     logger.info(f"🎭 Generating SOS token as reference for shot1")
        #     sos_reference = torch.randn_like(current_shot)
        #     scenario_shot_latents["shot1"] = sos_reference
        #     current_shot_name = "shot2"

        # scenario_shot_latents[current_shot_name] = current_shot
        # logger.info(f"🎭 Current shot assigned as: {current_shot_name}")
        # logger.info(f"🎭 total scenario : {[(k, v.shape) for k, v in scenario_shot_latents.items()]}")

        # Apply scenario preprocessing
        try:
            from ltxv_trainer.token_utils import preprocess_with_scenario, create_scenario_conditioning_mask

            tokenized_sequence, scenario_metadata = preprocess_with_scenario(
                shot_latents=shot_latents,
                scenario=scenario,
                token_embeddings=self.token_embeddings,
                device=device
            )

            # Convert tokenized sequence to packed format
            total_frames = tokenized_sequence.shape[0]
            seq_len = total_frames * latent_height * latent_width
            tokenized_flat = tokenized_sequence.view(total_frames, -1)  # (Total_F, H*W*C)
            tokenized_flat = tokenized_flat.view(seq_len, -1)  # (Total_F*H*W, C)
            packed_latents = tokenized_flat.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Total_F*H*W, C)

            # Create scenario conditioning mask
            bool_mask, strength_mask = create_scenario_conditioning_mask(
                metadata=scenario_metadata,
                conditioning_strength=1.0,
                stable_token_strength=0.5,
                transition_token_strength=0.8,
                target_shot="shot2",  # Use dynamic current shot name
                device=device
            )

            # Expand conditioning mask for spatial dimensions
            conditioning_mask = bool_mask.repeat_interleave(latent_height * latent_width)
            conditioning_mask = conditioning_mask.unsqueeze(0).expand(batch_size, -1)

            # Prepare video coordinates for the full tokenized sequence
            video_coords = self._prepare_video_ids(
                batch_size, total_frames, latent_height, latent_width,
                self.transformer_spatial_patch_size, self.transformer_temporal_patch_size, device
            )
            video_coords = self._scale_video_ids(
                video_coords, self.vae_spatial_compression_ratio,
                self.vae_temporal_compression_ratio, 0, device
            )

            # Calculate how many latents are from conditioning (reference) part
            if hasattr(scenario_metadata, 'shot_ranges'):
                prev_shot_range = scenario_metadata['shot_ranges'].get('prev_shot', (0, 0))
                extra_conditioning_num_latents = prev_shot_range[1] * latent_height * latent_width
            else:
                extra_conditioning_num_latents = num_latent_frames * latent_height * latent_width

            logger.info(f"🎭 Scenario latent preparation completed:")
            logger.info(f"🎭   Packed latents shape: {packed_latents.shape}")
            logger.info(f"🎭   Conditioning mask shape: {conditioning_mask.shape}")
            logger.info(f"🎭   Video coords shape: {video_coords.shape}")
            logger.info(f"🎭   Extra conditioning latents: {extra_conditioning_num_latents}")

            return packed_latents, conditioning_mask, video_coords, extra_conditioning_num_latents

        except Exception as e:
            logger.warning(f"🎭 Failed scenario latent preparation: {e}")
            logger.warning(f"🎭 Falling back to standard latent preparation")
            import traceback
            logger.warning(f"🎭 Scenario Debug: Traceback: {traceback.format_exc()}")


            # Fallback to standard processing
            packed_latents = self._pack_latents(
                current_latents,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size
            )

            conditioning_mask = torch.zeros(
                batch_size, packed_latents.shape[1], dtype=torch.bool, device=device
            )

            video_coords = self._prepare_video_ids(
                batch_size, num_latent_frames, latent_height, latent_width,
                self.transformer_spatial_patch_size, self.transformer_temporal_patch_size, device
            )
            video_coords = self._scale_video_ids(
                video_coords, self.vae_spatial_compression_ratio,
                self.vae_temporal_compression_ratio, 0, device
            )

            return packed_latents, conditioning_mask, video_coords, 0

    @property
    def guidance_scale(self):
        return self._guidance_scale

    def enable_debug_capture(self):
        """Enable debug latent capture for unified debug system."""
        self._debug_capture_enabled = True
        self._debug_valid_input_latents = None
        self._debug_valid_output_latents = None

    def disable_debug_capture(self):
        """Disable debug latent capture."""
        self._debug_capture_enabled = False
        self._debug_valid_input_latents = None
        self._debug_valid_output_latents = None

    def get_debug_latents(self):
        """Get captured debug latents."""
        return {
            'valid_input': getattr(self, '_debug_valid_input_latents', None),
            'valid_output': getattr(self, '_debug_valid_output_latents', None),
        }

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        conditions: Union[LTXVideoCondition, List[LTXVideoCondition]] = None,
        image: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        video: List[PipelineImageInput] = None,
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
        scenario: Optional[str] = None,
        output_reference_comparison: bool = False,
        return_latents: bool = False,
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
        shot_latents: Dict[str, torch.Tensor] = None,
        save_video: bool = True,
        output_dir: Optional[str] = None,
        video_filename: Optional[str] = None,
        fps: int = 24,
        validation_type: Optional[str] = None,
        step: Optional[int] = None,
        save_full_sequence: bool = False
    ):
        """
        Generate video using the SGMultiShotPipeline with scenario-based reference handling.

        Args:
            prompt: Text prompt for video generation
            scenario: Optional scenario string for token-based preprocessing (e.g., "shot1,stable,shot2,transition,shot3")
            shot_latents: Optional dictionary containing reference shot latents
            height: Video height
            width: Video width
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            generator: Random generator for reproducible results
            output_type: Output format ("pil" or "tensor")
            return_dict: Whether to return a dictionary
            save_video: Whether to save the generated video as MP4
            output_dir: Directory to save the video (defaults to current directory)
            video_filename: Filename for the saved video (auto-generated if not provided)
            fps: Frames per second for the saved video
            validation_type: Type of validation ("t2v_validation" or "v2v_validation")
            step: Training step number (included in filename if provided)
            save_full_sequence: Whether to save the full transformer output sequence (including reference tokens)

        Returns:
            Generated video frames

        Examples:
            # Basic generation
            video = pipeline("A cat playing in the garden", num_frames=161)

            # Multi-shot generation with scenario
            video = pipeline(
                prompt="A cat playing in the garden",
                scenario="shot1,stable,shot2,transition,shot3",
                shot_latents={"shot1": reference_latents}
            )

            # Generate and save video as MP4
            video = pipeline(
                prompt="A cat playing in the garden",
                num_frames=161,
                save_video=True,
                output_dir="./outputs",
                video_filename="cat_garden.mp4",
                fps=24
            )
        """
       

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if latents is not None:
            raise ValueError("Passing latents is not yet supported.")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            conditions=conditions,
            image=image,
            video=video,
            frame_index=frame_index,
            strength=strength,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            scenario=scenario,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None
        
        # logger.info(f"prompt embeds : {prompt_embeds}")


        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            if isinstance(prompt_embeds, dict):
                batch_size = prompt_embeds["prompt_embeds"].shape[0]
            elif isinstance(prompt_embeds, torch.Tensor):
                batch_size = prompt_embeds.shape[0]

        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions]

            strength = [condition.strength for condition in conditions]
            frame_index = [condition.frame_index for condition in conditions]
            image = [condition.image for condition in conditions]
            video = [condition.video for condition in conditions]
        elif image is not None or video is not None:
            if not isinstance(image, list):
                image = [image]
                num_conditions = 1
            elif isinstance(image, list):
                num_conditions = len(image)
            if not isinstance(video, list):
                video = [video]
                num_conditions = 1
            elif isinstance(video, list):
                num_conditions = len(video)

            if not isinstance(frame_index, list):
                frame_index = [frame_index] * num_conditions
            if not isinstance(strength, list):
                strength = [strength] * num_conditions

        device = self._execution_device

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        
        if self.do_classifier_free_guidance:
            logger.info(f"text prompt embeds : {prompt_embeds, negative_prompt_embeds}")
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        vae_dtype = self.vae.dtype

        conditioning_tensors = []
        is_conditioning_image_or_video = image is not None or video is not None
        if is_conditioning_image_or_video:
            for condition_image, condition_video, condition_frame_index, condition_strength in zip(
                image, video, frame_index, strength, strict=False
            ):
                if condition_image is not None:
                    condition_tensor = (
                        self.video_processor.preprocess(condition_image, height, width)
                        .unsqueeze(2)
                        .to(device, dtype=vae_dtype)
                    )
                elif condition_video is not None:
                    condition_tensor = self.video_processor.preprocess_video(condition_video, height, width)
                    num_frames_input = condition_tensor.size(2)
                    num_frames_output = self.trim_conditioning_sequence(
                        condition_frame_index, num_frames_input, num_frames
                    )
                    condition_tensor = condition_tensor[:, :, :num_frames_output]
                    condition_tensor = condition_tensor.to(device, dtype=vae_dtype)
                else:
                    raise ValueError("Either `image` or `video` must be provided for conditioning.")

                if condition_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                    raise ValueError(
                        f"Number of frames in the video must be of the form (k * {self.vae_temporal_compression_ratio} + 1) "
                        f"but got {condition_tensor.size(2)} frames."
                    )
                conditioning_tensors.append(condition_tensor)

        # 4. Prepare latent variables with scenario support
        num_channels_latents = self.transformer.config.in_channels

        # Use unified latent preparation method that handles scenarios
        if scenario is not None and self.use_tokens and self.token_embeddings is not None:
            # Use new scenario-based latent preparation
            latents, conditioning_mask, video_coords, extra_conditioning_num_latents = self.prepare_latents_with_scenario(
                scenario=scenario,
                shot_latents=shot_latents,  # Will be passed from caller if available
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            # Fall back to standard latent preparation with image/video conditioning
            latents, conditioning_mask, video_coords, extra_conditioning_num_latents = self.prepare_latents(
                conditioning_tensors,
                strength,
                frame_index,
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )

        # Initialize latents for conditioning if needed
        init_latents = latents.clone() if is_conditioning_image_or_video else None

        # Update video coordinates for classifier-free guidance
        if self.do_classifier_free_guidance:
            video_coords = torch.cat([video_coords, video_coords], dim=0)

        # 5. Prepare timesteps
        # latent dimensions already calculated above
        sigmas = linear_quadratic_schedule(num_inference_steps)
        timesteps = sigmas * 1000
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps=timesteps,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        if hasattr(self, '_debug_capture_enabled') and self._debug_capture_enabled:
            logger.info(f"inference input shape : {latents.shape}")
            self._debug_valid_input_latents = latents.detach().clone()
        else:
            logger.info(f"training inference input shape : {latents.shape}")


        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if image_cond_noise_scale > 0 and init_latents is not None:
                    # Add timestep-dependent noise to the hard-conditioning latents
                    # This helps with motion continuity, especially when conditioned on a single frame
                    latents = self.add_noise_to_image_conditioning_latents(
                        t / 1000.0,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        conditioning_mask,
                        generator,
                    )

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # Capture input latents for debugging (first timestep only)
                if i == 0:
                    self._capture_debug_latents("input", latents)
                if is_conditioning_image_or_video:
                    if conditioning_mask is not None:
                        conditioning_mask_model_input = (
                            torch.cat([conditioning_mask, conditioning_mask])
                            if self.do_classifier_free_guidance
                            else conditioning_mask
                        )
                    else:
                        conditioning_mask_model_input = None
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
                if is_conditioning_image_or_video:
                    if conditioning_mask_model_input is not None:
                        timestep = torch.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

                # Debug hook: store valid-input latents for unified debug system


                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    video_coords=video_coords,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    timestep, _ = timestep.chunk(2)

                denoised_latents = self.scheduler.step(
                    -noise_pred, t, latents, per_token_timesteps=timestep, return_dict=False
                )[0]
                if is_conditioning_image_or_video:
                    if conditioning_mask is not None:
                        tokens_to_denoise_mask = (t / 1000 - 1e-6 < (1.0 - conditioning_mask)).unsqueeze(-1)
                        latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
                    else:
                        latents = denoised_latents

                # Update latents
                if is_conditioning_image_or_video:
                    pass  # latents already updated above
                else:
                    latents = denoised_latents

                # Capture output latents for debugging (last timestep only)
                if i == len(timesteps) - 1:
                    self._capture_debug_latents("output", latents)
                    # Also capture for unified debug system
                    if hasattr(self, '_debug_capture_enabled') and self._debug_capture_enabled:
                        logger.info(f"batch inference output shape : {latents.shape}")
                        self._debug_valid_output_latents = latents.detach().clone()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()


        logger.info(f"total latent shape : {latents.shape}")

        # Offload all models
        self.maybe_free_model_hooks()

        # Decode latents to video if needed for return
        if not return_latents or output_type != "latent":
            # Decode final video for return
            if is_conditioning_image_or_video:
                logger.info(f"modified conditioning latents: {extra_conditioning_num_latents}")
                decode_latents = latents[:, extra_conditioning_num_latents:]
            else:
                decode_latents = latents

            # Unpack for decoding
            # Calculate actual number of frames from sequence length
            sequence_length = decode_latents.shape[1]  # [B, Seq, D]
            actual_num_frames = sequence_length // (latent_height * latent_width)
            logger.info(f"🔍 Decode latents: seq_len={sequence_length}, calculated_frames={actual_num_frames}, latent_h={latent_height}, latent_w={latent_width}")

            decode_latents_unpacked = self._unpack_latents(
                decode_latents,
                actual_num_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )

            # Denormalize and decode
            decode_latents_unpacked = self._denormalize_latents(
                decode_latents_unpacked, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )

            with torch.no_grad():
                # Prepare latents for decoding (based on original LTX pipeline logic)
                curr_latents = decode_latents_unpacked / self.vae.config.scaling_factor
                curr_latents = curr_latents.to(prompt_embeds.dtype)

                # Handle timestep conditioning based on VAE configuration
                if not self.vae.config.timestep_conditioning:
                    timestep = None
                else:
                    # Use decode_timestep parameter if available, otherwise default to 0.0
                    if 'decode_timestep' in locals() and decode_timestep is not None:
                        current_decode_timestep = decode_timestep
                        current_decode_noise_scale = decode_noise_scale
                    else:
                        # Default values for clean decoding
                        current_decode_timestep = 0.0
                        current_decode_noise_scale = None

                    noise = torch.randn(
                        curr_latents.shape, generator=generator, device=device, dtype=curr_latents.dtype
                    )

                    if not isinstance(current_decode_timestep, list):
                        current_decode_timestep = [current_decode_timestep] * curr_latents.shape[0]
                    if current_decode_noise_scale is None:
                        current_decode_noise_scale = current_decode_timestep
                    elif not isinstance(current_decode_noise_scale, list):
                        current_decode_noise_scale = [current_decode_noise_scale] * curr_latents.shape[0]

                    timestep = torch.tensor(current_decode_timestep, device=device, dtype=curr_latents.dtype)
                    decode_noise_scale_tensor = torch.tensor(current_decode_noise_scale, device=device, dtype=curr_latents.dtype)[
                        :, None, None, None, None
                    ]
                    curr_latents = (1 - decode_noise_scale_tensor) * curr_latents + decode_noise_scale_tensor * noise

                # Ensure VAE and latents are on the same device
                self.vae = self.vae.to(curr_latents.device)
                curr_latents = curr_latents.to(self.vae.dtype)

                # Decode using ltxv_utils.decode_video
                B, C, F, H, W = curr_latents.shape
                latents_seq = curr_latents.flatten(2).transpose(1, 2)
                video = decode_video(
                    vae=self.vae,
                    latents=latents_seq[0],
                    num_frames=F,
                    height=H,
                    width=W,
                    device=curr_latents.device,
                    dtype=curr_latents.dtype,
                    decode_timestep=timestep.item() if timestep is not None else 0.0
                )

            video = self.video_processor.postprocess_video(video, output_type=output_type)

            # Save video as MP4 if requested
            saved_video_path = None
            # logger.info(f"save_video : {save_video} video : {video}")
            if save_video and video is not None:
                saved_video_path = self._save_video_as_mp4(video, output_dir, video_filename, fps, prompt, scenario, validation_type, step)

            # Save full sequence video if requested
            full_sequence_video_path = None
            if save_full_sequence and latents is not None:
                logger.info("🎬 [FULL_SEQ] Generating full sequence video (including reference tokens)...")
                full_sequence_video_path = self._save_full_sequence_video(
                    latents=latents,
                    output_dir=output_dir,
                    fps=fps,
                    prompt=prompt,
                    scenario=scenario,
                    validation_type=validation_type,
                    step=step,
                    prompt_embeds=prompt_embeds,
                    latent_height=latent_height,
                    latent_width=latent_width
                )
        else:
            video = None
            saved_video_path = None

    def _save_full_sequence_video(
        self,
        latents: torch.Tensor,
        output_dir: Optional[str] = None,
        fps: int = 24,
        prompt: Optional[str] = None,
        scenario: Optional[str] = None,
        validation_type: Optional[str] = None,
        step: Optional[int] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        latent_height: int = None,
        latent_width: int = None
    ) -> str:
        """
        Save the full transformer output sequence as a video, including reference tokens.
        """
        try:
            # Determine output directory
            if output_dir is None:
                output_dir = os.getcwd()

            scenario_dir = os.path.join(output_dir, "scenario")
            os.makedirs(scenario_dir, exist_ok=True)

            # Create filename for full sequence using similar logic to _save_video_as_mp4
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Determine prefix based on validation type
            if validation_type == "t2v_validation":
                prefix = "full_sequence_t2v"
            elif validation_type == "v2v_validation":
                prefix = "full_sequence_v2v"
            else:
                # Fallback logic based on scenario presence
                if scenario:
                    prefix = "full_sequence_v2v"  # Has scenario = video-to-video
                else:
                    prefix = "full_sequence_t2v"  # No scenario = text-to-video

            # Create step suffix if step is provided
            step_suffix = f"_step_{step:06d}" if step is not None else ""

            # Use new naming convention based on scenario
            if scenario:
                # Try to extract shot information from scenario
                scenario_parts = scenario.split(",")
                shot_info = [part.strip() for part in scenario_parts if part.strip().startswith("shot")]

                if len(shot_info) >= 2:
                    # Multi-shot with transitions: full_sequence_prefix_shot1_transition_shot2_step_000123.mp4
                    video_filename = f"{prefix}_shot1_transition_shot2{step_suffix}_{timestamp}.mp4"
                elif len(shot_info) == 1:
                    # Single shot: full_sequence_prefix_shot1_step_000123.mp4
                    video_filename = f"{prefix}_shot1{step_suffix}_{timestamp}.mp4"
                else:
                    # Fallback with scenario info: full_sequence_prefix_scenario_info_step_000123.mp4
                    scenario_clean = scenario.replace(",", "_").replace(" ", "_")
                    video_filename = f"{prefix}_{scenario_clean}{step_suffix}_{timestamp}.mp4"
            else:
                # No scenario: full_sequence_prefix_step_000123.mp4
                video_filename = f"{prefix}{step_suffix}_{timestamp}.mp4"
            video_path = os.path.join(scenario_dir, video_filename)

            # Use the entire latents tensor without slicing
            logger.info(f"🎬 [FULL_SEQ] Decoding full sequence latents: {latents.shape}")

            # Denormalize and decode the full latents
            # Handle potential missing attributes in VAE config
            latents_mean = getattr(self.vae.config, 'latents_mean', None)
            latents_std = getattr(self.vae.config, 'latents_std', None)
            scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)

            # Unpack and denormalize the latents
            unpacked_latents = self._unpack_latents(
                latents,
                num_frames=latents.shape[1] // (latent_height * latent_width),  # Calculate frames from sequence length
                height=latent_height,
                width=latent_width
            )

            # Denormalize only if config values are available
            if latents_mean is not None and latents_std is not None:
                denormalized_latents = self._denormalize_latents(
                    unpacked_latents, latents_mean, latents_std, scaling_factor
                )
            else:
                # Use unpacked latents directly if normalization params are not available
                logger.warning("🎬 [FULL_SEQ] VAE config missing latents_mean/latents_std, using unpacked latents directly")
                denormalized_latents = unpacked_latents

            # Decode with VAE
            logger.info(f"🎬 [FULL_SEQ] VAE decoding latents shape: {denormalized_latents.shape}")
            with torch.no_grad():
                # Move VAE to same device and dtype
                self.vae = self.vae.to(denormalized_latents.device)
                denormalized_latents = denormalized_latents.to(self.vae.dtype)
                
                # Decode using ltxv_utils.decode_video
                B, C, F, H, W = denormalized_latents.shape
                latents_seq = denormalized_latents.flatten(2).transpose(1, 2)
                video_tensor = decode_video(
                    vae=self.vae,
                    latents=latents_seq[0],
                    num_frames=F,
                    height=H,
                    width=W,
                    device=denormalized_latents.device,
                    dtype=denormalized_latents.dtype,
                    decode_timestep=0.0
                )

            # Convert to video format expected by export_to_video
            video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
            video_frames = video_tensor.squeeze(0).permute(1, 2, 3, 0).cpu()  # [F, H, W, C]

            # Convert to PIL Images
            video_pil = []
            for frame_idx in range(video_frames.shape[0]):
                frame = video_frames[frame_idx]
                frame_pil = PIL.Image.fromarray((frame.to(torch.float32).cpu().numpy() * 255).astype(np.uint8))
                video_pil.append(frame_pil)

            # Save using export_to_video
            export_to_video(video_pil, video_path, fps=fps)

            logger.info(f"🎬 [FULL_SEQ] Full sequence video saved: {video_path}")
            return video_path

        except Exception as e:
            logger.error(f"🎬 [FULL_SEQ] Error saving full sequence video: {str(e)}")
            import traceback
            logger.error(f"🎬 [FULL_SEQ] Traceback: {traceback.format_exc()}")
            return None

        # Prepare return values
        if return_latents:
            # Return both generated video and latents for next shot
            # Extract output latents (skip conditioning part if present)
            if is_conditioning_image_or_video:
                output_latents = latents[:, extra_conditioning_num_latents:]
            else:
                output_latents = latents
            
            # Unpack to standard latent format [B, C, F, H, W]
            # Calculate actual number of frames from sequence length
            output_sequence_length = output_latents.shape[1]  # [B, Seq, D]
            actual_output_frames = output_sequence_length // (latent_height * latent_width)
            logger.info(f"🔍 Return latents: seq_len={output_sequence_length}, calculated_frames={actual_output_frames}, latent_h={latent_height}, latent_w={latent_width}")

            unpacked_latents = self._unpack_latents(
                output_latents,
                actual_output_frames,
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )

            # Denormalize latents for reuse
            unpacked_latents = self._denormalize_latents(
                unpacked_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )

            if not return_dict:
                if saved_video_path:
                    return (video, unpacked_latents, saved_video_path)
                else:
                    return (video, unpacked_latents)
            # Use dict to include latents since LTXPipelineOutput may not support latents attribute
            result = {"frames": video, "latents": unpacked_latents}
            if saved_video_path:
                result["saved_video_path"] = saved_video_path
            return result
        else:
            if not return_dict:
                if saved_video_path:
                    return (video, saved_video_path)
                else:
                    return (video,)
            result = LTXPipelineOutput(frames=video)
            if saved_video_path:
                # Since LTXPipelineOutput might not support extra attributes,
                # return dict instead when video is saved
                return {"frames": video, "saved_video_path": saved_video_path}
            return result

    def _save_video_as_mp4(
        self,
        video: List[PIL.Image.Image],
        output_dir: Optional[str] = None,
        video_filename: Optional[str] = None,
        fps: int = 24,
        prompt: Optional[str] = None,
        scenario: Optional[str] = None,
        validation_type: Optional[str] = None,
        step: Optional[int] = None
    ) -> str:
        """
        Save generated video frames as MP4 file using export_to_video.

        Args:
            video: List of PIL Images representing video frames
            output_dir: Directory to save the video (defaults to current directory)
            video_filename: Filename for the saved video (auto-generated if not provided)
            fps: Frames per second for the saved video
            prompt: Text prompt used for generation (used in auto-generated filename)
            scenario: Scenario string used (used in auto-generated filename)
            validation_type: Type of validation ("t2v_validation" or "v2v_validation")
            step: Training step number (included in auto-generated filename)

        Returns:
            str: Path to the saved video file
        """
        from datetime import datetime

        # Validate input
        if not video or len(video) == 0:
            logger.warning("No video frames to save")
            return None

        # Flatten nested list if needed
        if len(video) > 0 and isinstance(video[0], list):
            video = video[0]

        # Set default output directory
        if output_dir is None:
            output_dir = "."

        # Create scenario subfolder
        scenario_dir = os.path.join(output_dir, "scenario")
        os.makedirs(scenario_dir, exist_ok=True)

        # Generate filename if not provided
        if video_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Determine prefix based on validation type
            if validation_type == "t2v_validation":
                prefix = "t2v"
            elif validation_type == "v2v_validation":
                prefix = "v2v"
            else:
                # Fallback logic based on scenario presence
                if scenario:
                    prefix = "v2v"  # Has scenario = video-to-video
                else:
                    prefix = "t2v"  # No scenario = text-to-video

            # Create step suffix if step is provided
            step_suffix = f"_step_{step:06d}" if step is not None else ""

            # Use new naming convention based on scenario
            if scenario:
                # Try to extract shot information from scenario
                scenario_parts = scenario.split(",")
                shot_info = [part.strip() for part in scenario_parts if part.strip().startswith("shot")]

                if len(shot_info) >= 2:
                    # Multi-shot with transitions: prefix_shot1_transition_shot2_step_000123.mp4
                    video_filename = f"{prefix}_shot1_transition_shot2{step_suffix}_{timestamp}.mp4"
                elif len(shot_info) == 1:
                    # Single shot: prefix_shot1_step_000123.mp4
                    video_filename = f"{prefix}_shot1{step_suffix}_{timestamp}.mp4"
                else:
                    # Fallback with scenario info: prefix_scenario_info_step_000123.mp4
                    scenario_clean = scenario.replace(",", "_").replace(" ", "_")
                    video_filename = f"{prefix}_scenario_{scenario_clean}{step_suffix}_{timestamp}.mp4"
            else:
                # No scenario - single shot: prefix_single_shot_step_000123.mp4
                video_filename = f"{prefix}_single_shot{step_suffix}_{timestamp}.mp4"

        # Full path for the output file (save in scenario subfolder)
        output_path = os.path.join(scenario_dir, video_filename)

        try:
            # Use diffusers' export_to_video function for consistent video saving
            export_to_video(video, output_path, fps=fps)

            # Get video information for logging and metadata
            frame_count = len(video)
            if hasattr(video[0], 'size'):
                width, height = video[0].size
            else:
                # Fallback for non-PIL images
                width, height = 768, 448  # Default dimensions

            logger.info(f"🎬 Video saved successfully: {output_path}")
            logger.info(f"🎬 Video details: {frame_count} frames, {fps} FPS, {width}x{height}")

            # Save prompt information to JSON file
            self._save_prompt_info_to_json(
                scenario_dir=scenario_dir,
                video_filename=video_filename,
                output_path=output_path,
                prompt=prompt,
                scenario=scenario,
                fps=fps,
                frame_count=frame_count,
                resolution=(width, height)
            )

            return output_path

        except Exception as e:
            logger.error(f"🎬 ❌ Failed to save video using export_to_video: {e}")
            logger.error(f"🎬 Video input type: {type(video)}, length: {len(video) if video else 0}")
            if video and len(video) > 0:
                logger.error(f"🎬 First frame type: {type(video[0])}")
            return None

    def _save_prompt_info_to_json(
        self,
        scenario_dir: str,
        video_filename: str,
        output_path: str,
        prompt: Optional[str] = None,
        scenario: Optional[str] = None,
        fps: int = 24,
        frame_count: int = 0,
        resolution: Tuple[int, int] = (0, 0)
    ):
        """
        Save prompt and generation information to JSON file.

        Args:
            scenario_dir: Directory where the scenario folder is located
            video_filename: Name of the video file
            output_path: Full path to the saved video
            prompt: Text prompt used for generation
            scenario: Scenario string used
            fps: Frames per second
            frame_count: Number of frames in the video
            resolution: Video resolution (width, height)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create prompt info entry
            prompt_data = {
                "video_filename": video_filename,
                "video_path": output_path,
                "prompt": prompt,
                "scenario": scenario,
                "timestamp": timestamp,
                "generation_params": {
                    "fps": fps,
                    "frame_count": frame_count,
                    "resolution": {
                        "width": resolution[0],
                        "height": resolution[1]
                    }
                },
                "generation_type": "SGMultiShotPipeline"
            }

            # Store in instance for potential batch operations
            prompt_key = f"video_{timestamp}_{video_filename.replace('.mp4', '')}"
            self.prompt_info[prompt_key] = prompt_data

            # Save to JSON file
            json_path = os.path.join(scenario_dir, "prompt_info.json")

            # Load existing data if file exists
            existing_data = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load existing prompt info: {e}")

            # Merge with existing data
            existing_data.update(self.prompt_info)

            # Save updated data
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📝 Prompt information saved to: {json_path}")

        except Exception as e:
            logger.error(f"Failed to save prompt information: {e}")

    # """
    # Encode video to latent space for use as prev_latent conditioning.
    
    # Args:
    #     video: Video tensor, PIL images, or video frames
        
    # Returns:
    #     Latent tensor in sequence format [seq_len, channels] for conditioning
    # """
    # with torch.no_grad():
    #     # Handle different input types (PIL images, etc.)
    #     if isinstance(video, list):
    #         # Convert PIL images to tensor
    #         import numpy as np
    #         from PIL import Image
    #         frames = []
    #         for frame in video:
    #             if isinstance(frame, Image.Image):
    #                 frame_array = np.array(frame).transpose(2, 0, 1)  # HWC -> CHW
    #                 frames.append(torch.from_numpy(frame_array))
    #         video = torch.stack(frames, dim=0)  # [frames, channels, height, width]
        
    #     # Ensure video has batch dimension and correct format for VAE: [B, C, F, H, W]
    #     if video.dim() == 4:  # [frames, channels, height, width]
    #         video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, frames, channels, height, width] -> [1, channels, frames, height, width]
        
    #     # Move to device and normalize
    #     device = self.device
    #     video = video.to(device, dtype=torch.float32)
        
    #     # Normalize to [-1, 1] based on input range
    #     if video.max() > 1.0:  # PIL images are typically 0-255
    #         video = video / 255.0  # [0, 255] -> [0, 1]
    #         video = video * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    #     elif video.min() >= 0.0 and video.max() <= 1.0:  # Already in [0, 1]
    #         video = video * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
    #     # Encode using VAE
    #     latent_dist = self.vae.encode(video)
    #     latent = retrieve_latents(latent_dist)
        
    #     # Use standard VAE scaling
    #     latent = latent * self.vae.config.scaling_factor
            
    #     # Convert to sequence format [seq_len, channels]
    #     batch, latent_channels, latent_frames, latent_height, latent_width = latent.shape
    #     seq_len = latent_frames * latent_height * latent_width
    #     latent = latent.view(batch, latent_channels, seq_len).permute(0, 2, 1).squeeze(0)
        
    #     return latent