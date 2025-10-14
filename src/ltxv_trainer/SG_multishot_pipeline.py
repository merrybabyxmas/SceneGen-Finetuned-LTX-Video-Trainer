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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
from PIL import Image
import torch
import torch.nn as nn
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKLLTXVideo
from diffusers.models.transformers import LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import T5EncoderModel, T5TokenizerFast
from torchvision.transforms.functional import center_crop, resize

# [SG-PATCH:64TOKEN] Import token utilities for 64-case system
from ltxv_trainer.token_utils import VideoTokenEmbeddings

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
from ltxv_trainer import logger

# [SG-PATCH:BLEND] Import torch.nn for trainable gate parameter
import torch.nn as nn


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

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

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
        
        # Initialize video processor if not already done by parent
        if not hasattr(self, 'video_processor') or self.video_processor is None:
            vae_scale_factor = getattr(self.vae, 'spatial_compression_ratio', 32)
            self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)

        # [SG-PATCH:BLEND] Initialize trainable gate parameter for prev blending
        num_channels = 128  # Default LTX latent channels
        self.prev_gate = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

    # [SG-PATCH:64TOKEN] Token configuration method for 64-case system
    def set_token_config(self, 
                        sos_use: bool = True,
                        sos_mode: str = "trainable",
                        stable_use: bool = False,
                        stable_mode: str = "trainable",
                        transition_use: bool = False,
                        transition_mode: str = "trainable"):
        """
        Configure the 64-case token system.
        
        Args:
            sos_use: Whether to use SOS tokens (0/1)
            sos_mode: SOS token mode ('fixed' or 'trainable')
            stable_use: Whether to use stable tokens (0/1)
            stable_mode: Stable token mode ('fixed' or 'trainable')
            transition_use: Whether to use transition tokens (0/1)
            transition_mode: Transition token mode ('fixed' or 'trainable')
        """
        # Get latent dimensions from transformer config
        hidden_dim = getattr(self.transformer.config, 'hidden_size', 128)
        spatial_height = 14  # Default latent spatial dimensions
        spatial_width = 24
        
        # Initialize VideoTokenEmbeddings with 64-case configuration
        self.video_token_embeddings = VideoTokenEmbeddings(
            hidden_dim=hidden_dim,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            sos_use=sos_use,
            sos_mode=sos_mode,
            stable_use=stable_use,
            stable_mode=stable_mode,
            transition_use=transition_use,
            transition_mode=transition_mode
        )
        
        # Move to appropriate device if transformer is already on device
        if hasattr(self.transformer, 'device'):
            self.video_token_embeddings = self.video_token_embeddings.to(self.transformer.device)
            
        logger.info(f"🎭 Configured 64-case token system: {self.video_token_embeddings.get_token_config_string()}")

    # [SG-PATCH:BLEND] Safe LERP utility function
    @staticmethod
    def _safe_lerp(a: torch.Tensor, b: torch.Tensor, w: float | torch.Tensor) -> torch.Tensor:
        """
        Safe linear interpolation ensuring dtype consistency.
        
        Args:
            a: Start tensor
            b: End tensor  
            w: Weight for interpolation (0=a, 1=b)
            
        Returns:
            Interpolated tensor: (1-w)*a + w*b
        """
        if b.dtype != a.dtype:
            b = b.to(a.dtype)
        return torch.lerp(a, b, w)

    # [SG-PATCH:BLEND] Previous shot blending function
    def _blend_prev(self, prev: torch.Tensor, sos: torch.Tensor, mode: str, alpha: float = 0.2, gate: torch.nn.Parameter | None = None) -> torch.Tensor:
        """
        Blend previous shot latents with SOS token according to specified mode.
        
        Args:
            prev: Previous shot latents (B, C, F, H, W)
            sos: SOS token latents (B, C, F, H, W)
            mode: Blending mode - 'plain', 'lerp', 'temporal', 'trainable'
            alpha: Blending strength for lerp/temporal modes
            gate: Trainable gate parameter for trainable mode
            
        Returns:
            Blended latents (B, C, F, H, W)
        """
        prev = prev.to(sos.dtype)
        if mode == "plain":
            return prev
        elif mode == "lerp":
            return self._safe_lerp(prev, sos, alpha)
        elif mode == "temporal":
            F = prev.shape[2]
            w = torch.linspace(0, 1, F, device=prev.device, dtype=prev.dtype).view(1, 1, F, 1, 1)
            return prev * (1 - w) + sos * w
        elif mode == "trainable":
            assert gate is not None, "trainable blending requires a gate parameter"
            g = torch.sigmoid(gate).to(prev.dtype)
            return prev * (1 - g) + sos * g
        else:
            raise ValueError(f"Unknown blending mode: {mode}")
            
    # [SG-PATCH:TOKENS] Token insertion methods for 64-case system        
    def apply_token_insertions(self, latents_5d: torch.Tensor, config: dict) -> torch.Tensor:
        """
        Apply stable and transition token insertions based on configuration.
        
        Args:
            latents_5d: Input latents in 5D format (B, C, F, H, W)
            config: Configuration dictionary with token settings
            
        Returns:
            Latents with tokens inserted (B, C, F', H, W) where F' may be larger
        """
        if self.video_token_embeddings is None:
            return latents_5d
        
        # [SG-PATCH:STABLE] Apply stable token insertion between frames
        if self.video_token_embeddings.stable_use:
            def stable_fn(B, C, F, H, W, device, dtype):
                return self.video_token_embeddings.get_stable_token(device=device, batch_size=B)
            
            latents_5d = interleave_stable(latents_5d, stable_fn)
            logger.info(f"🔧 Applied stable tokens: {latents_5d.shape}")
        
        return latents_5d
        
    def apply_prev_blending(self, prev_latents: torch.Tensor, curr_latents: torch.Tensor, config: dict) -> torch.Tensor:
        """
        Apply previous shot blending according to configuration.
        
        Args:
            prev_latents: Previous shot latents (B, C, F, H, W)
            curr_latents: Current shot latents (B, C, F, H, W) 
            config: Configuration dictionary with blending settings
            
        Returns:
            Blended latents combining prev and curr shots
        """
        # [SG-PATCH:BLEND] Extract configuration
        blending_mode = config.get('prev_blending', 'plain')
        alpha = config.get('prev_alpha', 0.2)
        gate = getattr(self, '_trainable_gate', None) if blending_mode == 'trainable' else None
        
        # [SG-PATCH:TRANSITION] Apply transition token insertion between shots
        if self.video_token_embeddings and self.video_token_embeddings.transition_use:
            def trans_fn(B, C, F, H, W, device, dtype):
                return self.video_token_embeddings.get_transition_token(device=device, batch_size=B)
            
            combined_latents = insert_transition(prev_latents, curr_latents, trans_fn)
            logger.info(f"🔄 Applied transition tokens: {combined_latents.shape}")
            return combined_latents
        else:
            # Direct concatenation without transition tokens
            return torch.cat([prev_latents, curr_latents], dim=2)

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
        reference_latents=None,  # Add support for reference_latents
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

        # Add other kwargs (excluding reference_latents)
        for k, v in kwargs.items():
            if k != 'reference_latents':
                parent_kwargs[k] = v

        # Call parent class check_inputs
        super().check_inputs(**parent_kwargs)

        # Additional validation for reference_latents if needed
        if reference_latents is not None:
            if not isinstance(reference_latents, torch.Tensor):
                raise TypeError("reference_latents must be a torch.Tensor")
            if reference_latents.dim() != 5:
                raise ValueError("reference_latents must be a 5D tensor [B, C, F, H, W]")

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
        # New algorithm: all three parts (prev, transition, curr) are initialized as noisy
        latents_prev = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents_transition = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents_curr = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        



        video_ids = self._prepare_video_ids(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            patch_size_t=self.transformer_temporal_patch_size,
            patch_size=self.transformer_spatial_patch_size,
            device=device,
        )


        conditioning_mask, extra_conditioning_num_latents = None, 0
        
        video_ids_base = self._scale_video_ids(
            video_ids,
            scale_factor=self.vae_spatial_compression_ratio,
            scale_factor_t=self.vae_temporal_compression_ratio,
            frame_index=0,
            device=device,
        )
        
        # Combine coordinates for all three parts: prev + transition + curr
        offset = video_ids_base[:, 0].max(dim=1, keepdim=True)[0] + 1

        ids_prev = video_ids_base.clone()
        ids_trans = video_ids_base.clone()
        ids_curr = video_ids_base.clone()

        # --------------------------------------------------------------
        # [SG-PATCH: TEMPORAL OFFSET FIX]
        # transition은 prev와 curr의 중간 시점에 위치
        # --------------------------------------------------------------
        ids_trans[:, 0] = ids_trans[:, 0] + offset * 0.5
        ids_curr[:, 0]  = ids_curr[:, 0]  + offset

        # --------------------------------------------------------------
        # [SG-PATCH: PHASE CHANNEL]
        # 각 구간에 phase 스칼라 추가 (0=prev, 0.5=transition, 1=curr)
        # --------------------------------------------------------------
        phase_prev = torch.zeros_like(ids_prev[:, :1])          # (B, 1, Seq)
        phase_trans = torch.full_like(ids_trans[:, :1], 0.5)
        phase_curr = torch.ones_like(ids_curr[:, :1])

        ids_prev = torch.cat([ids_prev, phase_prev], dim=1)     # (B, 4, Seq)
        ids_trans = torch.cat([ids_trans, phase_trans], dim=1)
        ids_curr = torch.cat([ids_curr, phase_curr], dim=1)

        # --------------------------------------------------------------
        # [SG-PATCH: CONCAT ALL]
        # prev + transition + curr을 시간 순서로 결합
        # --------------------------------------------------------------
        video_ids = torch.cat([ids_prev, ids_trans, ids_curr], dim=2)  # (B, 4, 3*S_single)     
        # Pack latents for transformer input
        latents_prev = self._pack_latents(latents_prev, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
        latents_transition = self._pack_latents(latents_transition, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
        latents_curr = self._pack_latents(latents_curr, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)
        latents = torch.cat([latents_prev, latents_transition, latents_curr], dim=1) 
        logger.info(f"latents shape : {latents.shape}")

        # Create shot-level causal mask for new algorithm
        from ltxv_trainer.SG_training_strategy import create_shot_level_causal_mask
        single_shot_seq_len = latents_prev.shape[1]  # Each part has same length
        causal_mask = create_shot_level_causal_mask(single_shot_seq_len, device=device)
        
        # Expand for batch dimension: (batch_size, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        return latents, causal_mask, video_ids, extra_conditioning_num_latents

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
        reference_latents: Optional[torch.Tensor] = None,
        reference_video: Optional[List] = None,
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
    ):
        """
        Generate video using the SGMultiShotPipeline with SOS token and reference video support.
        
        Args:
            prompt: Text prompt for video generation
            reference_latents: Optional reference latents tensor for multi-shot generation
            height: Video height
            width: Video width
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            generator: Random generator for reproducible results
            output_type: Output format ("pil" or "tensor")
            return_dict: Whether to return a dictionary
        
        Returns:
            Generated video frames
        
        Examples:
        
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
            reference_latents=reference_latents,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
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

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, causal_mask, video_coords, extra_conditioning_num_latents = self.prepare_latents(
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

        # 4.5. Reference conditioning removed - using new algorithm with prompt-only conditioning
        reference_num_latents = 0

        video_coords = video_coords.float()
        # Reference latents removed - always apply frame rate scaling
        video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)

        init_latents = latents.clone() if is_conditioning_image_or_video else None
        # Set conditioning_mask to None since we're using causal_mask for attention control
        conditioning_mask = None

        if self.do_classifier_free_guidance:
            video_coords = torch.cat([video_coords, video_coords], dim=0)

        # 5. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
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
        logger.info(f"inference input shape : {latents.shape}")

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
                if is_conditioning_image_or_video or reference_latents is not None or reference_num_latents > 0:
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
                if is_conditioning_image_or_video or reference_latents is not None or reference_num_latents > 0:
                    if conditioning_mask_model_input is not None:
                        timestep = torch.min(timestep, (1 - conditioning_mask_model_input) * 1000.0)

                # Debug hook: store valid-input latents for unified debug system


                # Prepare causal mask for transformer if available
                transformer_kwargs = {
                    "hidden_states": latent_model_input,
                    "encoder_hidden_states": prompt_embeds,
                    "timestep": timestep,
                    "encoder_attention_mask": prompt_attention_mask,
                    "video_coords": video_coords,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                
                # Add causal mask if available (for new algorithm)
                if 'causal_mask' in locals() and causal_mask is not None:
                    # Expand causal mask for classifier-free guidance if needed
                    if self.do_classifier_free_guidance:
                        causal_mask_input = torch.cat([causal_mask, causal_mask], dim=0)
                    else:
                        causal_mask_input = causal_mask
                    transformer_kwargs["attention_mask"] = causal_mask_input
                    logger.info(f"attention causal mask shape : {causal_mask_input.shape}")
                
                noise_pred = self.transformer(**transformer_kwargs)[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    timestep, _ = timestep.chunk(2)

                denoised_latents = self.scheduler.step(
                    -noise_pred, t, latents, per_token_timesteps=timestep, return_dict=False
                )[0]
                if is_conditioning_image_or_video or reference_latents is not None or reference_num_latents > 0:
                    if conditioning_mask is not None:
                        tokens_to_denoise_mask = (t / 1000 - 1e-6 < (1.0 - conditioning_mask)).unsqueeze(-1)
                        latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
                    else:
                        latents = denoised_latents

                # Update latents
                if is_conditioning_image_or_video or reference_latents is not None or reference_num_latents > 0:
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

        # Handle reference latents output processing
        if reference_latents is not None and output_reference_comparison:
            # Split latents: [reference_latents, frame_conditions, target_latents]
            reference_latents_out = latents[:, :reference_num_latents]
            remaining_latents = latents[:, reference_num_latents:]

            # Remove frame conditioning from remaining latents if needed
            if is_conditioning_image_or_video:
                target_latents_out = remaining_latents[:, extra_conditioning_num_latents:]
            else:
                target_latents_out = remaining_latents

            # Process both reference and target latents
            videos = []
            for curr_latents in [reference_latents_out, target_latents_out]:
                if output_type == "latent":
                    curr_video = curr_latents
                else:
                    # 🧠 NEW: Exclude transition segment
                    total_frames = curr_latents.shape[2]
                    frames_per_shot = total_frames // 3

                    # Split into prev / transition / curr
                    latents_prev = curr_latents[:, :, :frames_per_shot]
                    latents_trans = curr_latents[:, :, frames_per_shot:2 * frames_per_shot]
                    latents_curr = curr_latents[:, :, 2 * frames_per_shot:]

                    # Keep only prev + curr for decoding
                    curr_latents = torch.cat([latents_prev, latents_curr], dim=2)
                    logger.info(
                        f"🔪 Excluded transition: prev={latents_prev.shape}, "
                        f"curr={latents_curr.shape}, combined={curr_latents.shape}"
                    )

                    curr_latents = self._denormalize_latents(
                        curr_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                    )
                    curr_latents = curr_latents.to(prompt_embeds.dtype)

                    if not self.vae.config.timestep_conditioning:
                        timestep = None
                    else:
                        noise = torch.randn(
                            curr_latents.shape, generator=generator, device=device, dtype=curr_latents.dtype
                        )
                        if not isinstance(decode_timestep, list):
                            decode_timestep = [decode_timestep] * batch_size
                        if decode_noise_scale is None:
                            decode_noise_scale = decode_timestep
                        elif not isinstance(decode_noise_scale, list):
                            decode_noise_scale = [decode_noise_scale] * batch_size

                        timestep = torch.tensor(decode_timestep, device=device, dtype=curr_latents.dtype)
                        decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=curr_latents.dtype)[
                            :, None, None, None, None
                        ]
                        curr_latents = (1 - decode_noise_scale) * curr_latents + decode_noise_scale * noise

                    curr_video = self.vae.decode(curr_latents, timestep, return_dict=False)[0]
                    curr_video = self.video_processor.postprocess_video(curr_video, output_type=output_type)
                videos.append(curr_video)

            # Concatenate videos side-by-side (along width dimension for visual output)
            if output_type == "latent":
                video = torch.cat(videos, dim=0)
            # For video tensors, shape is [B, C, F, H, W] or list of PIL images
            elif isinstance(videos[0], list):
                # Handle PIL images case - concatenate each frame side by side
                video = []
                for batch_idx in range(len(videos[0])):
                    combined_video = []
                    for frame_idx in range(len(videos[0][batch_idx])):
                        ref_frame = videos[0][batch_idx][frame_idx]
                        gen_frame = videos[1][batch_idx][frame_idx]
                        # Create side-by-side comparison
                        import PIL.Image

                        if isinstance(ref_frame, PIL.Image.Image) and isinstance(gen_frame, PIL.Image.Image):
                            combined_width = ref_frame.width + gen_frame.width
                            combined_height = max(ref_frame.height, gen_frame.height)
                            combined_frame = PIL.Image.new("RGB", (combined_width, combined_height))
                            combined_frame.paste(ref_frame, (0, 0))
                            combined_frame.paste(gen_frame, (ref_frame.width, 0))
                            combined_video.append(combined_frame)
                        else:
                            combined_video.append(gen_frame)  # Fallback to generated only
                    video.append(combined_video)
            else:
                # Handle tensor case - concatenate along width dimension (dim=4)
                video = torch.cat(videos, dim=4)
        else:
            # Regular processing - just remove conditioning parts and output generated video
            if reference_latents is not None or reference_num_latents > 0:
                # Remove reference latents
                latents = latents[:, reference_num_latents:]

            if is_conditioning_image_or_video:
                latents = latents[:, extra_conditioning_num_latents:]

            latents = self._unpack_latents(
                latents,
                latent_num_frames * 3,            # 기대 프레임 수 (3샷 구성)
                latent_height,
                latent_width,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )

            # --- 안전하게 프레임 분할 ---
            total_frames = latents.shape[2]       # [B, C, F, H, W]
            if total_frames % 3 != 0:
                logger.warning(
                    f"[Decode] total_frames={total_frames}가 3으로 나누어떨어지지 않습니다. "
                    "transition 길이 오차가 있을 수 있어 마지막 잉여 프레임을 버립니다."
                )
            frames_per_shot = total_frames // 3
            if frames_per_shot == 0:
                raise ValueError(f"[Decode] frames_per_shot가 0입니다. total_frames={total_frames}")

            # prev | transition | curr 로 정확히 슬라이스
            latents_prev = latents[:, :, 0:frames_per_shot]
            latents_trans = latents[:, :, frames_per_shot:2*frames_per_shot]
            latents_curr = latents[:, :, -frames_per_shot:]


            logger.info(
                f"🎞 Split latents: prev={latents_prev.shape}, trans={latents_trans.shape}, curr={latents_curr.shape}"
            )

            if output_type == "latent":
                # 단순히 latent로 반환할 경우는 병합 없이 그대로 반환
                video = torch.cat([latents_prev, latents_curr, latents_trans], dim=2)
            else:
                # --- 세 구간 각각 디코딩 ---
                decoded_segments = []
                for name, seg_lat in [("prev", latents_prev), ("curr", latents_curr), ("trans", latents_trans)]:
                    seg_lat = self._denormalize_latents(
                        seg_lat, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                    ).to(prompt_embeds.dtype)

                    if not self.vae.config.timestep_conditioning:
                        timestep = None
                    else:
                        noise = torch.randn(seg_lat.shape, generator=generator, device=device, dtype=seg_lat.dtype)
                        if not isinstance(decode_timestep, list):
                            decode_timestep = [decode_timestep] * batch_size
                        if decode_noise_scale is None:
                            decode_noise_scale = decode_timestep
                        elif not isinstance(decode_noise_scale, list):
                            decode_noise_scale = [decode_noise_scale] * batch_size

                        timestep = torch.tensor(decode_timestep, device=device, dtype=seg_lat.dtype)
                        decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=seg_lat.dtype)[
                            :, None, None, None, None
                        ]
                        seg_lat = (1 - decode_noise_scale) * seg_lat + decode_noise_scale * noise

                    seg_video = self.vae.decode(seg_lat, timestep, return_dict=False)[0]
                    seg_video = self.video_processor.postprocess_video(seg_video, output_type=output_type)
                    decoded_segments.append(seg_video)
                    
                    logger.info(f"decoded segments : {seg_video}")

                # --- 가로축(width dim=4)으로 결합 ---
                if isinstance(decoded_segments[0], torch.Tensor):
                    video = torch.cat(decoded_segments, dim=4)  # [B, C, F, H, W_total]
                    logger.info(f"🧩 Concatenated horizontally → {video.shape}")
                else:
                    # PIL 프레임 리스트일 경우
                    video = []
                    for batch_idx in range(len(decoded_segments[0])):
                        combined_frames = []
                        for frame_idx in range(len(decoded_segments[0][batch_idx])):
                            imgs = [decoded_segments[i][batch_idx][frame_idx] for i in range(3)]
                            w_total = sum(im.width for im in imgs)
                            h_max = max(im.height for im in imgs)
                            combined = Image.new("RGB", (w_total, h_max))
                            x_offset = 0
                            for im in imgs:
                                combined.paste(im, (x_offset, 0))
                                x_offset += im.width
                            combined_frames.append(combined)
                        video.append(combined_frames)
                    logger.info(f"🧩 Combined horizontally (PIL frames): {len(video[0])} frames")

        logger.info(f"total latent shape : {latents.shape}")

        # Offload all models
        self.maybe_free_model_hooks()

        # Prepare return values
        if return_latents:
            # Return both generated video and latents for next shot
            if reference_latents is not None or reference_num_latents > 0:
                # Unpack to standard latent format [B, C, F, H, W]
                unpacked_latents = self._unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
                logger.info(f"unpacked latents shape : {unpacked_latents.shape}")
                logger.info(f"latnets mean shape : {self.vae.latents_mean.shape}")
                logger.info(f"latnets std shape : {self.vae.latents_std.shape}")

                # Denormalize latents for reuse
                try:
                    unpacked_latents = self._denormalize_latents(
                        unpacked_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                    )
                    logger.info(f"🔍 Denormalization successful: {unpacked_latents.shape}")
                except Exception as denorm_error:
                    logger.error(f"❌ Denormalization failed: {denorm_error}")
                    logger.warning(f"🔄 Using raw unpacked latents without denormalization")
                    # Keep unpacked_latents as-is without denormalization
            else:
                # No reference case - return all latents
                if is_conditioning_image_or_video:
                    output_latents = latents[:, extra_conditioning_num_latents:]
                else:
                    output_latents = latents

                # Unpack to standard latent format [B, C, F, H, W]
                unpacked_latents = self._unpack_latents(
                    output_latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )

                # Denormalize latents for reuse
                try:
                    unpacked_latents = self._denormalize_latents(
                        unpacked_latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                    )
                    logger.info(f"🔍 Denormalization successful: {unpacked_latents.shape}")
                except Exception as denorm_error:
                    logger.error(f"❌ Denormalization failed: {denorm_error}")
                    logger.warning(f"🔄 Using raw unpacked latents without denormalization")
                    # Keep unpacked_latents as-is without denormalization

            # Debug: Check latents before return
            logger.info(f"🔍 PIPELINE RETURN - unpacked_latents type: {type(unpacked_latents)}")
            if unpacked_latents is not None:
                logger.info(f"🔍 PIPELINE RETURN - unpacked_latents shape: {unpacked_latents.shape}")
                logger.info(f"🔍 PIPELINE RETURN - unpacked_latents mean: {unpacked_latents.mean():.6f}")
            else:
                logger.error(f"❌ PIPELINE RETURN - unpacked_latents is None!")

            if not return_dict:
                return (video, unpacked_latents)
            # Use dict to include latents since LTXPipelineOutput may not support latents attribute
            return {"frames": video, "latents": unpacked_latents}
        else:
            # Debug: No latents return case
            logger.info(f"🔍 PIPELINE RETURN - return_latents=False, only returning video")
            if not return_dict:
                return (video,)
            return LTXPipelineOutput(frames=video)
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