"""
Training strategies for different conditioning modes (with prev-shot support).

- StandardTrainingStrategy:
    prev-shot(클린) + curr-shot(노이즈) 를 시퀀스 차원으로 concat
    → prev 구간은 전부 conditioning(True), curr의 첫 프레임만 선택적 conditioning
    → 타깃/로스는 curr-shot에만 적용 (마스킹됨)

- ReferenceVideoTrainingStrategy (IC-LoRA 스타일):
    기존 별도 ref_latents 디렉토리가 있으면 그대로 사용
    없으면 dataset이 제공한 prev-shot을 레퍼런스로 사용(자동 폴백)
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Union

import torch
import torch.nn.functional as F
from pydantic import BaseModel, computed_field
from torch import Tensor

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import get_rope_scale_factors, prepare_video_coordinates
from ltxv_trainer.timestep_samplers import TimestepSampler, UniformTimestepSampler

DEFAULT_FPS = 24  # FPS 메타가 없을 때 기본값


# --------------------------
# Packing/unpacking functions (from SGMultiShotPipeline)
# --------------------------
def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """
    Pack latents from [B, C, F, H, W] to [B, F // p_t * H // p * W // p, C * p_t * p * p].
    """
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


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    """
    Unpack latents from [B, S, D] to [B, C, F, H, W].
    """
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    """Normalize latents across the channel dimension [B, C, F, H, W]"""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


def _prepare_video_ids(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare video coordinate IDs."""
    # Debug prints to identify the issue
    logger.info(f"_prepare_video_ids: num_frames={num_frames}, height={height}, width={width}")

    # Check for zero dimensions
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid dimensions: num_frames={num_frames}, height={height}, width={width}")

    latent_sample_coords = torch.meshgrid(
        torch.arange(0, num_frames, patch_size_t, device=device),
        torch.arange(0, height, patch_size, device=device),
        torch.arange(0, width, patch_size, device=device),
        indexing="ij",
    )
    latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
    latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

    # Debug the tensor shapes before reshape
    seq_len = num_frames * height * width
    logger.info(f"latent_coords shape before reshape: {latent_coords.shape}, target seq_len: {seq_len}")

    if seq_len == 0:
        raise ValueError(f"Sequence length is 0: num_frames={num_frames} * height={height} * width={width} = {seq_len}")

    latent_coords = latent_coords.reshape(batch_size, -1, seq_len)
    return latent_coords


def _scale_video_ids(
    video_ids: torch.Tensor,
    scale_factor: int = 32,
    scale_factor_t: int = 8,
    frame_index: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """Scale video IDs (from pipeline)."""
    scaled_latent_coords = (
        video_ids
        * torch.tensor([scale_factor_t, scale_factor, scale_factor], device=video_ids.device)[None, :, None]
    )
    scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
    scaled_latent_coords[:, 0] += frame_index

    return scaled_latent_coords


def prepare_video_coords(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
    scale_factor: int = 32,
    scale_factor_t: int = 8,
    frame_index: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Prepare and scale video coordinates combining _prepare_video_ids and _scale_video_ids.
    This method follows the pattern from the standard ltxv_pipeline.
    """
    # First prepare video IDs
    video_ids = _prepare_video_ids(
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        device=device,
    )

    # Then scale them
    video_coords = _scale_video_ids(
        video_ids=video_ids,
        scale_factor=scale_factor,
        scale_factor_t=scale_factor_t,
        frame_index=frame_index,
        device=device,
    )

    return video_coords


def pc_cfm_loss(model_output: Tensor, X0: Tensor, X1c: Tensor, X1p: Tensor, lambda_val: float = 1.0) -> Tensor:
    """
    PC-CFM loss function with correct velocity field target.

    Args:
        model_output: Model prediction (velocity field v_t)
        X0: Initial noisy version of current shot (x_t)
        X1c: Clean current shot (x_1^c)
        X1p: Previous latent (x_1^p) or SOS token if first shot
        lambda_val: Lambda parameter for PC-CFM interpolation

    Returns:
        PC-CFM loss value
    """
    # PC-CFM velocity target: v_t = (x_1^c - x_t) + λ(x_1^p - x_1^c)
    # This represents the velocity field that moves from x_t towards a combination of current clean and previous
    velocity_target = (X1c - X0) + lambda_val * (X1p - X1c)
    return F.mse_loss(model_output, velocity_target)


# --------------------------
# Batch container (동일)
# --------------------------
class TrainingBatch(BaseModel):
    latents: Tensor                # (B, Seq, D)  # ex) Seq = (prev_seq + curr_seq)
    targets: Union[Tensor, dict]   # For StandardTraining: (B, curr_seq, D), For PC-CFM: dict with X0,X1c,X1p

    prompt_embeds: Tensor
    prompt_attention_mask: Tensor

    timesteps: Tensor              # (B, Seq)     # prev=0, curr=sampled
    sigmas: Tensor                 # (B, 1, 1)    # 노이즈 스케줄(브로드캐스트 용도)

    conditioning_mask: Tensor      # (B, Seq)     # True=conditioning(prev 전체 + curr 일부)

    num_frames: int
    height: int
    width: int
    fps: float

    rope_interpolation_scale: list[float]
    video_coords: Tensor | None = None
    
    # Additional info for extracting current part
    prev_seq_len: int = 0          # Length of previous sequence (0 if no prev)

    @computed_field
    @property
    def batch_size(self) -> int:
        return self.latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        return self.latents.shape[1]

    model_config = {"arbitrary_types_allowed": True}


# --------------------------
# Base strategy
# --------------------------
class TrainingStrategy(ABC):
    def __init__(self, conditioning_config: ConditioningConfig):
        self.conditioning_config = conditioning_config

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """
        ex) ["latents","conditions"]  or  {"latents":"latents","conditions":"conditions","ref_latents_dir":"ref_latents"}
        # dataset.sample_files : {"latent_conditions":[...], "text_conditions":[...]} 와의 매핑은 Dataset 쪽에서 처리
        """

    @abstractmethod
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        ...

    def _create_timesteps_from_conditioning_mask(
        self, conditioning_mask: Tensor, sampled_timestep_values: Tensor
    ) -> Tensor:
        """
        # conditioning_mask: (B, Seq)  True=conditioning → timestep=0
        # sampled_timestep_values: (B,) → 각 배치의 curr 구간에 복제

        return: timesteps (B, Seq)
        """
        expanded = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)  # (B, Seq)
        return torch.where(conditioning_mask, 0, expanded)

    def _create_first_frame_conditioning_mask(
        self, batch_size: int, sequence_length: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """
        curr-shot 내부에서 '첫 프레임'을 conditioning 처리할지 확률적으로 결정
        # True 범위: 첫 프레임의 (H*W) 토큰 → ex) (B, H*W)=True, 나머지 False
        """
        
        mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
        if (
            self.conditioning_config.first_frame_conditioning_p > 0
            and random.random() < self.conditioning_config.first_frame_conditioning_p
        ):
            first_frame_end = min(height * width, sequence_length)  # 안전장치
            mask[:, :first_frame_end] = True
        return mask

    @staticmethod
    def prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """
        모델 포워드 입력 규격 유지 (검증 파이프라인과 동일한 패킹 처리)
        """
        latents = batch.latents  # [B, seq_len, dim] format from dataset

        # 데이터셋의 sequence format을 검증 파이프라인과 동일하게 처리
        # 현재 latents는 이미 sequence format [B, seq_len, dim]이므로
        # 검증 파이프라인에서와 같이 packed latents로 간주

        # 만약 향후 5D tensor [B, C, F, H, W] format이 필요하다면:
        # 1. sequence를 5D로 unpacking: _unpack_latents 사용
        # 2. 다시 sequence로 packing: _pack_latents 사용
        # 하지만 현재는 이미 sequence format이므로 그대로 사용

        return {
            "hidden_states": latents,
            "encoder_hidden_states": batch.prompt_embeds,
            "timestep": batch.timesteps,
            "encoder_attention_mask": batch.prompt_attention_mask,
            "num_frames": batch.num_frames,
            "height": batch.height,
            "width": batch.width,
            "rope_interpolation_scale": batch.rope_interpolation_scale,
            "video_coords": batch.video_coords,
            "return_dict": False,
        }

    @abstractmethod
    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        ...


# --------------------------
# 유틸: dataset → 통일 언팩
# --------------------------
def _unpack_latent_entry(entry: Any) -> tuple[Tensor, int, int, int, float]:
    """
    entry 예시
    1) 텐서 직접 저장된 경우: entry = <Tensor: (B, Seq, D)>
    2) dict로 메타 포함: entry = {
            "latents": Tensor(B, Seq, D),
            "num_frames": Tensor([F,...]),
            "height": Tensor([H,...]),
            "width": Tensor([W,...]),
            "fps": Tensor([..., ...])  # 또는 없음
       }

    return: (latents, F, H, W, fps)
    """
    if torch.is_tensor(entry):
        # 메타가 없으면 합리적 기본값 추정 필요 → 여기서는 안전하게 더미 값
        # ex) (B, F*H*W, D) 구조라 가정할 수 없으므로, 호출부에서 메타를 별도로 전달하는 편이 더 정확.
        raise ValueError("Latent entry missing meta. Save dict with num_frames/height/width/fps.")
    else:
        lat = entry["latents"]                   # (B, Seq, D)
        F = int(entry["num_frames"][0].item())
        H = int(entry["height"][0].item())
        W = int(entry["width"][0].item())
        fps_t = entry.get("fps", None)
        if fps_t is not None and not torch.all(fps_t == fps_t[0]):
            logger.warning(f"Different FPS in batch: {fps_t.tolist()}, using first.")
        fps = float((fps_t[0].item() if fps_t is not None else DEFAULT_FPS))
        return lat, F, H, W, fps


def _unpack_condition_entry(conds: Any) -> tuple[Tensor, Tensor]:
    """
    conds 예시
    1) scene-level만 있는 경우: conds = {"prompt_embeds": (B, T, D), "prompt_attention_mask": (B, T)}
    2) prev/curr 구분 저장: conds = {
            "current_shot": {"prompt_embeds":..., "prompt_attention_mask":...},
            "prev_shot":    {"prompt_embeds":..., "prompt_attention_mask":...} (옵션)
       }
    """

    return conds["prompt_embeds"], conds["prompt_attention_mask"]


def _concat_prev_curr(prev_lat: Tensor | None, curr_lat: Tensor) -> tuple[Tensor, int, int]:
    """
    prev_lat: (B, Pseq, D) 또는 None
    curr_lat: (B, Cseq, D)
    return: (concat, Pseq, Cseq)

    # concat = [prev(clean) | curr(noisy)] @ seq-dim
    """
    if prev_lat is None:
        return curr_lat, 0, curr_lat.shape[1]
    return torch.cat([prev_lat, curr_lat], dim=1), prev_lat.shape[1], curr_lat.shape[1]


# ----------------------------------------
# Standard: prev-shot 지원 (새 규약에 맞게)
# ----------------------------------------
class StandardTrainingStrategy(TrainingStrategy):
    def __init__(self, conditioning_config: ConditioningConfig):
        super().__init__(conditioning_config)

    def get_data_sources(self) -> dict[str, str]:
        """
        표준 학습에 필요한 소스:
        - "latents": 현재 샷의 latent 데이터 (latent_conditions)
        - "conditions": scene/text 조건 (text_conditions)
        참고: prev_conditions는 데이터셋에서 자동으로 추가됨
        """
        return {"latents": "latent_conditions", "conditions": "text_conditions"}
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        # 1) Latents 언팩
        curr_lat, F, H, W, fps = _unpack_latent_entry(batch["latent_conditions"])

        # Debug print batch metadata
        logger.info(f"Standard batch metadata: F={F}, H={H}, W={W}, fps={fps}")
        logger.info(f"Current latent shape: {curr_lat.shape}")

        # Validate metadata
        if F <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"Invalid metadata from dataset: F={F}, H={H}, W={W}")

        prev_lat = None
        if batch.get("prev_conditions", None) is not None:
            prev_lat, _, _, _, _ = _unpack_latent_entry(batch["prev_conditions"])

        # 2) Conditions 언팩 (scene-level이면 curr와 동일)
        prompt_embeds, prompt_attention_mask = _unpack_condition_entry(batch["text_conditions"])
        
        
 
        # 3) 노이즈 샘플 & 시그마 (curr에만 적용)
        sigmas = timestep_sampler.sample_for(curr_lat)         # (B, Cseq, 1) 또는 전략 구현에 따라
        # ↓ 아래 연산에서 (B,1,1) 브로드캐스트를 기대하므로 reshape
        if sigmas.dim() > 3:
            raise ValueError("Unexpected sigma shape; expected (B,1,1) style broadcast.")
        sigmas = sigmas.view(curr_lat.shape[0], 1, 1)          # (B,1,1)

        noise = torch.randn_like(curr_lat, device=curr_lat.device)  # (B, Cseq, D)
        # CRITICAL FIX: Correct flow matching interpolation
        # Flow matching: x_t = (1-t) * x_0 + t * noise
        noisy_curr = (1 - sigmas) * curr_lat + sigmas * noise       # (B, Cseq, D)

        # 4) curr 내부 '첫 프레임 conditioning' 적용: 첫 프레임 토큰은 클린으로 대체
        first_mask_curr = self._create_first_frame_conditioning_mask(
            batch_size=curr_lat.shape[0],
            sequence_length=curr_lat.shape[1],
            height=H,
            width=W,
            device=curr_lat.device,
        )  # (B, Cseq) True=clean keep
        noisy_curr = torch.where(first_mask_curr.unsqueeze(-1), curr_lat, noisy_curr)

        # 5) prev + curr concat
        concat_lat, Pseq, _ = _concat_prev_curr(prev_lat, noisy_curr)  # (B, P+C, D)

        # 6) conditioning mask 구성
        #    prev 전체 True(조건), curr은 first_frame만 True
        if Pseq > 0:
            prev_mask = torch.ones(curr_lat.shape[0], Pseq, dtype=torch.bool, device=curr_lat.device)  # (B, Pseq)=True
            conditioning_mask = torch.cat([prev_mask, first_mask_curr], dim=1)  # (B, P+C)
        else:
            conditioning_mask = first_mask_curr  # (B, C)

        # 7) 타깃 구성: Only current part targets (no prev padding needed)
        targets = noise - curr_lat   # (B, Cseq, D) - Only current part targets

        # 8) timestep 생성: prev=0, curr=sigmas (no scaling for flow matching)
        sampled_t = sigmas.squeeze(-1).squeeze(-1)  # (B,) keep as float [0,1]
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_t)  # (B, P+C)

        # 9) ROPE scale & video coords (검증 파이프라인과 동일한 방식)
        rope_scale = get_rope_scale_factors(fps)

        if Pseq > 0:
            # VAE compression ratios
            vae_spatial_compression_ratio = 32
            vae_temporal_compression_ratio = 8
            transformer_spatial_patch_size = 1
            transformer_temporal_patch_size = 1

            # F, H, W from dataset are already latent dimensions - don't divide by compression ratios!
            latent_num_frames = F
            latent_height = H
            latent_width = W

            # Debug prints
            logger.info(f"Dataset latent dims (already compressed): F={F}, H={H}, W={W}")
            logger.info(f"Using as latent dims: frames={latent_num_frames}, height={latent_height}, width={latent_width}")

            # Check for invalid latent dimensions
            if latent_num_frames <= 0 or latent_height <= 0 or latent_width <= 0:
                raise ValueError(f"Invalid latent dimensions: frames={latent_num_frames}, height={latent_height}, width={latent_width}")

            # Generate video coordinates for both shots with continuous time indices
            # Total frames = 2 * latent_num_frames (prev + curr)
            total_frames = 2 * latent_num_frames
            scaled_video_ids = prepare_video_coords(
                batch_size=concat_lat.shape[0],
                num_frames=total_frames,
                height=latent_height,
                width=latent_width,
                patch_size=transformer_spatial_patch_size,
                patch_size_t=transformer_temporal_patch_size,
                scale_factor=vae_spatial_compression_ratio,
                scale_factor_t=vae_temporal_compression_ratio,
                frame_index=0,
                device=concat_lat.device,
            )

            # Apply ROPE scaling and convert to final format
            video_coords = scaled_video_ids.float()
            video_coords[:, 0] = video_coords[:, 0] * rope_scale[0]
            video_coords[:, 1] = video_coords[:, 1] * rope_scale[1]
            video_coords[:, 2] = video_coords[:, 2] * rope_scale[2]
        else:
            video_coords = None

        return TrainingBatch(
            latents=concat_lat,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=F,
            height=H,
            width=W,
            fps=fps,
            rope_interpolation_scale=rope_scale,
            video_coords=video_coords,
            prev_seq_len=Pseq,
        )
        
        
        

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """
        Direct MSE on current part only (no masking needed)
        - model_pred: (B, curr_seq, D) - only current part prediction
        - targets: (B, curr_seq, D) - only current part targets
        """
        # Handle both Tensor and dict targets (for compatibility with PC-CFM)
        if isinstance(batch.targets, dict):
            # This shouldn't happen for StandardTrainingStrategy, but handle gracefully
            raise ValueError("StandardTrainingStrategy received dict targets - use ReferenceVideoTrainingStrategy for PC-CFM")
        
        # Apply masking only to current part first-frame conditioning if needed
        if batch.prev_seq_len > 0:
            # Extract current part conditioning mask (skip prev part)
            curr_conditioning_mask = batch.conditioning_mask[:, batch.prev_seq_len:]  # (B, curr_seq)
        else:
            # No prev part, use full conditioning mask
            curr_conditioning_mask = batch.conditioning_mask  # (B, curr_seq)
        
        # Compute loss only on non-conditioning tokens of current part
        loss = (model_pred - batch.targets).pow(2)                          # (B, curr_seq, D)
        loss_mask = (~curr_conditioning_mask.unsqueeze(-1)).float()         # (B, curr_seq, 1)
        masked_loss = loss * loss_mask                                      # Apply mask
        
        num_valid_tokens = loss_mask.sum()
        if num_valid_tokens > 0:
            return masked_loss.sum() / num_valid_tokens
        else:
            return torch.tensor(0.0, device=loss.device, requires_grad=True)


class ReferenceVideoTrainingStrategy(TrainingStrategy):
    """Modified Reference video training strategy with PC-CFM loss.

    This strategy implements training with previous shot conditioning where:
    - Previous shot latents (clean) are concatenated with target latents (noised)
    - Uses text prompts for conditioning (like StandardTrainingStrategy)
    - Uses SOS token for first shot when no previous shot available
    - PC-CFM loss instead of masked MSE loss
    - Video coordinates are doubled to handle concatenated sequence
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize the modified reference strategy.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> dict[str, str]:
        """Modified reference training requires latents, text conditions, and prev conditions."""
        return {
            "latents": "latent_conditions", 
            "conditions": "text_conditions"
            # Note: prev_conditions will be automatically added by dataset
        }

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for modified reference training with prev latents and text conditions."""
        # 1) Unpack current shot latents
        curr_lat, F, H, W, fps = _unpack_latent_entry(batch["latent_conditions"])
        B, curr_seq_len, D = curr_lat.shape

        # Debug print batch metadata
        logger.info(f"ReferenceVideo batch metadata: F={F}, H={H}, W={W}, fps={fps}")
        logger.info(f"Current latent shape: {curr_lat.shape}")

        # Validate metadata
        if F <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"Invalid metadata from dataset: F={F}, H={H}, W={W}")

        # Check if we have previous conditions
        has_prev = batch.get("prev_conditions") is not None
        logger.info(f"Has previous conditions: {has_prev}")
        
        # 2) Get previous shot latents or use SOS token
        ref_lat = None
        if batch.get("prev_conditions") is not None:
            # Use previous shot as reference
            ref_lat, _, _, _, _ = _unpack_latent_entry(batch["prev_conditions"])
            # logger.debug(f"Using previous shot as reference: {ref_lat.shape}")
        else:
            # First shot: generate Gaussian noise as reference
            ref_lat = torch.randn(B, curr_seq_len, D, device=curr_lat.device, dtype=curr_lat.dtype)
            logger.info(f"Generated Gaussian noise SOS as reference: {ref_lat.shape}")
        
        ref_seq_len = ref_lat.shape[1]
        
        # 3) Unpack text conditions (same as StandardTrainingStrategy)
        prompt_embeds, prompt_attention_mask = _unpack_condition_entry(batch["text_conditions"])
        
        # 4) Sample timesteps and add noise to current shot only
        sigmas = timestep_sampler.sample_for(curr_lat)
        if sigmas.dim() > 3:
            raise ValueError("Unexpected sigma shape")
        sigmas = sigmas.view(B, 1, 1)  # (B, 1, 1)
        
        noise = torch.randn_like(curr_lat, device=curr_lat.device)
        # CRITICAL FIX: Correct flow matching interpolation
        # Flow matching: x_t = (1-t) * x_0 + t * noise, where t ∈ [0,1]
        # Here: x_t = (1-sigma) * curr_lat + sigma * noise
        noisy_curr = (1 - sigmas) * curr_lat + sigmas * noise  # X0 in PC-CFM
        
        # 5) Apply first frame conditioning to current shot
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=B,
            sequence_length=curr_seq_len,
            height=H,
            width=W,
            device=curr_lat.device,
        )
        noisy_curr = torch.where(target_conditioning_mask.unsqueeze(-1), curr_lat, noisy_curr)
        
        # 6) Create combined conditioning mask
        # Reference tokens are always conditioning
        ref_conditioning_mask = torch.ones(B, ref_seq_len, dtype=torch.bool, device=curr_lat.device)
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)
        
        # 7) Create timesteps based on conditioning mask
        sampled_t = sigmas.squeeze(-1).squeeze(-1)  # (B,)
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_t)
        
        # 8) Store components for PC-CFM loss (not regular targets)
        # We need to store: X0 (noisy), X1c (clean curr), X1p (prev/SOS)
        targets = {
            # CRITICAL FIX: Flow matching velocity target v = x_1 - x_0
            # For interpolated path x_t = (1-t)*x_0 + t*x_1, velocity is v = x_1 - x_0
            'XO-X1c' : noise - curr_lat,  # v = clean - noise (direction toward clean)
            'X0': noisy_curr,      # Initial noisy version
            'X1c': curr_lat,       # Clean current shot  
            'X1p': ref_lat,
            'X1p-X1c': ref_lat - curr_lat,        # Previous shot or SOS token
            'X0-X1p+X1c' : noise - ref_lat - curr_lat 
        }
        
        # 9) Concatenate reference and noisy current in sequence dimension
        combined_latents = torch.cat([ref_lat, noisy_curr], dim=1)  # (B, ref_seq + curr_seq, D)
        
        # 10) Prepare video coordinates (검증 파이프라인과 동일한 방식)
        rope_scale_factors = get_rope_scale_factors(fps)

        # VAE compression ratios
        vae_spatial_compression_ratio = 32
        vae_temporal_compression_ratio = 8
        transformer_spatial_patch_size = 1
        transformer_temporal_patch_size = 1

        # F, H, W from dataset are already latent dimensions - don't divide by compression ratios!
        latent_num_frames = F
        latent_height = H
        latent_width = W

        # Debug prints
        logger.info(f"ReferenceVideo - Dataset latent dims (already compressed): F={F}, H={H}, W={W}")
        logger.info(f"ReferenceVideo - Using as latent dims: frames={latent_num_frames}, height={latent_height}, width={latent_width}")

        # Check for invalid latent dimensions
        if latent_num_frames <= 0 or latent_height <= 0 or latent_width <= 0:
            raise ValueError(f"Invalid latent dimensions: frames={latent_num_frames}, height={latent_height}, width={latent_width}")

        # Generate video coordinates for both shots with continuous time indices
        # Total frames = 2 * latent_num_frames (reference + current)
        total_frames = 2 * latent_num_frames
        scaled_video_ids = prepare_video_coords(
            batch_size=B,
            num_frames=total_frames,
            height=latent_height,
            width=latent_width,
            patch_size=transformer_spatial_patch_size,
            patch_size_t=transformer_temporal_patch_size,
            scale_factor=vae_spatial_compression_ratio,
            scale_factor_t=vae_temporal_compression_ratio,
            frame_index=0,
            device=curr_lat.device,
        )

        # Apply ROPE scaling and convert to final format
        video_coords = scaled_video_ids.float()
        video_coords[:, 0] = video_coords[:, 0] * rope_scale_factors[0]
        video_coords[:, 1] = video_coords[:, 1] * rope_scale_factors[1]
        video_coords[:, 2] = video_coords[:, 2] * rope_scale_factors[2]
        
        return TrainingBatch(
            latents=combined_latents,
            targets=targets,  # Changed: now contains PC-CFM components
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=F,
            height=H,
            width=W,
            fps=fps,
            rope_interpolation_scale=rope_scale_factors,
            video_coords=video_coords,
            prev_seq_len=ref_seq_len,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute PC-CFM loss on target portion only."""
        # Extract target portion from model prediction (last part corresponds to current shot)
        target_seq_len = batch.targets['X1c'].shape[1]  # Current shot sequence length
        target_pred = model_pred[:, -target_seq_len:]  # Extract current shot prediction
        
        # Get PC-CFM loss components
        # X0 = batch.targets['X0']   # Noisy current shot
        X1c = batch.targets['X1c'] # Clean current shot
        X1p = batch.targets['X1p'] # Previous shot or SOS token
        
        # Apply PC-CFM loss: target = (X1c - X0) + lambda * (X1p - X1c)
        # CRITICAL FIX: Use proper lambda value for PC-CFM (typical range: 0.1-1.0)
        if X1p.shape[1] != X1c.shape[1]:
            # If reference (X1p) has different sequence length, we need to handle it
            # This can happen when prev shot has different length than current shot
            if X1p.shape[1] > X1c.shape[1]:
                # Truncate reference to match current shot length
                X1p = X1p[:, :X1c.shape[1], :]
            else:
                # Pad reference to match current shot length (repeat last frame)
                pad_len = X1c.shape[1] - X1p.shape[1]
                last_frame = X1p[:, -1:, :].repeat(1, pad_len, 1)
                X1p = torch.cat([X1p, last_frame], dim=1)
                
        # CRITICAL FIX: Re-enable PC-CFM loss with proper lambda
        # Standard PC-CFM: target = (X1c - X0) + λ(X1p - X1c) = XO-X1 + λ(X1p - X1c)
        # For PC-CFM, we need to match sequence lengths properly
        lambda_val = 0  # Balance between current and previous shot influence
        pc_cfm_target = batch.targets["XO-X1c"] + lambda_val * batch.targets["X0-X1p+X1c"]
        
        
        # Extract target portion conditioning mask for masking
        target_conditioning_mask = batch.conditioning_mask[:, -target_seq_len:]
        
        # Compute MSE loss between model prediction and PC-CFM target
        loss = (target_pred - pc_cfm_target).pow(2)
        
        # Apply mask to exclude conditioning tokens from loss
        loss_mask = (~target_conditioning_mask.unsqueeze(-1)).float()
        masked_loss = loss * loss_mask
        
        # Compute mean loss over valid (non-conditioning) tokens
        num_valid_tokens = loss_mask.sum()
        if num_valid_tokens > 0:
            return masked_loss.sum() / num_valid_tokens
        else:
            return torch.tensor(0.0, device=loss.device, requires_grad=True)


def get_training_strategy(conditioning_config: ConditioningConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.

    Args:
        conditioning_config: Configuration for conditioning behavior

    Returns:
        The appropriate training strategy instance

    Raises:
        ValueError: If conditioning mode is not supported
    """
    conditioning_mode = conditioning_config.mode

    if conditioning_mode == "none":
        strategy = StandardTrainingStrategy(conditioning_config)
    elif conditioning_mode == "reference_video":
        strategy = ReferenceVideoTrainingStrategy(conditioning_config)
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")

    logger.debug(f"🎯 Using {strategy.__class__.__name__}")
    return strategy

if __name__ == "__main__":
    import time
    import torch
    from torch.utils.data import DataLoader

    # --------------------------
    # 0) 재현/디바이스 설정
    # --------------------------
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------
    # 1) 데이터셋 로드
    # --------------------------
    from ltxv_trainer.SG_datasets import PrecomputedDataset


    data_root = "/home/jeongseon38/datasets/videos/splits/.precomputed"
    
    print("loading data!")
    # DataLoader는 여기선 1개만 뽑아보는 스모크 테스트 용도로 사용
    ds = PrecomputedDataset(data_root)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    print("data successfully loaded!")

    # --------------------------
    # 2) 설정(전략/샘플러)
    # --------------------------


    # 모드 선택: "none" → Standard, "reference_video" → IC-LoRA 스타일
    cfg = ConditioningConfig(
        mode="none",                      # "none" | "reference_video"
        first_frame_conditioning_p=0.5,   # 첫 프레임을 conditioning으로 사용할 확률
        # reference_latents_dir="ref_latents",  # 필요 시 활성화
    )
    sampler = UniformTimestepSampler(min_value=0.0, max_value=1.0)


    # 전략 생성
    strategy = get_training_strategy(cfg)
    print(f"[INFO] Using strategy: {strategy.__class__.__name__}")

    # --------------------------
    # 3) 배치 → TrainingBatch
    # --------------------------
    t0 = time.time()
    raw_batch = next(iter(dl))  # Dataset이 dict를 반환한다고 가정
    t1 = time.time()

    # 주의: Dataset에서 뽑힌 텐서들이 CPU일 수 있으므로, 여기선
    # prepare_batch 내부에서의 디바이스 가정에 맞춰 그대로 전달.
    # (전략 내부 연산은 입력 텐서의 device를 사용)

    training_batch = strategy.prepare_batch(raw_batch, sampler)
    t2 = time.time()

    print("------------- Prepared TrainingBatch -------------")
    print(f"latents              : {tuple(training_batch.latents.shape)}  (B, Seq, D)")
    print(f"targets              : {tuple(training_batch.targets.shape)}")
    print(f"prompt_embeds        : {tuple(training_batch.prompt_embeds.shape)}")
    print(f"prompt_attention_mask: {tuple(training_batch.prompt_attention_mask.shape)}")
    print(f"timesteps            : {tuple(training_batch.timesteps.shape)}")
    print(f"sigmas               : {tuple(training_batch.sigmas.shape)}")
    print(f"conditioning_mask    : {tuple(training_batch.conditioning_mask.shape)}  (True=conditioning)")
    print(f"num_frames / HxW / fps: {training_batch.num_frames} / {training_batch.height}x{training_batch.width} / {training_batch.fps}")
    if training_batch.video_coords is not None:
        print(f"video_coords         : {tuple(training_batch.video_coords.shape)}  (B, 3, Seq)")
    print(f"rope_scale           : {training_batch.rope_interpolation_scale}")
    print("--------------------------------------------------")
    print(f"[Timing] dataloader: {(t1 - t0):.3f}s, prepare_batch: {(t2 - t1):.3f}s")

    # --------------------------
    # 4) 더미 모델로 forward & loss
    # --------------------------
    # 실제 모델 입력 규격에 맞춰 dict를 구성
    model_inputs = strategy.prepare_model_inputs(training_batch)
    # 실제 모델이 없다면, 동일 shape의 더미 예측을 생성(평균 0, 분산 동일)
    # 여기서는 간단히 targets에 노이즈를 더한 값을 예측으로 사용
    with torch.no_grad():
        model_pred = training_batch.targets + 0.05 * torch.randn_like(training_batch.targets)

    loss = strategy.compute_loss(model_pred, training_batch)
    print(f"[Loss] masked MSE: {loss.item():.6f}")

    # --------------------------
    # 5) (선택) GPU 이동 테스트
    # --------------------------
    # 실제 학습 코드에선 모델/배치 텐서를 accelerator.device로 맞춰야 합니다.
    # 여기선 스모크 테스트 차원에서 latents만 잠깐 옮겨보기:
    try:
        if torch.cuda.is_available():
            _ = training_batch.latents.to(device)
            print(f"[Device] Latents moved to {device} OK.")
    except Exception as e:
        print(f"[WARN] Device move test failed: {e}")

    print("[DONE] Strategy smoke test finished.")



    