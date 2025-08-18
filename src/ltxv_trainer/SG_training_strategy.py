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
from typing import Any

import torch
from pydantic import BaseModel, computed_field
from torch import Tensor

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import get_rope_scale_factors, prepare_video_coordinates
from ltxv_trainer.timestep_samplers import TimestepSampler

DEFAULT_FPS = 24  # FPS 메타가 없을 때 기본값


# --------------------------
# Batch container (동일)
# --------------------------
class TrainingBatch(BaseModel):
    latents: Tensor                # (B, Seq, D)  # ex) Seq = (prev_seq + curr_seq)
    targets: Tensor                # (B, Seq, D)  # prev 구간은 0, curr 구간만 유효

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
        모델 포워드 입력 규격 유지
        """
        return {
            "hidden_states": batch.latents,
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
    if "current_shot" in conds:
        c = conds["current_shot"]
        return c["prompt_embeds"], c["prompt_attention_mask"]
    else:
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

    def get_data_sources(self) -> list[str]:
        """
        표준 학습에 필요한 소스:
        - "latents": {"current_shot": {...}, "prev_shot": {... or SOS}}  # ← Dataset이 보장
        - "conditions": scene/text 조건 (prev 구분이 있더라도 curr만 쓰면 충분)
        """
        return ["latents", "conditions"]

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        # 1) Latents 언팩
        lat_node = batch["latents"]  # ex) {"current_shot": <dict>, "prev_shot": <dict or Tensor or None>, ...}
        curr_lat, F, H, W, fps = _unpack_latent_entry(lat_node["current_shot"])
        # ex) curr_lat: (B, Cseq, D)

        prev_lat = None
        if lat_node.get("prev_shot", None) is not None:
            # prev_shot이 SOS 텐서/실샷 모두 가능 (Dataset이 생성)
            prev_lat, _, _, _, _ = _unpack_latent_entry(lat_node["prev_shot"])
            # ex) prev_lat: (B, Pseq, D)

        # 2) Conditions 언팩 (scene-level이면 curr와 동일)
        prompt_embeds, prompt_attention_mask = _unpack_condition_entry(batch["conditions"])

        # 3) 노이즈 샘플 & 시그마 (curr에만 적용)
        sigmas = timestep_sampler.sample_for(curr_lat)         # (B, Cseq, 1) 또는 전략 구현에 따라
        # ↓ 아래 연산에서 (B,1,1) 브로드캐스트를 기대하므로 reshape
        if sigmas.dim() > 3:
            raise ValueError("Unexpected sigma shape; expected (B,1,1) style broadcast.")
        sigmas = sigmas.view(curr_lat.shape[0], 1, 1)          # (B,1,1)

        noise = torch.randn_like(curr_lat, device=curr_lat.device)  # (B, Cseq, D)
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
        concat_lat, Pseq, Cseq = _concat_prev_curr(prev_lat, noisy_curr)  # (B, P+C, D)

        # 6) conditioning mask 구성
        #    prev 전체 True(조건), curr은 first_frame만 True
        if Pseq > 0:
            prev_mask = torch.ones(curr_lat.shape[0], Pseq, dtype=torch.bool, device=curr_lat.device)  # (B, Pseq)=True
            conditioning_mask = torch.cat([prev_mask, first_mask_curr], dim=1)  # (B, P+C)
        else:
            conditioning_mask = first_mask_curr  # (B, C)

        # 7) 타깃 구성: prev 구간은 0 (마스킹되므로 영향 X), curr 구간은 (noise - clean)
        targets_curr = noise - curr_lat  # (B, Cseq, D)
        if Pseq > 0:
            zeros_prev = torch.zeros(curr_lat.shape[0], Pseq, curr_lat.shape[2], device=curr_lat.device, dtype=targets_curr.dtype)
            targets = torch.cat([zeros_prev, targets_curr], dim=1)  # (B, P+C, D)
        else:
            targets = targets_curr  # (B, C, D)

        # 8) timestep 생성: prev=0, curr=round(sigmas*1000)
        sampled_t = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()  # (B,)
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_t)  # (B, P+C)

        # 9) ROPE scale & video coords (prev+curr 길이에 맞추어 준비)
        rope_scale = get_rope_scale_factors(fps)
        # seq_mult = 2 if Pseq > 0 else 1  # prev를 붙이면 2배

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
            video_coords=None,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """
        마스킹 MSE
        - prev 전체 & curr의 첫 프레임(조건)은 제외
        - targets는 prev 구간 0으로 채워져 있음 (안전)
        """
        loss = (model_pred - batch.targets).pow(2)                    # (B, Seq, D)
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()  # (B, Seq, 1)
        loss = loss.mul(loss_mask).div(loss_mask.mean())              # 평균 정규화
        return loss.mean()


# ------------------------------------------------------
# ReferenceVideo (IC-LoRA): prev-shot 자동 폴백 지원
# ------------------------------------------------------
class ReferenceVideoTrainingStrategy(TrainingStrategy):
    def __init__(self, conditioning_config: ConditioningConfig):
        super().__init__(conditioning_config)

    def get_data_sources(self) -> dict[str, str]:
        """
        기존: 별도 ref_latents_dir 필요.
        보강: ref_latents_dir가 없거나 batch에 없으면 dataset의 prev_shot을 레퍼런스로 사용(폴백)
        """
        mapping = {"latents": "latents", "conditions": "conditions"}
        if getattr(self.conditioning_config, "reference_latents_dir", None):
            mapping[self.conditioning_config.reference_latents_dir] = "ref_latents"
        return mapping

    def prepare_batch(self, batch: dict[str, dict[str, Tensor]], timestep_sampler: TimestepSampler) -> TrainingBatch:
        lat_node = batch["latents"]

        # 1) 타깃(latents) 언팩 (curr)
        curr_lat, F, H, W, fps = _unpack_latent_entry(lat_node["current_shot"])  # (B, Cseq, D)

        # 2) 레퍼런스 확보
        ref_lat = None
        # (a) 설정된 별도 디렉토리에서 제공되면 사용
        if "ref_latents" in batch:
            ref_lat, _, _, _, _ = _unpack_latent_entry(batch["ref_latents"])     # (B, Rseq, D)
        # (b) 없으면 prev_shot을 레퍼런스로 폴백
        elif lat_node.get("prev_shot", None) is not None:
            ref_lat, _, _, _, _ = _unpack_latent_entry(lat_node["prev_shot"])    # (B, Pseq, D)

        # 3) 텍스트 조건
        prompt_embeds, prompt_attention_mask = _unpack_condition_entry(batch["conditions"])

        # 4) curr에만 노이즈
        sigmas = timestep_sampler.sample_for(curr_lat)  # 기대 모양 브로드캐스트
        sigmas = sigmas.view(curr_lat.shape[0], 1, 1)
        noise = torch.randn_like(curr_lat, device=curr_lat.device)
        noisy_curr = (1 - sigmas) * curr_lat + sigmas * noise

        # 5) curr 첫 프레임 conditioning
        first_mask_curr = self._create_first_frame_conditioning_mask(
            batch_size=curr_lat.shape[0],
            sequence_length=curr_lat.shape[1],
            height=H,
            width=W,
            device=curr_lat.device,
        )
        noisy_curr = torch.where(first_mask_curr.unsqueeze(-1), curr_lat, noisy_curr)

        # 6) ref(항상 conditioning) + noisy_curr concat
        if ref_lat is None:
            # 레퍼런스가 전혀 없다면 표준전략과 동일 동작 (경고)
            logger.warning("No explicit ref_latents nor prev_shot; falling back to Standard-like behavior.")
            ref_mask = None
            combined, Rseq, Cseq = _concat_prev_curr(None, noisy_curr)
        else:
            combined, Rseq, Cseq = _concat_prev_curr(ref_lat, noisy_curr)
            ref_mask = torch.ones(curr_lat.shape[0], Rseq, dtype=torch.bool, device=curr_lat.device)

        # 7) conditioning mask
        if Rseq > 0:
            conditioning_mask = torch.cat([ref_mask, first_mask_curr], dim=1)  # (B, R+C)
        else:
            conditioning_mask = first_mask_curr  # (B, C)

        # 8) targets (ref 구간 0, curr 구간 noise - clean)
        targets_curr = noise - curr_lat
        if Rseq > 0:
            zeros_ref = torch.zeros(curr_lat.shape[0], Rseq, curr_lat.shape[2], device=curr_lat.device, dtype=targets_curr.dtype)
            targets = torch.cat([zeros_ref, targets_curr], dim=1)
        else:
            targets = targets_curr

        # 9) timesteps
        sampled_t = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()  # (B,)
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_t)  # (B, R+C)

        # 10) ROPE & coords (ref+curr 2배)
        rope_scale = get_rope_scale_factors(fps)
        seq_mult = 2 if Rseq > 0 else 1
        raw_coords = prepare_video_coordinates(
            num_frames=F, height=H, width=W,
            batch_size=combined.shape[0],
            sequence_multiplier=seq_mult,
            device=combined.device,
        )
        prescaled_f = raw_coords[..., 0] * rope_scale[0]
        prescaled_h = raw_coords[..., 1] * rope_scale[1]
        prescaled_w = raw_coords[..., 2] * rope_scale[2]
        video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)  # (B, 3, R+C)

        return TrainingBatch(
            latents=combined,
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
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """
        마스킹 MSE (ref + curr)
        - ref 구간: conditioning → 로스 제외
        - curr 첫 프레임: conditioning → 로스 제외
        """
        loss = (model_pred - batch.targets).pow(2)
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()


# --------------- Factory ---------------
def get_training_strategy(conditioning_config: ConditioningConfig) -> TrainingStrategy:
    mode = conditioning_config.mode
    if mode == "none":
        strategy = StandardTrainingStrategy(conditioning_config)
    elif mode == "reference_video":
        strategy = ReferenceVideoTrainingStrategy(conditioning_config)
    else:
        raise ValueError(f"Unknown conditioning mode: {mode}")

    logger.debug(f"🎯 Using {strategy.__class__.__name__}")
    return strategy
