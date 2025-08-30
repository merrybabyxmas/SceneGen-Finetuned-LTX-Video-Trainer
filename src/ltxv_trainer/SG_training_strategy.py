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
from ltxv_trainer.timestep_samplers import TimestepSampler, UniformTimestepSampler

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
        return {"latents": "latent_conditions", "conditions": "text_conditions"}
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        # 1) Latents 언팩
        curr_lat, F, H, W, fps = _unpack_latent_entry(batch["latent_conditions"])

        prev_lat = None
        if batch.get("prev_conditions", None) is not None:
            prev_lat, _, _, _, _ = _unpack_latent_entry(batch["prev_conditions"])

        # 2) Conditions 언팩 (scene-level이면 curr와 동일)
        prompt_embeds, prompt_attention_mask = _unpack_condition_entry(batch["text_conditions"])
        
        
        # print(f"-------------batch config-------------"
        #       f"curr_lat : {curr_lat.shape}"
        #       f"prev_lat : {prev_lat.shape}"
        #       f"prompt_embeds : {prompt_embeds.shape}"
        #       )

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
        
        
        if Pseq > 0:
            raw_coords = prepare_video_coordinates(
                num_frames=F, height=H, width=W,
                batch_size=concat_lat.shape[0],
                sequence_multiplier=2,              # ★ 핵심: prev + curr
                device=concat_lat.device,
            )
            prescaled_f = raw_coords[..., 0] * rope_scale[0]
            prescaled_h = raw_coords[..., 1] * rope_scale[1]
            prescaled_w = raw_coords[..., 2] * rope_scale[2]
            video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)  # (B, 3, P+C)
        else:
            video_coords = None

        # return TrainingBatch(
        #     latents=concat_lat,
        #     targets=targets,
        #     prompt_embeds=prompt_embeds,
        #     prompt_attention_mask=prompt_attention_mask,
        #     timesteps=timesteps,
        #     sigmas=sigmas,
        #     conditioning_mask=conditioning_mask,
        #     num_frames=F,
        #     height=H,
        #     width=W,
        #     fps=fps,
        #     rope_interpolation_scale=rope_scale,
        #     video_coords=video_coords,
        # )
        
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



if __name__ == "__main__":
    import os
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



    