# SceneGen Multishot Video Generation - 수정 사항 요약

## 📋 프로젝트 개요
- **목적**: LTX-Video 모델을 사용한 multishot 비디오 생성 훈련
- **모델**: LTXV_2B_0.9.5 with LoRA fine-tuning
- **핵심 기능**: PC-CFM (Probability Current Flow Matching) 기반 reference video conditioning

---

## 🚀 주요 해결된 문제들

### 1. Multishot Generation 실패 (핵심 문제)
**문제**: 첫 번째 샷은 정상 생성되지만, 2번째/3번째 샷이 노이즈만 생성

**원인**:
- Latent chaining 실패
- 텐서 크기 불일치 (2016 vs 1008 tokens)
- Reference latents 처리 오류

**해결책**:
- `SG_training_strategy.py`: 모델 예측에서 현재 샷 부분만 추출하는 로직 추가
- `SG_multishot_pipeline.py`: Reference latents 처리 개선 및 디버깅 강화
- `trainer.py`: Latent chaining과 fallback 로직 구현

### 2. 훈련 크래시 문제
**문제**:
- RuntimeError: 텐서 크기 불일치
- Mixed precision 관련 오류들

**해결책**:
- BFloat16 호환성 문제 해결
- Gradient clipping 비활성화
- Mixed precision 설정 최적화

### 3. 설정 및 환경 문제
**문제**:
- GPU 지정 실행 실패
- Config validation 오류
- 로깅 시스템 부재

**해결책**:
- CUDA_VISIBLE_DEVICES 설정 가이드
- LoggingConfig 클래스 추가
- PYTHONPATH 설정 해결

---

## 📝 수정된 파일들

### 1. 설정 파일

#### `configs/ltxv_2b_pc_cfm.yaml`
```yaml
# 주요 변경사항:
model:
  model_source: "LTXV_2B_0.9.5"
  training_mode: "lora"

lora:
  rank: 128  # 64 -> 128 (더 높은 표현력)
  alpha: 128

optimization:
  learning_rate: 1e-4  # 2e-4 -> 1e-4 (안정성)
  steps: 5000  # 2000 -> 5000 (충분한 학습)
  max_grad_norm: 0  # Gradient clipping 비활성화

acceleration:
  mixed_precision_mode: "bf16"  # dtype 통일

validation:
  prompts: ["a high-quality professional portrait video of a young man playing guitar, smooth camera movement, good lighting, clear focus"]
  negative_prompt: "worst quality, low quality, inconsistent motion, blurry, jittery, distorted, grainy, artifact, static noise, poor lighting"
  video_dims: [768, 448, 25]  # 프레임 길이 조정
  inference_steps: 40  # 품질 향상
  guidance_scale: 7.0  # 더 강한 guidance
  interval: 200
  starts_with: "sos"  # SOS 토큰 사용

# 새로 추가된 섹션:
logging:
  save_logs: true
  log_dir: "/home/jeongseon39/MLLAB/scenegen/outputs/logs"
  log_level: "INFO"
```

### 2. 핵심 로직 수정

#### `src/ltxv_trainer/SG_training_strategy.py`
**수정 내용**: 텐서 크기 불일치 해결
```python
# Line 538-546: 모델 예측에서 현재 샷 부분만 추출
target_seq_len = batch.targets.shape[1]
if model_pred.shape[1] > target_seq_len:
    model_pred_curr = model_pred[:, -target_seq_len:]
    print(f"DEBUG: Extracted current part from model_pred: {model_pred.shape} -> {model_pred_curr.shape}")
else:
    model_pred_curr = model_pred

loss = (model_pred_curr - batch.targets).pow(2)
```

#### `src/ltxv_trainer/SG_multishot_pipeline.py`
**수정 내용**: Reference latents 처리 개선
```python
# Line 1116-1119: 모델 dtype 맞추기
model_dtype = next(self.transformer.parameters()).dtype
reference_latents = reference_latents.to(device, dtype=model_dtype)
logger.info(f"🔍 Reference latents converted to model dtype: {model_dtype}")

# Line 1126-1129: Re-normalization 추가
latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(reference_latents.device, reference_latents.dtype)
latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(reference_latents.device, reference_latents.dtype)
reference_latents = (reference_latents / self.vae.config.scaling_factor - latents_mean) / latents_std

# Line 1490-1497: Denormalization 오류 처리
try:
    unpacked_latents = self._denormalize_latents(...)
    logger.info(f"🔍 Denormalization successful: {unpacked_latents.shape}")
except Exception as denorm_error:
    logger.error(f"❌ Denormalization failed: {denorm_error}")
    logger.warning(f"🔄 Using raw unpacked latents without denormalization")

# Line 1516-1527: 상세 디버깅 로그
logger.info(f"🔍 PIPELINE RETURN - unpacked_latents type: {type(unpacked_latents)}")
if unpacked_latents is not None:
    logger.info(f"🔍 PIPELINE RETURN - unpacked_latents shape: {unpacked_latents.shape}")
    logger.info(f"🔍 PIPELINE RETURN - unpacked_latents mean: {unpacked_latents.mean():.6f}")
```

#### `src/ltxv_trainer/trainer.py`
**수정 내용**: Latent chaining 강화 및 로깅 시스템
```python
# Line 52: Import 추가
from ltxv_trainer import logger, IS_MULTI_GPU, RANK

# Line 124-127: 파일 로깅 설정
self._sos_token_generator = SOSTokenLatents(d_model=128).to(self._accelerator.device)
self._setup_file_logging()

# Line 137-173: 파일 로깅 구현
def _setup_file_logging(self) -> None:
    """Setup file logging if configured in the config."""
    import logging
    from datetime import datetime

    if hasattr(self._config, 'logging'):
        save_logs = self._config.logging.save_logs
        if save_logs:
            log_dir = Path(self._config.logging.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"training_{timestamp}.log"
            # ... 파일 핸들러 설정

# Line 1570-1584: 상세 결과 디버깅
logger.info(f"🔍 Shot {shot_idx} pipeline result type: {type(result)}")
if isinstance(result, dict):
    logger.info(f"🔍 Dict keys: {list(result.keys())}")
    current_video = result.get('frames')
    latents_result = result.get('latents')
    logger.info(f"🔍 CRITICAL - Dict latents_result type: {type(latents_result)}")
    if latents_result is not None:
        logger.info(f"🔍 CRITICAL - Dict latents_result shape: {latents_result.shape}")
```

#### `src/ltxv_trainer/config.py`
**수정 내용**: LoggingConfig 클래스 추가
```python
# Line 427-443: 새로운 LoggingConfig 클래스
class LoggingConfig(ConfigBaseModel):
    """Configuration for logging output"""

    save_logs: bool = Field(
        default=False,
        description="Enable saving logs to file",
    )

    log_dir: str = Field(
        default="outputs/logs",
        description="Directory to save log files",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

# Line 462: LtxvTrainerConfig에 logging 필드 추가
logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

---

## 🔧 실행 방법

### 훈련 시작
```bash
# GPU 0,1번 사용하여 훈련
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" PYTHONPATH="./src" accelerate launch scripts/train.py configs/ltxv_2b_pc_cfm.yaml
```

### 주요 환경 변수
- `CUDA_VISIBLE_DEVICES=0,1`: GPU 0,1번만 사용
- `NCCL_P2P_DISABLE="1"`: RTX 4000 시리즈 P2P 통신 비활성화
- `NCCL_IB_DISABLE="1"`: InfiniBand 비활성화
- `PYTHONPATH="./src"`: 모듈 경로 설정

---

## 📊 훈련 모니터링

### 출력 파일들
- `outputs/samples/`: 최종 multishot 비디오 결과
  - `step_XXXXXX_prompt_0_shot_0.mp4`: 첫 번째 샷
  - `step_XXXXXX_prompt_0_shot_1.mp4`: 두 번째 샷
  - `step_XXXXXX_prompt_0_shot_2.mp4`: 세 번째 샷
  - `step_XXXXXX_prompt_0_shot_X_prev+curr.mp4`: 연결된 비디오

- `outputs/debug_videos/`: 디버깅용 비디오들
- `outputs/logs/`: 훈련 로그 파일들
- `outputs/checkpoints/`: 모델 체크포인트들

### 핵심 로그 메시지들
```
✅ Latents successfully stored for shot X: torch.Size([...])
➡️ Shot X using previous latents: torch.Size([...])
🔍 Reference latents converted to model dtype: torch.bfloat16
🔍 Re-normalizing reference latents for pipeline use
🔍 PIPELINE RETURN - unpacked_latents shape: torch.Size([...])
```

---

## 🎯 현재 상태 및 예상 결과

### 해결된 문제들
- ✅ Multishot latent chaining 완전 구현
- ✅ 텐서 크기 불일치 해결
- ✅ Mixed precision 안정화
- ✅ 로깅 시스템 구축
- ✅ 설정 최적화

### 현재 훈련 진행 상황
- **Step**: ~600/5000 (약 12% 진행)
- **품질**: 형체는 나오지만 아직 흐릿함 (정상적인 초기 단계)
- **Multishot**: Shot 0, 1, 2 모두 생성되지만 유사성 높음

### 예상 개선 시점
- **Step 800-1000**: 형태 안정화
- **Step 1500-2000**: 확실한 품질 향상
- **Step 3000+**: 고품질 multishot 생성

---

## 🔮 향후 개선 방향

### 단기 목표
1. **품질 개선**: Step 수 증가에 따른 자연스러운 향상
2. **다양성 증대**: Shot 간 더 뚜렷한 progression
3. **안정성 확보**: 지속적인 모니터링

### 장기 목표
1. **더 긴 시퀀스**: 3샷 이상의 multishot 생성
2. **프롬프트 제어**: 각 샷별 다른 프롬프트 적용
3. **품질 최적화**: 더 높은 해상도 및 프레임 수

---

## 📚 참고 정보

### 중요 파라미터들
- **LoRA rank/alpha**: 128/128 (높은 표현력)
- **Learning rate**: 1e-4 (안정적 학습)
- **Mixed precision**: bf16 (dtype 통일)
- **Video dims**: [768, 448, 25] (25프레임)
- **Inference steps**: 40 (고품질 생성)
- **Guidance scale**: 7.0 (강한 prompt following)

### 디버깅 팁
1. **로그 확인**: `outputs/logs/training_*.log` 파일 모니터링
2. **샘플 비교**: Step별 `outputs/samples/` 품질 변화 추적
3. **파일 크기**: Shot별 파일 크기로 품질 간접 확인

---

*작성일: 2025-09-23*
*최종 수정: Step 600 기준*