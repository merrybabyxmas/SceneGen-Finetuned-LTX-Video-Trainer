# LTXV Training Improvements

## 문제 분석 및 해결

### 1. 🔍 Train Loss 진동 원인 및 해결책

**주요 원인들:**
- **Flow Matching 파라미터화 문제**: `shifted_logit_normal` 샘플러가 multi-shot 학습에 부적합
- **Learning Rate 너무 높음**: 0.0002는 flow matching 모델에 과도함
- **Gradient Clipping 너무 강함**: 0.1로 설정되어 그래디언트 과도하게 잘림
- **Linear LR Scheduler**: 선형 감소가 flow matching에 최적이 아님

**해결책 (새 config: `ltxv_2b_0.9.5_improved.yaml`):**
```yaml
optimization:
  learning_rate: 1e-4      # 2e-4 → 1e-4 (50% 감소)
  max_grad_norm: 1.0       # 0.1 → 1.0 (10배 증가)
  scheduler_type: "cosine" # linear → cosine
  scheduler_params:
    eta_min: 1e-6         # 최소 학습률 설정

flow_matching:
  timestep_sampling_mode: "uniform"  # shifted_logit_normal → uniform
```

### 2. 🔇 로깅 최적화

**변경사항:**
- Wandb 로그 주기: 10 step → 25 step
- Progress bar 로그: 50 step → 100 step  
- Epoch 로그: 매번 → 5번마다
- Validation 로그: main process만

### 3. 📊 Timestep 정보 추가

**Debug frames 파일명 개선:**
- 기존: `step_001200_curr_clean_first_frame.png`
- 개선: `step_001200_t0.745_curr_clean_first_frame.png`

**Batch visualization 파일명 개선:**
- 기존: `epoch_00_batch_50_curr_curr_noisy.mp4`
- 개선: `epoch_00_batch_50_curr_curr_noisy_t0.745.mp4`

### 4. 🎥 ODE 중간 단계 저장

**새로운 기능:**
- Pipeline에 `save_intermediate_steps`, `save_step_interval` 파라미터 추가
- 50 스텝 중 매 10 스텝마다 중간 결과 저장
- `outputs/intermediate_steps/` 디렉토리에 저장
- 파일명: `step_001200_prompt_0_ode_step_10_t0.745.mp4`

## 📁 파일 변경 사항

### 수정된 파일들
1. **`src/ltxv_trainer/trainer.py`**
   - 로깅 빈도 감소
   - Timestep 정보 추가
   - 중간 단계 저장 메서드 추가

2. **`src/ltxv_trainer/ltxv_pipeline.py`**
   - 중간 단계 저장 기능 구현
   - ODE solver 루프에 저장 로직 추가

3. **`configs/ltxv_2b_0.9.5_improved.yaml`** (새 파일)
   - 최적화된 하이퍼파라미터 설정
   - 안정적인 학습을 위한 조정

## 🚀 사용법

### 개선된 config로 훈련 시작
```bash
cd dongwoo/SceneGen-Finetuned-LTX-Video-Trainer

PYTHONPATH=./src \
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1,2 \
python scripts/train_distributed.py configs/ltxv_2b_0.9.5_improved.yaml --num_processes 2
```

### 출력 파일들

**Debug frames:** `outputs/debug_frames/`
- `step_001200_t0.745_curr_clean_first_frame.png`
- `step_001200_t0.745_curr_noisy_first_frame.png` 
- `step_001200_t0.745_curr_denoised_first_frame.png`
- `step_001200_prev_clean_first_frame.png`

**Batch visualization:** `outputs/batch_visualization/`
- `epoch_00_batch_50_curr_curr_clean.mp4`
- `epoch_00_batch_50_curr_curr_noisy_t0.745.mp4`
- `epoch_00_batch_50_prev_prev_clean.mp4`

**ODE 중간 단계:** `outputs/intermediate_steps/`
- `step_001200_prompt_0_ode_step_00_t1.000.mp4`
- `step_001200_prompt_0_ode_step_10_t0.745.mp4`
- `step_001200_prompt_0_ode_step_20_t0.512.mp4`
- `step_001200_prompt_0_ode_step_30_t0.298.mp4`
- `step_001200_prompt_0_ode_step_40_t0.156.mp4`
- `step_001200_prompt_0_ode_step_49_t0.000.mp4`

## 🎯 예상 효과

1. **Loss 안정화**: Cosine scheduler와 적절한 learning rate로 smooth한 convergence
2. **학습 속도 향상**: 불필요한 로깅 제거로 I/O 부하 감소
3. **디버깅 효율성**: Timestep 정보로 더 정확한 분석 가능
4. **추론 과정 시각화**: ODE solver의 각 단계를 시각적으로 확인 가능

이제 더 안정적이고 효율적인 LTXV 파인튜닝을 진행할 수 있습니다! 🎉