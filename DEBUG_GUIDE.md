# 4-Scenario Debugging Guide for LTXV Multi-shot Generation

이 가이드는 LTXV 비디오 생성에서 4가지 시나리오를 테스트하여 multi-shot generation의 문제를 디버깅하는 방법을 설명합니다.

## 📋 4가지 시나리오

1. **Scenario 1: LoRA X & Single Shot Generation**
   - 기본 모델 사용 (LoRA 적용 안함)
   - 표준 LTXVideoPipeline 사용

2. **Scenario 2: LoRA X & Multi Shot Generation**
   - 기본 모델 사용 (LoRA 적용 안함)
   - SGMultiShotPipeline 사용 (1개 샷)

3. **Scenario 3: LoRA O & Single Shot Generation**
   - 학습된 LoRA 적용
   - 표준 LTXVideoPipeline 사용

4. **Scenario 4: LoRA O & Multi Shot Generation**
   - 학습된 LoRA 적용
   - SGMultiShotPipeline 사용 (1개 샷)

## 🚀 사용법

### 방법 1: Shell 스크립트 사용 (권장)

```bash
# 기본 프롬프트로 실행
./run_debug.sh

# 커스텀 프롬프트로 실행
./run_debug.sh "a professional portrait video of a person with blurry bokeh background"
```

### 방법 2: Python 스크립트 직접 실행

```bash
# PYTHONPATH 설정
export PYTHONPATH=./src:$PYTHONPATH

# 기본 설정으로 실행
python3 debug_four_scenarios.py

# 모든 옵션 지정
python3 debug_four_scenarios.py \
    --prompt "your custom prompt here" \
    --base-model "Lightricks/LTX-Video" \
    --lora-path "path/to/your/lora.safetensors" \
    --device cuda \
    --output-dir debug_outputs \
    --dtype bfloat16
```

## 📁 결과 파일

실행이 완료되면 `debug_outputs/` 디렉토리에 다음 파일들이 생성됩니다:

- `scenario_1_base_standard_YYYYMMDD_HHMMSS.mp4` - Scenario 1 비디오
- `scenario_2_base_multishot_YYYYMMDD_HHMMSS.mp4` - Scenario 2 비디오
- `scenario_3_lora_standard_YYYYMMDD_HHMMSS.mp4` - Scenario 3 비디오
- `scenario_4_lora_multishot_YYYYMMDD_HHMMSS.mp4` - Scenario 4 비디오
- `debug_report_YYYYMMDD_HHMMSS.json` - 상세한 결과 리포트

## 🔍 디버깅 분석 방법

### 1. 기본 기능 확인
먼저 Scenario 1이 성공하는지 확인하세요. 이것이 실패하면 기본 모델 로딩에 문제가 있습니다.

### 2. Multi-shot Pipeline 확인
Scenario 2가 성공하는지 확인하세요. 실패하면 SGMultiShotPipeline에 문제가 있을 수 있습니다.

### 3. LoRA 적용 확인
Scenario 3이 성공하는지 확인하세요. 실패하면 LoRA 가중치 로딩/적용에 문제가 있습니다.

### 4. 최종 통합 확인
Scenario 4가 성공하는지 확인하세요. 이것이 실패하면 LoRA + Multi-shot 조합에서 발생하는 문제입니다.

## 🛠️ 문제 해결

### 일반적인 에러들

**ModuleNotFoundError**:
```bash
# conda 환경 활성화 확인
conda activate your_env_name
```

**CUDA out of memory**:
```bash
# 더 작은 해상도로 테스트
python3 debug_four_scenarios.py --width 512 --height 288
```

**LoRA 로딩 실패**:
```bash
# LoRA 파일 경로 확인
ls -la outputs/checkpoints/*.safetensors
```

### 환경 요구사항

- Python 3.8+
- PyTorch 2.0+
- Diffusers 0.30+
- Transformers
- Accelerate
- SafeTensors

## 📊 결과 해석

각 시나리오의 성공/실패를 통해 다음을 파악할 수 있습니다:

| Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 | 추정 문제 |
|------------|------------|------------|------------|----------|
| ✅ | ✅ | ✅ | ❌ | LoRA + Multi-shot 호환성 |
| ✅ | ✅ | ❌ | ❌ | LoRA 가중치 로딩 |
| ✅ | ❌ | ✅ | ❌ | Multi-shot Pipeline |
| ❌ | ❌ | ❌ | ❌ | 기본 환경 설정 |

## 🔄 다음 단계

1. **결과 비교**: 생성된 비디오들을 시각적으로 비교
2. **로그 분석**: 콘솔 출력에서 에러 메시지 확인
3. **리포트 검토**: JSON 리포트에서 상세한 에러 정보 확인
4. **설정 조정**: 문제가 발견되면 해당 파이프라인의 설정 수정

이 디버깅 도구를 통해 multi-shot generation 문제의 정확한 원인을 파악하고 해결책을 찾을 수 있습니다.