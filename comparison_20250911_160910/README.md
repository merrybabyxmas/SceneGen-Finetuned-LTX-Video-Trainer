# Single Shot vs Multi-Shot Comparison

## Generation Settings
- **Model**: outputs/checkpoints/lora_weights_step_01500.safetensors
- **Prompt**: "the full shot of the earth"
- **Negative Prompt**: "jitter, artifacts, wobble, blur, distortion"
- **Dimensions**: 768x448x17
- **Steps**: 50
- **Guidance Scale**: 1.5
- **Seed**: 42
- **Device**: cuda:3
- **Multi-shot Count**: 2

## Results

### Single Shot
✅ Generated: single_shot.mp4

### Multi-Shot
✅ Generated 3 files:
- shot_01_the full shot of the earth.mp4
- shot_02_the full shot of the earth.mp4
- combined_multishot_2shots.mp4

## Analysis
- Single shot provides a baseline for quality and content
- Multi-shot should show temporal continuity and scene progression
- Compare for artifacts, consistency, and narrative flow

Generated on: 2025-09-11 16:10:07
