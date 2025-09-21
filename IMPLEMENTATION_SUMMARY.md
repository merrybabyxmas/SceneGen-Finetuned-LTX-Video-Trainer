# Token-Based Multi-Shot Video Generation Implementation

## Overview

This implementation adds comprehensive token-based multi-shot video generation capabilities to the LTX-Video trainer, including stable tokens for intra-shot continuity and transition tokens for inter-shot transitions, along with scenario-based inference control.

## ✅ Completed Components

### 1. Core Token System (`src/ltxv_trainer/token_utils.py`)

**VideoTokenEmbeddings Class:**
- Learnable stable token (s): For intra-shot temporal continuity
- Learnable transition token (t): For inter-shot transition modeling
- Initialized with small values (std=0.02) for stable training
- Hidden dimension: 3072 (matches transformer)

**Token Insertion Functions:**
- `insert_stable_tokens()`: Inserts stable tokens between frames within shots
  - Pattern: `[z1, s, z2, s, z3]` for 3 frames
- `insert_transition_token()`: Inserts transition token between shots
- `preprocess_multishot_with_tokens()`: Complete preprocessing pipeline

**Utility Functions:**
- `create_token_aware_conditioning_mask()`: Different strengths for token types
- `extract_token_positions()`: Position tracking for analysis

### 2. Training Strategy Integration (`src/ltxv_trainer/SG_training_strategy.py`)

**Key Updates:**
- Continuous time coordinate generation (fixed duplicate indices)
- Gaussian noise initialization (replaced SOSTokenLatents)
- Token-aware batch preparation with extensive logging
- Flow matching loss integration: `L = MSE(Vθ(x|xp), x1c - x0)`

**Coordinate Generation Fix:**
```python
# Before: torch.cat([ref_coords, curr_coords], dim=1)
# After: Single call with total_frames for continuous indexing
total_frames = 2 * latent_num_frames
scaled_video_ids = prepare_video_coords(
    batch_size=B,
    num_frames=total_frames,  # Continuous: [0..F-1, F..2F-1]
    height=latent_height,
    width=latent_width
)
```

### 3. Pipeline Updates (`src/ltxv_trainer/SG_multishot_pipeline.py`)

**Coordinate Generation:**
- Fixed continuous time indexing across multiple shots
- Token embeddings support integration
- Proper latent frame calculation scope

### 4. Scenario Inference System (`src/ltxv_trainer/scenario_inference.py`)

**ScenarioVideoGenerator Class:**
- Parse scenario strings: `"shot1,stable,shot2,transition,shot3"`
- Build complete token sequences from scenario descriptions
- Flow matching inference integration (with placeholder)
- Complete pipeline from scenario string to generated video

**Key Methods:**
- `parse_scenario()`: String parsing and validation
- `build_scenario_sequence()`: Token sequence construction
- `run_inference()`: Flow matching inference
- `generate_from_scenario()`: End-to-end generation

### 5. Debug and Validation Systems

**Mathematical Validation (`validate_token_logic.py`):**
- Token insertion calculations
- Sequence length verification
- Range boundary checks
- Conditioning mask logic

**Integration Tests:**
- `test_token_logging.py`: Logging functionality
- `test_complete_token_system.py`: Comprehensive system test
- `test_system_with_torch.py`: PyTorch environment test

## 🔧 Technical Details

### Token Insertion Logic

**Stable Tokens Within Shots:**
- Original: `[F1, F2, F3]` → With stables: `[F1, s, F2, s, F3]`
- Formula: `frames_with_stables = F + max(0, F-1)`

**Transition Tokens Between Shots:**
- `[prev_shot_with_stables, t, curr_shot_with_stables]`
- Total frames: `prev_with_stables + 1 + curr_with_stables`

**Conditioning Mask Strengths:**
- Regular frames: 1.0 (previous shot) / 0.0 (current shot training targets)
- Stable tokens: 0.5 (medium conditioning)
- Transition tokens: 0.8 (strong conditioning)

### Coordinate Generation

**Before (Problematic):**
```python
ref_coords = prepare_video_coords(..., num_frames=F)     # [0, 1, 2, ..., F-1]
curr_coords = prepare_video_coords(..., num_frames=F)    # [0, 1, 2, ..., F-1]
combined = torch.cat([ref_coords, curr_coords], dim=1)   # [0,1,2,F-1, 0,1,2,F-1]
```

**After (Fixed):**
```python
total_frames = 2 * F
combined_coords = prepare_video_coords(..., num_frames=total_frames)  # [0,1,2..F-1,F,F+1..2F-1]
```

## 🚀 Usage

### Training with Token Processing

```bash
conda activate ltxv
PYTHONPATH=./src python3 scripts/train_pc_cfm.py configs/ltxv_2b_pc_cfm.yaml
```

### Scenario-Based Inference

```python
from ltxv_trainer.scenario_inference import ScenarioVideoGenerator
from ltxv_trainer.token_utils import VideoTokenEmbeddings

# Initialize
token_embeddings = VideoTokenEmbeddings(hidden_dim=3072)
generator = ScenarioVideoGenerator(token_embeddings, transformer_model, device)

# Generate from scenario
scenario = "shot1,stable,shot2,transition,shot3"
generated_sequence, metadata = generator.generate_from_scenario(
    scenario, shot_latents_dict
)
```

### Testing

```bash
# Mathematical validation (no PyTorch required)
python3 validate_token_logic.py

# Complete system test (requires PyTorch)
conda activate ltxv
PYTHONPATH=./src python3 test_system_with_torch.py
```

## 📊 System Architecture

```
Input: Multi-shot video data
  ↓
VideoTokenEmbeddings: Learnable stable & transition tokens
  ↓
Token Insertion:
  - Stable tokens within shots: [z1, s, z2, s, z3]
  - Transition tokens between shots: [shot1_with_stables, t, shot2_with_stables]
  ↓
Coordinate Generation: Continuous time indices [0..F-1, F..2F-1, ...]
  ↓
Flow Matching Training: L = MSE(Vθ(x|xp), x1c - x0)
  ↓
Scenario Inference: "shot1,stable,shot2,transition,shot3" → Generated video
```

## 🧪 Validation Results

All mathematical validation tests pass:
- ✅ Token insertion calculations
- ✅ Sequence length computations
- ✅ Range boundary verification
- ✅ Conditioning mask assignments
- ✅ Token pattern generation

## 📝 Key Files Modified/Created

1. **Created:** `src/ltxv_trainer/token_utils.py` - Core token system
2. **Created:** `src/ltxv_trainer/scenario_inference.py` - Scenario-based inference
3. **Modified:** `src/ltxv_trainer/SG_training_strategy.py` - Training integration
4. **Modified:** `src/ltxv_trainer/SG_multishot_pipeline.py` - Pipeline fixes
5. **Modified:** `src/ltxv_trainer/trainer.py` - Debug fixes
6. **Created:** Multiple validation and test scripts

## 🎯 Next Steps

The system is now ready for:
1. **Training:** Run training with token processing enabled
2. **Evaluation:** Test multi-shot video generation quality
3. **Scenario Testing:** Experiment with different scenario patterns
4. **Fine-tuning:** Adjust token conditioning strengths based on results

## 🔍 Logging

Comprehensive logging is implemented throughout the system with emojis for easy identification:
- 🎭 Token embeddings and initialization
- 🔧 Stable token insertion
- 🔄 Transition token insertion
- 🎬 Multi-shot preprocessing
- 🚀 Training strategy operations
- 🧪 Testing and validation

Monitor logs during training to verify token processing is working correctly.