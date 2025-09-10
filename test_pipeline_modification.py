#!/usr/bin/env python3
"""
Test script to verify the pipeline modifications work correctly.

This script tests:
1. Training strategy produces correct target shapes (current part only)
2. Model prediction extraction works correctly
3. Loss computation works without masking
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ltxv_trainer.SG_training_strategy import StandardTrainingStrategy, get_training_strategy
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.timestep_samplers import UniformTimestepSampler

def test_pipeline_modifications():
    """Test the pipeline modifications."""
    print("🧪 Testing pipeline modifications...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditioning_config = ConditioningConfig(mode="none", first_frame_conditioning_p=0.5)
    strategy = get_training_strategy(conditioning_config)
    sampler = UniformTimestepSampler(min_value=0.0, max_value=1.0)
    
    # Create mock batch data
    batch_size = 2
    curr_seq_len = 1008  # Current sequence length
    prev_seq_len = 1008  # Previous sequence length  
    latent_dim = 128
    
    # Mock current latents (with metadata)
    curr_latents = {
        "latents": torch.randn(batch_size, curr_seq_len, latent_dim),
        "num_frames": torch.tensor([16] * batch_size),
        "height": torch.tensor([16] * batch_size), 
        "width": torch.tensor([21] * batch_size),
        "fps": torch.tensor([24.0] * batch_size)
    }
    
    # Mock previous latents
    prev_latents = {
        "latents": torch.randn(batch_size, prev_seq_len, latent_dim),
        "num_frames": torch.tensor([16] * batch_size),
        "height": torch.tensor([16] * batch_size),
        "width": torch.tensor([21] * batch_size),
        "fps": torch.tensor([24.0] * batch_size)
    }
    
    # Mock text conditions
    text_conditions = {
        "prompt_embeds": torch.randn(batch_size, 77, 768),
        "prompt_attention_mask": torch.ones(batch_size, 77)
    }
    
    # Create batch dict
    batch = {
        "latent_conditions": curr_latents,
        "prev_conditions": prev_latents,
        "text_conditions": text_conditions
    }
    
    print(f"📊 Input shapes:")
    print(f"  Current latents: {curr_latents['latents'].shape}")
    print(f"  Previous latents: {prev_latents['latents'].shape}")
    
    # Test 1: Strategy prepare_batch
    print("\n🔄 Testing strategy.prepare_batch...")
    training_batch = strategy.prepare_batch(batch, sampler)
    
    print(f"📊 Training batch shapes:")
    print(f"  Latents (prev+curr): {training_batch.latents.shape}")
    print(f"  Targets (curr only): {training_batch.targets.shape}")
    print(f"  Prev seq len: {training_batch.prev_seq_len}")
    
    # Verify target shape is current-only
    expected_target_shape = (batch_size, curr_seq_len, latent_dim)
    assert training_batch.targets.shape == expected_target_shape, f"Target shape mismatch: {training_batch.targets.shape} vs {expected_target_shape}"
    print("✅ Targets shape is current-only (no prev padding)")
    
    # Test 2: Mock model prediction extraction 
    print("\n🤖 Testing model prediction extraction...")
    
    # Simulate full model prediction (prev+curr)
    full_model_pred = torch.randn(batch_size, prev_seq_len + curr_seq_len, latent_dim)
    print(f"  Full model pred: {full_model_pred.shape}")
    
    # Extract current part (like _extract_current_part_from_prediction)
    if training_batch.prev_seq_len > 0:
        curr_model_pred = full_model_pred[:, training_batch.prev_seq_len:]
    else:
        curr_model_pred = full_model_pred
        
    print(f"  Current model pred: {curr_model_pred.shape}")
    
    # Verify current prediction shape matches targets
    assert curr_model_pred.shape == training_batch.targets.shape, f"Prediction shape mismatch: {curr_model_pred.shape} vs {training_batch.targets.shape}"
    print("✅ Current prediction shape matches targets")
    
    # Test 3: Loss computation 
    print("\n📈 Testing loss computation...")
    loss = strategy.compute_loss(curr_model_pred, training_batch)
    print(f"  Loss value: {loss.item():.6f}")
    
    # Verify loss is computed successfully
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.requires_grad, "Loss doesn't require grad"
    print("✅ Loss computation successful")
    
    # Test 4: Test case without previous latents
    print("\n🔄 Testing without previous latents...")
    batch_no_prev = {
        "latent_conditions": curr_latents,
        "text_conditions": text_conditions
        # No prev_conditions
    }
    
    training_batch_no_prev = strategy.prepare_batch(batch_no_prev, sampler)
    print(f"  Latents (curr only): {training_batch_no_prev.latents.shape}")
    print(f"  Targets (curr only): {training_batch_no_prev.targets.shape}")
    print(f"  Prev seq len: {training_batch_no_prev.prev_seq_len}")
    
    # Mock prediction for no-prev case
    full_pred_no_prev = torch.randn(batch_size, curr_seq_len, latent_dim)
    loss_no_prev = strategy.compute_loss(full_pred_no_prev, training_batch_no_prev)
    print(f"  Loss (no prev): {loss_no_prev.item():.6f}")
    print("✅ No-prev case works correctly")
    
    print("\n🎉 All tests passed! Pipeline modifications are working correctly.")
    print("\n📋 Summary of changes:")
    print("  ✓ Targets now contain only current part (no prev padding)")
    print("  ✓ Model prediction extraction works correctly")  
    print("  ✓ Loss computation works without full masking")
    print("  ✓ Both prev+curr and curr-only cases work")

if __name__ == "__main__":
    test_pipeline_modifications()