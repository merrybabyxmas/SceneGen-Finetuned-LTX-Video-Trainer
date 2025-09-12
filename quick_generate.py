#!/usr/bin/env python
"""
Quick video generation script - simple interface for testing.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate_video import VideoGenerator

def quick_generate():
    """Quick generation with latest checkpoint."""
    
    # Use the latest checkpoint automatically
    checkpoint_dir = Path("outputs/checkpoints")
    if not checkpoint_dir.exists():
        print("❌ No checkpoints directory found!")
        return
        
    # Find latest LoRA checkpoint (not ComfyUI format)
    lora_checkpoints = list(checkpoint_dir.glob("lora_weights_step_*.safetensors"))
    if not lora_checkpoints:
        print("❌ No LoRA checkpoints found!")
        return
        
    # Sort by step number and get latest
    latest_checkpoint = sorted(lora_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    
    print(f"🔄 Using checkpoint: {latest_checkpoint}")
    
    # Create generator
    generator = VideoGenerator(
        checkpoint_path=str(latest_checkpoint),
        model_version="LTXV_2B_0.9.5",
        device="cuda"
    )
    
    # Load model
    generator.load_model()
    
    # Test prompts based on your training config
    test_prompts = [
        "a guy talking in front of a wooden piece of furniture",
        "a woman walking through a garden with flowers blooming",
        "a cat playing with a ball on a wooden floor",
        "a person cooking in a modern kitchen",
        "children playing in a park on a sunny day"
    ]
    
    print("Available test prompts:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i}. {prompt}")
    
    # Get user choice
    try:
        choice = input("\nSelect prompt (1-5) or enter custom prompt: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(test_prompts):
            selected_prompt = test_prompts[int(choice) - 1]
        else:
            selected_prompt = choice
            
        print(f"\n🎬 Generating video with prompt: '{selected_prompt}'")
        
        # Generate video
        output_path = generator.generate_video(
            prompt=selected_prompt,
            negative_prompt="jitter, artifacts, wobble, blur, distortion",
            width=768,
            height=448,
            num_frames=17,
            num_inference_steps=50,
            guidance_scale=1.5,
            seed=42
        )
        
        print(f"\n🎉 Video generated successfully!")
        print(f"📁 Saved to: {output_path}")
        
    except KeyboardInterrupt:
        print("\n❌ Generation cancelled by user")
    except Exception as e:
        print(f"❌ Error during generation: {e}")


if __name__ == "__main__":
    quick_generate()