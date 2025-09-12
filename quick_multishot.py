#!/usr/bin/env python
"""
Quick multi-shot generation script with predefined scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate_multishot_video import MultiShotVideoGenerator

def quick_multishot_generate():
    """Quick multi-shot generation with latest checkpoint and predefined scenarios."""
    
    # Use the latest checkpoint automatically
    checkpoint_dir = Path("outputs/checkpoints")
    if not checkpoint_dir.exists():
        print("❌ No checkpoints directory found!")
        return
        
    # Find latest LoRA checkpoint
    lora_checkpoints = list(checkpoint_dir.glob("lora_weights_step_*.safetensors"))
    if not lora_checkpoints:
        print("❌ No LoRA checkpoints found!")
        return
        
    # Sort by step number and get latest
    latest_checkpoint = sorted(lora_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    
    print(f"🔄 Using checkpoint: {latest_checkpoint}")
    
    # Create generator
    generator = MultiShotVideoGenerator(
        checkpoint_path=str(latest_checkpoint),
        model_version="LTXV_2B_0.9.5",
        device="cuda:0"
    )
    
    # Load model
    generator.load_model()
    
    # Multi-shot scenarios based on your training data
    scenarios = [
        {
            "prompt": "a guy talking in front of a wooden piece of furniture",
            "description": "Training validation prompt - indoor conversation scene",
            "shots": 3
        },
        {
            "prompt": "a woman walking through a garden with flowers blooming",
            "description": "Outdoor nature scene with movement",
            "shots": 4
        },
        {
            "prompt": "children playing in a park on a sunny day",
            "description": "Dynamic outdoor scene with multiple subjects",
            "shots": 3
        },
        {
            "prompt": "a person cooking in a modern kitchen",
            "description": "Indoor activity scene with object interactions",
            "shots": 4
        },
        {
            "prompt": "a cat walking across a wooden floor then sitting by a window",
            "description": "Pet scene with location transition",
            "shots": 3
        }
    ]
    
    print("\n🎬 Available Multi-Shot Scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['description']}")
        print(f"   Prompt: \"{scenario['prompt']}\"")
        print(f"   Shots: {scenario['shots']}")
        print()
    
    # Get user choice
    try:
        choice = input("Select scenario (1-5) or enter custom prompt: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(scenarios):
            selected_scenario = scenarios[int(choice) - 1]
            prompt = selected_scenario["prompt"]
            num_shots = selected_scenario["shots"]
            print(f"\n🎬 Selected scenario: {selected_scenario['description']}")
        else:
            prompt = choice
            num_shots = int(input("Enter number of shots (2-5): ").strip() or "3")
            num_shots = max(2, min(5, num_shots))  # Clamp between 2-5
        
        print(f"\n🎥 Generating {num_shots}-shot video:")
        print(f"📝 Prompt: '{prompt}'")
        print(f"🎯 This will create sequential shots with:")
        print(f"   • Shot 1: SOS token conditioning (fresh start)")
        print(f"   • Shots 2-{num_shots}: Previous shot conditioning (narrative flow)")
        
        # Generate multi-shot video
        video_paths = generator.generate_multi_shot_video(
            prompt=prompt,
            negative_prompt="jitter, artifacts, wobble, blur, distortion, inconsistent motion",
            num_shots=num_shots,
            width=768,
            height=448,
            num_frames=17,
            num_inference_steps=50,
            guidance_scale=1.5,
            seed=42
        )
        
        print(f"\n🎉 Multi-shot generation completed!")
        print(f"📂 Generated files:")
        for path in video_paths:
            print(f"   • {Path(path).name}")
        
    except KeyboardInterrupt:
        print("\n❌ Generation cancelled by user")
    except Exception as e:
        print(f"❌ Error during generation: {e}")


if __name__ == "__main__":
    quick_multishot_generate()