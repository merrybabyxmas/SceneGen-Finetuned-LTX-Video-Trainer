#!/usr/bin/env python
"""
Compare single shot vs multishot video generation with identical settings.

This script generates both single shot and multishot videos using the same:
- Model checkpoint
- Prompt
- Seed
- Generation parameters

Outputs videos side-by-side for easy comparison.
"""

import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate_video import VideoGenerator
from generate_multishot_video import MultiShotVideoGenerator


def compare_generation(
    checkpoint_path: str,
    prompt: str,
    negative_prompt: str = "jitter, artifacts, wobble, blur, distortion",
    width: int = 768,
    height: int = 448,
    num_frames: int = 17,
    num_shots: int = 2,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: int = 42,
    device: str = "cuda:0",
    output_dir: str = None
):
    """
    Generate both single shot and multishot videos for comparison.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"comparison_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"🎬 Starting single shot vs multishot comparison")
    print(f"📁 Output directory: {output_path}")
    print(f"🎯 Settings:")
    print(f"   Prompt: {prompt}")
    print(f"   Dimensions: {width}x{height}x{num_frames}")
    print(f"   Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"   Seed: {seed}, Device: {device}")
    print(f"   Multishot: {num_shots} shots")
    print()
    
    results = {}
    
    # 1. Generate Single Shot
    print("🎥 Generating Single Shot Video...")
    try:
        single_generator = VideoGenerator(
            checkpoint_path=checkpoint_path,
            device=device
        )
        single_generator.load_model()
        
        single_output = single_generator.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_path=str(output_path / "single_shot.mp4")
        )
        
        results['single_shot'] = single_output
        print(f"✅ Single shot generated: {single_output}")
        
        # Clean up memory
        del single_generator
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Single shot generation failed: {e}")
        results['single_shot'] = None
    
    print()
    
    # 2. Generate Multi-Shot
    print(f"🎥 Generating Multi-Shot Video ({num_shots} shots)...")
    try:
        multi_generator = MultiShotVideoGenerator(
            checkpoint_path=checkpoint_path,
            device=device
        )
        multi_generator.load_model()
        
        multishot_outputs = multi_generator.generate_multi_shot_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_shots=num_shots,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_dir=str(output_path / "multishot")
        )
        
        results['multishot'] = multishot_outputs
        print(f"✅ Multi-shot generated: {len(multishot_outputs)} files")
        for i, path in enumerate(multishot_outputs, 1):
            print(f"   {i}. {Path(path).name}")
        
        # Clean up memory
        del multi_generator
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Multi-shot generation failed: {e}")
        results['multishot'] = None
    
    print()
    
    # 3. Create Comparison Summary
    print("📊 Comparison Summary:")
    print("=" * 50)
    
    if results['single_shot']:
        single_size = Path(results['single_shot']).stat().st_size / 1024
        print(f"Single Shot:")
        print(f"  📄 File: {Path(results['single_shot']).name}")
        print(f"  📏 Size: {single_size:.1f} KB")
    else:
        print(f"Single Shot: ❌ Failed")
    
    print()
    
    if results['multishot']:
        print(f"Multi-Shot ({num_shots} shots):")
        total_size = 0
        for path in results['multishot']:
            if Path(path).exists():
                size = Path(path).stat().st_size / 1024
                total_size += size
                print(f"  📄 {Path(path).name}: {size:.1f} KB")
        print(f"  📏 Total size: {total_size:.1f} KB")
    else:
        print(f"Multi-Shot: ❌ Failed")
    
    print()
    print(f"🎉 Comparison complete! Results saved in: {output_path}")
    
    # Create README with comparison details
    readme_content = f"""# Single Shot vs Multi-Shot Comparison

## Generation Settings
- **Model**: {checkpoint_path}
- **Prompt**: "{prompt}"
- **Negative Prompt**: "{negative_prompt}"
- **Dimensions**: {width}x{height}x{num_frames}
- **Steps**: {num_inference_steps}
- **Guidance Scale**: {guidance_scale}
- **Seed**: {seed}
- **Device**: {device}
- **Multi-shot Count**: {num_shots}

## Results

### Single Shot
{"✅ Generated: " + Path(results['single_shot']).name if results['single_shot'] else "❌ Failed"}

### Multi-Shot
"""
    
    if results['multishot']:
        readme_content += f"✅ Generated {len(results['multishot'])} files:\n"
        for path in results['multishot']:
            readme_content += f"- {Path(path).name}\n"
    else:
        readme_content += "❌ Failed\n"
    
    readme_content += f"""
## Analysis
- Single shot provides a baseline for quality and content
- Multi-shot should show temporal continuity and scene progression
- Compare for artifacts, consistency, and narrative flow

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Comparison details saved: {readme_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare single shot vs multishot video generation")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to LoRA checkpoint file (.safetensors)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative_prompt", 
        type=str, 
        default="jitter, artifacts, wobble, blur, distortion",
        help="Negative prompt"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=768,
        help="Video width"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=448,
        help="Video height"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=17,
        help="Number of frames"
    )
    parser.add_argument(
        "--num_shots", 
        type=int, 
        default=2,
        help="Number of shots for multishot generation"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=1.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_generation(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_shots=args.num_shots,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        print("🎬 Single Shot vs Multi-Shot Comparison Tool")
        print("\nExample usage:")
        print("python compare_single_vs_multishot.py \\")
        print("  --checkpoint outputs/checkpoints/lora_weights_step_01500.safetensors \\")
        print("  --prompt 'a bunch of people playing minecraft' \\")
        print("  --num_shots 2 --steps 20 --device cuda:1")
        print("\nFor help: python compare_single_vs_multishot.py --help")
    else:
        main()