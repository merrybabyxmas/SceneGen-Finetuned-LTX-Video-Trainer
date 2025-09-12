#!/usr/bin/env python
"""
Multi-shot video generation script for SceneGen-Finetuned LTX-Video model.

This script generates sequential multi-shot videos using:
1. SOS token conditioning for the first shot
2. Previous shot conditioning for subsequent shots
3. Same prompt across all shots for narrative consistency
"""

import sys
from pathlib import Path
from typing import Optional, List
import argparse
import torch
from diffusers.utils import export_to_video
from torch.amp import autocast
import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ltxv_trainer.model_loader import load_ltxv_components
from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
from ltxv_trainer.SG_datasets import SOSTokenLatents


class MultiShotVideoGenerator:
    """Multi-shot video generator using SceneGen training approach."""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_version: str = "LTXV_2B_0.9.5",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the multi-shot video generator.
        
        Args:
            checkpoint_path: Path to the LoRA checkpoint (.safetensors)
            model_version: Base model version to use
            device: Device to run on
            torch_dtype: Torch data type for inference
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_version = model_version
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipeline = None
        self.sos_token_generator = None
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
    def load_model(self):
        """Load the multi-shot pipeline and apply LoRA weights."""
        print(f"🔄 Loading SceneGen multi-shot pipeline...")
        print(f"Base model: {self.model_version}")
        print(f"Target device: {self.device}")
        
        # Set CUDA device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)
            print(f"Set CUDA device to: {self.device}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before loading: {torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB")
        
        # Load base model components
        components = load_ltxv_components(
            model_source=self.model_version,
            load_text_encoder_in_8bit=False,  # Disable 8-bit for device consistency
            transformer_dtype=self.torch_dtype,
            vae_dtype=self.torch_dtype
        )
        
        # Move all components to the correct device
        print(f"Moving components to {self.device}...")
        components.transformer = components.transformer.to(self.device, dtype=self.torch_dtype)
        components.vae = components.vae.to(self.device, dtype=self.torch_dtype)
        components.text_encoder = components.text_encoder.to(self.device)
        components.scheduler = components.scheduler
        # Ensure tokenizer doesn't have device conflicts
        if hasattr(components.tokenizer, 'to'):
            components.tokenizer = components.tokenizer.to(self.device)
        print("✅ All components moved to device")
        
        # Initialize SOS token generator
        self.sos_token_generator = SOSTokenLatents(d_model=128, use_zero_init=False)
        self.sos_token_generator = self.sos_token_generator.to(self.device)
        print(f"✅ SOS token generator moved to {self.device}")
        
        # Create multi-shot pipeline (but without clean replacement)
        self.pipeline = SGMultiShotPipeline(
            scheduler=components.scheduler,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            transformer=components.transformer,
            sos_token_generator=self.sos_token_generator,
        )
        
        # Set device override and ensure all components are on correct device
        self.pipeline._device_override = torch.device(self.device)
        
        # Force move pipeline to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Explicitly ensure all sub-components are on target device
        if hasattr(self.pipeline, 'text_encoder'):
            self.pipeline.text_encoder = self.pipeline.text_encoder.to(self.device)
        if hasattr(self.pipeline, 'tokenizer') and hasattr(self.pipeline.tokenizer, 'to'):
            self.pipeline.tokenizer = self.pipeline.tokenizer.to(self.device)
            
        print(f"✅ Multi-shot pipeline loaded on {self.device}")
        
        # Load LoRA weights
        print(f"Loading LoRA weights from: {self.checkpoint_path}")
        self.pipeline.load_lora_weights(str(self.checkpoint_path))
        
        # Disable CPU offload for debugging device issues
        print("⚠️ CPU offload disabled for device consistency")
        
        # Additional memory optimizations
        if hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing()
            print("✅ Attention slicing enabled")
        
        if hasattr(self.pipeline, 'enable_vae_slicing'):
            self.pipeline.enable_vae_slicing()
            print("✅ VAE slicing enabled")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after loading: {torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB")
        
        print("✅ Multi-shot pipeline loaded successfully!")
        
    def generate_multi_shot_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_shots: int = 3,
        width: int = 768,
        height: int = 448,
        num_frames: int = 17,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate a multi-shot video sequence.
        
        Args:
            prompt: Text prompt for the entire sequence
            negative_prompt: Negative prompt
            num_shots: Number of shots to generate
            width: Video width in pixels
            height: Video height in pixels  
            num_frames: Number of frames per shot
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for prompt following
            seed: Random seed for reproducibility
            output_dir: Directory to save videos
            
        Returns:
            List of paths to generated video files
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")
            
        if seed is not None:
            torch.manual_seed(seed)
            
        # Create output directory
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"multishot_videos_{timestamp}"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"🎬 Generating {num_shots}-shot video sequence...")
        print(f"Prompt: {prompt}")
        print(f"Dimensions: {width}x{height}x{num_frames}")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        print(f"Output directory: {output_dir}")
        
        video_paths = []
        prev_latent = None
        
        # Generate each shot sequentially
        for shot_idx in range(num_shots):
            print(f"\n🎥 Generating shot {shot_idx + 1}/{num_shots}...")
            
            # Set up generator for reproducibility
            shot_seed = seed + shot_idx if seed is not None else None
            generator = torch.Generator(device=self.device).manual_seed(shot_seed) if shot_seed else None
            
            # Clear GPU cache before each shot
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU memory before shot {shot_idx + 1}: {torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB")
            
            # Prepare pipeline inputs
            pipeline_inputs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_reference_comparison": True,
            }
            
            # Handle conditioning for multi-shot generation
            if shot_idx == 0:
                # First shot: use SOS token conditioning
                pipeline_inputs["use_sos_conditioning"] = True
                print("   🏁 Using SOS token conditioning for first shot")
            else:
                # Subsequent shots: use previous shot as conditioning
                pipeline_inputs["prev_latent"] = prev_latent
                pipeline_inputs["use_prev_conditioning"] = True
                print(f"   🔗 Using previous shot conditioning for shot {shot_idx + 1}")
            
            try:
                # Generate the shot
                with torch.inference_mode():
                    # Fix device type handling for autocast
                    device_type = self.device.split(':')[0] if isinstance(self.device, str) else str(self.device).split(':')[0]
                    with autocast(device_type, dtype=self.torch_dtype):
                        # Use multishot pipeline with modified conditioning
                        result = self.pipeline(**pipeline_inputs)
                        
                        if hasattr(result, 'frames') and result.frames:
                            video = result.frames[0]  # Get first video from batch
                            
                            # Save the video
                            safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in prompt)[:50]
                            video_filename = f"shot_{shot_idx+1:02d}_{safe_prompt}.mp4"
                            video_path = output_dir / video_filename
                            
                            export_to_video(video, str(video_path), fps=8)
                            video_paths.append(str(video_path))
                            
                            print(f"   ✅ Shot {shot_idx + 1} saved: {video_filename}")
                            
                            # Store latent for next shot conditioning
                            if hasattr(result, 'latents') and result.latents is not None:
                                prev_latent = result.latents.clone()
                                print(f"   📦 Stored latent for next shot: {prev_latent.shape}")
                            elif shot_idx < num_shots - 1:
                                print(f"   ⚠️ Warning: No latent stored for next shot")
                                
                        else:
                            print(f"   ❌ Failed to generate shot {shot_idx + 1}: No frames in result")
                            
            except Exception as e:
                print(f"   ❌ Error generating shot {shot_idx + 1}: {e}")
                continue
                
        # Create a combined video if we have multiple shots
        if len(video_paths) > 1:
            try:
                combined_path = output_dir / f"combined_multishot_{len(video_paths)}shots.mp4"
                self._create_combined_video(video_paths, str(combined_path))
                video_paths.append(str(combined_path))
                print(f"\n🎞️ Combined video created: {combined_path.name}")
            except Exception as e:
                print(f"\n⚠️ Could not create combined video: {e}")
        
        print(f"\n🎉 Multi-shot generation complete!")
        print(f"📁 Generated {len(video_paths)} files in: {output_dir}")
        
        return video_paths
    
    def _create_combined_video(self, video_paths: List[str], output_path: str):
        """Create a combined video from individual shots using ffmpeg."""
        import subprocess
        
        # Create a text file listing all videos
        list_file = Path(output_path).parent / "video_list.txt"
        
        with open(list_file, 'w') as f:
            for video_path in video_paths[:-1] if video_paths[-1].endswith('combined_multishot') else video_paths:
                f.write(f"file '{Path(video_path).absolute()}'\n")
        
        # Use ffmpeg to concatenate videos
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            list_file.unlink()  # Clean up temporary file
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            list_file.unlink(missing_ok=True)
            raise


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Generate multi-shot videos with SceneGen")
    
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
        help="Text prompt for the entire multi-shot sequence"
    )
    parser.add_argument(
        "--negative_prompt", 
        type=str, 
        default="jitter, artifacts, wobble, blur, distortion",
        help="Negative prompt"
    )
    parser.add_argument(
        "--num_shots", 
        type=int, 
        default=3,
        help="Number of shots to generate"
    )
    parser.add_argument(
        "--model_version", 
        type=str, 
        default="LTXV_2B_0.9.5",
        help="Base model version"
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
        help="Number of frames per shot"
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
        "--output_dir", 
        type=str,
        help="Output directory for videos"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = MultiShotVideoGenerator(
        checkpoint_path=args.checkpoint,
        model_version=args.model_version,
        device=args.device
    )
    
    # Load model
    generator.load_model()
    
    # Generate multi-shot video
    video_paths = generator.generate_multi_shot_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_shots=args.num_shots,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print(f"\n🎯 Final Results:")
    for i, path in enumerate(video_paths, 1):
        print(f"  {i}. {Path(path).name}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        print("🎬 SceneGen Multi-Shot Video Generator")
        print("\nExample usage:")
        print("python generate_multishot_video.py \\")
        print("  --checkpoint outputs/checkpoints/lora_weights_step_00800.safetensors \\")
        print("  --prompt 'a woman walking through a garden with flowers blooming' \\")
        print("  --num_shots 4 --device cuda:0")
        print("\nFor help: python generate_multishot_video.py --help")
    else:
        main()