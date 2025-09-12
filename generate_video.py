#!/usr/bin/env python
"""
Video generation script for SceneGen-Finetuned LTX-Video model.

This script loads a trained LoRA checkpoint and generates videos from text prompts
using the fine-tuned LTXV model.
"""

import sys
from pathlib import Path
from typing import Optional
import argparse
import torch
from diffusers.utils import export_to_video

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ltxv_trainer.model_loader import load_ltxv_components
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline


class VideoGenerator:
    """Video generator using trained LTXV model with LoRA weights."""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_version: str = "LTXV_2B_0.9.5",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the video generator.
        
        Args:
            checkpoint_path: Path to the LoRA checkpoint (.safetensors)
            model_version: Base model version to use
            device: Device to run on ('cuda' or 'cpu')
            torch_dtype: Torch data type for inference
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_version = model_version
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipeline = None
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
    def load_model(self):
        """Load the base model and apply LoRA weights."""
        print(f"Loading base model: {self.model_version}")
        print(f"Target device: {self.device}")
        
        # Set CUDA device first
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)
            print(f"Set CUDA device to: {self.device}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before loading: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f}GB")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB")
        
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
        print("✅ All components moved to device")
        
        # Create pipeline
        self.pipeline = LTXConditionPipeline(
            transformer=components.transformer,
            scheduler=components.scheduler,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
        )
        
        # Ensure pipeline is on correct device
        print(f"Ensuring pipeline is on device: {self.device}")
        # Don't call .to() again since components are already on device
        
        # Load LoRA weights
        print(f"Loading LoRA weights from: {self.checkpoint_path}")
        self.pipeline.load_lora_weights(str(self.checkpoint_path))
        
        # Enable memory optimizations
        try:
            self.pipeline.enable_model_cpu_offload()
            print("✅ CPU offload enabled")
        except:
            print("⚠️ CPU offload not available, using sequential CPU offload")
            self.pipeline.enable_sequential_cpu_offload()
        
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
        
        print("✅ Model loaded successfully!")
        
    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 768,
        height: int = 448,
        num_frames: int = 17,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt to avoid certain content
            width: Video width in pixels
            height: Video height in pixels  
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for prompt following
            seed: Random seed for reproducibility
            output_path: Custom output path (optional)
            
        Returns:
            Path to the generated video file
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if seed is not None:
            torch.manual_seed(seed)
            
        print(f"🎬 Generating video...")
        print(f"Prompt: {prompt}")
        print(f"Dimensions: {width}x{height}x{num_frames}")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        
        # Generate video with memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before generation: {torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB")
        
        with torch.inference_mode():
            # Ensure we're using the correct device for generation
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
            
            video = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
            
        # Save video
        if output_path is None:
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in prompt)[:50]
            output_path = f"generated_video_{safe_prompt}_{timestamp}.mp4"
            
        export_to_video(video, output_path, fps=8)
        
        print(f"✅ Video generated: {output_path}")
        return output_path


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Generate videos with SceneGen-Finetuned LTX-Video")
    
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
        help="Number of frames"
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
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output video file path"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:3",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = VideoGenerator(
        checkpoint_path=args.checkpoint,
        model_version=args.model_version,
        device=args.device
    )
    
    # Load model
    generator.load_model()
    
    # Generate video
    output_path = generator.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=args.output
    )
    
    print(f"🎉 Done! Video saved to: {output_path}")


if __name__ == "__main__":
    # Example usage as a script
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python generate_video.py --checkpoint outputs/checkpoints/lora_weights_step_03000.safetensors --prompt 'a guy talking in front of a wooden piece of furniture'")
        print("\nFor help: python generate_video.py --help")
    else:
        main()