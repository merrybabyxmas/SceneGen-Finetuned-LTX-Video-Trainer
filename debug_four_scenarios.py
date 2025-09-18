#!/usr/bin/env python3
"""
4-Scenario Debugging Script for LTXV Video Generation

This script tests 4 different combinations to help debug multi-shot video generation:

1. LoRA X & single shot generation: Base model + standard LTXVideoPipeline
2. LoRA X & multi shot generation: Base model + SGMultiShotPipeline (1 shot)
3. LoRA O & single shot generation: LoRA + standard LTXVideoPipeline
4. LoRA O & multi shot generation: LoRA + SGMultiShotPipeline (1 shot)
"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import time 


import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXVideoTransformer3DModel
)
from diffusers.utils import export_to_video
from transformers import T5EncoderModel, T5TokenizerFast
from safetensors.torch import load_file
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
from ltxv_trainer.SG_validation_pipeline import MultiShotValidationPipeline
from ltxv_trainer.SG_datasets import SOSTokenLatents
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer import logger


class ScenarioDebugger:
    """Debug four different video generation scenarios."""

    def __init__(
        self,
        base_model_id: str = "Lightricks/LTX-Video-0.9.5",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        output_dir: str = "debug_outputs"
    ):
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self.device = device
        self.dtype = dtype
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Common generation parameters
        self.generation_params = {
            "width": 768,  # Reduce resolution to save memory
            "height": 448,  # Reduce resolution to save memory
            "num_frames": 17,
            "num_inference_steps": 50,  # Reduce steps for faster testing
            "guidance_scale": 3.5,
        }

        self.loaded_components = {}
        self.results = {}

    def load_base_components(self):
        """Load base model components with memory optimizations."""
        logger.info("Loading base model components with memory optimizations...")

        # Load tokenizer (CPU only)
        self.loaded_components["tokenizer"] = T5TokenizerFast.from_pretrained(
            self.base_model_id, subfolder="tokenizer"
        )

        # Load scheduler (CPU only)
        self.loaded_components["scheduler"] = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.base_model_id, subfolder="scheduler"
        )

        # Clear CUDA cache before loading models
        torch.cuda.empty_cache()

        # Load VAE with reduced precision
        logger.info("Loading VAE...")
        self.loaded_components["vae"] = AutoencoderKLLTXVideo.from_pretrained(
            self.base_model_id, subfolder="vae", torch_dtype=self.dtype
        )
        # Enable memory efficient attention
        if hasattr(self.loaded_components["vae"], "enable_xformers_memory_efficient_attention"):
            try:
                self.loaded_components["vae"].enable_xformers_memory_efficient_attention()
                logger.info("✅ VAE: Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"VAE: Could not enable xformers attention: {e}")

        # Load text encoder without quantization to avoid device placement issues
        logger.info("Loading text encoder...")
        self.loaded_components["text_encoder"] = T5EncoderModel.from_pretrained(
            self.base_model_id, subfolder="text_encoder", torch_dtype=self.dtype
        )
        logger.info("✅ Text encoder loaded successfully")

        # Clear cache again
        torch.cuda.empty_cache()

        # Load transformer with optimizations
        logger.info("Loading transformer...")
        self.loaded_components["transformer"] = LTXVideoTransformer3DModel.from_pretrained(
            self.base_model_id, subfolder="transformer", torch_dtype=self.dtype
        )

        # Enable memory efficient attention for transformer
        if hasattr(self.loaded_components["transformer"], "enable_xformers_memory_efficient_attention"):
            try:
                self.loaded_components["transformer"].enable_xformers_memory_efficient_attention()
                logger.info("✅ Transformer: Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Transformer: Could not enable xformers attention: {e}")

        # Move models to device one by one to avoid peak memory usage
        logger.info("Moving models to device...")
        self.loaded_components["vae"] = self.loaded_components["vae"].to(self.device)
        torch.cuda.empty_cache()

        # Move models to device (skip 8-bit quantized models as they're already on correct device)
        try:
            self.loaded_components["text_encoder"] = self.loaded_components["text_encoder"].to(self.device)
        except ValueError as e:
            if "8-bit" in str(e):
                logger.info("✅ Text encoder: 8-bit model already on correct device")
            else:
                raise e
        torch.cuda.empty_cache()

        self.loaded_components["transformer"] = self.loaded_components["transformer"].to(self.device)
        torch.cuda.empty_cache()

        logger.info("✅ Base components loaded successfully with optimizations")

    def apply_lora_weights(self, transformer):
        """Apply LoRA weights to transformer if available."""
        if not self.lora_path or not Path(self.lora_path).exists():
            logger.warning(f"LoRA path not found: {self.lora_path}")
            return transformer

        logger.info(f"Loading LoRA weights from: {self.lora_path}")

        try:
            # Load LoRA state dict
            if Path(self.lora_path).suffix == '.safetensors':
                lora_state_dict = load_file(self.lora_path)
            else:
                lora_state_dict = torch.load(self.lora_path, map_location="cpu")

            # Apply LoRA weights - simple approach for debugging
            transformer_state_dict = transformer.state_dict()

            # Filter and apply LoRA weights
            for key, value in lora_state_dict.items():
                if key in transformer_state_dict:
                    logger.info(f"Applying LoRA weight: {key}")
                    transformer_state_dict[key] = value.to(transformer.device)

            transformer.load_state_dict(transformer_state_dict, strict=False)
            logger.info("✅ LoRA weights applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply LoRA weights: {e}")
            logger.error(traceback.format_exc())

        return transformer

    def create_standard_pipeline(self, use_lora: bool = False):
        """Create standard LTXConditionPipeline with memory optimizations."""
        transformer = self.loaded_components["transformer"]

        if use_lora:
            transformer = self.apply_lora_weights(transformer)

        pipeline = LTXConditionPipeline(
            vae=self.loaded_components["vae"],
            text_encoder=self.loaded_components["text_encoder"],
            tokenizer=self.loaded_components["tokenizer"],
            scheduler=self.loaded_components["scheduler"],
            transformer=transformer,
        )

        # Enable memory optimizations
        pipeline.enable_model_cpu_offload()

        # Enable sequential CPU offload for maximum memory savings
        try:
            pipeline.enable_sequential_cpu_offload()
            logger.info("✅ Pipeline: Enabled sequential CPU offload")
        except Exception as e:
            logger.warning(f"Could not enable sequential CPU offload: {e}")

        # Enable memory efficient attention if available
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ Pipeline: Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers attention: {e}")

        return pipeline

    def create_multishot_pipeline(self, use_lora: bool = False):
        """Create MultiShotValidationPipeline with memory optimizations."""
        transformer = self.loaded_components["transformer"]

        if use_lora:
            transformer = self.apply_lora_weights(transformer)

        # Create SOS token generator
        sos_token_generator = SOSTokenLatents(
            d_model=128,  # Default SOS latent dimension
            use_zero_init=False
        ).to(self.device)

        # First create the base SGMultiShotPipeline
        sg_pipeline = SGMultiShotPipeline(
            vae=self.loaded_components["vae"],
            text_encoder=self.loaded_components["text_encoder"],
            tokenizer=self.loaded_components["tokenizer"],
            scheduler=self.loaded_components["scheduler"],
            transformer=transformer,
            sos_token_generator=sos_token_generator,
        )

        # Enable memory optimizations for base pipeline
        if hasattr(sg_pipeline, "enable_model_cpu_offload"):
            sg_pipeline.enable_model_cpu_offload()

        if hasattr(sg_pipeline, "enable_sequential_cpu_offload"):
            try:
                sg_pipeline.enable_sequential_cpu_offload()
                logger.info("✅ Base Pipeline: Enabled sequential CPU offload")
            except Exception as e:
                logger.warning(f"Could not enable sequential CPU offload: {e}")

        # Create MultiShotValidationPipeline wrapper
        pipeline = MultiShotValidationPipeline(
            pipeline=sg_pipeline,
            device=self.device,
            accelerator=None,  # Not needed for inference
            sos_token_generator=sos_token_generator,
        )

        logger.info("✅ Multishot Validation Pipeline created")
        return pipeline

    def generate_video(self, pipeline, prompt: str, scenario_name: str, use_multishot: bool = False):
        """Generate video using the given pipeline."""
        logger.info(f"Generating video for scenario: {scenario_name}")

        set_seed(42)  # For reproducible results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if use_multishot and hasattr(pipeline, 'generate_multi_shot_sequence'):
                # Use multi-shot generation
                video_paths = pipeline.generate_multi_shot_sequence(
                    prompt=prompt,
                    output_dir=self.output_dir,
                    global_step=0,  # Not used for naming
                    video_dims=(self.generation_params["width"],
                               self.generation_params["height"],
                               self.generation_params["num_frames"]),
                    num_shots=1,
                    inference_steps=self.generation_params["num_inference_steps"],
                    guidance_scale=self.generation_params["guidance_scale"],
                    seed=42,
                )

                # Rename generated videos to include scenario name
                renamed_paths = []
                for i, original_path in enumerate(video_paths):
                    new_path = self.output_dir / f"{scenario_name}_shot_{i+1}_{timestamp}.mp4"
                    if Path(original_path).exists():
                        Path(original_path).rename(new_path)
                        renamed_paths.append(str(new_path))
                        logger.info(f"✅ Shot {i+1} saved: {new_path}")

                # Return the first shot path as primary result, but log all
                primary_video_path = renamed_paths[0] if renamed_paths else None
                logger.info(f"✅ Multi-shot generation completed: {len(renamed_paths)} shots saved")
                return primary_video_path, True, None

            else:
                # Use standard single-shot generation
                generator = torch.Generator(device=self.device).manual_seed(42)
                result = pipeline(
                    prompt=prompt,
                    generator=generator,
                    **self.generation_params
                )
                frames = result.frames[0] if hasattr(result, 'frames') else result

                # Save video
                video_path = self.output_dir / f"{scenario_name}_{timestamp}.mp4"
                export_to_video(frames, str(video_path), fps=8)

                logger.info(f"✅ Single-shot video saved: {video_path}")
                return str(video_path), True, None

        except Exception as e:
            error_msg = f"Generation failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None, False, error_msg

    def run_scenario_1(self, prompt: str):
        """Scenario 1: LoRA X & single shot generation (Base model + standard pipeline)"""
        logger.info("🔬 Running Scenario 1: Base Model + Standard Pipeline")

        pipeline = None
        try:
            pipeline = self.create_standard_pipeline(use_lora=False)
            video_path, success, error = self.generate_video(
                pipeline, prompt, "scenario_1_base_standard", use_multishot=False
            )
        except Exception as e:
            video_path, success, error = None, False, str(e)
        finally:
            if pipeline:
                self.cleanup_pipeline(pipeline)

        self.results["scenario_1"] = {
            "description": "Base Model + Standard LTXConditionPipeline",
            "lora": False,
            "multishot": False,
            "video_path": video_path,
            "success": success,
            "error": error
        }

    def run_scenario_2(self, prompt: str):
        """Scenario 2: LoRA X & multi shot generation (Base model + multishot pipeline, 1 shot)"""
        logger.info("🔬 Running Scenario 2: Base Model + Multishot Pipeline (1 shot)")

        pipeline = None
        try:
            pipeline = self.create_multishot_pipeline(use_lora=False)
            video_path, success, error = self.generate_video(
                pipeline, prompt, "scenario_2_base_multishot", use_multishot=True
            )
        except Exception as e:
            video_path, success, error = None, False, str(e)
        finally:
            if pipeline:
                self.cleanup_pipeline(pipeline)

        self.results["scenario_2"] = {
            "description": "Base Model + SGMultiShotPipeline (1 shot)",
            "lora": False,
            "multishot": True,
            "video_path": video_path,
            "success": success,
            "error": error
        }

    def run_scenario_3(self, prompt: str):
        """Scenario 3: LoRA O & single shot generation (LoRA + standard pipeline)"""
        logger.info("🔬 Running Scenario 3: LoRA Model + Standard Pipeline")

        pipeline = None
        try:
            pipeline = self.create_standard_pipeline(use_lora=True)
            video_path, success, error = self.generate_video(
                pipeline, prompt, "scenario_3_lora_standard", use_multishot=False
            )
        except Exception as e:
            video_path, success, error = None, False, str(e)
        finally:
            if pipeline:
                self.cleanup_pipeline(pipeline)

        self.results["scenario_3"] = {
            "description": "LoRA Model + Standard LTXConditionPipeline",
            "lora": True,
            "multishot": False,
            "video_path": video_path,
            "success": success,
            "error": error
        }

    def run_scenario_4(self, prompt: str):
        """Scenario 4: LoRA O & multi shot generation (LoRA + multishot pipeline, 1 shot)"""
        logger.info("🔬 Running Scenario 4: LoRA Model + Multishot Pipeline (1 shot)")

        pipeline = None
        try:
            pipeline = self.create_multishot_pipeline(use_lora=True)
            video_path, success, error = self.generate_video(
                pipeline, prompt, "scenario_4_lora_multishot", use_multishot=True
            )
        except Exception as e:
            video_path, success, error = None, False, str(e)
        finally:
            if pipeline:
                self.cleanup_pipeline(pipeline)

        self.results["scenario_4"] = {
            "description": "LoRA Model + SGMultiShotPipeline (1 shot)",
            "lora": True,
            "multishot": True,
            "video_path": video_path,
            "success": success,
            "error": error
        }

    def run_all_scenarios(self, prompt: str):
        """Run all four scenarios."""
        logger.info("🚀 Starting 4-Scenario Debug Session")
        logger.info("="*60)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Base Model: {self.base_model_id}")
        logger.info(f"LoRA Path: {self.lora_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("="*60)

        # Load base components once
        self.load_base_components()

        # Run all scenarios
        scenarios = [
            ("Scenario 1", self.run_scenario_1),
            ("Scenario 2", self.run_scenario_2),
            ("Scenario 3", self.run_scenario_3),
            ("Scenario 4", self.run_scenario_4),
        ]

        for scenario_name, scenario_func in scenarios:
            try:
                logger.info(f"\n{'='*20} {scenario_name} {'='*20}")
                scenario_func(prompt)
            except Exception as e:
                logger.error(f"❌ {scenario_name} failed with error: {e}")
                logger.error(traceback.format_exc())

        # Generate summary report
        self.generate_report()

    def cleanup_memory(self):
        """Clean up CUDA memory and run garbage collection."""
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            logger.info(f"🧹 Memory cleaned. GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB used")

    def cleanup_pipeline(self, pipeline):
        """Clean up pipeline and free memory."""
        if hasattr(pipeline, 'maybe_free_model_hooks'):
            pipeline.maybe_free_model_hooks()
        del pipeline
        self.cleanup_memory()

    def generate_report(self):
        """Generate a summary report of all scenarios."""
        logger.info("\n" + "="*60)
        logger.info("📊 SCENARIO COMPARISON REPORT")
        logger.info("="*60)

        successful_scenarios = 0

        for scenario_id, result in self.results.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            logger.info(f"\n{scenario_id.upper()}: {result['description']}")
            logger.info(f"Status: {status}")
            logger.info(f"LoRA: {'✓' if result['lora'] else '✗'}")
            logger.info(f"Multishot: {'✓' if result['multishot'] else '✗'}")

            if result["success"]:
                logger.info(f"Video: {result['video_path']}")
                successful_scenarios += 1
            else:
                logger.info(f"Error: {result['error']}")

        logger.info(f"\n📈 Overall Results: {successful_scenarios}/{len(self.results)} scenarios successful")

        # Save detailed report
        report_path = self.output_dir / f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"📄 Detailed report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="4-Scenario LTXV Video Generation Debugger")

    parser.add_argument(
        "--prompt",
        type=str,
        default="a professional quality video of the earth",
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Lightricks/LTX-Video-0.9.5",
        help="Base model identifier"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug_outputs",
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype"
    )

    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 예: 20250915_003012
    args.output_dir = os.path.join(args.output_dir, timestamp)

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # Check for LoRA path
    if args.lora_path is None:
        # Try to find LoRA checkpoint in outputs directory
        outputs_dir = Path("outputs/checkpoints")
        if outputs_dir.exists():
            lora_files = list(outputs_dir.glob("*lora*.safetensors"))
            if lora_files:
                args.lora_path = str(lora_files[0])  # Use the first LoRA file found
                logger.info(f"Auto-detected LoRA checkpoint: {args.lora_path}")
            else:
                logger.warning("No LoRA checkpoint found in outputs/checkpoints/")
        else:
            logger.warning("outputs/checkpoints/ directory not found")

    # Create debugger and run scenarios
    debugger = ScenarioDebugger(
        base_model_id=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=dtype,
        output_dir=args.output_dir
    )

    debugger.run_all_scenarios(args.prompt)


if __name__ == "__main__":
    main()