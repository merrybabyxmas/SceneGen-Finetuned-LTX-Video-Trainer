#!/usr/bin/env python3
"""
SceneGen Fine-tuned LTX-Video Inference Script

This script loads a fine-tuned LTX-Video model and generates multi-shot videos
using the Scene Generation approach with reference latents conditioning.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import json

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel, T5TokenizerFast
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
from ltxv_trainer.SG_datasets import SOSTokenLatents
from ltxv_trainer.scenario_video_saver import ScenarioVideoSaver
from ltxv_trainer import logger


def load_model_components(
    model_path: str,
    base_model_id: str = "Lightricks/LTX-Video",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load all model components for inference.

    Args:
        model_path: Path to fine-tuned model checkpoint
        base_model_id: Base model identifier for loading components
        device: Target device
        dtype: Model dtype

    Returns:
        Dictionary containing all loaded model components
    """
    logger.info(f"Loading model components from {model_path}")

    # Load base model components
    logger.info("Loading VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        base_model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)

    logger.info("Loading text encoder...")
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)

    logger.info("Loading tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained(
        base_model_id, subfolder="tokenizer"
    )

    logger.info("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_id, subfolder="scheduler"
    )

    # Load transformer
    logger.info("Loading transformer...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        base_model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)

    # Load fine-tuned weights if available
    checkpoint_path = Path(model_path)
    if checkpoint_path.is_file():
        logger.info(f"Loading weights from {checkpoint_path}")

        # Check if it's a LoRA checkpoint
        if "lora" in checkpoint_path.name.lower() or "comfy" in checkpoint_path.name.lower():
            # Load LoRA weights
            try:
                from safetensors.torch import load_file
                if checkpoint_path.suffix == '.safetensors':
                    lora_state_dict = load_file(checkpoint_path)
                else:
                    lora_state_dict = torch.load(checkpoint_path, map_location="cpu")

                # Apply LoRA weights to transformer
                logger.info("Applying LoRA weights...")

                # Get current transformer state dict
                transformer_state_dict = transformer.state_dict()

                # Group LoRA weights by layer
                lora_weights = {}
                for name, param in lora_state_dict.items():
                    if '.lora_up.' in name or '.lora_down.' in name:
                        # Extract the base layer name (everything before .lora_up/.lora_down)
                        if '.lora_up.' in name:
                            base_name = name.replace('.lora_up.', '.')
                            lora_type = 'up'
                        else:
                            base_name = name.replace('.lora_down.', '.')
                            lora_type = 'down'

                        # Remove the final .weight if present
                        if base_name.endswith('.weight'):
                            base_name = base_name[:-7]

                        if base_name not in lora_weights:
                            lora_weights[base_name] = {}
                        lora_weights[base_name][lora_type] = param.to(device, dtype=dtype)

                # Apply LoRA modifications
                alpha = 64  # From config
                rank = 64   # From config
                scale = alpha / rank

                for base_name, lora_params in lora_weights.items():
                    if 'up' in lora_params and 'down' in lora_params:
                        # Find the corresponding weight in transformer
                        weight_name = base_name + '.weight'
                        if weight_name in transformer_state_dict:
                            base_weight = transformer_state_dict[weight_name]
                            lora_up = lora_params['up']
                            lora_down = lora_params['down']

                            # Apply LoRA: base_weight + lora_up @ lora_down * scale
                            delta_weight = torch.mm(lora_up, lora_down) * scale
                            new_weight = base_weight + delta_weight
                            transformer_state_dict[weight_name] = new_weight
                            logger.debug(f"Applied LoRA to {weight_name}")

                # Load the modified state dict
                transformer.load_state_dict(transformer_state_dict)
                logger.info(f"Successfully loaded LoRA weights from {checkpoint_path}")

            except Exception as e:
                logger.warning(f"Failed to load as LoRA weights: {e}")
                logger.info("Trying to load as full checkpoint...")

                # Fallback to full checkpoint loading
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "transformer" in checkpoint:
                    transformer.load_state_dict(checkpoint["transformer"])
                else:
                    transformer.load_state_dict(checkpoint)
                logger.info(f"Loaded as full checkpoint from {checkpoint_path}")
        else:
            # Load as full checkpoint
            if checkpoint_path.suffix == '.safetensors':
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
                transformer.load_state_dict(state_dict)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "transformer" in checkpoint:
                    transformer.load_state_dict(checkpoint["transformer"])
                else:
                    transformer.load_state_dict(checkpoint)
            logger.info(f"Loaded full checkpoint from {checkpoint_path}")

    elif checkpoint_path.is_dir():
        # Directory with separate files
        transformer_path = checkpoint_path / "transformer" / "diffusion_pytorch_model.safetensors"
        if transformer_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
            transformer.load_state_dict(state_dict)
            logger.info(f"Loaded transformer from {transformer_path}")
        else:
            # Try loading from transformers format
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                checkpoint_path, torch_dtype=dtype
            ).to(device)
            logger.info(f"Loaded transformer from directory {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint path {checkpoint_path} not found. Using base model weights only.")

    # Create SOS token generator
    logger.info("Creating SOS token generator...")
    sos_token_generator = SOSTokenLatents(
        d_model=transformer.config.in_channels,
        use_zero_init=False
    ).to(device)

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "transformer": transformer,
        "sos_token_generator": sos_token_generator,
    }


def create_pipeline(components: dict) -> SGMultiShotPipeline:
    """Create the multi-shot pipeline from loaded components."""
    pipeline = SGMultiShotPipeline(
        vae=components["vae"],
        text_encoder=components["text_encoder"],
        tokenizer=components["tokenizer"],
        scheduler=components["scheduler"],
        transformer=components["transformer"],
        sos_token_generator=components["sos_token_generator"],
    )

    # Enable memory efficient attention if available
    if hasattr(pipeline.transformer, "enable_xformers_memory_efficient_attention"):
        try:
            pipeline.transformer.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    # Enable CPU offloading for memory efficiency
    try:
        pipeline.enable_model_cpu_offload()
        logger.info("Enabled CPU offloading")
    except Exception as e:
        logger.warning(f"Could not enable CPU offloading: {e}")

    return pipeline


def generate_single_shot(
    pipeline: SGMultiShotPipeline,
    prompt: str,
    reference_latents: Optional[torch.Tensor] = None,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 448,
    width: int = 704,
    num_frames: int = 121,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.0,
    generator: Optional[torch.Generator] = None,
    return_latents: bool = True,
) -> dict:
    """
    Generate a single shot using the pipeline.

    Args:
        pipeline: The SGMultiShotPipeline
        prompt: Text prompt for generation
        reference_latents: Optional reference latents for conditioning
        negative_prompt: Negative prompt
        height: Video height
        width: Video width
        num_frames: Number of frames
        num_inference_steps: Denoising steps
        guidance_scale: CFG guidance scale
        generator: Random generator
        return_latents: Whether to return latents for next shot

    Returns:
        Dictionary containing generated video and optionally latents
    """
    logger.info(f"Generating shot with prompt: '{prompt[:50]}...'")

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        reference_latents=reference_latents,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        return_latents=return_latents,
        output_type="pil",
    )

    if return_latents:
        return {
            "frames": result["frames"],
            "latents": result["latents"],
        }
    else:
        return {"frames": result.frames if hasattr(result, 'frames') else result["frames"]}


def generate_multi_shot_sequence(
    pipeline: SGMultiShotPipeline,
    prompts: List[str],
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 448,
    width: int = 704,
    num_frames: int = 121,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.0,
    seed: Optional[int] = None,
    output_dir: str = "outputs",
    scenario_string: Optional[str] = None,
    save_individual_shots: bool = True,
) -> List[str]:
    """
    Generate a multi-shot video sequence.

    Args:
        pipeline: The SGMultiShotPipeline
        prompts: List of prompts for each shot
        negative_prompt: Negative prompt
        height: Video height
        width: Video width
        num_frames: Number of frames per shot
        num_inference_steps: Denoising steps
        guidance_scale: CFG guidance scale
        seed: Random seed
        output_dir: Base output directory
        scenario_string: Optional scenario string for organized output directories
        save_individual_shots: Whether to save individual shots

    Returns:
        List of output video paths
    """
    # Initialize scenario video saver if scenario is provided
    if scenario_string:
        video_saver = ScenarioVideoSaver(output_dir)
        logger.info(f"🎬 Using scenario-based saving: '{scenario_string}'")
    else:
        os.makedirs(output_dir, exist_ok=True)
        video_saver = None

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    previous_latents = None
    output_paths = []
    all_frames = []

    for shot_idx, prompt in enumerate(prompts):
        logger.info(f"\n=== Generating Shot {shot_idx + 1}/{len(prompts)} ===")

        # Generate current shot
        result = generate_single_shot(
            pipeline=pipeline,
            prompt=prompt,
            reference_latents=previous_latents,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            return_latents=shot_idx < len(prompts) - 1,  # Don't need latents for last shot
        )

        frames = result["frames"][0]  # Get first (and only) batch
        all_frames.extend(frames)

        # Store shot frames for later saving
        if shot_idx == 0:
            shot_frames_list = []
        shot_frames_list.append(frames)

        # Update previous latents for next shot
        if "latents" in result:
            previous_latents = result["latents"]
            logger.info(f"Using latents from shot {shot_idx + 1} as reference for next shot")

    # Save videos using scenario-based or traditional approach
    if video_saver and scenario_string:
        # Use scenario-based saving
        saved_paths_dict = video_saver.save_inference_videos(
            scenario_string=scenario_string,
            shot_frames_list=shot_frames_list,
            full_sequence_frames=all_frames,
            fps=24,
            save_individual_shots=save_individual_shots
        )

        output_paths = []
        if save_individual_shots:
            output_paths.extend([str(p) for p in saved_paths_dict['individual_shots']])
        output_paths.extend([str(p) for p in saved_paths_dict['full_sequence']])

        logger.info(f"🎬 ✅ Scenario-based saving completed: {len(output_paths)} files")
    else:
        # Traditional saving approach
        output_paths = []

        # Save individual shots if requested
        if save_individual_shots:
            for shot_idx, frames in enumerate(shot_frames_list):
                shot_path = os.path.join(output_dir, f"shot_{shot_idx + 1:02d}.mp4")
                export_to_video(frames, shot_path, fps=24)
                output_paths.append(shot_path)
                logger.info(f"Saved shot {shot_idx + 1} to {shot_path}")

        # Save complete sequence
        full_sequence_path = os.path.join(output_dir, "full_sequence.mp4")
        export_to_video(all_frames, full_sequence_path, fps=24)
        output_paths.append(full_sequence_path)
        logger.info(f"Saved full sequence to {full_sequence_path}")

    return output_paths


def load_prompts_from_file(prompts_file: str) -> List[str]:
    """Load prompts from text file (one per line) or JSON file."""
    prompts_path = Path(prompts_file)

    if prompts_path.suffix.lower() == ".json":
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "prompts" in data:
            return data["prompts"]
        else:
            raise ValueError("JSON file should contain a list of prompts or a dict with 'prompts' key")
    else:
        # Text file - one prompt per line
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate videos using fine-tuned LTX-Video model")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint or directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Lightricks/LTX-Video",
        help="Base model identifier"
    )

    # Generation arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        help="Text prompts for generation (can specify multiple for multi-shot)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="File containing prompts (JSON or text file with one prompt per line)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt"
    )

    # Video parameters
    parser.add_argument("--height", type=int, default=448, help="Video height")
    parser.add_argument("--width", type=int, default=704, help="Video width")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames per shot")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--scenario", type=str, help="Scenario string for organizing outputs (e.g., 'shot1,stable,shot2,transition,shot3')")
    parser.add_argument("--no_individual_shots", action="store_true", help="Don't save individual shots")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Model dtype")

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Get prompts
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
    elif args.prompts:
        prompts = args.prompts
    else:
        # Default demo prompts
        prompts = [
            "A serene mountain landscape with a crystal clear lake reflecting snow-capped peaks, morning mist rising from the water.",
            "The same lake scene as camera slowly pans across the water, revealing a small wooden dock with a red kayak.",
            "Close-up of the red kayak gently rocking in the calm water, with mountains visible in the background.",
        ]
        logger.info("Using default demo prompts")

    logger.info(f"Generating sequence with {len(prompts)} shots")
    for i, prompt in enumerate(prompts):
        logger.info(f"Shot {i+1}: {prompt[:80]}...")

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Load model
    logger.info("Loading model components...")
    components = load_model_components(
        model_path=args.model_path,
        base_model_id=args.base_model,
        device=args.device,
        dtype=dtype,
    )

    # Create pipeline
    logger.info("Creating pipeline...")
    pipeline = create_pipeline(components)

    # Generate sequence
    logger.info("Starting generation...")
    if args.scenario:
        logger.info(f"🎬 Using scenario: '{args.scenario}'")

    output_paths = generate_multi_shot_sequence(
        pipeline=pipeline,
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_dir=args.output_dir,
        scenario_string=args.scenario,
        save_individual_shots=not args.no_individual_shots,
    )

    logger.info("\n=== Generation Complete ===")
    logger.info(f"Generated {len(output_paths)} videos:")
    for path in output_paths:
        logger.info(f"  - {path}")

    # Save generation config
    config_path = os.path.join(args.output_dir, "generation_config.json")
    config = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "prompts": prompts,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "output_paths": output_paths,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved generation config to {config_path}")


if __name__ == "__main__":
    main()