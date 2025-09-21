#!/usr/bin/env python3
"""
Single Prompt Multi-View Video Generation Inference Script

This script generates multiple camera angles/viewpoints from a single text prompt
using scenario strings to control view transitions and camera movements.

Example:
    python single_prompt_multiview_inference.py \
        --model_path outputs/checkpoints/lora_weights_step_01500.safetensors \
        --prompt "woman walking through a beautiful garden" \
        --scenario "wide,stable,medium,transition,close" \
        --output_dir outputs
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXVideoTransformer3DModel
from transformers import T5EncoderModel, T5TokenizerFast

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ltxv_trainer.single_prompt_multiview import SinglePromptMultiViewGenerator
from ltxv_trainer.token_utils import VideoTokenEmbeddings
from ltxv_trainer.scenario_video_saver import ScenarioVideoSaver
from ltxv_trainer import logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def load_model_components(
    model_path: str,
    base_model_id: str = "Lightricks/LTX-Video",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    Load all model components for single prompt multi-view inference.

    Args:
        model_path: Path to fine-tuned model checkpoint
        base_model_id: Base model identifier
        device: Target device
        dtype: Model dtype

    Returns:
        Dictionary containing all loaded model components
    """
    logger.info(f"🚀 Loading model components for single prompt multi-view generation")

    # Load VAE
    logger.info("Loading VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        base_model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)

    # Load text encoder and tokenizer
    logger.info("Loading text encoder and tokenizer...")
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(
        base_model_id, subfolder="tokenizer"
    )

    # Load scheduler
    logger.info("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_id, subfolder="scheduler"
    )

    # Load transformer
    logger.info("Loading transformer...")
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        base_model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)

    # Load fine-tuned weights
    checkpoint_path = Path(model_path)
    if checkpoint_path.exists():
        logger.info(f"Loading fine-tuned weights from {checkpoint_path}")
        if checkpoint_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            transformer.load_state_dict(state_dict, strict=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "transformer" in checkpoint:
                transformer.load_state_dict(checkpoint["transformer"])
            else:
                transformer.load_state_dict(checkpoint)
        logger.info("✅ Fine-tuned weights loaded successfully")

    # Initialize token embeddings
    logger.info("Initializing video token embeddings...")
    token_embeddings = VideoTokenEmbeddings(hidden_dim=transformer.config.in_channels)
    token_embeddings.to(device)

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "transformer": transformer,
        "token_embeddings": token_embeddings,
    }


def generate_multiview_video(
    prompt: str,
    scenario_string: str,
    model_components: Dict[str, Any],
    height: int = 448,
    width: int = 704,
    num_frames: int = 121,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.0,
    seed: Optional[int] = None,
    device: str = "cuda"
) -> tuple[torch.Tensor, Dict]:
    """
    Generate multi-view video from single prompt and scenario.

    Args:
        prompt: Single text prompt for all views
        scenario_string: View scenario like "wide,stable,medium,transition,close"
        model_components: Loaded model components
        height: Video height
        width: Video width
        num_frames: Number of frames per view
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale
        seed: Random seed
        device: Target device

    Returns:
        Tuple of (generated_sequence, metadata)
    """
    logger.info(f"🎬 Generating multi-view video:")
    logger.info(f"🎬   Prompt: '{prompt}'")
    logger.info(f"🎬   Scenario: '{scenario_string}'")

    if seed is not None:
        set_seed(seed)

    # Calculate latent dimensions
    vae = model_components["vae"]
    latent_height = height // vae.config.scaling_factor
    latent_width = width // vae.config.scaling_factor

    # Create multi-view generator
    generator = SinglePromptMultiViewGenerator(
        token_embeddings=model_components["token_embeddings"],
        transformer_model=model_components["transformer"],
        device=torch.device(device)
    )

    # Generate multi-view sequence
    generated_sequence, metadata = generator.generate_multiview_video(
        scenario_string=scenario_string,
        base_prompt=prompt,
        height=latent_height,
        width=latent_width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    logger.info(f"🎬 ✅ Multi-view generation completed: {generated_sequence.shape}")
    return generated_sequence, metadata


def decode_latents_to_frames(
    latent_sequence: torch.Tensor,
    vae: AutoencoderKLLTXVideo,
    batch_size: int = 4
) -> List:
    """
    Decode latent sequence to video frames.

    Args:
        latent_sequence: Generated latent sequence [T, H, W, C]
        vae: VAE decoder
        batch_size: Batch size for decoding

    Returns:
        List of decoded video frames
    """
    logger.info(f"🎨 Decoding latents to frames: {latent_sequence.shape}")

    frames = []
    total_frames = latent_sequence.shape[0]

    # Process in batches
    for i in range(0, total_frames, batch_size):
        end_idx = min(i + batch_size, total_frames)
        batch_latents = latent_sequence[i:end_idx]

        # Add batch dimension and rearrange for VAE
        batch_latents = batch_latents.unsqueeze(0)  # [1, T, H, W, C]
        batch_latents = batch_latents.permute(0, 4, 1, 2, 3)  # [1, C, T, H, W]

        with torch.no_grad():
            decoded = vae.decode(batch_latents).sample

        # Convert to CPU and rearrange
        decoded = decoded.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
        decoded = ((decoded + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        for frame in decoded:
            frames.append(frame.cpu().numpy())

        logger.info(f"🎨 Decoded frames {i+1}-{end_idx}/{total_frames}")

    logger.info(f"🎨 ✅ Decoded {len(frames)} frames")
    return frames


def extract_view_segments(
    frames: List,
    metadata: Dict
) -> Dict[str, List]:
    """
    Extract individual view segments from the full frame sequence.

    Args:
        frames: List of all frames
        metadata: Generation metadata with view ranges

    Returns:
        Dictionary mapping view names to frame lists
    """
    logger.info(f"✂️ Extracting {metadata['view_count']} view segments")

    view_segments = {}
    for view_name, (start, end) in metadata['view_ranges'].items():
        view_frames = frames[start:end]
        view_segments[view_name] = view_frames
        logger.info(f"✂️ Extracted {view_name}: frames {start}-{end} ({len(view_frames)} frames)")

    return view_segments


def main():
    parser = argparse.ArgumentParser(description="Single prompt multi-view video generation")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Lightricks/LTX-Video",
        help="Base model identifier"
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Single text prompt for all views"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="View scenario string like 'wide,stable,medium,transition,close'"
    )

    # Video parameters
    parser.add_argument("--height", type=int, default=448, help="Video height")
    parser.add_argument("--width", type=int, default=704, help="Video width")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames per view")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--save_individual_views", action="store_true", help="Save individual view segments")
    parser.add_argument("--no_decode", action="store_true", help="Skip VAE decoding (save latents only)")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Model dtype")

    args = parser.parse_args()

    # Set up device and dtype
    device = args.device
    dtype = getattr(torch, args.dtype)

    logger.info(f"🚀 Starting single prompt multi-view inference")
    logger.info(f"🚀   Prompt: '{args.prompt}'")
    logger.info(f"🚀   Scenario: '{args.scenario}'")
    logger.info(f"🚀   Device: {device}, dtype: {dtype}")

    # Load model components
    model_components = load_model_components(
        model_path=args.model_path,
        base_model_id=args.base_model,
        device=device,
        dtype=dtype
    )

    # Generate multi-view video
    generated_sequence, metadata = generate_multiview_video(
        prompt=args.prompt,
        scenario_string=args.scenario,
        model_components=model_components,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=device
    )

    # Initialize video saver with enhanced scenario name
    scenario_with_prompt = f"{args.scenario}__{args.prompt.replace(' ', '_')[:50]}"
    video_saver = ScenarioVideoSaver(args.output_dir)

    if not args.no_decode:
        # Decode latents to frames
        frames = decode_latents_to_frames(
            generated_sequence,
            model_components["vae"]
        )

        # Save full sequence
        saved_paths = video_saver.save_scenario_videos(
            scenario_string=scenario_with_prompt,
            video_frames_list=[frames],
            video_names=["full_multiview_sequence"],
            fps=args.fps
        )

        # Save individual view segments if requested
        if args.save_individual_views:
            view_segments = extract_view_segments(frames, metadata)

            view_videos = []
            view_names = []
            for view_name, view_frames in view_segments.items():
                view_videos.append(view_frames)
                view_names.append(f"{view_name}")

            if view_videos:
                view_saved_paths = video_saver.save_scenario_videos(
                    scenario_string=scenario_with_prompt,
                    video_frames_list=view_videos,
                    video_names=view_names,
                    fps=args.fps
                )
                saved_paths.extend(view_saved_paths)

        logger.info(f"🎬 ✅ Videos saved to: {[str(p) for p in saved_paths]}")

    else:
        # Save latents
        scenario_dir = video_saver.create_scenario_directory(scenario_with_prompt)
        latents_path = scenario_dir / "generated_latents.pt"
        torch.save({
            'latents': generated_sequence,
            'metadata': metadata
        }, latents_path)
        logger.info(f"💾 Latents saved to: {latents_path}")

    # Print summary
    logger.info(f"\n🎉 Single prompt multi-view generation completed!")
    logger.info(f"📝 Summary:")
    logger.info(f"   Prompt: '{args.prompt}'")
    logger.info(f"   Scenario: '{args.scenario}'")
    logger.info(f"   Views generated: {metadata['view_count']}")
    logger.info(f"   Total frames: {metadata['total_frames']}")
    logger.info(f"   View prompts: {metadata['view_prompts']}")


if __name__ == "__main__":
    main()