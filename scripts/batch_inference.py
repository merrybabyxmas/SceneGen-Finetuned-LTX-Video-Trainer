#!/usr/bin/env python3
"""
Batch Inference Script for SceneGen Fine-tuned LTX-Video

This script performs batch inference on multiple prompt sets,
useful for evaluation or generating multiple video sequences.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

import torch
from accelerate.utils import set_seed
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ltxv_trainer import logger
from inference import load_model_components, create_pipeline, generate_multi_shot_sequence


def load_batch_config(config_file: str) -> Dict[str, Any]:
    """Load batch inference configuration from JSON file."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["sequences"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config file")

    return config


def run_batch_inference(
    pipeline,
    sequences: List[Dict[str, Any]],
    base_config: Dict[str, Any],
    output_base_dir: str,
) -> List[Dict[str, Any]]:
    """
    Run batch inference on multiple sequences.

    Args:
        pipeline: The SGMultiShotPipeline
        sequences: List of sequence configurations
        base_config: Base configuration for generation parameters
        output_base_dir: Base output directory

    Returns:
        List of results with paths and metadata
    """
    results = []

    for seq_idx, sequence in enumerate(tqdm(sequences, desc="Generating sequences")):
        sequence_name = sequence.get("name", f"sequence_{seq_idx:03d}")
        logger.info(f"\n=== Processing Sequence: {sequence_name} ===")

        # Create output directory for this sequence
        seq_output_dir = os.path.join(output_base_dir, sequence_name)
        os.makedirs(seq_output_dir, exist_ok=True)

        # Get prompts
        prompts = sequence["prompts"]
        logger.info(f"Generating {len(prompts)} shots")

        # Use sequence-specific config or fall back to base config
        generation_config = {**base_config}
        if "config" in sequence:
            generation_config.update(sequence["config"])

        # Set seed for this sequence if specified
        if "seed" in sequence:
            set_seed(sequence["seed"])
            generation_config["seed"] = sequence["seed"]

        start_time = time.time()

        try:
            # Generate sequence
            output_paths = generate_multi_shot_sequence(
                pipeline=pipeline,
                prompts=prompts,
                output_dir=seq_output_dir,
                **generation_config,
            )

            generation_time = time.time() - start_time

            # Record results
            result = {
                "sequence_name": sequence_name,
                "sequence_idx": seq_idx,
                "prompts": prompts,
                "output_paths": output_paths,
                "generation_time": generation_time,
                "config": generation_config,
                "status": "success",
            }

            logger.info(f"✅ Completed {sequence_name} in {generation_time:.1f}s")

        except Exception as e:
            logger.error(f"❌ Failed to generate {sequence_name}: {e}")
            result = {
                "sequence_name": sequence_name,
                "sequence_idx": seq_idx,
                "prompts": prompts,
                "error": str(e),
                "status": "failed",
            }

        results.append(result)

        # Save intermediate results
        intermediate_results_path = os.path.join(output_base_dir, "batch_results_intermediate.json")
        with open(intermediate_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch inference for SceneGen fine-tuned LTX-Video")

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

    # Batch configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON file containing batch inference configuration"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_outputs",
        help="Base output directory"
    )

    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Load batch configuration
    logger.info(f"Loading batch configuration from {args.config}")
    batch_config = load_batch_config(args.config)

    sequences = batch_config["sequences"]
    base_generation_config = batch_config.get("default_config", {})

    logger.info(f"Loaded {len(sequences)} sequences for batch processing")

    # Set default values for base config
    default_config = {
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "height": 448,
        "width": 704,
        "num_frames": 121,
        "num_inference_steps": 40,
        "guidance_scale": 3.0,
        "save_individual_shots": True,
    }

    final_config = {**default_config, **base_generation_config}

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run batch inference
    logger.info(f"Starting batch inference with {len(sequences)} sequences...")
    start_time = time.time()

    results = run_batch_inference(
        pipeline=pipeline,
        sequences=sequences,
        base_config=final_config,
        output_base_dir=args.output_dir,
    )

    total_time = time.time() - start_time

    # Save final results
    results_path = os.path.join(args.output_dir, "batch_results.json")
    final_results = {
        "config_file": args.config,
        "model_path": args.model_path,
        "base_model": args.base_model,
        "total_sequences": len(sequences),
        "successful_sequences": len([r for r in results if r["status"] == "success"]),
        "failed_sequences": len([r for r in results if r["status"] == "failed"]),
        "total_time": total_time,
        "results": results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Print summary
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "failed"])

    logger.info("\n" + "="*50)
    logger.info("BATCH INFERENCE COMPLETE")
    logger.info("="*50)
    logger.info(f"Total sequences: {len(sequences)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Average time per sequence: {total_time/len(sequences):.1f}s")
    logger.info(f"Results saved to: {results_path}")

    if failed > 0:
        logger.info(f"\nFailed sequences:")
        for result in results:
            if result["status"] == "failed":
                logger.info(f"  - {result['sequence_name']}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()