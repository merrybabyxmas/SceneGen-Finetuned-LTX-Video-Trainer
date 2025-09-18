#!/usr/bin/env python3
"""
Test script for SceneGen inference to verify everything works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from ltxv_trainer import logger


def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")

    try:
        from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
        from ltxv_trainer.SG_datasets import SOSTokenLatents
        from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXVideoTransformer3DModel
        from transformers import T5EncoderModel, T5TokenizerFast
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_model_loading():
    """Test loading base model components (without fine-tuned weights)."""
    logger.info("Testing base model loading...")

    try:
        base_model_id = "Lightricks/LTX-Video"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Test loading each component
        logger.info("Loading tokenizer...")
        tokenizer = T5TokenizerFast.from_pretrained(base_model_id, subfolder="tokenizer")

        logger.info("Loading scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_id, subfolder="scheduler")

        if torch.cuda.is_available():
            logger.info("Loading VAE...")
            vae = AutoencoderKLLTXVideo.from_pretrained(
                base_model_id, subfolder="vae", torch_dtype=dtype
            ).to(device)

            logger.info("Loading text encoder...")
            text_encoder = T5EncoderModel.from_pretrained(
                base_model_id, subfolder="text_encoder", torch_dtype=dtype
            ).to(device)

            logger.info("Loading transformer...")
            transformer = LTXVideoTransformer3DModel.from_pretrained(
                base_model_id, subfolder="transformer", torch_dtype=dtype
            ).to(device)

            logger.info("Creating SOS token generator...")
            sos_token_generator = SOSTokenLatents(
                d_model=transformer.config.in_channels, use_zero_init=False
            ).to(device)

            logger.info("Creating pipeline...")
            from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
            pipeline = SGMultiShotPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                transformer=transformer,
                sos_token_generator=sos_token_generator,
            )

            logger.info("✅ Base model loading successful")
            return True
        else:
            logger.info("✅ Basic components loading successful (CPU mode)")
            return True

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False


def test_sos_token_generation():
    """Test SOS token generation."""
    logger.info("Testing SOS token generation...")

    try:
        from ltxv_trainer.SG_datasets import SOSTokenLatents

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sos_generator = SOSTokenLatents(d_model=128, use_zero_init=False).to(device)

        # Test sequence format
        seq_latents = sos_generator(seq_len=100, device=device)
        logger.info(f"Sequence format: {seq_latents.shape}")
        assert seq_latents.shape == (100, 128)

        # Test grid format
        grid_latents = sos_generator(batch_size=1, num_frames=3, height=14, width=24, device=device)
        logger.info(f"Grid format: {grid_latents.shape}")
        assert grid_latents.shape == (1, 128, 3, 14, 24)

        logger.info("✅ SOS token generation successful")
        return True

    except Exception as e:
        logger.error(f"❌ SOS token generation failed: {e}")
        return False


def test_inference_script():
    """Test if inference script can be imported and basic functions work."""
    logger.info("Testing inference script...")

    try:
        # Import inference functions
        sys.path.append(str(Path(__file__).parent))

        # Test imports from inference script
        from inference import load_prompts_from_file

        # Test prompt loading
        demo_prompts_path = Path(__file__).parent.parent / "examples" / "demo_prompts.json"
        if demo_prompts_path.exists():
            prompts = load_prompts_from_file(str(demo_prompts_path))
            logger.info(f"Loaded {len(prompts)} demo prompts")
            assert len(prompts) > 0

        logger.info("✅ Inference script test successful")
        return True

    except Exception as e:
        logger.error(f"❌ Inference script test failed: {e}")
        return False


def main():
    logger.info("🧪 Starting SceneGen Inference Tests")
    logger.info("="*50)

    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("SOS Token Test", test_sos_token_generation),
        ("Inference Script Test", test_inference_script),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test {test_name} failed!")

    logger.info("\n" + "="*50)
    logger.info("🏁 Test Results")
    logger.info("="*50)
    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.info("🎉 All tests passed! Inference setup is ready.")
    else:
        logger.error(f"❌ {total - passed} test(s) failed. Please check the errors above.")

    # System info
    logger.info(f"\n📊 System Information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)