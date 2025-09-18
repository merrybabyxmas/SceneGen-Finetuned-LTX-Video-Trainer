import os  # noqa: I001
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock

import numpy as np 
from PIL import Image

import rich
import torch
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Group,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F  # noqa: N812


from ltxv_trainer import logger
from ltxv_trainer.config import LtxvTrainerConfig
# from ltxv_trainer.datasets import PrecomputedDataset
from ltxv_trainer.SG_datasets import PrecomputedDataset, SOSTokenLatents

from ltxv_trainer.hf_hub_utils import push_to_hub
from ltxv_trainer.model_loader import load_ltxv_components
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
from ltxv_trainer.SG_validation_pipeline import create_multi_shot_validation_pipeline

from ltxv_trainer.quantization import quantize_model
from ltxv_trainer.timestep_samplers import SAMPLERS
# from ltxv_trainer.training_strategies import get_training_strategy
from ltxv_trainer.SG_training_strategy import get_training_strategy

from ltxv_trainer.utils import get_gpu_memory_gb, open_image_as_srgb, convert_checkpoint
from ltxv_trainer.video_utils import read_video
from ltxv_trainer.ltxv_utils import decode_video
import PIL.Image


# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

COMPILE_WARMUP_STEPS = 5
MEMORY_CHECK_INTERVAL = 500


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    compilation_time_seconds: Optional[float]  # Only if compile_with_inductor=True
    training_time: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


class LtxvTrainer:
    def __init__(self, trainer_config: LtxvTrainerConfig) -> None:
        self._config = trainer_config
        self._print_config(trainer_config)
        self._setup_accelerator()
        self._load_models()
        self._compile_transformer()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._current_epoch = 0
        self._checkpoint_paths = []
        self._init_wandb()
        self._training_strategy = get_training_strategy(self._config.conditioning)
        # No longer using SOS token generator - using Gaussian noise instead
        self._sos_token_generator = None






        # Store current training batch for validation
        self._current_training_batch = None

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        # Use the same seed for all processes and ensure deterministic operations
        set_seed(cfg.seed)

        if cfg.model.training_mode == "lora" and not cfg.model.load_checkpoint:
            self._init_lora_weights()

        self._init_optimizer()
        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        logger.info("🚀 Starting training...")

        # Create progress columns with simplified styling
        if disable_progress_bars or not IS_MAIN_PROCESS:
            train_progress = MagicMock()
            sample_progress = MagicMock()
            live = nullcontext()
            if IS_MAIN_PROCESS:
                logger.warning("Progress bars disabled. Status messages will be printed occasionally instead.")
        else:
            train_progress = Progress(
                TextColumn("Training Step"),
                MofNCompleteColumn(),
                BarColumn(bar_width=40, style="blue"),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TextColumn("Time/Step: {task.fields[step_time]:.2f}s"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
            )

            # Create a separate progress instance for sampling
            sample_progress = Progress(
                TextColumn("Sampling validation videos"),
                MofNCompleteColumn(),
                BarColumn(bar_width=40, style="blue"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
            )

            live = Live(Panel(Group(train_progress, sample_progress)), refresh_per_second=2)

        self._transformer.train()
        self._global_step = 0

        # For tracking compilation time
        compilation_time = None
        peak_mem_during_training = start_mem

        # Track when actual training starts (after compilation)
        actual_training_start = None

        sampled_videos_paths = None

        with live:
            task = train_progress.add_task(
                "Training",
                total=cfg.optimization.steps,
                loss=0.0,
                lr=cfg.optimization.learning_rate,
                step_time=0.0,
            )

            if cfg.validation.interval and IS_MAIN_PROCESS and not cfg.validation.skip_initial_validation:
                sampled_videos_paths = self._sample_videos(sample_progress)
            self._accelerator.wait_for_everyone()

            for step in range(cfg.optimization.steps * cfg.optimization.gradient_accumulation_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self._current_epoch += 1  # Increment epoch when dataloader resets
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)
                    if self._current_epoch % 5 == 0:  # Log every 5th epoch only
                        logger.info(f"📈 Starting epoch {self._current_epoch}")

                # Measure compilation time (first COMPILE_WARMUP_STEPS steps)
                if step == COMPILE_WARMUP_STEPS and cfg.acceleration.compile_with_inductor:
                    compilation_time = time.time() - train_start_time
                    actual_training_start = time.time()
                elif step == COMPILE_WARMUP_STEPS and not cfg.acceleration.compile_with_inductor:
                    actual_training_start = train_start_time

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    loss = self._training_step(batch, is_optimization_step)
                    self._accelerator.backward(loss)

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()
                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                        and IS_MAIN_PROCESS
                    ):
                        sampled_videos_paths = self._sample_videos(sample_progress)
                        if sampled_videos_paths and self._config.wandb.log_validation_videos:
                            self._log_validation_videos(sampled_videos_paths, cfg.validation.prompts)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                        and IS_MAIN_PROCESS
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress
                    if IS_MAIN_PROCESS:
                        current_lr = self._optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - train_start_time
                        progress_percentage = self._global_step / cfg.optimization.steps
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."

                        step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                        train_progress.update(
                            task,
                            advance=1 if is_optimization_step else 0,
                            loss=loss.item(),
                            lr=current_lr,
                            step_time=step_time,
                            total_time=total_time,
                        )

                        # Log essential metrics to W&B only every 25 steps
                        if self._global_step % 25 == 0:
                            self._log_metrics(
                                {
                                    "train/loss": loss.item(),
                                    "train/learning_rate": current_lr,
                                    "train/global_step": self._global_step,
                                }
                            )

                        if disable_progress_bars and self._global_step % 100 == 0:
                            logger.info(
                                f"Step {self._global_step}/{cfg.optimization.steps} - Loss: {loss.item():.4f}"
                            )

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second excluding compilation time if needed
        if cfg.acceleration.compile_with_inductor:
            training_time = train_end_time - actual_training_start
            steps_per_second = (cfg.optimization.steps - COMPILE_WARMUP_STEPS) / training_time
        else:
            training_time = train_end_time - train_start_time
            steps_per_second = cfg.optimization.steps / training_time

        samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=train_end_time - train_start_time,
            training_time=training_time,
            compilation_time_seconds=compilation_time,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        train_progress.remove_task(task)
        self._accelerator.end_training()

        if IS_MAIN_PROCESS:
            saved_path = self._save_checkpoint()
            comfy_path = saved_path.parent / f"comfy_{saved_path.name}"
            convert_checkpoint(
                input_path=str(saved_path),
                to_comfy=True,
                output_path=str(comfy_path),
            )

            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, comfy_path, sampled_videos_paths, self._config)

            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/training_time_minutes": stats.training_time / 60,
                        "stats/compilation_time_seconds": stats.compilation_time_seconds,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.end_training()
        
        # Initialize comfy_path to None by default
        comfy_path = None
        if IS_MAIN_PROCESS and 'saved_path' in locals():
            comfy_path = saved_path.parent / f"comfy_{saved_path.name}"

        return comfy_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]], is_optimization_step: bool = False) -> Tensor:
        """Perform a single training step using the configured strategy."""
        # Store current batch for validation
        self._current_training_batch = batch

        # Use strategy to prepare the training batch
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)


        # Use strategy to prepare model inputs
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)

        # Run transformer forward pass
        model_pred = self._transformer(**model_inputs)[0]

        # Save unified debug MP4s if enabled
        if (IS_MAIN_PROCESS and
            hasattr(self._config.debug, 'debug_interval') and
            self._global_step % self._config.debug.debug_interval == 0):
            self._save_unified_debug_mp4s(batch, training_batch, model_pred)

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(model_pred, training_batch)

        return loss


    @torch.no_grad()
    def _save_unified_debug_mp4s(self, batch: dict[str, dict[str, Tensor]], training_batch, model_pred: Tensor = None) -> None:
        """Save debug MP4s for three scenarios: batch_latents, sos_latents, sos_video."""
        try:
            # Skip if not at debug interval
            if self._global_step % self._config.debug.debug_interval != 0:
                return

            # Store batch data for access in _run_debug_scenario
            self._current_batch_data = batch

            logger.info("🎬 Saving unified debug MP4s...")

            # Create debug directory
            output_root = getattr(self._config, 'output_root', 'outputs')
            debug_dir = Path(output_root) / "debug_videos"
            debug_dir.mkdir(exist_ok=True, parents=True)

            # Get training batch prev conditions for scenarios
            if training_batch.prev_seq_len > 0:
                # Extract previous part from latents
                prev_conditions = training_batch.latents[:, :training_batch.prev_seq_len]  # [1, prev_seq_len, 128]
            else:
                # No previous sequence, create dummy prev conditions
                prev_conditions = torch.zeros(1, 1008, 128, device=training_batch.latents.device)

            # Common parameters for all scenarios
            prompt = "a professional portrait video of the earth"
            negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

            # Convert latents to 5D format [B, C, F, H, W] for pipeline
            def convert_to_5d_latents(latents_2d):
                # latents_2d: [B, seq_len, channels] -> [B, channels, frames, height, width]
                batch_size, seq_len, channels = latents_2d.shape

                # Use actual training batch dimensions
                if hasattr(training_batch, 'num_frames') and hasattr(training_batch, 'height') and hasattr(training_batch, 'width'):
                    # For multishot: total frames = prev_frames + curr_frames
                    total_frames = training_batch.num_frames   # prev + curr
                    height = training_batch.height
                    width = training_batch.width
                else:
                    # Fallback: calculate from sequence length and assume square spatial
                    # seq_len = total_frames * height * width
                    # Try common combinations
                    total_frames = 6  # 3 prev + 3 curr
                    spatial_tokens = seq_len // total_frames
                    height = width = int(spatial_tokens ** 0.5)

                expected_tokens = total_frames * height * width
                if expected_tokens != seq_len:
                    logger.warning(f"Token mismatch: expected {expected_tokens}, got {seq_len}. Adjusting frames.")
                    # Recalculate frames based on actual sequence length
                    total_frames = seq_len // (height * width)

                logger.info(f"Convert to 5D: [{batch_size}, {seq_len}, {channels}] -> [{batch_size}, {channels}, {total_frames}, {height}, {width}]")
                return latents_2d.transpose(1, 2).reshape(batch_size, channels, total_frames, height, width)

            # Scenario 1: batch_latents (use training batch latents directly)
            # training_batch.latents already contains [clean prev + noisy curr]
            batch_input_latents = training_batch.latents
            prev_input_latents, prev_outout_latents = batch_input_latents.chunk(2,dim = 1)
            logger.info(f"Batch input latents (direct from training): {prev_input_latents.shape}")
            logger.info(f"  prev_seq_len: {training_batch.prev_seq_len}, total_seq_len: {batch_input_latents.shape[1]}")

            prev_input_latents_5d = convert_to_5d_latents(prev_input_latents)
            logger.info(f"batch latents for batch : {prev_input_latents_5d.shape}")
            self._run_debug_scenario(
                scenario_name="batch_latents",
                reference_latents=prev_input_latents_5d,
                reference_video=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                debug_dir=debug_dir,
                training_batch=training_batch
            )

            # Scenario 2: sos_latents (use Gaussian noise instead of SOS token)
            seq_len = 1008
            d_model = prev_conditions.shape[-1]  # Use same channel dimension as prev_conditions
            sos_latents = torch.randn(seq_len, d_model, device=prev_conditions.device, dtype=prev_conditions.dtype)
            logger.info(f"SOS latents shape: {sos_latents.shape}")

            # Reshape to match expected format [1, 1008, 128] -> [1, 128, 3, 14, 24]
            sos_latents_2d = sos_latents.unsqueeze(0)  # [1008, 128] -> [1, 1008, 128]
            logger.info(f"SOS latents 2D shape: {sos_latents_2d.shape}")
            sos_latents_5d = convert_to_5d_latents(sos_latents_2d)
            logger.info(f"latents for sos latents : {sos_latents_5d.shape}")

            self._run_debug_scenario(
                scenario_name="sos_latents",
                reference_latents=sos_latents_5d,
                reference_video=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                debug_dir=debug_dir,
                training_batch=training_batch
            )

            # Scenario 3: sos_video (use dummy video frames)
            dummy_video = self._create_dummy_video(device=prev_conditions.device)

            self._run_debug_scenario(
                scenario_name="sos_video",
                reference_latents=None,
                reference_video=dummy_video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                debug_dir=debug_dir,
                training_batch=training_batch
            )

        except Exception as e:
            logger.warning(f"Failed to save unified debug MP4s: {e}")

    def _run_debug_scenario(self, scenario_name: str, reference_latents, reference_video, prompt: str, negative_prompt: str, debug_dir: Path, training_batch):
        """Run validation for a specific debug scenario and save input/output videos."""
        try:
            logger.info(f"🎯 Running debug scenario: {scenario_name}")

            # Create SGMultiShotPipeline for debug validation
            from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
            from copy import deepcopy

            # Get device from accelerator
            device = self._accelerator.device

            debug_pipeline = SGMultiShotPipeline(
                scheduler=deepcopy(self._scheduler),
                vae=self._accelerator.unwrap_model(self._vae),
                text_encoder=self._accelerator.unwrap_model(self._text_encoder),
                tokenizer=self._tokenizer,
                transformer=self._accelerator.unwrap_model(self._transformer),
                sos_token_generator=None,
            )
            debug_pipeline.set_progress_bar_config(disable=True)

            # Move pipeline to device
            debug_pipeline = debug_pipeline.to(device)

            # Enable debug capture in pipeline
            debug_pipeline.enable_debug_capture()

            # Ensure inputs are on correct device
            if reference_latents is not None:
                reference_latents = reference_latents.to(device)
            if reference_video is not None:
                reference_video = reference_video.to(device)

            # Prepare pipeline inputs based on scenario
            if scenario_name == "batch_latents":
                # Use batch-specific prompt_embeds for batch_latents scenario
                pipeline_inputs = {
                    "prompt_embeds": training_batch.prompt_embeds,
                    "prompt_attention_mask": training_batch.prompt_attention_mask,
                    "negative_prompt_embeds": None,  # Can be generated if needed
                    "height": 448,
                    "width": 768,
                    "num_frames": 17,
                    "num_videos_per_prompt": 1,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "generator": torch.Generator(device=device).manual_seed(42),
                    "reference_latents": reference_latents,
                    "reference_video": reference_video,
                }
            else:
                # Use global prompt for other scenarios (sos_latents, sos_video)
                pipeline_inputs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "height": 448,
                    "width": 768,
                    "num_frames": 17,
                    "num_videos_per_prompt": 1,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                    "generator": torch.Generator(device=device).manual_seed(42),
                    "reference_latents": reference_latents,
                    "reference_video": reference_video,
                }

            # Run pipeline
            with autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                if scenario_name == "batch_latents":
                    # For batch_latents, temporarily bypass check_inputs validation
                    # since we're using prompt_embeds instead of prompt
                    original_check_inputs = debug_pipeline.check_inputs
                    debug_pipeline.check_inputs = lambda *args, **kwargs: None
                    try:
                        result = debug_pipeline(**pipeline_inputs)
                    finally:
                        debug_pipeline.check_inputs = original_check_inputs
                else:
                    result = debug_pipeline(**pipeline_inputs)
                # logger.info(f"result : {result}")

            # Get debug latents
            debug_latents = debug_pipeline.get_debug_latents()
            logger.info(f"debug input latents : {debug_latents['valid_input'].shape}")

            # Save input video
            if debug_latents.get('valid_input') is not None:
                input_path = debug_dir / f"step_{self._global_step:06d}_valid_input_{scenario_name}.mp4"
                self._save_debug_latents_as_video(debug_latents['valid_input'], input_path, training_batch)
                logger.info(f"✅ Saved {scenario_name} input: {input_path.name}")

            # Save output video
            if debug_latents.get('valid_output') is not None:
                output_path = debug_dir / f"step_{self._global_step:06d}_valid_output_{scenario_name}.mp4"
                self._save_debug_latents_as_video(debug_latents['valid_output'], output_path, training_batch)
                logger.info(f"✅ Saved {scenario_name} output: {output_path.name}")

            # For batch_latents scenario, also save the real batch latents (prev_clean + curr_clean)
            if scenario_name == "batch_latents":
                try:
                    # Get the original batch data that was passed to _save_unified_debug_mp4s
                    # The batch parameter should contain the clean data
                    if hasattr(self, '_current_batch_data'):
                        batch_data = self._current_batch_data
                    else:
                        batch_data = None
                        logger.warning("No batch data available for real latents")

                    real_batch_latents = None

                    # Try to get real latents from various sources
                    if batch_data is not None:
                        logger.info(f"🔍 Batch data keys: {list(batch_data.keys()) if isinstance(batch_data, dict) else 'Not a dict'}")

                        # Check if batch_data has clean latents
                        if isinstance(batch_data, dict):
                            # Method 1: Look for direct clean latents in batch data
                            if 'clean_latents' in batch_data:
                                real_batch_latents = batch_data['clean_latents']
                                logger.info(f"✅ Found clean_latents in batch data: {real_batch_latents.shape}")

                            # Method 2: Reconstruct from latent_conditions and prev_conditions
                            elif 'latent_conditions' in batch_data:
                                logger.info("🔄 Reconstructing clean latents from latent_conditions and prev_conditions")

                                # Get current clean latents from latent_conditions
                                curr_conditions = batch_data['latent_conditions']
                                if torch.is_tensor(curr_conditions):
                                    curr_clean = curr_conditions
                                elif isinstance(curr_conditions, dict) and 'latents' in curr_conditions:
                                    curr_clean = curr_conditions['latents']
                                else:
                                    curr_clean = None

                                # Get previous clean latents from prev_conditions
                                prev_conditions = batch_data.get('prev_conditions', None)
                                if prev_conditions is not None:
                                    if torch.is_tensor(prev_conditions):
                                        prev_clean = prev_conditions
                                    elif isinstance(prev_conditions, dict) and 'latents' in prev_conditions:
                                        prev_clean = prev_conditions['latents']
                                    else:
                                        prev_clean = None
                                else:
                                    prev_clean = None

                                # Combine prev + curr clean latents
                                if curr_clean is not None:
                                    if prev_clean is not None:
                                        real_batch_latents = torch.cat([prev_clean, curr_clean], dim=1)
                                        logger.info(f"✅ Combined prev_clean + curr_clean: {real_batch_latents.shape}")
                                        logger.info(f"   prev_clean: {prev_clean.shape}, curr_clean: {curr_clean.shape}")
                                    else:
                                        real_batch_latents = curr_clean
                                        logger.info(f"✅ Using curr_clean only: {real_batch_latents.shape}")

                            # Method 3: Look through all tensor values in batch_data
                            else:
                                for key, value in batch_data.items():
                                    if torch.is_tensor(value) and len(value.shape) == 3:  # [batch, seq, dim]
                                        logger.info(f"🔍 Found tensor in batch_data['{key}']: {value.shape}")
                                        if 'clean' in key.lower() or ('latent' in key.lower() and 'noisy' not in key.lower()):
                                            real_batch_latents = value
                                            logger.info(f"✅ Using {key} as real batch latents: {real_batch_latents.shape}")
                                            break

                    # Fallback: Try to reconstruct from training_batch using targets
                    if real_batch_latents is None and hasattr(training_batch, 'targets') and hasattr(training_batch, 'latents'):
                        logger.info("🔄 Trying to reconstruct clean latents from training_batch")

                        # In prepare_batch: targets = noise - curr_lat and latents contains noisy data
                        # To get clean latents: clean = noisy - (noise - clean) = noisy - targets + clean
                        # But actually: targets = noise - clean, so clean = noise - targets
                        # And noisy = (1-sigma)*clean + sigma*noise
                        # So: clean = (noisy - sigma*noise) / (1-sigma)
                        # But we can also try: clean ≈ noisy - targets (approximation)

                        noisy_latents = training_batch.latents  # [B, seq_len, D] - contains noisy data
                        targets = training_batch.targets        # [B, seq_len, D] - contains (noise - clean)

                        # Method 1: Try reverse calculation from targets
                        # Since targets = noise - clean in curr region, and prev region has targets=0
                        # We can reconstruct clean latents for the curr region
                        if hasattr(training_batch, 'prev_seq_len'):
                            prev_seq_len = training_batch.prev_seq_len

                            # For prev region: latents should already be clean (no noise added)
                            prev_clean = noisy_latents[:, :prev_seq_len]  # Should be clean already

                            # For curr region: need to reconstruct clean from noisy and targets
                            noisy_curr = noisy_latents[:, prev_seq_len:]  # Noisy current latents
                            targets_curr = targets[:, prev_seq_len:]      # noise - clean for current

                            # Try to get clean current latents
                            # We know: targets_curr = noise - clean_curr
                            # And: noisy_curr = (1-sigma)*clean_curr + sigma*noise
                            # Let's try approximation: clean_curr ≈ noisy_curr - targets_curr
                            clean_curr_approx = noisy_curr - targets_curr

                            # Combine prev (already clean) + curr (reconstructed clean)
                            real_batch_latents = torch.cat([prev_clean, clean_curr_approx], dim=1)
                            logger.info(f"✅ Reconstructed clean latents from targets: {real_batch_latents.shape}")
                            logger.info(f"   prev_clean: {prev_clean.shape}, clean_curr: {clean_curr_approx.shape}")
                        else:
                            # Single shot case: try to reconstruct clean from noisy - targets
                            real_batch_latents = noisy_latents - targets
                            logger.info(f"✅ Reconstructed clean latents (single shot): {real_batch_latents.shape}")

                    # Save the real batch latents if we found them
                    if real_batch_latents is not None:
                        real_path = debug_dir / f"step_{self._global_step:06d}_valid_real_{scenario_name}.mp4"
                        self._save_debug_latents_as_video(real_batch_latents, real_path, training_batch)
                        logger.info(f"✅ Saved {scenario_name} real batch (clean latents): {real_path.name}")
                    else:
                        logger.warning(f"❌ Could not find real batch latents in any available data source")

                except Exception as e:
                    logger.warning(f"Failed to save real batch latents for {scenario_name}: {e}")
                    import traceback
                    logger.warning(f"Traceback: {traceback.format_exc()}")

            # Disable debug capture
            debug_pipeline.disable_debug_capture()

        except Exception as e:
            logger.warning(f"Failed to run debug scenario {scenario_name}: {e}")

    def _create_dummy_video(self, device) -> torch.Tensor:
        """Create a dummy video tensor for sos_video scenario."""
        # Create 17 frames of 448x768 with simple pattern
        frames = []
        for i in range(17):
            # Create a simple gradient pattern
            frame = torch.zeros(3, 448, 768, device=device)
            frame[0] = (i / 16.0)  # Red channel gradient over time
            frame[1] = 0.5  # Green channel constant
            frame[2] = 1.0 - (i / 16.0)  # Blue channel inverse gradient
            frames.append(frame)

        # Stack to [F, C, H, W] format
        dummy_video = torch.stack(frames)  # [17, 3, 448, 768]
        return dummy_video

    def _save_debug_latents_as_video(self, latents: torch.Tensor, output_path: Path, training_batch):
        """Save debug latents as MP4 video."""
        try:
            # Use debug GPU
            debug_device = torch.device(f"cuda:{self._config.debug.debug_gpu_id}")

            # Move VAE to debug device
            self._vae.to(debug_device)
            latents = latents.to(debug_device)
            logger.info(f"[save] latents : {latents.shape}")

            # Decode latents to video
            with autocast(debug_device.type, dtype=torch.bfloat16):
                # Take first batch item and decode
                video_latents = latents[0:1]  # [1, seq_len, 128]
                # Use actual training batch dimensions
                if hasattr(self, '_current_training_batch') and self._current_training_batch is not None:
                    # logger.info(f"  training batch : {self._current_training_batch}")
                    num_frames = self._current_training_batch["latent_conditions"]["num_frames"] *2
                    height = self._current_training_batch["latent_conditions"]["height"]
                    width = self._current_training_batch["latent_conditions"]["width"]
                else:
                    # Fallback dimensions
                    num_frames = 6
                    height = 14  # 448 // 8
                    width = 24   # 768 // 8

                from .ltxv_utils import decode_video
                logger.info(f"video latents : {video_latents.shape}")
                video_latents = video_latents.squeeze(0)
                decoded = decode_video(
                    vae=self._vae,
                    latents=video_latents,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    device=debug_device,
                    dtype=torch.bfloat16
                )
            logger.info(f"decoded shape:{decoded.shape}")
            decoded = decoded.squeeze(0)

            # Convert to video format and save
            video_tensor = decoded.permute(1, 2, 3, 0)  # (F, H, W, 3)
            video_tensor = video_tensor.clamp(-1, 1).add(1).div(2)  # [-1,1] -> [0,1]
            logger.info(f"[save] video tensor : { video_tensor.shape}")

            # Convert to numpy and scale to [0, 255]f
            video_np = video_tensor.cpu().float().numpy()
            video_np = (video_np * 255).astype(np.uint8)

            # Convert to list of PIL Images for video export
            from PIL import Image
            video_frames = [Image.fromarray(frame) for frame in video_np]

            # Save as MP4 video
            from diffusers.utils import export_to_video
            export_to_video(video_frames, str(output_path), fps=24)

            # Move VAE back to CPU
            self._vae.to("cpu")

        except Exception as e:
            logger.warning(f"Failed to save debug video {output_path}: {e}")
            try:
                self._vae.to("cpu")
            except:
                pass


    @staticmethod
    def _print_config(config: BaseModel) -> None:
        """Print the configuration as a nicely formatted table."""
        if not IS_MAIN_PROCESS:
            return

        from rich.table import Table

        table = Table(title="⚙️ Training Configuration", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="bold cyan")

        def flatten_config(cfg: BaseModel, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for field, value in cfg:
                full_field = f"{prefix}.{field}" if prefix else field
                if isinstance(value, BaseModel):
                    # Recursively flatten nested config
                    rows.extend(flatten_config(value, full_field))
                elif isinstance(value, (list, tuple, set)):
                    # Format collections
                    value_str = ", ".join(str(item) for item in value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
                else:
                    # Add simple values
                    value_str = str(value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
            return rows

        for param, value in flatten_config(config):
            table.add_row(param, value)

        rich.print(table)

    def _load_models(self) -> None:
        """Load the LTXV model components."""

        # Load all model components using the new loader
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        components = load_ltxv_components(
            model_source=self._config.model.model_source,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
            transformer_dtype=transformer_dtype,
            vae_dtype=torch.bfloat16,
        )

        # Prepare components with accelerator
        self._scheduler = components.scheduler
        self._tokenizer = components.tokenizer
        self._text_encoder = components.text_encoder
        self._vae = components.vae
        self._transformer = components.transformer

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.warning(f"Quantizing model with precision: {self._config.acceleration.quantization}")
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

    def _save_batch_visualization(self, training_batch, prev_clean_latents, curr_clean_latents, full_latents, model_pred: Tensor = None, batch_viz_dir = "outputs/batchvisualization"):
        """Save batch visualization videos for debugging."""
        try:
            
            batch_idx = 0
            debug_device = torch.device(f"cuda:{self._config.debug.debug_gpu_id}")

            logger.info(f"[batch visualization info]\n"
                        f"prev conditions shape : {prev_clean_latents.shape}\n"
                        f"curr conditions shape : {curr_clean_latents.shape}\n"
                        f"total latents shape : {full_latents.shape}")
                        
            
            # Separate prev (clean SOS) and curr (noisy) portions
            if prev_clean_latents is not None:
                prev_seq_len = prev_clean_latents.shape[1]  # 1008
                noisy_curr_latents = full_latents[:, prev_seq_len:]  # (1, 1008, 128) - current with noise
                clean_prev_latents_from_batch = full_latents[:, :prev_seq_len]  # (1, 1008, 128) - prev (should be clean)
            else:
                # No previous shot, all latents are current
                noisy_curr_latents = full_latents
                clean_prev_latents_from_batch = None
            
            # VAE latent channels
            vae_channels = 128
            
            # Helper function to decode and save latents as video
            def decode_and_save_video(latents, suffix, step, shot_type=""):
                try:
                    # Calculate total tokens for the video (F * H * W) using actual batch metadata
                    total_tokens = training_batch.num_frames * training_batch.height * training_batch.width
                    
                    if latents.shape[1] >= total_tokens:
                        # Extract full video latents
                        video_latents = latents[:, :total_tokens]  # (1, F*H*W, 128)
                        
                        # Reshape to 5D VAE format for full video: (batch, channels, frames, height, width)
                        batch_F = training_batch.num_frames
                        batch_H = training_batch.height 
                        batch_W = training_batch.width
                        reshaped = video_latents.transpose(1, 2).reshape(1, vae_channels, batch_F, batch_H, batch_W)
                        reshaped = reshaped.to(debug_device)
                        
                        # Decode full video using ltxv_utils decode_video
                        with autocast(debug_device.type, dtype=torch.bfloat16):
                            decoded = decode_video(
                                vae=self._vae,
                                latents=video_latents,
                                num_frames=batch_F,
                                height=batch_H,
                                width=batch_W,
                                device=debug_device,
                                dtype=torch.bfloat16
                            )  # decode_video returns (3, F, H, W)
                        decoded = decoded.squeeze(0)
                        logger.info(f"decoded shape : {decoded.shape}")
                        # Convert to video format and save
                        # decode_video returns (3, F, H, W)
                        video_tensor = decoded.permute(1, 2, 3, 0)  # (F, H, W, 3)
                        video_tensor = video_tensor.clamp(-1, 1).add(1).div(2)  # [-1,1] -> [0,1]
                        
                        # Convert to numpy and scale to [0, 255]
                        video_np = video_tensor.cpu().float().numpy()
                        video_np = (video_np * 255).astype(np.uint8)
                        
                        # Convert to list of PIL Images for video export
                        video_frames = [Image.fromarray(frame) for frame in video_np]
                        
                        # Save as MP4 video with timestep info
                        step_str = f"epoch_{self._current_epoch:02d}_batch_{step:02d}"
                        shot_prefix = f"{shot_type}_" if shot_type else ""
                        
                        # Add timestep info for noisy videos
                        timestep_str = ""
                        if "noisy" in suffix and hasattr(training_batch, 'sigmas'):
                            timestep_val = float(training_batch.sigmas[batch_idx, 0, 0].item())
                            timestep_str = f"_t{timestep_val:.3f}"
                        
                        video_path = batch_viz_dir / f"{step_str}_{shot_prefix}{suffix}{timestep_str}.mp4"
                        
                        # Use diffusers export_to_video function
                        from diffusers.utils import export_to_video
                        export_to_video(video_frames, str(video_path), fps=24)
                        
                        return True
                        
                    else:
                        logger.warning(f"Not enough tokens for full video processing")
                        return False
                        
                except Exception as e:
                    logger.warning(f"Failed to decode video {suffix}: {e}")
                    return False
            
            # Save visualizations for this batch
            step = self._global_step
            saved_count = 0
            
            # 1. Save prev latents (SOS/previous shot) - clean
            if prev_clean_latents is not None:
                shot_type = "SOS" if step == 1 else "prev"
                if decode_and_save_video(prev_clean_latents, "prev_clean", step, shot_type):
                    saved_count += 1
                    
                # Also save prev from training batch to verify it's clean
                if decode_and_save_video(clean_prev_latents_from_batch, "prev_from_training_batch", step, shot_type):
                    saved_count += 1
                
                if step == 1:
                    logger.info(f"Batch {step}: Using SOS token as previous latent (first shot)")
                else:
                    logger.info(f"Batch {step}: Using previous shot latent")
            else:
                # Should not happen in multi-shot training
                logger.warning(f"Batch {step}: No previous latent found - this shouldn't happen in multi-shot training")
            
            # 2. Save curr latents - clean (before noise)
            if decode_and_save_video(curr_clean_latents, "curr_clean", step, "curr"):
                saved_count += 1
            
            # 3. Save curr latents - noisy (after noise)
            if decode_and_save_video(noisy_curr_latents, "curr_noisy", step, "curr"):
                saved_count += 1
            
            # 4. Save curr latents - denoised (model prediction) 
            if model_pred is not None:
                # For ReferenceVideoTrainingStrategy, model_pred has full sequence length (prev + curr)
                # Need to extract current part only
                if hasattr(training_batch, 'prev_seq_len') and training_batch.prev_seq_len > 0:
                    # ReferenceVideoTrainingStrategy: extract current part from full prediction
                    curr_seq_len = noisy_curr_latents.shape[1]
                    model_pred_curr = model_pred[batch_idx:batch_idx+1, -curr_seq_len:]  # Extract current part
                    logger.info(f"ReferenceVideo: Extracted current part from model_pred: {model_pred_curr.shape}")
                else:
                    # StandardTrainingStrategy: model_pred is already current part only
                    model_pred_curr = model_pred[batch_idx:batch_idx+1]
                    logger.info(f"Standard: Using full model_pred as current: {model_pred_curr.shape}")
                
                # Check if we're using PC-CFM strategy (targets is dict)
                if isinstance(training_batch.targets, dict):
                    # PC-CFM denoised calculation
                    # In PC-CFM, model predicts the velocity field v_t
                    # The denoised estimate is: x_0 = x_t - t * v_t (where t is timestep, not sigma)
                    timesteps = training_batch.timesteps[batch_idx:batch_idx+1]  # (1, seq_len)

                    # Extract timestep for current part (skip prev part if exists)
                    curr_timesteps = timesteps[:, training_batch.prev_seq_len:]  # (1, curr_seq_len)
                    t_values = curr_timesteps.mean(dim=1, keepdim=True).unsqueeze(-1)  # (1, 1, 1) for broadcasting

                    # PC-CFM: move backward along the predicted velocity field using timestep
                    denoised_curr_latents = noisy_curr_latents - t_values * model_pred_curr
                    logger.info(f"PC-CFM denoised calculation: x_t - t * velocity_pred (t={t_values.item():.3f})")
                else:
                    # Standard strategy: model predicts noise (epsilon parameterization)
                    denoised_curr_latents = noisy_curr_latents - model_pred_curr
                    logger.info(f"Standard denoised calculation: noisy - noise_pred")
                
                if decode_and_save_video(denoised_curr_latents, "curr_denoised", step, "curr"):
                    saved_count += 1
                    logger.info(f"✓ Denoised video saved for batch {step}")
                else:
                    logger.warning(f"❌ Failed to save denoised video for batch {step}")
            
            # 4. Additional verification messages
            if step == 1:  # First batch
                logger.info("🔍 First batch detected")
                logger.info("   SOS token is being used as prev latent")
                logger.info("   Check: epoch_00_batch_01_SOS_prev_clean.mp4 contains the SOS token")
            elif step == 2:  # Second batch
                logger.info("🔍 Second batch detected") 
                logger.info("   Check: epoch_00_batch_01_curr_curr_clean.mp4 should match epoch_00_batch_02_prev_prev_clean.mp4")
                logger.info("   This verifies the SOS→curr transition is working correctly")
            
            if saved_count > 0:
                logger.info(f"🎥 Batch visualization saved: {saved_count} videos for batch {step}")
            
            # Move VAE back to CPU
            self._vae.to("cpu")

        except Exception as e:
            logger.warning(f"Failed to save batch visualization: {e}")
            try:
                self._vae.to("cpu")
            except:
                pass

    @staticmethod
    def _print_config(config: BaseModel) -> None:
        """Print the configuration as a nicely formatted table."""
        if not IS_MAIN_PROCESS:
            return

        from rich.table import Table

        table = Table(title="⚙️ Training Configuration", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="bold cyan")

        def flatten_config(cfg: BaseModel, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for field, value in cfg:
                full_field = f"{prefix}.{field}" if prefix else field
                if isinstance(value, BaseModel):
                    # Recursively flatten nested config
                    rows.extend(flatten_config(value, full_field))
                elif isinstance(value, (list, tuple, set)):
                    # Format list/tuple/set values
                    value_str = ", ".join(str(item) for item in value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
                else:
                    # Add simple values
                    value_str = str(value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
            return rows

        for param, value in flatten_config(config):
            table.add_row(param, value)

        rich.print(table)

    def _load_models(self) -> None:
        """Load the LTXV model components."""

        # Load all model components using the new loader
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        components = load_ltxv_components(
            model_source=self._config.model.model_source,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
            transformer_dtype=transformer_dtype,
            vae_dtype=torch.bfloat16,
        )

        # Prepare components with accelerator
        self._scheduler = components.scheduler
        self._tokenizer = components.tokenizer
        self._text_encoder = components.text_encoder
        self._vae = components.vae
        self._transformer = components.transformer

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.warning(f"Quantizing model with precision: {self._config.acceleration.quantization}")
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        # Freeze all models. We later unfreeze the transformer based on training mode.
        self._text_encoder.requires_grad_(False)
        self._vae.requires_grad_(False)
        self._transformer.requires_grad_(False)

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def _compile_transformer(self) -> None:
        """Compile the transformer model with Torch Inductor."""

        if not self._config.acceleration.compile_with_inductor:
            return

        torch._dynamo.config.inline_inbuilt_nn_modules = True
        torch._dynamo.config.cache_size_limit = 128

        compile_module = partial(torch.compile, mode=self._config.acceleration.compilation_mode)
        self._transformer.transformer_blocks = nn.ModuleList(
            [compile_module(block) for block in self._transformer.transformer_blocks],
        )

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        # Enable gradient checkpointing if requested (must be before accelerator.prepare)
        if self._config.optimization.enable_gradient_checkpointing:
            self._transformer.enable_gradient_checkpointing()
            if IS_MAIN_PROCESS:
                logger.info("✅ Gradient checkpointing enabled")

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        if IS_MAIN_PROCESS:
            logger.info(f"Trainable params: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        if IS_MAIN_PROCESS:
            logger.info(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        self._transformer.add_adapter(lora_config)

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config."""
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            return

        transformer = self._accelerator.unwrap_model(self._transformer)

        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")
        state_dict = load_file(checkpoint_path)

        if self._config.model.training_mode == "full":
            transformer.load_state_dict(state_dict)
        else:  # LoRA mode
            # Adjust layer names to match PEFT format
            state_dict = {k.replace("transformer.", "", 1): v for k, v in state_dict.items()}
            state_dict = {k.replace("lora_A", "lora_A.default", 1): v for k, v in state_dict.items()}
            state_dict = {k.replace("lora_B", "lora_B.default", 1): v for k, v in state_dict.items()}

            # Load LoRA weights and verify all weights were loaded
            _, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
            if unexpected_keys:
                raise ValueError(f"Failed to load some LoRA weights: {unexpected_keys}")

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""
        prepare = self._accelerator.prepare
        self._vae = prepare(self._vae).to("cpu")
        self._transformer = prepare(self._transformer)
        self._text_encoder = prepare(self._text_encoder)

        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder = self._text_encoder.to("cpu")

        # SOS token generator is now None, no need to move to device


    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()

            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root, 
                data_sources=data_sources,
                dataset_size=self._config.data.dataset_size
            )
            if IS_MAIN_PROCESS:
                logger.info(f"Dataset loaded: {len(self._dataset):,} samples")

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # type: ignore

            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        # Add scheduler initialization
        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            # Remove par784ameters that LinearLR doesn't support
            params.pop("eta_min", None)  # LinearLR doesn't support eta_min
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            eta_min = params.pop("eta_min", 0)
            # Ensure eta_min is a float (in case it's passed as string from YAML)
            if isinstance(eta_min, str):
                eta_min = float(eta_min)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=eta_min,
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            eta_min = params.pop("eta_min", 5e-5)
            # Ensure eta_min is a float (in case it's passed as string from YAML)
            if isinstance(eta_min, str):
                eta_min = float(eta_min)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),  # First restart cycle length
                T_mult=params.pop("T_mult", 1),  # Multiplicative factor for cycle lengths
                eta_min=eta_min,
                **params,
            )
        elif scheduler_type == "polynomial":
            # Remove parameters that PolynomialLR doesn't support
            params.pop("eta_min", None)  # PolynomialLR doesn't support eta_min
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            # Remove parameters that StepLR doesn't support
            params.pop("eta_min", None)  # StepLR doesn't support eta_min
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )

        # Log information about distributed training
        if self._accelerator.num_processes > 1:
            logger.info(f"Distributed training: {self._accelerator.num_processes} processes, global batch: {self._config.optimization.batch_size * self._accelerator.num_processes}")

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_videos(self, progress: Progress) -> list[Path] | None:
        """Run validation by generating images from validation prompts."""

        self._vae.to(self._accelerator.device)
        # Model is already in the correct device if loaded in 8-bit.
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to(self._accelerator.device)

        # Check if multi-shot validation is enabled
        if self._config.validation.enable_multi_shot:
            return self._sample_multi_shot_videos(progress)

        use_images = self._config.validation.images is not None

        pipeline = SGMultiShotPipeline(
            scheduler=deepcopy(self._scheduler),
            vae=self._accelerator.unwrap_model(self._vae),
            text_encoder=self._accelerator.unwrap_model(self._text_encoder),
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer),
            sos_token_generator=None,
        )
        pipeline.set_progress_bar_config(disable=True)

        # Create a task in the sampling progress
        task = progress.add_task(
            "sampling",
            total=len(self._config.validation.prompts),
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        i = 0
        for j, prompt in enumerate(self._config.validation.prompts):
            generator = torch.Generator(device=self._accelerator.device).manual_seed(self._config.validation.seed)

            # Generate video
            width, height, frames = self._config.validation.video_dims

            pipeline_inputs = {
                "prompt": prompt,
                "negative_prompt": self._config.validation.negative_prompt,
                "width": width,
                "height": height,
                "num_frames": frames,
                "num_inference_steps": self._config.validation.inference_steps,
                "guidance_scale": self._config.validation.guidance_scale,
                "generator": generator,
                "output_reference_comparison": True,
                "save_intermediate_steps": True,
                "save_step_interval": max(1, self._config.validation.inference_steps // 5),  # Save every 20% of steps
            }

            # Load and add first frame image, if provided
            if use_images:
                image_path = self._config.validation.images[j]
                image = open_image_as_srgb(image_path)
                if image.size != (height, width):
                    # Resize and center crop the image to match the validation video dimensions
                    image = F.resize(image, size=min(width, height))
                    image = F.center_crop(image, output_size=(width, height))
                pipeline_inputs["image"] = image

            # Load and add reference video, if provided
            if self._config.validation.reference_videos is not None:
                video_path = self._config.validation.reference_videos[j]
                ref_video, _ = read_video(video_path)[:frames]
                logger.info(f"ref video shape : {ref_video.shape}")
                pipeline_inputs["reference_video"] = ref_video

            with autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                result = pipeline(**pipeline_inputs)
                videos = result.frames
                
                # Save intermediate steps if available
                if hasattr(result, 'intermediate_latents') and result.intermediate_latents:
                    self._save_intermediate_validation_steps(result.intermediate_latents, prompt, j)

            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                export_to_video(video, str(video_path), fps=24)
                video_paths.append(video_path)
                i += 1
            if hasattr(progress, 'update'):
                progress.update(task, advance=1)

        if hasattr(progress, 'remove_task'):
            progress.remove_task(task)

        # Move unused components back to CPU.
        self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")

        if IS_MAIN_PROCESS:
            logger.info(f"🎥 Validation samples saved for step {self._global_step}")
        return video_paths

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_multi_shot_videos(self, progress: Progress) -> list[Path] | None:
        """Run multi-shot validation by generating sequential video shots from a single prompt."""
        
        # Create SGMultiShotPipeline for multi-shot validation
        multi_shot_pipeline = SGMultiShotPipeline(
            scheduler=deepcopy(self._scheduler),  # Copy scheduler like single-shot validation
            vae=self._accelerator.unwrap_model(self._vae),  # Unwrap like single-shot validation
            text_encoder=self._accelerator.unwrap_model(self._text_encoder),  # Unwrap like single-shot validation
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer),  # Unwrap like single-shot validation
            sos_token_generator=None,
        )
        multi_shot_pipeline.set_progress_bar_config(disable=True)
        
        # Create output directory for multi-shot samples
        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        all_video_paths = []
        
        # Create a task in the sampling progress
        task = progress.add_task(
            "multi-shot sampling",
            total=len(self._config.validation.prompts),
        )
        
        # Generate multi-shot sequence for each validation prompt
        for i, prompt in enumerate(self._config.validation.prompts):
            
            try:
                # Generate multiple shots sequentially using reference latents
                width, height, frames = self._config.validation.video_dims
                num_shots = self._config.validation.num_shots
                sequence_videos = []
                previous_shot_latents = None
                previous_shot_video = None  # Store previous shot video for concatenation
                
                for shot_idx in range(num_shots):
                    generator = torch.Generator(device=self._accelerator.device).manual_seed(
                        self._config.validation.seed
                    )
                    
                    # Pipeline inputs
                    pipeline_inputs = {
                        "prompt": prompt,
                        "negative_prompt": self._config.validation.negative_prompt,
                        "conditions": [],  # Empty conditions list for multishot
                        "image": None,     # No image conditioning for multishot
                        "video": None,     # No video conditioning for multishot
                        "frame_index": None,  # No frame indexing for multishot
                        "strength": None,     # No strength parameter for multishot
                        "width": width,
                        "height": height,
                        "num_frames": frames,
                        "num_inference_steps": self._config.validation.inference_steps,
                        "guidance_scale": self._config.validation.guidance_scale,
                        "generator": generator,
                        "output_type": "pil",
                        "return_latents": True,  # Return latents for next shot conditioning
                    }

                    # Add reference latents based on shot index and starts_with config
                    if shot_idx == 0:
                        # First shot: conditioning based on starts_with setting
                        if self._config.validation.starts_with == "batch" and self._current_training_batch is not None:
                            # Use current training batch's prev condition
                            batch_prev_conditions = self._current_training_batch['prev_conditions']
                            if batch_prev_conditions is not None:
                                if not torch.is_tensor(batch_prev_conditions):
                                    batch_prev_latents = batch_prev_conditions["latents"][0:1]  # Take first batch item [1, seq, D]
                                else:
                                    batch_prev_latents = batch_prev_conditions[0:1]  # [1, seq, D]

                                logger.info(f"1st shot: Training batch prev condition shape : {batch_prev_latents.shape}")

                                # Convert from sequence format [B, seq, D] to 5D tensor [B, C, F, H, W]
                                # Use the same approach as batch_visualization (which works correctly)
                                latent_num_frames = (frames - 1) // 8 + 1  # VAE temporal compression ratio
                                latent_height = height // 32  # VAE spatial compression ratio
                                latent_width = width // 32

                                try:
                                    # Use the same reshaping approach as batch_visualization (lines 970-971)
                                    vae_channels = 128
                                    total_tokens = latent_num_frames * latent_height * latent_width

                                    if batch_prev_latents.shape[1] == total_tokens:
                                        # Extract video latents and reshape like batch_visualization does
                                        video_latents = batch_prev_latents[:, :total_tokens]  # (1, F*H*W, 128)
                                        reshaped_latents = video_latents.transpose(1, 2).reshape(
                                            1, vae_channels, latent_num_frames, latent_height, latent_width
                                        )  # (1, 128, F, H, W)

                                        # The pipeline will apply normalization, but we want to use the latents as-is
                                        # So we pre-apply inverse normalization to counteract the pipeline's normalization
                                        # This way: inverse_normalize(latents) -> pipeline_normalize(inverse_normalize(latents)) = latents
                                        try:
                                            if (hasattr(multi_shot_pipeline.vae, 'latents_mean') and
                                                multi_shot_pipeline.vae.latents_mean is not None):

                                               pass
                                            else:
                                                pipeline_inputs["reference_latents"] = reshaped_latents
                                                logger.info(f"1st shot: Using reshaped batch prev condition without normalization : {reshaped_latents.shape}")
                                        except Exception as norm_e:
                                            logger.warning(f"1st shot: Failed to apply inverse normalization ({norm_e}), using raw latents")
                                            pipeline_inputs["reference_latents"] = reshaped_latents
                                    else:
                                        logger.warning(f"1st shot: Token count mismatch {batch_prev_latents.shape[1]} vs {total_tokens}, falling back to SOS")
                                        raise ValueError("Token count mismatch")

                                except Exception as e:
                                    logger.warning(f"1st shot: Failed to reshape batch prev condition ({e}), falling back to SOS")
                                    # Fallback to SOS if reshaping fails
                                    sos_latents = self._sos_token_generator(
                                        batch_size=1,
                                        num_frames=latent_num_frames,
                                        height=latent_height,
                                        width=latent_width,
                                        device=self._accelerator.device
                                    )
                                    pipeline_inputs["reference_latents"] = sos_latents
                            else:
                                logger.warning("1st shot: No prev conditions in training batch, falling back to SOS")
                                # Fallback to SOS if no prev conditions
                                latent_num_frames = (frames - 1) // 8 + 1
                                latent_height = height // 32
                                latent_width = width // 32
                                sos_latents = self._sos_token_generator(
                                    batch_size=1,
                                    num_frames=latent_num_frames,
                                    height=latent_height,
                                    width=latent_width,
                                    device=self._accelerator.device
                                )
                                pipeline_inputs["reference_latents"] = sos_latents
                        else:
                            # Default: use SOS token conditioning
                            latent_num_frames = (frames - 1) // 8 + 1  # VAE temporal compression ratio
                            latent_height = height // 32  # VAE spatial compression ratio
                            latent_width = width // 32
                            latent_seq = latent_num_frames * latent_height * latent_width
                            logger.info(f"1st shot: Generating SOS latents : sequence {latent_seq} = {latent_num_frames} x {latent_height} x {latent_width}")

                            sos_latents = self._sos_token_generator(
                                batch_size=1,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                device=self._accelerator.device
                            )
                            logger.info(f"sos latents shape : {sos_latents.shape}")
                            pipeline_inputs["reference_latents"] = sos_latents
                    else:
                        # Subsequent shots: use previous shot latents as conditioning
                        if previous_shot_latents is not None:
                            pipeline_inputs["reference_latents"] = previous_shot_latents
                    
                    # Generate current shot
                    with autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                        result = multi_shot_pipeline(**pipeline_inputs)

                        # Handle different result formats (dict or object)
                        if isinstance(result, dict):
                            current_video = result.get('frames')
                            latents_result = result.get('latents')
                        else:
                            current_video = getattr(result, 'frames', None)
                            latents_result = getattr(result, 'latents', None)

                        if current_video is not None:
                            if isinstance(current_video, list):
                                current_video = current_video[0]  # Get first video from batch

                    # Save current shot
                    shot_path = output_dir / f"step_{self._global_step:06d}_prompt_{i}_shot_{shot_idx}.mp4"
                    export_to_video(current_video, str(shot_path), fps=24)
                    sequence_videos.append(shot_path)

                    # Save concatenated prev+curr video for shots after the first
                    if shot_idx > 0 and previous_shot_video is not None:
                        try:
                            # Concatenate previous and current video frames
                            concatenated_frames = previous_shot_video + current_video
                            concat_path = output_dir / f"step_{self._global_step:06d}_prompt_{i}_shot_{shot_idx}_prev+curr.mp4"
                            export_to_video(concatenated_frames, str(concat_path), fps=24)
                            logger.info(f"Saved concatenated prev+curr video: {concat_path.name} ({len(previous_shot_video)} + {len(current_video)} = {len(concatenated_frames)} frames)")
                        except Exception as e:
                            logger.warning(f"Failed to create concatenated video for shot {shot_idx}: {e}")

                    # Store current video and latents for next shot
                    previous_shot_video = current_video
                    if shot_idx < num_shots - 1:  # Don't need to store latents for the last shot
                        if latents_result is not None:
                            previous_shot_latents = latents_result.clone() if hasattr(latents_result, 'clone') else latents_result
                        else:
                            logger.warning(f"No latents returned for shot {shot_idx}, multishot sequence may be broken")
                            previous_shot_latents = None
                
                all_video_paths.extend(sequence_videos)
                logger.info(f"Generated {len(sequence_videos)} shots for prompt: '{prompt}'")
                
            except Exception as e:
                logger.error(f"Failed to generate multi-shot sequence for prompt {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Update progress
            if hasattr(progress, 'update'):
                progress.update(task, advance=1)
        
        # Remove progress task
        if hasattr(progress, 'remove_task'):
            progress.remove_task(task)
        
        # Move unused components back to CPU
        self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")
        
        if all_video_paths:
            if IS_MAIN_PROCESS:
                logger.info(f"🎬 Multi-shot validation samples saved for step {self._global_step}")
            return all_video_paths
        else:
            if IS_MAIN_PROCESS:
                logger.warning("Multi-shot validation failed")
            return None

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training time: {stats.training_time / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.compilation_time_seconds is not None:
            stats_str += f"\n - Compilation time: {stats.compilation_time_seconds:.1f} seconds\n"
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path:
        """Save the model weights."""

        # Create checkpoints directory if it doesn't exist
        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Create filename with step number
        prefix = "model" if self._config.model.training_mode == "full" else "lora"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get model state dict
        unwrapped_model = self._accelerator.unwrap_model(self._transformer)

        if self._config.model.training_mode == "full":
            state_dict = unwrapped_model.state_dict()
            save_file(state_dict, saved_weights_path)
            logger.info(f"💾 Model checkpoint saved: step {self._global_step}")
        elif self._config.model.training_mode == "lora":
            state_dict = get_peft_model_state_dict(unwrapped_model)
            # Adjust layer names to standard formatting.
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            LTXConditionPipeline.save_lora_weights(
                save_directory=save_dir,
                transformer_lora_layers=state_dict,
                weight_name=filename,
            )
            logger.info(f"💾 LoRA checkpoint saved: step {self._global_step}")
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        # Keep track of checkpoint paths, and cleanup old checkpoints if needed
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()

        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            # Update the list to only contain kept checkpoints
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info("💾 Training configuration saved")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
    
    def _save_intermediate_validation_steps(self, intermediate_latents: list, prompt: str, prompt_idx: int) -> None:
        """Save intermediate validation steps as videos for ODE visualization."""
        try:
            from pathlib import Path
            
            # Create intermediate steps directory
            intermediate_dir = Path(self._config.output_dir) / "intermediate_steps"
            intermediate_dir.mkdir(exist_ok=True, parents=True)
            
            # Move VAE to device temporarily
            self._vae.to(self._accelerator.device)
            
            for step_data in intermediate_latents:
                step_num = step_data['step']
                timestep = step_data['timestep']
                latents = step_data['latents'].to(self._accelerator.device)
                
                # Decode latents to video frames
                with autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                    # Simple decode using VAE
                    decoded = self._vae.decode(latents)[0]
                    
                    # Convert to video format (B, C, F, H, W) -> (B, F, H, W, C)
                    decoded = decoded.permute(0, 2, 3, 4, 1).cpu()
                    decoded = decoded.clamp(-1, 1).add(1).div(2)  # [-1,1] -> [0,1]
                    
                    # Convert to list of PIL images
                    frames = []
                    for b in range(decoded.shape[0]):
                        for f in range(decoded.shape[1]):
                            frame = decoded[b, f].numpy()
                            frame = (frame * 255).astype('uint8')
                            frames.append(PIL.Image.fromarray(frame))
                    
                    # Save as video
                    video_path = intermediate_dir / f"step_{self._global_step:06d}_prompt_{prompt_idx}_ode_step_{step_num:02d}_t{timestep:.3f}.mp4"
                    export_to_video(frames, str(video_path), fps=24)
            
            # Move VAE back to CPU
            self._vae.to("cpu")
            
            logger.info(f"🎬 Saved {len(intermediate_latents)} intermediate ODE steps for validation")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate validation steps: {e}")
            try:
                self._vae.to("cpu")
            except:
                pass

    def _log_validation_videos(self, video_paths: list[Path], prompts: list[str]) -> None:
        """Log validation videos to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Create lists of videos with their captions
        validation_videos = [
            wandb.Video(str(video_path), caption=prompt, format="mp4")
            for video_path, prompt in zip(video_paths, prompts, strict=False)
        ]

        # Log all videos at once
        self._wandb_run.log(
            {
                "validation_videos": validation_videos,
            },
            step=self._global_step,
        )
