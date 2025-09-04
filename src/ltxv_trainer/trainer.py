import os  # noqa: I001
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock

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
from ltxv_trainer.SG_datasets import PrecomputedDataset

from ltxv_trainer.hf_hub_utils import push_to_hub
from ltxv_trainer.model_loader import load_ltxv_components
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.SG_validation_pipeline import create_multi_shot_validation_pipeline

from ltxv_trainer.quantization import quantize_model
from ltxv_trainer.timestep_samplers import SAMPLERS
# from ltxv_trainer.training_strategies import get_training_strategy
from ltxv_trainer.SG_training_strategy import get_training_strategy

from ltxv_trainer.utils import get_gpu_memory_gb, open_image_as_srgb, convert_checkpoint
from ltxv_trainer.video_utils import read_video


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

                        # Log essential metrics to W&B only every 10 steps
                        if self._global_step % 10 == 0:
                            self._log_metrics(
                                {
                                    "train/loss": loss.item(),
                                    "train/learning_rate": current_lr,
                                    "train/global_step": self._global_step,
                                }
                            )

                        if disable_progress_bars and self._global_step % 50 == 0:
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
        # Use strategy to prepare the training batch
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)


        # Use strategy to prepare model inputs
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)

        # Run transformer forward pass
        model_pred = self._transformer(**model_inputs)[0]

        # Save batch visualization for first 2 batches (only on optimization steps)
        if (IS_MAIN_PROCESS and 
            is_optimization_step and
            hasattr(self._config.debug, 'enable_batch_visualization') and 
            self._config.debug.enable_batch_visualization and 
            self._global_step % self._config.debug.debug_interval == 0):
            self._save_batch_visualization(batch, training_batch, model_pred)

        # Save debug frames with model prediction if enabled in config
        if (IS_MAIN_PROCESS and 
            self._config.debug.enable_debug_frames and 
            self._global_step % self._config.debug.debug_interval == 0):
            self._save_debug_frames(batch, training_batch, model_pred)

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(model_pred, training_batch)

        return loss

    @torch.no_grad()
    def _save_debug_frames(self, batch: dict[str, dict[str, Tensor]], training_batch, model_pred: Tensor = None) -> None:
        """Save debug frames to visualize training progress."""
        try:
            from PIL import Image
            import numpy as np
            
            # Create debug directory
            debug_dir = Path(self._config.output_dir) / "debug_frames"
            debug_dir.mkdir(exist_ok=True, parents=True)
            
            # Use separate debug GPU with availability check
            try:
                debug_gpu_id = self._config.debug.debug_gpu_id
                # Check if debug GPU is available and different from current device
                if debug_gpu_id < torch.cuda.device_count() and debug_gpu_id != self._accelerator.device.index:
                    debug_device = torch.device(f"cuda:{debug_gpu_id}")
                    logger.info(f"🔍 Using separate debug GPU: {debug_device}")
                else:
                    # Fallback to training GPU if debug GPU not available
                    debug_device = self._accelerator.device
                    logger.info(f"🔍 Debug GPU {debug_gpu_id} not available, using training GPU: {debug_device}")
            except Exception as e:
                debug_device = self._accelerator.device
                logger.warning(f"🔍 Debug GPU setup failed, using training GPU: {e}")
            
            # Move VAE to debug device temporarily
            self._vae.to(debug_device)
            
            # Get first sample from batch
            batch_idx = 0
            
            # Extract original clean latents from the raw batch
            curr_lat_dict = batch["latent_conditions"]
            if torch.is_tensor(curr_lat_dict):
                curr_clean_latents = curr_lat_dict[batch_idx:batch_idx+1]  # (1, Seq, D)
            else:
                curr_clean_latents = curr_lat_dict["latents"][batch_idx:batch_idx+1]  # (1, Seq, D)
            
            # Extract previous clean latents if available
            prev_clean_latents = None
            if batch.get("prev_conditions", None) is not None:
                prev_dict = batch["prev_conditions"]
                if not torch.is_tensor(prev_dict):
                    prev_clean_latents = prev_dict["latents"][batch_idx:batch_idx+1]
                else:
                    prev_clean_latents = prev_dict[batch_idx:batch_idx+1]
            
            # Get noisy latents from training batch (first sample)
            # For multi-shot: extract current shot portion
            if hasattr(training_batch, 'latents'):
                full_latents = training_batch.latents[batch_idx:batch_idx+1]  # (1, Total_Seq, D)
                
                if prev_clean_latents is not None:
                    # Extract current shot portion (after previous shot)
                    prev_seq_len = prev_clean_latents.shape[1]
                    curr_noisy_latents = full_latents[:, prev_seq_len:]  # Current shot with noise
                else:
                    curr_noisy_latents = full_latents  # No previous shot, all is current
            else:
                return  # Skip if no latents found
            
            # Get dimensions from training batch
            H, W = training_batch.height, training_batch.width
            frames_per_sample = H * W
            
            # Extract first frame latents (HxW tokens)
            if curr_clean_latents.shape[1] >= frames_per_sample:
                curr_clean_first_frame = curr_clean_latents[:, :frames_per_sample]  # (1, H*W, D)
                curr_noisy_first_frame = curr_noisy_latents[:, :frames_per_sample]  # (1, H*W, D)
                
                # Extract previous first frame if available
                prev_clean_first_frame = None
                if prev_clean_latents is not None and prev_clean_latents.shape[1] >= frames_per_sample:
                    prev_clean_first_frame = prev_clean_latents[:, :frames_per_sample]  # (1, H*W, D)
                
                logger.info(f"Current clean frame shape: {curr_clean_first_frame.shape}")
                if prev_clean_first_frame is not None:
                    logger.info(f"Previous clean frame shape: {prev_clean_first_frame.shape}")
                logger.info(f"Dimensions H={H}, W={W}, frames_per_sample={frames_per_sample}")
                
                # Calculate proper VAE latent dimensions
                # LTX Video VAE uses 32x spatial downsampling, 8x temporal downsampling
                vae_h = H // 1  # Spatial downsampling factor
                vae_w = W // 1  
                vae_channels = 128  # LTX latent channels
                
                logger.info(f"VAE dimensions: {vae_channels}x{vae_h}x{vae_w}")
                
                # Reshape properly: (1, H*W, 128) -> (1, 128, vae_h, vae_w)
                try:
                    # Take only the spatial part (first frame)
                    spatial_tokens = vae_h * vae_w
                    if curr_clean_first_frame.shape[1] >= spatial_tokens:
                        # Current latents
                        curr_clean_spatial = curr_clean_first_frame[:, :spatial_tokens, :]  # (1, spatial_tokens, 128)
                        curr_noisy_spatial = curr_noisy_first_frame[:, :spatial_tokens, :]
                        
                        # Previous latents (if available)
                        prev_clean_spatial = None
                        if prev_clean_first_frame is not None:
                            prev_clean_spatial = prev_clean_first_frame[:, :spatial_tokens, :]
                        
                        # Reshape to proper 5D VAE format: (batch, channels, frames, height, width)
                        vae_f = 1  # Single frame for debugging
                        curr_clean_reshaped = curr_clean_spatial.transpose(1, 2).reshape(1, vae_channels, vae_f, vae_h, vae_w)
                        curr_noisy_reshaped = curr_noisy_spatial.transpose(1, 2).reshape(1, vae_channels, vae_f, vae_h, vae_w)
                        
                        prev_clean_reshaped = None
                        if prev_clean_spatial is not None:
                            prev_clean_reshaped = prev_clean_spatial.transpose(1, 2).reshape(1, vae_channels, vae_f, vae_h, vae_w)

                        logger.info(f"Final VAE input shapes: curr_clean={curr_clean_reshaped.shape}, curr_noisy={curr_noisy_reshaped.shape}")
                        if prev_clean_reshaped is not None:
                            logger.info(f"Previous clean shape: {prev_clean_reshaped.shape}")
                        
                        # Move latents to debug device
                        curr_clean_reshaped = curr_clean_reshaped.to(debug_device)
                        curr_noisy_reshaped = curr_noisy_reshaped.to(debug_device)
                        if prev_clean_reshaped is not None:
                            prev_clean_reshaped = prev_clean_reshaped.to(debug_device)
                    else:
                        logger.warning(f"Not enough spatial tokens: {curr_clean_first_frame.shape[1]} < {spatial_tokens}")
                        return
                except Exception as e:
                    logger.error(f"Failed to reshape latents: {e}")
                    return
                # Decode with VAE on debug device
                with autocast(debug_device.type, dtype=torch.bfloat16):
                    try:
                        # Use timestep tensor for VAE decode
                        timestep = torch.zeros(1, device=debug_device, dtype=torch.long)
                        
                        # Decode current latents
                        curr_clean_result = self._vae.decode(curr_clean_reshaped / self._vae.config.scaling_factor, timestep, return_dict=False)
                        curr_noisy_result = self._vae.decode(curr_noisy_reshaped / self._vae.config.scaling_factor, timestep, return_dict=False)
                        
                        # Decode previous latents if available
                        prev_clean_result = None
                        if prev_clean_reshaped is not None:
                            prev_clean_result = self._vae.decode(prev_clean_reshaped / self._vae.config.scaling_factor, timestep, return_dict=False)
                        
                        # Handle different return formats for current latents
                        if isinstance(curr_clean_result, (list, tuple)):
                            curr_clean_decoded = curr_clean_result[0]
                        else:
                            curr_clean_decoded = curr_clean_result
                            
                        if isinstance(curr_noisy_result, (list, tuple)):
                            curr_noisy_decoded = curr_noisy_result[0] 
                        else:
                            curr_noisy_decoded = curr_noisy_result
                        
                        # Handle previous latents
                        prev_clean_decoded = None
                        if prev_clean_result is not None:
                            if isinstance(prev_clean_result, (list, tuple)):
                                prev_clean_decoded = prev_clean_result[0]
                            else:
                                prev_clean_decoded = prev_clean_result
                        
                        # Compute denoised latents if model prediction is available
                        curr_denoised_decoded = None
                        if model_pred is not None:
                            try:
                                # Get current portion of model prediction
                                if prev_clean_latents is not None:
                                    prev_seq_len = prev_clean_latents.shape[1]
                                    curr_model_pred = model_pred[batch_idx:batch_idx+1, prev_seq_len:]
                                else:
                                    curr_model_pred = model_pred[batch_idx:batch_idx+1]
                                
                                # Extract first frame from model prediction
                                curr_model_pred_first = curr_model_pred[:, :frames_per_sample]  # (1, H*W, D)
                                curr_model_pred_spatial = curr_model_pred_first[:, :spatial_tokens, :]
                                curr_model_pred_reshaped = curr_model_pred_spatial.transpose(1, 2).reshape(1, vae_channels, vae_f, vae_h, vae_w)
                                curr_model_pred_reshaped = curr_model_pred_reshaped.to(debug_device)
                                
                                # Compute denoised latents using flow matching
                                # For flow matching: denoised = noisy - sigma * predicted_velocity
                                sigmas = training_batch.sigmas[batch_idx:batch_idx+1].to(debug_device)  # (1, 1, 1)
                                
                                # Flow matching denoising: x_0 = x_t - sigma * v_pred
                                # where v_pred is the velocity field predicted by the model
                                sigma_reshaped = sigmas.view(-1, 1, 1, 1, 1)  # Shape for broadcasting
                                
                                # Compute denoised latents using flow matching
                                curr_denoised_reshaped = curr_noisy_reshaped - sigma_reshaped * curr_model_pred_reshaped
                                
                                # Decode denoised latents
                                curr_denoised_result = self._vae.decode(curr_denoised_reshaped / self._vae.config.scaling_factor, timestep, return_dict=False)
                                
                                if isinstance(curr_denoised_result, (list, tuple)):
                                    curr_denoised_decoded = curr_denoised_result[0]
                                else:
                                    curr_denoised_decoded = curr_denoised_result
                                    
                            except Exception as e:
                                logger.warning(f"Failed to compute denoised frame: {e}")
                                curr_denoised_decoded = None
                            
                    except Exception as e:
                        logger.warning(f"VAE decode failed: {e}")
                        return
                
                # Convert to images and save
                def tensor_to_image(tensor):
                    # Handle 5D VAE output: (1, 3, 1, H, W) -> (H, W, 3)
                    logger.info(f"tensor_to_image input shape: {tensor.shape}")
                    
                    # Remove batch and frame dimensions: (1, 3, 1, H, W) -> (3, H, W)
                    img = tensor.squeeze(0).squeeze(1) if tensor.dim() == 5 else tensor.squeeze(0)
                    
                    # Normalize and convert: (3, H, W) -> (H, W, 3)
                    img = img.clamp(-1, 1).add(1).div(2)  # [-1,1] -> [0,1]
                    img = img.permute(1, 2, 0).cpu().float().numpy()  # Convert to float32 before numpy
                    img = (img * 255).astype(np.uint8)
                    return Image.fromarray(img)
                
                # Convert current latents to images
                curr_clean_img = tensor_to_image(curr_clean_decoded)
                curr_noisy_img = tensor_to_image(curr_noisy_decoded)
                
                # Convert previous latents to images if available
                prev_clean_img = None
                if prev_clean_decoded is not None:
                    prev_clean_img = tensor_to_image(prev_clean_decoded)
                
                # Convert denoised latents to images if available
                curr_denoised_img = None
                if curr_denoised_decoded is not None:
                    curr_denoised_img = tensor_to_image(curr_denoised_decoded)
                
                # Save images
                step_str = f"{self._global_step:06d}"
                
                # Save current frames
                curr_clean_img.save(debug_dir / f"step_{step_str}_curr_clean_first_frame.png")
                curr_noisy_img.save(debug_dir / f"step_{step_str}_curr_noisy_first_frame.png")
                
                # Save denoised frame if available
                if curr_denoised_img is not None:
                    curr_denoised_img.save(debug_dir / f"step_{step_str}_curr_denoised_first_frame.png")
                
                # Save previous frame if available
                if prev_clean_img is not None:
                    prev_clean_img.save(debug_dir / f"step_{step_str}_prev_clean_first_frame.png")
                
                saved_files = ["curr_clean", "curr_noisy"]
                if curr_denoised_img is not None:
                    saved_files.append("curr_denoised")
                if prev_clean_img is not None:
                    saved_files.append("prev_clean")
                    
                logger.info(f"🔍 Debug frames saved for step {self._global_step}: {', '.join(saved_files)}")
            
            # Move VAE back to CPU
            self._vae.to("cpu")
            
        except Exception as e:
            logger.warning(f"Failed to save debug frames: {e}")
            # Ensure VAE is back on CPU even if error occurs
            try:
                self._vae.to("cpu")
            except:
                pass

    @torch.no_grad()
    def _save_denoised_debug_frame(self, batch: dict[str, dict[str, Tensor]], training_batch, model_pred: Tensor) -> None:
        """Save denoised frame using model prediction."""
        try:
            from PIL import Image
            import numpy as np
            
            debug_dir = Path(self._config.output_dir) / "debug_frames"
            debug_dir.mkdir(exist_ok=True, parents=True)
            
            # Use separate debug GPU
            debug_device = torch.device(f"cuda:{self._config.debug.debug_gpu_id}")
            
            # Move VAE to debug device temporarily
            self._vae.to(debug_device)
            
            batch_idx = 0
            
            # Get current shot portion from noisy latents in training batch
            full_latents = training_batch.latents[batch_idx:batch_idx+1]  # (1, Total_Seq, D)
            
            # Extract current shot portion if there's previous shot
            prev_lat = None
            if batch.get("prev_conditions", None) is not None:
                prev_dict = batch["prev_conditions"]
                if not torch.is_tensor(prev_dict):
                    prev_lat = prev_dict["latents"][batch_idx:batch_idx+1]
                else:
                    prev_lat = prev_dict[batch_idx:batch_idx+1]
            
            if prev_lat is not None:
                prev_seq_len = prev_lat.shape[1]
                noisy_latents = full_latents[:, prev_seq_len:]  # Current shot with noise
                model_pred_curr = model_pred[batch_idx:batch_idx+1, prev_seq_len:]  # Model prediction for current shot
            else:
                noisy_latents = full_latents
                model_pred_curr = model_pred[batch_idx:batch_idx+1]
            
            # Get dimensions
            H, W = training_batch.height, training_batch.width
            frames_per_sample = H * W
            
            if noisy_latents.shape[1] >= frames_per_sample:
                # Extract first frame
                noisy_first_frame = noisy_latents[:, :frames_per_sample]  # (1, H*W, D)
                pred_noise_first_frame = model_pred_curr[:, :frames_per_sample]  # (1, H*W, D)
                
                # Compute denoised latent using flow matching: x_0 = x_t - sigma * v_pred
                # Note: This assumes model predicts velocity field (flow matching parameterization)
                sigma = training_batch.sigmas[batch_idx:batch_idx+1, 0, 0].to(debug_device)  # (1,)
                denoised_first_frame = noisy_first_frame - sigma.view(-1, 1, 1) * pred_noise_first_frame
                
                # Calculate proper VAE latent dimensions
                vae_h = H // 1  # Spatial downsampling factor
                vae_w = W // 1  
                vae_channels = 128  # LTX latent channels
                
                # Reshape properly for VAE
                try:
                    spatial_tokens = vae_h * vae_w
                    if denoised_first_frame.shape[1] >= spatial_tokens:
                        denoised_spatial = denoised_first_frame[:, :spatial_tokens, :]  # (1, spatial_tokens, 128)
                        # Reshape to 5D VAE format
                        vae_f = 1  # Single frame for debugging
                        denoised_reshaped = denoised_spatial.transpose(1, 2).reshape(1, vae_channels, vae_f, vae_h, vae_w)
                        # Move latents to debug device
                        denoised_reshaped = denoised_reshaped.to(debug_device)
                    else:
                        logger.warning(f"Not enough spatial tokens for denoised: {denoised_first_frame.shape[1]} < {spatial_tokens}")
                        return
                except Exception as e:
                    logger.error(f"Failed to reshape denoised latents: {e}")
                    return
                
                # Decode with VAE on debug device
                with autocast(debug_device.type, dtype=torch.bfloat16):
                    try:
                        # Use timestep tensor for VAE decode
                        timestep = torch.zeros(1, device=debug_device, dtype=torch.long)
                        denoised_result = self._vae.decode(denoised_reshaped / self._vae.config.scaling_factor, timestep, return_dict=False)
                        
                        # Handle different return formats
                        if isinstance(denoised_result, (list, tuple)):
                            denoised_decoded = denoised_result[0]
                        else:
                            denoised_decoded = denoised_result
                            
                    except Exception as e:
                        logger.warning(f"VAE decode failed: {e}")
                        return
                
                # Convert to image and save
                def tensor_to_image(tensor):
                    # Handle 5D VAE output: (1, 3, 1, H, W) -> (H, W, 3)
                    logger.info(f"tensor_to_image input shape (denoised): {tensor.shape}")
                    
                    # Remove batch and frame dimensions: (1, 3, 1, H, W) -> (3, H, W)
                    img = tensor.squeeze(0).squeeze(1) if tensor.dim() == 5 else tensor.squeeze(0)
                    
                    # Normalize and convert: (3, H, W) -> (H, W, 3)
                    img = img.clamp(-1, 1).add(1).div(2)  # [-1,1] -> [0,1]
                    img = img.permute(1, 2, 0).cpu().float().numpy()  # Convert to float32 before numpy
                    img = (img * 255).astype(np.uint8)
                    return Image.fromarray(img)
                
                denoised_img = tensor_to_image(denoised_decoded)
                
                # Save image
                step_str = f"{self._global_step:06d}"
                denoised_img.save(debug_dir / f"step_{step_str}_denoised_first_frame.png")
                
            # Move VAE back to CPU
            self._vae.to("cpu")
            
        except Exception as e:
            logger.warning(f"Failed to save denoised debug frame: {e}")
            try:
                self._vae.to("cpu")
            except:
                pass

    @torch.no_grad()
    def _save_batch_visualization(self, batch: dict[str, dict[str, Tensor]], training_batch, model_pred: Tensor = None) -> None:
        """Save batch visualization showing prev/curr latents and transitions for first 2 batches."""
        try:
            from PIL import Image
            import numpy as np
            
            # Create batch visualization directory
            batch_viz_dir = Path(self._config.output_dir) / "batch_visualization"
            batch_viz_dir.mkdir(exist_ok=True, parents=True)
            
            # Use debug GPU
            try:
                debug_gpu_id = self._config.debug.debug_gpu_id
                if debug_gpu_id < torch.cuda.device_count() and debug_gpu_id != self._accelerator.device.index:
                    debug_device = torch.device(f"cuda:{debug_gpu_id}")
                else:
                    debug_device = self._accelerator.device
            except Exception:
                debug_device = self._accelerator.device
            
            # Move VAE to debug device temporarily
            self._vae.to(debug_device)
            
            batch_idx = 0  # Focus on first sample in batch
            
            # Extract original clean latents from raw batch
            curr_lat_dict = batch["latent_conditions"]
            if torch.is_tensor(curr_lat_dict):
                curr_clean_latents = curr_lat_dict[batch_idx:batch_idx+1]  # (1, 1008, 128)
            else:
                curr_clean_latents = curr_lat_dict["latents"][batch_idx:batch_idx+1]  # (1, 1008, 128)
            
            # Extract prev latents (SOS) if available
            prev_clean_latents = None
            if batch.get("prev_conditions", None) is not None:
                prev_dict = batch["prev_conditions"]
                if not torch.is_tensor(prev_dict):
                    prev_clean_latents = prev_dict["latents"][batch_idx:batch_idx+1]  # (1, 1008, 128)
                else:
                    prev_clean_latents = prev_dict[batch_idx:batch_idx+1]  # (1, 1008, 128)
            
            # Get concatenated noisy latents from training batch
            full_latents = training_batch.latents[batch_idx:batch_idx+1]  # (1, 2016, 128) or (1, 1008, 128)
            
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
                        
                        # Decode full video
                        with autocast(debug_device.type, dtype=torch.bfloat16):
                            timestep = torch.zeros(1, device=debug_device, dtype=torch.long)
                            result = self._vae.decode(reshaped / self._vae.config.scaling_factor, timestep, return_dict=False)
                            
                            if isinstance(result, (list, tuple)):
                                decoded = result[0]  # (1, 3, F, H, W)
                            else:
                                decoded = result
                        
                        # Convert to video format and save
                        # decoded: (1, 3, F, H, W) -> (F, H, W, 3) for video export
                        video_tensor = decoded.squeeze(0)  # (3, F, H, W)
                        video_tensor = video_tensor.permute(1, 2, 3, 0)  # (F, H, W, 3)
                        video_tensor = video_tensor.clamp(-1, 1).add(1).div(2)  # [-1,1] -> [0,1]
                        
                        # Convert to numpy and scale to [0, 255]
                        video_np = video_tensor.cpu().float().numpy()
                        video_np = (video_np * 255).astype(np.uint8)
                        
                        # Convert to list of PIL Images for video export
                        video_frames = [Image.fromarray(frame) for frame in video_np]
                        
                        # Save as MP4 video
                        step_str = f"epoch_{self._current_epoch:02d}_batch_{step:02d}"
                        shot_prefix = f"{shot_type}_" if shot_type else ""
                        video_path = batch_viz_dir / f"{step_str}_{shot_prefix}{suffix}.mp4"
                        
                        # Use diffusers export_to_video function
                        from diffusers.utils import export_to_video
                        export_to_video(video_frames, str(video_path), fps=24)
                        
                        return True
                        
                    else:
                        logger.warning(f"Not enough tokens for full video: {latents.shape[1]} < {total_tokens}")
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
                # Extract current shot portion from model prediction
                if prev_clean_latents is not None:
                    prev_seq_len = prev_clean_latents.shape[1]
                    model_pred_curr = model_pred[batch_idx:batch_idx+1, prev_seq_len:]  # Current shot prediction
                else:
                    model_pred_curr = model_pred[batch_idx:batch_idx+1]
                
                # Compute denoised latent: noisy_latent - predicted_noise (epsilon parameterization)
                denoised_curr_latents = noisy_curr_latents - model_pred_curr
                
                if decode_and_save_video(denoised_curr_latents, "curr_denoised", step, "curr"):
                    saved_count += 1
            
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

        # Enable gradient checkpointing if requested
        if self._config.optimization.enable_gradient_checkpointing:
            self._transformer.enable_gradient_checkpointing()

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

            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
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
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.get("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),  # First restart cycle length
                T_mult=params.pop("T_mult", 1),  # Multiplicative factor for cycle lengths
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
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

        pipeline = LTXConditionPipeline(
            scheduler=deepcopy(self._scheduler),
            vae=self._accelerator.unwrap_model(self._vae),
            text_encoder=self._accelerator.unwrap_model(self._text_encoder),
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer),
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
                pipeline_inputs["reference_video"] = ref_video

            with autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                result = pipeline(**pipeline_inputs)
                videos = result.frames

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

        logger.info(f"🎥 Validation samples saved for step {self._global_step}")
        return video_paths

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_multi_shot_videos(self, progress: Progress) -> list[Path] | None:
        """Run multi-shot validation by generating sequential video shots from a single prompt."""
        
        # Create multi-shot validation pipeline (use same safe approach as single-shot)
        multi_shot_pipeline = create_multi_shot_validation_pipeline(
            scheduler=deepcopy(self._scheduler),  # Copy scheduler like single-shot validation
            vae=self._accelerator.unwrap_model(self._vae),  # Unwrap like single-shot validation
            text_encoder=self._accelerator.unwrap_model(self._text_encoder),  # Unwrap like single-shot validation
            tokenizer=self._tokenizer,
            transformer=self._accelerator.unwrap_model(self._transformer),  # Unwrap like single-shot validation
            device=self._accelerator.device,
            accelerator=self._accelerator,
            d_model=self._config.validation.sos_latent_dim
        )
        
        # Create output directory for multi-shot samples
        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        all_video_paths = []
        
        # Generate multi-shot sequence for each validation prompt
        for i, prompt in enumerate(self._config.validation.prompts):
            
            try:
                video_paths = multi_shot_pipeline.generate_multi_shot_sequence(
                    prompt=prompt,  # Single prompt for all shots
                    output_dir=output_dir,
                    global_step=self._global_step,
                    video_dims=self._config.validation.video_dims,
                    num_shots=self._config.validation.num_shots,
                    inference_steps=self._config.validation.inference_steps,
                    guidance_scale=self._config.validation.guidance_scale,
                    negative_prompt=self._config.validation.negative_prompt,
                    seed=self._config.validation.seed   # Different seed for each sequence
                )
                
                all_video_paths.extend(video_paths)
                
            except Exception as e:
                logger.error(f"Failed to generate multi-shot sequence for prompt {i+1}: {e}")
                continue
        
        # Move unused components back to CPU
        self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")
        
        if all_video_paths:
            logger.info(f"🎬 Multi-shot validation samples saved for step {self._global_step}")
            return all_video_paths
        else:
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
