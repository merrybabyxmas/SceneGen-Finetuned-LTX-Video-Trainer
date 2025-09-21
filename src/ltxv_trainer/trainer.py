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
import traceback

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
        self._training_strategy = get_training_strategy(self._config.conditioning)
        self._compile_transformer()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._current_epoch = 0
        self._checkpoint_paths = []
        self._init_wandb()

        # Load token embeddings from checkpoint if available
        self._load_token_embeddings_from_checkpoint()

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

            # if cfg.validation.interval and IS_MAIN_PROCESS and not cfg.validation.skip_initial_validation:
            #     sampled_videos_paths = self._sample_videos(sample_progress)
            # self._accelerator.wait_for_everyone()

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
                        # Run unified validation with both T2V and V2V inference
                        scenario = "shot1,transition,shot2"  # Default scenario, can be configurable
                        self.run_unified_validation(batch, scenario)

                        # # Also run original validation for compatibility (if needed)
                        # if cfg.validation.prompts:  # Only if validation prompts are configured
                        #     sampled_videos_paths = self._sample_videos(sample_progress)
                        #     if sampled_videos_paths and self._config.wandb.log_validation_videos:
                        #         self._log_validation_videos(sampled_videos_paths, cfg.validation.prompts)

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
        logger.info(f"Training Start!")
        

        # Use strategy to prepare the training batch
        training_batch = self._training_strategy.prepare_batch(batch, self._timestep_sampler)


        # Use strategy to prepare model inputs
        model_inputs = self._training_strategy.prepare_model_inputs(training_batch)

        # Run transformer forward pass
        model_pred = self._transformer(**model_inputs)[0]
        


        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(model_pred, training_batch)
        logger.info(f"Training End!")
        

        return loss




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

        # Collect transformer parameters
        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]

        # Add token embeddings from training strategy if available
        if hasattr(self._training_strategy, 'token_embeddings') and self._training_strategy.token_embeddings is not None:
            token_params = [p for p in self._training_strategy.token_embeddings.parameters() if p.requires_grad]
            self._trainable_params.extend(token_params)
            if IS_MAIN_PROCESS:
                logger.info(f"🎭 Added {len(token_params)} token embedding parameters to optimizer")
                token_param_count = sum(p.numel() for p in token_params)
                logger.info(f"🎭   Token parameters: {token_param_count:,}")

        if IS_MAIN_PROCESS:
            total_params = sum(p.numel() for p in self._trainable_params)
            logger.info(f"Trainable params (total): {total_params:,}")

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

        # Separate token embeddings from transformer state dict
        token_state_dict = {}
        transformer_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("token_embeddings."):
                # Remove prefix and store in token state dict
                token_key = key[len("token_embeddings."):]
                token_state_dict[token_key] = value
            else:
                transformer_state_dict[key] = value

        # Load transformer weights
        if self._config.model.training_mode == "full":
            transformer.load_state_dict(transformer_state_dict)
        else:  # LoRA mode
            # Adjust layer names to match PEFT format
            transformer_state_dict = {k.replace("transformer.", "", 1): v for k, v in transformer_state_dict.items()}
            transformer_state_dict = {k.replace("lora_A", "lora_A.default", 1): v for k, v in transformer_state_dict.items()}
            transformer_state_dict = {k.replace("lora_B", "lora_B.default", 1): v for k, v in transformer_state_dict.items()}

            # Load LoRA weights and verify all weights were loaded
            _, unexpected_keys = transformer.load_state_dict(transformer_state_dict, strict=False)
            if unexpected_keys:
                raise ValueError(f"Failed to load some LoRA weights: {unexpected_keys}")

        # Store token state dict for later loading (after training strategy is initialized)
        if token_state_dict:
            self._pending_token_state_dict = token_state_dict
            logger.info(f"🎭 Found {len(token_state_dict)} token embedding parameters in checkpoint (will load after strategy init)")
        else:
            self._pending_token_state_dict = None

    def _load_token_embeddings_from_checkpoint(self) -> None:
        """Load token embeddings from checkpoint after training strategy is initialized."""
        if not hasattr(self, '_pending_token_state_dict') or self._pending_token_state_dict is None:
            return

        if not hasattr(self._training_strategy, 'token_embeddings') or self._training_strategy.token_embeddings is None:
            logger.warning("🎭 Token embeddings found in checkpoint but training strategy has no token embeddings")
            return

        try:
            self._training_strategy.token_embeddings.load_state_dict(self._pending_token_state_dict)
            logger.info(f"🎭 ✅ Loaded {len(self._pending_token_state_dict)} token embedding parameters from checkpoint")
        except Exception as e:
            logger.error(f"🎭 ❌ Failed to load token embeddings from checkpoint: {e}")

        # Clean up pending state dict
        self._pending_token_state_dict = None

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

            # Add token embeddings to state dict if available
            if hasattr(self._training_strategy, 'token_embeddings') and self._training_strategy.token_embeddings is not None:
                token_state_dict = self._training_strategy.token_embeddings.state_dict()
                # Prefix token parameters to avoid conflicts
                for key, value in token_state_dict.items():
                    state_dict[f"token_embeddings.{key}"] = value
                logger.info(f"🎭 Added {len(token_state_dict)} token embedding parameters to checkpoint")

            save_file(state_dict, saved_weights_path)
            logger.info(f"💾 Model checkpoint saved: step {self._global_step}")

        elif self._config.model.training_mode == "lora":
            state_dict = get_peft_model_state_dict(unwrapped_model)
            # Adjust layer names to standard formatting.
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            # Add token embeddings to LoRA checkpoint if available
            if hasattr(self._training_strategy, 'token_embeddings') and self._training_strategy.token_embeddings is not None:
                token_state_dict = self._training_strategy.token_embeddings.state_dict()
                # Prefix token parameters to avoid conflicts
                for key, value in token_state_dict.items():
                    state_dict[f"token_embeddings.{key}"] = value
                logger.info(f"🎭 Added {len(token_state_dict)} token embedding parameters to LoRA checkpoint")

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

    def run_unified_validation(self, training_batch, scenario: str = "shot1,transition,shot2"):
        """Run unified validation for both T2V and V2V inference with scenario-driven approach.

        Args:
            training_batch: The current training batch containing prev_conditions and prompt_embeds
            scenario: Scenario string (e.g., "shot1,transition,shot2,stable,shot3")
        """
        if not self._config.validation.interval:
            return

        logger.info("🔥 " + "=" * 80)
        logger.info("🔥 UNIFIED VALIDATION STARTED")
        logger.info("🔥 " + "=" * 80)
        logger.info(f"🔥 [MAIN] Scenario: '{scenario}'")
        logger.info(f"🔥 [MAIN] Step: {self._global_step}")

        # Create validation output directory
        scenario_name = scenario.replace(",", "_")
        validation_dir = Path(self._config.output_dir) / "validation_debug" / scenario_name
        validation_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"🔥 [MAIN] Output directory: {validation_dir}")

        # Get a prompt from config for T2V
        validation_prompt = (
            self._config.validation.prompts[0] if self._config.validation.prompts
            else "a woman walking through a beautiful garden with flowers blooming"
        )
        logger.info(f"🔥 [MAIN] Validation prompt: {validation_prompt}")

        try:
            logger.info("🔥 [MAIN] Starting T2V validation...")
            # Run T2V validation (Text → Video)
            self._run_t2v_validation(scenario, validation_prompt, validation_dir)
            logger.info("🔥 [MAIN] T2V validation completed!")

            logger.info("🔥 [MAIN] Starting V2V validation...")
            self._run_v2v_validation(scenario, training_batch, validation_dir)
            logger.info("🔥 [MAIN] V2V validation completed!")

            logger.info("🔥 [MAIN] ✅ UNIFIED VALIDATION COMPLETED")
            logger.info("🔥 " + "=" * 80)

        except Exception as e:
            logger.error("🔥 [MAIN] ❌ UNIFIED VALIDATION FAILED")
            logger.error(f"🔥 [MAIN] Error: {e}")
            logger.info("🔥 " + "=" * 80)

    def _run_t2v_validation(self, scenario: str, prompt: str, output_dir: Path):
        """Run T2V inference validation (Text → Video).

        SOS token is initialized from pure random noise.
        Saves noisy input and denoised output pairs for each shot.
        """
        logger.info("=" * 80)
        logger.info("🎯 T2V VALIDATION STARTED")
        logger.info("=" * 80)
        logger.info(f"🎯 [T2V] Scenario: {scenario}")
        logger.info(f"🎯 [T2V] Prompt: {prompt}")
        logger.info(f"🎯 [T2V] Output directory: {output_dir}")

        try:
            # Import here to avoid circular import
            from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
            from ltxv_trainer.token_utils import preprocess_with_scenario
            from copy import deepcopy
            import json

            device = self._accelerator.device

            # Create SGMultiShotPipeline for T2V validation
            t2v_pipeline = SGMultiShotPipeline(
                scheduler=deepcopy(self._scheduler),
                vae=self._accelerator.unwrap_model(self._vae),
                text_encoder=self._accelerator.unwrap_model(self._text_encoder),
                tokenizer=self._tokenizer,
                transformer=self._accelerator.unwrap_model(self._transformer),
                sos_token_generator=self._sos_token_generator,
                use_tokens=True,  # Enable token embeddings for scenario processing
                hidden_dim=128,   # Match the transformer hidden dimension
            )

            # **FIX: Transfer trained token embeddings to validation pipeline**
            if hasattr(self._training_strategy, 'token_embeddings') and self._training_strategy.token_embeddings is not None:
                if hasattr(t2v_pipeline, 'token_embeddings') and t2v_pipeline.token_embeddings is not None:
                    # Copy the trained token embeddings state dict
                    trained_token_state = self._training_strategy.token_embeddings.state_dict()
                    t2v_pipeline.token_embeddings.load_state_dict(trained_token_state)
                    logger.info(f"🎯 [T2V] ✅ Transferred {len(trained_token_state)} trained token embeddings to validation pipeline")
                else:
                    logger.warning("🎯 [T2V] ⚠️ Validation pipeline has no token embeddings to load trained weights into")
            else:
                logger.warning("🎯 [T2V] ⚠️ No trained token embeddings found in training strategy")
            t2v_pipeline.set_progress_bar_config(disable=True)

            # Debug logging for device mismatch
            logger.info(f"🎯 [T2V] Device setup: target device = {device}")
            logger.info(f"🎯 [T2V] VAE device: {next(t2v_pipeline.vae.parameters()).device}")
            logger.info(f"🎯 [T2V] Transformer device: {next(t2v_pipeline.transformer.parameters()).device}")
            logger.info(f"🎯 [T2V] Text encoder device: {next(t2v_pipeline.text_encoder.parameters()).device}")

            t2v_pipeline = t2v_pipeline.to(device)

            # Verify after move to device
            logger.info(f"🎯 [T2V] After pipeline.to(device):")
            logger.info(f"🎯 [T2V] VAE device: {next(t2v_pipeline.vae.parameters()).device}")
            logger.info(f"🎯 [T2V] Transformer device: {next(t2v_pipeline.transformer.parameters()).device}")

            t2v_pipeline.enable_debug_capture()

            # Generate random SOS latents as starting point for shot1
            frames, height, width = 17, 448, 768  # Default dimensions
            latent_num_frames = (frames - 1) // 8 + 1
            latent_height = height // 32
            latent_width = width // 32
            total_tokens = latent_num_frames * latent_height * latent_width

            # For T2V: shot1 is always initialized from pure random noise (SOS token)
            sos_latents = torch.randn(
                1, latent_num_frames, latent_height, latent_width,128,
                device=device, dtype=torch.bfloat16
            )
            logger.info(f"🎯 [T2V] Generated random SOS latents for shot1: {sos_latents.shape}")
            logger.info(f"🎯 [T2V] SOS latents device: {sos_latents.device}, dtype: {sos_latents.dtype}")
            # Set shot_latents on the pipeline directly
            shot_latents = {"shot1": sos_latents[0],
                            "shot2": sos_latents[0]}

            pipeline_inputs = {
                "prompt": prompt,
                "negative_prompt": self._config.validation.negative_prompt,
                "height": height,
                "width": width,
                "num_frames": frames,
                "num_videos_per_prompt": 1,
                "num_inference_steps": self._config.validation.inference_steps,
                "guidance_scale": self._config.validation.guidance_scale,
                "generator": torch.Generator(device=device).manual_seed(42),
                "scenario": scenario,
                "shot_latents": shot_latents,
                "validation_type": "t2v_validation",
                "step": self._global_step
            }
            # First run the full scenario for overall processing (keeping original functionality)
            with autocast(device.type, dtype=torch.bfloat16):
                result = t2v_pipeline(**pipeline_inputs)

            # Get debug latents
            debug_latents = t2v_pipeline.get_debug_latents()

            # Save metadata
            metadata = {
                "mode": "T2V",
                "scenario": scenario,
                "prompt": prompt,
                "step": self._global_step,
                "sos_shape": list(sos_latents.shape)
            }

            metadata_path = output_dir / f"t2v_metadata_step_{self._global_step:06d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Parse scenario to understand shot sequence
            shots = [s.strip() for s in scenario.split(',')]
            shot_names = []
            connectors = []

            # Extract shot names and connectors
            for i, item in enumerate(shots):
                if i % 2 == 0:  # Even indices are shot names
                    shot_names.append(item)
                else:  # Odd indices are connectors
                    connectors.append(item)

            # Generate each shot individually to get proper noisy/denoised pairs
            # For scenario "shot1,transition,shot2,stable,shot3" we need:
            # sos_transition_noisyshot0.mp4, sos_transition_denoisedshot0.mp4
            # denoised_shot0_transition_noisyshot1.mp4, denoised_shot0_transition_denoisedshot1.mp4
            # denoised_shot1_stable_noisyshot2.mp4, denoised_shot1_stable_denoisedshot2.mp4

            prev_latents = sos_latents_5d  # Start with batch prev latents
            prev_state = f"shot1"
            curr_latents = curr_noise_5d
            curr_state = f"shot2"


            for shot_idx, shot_name in enumerate(shot_names):
                # Determine connector for this shot
                connector = connectors[shot_idx] if shot_idx < len(connectors) else "stable"

                # Generate filenames with 0-indexed shot numbers (T2V)
                noisy_filename = f"t2v_{prev_state}_{connector}_noisyshot{shot_idx+2}.mp4"
                denoised_filename = f"t2v_{prev_state}_{connector}_denoisedshot{shot_idx+2}.mp4"

                if f"shot{shot_idx+2}" in shot_names:
                    shot_latents_iter = {f"shot{shot_idx+1}": prev_latents[0],
                                        f"shot{shot_idx+2}": curr_latents[0]}
                    
                    # Create individual shot pipeline inputs
                    shot_pipeline_inputs = {
                        "prompt": prompt,
                        "negative_prompt": self._config.validation.negative_prompt,
                        "height": height,
                        "width": width,
                        "num_frames": frames,
                        "num_videos_per_prompt": 1,
                        "num_inference_steps": self._config.validation.inference_steps,
                        "guidance_scale": self._config.validation.guidance_scale,
                        "generator": torch.Generator(device=device).manual_seed(42 + shot_idx),
                        "scenario": f"shot{shot_idx+1}",
                        "validation_type": "t2v_validation",
                        "step": self._global_step
                    }

                    # Run pipeline for this shot
                    with autocast(device.type, dtype=torch.bfloat16):
                        shot_result = t2v_pipeline(**shot_pipeline_inputs)

                    # Get debug latents for this shot
                    shot_debug_latents = t2v_pipeline.get_debug_latents()

                    # Get denoised output for next shot (no video saving)
                    if shot_debug_latents.get('valid_output') is not None:
                        logger.info(f"🎯 [T2V] ✅ Shot {shot_idx} denoised output ready")

                        # Convert packed latents back to 5D format for next shot
                        packed_latents = shot_debug_latents['valid_output']
                        if packed_latents.dim() == 3 and packed_latents.shape[1] >= total_tokens:
                            # Extract only the target video portion (remove reference part if present)
                            if packed_latents.shape[1] > total_tokens:
                                # Remove reference latents from the beginning
                                target_latents = packed_latents[:, -total_tokens:, :]
                            else:
                                target_latents = packed_latents
                            sequence_length = target_latents.shape[1]  # [B, Seq, D]
                            actual_num_frames = sequence_length // (latent_height * latent_width)
                            # Unpack to 5D format with proper error handling
                            try:
                                prev_latents = t2v_pipeline._unpack_latents(
                                    target_latents,
                                    actual_num_frames,
                                    latent_height,
                                    latent_width,
                                    t2v_pipeline.transformer_spatial_patch_size,
                                    t2v_pipeline.transformer_temporal_patch_size,
                                )
                            except Exception as reshape_error:
                                logger.warning(f"🎯 [T2V] ⚠️ Unpack failed {reshape_error}, target_latents: {target_latents.shape}")
                                # Calculate expected dimensions
                                expected_tokens = latent_num_frames * latent_height * latent_width
                                logger.info(f"🎯 [T2V] Expected tokens: {expected_tokens}, actual: {target_latents.shape[1]}")
                                # Use SOS latents as fallback
                                prev_latents = sos_latents
                                logger.warning(f"🎯 [T2V] ⚠️ Using SOS latents as fallback due to reshape error")
                            logger.info(f"🎯 [T2V] 🔄 Converted packed latents {packed_latents.shape} to 5D {prev_latents.shape} for next shot")
                        else:
                            # Fallback: use original SOS latents
                            prev_latents = sos_latents
                            logger.warning(f"Failed to convert debug latents to 5D, using SOS latents")

                    # Update prev_state for next iteration
                    prev_state = f"denoised_shot{shot_idx}"

                logger.info(f"✅ T2V validation completed")

                # Return denoised result
                if debug_latents.get('valid_output') is not None:
                    return debug_latents['valid_output']
                else:
                    return None

        except Exception as e:
            logger.error("🎯 [T2V] ❌ VALIDATION FAILED")
            logger.error(f"🎯 [T2V] Error: {e}")
            logger.error(f"🎯 [T2V] Traceback: {traceback.format_exc()}")
            logger.info("🎯 [T2V] " + "=" * 50)
            return None
        finally:
            logger.info("🎯 [T2V] ✅ VALIDATION COMPLETED")
            logger.info("🎯 [T2V] " + "=" * 50)
            

    def _run_v2v_validation(self, scenario: str, training_batch, output_dir: Path):
        """Run V2V inference validation (Video → Video).

        The reference is the prev latents from the actual training batch.
        SOS is initialized with the batch's prev latents.
        """
        logger.info("=" * 80)
        logger.info("🎬 V2V VALIDATION STARTED")
        logger.info("=" * 80)
        logger.info(f"🎬 [V2V] Scenario: {scenario}")
        logger.info(f"🎬 [V2V] Output directory: {output_dir}")

        try:
            # Import here to avoid circular import
            from ltxv_trainer.SG_multishot_pipeline import SGMultiShotPipeline
            from ltxv_trainer.token_utils import preprocess_with_scenario
            from copy import deepcopy
            import json

            device = self._accelerator.device

            # Create SGMultiShotPipeline for V2V validation
            v2v_pipeline = SGMultiShotPipeline(
                scheduler=deepcopy(self._scheduler),
                vae=self._accelerator.unwrap_model(self._vae),
                text_encoder=self._accelerator.unwrap_model(self._text_encoder),
                tokenizer=self._tokenizer,
                transformer=self._accelerator.unwrap_model(self._transformer),
                sos_token_generator=None,  # Will use batch latents instead
                use_tokens=True,  # Enable token embeddings for scenario processing
                hidden_dim=128,   # Match the transformer hidden dimension
            )

            # **FIX: Transfer trained token embeddings to validation pipeline**
            if hasattr(self._training_strategy, 'token_embeddings') and self._training_strategy.token_embeddings is not None:
                if hasattr(v2v_pipeline, 'token_embeddings') and v2v_pipeline.token_embeddings is not None:
                    # Copy the trained token embeddings state dict
                    trained_token_state = self._training_strategy.token_embeddings.state_dict()
                    v2v_pipeline.token_embeddings.load_state_dict(trained_token_state)
                    logger.info(f"🎬 [V2V] ✅ Transferred {len(trained_token_state)} trained token embeddings to validation pipeline")
                else:
                    logger.warning("🎬 [V2V] ⚠️ Validation pipeline has no token embeddings to load trained weights into")
            else:
                logger.warning("🎬 [V2V] ⚠️ No trained token embeddings found in training strategy")

            v2v_pipeline.set_progress_bar_config(disable=True)
            v2v_pipeline = v2v_pipeline.to(device)
            v2v_pipeline.enable_debug_capture()

            # Extract SOS from training batch prev_conditions
            prev_conditions = training_batch.get('prev_conditions') or getattr(training_batch, 'prev_conditions', None)

            if prev_conditions is None:
                logger.warning("🎬 [V2V] ⚠️ No prev_conditions in training batch, skipping validation")
                return

            # Get batch latents
            if torch.is_tensor(prev_conditions):
                batch_prev_latents = prev_conditions[0:1]  # Take first item [1, seq, D]
            elif isinstance(prev_conditions, dict) and 'latents' in prev_conditions:
                batch_prev_latents = prev_conditions['latents'][0:1]
            else:
                logger.warning("🎬 [V2V] ⚠️ Invalid prev_conditions format, skipping validation")
                return

            # Convert batch latents to 5D format for shot1 initialization
            frames, height, width = 17, 448, 768  # Default dimensions
            latent_num_frames = (frames - 1) // 8 + 1
            latent_height = height // 32
            latent_width = width // 32
            vae_channels = 128
            total_tokens = latent_num_frames * latent_height * latent_width

            # Simple extraction as requested
            video_latents = batch_prev_latents[:, :total_tokens]
            # For V2V: shot1 is always initialized from batch's prev latents (not random noise)
            sos_latents_5d = video_latents.transpose(1, 2).reshape(
                1, latent_num_frames, latent_height, latent_width, vae_channels
            ).to(device)
            logger.info(f"🎬 [V2V] Using batch prev latents for shot1: {sos_latents_5d.shape}")
            
            curr_noise_5d = torch.randn(
                1, latent_num_frames, latent_height, latent_width,128,
                device=device, dtype=torch.bfloat16
            )
            shot_latents = {
                "shot1" : sos_latents_5d[0],
                "shot2" : curr_noise_5d[0]
            }
            # Get prompt embeds from training batch
            prompt_embeds = getattr(training_batch, 'prompt_embeds', None)
            prompt_attention_mask = getattr(training_batch, 'prompt_attention_mask', None)

            # Debug logging for prompt_embeds issue
            logger.info(f"🎬 [V2V] Training batch type: {type(training_batch)}")
            if hasattr(training_batch, '__dict__'):
                available_attrs = list(training_batch.__dict__.keys())[:10]
                logger.info(f"🎬 [V2V] Available attributes: {available_attrs}")
            elif isinstance(training_batch, dict):
                available_keys = list(training_batch.keys())[:10]
                logger.info(f"🎬 [V2V] Available dict keys: {available_keys}")
            else:
                logger.info(f"🎬 [V2V] ⚠️ Training batch is neither object nor dict")

            # Check if training_batch is a dict
            if prompt_embeds is None and isinstance(training_batch, dict):
                prompt_embeds = training_batch.get('text_conditions', None)
                prompt_attention_mask = training_batch.get('prompt_attention_mask', None)
                logger.info(f"🎬 [V2V] Found prompt_embeds in dict: {prompt_embeds is not None}")

            # Handle case where prompt_embeds is a dict (extract tensor)
            if isinstance(prompt_embeds, dict):
                logger.info(f"🎬 [V2V] Prompt_embeds is dict with keys: {list(prompt_embeds.keys())}")
                prompt_embeddings = prompt_embeds["prompt_embeds"]
                prompt_attention_mask = prompt_embeds["prompt_attention_mask"]
                

            if prompt_embeds is None:
                logger.warning("🎬 [V2V] ⚠️ No prompt_embeds found in training batch, skipping validation")
                if hasattr(training_batch, 'prompt'):
                    logger.info(f"🎬 [V2V] But found 'prompt' attribute: {getattr(training_batch, 'prompt', 'None')}")
                return


            # Prepare pipeline inputs for V2V
            pipeline_inputs = {
                "prompt_embeds": prompt_embeddings,
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_embeds": None,
                "height": height,
                "width": width,
                "num_frames": frames,
                "num_videos_per_prompt": 1,
                "num_inference_steps": self._config.validation.inference_steps,
                "guidance_scale": self._config.validation.guidance_scale,
                "generator": torch.Generator(device=device).manual_seed(42),
                "scenario": scenario,
                "shot_latents": shot_latents,
                "validation_type": "v2v_validation",
                "step": self._global_step
            }

            # First run the full scenario for overall processing (keeping original functionality)
            with autocast(device.type, dtype=torch.bfloat16):
                # Temporarily bypass check_inputs validation for prompt_embeds
                original_check_inputs = v2v_pipeline.check_inputs
                v2v_pipeline.check_inputs = lambda *args, **kwargs: None
                try:
                    result = v2v_pipeline(**pipeline_inputs)
                finally:
                    v2v_pipeline.check_inputs = original_check_inputs

            # Get debug latents
            debug_latents = v2v_pipeline.get_debug_latents()

            # Get the actual prompt text for metadata (decode from embeddings if possible)
            prompt_text = "training_batch_prompt"
            if hasattr(training_batch, 'prompt') and training_batch.prompt:
                prompt_text = training_batch.prompt[0] if isinstance(training_batch.prompt, (list, tuple)) else str(training_batch.prompt)

            # Save metadata
            metadata = {
                "mode": "V2V",
                "scenario": scenario,
                "prompt": prompt_text,
                "step": self._global_step,
                "batch_latents_shape": list(batch_prev_latents.shape),
                "sos_5d_shape": list(sos_latents_5d.shape)
            }

            metadata_path = output_dir / f"v2v_metadata_step_{self._global_step:06d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Parse scenario to understand shot sequence
            shots = [s.strip() for s in scenario.split(',')]
            shot_names = []
            connectors = []

            # Extract shot names and connectors
            for i, item in enumerate(shots):
                if i % 2 == 0:  # Even indices are shot names
                    shot_names.append(item)
                else:  # Odd indices are connectors
                    connectors.append(item)

            # Generate each shot individually to get proper noisy/denoised pairs
            # For scenario "shot1,transition,shot2,stable,shot3" we need:
            # sos_transition_noisyshot0.mp4, sos_transition_denoisedshot0.mp4
            # denoised_shot0_transition_noisyshot1.mp4, denoised_shot0_transition_denoisedshot1.mp4
            # denoised_shot1_stable_noisyshot2.mp4, denoised_shot1_stable_denoisedshot2.mp4

            prev_latents = sos_latents_5d  # Start with batch prev latents
            prev_state = f"shot1"
            curr_latents = curr_noise_5d
            curr_state = f"shot2"

            for shot_idx, shot_name in enumerate(shot_names):
                # Determine connector for this shot
                connector = connectors[shot_idx] if shot_idx < len(connectors) else "stable"

                # Generate filenames with 0-indexed shot numbers (V2V)
                noisy_filename = f"v2v_{prev_state}_{connector}_noisyshot{shot_idx+2}.mp4"
                denoised_filename = f"v2v_{prev_state}_{connector}_denoisedshot{shot_idx+2}.mp4"

                # Set shot_latents on the pipeline directly for individual shot
                if f"shot{shot_idx+2}" in shot_names:
                    shot_latents_iter = {f"shot{shot_idx+1}": prev_latents[0],
                                         f"shot{shot_idx+2}": curr_latents[0]}
                    
                    # Create individual shot pipeline inputs
                    shot_pipeline_inputs = {
                        "prompt_embeds": prompt_embeds,
                        "prompt_attention_mask": prompt_attention_mask,
                        "negative_prompt_embeds": None,
                        "height": height,
                        "width": width,
                        "num_frames": frames,
                        "num_videos_per_prompt": 1,
                        "num_inference_steps": self._config.validation.inference_steps,
                        "guidance_scale": self._config.validation.guidance_scale,
                        "generator": torch.Generator(device=device).manual_seed(42 + shot_idx),
                        "return_latents": True,
                        "return_dict": True,
                        "scenario": f"shot{shot_idx+1},{connector},shot{shot_idx+2}",
                        "shot_latents": shot_latents_iter,
                        "validation_type": "v2v_validation",
                        "step": self._global_step
                    }

                    # Run pipeline for this shot
                    with autocast(device.type, dtype=torch.bfloat16):
                        # Temporarily bypass check_inputs validation for prompt_embeds
                        original_check_inputs = v2v_pipeline.check_inputs
                        v2v_pipeline.check_inputs = lambda *args, **kwargs: None
                        try:
                            shot_result = v2v_pipeline(**shot_pipeline_inputs)
                            
                        finally:
                            v2v_pipeline.check_inputs = original_check_inputs
                            
                        

                    # Get debug latents for this shot
                    shot_debug_latents = v2v_pipeline.get_debug_latents()

                    # Get denoised output for next shot (no video saving)
                    if shot_debug_latents.get('valid_output') is not None:
                        logger.info(f"✅ V2V shot {shot_idx+2} denoised output ready")

                        # Convert packed latents back to 5D format for next shot
                        packed_latents = shot_debug_latents['valid_output']
                        if packed_latents.dim() == 3 and packed_latents.shape[1] >= total_tokens:
                            # Extract only the target video portion (remove reference part if present)
                            if packed_latents.shape[1] > total_tokens:
                                # Remove reference latents from the beginning
                                target_latents = packed_latents[:, -total_tokens:, :]
                            else:
                                target_latents = packed_latents
                            sequence_length = target_latents.shape[1]  # [B, Seq, D]
                            actual_num_frames = sequence_length // (latent_height * latent_width)
                            # Unpack to 5D format
                            prev_latents = v2v_pipeline._unpack_latents(
                                target_latents,
                                actual_num_frames,
                                latent_height,
                                latent_width,
                                v2v_pipeline.transformer_spatial_patch_size,
                                v2v_pipeline.transformer_temporal_patch_size,
                            )
                            logger.info(f"🔄 V2V: Converted packed latents {packed_latents.shape} to 5D {prev_latents.shape} for next shot")
                        else:
                            # Fallback: use original SOS latents
                            prev_latents = sos_latents_5d
                            logger.warning(f"V2V: Failed to convert debug latents to 5D, using SOS latents")

                    # Update prev_state for next iteration
                    prev_state = f"denoised_shot{shot_idx}"

                logger.info("🎬 [V2V] ✅ VALIDATION COMPLETED")

                # Return denoised result
                if debug_latents.get('valid_output') is not None:
                    return debug_latents['valid_output']
                else:
                    return None

        except Exception as e:
            logger.error("🎬 [V2V] ❌ VALIDATION FAILED")
            logger.error(f"🎬 [V2V] Error: {e}")
            logger.error(f"🎬 [V2V] Traceback: {traceback.format_exc()}")
            logger.info("🎬 [V2V] " + "=" * 50)
            return None
        finally:
            logger.info("🎬 [V2V] " + "=" * 50)

