#!/usr/bin/env python3

"""
Decode precomputed dataset using trainer's save_batch_visualization method.

This script uses the trainer's visualization functionality to decode precomputed
tensors with the exact same logic as used during training.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import sys
sys.path.append('src')
sys.path.append('.')

from ltxv_trainer.SG_datasets import PrecomputedDataset
from ltxv_trainer.SG_training_strategy import StandardTrainingStrategy, ConditioningConfig
from ltxv_trainer.timestep_samplers import UniformTimestepSampler
from ltxv_trainer.model_loader import load_vae

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Decode precomputed dataset using trainer's visualization method.",
)


class TrainerStyleDecoder:
    """Decoder that uses trainer's save_batch_visualization logic"""
    
    def __init__(self, vae, device: torch.device, output_dir: Path):
        self.vae = vae
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @torch.no_grad()
    def decode_batch_with_trainer_logic(self, batch_data, training_batch, global_step: int):
        """Use trainer's exact decode_and_save_video logic"""
        
        from PIL import Image
        import numpy as np
        
        # Move VAE to device
        self.vae.to(self.device)
        
        batch_idx = 0  # Focus on first sample
        
        # Extract clean latents from raw batch (same as trainer)
        curr_lat_dict = batch_data["latent_conditions"]
        if torch.is_tensor(curr_lat_dict):
            curr_clean_latents = curr_lat_dict[batch_idx:batch_idx+1]
        else:
            curr_clean_latents = curr_lat_dict["latents"][batch_idx:batch_idx+1]
            
        # Extract prev latents if available
        prev_clean_latents = None
        if batch_data.get("prev_conditions", None) is not None:
            prev_dict = batch_data["prev_conditions"]
            if not torch.is_tensor(prev_dict):
                prev_clean_latents = prev_dict["latents"][batch_idx:batch_idx+1]
            else:
                prev_clean_latents = prev_dict[batch_idx:batch_idx+1]
                
        # Get concatenated latents from training batch
        full_latents = training_batch.latents[batch_idx:batch_idx+1]
        
        # Separate prev and curr portions
        if prev_clean_latents is not None:
            prev_seq_len = prev_clean_latents.shape[1]
            noisy_curr_latents = full_latents[:, prev_seq_len:]
            clean_prev_latents_from_batch = full_latents[:, :prev_seq_len]
        else:
            noisy_curr_latents = full_latents
            clean_prev_latents_from_batch = None
            
        vae_channels = 128
        
        # Trainer's decode_and_save_video function (slightly modified for standalone use)
        def decode_and_save_video(latents, suffix, step, shot_type=""):
            try:
                # Calculate total tokens using training batch metadata
                total_tokens = training_batch.num_frames * training_batch.height * training_batch.width
                
                if latents.shape[1] >= total_tokens:
                    # Extract full video latents
                    video_latents = latents[:, :total_tokens]
                    
                    # Reshape to 5D VAE format
                    batch_F = training_batch.num_frames
                    batch_H = training_batch.height
                    batch_W = training_batch.width
                    reshaped = video_latents.transpose(1, 2).reshape(1, vae_channels, batch_F, batch_H, batch_W)
                    reshaped = reshaped.to(self.device)
                    
                    # Decode using trainer's exact method (with the noise issue)
                    device_type = self.device.type if self.device.type != "cuda" else "cuda"
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        timestep = torch.zeros(1, device=self.device, dtype=torch.long)
                        result = self.vae.decode(reshaped / self.vae.config.scaling_factor, timestep, return_dict=False)
                        
                        if isinstance(result, (list, tuple)):
                            decoded = result[0]  # (1, 3, F, H, W)
                        else:
                            decoded = result
                            
                    console.print(f"[yellow]Decoded shape: {decoded.shape}[/yellow]")
                    
                    # Convert to video format (trainer's exact method)
                    video_tensor = decoded.squeeze(0)  # (3, F, H, W)
                    video_tensor = video_tensor.permute(1, 2, 3, 0)  # (F, H, W, 3)
                    video_tensor = video_tensor.clamp(-1, 1).add(1).div(2)  # [-1,1] -> [0,1]
                    
                    # Convert to numpy and scale to [0, 255]
                    video_np = video_tensor.cpu().float().numpy()
                    video_np = (video_np * 255).astype(np.uint8)
                    
                    # Convert to list of PIL Images
                    video_frames = [Image.fromarray(frame) for frame in video_np]
                    
                    # Save as MP4 video
                    step_str = f"step_{step:04d}"
                    shot_prefix = f"{shot_type}_" if shot_type else ""
                    video_path = self.output_dir / f"{step_str}_{shot_prefix}{suffix}.mp4"
                    
                    # Use diffusers export_to_video
                    from diffusers.utils import export_to_video
                    export_to_video(video_frames, str(video_path), fps=24)
                    
                    console.print(f"[green]Saved: {video_path.name}[/green]")
                    return True
                    
                else:
                    console.print(f"[yellow]Not enough tokens: {latents.shape[1]} < {total_tokens}[/yellow]")
                    return False
                    
            except Exception as e:
                console.print(f"[red]Failed to decode {suffix}: {e}[/red]")
                return False
                
        # Save visualizations (same as trainer)
        saved_count = 0
        
        # 1. Save prev latents (clean)
        if prev_clean_latents is not None:
            shot_type = "SOS" if global_step == 1 else "prev"
            if decode_and_save_video(prev_clean_latents, "prev_clean", global_step, shot_type):
                saved_count += 1
                
            # Also save prev from training batch
            if decode_and_save_video(clean_prev_latents_from_batch, "prev_from_batch", global_step, shot_type):
                saved_count += 1
                
        # 2. Save curr latents (clean)
        if decode_and_save_video(curr_clean_latents, "curr_clean", global_step, "curr"):
            saved_count += 1
            
        # 3. Save curr latents (noisy)
        if decode_and_save_video(noisy_curr_latents, "curr_noisy", global_step, "curr"):
            saved_count += 1
            
        console.print(f"[blue]Saved {saved_count} videos for step {global_step}[/blue]")
        
        # Move VAE back to CPU
        self.vae.to("cpu")


@app.command()
def main(
    data_root: str = typer.Option(
        ...,
        help="Root directory containing precomputed data",
    ),
    output_dir: str = typer.Option(
        ...,
        help="Directory to save decoded videos",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size (should be 1 for proper visualization)",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    model_source: str = typer.Option(
        default="LTXV_2B_0.9.5",
        help="Model source for VAE decoder",
    ),
    max_batches: int = typer.Option(
        default=5,
        help="Maximum number of batches to process",
    ),
) -> None:
    """Decode precomputed data using trainer's visualization method.
    
    This reproduces the exact same decoding logic used in trainer validation,
    including any noise artifacts that may occur.
    """
    
    data_root_path = Path(data_root)
    output_path = Path(output_dir)
    
    if not data_root_path.exists():
        raise typer.BadParameter(f"Data root does not exist: {data_root_path}")
        
    device_obj = torch.device(device)
    
    # Load dataset
    console.print(f"[bold]Loading PrecomputedDataset from {data_root_path}...[/bold]")
    dataset = PrecomputedDataset(str(data_root_path))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load VAE
    console.print(f"[bold]Loading VAE ({model_source})...[/bold]")
    vae = load_vae(model_source, dtype=torch.bfloat16)
    
    # Setup training strategy for batch preparation
    conditioning_config = ConditioningConfig(mode="none", first_frame_conditioning_p=0.5)
    strategy = StandardTrainingStrategy(conditioning_config)
    timestep_sampler = UniformTimestepSampler(min_value=0.0, max_value=1.0)
    
    # Create decoder
    decoder = TrainerStyleDecoder(vae, device_obj, output_path)
    
    # Process batches
    total_batches = min(max_batches, len(dataloader))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Decoding with trainer logic", total=total_batches)
        
        for batch_idx, raw_batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            try:
                # Prepare training batch (same as trainer)
                training_batch = strategy.prepare_batch(raw_batch, timestep_sampler)
                
                # Decode using trainer's visualization logic
                decoder.decode_batch_with_trainer_logic(raw_batch, training_batch, batch_idx + 1)
                
            except Exception as e:
                console.print(f"[red]Error processing batch {batch_idx}: {e}[/red]")
                continue
                
            progress.advance(task)
    
    console.print(f"[bold green]Decoding complete! Videos saved to {output_path}[/bold green]")


if __name__ == "__main__":
    app()