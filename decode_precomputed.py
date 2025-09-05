#!/usr/bin/env python3

"""
Decode precomputed dataset tensors back to videos using existing decoder.

This script loads tensors from PrecomputedDataset and decodes them back to videos
using the VAE decoder from the existing decode_latents implementation.

Basic usage:
    python decode_precomputed.py --data-root /path/to/precomputed/data --output-dir /path/to/output
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

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Decode precomputed dataset tensors back to videos.",
)


# Instead of importing LatentsDecoder, we'll create our own decoder class
from ltxv_trainer.model_loader import load_vae
from ltxv_trainer.ltxv_utils import decode_video

class SimpleVAEDecoder:
    def __init__(self, model_source: str, device: str = "cuda", vae_tiling: bool = False):
        self.device = torch.device(device)
        with console.status(f"[bold]Loading VAE model from {model_source}...", spinner="dots"):
            self.vae = load_vae(model_source, dtype=torch.bfloat16).to(self.device)
            if vae_tiling:
                self.vae.enable_tiling()

def _concat_prev_curr_latents(prev_lat: torch.Tensor | None, curr_lat: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """
    Concatenate prev and current shot latents along sequence dimension.
    Args:
        prev_lat: (B, Pseq, D) or None
        curr_lat: (B, Cseq, D)
    Returns:
        (concat_latents, prev_seq_len, curr_seq_len)
    """
    if prev_lat is None:
        return curr_lat, 0, curr_lat.shape[1]
    console.print(f"prev lat shape : {prev_lat.shape}")
    return torch.cat([prev_lat, curr_lat], dim=0), prev_lat.shape[1], curr_lat.shape[1]


@torch.inference_mode()
def decode_precomputed_batch(decoder: SimpleVAEDecoder, batch_data, output_dir: Path, batch_idx: int, 
                           concatenate_shots: bool = False):
    """
    Decode a batch of precomputed data.
    Args:
        decoder: VAE decoder
        batch_data: Batch data from PrecomputedDataset
        output_dir: Output directory
        batch_idx: Batch index for naming
        concatenate_shots: If True, concatenate prev + current shots for full video
    """
    batch_size = len(batch_data["idx"])
    
    for i in range(batch_size):
        # Extract single sample data
        idx = batch_data["idx"][i].item()
        
        # Get current shot latent conditions
        curr_latent_data = {}
        if "latent_conditions" in batch_data:
            latent_conditions = batch_data["latent_conditions"]
            curr_latent_data["latents"] = latent_conditions["latents"][i]
            curr_latent_data["num_frames"] = latent_conditions["num_frames"][i].item()
            curr_latent_data["height"] = latent_conditions["height"][i].item() 
            curr_latent_data["width"] = latent_conditions["width"][i].item()
            
            # Add fps if available
            if "fps" in latent_conditions:
                curr_latent_data["fps"] = latent_conditions["fps"][i].item()
            else:
                curr_latent_data["fps"] = 24  # default fps
        else:
            console.print(f"[yellow]Warning: No latent_conditions found for sample {idx}[/yellow]")
            continue
            
        # Handle prev shot if concatenation is requested
        if concatenate_shots and "prev_conditions" in batch_data:
            prev_conditions = batch_data["prev_conditions"]
            
            # Check if prev_conditions is a dict with latents or direct tensor
            if isinstance(prev_conditions, dict) and "latents" in prev_conditions:
                prev_latents = prev_conditions["latents"][i]
            else:
                # Assume prev_conditions is the latent tensor directly
                prev_latents = prev_conditions[i]
                
            curr_latents = curr_latent_data["latents"]
            
            # Debug: Print tensor shapes
            console.print(f"[yellow]Debug - prev_latents shape: {prev_latents.shape}[/yellow]")
            console.print(f"[yellow]Debug - curr_latents shape: {curr_latents.shape}[/yellow]")
            
            # Ensure both tensors have same number of dimensions
            if prev_latents.dim() != curr_latents.dim():
                if prev_latents.dim() == 2 and curr_latents.dim() == 3:
                    # Add batch dimension to prev_latents
                    prev_latents = prev_latents.unsqueeze(0)
                    console.print(f"[yellow]Added batch dim to prev_latents: {prev_latents.shape}[/yellow]")
                elif prev_latents.dim() == 3 and curr_latents.dim() == 2:
                    # Add batch dimension to curr_latents
                    curr_latents = curr_latents.unsqueeze(0)
                    console.print(f"[yellow]Added batch dim to curr_latents: {curr_latents.shape}[/yellow]")
            
            # Concatenate prev + current shots
            concat_latents, prev_seq_len, curr_seq_len = _concat_prev_curr_latents(prev_latents, curr_latents)
            
            # If we have batch dimension, remove it for decode_video
            if concat_latents.dim() == 3:
                decode_latents = concat_latents[0]  # Take first (and only) batch item
            else:
                decode_latents = concat_latents
            
            # Update metadata for concatenated video
            total_frames = curr_latent_data["num_frames"] * 2  # Assuming prev and curr have same frame count
            output_filename = f"concat_prev_curr_{batch_idx:04d}_{i:02d}_idx_{idx}.mp4"
            
            console.print(f"[blue]Concatenating prev ({prev_seq_len}) + curr ({curr_seq_len}) shots[/blue]")
            console.print(f"[blue]Final decode_latents shape: {decode_latents.shape}[/blue]")
            
        else:
            # Decode only current shot
            decode_latents = curr_latent_data["latents"]
            total_frames = curr_latent_data["num_frames"]
            output_filename = f"decoded_current_{batch_idx:04d}_{i:02d}_idx_{idx}.mp4"
            
        output_path = output_dir / output_filename
        
        try:
            # Use the proper decode_video function
            video = decode_video(
                vae=decoder.vae,
                latents=decode_latents,
                num_frames=total_frames,
                height=curr_latent_data["height"],
                width=curr_latent_data["width"],
                device=decoder.device,
                patch_size=1,
                patch_size_t=1,
                generator=None,
            )
            
            video = video[0]  # Remove batch dimension
            
            # Convert to uint8 for saving
            video = (video * 255).round().clamp(0, 255).to(torch.uint8)
            video = video.permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]
            
            # Save as MP4
            import torchvision
            from fractions import Fraction
            
            fps = curr_latent_data.get("fps", 24)
            torchvision.io.write_video(
                str(output_path),
                video.cpu(),
                fps=Fraction(fps).limit_denominator(1000),
                video_codec="h264",
                options={"crf": "18"},
            )
            
            console.print(f"[green]Saved: {output_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error processing sample {idx}: {e}[/red]")
            continue


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
        default=4,
        help="Batch size for processing",
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
        default=None,
        help="Maximum number of batches to process (for testing)",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    concatenate_shots: bool = typer.Option(
        default=False,
        help="Concatenate previous and current shots for full video decode",
    ),
) -> None:
    """Decode precomputed dataset tensors back to videos.
    
    This script loads the PrecomputedDataset and decodes the stored tensors
    back to video files using the VAE decoder.
    
    Examples:
        # Basic usage - decode current shots only
        python decode_precomputed.py --data-root /path/to/precomputed --output-dir /path/to/videos
        
        # Decode concatenated prev + current shots for full videos
        python decode_precomputed.py --data-root /path/to/precomputed --output-dir /path/to/videos --concatenate-shots
        
        # Process only first 5 batches for testing
        python decode_precomputed.py --data-root /path/to/precomputed --output-dir /path/to/videos --max-batches 5
    """
    data_root_path = Path(data_root)
    output_path = Path(output_dir)
    
    if not data_root_path.exists():
        raise typer.BadParameter(f"Data root directory does not exist: {data_root_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    console.print(f"[bold]Loading PrecomputedDataset from {data_root_path}...[/bold]")
    try:
        dataset = PrecomputedDataset(str(data_root_path))
        console.print(f"[green]Dataset loaded with {len(dataset)} samples[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        return
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep original order for debugging
        num_workers=0   # Single process to avoid multiprocessing issues
    )
    
    # Initialize decoder
    console.print(f"[bold]Loading VAE decoder ({model_source})...[/bold]")
    try:
        decoder = SimpleVAEDecoder(model_source=model_source, device=device, vae_tiling=vae_tiling)
        console.print("[green]VAE decoder loaded successfully[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load decoder: {e}[/red]")
        return
    
    # Process batches
    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Decoding batches", total=total_batches)
        
        for batch_idx, batch_data in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                decode_precomputed_batch(decoder, batch_data, output_path, batch_idx, concatenate_shots)
            except Exception as e:
                console.print(f"[red]Error processing batch {batch_idx}: {e}[/red]")
                continue
            
            progress.advance(task)
    
    console.print(f"[bold green]Decoding complete! Videos saved to {output_path}[/bold green]")


if __name__ == "__main__":
    app()