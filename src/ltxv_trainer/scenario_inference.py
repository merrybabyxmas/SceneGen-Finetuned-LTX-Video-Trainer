"""
Scenario-based inference for video generation with stable and transition tokens.

Supports parsing scenario strings and generating videos with appropriate token placement
for controlling intra-shot continuity (stable tokens) vs inter-shot transitions.
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union
import logging
import re

from ltxv_trainer.token_utils import (
    VideoTokenEmbeddings,
    insert_stable_tokens,
    insert_transition_token
)

logger = logging.getLogger(__name__)


class ScenarioVideoGenerator:
    """
    Generator for scenario-based video generation with token control.
    """

    def __init__(
        self,
        token_embeddings: VideoTokenEmbeddings,
        transformer_model,
        device: torch.device
    ):
        """
        Initialize the scenario video generator.

        Args:
            token_embeddings: Video token embeddings with stable and transition tokens
            transformer_model: The transformer model for video generation
            device: Device for computation
        """
        self.token_embeddings = token_embeddings
        self.transformer_model = transformer_model
        self.device = device

        logger.info(f"🎬 ScenarioVideoGenerator initialized on device: {device}")

    def parse_scenario(self, scenario_string: str) -> List[str]:
        """
        Parse scenario string into tokens.

        Args:
            scenario_string: Comma-separated scenario like "shot1,stable,shot2,transition,shot3"

        Returns:
            List of tokens: ["shot1", "stable", "shot2", "transition", "shot3"]
        """
        # Clean and split the scenario string
        tokens = [token.strip() for token in scenario_string.split(',')]

        # Validate token structure
        if len(tokens) < 1:
            raise ValueError("Scenario must contain at least one shot")

        # Check for valid alternating pattern: shot, [connector], shot, [connector], ...
        for i, token in enumerate(tokens):
            if i % 2 == 0:  # Even indices should be shots
                if token in ['stable', 'transition']:
                    raise ValueError(f"Expected shot name at position {i}, got connector '{token}'")
            else:  # Odd indices should be connectors
                if token not in ['stable', 'transition']:
                    logger.warning(f"Unknown connector '{token}' at position {i}, treating as 'stable'")
                    tokens[i] = 'stable'

        logger.info(f"🎭 Parsed scenario: {tokens}")
        return tokens

    def prepare_shot_latents_with_tokens(
        self,
        shot_latents: Tensor,
        shot_name: str
    ) -> Tensor:
        """
        Prepare shot latents by inserting stable tokens within the shot.

        Args:
            shot_latents: Shot latents [F, H, W, D]
            shot_name: Name of the shot for logging

        Returns:
            Shot latents with stable tokens inserted [F', H, W, D]
        """
        stable_token = self.token_embeddings.get_stable_token(self.device)
        shot_with_stables = insert_stable_tokens(shot_latents, stable_token)

        logger.info(f"🔧 Prepared '{shot_name}': {shot_latents.shape} → {shot_with_stables.shape}")
        return shot_with_stables

    def build_scenario_sequence(
        self,
        scenario_tokens: List[str],
        shot_latents_dict: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict]:
        """
        Build the complete latent sequence based on scenario tokens.

        Args:
            scenario_tokens: Parsed scenario tokens
            shot_latents_dict: Dictionary mapping shot names to latent tensors [F, H, W, D]

        Returns:
            Tuple of:
            - combined_latents: Full sequence [Total_F, H, W, D]
            - metadata: Information about the construction
        """
        logger.info(f"🏗️  Building scenario sequence from {len(scenario_tokens)} tokens")

        # Get token embeddings
        stable_token = self.token_embeddings.get_stable_token(self.device)
        transition_token = self.token_embeddings.get_transition_token(self.device)

        sequence_parts = []
        metadata = {
            'shot_ranges': {},
            'stable_token_positions': [],
            'transition_token_positions': [],
            'total_frames': 0
        }

        current_position = 0

        for i, token in enumerate(scenario_tokens):
            if i % 2 == 0:  # Shot token
                shot_name = token
                if shot_name not in shot_latents_dict:
                    raise ValueError(f"Shot '{shot_name}' not found in provided latents")

                # Prepare shot with internal stable tokens
                shot_latents = shot_latents_dict[shot_name]
                shot_with_stables = self.prepare_shot_latents_with_tokens(shot_latents, shot_name)

                # Record shot range in final sequence
                shot_frames = shot_with_stables.shape[0]
                metadata['shot_ranges'][shot_name] = (current_position, current_position + shot_frames)

                sequence_parts.append(shot_with_stables)
                current_position += shot_frames

                logger.info(f"🎬 Added shot '{shot_name}' at frames [{metadata['shot_ranges'][shot_name][0]}:{metadata['shot_ranges'][shot_name][1]}]")

            else:  # Connector token
                connector = token
                F, H, W, D = sequence_parts[-1].shape[1:]  # Get spatial dims from last shot

                if connector == 'stable':
                    # Insert stable token
                    stable_spatial = stable_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, H, W, D)
                    sequence_parts.append(stable_spatial)
                    metadata['stable_token_positions'].append(current_position)
                    logger.info(f"🔧 Added stable token at position {current_position}")

                elif connector == 'transition':
                    # Insert transition token
                    transition_spatial = transition_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, H, W, D)
                    sequence_parts.append(transition_spatial)
                    metadata['transition_token_positions'].append(current_position)
                    logger.info(f"🔄 Added transition token at position {current_position}")

                current_position += 1

        # Combine all parts
        combined_latents = torch.cat(sequence_parts, dim=0)
        metadata['total_frames'] = combined_latents.shape[0]

        logger.info(f"🏗️  ✅ Scenario sequence built: {combined_latents.shape}")
        logger.info(f"🏗️     Shot ranges: {metadata['shot_ranges']}")
        logger.info(f"🏗️     Stable tokens at: {metadata['stable_token_positions']}")
        logger.info(f"🏗️     Transition tokens at: {metadata['transition_token_positions']}")

        return combined_latents, metadata

    def run_inference(
        self,
        scenario_sequence: Tensor,
        conditioning_mask: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5
    ) -> Tensor:
        """
        Run the transformer inference on the scenario sequence.

        Args:
            scenario_sequence: Combined latent sequence [Total_F, H, W, D]
            conditioning_mask: Optional conditioning mask for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation

        Returns:
            Generated latent sequence [Total_F, H, W, D]
        """
        logger.info(f"🚀 Running inference on sequence: {scenario_sequence.shape}")
        logger.info(f"🚀   Inference steps: {num_inference_steps}, guidance: {guidance_scale}")

        # Add batch dimension if needed
        if scenario_sequence.dim() == 4:
            scenario_sequence = scenario_sequence.unsqueeze(0)  # [1, Total_F, H, W, D]

        B, Total_F, H, W, D = scenario_sequence.shape

        # Convert to transformer input format (flatten spatial dimensions)
        # From [B, Total_F, H, W, D] to [B, Total_F*H*W, D]
        seq_len = Total_F * H * W
        transformer_input = scenario_sequence.view(B, seq_len, D)

        logger.info(f"🚀   Transformer input shape: {transformer_input.shape}")

        # Create conditioning mask if not provided
        if conditioning_mask is None:
            # Default: condition on everything (all frames are "reference")
            conditioning_mask = torch.ones(B, seq_len, dtype=torch.bool, device=self.device)
            logger.info(f"🚀   Created default conditioning mask: all frames conditioned")
        else:
            logger.info(f"🚀   Using provided conditioning mask: {conditioning_mask.shape}")

        # Run transformer inference (this would be replaced with actual model call)
        with torch.no_grad():
            # Placeholder for actual transformer call
            # In practice, this would involve the flow matching denoising process
            generated_latents = self._run_flow_matching_inference(
                transformer_input,
                conditioning_mask,
                num_inference_steps,
                guidance_scale
            )

        # Convert back to spatial format
        generated_sequence = generated_latents.view(B, Total_F, H, W, D)

        # Remove batch dimension if it was added
        if B == 1:
            generated_sequence = generated_sequence.squeeze(0)

        logger.info(f"🚀 ✅ Inference completed: {generated_sequence.shape}")
        return generated_sequence

    def _run_flow_matching_inference(
        self,
        latents: Tensor,
        conditioning_mask: Tensor,
        num_steps: int,
        guidance_scale: float
    ) -> Tensor:
        """
        Placeholder for flow matching inference implementation.

        In practice, this would implement the full flow matching denoising process.
        """
        logger.info(f"🔄 Running flow matching inference (placeholder)")

        # Placeholder: return slightly modified input
        # In real implementation, this would be the denoising loop
        return latents + torch.randn_like(latents) * 0.01

    def generate_from_scenario(
        self,
        scenario_string: str,
        shot_latents_dict: Dict[str, Tensor],
        conditioning_mask: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5
    ) -> Tuple[Tensor, Dict]:
        """
        Complete inference pipeline from scenario string to generated video.

        Args:
            scenario_string: Scenario description like "shot1,stable,shot2,transition,shot3"
            shot_latents_dict: Dictionary mapping shot names to latent tensors [F, H, W, D]
            conditioning_mask: Optional conditioning mask
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale

        Returns:
            Tuple of:
            - generated_sequence: Generated video latents [Total_F, H, W, D]
            - metadata: Information about the generation process
        """
        logger.info(f"🎬 Starting scenario-based generation: '{scenario_string}'")

        # Step 1: Parse scenario
        scenario_tokens = self.parse_scenario(scenario_string)

        # Step 2: Build scenario sequence
        scenario_sequence, metadata = self.build_scenario_sequence(
            scenario_tokens, shot_latents_dict
        )

        # Step 3: Run inference
        generated_sequence = self.run_inference(
            scenario_sequence,
            conditioning_mask,
            num_inference_steps,
            guidance_scale
        )

        # Update metadata with generation info
        metadata.update({
            'scenario_string': scenario_string,
            'scenario_tokens': scenario_tokens,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'generated_shape': generated_sequence.shape
        })

        logger.info(f"🎬 ✅ Scenario generation completed!")
        logger.info(f"🎬    Input scenario: '{scenario_string}'")
        logger.info(f"🎬    Generated shape: {generated_sequence.shape}")
        logger.info(f"🎬    Total frames: {metadata['total_frames']}")

        return generated_sequence, metadata


def create_sample_shot_latents(
    num_shots: int = 3,
    frames_per_shot: int = 8,
    height: int = 56,
    width: int = 96,
    channels: int = 3072,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Tensor]:
    """
    Create sample shot latents for testing.

    Args:
        num_shots: Number of shots to create
        frames_per_shot: Frames per shot
        height: Latent height
        width: Latent width
        channels: Latent channels
        device: Device for tensors

    Returns:
        Dictionary mapping shot names to latent tensors
    """
    shot_latents = {}

    for i in range(num_shots):
        shot_name = f"shot{i+1}"
        # Create random latents with slight variation per shot
        latents = torch.randn(frames_per_shot, height, width, channels, device=device)
        latents = latents * (0.5 + i * 0.1)  # Vary scale per shot
        shot_latents[shot_name] = latents

    logger.info(f"🎭 Created {num_shots} sample shots with {frames_per_shot} frames each")
    return shot_latents


# Example usage and testing functions
def test_scenario_inference():
    """Test the scenario inference pipeline."""
    logger.info("🧪 Testing scenario inference pipeline")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create token embeddings
    token_embeddings = VideoTokenEmbeddings(hidden_dim=3072)

    # Create sample transformer (placeholder)
    class MockTransformer:
        pass
    transformer = MockTransformer()

    # Create generator
    generator = ScenarioVideoGenerator(token_embeddings, transformer, device)

    # Create sample shot latents
    shot_latents = create_sample_shot_latents(num_shots=3, device=device)

    # Test scenarios
    test_scenarios = [
        "shot1,stable,shot2,transition,shot3",
        "shot1,transition,shot2",
        "shot1,stable,shot2,stable,shot3",
        "shot1,transition,shot2,transition,shot3"
    ]

    for scenario in test_scenarios:
        logger.info(f"\n🎬 Testing scenario: '{scenario}'")
        try:
            generated_sequence, metadata = generator.generate_from_scenario(
                scenario, shot_latents
            )
            logger.info(f"✅ Scenario '{scenario}' completed successfully")
            logger.info(f"   Generated shape: {generated_sequence.shape}")
            logger.info(f"   Total frames: {metadata['total_frames']}")
        except Exception as e:
            logger.error(f"❌ Scenario '{scenario}' failed: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_scenario_inference()