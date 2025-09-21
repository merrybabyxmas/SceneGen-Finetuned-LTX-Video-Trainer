"""
Single Prompt Multi-View Video Generation

Generates multiple camera angles/viewpoints from a single text prompt using
scenario strings to control camera movements and view transitions.
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import logging
import re

from ltxv_trainer.token_utils import (
    VideoTokenEmbeddings,
    preprocess_multishot_with_tokens,
    create_token_aware_conditioning_mask
)

logger = logging.getLogger(__name__)


class SinglePromptMultiViewGenerator:
    """
    Generates multiple camera views/angles from a single text prompt.
    """

    def __init__(
        self,
        token_embeddings: VideoTokenEmbeddings,
        transformer_model,
        device: torch.device
    ):
        """
        Initialize the single prompt multi-view generator.

        Args:
            token_embeddings: Video token embeddings with stable and transition tokens
            transformer_model: The transformer model for video generation
            device: Device for computation
        """
        self.token_embeddings = token_embeddings
        self.transformer_model = transformer_model
        self.device = device

        # Define view/angle templates
        self.view_templates = {
            'wide': "wide shot of {prompt}",
            'medium': "medium shot of {prompt}",
            'close': "close-up of {prompt}",
            'extreme_close': "extreme close-up of {prompt}",
            'bird_eye': "bird's eye view of {prompt}",
            'low_angle': "low angle shot of {prompt}",
            'high_angle': "high angle shot of {prompt}",
            'side': "side view of {prompt}",
            'behind': "view from behind of {prompt}",
            'front': "front view of {prompt}",
            'tracking': "tracking shot following {prompt}",
            'panning': "panning shot across {prompt}",
            'dolly': "dolly shot moving towards {prompt}",
            'overhead': "overhead shot of {prompt}",
            'dutch': "dutch angle shot of {prompt}"
        }

        logger.info(f"🎬 SinglePromptMultiViewGenerator initialized with {len(self.view_templates)} view types")

    def parse_multiview_scenario(self, scenario_string: str, base_prompt: str) -> Tuple[List[str], List[str]]:
        """
        Parse multi-view scenario string and generate view-specific prompts.

        Args:
            scenario_string: Scenario like "wide,stable,close,transition,medium"
            base_prompt: Base prompt like "woman walking in garden"

        Returns:
            Tuple of (view_tokens, view_prompts)
        """
        logger.info(f"🎭 Parsing multi-view scenario: '{scenario_string}' with base prompt: '{base_prompt}'")

        # Split scenario into tokens
        tokens = [token.strip() for token in scenario_string.split(',')]

        view_tokens = []
        view_prompts = []

        for i, token in enumerate(tokens):
            if i % 2 == 0:  # Even indices should be view types
                view_type = token.lower()

                # Check if it's a known view type
                if view_type in self.view_templates:
                    view_prompt = self.view_templates[view_type].format(prompt=base_prompt)
                else:
                    # If not a known view type, treat as custom view description
                    view_prompt = f"{token} of {base_prompt}"
                    logger.warning(f"Unknown view type '{view_type}', using custom format")

                view_tokens.append(token)
                view_prompts.append(view_prompt)
                logger.info(f"🎥 View {len(view_prompts)}: '{token}' → '{view_prompt}'")

            else:  # Odd indices should be connectors (stable/transition)
                if token not in ['stable', 'transition']:
                    logger.warning(f"Unknown connector '{token}' at position {i}, treating as 'stable'")
                    tokens[i] = 'stable'
                view_tokens.append(token)

        logger.info(f"🎭 Generated {len(view_prompts)} view-specific prompts from base prompt")
        return view_tokens, view_prompts

    def create_multiview_latents(
        self,
        view_prompts: List[str],
        height: int = 56,
        width: int = 96,
        num_frames: int = 8,
        channels: int = 3072
    ) -> Dict[str, Tensor]:
        """
        Create latent representations for each view.

        Args:
            view_prompts: List of view-specific prompts
            height: Latent height
            width: Latent width
            num_frames: Frames per view
            channels: Latent channels

        Returns:
            Dictionary mapping view names to latent tensors
        """
        logger.info(f"🎬 Creating multi-view latents for {len(view_prompts)} views")

        view_latents = {}
        for i, prompt in enumerate(view_prompts):
            view_name = f"view{i+1}"

            # Create latents (in practice, these would come from encoded reference videos or text conditioning)
            latents = torch.randn(
                num_frames, height, width, channels,
                device=self.device, dtype=torch.float32
            )

            # Add some variation based on view index to make views distinct
            variation_factor = 0.1 + (i * 0.05)
            latents = latents * variation_factor

            view_latents[view_name] = latents
            logger.info(f"🎥 Created latents for {view_name}: {latents.shape} (variation: {variation_factor:.2f})")

        return view_latents

    def build_multiview_sequence(
        self,
        view_tokens: List[str],
        view_latents: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict]:
        """
        Build shot-adapter style multi-view sequence with tokens.

        Args:
            view_tokens: Parsed view tokens including connectors
            view_latents: Dictionary mapping view names to latent tensors

        Returns:
            Tuple of (combined_sequence, metadata)
        """
        logger.info(f"🎭 Building shot-adapter multi-view sequence from {len(view_tokens)} tokens")

        # For shot-adapter, we build pairwise sequences: prev_view → transition → curr_view
        view_names = [view_tokens[i] for i in range(0, len(view_tokens), 2)]
        connectors = [view_tokens[i] for i in range(1, len(view_tokens), 2)]

        logger.info(f"🎭 Views: {view_names}")
        logger.info(f"🎭 Connectors: {connectors}")

        if len(view_names) < 2:
            raise ValueError("Shot-adapter requires at least 2 views")

        # Build sequence using shot-adapter preprocessing
        from ltxv_trainer.token_utils import preprocess_shot_adapter_with_tokens

        final_sequences = []
        all_metadata = []

        for i in range(len(view_names) - 1):
            prev_view_name = f"view{i+1}"
            curr_view_name = f"view{i+2}"

            prev_latents = view_latents[prev_view_name]
            curr_latents = view_latents[curr_view_name]

            logger.info(f"🎭 Processing view pair: {prev_view_name} → {curr_view_name}")
            logger.info(f"🎭   Prev: {prev_latents.shape}, Curr: {curr_latents.shape}")

            # Apply shot-adapter token preprocessing
            tokenized_sequence, pair_metadata = preprocess_shot_adapter_with_tokens(
                prev_shot_latents=prev_latents,
                curr_shot_latents=curr_latents,
                token_embeddings=self.token_embeddings,
                device=self.device
            )

            # For multi-view, we only keep the current part (since previous is conditioning)
            # But for full sequence generation, we keep everything
            final_sequences.append(tokenized_sequence)
            all_metadata.append(pair_metadata)

        # For now, return the last sequence (final view transition)
        # In practice, you might want to chain multiple transitions
        combined_sequence = final_sequences[-1]
        metadata = all_metadata[-1]

        # Update metadata for multi-view context
        metadata.update({
            'view_count': len(view_names),
            'view_names': view_names,
            'connectors': connectors,
            'shot_adapter_pairs': len(final_sequences)
        })

        logger.info(f"🎭 ✅ Shot-adapter multi-view sequence built: {combined_sequence.shape}")
        logger.info(f"🎭     Views: {metadata['view_count']}, Total frames: {metadata['total_frames']}")
        logger.info(f"🎭     Shot-adapter pairs: {metadata['shot_adapter_pairs']}")

        return combined_sequence, metadata

    def generate_multiview_video(
        self,
        scenario_string: str,
        base_prompt: str,
        height: int = 56,
        width: int = 96,
        num_frames: int = 8,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5
    ) -> Tuple[Tensor, Dict]:
        """
        Generate multi-view video from single prompt and scenario.

        Args:
            scenario_string: View scenario like "wide,stable,close,transition,medium"
            base_prompt: Base text prompt like "woman walking in garden"
            height: Latent height
            width: Latent width
            num_frames: Frames per view
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale

        Returns:
            Tuple of (generated_sequence, metadata)
        """
        logger.info(f"🎬 Starting single prompt multi-view generation")
        logger.info(f"🎬   Base prompt: '{base_prompt}'")
        logger.info(f"🎬   View scenario: '{scenario_string}'")

        # Step 1: Parse scenario and generate view-specific prompts
        view_tokens, view_prompts = self.parse_multiview_scenario(scenario_string, base_prompt)

        # Step 2: Create latents for each view
        view_latents = self.create_multiview_latents(
            view_prompts, height, width, num_frames
        )

        # Step 3: Build multi-view sequence with tokens
        multiview_sequence, metadata = self.build_multiview_sequence(
            view_tokens, view_latents
        )

        # Step 4: Run inference (placeholder for actual model inference)
        logger.info(f"🚀 Running inference on multi-view sequence: {multiview_sequence.shape}")

        # Add batch dimension for inference
        if multiview_sequence.dim() == 4:
            multiview_sequence = multiview_sequence.unsqueeze(0)

        # Placeholder inference (replace with actual model call)
        with torch.no_grad():
            # In practice, this would be the flow matching inference
            generated_sequence = multiview_sequence + torch.randn_like(multiview_sequence) * 0.01

        # Remove batch dimension
        if generated_sequence.shape[0] == 1:
            generated_sequence = generated_sequence.squeeze(0)

        # Update metadata
        metadata.update({
            'base_prompt': base_prompt,
            'scenario_string': scenario_string,
            'view_tokens': view_tokens,
            'view_prompts': view_prompts,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'generated_shape': generated_sequence.shape
        })

        logger.info(f"🎬 ✅ Multi-view generation completed!")
        logger.info(f"🎬    Base prompt: '{base_prompt}'")
        logger.info(f"🎬    Views generated: {metadata['view_count']}")
        logger.info(f"🎬    Final shape: {generated_sequence.shape}")

        return generated_sequence, metadata


def create_multiview_generator(
    token_embeddings: VideoTokenEmbeddings,
    transformer_model,
    device: torch.device
) -> SinglePromptMultiViewGenerator:
    """
    Factory function to create a multi-view generator.
    """
    return SinglePromptMultiViewGenerator(token_embeddings, transformer_model, device)


# Example usage and testing
def test_multiview_generation():
    """Test the multi-view generation system."""
    logger.info("🧪 Testing single prompt multi-view generation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create token embeddings
    token_embeddings = VideoTokenEmbeddings(hidden_dim=3072)

    # Create mock transformer
    class MockTransformer:
        pass

    transformer = MockTransformer()

    # Create generator
    generator = SinglePromptMultiViewGenerator(token_embeddings, transformer, device)

    # Test scenarios with single prompt
    base_prompt = "woman walking through a beautiful garden"
    test_scenarios = [
        "wide,stable,medium,stable,close",
        "wide,transition,close,transition,medium",
        "bird_eye,transition,front,stable,side",
        "tracking,stable,panning,transition,close"
    ]

    for scenario in test_scenarios:
        logger.info(f"\n🎬 Testing scenario: '{scenario}' with prompt: '{base_prompt}'")
        try:
            generated_sequence, metadata = generator.generate_multiview_video(
                scenario_string=scenario,
                base_prompt=base_prompt
            )
            logger.info(f"✅ Generated multi-view video: {generated_sequence.shape}")
            logger.info(f"   Views: {metadata['view_count']}")
            logger.info(f"   View prompts: {metadata['view_prompts']}")
        except Exception as e:
            logger.error(f"❌ Failed: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_multiview_generation()