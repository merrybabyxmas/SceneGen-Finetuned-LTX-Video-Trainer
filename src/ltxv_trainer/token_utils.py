"""
Token utilities for multi-shot video generation with stable and transition tokens.

Implements:
1. Stable Token (s): Inserted between latent frames within a single shot for intra-shot temporal continuity
2. Transition Token (t): Inserted between shots for inter-shot transition modeling
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoTokenEmbeddings(nn.Module):
    """
    Learnable embeddings for stable and transition tokens in multi-shot video generation.
    """

    def __init__(self, hidden_dim: int = 128, spatial_height: int = 24, spatial_width: int = 14):
        """
        Initialize video token embeddings.

        Args:
            hidden_dim: Dimension of the latent space (should match transformer hidden size)
            spatial_height: Height dimension of the latent space
            spatial_width: Width dimension of the latent space
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.token_dim = spatial_height * spatial_width * hidden_dim

        logger.info(f"🎭 Initializing VideoTokenEmbeddings with hidden_dim={hidden_dim}, spatial={spatial_height}x{spatial_width}")
        logger.info(f"🎭 Token dimension: H*W*D = {spatial_height}*{spatial_width}*{hidden_dim} = {self.token_dim}")

        # Learnable token embeddings with H*W*D dimensions
        self.stable_token = nn.Parameter(torch.randn(self.token_dim))
        self.transition_token = nn.Parameter(torch.randn(self.token_dim))

        # Initialize with small values for stable training
        nn.init.normal_(self.stable_token, std=0.02)
        nn.init.normal_(self.transition_token, std=0.02)

        logger.info(f"🎭   Stable token initialized: shape={self.stable_token.shape}, std=0.02")
        logger.info(f"🎭   Transition token initialized: shape={self.transition_token.shape}, std=0.02")
        logger.info(f"🎭   VideoTokenEmbeddings ready for training!")

    def get_stable_token(self, device: torch.device, target_shape: tuple = None) -> Tensor:
        """Get stable token for intra-shot continuity, reshaped to [H, W, D] or target_shape."""
        token = self.stable_token.to(device)
        if target_shape is not None:
            H, W, D = target_shape
            return token.reshape(H, W, D)
        return token.reshape(self.spatial_height, self.spatial_width, self.hidden_dim)

    def get_transition_token(self, device: torch.device, target_shape: tuple = None) -> Tensor:
        """Get transition token for inter-shot transitions, reshaped to [H, W, D] or target_shape."""
        token = self.transition_token.to(device)
        if target_shape is not None:
            H, W, D = target_shape
            return token.reshape(H, W, D)
        return token.reshape(self.spatial_height, self.spatial_width, self.hidden_dim)


def insert_stable_tokens(latents: Tensor, stable_token_or_embeddings, device: torch.device = None) -> Tensor:
    """
    Insert stable tokens between latent frames within a shot.

    Args:
        latents: Shot latents [F, H, W, D] where F=frames, H=height, W=width, D=channels
        stable_token_or_embeddings: Either a stable token tensor [H, W, D] or VideoTokenEmbeddings instance
        device: Device for token embeddings (only used if stable_token_or_embeddings is VideoTokenEmbeddings)

    Returns:
        Tokenized latents with stable tokens inserted: [F + (F-1), H, W, D]
        Example: [z1, s, z2, s, z3] for 3 frames
    """
    logger.info(f"latents shape for stable tokens:  {latents.shape}")
    F, H, W, D = latents.shape
    latents_device = latents.device

    # Handle both token tensor and VideoTokenEmbeddings
    if hasattr(stable_token_or_embeddings, 'get_stable_token'):
        # It's a VideoTokenEmbeddings instance
        target_device = device if device is not None else latents_device
        stable_token = stable_token_or_embeddings.get_stable_token(target_device, target_shape=(H, W, D))
    else:
        # It's already a token tensor
        stable_token = stable_token_or_embeddings

    logger.info(f"🔧 Inserting stable tokens: input shape {latents.shape}, stable_token shape {stable_token.shape}")

    if F <= 1:
        # Single frame or empty, no tokens to insert
        logger.info(f"🔧 Single frame or empty ({F} frames), no stable tokens inserted")
        return latents
    logger.info(f"stable token shape : {stable_token.shape}")
    logger.info(f"latents token shape : {latents.shape}")

    # Stable token should match latent spatial dimensions [H, W, D], just add frame dimension
    stable_spatial = stable_token.unsqueeze(0)  # [1, H, W, D]
    logger.info(f"🔧 Stable token reshaped to frame format: {stable_spatial.shape}")

    # Insert stable tokens between frames
    result_frames = []
    stable_positions = []
    for i in range(F):
        result_frames.append(latents[i:i+1])  # [1, H, W, D]
        if i < F - 1:  # Don't add stable token after last frame
            result_frames.append(stable_spatial)
            stable_positions.append(len(result_frames) - 1)  # Record position of stable token

    # Concatenate along frame dimension
    tokenized_latents = torch.cat(result_frames, dim=0)  # [F + (F-1), H, W, D]

    num_stable_tokens = len(stable_positions)
    logger.info(f"🔧 Inserted {num_stable_tokens} stable tokens at positions: {stable_positions}")
    logger.info(f"🔧 Result shape: {tokenized_latents.shape} (original: {F} → with stables: {tokenized_latents.shape[0]})")

    return tokenized_latents


def insert_transition_token(prev_latents: Tensor, curr_latents: Tensor,
                          transition_token_or_embeddings, device: torch.device = None) -> Tensor:
    """
    Insert transition token between previous and current shot latents.

    Args:
        prev_latents: Previous shot latents (already with stable tokens) [Fp, H, W, D]
        curr_latents: Current shot latents (already with stable tokens) [Fc, H, W, D]
        transition_token_or_embeddings: Either a transition token tensor [H, W, D] or VideoTokenEmbeddings instance
        device: Device for token embeddings (only used if transition_token_or_embeddings is VideoTokenEmbeddings)

    Returns:
        Combined latents with transition token: [Fp + 1 + Fc, H, W, D]
        Example: [prev_shot_with_stables, t, curr_shot_with_stables]
    """
    Fp, H, W, D = prev_latents.shape
    Fc = curr_latents.shape[0]
    latents_device = prev_latents.device

    # Handle both token tensor and VideoTokenEmbeddings
    if hasattr(transition_token_or_embeddings, 'get_transition_token'):
        # It's a VideoTokenEmbeddings instance
        target_device = device if device is not None else latents_device
        transition_token = transition_token_or_embeddings.get_transition_token(target_device, target_shape=(H, W, D))
    else:
        # It's already a token tensor
        transition_token = transition_token_or_embeddings

    logger.info(f"🔄 Inserting transition token:")
    logger.info(f"🔄   Prev latents shape: {prev_latents.shape}")
    logger.info(f"🔄   Curr latents shape: {curr_latents.shape}")
    logger.info(f"🔄   Transition token shape: {transition_token.shape}")

    # Transition token should match latent spatial dimensions [H, W, D], just add frame dimension
    transition_spatial = transition_token.unsqueeze(0)  # [1, H, W, D]
    logger.info(f"🔄   Transition token reshaped to frame format: {transition_spatial.shape}")

    # Record transition token position
    transition_position = Fp  # Position in final sequence where transition token is inserted

    # Combine: [prev_latents, transition_token, curr_latents]
    combined_latents = torch.cat([
        prev_latents,      # [Fp, H, W, D]
        transition_spatial,  # [1, H, W, D]
        curr_latents       # [Fc, H, W, D]
    ], dim=0)  # [Fp + 1 + Fc, H, W, D]

    total_frames = combined_latents.shape[0]
    logger.info(f"🔄 Transition token inserted at position {transition_position}")
    logger.info(f"🔄 Combined shape: {combined_latents.shape} (total frames: {total_frames})")
    logger.info(f"🔄 Frame breakdown: prev={Fp} + transition=1 + curr={Fc} = {total_frames}")

    return combined_latents


def preprocess_shot_adapter_with_tokens(
    prev_shot_latents: Tensor,
    curr_shot_latents: Tensor,
    token_embeddings: VideoTokenEmbeddings,
    device: torch.device
) -> Tuple[Tensor, dict]:
    """
    Shot-adapter style preprocessing with stable and transition tokens.

    Args:
        prev_shot_latents: Previous shot latents [F, H, W, D]
        curr_shot_latents: Current shot latents [F, H, W, D]
        token_embeddings: Video token embeddings module
        device: Target device

    Returns:
        Tuple of:
        - tokenized_sequence: [prev_with_stables, transition_token, curr_with_stables] [Total_F, H, W, D]
        - metadata: Dict with sequence information
    """
    logger.info(f"🎬 Shot-adapter token preprocessing:")
    logger.info(f"🎬   Prev shot: {prev_shot_latents.shape}")
    logger.info(f"🎬   Curr shot: {curr_shot_latents.shape}")

    # Step 1: Insert stable tokens within each shot (pass embeddings instance)
    logger.info(f"🎬 Step 1: Inserting stable tokens within shots...")
    prev_with_stables = insert_stable_tokens(prev_shot_latents, token_embeddings, device)
    curr_with_stables = insert_stable_tokens(curr_shot_latents, token_embeddings, device)

    # Step 2: Insert transition token between shots (pass embeddings instance)
    logger.info(f"🎬 Step 2: Inserting transition token between shots...")
    tokenized_sequence = insert_transition_token(prev_with_stables, curr_with_stables, token_embeddings, device)

    # Create metadata
    F_prev_orig, F_curr_orig = prev_shot_latents.shape[0], curr_shot_latents.shape[0]
    F_prev_with_stables = prev_with_stables.shape[0]
    F_curr_with_stables = curr_with_stables.shape[0]
    transition_pos = F_prev_with_stables

    metadata = {
        'original_prev_frames': F_prev_orig,
        'original_curr_frames': F_curr_orig,
        'prev_with_stables_frames': F_prev_with_stables,
        'curr_with_stables_frames': F_curr_with_stables,
        'transition_token_position': transition_pos,
        'total_frames': tokenized_sequence.shape[0],
        'stable_tokens_prev': max(0, F_prev_orig - 1),
        'stable_tokens_curr': max(0, F_curr_orig - 1),
        'prev_shot_range': (0, F_prev_with_stables),
        'transition_range': (transition_pos, transition_pos + 1),
        'curr_shot_range': (transition_pos + 1, tokenized_sequence.shape[0])
    }

    logger.info(f"🎬 ✅ Shot-adapter preprocessing completed!")
    logger.info(f"🎬   Final sequence: {tokenized_sequence.shape}")
    logger.info(f"🎬   Prev range: {metadata['prev_shot_range']}")
    logger.info(f"🎬   Transition pos: {metadata['transition_token_position']}")
    logger.info(f"🎬   Curr range: {metadata['curr_shot_range']}")

    return tokenized_sequence, metadata


def preprocess_multishot_with_tokens(
    prev_shot_latents: Tensor,
    curr_shot_latents: Tensor,
    token_embeddings: VideoTokenEmbeddings,
    device: torch.device
) -> Tuple[Tensor, dict]:
    """
    Complete preprocessing function for multi-shot video generation with tokens.

    Args:
        prev_shot_latents: Previous shot latents [F1, H, W, D]
        curr_shot_latents: Current shot latents [F2, H, W, D]
        token_embeddings: Video token embeddings module
        device: Target device

    Returns:
        Tuple of:
        - tokenized_sequence: Final sequence ready for patchify [Total_F, H, W, D]
        - metadata: Dict with sequence information for loss computation
    """
    logger.info(f"🎬 Starting multi-shot token preprocessing:")
    logger.info(f"🎬   Prev shot latents: {prev_shot_latents.shape}")
    logger.info(f"🎬   Curr shot latents: {curr_shot_latents.shape}")
    logger.info(f"🎬   Device: {device}")

    # Step 1: Insert stable tokens within each shot (pass embeddings instance)
    logger.info(f"🎬 Step 1: Inserting stable tokens within shots...")
    prev_with_stables = insert_stable_tokens(prev_shot_latents, token_embeddings, device)
    curr_with_stables = insert_stable_tokens(curr_shot_latents, token_embeddings, device)

    # Step 2: Insert transition token between shots (pass embeddings instance)
    logger.info(f"🎬 Step 2: Inserting transition token between shots...")
    tokenized_sequence = insert_transition_token(prev_with_stables, curr_with_stables, token_embeddings, device)

    # Create metadata for loss computation and analysis
    F1, F2 = prev_shot_latents.shape[0], curr_shot_latents.shape[0]
    F1_with_stables = prev_with_stables.shape[0]
    F2_with_stables = curr_with_stables.shape[0]

    logger.info(f"🎬 Step 3: Creating metadata...")
    logger.info(f"🎬   Original frames: prev={F1}, curr={F2}")
    logger.info(f"🎬   With stable tokens: prev={F1_with_stables}, curr={F2_with_stables}")
    logger.info(f"🎬   Total frames in final sequence: {tokenized_sequence.shape[0]}")

    metadata = {
        'original_prev_frames': F1,
        'original_curr_frames': F2,
        'prev_with_stables_frames': F1_with_stables,
        'curr_with_stables_frames': F2_with_stables,
        'transition_token_position': F1_with_stables,  # Index where transition token is inserted
        'total_frames': tokenized_sequence.shape[0],
        'stable_tokens_prev': max(0, F1 - 1),  # Number of stable tokens in prev shot
        'stable_tokens_curr': max(0, F2 - 1),  # Number of stable tokens in curr shot
        'prev_shot_range': (0, F1_with_stables),  # Range of prev shot frames in final sequence
        'transition_range': (F1_with_stables, F1_with_stables + 1),  # Range of transition token
        'curr_shot_range': (F1_with_stables + 1, tokenized_sequence.shape[0])  # Range of curr shot frames
    }

    logger.info(f"🎬   Stable tokens count: prev={metadata['stable_tokens_prev']}, curr={metadata['stable_tokens_curr']}")
    logger.info(f"🎬   Frame ranges: prev={metadata['prev_shot_range']}, transition={metadata['transition_range']}, curr={metadata['curr_shot_range']}")
    logger.info(f"🎬   Transition token position: {metadata['transition_token_position']}")
    logger.info(f"🎬 ✅ Token preprocessing completed! Final shape: {tokenized_sequence.shape}")

    return tokenized_sequence, metadata


def create_shot_adapter_conditioning_mask(
    metadata: dict,
    conditioning_strength: float = 1.0,
    stable_token_strength: float = 0.5,
    transition_token_strength: float = 0.8,
    device: torch.device = None
) -> Tuple[Tensor, Tensor]:
    """
    Create shot-adapter style conditioning mask for flow matching training.

    Returns separate boolean conditioning mask for timesteps and float strength mask for attention/loss.

    In shot-adapter:
    - Previous shot (xp): Full conditioning (True, strength 1.0)
    - Stable tokens in prev shot: Conditioning (True, strength 0.5)
    - Transition token: Conditioning (True, strength 0.8)
    - Current shot (x1c): NO conditioning (False, strength 0.0) - these are training targets
    - Stable tokens in curr shot: Conditioning (True, strength 0.5)

    Args:
        metadata: Metadata from preprocess_shot_adapter_with_tokens
        conditioning_strength: Strength for previous shot frames
        stable_token_strength: Strength for stable tokens
        transition_token_strength: Strength for transition token

    Returns:
        Tuple of:
        - Boolean conditioning mask [Total_F] (True=conditioning, False=target)
        - Float strength mask [Total_F] with conditioning strengths
    """
    total_frames = metadata['total_frames']
    bool_mask = torch.zeros(total_frames, dtype=torch.bool, device=device)
    strength_mask = torch.zeros(total_frames, device=device)

    logger.info(f"🎭 Creating shot-adapter conditioning masks for {total_frames} frames")

    # Get ranges
    prev_start, prev_end = metadata['prev_shot_range']
    trans_start, trans_end = metadata['transition_range']
    curr_start, curr_end = metadata['curr_shot_range']

    # Previous shot: condition on everything (including stables)
    for i in range(prev_start, prev_end):
        local_idx = i - prev_start
        bool_mask[i] = True  # All previous shot tokens are conditioned
        # In prev shot: [frame, stable, frame, stable, ...]
        if local_idx % 2 == 0:  # Frame
            strength_mask[i] = conditioning_strength
        else:  # Stable token
            strength_mask[i] = stable_token_strength

    logger.info(f"🎭   Previous shot [{prev_start}:{prev_end}] → conditioning: True, frames: {conditioning_strength}, stables: {stable_token_strength}")

    # Transition token: strong conditioning
    bool_mask[trans_start:trans_end] = True
    strength_mask[trans_start:trans_end] = transition_token_strength
    logger.info(f"🎭   Transition token [{trans_start}:{trans_end}] → conditioning: True, strength: {transition_token_strength}")

    # Current shot: NO conditioning on frames (training targets), but condition on stable tokens
    for i in range(curr_start, curr_end):
        local_idx = i - curr_start
        # In curr shot: [frame, stable, frame, stable, ...]
        if local_idx % 2 == 0:  # Frame - NO conditioning (training target)
            bool_mask[i] = False
            strength_mask[i] = 0.0
        else:  # Stable token - medium conditioning
            bool_mask[i] = True
            strength_mask[i] = stable_token_strength

    logger.info(f"🎭   Current shot [{curr_start}:{curr_end}] → frames: False (targets), stables: True")
    logger.info(f"🎭   Boolean mask: {bool_mask.tolist()}")
    logger.info(f"🎭   Strength mask: {strength_mask.tolist()}")

    return bool_mask, strength_mask


def create_token_aware_conditioning_mask(
    metadata: dict,
    conditioning_strength: float = 1.0,
    stable_token_strength: float = 0.5,
    transition_token_strength: float = 0.8,
    device: torch.device = None
) -> Tuple[Tensor, Tensor]:
    """
    Create conditioning mask that accounts for different token types.

    Returns separate boolean conditioning mask for timesteps and float strength mask for attention/loss.

    Args:
        metadata: Metadata from preprocess_multishot_with_tokens
        conditioning_strength: Strength for regular latent frames
        stable_token_strength: Strength for stable tokens (usually lower)
        transition_token_strength: Strength for transition token

    Returns:
        Tuple of:
        - Boolean conditioning mask [Total_F] (True=conditioning, False=target)
        - Float strength mask [Total_F] with conditioning strengths
    """
    total_frames = metadata['total_frames']
    bool_mask = torch.ones(total_frames, dtype=torch.bool, device=device)
    strength_mask = torch.ones(total_frames, device=device)

    logger.info(f"🎭 Creating token-aware conditioning masks for {total_frames} frames")
    logger.info(f"🎭   Conditioning strengths: regular={conditioning_strength}, stable={stable_token_strength}, transition={transition_token_strength}")

    # Set conditioning strengths based on token types
    prev_start, prev_end = metadata['prev_shot_range']
    trans_start, trans_end = metadata['transition_range']
    curr_start, curr_end = metadata['curr_shot_range']

    # Previous shot: full conditioning strength
    bool_mask[prev_start:prev_end] = True
    strength_mask[prev_start:prev_end] = conditioning_strength
    logger.info(f"🎭   Previous shot [{prev_start}:{prev_end}] → conditioning: True, strength: {conditioning_strength}")

    # Transition token: medium strength
    bool_mask[trans_start:trans_end] = True
    strength_mask[trans_start:trans_end] = transition_token_strength
    logger.info(f"🎭   Transition token [{trans_start}:{trans_end}] → conditioning: True, strength: {transition_token_strength}")

    # Current shot: conditioning based on frame type
    curr_frames = curr_end - curr_start
    stable_positions = []
    frame_positions = []

    for i in range(curr_frames):
        global_idx = curr_start + i
        local_idx = i

        # Check if this is a stable token or regular frame in current shot
        # In current shot with stables: [frame, stable, frame, stable, ...]
        # So odd indices (1, 3, 5, ...) are stable tokens
        if local_idx % 2 == 1:  # Stable token
            bool_mask[global_idx] = True
            strength_mask[global_idx] = stable_token_strength
            stable_positions.append(global_idx)
        else:  # Regular frame - these should typically have low/zero conditioning for training
            bool_mask[global_idx] = False  # Current frames are not conditioned during training
            strength_mask[global_idx] = 0.0
            frame_positions.append(global_idx)

    logger.info(f"🎭   Current shot stable tokens {stable_positions} → conditioning: True, strength: {stable_token_strength}")
    logger.info(f"🎭   Current shot frames {frame_positions} → conditioning: False (training targets)")
    logger.info(f"🎭   Boolean mask: {bool_mask.tolist()}")
    logger.info(f"🎭   Strength mask: {strength_mask.tolist()}")

    return bool_mask, strength_mask


def preprocess_with_scenario(
    shot_latents: dict[str, Tensor],
    scenario: str,
    token_embeddings: VideoTokenEmbeddings,
    device: torch.device
) -> Tuple[Tensor, dict]:
    """
    Process shot latents according to scenario string with stable and transition tokens.

    Args:
        shot_latents: Dictionary mapping shot names to latent tensors [F, H, W, D]
        scenario: Scenario string like "shot1,stable,shot2,transition,shot3"
        token_embeddings: Video token embeddings module
        device: Target device

    Returns:
        Tuple of:
        - sequence: Concatenated sequence with tokens [Total_F, H, W, D]
        - metadata: Information about the sequence construction
    """
    logger.info(f"🎬 Processing scenario: '{scenario}'")
    logger.info(f"🎬 Available shots: {list(shot_latents.keys())}")

    # Parse scenario (handle both string and list input)
    logger.info(f"🎬 Raw scenario type: {type(scenario)}, value: {scenario}")
    if isinstance(scenario, str):
        tokens = [token.strip() for token in scenario.split(',')]
    elif isinstance(scenario, list):
        # If list has one element that's a comma-separated string, split it
        if len(scenario) == 1 and isinstance(scenario[0], str) and ',' in scenario[0]:
            tokens = [token.strip() for token in scenario[0].split(',')]
        else:
            tokens = [str(token).strip() for token in scenario]
    else:
        raise ValueError(f"Scenario must be string or list, got {type(scenario)}")
    logger.info(f"🎬 Parsed tokens: {tokens}")

    # Validate token structure
    if len(tokens) < 1:
        raise ValueError("Scenario must contain at least one shot")

    # Will use token embeddings instance directly

    sequence_parts = []
    metadata = {
        'scenario': scenario,
        'tokens': tokens,
        'shot_ranges': {},
        'stable_token_positions': [],
        'transition_token_positions': [],
        'total_frames': 0,
        'shot_count': 0
    }

    current_position = 0

    for i, token in enumerate(tokens):
        if i % 2 == 0:  # Even indices should be shot names
            shot_name = token
            if shot_name not in shot_latents:
                raise ValueError(f"Shot '{shot_name}' not found in provided latents")

            # Get shot latents and insert stable tokens within the shot
            shot_frames = shot_latents[shot_name]


            shot_with_stables = insert_stable_tokens(shot_frames, token_embeddings, device)
            logger.info(f"shot frames : {shot_frames.shape}")
            logger.info(f"shot w stables : {shot_with_stables.shape}")

            # Record shot range in final sequence
            shot_frames_count = shot_with_stables.shape[0]
            metadata['shot_ranges'][shot_name] = (current_position, current_position + shot_frames_count)
            metadata['shot_count'] += 1

            sequence_parts.append(shot_with_stables)
            current_position += shot_frames_count

            logger.info(f"🎥 Added shot '{shot_name}' at frames [{metadata['shot_ranges'][shot_name][0]}:{metadata['shot_ranges'][shot_name][1]}]")

        else:  # Odd indices should be connectors
            connector = token
            if len(sequence_parts) == 0:
                raise ValueError(f"Connector '{connector}' at position {i} without preceding shot")
            
            logger.info(f"  last shot shape : {sequence_parts[-1].shape[1:]}")

            # Get spatial dimensions from last shot
            last_shot_shape = sequence_parts[-1].shape
            H, W, D = last_shot_shape[1], last_shot_shape[2], last_shot_shape[3]

            if connector == 'stable':
                # Insert stable token for smooth transition
                stable_spatial = token_embeddings.get_stable_token(device, target_shape=(H, W, D)).unsqueeze(0)
                sequence_parts.append(stable_spatial)
                metadata['stable_token_positions'].append(current_position)
                logger.info(f"🔧 Added stable token at position {current_position}")

            elif connector == 'transition':
                # Insert transition token for sharp transition
                transition_spatial = token_embeddings.get_transition_token(device, target_shape=(H, W, D)).unsqueeze(0)
                sequence_parts.append(transition_spatial)
                metadata['transition_token_positions'].append(current_position)
                logger.info(f"🔄 Added transition token at position {current_position}")

            else:
                logger.warning(f"Unknown connector '{connector}', treating as 'stable'")
                stable_spatial = token_embeddings.get_stable_token(device, target_shape=(H, W, D)).unsqueeze(0)
                sequence_parts.append(stable_spatial)
                metadata['stable_token_positions'].append(current_position)

            current_position += 1

    # Combine all parts
    if not sequence_parts:
        raise ValueError("No sequence parts generated from scenario")

    combined_sequence = torch.cat(sequence_parts, dim=0)
    metadata['total_frames'] = combined_sequence.shape[0]

    logger.info(f"🎬 ✅ Scenario processing completed!")
    logger.info(f"🎬   Final sequence: {combined_sequence.shape}")
    logger.info(f"🎬   Shots: {metadata['shot_count']}")
    logger.info(f"🎬   Total frames: {metadata['total_frames']}")
    logger.info(f"🎬   Stable tokens: {len(metadata['stable_token_positions'])}")
    logger.info(f"🎬   Transition tokens: {len(metadata['transition_token_positions'])}")

    return combined_sequence, metadata


def create_scenario_conditioning_mask(
    metadata: dict,
    conditioning_strength: float = 1.0,
    stable_token_strength: float = 0.5,
    transition_token_strength: float = 0.8,
    target_shot: Optional[str] = None,
    device: torch.device = None
) -> Tuple[Tensor, Tensor]:
    """
    Create conditioning mask for scenario-based processing.

    Returns separate boolean conditioning mask for timesteps and float strength mask for attention/loss.

    Args:
        metadata: Metadata from preprocess_with_scenario
        conditioning_strength: Strength for conditioning shots
        stable_token_strength: Strength for stable tokens
        transition_token_strength: Strength for transition tokens
        target_shot: If specified, this shot has False conditioning (training target)

    Returns:
        Tuple of:
        - Boolean conditioning mask [Total_F] (True=conditioning, False=target)
        - Float strength mask [Total_F] with conditioning strengths
    """
    total_frames = metadata['total_frames']
    bool_mask = torch.ones(total_frames, dtype=torch.bool, device=device)
    strength_mask = torch.ones(total_frames, device=device)

    logger.info(f"🎭 Creating scenario conditioning masks for {total_frames} frames")
    if target_shot:
        logger.info(f"🎭   Target shot (no conditioning): {target_shot}")

    # Set shot conditioning
    for shot_name, (start, end) in metadata['shot_ranges'].items():
        if target_shot and shot_name == target_shot:
            # Target shot: set frames to False, stable tokens to True with medium strength
            for i in range(start, end):
                shot_local_idx = i - start
                if shot_local_idx % 2 == 0:  # Frame
                    bool_mask[i] = False  # Training target
                    strength_mask[i] = 0.0
                else:  # Stable token
                    bool_mask[i] = True
                    strength_mask[i] = stable_token_strength
            logger.info(f"🎭   Shot '{shot_name}' [{start}:{end}] → frames: False (target), stables: True")
        else:
            # Conditioning shot: set frames to True, stable tokens to True with medium strength
            for i in range(start, end):
                shot_local_idx = i - start
                if shot_local_idx % 2 == 0:  # Frame
                    bool_mask[i] = True
                    strength_mask[i] = conditioning_strength
                else:  # Stable token
                    bool_mask[i] = True
                    strength_mask[i] = stable_token_strength
            logger.info(f"🎭   Shot '{shot_name}' [{start}:{end}] → frames: True, stables: True")

    # Set stable token positions
    for pos in metadata['stable_token_positions']:
        bool_mask[pos] = True
        strength_mask[pos] = stable_token_strength

    # Set transition token positions
    for pos in metadata['transition_token_positions']:
        bool_mask[pos] = True
        strength_mask[pos] = transition_token_strength

    logger.info(f"🎭   Inter-shot stable tokens: {len(metadata['stable_token_positions'])} at {metadata['stable_token_positions']}")
    logger.info(f"🎭   Transition tokens: {len(metadata['transition_token_positions'])} at {metadata['transition_token_positions']}")

    return bool_mask, strength_mask


def extract_token_positions(metadata: dict) -> dict:
    """
    Extract positions of different token types for analysis and loss computation.

    Args:
        metadata: Metadata from preprocess_multishot_with_tokens

    Returns:
        Dictionary with token position information
    """
    positions = {
        'stable_token_positions': [],
        'transition_token_positions': [],
        'regular_frame_positions': []
    }

    prev_start, prev_end = metadata['prev_shot_range']
    trans_start, trans_end = metadata['transition_range']
    curr_start, curr_end = metadata['curr_shot_range']

    # Previous shot positions
    for i in range(prev_start, prev_end):
        local_idx = i - prev_start
        if local_idx % 2 == 1:  # Stable tokens are at odd positions
            positions['stable_token_positions'].append(i)
        else:
            positions['regular_frame_positions'].append(i)

    # Transition token positions
    for i in range(trans_start, trans_end):
        positions['transition_token_positions'].append(i)

    # Current shot positions
    for i in range(curr_start, curr_end):
        local_idx = i - curr_start
        if local_idx % 2 == 1:  # Stable tokens are at odd positions
            positions['stable_token_positions'].append(i)
        else:
            positions['regular_frame_positions'].append(i)

    return positions