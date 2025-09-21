"""
Scenario-based video saving utilities for multi-shot video generation.

Handles automatic directory creation and video saving based on scenario strings.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from diffusers.utils import export_to_video

logger = logging.getLogger(__name__)


class ScenarioVideoSaver:
    """
    Manages saving generated videos with scenario-based directory organization.
    """

    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize the scenario video saver.

        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        logger.info(f"🎬 ScenarioVideoSaver initialized with base dir: {self.base_output_dir}")

    def sanitize_scenario_name(self, scenario_string: str) -> str:
        """
        Convert scenario string to safe directory name.

        Args:
            scenario_string: Raw scenario like "shot1,stable,shot2,transition,shot3"

        Returns:
            Sanitized directory name like "shot1_stable_shot2_transition_shot3"
        """
        # Replace commas with underscores and remove any unsafe characters
        sanitized = re.sub(r'[,\s]+', '_', scenario_string.strip())
        sanitized = re.sub(r'[^\w\-_]', '', sanitized)

        # Ensure it's not too long (max 100 chars)
        if len(sanitized) > 100:
            sanitized = sanitized[:100]

        logger.info(f"🎭 Sanitized scenario name: '{scenario_string}' → '{sanitized}'")
        return sanitized

    def create_scenario_directory(self, scenario_string: str) -> Path:
        """
        Create output directory for a specific scenario.

        Args:
            scenario_string: Scenario description

        Returns:
            Path to the created scenario directory
        """
        scenario_name = self.sanitize_scenario_name(scenario_string)
        scenario_dir = self.base_output_dir / scenario_name

        # Create directory if it doesn't exist
        scenario_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Created scenario directory: {scenario_dir}")
        return scenario_dir

    def save_scenario_videos(
        self,
        scenario_string: str,
        video_frames_list: List[List],
        video_names: Optional[List[str]] = None,
        fps: int = 24,
        step_number: Optional[int] = None
    ) -> List[Path]:
        """
        Save videos for a specific scenario.

        Args:
            scenario_string: Scenario description
            video_frames_list: List of video frame sequences
            video_names: Optional custom names for videos
            fps: Frames per second for output videos
            step_number: Optional step number for training checkpoints

        Returns:
            List of saved video file paths
        """
        scenario_dir = self.create_scenario_directory(scenario_string)
        saved_paths = []

        logger.info(f"🎬 Saving {len(video_frames_list)} videos for scenario: '{scenario_string}'")

        for i, frames in enumerate(video_frames_list):
            # Determine filename
            if video_names and i < len(video_names):
                base_name = video_names[i]
            elif step_number is not None:
                base_name = f"step_{step_number:04d}_{i:02d}"
            else:
                base_name = f"video_{i:02d}"

            # Ensure .mp4 extension
            if not base_name.endswith('.mp4'):
                base_name += '.mp4'

            video_path = scenario_dir / base_name

            # Save video
            export_to_video(frames, str(video_path), fps=fps)
            saved_paths.append(video_path)

            logger.info(f"💾 Saved video: {video_path} ({len(frames)} frames)")

        logger.info(f"🎬 ✅ Saved {len(saved_paths)} videos for scenario '{scenario_string}'")
        return saved_paths

    def save_training_step_videos(
        self,
        scenario_string: str,
        video_frames_list: List[List],
        step_number: int,
        fps: int = 24
    ) -> List[Path]:
        """
        Save videos from a training step with step numbering.

        Args:
            scenario_string: Scenario description
            video_frames_list: List of video frame sequences
            step_number: Training step number
            fps: Frames per second

        Returns:
            List of saved video file paths
        """
        return self.save_scenario_videos(
            scenario_string=scenario_string,
            video_frames_list=video_frames_list,
            fps=fps,
            step_number=step_number
        )

    def save_inference_videos(
        self,
        scenario_string: str,
        shot_frames_list: List[List],
        full_sequence_frames: Optional[List] = None,
        fps: int = 24,
        save_individual_shots: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Save inference videos with shot-based organization.

        Args:
            scenario_string: Scenario description
            shot_frames_list: List of frames for each shot
            full_sequence_frames: Optional full sequence frames
            fps: Frames per second
            save_individual_shots: Whether to save individual shots

        Returns:
            Dictionary with saved paths organized by type
        """
        scenario_dir = self.create_scenario_directory(scenario_string)
        saved_paths = {
            'individual_shots': [],
            'full_sequence': []
        }

        logger.info(f"🎬 Saving inference results for scenario: '{scenario_string}'")

        # Save individual shots
        if save_individual_shots:
            for shot_idx, frames in enumerate(shot_frames_list):
                shot_filename = f"shot_{shot_idx + 1:02d}.mp4"
                shot_path = scenario_dir / shot_filename

                export_to_video(frames, str(shot_path), fps=fps)
                saved_paths['individual_shots'].append(shot_path)

                logger.info(f"💾 Saved shot {shot_idx + 1}: {shot_path}")

        # Save full sequence if provided
        if full_sequence_frames is not None:
            full_sequence_path = scenario_dir / "full_sequence.mp4"
            export_to_video(full_sequence_frames, str(full_sequence_path), fps=fps)
            saved_paths['full_sequence'].append(full_sequence_path)

            logger.info(f"💾 Saved full sequence: {full_sequence_path}")

        logger.info(f"🎬 ✅ Inference saving completed for scenario '{scenario_string}'")
        return saved_paths

    def list_scenario_directories(self) -> List[str]:
        """
        List all existing scenario directories.

        Returns:
            List of scenario directory names
        """
        if not self.base_output_dir.exists():
            return []

        scenario_dirs = [
            d.name for d in self.base_output_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

        logger.info(f"📋 Found {len(scenario_dirs)} scenario directories: {scenario_dirs}")
        return scenario_dirs

    def get_scenario_videos(self, scenario_string: str) -> List[Path]:
        """
        Get all video files for a specific scenario.

        Args:
            scenario_string: Scenario description

        Returns:
            List of video file paths
        """
        scenario_name = self.sanitize_scenario_name(scenario_string)
        scenario_dir = self.base_output_dir / scenario_name

        if not scenario_dir.exists():
            logger.warning(f"⚠️  Scenario directory not found: {scenario_dir}")
            return []

        video_files = list(scenario_dir.glob("*.mp4"))
        logger.info(f"🎥 Found {len(video_files)} videos for scenario '{scenario_string}'")

        return sorted(video_files)

    def clear_scenario_directory(self, scenario_string: str) -> bool:
        """
        Clear all videos from a scenario directory.

        Args:
            scenario_string: Scenario description

        Returns:
            True if successful, False otherwise
        """
        scenario_name = self.sanitize_scenario_name(scenario_string)
        scenario_dir = self.base_output_dir / scenario_name

        if not scenario_dir.exists():
            logger.warning(f"⚠️  Scenario directory not found: {scenario_dir}")
            return False

        video_files = list(scenario_dir.glob("*.mp4"))
        for video_file in video_files:
            try:
                video_file.unlink()
                logger.info(f"🗑️  Deleted: {video_file}")
            except Exception as e:
                logger.error(f"❌ Failed to delete {video_file}: {e}")
                return False

        logger.info(f"🧹 Cleared {len(video_files)} videos from scenario '{scenario_string}'")
        return True


def create_scenario_saver(base_output_dir: str = "outputs") -> ScenarioVideoSaver:
    """
    Factory function to create a ScenarioVideoSaver.

    Args:
        base_output_dir: Base output directory

    Returns:
        Configured ScenarioVideoSaver instance
    """
    return ScenarioVideoSaver(base_output_dir)


# Example usage functions
def save_generated_videos_by_scenario(
    scenario_string: str,
    video_frames_list: List[List],
    base_output_dir: str = "outputs",
    fps: int = 24,
    step_number: Optional[int] = None
) -> List[Path]:
    """
    Convenience function to save videos by scenario.

    Args:
        scenario_string: Scenario description
        video_frames_list: List of video frame sequences
        base_output_dir: Base output directory
        fps: Frames per second
        step_number: Optional step number

    Returns:
        List of saved video paths
    """
    saver = ScenarioVideoSaver(base_output_dir)
    return saver.save_scenario_videos(
        scenario_string=scenario_string,
        video_frames_list=video_frames_list,
        fps=fps,
        step_number=step_number
    )