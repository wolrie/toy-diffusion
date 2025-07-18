"""Configuration for visualization settings in the diffusion model."""

from dataclasses import dataclass
from typing import Union

from config.base import Config

from .enums import GifType


@dataclass
class VisualizationGeneralConfig(Config):
    """General visualization configuration."""

    n_samples: int = 2000
    n_trajectory_samples: int = 300
    create_visualizations: bool = True

    def validate(self) -> None:
        """Validate visualization general configuration settings."""
        if self.n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if self.n_trajectory_samples <= 0:
            raise ValueError("Number of trajectory samples must be positive")
        if not isinstance(self.create_visualizations, bool):
            raise ValueError("Create visualizations must be a boolean value")


@dataclass
class VisualizationGifConfig(Config):
    """GIF animation configuration."""

    create_gif: bool = True
    n_frames: int = 20
    fps: int = 4
    gif_type: Union[str, GifType] = GifType.STANDARD

    def validate(self) -> None:
        """Validate GIF configuration settings."""
        if not isinstance(self.create_gif, bool):
            raise ValueError("Create GIF must be a boolean value")
        if self.n_frames <= 0:
            raise ValueError("Number of frames must be positive")
        if self.fps <= 0:
            raise ValueError("Frames per second (FPS) must be positive")
        if not isinstance(self.gif_type, GifType):
            try:
                self.gif_type = GifType(self.gif_type)
            except (ValueError, TypeError):
                raise ValueError(f"GIF type must be in {list(GifType)}")


@dataclass
class VisualizationProgressionConfig(Config):
    """Progression strip visualization configuration."""

    create_progression_strip: bool = True
    frames: int = 5

    def validate(self) -> None:
        """Validate progression strip configuration settings."""
        if not isinstance(self.create_progression_strip, bool):
            raise ValueError("Create progression strip must be a boolean value")
        if self.frames <= 0:
            raise ValueError("Number of frames for progression strip must be positive")


@dataclass
class VisualizationDisplayConfig(Config):
    """Display and figure configuration."""

    figure_dpi: int = 150

    def validate(self) -> None:
        """Validate display configuration settings."""
        if self.figure_dpi <= 0:
            raise ValueError("Figure DPI must be positive")


@dataclass
class VisualizationConfig(Config):
    """Complete visualization configuration."""

    general: VisualizationGeneralConfig
    gif: VisualizationGifConfig
    progression: VisualizationProgressionConfig
    display: VisualizationDisplayConfig

    def validate(self) -> None:
        """Validate visualization configuration settings."""
        self.general.validate()
        self.gif.validate()
        self.progression.validate()
        self.display.validate()
