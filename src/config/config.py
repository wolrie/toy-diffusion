"""
Configuration Management - Centralized configuration following Single Responsibility Principle.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data generation configuration."""

    n_data_points: int = 2000
    noise_level: float = 0.05
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    timesteps: int = 100
    beta_min: float = 0.0004
    beta_max: float = 0.04
    hidden_dim: int = 256


@dataclass
class TrainingConfig:
    """Training configuration."""

    n_epochs: int = 7000
    learning_rate: float = 1e-3
    batch_size: int = 128
    scheduler_type: str = "cosine"
    scheduler_eta_min: float = 1e-5


@dataclass
class VisualizationGeneralConfig:
    """General visualization configuration."""

    n_samples: int = 2000
    n_trajectory_samples: int = 300
    create_visualizations: bool = True


@dataclass
class VisualizationGifConfig:
    """GIF animation configuration."""

    create_gif: bool = True
    n_frames: int = 20
    fps: int = 4
    gif_type: str = "standard"  # "standard", "side_by_side", "progression"


@dataclass
class VisualizationProgressionConfig:
    """Progression strip visualization configuration."""

    create_progression_strip: bool = True
    frames: int = 5


@dataclass
class VisualizationDisplayConfig:
    """Display and figure configuration."""

    figure_dpi: int = 150


@dataclass
class VisualizationConfig:
    """Complete visualization configuration."""

    general: VisualizationGeneralConfig
    gif: VisualizationGifConfig
    progression: VisualizationProgressionConfig
    display: VisualizationDisplayConfig


@dataclass
class ExecutionConfig:
    """Execution and runtime configuration."""

    device: str = "auto"  # "cpu", "cuda", "auto"
    verbose: bool = False
    quiet: bool = False


@dataclass
class OutputConfig:
    """Output and saving configuration."""

    save_model: bool = False
    save_config: bool = False
    experiment_name: Optional[str] = None
    output_dir: str = "."


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    visualization: VisualizationConfig
    execution: ExecutionConfig
    output: OutputConfig

    @classmethod
    def default(cls) -> "ExperimentConfig":
        """Create default configuration."""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            visualization=VisualizationConfig(
                general=VisualizationGeneralConfig(),
                gif=VisualizationGifConfig(),
                progression=VisualizationProgressionConfig(),
                display=VisualizationDisplayConfig(),
            ),
            execution=ExecutionConfig(),
            output=OutputConfig(),
        )

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate data config
        if self.data.n_data_points <= 0:
            raise ValueError("Number of data points must be positive")
        if self.data.noise_level < 0:
            raise ValueError("Noise level must be non-negative")

        # Validate model config
        if self.model.timesteps <= 0:
            raise ValueError("Timesteps must be positive")
        if self.model.beta_min <= 0 or self.model.beta_max <= 0:
            raise ValueError("Beta values must be positive")
        if self.model.beta_min >= self.model.beta_max:
            raise ValueError("Beta_min must be less than beta_max")
        if self.model.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")

        # Validate training config
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.training.n_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.training.scheduler_type not in ["cosine", "step"]:
            raise ValueError("Scheduler type must be 'cosine' or 'step'")

        # Validate visualization config
        if self.visualization.general.n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if self.visualization.gif.n_frames <= 0:
            raise ValueError("Number of GIF frames must be positive")
        if self.visualization.gif.fps <= 0:
            raise ValueError("GIF FPS must be positive")
        if self.visualization.display.figure_dpi <= 0:
            raise ValueError("Figure DPI must be positive")
        if self.visualization.gif.gif_type not in [
            "standard",
            "side_by_side",
            "progression",
        ]:
            raise ValueError("GIF type must be 'standard', 'side_by_side', or 'progression'")

        # Validate execution config
        if self.execution.device not in ["cpu", "cuda", "auto"]:
            raise ValueError("Device must be 'cpu', 'cuda', or 'auto'")
        if self.execution.verbose and self.execution.quiet:
            raise ValueError("Cannot set both verbose and quiet to True")
