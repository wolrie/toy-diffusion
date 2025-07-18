"""Configuration Management"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.config import DataConfig
from domain.config import ModelConfig
from training.config import TrainingConfig
from training.enums import DeviceType
from visualization.config import (
    VisualizationConfig,
    VisualizationDisplayConfig,
    VisualizationGeneralConfig,
    VisualizationGifConfig,
    VisualizationProgressionConfig,
)

from .base import Config


@dataclass
class ExecutionConfig(Config):
    """Execution and runtime configuration."""

    device: DeviceType = DeviceType.AUTO
    verbose: bool = False
    quiet: bool = False

    def validate(self) -> None:
        """Validate execution configuration settings."""
        if not isinstance(self.device, DeviceType):
            try:
                self.device = DeviceType(self.device)
            except (ValueError, TypeError):
                raise ValueError(f"Device must be in {list(DeviceType)}")
        if not isinstance(self.verbose, bool):
            raise ValueError("Verbose must be a boolean value")
        if not isinstance(self.quiet, bool):
            raise ValueError("Quiet must be a boolean value")
        if self.verbose and self.quiet:
            raise ValueError("Cannot set both verbose and quiet to True")


@dataclass
class OutputConfig(Config):
    """Output and saving configuration."""

    save_model: bool = False
    save_config: bool = False
    experiment_name: Optional[str] = None
    output_dir: str = "outputs"

    def validate(self) -> None:
        """Validate output configuration settings."""
        if not isinstance(self.save_model, bool):
            raise ValueError("Save model must be a boolean value")
        if not isinstance(self.save_config, bool):
            raise ValueError("Save config must be a boolean value")
        if self.experiment_name and not isinstance(self.experiment_name, str):
            raise ValueError("Experiment name must be a string")
        if not isinstance(self.output_dir, str):
            raise ValueError("Output directory must be a string")


@dataclass
class ExperimentConfig(Config):
    """Complete experiment configuration."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    visualization: VisualizationConfig
    execution: ExecutionConfig
    output: OutputConfig

    @classmethod
    def default(cls) -> ExperimentConfig:
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

    def validate(self) -> None:
        """Validate the entire experiment configuration."""
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.visualization.validate()
        self.execution.validate()
        self.output.validate()
