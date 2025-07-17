"""Configuration Loader - Loads config from external files."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

import tomli_w

from .config import (
    DataConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    VisualizationConfig,
    VisualizationDisplayConfig,
    VisualizationGeneralConfig,
    VisualizationGifConfig,
    VisualizationProgressionConfig,
)

try:
    # Python 3.11+ has built-in tomllib
    import tomllib
except ImportError:
    # Python <3.11 needs tomli
    import tomli as tomllib


class ConfigurationLoader:
    """Configuration loader supporting multiple formats."""

    @staticmethod
    def load_toml(config_path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from TOML file.

        Args:
            config_path: Path to TOML configuration file

        Returns:
            ExperimentConfig: Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn"t exist
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return ConfigurationLoader._dict_to_config(data)

    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        # Extract nested configurations
        data_config = DataConfig(**data.get("data", {}))
        model_config = ModelConfig(**data.get("model", {}))
        training_config = TrainingConfig(**data.get("training", {}))
        # Handle nested visualization configuration
        viz_data = data.get("visualization", {})
        visualization_config = VisualizationConfig(
            general=VisualizationGeneralConfig(**viz_data.get("general", {})),
            gif=VisualizationGifConfig(**viz_data.get("gif", {})),
            progression=VisualizationProgressionConfig(**viz_data.get("progression", {})),
            display=VisualizationDisplayConfig(**viz_data.get("display", {})),
        )
        execution_config = ExecutionConfig(**data.get("execution", {}))
        output_config = OutputConfig(**data.get("output", {}))

        return ExperimentConfig(
            data=data_config,
            model=model_config,
            training=training_config,
            visualization=visualization_config,
            execution=execution_config,
            output=output_config,
        )

    @staticmethod
    def save_to_file(config: ExperimentConfig, config_path: Union[str, Path]) -> None:
        """
        Save configuration to TOML file.

        Args:
            config: Configuration to save
            config_path: Path where to save the TOML configuration
        """
        config_path = Path(config_path)
        suffix = config_path.suffix.lower()

        if suffix != ".toml":
            raise ValueError(
                f"Unsupported file format for saving: {suffix}. Only TOML (.toml) files are supported."
            )

        # Convert config to dictionary
        config_dict = asdict(config)

        # Save as TOML
        with open(config_path, "wb") as f:
            tomli_w.dump(config_dict, f)

    @staticmethod
    def create_default_config_file(config_path: Union[str, Path]) -> None:
        """
        Create a default configuration file.

        Args:
            config_path: Path where to create the configuration file
        """
        default_config = ExperimentConfig.default()
        ConfigurationLoader.save_to_file(default_config, config_path)

    @staticmethod
    def validate_config_file(config_path: Union[str, Path]) -> bool:
        """
        Validate a configuration file.

        Args:
            config_path: Path to configuration file to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            config = ConfigurationLoader.load_toml(config_path)
            # Validation happens in ExperimentConfig.__post_init__
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
