"""Unit tests for configuration management."""

from pathlib import Path

import pytest

from config import ConfigurationLoader, ExecutionConfig, ExperimentConfig, LoggingConfig
from exceptions import ConfigurationError
from logger import LogLevel
from training.enums import DeviceType


@pytest.mark.unit
class TestConfigurationLoader:
    """Test configuration loading and validation."""

    def test_load_minimal_config(self, minimal_config: ExperimentConfig) -> None:
        """Test loading minimal configuration."""
        assert minimal_config.data.n_data_points == 10
        assert minimal_config.model.timesteps == 5
        assert minimal_config.training.n_epochs == 2
        assert minimal_config.logging.level == LogLevel.DEBUG

    def test_load_production_config(self, production_config: ExperimentConfig) -> None:
        """Test loading production configuration."""
        assert production_config.data.n_data_points == 100
        assert production_config.model.timesteps == 20
        assert production_config.training.n_epochs == 10
        assert production_config.logging.level == LogLevel.INFO
        assert production_config.logging.use_json_format is True

    def test_invalid_config_file_path(self) -> None:
        """Test error handling for non-existent config file."""
        with pytest.raises(ConfigurationError):
            ConfigurationLoader.load_toml("nonexistent.toml")

    def test_save_and_reload_config(self, minimal_config: ExperimentConfig, tmp_path: Path) -> None:
        """Test saving and reloading configuration."""
        save_path = tmp_path / "test_config.toml"

        # Convert enum to string and handle None values for serialization
        minimal_config.logging.level = minimal_config.logging.level.value
        if minimal_config.logging.log_file is None:
            minimal_config.logging.log_file = ""
        elif isinstance(minimal_config.logging.log_file, Path):
            minimal_config.logging.log_file = str(minimal_config.logging.log_file)

        ConfigurationLoader.save_to_file(minimal_config, save_path)
        assert save_path.exists()

        reloaded_config = ConfigurationLoader.load_toml(save_path)
        assert reloaded_config.data.n_data_points == minimal_config.data.n_data_points


@pytest.mark.unit
class TestLoggingConfig:
    """Test logging configuration validation."""

    def test_valid_log_level_string(self) -> None:
        """Test valid log level string conversion."""
        config = LoggingConfig(level="WARNING")
        config.validate()
        assert config.level == LogLevel.WARNING

    def test_invalid_log_level_string(self) -> None:
        """Test invalid log level string raises error."""
        with pytest.raises(ConfigurationError, match="Log level must be one of"):
            config = LoggingConfig(level="INVALID")
            config.validate()

    def test_empty_log_file_becomes_none(self) -> None:
        """Test empty log file string becomes None."""
        config = LoggingConfig(log_file="")
        config.validate()
        assert config.log_file is None

    def test_log_file_string_becomes_path(self) -> None:
        """Test log file string becomes Path object."""
        config = LoggingConfig(log_file="test.log")
        config.validate()
        assert isinstance(config.log_file, Path)
        assert str(config.log_file) == "test.log"

    @pytest.mark.parametrize(
        "json_format,console",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_valid_boolean_combinations(self, json_format: bool, console: bool) -> None:
        """Test all valid boolean combinations."""
        config = LoggingConfig(use_json_format=json_format, enable_console=console)
        config.validate()  # Should not raise


@pytest.mark.unit
class TestExecutionConfig:
    """Test execution configuration validation."""

    def test_device_string_conversion(self) -> None:
        """Test device string conversion."""
        config = ExecutionConfig(device="cpu")
        config.validate()
        assert config.device == DeviceType.CPU

    def test_verbose_and_quiet_conflict(self) -> None:
        """Test that verbose and quiet cannot both be True."""
        with pytest.raises(ConfigurationError, match="Cannot set both verbose and quiet"):
            config = ExecutionConfig(verbose=True, quiet=True)
            config.validate()

    @pytest.mark.parametrize(
        "verbose,quiet",
        [
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_valid_verbose_quiet_combinations(self, verbose: bool, quiet: bool) -> None:
        """Test valid verbose/quiet combinations."""
        config = ExecutionConfig(verbose=verbose, quiet=quiet)
        config.validate()  # Should not raise


@pytest.mark.unit
class TestExperimentConfig:
    """Test complete experiment configuration."""

    def test_default_config_validation(self) -> None:
        """Test default configuration is valid."""
        config = ExperimentConfig.default()
        config.validate()  # Should not raise

        assert config.data is not None
        assert config.model is not None
        assert config.training is not None
        assert config.visualization is not None
        assert config.execution is not None
        assert config.output is not None
        assert config.logging is not None

    def test_nested_validation_propagation(self, invalid_config_path: Path) -> None:
        """Test that validation errors in nested configs are caught."""
        with pytest.raises((ConfigurationError, ValueError)):
            ConfigurationLoader.load_toml(invalid_config_path)

    def test_config_modification(self, minimal_config: ExperimentConfig) -> None:
        """Test configuration can be modified and validated."""
        minimal_config.data.n_data_points = 100
        minimal_config.validate()  # Should still be valid

        assert minimal_config.data.n_data_points == 100

    def test_config_serialization(self, minimal_config: ExperimentConfig) -> None:
        """Test configuration can be serialized to dict."""
        from dataclasses import asdict

        config_dict = asdict(minimal_config)
        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "model" in config_dict
        assert config_dict["data"]["n_data_points"] == 10
