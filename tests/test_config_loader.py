"""
Tests for configuration loader - External configuration file support.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import mock_open

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    VisualizationConfig,
)
from config.config_loader import ConfigurationLoader


class TestConfigurationLoader:
    """Test cases for ConfigurationLoader."""

    def test_load_toml_config(self, temp_dir):
        """Test loading TOML configuration."""
        toml_content = """
[data]
n_data_points = 300
noise_level = 0.3

[model]
timesteps = 150
hidden_dim = 256

[training]
n_epochs = 100
batch_size = 64

[visualization]
n_samples = 300
n_gif_frames = 30
"""

        config_path = Path(temp_dir) / "test_config.toml"
        with open(config_path, "w") as f:
            f.write(toml_content)

        try:
            config = ConfigurationLoader.load_toml(config_path)
            assert isinstance(config, ExperimentConfig)
            assert config.data.n_data_points == 300
            assert config.model.timesteps == 150
        except ImportError:
            pytest.skip("tomli not available")

    def test_unsupported_format(self, temp_dir):
        """Test error handling for unsupported file format."""
        config_path = Path(temp_dir) / "test_config.json"
        config_path.write_text("invalid format")

        with pytest.raises(ValueError, match="Only TOML"):
            ConfigurationLoader.load_toml(config_path)

    def test_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            ConfigurationLoader.load_toml("nonexistent.toml")

    def test_save_toml_config(self, temp_dir):
        """Test saving TOML configuration."""
        config = ExperimentConfig.default()
        config_path = Path(temp_dir) / "saved_config.toml"

        ConfigurationLoader.save_to_file(config, config_path)

        # Verify file exists and has proper structure
        assert config_path.exists()

        # Verify we can load it back
        loaded_config = ConfigurationLoader.load_toml(config_path)
        assert isinstance(loaded_config, ExperimentConfig)

        # Verify content has proper TOML structure
        with open(config_path, "r") as f:
            content = f.read()
        assert "[data]" in content
        assert "[model]" in content
        assert "[training]" in content
        assert "[visualization]" in content

    def test_save_unsupported_format(self, temp_dir):
        """Test error handling for unsupported save format."""
        config = ExperimentConfig.default()
        config_path = Path(temp_dir) / "config.json"

        with pytest.raises(ValueError, match="Only TOML"):
            ConfigurationLoader.save_to_file(config, config_path)

    def test_dict_to_config(self):
        """Test conversion from dictionary to ExperimentConfig."""
        config_dict = {
            "data": {"n_data_points": 500, "noise_level": 0.15},
            "model": {"timesteps": 75, "hidden_dim": 96},
            "training": {"n_epochs": 25, "batch_size": 24},
            "visualization": {"n_samples": 500, "n_gif_frames": 15},
        }

        config = ConfigurationLoader._dict_to_config(config_dict)

        assert isinstance(config, ExperimentConfig)
        assert config.data.n_data_points == 500
        assert config.model.timesteps == 75
        assert config.training.n_epochs == 25
        assert config.visualization.n_samples == 500

    def test_dict_to_config_partial(self):
        """Test conversion with partial dictionary (uses defaults)."""
        config_dict = {
            "data": {"n_data_points": 200},
            "model": {"timesteps": 50},
        }

        config = ConfigurationLoader._dict_to_config(config_dict)

        assert isinstance(config, ExperimentConfig)
        assert config.data.n_data_points == 200
        assert config.data.noise_level == 0.05  # Default value
        assert config.model.timesteps == 50
        assert config.training.n_epochs == 7000  # Default value

    def test_validate_config_file(self, temp_dir):
        """Test configuration file validation."""
        # Valid config
        valid_toml = """
[data]
n_data_points = 100
noise_level = 0.1

[model]
timesteps = 50
hidden_dim = 64

[training]
n_epochs = 10
batch_size = 16

[visualization]
n_samples = 100
"""

        valid_path = Path(temp_dir) / "valid_config.toml"
        with open(valid_path, "w") as f:
            f.write(valid_toml)

        assert ConfigurationLoader.validate_config_file(valid_path) is True

        # Invalid config (negative batch size)
        invalid_toml = """
[data]
n_data_points = 100

[model]
timesteps = 50

[training]
n_epochs = 10
batch_size = -1

[visualization]
n_samples = 100
"""

        invalid_path = Path(temp_dir) / "invalid_config.toml"
        with open(invalid_path, "w") as f:
            f.write(invalid_toml)

        assert ConfigurationLoader.validate_config_file(invalid_path) is False

    def test_create_default_config_file(self, temp_dir):
        """Test creating default configuration file."""
        config_path = Path(temp_dir) / "default_config.toml"

        ConfigurationLoader.create_default_config_file(config_path)

        assert config_path.exists()

        # Verify it's a valid config
        loaded_config = ConfigurationLoader.load_toml(config_path)
        assert isinstance(loaded_config, ExperimentConfig)


class TestConfigurationHelper:
    """Test cases for ConfigurationHelper."""

    def test_compare_configs(self):
        """Test configuration comparison."""
        config1 = ExperimentConfig.default()

        # Create modified config
        config2 = ExperimentConfig.default()
        config2.data.n_data_points = 500
        config2.training.n_epochs = 100

        differences = ConfigurationHelper.compare_configs(config1, config2)

        assert "data" in differences
        assert "training" in differences
        assert "n_data_points" in differences["data"]
        assert "n_epochs" in differences["training"]

        # Check difference values
        assert differences["data"]["n_data_points"]["config1"] == 2000
        assert differences["data"]["n_data_points"]["config2"] == 500
        assert differences["training"]["n_epochs"]["config1"] == 7000
        assert differences["training"]["n_epochs"]["config2"] == 100

    def test_compare_identical_configs(self):
        """Test comparison of identical configurations."""
        config1 = ExperimentConfig.default()
        config2 = ExperimentConfig.default()

        differences = ConfigurationHelper.compare_configs(config1, config2)

        assert differences == {}

    def test_print_env_variables(self, capsys):
        """Test printing environment variables."""
        ConfigurationHelper.print_env_variables()

        captured = capsys.readouterr()
        output = captured.out

        # Check that sections are printed
        assert "DATA" in output
        assert "MODEL" in output
        assert "TRAINING" in output
        assert "VISUALIZATION" in output

        # Check that some specific variables are mentioned
        assert "DIFFUSION_DATA_N_DATA_POINTS" in output
        assert "DIFFUSION_TRAINING_N_EPOCHS" in output

    def test_print_env_variables_custom_prefix(self, capsys):
        """Test printing environment variables with custom prefix."""
        ConfigurationHelper.print_env_variables(env_prefix="CUSTOM")

        captured = capsys.readouterr()
        output = captured.out

        assert "CUSTOM_DATA_N_DATA_POINTS" in output
        assert "CUSTOM_TRAINING_N_EPOCHS" in output

    def test_create_example_configs(self, temp_dir):
        """Test creating example configuration files."""
        original_cwd = os.getcwd()

        try:
            # Change to temp directory
            os.chdir(temp_dir)

            ConfigurationHelper.create_example_configs()

            # Check that TOML file was created
            etc_dir = Path("etc")
            assert etc_dir.exists()
            assert (etc_dir / "default_config.toml").exists()

            # Verify it's loadable
            config = ConfigurationLoader.load_toml(etc_dir / "default_config.toml")
            assert isinstance(config, ExperimentConfig)

        finally:
            os.chdir(original_cwd)


class TestConfigurationIntegration:
    """Integration tests for configuration loading."""

    def test_roundtrip_toml(self, temp_dir):
        """Test save and load roundtrip with TOML."""
        original_config = ExperimentConfig.default()
        original_config.data.n_data_points = 123
        original_config.training.n_epochs = 456

        config_path = Path(temp_dir) / "roundtrip.toml"

        # Save and load
        ConfigurationLoader.save_to_file(original_config, config_path)
        loaded_config = ConfigurationLoader.load_toml(config_path)

        # Should be identical
        assert loaded_config.data.n_data_points == 123
        assert loaded_config.training.n_epochs == 456
        assert loaded_config.model.timesteps == original_config.model.timesteps

    def test_config_with_validation_errors(self, temp_dir):
        """Test configuration with validation errors."""
        # Create config with invalid values
        invalid_toml = """
[data]
n_data_points = 0

[model]
timesteps = -1

[training]
batch_size = 0

[visualization]
n_samples = 100
"""

        config_path = Path(temp_dir) / "invalid_config.toml"
        with open(config_path, "w") as f:
            f.write(invalid_toml)

        # Should raise validation error
        with pytest.raises(ValueError):
            ConfigurationLoader.load_toml(config_path)

    def test_partial_config_loading(self, temp_dir):
        """Test loading configuration with missing sections."""
        # Config with only some sections
        partial_toml = """
[data]
n_data_points = 100

[model]
timesteps = 50
"""

        config_path = Path(temp_dir) / "partial_config.toml"
        with open(config_path, "w") as f:
            f.write(partial_toml)

        config = ConfigurationLoader.load_toml(config_path)

        # Should use defaults for missing sections
        assert config.data.n_data_points == 100
        assert config.model.timesteps == 50
        assert config.training.n_epochs == 7000  # Default
        assert config.visualization.n_samples == 2000  # Default
