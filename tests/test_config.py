"""
Tests for config module - Configuration management and validation.
"""

import pytest

from config.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    VisualizationConfig,
    VisualizationDisplayConfig,
    VisualizationGeneralConfig,
    VisualizationGifConfig,
    VisualizationProgressionConfig,
)


class TestDataConfig:
    """Test cases for DataConfig."""

    def test_default_initialization(self):
        """Test default data configuration."""
        config = DataConfig()

        assert config.n_data_points == 2000
        assert config.noise_level == 0.05  # Updated in user's file
        assert config.random_state == 42

    def test_custom_initialization(self):
        """Test custom data configuration."""
        config = DataConfig(n_data_points=1000, noise_level=0.3, random_state=123)

        assert config.n_data_points == 1000
        assert config.noise_level == 0.3
        assert config.random_state == 123

    def test_dataclass_properties(self):
        """Test dataclass properties."""
        config = DataConfig()

        # Test that it's a dataclass
        assert hasattr(config, "__dataclass_fields__")

        # Test field types
        fields = config.__dataclass_fields__
        assert fields["n_data_points"].type == int
        assert fields["noise_level"].type == float
        assert fields["random_state"].type == int


class TestModelConfig:
    """Test cases for ModelConfig."""

    def test_default_initialization(self):
        """Test default model configuration."""
        config = ModelConfig()

        assert config.timesteps == 100
        assert config.beta_min == 0.0004
        assert config.beta_max == 0.04
        assert config.hidden_dim == 256

    def test_custom_initialization(self):
        """Test custom model configuration."""
        config = ModelConfig(timesteps=50, beta_min=0.001, beta_max=0.02, hidden_dim=128)

        assert config.timesteps == 50
        assert config.beta_min == 0.001
        assert config.beta_max == 0.02
        assert config.hidden_dim == 128

    def test_beta_relationship(self):
        """Test that beta_min < beta_max."""
        # This is a logical test, not enforced by the dataclass
        config = ModelConfig()
        assert config.beta_min < config.beta_max


class TestTrainingConfig:
    """Test cases for TrainingConfig."""

    def test_default_initialization(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.n_epochs == 7000
        assert config.learning_rate == 1e-3
        assert config.batch_size == 128
        assert config.scheduler_type == "cosine"
        assert config.scheduler_eta_min == 1e-5

    def test_custom_initialization(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            n_epochs=1000,
            learning_rate=5e-4,
            batch_size=64,
            scheduler_type="step",
            scheduler_eta_min=1e-6,
        )

        assert config.n_epochs == 1000
        assert config.learning_rate == 5e-4
        assert config.batch_size == 64
        assert config.scheduler_type == "step"
        assert config.scheduler_eta_min == 1e-6


class TestVisualizationConfig:
    """Test cases for VisualizationConfig."""

    def test_default_initialization(self):
        """Test default visualization configuration."""
        config = VisualizationConfig(
            general=VisualizationGeneralConfig(),
            gif=VisualizationGifConfig(),
            progression=VisualizationProgressionConfig(),
            display=VisualizationDisplayConfig(),
        )

        assert config.general.n_samples == 2000
        assert config.general.n_trajectory_samples == 300
        assert config.gif.n_frames == 20
        assert config.gif.fps == 4
        assert config.display.figure_dpi == 150

    def test_custom_initialization(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            general=VisualizationGeneralConfig(n_samples=1000, n_trajectory_samples=200),
            gif=VisualizationGifConfig(n_frames=10, fps=8),
            progression=VisualizationProgressionConfig(),
            display=VisualizationDisplayConfig(figure_dpi=300),
        )

        assert config.general.n_samples == 1000
        assert config.general.n_trajectory_samples == 200
        assert config.gif.n_frames == 10
        assert config.gif.fps == 8
        assert config.display.figure_dpi == 300


class TestExperimentConfig:
    """Test cases for ExperimentConfig."""

    def test_default_initialization(self):
        """Test default experiment configuration."""
        config = ExperimentConfig.default()

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.visualization, VisualizationConfig)

    def test_custom_initialization(self):
        """Test custom experiment configuration."""
        data_config = DataConfig(n_data_points=500)
        model_config = ModelConfig(timesteps=50)
        training_config = TrainingConfig(n_epochs=100)
        viz_config = VisualizationConfig(n_samples=100)

        config = ExperimentConfig(
            data=data_config,
            model=model_config,
            training=training_config,
            visualization=viz_config,
        )

        assert config.data.n_data_points == 500
        assert config.model.timesteps == 50
        assert config.training.n_epochs == 100
        assert config.visualization.n_samples == 100

    def test_validation_positive_batch_size(self):
        """Test validation for positive batch size."""
        training_config = TrainingConfig(batch_size=0)

        with pytest.raises(ValueError, match="Batch size must be positive"):
            ExperimentConfig(
                data=DataConfig(),
                model=ModelConfig(),
                training=training_config,
                visualization=VisualizationConfig(),
            )

    def test_validation_positive_timesteps(self):
        """Test validation for positive timesteps."""
        model_config = ModelConfig(timesteps=0)

        with pytest.raises(ValueError, match="Timesteps must be positive"):
            ExperimentConfig(
                data=DataConfig(),
                model=model_config,
                training=TrainingConfig(),
                visualization=VisualizationConfig(),
            )

    def test_validation_positive_data_points(self):
        """Test validation for positive data points."""
        data_config = DataConfig(n_data_points=0)

        with pytest.raises(ValueError, match="Number of data points must be positive"):
            ExperimentConfig(
                data=data_config,
                model=ModelConfig(),
                training=TrainingConfig(),
                visualization=VisualizationConfig(),
            )

    def test_validation_negative_batch_size(self):
        """Test validation for negative batch size."""
        training_config = TrainingConfig(batch_size=-10)

        with pytest.raises(ValueError, match="Batch size must be positive"):
            ExperimentConfig(
                data=DataConfig(),
                model=ModelConfig(),
                training=training_config,
                visualization=VisualizationConfig(),
            )

    def test_validation_passes_for_valid_config(self):
        """Test that validation passes for valid configuration."""
        # This should not raise any exception
        config = ExperimentConfig.default()

        # Verify it was created successfully
        assert isinstance(config, ExperimentConfig)
        assert config.training.batch_size > 0
        assert config.model.timesteps > 0
        assert config.data.n_data_points > 0


class TestConfigurationIntegration:
    """Integration tests for configuration management."""

    def test_config_field_access(self):
        """Test accessing nested configuration fields."""
        config = ExperimentConfig.default()

        # Test direct access
        assert config.data.n_data_points == 2000
        assert config.model.timesteps == 100
        assert config.training.batch_size == 128
        assert config.visualization.n_samples == 2000

    def test_config_modification(self):
        """Test modifying configuration values."""
        config = ExperimentConfig.default()

        # Modify values
        config.data.n_data_points = 1000
        config.model.timesteps = 50
        config.training.batch_size = 64

        assert config.data.n_data_points == 1000
        assert config.model.timesteps == 50
        assert config.training.batch_size == 64

    def test_config_serialization_compatibility(self):
        """Test that configs can be converted to dict format."""
        config = ExperimentConfig.default()

        # Convert to dict (useful for serialization)
        from dataclasses import asdict

        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert "visualization" in config_dict

        # Check nested structure
        assert isinstance(config_dict["data"], dict)
        assert config_dict["data"]["n_data_points"] == 2000

    def test_config_copy(self):
        """Test configuration copying."""
        from dataclasses import replace

        config = ExperimentConfig.default()

        # Create modified copy
        modified_config = replace(config, data=replace(config.data, n_data_points=500))

        # Original should be unchanged
        assert config.data.n_data_points == 2000
        assert modified_config.data.n_data_points == 500

    def test_edge_case_values(self):
        """Test configuration with edge case values."""
        # Test with minimum reasonable values
        config = ExperimentConfig(
            data=DataConfig(n_data_points=1, noise_level=0.0, random_state=0),
            model=ModelConfig(timesteps=1, beta_min=1e-6, beta_max=1e-5, hidden_dim=1),
            training=TrainingConfig(n_epochs=1, learning_rate=1e-6, batch_size=1),
            visualization=VisualizationConfig(n_samples=1, n_trajectory_samples=1, n_gif_frames=1),
        )

        # Should pass validation
        assert isinstance(config, ExperimentConfig)

        # Test with large values
        config_large = ExperimentConfig(
            data=DataConfig(n_data_points=100000, noise_level=10.0),
            model=ModelConfig(timesteps=1000, hidden_dim=2048),
            training=TrainingConfig(n_epochs=100000, batch_size=1024),
            visualization=VisualizationConfig(n_samples=10000, figure_dpi=600),
        )

        # Should also pass validation
        assert isinstance(config_large, ExperimentConfig)

    def test_config_type_consistency(self):
        """Test that configuration types are consistent."""
        config = ExperimentConfig.default()

        # Check types
        assert isinstance(config.data.n_data_points, int)
        assert isinstance(config.data.noise_level, float)
        assert isinstance(config.data.random_state, int)

        assert isinstance(config.model.timesteps, int)
        assert isinstance(config.model.beta_min, float)
        assert isinstance(config.model.beta_max, float)
        assert isinstance(config.model.hidden_dim, int)

        assert isinstance(config.training.n_epochs, int)
        assert isinstance(config.training.learning_rate, float)
        assert isinstance(config.training.batch_size, int)
        assert isinstance(config.training.scheduler_type, str)

        assert isinstance(config.visualization.general.n_samples, int)
        assert isinstance(config.visualization.gif.n_frames, int)
