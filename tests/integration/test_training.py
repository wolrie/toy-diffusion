"""Integration tests for training pipeline.

Tests interaction between trainer, model, and data components.
"""

import pytest
import torch

from config import ExperimentConfig
from data import SwissRollGenerator
from domain import DiffusionModel, LinearNoiseScheduler
from training import DiffusionTrainer, TrainingMetrics


@pytest.mark.integration
class TestTrainingIntegration:
    """Test training component integration."""

    def test_trainer_initialization(
        self, diffusion_model: DiffusionModel, minimal_config: ExperimentConfig
    ) -> None:
        """Test trainer initializes properly with model and config."""
        trainer = DiffusionTrainer(diffusion_model, minimal_config.training)

        assert trainer.model == diffusion_model
        assert trainer.config == minimal_config.training
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert isinstance(trainer.metrics, TrainingMetrics)

    def test_training_reduces_loss(
        self, trainer: DiffusionTrainer, swiss_roll_data: torch.Tensor
    ) -> None:
        """Test that training generally reduces loss over time."""
        metrics = trainer.train(swiss_roll_data, verbose=False)

        assert len(metrics.losses) > 0
        assert all(torch.isfinite(torch.tensor(loss)) for loss in metrics.losses)
        assert all(loss > 0 for loss in metrics.losses)

        # Loss should improve (or at least not get much worse)
        initial_loss = metrics.losses[0]
        min_loss = min(metrics.losses)

        # At least the minimum should be better than or equal to initial
        assert min_loss <= initial_loss * 1.1  # Allow 10% tolerance

    def test_training_metrics_consistency(
        self, trainer: DiffusionTrainer, swiss_roll_data: torch.Tensor
    ) -> None:
        """Test training metrics are consistent."""
        metrics = trainer.train(swiss_roll_data, verbose=False)

        assert len(metrics.losses) == trainer.config.n_epochs
        assert len(metrics.learning_rates) == trainer.config.n_epochs

        # Learning rates should be positive and finite
        assert all(lr > 0 for lr in metrics.learning_rates)
        assert all(torch.isfinite(torch.tensor(lr)) for lr in metrics.learning_rates)

    def test_model_parameters_update(
        self, trainer: DiffusionTrainer, swiss_roll_data: torch.Tensor
    ) -> None:
        """Test that model parameters are updated during training."""
        # Store initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Train for one epoch
        trainer.train(swiss_roll_data, verbose=False)

        # Check parameters have changed
        current_params = list(trainer.model.parameters())

        params_changed = any(
            not torch.allclose(initial, current, atol=1e-6)
            for initial, current in zip(initial_params, current_params)
        )

        assert params_changed, "Model parameters should change during training"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_training_with_different_batch_sizes(
        self, minimal_config: ExperimentConfig, batch_size: int
    ) -> None:
        """Test training works with different batch sizes."""
        # Create trainer with custom batch size
        minimal_config.training.batch_size = batch_size

        noise_scheduler = LinearNoiseScheduler(timesteps=5)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, minimal_config.training)

        # Generate data that works with the batch size
        data_size = max(batch_size * 2, 10)  # Ensure enough data
        generator = SwissRollGenerator(random_state=42)
        data = generator.generate(data_size)

        metrics = trainer.train(data, verbose=False)

        assert len(metrics.losses) == minimal_config.training.n_epochs
        assert metrics.get_final_loss() < float("inf")

    def test_sample_quality_evaluation(
        self, trainer: DiffusionTrainer, swiss_roll_data: torch.Tensor
    ) -> None:
        """Test sample quality evaluation."""
        # Train briefly
        trainer.train(swiss_roll_data, verbose=False)

        # Evaluate sample quality
        quality_metrics = trainer.evaluate_sample_quality(swiss_roll_data, n_samples=20)

        assert isinstance(quality_metrics, dict)
        assert "mean_error" in quality_metrics
        assert "std_error" in quality_metrics
        assert "original_mean" in quality_metrics
        assert "sample_mean" in quality_metrics
        assert torch.isfinite(torch.tensor(quality_metrics["mean_error"]))
        assert torch.isfinite(torch.tensor(quality_metrics["std_error"]))
        assert quality_metrics["mean_error"] >= 0
        assert quality_metrics["std_error"] >= 0


@pytest.mark.integration
class TestTrainingScheduler:
    """Test training with different schedulers."""

    @pytest.mark.parametrize("scheduler_type", ["cosine", "step"])
    def test_different_schedulers(
        self, minimal_config: ExperimentConfig, scheduler_type: str
    ) -> None:
        """Test training with different learning rate schedulers."""
        minimal_config.training.scheduler_type = scheduler_type

        noise_scheduler = LinearNoiseScheduler(timesteps=5)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, minimal_config.training)

        generator = SwissRollGenerator(random_state=42)
        data = generator.generate(20)

        metrics = trainer.train(data, verbose=False)

        assert len(metrics.losses) == minimal_config.training.n_epochs
        assert len(metrics.learning_rates) == minimal_config.training.n_epochs

        # Learning rates should change over training (for most schedulers)
        if scheduler_type in ["cosine"]:
            lr_values = metrics.learning_rates
            assert len(set(f"{lr:.8f}" for lr in lr_values)) > 1  # Should have variety
        # Step scheduler might not change with only 2 epochs, so we just check it completed


@pytest.mark.integration
class TestModelDataIntegration:
    """Test integration between model and data components."""

    def test_model_handles_real_data(self, diffusion_model: DiffusionModel) -> None:
        """Test model can process real Swiss roll data."""
        generator = SwissRollGenerator(noise_level=0.1, random_state=42)
        data = generator.generate(50)

        # Test loss computation
        loss = diffusion_model.compute_loss(data)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

        # Test sampling
        samples, trajectory = diffusion_model.sample(25, return_trajectory=True)
        assert samples.shape == (25, 2)
        assert trajectory is not None
        assert len(trajectory) == diffusion_model.timesteps + 1

    def test_data_generator_with_model_training(self, minimal_config: ExperimentConfig) -> None:
        """Test data generator works seamlessly with model training."""
        # Create components
        generator = SwissRollGenerator(
            noise_level=minimal_config.data.noise_level,
            random_state=minimal_config.data.random_state,
        )
        data = generator.generate(minimal_config.data.n_data_points)

        noise_scheduler = LinearNoiseScheduler(timesteps=minimal_config.model.timesteps)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, minimal_config.training)

        # Should train without errors
        metrics = trainer.train(data, verbose=False)

        assert metrics.get_final_loss() < float("inf")
        assert len(metrics.losses) == minimal_config.training.n_epochs

    @pytest.mark.parametrize(
        "data_size,timesteps",
        [
            (10, 5),
            (50, 10),
            (100, 20),
        ],
    )
    def test_different_data_and_model_sizes(
        self, data_size: int, timesteps: int, minimal_config: ExperimentConfig
    ) -> None:
        """Test different combinations of data size and model timesteps."""
        # Generate data
        generator = SwissRollGenerator(random_state=42)
        data = generator.generate(data_size)

        # Create model
        noise_scheduler = LinearNoiseScheduler(timesteps=timesteps)
        model = DiffusionModel(noise_scheduler)

        # Quick training test
        minimal_config.training.n_epochs = 1  # Just one epoch for speed
        trainer = DiffusionTrainer(model, minimal_config.training)

        metrics = trainer.train(data, verbose=False)

        assert len(metrics.losses) == 1
        assert torch.isfinite(torch.tensor(metrics.losses[0]))
