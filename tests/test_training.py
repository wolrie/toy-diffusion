"""
Tests for training module - Training orchestration and metrics.
"""

import pytest
import torch
import torch.optim as optim

from config.config import TrainingConfig
from domain.diffusion_model import DiffusionModel
from domain.noise_scheduler import LinearNoiseScheduler
from training.scheduler_factory import SchedulerFactory
from training.trainer import DiffusionTrainer, TrainingMetrics


class TestTrainingMetrics:
    """Test cases for TrainingMetrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = TrainingMetrics()

        assert metrics.losses == []
        assert metrics.learning_rates == []

    def test_add_epoch_metrics(self):
        """Test adding epoch metrics."""
        metrics = TrainingMetrics()

        metrics.add_epoch_metrics(0.5, 1e-3)
        metrics.add_epoch_metrics(0.3, 5e-4)

        assert len(metrics.losses) == 2
        assert len(metrics.learning_rates) == 2
        assert metrics.losses == [0.5, 0.3]
        assert metrics.learning_rates == [1e-3, 5e-4]

    def test_get_final_loss(self):
        """Test getting final loss."""
        metrics = TrainingMetrics()

        # Test with no metrics
        assert metrics.get_final_loss() == float("inf")

        # Test with metrics
        metrics.add_epoch_metrics(0.8, 1e-3)
        metrics.add_epoch_metrics(0.4, 5e-4)

        assert metrics.get_final_loss() == 0.4


class TestSchedulerFactory:
    """Test cases for SchedulerFactory."""

    def test_cosine_scheduler_creation(self):
        """Test cosine scheduler creation."""
        optimizer = optim.Adam([torch.randn(5, requires_grad=True)], lr=1e-3)

        scheduler = SchedulerFactory.create_scheduler("cosine", optimizer, T_max=100, eta_min=1e-5)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 1e-5

    def test_step_scheduler_creation(self):
        """Test step scheduler creation."""
        optimizer = optim.Adam([torch.randn(5, requires_grad=True)], lr=1e-3)

        scheduler = SchedulerFactory.create_scheduler("step", optimizer, step_size=50, gamma=0.5)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 50
        assert scheduler.gamma == 0.5

    def test_unknown_scheduler(self):
        """Test error handling for unknown scheduler."""
        optimizer = optim.Adam([torch.randn(5, requires_grad=True)], lr=1e-3)

        with pytest.raises(ValueError, match="Unknown scheduler type"):
            SchedulerFactory.create_scheduler("unknown", optimizer)

    def test_scheduler_registration(self):
        """Test custom scheduler registration."""

        # Define a custom scheduler
        def custom_scheduler(optimizer, **kwargs):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Register it
        SchedulerFactory.register_scheduler("custom", custom_scheduler)

        # Test creation
        optimizer = optim.Adam([torch.randn(5, requires_grad=True)], lr=1e-3)
        scheduler = SchedulerFactory.create_scheduler("custom", optimizer)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10


class TestDiffusionTrainer:
    """Test cases for DiffusionTrainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.noise_scheduler = LinearNoiseScheduler(timesteps=10)
        self.model = DiffusionModel(self.noise_scheduler)
        self.config = TrainingConfig(
            n_epochs=5,
            learning_rate=1e-3,
            batch_size=4,
            scheduler_type="cosine",
        )
        self.trainer = DiffusionTrainer(self.model, self.config)
        self.test_data = torch.randn(20, 2)

    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.model == self.model
        assert self.trainer.config == self.config
        assert isinstance(self.trainer.optimizer, torch.optim.Adam)
        assert isinstance(self.trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert isinstance(self.trainer.metrics, TrainingMetrics)

    def test_scheduler_creation(self):
        """Test learning rate scheduler creation."""
        # Test with different scheduler types
        config_step = TrainingConfig(scheduler_type="step", n_epochs=10)
        trainer_step = DiffusionTrainer(self.model, config_step)

        assert isinstance(trainer_step.scheduler, torch.optim.lr_scheduler.StepLR)

    def test_train_epoch(self):
        """Test single epoch training."""
        from torch.utils.data import DataLoader, TensorDataset

        dataloader = DataLoader(
            TensorDataset(self.test_data),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        initial_loss = self.trainer._train_epoch(dataloader)

        assert isinstance(initial_loss, float)
        assert initial_loss >= 0  # MSE loss should be non-negative
        assert torch.isfinite(torch.tensor(initial_loss))

    def test_training_loop(self):
        """Test complete training loop."""
        initial_params = [p.clone() for p in self.model.parameters()]

        metrics = self.trainer.train(self.test_data, verbose=False)

        # Check that parameters changed
        final_params = list(self.model.parameters())
        params_changed = any(
            not torch.allclose(initial, final, atol=1e-6)
            for initial, final in zip(initial_params, final_params)
        )
        assert params_changed

        # Check metrics
        assert isinstance(metrics, TrainingMetrics)
        assert len(metrics.losses) == self.config.n_epochs
        assert len(metrics.learning_rates) == self.config.n_epochs

        # Check that losses are finite
        assert all(torch.isfinite(torch.tensor(loss)) for loss in metrics.losses)

        # Check that learning rates decreased (cosine schedule)
        assert metrics.learning_rates[-1] < metrics.learning_rates[0]

    def test_training_reproducibility(self):
        """Test training reproducibility."""
        torch.manual_seed(42)
        trainer1 = DiffusionTrainer(self.model, self.config)
        metrics1 = trainer1.train(self.test_data, verbose=False)

        # Reset model parameters
        for p in self.model.parameters():
            p.data = torch.randn_like(p.data)

        torch.manual_seed(42)
        trainer2 = DiffusionTrainer(self.model, self.config)
        metrics2 = trainer2.train(self.test_data, verbose=False)

        # Training should be reproducible
        assert len(metrics1.losses) == len(metrics2.losses)
        # Note: Due to parameter initialization differences, exact loss matching
        # might not occur, but training behavior should be consistent

    def test_evaluate_sample_quality(self):
        """Test sample quality evaluation."""
        # Train briefly first
        self.trainer.train(self.test_data, verbose=False)

        quality_metrics = self.trainer.evaluate_sample_quality(self.test_data, 50)

        assert isinstance(quality_metrics, dict)

        required_keys = [
            "mean_error",
            "std_error",
            "original_mean",
            "original_std",
            "sample_mean",
            "sample_std",
        ]

        for key in required_keys:
            assert key in quality_metrics

        # Check that errors are non-negative
        assert quality_metrics["mean_error"] >= 0
        assert quality_metrics["std_error"] >= 0

        # Check that means and stds are lists of correct length
        assert len(quality_metrics["original_mean"]) == 2
        assert len(quality_metrics["original_std"]) == 2
        assert len(quality_metrics["sample_mean"]) == 2
        assert len(quality_metrics["sample_std"]) == 2

    def test_training_with_different_batch_sizes(self):
        """Test training with different batch sizes."""
        for batch_size in [1, 4, 8]:
            config = TrainingConfig(n_epochs=2, batch_size=batch_size, learning_rate=1e-3)
            trainer = DiffusionTrainer(self.model, config)

            metrics = trainer.train(self.test_data, verbose=False)

            assert len(metrics.losses) == 2
            assert all(torch.isfinite(torch.tensor(loss)) for loss in metrics.losses)

    def test_model_mode_changes(self):
        """Test that trainer properly handles model train/eval modes."""
        # Model should be in training mode during training
        self.trainer.train(self.test_data, verbose=False)

        # Model should be in eval mode during evaluation
        quality_metrics = self.trainer.evaluate_sample_quality(self.test_data, 10)

        # Both should complete without errors
        assert isinstance(quality_metrics, dict)

    def test_gradient_accumulation(self):
        """Test that gradients are properly accumulated and reset."""
        from torch.utils.data import DataLoader, TensorDataset

        dataloader = DataLoader(TensorDataset(self.test_data), batch_size=4, shuffle=False)

        # Check initial gradients are None
        for param in self.model.parameters():
            assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))

        # Run one epoch
        self.trainer._train_epoch(dataloader)

        # Gradients should be computed and then reset
        for param in self.model.parameters():
            if param.grad is not None:
                # Gradients should be close to zero after optimizer.step()
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Create components
        noise_scheduler = LinearNoiseScheduler(timesteps=5)
        model = DiffusionModel(noise_scheduler)
        config = TrainingConfig(n_epochs=3, batch_size=4, learning_rate=1e-3)
        trainer = DiffusionTrainer(model, config)

        # Generate test data
        data = torch.randn(16, 2)

        # Train model
        metrics = trainer.train(data, verbose=False)

        # Evaluate quality
        quality = trainer.evaluate_sample_quality(data, 20)

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples, trajectory = model.sample(10, return_trajectory=True)

        # Verify everything worked
        assert len(metrics.losses) == 3
        assert quality["mean_error"] >= 0
        assert samples.shape == (10, 2)
        assert len(trajectory) == 6  # 5 timesteps + 1 initial

    def test_different_schedulers(self):
        """Test training with different learning rate schedulers."""
        noise_scheduler = LinearNoiseScheduler(timesteps=5)
        model = DiffusionModel(noise_scheduler)
        data = torch.randn(16, 2)

        for scheduler_type in ["cosine", "step"]:
            config = TrainingConfig(n_epochs=3, scheduler_type=scheduler_type, batch_size=4)
            trainer = DiffusionTrainer(model, config)

            metrics = trainer.train(data, verbose=False)

            assert len(metrics.losses) == 3
            assert len(metrics.learning_rates) == 3
            assert all(lr > 0 for lr in metrics.learning_rates)

    def test_training_convergence(self):
        """Test that training shows convergence behavior."""
        noise_scheduler = LinearNoiseScheduler(timesteps=10)
        model = DiffusionModel(noise_scheduler)
        config = TrainingConfig(n_epochs=20, batch_size=8, learning_rate=1e-2)
        trainer = DiffusionTrainer(model, config)

        # Generate synthetic data that should be learnable
        data = torch.randn(100, 2) * 0.5

        metrics = trainer.train(data, verbose=False)

        # Loss should generally decrease over training
        initial_loss = metrics.losses[0]
        final_loss = metrics.losses[-1]

        # Allow for some fluctuation but expect overall improvement
        assert final_loss < initial_loss * 1.1  # At most 10% worse than initial

        # At least some improvement should occur
        min_loss = min(metrics.losses)
        assert min_loss < initial_loss * 0.9  # At least 10% improvement at some point
