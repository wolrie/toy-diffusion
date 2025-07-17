"""
Tests for domain module - Core diffusion model logic.
"""

import os
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from domain.diffusion_model import DiffusionModel, DiffusionNetwork
from domain.noise_scheduler import LinearNoiseScheduler


class TestLinearNoiseScheduler:
    """Test cases for LinearNoiseScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = LinearNoiseScheduler(timesteps=100)

        assert scheduler.timesteps == 100
        assert scheduler.betas.shape == (100,)
        assert scheduler.alphas.shape == (100,)
        assert scheduler.alphas_cumprod.shape == (100,)

        # Test that betas are increasing
        assert torch.all(scheduler.betas[1:] >= scheduler.betas[:-1])

        # Test that alphas = 1 - betas
        assert torch.allclose(scheduler.alphas, 1.0 - scheduler.betas)

    def test_forward_process(self):
        """Test forward diffusion process."""
        scheduler = LinearNoiseScheduler(timesteps=10)

        # Create test data
        x0 = torch.randn(5, 2)
        t = torch.tensor([0, 1, 2, 3, 4])

        # Test forward process
        x_t = scheduler.forward_process(x0, t)

        assert x_t.shape == x0.shape
        assert torch.isfinite(x_t).all()

        # Test with specific noise
        noise = torch.randn_like(x0)
        x_t_with_noise = scheduler.forward_process(x0, t, noise)

        assert x_t_with_noise.shape == x0.shape
        assert torch.isfinite(x_t_with_noise).all()

    def test_posterior_params(self):
        """Test posterior parameter computation."""
        scheduler = LinearNoiseScheduler(timesteps=10)

        t = torch.tensor([0, 1, 2, 3, 4])
        params = scheduler.get_posterior_params(t)

        # Should return 4 tensors
        assert len(params) == 4

        # All should have correct batch dimension
        for param in params:
            assert param.shape[0] == len(t)
            assert torch.isfinite(param).all()

    def test_x0_prediction(self):
        """Test x0 prediction from noise."""
        scheduler = LinearNoiseScheduler(timesteps=10)

        # Create test data
        x0 = torch.randn(5, 2)
        t = torch.tensor([1, 2, 3, 4, 5])
        noise = torch.randn_like(x0)

        # Forward process
        x_t = scheduler.forward_process(x0, t, noise)

        # Predict x0 from noise
        predicted_x0 = scheduler.predict_x0_from_noise(x_t, t, noise)

        assert predicted_x0.shape == x0.shape
        assert torch.allclose(predicted_x0, x0, atol=1e-5)


class TestDiffusionNetwork:
    """Test cases for DiffusionNetwork."""

    def test_initialization(self):
        """Test network initialization."""
        network = DiffusionNetwork()

        # Test default parameters
        assert isinstance(network.network, torch.nn.Sequential)

        # Test custom parameters
        network_custom = DiffusionNetwork(input_dim=5, hidden_dim=128, output_dim=3)
        assert isinstance(network_custom.network, torch.nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass."""
        network = DiffusionNetwork()

        # Test with batch of data
        x = torch.randn(10, 3)
        output = network(x)

        assert output.shape == (10, 2)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test that gradients flow through network."""
        network = DiffusionNetwork()

        x = torch.randn(5, 3, requires_grad=True)
        output = network(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestDiffusionModel:
    """Test cases for DiffusionModel."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = LinearNoiseScheduler(timesteps=10)
        self.model = DiffusionModel(self.scheduler)

    def test_initialization(self):
        """Test model initialization."""
        assert self.model.noise_scheduler == self.scheduler
        assert self.model.timesteps == 10
        assert isinstance(self.model.network, DiffusionNetwork)

        # Test with custom network
        custom_network = DiffusionNetwork(hidden_dim=128)
        model_custom = DiffusionModel(self.scheduler, custom_network)
        assert model_custom.network == custom_network

    def test_noise_prediction(self):
        """Test noise prediction."""
        x_t = torch.randn(5, 2)
        t = torch.tensor([0, 1, 2, 3, 4])

        predicted_noise = self.model.predict_noise(x_t, t)

        assert predicted_noise.shape == x_t.shape
        assert torch.isfinite(predicted_noise).all()

    def test_reverse_step(self):
        """Test reverse diffusion step."""
        x_t = torch.randn(5, 2)
        t = torch.tensor([1, 2, 3, 4, 5])

        x_prev = self.model.reverse_step(x_t, t)

        assert x_prev.shape == x_t.shape
        assert torch.isfinite(x_prev).all()

        # Test final step (t=0) - should not add noise
        t_zero = torch.tensor([0, 0, 0, 0, 0])
        x_final = self.model.reverse_step(x_t, t_zero)

        assert x_final.shape == x_t.shape
        assert torch.isfinite(x_final).all()

    def test_sampling(self):
        """Test sample generation."""
        # Test without trajectory
        samples, trajectory = self.model.sample(10, return_trajectory=False)

        assert samples.shape == (10, 2)
        assert torch.isfinite(samples).all()
        assert trajectory is None

        # Test with trajectory
        samples, trajectory = self.model.sample(10, return_trajectory=True)

        assert samples.shape == (10, 2)
        assert torch.isfinite(samples).all()
        assert trajectory is not None
        assert len(trajectory) == self.model.timesteps + 1  # +1 for initial noise

        for step in trajectory:
            assert step.shape == (10, 2)
            assert torch.isfinite(step).all()

    def test_loss_computation(self):
        """Test training loss computation."""
        x0 = torch.randn(8, 2)

        loss = self.model.compute_loss(x0)

        assert loss.numel() == 1  # Scalar loss
        assert torch.isfinite(loss).all()
        assert loss.item() >= 0  # MSE loss should be non-negative

    def test_training_mode_effects(self):
        """Test that training mode affects behavior."""
        x0 = torch.randn(5, 2)

        # Test in training mode
        self.model.train()
        loss_train = self.model.compute_loss(x0)

        # Test in eval mode
        self.model.eval()
        with torch.no_grad():
            loss_eval = self.model.compute_loss(x0)

        # Both should be valid losses
        assert torch.isfinite(loss_train).all()
        assert torch.isfinite(loss_eval).all()

    def test_reproducibility(self):
        """Test that sampling is reproducible with same random seed."""
        torch.manual_seed(42)
        samples1, _ = self.model.sample(10)

        torch.manual_seed(42)
        samples2, _ = self.model.sample(10)

        assert torch.allclose(samples1, samples2, atol=1e-6)


class TestDiffusionModelIntegration:
    """Integration tests for diffusion model components."""

    def test_end_to_end_pipeline(self):
        """Test complete forward and reverse process."""
        scheduler = LinearNoiseScheduler(timesteps=5)
        model = DiffusionModel(scheduler)

        # Create clean data
        x0 = torch.randn(10, 2)

        # Test complete pipeline
        # 1. Compute loss (simulates training)
        loss = model.compute_loss(x0)
        assert torch.isfinite(loss).all()

        # 2. Generate samples
        samples, trajectory = model.sample(10, return_trajectory=True)
        assert samples.shape == x0.shape
        assert len(trajectory) == 6  # 5 timesteps + initial noise

        # 3. Verify trajectory progression
        for i in range(len(trajectory) - 1):
            # Each step should be different (due to denoising)
            assert not torch.allclose(trajectory[i], trajectory[i + 1], atol=1e-3)

    def test_batch_consistency(self):
        """Test that batch processing is consistent."""
        scheduler = LinearNoiseScheduler(timesteps=5)
        model = DiffusionModel(scheduler)

        # Test with different batch sizes
        for batch_size in [1, 5, 10]:
            x0 = torch.randn(batch_size, 2)

            loss = model.compute_loss(x0)
            assert torch.isfinite(loss).all()

            samples, _ = model.sample(batch_size)
            assert samples.shape == (batch_size, 2)
            assert torch.isfinite(samples).all()
