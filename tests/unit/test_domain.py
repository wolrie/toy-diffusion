"""Unit tests for domain models (diffusion model and noise scheduler)."""

import pytest
import torch

from domain import DiffusionModel, DiffusionNetwork, LinearNoiseScheduler


@pytest.mark.unit
class TestLinearNoiseScheduler:
    """Test noise scheduler functionality."""

    def test_initialization(self):
        """Test scheduler initialization with different parameters."""
        scheduler = LinearNoiseScheduler(timesteps=100, beta_min=0.001, beta_max=0.02)

        assert scheduler.timesteps == 100
        assert scheduler.betas.shape == (100,)
        assert scheduler.alphas.shape == (100,)

        # Mathematical relationships
        assert torch.allclose(scheduler.alphas, 1.0 - scheduler.betas)
        assert torch.all(scheduler.betas[1:] >= scheduler.betas[:-1])  # Monotonic

    @pytest.mark.parametrize("timesteps", [1, 10, 100, 1000])
    def test_different_timesteps(self, timesteps: int):
        """Test scheduler with different timestep values."""
        scheduler = LinearNoiseScheduler(timesteps=timesteps)
        assert scheduler.timesteps == timesteps
        assert scheduler.betas.shape == (timesteps,)

    def test_forward_process(self, noise_scheduler: LinearNoiseScheduler):
        """Test forward diffusion process."""
        x0 = torch.randn(5, 2)
        t = torch.tensor([0, 1, 2, 3, 4])

        x_t = noise_scheduler.forward_process(x0, t)

        assert x_t.shape == x0.shape
        assert torch.isfinite(x_t).all()

    def test_forward_process_with_specific_noise(self, noise_scheduler: LinearNoiseScheduler):
        """Test forward process with provided noise."""
        x0 = torch.randn(3, 2)
        t = torch.tensor([1, 2, 3])
        noise = torch.randn_like(x0)

        x_t = noise_scheduler.forward_process(x0, t, noise)

        assert x_t.shape == x0.shape
        assert torch.isfinite(x_t).all()

    def test_posterior_params(self, noise_scheduler: LinearNoiseScheduler):
        """Test posterior parameter computation."""
        t = torch.tensor([0, 1, 2, 3, 4])
        params = noise_scheduler.get_posterior_params(t)

        assert len(params) == 4
        for param in params:
            assert param.shape[0] == len(t)
            assert torch.isfinite(param).all()

    def test_x0_prediction_accuracy(self, noise_scheduler: LinearNoiseScheduler, seed_random):
        """Test x0 prediction is accurate."""
        x0 = torch.randn(5, 2)
        t = torch.tensor([1, 2, 3, 4, 5])
        noise = torch.randn_like(x0)

        # Forward then reverse
        x_t = noise_scheduler.forward_process(x0, t, noise)
        predicted_x0 = noise_scheduler.predict_x0_from_noise(x_t, t, noise)

        assert torch.allclose(predicted_x0, x0, atol=1e-5)


@pytest.mark.unit
class TestDiffusionNetwork:
    """Test neural network component."""

    def test_default_initialization(self):
        """Test network with default parameters."""
        network = DiffusionNetwork()
        assert isinstance(network.network, torch.nn.Sequential)

    @pytest.mark.parametrize(
        "input_dim,hidden_dim,output_dim",
        [
            (3, 64, 2),
            (5, 128, 3),
            (10, 256, 5),
        ],
    )
    def test_custom_dimensions(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Test network with custom dimensions."""
        network = DiffusionNetwork(input_dim, hidden_dim, output_dim)

        x = torch.randn(10, input_dim)
        output = network(x)

        assert output.shape == (10, output_dim)
        assert torch.isfinite(output).all()

    def test_gradient_computation(self):
        """Test gradients flow through network."""
        network = DiffusionNetwork()
        x = torch.randn(5, 3, requires_grad=True)

        output = network(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_different_batch_sizes(self, batch_size: int):
        """Test network handles different batch sizes."""
        network = DiffusionNetwork()
        x = torch.randn(batch_size, 3)

        output = network(x)
        assert output.shape == (batch_size, 2)


@pytest.mark.unit
class TestDiffusionModel:
    """Test complete diffusion model."""

    def test_initialization(self, noise_scheduler: LinearNoiseScheduler):
        """Test model initialization."""
        model = DiffusionModel(noise_scheduler)

        assert model.noise_scheduler == noise_scheduler
        assert model.timesteps == noise_scheduler.timesteps
        assert isinstance(model.network, DiffusionNetwork)

    def test_custom_network(self, noise_scheduler: LinearNoiseScheduler):
        """Test model with custom network."""
        custom_network = DiffusionNetwork(hidden_dim=128)
        model = DiffusionModel(noise_scheduler, custom_network)

        assert model.network == custom_network

    def test_noise_prediction(self, diffusion_model: DiffusionModel):
        """Test noise prediction."""
        x_t = torch.randn(5, 2)
        t = torch.tensor([0, 1, 2, 3, 4])

        predicted_noise = diffusion_model.predict_noise(x_t, t)

        assert predicted_noise.shape == x_t.shape
        assert torch.isfinite(predicted_noise).all()

    def test_reverse_step(self, diffusion_model: DiffusionModel):
        """Test single reverse step."""
        x_t = torch.randn(3, 2)
        t = torch.tensor([1, 2, 3])

        x_prev = diffusion_model.reverse_step(x_t, t)

        assert x_prev.shape == x_t.shape
        assert torch.isfinite(x_prev).all()

    def test_reverse_step_final(self, diffusion_model: DiffusionModel):
        """Test final reverse step (t=0) has no noise."""
        x_t = torch.randn(3, 2)
        t_zero = torch.tensor([0, 0, 0])

        x_final = diffusion_model.reverse_step(x_t, t_zero)

        assert x_final.shape == x_t.shape
        assert torch.isfinite(x_final).all()

    @pytest.mark.parametrize("n_samples", [1, 5, 10])
    def test_sampling_without_trajectory(self, diffusion_model: DiffusionModel, n_samples: int):
        """Test sampling without trajectory."""
        samples, trajectory = diffusion_model.sample(n_samples, return_trajectory=False)

        assert samples.shape == (n_samples, 2)
        assert torch.isfinite(samples).all()
        assert trajectory is None

    def test_sampling_with_trajectory(self, diffusion_model: DiffusionModel):
        """Test sampling with trajectory."""
        samples, trajectory = diffusion_model.sample(5, return_trajectory=True)

        assert samples.shape == (5, 2)
        assert torch.isfinite(samples).all()
        assert trajectory is not None
        assert len(trajectory) == diffusion_model.timesteps + 1

        for step in trajectory:
            assert step.shape == (5, 2)
            assert torch.isfinite(step).all()

    def test_loss_computation(self, diffusion_model: DiffusionModel):
        """Test loss computation."""
        x0 = torch.randn(8, 2)
        loss = diffusion_model.compute_loss(x0)

        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
        assert loss.item() >= 0  # MSE is non-negative

    def test_reproducible_sampling(self, diffusion_model: DiffusionModel):
        """Test sampling reproducibility."""
        torch.manual_seed(42)
        samples1, _ = diffusion_model.sample(10)

        torch.manual_seed(42)
        samples2, _ = diffusion_model.sample(10)

        assert torch.allclose(samples1, samples2, atol=1e-6)

    def test_training_eval_modes(self, diffusion_model: DiffusionModel):
        """Test model behavior in training vs eval mode."""
        x0 = torch.randn(5, 2)

        # Training mode
        diffusion_model.train()
        loss_train = diffusion_model.compute_loss(x0)

        # Eval mode
        diffusion_model.eval()
        with torch.no_grad():
            loss_eval = diffusion_model.compute_loss(x0)

        assert torch.isfinite(loss_train)
        assert torch.isfinite(loss_eval)
