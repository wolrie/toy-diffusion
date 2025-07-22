"""Diffusion Model - Core domain logic for diffusion models."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise_scheduler import NoiseSchedulerInterface


class DiffusionNetwork(nn.Module):
    """Neural network for noise prediction."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, output_dim: int = 2) -> None:
        """Initialize the diffusion network."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.network(x)
        return output


class DiffusionModel(nn.Module):
    """Main diffusion model implementing DDPM algorithm."""

    def __init__(
        self,
        noise_scheduler: NoiseSchedulerInterface,
        network: Optional[DiffusionNetwork] = None,
    ) -> None:
        """Initialize the diffusion model with a noise scheduler and optional network."""
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.network = network or DiffusionNetwork()
        self.timesteps = noise_scheduler.timesteps

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise that was added to create x_t."""
        t_normalized = t.float() / self.timesteps
        network_input = torch.cat([x_t, t_normalized.unsqueeze(1)], dim=1)
        predicted_noise: torch.Tensor = self.network(network_input)
        return predicted_noise

    def reverse_step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse diffusion step."""
        # Predict noise
        predicted_noise = self.predict_noise(x_t, t)

        # Get posterior parameters from noise scheduler
        (
            sqrt_recip_alphas_t,
            betas_t,
            sqrt_one_minus_alphas_cumprod_t,
            posterior_variance_t,
        ) = self.noise_scheduler.get_posterior_params(t)

        # Compute posterior mean
        posterior_mean = sqrt_recip_alphas_t * (
            x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise
        )

        # Add noise for t > 0
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return posterior_mean

    def sample(
        self, n_samples: int, return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Generate samples from the model."""
        x = torch.randn(n_samples, 2)
        trajectory = [x.clone()] if return_trajectory else None

        for i in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), i, dtype=torch.long)
            x = self.reverse_step(x, t)

            if return_trajectory and trajectory is not None:
                trajectory.append(x.clone())

        return (x, trajectory) if return_trajectory else (x, None)

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute training loss (noise prediction MSE)."""
        batch_size = x0.shape[0]

        # Random timestep
        t = torch.randint(0, self.timesteps, (batch_size,))

        # Add noise
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.forward_process(x0, t, noise)

        # Predict noise
        predicted_noise = self.predict_noise(x_t, t)

        # MSE loss between actual and predicted noise
        return F.mse_loss(predicted_noise, noise)
