"""Noise Scheduler - Handles noise schedule and forward/reverse process computations."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class NoiseSchedulerInterface(ABC):
    """Interface for noise schedulers."""

    @property
    @abstractmethod
    def timesteps(self) -> int:
        """Number of diffusion timesteps."""
        pass

    @abstractmethod
    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to clean data."""
        pass

    @abstractmethod
    def get_posterior_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get parameters for posterior distribution."""
        pass


class LinearNoiseScheduler(nn.Module, NoiseSchedulerInterface):
    """Linear noise scheduler implementation."""

    def __init__(self, timesteps: int, beta_min: float = 0.0004, beta_max: float = 0.04) -> None:
        """Initialize the linear noise scheduler."""
        super().__init__()
        self._timesteps = timesteps

        # Linear noise schedule
        self.register_buffer("betas", torch.linspace(beta_min, beta_max, self._timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

        # Forward process parameters
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )

        # Reverse process parameters
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))

        # Posterior variance computation
        alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )

    @property
    def timesteps(self) -> int:
        """Number of diffusion timesteps."""
        return self._timesteps

    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to clean data according to forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def get_posterior_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean coefficients and variance for reverse process."""
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].unsqueeze(1)
        betas_t = self.betas[t].unsqueeze(1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        posterior_variance_t = self.posterior_variance[t].unsqueeze(1)

        return (
            sqrt_recip_alphas_t,
            betas_t,
            sqrt_one_minus_alphas_cumprod_t,
            posterior_variance_t,
        )

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """Recover x0 from noise prediction."""
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

        return (x_t - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
