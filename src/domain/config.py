"""Configuration for the diffusion model's architecture and training settings."""

from dataclasses import dataclass

from config.base import Config


@dataclass
class ModelConfig(Config):
    """Model architecture configuration."""

    timesteps: int = 100
    beta_min: float = 0.0004
    beta_max: float = 0.04
    hidden_dim: int = 256

    def validate(self) -> None:
        """Validate model configuration settings."""
        if self.timesteps <= 0:
            raise ValueError("Timesteps must be positive")
        if self.beta_min <= 0 or self.beta_max <= 0:
            raise ValueError("Beta values must be positive")
        if self.beta_min >= self.beta_max:
            raise ValueError("Beta_min must be less than beta_max")
        if self.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")
