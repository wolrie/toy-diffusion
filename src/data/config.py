"""Configuration for data generation in the diffusion model."""

from dataclasses import dataclass

from config.base import Config


@dataclass
class DataConfig(Config):
    """Data generation configuration."""

    n_data_points: int = 2000
    noise_level: float = 0.05
    random_state: int = 42

    def validate(self) -> None:
        """Validate data configuration settings."""
        if self.n_data_points <= 0:
            raise ValueError("Number of data points must be positive")
        if self.noise_level < 0:
            raise ValueError("Noise level must be non-negative")
        if not isinstance(self.random_state, int):
            raise ValueError("Random state must be an integer")
