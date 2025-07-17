"""
Swiss Roll Data Generator - Generates Swiss Roll datasets.

Follows Single Responsibility Principle - only handles Swiss Roll data generation.
"""

from typing import Any, Dict

import numpy as np
import torch

from .data_interface import DataGeneratorInterface


class SwissRollGenerator(DataGeneratorInterface):
    """
    Swiss Roll data generator.

    Follows Single Responsibility Principle - only generates Swiss Roll data.
    """

    def __init__(self, noise_level: float = 0.2, random_state: int = 42):
        self.noise_level = noise_level
        self.random_state = random_state
        self._setup_random_state()

    def _setup_random_state(self):
        """Setup random state for reproducibility."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def generate(self, n_samples: int, **kwargs) -> torch.Tensor:
        """
        Generate Swiss Roll data.

        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters (noise_level override)

        Returns:
            torch.Tensor: Generated Swiss Roll data of shape (n_samples, 2)
        """
        noise_level = kwargs.get("noise_level", self.noise_level)

        # Generate Swiss Roll using same logic as original
        t = torch.rand(n_samples) * 3 * np.pi + 0.5 * np.pi

        # Swiss roll parametric equations
        x = t * torch.cos(t)
        y = t * torch.sin(t)

        # Normalize to reasonable range
        x = (x - x.mean()) / x.std() * 0.5
        y = (y - y.mean()) / y.std() * 0.5

        # Add noise
        if noise_level > 0:
            noise_x = torch.randn(n_samples) * noise_level
            noise_y = torch.randn(n_samples) * noise_level
            x += noise_x
            y += noise_y

        # Stack and return
        data = torch.stack([x, y], dim=1)
        return data.float()

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the Swiss Roll data generation parameters."""
        return {
            "data_type": "swiss_roll",
            "noise_level": self.noise_level,
            "random_state": self.random_state,
            "dimensions": 2,
            "description": "Swiss Roll manifold data with Gaussian noise",
        }
