"""Data Generator Interface - Defines contract for data generators."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class DataGeneratorInterface(ABC):
    """Interface for data generators."""

    @abstractmethod
    def generate(self, n_samples: int, **kwargs: Any) -> torch.Tensor:
        """Generate n_samples of data."""
        pass

    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the generated data."""
        pass
