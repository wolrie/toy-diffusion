"""Interfaces for training components."""

from dataclasses import dataclass
from typing import Union

from config.base import Config

from .enums import SchedulerType


@dataclass
class TrainingConfig(Config):
    """Training configuration."""

    n_epochs: int = 7000
    learning_rate: float = 1e-3
    batch_size: int = 128
    scheduler_type: Union[str, SchedulerType] = SchedulerType.COSINE
    scheduler_eta_min: float = 1e-5

    def validate(self) -> None:
        """Validate training configuration settings."""
        if self.n_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not isinstance(self.scheduler_type, SchedulerType):
            try:
                self.scheduler_type = SchedulerType(self.scheduler_type)
            except (ValueError, TypeError):
                raise ValueError(f"Scheduler type must be in {list(SchedulerType)}")
        if self.scheduler_eta_min < 0:
            raise ValueError("Scheduler eta_min must be non-negative")
