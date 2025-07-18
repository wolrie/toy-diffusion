"""Base configuration interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Config(ABC):
    """Base class for all configurations."""

    def __post_init__(self) -> None:
        """Post-initialization hook for validation."""
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration settings."""
        pass
