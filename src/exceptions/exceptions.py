"""Simple exception hierarchy."""


class DiffusionError(Exception):
    """Base exception for all diffusion model errors."""

    pass


class ConfigurationError(DiffusionError):
    """Raised when there's an error with configuration."""

    pass


class DataError(DiffusionError):
    """Raised when there's an error with data generation or processing."""

    pass


class ModelError(DiffusionError):
    """Raised when there's an error with the diffusion model."""

    pass


class TrainingError(DiffusionError):
    """Raised when there's an error during training."""

    pass


class VisualizationError(DiffusionError):
    """Raised when there's an error with visualization."""

    pass
