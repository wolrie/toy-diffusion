"""Logging interface definitions."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Protocol

from .logging import LogLevel


class LoggerInterface(Protocol):
    """Protocol for logger instances."""

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        ...

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        ...

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        ...


class LoggerFactoryInterface(ABC):
    """Abstract interface for logger factories."""

    @abstractmethod
    def get_logger(self, name: str) -> LoggerInterface:
        """Get or create a logger with the given name."""
        pass

    @abstractmethod
    def configure(
        self,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        use_json_format: bool = False,
        enable_console: bool = True,
    ) -> None:
        """Configure the logging system."""
        pass


class StructuredLoggerInterface(LoggerInterface, Protocol):
    """Protocol for structured loggers with context support."""

    def log_with_context(self, level: LogLevel, message: str, **context: Any) -> None:
        """Log a message with additional context."""
        ...
