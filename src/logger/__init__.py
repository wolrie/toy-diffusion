"""Custom logging utilities and configuration."""

from .interface import LoggerFactoryInterface, LoggerInterface, StructuredLoggerInterface
from .logging import (
    LogLevel,
    configure_logging,
    critical,
    debug,
    error,
    get_logger,
    info,
    log_with_context,
    warning,
)

__all__ = [
    # Interfaces
    "LoggerInterface",
    "LoggerFactoryInterface",
    "StructuredLoggerInterface",
    # Implementation
    "LogLevel",
    "get_logger",
    "configure_logging",
    "log_with_context",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]
