"""Enumeration for logging settings."""

from enum import Enum


class LogLevel(Enum):
    """Logging levels enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
