"""Structured logging system for the diffusion model project."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .enums import LogLevel


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class DiffusionLogger:
    """Main logger class for the diffusion model project."""

    _instance: Optional[DiffusionLogger] = None
    _loggers: Dict[str, logging.Logger] = {}

    def __new__(cls) -> DiffusionLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._setup_root_logger()

    def _setup_root_logger(self) -> None:
        """Setup the root logger with default configuration."""
        root_logger = logging.getLogger("diffusion")
        root_logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        self._add_console_handler(root_logger)

    def _add_console_handler(self, logger: logging.Logger, use_json: bool = False) -> None:
        """Add console handler to logger."""
        console_handler = logging.StreamHandler(sys.stdout)

        if use_json:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        logger.addHandler(console_handler)

    def _add_file_handler(
        self, logger: logging.Logger, log_file: Path, use_json: bool = True
    ) -> None:
        """Add file handler to logger."""
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)

        if use_json:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(f"diffusion.{name}")
        return self._loggers[name]

    def configure(
        self,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        use_json_format: bool = False,
        enable_console: bool = True,
    ) -> None:
        """Configure the logging system."""
        root_logger = logging.getLogger("diffusion")
        root_logger.setLevel(getattr(logging, level.value))

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler if enabled
        if enable_console:
            self._add_console_handler(root_logger, use_json_format)

        # Add file handler if specified
        if log_file:
            self._add_file_handler(root_logger, log_file, use_json_format)

    def log_with_context(
        self, logger_name: str, level: LogLevel, message: str, **context: Any
    ) -> None:
        """Log a message with additional context."""
        logger = self.get_logger(logger_name)
        record = logging.LogRecord(
            name=logger.name,
            level=getattr(logging, level.value),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.extra_fields = context
        logger.handle(record)


# Global logger instance
_logger_instance = DiffusionLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: The name of the logger (typically module name)

    Returns:
        A configured logger instance
    """
    return _logger_instance.get_logger(name)


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[Path] = None,
    use_json_format: bool = False,
    enable_console: bool = True,
) -> None:
    """Configure the global logging system."""
    _logger_instance.configure(level, log_file, use_json_format, enable_console)


def log_with_context(logger_name: str, level: LogLevel, message: str, **context: Any) -> None:
    """Log a message with additional context."""
    _logger_instance.log_with_context(logger_name, level, message, **context)


# Convenience functions
def debug(message: str, logger_name: str = "main", **context: Any) -> None:
    """Log a debug message."""
    log_with_context(logger_name, LogLevel.DEBUG, message, **context)


def info(message: str, logger_name: str = "main", **context: Any) -> None:
    """Log an info message."""
    log_with_context(logger_name, LogLevel.INFO, message, **context)


def warning(message: str, logger_name: str = "main", **context: Any) -> None:
    """Log a warning message."""
    log_with_context(logger_name, LogLevel.WARNING, message, **context)


def error(message: str, logger_name: str = "main", **context: Any) -> None:
    """Log an error message."""
    log_with_context(logger_name, LogLevel.ERROR, message, **context)


def critical(message: str, logger_name: str = "main", **context: Any) -> None:
    """Log a critical message."""
    log_with_context(logger_name, LogLevel.CRITICAL, message, **context)
