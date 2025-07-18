"""Enums for different settings in the training process."""

from enum import Enum


class DeviceType(str, Enum):
    """Enumeration for different device types."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class SchedulerType(str, Enum):
    """Enumeration for different types of schedulers."""

    COSINE = "cosine"
    STEP = "step"
