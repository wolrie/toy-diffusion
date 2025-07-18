"""Enums for different settings in the training process."""

from enum import StrEnum


class DeviceType(StrEnum):
    """Enumeration for different device types."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class SchedulerType(StrEnum):
    """Enumeration for different types of schedulers."""

    COSINE = "cosine"
    STEP = "step"
