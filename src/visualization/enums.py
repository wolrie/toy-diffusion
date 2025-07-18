"""Enums for different settings during visualization."""

from enum import StrEnum


class GifType(StrEnum):
    """Enumeration for different types of GIFs."""

    STANDARD = "standard"
    SIDE_BY_SIDE = "side_by_side"
    PROGRESSION = "progression"
