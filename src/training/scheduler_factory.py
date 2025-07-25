"""Scheduler Factory - Creates learning rate schedulers."""

from typing import Any, Callable, Dict, Union

import torch

from exceptions import TrainingError

from .enums import SchedulerType

KeyType = Union[str, SchedulerType]


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""

    _schedulers: Dict[KeyType, Callable] = {}

    @classmethod
    def register_scheduler(cls, name: str, scheduler_fn: Callable) -> None:
        """Register a new scheduler type."""
        cls._schedulers[name] = scheduler_fn

    @classmethod
    def create_scheduler(
        cls, name: str, optimizer: torch.optim.Optimizer, **kwargs: Any
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create a scheduler by name."""
        if name not in cls._schedulers:
            raise TrainingError(f"Unknown scheduler type: {name}")

        scheduler: torch.optim.lr_scheduler._LRScheduler = cls._schedulers[name](
            optimizer, **kwargs
        )
        return scheduler


# Register default schedulers
def _create_cosine_scheduler(
    optimizer: torch.optim.Optimizer, **kwargs: Any
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Create cosine annealing scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=kwargs.get("T_max", 1000),
        eta_min=kwargs.get("eta_min", 1e-5),
    )


def _create_step_scheduler(
    optimizer: torch.optim.Optimizer, **kwargs: Any
) -> torch.optim.lr_scheduler.StepLR:
    """Create step scheduler."""
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=kwargs.get("step_size", 1000),
        gamma=kwargs.get("gamma", 0.5),
    )


# Register schedulers
SchedulerFactory.register_scheduler(SchedulerType.COSINE, _create_cosine_scheduler)
SchedulerFactory.register_scheduler(SchedulerType.STEP, _create_step_scheduler)
