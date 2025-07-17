"""Diffusion Trainer - Handles training orchestration."""

from typing import List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.config import TrainingConfig
from domain.diffusion_model import DiffusionModel

from .scheduler_factory import SchedulerFactory


class TrainingMetrics:
    """Container for training metrics."""

    def __init__(self) -> None:
        """Initialize training metrics."""
        self.losses: List[float] = []
        self.learning_rates: List[float] = []

    def add_epoch_metrics(self, loss: float, lr: float):
        """Add metrics for an epoch."""
        self.losses.append(loss)
        self.learning_rates.append(lr)

    def get_final_loss(self) -> float:
        """Get the final training loss."""
        return self.losses[-1] if self.losses else float("inf")


class DiffusionTrainer:
    """Trainer for diffusion models."""

    def __init__(self, model: DiffusionModel, config: TrainingConfig) -> None:
        """Initialize the trainer with a model and configuration."""
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = self._create_scheduler()
        self.metrics = TrainingMetrics()

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler using factory."""
        return SchedulerFactory.create_scheduler(
            self.config.scheduler_type,
            self.optimizer,
            T_max=self.config.n_epochs,
            eta_min=self.config.scheduler_eta_min,
        )

    def train(self, data: torch.Tensor, verbose: bool = True) -> TrainingMetrics:
        """Train the diffusion model.

        Args:
            data: Training data
            verbose: Whether to show progress bar

        Returns:
            TrainingMetrics: Training metrics
        """
        dataloader = DataLoader(
            TensorDataset(data),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Training loop
        self.model.train()
        iterator = (
            tqdm(range(self.config.n_epochs), desc="Training")
            if verbose
            else range(self.config.n_epochs)
        )

        for epoch in iterator:
            epoch_loss = self._train_epoch(dataloader)
            self.scheduler.step()

            # Record metrics
            current_lr = self.scheduler.get_last_lr()[0]
            self.metrics.add_epoch_metrics(epoch_loss, current_lr)

            # Update progress bar
            if verbose and epoch % 500 == 0:
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        {
                            "loss": f"{epoch_loss:.4f}",
                            "lr": f"{current_lr:.6f}",
                        }
                    )
                else:
                    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, LR = {current_lr:.6f}")

        return self.metrics

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        epoch_loss = 0.0
        for batch in dataloader:
            batch_data = batch[0]
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch_data)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def evaluate_sample_quality(self, original_data: torch.Tensor, n_samples: int) -> dict:
        """Evaluate sample quality against original data.

        Args:
            original_data: Original training data
            n_samples: Number of samples to generate

        Returns:
            dict: Quality metrics
        """
        self.model.eval()
        with torch.no_grad():
            samples, _ = self.model.sample(n_samples)

        # Compute quality metrics
        original_mean = original_data.mean(dim=0)
        original_std = original_data.std(dim=0)
        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)

        mean_error = torch.norm(original_mean - sample_mean).item()
        std_error = torch.norm(original_std - sample_std).item()

        return {
            "mean_error": mean_error,
            "std_error": std_error,
            "original_mean": original_mean.tolist(),
            "original_std": original_std.tolist(),
            "sample_mean": sample_mean.tolist(),
            "sample_std": sample_std.tolist(),
        }
