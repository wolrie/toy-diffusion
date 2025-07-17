"""
Plot Manager - Handles static plot creation and management.

Follows Single Responsibility Principle - only handles static plotting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.trainer import TrainingMetrics

# # Import TrainingMetrics with fallback for different execution contexts
# import sys
# import os

# # Add parent directory to path if needed
# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

# try:
#     from training.trainer import TrainingMetrics
# except ImportError:
#     try:
#         from ..training.trainer import TrainingMetrics
#     except ImportError:
#         # Last resort - add src directory
#         src_dir = os.path.dirname(os.path.dirname(__file__))
#         sys.path.insert(0, src_dir)
#         from training.trainer import TrainingMetrics


class PlotManager:
    """Manages static plot creation for diffusion model results."""

    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 150) -> None:
        """Initialize plot manager.

        Args:
            figsize: Default figure size for plots
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.default_style = {
            "alpha": 0.6,
            "scatter_size": 20,
            "grid_alpha": 0.3,
            "xlim": (-1.5, 1.5),
            "ylim": (-1.5, 1.5),
        }

    def create_data_comparison_plot(
        self,
        original_data: torch.Tensor,
        generated_data: torch.Tensor,
        title: str = "Data Comparison",
    ) -> plt.Figure:
        """Create a comparison plot between original and generated data.

        Args:
            original_data: Original training data
            generated_data: Generated samples
            title: Plot title

        Returns:
            matplotlib.pyplot.Figure: The created figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Convert to numpy
        orig_np = original_data.numpy() if torch.is_tensor(original_data) else original_data
        gen_np = generated_data.numpy() if torch.is_tensor(generated_data) else generated_data

        # Original data
        axes[0].scatter(
            orig_np[:, 0],
            orig_np[:, 1],
            alpha=self.default_style["alpha"],
            s=self.default_style["scatter_size"],
            c="blue",
        )
        axes[0].set_title("Original Data")
        axes[0].set_xlim(self.default_style["xlim"])
        axes[0].set_ylim(self.default_style["ylim"])
        axes[0].grid(True, alpha=self.default_style["grid_alpha"])

        # Generated data
        axes[1].scatter(
            gen_np[:, 0],
            gen_np[:, 1],
            alpha=self.default_style["alpha"],
            s=self.default_style["scatter_size"],
            c="red",
        )
        axes[1].set_title("Generated Samples")
        axes[1].set_xlim(self.default_style["xlim"])
        axes[1].set_ylim(self.default_style["ylim"])
        axes[1].grid(True, alpha=self.default_style["grid_alpha"])

        # Overlay comparison
        axes[2].scatter(
            orig_np[:, 0],
            orig_np[:, 1],
            alpha=0.5,
            s=25,
            c="blue",
            label="Original",
        )
        axes[2].scatter(
            gen_np[:, 0],
            gen_np[:, 1],
            alpha=0.5,
            s=25,
            c="red",
            label="Generated",
        )
        axes[2].set_title("Overlay Comparison")
        axes[2].set_xlim(self.default_style["xlim"])
        axes[2].set_ylim(self.default_style["ylim"])
        axes[2].legend()
        axes[2].grid(True, alpha=self.default_style["grid_alpha"])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig

    def create_training_metrics_plot(
        self, metrics: TrainingMetrics, title: str = "Training Metrics"
    ) -> plt.Figure:
        """Create training metrics visualization.

        Args:
            metrics: Training metrics object
            title: Plot title

        Returns:
            matplotlib.pyplot.Figure: The created figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        axes[0].plot(metrics.losses, color="blue", linewidth=2)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=self.default_style["grid_alpha"])

        # Learning rate curve
        axes[1].plot(metrics.learning_rates, color="green", linewidth=2)
        axes[1].set_title("Learning Rate Schedule")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=self.default_style["grid_alpha"])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig

    def create_distribution_comparison_plot(
        self,
        original_data: torch.Tensor,
        generated_data: torch.Tensor,
        title: str = "Distribution Comparison",
    ) -> plt.Figure:
        """Create distribution comparison plots.

        Args:
            original_data: Original training data
            generated_data: Generated samples
            title: Plot title

        Returns:
            matplotlib.pyplot.Figure: The created figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Convert to numpy
        orig_np = original_data.numpy() if torch.is_tensor(original_data) else original_data
        gen_np = generated_data.numpy() if torch.is_tensor(generated_data) else generated_data

        # X distribution
        axes[0].hist(
            orig_np[:, 0],
            bins=30,
            alpha=0.7,
            density=True,
            label="Original X",
            color="blue",
        )
        axes[0].hist(
            gen_np[:, 0],
            bins=30,
            alpha=0.7,
            density=True,
            label="Generated X",
            color="red",
        )
        axes[0].set_title("X Distribution")
        axes[0].legend()
        axes[0].grid(True, alpha=self.default_style["grid_alpha"])

        # Y distribution
        axes[1].hist(
            orig_np[:, 1],
            bins=30,
            alpha=0.7,
            density=True,
            label="Original Y",
            color="blue",
        )
        axes[1].hist(
            gen_np[:, 1],
            bins=30,
            alpha=0.7,
            density=True,
            label="Generated Y",
            color="red",
        )
        axes[1].set_title("Y Distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=self.default_style["grid_alpha"])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig

    def create_trajectory_plot(
        self,
        trajectory: List[torch.Tensor],
        original_data: Optional[torch.Tensor] = None,
        title: str = "Denoising Trajectory",
    ) -> plt.Figure:
        """Create trajectory visualization with color-coded timesteps.

        Args:
            trajectory: List of tensors representing denoising trajectory
            original_data: Optional original data for background
            title: Plot title

        Returns:
            matplotlib.pyplot.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot original data as background if provided
        if original_data is not None:
            data_np = original_data.numpy() if torch.is_tensor(original_data) else original_data
            ax.scatter(
                data_np[:, 0],
                data_np[:, 1],
                alpha=0.3,
                s=10,
                c="lightblue",
                label="Original Data",
            )

        # Plot trajectory with color coding
        timesteps = np.arange(len(trajectory))

        for i, traj_step in enumerate(trajectory):
            traj_np = traj_step.numpy() if torch.is_tensor(traj_step) else traj_step
            colors = np.full(traj_np.shape[0], timesteps[i])

            scatter = ax.scatter(
                traj_np[:, 0],
                traj_np[:, 1],
                alpha=0.5,
                s=15,
                c=colors,
                cmap="viridis",
                vmin=0,
                vmax=len(trajectory) - 1,
            )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Timestep", rotation=270, labelpad=15)

        ax.set_title(title)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=self.default_style["grid_alpha"])

        if original_data is not None:
            ax.legend()

        plt.tight_layout()

        return fig

    def create_comprehensive_results_plot(
        self,
        original_data: torch.Tensor,
        generated_data: torch.Tensor,
        trajectory: List[torch.Tensor],
        metrics: TrainingMetrics,
        title: str = "Comprehensive Training Results",
    ) -> plt.Figure:
        """Create a comprehensive results plot with all visualizations.

        Args:
            original_data: Original training data
            generated_data: Generated samples
            trajectory: Denoising trajectory
            metrics: Training metrics
            title: Plot title

        Returns:
            matplotlib.pyplot.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Convert to numpy
        orig_np = original_data.numpy() if torch.is_tensor(original_data) else original_data
        gen_np = generated_data.numpy() if torch.is_tensor(generated_data) else generated_data

        # Original data
        axes[0, 0].scatter(orig_np[:, 0], orig_np[:, 1], alpha=0.6, s=20, c="blue")
        axes[0, 0].set_title("Original Data")
        axes[0, 0].set_xlim(self.default_style["xlim"])
        axes[0, 0].set_ylim(self.default_style["ylim"])
        axes[0, 0].grid(True, alpha=self.default_style["grid_alpha"])

        # Generated samples
        axes[0, 1].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.6, s=20, c="red")
        axes[0, 1].set_title("Generated Samples")
        axes[0, 1].set_xlim(self.default_style["xlim"])
        axes[0, 1].set_ylim(self.default_style["ylim"])
        axes[0, 1].grid(True, alpha=self.default_style["grid_alpha"])

        # Overlay comparison
        axes[0, 2].scatter(
            orig_np[:, 0],
            orig_np[:, 1],
            alpha=0.5,
            s=25,
            c="blue",
            label="Original",
        )
        axes[0, 2].scatter(
            gen_np[:, 0],
            gen_np[:, 1],
            alpha=0.5,
            s=25,
            c="red",
            label="Generated",
        )
        axes[0, 2].set_title("Overlay Comparison")
        axes[0, 2].set_xlim(self.default_style["xlim"])
        axes[0, 2].set_ylim(self.default_style["ylim"])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=self.default_style["grid_alpha"])

        # Trajectory with colorbar
        timesteps = np.arange(len(trajectory))
        for i, traj_step in enumerate(trajectory):
            traj_np = traj_step.numpy() if torch.is_tensor(traj_step) else traj_step
            colors = np.full(traj_np.shape[0], timesteps[i])
            scatter = axes[1, 0].scatter(
                traj_np[:, 0],
                traj_np[:, 1],
                alpha=0.5,
                s=15,
                c=colors,
                cmap="viridis",
                vmin=0,
                vmax=len(trajectory) - 1,
            )

        axes[1, 0].set_title("Denoising Trajectory")
        axes[1, 0].set_xlim(-2, 2)
        axes[1, 0].set_ylim(-2, 2)
        axes[1, 0].grid(True, alpha=self.default_style["grid_alpha"])

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label("Timestep", rotation=270, labelpad=15)

        # Loss curve
        axes[1, 1].plot(metrics.losses)
        axes[1, 1].set_title("Training Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=self.default_style["grid_alpha"])

        # Distribution comparison
        axes[1, 2].hist(
            orig_np[:, 0],
            bins=30,
            alpha=0.7,
            density=True,
            label="Original X",
            color="blue",
        )
        axes[1, 2].hist(
            gen_np[:, 0],
            bins=30,
            alpha=0.7,
            density=True,
            label="Generated X",
            color="red",
        )
        axes[1, 2].set_title("X Distribution")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=self.default_style["grid_alpha"])

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        return fig

    def create_denoising_progression_strip(
        self,
        trajectory: List[torch.Tensor],
        original_data: torch.Tensor,
        n_frames: int = 5,
        title: str = "Denoising Progression",
    ) -> plt.Figure:
        """Create a horizontal strip showing denoising progression at different time steps.

        Args:
            trajectory: List of tensors representing denoising steps
            original_data: Original training data for background
            n_frames: Number of frames to show in the strip
            title: Title for the entire figure

        Returns:
            plt.Figure: The created figure
        """
        # Sample frames from trajectory
        total_steps = len(trajectory)
        if n_frames > total_steps:
            n_frames = total_steps

        frame_indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)
        sampled_trajectory = [trajectory[i] for i in frame_indices]

        # Convert data to numpy
        data_np = original_data.numpy() if torch.is_tensor(original_data) else original_data

        # Create figure with subplots in a row
        fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4))

        # Handle case where n_frames = 1 (axes is not an array)
        if n_frames == 1:
            axes = [axes]

        for i, (ax, frame_idx) in enumerate(zip(axes, frame_indices)):
            # Set up subplot
            ax.set_xlim(self.default_style["xlim"])
            ax.set_ylim(self.default_style["ylim"])
            ax.grid(True, alpha=self.default_style["grid_alpha"])

            # Plot original data as background
            ax.scatter(
                data_np[:, 0],
                data_np[:, 1],
                alpha=0.3,
                s=self.default_style["scatter_size"],
                c="blue",
                label="Original Data",
            )

            # Plot current trajectory step
            current_traj = sampled_trajectory[i]
            current_traj_np = (
                current_traj.numpy() if torch.is_tensor(current_traj) else current_traj
            )

            ax.scatter(
                current_traj_np[:, 0],
                current_traj_np[:, 1],
                alpha=0.7,
                s=self.default_style["scatter_size"],
                c="red",
                label="Generated Samples",
            )

            # Set title for each subplot
            ax.set_title(f"Step {frame_idx}")

            # Add labels only to the first subplot
            if i == 0:
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.legend(loc="upper right")
            else:
                ax.set_xlabel("X")
                ax.set_ylabel("")

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        return fig

    def save_figure(self, fig: plt.Figure, filepath: Path, close_after_save: bool = True) -> None:
        """Save figure to file.

        Args:
            fig: Matplotlib figure to save
            filepath: Path to save the figure
            close_after_save: Whether to close the figure after saving
        """
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")

        if close_after_save:
            plt.close(fig)

    def set_style(self, style_dict: Dict[str, Any]) -> None:
        """Update default style settings.

        Args:
            style_dict: Dictionary with style parameters to update
        """
        self.default_style.update(style_dict)
