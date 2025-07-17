"""Diffusion Visualizer - Main interface for all visualization tasks."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config.config import VisualizationConfig
from training.trainer import TrainingMetrics

from .gif_creator import GifCreator
from .plot_manager import PlotManager

# Import modules with fallback for different execution contexts
# import sys
# import os

# Add parent directory to path if needed
# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)


# try:
#     from visualization.plot_manager import PlotManager
#     from visualization.gif_creator import GifCreator
# except ImportError:
#     try:
#         from .plot_manager import PlotManager
#         from .gif_creator import GifCreator
#         from ..training.trainer import TrainingMetrics
#         from ..config.config import VisualizationConfig
#     except ImportError:
#         # Last resort - add src directory
#         src_dir = os.path.dirname(os.path.dirname(__file__))
#         sys.path.insert(0, src_dir)
#         from visualization.plot_manager import PlotManager
#         from visualization.gif_creator import GifCreator
#         from training.trainer import TrainingMetrics
#         from config.config import VisualizationConfig


class DiffusionVisualizer:
    """Main visualizer interface for diffusion model results."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize visualizer with configuration.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.plot_manager = PlotManager(figsize=(15, 10), dpi=config.display.figure_dpi)
        self.gif_creator = GifCreator(
            figsize=(10, 8), fps=config.gif.fps, dpi=config.display.figure_dpi
        )

    def create_training_results(
        self,
        original_data: torch.Tensor,
        generated_samples: torch.Tensor,
        trajectory: Optional[List[torch.Tensor]],
        metrics: TrainingMetrics,
        output_dir: Path,
        experiment_name: str = "training_results",
    ) -> Dict[str, Path]:
        """Create comprehensive training results visualization.

        Args:
            original_data: Original training data
            generated_samples: Generated samples from the model
            trajectory: Optional denoising trajectory
            metrics: Training metrics
            output_dir: Directory to save visualizations
            experiment_name: Name for the experiment files

        Returns:
            Dict[str, Path]: Dictionary mapping visualization types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        print("📊 Creating training results visualizations...")

        # 1. Comprehensive results plot
        if trajectory is not None:
            fig = self.plot_manager.create_comprehensive_results_plot(
                original_data,
                generated_samples,
                trajectory,
                metrics,
                title=f"Training Results: {experiment_name}",
            )
            results_path = output_dir / f"{experiment_name}_comprehensive.png"
            self.plot_manager.save_figure(fig, results_path)
            results["comprehensive"] = results_path
            print(f"   Comprehensive results: {results_path}")

        # 2. Data comparison plot
        fig = self.plot_manager.create_data_comparison_plot(
            original_data,
            generated_samples,
            title=f"Data Comparison: {experiment_name}",
        )
        comparison_path = output_dir / f"{experiment_name}_comparison.png"
        self.plot_manager.save_figure(fig, comparison_path)
        results["comparison"] = comparison_path
        print(f"   Data comparison: {comparison_path}")

        # 3. Training metrics plot
        fig = self.plot_manager.create_training_metrics_plot(
            metrics, title=f"Training Metrics: {experiment_name}"
        )
        metrics_path = output_dir / f"{experiment_name}_metrics.png"
        self.plot_manager.save_figure(fig, metrics_path)
        results["metrics"] = metrics_path
        print(f"   Training metrics: {metrics_path}")

        # 4. Distribution comparison
        fig = self.plot_manager.create_distribution_comparison_plot(
            original_data,
            generated_samples,
            title=f"Distribution Comparison: {experiment_name}",
        )
        distribution_path = output_dir / f"{experiment_name}_distributions.png"
        self.plot_manager.save_figure(fig, distribution_path)
        results["distributions"] = distribution_path
        print(f"   Distributions: {distribution_path}")

        # 5. Trajectory plot (if available)
        if trajectory is not None:
            fig = self.plot_manager.create_trajectory_plot(
                trajectory,
                original_data,
                title=f"Denoising Trajectory: {experiment_name}",
            )
            trajectory_path = output_dir / f"{experiment_name}_trajectory.png"
            self.plot_manager.save_figure(fig, trajectory_path)
            results["trajectory"] = trajectory_path
            print(f"   Trajectory plot: {trajectory_path}")

        return results

    def create_denoising_gif(
        self,
        trajectory: List[torch.Tensor],
        original_data: torch.Tensor,
        output_dir: Path,
        experiment_name: str = "denoising",
        gif_type: str = "standard",
    ) -> Path:
        """Create denoising trajectory GIF.

        Args:
            trajectory: Denoising trajectory
            original_data: Original training data
            output_dir: Directory to save GIF
            experiment_name: Name for the experiment files
            gif_type: Type of GIF ('standard', 'side_by_side', 'progression')

        Returns:
            Path: Path to the created GIF
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gif_path = output_dir / f"{experiment_name}_trajectory.gif"

        print(f"🎬 Creating {gif_type} denoising GIF...")

        if gif_type == "standard":
            return self.gif_creator.create_trajectory_gif(
                trajectory,
                original_data,
                gif_path,
                n_frames=self.config.gif.n_frames,
            )
        elif gif_type == "side_by_side":
            return self.gif_creator.create_side_by_side_gif(
                trajectory,
                original_data,
                gif_path,
                n_frames=self.config.gif.n_frames,
            )
        elif gif_type == "progression":
            return self.gif_creator.create_progression_gif(
                trajectory,
                original_data,
                gif_path,
                n_frames=self.config.gif.n_frames,
            )
        else:
            raise ValueError(f"Unknown GIF type: {gif_type}")

    def create_denoising_progression_strip(
        self,
        trajectory: List[torch.Tensor],
        original_data: torch.Tensor,
        output_dir: Path,
        experiment_name: str = "denoising_progression",
        n_frames: int = 5,
    ) -> Path:
        """Create a horizontal strip showing denoising progression at different time steps.

        Args:
            trajectory: Denoising trajectory
            original_data: Original training data
            output_dir: Directory to save visualization
            experiment_name: Name for the experiment files
            n_frames: Number of frames to show in the strip

        Returns:
            Path: Path to the saved visualization
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 Creating denoising progression strip with {n_frames} frames...")

        # Create the strip visualization
        fig = self.plot_manager.create_denoising_progression_strip(
            trajectory=trajectory,
            original_data=original_data,
            n_frames=n_frames,
            title="Denoising Progression",
        )

        # Save the figure
        output_path = output_dir / f"{experiment_name}_strip.png"
        self.plot_manager.save_figure(fig, output_path)

        print(f"   Progression strip saved to: {output_path}")
        return output_path

    def create_comparison_visualization(
        self,
        datasets: Dict[str, torch.Tensor],
        output_dir: Path,
        title: str = "Dataset Comparison",
    ) -> Path:
        """Create comparison visualization for multiple datasets.

        Args:
            datasets: Dictionary mapping dataset names to data tensors
            output_dir: Directory to save visualization
            title: Plot title

        Returns:
            Path: Path to the saved comparison plot
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))

        if n_datasets == 1:
            axes = [axes]

        colors = ["blue", "red", "green", "orange", "purple", "brown"]

        for i, (name, data) in enumerate(datasets.items()):
            data_np = data.numpy() if torch.is_tensor(data) else data
            color = colors[i % len(colors)]

            axes[i].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, s=20, c=color)
            axes[i].set_title(name)
            axes[i].set_xlim(-1.5, 1.5)
            axes[i].set_ylim(-1.5, 1.5)
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        comparison_path = output_dir / "dataset_comparison.png"
        self.plot_manager.save_figure(fig, comparison_path)

        print(f"   Dataset comparison: {comparison_path}")
        return comparison_path

    def create_quality_metrics_visualization(
        self,
        quality_metrics: Dict[str, Any],
        output_dir: Path,
        experiment_name: str = "quality_metrics",
    ) -> Path:
        """Create visualization for quality metrics.

        Args:
            quality_metrics: Dictionary with quality metrics
            output_dir: Directory to save visualization
            experiment_name: Name for the experiment files

        Returns:
            Path: Path to the saved metrics plot
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Mean comparison
        original_mean = quality_metrics["original_mean"]
        sample_mean = quality_metrics["sample_mean"]

        x_labels = ["X", "Y"]
        x_pos = range(len(x_labels))

        axes[0].bar(
            [x - 0.2 for x in x_pos],
            original_mean,
            0.4,
            label="Original",
            alpha=0.7,
        )
        axes[0].bar(
            [x + 0.2 for x in x_pos],
            sample_mean,
            0.4,
            label="Generated",
            alpha=0.7,
        )
        axes[0].set_xlabel("Dimension")
        axes[0].set_ylabel("Mean Value")
        axes[0].set_title("Mean Comparison")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(x_labels)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Standard deviation comparison
        original_std = quality_metrics["original_std"]
        sample_std = quality_metrics["sample_std"]

        axes[1].bar(
            [x - 0.2 for x in x_pos],
            original_std,
            0.4,
            label="Original",
            alpha=0.7,
        )
        axes[1].bar(
            [x + 0.2 for x in x_pos],
            sample_std,
            0.4,
            label="Generated",
            alpha=0.7,
        )
        axes[1].set_xlabel("Dimension")
        axes[1].set_ylabel("Standard Deviation")
        axes[1].set_title("Standard Deviation Comparison")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(x_labels)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Quality Metrics: {experiment_name}", fontsize=16)
        plt.tight_layout()

        metrics_path = output_dir / f"{experiment_name}_quality_metrics.png"
        self.plot_manager.save_figure(fig, metrics_path)

        print(f"   Quality metrics: {metrics_path}")
        return metrics_path

    def update_config(self, config: VisualizationConfig) -> None:
        """Update visualization configuration.

        Args:
            config: New visualization configuration
        """
        self.config = config
        self.plot_manager.dpi = config.display.figure_dpi
        self.gif_creator.fps = config.gif.fps
        self.gif_creator.dpi = config.display.figure_dpi

    def set_style(self, style_dict: Dict[str, Any]) -> None:
        """Update visualization style settings.

        Args:
            style_dict: Dictionary with style parameters
        """
        self.plot_manager.set_style(style_dict)
        self.gif_creator.set_style(style_dict)
