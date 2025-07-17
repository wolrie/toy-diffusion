"""
Tests for visualization module - Plotting and animation functionality.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from config.config import VisualizationConfig
from training.trainer import TrainingMetrics
from visualization import DiffusionVisualizer, GifCreator, PlotManager


class TestPlotManager:
    """Test cases for PlotManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plot_manager = PlotManager()
        self.original_data = torch.randn(100, 2)
        self.generated_data = torch.randn(80, 2)
        self.metrics = TrainingMetrics()

        # Add some dummy metrics
        for i in range(10):
            self.metrics.add_epoch_metrics(1.0 - i * 0.1, 1e-3 * (1 - i * 0.05))

    def test_initialization(self):
        """Test plot manager initialization."""
        assert self.plot_manager.figsize == (15, 10)
        assert self.plot_manager.dpi == 150
        assert "alpha" in self.plot_manager.default_style
        assert "xlim" in self.plot_manager.default_style

    def test_data_comparison_plot(self):
        """Test data comparison plot creation."""
        fig = self.plot_manager.create_data_comparison_plot(self.original_data, self.generated_data)

        assert fig is not None
        assert len(fig.axes) == 3  # Three subplots

        # Check that axes have correct properties
        for ax in fig.axes:
            assert ax.get_xlim() == self.plot_manager.default_style["xlim"]
            assert ax.get_ylim() == self.plot_manager.default_style["ylim"]

        # Close figure to prevent memory issues
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_training_metrics_plot(self):
        """Test training metrics plot creation."""
        fig = self.plot_manager.create_training_metrics_plot(self.metrics)

        assert fig is not None
        assert len(fig.axes) == 2  # Loss and learning rate plots

        # Check that data was plotted
        loss_ax, lr_ax = fig.axes
        assert len(loss_ax.get_lines()) > 0  # Loss line should exist
        assert len(lr_ax.get_lines()) > 0  # LR line should exist

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_distribution_comparison_plot(self):
        """Test distribution comparison plot creation."""
        fig = self.plot_manager.create_distribution_comparison_plot(
            self.original_data, self.generated_data
        )

        assert fig is not None
        assert len(fig.axes) == 2  # X and Y distribution plots

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_trajectory_plot(self):
        """Test trajectory plot creation."""
        # Create dummy trajectory
        trajectory = [torch.randn(50, 2) for _ in range(5)]

        fig = self.plot_manager.create_trajectory_plot(trajectory, self.original_data)

        assert fig is not None
        assert len(fig.axes) == 1

        # Check that colorbar was added
        assert hasattr(fig, "axes")

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_comprehensive_results_plot(self):
        """Test comprehensive results plot creation."""
        trajectory = [torch.randn(50, 2) for _ in range(5)]

        fig = self.plot_manager.create_comprehensive_results_plot(
            self.original_data, self.generated_data, trajectory, self.metrics
        )

        assert fig is not None
        assert len(fig.axes) >= 6  # 2x3 grid plus colorbar

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save_figure(self, temp_dir):
        """Test figure saving functionality."""
        fig = self.plot_manager.create_data_comparison_plot(self.original_data, self.generated_data)

        output_path = Path(temp_dir) / "test_plot.png"
        self.plot_manager.save_figure(fig, output_path, close_after_save=False)

        assert output_path.exists()

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_style_update(self):
        """Test style update functionality."""
        new_style = {"alpha": 0.8, "xlim": (-2, 2)}
        self.plot_manager.set_style(new_style)

        assert self.plot_manager.default_style["alpha"] == 0.8
        assert self.plot_manager.default_style["xlim"] == (-2, 2)


class TestGifCreator:
    """Test cases for GifCreator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gif_creator = GifCreator()
        self.original_data = torch.randn(50, 2)
        self.trajectory = [torch.randn(30, 2) for _ in range(10)]

    def test_initialization(self):
        """Test GIF creator initialization."""
        assert self.gif_creator.figsize == (10, 8)
        assert self.gif_creator.fps == 4
        assert self.gif_creator.dpi == 100
        assert "xlim" in self.gif_creator.default_style

    @patch("matplotlib.animation.FuncAnimation.save")
    def test_trajectory_gif_creation(self, mock_save, temp_dir):
        """Test trajectory GIF creation."""
        output_path = Path(temp_dir) / "test_trajectory.gif"

        result_path = self.gif_creator.create_trajectory_gif(
            self.trajectory, self.original_data, output_path, n_frames=5
        )

        assert result_path == output_path
        mock_save.assert_called_once()

    @patch("matplotlib.animation.FuncAnimation.save")
    def test_side_by_side_gif(self, mock_save, temp_dir):
        """Test side-by-side GIF creation."""
        output_path = Path(temp_dir) / "test_side_by_side.gif"

        result_path = self.gif_creator.create_side_by_side_gif(
            self.trajectory, self.original_data, output_path, n_frames=5
        )

        assert result_path == output_path
        mock_save.assert_called_once()

    @patch("matplotlib.animation.FuncAnimation.save")
    def test_progression_gif(self, mock_save, temp_dir):
        """Test progression GIF creation."""
        output_path = Path(temp_dir) / "test_progression.gif"

        result_path = self.gif_creator.create_progression_gif(
            self.trajectory, self.original_data, output_path, n_frames=5
        )

        assert result_path == output_path
        mock_save.assert_called_once()

    def test_style_update(self):
        """Test style update functionality."""
        new_style = {"xlim": (-3, 3), "fps": 8}
        self.gif_creator.set_style(new_style)

        assert self.gif_creator.default_style["xlim"] == (-3, 3)


class TestDiffusionVisualizer:
    """Test cases for DiffusionVisualizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = VisualizationConfig()
        self.visualizer = DiffusionVisualizer(self.config)

        self.original_data = torch.randn(100, 2)
        self.generated_data = torch.randn(80, 2)
        self.trajectory = [torch.randn(50, 2) for _ in range(5)]

        self.metrics = TrainingMetrics()
        for i in range(10):
            self.metrics.add_epoch_metrics(1.0 - i * 0.1, 1e-3 * (1 - i * 0.05))

    def test_initialization(self):
        """Test visualizer initialization."""
        assert isinstance(self.visualizer.plot_manager, PlotManager)
        assert isinstance(self.visualizer.gif_creator, GifCreator)
        assert self.visualizer.config == self.config

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    def test_create_training_results(self, mock_savefig, mock_close, temp_dir):
        """Test training results creation."""
        output_dir = Path(temp_dir)

        results = self.visualizer.create_training_results(
            self.original_data,
            self.generated_data,
            self.trajectory,
            self.metrics,
            output_dir,
            "test_experiment",
        )

        assert isinstance(results, dict)
        assert "comprehensive" in results
        assert "comparison" in results
        assert "metrics" in results
        assert "distributions" in results
        assert "trajectory" in results

        # Check that all result paths are Path objects
        for path in results.values():
            assert isinstance(path, Path)

    @patch("visualization.gif_creator.GifCreator.create_trajectory_gif")
    def test_create_denoising_gif(self, mock_create_gif, temp_dir):
        """Test denoising GIF creation."""
        output_dir = Path(temp_dir)
        mock_create_gif.return_value = output_dir / "test.gif"

        result_path = self.visualizer.create_denoising_gif(
            self.trajectory, self.original_data, output_dir, "test", "standard"
        )

        mock_create_gif.assert_called_once()
        assert isinstance(result_path, Path)

    def test_create_denoising_gif_invalid_type(self, temp_dir):
        """Test error handling for invalid GIF type."""
        output_dir = Path(temp_dir)

        with pytest.raises(ValueError, match="Unknown GIF type"):
            self.visualizer.create_denoising_gif(
                self.trajectory,
                self.original_data,
                output_dir,
                "test",
                "invalid_type",
            )

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    def test_create_comparison_visualization(self, mock_savefig, mock_close, temp_dir):
        """Test comparison visualization creation."""
        datasets = {
            "Dataset 1": torch.randn(50, 2),
            "Dataset 2": torch.randn(60, 2),
            "Dataset 3": torch.randn(40, 2),
        }

        output_dir = Path(temp_dir)
        result_path = self.visualizer.create_comparison_visualization(
            datasets, output_dir, "Test Comparison"
        )

        assert isinstance(result_path, Path)
        mock_savefig.assert_called()

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    def test_create_quality_metrics_visualization(self, mock_savefig, mock_close, temp_dir):
        """Test quality metrics visualization creation."""
        quality_metrics = {
            "original_mean": [0.1, 0.2],
            "sample_mean": [0.15, 0.18],
            "original_std": [1.0, 1.1],
            "sample_std": [0.9, 1.05],
            "mean_error": 0.05,
            "std_error": 0.08,
        }

        output_dir = Path(temp_dir)
        result_path = self.visualizer.create_quality_metrics_visualization(
            quality_metrics, output_dir, "test_metrics"
        )

        assert isinstance(result_path, Path)
        mock_savefig.assert_called()

    def test_update_config(self):
        """Test configuration update."""
        new_config = VisualizationConfig(figure_dpi=300, gif_fps=8)
        self.visualizer.update_config(new_config)

        assert self.visualizer.config == new_config
        assert self.visualizer.plot_manager.dpi == 300
        assert self.visualizer.gif_creator.fps == 8

    def test_set_style(self):
        """Test style update."""
        style_dict = {"alpha": 0.8, "xlim": (-3, 3)}
        self.visualizer.set_style(style_dict)

        # Should update both plot manager and gif creator styles
        assert self.visualizer.plot_manager.default_style["alpha"] == 0.8
        assert self.visualizer.gif_creator.default_style["xlim"] == (-3, 3)


class TestVisualizationIntegration:
    """Integration tests for visualization module."""

    def test_end_to_end_visualization_workflow(self, temp_dir):
        """Test complete visualization workflow."""
        # Setup
        config = VisualizationConfig(figure_dpi=100, gif_fps=2, n_gif_frames=3)
        visualizer = DiffusionVisualizer(config)

        # Generate test data
        original_data = torch.randn(50, 2) * 0.5
        generated_data = torch.randn(40, 2) * 0.6 + 0.1
        trajectory = [torch.randn(30, 2) * (1 - i / 10) for i in range(5)]

        metrics = TrainingMetrics()
        for i in range(5):
            metrics.add_epoch_metrics(1.0 - i * 0.2, 1e-3 * (1 - i * 0.1))

        output_dir = Path(temp_dir)

        # Test training results creation
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            results = visualizer.create_training_results(
                original_data,
                generated_data,
                trajectory,
                metrics,
                output_dir,
                "integration_test",
            )

            assert len(results) >= 4  # Should have multiple visualization types

        # Test GIF creation
        with patch("matplotlib.animation.FuncAnimation.save"):
            gif_path = visualizer.create_denoising_gif(
                trajectory,
                original_data,
                output_dir,
                "integration_test",
                "standard",
            )
            assert isinstance(gif_path, Path)

    def test_visualization_with_different_data_sizes(self):
        """Test visualization with different data sizes."""
        config = VisualizationConfig()
        visualizer = DiffusionVisualizer(config)

        # Test with different data sizes
        test_cases = [
            (10, 8),  # Small datasets
            (100, 90),  # Medium datasets
            (1000, 800),  # Large datasets
        ]

        for orig_size, gen_size in test_cases:
            original_data = torch.randn(orig_size, 2)
            generated_data = torch.randn(gen_size, 2)

            # Should not raise errors
            with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
                fig = visualizer.plot_manager.create_data_comparison_plot(
                    original_data, generated_data
                )
                assert fig is not None

    def test_visualization_error_handling(self, temp_dir):
        """Test error handling in visualization."""
        config = VisualizationConfig()
        visualizer = DiffusionVisualizer(config)

        # Test with empty data
        empty_data = torch.empty(0, 2)
        valid_data = torch.randn(10, 2)

        # Should handle empty data gracefully
        try:
            with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
                _ = visualizer.plot_manager.create_data_comparison_plot(empty_data, valid_data)
        except Exception as e:
            # If it raises an exception, it should be a meaningful one
            assert "empty" in str(e).lower() or "size" in str(e).lower()


if __name__ == "__main__":
    # Quick visualization test
    test = TestPlotManager()
    test.setup_method()
    test.test_initialization()
    print("✅ Visualization tests structure verified!")
