#!/usr/bin/env python3
"""Diffusion Model Training Script

A flexible training script that supports external configuration files,
command-line arguments, and environment variable overrides.

Usage:
    python scripts/train.py --config etc/default_config.toml
    python scripts/train.py --config etc/quick_test.toml --output results/
    python scripts/train.py --epochs 1000 --batch-size 64
    DIFFUSION_TRAINING_N_EPOCHS=500 python scripts/train.py --config \\
        etc/default_config.toml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config import ConfigurationLoader, ExperimentConfig
from data import SwissRollGenerator
from domain import DiffusionModel, LinearNoiseScheduler
from training import DiffusionTrainer, enums
from visualization import DiffusionVisualizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a diffusion model with TOML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --config etc/quick_test.toml
  python scripts/train.py --config etc/default_config.toml --output results/experiment1/
  python scripts/train.py --name my_experiment
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="etc/default_config.toml",
        help="Path to TOML configuration file (default: etc/default_config.toml)",
    )
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--name", type=str, help="Override experiment name")

    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> ExperimentConfig:
    """Load configuration from various sources."""
    print(f"📝 Loading configuration from: {args.config}")
    config = ConfigurationLoader.load_toml(args.config)

    # Apply minimal CLI overrides
    if args.output:
        config.output.output_dir = args.output
    if args.name:
        config.output.experiment_name = args.name

    return config


def setup_device(config: ExperimentConfig) -> torch.device:
    """Setup computation device based on config."""
    if config.execution.device == enums.DeviceType.AUTO:
        device = torch.device(
            enums.DeviceType.CUDA if torch.cuda.is_available() else enums.DeviceType.CPU
        )
    else:
        device = torch.device(config.execution.device)
    print(f"🖥️  Using device: {device}")
    return device


def setup_output_directory(config: ExperimentConfig) -> Path:
    """Setup output directory for results based on config."""
    if config.output.experiment_name:
        output_path: Path = Path(config.output.output_dir) / config.output.experiment_name
    else:
        # Generate unique name with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(config.output.output_dir) / f"diffusion_experiment_{timestamp}"

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    return output_path


def print_configuration(config: ExperimentConfig) -> None:
    """Print configuration summary based on config verbosity settings."""
    if config.execution.quiet:
        return

    if config.execution.verbose:
        print("\n📊 Configuration Details:")
        print(f"   Data: {config.data.n_data_points} points, noise={config.data.noise_level}")
        print(f"   Model: {config.model.timesteps} timesteps, hidden_dim={config.model.hidden_dim}")
        print(
            f"   Training: {config.training.n_epochs} epochs, "
            f"batch_size={config.training.batch_size}"
        )
        print(
            f"   Learning rate: {config.training.learning_rate} "
            f"({config.training.scheduler_type} scheduler)"
        )
        print(f"   Visualization: {config.visualization.general.n_samples} samples")
        print(f"   Device: {config.execution.device}")
        print(f"   Output: {config.output.output_dir}")
    else:
        print(
            f"📊 Config: {config.data.n_data_points} data points, "
            f"{config.training.n_epochs} epochs, batch_size={config.training.batch_size}"
        )


def create_visualizations(
    data: torch.Tensor,
    samples: torch.Tensor,
    trajectory: Optional[List[torch.Tensor]],
    metrics: Any,
    output_path: Path,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Create and save visualizations using the visualization module."""
    if not config.visualization.general.create_visualizations:
        return {}

    print("📊 Creating visualizations...")

    # Initialize visualizer
    visualizer = DiffusionVisualizer(config.visualization)

    # Create comprehensive training results
    results: Dict[str, Any] = visualizer.create_training_results(
        original_data=data,
        generated_samples=samples,
        trajectory=trajectory,
        metrics=metrics,
        output_dir=output_path,
        experiment_name="training_results",
    )

    print(f"   Training results saved to: {output_path}")

    # Create GIF if requested and trajectory is available
    if config.visualization.gif.create_gif and trajectory is not None:
        try:
            gif_path = visualizer.create_denoising_gif(
                trajectory=trajectory,
                original_data=data,
                output_dir=output_path,
                experiment_name="denoising",
                gif_type=config.visualization.gif.gif_type,
            )
            print(f"   GIF saved to: {gif_path}")
        except Exception as e:
            print(f"   Skipping GIF creation: {e}")

    # Create progression strip visualization if trajectory is available
    if config.visualization.progression.create_progression_strip and trajectory is not None:
        try:
            strip_path = visualizer.create_denoising_progression_strip(
                trajectory=trajectory,
                original_data=data,
                output_dir=output_path,
                experiment_name="denoising_progression",
                n_frames=config.visualization.progression.frames,
            )
            print(f"   Progression strip saved to: {strip_path}")
        except Exception as e:
            print(f"   Skipping progression strip creation: {e}")

    return results


def save_results(
    model: DiffusionModel,
    config: ExperimentConfig,
    metrics: Any,
    output_path: Path,
) -> None:
    """Save training results and artifacts based on config settings."""
    # Save model if requested
    if config.output.save_model:
        model_path = output_path / "model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to: {model_path}")

    # Save configuration if requested
    if config.output.save_config:
        config_path = output_path / "config.toml"
        ConfigurationLoader.save_to_file(config, config_path)
        print(f"💾 Configuration saved to: {config_path}")

    # Save training metrics
    metrics_path = output_path / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Final Loss: {metrics.get_final_loss():.6f}\n")
        f.write(f"Total Epochs: {len(metrics.losses)}\n")
        f.write(f"Initial Loss: {metrics.losses[0]:.6f}\n")
        f.write(f"Best Loss: {min(metrics.losses):.6f}\n")
        f.write(f"Final Learning Rate: {metrics.learning_rates[-1]:.8f}\n")

    print(f"📈 Metrics saved to: {metrics_path}")


def main() -> None:
    """Main training function."""
    # Parse command line arguments
    print("🚀 Starting diffusion model training...")

    args = parse_arguments()
    config = load_configuration(args)
    print_configuration(config)

    output_path = setup_output_directory(config)
    device = setup_device(config)

    # Generate data
    if not config.execution.quiet:
        print("🎲 Generating Swiss Roll data...")
    data_generator = SwissRollGenerator(
        noise_level=config.data.noise_level,
        random_state=config.data.random_state,
    )
    data = data_generator.generate(config.data.n_data_points)
    data = data.to(device)

    if config.execution.verbose:
        print(f"   Generated {data.shape[0]} data points")
        print(
            f"   Data range: X[{data[:, 0].min():.3f}, {data[:, 0].max():.3f}], "
            f"Y[{data[:, 1].min():.3f}, {data[:, 1].max():.3f}]"
        )

    # Create model
    if not config.execution.quiet:
        print("🧠 Creating diffusion model...")
    noise_scheduler = LinearNoiseScheduler(
        timesteps=config.model.timesteps,
        beta_min=config.model.beta_min,
        beta_max=config.model.beta_max,
    )
    model = DiffusionModel(noise_scheduler).to(device)

    if config.execution.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {total_params:,}")

    # Train model
    if not config.execution.quiet:
        print("🏋️  Training model...")
    trainer = DiffusionTrainer(model, config.training)

    start_time = time.time()
    metrics = trainer.train(data, verbose=not config.execution.quiet)
    training_time = time.time() - start_time

    if not config.execution.quiet:
        print(f"   Training completed in {training_time:.1f}s")
        print(f"   Final loss: {metrics.get_final_loss():.4f}")

    # Generate samples
    if not config.execution.quiet:
        print("🎨 Generating samples...")
    model.eval()
    with torch.no_grad():
        samples, trajectory = model.sample(
            config.visualization.general.n_samples,
            return_trajectory=config.visualization.gif.create_gif
            or config.visualization.progression.create_progression_strip,
        )
        samples = samples.cpu()
        if trajectory is not None:
            trajectory = [t.cpu() for t in trajectory]

    # Evaluate sample quality
    quality_metrics = trainer.evaluate_sample_quality(
        data.cpu(), config.visualization.general.n_samples
    )

    if not config.execution.quiet:
        print("\n📈 Results Summary:")
        print(f"   Final loss: {metrics.get_final_loss():.4f}")
        print(f"   Mean error: {quality_metrics['mean_error']:.4f}")
        print(f"   Std error: {quality_metrics['std_error']:.4f}")
        print(f"   Training time: {training_time:.1f}s")

    # Create visualizations
    create_visualizations(data.cpu(), samples, trajectory, metrics, output_path, config)

    # Create quality metrics visualization
    if config.visualization.general.create_visualizations:
        try:
            visualizer = DiffusionVisualizer(config.visualization)
            _ = visualizer.create_quality_metrics_visualization(
                quality_metrics=quality_metrics,
                output_dir=output_path,
                experiment_name="quality_metrics",
            )
        except Exception as e:
            print(f"   Skipping quality metrics visualization: {e}")

    # Save results
    save_results(model, config, metrics, output_path)

    if not config.execution.quiet:
        print("\n✅ Training completed successfully!")
        print(f"   All results saved to: {output_path}")


if __name__ == "__main__":
    main()
