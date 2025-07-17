#!/usr/bin/env python3
"""Configuration Manager - CLI utility for managing diffusion model configs.

Usage:
    python config_manager.py create default_config.yaml
    python config_manager.py validate config.json
    python config_manager.py convert config.yaml config.json
    python config_manager.py env-vars
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import ConfigurationHelper, ConfigurationLoader, ExperimentConfig


def create_config(output_path: str, config_type: str = "default"):
    """Create a new configuration file."""
    output_path = Path(output_path)

    if config_type == "default":
        config = ExperimentConfig.default()
    elif config_type == "quick":
        # Quick test configuration
        from config.config import DataConfig, ModelConfig, TrainingConfig, VisualizationConfig

        config = ExperimentConfig(
            data=DataConfig(n_data_points=100, noise_level=0.1),
            model=ModelConfig(timesteps=10, hidden_dim=64),
            training=TrainingConfig(n_epochs=50, batch_size=16, learning_rate=0.01),
            visualization=VisualizationConfig(n_samples=100, n_gif_frames=10),
        )
    elif config_type == "high_quality":
        # High quality configuration
        from config.config import DataConfig, ModelConfig, TrainingConfig, VisualizationConfig

        config = ExperimentConfig(
            data=DataConfig(n_data_points=5000, noise_level=0.15),
            model=ModelConfig(timesteps=200, hidden_dim=512),
            training=TrainingConfig(n_epochs=15000, batch_size=256, learning_rate=0.0005),
            visualization=VisualizationConfig(n_samples=5000, n_gif_frames=30, figure_dpi=300),
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    ConfigurationLoader.save_to_file(config, output_path)
    print(f"✅ Created {config_type} configuration: {output_path}")


def validate_config(config_path: str):
    """Validate a configuration file."""
    is_valid = ConfigurationLoader.validate_config_file(config_path)

    if is_valid:
        print(f"✅ Configuration file is valid: {config_path}")

        # Load and display config details
        config = ConfigurationLoader.load_toml(config_path)
        print("\nConfiguration details:")
        print(f"  Data points: {config.data.n_data_points}")
        print(f"  Noise level: {config.data.noise_level}")
        print(f"  Timesteps: {config.model.timesteps}")
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  Epochs: {config.training.n_epochs}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")
    else:
        print(f"❌ Configuration file is invalid: {config_path}")
        return False

    return True


def convert_config(input_path: str, output_path: str):
    """Convert configuration from one format to another."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False

    try:
        # Load from input format
        config = ConfigurationLoader.load_toml(input_path)

        # Save to output format
        ConfigurationLoader.save_to_file(config, output_path)

        print(f"✅ Converted {input_path} -> {output_path}")
        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def compare_configs(config1_path: str, config2_path: str):
    """Compare two configuration files."""
    try:
        config1 = ConfigurationLoader.load_toml(config1_path)
        config2 = ConfigurationLoader.load_toml(config2_path)

        differences = ConfigurationHelper.compare_configs(config1, config2)

        if not differences:
            print("✅ Configurations are identical")
        else:
            print("📊 Configuration differences:")
            for section, diffs in differences.items():
                print(f"\n[{section.upper()}]")
                for field, values in diffs.items():
                    print(f"  {field}: ")
                    print(f"    {config1_path}: {values['config1']}")
                    print(f"    {config2_path}: {values['config2']}")

        return True

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False


def print_env_variables():
    """Print available environment variables."""
    ConfigurationHelper.print_env_variables()


def show_examples():
    """Show example usage."""
    print("Configuration Manager - Example Usage")
    print("=====================================")
    print()
    print("Create configurations:")
    print("  python config_manager.py create my_config.yaml")
    print("  python config_manager.py create quick_test.json --type quick")
    print("  python config_manager.py create high_quality.yaml --type " "high_quality")
    print()
    print("Validate configurations:")
    print("  python config_manager.py validate my_config.yaml")
    print()
    print("Convert between formats:")
    print("  python config_manager.py convert config.yaml config.json")
    print("  python config_manager.py convert config.json config.yaml")
    print()
    print("Compare configurations:")
    print("  python config_manager.py compare config1.yaml config2.yaml")
    print()
    print("Environment variables:")
    print("  python config_manager.py env-vars")
    print()
    print("Environment variable example:")
    print("  export DIFFUSION_TRAINING_N_EPOCHS=1000")
    print("  export DIFFUSION_DATA_NOISE_LEVEL=0.3")
    print("  python config_manager.py validate config.yaml")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Configuration Manager for Diffusion Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new configuration file")
    create_parser.add_argument("output", help="Output configuration file path")
    create_parser.add_argument(
        "--type",
        choices=["default", "quick", "high_quality"],
        default="default",
        help="Configuration type",
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument("config", help="Configuration file path")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert configuration format")
    convert_parser.add_argument("input", help="Input configuration file")
    convert_parser.add_argument("output", help="Output configuration file")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two configuration files")
    compare_parser.add_argument("config1", help="First configuration file")
    compare_parser.add_argument("config2", help="Second configuration file")

    # Environment variables command
    subparsers.add_parser("env-vars", help="Print available environment variables")

    # Examples command
    subparsers.add_parser("examples", help="Show usage examples")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "create":
            create_config(args.output, args.type)
        elif args.command == "validate":
            validate_config(args.config)
        elif args.command == "convert":
            convert_config(args.input, args.output)
        elif args.command == "compare":
            compare_configs(args.config1, args.config2)
        elif args.command == "env-vars":
            print_env_variables()
        elif args.command == "examples":
            show_examples()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
