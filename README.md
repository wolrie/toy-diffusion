# Toy Diffusion

A clean, well-architected implementation of diffusion models demonstrating software engineering best practices and SOLID principles.

## Architecture Overview

This project implements a toy diffusion model that follows clean architecture principles:

### SOLID Principles Implementation

- **Single Responsibility Principle (SRP)**: Each module has a single, well-defined responsibility
  - `domain/`: Core diffusion model logic
  - `data/`: Data generation and management
  - `training/`: Training orchestration
  - `visualization/`: Plotting and animation
  - `config/`: Configuration management

- **Open/Closed Principle (OCP)**: Components are open for extension, closed for modification
  - New data generators can be added via `DataGeneratorInterface`
  - New noise schedulers can be implemented via `NoiseSchedulerInterface`
  - New learning rate schedulers via `SchedulerFactory`

- **Liskov Substitution Principle (LSP)**: Implementations can be substituted for their interfaces
  - Any `DataGeneratorInterface` implementation works with the training pipeline
  - Any `NoiseSchedulerInterface` implementation works with `DiffusionModel`

- **Interface Segregation Principle (ISP)**: Clients depend only on interfaces they use
  - Separate interfaces for data generation, noise scheduling, etc.

- **Dependency Inversion Principle (DIP)**: High-level modules depend on abstractions
  - `DiffusionModel` depends on `NoiseSchedulerInterface`, not concrete implementations
  - `DiffusionTrainer` works with any `DiffusionModel`

### Project Structure

```
toy-diffusion/
├── src/              # Source code following clean architecture
│   ├── domain/       # Core diffusion model logic
│   ├── data/         # Data generation and management
│   ├── training/     # Training orchestration
│   ├── config/       # Configuration management
│   └── visualization/ # Plotting, animations, and visual analytics
├── etc/              # Configuration files (TOML)
│   ├── default_config.toml
│   ├── quick_test.toml
│   └── high_quality.toml
├── scripts/          # Command-line tools
│   ├── train.py      # Flexible training script
│   └── config_manager.py # Configuration utilities
├── tests/            # Comprehensive test suite
└── examples/         # Usage examples
```

## Quick Start

### Using External Configuration (Recommended)

```python
from src.config import ConfigurationLoader
from src.domain import DiffusionModel, LinearNoiseScheduler
from src.data import SwissRollGenerator
from src.training import DiffusionTrainer

# Load configuration from file
config = ConfigurationLoader.load_toml("etc/default_config.toml")

# Generate data
data_generator = SwissRollGenerator(noise_level=config.data.noise_level)
data = data_generator.generate(config.data.n_data_points)

# Create model
noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
model = DiffusionModel(noise_scheduler)

# Train model
trainer = DiffusionTrainer(model, config.training)
metrics = trainer.train(data)

# Generate samples
samples, _ = model.sample(config.visualization.n_samples)
```

### Using Default Configuration

```python
from src.config import ExperimentConfig
# ... rest same as above but with:
config = ExperimentConfig.default()
```

## Features

- **Clean Architecture**: Separation of concerns following SOLID principles
- **Flexible Configuration**: External TOML config files with environment variable overrides
- **Comprehensive Testing**: Full test suite with unit, integration, and performance tests
- **Extensible Design**: Easy to add new data generators, schedulers, and components
- **Professional Visualization**: Comprehensive plotting, animations, and visual analytics
- **Modular Visualization**: Separate components for static plots, GIF creation, and result analysis
- **Type Safety**: Type hints and validation throughout
- **CLI Tools**: Command-line utilities for configuration management

## Configuration System

### Configuration Files

The project supports external configuration files in TOML format:

```bash
# Use provided configurations
python examples/config_example.py --config etc/quick_test.toml
python examples/config_example.py --config etc/default_config.toml
python examples/config_example.py --config etc/high_quality.toml

# Or use the dedicated training script
python scripts/train.py --config etc/default_config.toml
python scripts/train.py --preset quick --output results/
python scripts/train.py --epochs 100 --batch-size 32

# Create your own configuration
python scripts/config_manager.py create my_config.toml --type default
python scripts/config_manager.py create my_config.toml --type quick
```

### Available Configurations

Multiple configuration presets are provided:

**Standard Configuration:**
- `etc/default_config.toml`

**Quick Test Configuration:**
- `etc/quick_test.toml`

**High Quality Configuration:**
- `etc/high_quality.toml`

### Environment Variable Overrides

Override any configuration parameter using environment variables:

```bash
export DIFFUSION_TRAINING_N_EPOCHS=1000
export DIFFUSION_DATA_NOISE_LEVEL=0.3
python examples/config_example.py --config my_config.toml --use-env-override
```

### Training Script

The dedicated training script provides maximum flexibility:

```bash
# Use configuration files
python scripts/train.py --config etc/default_config.toml
python scripts/train.py --config etc/quick_test.toml --output results/experiment1/

# Use preset configurations
python scripts/train.py --preset quick
python scripts/train.py --preset high_quality --output results/

# Override specific parameters
python scripts/train.py --config etc/default_config.toml --epochs 500 --batch-size 64
python scripts/train.py --preset quick --learning-rate 0.01 --no-gif

# Environment variable support
DIFFUSION_TRAINING_N_EPOCHS=1000 python scripts/train.py --config etc/default_config.toml

# Save results and models
python scripts/train.py --config etc/default_config.toml --save-model --save-config --name my_experiment
```

### Configuration Management CLI

```bash
# Create configurations
diffusion-config create my_config.toml --type quick

# Validate configurations
diffusion-config validate my_config.toml

# Compare configurations
diffusion-config compare config1.toml config2.toml

# Show available environment variables
diffusion-config env-vars
```

## Installation

```bash
# Basic installation
pip install -e .

# With configuration file support (YAML/TOML)
pip install -e ".[config]"

# Full development installation
pip install -e ".[all]"
```

After installation, you'll have access to command-line tools:
- `diffusion-train` - Flexible training script
- `diffusion-config` - Configuration management utilities

## Command-Line Tools

### Training Tool

```bash
# Quick training with presets
diffusion-train --preset quick
diffusion-train --preset high_quality --output results/

# Custom configurations
diffusion-train --config etc/my_config.toml --save-model
diffusion-train --epochs 100 --batch-size 32 --learning-rate 0.01

# Advanced usage
diffusion-train --config etc/default_config.toml --env-override --verbose --save-config
```

### Configuration Tool

```bash
# Manage configurations
diffusion-config create my_config.toml --type quick
diffusion-config validate etc/default_config.toml
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with coverage
pytest --cov=src tests/

# Run specific test files
pytest tests/test_domain.py
pytest tests/test_config.py
```

Test categories:
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test complete workflows end-to-end
- **Performance tests**: Monitor training convergence and timing
- **Configuration tests**: Validate config loading and validation

## Getting Started Example

Here's a complete workflow from configuration to training:

```bash
# 1. Install the package
pip install -e ".[config]"

# 2. Quick test run with preset configuration
diffusion-train --preset quick --output results/quick_test --verbose

# 3. Custom training with configuration file
diffusion-train --config etc/default_config.toml \
                --epochs 1000 \
                --batch-size 64 \
                --save-model \
                --save-config \
                --name my_experiment \
                --output results/

# 4. High-quality training with environment overrides
export DIFFUSION_TRAINING_N_EPOCHS=5000
export DIFFUSION_MODEL_HIDDEN_DIM=512
diffusion-train --config etc/high_quality.toml \
                --env-override \
                --output results/high_quality_run

# 5. Manage configurations
diffusion-config create custom_config.toml --type default
diffusion-config validate custom_config.toml
diffusion-config compare etc/default_config.toml custom_config.toml
```

This example demonstrates:
- ✅ Clean separation of configuration from code
- ✅ Flexible parameter overrides
- ✅ Organized output management
- ✅ Comprehensive result saving
- ✅ Environment variable integration

The architecture makes it easy to run reproducible experiments, compare different configurations, and scale from quick prototypes to production training runs.

## Troubleshooting

### Import Errors
If you encounter import errors when running scripts:

1. **Install dependencies**: `pip install torch matplotlib numpy tqdm pillow`
2. **Run from project root**: Always execute scripts from the project root directory
3. **Test structure**: Run `python test_visualization_imports.py` to verify imports

### Common Issues
- **"No module named 'torch'"**: Install PyTorch with `pip install torch`
- **"No module named 'matplotlib'"**: Install with `pip install matplotlib`
- **Import path issues**: Ensure you're running from the project root directory

### Testing the Setup
```bash
# Test import structure (no dependencies required)
python tests/integration/test_visualization_imports.py

# Test training script structure
python scripts/test_train_structure.py

# Run actual training (requires dependencies)
python scripts/train.py --preset quick
```
