# Toy Diffusion

An implementation of a toy diffusion model for 2D data generation by example.

TODOs:

* drop scripts/config_manager.py
* add/update README
* add data assets to be used in github pages (gif, plots, docs/references)
* add github pages

## Architecture

```
toy-diffusion/
├── src/
│   ├── config/          # Configuration management
│   ├── data/           # Data generation (Swiss Roll)
│   ├── domain/         # Core diffusion model logic
│   ├── training/       # Training pipeline
│   ├── visualization/  # Plotting and animations
│   ├── logger/         # Structured logging
│   └── exceptions/     # Custom exceptions
├── scripts/
│   └── train.py        # Training script
├── tests/              # Unit, integration, and E2E tests
├── etc/                # Configuration files
└── outputs/            # Generated results and visualizations
```

## Features

- **Clean Architecture**: Follows SOLID principles with clear separation of concerns
- **Swiss Roll Dataset**: Generates and learns from 2D Swiss Roll manifold data
- **DDPM Algorithm**: Implements Denoising Diffusion Probabilistic Models
- **Type Safety**: Full type annotations and comprehensive testing
- **Structured Logging**: JSON-based logging with configurable output
- **Configuration Management**: TOML-based configuration with validation

## Requirements

- Python ≥3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash

# Install with uv (recommended); flag is needed for development only
uv sync [--extra=dev]

# Or with pip
pip install -e ".[dev]"
```

## Usage

### Training

```bash
# Train with default settings
python scripts/train.py

# Use custom configuration
python scripts/train.py --config etc/custom.toml

# Quick test run
python scripts/train.py --config etc/quick_test.toml
```

### Python API

```python
from config import ConfigurationLoader
from data import SwissRollGenerator
from domain import DiffusionModel, LinearNoiseScheduler
from training import DiffusionTrainer

# Load configuration
config = ConfigurationLoader.load_toml("etc/default_config.toml")

# Generate data
data_generator = SwissRollGenerator(noise_level=0.1, random_state=42)
data = data_generator.generate(1000)

# Create and train model
noise_scheduler = LinearNoiseScheduler(timesteps=100)
model = DiffusionModel(noise_scheduler)
trainer = DiffusionTrainer(model, config.training)

# Train and generate samples
metrics = trainer.train(data)
samples, trajectory = model.sample(100, return_trajectory=True)
```

## Configuration

Configurations are stored in `etc/` as TOML files:

- `default_config.toml`: Standard settings
- `quick_test.toml`: Fast configuration for testing
- `high_quality.toml`: High-quality generation settings

**Example configuration**

```toml
[data]
n_data_points = 1000
noise_level = 0.1
random_state = 42

[model]
timesteps = 100
beta_min = 0.0001
beta_max = 0.02

[training]
n_epochs = 100
batch_size = 32
learning_rate = 0.001

[logging]
level = "INFO"
use_json_format = true
log_file = "diffusion.log"
```

## Output

Training generates timestamped results in `outputs/`:

- **Training metrics**: Loss curves and learning rate schedules
- **Sample visualizations**: Generated data points and comparisons
- **Denoising trajectory**: GIF showing the reverse diffusion process
- **Quality metrics**: Evaluation of generated samples vs. original data

## Development

The project includes comprehensive unit, integration, and end-to-end tests with pytest, type checking with mypy, and code formatting with black/isort. Run tests with `uv run pytest`.
