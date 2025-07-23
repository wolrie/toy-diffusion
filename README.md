# Toy Diffusion

[![Tests](https://github.com/wolrie/toy-diffusion/actions/workflows/test.yml/badge.svg)](https://github.com/wolrie/toy-diffusion/actions/workflows/test.yml) [![Deploy GitHub Pages](https://github.com/wolrie/toy-diffusion/actions/workflows/pages.yml/badge.svg)](https://github.com/wolrie/toy-diffusion/actions/workflows/pages.yml)

This repo was written to serve as a starting point for a technical talk on diffusion language models. It contains an minimalistic implementation for a toy diffusion model for 2d data generation. This implementation demonstrates the core concepts of diffusion probabilistic models using a simple 2D Swiss Roll dataset.

The idea for this implementation was spawned by two papers (see the references [below](#7-references) for more info and links):

- **Forward/reverse diffusion framework**: Based on [1] - the original thermodynamics-inspired approach. The swiss roll dataset was also used there.
- **DDPM training algorithm**: Follows [2] - the practical denoising objective and simplified training procedure.

This toy implementation follows the DDPM framework from [2], specifically:

- **Noise Scheduling:** Linear beta scheduling as described in the DDPM paper
- **Loss Function:** Simplified denoising objective (MSE between predicted and actual noise)
- **Reverse Process:** Single-step denoising with learned noise prediction
- **Architecture:** Simple MLP network suitable for 2D data (Swiss Roll manifold)

## Table of Contents

1. [Architecture](#1-architecture)
1. [Setup](#2-setup)
1. [Usage](#3-usage)
1. [Configuration](#4-configuration)
1. [Output](#5-output)
1. [Development](#6-development)
1. [References](#7-references)

---

## 1. Architecture

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

## 2. Setup

### 2.1 Requirements

- Python ≥3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 2.1 Installation

```bash

# Install with uv (recommended); flag is needed for development only
uv sync [--extra=dev]

# Or with pip
pip install -e ".[dev]"
```

## 3. Usage

### 3.1 Training

```bash
# Train with default settings
python scripts/train.py

# Use custom configuration
python scripts/train.py --config etc/custom.toml

# Quick test run
python scripts/train.py --config etc/quick_test.toml
```

### 3.2 Python API

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

## 4. Configuration

Configurations are stored in `etc/` as TOML files. Check them for comments and further info.

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

## 5. Output

Training generates timestamped results in `outputs/`:

- **Training metrics**: Loss curves and learning rate schedules
- **Sample visualizations**: Generated data points and comparisons
- **Denoising trajectory**: GIF showing the reverse diffusion process
- **Quality metrics**: Evaluation of generated samples vs. original data

## 6. Development

The project includes comprehensive unit, integration, and end-to-end tests with pytest, type checking with mypy, and code formatting with black/isort. Run tests with `uv run pytest`.

## 7. References

[1] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 2256-2265).
**Links:** [PMLR](https://proceedings.mlr.press/v37/sohl-dickstein15.html) | [ArXiv](https://arxiv.org/abs/1503.03585) | [GitHub](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)

[2] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 6840-6851).
**Links:** [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) | [ArXiv](https://arxiv.org/abs/2006.11239) | [GitHub](https://github.com/hojonathanho/diffusion)
