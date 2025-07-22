"""
Pytest configuration and shared fixtures.
"""

from pathlib import Path
from typing import Generator

import pytest
import torch

# Import all necessary modules
from config import ConfigurationLoader, ExperimentConfig
from data import SwissRollGenerator
from domain import DiffusionModel, LinearNoiseScheduler
from training import DiffusionTrainer

# Test assets directory
ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture
def assets_dir() -> Path:
    """Path to test assets directory."""
    return ASSETS_DIR


@pytest.fixture
def minimal_config() -> ExperimentConfig:
    """Minimal configuration for fast testing."""
    config_path = ASSETS_DIR / "minimal_config.toml"
    return ConfigurationLoader.load_toml(config_path)


@pytest.fixture
def production_config() -> ExperimentConfig:
    """Production-like configuration for realistic testing."""
    config_path = ASSETS_DIR / "production_config.toml"
    return ConfigurationLoader.load_toml(config_path)


@pytest.fixture
def invalid_config_path() -> Path:
    """Path to invalid configuration for error testing."""
    return ASSETS_DIR / "invalid_config.toml"


@pytest.fixture
def seed_random() -> Generator[None, None, None]:
    """Seed random number generators for reproducible tests."""
    torch.manual_seed(42)
    yield
    # Cleanup if needed


@pytest.fixture
def noise_scheduler() -> LinearNoiseScheduler:
    """Standard noise scheduler for testing."""
    return LinearNoiseScheduler(timesteps=10)


@pytest.fixture
def diffusion_model(noise_scheduler: LinearNoiseScheduler) -> DiffusionModel:
    """Standard diffusion model for testing."""
    return DiffusionModel(noise_scheduler)


@pytest.fixture
def swiss_roll_data() -> torch.Tensor:
    """Sample Swiss roll data for testing."""
    generator = SwissRollGenerator(noise_level=0.1, random_state=42)
    data: torch.Tensor = generator.generate(50)
    return data


@pytest.fixture
def trainer(diffusion_model: DiffusionModel, minimal_config: ExperimentConfig) -> DiffusionTrainer:
    """Standard trainer for testing."""
    return DiffusionTrainer(diffusion_model, minimal_config.training)


@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory for test outputs."""
    temp_dir: Path = tmp_path_factory.mktemp("diffusion_test_outputs")
    return temp_dir


# Custom markers for test organization
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]
