"""
Test configuration and fixtures for pytest.
"""

import os
import sys

import numpy as np
import pytest
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    VisualizationConfig,
)
from data.swiss_roll_generator import SwissRollGenerator
from domain.diffusion_model import DiffusionModel
from domain.noise_scheduler import LinearNoiseScheduler
from training.trainer import DiffusionTrainer


@pytest.fixture
def test_config():
    """Fixture providing a test configuration."""
    return ExperimentConfig(
        data=DataConfig(n_data_points=50, noise_level=0.1, random_state=42),
        model=ModelConfig(timesteps=5, hidden_dim=32),
        training=TrainingConfig(n_epochs=3, batch_size=8, learning_rate=1e-2),
        visualization=VisualizationConfig(n_samples=20),
    )


@pytest.fixture
def test_data():
    """Fixture providing test data."""
    generator = SwissRollGenerator(noise_level=0.1, random_state=42)
    return generator.generate(50)


@pytest.fixture
def noise_scheduler():
    """Fixture providing a test noise scheduler."""
    return LinearNoiseScheduler(timesteps=5)


@pytest.fixture
def diffusion_model(noise_scheduler):
    """Fixture providing a test diffusion model."""
    return DiffusionModel(noise_scheduler)


@pytest.fixture
def trainer(diffusion_model, test_config):
    """Fixture providing a test trainer."""
    return DiffusionTrainer(diffusion_model, test_config.training)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Auto-use fixture to set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup if needed


@pytest.fixture
def small_test_data():
    """Fixture providing minimal test data for quick tests."""
    generator = SwissRollGenerator(noise_level=0.05, random_state=42)
    return generator.generate(10)


@pytest.fixture
def large_test_data():
    """Fixture providing larger test data for integration tests."""
    generator = SwissRollGenerator(noise_level=0.2, random_state=42)
    return generator.generate(500)


class TestDataProvider:
    """Helper class for providing test data with different configurations."""

    @staticmethod
    def get_minimal_config():
        """Get minimal configuration for fast tests."""
        return ExperimentConfig(
            data=DataConfig(n_data_points=10, noise_level=0.05),
            model=ModelConfig(timesteps=3, hidden_dim=16),
            training=TrainingConfig(n_epochs=2, batch_size=4, learning_rate=1e-2),
        )

    @staticmethod
    def get_standard_config():
        """Get standard configuration for normal tests."""
        return ExperimentConfig(
            data=DataConfig(n_data_points=100, noise_level=0.2),
            model=ModelConfig(timesteps=10, hidden_dim=64),
            training=TrainingConfig(n_epochs=5, batch_size=16, learning_rate=1e-3),
        )

    @staticmethod
    def get_comprehensive_config():
        """Get comprehensive configuration for integration tests."""
        return ExperimentConfig(
            data=DataConfig(n_data_points=1000, noise_level=0.2),
            model=ModelConfig(timesteps=50, hidden_dim=128),
            training=TrainingConfig(n_epochs=20, batch_size=32, learning_rate=1e-3),
        )


@pytest.fixture
def test_data_provider():
    """Fixture providing the test data provider."""
    return TestDataProvider()


# Custom markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Skip slow tests by default unless specifically requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle slow tests."""
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


# Cleanup fixtures
@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Reset random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    # Force garbage collection if needed
    import gc

    gc.collect()


# Performance monitoring fixture
@pytest.fixture
def performance_monitor():
    """Fixture for monitoring test performance."""
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        def get_duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return PerformanceMonitor()


# Temporary directory fixture for file operations
@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
