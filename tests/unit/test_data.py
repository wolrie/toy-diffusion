"""Unit tests for data generation components."""

import pytest
import torch

from data import SwissRollGenerator
from exceptions import DataError


@pytest.mark.unit
class TestSwissRollGenerator:
    """Test Swiss Roll data generator."""

    def test_default_initialization(self) -> None:
        """Test generator with default parameters."""
        generator = SwissRollGenerator()

        assert generator.noise_level == 0.2
        assert generator.random_state == 42

    @pytest.mark.parametrize(
        "noise_level,random_state",
        [
            (0.0, 123),
            (0.1, 456),
            (0.5, 789),
            (1.0, 0),
        ],
    )
    def test_custom_initialization(self, noise_level: float, random_state: int) -> None:
        """Test generator with custom parameters."""
        generator = SwissRollGenerator(noise_level=noise_level, random_state=random_state)

        assert generator.noise_level == noise_level
        assert generator.random_state == random_state

    @pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
    def test_data_generation_shapes(self, n_samples: int) -> None:
        """Test data generation with different sample counts."""
        generator = SwissRollGenerator(random_state=42)
        data = generator.generate(n_samples)

        assert data.shape == (n_samples, 2)
        assert data.dtype == torch.float32
        # Check all values are finite
        finite_check = torch.isfinite(data).all()
        assert finite_check, "Generated data contains non-finite values"

    def test_reproducibility(self) -> None:
        """Test data generation is reproducible."""
        generator1 = SwissRollGenerator(random_state=42)
        generator2 = SwissRollGenerator(random_state=42)

        data1 = generator1.generate(50)
        data2 = generator2.generate(50)

        # Use a more reasonable tolerance for reproducibility
        assert torch.allclose(data1, data2, atol=1e-5)

    def test_different_seeds_produce_different_data(self) -> None:
        """Test different seeds produce different data."""
        generator1 = SwissRollGenerator(random_state=42)
        generator2 = SwissRollGenerator(random_state=123)

        data1 = generator1.generate(50)
        data2 = generator2.generate(50)

        assert not torch.allclose(data1, data2, atol=1e-3)

    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.1, 0.5, 1.0])
    def test_noise_levels(self, noise_level: float) -> None:
        """Test different noise levels."""
        generator = SwissRollGenerator(noise_level=noise_level, random_state=42)
        data = generator.generate(100)

        assert torch.isfinite(data).all()
        assert data.shape == (100, 2)

    def test_extreme_noise_levels(self) -> None:
        """Test extreme noise levels."""
        # Very high noise
        generator = SwissRollGenerator(noise_level=10.0, random_state=42)
        data = generator.generate(50)

        assert torch.isfinite(data).all()
        assert data.shape == (50, 2)

        # Zero noise
        generator_no_noise = SwissRollGenerator(noise_level=0.0, random_state=42)
        data_no_noise = generator_no_noise.generate(50)

        assert torch.isfinite(data_no_noise).all()
        assert data_no_noise.shape == (50, 2)

    def test_data_info(self) -> None:
        """Test data info method returns correct information."""
        generator = SwissRollGenerator(noise_level=0.15, random_state=123)
        info = generator.get_data_info()

        assert isinstance(info, dict)
        assert info["data_type"] == "swiss_roll"
        assert info["noise_level"] == 0.15
        assert info["random_state"] == 123
        assert info["dimensions"] == 2
        assert "description" in info
        assert isinstance(info["description"], str)

    def test_data_range_reasonable(self) -> None:
        """Test generated data is in reasonable range."""
        generator = SwissRollGenerator(noise_level=0.1, random_state=42)
        data = generator.generate(100)

        # Data should be roughly centered around zero with reasonable spread
        mean = data.mean(dim=0)
        std = data.std(dim=0)

        assert torch.abs(mean).max() < 2.0  # Roughly centered
        assert std.min() > 0.1  # Has some spread
        assert std.max() < 10.0  # Not too spread out

    def test_noise_override_in_generate(self) -> None:
        """Test noise level can be overridden in generate method."""
        generator = SwissRollGenerator(noise_level=0.1, random_state=42)

        data1 = generator.generate(50, noise_level=0.0)  # Override to no noise
        data2 = generator.generate(50, noise_level=1.0)  # Override to high noise

        # Both should be valid but different due to noise levels
        assert torch.isfinite(data1).all()
        assert torch.isfinite(data2).all()
        assert not torch.allclose(data1, data2, atol=0.1)

    def test_data_distribution_properties(self) -> None:
        """Test statistical properties of generated data."""
        generator = SwissRollGenerator(noise_level=0.05, random_state=42)
        data = generator.generate(1000)  # Large sample for statistics

        # Should have some correlation structure (Swiss roll is not random)
        cov_matrix = torch.cov(data.T)
        assert cov_matrix.shape == (2, 2)
        assert torch.isfinite(cov_matrix).all()

        # Diagonal elements (variances) should be positive
        assert cov_matrix[0, 0] > 0
        assert cov_matrix[1, 1] > 0
