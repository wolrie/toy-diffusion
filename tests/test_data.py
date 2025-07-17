"""
Tests for data module - Data generation and management.
"""

import numpy as np
import torch

from data.data_interface import DataGeneratorInterface
from data.swiss_roll_generator import SwissRollGenerator


class TestSwissRollGenerator:
    """Test cases for SwissRollGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = SwissRollGenerator()

        assert generator.noise_level == 0.2
        assert generator.random_state == 42

        # Test custom parameters
        generator_custom = SwissRollGenerator(noise_level=0.5, random_state=123)
        assert generator_custom.noise_level == 0.5
        assert generator_custom.random_state == 123

    def test_interface_compliance(self):
        """Test that SwissRollGenerator implements DataGeneratorInterface."""
        generator = SwissRollGenerator()

        assert isinstance(generator, DataGeneratorInterface)
        assert hasattr(generator, "generate")
        assert hasattr(generator, "get_data_info")

    def test_data_generation(self):
        """Test basic data generation."""
        generator = SwissRollGenerator()

        # Test different sample sizes
        for n_samples in [10, 100, 1000]:
            data = generator.generate(n_samples)

            assert data.shape == (n_samples, 2)
            assert data.dtype == torch.float32
            assert torch.isfinite(data).all()

    def test_data_quality(self):
        """Test data quality and distribution."""
        generator = SwissRollGenerator(noise_level=0.1)
        data = generator.generate(1000)

        # Test data range (should be roughly in [-2, 2] range after normalization)
        assert data.min() > -3.0
        assert data.max() < 3.0

        # Test that data has expected structure (spiral-like)
        # Swiss roll should have some correlation between dimensions
        correlation = torch.corrcoef(data.T)[0, 1]
        assert abs(correlation) > 0.1  # Should have some correlation

        # Test that data is not just noise (should have structure)
        std_x, std_y = data.std(dim=0)
        assert std_x > 0.1 and std_y > 0.1

    def test_reproducibility(self):
        """Test that generation is reproducible."""
        generator1 = SwissRollGenerator(random_state=42)
        generator2 = SwissRollGenerator(random_state=42)

        data1 = generator1.generate(100)
        data2 = generator2.generate(100)

        assert torch.allclose(data1, data2, atol=1e-6)

    def test_noise_level_effect(self):
        """Test that noise level affects data variance."""
        generator_low = SwissRollGenerator(noise_level=0.01, random_state=42)
        generator_high = SwissRollGenerator(noise_level=0.5, random_state=42)

        data_low = generator_low.generate(1000)
        data_high = generator_high.generate(1000)

        # Higher noise should lead to higher variance
        var_low = data_low.var(dim=0).mean()
        var_high = data_high.var(dim=0).mean()

        assert var_high > var_low

    def test_zero_noise(self):
        """Test generation with zero noise."""
        generator = SwissRollGenerator(noise_level=0.0)
        data = generator.generate(100)

        assert data.shape == (100, 2)
        assert torch.isfinite(data).all()

        # With zero noise, should be very smooth
        # (low variance in local neighborhoods)
        sorted_data = data[torch.argsort(data[:, 0])]
        local_variance = torch.var(sorted_data[1:] - sorted_data[:-1], dim=0)
        assert local_variance.mean() < 0.1

    def test_parameter_override(self):
        """Test parameter override in generate method."""
        generator = SwissRollGenerator(noise_level=0.2)

        # Generate with different noise level
        data_low = generator.generate(100, noise_level=0.01)
        data_high = generator.generate(100, noise_level=0.5)

        # Should use overridden noise levels
        var_low = data_low.var(dim=0).mean()
        var_high = data_high.var(dim=0).mean()

        assert var_high > var_low

    def test_get_data_info(self):
        """Test data info method."""
        generator = SwissRollGenerator(noise_level=0.3, random_state=123)
        info = generator.get_data_info()

        assert isinstance(info, dict)
        assert info["data_type"] == "swiss_roll"
        assert info["noise_level"] == 0.3
        assert info["random_state"] == 123
        assert info["dimensions"] == 2
        assert "description" in info

    def test_large_sample_generation(self):
        """Test generation of large samples."""
        generator = SwissRollGenerator()

        # Test with large number of samples
        data = generator.generate(10000)

        assert data.shape == (10000, 2)
        assert torch.isfinite(data).all()

        # Should still have proper statistics
        mean = data.mean(dim=0)
        std = data.std(dim=0)

        # Mean should be close to zero (normalized)
        assert torch.abs(mean).max() < 0.1

        # Standard deviation should be reasonable
        assert std.min() > 0.1
        assert std.max() < 2.0

    def test_edge_cases(self):
        """Test edge cases."""
        generator = SwissRollGenerator()

        # Test with single sample
        data = generator.generate(1)
        assert data.shape == (1, 2)
        assert torch.isfinite(data).all()

        # Test with very small noise
        data_small_noise = generator.generate(10, noise_level=1e-6)
        assert torch.isfinite(data_small_noise).all()

        # Test with very large noise
        data_large_noise = generator.generate(10, noise_level=10.0)
        assert torch.isfinite(data_large_noise).all()


class TestDataInterface:
    """Test cases for DataGeneratorInterface."""

    def test_interface_methods(self):
        """Test that interface has required methods."""
        # This is more of a documentation test
        interface_methods = ["generate", "get_data_info"]

        for method in interface_methods:
            assert hasattr(DataGeneratorInterface, method)

    def test_swiss_roll_implements_interface(self):
        """Test that SwissRollGenerator properly implements interface."""
        generator = SwissRollGenerator()

        # Test generate method signature
        data = generator.generate(10)
        assert isinstance(data, torch.Tensor)

        # Test get_data_info method signature
        info = generator.get_data_info()
        assert isinstance(info, dict)


class TestDataGeneration:
    """Integration tests for data generation."""

    def test_multiple_generators_consistency(self):
        """Test that multiple generators with same parameters produce same results."""
        params = {"noise_level": 0.15, "random_state": 999}

        generator1 = SwissRollGenerator(**params)
        generator2 = SwissRollGenerator(**params)

        data1 = generator1.generate(50)
        data2 = generator2.generate(50)

        assert torch.allclose(data1, data2, atol=1e-6)

    def test_different_random_states(self):
        """Test that different random states produce different results."""
        generator1 = SwissRollGenerator(random_state=42)
        generator2 = SwissRollGenerator(random_state=123)

        data1 = generator1.generate(100)
        data2 = generator2.generate(100)

        # Should be different
        assert not torch.allclose(data1, data2, atol=1e-3)

    def test_data_statistics_stability(self):
        """Test that data statistics are stable across different generations."""
        generator = SwissRollGenerator(random_state=42)

        # Generate multiple batches
        means = []
        stds = []

        for _ in range(10):
            data = generator.generate(1000)
            means.append(data.mean(dim=0))
            stds.append(data.std(dim=0))

        means = torch.stack(means)
        stds = torch.stack(stds)

        # Statistics should be similar across generations
        mean_variance = means.var(dim=0)
        std_variance = stds.var(dim=0)

        assert mean_variance.max() < 0.01  # Means should be stable
        assert std_variance.max() < 0.01  # Stds should be stable

    def test_spiral_structure(self):
        """Test that generated data has spiral structure."""
        generator = SwissRollGenerator(noise_level=0.05, random_state=42)
        data = generator.generate(1000)

        # Convert to numpy for easier analysis
        data_np = data.numpy()

        # Compute angles from origin
        angles = np.arctan2(data_np[:, 1], data_np[:, 0])

        # Compute distances from origin
        distances = np.sqrt(data_np[:, 0] ** 2 + data_np[:, 1] ** 2)

        # For a spiral, there should be correlation between angle and distance
        correlation = np.corrcoef(angles, distances)[0, 1]

        # Should have some correlation (not perfect due to noise and normalization)
        assert abs(correlation) > 0.1
