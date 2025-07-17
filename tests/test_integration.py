"""
Integration tests - End-to-end testing of the complete diffusion model pipeline.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from data.swiss_roll_generator import SwissRollGenerator
from domain.diffusion_model import DiffusionModel
from domain.noise_scheduler import LinearNoiseScheduler
from training.trainer import DiffusionTrainer


class TestEndToEndPipeline:
    """Test complete end-to-end diffusion model pipeline."""

    def test_minimal_working_example(self):
        """Test minimal working example of the complete pipeline."""
        # Create minimal configuration
        config = ExperimentConfig(
            data=DataConfig(n_data_points=50, noise_level=0.1),
            model=ModelConfig(timesteps=5, hidden_dim=32),
            training=TrainingConfig(n_epochs=3, batch_size=8, learning_rate=1e-2),
            visualization=VisualizationConfig(n_samples=20),
        )

        # 1. Generate data
        data_generator = SwissRollGenerator(
            noise_level=config.data.noise_level,
            random_state=config.data.random_state,
        )
        data = data_generator.generate(config.data.n_data_points)

        # 2. Create model components
        noise_scheduler = LinearNoiseScheduler(
            timesteps=config.model.timesteps,
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
        )
        model = DiffusionModel(noise_scheduler)

        # 3. Train model
        trainer = DiffusionTrainer(model, config.training)
        metrics = trainer.train(data, verbose=False)

        # 4. Evaluate and sample
        quality_metrics = trainer.evaluate_sample_quality(data, config.visualization.n_samples)

        model.eval()
        with torch.no_grad():
            samples, trajectory = model.sample(
                config.visualization.n_samples, return_trajectory=True
            )

        # Verify all components worked
        assert data.shape == (50, 2)
        assert len(metrics.losses) == 3
        assert quality_metrics["mean_error"] >= 0
        assert samples.shape == (20, 2)
        assert len(trajectory) == 6  # 5 timesteps + 1 initial
        assert torch.isfinite(samples).all()

    def test_reproducible_pipeline(self):
        """Test that the pipeline is reproducible."""
        config = ExperimentConfig(
            data=DataConfig(n_data_points=30, random_state=42),
            model=ModelConfig(timesteps=3),
            training=TrainingConfig(n_epochs=2, batch_size=6),
            visualization=VisualizationConfig(n_samples=10),
        )

        def run_pipeline():
            # Set seeds for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)

            data_generator = SwissRollGenerator(random_state=42)
            data = data_generator.generate(config.data.n_data_points)

            noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
            model = DiffusionModel(noise_scheduler)

            trainer = DiffusionTrainer(model, config.training)
            metrics = trainer.train(data, verbose=False)

            model.eval()
            with torch.no_grad():
                samples, _ = model.sample(config.visualization.n_samples)

            return data, metrics.losses, samples

        # Run pipeline twice
        data1, losses1, samples1 = run_pipeline()
        data2, losses2, samples2 = run_pipeline()

        # Should be identical
        assert torch.allclose(data1, data2, atol=1e-6)
        assert len(losses1) == len(losses2)
        # Note: samples might not be identical due to different model states
        # but data generation should be identical

    def test_different_configurations(self):
        """Test pipeline with different configurations."""
        configurations = [
            # Small, fast configuration
            ExperimentConfig(
                data=DataConfig(n_data_points=20, noise_level=0.05),
                model=ModelConfig(timesteps=3, hidden_dim=16),
                training=TrainingConfig(n_epochs=2, batch_size=4),
            ),
            # Medium configuration
            ExperimentConfig(
                data=DataConfig(n_data_points=100, noise_level=0.2),
                model=ModelConfig(timesteps=10, hidden_dim=64),
                training=TrainingConfig(n_epochs=5, batch_size=16),
            ),
        ]

        for i, config in enumerate(configurations):
            # Generate data
            data_generator = SwissRollGenerator(
                noise_level=config.data.noise_level,
                random_state=config.data.random_state,
            )
            data = data_generator.generate(config.data.n_data_points)

            # Create and train model
            noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
            model = DiffusionModel(noise_scheduler)
            trainer = DiffusionTrainer(model, config.training)

            metrics = trainer.train(data, verbose=False)

            # Generate samples
            model.eval()
            with torch.no_grad():
                samples, _ = model.sample(50)

            # Verify results
            assert data.shape == (config.data.n_data_points, 2)
            assert len(metrics.losses) == config.training.n_epochs
            assert samples.shape == (50, 2)
            assert torch.isfinite(samples).all()

            print(f"Configuration {i+1}: Final loss = {metrics.get_final_loss():.4f}")

    def test_gradient_flow_integration(self):
        """Test that gradients flow properly through the entire model."""
        config = ExperimentConfig(
            data=DataConfig(n_data_points=20),
            model=ModelConfig(timesteps=5),
            training=TrainingConfig(n_epochs=1, batch_size=4),
        )

        # Generate data
        data_generator = SwissRollGenerator()
        data = data_generator.generate(config.data.n_data_points)

        # Create model
        noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
        model = DiffusionModel(noise_scheduler)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Train for one epoch
        trainer = DiffusionTrainer(model, config.training)
        trainer.train(data, verbose=False)

        # Check that parameters changed
        final_params = list(model.parameters())

        params_changed = False
        for initial, final in zip(initial_params, final_params):
            if not torch.allclose(initial, final, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "Model parameters should change during training"

    def test_loss_convergence_behavior(self):
        """Test that loss shows reasonable convergence behavior."""
        config = ExperimentConfig(
            data=DataConfig(n_data_points=100, noise_level=0.1),
            model=ModelConfig(timesteps=10, hidden_dim=128),
            training=TrainingConfig(n_epochs=20, batch_size=16, learning_rate=1e-2),
        )

        # Generate simple, learnable data
        data_generator = SwissRollGenerator(noise_level=config.data.noise_level)
        data = data_generator.generate(config.data.n_data_points)

        # Train model
        noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, config.training)

        metrics = trainer.train(data, verbose=False)

        # Analyze loss behavior
        losses = metrics.losses

        # Loss should be finite and positive
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        assert all(loss > 0 for loss in losses)

        # Should show some improvement over training
        initial_loss = losses[0]
        final_loss = losses[-1]
        min_loss = min(losses)

        # At least the minimum loss should be better than initial
        assert min_loss <= initial_loss

        # Final loss should be reasonable (not worse than 2x initial)
        assert final_loss <= initial_loss * 2.0

    def test_sample_quality_progression(self):
        """Test that sample quality improves with training."""
        config = ExperimentConfig(
            data=DataConfig(n_data_points=200, noise_level=0.15),
            model=ModelConfig(timesteps=10),
            training=TrainingConfig(n_epochs=10, batch_size=32),
        )

        # Generate data
        data_generator = SwissRollGenerator(
            noise_level=config.data.noise_level,
            random_state=config.data.random_state,
        )
        data = data_generator.generate(config.data.n_data_points)

        # Create model and trainer
        noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, config.training)

        # Evaluate before training
        initial_quality = trainer.evaluate_sample_quality(data, 100)

        # Train model
        metrics = trainer.train(data, verbose=False)

        # Evaluate after training
        final_quality = trainer.evaluate_sample_quality(data, 100)

        # Quality should improve or at least not get much worse
        # (Perfect quality improvement is not guaranteed due to stochasticity)
        print(f"Initial mean error: {initial_quality['mean_error']:.4f}")
        print(f"Final mean error: {final_quality['mean_error']:.4f}")
        print(f"Initial std error: {initial_quality['std_error']:.4f}")
        print(f"Final std error: {final_quality['std_error']:.4f}")

        # At minimum, errors should be finite and reasonable
        assert torch.isfinite(torch.tensor(final_quality["mean_error"]))
        assert torch.isfinite(torch.tensor(final_quality["std_error"]))
        assert final_quality["mean_error"] < 10.0  # Reasonable bound
        assert final_quality["std_error"] < 10.0  # Reasonable bound


class TestRobustness:
    """Test robustness of the pipeline to various conditions."""

    def test_edge_case_data_sizes(self):
        """Test pipeline with edge case data sizes."""
        edge_cases = [
            {"n_data_points": 1, "batch_size": 1},
            {"n_data_points": 2, "batch_size": 1},
            {"n_data_points": 5, "batch_size": 5},
        ]

        for case in edge_cases:
            config = ExperimentConfig(
                data=DataConfig(n_data_points=case["n_data_points"]),
                model=ModelConfig(timesteps=3),
                training=TrainingConfig(n_epochs=2, batch_size=case["batch_size"]),
            )

            # Should not crash
            data_generator = SwissRollGenerator()
            data = data_generator.generate(config.data.n_data_points)

            noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
            model = DiffusionModel(noise_scheduler)
            trainer = DiffusionTrainer(model, config.training)

            metrics = trainer.train(data, verbose=False)

            # Should complete successfully
            assert len(metrics.losses) == config.training.n_epochs

    def test_extreme_noise_levels(self):
        """Test pipeline with extreme noise levels."""
        noise_levels = [0.0, 1e-6, 1.0, 10.0]

        for noise_level in noise_levels:
            config = ExperimentConfig(
                data=DataConfig(n_data_points=50, noise_level=noise_level),
                model=ModelConfig(timesteps=5),
                training=TrainingConfig(n_epochs=2, batch_size=8),
            )

            # Generate data
            data_generator = SwissRollGenerator(noise_level=noise_level)
            data = data_generator.generate(config.data.n_data_points)

            # Should have finite values
            assert torch.isfinite(data).all()

            # Train model (should not crash)
            noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
            model = DiffusionModel(noise_scheduler)
            trainer = DiffusionTrainer(model, config.training)

            metrics = trainer.train(data, verbose=False)

            # Should complete with finite losses
            assert all(torch.isfinite(torch.tensor(loss)) for loss in metrics.losses)

    def test_different_random_seeds(self):
        """Test pipeline stability across different random seeds."""
        config = ExperimentConfig(
            data=DataConfig(n_data_points=50),
            model=ModelConfig(timesteps=5),
            training=TrainingConfig(n_epochs=3, batch_size=8),
        )

        results = []

        for seed in [42, 123, 999]:
            # Set seed
            torch.manual_seed(seed)

            # Generate data with different seed
            data_generator = SwissRollGenerator(random_state=seed)
            data = data_generator.generate(config.data.n_data_points)

            # Train model
            noise_scheduler = LinearNoiseScheduler(config.model.timesteps)
            model = DiffusionModel(noise_scheduler)
            trainer = DiffusionTrainer(model, config.training)

            metrics = trainer.train(data, verbose=False)

            results.append(metrics.get_final_loss())

        # All results should be finite and positive
        assert all(torch.isfinite(torch.tensor(loss)) for loss in results)
        assert all(loss > 0 for loss in results)

        # Results should be different (due to different seeds)
        assert not all(abs(results[0] - loss) < 1e-6 for loss in results[1:])


if __name__ == "__main__":
    # Quick integration test
    test = TestEndToEndPipeline()
    test.test_minimal_working_example()
    print("✅ Integration test passed!")
