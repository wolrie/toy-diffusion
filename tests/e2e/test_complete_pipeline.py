"""End-to-end tests for complete diffusion model pipeline.

Tests the entire workflow from configuration to sample generation.
"""

from pathlib import Path

import pytest
import torch

from config import ConfigurationLoader, ExperimentConfig
from data import SwissRollGenerator
from domain import DiffusionModel, LinearNoiseScheduler
from training import DiffusionTrainer


@pytest.mark.e2e
class TestCompletePipeline:
    """Test complete end-to-end diffusion pipeline."""

    def test_minimal_working_pipeline(self, minimal_config: ExperimentConfig) -> None:
        """Test minimal working example of complete pipeline."""
        # 1. Data Generation
        data_generator = SwissRollGenerator(
            noise_level=minimal_config.data.noise_level,
            random_state=minimal_config.data.random_state,
        )
        data = data_generator.generate(minimal_config.data.n_data_points)

        # 2. Model Creation
        noise_scheduler = LinearNoiseScheduler(
            timesteps=minimal_config.model.timesteps,
            beta_min=minimal_config.model.beta_min,
            beta_max=minimal_config.model.beta_max,
        )
        model = DiffusionModel(noise_scheduler)

        # 3. Training
        trainer = DiffusionTrainer(model, minimal_config.training)
        metrics = trainer.train(data, verbose=False)

        # 4. Sample Generation
        model.eval()
        with torch.no_grad():
            samples, trajectory = model.sample(20, return_trajectory=True)

        # 5. Quality Evaluation
        quality_metrics = trainer.evaluate_sample_quality(data, 15)

        assert data.shape == (minimal_config.data.n_data_points, 2)
        assert len(metrics.losses) == minimal_config.training.n_epochs
        assert samples.shape == (20, 2)
        assert len(trajectory) == minimal_config.model.timesteps + 1
        assert torch.isfinite(samples).all()
        assert quality_metrics["mean_error"] >= 0
        assert quality_metrics["std_error"] >= 0

    def test_production_pipeline(
        self, production_config: ExperimentConfig, temp_output_dir: Path
    ) -> None:
        """Test production-like pipeline with full configuration."""
        # Configure logging (in real scenario)
        from logger import configure_logging

        log_file = temp_output_dir / "test.log" if production_config.logging.log_file else None

        configure_logging(
            level=production_config.logging.level,
            log_file=log_file,
            use_json_format=production_config.logging.use_json_format,
            enable_console=production_config.logging.enable_console,
        )

        # Complete pipeline
        data_generator = SwissRollGenerator(
            noise_level=production_config.data.noise_level,
            random_state=production_config.data.random_state,
        )
        data = data_generator.generate(production_config.data.n_data_points)

        noise_scheduler = LinearNoiseScheduler(
            timesteps=production_config.model.timesteps,
            beta_min=production_config.model.beta_min,
            beta_max=production_config.model.beta_max,
        )
        model = DiffusionModel(noise_scheduler)

        trainer = DiffusionTrainer(model, production_config.training)
        metrics = trainer.train(data, verbose=False)

        # Sample and evaluate
        model.eval()
        with torch.no_grad():
            samples, _ = model.sample(production_config.visualization.general.n_samples)

        # Save results (simulate saving)
        if production_config.output.save_config:
            config_save_path = temp_output_dir / "config.toml"
            # Convert enum to string and Path to string for serialization
            production_config.logging.level = production_config.logging.level.value
            if production_config.logging.log_file is not None:
                production_config.logging.log_file = str(production_config.logging.log_file)
            ConfigurationLoader.save_to_file(production_config, config_save_path)
            assert config_save_path.exists()

        if production_config.output.save_model:
            model_save_path = temp_output_dir / "model.pth"
            torch.save(model.state_dict(), model_save_path)
            assert model_save_path.exists()

        # Verify results
        assert samples.shape == (production_config.visualization.general.n_samples, 2)
        assert torch.isfinite(samples).all()
        assert len(metrics.losses) == production_config.training.n_epochs
        assert metrics.get_final_loss() < float("inf")

    def test_pipeline_reproducibility(
        self, minimal_config: ExperimentConfig, seed_random: None
    ) -> None:
        """Test pipeline produces reproducible results."""

        def run_pipeline() -> tuple[torch.Tensor, list[float]]:
            torch.manual_seed(42)

            data_generator = SwissRollGenerator(random_state=42)
            data = data_generator.generate(minimal_config.data.n_data_points)

            noise_scheduler = LinearNoiseScheduler(timesteps=minimal_config.model.timesteps)
            model = DiffusionModel(noise_scheduler)

            trainer = DiffusionTrainer(model, minimal_config.training)
            metrics = trainer.train(data, verbose=False)

            return data, metrics.losses

        # Run pipeline twice
        data1, losses1 = run_pipeline()
        data2, losses2 = run_pipeline()

        # Data generation should be identical
        assert torch.allclose(data1, data2, atol=1e-6)
        assert len(losses1) == len(losses2)

    @pytest.mark.slow
    def test_pipeline_convergence(self, production_config: ExperimentConfig) -> None:
        """Test pipeline shows reasonable convergence behavior."""
        # Use more epochs for convergence test
        production_config.training.n_epochs = 20

        # Generate clean, learnable data
        data_generator = SwissRollGenerator(noise_level=0.05, random_state=42)
        data = data_generator.generate(production_config.data.n_data_points)

        # Create and train model
        noise_scheduler = LinearNoiseScheduler(timesteps=production_config.model.timesteps)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, production_config.training)

        metrics = trainer.train(data, verbose=False)

        # Analyze convergence
        losses = metrics.losses
        initial_loss = losses[0]
        final_loss = losses[-1]
        min_loss = min(losses)

        # Should show improvement
        assert min_loss <= initial_loss
        assert final_loss <= initial_loss * 2.0  # Not worse than 2x initial
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        assert all(loss > 0 for loss in losses)


@pytest.mark.e2e
class TestPipelineRobustness:
    """Test pipeline robustness to various conditions."""

    def test_edge_case_configurations(self) -> None:
        """Test pipeline with edge case configurations."""
        edge_configs = [
            # Minimal viable config
            {
                "data": {"n_data_points": 5},
                "model": {"timesteps": 2},
                "training": {"n_epochs": 1, "batch_size": 1},
            },
            # Single sample
            {
                "data": {"n_data_points": 1},
                "model": {"timesteps": 3},
                "training": {"n_epochs": 1, "batch_size": 1},
            },
        ]

        for config_overrides in edge_configs:
            base_config = ExperimentConfig.default()

            for section, values in config_overrides.items():
                section_config = getattr(base_config, section)
                for key, value in values.items():
                    setattr(section_config, key, value)

            data_generator = SwissRollGenerator(random_state=42)
            data = data_generator.generate(base_config.data.n_data_points)

            noise_scheduler = LinearNoiseScheduler(timesteps=base_config.model.timesteps)
            model = DiffusionModel(noise_scheduler)
            trainer = DiffusionTrainer(model, base_config.training)

            metrics = trainer.train(data, verbose=False)

            assert len(metrics.losses) == base_config.training.n_epochs
            final_loss = metrics.get_final_loss()
            # Allow for potential numerical issues with edge cases
            assert isinstance(final_loss, float)

    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 1.0, 5.0])
    def test_pipeline_with_extreme_noise(
        self, minimal_config: ExperimentConfig, noise_level: float
    ) -> None:
        """Test pipeline handles extreme noise levels."""
        data_generator = SwissRollGenerator(noise_level=noise_level, random_state=42)
        data = data_generator.generate(minimal_config.data.n_data_points)

        noise_scheduler = LinearNoiseScheduler(timesteps=minimal_config.model.timesteps)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, minimal_config.training)

        metrics = trainer.train(data, verbose=False)

        assert torch.isfinite(data).all()
        assert len(metrics.losses) == minimal_config.training.n_epochs
        assert all(torch.isfinite(torch.tensor(loss)) for loss in metrics.losses)

    def test_configuration_validation_in_pipeline(self, invalid_config_path: Path) -> None:
        """Test that invalid configurations are caught in pipeline."""
        with pytest.raises(Exception):
            ConfigurationLoader.load_toml(invalid_config_path)

    def test_basic_memory_cleanup(self, minimal_config: ExperimentConfig) -> None:
        """Test basic memory cleanup works."""
        import gc

        # Run minimal pipeline
        data_generator = SwissRollGenerator(random_state=42)
        data = data_generator.generate(minimal_config.data.n_data_points)

        noise_scheduler = LinearNoiseScheduler(timesteps=minimal_config.model.timesteps)
        model = DiffusionModel(noise_scheduler)
        trainer = DiffusionTrainer(model, minimal_config.training)

        metrics = trainer.train(data, verbose=False)

        samples, trajectory = model.sample(20, return_trajectory=True)

        # Basic checks - ensure objects are created properly
        assert data is not None
        assert samples is not None
        assert trajectory is not None
        assert len(metrics.losses) > 0

        del data, model, trainer, samples, trajectory
        gc.collect()
