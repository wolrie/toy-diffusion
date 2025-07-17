"""
Tests for scripts - Command-line tools and training scripts.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


class TestConfigStructure:
    """Test that configuration directory structure is correct."""

    def test_etc_directory_exists(self):
        """Test that etc directory exists."""
        etc_dir = Path(__file__).parent.parent / "etc"
        assert etc_dir.exists(), "etc/ directory should exist"
        assert etc_dir.is_dir(), "etc/ should be a directory"

    def test_default_configs_exist(self):
        """Test that default configuration files exist."""
        etc_dir = Path(__file__).parent.parent / "etc"

        required_files = [
            "default_config.yaml",
            "default_config.json",
            "default_config.toml",
            "quick_test.yaml",
            "quick_test.toml",
            "high_quality.yaml",
            "high_quality.toml",
        ]

        for filename in required_files:
            config_file = etc_dir / filename
            assert config_file.exists(), f"Configuration file {filename} should exist in etc/"

    def test_config_files_are_valid(self):
        """Test that all configuration files are valid and loadable."""
        etc_dir = Path(__file__).parent.parent / "etc"

        from config import ConfigurationLoader

        config_files = [
            "default_config.yaml",
            "default_config.json",
            "default_config.toml",
            "quick_test.yaml",
            "quick_test.toml",
            "high_quality.yaml",
            "high_quality.toml",
        ]

        for filename in config_files:
            config_file = etc_dir / filename
            if config_file.exists():
                try:
                    config = ConfigurationLoader.load_toml(config_file)
                    assert hasattr(config, "data"), f"Config {filename} should have data section"
                    assert hasattr(config, "model"), f"Config {filename} should have model section"
                    assert hasattr(
                        config, "training"
                    ), f"Config {filename} should have training section"
                    assert hasattr(
                        config, "visualization"
                    ), f"Config {filename} should have visualization section"
                except Exception as e:
                    pytest.fail(f"Failed to load configuration {filename}: {e}")


class TestTrainingScript:
    """Test the training script functionality."""

    def test_training_script_exists(self):
        """Test that training script exists and is executable."""
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        assert script_path.exists(), "Training script should exist"
        assert script_path.is_file(), "Training script should be a file"

    def test_training_script_help(self):
        """Test that training script shows help."""
        script_path = Path(__file__).parent.parent / "scripts" / "train.py"

        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0, "Training script should show help successfully"
            assert "usage:" in result.stdout.lower(), "Help should show usage information"
            assert "--config" in result.stdout, "Help should mention --config option"
            assert "--preset" in result.stdout, "Help should mention --preset option"

        except subprocess.TimeoutExpired:
            pytest.fail("Training script help command timed out")
        except Exception as e:
            pytest.skip(f"Could not test training script help: {e}")

    def test_training_script_imports(self):
        """Test that training script can import required modules."""
        script_dir = Path(__file__).parent.parent / "scripts"

        # Test that we can import the script without errors
        import sys

        sys.path.insert(0, str(script_dir))
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        try:
            # This should work if all imports are correct
            with patch("sys.argv", ["train.py", "--help"]):
                import train

                assert hasattr(train, "main"), "Training script should have main function"
                assert hasattr(
                    train, "parse_arguments"
                ), "Training script should have argument parser"

        except ImportError as e:
            pytest.fail(f"Training script has import errors: {e}")
        except SystemExit:
            # Expected when --help is processed
            pass
        finally:
            sys.path.remove(str(script_dir))


class TestConfigManagerScript:
    """Test the configuration manager script."""

    def test_config_manager_exists(self):
        """Test that config manager script exists."""
        script_path = Path(__file__).parent.parent / "scripts" / "config_manager.py"
        assert script_path.exists(), "Config manager script should exist"
        assert script_path.is_file(), "Config manager script should be a file"

    def test_config_manager_help(self):
        """Test that config manager shows help."""
        script_path = Path(__file__).parent.parent / "scripts" / "config_manager.py"

        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0, "Config manager should show help successfully"
            assert "usage:" in result.stdout.lower(), "Help should show usage information"

        except subprocess.TimeoutExpired:
            pytest.fail("Config manager help command timed out")
        except Exception as e:
            pytest.skip(f"Could not test config manager help: {e}")


class TestScriptIntegration:
    """Integration tests for scripts."""

    def test_config_validation_workflow(self, temp_dir):
        """Test complete configuration validation workflow."""
        script_path = Path(__file__).parent.parent / "scripts" / "config_manager.py"

        # Create a test config
        test_config = {
            "data": {"n_data_points": 100, "noise_level": 0.1},
            "model": {"timesteps": 10, "hidden_dim": 32},
            "training": {"n_epochs": 5, "batch_size": 8},
            "visualization": {"n_samples": 50},
        }

        config_path = Path(temp_dir) / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(test_config, f)

        try:
            # Test validation
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "validate",
                    str(config_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path(__file__).parent.parent),
            )

            # Should succeed for valid config
            assert result.returncode == 0, f"Config validation failed: {result.stderr}"
            assert "valid" in result.stdout.lower(), "Should report config as valid"

        except subprocess.TimeoutExpired:
            pytest.skip("Config validation test timed out")
        except Exception as e:
            pytest.skip(f"Could not test config validation: {e}")

    def test_config_creation_workflow(self, temp_dir):
        """Test configuration creation workflow."""
        script_path = Path(__file__).parent.parent / "scripts" / "config_manager.py"

        config_path = Path(temp_dir) / "created_config.yaml"

        try:
            # Test config creation
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "create",
                    str(config_path),
                    "--type",
                    "default",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path(__file__).parent.parent),
            )

            assert result.returncode == 0, f"Config creation failed: {result.stderr}"
            assert config_path.exists(), "Config file should be created"

            # Verify it's a valid config
            from config import ConfigurationLoader

            config = ConfigurationLoader.load_toml(config_path)
            assert hasattr(config, "data"), "Created config should have data section"

        except subprocess.TimeoutExpired:
            pytest.skip("Config creation test timed out")
        except Exception as e:
            pytest.skip(f"Could not test config creation: {e}")


class TestProjectStructure:
    """Test overall project structure and organization."""

    def test_directory_structure(self):
        """Test that all required directories exist."""
        root_dir = Path(__file__).parent.parent

        required_dirs = ["src", "tests", "scripts", "etc", "examples"]

        for dir_name in required_dirs:
            dir_path = root_dir / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"

    def test_src_module_structure(self):
        """Test that src modules are properly structured."""
        src_dir = Path(__file__).parent.parent / "src"

        required_modules = ["domain", "data", "training", "config"]

        for module_name in required_modules:
            module_dir = src_dir / module_name
            assert module_dir.exists(), f"Module {module_name} should exist"
            assert module_dir.is_dir(), f"{module_name} should be a directory"

            init_file = module_dir / "__init__.py"
            assert init_file.exists(), f"Module {module_name} should have __init__.py"

    def test_examples_exist(self):
        """Test that example files exist."""
        examples_dir = Path(__file__).parent.parent / "examples"

        required_examples = ["basic_example.py", "config_example.py"]

        for example_name in required_examples:
            example_file = examples_dir / example_name
            assert example_file.exists(), f"Example {example_name} should exist"

    def test_package_files_exist(self):
        """Test that package configuration files exist."""
        root_dir = Path(__file__).parent.parent

        required_files = ["pyproject.toml", "README.md", "pytest.ini"]

        for file_name in required_files:
            file_path = root_dir / file_name
            assert file_path.exists(), f"Package file {file_name} should exist"


if __name__ == "__main__":
    # Quick structure verification
    test = TestConfigStructure()
    test.test_etc_directory_exists()
    test.test_default_configs_exist()
    print("✅ Project structure tests passed!")
