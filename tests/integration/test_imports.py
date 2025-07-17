#!/usr/bin/env python3
"""
Test import structure to verify the visualization module imports work correctly.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_basic_imports():
    """Test basic module imports without external dependencies."""
    print("Testing basic module structure...")

    try:
        # Test config imports
        from config.config import ExperimentConfig

        print("✅ Config module imports work")

        # Test data interface (no torch dependency)
        from data.data_interface import DataGeneratorInterface

        print("✅ Data interface imports work")

        # Test domain interfaces (will fail without torch but we can check structure)
        try:
            from domain.noise_scheduler import NoiseSchedulerInterface

            print("✅ Domain interfaces import work")
        except ImportError as e:
            if "torch" in str(e):
                print("⚠️  Domain imports require torch (expected)")
            else:
                print(f"❌ Domain import error: {e}")
                return False

        # Test visualization imports (the main issue we're fixing)
        try:
            print("Testing visualization imports...")
            from visualization import DiffusionVisualizer, GifCreator, PlotManager

            print("✅ Visualization module imports work!")

            # Test that we can instantiate (will fail without matplotlib but structure should work)
            try:
                from config.config import VisualizationConfig

                config = VisualizationConfig()
                visualizer = DiffusionVisualizer(config)
                print("✅ Visualization instantiation works!")
            except ImportError as e:
                if any(lib in str(e) for lib in ["matplotlib", "torch", "numpy"]):
                    print("⚠️  Visualization requires matplotlib/torch/numpy (expected)")
                else:
                    print(f"❌ Visualization instantiation error: {e}")
                    return False

        except ImportError as e:
            print(f"❌ Visualization import error: {e}")
            return False

        print("\n🎉 All import structure tests passed!")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_training_script_structure():
    """Test that the training script can at least import its modules."""
    print("\nTesting training script import structure...")

    # Test that training script can import config without external deps
    try:
        from config import ConfigurationLoader, ExperimentConfig

        print("✅ Training script can import config modules")

        # Test that all required modules are at least importable
        config = ExperimentConfig.default()
        print("✅ Can create default configuration")

        return True

    except Exception as e:
        print(f"❌ Training script structure test failed: {e}")
        return False


def main():
    """Run import tests."""
    print("🔍 Testing Import Structure")
    print("=" * 40)

    basic_ok = test_basic_imports()
    script_ok = test_training_script_structure()

    print("\n" + "=" * 40)
    if basic_ok and script_ok:
        print("✅ All import structure tests passed!")
        print("The relative import issue has been resolved.")
        print("\nTo run the training script, install dependencies:")
        print("  pip install torch matplotlib numpy tqdm pillow")
        return 0
    else:
        print("❌ Some import structure tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
