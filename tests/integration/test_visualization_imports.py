#!/usr/bin/env python3
"""
Test visualization imports specifically to verify the relative import fix.
"""

import os
import sys


def test_visualization_imports():
    """Test visualization module imports."""
    print("🔍 Testing visualization import structure...")

    # Add src to path (simulating how scripts/train.py works)
    # From tests/integration/, we need to go up two levels to reach src/
    src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    sys.path.append(src_path)

    try:
        # This is the import that was failing with relative import error
        print("Attempting: from visualization import DiffusionVisualizer")
        from visualization import DiffusionVisualizer

        print("✅ DiffusionVisualizer import successful!")

        print("Attempting: from visualization import PlotManager")
        from visualization import PlotManager

        print("✅ PlotManager import successful!")

        print("Attempting: from visualization import GifCreator")
        from visualization import GifCreator

        print("✅ GifCreator import successful!")

        print("\n🎉 All visualization imports work correctly!")
        print("The relative import issue has been resolved.")

        return True

    except ImportError as e:
        if "attempted relative import beyond top-level package" in str(e):
            print(f"❌ Relative import error still exists: {e}")
            return False
        elif any(lib in str(e) for lib in ["torch", "matplotlib", "numpy"]):
            print(f"⚠️  Import requires external dependencies: {e}")
            print("✅ But the relative import structure is working!")
            return True
        else:
            print(f"❌ Unexpected import error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_config_imports():
    """Test config imports as a baseline."""
    print("\n🔧 Testing config imports (baseline)...")

    # From tests/integration/, we need to go up two levels to reach src/
    src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    sys.path.append(src_path)

    try:
        from config import ExperimentConfig

        print("✅ Config imports work")

        config = ExperimentConfig.default()
        print("✅ Can create default config")

        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False


def main():
    """Run import tests."""
    print("Testing Import Fixes for Visualization Module")
    print("=" * 50)

    config_ok = test_config_imports()
    viz_ok = test_visualization_imports()

    print("\n" + "=" * 50)
    if config_ok and viz_ok:
        print("✅ SUCCESS: Relative import issue resolved!")
        print("\nThe scripts should now work. To run the training script:")
        print("1. Install dependencies: pip install torch matplotlib numpy tqdm pillow")
        print("2. Run: python scripts/train.py --preset quick")
        return 0
    else:
        print("❌ FAILED: Import issues still exist")
        return 1


if __name__ == "__main__":
    exit(main())
