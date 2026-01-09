"""
Quick verification test to ensure test suite is properly configured.

Run this first to verify everything is working:
    python tests/test_quick_verify.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    # Test pytest
    import pytest
    print(f"✅ pytest {pytest.__version__}")

    # Test pytest-asyncio
    import pytest_asyncio
    print(f"✅ pytest-asyncio installed")

    # Test local modules
    from local_model_manager.core.llm_loader import LLMLoader, GPUMemoryMonitor
    from local_model_manager.core.parallel_manager import ParallelModelManager
    from local_model_manager.core.resource_manager import ResourceManager
    from local_model_manager.monitoring.gpu_monitor import GPUMonitor
    from local_model_manager.monitoring.memory_optimizer import MemoryOptimizer
    from local_model_manager.api.client import LocalModelClient
    print("✅ All local_model_manager modules imported")

    # Test conftest fixtures
    from tests.conftest import test_config_path
    print("✅ Conftest accessible")

    return True


def test_fixtures():
    """Test that fixtures work."""
    print("\nTesting fixtures...")

    import pytest
    from tests.conftest import mock_gpu, mock_llama_model

    # Create mock objects
    gpu = mock_gpu()
    print(f"✅ Mock GPU created: {gpu.name}")

    model = mock_llama_model()
    print(f"✅ Mock Llama model created")

    return True


def test_pytest_config():
    """Test pytest configuration."""
    print("\nTesting pytest configuration...")

    import pytest
    import configparser

    # Read pytest.ini
    config_path = Path(__file__).parent / "pytest.ini"
    assert config_path.exists(), "pytest.ini not found"
    print(f"✅ pytest.ini found at {config_path}")

    # Parse config
    config = configparser.ConfigParser()
    config.read(config_path)
    assert 'pytest' in config.sections()
    print("✅ pytest.ini is valid")

    # Check markers
    markers = config['pytest']['markers'].strip().split('\n')
    print(f"✅ {len(markers)} markers defined")

    return True


def test_file_structure():
    """Test that all test files exist."""
    print("\nTesting file structure...")

    test_dir = Path(__file__).parent
    test_files = [
        "conftest.py",
        "pytest.ini",
        "test_model_loader.py",
        "test_parallel_manager.py",
        "test_resource_manager.py",
        "test_gpu_monitor.py",
        "test_memory_optimizer.py",
        "test_api.py",
        "test_client.py",
        "test_integration.py",
        "run_tests.py",
        "README.md",
        "__init__.py"
    ]

    for test_file in test_files:
        file_path = test_dir / test_file
        assert file_path.exists(), f"{test_file} not found"
        print(f"✅ {test_file} exists")

    return True


def main():
    """Run all verification tests."""
    print("="*70)
    print("Local Model Manager - Test Suite Verification")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("Fixtures", test_fixtures),
        ("Pytest Config", test_pytest_config),
        ("File Structure", test_file_structure)
    ]

    failed = []

    for name, test_func in tests:
        try:
            if not test_func():
                failed.append(name)
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            failed.append(name)

    print("\n" + "="*70)
    if failed:
        print(f"❌ Verification failed: {', '.join(failed)}")
        return 1
    else:
        print("✅ All verification tests passed!")
        print("\nYou can now run the full test suite:")
        print("  python tests/run_tests.py")
        print("\nOr run specific tests:")
        print("  pytest tests/test_model_loader.py -v")
        return 0


if __name__ == "__main__":
    sys.exit(main())
