# Local Model Manager - Test Suite

Comprehensive test suite for the local-model-manager package with 80%+ coverage target.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Coverage](#coverage)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)

## 🎯 Overview

This test suite provides comprehensive testing for the local-model-manager package including:

- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: End-to-end workflow testing
- **Mocked Dependencies**: External dependencies (GPU, llama-cpp-python) are mocked
- **Async Support**: Full pytest-asyncio integration for testing async code
- **Coverage Reporting**: Detailed coverage reports with HTML output

**Target Coverage**: 80%+

## 📁 Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── pytest.ini                  # Pytest settings
├── run_tests.py                # Test runner script
├── README.md                   # This file
├── test_model_loader.py        # Model loading/unloading tests
├── test_parallel_manager.py    # Parallel execution tests
├── test_resource_manager.py    # Resource management tests
├── test_gpu_monitor.py         # GPU monitoring tests
├── test_memory_optimizer.py    # Memory optimization tests
├── test_api.py                 # FastAPI endpoint tests
├── test_client.py              # Async client tests
└── test_integration.py         # End-to-end workflow tests
```

## 🔧 Installation

Install test dependencies:

```bash
# Install package with test dependencies
pip install -e ".[dev]"

# Or install manually
pip install pytest pytest-asyncio pytest-cov pytest-xdist
```

## 🚀 Running Tests

### Quick Start

Run all tests with coverage:

```bash
cd /mnt/c/users/casey/local-model-manager
python tests/run_tests.py
```

### Using the Test Runner

The `run_tests.py` script provides convenient options:

```bash
# Run all tests with coverage (default)
python tests/run_tests.py

# Run without coverage
python tests/run_tests.py --no-coverage

# Run only unit tests
python tests/run_tests.py --unit

# Run only integration tests
python tests/run_tests.py --integration

# Run only slow tests
python tests/run_tests.py --slow

# Run specific test file
python tests/run_tests.py --file test_model_loader.py

# Run specific test
python tests/run_tests.py --test test_model_loader.py::TestGPUMemoryMonitor::test_init

# Run tests matching pattern
python tests/run_tests.py --pattern "gpu"

# Run tests in parallel (requires pytest-xdist)
python tests/run_tests.py --parallel

# List all tests without running
python tests/run_tests.py --list

# Generate coverage report only
python tests/run_tests.py --coverage-report
```

### Using pytest Directly

You can also use pytest directly:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=local_model_manager --cov-report=html

# Run specific file
pytest tests/test_model_loader.py

# Run specific test
pytest tests/test_model_loader.py::TestGPUMemoryMonitor::test_init

# Run with marker
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Verbose output
pytest -v tests/

# Show output (don't capture)
pytest -s tests/

# Parallel execution
pytest -n auto tests/
```

## 📊 Test Categories

### Unit Tests (`@pytest.mark.unit`)

Fast, isolated tests that mock external dependencies:

- **test_model_loader.py**: Model loading, unloading, switching
- **test_parallel_manager.py**: Task queue, priority handling, execution
- **test_resource_manager.py**: Resource allocation, switching strategies
- **test_gpu_monitor.py**: GPU state capture, memory tracking
- **test_memory_optimizer.py**: Garbage collection, memory cleanup

Run unit tests only:
```bash
pytest -m unit tests/
```

### Integration Tests (`@pytest.mark.integration`)

End-to-end tests that test complete workflows:

- **test_integration.py**: Complete model lifecycle, parallel processing, API integration

Run integration tests only:
```bash
pytest -m integration tests/
```

### Slow Tests (`@pytest.mark.slow`)

Tests that take longer to run (stress tests, performance tests):

- Various performance and stress tests throughout the suite

Run slow tests only:
```bash
pytest -m slow tests/
```

### GPU Tests (`@pytest.mark.gpu`)

Tests that require GPU access (mostly mocked in this suite):

```bash
pytest -m gpu tests/
```

## 📈 Coverage

### View Coverage Reports

After running tests with coverage:

```bash
# Generate HTML coverage report
pytest tests/ --cov=local_model_manager --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Configuration

Coverage is configured in `pytest.ini`:

- **Target**: 80% minimum coverage
- **Reports**: HTML, terminal, XML
- **Omitted**: Test files, examples, scripts

### Current Coverage

View current coverage:

```bash
pytest tests/ --cov=local_model_manager --cov-report=term-missing
```

## ✍️ Writing Tests

### Test Structure

Follow this pattern for new tests:

```python
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch

class TestFeature:
    """Test feature description."""

    @pytest.fixture
    def setup(self):
        """Create test fixtures."""
        # Setup code
        yield
        # Teardown code

    @pytest.mark.unit
    def test_specific_behavior(self, setup):
        """Test that specific behavior works correctly."""
        # Arrange
        # Act
        # Assert
        assert expected == actual

    @pytest.mark.unit
    async def test_async_behavior(self, setup):
        """Test async behavior."""
        # Arrange
        # Act
        result = await async_function()
        # Assert
        assert result is not None
```

### Using Fixtures

Shared fixtures are in `conftest.py`:

```python
def test_with_fixture(mock_gpu, mock_llama_model):
    """Test using shared fixtures."""
    # Use mocked GPU and model
    assert mock_gpu.memoryTotal > 0
```

### Mocking External Dependencies

```python
@patch('local_model_manager.core.llm_loader.Llama')
def test_with_mock(mock_llama):
    """Test with mocked llama-cpp-python."""
    mock_llama.return_value = MagicMock()
    # Test code
```

### Async Tests

Use pytest-asyncio for async tests:

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

## 🔄 CI/CD Integration

### GitHub Actions

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          python tests/run_tests.py --coverage-report
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: Run tests
        entry: python tests/run_tests.py --unit
        language: system
        pass_filenames: false
```

## 📝 Test Checklist

When writing new tests, ensure:

- [ ] Tests are properly marked (unit/integration/slow)
- [ ] External dependencies are mocked
- [ ] Fixtures are used for common setup
- [ ] Both success and failure cases are tested
- [ ] Edge cases are covered
- [ ] Error handling is tested
- [ ] Async functions use `@pytest.mark.asyncio` or `async def`
- [ ] Tests are independent (can run in any order)
- [ ] Descriptive test names and docstrings
- [ ] Coverage remains above 80%

## 🐛 Debugging Tests

### Run with Output

```bash
pytest -s tests/test_file.py  # Don't capture output
pytest -vv tests/test_file.py  # Very verbose
```

### Debug with pdb

```bash
pytest --pdb tests/test_file.py  # Drop into debugger on failure
pytest --trace tests/test_file.py  # Trace execution
```

### Run Last Failed

```bash
pytest --lf  # Run last failed tests
pytest --ff  # Run failed first, then others
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

## 🤝 Contributing

When contributing tests:

1. Follow existing test patterns
2. Keep tests independent and fast
3. Mock external dependencies
4. Maintain >80% coverage
5. Add docstrings to test classes and methods
6. Use descriptive test names

## 📄 License

Same license as the main local-model-manager package.
