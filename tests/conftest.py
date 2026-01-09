"""
Pytest configuration and shared fixtures for local-model-manager tests.

This module provides common fixtures, mocks, and test utilities used across
all test modules.
"""

import asyncio
import os
import tempfile
import yaml
from pathlib import Path
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures for Test Configuration
# =============================================================================

@pytest.fixture
def test_config_path() -> Generator[Path, None, None]:
    """Create a temporary test configuration file."""
    config_data = {
        'system': {
            'max_concurrent_models': 3,
            'safety_margin_gb': 1.0,
            'default_context_size': 4096,
            'default_batch_size': 512
        },
        'models': {
            'test-model-1': {
                'name': 'Test Model 1',
                'repo_id': 'test/test-model-1',
                'gguf_file': 'test-model-1.Q4_K_M.gguf',
                'context_size': 4096,
                'gpu_layers': 35,
                'batch_size': 512,
                'max_tokens': 2048,
                'temperature': 0.7,
                'top_p': 0.9,
                'repeat_penalty': 1.1,
                'estimated_vram_gb': 4.5,
                'specialization': 'code, reasoning, technical'
            },
            'test-model-2': {
                'name': 'Test Model 2',
                'repo_id': 'test/test-model-2',
                'gguf_file': 'test-model-2.Q4_K_M.gguf',
                'context_size': 2048,
                'gpu_layers': 25,
                'batch_size': 256,
                'max_tokens': 1024,
                'temperature': 0.8,
                'top_p': 0.95,
                'repeat_penalty': 1.0,
                'estimated_vram_gb': 3.0,
                'specialization': 'creative, writing, general'
            },
            'test-model-3': {
                'name': 'Test Model 3',
                'repo_id': 'test/test-model-3',
                'gguf_file': 'test-model-3.Q4_K_M.gguf',
                'context_size': 8192,
                'gpu_layers': 30,
                'batch_size': 1024,
                'max_tokens': 4096,
                'temperature': 0.6,
                'top_p': 0.85,
                'repeat_penalty': 1.15,
                'estimated_vram_gb': 6.0,
                'specialization': 'analysis, summarization, general'
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def test_models_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create mock model files
        for model_id in ['test-model-1', 'test-model-2', 'test-model-3']:
            model_dir = models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            # Create a dummy file to simulate model
            (model_dir / f"{model_id}.Q4_K_M.gguf").write_text("mock model data")

        yield models_dir


@pytest.fixture
def test_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yield cache_dir


# =============================================================================
# Mock Fixtures for External Dependencies
# =============================================================================

@pytest.fixture
def mock_llama_model():
    """Mock llama-cpp-python Llama model."""
    mock = MagicMock()
    mock.return_value = MagicMock()
    mock.return_value.__call__ = MagicMock(
        return_value={
            'choices': [{'text': 'Test response'}],
            'usage': {'completion_tokens': 50}
        }
    )
    return mock


@pytest.fixture
def mock_gpu():
    """Mock GPUUtil GPU object."""
    mock = MagicMock()
    mock.name = "NVIDIA GeForce RTX 3090"
    mock.memoryTotal = 24576  # 24GB in MB
    mock.memoryUsed = 8192     # 8GB in MB
    mock.memoryFree = 16384    # 16GB in MB
    mock.temperature = 65.0
    mock.load = 0.45
    mock.powerLoad = 250.0
    mock.driver = "535.104.05"
    return mock


@pytest.fixture
def mock_gputil(mock_gpu):
    """Mock GPUtil module."""
    with patch('GPUtil.getGPUs') as mock_get_gpus:
        mock_get_gpus.return_value = [mock_gpu]
        yield mock_get_gpus


@pytest.fixture
def mock_torch_cuda():
    """Mock PyTorch CUDA."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.memory_allocated', return_value=8 * 1024**3), \
         patch('torch.cuda.empty_cache'), \
         patch('torch.cuda.synchronize'), \
         patch('torch.cuda.reset_peak_memory_stats'):
        yield


@pytest.fixture
def mock_psutil():
    """Mock psutil process monitoring."""
    with patch('psutil.Process') as mock_process:
        proc = MagicMock()
        proc.memory_info.return_value = MagicMock(rss=8 * 1024**3)  # 8GB
        mock_process.return_value = proc
        yield mock_process


@pytest.fixture
def mock_nvidia_smi():
    """Mock nvidia-smi subprocess calls."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345,python,4096\n12346,test_app,2048\n"
        )
        yield mock_run


@pytest.fixture
def mock_aiohttp_response():
    """Mock aiohttp client response."""
    mock = MagicMock()
    mock.status = 200
    mock.headers = {'content-length': '1000'}

    async def mock_json():
        return {"result": "success"}

    mock.json = mock_json

    async def mock_text():
        return "success"

    mock.text = mock_text
    return mock


@pytest.fixture
def mock_aiohttp_session(mock_aiohttp_response):
    """Mock aiohttp client session."""
    with patch('aiohttp.ClientSession') as mock_session:
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock()
        session.get = MagicMock(return_value=mock_aiohttp_response)
        session.post = MagicMock(return_value=mock_aiohttp_response)
        session.request = MagicMock(return_value=mock_aiohttp_response)
        session.close = AsyncMock()
        mock_session.return_value = session
        yield mock_session


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "Summarize the benefits of renewable energy.",
        "Create a short story about a robot learning to paint."
    ]


@pytest.fixture
def sample_generation_params():
    """Sample generation parameters."""
    return {
        'max_tokens': 100,
        'temperature': 0.7,
        'top_p': 0.9,
        'repeat_penalty': 1.1,
        'stop': []
    }


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        'task_id': 'test-task-001',
        'model_id': 'test-model-1',
        'task_type': 'code',
        'priority': 3,  # HIGH
        'prompt': 'Write a hello world function in Python.',
        'params': {'max_tokens': 100, 'temperature': 0.2}
    }


# =============================================================================
# Async Event Loop Fixtures
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_context():
    """Provide async context setup and teardown."""
    setup_done = False

    async def setup():
        nonlocal setup_done
        # Setup code here
        setup_done = True

    async def teardown():
        # Teardown code here
        pass

    await setup()
    yield
    await teardown()


# =============================================================================
# Test Utilities
# =============================================================================

@pytest.fixture
def wait_for_condition():
    """Utility to wait for a condition with timeout."""
    async def _wait_for(condition, timeout=5.0, interval=0.1):
        """Wait for condition to be True or timeout."""
        start_time = asyncio.get_event_loop().time()
        while True:
            if condition():
                return True
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False
            await asyncio.sleep(interval)

    return _wait_for_condition


@pytest.fixture
def create_temp_file():
    """Utility to create temporary files."""
    def _create(content: str, suffix: str = '.tmp') -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return Path(path)

    return _create


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    metrics = {
        'operation_times': [],
        'memory_usage': [],
        'gpu_usage': []
    }

    def record_operation(operation_name: str, duration: float):
        metrics['operation_times'].append({
            'operation': operation_name,
            'duration': duration
        })

    def record_memory(memory_mb: float):
        metrics['memory_usage'].append(memory_mb)

    def record_gpu(gpu_mb: float):
        metrics['gpu_usage'].append(gpu_mb)

    yield {
        'record_operation': record_operation,
        'record_memory': record_memory,
        'record_gpu': record_gpu,
        'metrics': metrics
    }


# =============================================================================
# Marker Registration
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network"
    )
    config.addinivalue_line(
        "markers", "async: mark test as using async/await"
    )
