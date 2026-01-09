"""
Tests for FastAPI endpoints.

Tests the API server including:
- Model management endpoints
- Text generation endpoints
- Task status endpoints
- Memory optimization endpoints
- System status endpoints
- Health check
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path

# Import after setting path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_model_manager.api.server import app


# =============================================================================
# Test Client Setup
# =============================================================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_app_state():
    """Mock app state dependencies."""
    with patch('local_model_manager.api.server.LLMLoader') as mock_loader, \
         patch('local_model_manager.api.server.GPUMonitor') as mock_monitor, \
         patch('local_model_manager.api.server.MemoryOptimizer') as mock_optimizer, \
         patch('local_model_manager.api.server.ResourceManager') as mock_resource, \
         patch('local_model_manager.api.server.ParallelModelManager') as mock_parallel:

        # Configure mocks
        loader = MagicMock()
        loader.list_loaded_models = Mock(return_value=['model-1', 'model-2'])
        loader.get_memory_info = Mock(return_value={
            'total_vram_gb': 24.0,
            'used_vram_gb': 8.0,
            'available_vram_gb': 16.0,
            'loaded_models': ['model-1', 'model-2'],
            'max_concurrent_models': 3
        })
        loader.downloader = MagicMock()
        loader.downloader.configs = {
            'model-1': MagicMock(name='Model 1', estimated_vram_gb=4.5, specialization='code'),
            'model-2': MagicMock(name='Model 2', estimated_vram_gb=3.0, specialization='creative')
        }
        loader.load_model = AsyncMock(return_value=True)
        loader.unload_model = AsyncMock(return_value=True)
        loader.switch_model = AsyncMock(return_value=True)
        mock_loader.return_value = loader

        monitor = MagicMock()
        monitor.get_current_snapshot = Mock(return_value=MagicMock(
            temperature_c=65.0,
            utilization_percent=45.0,
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            power_usage_watts=250.0
        ))
        monitor.start_monitoring = AsyncMock()
        monitor.stop_monitoring = AsyncMock()
        mock_monitor.return_value = monitor

        optimizer = MagicMock()
        optimizer.optimize_memory = AsyncMock(return_value=MagicMock(
            success=True,
            memory_freed_gb=1.5,
            optimization_time_s=2.0,
            optimizations_applied=['GC', 'Torch'],
            recommendations=[]
        ))
        optimizer.get_optimization_stats = Mock(return_value={})
        mock_optimizer.return_value = optimizer

        resource = MagicMock()
        resource.start = AsyncMock()
        resource.stop = AsyncMock()
        resource.model_resources = {
            'model-1': MagicMock(status=MagicMock(value='loaded'), current_tasks=1, total_tasks_processed=10,
                               average_response_time=0.5, error_count=0),
            'model-2': MagicMock(status=MagicMock(value='loaded'), current_tasks=0, total_tasks_processed=5,
                               average_response_time=0.7, error_count=1)
        }
        resource.set_switching_strategy = Mock()
        mock_resource.return_value = resource

        parallel = MagicMock()
        parallel.start_processor = AsyncMock()
        parallel.stop_processor = AsyncMock()
        parallel.create_task = Mock(return_value=MagicMock(
            task_id='test-001',
            model_id='model-1',
            prompt='Test',
            task_type='code',
            priority=3
        ))
        parallel.submit_task = AsyncMock(return_value='test-001')
        parallel.wait_for_task = AsyncMock(return_value=MagicMock(
            task_id='test-001',
            model_id='model-1',
            result='Generated text',
            tokens_generated=50,
            time_taken=1.5,
            success=True,
            error_message=None
        ))
        parallel.get_task_status = AsyncMock(return_value={
            'task_id': 'test-001',
            'status': 'completed',
            'success': True
        })
        parallel.get_queue_status = Mock(return_value={
            'queued_tasks': 0,
            'running_tasks': 1,
            'completed_tasks': 10
        })
        mock_parallel.return_value = parallel

        # Set app state (normally done in lifespan)
        app.state.llm_loader = loader
        app.state.gpu_monitor = monitor
        app.state.memory_optimizer = optimizer
        app.state.resource_manager = resource
        app.state.parallel_manager = parallel

        yield {
            'loader': loader,
            'monitor': monitor,
            'optimizer': optimizer,
            'resource': resource,
            'parallel': parallel
        }


# =============================================================================
# Test Basic Endpoints
# =============================================================================

class TestBasicEndpoints:
    """Test basic API endpoints."""

    @pytest.mark.unit
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


# =============================================================================
# Test Model Management
# =============================================================================

class TestModelManagement:
    """Test model management endpoints."""

    @pytest.mark.unit
    def test_list_models(self, client, mock_app_state):
        """Test listing models."""
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.unit
    def test_load_model(self, client, mock_app_state):
        """Test loading a model."""
        response = client.post("/models/load", json={"model_id": "model-1"})
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @pytest.mark.unit
    def test_unload_model(self, client, mock_app_state):
        """Test unloading a model."""
        response = client.post("/models/unload/model-1")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @pytest.mark.unit
    def test_switch_model(self, client, mock_app_state):
        """Test switching models."""
        response = client.post("/models/switch", json={
            "from_model": "model-1",
            "to_model": "model-2"
        })
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @pytest.mark.unit
    def test_set_switching_strategy(self, client, mock_app_state):
        """Test setting switching strategy."""
        response = client.post("/models/switching-strategy", json={"strategy": "lru"})
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


# =============================================================================
# Test Text Generation
# =============================================================================

class TestTextGeneration:
    """Test text generation endpoints."""

    @pytest.mark.unit
    def test_generate_text(self, client, mock_app_state):
        """Test synchronous text generation."""
        response = client.post("/generate", json={
            "prompt": "Write hello world",
            "task_type": "code",
            "max_tokens": 100
        })
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "text" in data
        assert data["success"] is True

    @pytest.mark.unit
    def test_generate_text_async(self, client, mock_app_state):
        """Test async text generation."""
        response = client.post("/generate/async", json={
            "prompt": "Write hello world",
            "task_type": "code"
        })
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "status" in data


# =============================================================================
# Test Task Management
# =============================================================================

class TestTaskManagement:
    """Test task management endpoints."""

    @pytest.mark.unit
    def test_get_task_status(self, client, mock_app_state):
        """Test getting task status."""
        response = client.get("/tasks/test-001")
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "status" in data

    @pytest.mark.unit
    def test_list_tasks(self, client, mock_app_state):
        """Test listing tasks."""
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "queue_status" in data


# =============================================================================
# Test System Status
# =============================================================================

class TestSystemStatus:
    """Test system status endpoints."""

    @pytest.mark.unit
    def test_get_system_status(self, client, mock_app_state):
        """Test getting system status."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "loaded_models" in data
        assert "total_vram_gb" in data
        assert "gpu_temperature_c" in data
        assert "queued_tasks" in data

    @pytest.mark.unit
    def test_health_check(self, client, mock_app_state):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]


# =============================================================================
# Test Memory Management
# =============================================================================

class TestMemoryManagement:
    """Test memory management endpoints."""

    @pytest.mark.unit
    def test_optimize_memory(self, client, mock_app_state):
        """Test memory optimization."""
        response = client.post("/memory/optimize?aggressive=false")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "memory_freed_gb" in data

    @pytest.mark.unit
    def test_get_memory_stats(self, client, mock_app_state):
        """Test getting memory statistics."""
        response = client.get("/memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert "memory_stats" in data
        assert "optimization_stats" in data


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.unit
    def test_load_nonexistent_model(self, client, mock_app_state):
        """Test loading non-existent model."""
        mock_app_state['loader'].load_model = AsyncMock(return_value=False)
        response = client.post("/models/load", json={"model_id": "nonexistent"})
        assert response.status_code == 400

    @pytest.mark.unit
    def test_get_nonexistent_task(self, client, mock_app_state):
        """Test getting status of non-existent task."""
        mock_app_state['parallel'].get_task_status = AsyncMock(return_value=None)
        response = client.get("/tasks/nonexistent")
        assert response.status_code == 404

    @pytest.mark.unit
    def test_invalid_switching_strategy(self, client, mock_app_state):
        """Test setting invalid switching strategy."""
        response = client.post("/models/switching-strategy", json={"strategy": "invalid"})
        assert response.status_code == 400
