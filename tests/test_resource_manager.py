"""
Tests for resource management and model switching functionality.

Tests the ResourceManager class including:
- Model allocation and deallocation
- Switching strategies (LRU, LFU, PRIORITY, SPECIALIZATION, HYBRID)
- Automatic model switching
- Performance tracking
- Memory-based switching
- Idle model cleanup
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any

from local_model_manager.core.resource_manager import (
    ResourceManager,
    ModelStatus,
    ModelResourceInfo,
    ResourceAllocation,
    ModelSwitchingStrategy
)


# =============================================================================
# Test ResourceManager Initialization
# =============================================================================

class TestResourceManagerInit:
    """Test ResourceManager initialization."""

    @pytest.fixture
    def mock_loader(self):
        """Create mock LLMLoader."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.memory_monitor.total_vram_gb = 24.0
        loader.memory_monitor.get_vram_usage = Mock(return_value=8.0)
        loader.downloader = MagicMock()
        loader.downloader.configs = {}
        loader.models = {}
        return loader

    @pytest.fixture
    def manager(self, mock_loader):
        """Create ResourceManager instance."""
        manager = ResourceManager(mock_loader)
        return manager

    @pytest.mark.unit
    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.switching_strategy == ModelSwitchingStrategy.HYBRID
        assert manager.auto_switch_threshold == 0.8
        assert manager.max_idle_time == 300
        assert manager.model_resources == {}
        assert manager.resource_allocations == {}

    @pytest.mark.unit
    async def test_start(self, manager, mock_loader):
        """Test starting the resource manager."""
        # Add some model configs
        mock_loader.downloader.configs = {
            'model-1': MagicMock(estimated_vram_gb=4.5, specialization='code'),
            'model-2': MagicMock(estimated_vram_gb=3.0, specialization='creative')
        }

        await manager.start()
        assert manager._running is True
        assert 'model-1' in manager.model_resources
        assert 'model-2' in manager.model_resources
        assert manager._cleanup_task is not None

        await manager.stop()

    @pytest.mark.unit
    async def test_stop(self, manager):
        """Test stopping the resource manager."""
        await manager.start()
        await manager.stop()
        assert manager._running is False


# =============================================================================
# Test Model Resource Initialization
# =============================================================================

class TestModelResources:
    """Test model resource management."""

    @pytest.fixture
    def manager(self):
        """Create manager with test models."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.memory_monitor.total_vram_gb = 24.0
        loader.models = {}
        loader.downloader = MagicMock()

        # Create model configs
        loader.downloader.configs = {
            'model-1': MagicMock(
                estimated_vram_gb=4.5,
                specialization='code, reasoning'
            ),
            'model-2': MagicMock(
                estimated_vram_gb=3.0,
                specialization='creative, writing'
            ),
            'model-3': MagicMock(
                estimated_vram_gb=6.0,
                specialization='general, analysis'
            )
        }

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.unit
    async def test_initialize_model_resources(self, manager):
        """Test model resource initialization."""
        await manager._initialize_model_resources()

        assert 'model-1' in manager.model_resources
        assert 'model-2' in manager.model_resources
        assert 'model-3' in manager.model_resources

        resource = manager.model_resources['model-1']
        assert resource.status == ModelStatus.UNLOADED
        assert resource.vram_usage_gb == 4.5
        assert resource.load_count == 0

    @pytest.mark.unit
    def test_calculate_specialization_scores(self, manager):
        """Test specialization score calculation."""
        scores = manager._calculate_specialization_scores('code, reasoning')

        assert 'code' in scores
        assert 'creative' in scores
        assert scores['code'] > scores['creative']

    @pytest.mark.unit
    async def test_get_best_model_for_task(self, manager):
        """Test getting best model for task type."""
        await manager._initialize_model_resources()

        # Mark model-1 as loaded
        manager.model_resources['model-1'].status = ModelStatus.LOADED
        manager.model_resources['model-1'].average_response_time = 0.5
        manager.model_resources['model-1'].current_tasks = 1

        # Code task should prefer model-1
        best = manager.get_best_model_for_task('code')
        assert best == 'model-1'


# =============================================================================
# Test Model Allocation
# =============================================================================

class TestModelAllocation:
    """Test model allocation functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.models = {}
        loader.load_model = AsyncMock(return_value=True)
        loader.downloader = MagicMock()
        loader.downloader.configs = {
            'model-1': MagicMock(estimated_vram_gb=4.5, specialization='code')
        }

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.unit
    async def test_allocate_model_already_loaded(self, manager):
        """Test allocating an already loaded model."""
        await manager._initialize_model_resources()
        manager.model_resources['model-1'].status = ModelStatus.LOADED
        manager.llm_loader.models = {'model-1': MagicMock()}

        success = await manager.allocate_model('model-1', 'task-001', priority=5)
        assert success is True
        assert 'task-001' in manager.resource_allocations
        assert manager.model_resources['model-1'].current_tasks == 1

    @pytest.mark.unit
    async def test_allocate_model_not_loaded(self, manager):
        """Test allocating a model that's not loaded."""
        await manager._initialize_model_resources()

        success = await manager.allocate_model('model-1', 'task-002', priority=5)
        assert success is True
        assert manager.model_resources['model-1'].status == ModelStatus.LOADED
        assert manager.model_resources['model-1'].load_count == 1

    @pytest.mark.unit
    async def test_deallocate_model(self, manager):
        """Test deallocating a model."""
        await manager._initialize_model_resources()
        manager.model_resources['model-1'].status = ModelStatus.LOADED

        # Allocate
        await manager.allocate_model('model-1', 'task-003', priority=5)
        assert manager.model_resources['model-1'].current_tasks == 1

        # Deallocate
        await manager.deallocate_model('task-003')
        assert 'task-003' not in manager.resource_allocations
        assert manager.model_resources['model-1'].current_tasks == 0
        assert manager.model_resources['model-1'].total_tasks_processed == 1


# =============================================================================
# Test Switching Strategies
# =============================================================================

class TestSwitchingStrategies:
    """Test different model switching strategies."""

    @pytest.fixture
    def manager(self):
        """Create manager with loaded models."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.models = {
            'model-1': MagicMock(),
            'model-2': MagicMock(),
            'model-3': MagicMock()
        }
        loader.unload_model = AsyncMock(return_value=True)
        loader.downloader = MagicMock()
        loader.downloader.configs = {}

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.unit
    async def test_lru_strategy(self, manager):
        """Test Least Recently Used strategy."""
        await manager._initialize_model_resources()
        manager.switching_strategy = ModelSwitchingStrategy.LRU

        # Set last_used times
        current_time = time.time()
        manager.model_resources['model-1'].last_used = current_time - 1000
        manager.model_resources['model-2'].last_used = current_time - 500
        manager.model_resources['model-3'].last_used = current_time - 100

        candidates = await manager._get_unloading_candidates()
        # model-1 should be first (oldest)
        assert candidates[0] == 'model-1'

    @pytest.mark.unit
    async def test_lfu_strategy(self, manager):
        """Test Least Frequently Used strategy."""
        await manager._initialize_model_resources()
        manager.switching_strategy = ModelSwitchingStrategy.LFU

        # Set task counts
        manager.model_resources['model-1'].total_tasks_processed = 10
        manager.model_resources['model-2'].total_tasks_processed = 50
        manager.model_resources['model-3'].total_tasks_processed = 100

        candidates = await manager._get_unloading_candidates()
        # model-1 should be first (least used)
        assert candidates[0] == 'model-1'

    @pytest.mark.unit
    async def test_priority_strategy(self, manager):
        """Test priority-based strategy."""
        await manager._initialize_model_resources()
        manager.switching_strategy = ModelSwitchingStrategy.PRIORITY

        # Add allocations with different priorities
        manager.resource_allocations['task-1'] = ResourceAllocation(
            model_id='model-1',
            allocated_vram_gb=4.5,
            priority=1,  # Low priority
            expires_at=time.time() + 3600,
            task_id='task-1'
        )
        manager.resource_allocations['task-2'] = ResourceAllocation(
            model_id='model-2',
            allocated_vram_gb=3.0,
            priority=10,  # High priority
            expires_at=time.time() + 3600,
            task_id='task-2'
        )

        candidates = await manager._get_unloading_candidates()
        # model-1 should be first (lower priority tasks)
        assert candidates[0] == 'model-1'

    @pytest.mark.unit
    async def test_specialization_strategy(self, manager):
        """Test specialization-based strategy."""
        await manager._initialize_model_resources()
        manager.switching_strategy = ModelSwitchingStrategy.SPECIALIZATION

        # Set specializations
        manager.model_resources['model-1'].specialization_scores = {
            'code': 1.0,
            'creative': 0.3
        }
        manager.model_resources['model-2'].specialization_scores = {
            'code': 0.3,
            'creative': 1.0
        }
        manager.model_resources['model-3'].specialization_scores = {
            'code': 0.5,
            'creative': 0.5
        }

        candidates = await manager._get_unloading_candidates()
        # Should return all models (simplified test)
        assert len(candidates) > 0

    @pytest.mark.unit
    async def test_hybrid_strategy(self, manager):
        """Test hybrid strategy combining multiple factors."""
        await manager._initialize_model_resources()
        manager.switching_strategy = ModelSwitchingStrategy.HYBRID

        # Set various metrics
        current_time = time.time()
        manager.model_resources['model-1'].last_used = current_time - 1000
        manager.model_resources['model-1'].load_count = 10
        manager.model_resources['model-1'].current_tasks = 0
        manager.model_resources['model-1'].error_count = 5

        manager.model_resources['model-2'].last_used = current_time - 100
        manager.model_resources['model-2'].load_count = 100
        manager.model_resources['model-2'].current_tasks = 5
        manager.model_resources['model-2'].error_count = 0

        candidates = await manager._get_unloading_candidates()
        # model-1 should be candidate (old, few loads, no tasks, many errors)
        assert 'model-1' in candidates

    @pytest.mark.unit
    def test_set_switching_strategy(self, manager):
        """Test changing switching strategy."""
        manager.set_switching_strategy(ModelSwitchingStrategy.LRU)
        assert manager.switching_strategy == ModelSwitchingStrategy.LRU

        manager.set_switching_strategy(ModelSwitchingStrategy.LFU)
        assert manager.switching_strategy == ModelSwitchingStrategy.LFU


# =============================================================================
# Test Memory Management
# =============================================================================

class TestMemoryManagement:
    """Test memory-based model switching."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.memory_monitor.total_vram_gb = 24.0
        loader.memory_monitor.get_available_vram = Mock(return_value=2.0)
        loader.models = {
            'model-1': MagicMock(),
            'model-2': MagicMock()
        }
        loader.unload_model = AsyncMock(return_value=True)
        loader.get_memory_info = Mock(return_value={
            'total_vram_gb': 24.0,
            'used_vram_gb': 20.0,
            'available_vram_gb': 4.0
        })
        loader.downloader = MagicMock()
        loader.downloader.configs = {}

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.unit
    async def test_make_space_for_model(self, manager):
        """Test making space for a model."""
        await manager._initialize_model_resources()
        manager.model_resources['model-1'].vram_usage_gb = 4.5
        manager.model_resources['model-2'].vram_usage_gb = 3.0

        # Need 6GB but only 2GB available
        await manager._make_space_for_model('model-3')

        # Should unload models to make space
        assert manager.llm_loader.unload_model.called

    @pytest.mark.unit
    async def test_ensure_model_loaded_sufficient_memory(self, manager):
        """Test loading model when sufficient memory available."""
        await manager._initialize_model_resources()
        manager.memory_monitor.get_available_vram = Mock(return_value=10.0)
        manager.llm_loader.load_model = AsyncMock(return_value=True)

        result = await manager._ensure_model_loaded('model-1')
        assert result is True
        assert manager.model_resources['model-1'].status == ModelStatus.LOADED

    @pytest.mark.unit
    async def test_ensure_model_loaded_insufficient_memory(self, manager):
        """Test loading model when insufficient memory."""
        await manager._initialize_model_resources()
        manager.memory_monitor.get_available_vram = Mock(return_value=1.0)
        manager.llm_loader.load_model = AsyncMock(return_value=True)
        manager.llm_loader.models = {}

        result = await manager._ensure_model_loaded('model-1')
        assert result is True  # Should still try to load


# =============================================================================
# Test Performance Tracking
# =============================================================================

class TestPerformanceTracking:
    """Test performance tracking functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.models = {}
        loader.downloader = MagicMock()
        loader.downloader.configs = {}

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.unit
    async def test_update_model_performance(self, manager):
        """Test updating model performance metrics."""
        await manager._initialize_model_resources()

        # Update with successful request
        await manager.update_model_performance('model-1', 0.5, True)
        assert manager.model_resources['model-1'].average_response_time == 0.5
        assert manager.model_resources['model-1'].error_count == 0

        # Update with failed request
        await manager.update_model_performance('model-1', 0.7, False)
        assert manager.model_resources['model-1'].error_count == 1

    @pytest.mark.unit
    async def test_performance_history_tracking(self, manager):
        """Test that performance history is tracked."""
        await manager._initialize_model_resources()

        for i in range(10):
            await manager.update_model_performance('model-1', 0.5, True)

        # Check history exists
        assert 'model-1' in manager.model_performance_history
        assert len(manager.model_performance_history['model-1']) == 10


# =============================================================================
# Test Cleanup Functionality
# =============================================================================

class TestCleanup:
    """Test cleanup functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.memory_monitor.get_vram_usage = Mock(return_value=12.0)
        loader.memory_monitor.total_vram_gb = 24.0
        loader.models = {
            'model-1': MagicMock(),
            'model-2': MagicMock()
        }
        loader.unload_model = AsyncMock(return_value=True)
        loader.downloader = MagicMock()
        loader.downloader.configs = {}

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.unit
    async def test_cleanup_idle_resources(self, manager):
        """Test cleanup of idle models."""
        await manager._initialize_model_resources()

        # Mark model-1 as loaded and idle
        manager.model_resources['model-1'].status = ModelStatus.LOADED
        manager.model_resources['model-1'].last_used = time.time() - 400  # 6+ minutes ago
        manager.model_resources['model-1'].current_tasks = 0

        # Mark model-2 as loaded and active
        manager.model_resources['model-2'].status = ModelStatus.LOADED
        manager.model_resources['model-2'].last_used = time.time() - 60
        manager.model_resources['model-2'].current_tasks = 2

        await manager._cleanup_idle_resources()

        # model-1 should be unloaded
        assert manager.llm_loader.unload_model.called

    @pytest.mark.unit
    async def test_cleanup_expired_allocations(self, manager):
        """Test cleanup of expired allocations."""
        await manager._initialize_model_resources()

        # Add expired allocation
        manager.resource_allocations['expired-task'] = ResourceAllocation(
            model_id='model-1',
            allocated_vram_gb=4.5,
            priority=5,
            expires_at=time.time() - 100,  # Expired
            task_id='expired-task'
        )

        await manager._cleanup_idle_resources()

        # Should be removed
        assert 'expired-task' not in manager.resource_allocations


# =============================================================================
# Test Status Reporting
# =============================================================================

class TestStatusReporting:
    """Test status reporting functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.memory_monitor.total_vram_gb = 24.0
        loader.memory_monitor.get_vram_usage = Mock(return_value=12.0)
        loader.models = {'model-1': MagicMock()}
        loader.get_memory_info = Mock(return_value={
            'total_vram_gb': 24.0,
            'used_vram_gb': 12.0,
            'available_vram_gb': 12.0
        })
        loader.downloader = MagicMock()
        loader.downloader.configs = {}

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.unit
    async def test_get_resource_status(self, manager):
        """Test getting resource status."""
        await manager._initialize_model_resources()
        manager.model_resources['model-1'].status = ModelStatus.LOADED
        manager.model_resources['model-1'].current_tasks = 2

        status = manager.get_resource_status()

        assert 'loaded_models' in status
        assert 'model_resources' in status
        assert 'active_allocations' in status
        assert 'memory_info' in status
        assert 'switching_strategy' in status

        assert 'model-1' in status['loaded_models']


# =============================================================================
# Integration Tests
# =============================================================================

class TestResourceManagerIntegration:
    """Integration tests for resource manager."""

    @pytest.fixture
    def real_manager(self):
        """Create realistic manager."""
        loader = MagicMock()
        loader.memory_monitor = MagicMock()
        loader.memory_monitor.total_vram_gb = 24.0
        loader.memory_monitor.get_vram_usage = Mock(return_value=8.0)
        loader.memory_monitor.get_available_vram = Mock(return_value=16.0)
        loader.models = {}
        loader.load_model = AsyncMock(return_value=True)
        loader.unload_model = AsyncMock(return_value=True)
        loader.get_memory_info = Mock(return_value={
            'total_vram_gb': 24.0,
            'used_vram_gb': 8.0,
            'available_vram_gb': 16.0,
            'loaded_models': [],
            'max_concurrent_models': 3
        })
        loader.downloader = MagicMock()
        loader.downloader.configs = {
            'model-1': MagicMock(estimated_vram_gb=4.5, specialization='code'),
            'model-2': MagicMock(estimated_vram_gb=3.0, specialization='creative'),
            'model-3': MagicMock(estimated_vram_gb=6.0, specialization='analysis')
        }

        manager = ResourceManager(loader)
        return manager

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_allocation_workflow(self, real_manager):
        """Test complete allocation workflow."""
        await real_manager.start()

        # Allocate models for tasks
        await real_manager.allocate_model('model-1', 'task-1', priority=5)
        await real_manager.allocate_model('model-2', 'task-2', priority=5)

        assert 'task-1' in real_manager.resource_allocations
        assert 'task-2' in real_manager.resource_allocations

        # Update performance
        await real_manager.update_model_performance('model-1', 0.5, True)
        await real_manager.update_model_performance('model-2', 0.7, True)

        # Deallocate
        await real_manager.deallocate_model('task-1')
        await real_manager.deallocate_model('task-2')

        assert 'task-1' not in real_manager.resource_allocations
        assert 'task-2' not in real_manager.resource_allocations

        await real_manager.stop()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_automatic_switching(self, real_manager):
        """Test automatic model switching based on memory."""
        await real_manager.start()

        # Allocate models until memory is high
        await real_manager.allocate_model('model-1', 'task-1', priority=5)
        await real_manager.allocate_model('model-2', 'task-2', priority=5)

        # Simulate high memory usage
        real_manager.memory_monitor.get_vram_usage = Mock(return_value=20.0)
        real_manager.llm_loader.get_memory_info = Mock(return_value={
            'total_vram_gb': 24.0,
            'used_vram_gb': 20.0,
            'available_vram_gb': 4.0
        })

        # Try to allocate third model (should trigger switching)
        await real_manager.allocate_model('model-3', 'task-3', priority=5)

        # Should have unloaded some models
        assert real_manager.llm_loader.unload_model.called

        await real_manager.stop()
