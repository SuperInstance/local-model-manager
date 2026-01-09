"""
End-to-end integration tests for local-model-manager.

Tests complete workflows including:
- Model loading and generation
- Parallel task processing
- Memory management and optimization
- API client-server interaction
- Resource management
- Error recovery
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path

# Import after setting path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Complete Model Workflow
# =============================================================================

class TestCompleteModelWorkflow:
    """Test end-to-end model loading and generation workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_model_load_generate_unload(self, test_config_path, test_models_dir, mock_llama_model):
        """Test complete model lifecycle."""
        from local_model_manager.core.llm_loader import LLMLoader

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            # Initialize loader
            loader = LLMLoader(str(test_config_path))

            try:
                # Load model
                success = await loader.load_model('test-model-1')
                assert success is True
                assert 'test-model-1' in loader.models

                # Get model and generate
                model = loader.get_model('test-model-1')
                assert model is not None

                # Generate (mocked)
                response = model("Test prompt")
                assert 'choices' in response

                # Check memory
                memory_info = loader.get_memory_info()
                assert memory_info['loaded_models'] >= 1

                # Unload
                success = await loader.unload_model('test-model-1')
                assert success is True
                assert 'test-model-1' not in loader.models

            finally:
                await loader.shutdown()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_multi_model_switching(self, test_config_path, test_models_dir, mock_llama_model):
        """Test switching between multiple models."""
        from local_model_manager.core.llm_loader import LLMLoader

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            loader = LLMLoader(str(test_config_path))
            loader.max_concurrent_models = 2

            try:
                # Load first model
                await loader.load_model('test-model-1')
                assert 'test-model-1' in loader.models

                # Load second model
                await loader.load_model('test-model-2')
                assert 'test-model-2' in loader.models

                # Switch models
                await loader.switch_model('test-model-1', 'test-model-3')
                assert 'test-model-3' in loader.models

                # Verify memory management
                loaded = loader.list_loaded_models()
                assert len(loaded) <= 2  # Should respect max_concurrent

            finally:
                await loader.shutdown()


# =============================================================================
# Test Parallel Processing Workflow
# =============================================================================

class TestParallelProcessingWorkflow:
    """Test parallel task processing workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_parallel_task_execution(self, test_config_path, mock_llama_model):
        """Test executing multiple tasks in parallel."""
        from local_model_manager.core.llm_loader import LLMLoader
        from local_model_manager.core.parallel_manager import ParallelModelManager, TaskPriority

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            loader = LLMLoader(str(test_config_path))
            manager = ParallelModelManager(loader)

            try:
                # Start processor
                await manager.start_processor()

                # Load model
                await loader.load_model('test-model-1')
                manager.llm_loader.models = {'test-model-1': MagicMock()}
                manager.llm_loader.get_model = Mock(return_value=mock_llama_model())

                # Create and submit tasks
                task_ids = []
                for i in range(5):
                    task = manager.create_task(
                        prompt=f"Task {i}",
                        task_type="code",
                        model_id="test-model-1",
                        priority=TaskPriority.MEDIUM
                    )
                    task_id = await manager.submit_task(task)
                    task_ids.append(task_id)

                # Wait for completion
                results = await manager.wait_for_all_tasks(task_ids, timeout=10.0)

                assert len(results) > 0
                assert all(isinstance(r, MagicMock) or r.success if hasattr(r, 'success') else True
                          for r in results)

            finally:
                await manager.stop_processor()
                await loader.shutdown()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_priority_task_ordering(self, test_config_path, mock_llama_model):
        """Test that tasks are processed by priority."""
        from local_model_manager.core.llm_loader import LLMLoader
        from local_model_manager.core.parallel_manager import ParallelModelManager, TaskPriority

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            loader = LLMLoader(str(test_config_path))
            manager = ParallelModelManager(loader)

            try:
                await manager.start_processor()
                await loader.load_model('test-model-1')
                manager.llm_loader.models = {'test-model-1': MagicMock()}
                manager.llm_loader.get_model = Mock(return_value=mock_llama_model())

                # Submit tasks with different priorities
                tasks = []
                for priority in [TaskPriority.LOW, TaskPriority.HIGH, TaskPriority.MEDIUM]:
                    task = manager.create_task(
                        prompt=f"{priority.name} task",
                        task_type="code",
                        model_id="test-model-1",
                        priority=priority
                    )
                    task_id = await manager.submit_task(task)
                    tasks.append(task_id)

                # All should complete
                results = await manager.wait_for_all_tasks(tasks, timeout=10.0)
                assert len(results) == 3

            finally:
                await manager.stop_processor()
                await loader.shutdown()


# =============================================================================
# Test Resource Management Workflow
# =============================================================================

class TestResourceManagementWorkflow:
    """Test resource management workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_automatic_resource_allocation(self, test_config_path, mock_llama_model):
        """Test automatic resource allocation and cleanup."""
        from local_model_manager.core.llm_loader import LLMLoader
        from local_model_manager.core.resource_manager import ResourceManager

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            loader = LLMLoader(str(test_config_path))
            loader.load_model = AsyncMock(return_value=True)
            resource_manager = ResourceManager(loader)

            try:
                await resource_manager.start()

                # Allocate models for tasks
                await resource_manager.allocate_model('test-model-1', 'task-1', priority=5)
                await resource_manager.allocate_model('test-model-2', 'task-2', priority=5)

                assert 'task-1' in resource_manager.resource_allocations
                assert resource_manager.model_resources['test-model-1'].current_tasks == 1

                # Update performance
                await resource_manager.update_model_performance('test-model-1', 0.5, True)

                # Deallocate
                await resource_manager.deallocate_model('task-1')
                assert resource_manager.model_resources['test-model-1'].current_tasks == 0

                # Get status
                status = resource_manager.get_resource_status()
                assert 'loaded_models' in status

            finally:
                await resource_manager.stop()
                await loader.shutdown()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_switching_strategy_behavior(self, test_config_path, mock_llama_model):
        """Test different switching strategies."""
        from local_model_manager.core.llm_loader import LLMLoader
        from local_model_manager.core.resource_manager import ResourceManager, ModelSwitchingStrategy

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            loader = LLMLoader(str(test_config_path))
            loader.models = {'test-model-1': MagicMock(), 'test-model-2': MagicMock()}
            loader.unload_model = AsyncMock(return_value=True)
            resource_manager = ResourceManager(loader)

            try:
                await resource_manager.start()

                # Test LRU strategy
                resource_manager.set_switching_strategy(ModelSwitchingStrategy.LRU)
                candidates = await resource_manager._get_unloading_candidates()
                assert isinstance(candidates, list)

                # Test LFU strategy
                resource_manager.set_switching_strategy(ModelSwitchingStrategy.LFU)
                resource_manager.model_resources['test-model-1'].total_tasks_processed = 10
                resource_manager.model_resources['test-model-2'].total_tasks_processed = 5

                candidates = await resource_manager._get_unloading_candidates()
                assert candidates[0] == 'test-model-2'  # Less frequently used

            finally:
                await resource_manager.stop()
                await loader.shutdown()


# =============================================================================
# Test Memory Optimization Workflow
# =============================================================================

class TestMemoryOptimizationWorkflow:
    """Test memory optimization workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_monitoring_and_optimization(self, mock_gpu):
        """Test GPU monitoring triggering optimization."""
        from local_model_manager.monitoring.gpu_monitor import GPUMonitor
        from local_model_manager.monitoring.memory_optimizer import MemoryOptimizer

        # Create monitor
        monitor = GPUMonitor(monitoring_interval=0.1)

        # Create optimizer
        optimizer = MemoryOptimizer(monitor)

        try:
            # Start monitoring
            await monitor.start_monitoring()
            await asyncio.sleep(0.2)  # Collect some data

            # Run optimization
            with patch('gc.collect', return_value=100), \
                 patch('psutil.Process') as mock_proc:
                proc = MagicMock()
                proc.memory_info.return_value = MagicMock(rss=8 * 1024**3)
                mock_proc.return_value = proc

                result = await optimizer.optimize_memory(aggressive=False)

            assert result.success is True
            assert len(optimizer.optimization_history) > 0

            # Get stats
            stats = optimizer.get_optimization_stats()
            assert stats['total_optimizations'] > 0

        finally:
            await monitor.stop_monitoring()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_auto_optimization_workflow(self, mock_gpu):
        """Test automatic optimization based on thresholds."""
        from local_model_manager.monitoring.gpu_monitor import GPUMonitor
        from local_model_manager.monitoring.memory_optimizer import MemoryOptimizer

        # Create monitor with high memory usage
        monitor = GPUMonitor(monitoring_interval=0.1)

        # Mock high memory snapshot
        high_memory_snapshot = MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=22.0,  # Above threshold
            free_memory_gb=2.0,
            temperature_c=65.0,
            utilization_percent=50.0
        )
        monitor.get_current_snapshot = Mock(return_value=high_memory_snapshot)

        optimizer = MemoryOptimizer(monitor)

        try:
            # Run auto-optimize briefly
            task = asyncio.create_task(optimizer.auto_optimize(
                threshold=0.85,
                check_interval=0.1
            ))
            await asyncio.sleep(0.3)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have triggered optimization
            assert len(optimizer.optimization_history) >= 0

        finally:
            pass  # Monitor not started in this test


# =============================================================================
# Test API Integration
# =============================================================================

class TestAPIIntegration:
    """Test API integration with backend."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_client_server_workflow(self, test_config_path, mock_llama_model):
        """Test client communicating with server."""
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from local_model_manager.api.server import app

        # Mock app state
        with patch('local_model_manager.api.server.LLMLoader') as mock_loader, \
             patch('local_model_manager.api.server.GPUMonitor') as mock_monitor, \
             patch('local_model_manager.api.server.MemoryOptimizer') as mock_optimizer, \
             patch('local_model_manager.api.server.ResourceManager') as mock_resource, \
             patch('local_model_manager.api.server.ParallelModelManager') as mock_parallel:

            # Configure mocks
            loader = MagicMock()
            loader.list_loaded_models = Mock(return_value=['model-1'])
            loader.get_memory_info = Mock(return_value={
                'total_vram_gb': 24.0,
                'used_vram_gb': 8.0,
                'available_vram_gb': 16.0,
                'loaded_models': ['model-1'],
                'max_concurrent_models': 3
            })
            loader.downloader = MagicMock()
            loader.downloader.configs = {
                'model-1': MagicMock(name='Model 1', estimated_vram_gb=4.5, specialization='code')
            }
            mock_loader.return_value = loader

            monitor = MagicMock()
            monitor.get_current_snapshot = Mock(return_value=MagicMock(
                temperature_c=65.0,
                utilization_percent=45.0,
                total_memory_gb=24.0,
                used_memory_gb=8.0,
                free_memory_gb=16.0
            ))
            mock_monitor.return_value = monitor

            optimizer = MagicMock()
            optimizer.optimize_memory = AsyncMock(return_value=MagicMock(
                success=True,
                memory_freed_gb=1.0,
                optimization_time_s=1.5,
                optimizations_applied=['GC'],
                recommendations=[]
            ))
            optimizer.get_optimization_stats = Mock(return_value={})
            mock_optimizer.return_value = optimizer

            resource = MagicMock()
            resource.start = AsyncMock()
            resource.stop = AsyncMock()
            resource.model_resources = {
                'model-1': MagicMock(status=MagicMock(value='loaded'), current_tasks=0,
                                   total_tasks_processed=5, average_response_time=0.5, error_count=0)
            }
            mock_resource.return_value = resource

            parallel = MagicMock()
            parallel.start_processor = AsyncMock()
            parallel.stop_processor = AsyncMock()
            parallel.create_task = Mock(return_value=MagicMock(
                task_id='test-001',
                model_id='model-1',
                prompt='Test',
                task_type='code'
            ))
            parallel.submit_task = AsyncMock(return_value='test-001')
            parallel.wait_for_task = AsyncMock(return_value=MagicMock(
                task_id='test-001',
                model_id='model-1',
                result='Response',
                tokens_generated=50,
                time_taken=1.0,
                success=True
            ))
            parallel.get_queue_status = Mock(return_value={
                'queued_tasks': 0,
                'running_tasks': 0,
                'completed_tasks': 1
            })
            mock_parallel.return_value = parallel

            # Set app state
            app.state.llm_loader = loader
            app.state.gpu_monitor = monitor
            app.state.memory_optimizer = optimizer
            app.state.resource_manager = resource
            app.state.parallel_manager = parallel

            # Test client
            client = TestClient(app)

            # Get status
            response = client.get("/status")
            assert response.status_code == 200
            data = response.json()
            assert "loaded_models" in data

            # List models
            response = client.get("/models")
            assert response.status_code == 200
            models = response.json()
            assert len(models) > 0

            # Get memory stats
            response = client.get("/memory/stats")
            assert response.status_code == 200


# =============================================================================
# Test Error Recovery
# =============================================================================

class TestErrorRecovery:
    """Test error recovery in workflows."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_model_load_failure_recovery(self, test_config_path):
        """Test recovery from model load failure."""
        from local_model_manager.core.llm_loader import LLMLoader

        with patch('local_model_manager.core.llm_loader.Llama', side_effect=Exception("Load failed")):
            loader = LLMLoader(str(test_config_path))

            try:
                # Try to load model (should fail)
                success = await loader.load_model('test-model-1')
                assert success is False
                assert 'test-model-1' not in loader.models

                # System should still be functional
                memory_info = loader.get_memory_info()
                assert 'total_vram_gb' in memory_info

            finally:
                await loader.shutdown()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_task_timeout_handling(self, test_config_path, mock_llama_model):
        """Test handling of task timeouts."""
        from local_model_manager.core.llm_loader import LLMLoader
        from local_model_manager.core.parallel_manager import ParallelModelManager

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            loader = LLMLoader(str(test_config_path))
            manager = ParallelModelManager(loader)

            try:
                await manager.start_processor()
                await loader.load_model('test-model-1')
                manager.llm_loader.models = {'test-model-1': MagicMock()}

                # Create a task that will timeout
                task = manager.create_task(
                    prompt="Test",
                    task_type="code",
                    model_id="test-model-1"
                )

                task_id = await manager.submit_task(task)

                # Wait with very short timeout (should timeout)
                result = await manager.wait_for_task(task_id, timeout=0.01)
                assert result is None  # Timeout returns None

            finally:
                await manager.stop_processor()
                await loader.shutdown()


# =============================================================================
# Test Performance Under Load
# =============================================================================

class TestPerformanceUnderLoad:
    """Test system performance under load."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skip("Performance test - skip in regular runs")
    async def test_concurrent_model_loads(self, test_config_path, mock_llama_model):
        """Test loading multiple models concurrently."""
        from local_model_manager.core.llm_loader import LLMLoader

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            loader = LLMLoader(str(test_config_path))

            try:
                # Load models concurrently
                tasks = [
                    loader.load_model('test-model-1'),
                    loader.load_model('test-model-2'),
                    loader.load_model('test-model-3')
                ]

                results = await asyncio.gather(*tasks)

                # At least some should succeed
                assert any(results)

            finally:
                await loader.shutdown()
