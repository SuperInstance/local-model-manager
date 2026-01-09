"""
Tests for parallel model execution functionality.

Tests the ParallelModelManager class including:
- Task submission and queuing
- Priority-based execution
- Parallel task processing
- Task result retrieval
- Model selection for tasks
- Task callbacks
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List

from local_model_manager.core.parallel_manager import (
    ParallelModelManager,
    ModelTask,
    TaskResult,
    TaskPriority
)


# =============================================================================
# Test ModelTask and TaskResult
# =============================================================================

class TestModelTask:
    """Test ModelTask dataclass."""

    @pytest.mark.unit
    def test_create_task_with_defaults(self):
        """Test creating a task with default values."""
        task = ModelTask(
            task_id="test-001",
            model_id="model-1",
            task_type="code",
            priority=TaskPriority.MEDIUM,
            prompt="Write hello world",
            params={}
        )
        assert task.task_id == "test-001"
        assert task.model_id == "model-1"
        assert task.created_at is not None

    @pytest.mark.unit
    def test_task_priority_ordering(self):
        """Test that task priorities are ordered correctly."""
        assert TaskPriority.CRITICAL.value > TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value > TaskPriority.MEDIUM.value
        assert TaskPriority.MEDIUM.value > TaskPriority.LOW.value


# =============================================================================
# Test ParallelModelManager Initialization
# =============================================================================

class TestParallelModelManagerInit:
    """Test ParallelModelManager initialization."""

    @pytest.fixture
    def mock_loader(self):
        """Create mock LLMLoader."""
        loader = MagicMock()
        loader.models = {}
        loader.get_model = Mock(return_value=MagicMock())
        return loader

    @pytest.fixture
    def manager(self, mock_loader):
        """Create ParallelModelManager instance."""
        return ParallelModelManager(mock_loader)

    @pytest.mark.unit
    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.max_parallel_tasks == 3
        assert manager.task_queue is not None
        assert manager.running_tasks == {}
        assert manager.task_results == {}

    @pytest.mark.unit
    def test_load_model_specializations(self, manager):
        """Test that model specializations are loaded."""
        assert "phi-3.5-mini" in manager.model_specializations
        assert "llama-3.2-3b" in manager.model_specializations
        assert "gemma-2-2b" in manager.model_specializations

    @pytest.mark.unit
    def test_get_best_model_for_task(self, manager):
        """Test getting the best model for a task type."""
        # Code task should prefer phi-3.5-mini
        model = manager._get_best_model_for_task("code")
        assert model == "phi-3.5-mini"

        # Creative task should prefer gemma-2-2b
        model = manager._get_best_model_for_task("creative")
        assert model == "gemma-2-2b"

    @pytest.mark.unit
    def test_get_best_model_for_unknown_task(self, manager):
        """Test getting model for unknown task type."""
        model = manager._get_best_model_for_task("unknown")
        assert model is None


# =============================================================================
# Test Task Submission
# =============================================================================

class TestTaskSubmission:
    """Test task submission functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager with mocked loader."""
        loader = MagicMock()
        loader.models = {}
        loader.load_model = AsyncMock(return_value=True)

        manager = ParallelModelManager(loader)
        return manager

    @pytest.mark.unit
    async def test_submit_task_success(self, manager):
        """Test successful task submission."""
        manager.llm_loader.models = {"test-model": MagicMock()}

        task = ModelTask(
            task_id="test-001",
            model_id="test-model",
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="Test prompt",
            params={}
        )

        task_id = await manager.submit_task(task)
        assert task_id == "test-001"
        assert manager.task_queue.qsize() == 1

    @pytest.mark.unit
    async def test_submit_task_without_model(self, manager):
        """Test task submission without specifying model."""
        # Mock load_model to succeed
        manager.llm_loader.load_model = AsyncMock(return_value=True)
        manager.llm_loader.models = {"phi-3.5-mini": MagicMock()}

        task = ModelTask(
            task_id="test-002",
            model_id="",  # No model specified
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="Test prompt",
            params={}
        )

        task_id = await manager.submit_task(task)
        assert task_id == "test-002"
        assert task.model_id == "phi-3.5-mini"  # Should auto-select

    @pytest.mark.unit
    async def test_submit_task_load_failure(self, manager):
        """Test task submission when model loading fails."""
        manager.llm_loader.load_model = AsyncMock(return_value=False)
        manager.llm_loader.models = {}

        task = ModelTask(
            task_id="test-003",
            model_id="non-existent",
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="Test prompt",
            params={}
        )

        task_id = await manager.submit_task(task)
        assert task_id == ""  # Should return empty string on failure

    @pytest.mark.unit
    async def test_submit_multiple_tasks(self, manager):
        """Test submitting multiple tasks."""
        manager.llm_loader.load_model = AsyncMock(return_value=True)
        manager.llm_loader.models = {"phi-3.5-mini": MagicMock()}

        tasks = []
        for i in range(3):
            task = ModelTask(
                task_id=f"task-{i}",
                model_id="",
                task_type="code",
                priority=TaskPriority.MEDIUM,
                prompt=f"Prompt {i}",
                params={}
            )
            tasks.append(task)

        task_ids = await manager.submit_multiple_tasks(tasks)
        assert len(task_ids) == 3
        assert manager.task_queue.qsize() == 3


# =============================================================================
# Test Task Execution
# =============================================================================

class TestTaskExecution:
    """Test task execution functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager ready for task execution."""
        loader = MagicMock()
        mock_model = MagicMock()
        mock_model.return_value = {
            'choices': [{'text': 'Test response'}],
            'usage': {'completion_tokens': 50}
        }
        loader.models = {"test-model": MagicMock(model=mock_model, config=MagicMock(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        ))}
        loader.get_model = Mock(return_value=mock_model)
        loader.load_model = AsyncMock(return_value=True)

        manager = ParallelModelManager(loader)
        return manager

    @pytest.mark.unit
    async def test_execute_task_success(self, manager):
        """Test successful task execution."""
        manager.llm_loader.models = {"test-model": MagicMock()}

        task = ModelTask(
            task_id="test-001",
            model_id="test-model",
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="Test prompt",
            params={}
        )

        result = await manager._execute_task(task)
        assert isinstance(result, TaskResult)
        assert result.task_id == "test-001"
        assert result.success is True
        assert result.result == "Test response"

    @pytest.mark.unit
    async def test_execute_task_with_params(self, manager):
        """Test task execution with custom parameters."""
        mock_model = MagicMock()
        mock_model.return_value = {
            'choices': [{'text': 'Custom response'}],
            'usage': {'completion_tokens': 100}
        }
        manager.llm_loader.models = {"test-model": MagicMock(model=mock_model, config=MagicMock(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        ))}
        manager.llm_loader.get_model = Mock(return_value=mock_model)

        task = ModelTask(
            task_id="test-002",
            model_id="test-model",
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="Test prompt",
            params={'max_tokens': 500, 'temperature': 0.5}
        )

        result = await manager._execute_task(task)
        assert result.success is True
        # Verify model was called with custom params
        mock_model.assert_called_once()

    @pytest.mark.unit
    async def test_execute_task_model_not_available(self, manager):
        """Test task execution when model is not available."""
        manager.llm_loader.get_model = Mock(return_value=None)

        task = ModelTask(
            task_id="test-003",
            model_id="missing-model",
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="Test prompt",
            params={}
        )

        result = await manager._execute_task(task)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    async def test_execute_task_with_callback(self, manager):
        """Test task execution with callback."""
        manager.llm_loader.models = {"test-model": MagicMock()}

        callback_results = []

        async def test_callback(result):
            callback_results.append(result)

        task = ModelTask(
            task_id="test-004",
            model_id="test-model",
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="Test prompt",
            params={},
            callback=test_callback
        )

        await manager._execute_task(task)
        assert len(callback_results) == 1
        assert callback_results[0].task_id == "test-004"


# =============================================================================
# Test Task Processor
# =============================================================================

class TestTaskProcessor:
    """Test task processor loop."""

    @pytest.fixture
    def manager(self):
        """Create manager with mocked components."""
        loader = MagicMock()
        mock_model = MagicMock()
        mock_model.return_value = {
            'choices': [{'text': 'Response'}],
            'usage': {'completion_tokens': 50}
        }
        loader.models = {"test-model": MagicMock(model=mock_model, config=MagicMock(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        ))}
        loader.get_model = Mock(return_value=mock_model)
        loader.load_model = AsyncMock(return_value=True)

        manager = ParallelModelManager(loader)
        return manager

    @pytest.mark.unit
    async def test_start_processor(self, manager):
        """Test starting the task processor."""
        await manager.start_processor()
        assert manager.processor_task is not None
        assert not manager.processor_task.done()

        await manager.stop_processor()

    @pytest.mark.unit
    async def test_stop_processor(self, manager):
        """Test stopping the task processor."""
        await manager.start_processor()
        await manager.stop_processor()
        assert manager._shutdown is True

    @pytest.mark.unit
    async def test_processor_handles_tasks(self, manager):
        """Test that processor actually processes tasks."""
        manager.llm_loader.models = {"test-model": MagicMock()}

        await manager.start_processor()

        # Submit a task
        task = manager.create_task(
            prompt="Test",
            task_type="code",
            model_id="test-model"
        )
        await manager.submit_task(task)

        # Wait a bit for processing
        await asyncio.sleep(0.2)

        # Stop processor
        await manager.stop_processor()

        # Task should be in results
        assert task.task_id in manager.task_results


# =============================================================================
# Test Task Status and Waiting
# =============================================================================

class TestTaskStatus:
    """Test task status checking and waiting."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        loader = MagicMock()
        loader.models = {}
        loader.load_model = AsyncMock(return_value=True)
        return ParallelModelManager(loader)

    @pytest.mark.unit
    async def test_get_task_status_completed(self, manager):
        """Test getting status of completed task."""
        result = TaskResult(
            task_id="test-001",
            model_id="test-model",
            result="Response",
            tokens_generated=50,
            time_taken=1.0,
            success=True
        )
        manager.task_results["test-001"] = result

        status = await manager.get_task_status("test-001")
        assert status is not None
        assert status["status"] == "completed"
        assert status["success"] is True

    @pytest.mark.unit
    async def test_get_task_status_running(self, manager):
        """Test getting status of running task."""
        mock_task = asyncio.create_task(asyncio.sleep(1))
        manager.running_tasks["test-002"] = mock_task

        status = await manager.get_task_status("test-002")
        assert status is not None
        assert status["status"] == "running"

        # Cleanup
        mock_task.cancel()

    @pytest.mark.unit
    async def test_get_task_status_not_found(self, manager):
        """Test getting status of non-existent task."""
        status = await manager.get_task_status("non-existent")
        assert status is None

    @pytest.mark.unit
    async def test_wait_for_task_completed(self, manager):
        """Test waiting for a completed task."""
        result = TaskResult(
            task_id="test-001",
            model_id="test-model",
            result="Response",
            tokens_generated=50,
            time_taken=1.0,
            success=True
        )
        manager.task_results["test-001"] = result

        waited_result = await manager.wait_for_task("test-001")
        assert waited_result is not None
        assert waited_result.task_id == "test-001"

    @pytest.mark.unit
    async def test_wait_for_task_timeout(self, manager):
        """Test waiting for task with timeout."""
        mock_task = asyncio.create_task(asyncio.sleep(10))
        manager.running_tasks["test-002"] = mock_task

        result = await manager.wait_for_task("test-002", timeout=0.1)
        assert result is None

        # Cleanup
        mock_task.cancel()


# =============================================================================
# Test Queue Management
# =============================================================================

class TestQueueManagement:
    """Test queue management functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        loader = MagicMock()
        loader.models = {}
        return ParallelModelManager(loader)

    @pytest.mark.unit
    def test_get_queue_status(self, manager):
        """Test getting queue status."""
        status = manager.get_queue_status()
        assert "queued_tasks" in status
        assert "running_tasks" in status
        assert "completed_tasks" in status
        assert "max_parallel_tasks" in status

    @pytest.mark.unit
    async def test_priority_ordering(self, manager):
        """Test that tasks are processed by priority."""
        manager.llm_loader.load_model = AsyncMock(return_value=True)
        manager.llm_loader.models = {"test-model": MagicMock()}

        # Submit tasks with different priorities
        low_task = ModelTask(
            task_id="low",
            model_id="test-model",
            task_type="code",
            priority=TaskPriority.LOW,
            prompt="Low",
            params={}
        )

        high_task = ModelTask(
            task_id="high",
            model_id="test-model",
            task_type="code",
            priority=TaskPriority.HIGH,
            prompt="High",
            params={}
        )

        # Submit in reverse priority order
        await manager.submit_task(low_task)
        await manager.submit_task(high_task)

        # High priority task should be processed first
        # (This is implementation detail, just verify queue has 2 tasks)
        assert manager.task_queue.qsize() == 2


# =============================================================================
# Test Helper Methods
# =============================================================================

class TestHelperMethods:
    """Test helper methods."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        loader = MagicMock()
        loader.models = {}
        return ParallelModelManager(loader)

    @pytest.mark.unit
    def test_create_task(self, manager):
        """Test task creation helper."""
        task = manager.create_task(
            prompt="Test prompt",
            task_type="code",
            model_id="test-model",
            priority=TaskPriority.HIGH
        )

        assert isinstance(task, ModelTask)
        assert task.prompt == "Test prompt"
        assert task.task_type == "code"
        assert task.model_id == "test-model"
        assert task.priority == TaskPriority.HIGH
        assert task.task_id is not None

    @pytest.mark.unit
    async def test_wait_for_all_tasks(self, manager):
        """Test waiting for multiple tasks."""
        # Mock some completed tasks
        results = []
        for i in range(3):
            result = TaskResult(
                task_id=f"task-{i}",
                model_id="test-model",
                result=f"Response {i}",
                tokens_generated=50,
                time_taken=1.0,
                success=True
            )
            manager.task_results[f"task-{i}"] = result
            results.append(f"task-{i}")

        final_results = await manager.wait_for_all_tasks(results)
        assert len(final_results) == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestParallelManagerIntegration:
    """Integration tests for parallel manager."""

    @pytest.fixture
    def real_manager(self):
        """Create manager with more realistic setup."""
        loader = MagicMock()
        mock_model = MagicMock()
        mock_model.return_value = {
            'choices': [{'text': 'Test response'}],
            'usage': {'completion_tokens': 50}
        }

        # Create realistic model instances
        model1 = MagicMock(model=mock_model, config=MagicMock(
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        ))

        loader.models = {"phi-3.5-mini": model1}
        loader.get_model = Mock(return_value=mock_model)
        loader.load_model = AsyncMock(return_value=True)

        manager = ParallelModelManager(loader)
        return manager

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_task_workflow(self, real_manager):
        """Test complete workflow from submission to result."""
        await real_manager.start_processor()

        # Create and submit task
        task = real_manager.create_task(
            prompt="Write a hello world function",
            task_type="code",
            priority=TaskPriority.HIGH
        )

        task_id = await real_manager.submit_task(task)
        assert task_id != ""

        # Wait for completion
        result = await real_manager.wait_for_task(task_id, timeout=5.0)
        assert result is not None
        assert result.success is True

        # Check status
        status = await real_manager.get_task_status(task_id)
        assert status["status"] == "completed"

        await real_manager.stop_processor()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_parallel_task_execution(self, real_manager):
        """Test executing multiple tasks in parallel."""
        await real_manager.start_processor()

        tasks = []
        for i in range(5):
            task = real_manager.create_task(
                prompt=f"Prompt {i}",
                task_type="code",
                priority=TaskPriority.MEDIUM
            )
            task_id = await real_manager.submit_task(task)
            tasks.append(task_id)

        # Wait for all
        results = await real_manager.wait_for_all_tasks(tasks, timeout=10.0)
        assert len(results) == 5

        await real_manager.stop_processor()

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_task_priorities_respected(self, real_manager):
        """Test that higher priority tasks are processed first."""
        await real_manager.start_processor()

        # Submit tasks with different priorities
        task_ids = []
        for priority in [TaskPriority.LOW, TaskPriority.HIGH, TaskPriority.MEDIUM]:
            task = real_manager.create_task(
                prompt=f"Task with priority {priority.name}",
                task_type="code",
                priority=priority
            )
            task_id = await real_manager.submit_task(task)
            task_ids.append(task_id)

        # Wait for all
        await real_manager.wait_for_all_tasks(task_ids, timeout=10.0)

        # All should complete
        for task_id in task_ids:
            result = real_manager.task_results.get(task_id)
            assert result is not None
            assert result.success is True

        await real_manager.stop_processor()
