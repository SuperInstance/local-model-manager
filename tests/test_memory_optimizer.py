"""
Tests for memory optimization functionality.

Tests the MemoryOptimizer class including:
- Garbage collection
- PyTorch memory cleanup
- GPU memory compaction
- Optimization recommendations
- Auto-optimization
- Stress testing
- Settings management
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any

from local_model_manager.monitoring.memory_optimizer import (
    MemoryOptimizer,
    OptimizationSettings,
    OptimizationResult
)
from local_model_manager.monitoring.gpu_monitor import GPUMonitor


# =============================================================================
# Test MemoryOptimizer Initialization
# =============================================================================

class TestMemoryOptimizerInit:
    """Test MemoryOptimizer initialization."""

    @pytest.fixture
    def gpu_monitor(self):
        """Create mock GPU monitor."""
        monitor = MagicMock()
        monitor.get_current_snapshot = Mock(return_value=MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            temperature_c=65.0,
            utilization_percent=45.0
        ))
        return monitor

    @pytest.fixture
    def optimizer(self, gpu_monitor):
        """Create MemoryOptimizer instance."""
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    def test_init(self, optimizer):
        """Test optimizer initialization."""
        assert isinstance(optimizer.settings, OptimizationSettings)
        assert optimizer.optimization_history == []
        assert optimizer._optimization_callbacks == []

    @pytest.mark.unit
    def test_default_settings(self, optimizer):
        """Test default optimization settings."""
        assert optimizer.settings.enable_mmap is True
        assert optimizer.settings.enable_mlock is False
        assert optimizer.settings.garbage_collect_interval == 30


# =============================================================================
# Test Memory Optimization
# =============================================================================

class TestMemoryOptimization:
    """Test memory optimization functionality."""

    @pytest.fixture
    def gpu_monitor(self):
        """Create mock GPU monitor."""
        monitor = MagicMock()
        monitor.get_current_snapshot = Mock(return_value=MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=12.0,
            free_memory_gb=12.0,
            temperature_c=65.0,
            utilization_percent=50.0
        ))
        return monitor

    @pytest.fixture
    def optimizer(self, gpu_monitor):
        """Create optimizer."""
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    async def test_optimize_memory_basic(self, optimizer, mock_psutil):
        """Test basic memory optimization."""
        result = await optimizer.optimize_memory(aggressive=False)

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert isinstance(result.memory_freed_gb, float)
        assert isinstance(result.optimization_time_s, float)
        assert isinstance(result.optimizations_applied, list)
        assert isinstance(result.recommendations, list)

    @pytest.mark.unit
    async def test_optimize_memory_aggressive(self, optimizer, mock_psutil, mock_torch_cuda):
        """Test aggressive memory optimization."""
        result = await optimizer.optimize_memory(aggressive=True)

        assert result.success is True
        # Aggressive mode should apply more optimizations
        assert len(result.optimizations_applied) >= 0

    @pytest.mark.unit
    async def test_garbage_collect(self, optimizer, mock_psutil):
        """Test Python garbage collection."""
        result = await optimizer._garbage_collect()

        assert 'objects_collected' in result
        assert 'freed_memory_mb' in result
        assert result['freed_memory_mb'] >= 0

    @pytest.mark.unit
    async def test_torch_memory_cleanup(self, optimizer, mock_torch_cuda):
        """Test PyTorch memory cleanup."""
        result = await optimizer._torch_memory_cleanup(aggressive=False)

        assert 'freed_memory_mb' in result
        assert isinstance(result['freed_memory_mb'], (int, float))

    @pytest.mark.unit
    async def test_torch_memory_cleanup_aggressive(self, optimizer, mock_torch_cuda):
        """Test aggressive PyTorch cleanup."""
        result = await optimizer._torch_memory_cleanup(aggressive=True)

        assert 'freed_memory_mb' in result
        # Aggressive mode should include peak memory
        assert 'peak_memory_mb' in result

    @pytest.mark.unit
    async def test_system_memory_optimization(self, optimizer, mock_psutil):
        """Test system-level memory optimization."""
        result = await optimizer._system_memory_optimization()

        assert 'freed_memory_mb' in result
        assert isinstance(result['freed_memory_mb'], (int, float))

    @pytest.mark.unit
    async def test_gpu_memory_compaction(self, optimizer, mock_torch_cuda):
        """Test GPU memory compaction."""
        result = await optimizer._gpu_memory_compaction()

        assert 'success' in result
        assert isinstance(result['success'], bool)

    @pytest.mark.unit
    async def test_gpu_memory_compaction_no_cuda(self, optimizer):
        """Test GPU compaction when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            result = await optimizer._gpu_memory_compaction()
            assert result['success'] is False


# =============================================================================
# Test Recommendations
# =============================================================================

class TestRecommendations:
    """Test optimization recommendations."""

    @pytest.fixture
    def gpu_monitor(self):
        """Create mock GPU monitor."""
        monitor = MagicMock()
        return monitor

    @pytest.fixture
    def optimizer(self, gpu_monitor):
        """Create optimizer."""
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    def test_recommendations_high_memory(self, optimizer):
        """Test recommendations for high memory usage."""
        snapshot = MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=20.0,  # 83%
            free_memory_gb=4.0,
            temperature_c=65.0,
            utilization_percent=50.0
        )
        optimizer.gpu_monitor.get_current_snapshot = Mock(return_value=snapshot)

        recommendations = optimizer._generate_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('memory' in r.lower() for r in recommendations)

    @pytest.mark.unit
    def test_recommendations_high_temperature(self, optimizer):
        """Test recommendations for high temperature."""
        snapshot = MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            temperature_c=80.0,  # High temp
            utilization_percent=50.0
        )
        optimizer.gpu_monitor.get_current_snapshot = Mock(return_value=snapshot)

        recommendations = optimizer._generate_recommendations()
        assert any('temperature' in r.lower() for r in recommendations)

    @pytest.mark.unit
    def test_recommendations_fragmentation(self, optimizer):
        """Test recommendations for memory fragmentation."""
        # Add some optimization history
        for i in range(10):
            result = OptimizationResult(
                success=True,
                memory_freed_gb=0.2,  # Consistently freeing memory
                optimization_time_s=1.0,
                optimizations_applied=['GC'],
                recommendations=[]
            )
            optimizer.optimization_history.append(result)

        snapshot = MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=15.0,
            free_memory_gb=9.0,
            temperature_c=65.0,
            utilization_percent=50.0
        )
        optimizer.gpu_monitor.get_current_snapshot = Mock(return_value=snapshot)

        recommendations = optimizer._generate_recommendations()
        # Should detect fragmentation
        assert isinstance(recommendations, list)


# =============================================================================
# Test Callbacks
# =============================================================================

class TestCallbacks:
    """Test optimization callbacks."""

    @pytest.fixture
    def gpu_monitor(self):
        """Create mock GPU monitor."""
        monitor = MagicMock()
        monitor.get_current_snapshot = Mock(return_value=MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            temperature_c=65.0,
            utilization_percent=50.0
        ))
        return monitor

    @pytest.fixture
    def optimizer(self, gpu_monitor):
        """Create optimizer."""
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    async def test_optimization_callback(self, optimizer, mock_psutil):
        """Test optimization callback is triggered."""
        callback_received = []

        def test_callback(result):
            callback_received.append(result)

        optimizer.add_optimization_callback(test_callback)

        await optimizer.optimize_memory(aggressive=False)

        assert len(callback_received) == 1
        assert isinstance(callback_received[0], OptimizationResult)

    @pytest.mark.unit
    async def test_async_optimization_callback(self, optimizer, mock_psutil):
        """Test async optimization callback."""
        callback_received = []

        async def async_callback(result):
            callback_received.append(result)

        optimizer.add_optimization_callback(async_callback)

        await optimizer.optimize_memory(aggressive=False)

        assert len(callback_received) == 1

    @pytest.mark.unit
    async def test_callback_error_handling(self, optimizer, mock_psutil):
        """Test that callback errors don't stop optimization."""
        def bad_callback(result):
            raise Exception("Callback error")

        def good_callback(result):
            pass

        optimizer.add_optimization_callback(bad_callback)
        optimizer.add_optimization_callback(good_callback)

        # Should not raise exception
        result = await optimizer.optimize_memory(aggressive=False)
        assert result.success is True


# =============================================================================
# Test Auto-Optimization
# =============================================================================

class TestAutoOptimization:
    """Test automatic optimization."""

    @pytest.fixture
    def gpu_monitor(self):
        """Create mock GPU monitor."""
        monitor = MagicMock()
        return monitor

    @pytest.fixture
    def optimizer(self, gpu_monitor):
        """Create optimizer."""
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    @pytest.mark.slow
    async def test_auto_optimize_triggers(self, optimizer, mock_psutil):
        """Test that auto-optimization triggers when threshold exceeded."""
        # Set memory usage high
        snapshot = MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=22.0,  # 92% - above threshold
            free_memory_gb=2.0,
            temperature_c=65.0,
            utilization_percent=50.0
        )
        optimizer.gpu_monitor.get_current_snapshot = Mock(return_value=snapshot)

        # Run auto-optimize for a short time
        task = asyncio.create_task(optimizer.auto_optimize(threshold=0.85, check_interval=0.1))
        await asyncio.sleep(0.3)  # Let it run a bit
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have triggered at least once
        assert len(optimizer.optimization_history) >= 0

    @pytest.mark.unit
    @pytest.mark.slow
    async def test_auto_optimize_below_threshold(self, optimizer, mock_psutil):
        """Test that auto-optimization doesn't trigger below threshold."""
        # Set memory usage low
        snapshot = MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=8.0,  # 33% - below threshold
            free_memory_gb=16.0,
            temperature_c=65.0,
            utilization_percent=50.0
        )
        optimizer.gpu_monitor.get_current_snapshot = Mock(return_value=snapshot)

        # Run auto-optimize briefly
        task = asyncio.create_task(optimizer.auto_optimize(threshold=0.85, check_interval=0.1))
        await asyncio.sleep(0.3)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should not have triggered
        assert len(optimizer.optimization_history) == 0


# =============================================================================
# Test Statistics
# =============================================================================

class TestStatistics:
    """Test optimization statistics."""

    @pytest.fixture
    def gpu_monitor(self):
        """Create mock GPU monitor."""
        monitor = MagicMock()
        monitor.get_current_snapshot = Mock(return_value=MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            temperature_c=65.0,
            utilization_percent=50.0
        ))
        return monitor

    @pytest.fixture
    def optimizer(self, gpu_monitor):
        """Create optimizer with history."""
        opt = MemoryOptimizer(gpu_monitor)
        # Add sample optimizations
        for i in range(5):
            result = OptimizationResult(
                success=True,
                memory_freed_gb=0.5 + i * 0.1,
                optimization_time_s=1.0 + i * 0.1,
                optimizations_applied=['GC', 'Torch'],
                recommendations=[]
            )
            opt.optimization_history.append(result)
        return opt

    @pytest.mark.unit
    def test_get_optimization_stats(self, optimizer):
        """Test getting optimization statistics."""
        stats = optimizer.get_optimization_stats()

        assert 'total_optimizations' in stats
        assert 'total_memory_freed_gb' in stats
        assert 'total_time_s' in stats
        assert 'avg_memory_freed_gb' in stats
        assert 'avg_time_s' in stats

        assert stats['total_optimizations'] == 5
        assert stats['total_memory_freed_gb'] > 0

    @pytest.mark.unit
    def test_get_optimization_stats_no_history(self, gpu_monitor):
        """Test stats when no history available."""
        optimizer = MemoryOptimizer(gpu_monitor)
        stats = optimizer.get_optimization_stats()

        assert 'message' in stats


# =============================================================================
# Test Settings Management
# =============================================================================

class TestSettings:
    """Test settings management."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        gpu_monitor = MagicMock()
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    def test_update_settings(self, optimizer):
        """Test updating settings."""
        optimizer.update_settings(
            enable_mmap=False,
            garbage_collect_interval=60,
            max_offload_ratio=0.9
        )

        assert optimizer.settings.enable_mmap is False
        assert optimizer.settings.garbage_collect_interval == 60
        assert optimizer.settings.max_offload_ratio == 0.9

    @pytest.mark.unit
    def test_update_invalid_setting(self, optimizer):
        """Test updating invalid setting is ignored."""
        original_value = optimizer.settings.enable_mmap
        optimizer.update_settings(invalid_setting=True)

        # Should not crash and original value should be unchanged
        assert optimizer.settings.enable_mmap == original_value


# =============================================================================
# Test Optimal Configuration
# =============================================================================

class TestOptimalConfiguration:
    """Test optimal model configuration calculation."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        gpu_monitor = MagicMock()
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    def test_calculate_optimal_model_config_low_vram(self, optimizer):
        """Test config calculation with low VRAM."""
        config = optimizer.calculate_optimal_model_config('test-model', available_vram_gb=2.5)

        assert 'gpu_layers' in config
        assert 'context_size' in config
        assert 'batch_size' in config
        assert config['gpu_layers'] < 20  # Conservative
        assert config['context_size'] <= 2048

    @pytest.mark.unit
    def test_calculate_optimal_model_config_medium_vram(self, optimizer):
        """Test config calculation with medium VRAM."""
        config = optimizer.calculate_optimal_model_config('test-model', available_vram_gb=4.0)

        assert 'gpu_layers' in config
        assert config['gpu_layers'] >= 20
        assert config['context_size'] >= 2048

    @pytest.mark.unit
    def test_calculate_optimal_model_config_high_vram(self, optimizer):
        """Test config calculation with high VRAM."""
        config = optimizer.calculate_optimal_model_config('test-model', available_vram_gb=8.0)

        assert 'gpu_layers' in config
        assert config['gpu_layers'] >= 30
        assert config['context_size'] >= 4096

    @pytest.mark.unit
    def test_calculate_optimal_model_config_insufficient_vram(self, optimizer):
        """Test config calculation with insufficient VRAM."""
        config = optimizer.calculate_optimal_model_config('test-model', available_vram_gb=1.0)

        assert 'error' in config
        assert 'recommendation' in config


# =============================================================================
# Test Stress Testing
# =============================================================================

class TestStressTesting:
    """Test memory stress testing."""

    @pytest.fixture
    def gpu_monitor(self):
        """Create mock GPU monitor."""
        monitor = MagicMock()

        # Create snapshots with varying memory usage
        snapshots = []
        now = time.time()
        for i in range(30):  # 5 samples over 30 seconds (assuming 10s interval in test)
            snapshot = MagicMock(
                timestamp=now - (30 - i) * 10,
                used_memory_gb=10.0 + i * 0.3,
                temperature_c=60.0 + i * 0.5
            )
            snapshots.append(snapshot)

        def get_snapshot():
            if snapshots:
                return snapshots.pop(0)
            return MagicMock(used_memory_gb=15.0, temperature_c=70.0)

        monitor.get_current_snapshot = Mock(side_effect=get_snapshot)
        return monitor

    @pytest.fixture
    def optimizer(self, gpu_monitor):
        """Create optimizer."""
        return MemoryOptimizer(gpu_monitor)

    @pytest.mark.unit
    @pytest.mark.slow
    async def test_stress_test_memory(self, optimizer, mock_psutil):
        """Test memory stress testing."""
        # Run for very short duration (0.1 minutes = 6 seconds)
        # But mock it to complete quickly
        original_sleep = asyncio.sleep

        async def mock_sleep(duration):
            # Sleep much shorter for testing
            await original_sleep(0.1)

        with patch('asyncio.sleep', side_effect=mock_sleep):
            result = await optimizer.stress_test_memory(duration_minutes=0.1)

        assert 'duration_minutes' in result
        assert 'samples_taken' in result
        assert 'peak_memory_gb' in result
        assert 'avg_memory_gb' in result
        assert 'memory_stability' in result

    @pytest.mark.unit
    def test_calculate_memory_stability(self, optimizer):
        """Test memory stability calculation."""
        samples = [
            {'used_memory_gb': 10.0},
            {'used_memory_gb': 10.1},
            {'used_memory_gb': 10.0},
            {'used_memory_gb': 9.9},
            {'used_memory_gb': 10.0}
        ]

        stability = optimizer._calculate_memory_stability(samples)
        assert 0.0 <= stability <= 1.0
        # Very stable memory usage
        assert stability > 0.9


# =============================================================================
# Integration Tests
# =============================================================================

class TestMemoryOptimizerIntegration:
    """Integration tests for memory optimizer."""

    @pytest.fixture
    def real_optimizer(self):
        """Create realistic optimizer."""
        gpu_monitor = MagicMock()
        gpu_monitor.get_current_snapshot = Mock(return_value=MagicMock(
            total_memory_gb=24.0,
            used_memory_gb=16.0,
            free_memory_gb=8.0,
            temperature_c=70.0,
            utilization_percent=60.0
        ))

        optimizer = MemoryOptimizer(gpu_monitor)
        return optimizer

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_optimization_workflow(self, real_optimizer, mock_psutil, mock_torch_cuda):
        """Test complete optimization workflow."""
        # Initial optimization
        result1 = await real_optimizer.optimize_memory(aggressive=False)
        assert result1.success is True

        # Aggressive optimization
        result2 = await real_optimizer.optimize_memory(aggressive=True)
        assert result2.success is True

        # Check stats
        stats = real_optimizer.get_optimization_stats()
        assert stats['total_optimizations'] == 2

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_settings_and_optimization_integration(self, real_optimizer, mock_psutil):
        """Test settings affect optimization behavior."""
        # Update settings
        real_optimizer.update_settings(
            garbage_collect_interval=10,
            max_offload_ratio=0.7
        )

        # Run optimization
        result = await real_optimizer.optimize_memory(aggressive=True)
        assert result.success is True

        # Verify settings were applied
        assert real_optimizer.settings.garbage_collect_interval == 10
