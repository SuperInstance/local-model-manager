"""
Tests for GPU monitoring functionality.

Tests the GPUMonitor class including:
- GPU state capture
- Memory history tracking
- Process monitoring
- Alert triggering
- Statistics calculation
- Trend analysis
- Data export
"""

import pytest
import asyncio
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List

from local_model_manager.monitoring.gpu_monitor import (
    GPUMonitor,
    GPUMemorySnapshot,
    MemoryTrend
)


# =============================================================================
# Test GPUMonitor Initialization
# =============================================================================

class TestGPUMonitorInit:
    """Test GPUMonitor initialization."""

    @pytest.mark.unit
    def test_init_with_defaults(self, mock_gpu):
        """Test initialization with default parameters."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            assert monitor.monitoring_interval == 2.0
            assert monitor.history_size == 1000
            assert len(monitor.memory_history) == 0
            assert monitor._monitoring is False

    @pytest.mark.unit
    def test_init_with_custom_params(self, mock_gpu):
        """Test initialization with custom parameters."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor(
                monitoring_interval=1.0,
                history_size=500
            )
            assert monitor.monitoring_interval == 1.0
            assert monitor.history_size == 500

    @pytest.mark.unit
    def test_initialize_gpu_info(self, mock_gpu):
        """Test GPU info initialization."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            assert 'name' in monitor.gpu_info
            assert 'total_memory_gb' in monitor.gpu_info
            assert monitor.gpu_info['name'] == "NVIDIA GeForce RTX 3090"

    @pytest.mark.unit
    def test_initialize_no_gpu(self):
        """Test initialization when no GPU is available."""
        with patch('GPUtil.getGPUs', return_value=[]):
            monitor = GPUMonitor()
            assert monitor.gpu_info == {}


# =============================================================================
# Test Monitoring Control
# =============================================================================

class TestMonitoringControl:
    """Test monitoring start/stop functionality."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor instance."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor(monitoring_interval=0.1)  # Fast interval for testing
            return monitor

    @pytest.mark.unit
    async def test_start_monitoring(self, monitor):
        """Test starting monitoring."""
        await monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_task is not None
        await monitor.stop_monitoring()

    @pytest.mark.unit
    async def test_stop_monitoring(self, monitor):
        """Test stopping monitoring."""
        await monitor.start_monitoring()
        await monitor.stop_monitoring()
        assert monitor._monitoring is False

    @pytest.mark.unit
    async def test_start_already_monitoring(self, monitor):
        """Test starting when already monitoring."""
        await monitor.start_monitoring()
        task = monitor._monitor_task

        # Try to start again
        await monitor.start_monitoring()

        # Should use same task
        assert monitor._monitor_task == task

        await monitor.stop_monitoring()


# =============================================================================
# Test Snapshot Capture
# =============================================================================

class TestSnapshotCapture:
    """Test GPU snapshot capture functionality."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor instance."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            return monitor

    @pytest.mark.unit
    def test_capture_gpu_snapshot(self, monitor, mock_gpu):
        """Test capturing GPU snapshot."""
        snapshot = monitor._capture_gpu_snapshot()

        assert snapshot is not None
        assert isinstance(snapshot, GPUMemorySnapshot)
        assert snapshot.total_memory_gb > 0
        assert snapshot.used_memory_gb > 0
        assert snapshot.free_memory_gb > 0
        assert snapshot.temperature_c > 0

    @pytest.mark.unit
    def test_capture_gpu_snapshot_no_gpu(self, monitor):
        """Test snapshot when no GPU available."""
        with patch('GPUtil.getGPUs', return_value=[]):
            snapshot = monitor._capture_gpu_snapshot()
            assert snapshot is None

    @pytest.mark.unit
    def test_get_gpu_processes(self, monitor, mock_nvidia_smi):
        """Test getting GPU processes."""
        processes = monitor._get_gpu_processes()
        assert isinstance(processes, list)
        assert len(processes) > 0
        assert 'pid' in processes[0]
        assert 'name' in processes[0]
        assert 'memory_mb' in processes[0]

    @pytest.mark.unit
    def test_get_gpu_processes_error(self, monitor):
        """Test handling errors when getting processes."""
        with patch('subprocess.run', side_effect=Exception("nvidia-smi failed")):
            processes = monitor._get_gpu_processes()
            assert processes == []


# =============================================================================
# Test Memory History
# =============================================================================

class TestMemoryHistory:
    """Test memory history tracking."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor with fast interval."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor(monitoring_interval=0.05)
            return monitor

    @pytest.mark.unit
    async def test_monitoring_loop_records_history(self, monitor):
        """Test that monitoring loop records history."""
        await monitor.start_monitoring()
        await asyncio.sleep(0.2)  # Wait for a few snapshots
        await monitor.stop_monitoring()

        assert len(monitor.memory_history) > 0

    @pytest.mark.unit
    def test_get_current_snapshot(self, monitor):
        """Test getting current snapshot."""
        # Add a snapshot
        snapshot = GPUMemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            temperature_c=65.0,
            utilization_percent=45.0,
            power_usage_watts=250.0,
            processes=[]
        )
        monitor.memory_history.append(snapshot)

        current = monitor.get_current_snapshot()
        assert current is not None
        assert current.used_memory_gb == 8.0

    @pytest.mark.unit
    def test_get_current_snapshot_empty(self, monitor):
        """Test getting snapshot when history is empty."""
        current = monitor.get_current_snapshot()
        assert current is None

    @pytest.mark.unit
    def test_get_memory_history(self, monitor):
        """Test getting memory history for duration."""
        # Add snapshots
        now = time.time()
        for i in range(10):
            snapshot = GPUMemorySnapshot(
                timestamp=now - (i * 60),  # Every minute
                total_memory_gb=24.0,
                used_memory_gb=8.0 + i,
                free_memory_gb=16.0 - i,
                temperature_c=65.0,
                utilization_percent=45.0,
                power_usage_watts=250.0,
                processes=[]
            )
            monitor.memory_history.append(snapshot)

        # Get last 5 minutes
        history = monitor.get_memory_history(duration_minutes=5)
        assert len(history) <= 6  # Should include recent snapshots


# =============================================================================
# Test Statistics
# =============================================================================

class TestStatistics:
    """Test statistics calculation."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor with sample data."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            # Add sample snapshots
            now = time.time()
            for i in range(10):
                snapshot = GPUMemorySnapshot(
                    timestamp=now - (i * 10),
                    total_memory_gb=24.0,
                    used_memory_gb=8.0 + i,
                    free_memory_gb=16.0 - i,
                    temperature_c=60.0 + i,
                    utilization_percent=40.0 + i,
                    power_usage_watts=200.0 + i * 10,
                    processes=[]
                )
                monitor.memory_history.append(snapshot)
            return monitor

    @pytest.mark.unit
    def test_get_memory_stats(self, monitor):
        """Test getting memory statistics."""
        stats = monitor.get_memory_stats()

        assert 'current' in stats
        assert 'recent_stats' in stats
        assert 'gpu_info' in stats
        assert 'active_processes' in stats

        current = stats['current']
        assert 'total_gb' in current
        assert 'used_gb' in current
        assert 'usage_percent' in current
        assert 'temperature_c' in current

    @pytest.mark.unit
    def test_get_memory_stats_no_data(self, monitor):
        """Test getting stats when no data available."""
        monitor.memory_history.clear()
        stats = monitor.get_memory_stats()
        assert 'error' in stats

    @pytest.mark.unit
    def test_calculate_trend_increasing(self, monitor):
        """Test trend calculation for increasing values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = monitor._calculate_trend(values)
        assert trend == "increasing"

    @pytest.mark.unit
    def test_calculate_trend_decreasing(self, monitor):
        """Test trend calculation for decreasing values."""
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = monitor._calculate_trend(values)
        assert trend == "decreasing"

    @pytest.mark.unit
    def test_calculate_trend_stable(self, monitor):
        """Test trend calculation for stable values."""
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        trend = monitor._calculate_trend(values)
        assert trend == "stable"

    @pytest.mark.unit
    def test_calculate_trend_insufficient_data(self, monitor):
        """Test trend calculation with insufficient data."""
        values = [1.0]
        trend = monitor._calculate_trend(values)
        assert trend == "stable"


# =============================================================================
# Test Alerts
# =============================================================================

class TestAlerts:
    """Test alert functionality."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            return monitor

    @pytest.mark.unit
    async def test_high_memory_alert(self, monitor):
        """Test high memory alert."""
        alert_received = []

        async def alert_callback(alert_type, data):
            alert_received.append((alert_type, data))

        monitor.add_alert_callback(alert_callback)

        # Create high memory snapshot
        snapshot = GPUMemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=24.0,
            used_memory_gb=22.0,  # 92% usage
            free_memory_gb=2.0,
            temperature_c=65.0,
            utilization_percent=45.0,
            power_usage_watts=250.0,
            processes=[]
        )

        await monitor._process_snapshot(snapshot)

        assert len(alert_received) > 0
        assert alert_received[0][0] == "high_memory"

    @pytest.mark.unit
    async def test_high_temperature_alert(self, monitor):
        """Test high temperature alert."""
        alert_received = []

        async def alert_callback(alert_type, data):
            alert_received.append((alert_type, data))

        monitor.add_alert_callback(alert_callback)

        # Create high temperature snapshot
        snapshot = GPUMemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            temperature_c=85.0,  # High temperature
            utilization_percent=45.0,
            power_usage_watts=250.0,
            processes=[]
        )

        await monitor._process_snapshot(snapshot)

        assert len(alert_received) > 0
        assert alert_received[0][0] == "high_temperature"

    @pytest.mark.unit
    async def test_alert_callback_error(self, monitor):
        """Test handling of errors in alert callbacks."""
        def bad_callback(alert_type, data):
            raise Exception("Callback error")

        def good_callback(alert_type, data):
            pass

        monitor.add_alert_callback(bad_callback)
        monitor.add_alert_callback(good_callback)

        # Create alert snapshot
        snapshot = GPUMemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=24.0,
            used_memory_gb=22.0,
            free_memory_gb=2.0,
            temperature_c=65.0,
            utilization_percent=45.0,
            power_usage_watts=250.0,
            processes=[]
        )

        # Should not raise exception
        await monitor._process_snapshot(snapshot)


# =============================================================================
# Test Model Memory Tracking
# =============================================================================

class TestModelMemoryTracking:
    """Test per-model memory tracking."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            return monitor

    @pytest.mark.unit
    def test_track_model_memory(self, monitor):
        """Test tracking memory for a model."""
        monitor.track_model_memory('model-1', 4.5, 0.5)
        monitor.track_model_memory('model-1', 5.0, 0.6)

        assert 'model-1' in monitor.model_memory_trends
        trend = monitor.model_memory_trends['model-1']
        assert len(trend.memory_usage_gb) == 2
        assert trend.peak_memory_gb == 5.0

    @pytest.mark.unit
    def test_get_model_memory_trend(self, monitor):
        """Test getting memory trend for a model."""
        monitor.track_model_memory('model-1', 4.5, 0.5)

        trend = monitor.get_model_memory_trend('model-1')
        assert trend is not None
        assert 'model_id' in trend
        assert 'peak_usage_gb' in trend
        assert 'avg_response_time' in trend

    @pytest.mark.unit
    def test_get_model_memory_trend_not_found(self, monitor):
        """Test getting trend for non-existent model."""
        trend = monitor.get_model_memory_trend('non-existent')
        assert trend is None


# =============================================================================
# Test Optimization Recommendations
# =============================================================================

class TestOptimizationRecommendations:
    """Test optimization recommendations."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            return monitor

    @pytest.mark.unit
    def test_optimize_gpu_settings_high_memory(self, monitor):
        """Test recommendations for high memory usage."""
        # Create high memory snapshot
        snapshot = GPUMemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=24.0,
            used_memory_gb=20.0,  # 83% usage
            free_memory_gb=4.0,
            temperature_c=65.0,
            utilization_percent=45.0,
            power_usage_watts=250.0,
            processes=[]
        )
        monitor.memory_history.append(snapshot)

        recommendations = monitor.optimize_gpu_settings()
        assert 'recommendations' in recommendations
        assert len(recommendations['recommendations']) > 0

    @pytest.mark.unit
    def test_optimize_gpu_settings_high_temperature(self, monitor):
        """Test recommendations for high temperature."""
        snapshot = GPUMemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=24.0,
            used_memory_gb=8.0,
            free_memory_gb=16.0,
            temperature_c=82.0,  # High temperature
            utilization_percent=45.0,
            power_usage_watts=250.0,
            processes=[]
        )
        monitor.memory_history.append(snapshot)

        recommendations = monitor.optimize_gpu_settings()
        assert any('temperature' in r['type'] for r in recommendations['recommendations'])

    @pytest.mark.unit
    def test_optimize_gpu_settings_no_data(self, monitor):
        """Test recommendations when no data available."""
        recommendations = monitor.optimize_gpu_settings()
        assert 'error' in recommendations


# =============================================================================
# Test Data Export
# =============================================================================

class TestDataExport:
    """Test data export functionality."""

    @pytest.fixture
    def monitor(self, mock_gpu):
        """Create monitor with sample data."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor()
            # Add sample snapshots
            now = time.time()
            for i in range(5):
                snapshot = GPUMemorySnapshot(
                    timestamp=now - (i * 60),
                    total_memory_gb=24.0,
                    used_memory_gb=8.0 + i,
                    free_memory_gb=16.0 - i,
                    temperature_c=65.0,
                    utilization_percent=45.0,
                    power_usage_watts=250.0,
                    processes=[]
                )
                monitor.memory_history.append(snapshot)
            return monitor

    @pytest.mark.unit
    def test_export_monitoring_data(self, monitor):
        """Test exporting monitoring data to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)

        try:
            monitor.export_monitoring_data(str(export_path), duration_minutes=60)

            assert export_path.exists()

            with open(export_path, 'r') as f:
                data = json.load(f)

            assert 'snapshots' in data
            assert 'gpu_info' in data
            assert len(data['snapshots']) > 0

        finally:
            if export_path.exists():
                export_path.unlink()

    @pytest.mark.unit
    def test_export_with_model_trends(self, monitor):
        """Test export includes model trends."""
        monitor.track_model_memory('model-1', 4.5, 0.5)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)

        try:
            monitor.export_monitoring_data(str(export_path), duration_minutes=60)

            with open(export_path, 'r') as f:
                data = json.load(f)

            assert 'model_trends' in data
            assert 'model-1' in data['model_trends']

        finally:
            if export_path.exists():
                export_path.unlink()


# =============================================================================
# Integration Tests
# =============================================================================

class TestGPUMonitorIntegration:
    """Integration tests for GPU monitor."""

    @pytest.fixture
    def real_monitor(self, mock_gpu):
        """Create realistic monitor."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMonitor(monitoring_interval=0.1)
            return monitor

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_monitoring_workflow(self, real_monitor):
        """Test complete monitoring workflow."""
        # Start monitoring
        await real_monitor.start_monitoring()

        # Let it collect some data
        await asyncio.sleep(0.5)

        # Check history
        assert len(real_monitor.memory_history) > 0

        # Get stats
        stats = real_monitor.get_memory_stats()
        assert 'current' in stats

        # Stop monitoring
        await real_monitor.stop_monitoring()

        assert real_monitor._monitoring is False

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_alert_workflow(self, real_monitor):
        """Test alert triggering workflow."""
        alerts_received = []

        async def alert_handler(alert_type, data):
            alerts_received.append({'type': alert_type, 'data': data})

        real_monitor.add_alert_callback(alert_handler)

        await real_monitor.start_monitoring()
        await asyncio.sleep(0.3)
        await real_monitor.stop_monitoring()

        # Check if any alerts were triggered (depends on GPU state)
        # This test just ensures the workflow doesn't crash
        assert isinstance(alerts_received, list)
