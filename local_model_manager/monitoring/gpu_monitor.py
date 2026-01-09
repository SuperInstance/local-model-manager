import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import psutil
import GPUtil
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GPUMemorySnapshot:
    timestamp: float
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    temperature_c: float
    utilization_percent: float
    power_usage_watts: float
    processes: List[Dict[str, Any]]

@dataclass
class MemoryTrend:
    model_id: str
    timestamps: List[float] = field(default_factory=list)
    memory_usage_gb: List[float] = field(default_factory=list)
    avg_response_time: float = 0.0
    peak_memory_gb: float = 0.0
    memory_efficiency: float = 0.0

class GPUMonitor:
    def __init__(self, monitoring_interval: float = 2.0, history_size: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.memory_history: deque = deque(maxlen=history_size)
        self.model_memory_trends: Dict[str, MemoryTrend] = {}
        self.alert_callbacks: List[Callable] = []
        self._monitoring = False
        self._monitor_task = None
        self._lock = threading.Lock()
        self.gpu_info = self._initialize_gpu_info()

    def _initialize_gpu_info(self) -> Dict[str, Any]:
        """Initialize GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "name": gpu.name,
                    "total_memory_gb": gpu.memoryTotal / 1024,
                    "driver_version": getattr(gpu, 'driver', 'Unknown'),
                    "temperature": gpu.temperature
                }
            else:
                logger.warning("No GPU detected")
                return {}
        except Exception as e:
            logger.error(f"Failed to initialize GPU info: {e}")
            return {}

    async def start_monitoring(self):
        """Start GPU monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("GPU monitoring started")

    async def stop_monitoring(self):
        """Stop GPU monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("GPU monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                snapshot = self._capture_gpu_snapshot()
                if snapshot:
                    with self._lock:
                        self.memory_history.append(snapshot)
                    await self._process_snapshot(snapshot)
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def _capture_gpu_snapshot(self) -> Optional[GPUMemorySnapshot]:
        """Capture current GPU state"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu = gpus[0]

            # Get process information using nvidia-smi
            processes = self._get_gpu_processes()

            snapshot = GPUMemorySnapshot(
                timestamp=time.time(),
                total_memory_gb=gpu.memoryTotal / 1024,
                used_memory_gb=gpu.memoryUsed / 1024,
                free_memory_gb=gpu.memoryFree / 1024,
                temperature_c=gpu.temperature,
                utilization_percent=gpu.load * 100,
                power_usage_watts=getattr(gpu, 'powerLoad', 0),
                processes=processes
            )

            return snapshot

        except Exception as e:
            logger.error(f"Failed to capture GPU snapshot: {e}")
            return None

    def _get_gpu_processes(self) -> List[Dict[str, Any]]:
        """Get GPU process information using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return []

            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        processes.append({
                            "pid": int(parts[0].strip()),
                            "name": parts[1].strip(),
                            "memory_mb": int(parts[2].strip())
                        })

            return processes

        except Exception as e:
            logger.debug(f"Could not get GPU processes: {e}")
            return []

    async def _process_snapshot(self, snapshot: GPUMemorySnapshot):
        """Process and analyze the snapshot"""
        # Check for memory alerts
        memory_usage_ratio = snapshot.used_memory_gb / snapshot.total_memory_gb
        if memory_usage_ratio > 0.9:
            await self._trigger_alert("high_memory", {
                "usage_ratio": memory_usage_ratio,
                "used_gb": snapshot.used_memory_gb,
                "total_gb": snapshot.total_memory_gb
            })

        # Check for temperature alerts
        if snapshot.temperature_c > 80:
            await self._trigger_alert("high_temperature", {
                "temperature": snapshot.temperature_c
            })

    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger alerts to registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, data)
                else:
                    callback(alert_type, data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add a callback for alerts"""
        self.alert_callbacks.append(callback)

    def get_current_snapshot(self) -> Optional[GPUMemorySnapshot]:
        """Get the most recent snapshot"""
        with self._lock:
            return self.memory_history[-1] if self.memory_history else None

    def get_memory_history(self, duration_minutes: int = 10) -> List[GPUMemorySnapshot]:
        """Get memory history for the specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)

        with self._lock:
            return [snapshot for snapshot in self.memory_history
                   if snapshot.timestamp >= cutoff_time]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        current = self.get_current_snapshot()
        if not current:
            return {"error": "No GPU data available"}

        # Calculate statistics from recent history
        recent_history = self.get_memory_history(5)  # Last 5 minutes
        memory_values = [s.used_memory_gb for s in recent_history]

        stats = {
            "current": {
                "total_gb": current.total_memory_gb,
                "used_gb": current.used_memory_gb,
                "free_gb": current.free_memory_gb,
                "usage_percent": (current.used_memory_gb / current.total_memory_gb) * 100,
                "temperature_c": current.temperature_c,
                "utilization_percent": current.utilization_percent,
                "power_watts": current.power_usage_watts
            },
            "recent_stats": {
                "avg_usage_gb": sum(memory_values) / len(memory_values) if memory_values else 0,
                "peak_usage_gb": max(memory_values) if memory_values else 0,
                "min_usage_gb": min(memory_values) if memory_values else 0,
                "trend_direction": self._calculate_trend(memory_values)
            },
            "gpu_info": self.gpu_info,
            "active_processes": len(current.processes)
        }

        return stats

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate memory usage trend"""
        if len(values) < 2:
            return "stable"

        # Simple linear regression to determine trend
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))

        if n * sum_x2 - sum_x * sum_x == 0:
            return "stable"

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def track_model_memory(self, model_id: str, memory_usage_gb: float, response_time: float):
        """Track memory usage for a specific model"""
        if model_id not in self.model_memory_trends:
            self.model_memory_trends[model_id] = MemoryTrend(model_id=model_id)

        trend = self.model_memory_trends[model_id]
        current_time = time.time()

        trend.timestamps.append(current_time)
        trend.memory_usage_gb.append(memory_usage_gb)

        # Update peak memory
        trend.peak_memory_gb = max(trend.peak_memory_gb, memory_usage_gb)

        # Update average response time (exponential moving average)
        alpha = 0.1
        if trend.avg_response_time == 0:
            trend.avg_response_time = response_time
        else:
            trend.avg_response_time = (alpha * response_time +
                                      (1 - alpha) * trend.avg_response_time)

        # Calculate memory efficiency (tokens per GB per second)
        if memory_usage_gb > 0 and response_time > 0:
            # This would need token count for accurate calculation
            trend.memory_efficiency = 1.0 / (memory_usage_gb * response_time)

        # Keep only recent data
        if len(trend.timestamps) > 100:
            trend.timestamps = trend.timestamps[-50:]
            trend.memory_usage_gb = trend.memory_usage_gb[-50:]

    def get_model_memory_trend(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get memory trend for a specific model"""
        if model_id not in self.model_memory_trends:
            return None

        trend = self.model_memory_trends[model_id]
        return {
            "model_id": trend.model_id,
            "current_usage_gb": trend.memory_usage_gb[-1] if trend.memory_usage_gb else 0,
            "peak_usage_gb": trend.peak_memory_gb,
            "avg_response_time": trend.avg_response_time,
            "memory_efficiency": trend.memory_efficiency,
            "data_points": len(trend.timestamps),
            "recent_trend": self._calculate_trend(trend.memory_usage_gb[-10:])
        }

    def optimize_gpu_settings(self) -> Dict[str, Any]:
        """Provide optimization recommendations"""
        current = self.get_current_snapshot()
        if not current:
            return {"error": "No GPU data available"}

        recommendations = []

        # Memory usage recommendations
        memory_ratio = current.used_memory_gb / current.total_memory_gb
        if memory_ratio > 0.8:
            recommendations.append({
                "type": "memory",
                "severity": "high",
                "message": "High memory usage detected. Consider unloading unused models or reducing batch sizes."
            })
        elif memory_ratio > 0.6:
            recommendations.append({
                "type": "memory",
                "severity": "medium",
                "message": "Moderate memory usage. Monitor closely when loading additional models."
            })

        # Temperature recommendations
        if current.temperature_c > 80:
            recommendations.append({
                "type": "temperature",
                "severity": "high",
                "message": f"GPU temperature high ({current.temperature_c:.1f}°C). Check cooling."
            })
        elif current.temperature_c > 70:
            recommendations.append({
                "type": "temperature",
                "severity": "medium",
                "message": f"GPU temperature elevated ({current.temperature_c:.1f}°C). Monitor cooling."
            })

        # Utilization recommendations
        if current.utilization_percent < 20 and memory_ratio > 0.5:
            recommendations.append({
                "type": "utilization",
                "severity": "low",
                "message": "Low GPU utilization with high memory usage. Consider optimizing batch sizes."
            })

        # Process recommendations
        if len(current.processes) > 5:
            recommendations.append({
                "type": "processes",
                "severity": "medium",
                "message": f"Many GPU processes ({len(current.processes)}). Check for zombie processes."
            })

        return {
            "recommendations": recommendations,
            "current_stats": self.get_memory_stats()
        }

    def export_monitoring_data(self, filepath: str, duration_minutes: int = 60):
        """Export monitoring data to file"""
        history = self.get_memory_history(duration_minutes)

        data = {
            "export_timestamp": time.time(),
            "duration_minutes": duration_minutes,
            "gpu_info": self.gpu_info,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "used_memory_gb": s.used_memory_gb,
                    "free_memory_gb": s.free_memory_gb,
                    "temperature_c": s.temperature_c,
                    "utilization_percent": s.utilization_percent,
                    "power_usage_watts": s.power_usage_watts
                }
                for s in history
            ],
            "model_trends": {
                model_id: {
                    "peak_usage_gb": trend.peak_memory_gb,
                    "avg_response_time": trend.avg_response_time,
                    "memory_efficiency": trend.memory_efficiency,
                    "data_points": len(trend.timestamps)
                }
                for model_id, trend in self.model_memory_trends.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Monitoring data exported to {filepath}")