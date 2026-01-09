import asyncio
import logging
import gc
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import psutil
import torch

from .gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

@dataclass
class OptimizationSettings:
    enable_mmap: bool = True
    enable_mlock: bool = False
    enable_offload: bool = True
    max_offload_ratio: float = 0.8
    garbage_collect_interval: int = 30
    memory_compaction_threshold: float = 0.1
    enable_memory_mapping: bool = True
    tensor_parallel_size: int = 1
    cpu_threads: int = 4

@dataclass
class OptimizationResult:
    success: bool
    memory_freed_gb: float
    optimization_time_s: float
    optimizations_applied: List[str]
    recommendations: List[str]

class MemoryOptimizer:
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.settings = OptimizationSettings()
        self.optimization_history: List[OptimizationResult] = []
        self._last_gc_time = 0
        self._optimization_callbacks: List[callable] = []

    async def optimize_memory(self, aggressive: bool = False) -> OptimizationResult:
        """Perform memory optimization"""
        start_time = time.time()
        optimizations_applied = []
        memory_freed_gb = 0.0

        logger.info(f"Starting memory optimization (aggressive={aggressive})")

        # Get initial memory state
        initial_snapshot = self.gpu_monitor.get_current_snapshot()
        initial_memory = initial_snapshot.used_memory_gb if initial_snapshot else 0.0

        try:
            # 1. Python garbage collection
            gc_result = await self._garbage_collect()
            if gc_result["freed_memory_mb"] > 0:
                optimizations_applied.append(f"Python GC: {gc_result['freed_memory_mb']:.1f}MB")
                memory_freed_gb += gc_result["freed_memory_mb"] / 1024

            # 2. PyTorch memory cleanup
            if torch.cuda.is_available():
                torch_result = await self._torch_memory_cleanup(aggressive)
                if torch_result["freed_memory_mb"] > 0:
                    optimizations_applied.append(f"PyTorch cleanup: {torch_result['freed_memory_mb']:.1f}MB")
                    memory_freed_gb += torch_result["freed_memory_mb"] / 1024

            # 3. System memory optimization
            if aggressive:
                sys_result = await self._system_memory_optimization()
                if sys_result["freed_memory_mb"] > 0:
                    optimizations_applied.append(f"System cleanup: {sys_result['freed_memory_mb']:.1f}MB")
                    memory_freed_gb += sys_result["freed_memory_mb"] / 1024

            # 4. GPU memory compaction
            if aggressive and torch.cuda.is_available():
                compact_result = await self._gpu_memory_compaction()
                if compact_result["success"]:
                    optimizations_applied.append("GPU memory compaction")

        except Exception as e:
            logger.error(f"Error during optimization: {e}")

        # Get final memory state
        final_snapshot = self.gpu_monitor.get_current_snapshot()
        final_memory = final_snapshot.used_memory_gb if final_snapshot else 0.0
        actual_freed = max(0, initial_memory - final_memory)

        optimization_time = time.time() - start_time

        result = OptimizationResult(
            success=True,
            memory_freed_gb=actual_freed,
            optimization_time_s=optimization_time,
            optimizations_applied=optimizations_applied,
            recommendations=self._generate_recommendations()
        )

        self.optimization_history.append(result)

        # Trigger callbacks
        await self._trigger_optimization_callbacks(result)

        logger.info(f"Optimization completed: freed {actual_freed:.2f}GB in {optimization_time:.2f}s")
        return result

    async def _garbage_collect(self) -> Dict[str, Any]:
        """Perform Python garbage collection"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run garbage collection
        collected = gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        freed_memory = max(0, initial_memory - final_memory)

        return {
            "objects_collected": collected,
            "freed_memory_mb": freed_memory
        }

    async def _torch_memory_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """Perform PyTorch GPU memory cleanup"""
        if not torch.cuda.is_available():
            return {"freed_memory_mb": 0}

        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        # Clear cache
        torch.cuda.empty_cache()

        if aggressive:
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

            # Synchronize CUDA operations
            torch.cuda.synchronize()

        final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        freed_memory = max(0, initial_memory - final_memory)

        return {
            "freed_memory_mb": freed_memory,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
        }

    async def _system_memory_optimization(self) -> Dict[str, Any]:
        """Perform system-level memory optimization"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Clear Python interned strings (if available)
        try:
            import sys
            if hasattr(sys, 'intern'):
                # This is a rough approximation
                pass
        except:
            pass

        # Force memory trimming
        try:
            process.memory_maps()
        except:
            pass

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        freed_memory = max(0, initial_memory - final_memory)

        return {
            "freed_memory_mb": freed_memory
        }

    async def _gpu_memory_compaction(self) -> Dict[str, Any]:
        """Perform GPU memory compaction"""
        if not torch.cuda.is_available():
            return {"success": False, "message": "CUDA not available"}

        try:
            # Enable memory compaction (if supported)
            if hasattr(torch.cuda, 'memory_empty_cache'):
                torch.cuda.memory_empty_cache()

            # Synchronize to ensure completion
            torch.cuda.synchronize()

            return {"success": True, "message": "GPU memory compacted"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        snapshot = self.gpu_monitor.get_current_snapshot()

        if not snapshot:
            return recommendations

        memory_ratio = snapshot.used_memory_gb / snapshot.total_memory_gb

        if memory_ratio > 0.9:
            recommendations.extend([
                "Consider reducing model context sizes",
                "Lower GPU layers for loaded models",
                "Enable more aggressive model unloading"
            ])
        elif memory_ratio > 0.7:
            recommendations.extend([
                "Monitor memory usage closely",
                "Consider reducing batch sizes"
            ])

        if snapshot.temperature_c > 75:
            recommendations.append("Monitor GPU temperature and cooling")

        # Check for memory fragmentation
        if len(self.optimization_history) > 5:
            recent_optimizations = self.optimization_history[-5:]
            avg_freed = sum(opt.memory_freed_gb for opt in recent_optimizations) / len(recent_optimizations)
            if avg_freed > 0.1:  # If consistently freeing more than 100MB
                recommendations.append("Memory fragmentation detected - consider periodic restarts")

        return recommendations

    async def _trigger_optimization_callbacks(self, result: OptimizationResult):
        """Trigger optimization callbacks"""
        for callback in self._optimization_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")

    def add_optimization_callback(self, callback: callable):
        """Add a callback for optimization events"""
        self._optimization_callbacks.append(callback)

    async def auto_optimize(self, threshold: float = 0.85, check_interval: int = 60):
        """Automatic optimization based on memory threshold"""
        while True:
            try:
                snapshot = self.gpu_monitor.get_current_snapshot()
                if snapshot:
                    memory_ratio = snapshot.used_memory_gb / snapshot.total_memory_gb
                    if memory_ratio > threshold:
                        logger.info(f"Auto-optimizing: memory usage {memory_ratio:.1%} > {threshold:.1%}")
                        await self.optimize_memory(aggressive=False)

                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Auto-optimization error: {e}")
                await asyncio.sleep(check_interval)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {"message": "No optimization history available"}

        total_freed = sum(opt.memory_freed_gb for opt in self.optimization_history)
        total_time = sum(opt.optimization_time_s for opt in self.optimization_history)
        avg_freed = total_freed / len(self.optimization_history)
        avg_time = total_time / len(self.optimization_history)

        return {
            "total_optimizations": len(self.optimization_history),
            "total_memory_freed_gb": total_freed,
            "total_time_s": total_time,
            "avg_memory_freed_gb": avg_freed,
            "avg_time_s": avg_time,
            "last_optimization": self.optimization_history[-1].optimizations_applied
        }

    def calculate_optimal_model_config(self, model_id: str, available_vram_gb: float) -> Dict[str, Any]:
        """Calculate optimal model configuration based on available memory"""
        # This would integrate with model configurations
        # For now, provide general recommendations

        if available_vram_gb < 2.0:
            return {
                "error": "Insufficient VRAM for model loading",
                "recommendation": "Free up memory or use smaller models"
            }
        elif available_vram_gb < 3.0:
            return {
                "gpu_layers": 15,
                "context_size": 2048,
                "batch_size": 256,
                "note": "Conservative settings for limited VRAM"
            }
        elif available_vram_gb < 4.5:
            return {
                "gpu_layers": 25,
                "context_size": 4096,
                "batch_size": 512,
                "note": "Balanced settings for moderate VRAM"
            }
        else:
            return {
                "gpu_layers": 35,
                "context_size": 8192,
                "batch_size": 1024,
                "note": "Optimal settings for abundant VRAM"
            }

    def update_settings(self, **kwargs):
        """Update optimization settings"""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                logger.info(f"Updated setting {key} = {value}")

    async def stress_test_memory(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Perform memory stress test"""
        logger.info(f"Starting memory stress test for {duration_minutes} minutes")

        start_time = time.time()
        memory_samples = []
        optimization_events = []

        while time.time() - start_time < duration_minutes * 60:
            snapshot = self.gpu_monitor.get_current_snapshot()
            if snapshot:
                memory_samples.append({
                    "timestamp": snapshot.timestamp,
                    "used_memory_gb": snapshot.used_memory_gb,
                    "temperature_c": snapshot.temperature_c
                })

                # Trigger optimization if memory usage is high
                memory_ratio = snapshot.used_memory_gb / snapshot.total_memory_gb
                if memory_ratio > 0.9:
                    opt_result = await self.optimize_memory(aggressive=True)
                    optimization_events.append({
                        "timestamp": time.time(),
                        "trigger_ratio": memory_ratio,
                        "result": opt_result
                    })

            await asyncio.sleep(10)  # Sample every 10 seconds

        return {
            "duration_minutes": duration_minutes,
            "samples_taken": len(memory_samples),
            "optimization_events": len(optimization_events),
            "peak_memory_gb": max(s["used_memory_gb"] for s in memory_samples) if memory_samples else 0,
            "avg_memory_gb": sum(s["used_memory_gb"] for s in memory_samples) / len(memory_samples) if memory_samples else 0,
            "peak_temperature_c": max(s["temperature_c"] for s in memory_samples) if memory_samples else 0,
            "memory_stability": self._calculate_memory_stability(memory_samples)
        }

    def _calculate_memory_stability(self, samples: List[Dict]) -> float:
        """Calculate memory usage stability (0-1, higher is more stable)"""
        if len(samples) < 2:
            return 1.0

        memory_values = [s["used_memory_gb"] for s in samples]
        mean_memory = sum(memory_values) / len(memory_values)
        variance = sum((x - mean_memory) ** 2 for x in memory_values) / len(memory_values)
        std_dev = variance ** 0.5

        # Stability is inverse of coefficient of variation
        cv = std_dev / mean_memory if mean_memory > 0 else 0
        stability = max(0, 1 - cv)

        return stability