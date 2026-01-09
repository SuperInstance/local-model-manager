import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque

from .llm_loader import LLMLoader, GPUMemoryMonitor

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"
    LOADING = "loading"
    UNLOADING = "unloading"
    ERROR = "error"

@dataclass
class ModelResourceInfo:
    model_id: str
    status: ModelStatus
    vram_usage_gb: float
    last_used: float
    load_count: int
    current_tasks: int
    total_tasks_processed: int
    average_response_time: float
    error_count: int
    specialization_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class ResourceAllocation:
    model_id: str
    allocated_vram_gb: float
    priority: int
    expires_at: float
    task_id: str

class ModelSwitchingStrategy(Enum):
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    PRIORITY = "priority_based"
    SPECIALIZATION = "specialization_based"
    HYBRID = "hybrid"

class ResourceManager:
    def __init__(self, llm_loader: LLMLoader):
        self.llm_loader = llm_loader
        self.memory_monitor = llm_loader.memory_monitor
        self.model_resources: Dict[str, ModelResourceInfo] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.switching_strategy = ModelSwitchingStrategy.HYBRID
        self.auto_switch_threshold = 0.8  # Memory usage threshold
        self.max_idle_time = 300  # 5 minutes
        self.cleanup_interval = 60  # 1 minute
        self.task_history = deque(maxlen=1000)
        self.model_performance_history = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._running = False

    async def start(self):
        """Start the resource manager"""
        if self._running:
            return

        self._running = True
        await self._initialize_model_resources()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Resource manager started")

    async def stop(self):
        """Stop the resource manager"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource manager stopped")

    async def _initialize_model_resources(self):
        """Initialize resource tracking for all configured models"""
        for model_id, config in self.llm_loader.downloader.configs.items():
            self.model_resources[model_id] = ModelResourceInfo(
                model_id=model_id,
                status=ModelStatus.UNLOADED,
                vram_usage_gb=config.estimated_vram_gb,
                last_used=0,
                load_count=0,
                current_tasks=0,
                total_tasks_processed=0,
                average_response_time=0,
                error_count=0,
                specialization_scores=self._calculate_specialization_scores(config.specialization)
            )

    def _calculate_specialization_scores(self, specialization: str) -> Dict[str, float]:
        """Calculate specialization scores for different task types"""
        specializations = specialization.split(", ")
        task_types = ["code", "creative", "analysis", "general", "conversation", "summarization"]
        scores = {}

        for task_type in task_types:
            if any(s in task_type for s in specializations):
                scores[task_type] = 1.0
            elif "general" in specializations:
                scores[task_type] = 0.7
            else:
                scores[task_type] = 0.3

        return scores

    async def allocate_model(self, model_id: str, task_id: str, priority: int = 5) -> bool:
        """Allocate a model for a specific task"""
        async with self._lock:
            # Check if model is loaded
            if model_id not in self.llm_loader.models:
                # Check if we need to switch models
                if not await self._ensure_model_loaded(model_id):
                    return False

            # Create resource allocation
            allocation = ResourceAllocation(
                model_id=model_id,
                allocated_vram_gb=self.model_resources[model_id].vram_usage_gb,
                priority=priority,
                expires_at=time.time() + 3600,  # 1 hour default
                task_id=task_id
            )

            self.resource_allocations[task_id] = allocation
            self.model_resources[model_id].current_tasks += 1
            self.model_resources[model_id].last_used = time.time()

            logger.debug(f"Allocated model {model_id} for task {task_id}")
            return True

    async def deallocate_model(self, task_id: str):
        """Deallocate a model after task completion"""
        async with self._lock:
            if task_id in self.resource_allocations:
                allocation = self.resource_allocations[task_id]
                model_id = allocation.model_id

                if model_id in self.model_resources:
                    self.model_resources[model_id].current_tasks = max(
                        0, self.model_resources[model_id].current_tasks - 1
                    )
                    self.model_resources[model_id].total_tasks_processed += 1

                del self.resource_allocations[task_id]
                logger.debug(f"Deallocated model {model_id} for task {task_id}")

    async def _ensure_model_loaded(self, model_id: str) -> bool:
        """Ensure a model is loaded, switching if necessary"""
        # Check current memory usage
        memory_info = self.llm_loader.get_memory_info()
        memory_usage_ratio = memory_info["used_vram_gb"] / memory_info["total_vram_gb"]

        # If memory usage is high, try to make space
        if memory_usage_ratio > self.auto_switch_threshold:
            await self._make_space_for_model(model_id)

        # Load the model if not already loaded
        if model_id not in self.llm_loader.models:
            self.model_resources[model_id].status = ModelStatus.LOADING
            success = await self.llm_loader.load_model(model_id)

            if success:
                self.model_resources[model_id].status = ModelStatus.LOADED
                self.model_resources[model_id].load_count += 1
            else:
                self.model_resources[model_id].status = ModelStatus.ERROR
                self.model_resources[model_id].error_count += 1

            return success

        return True

    async def _make_space_for_model(self, target_model_id: str):
        """Make space for a model by switching out other models"""
        target_memory = self.model_resources[target_model_id].vram_usage_gb
        available_memory = self.memory_monitor.get_available_vram()

        if available_memory >= target_memory:
            return

        # Get candidates for unloading
        candidates = await self._get_unloading_candidates()
        freed_memory = 0

        for candidate in candidates:
            if available_memory + freed_memory >= target_memory:
                break

            if candidate in self.llm_loader.models:
                await self.llm_loader.unload_model(candidate)
                self.model_resources[candidate].status = ModelStatus.UNLOADED
                freed_memory += self.model_resources[candidate].vram_usage_gb
                logger.info(f"Unloaded model {candidate} to free space for {target_model_id}")

    async def _get_unloading_candidates(self) -> List[str]:
        """Get models that can be unloaded based on current strategy"""
        loaded_models = list(self.llm_loader.models.keys())
        if not loaded_models:
            return []

        if self.switching_strategy == ModelSwitchingStrategy.LRU:
            # Least Recently Used
            return sorted(loaded_models,
                         key=lambda x: self.model_resources[x].last_used)

        elif self.switching_strategy == ModelSwitchingStrategy.LFU:
            # Least Frequently Used
            return sorted(loaded_models,
                         key=lambda x: self.model_resources[x].total_tasks_processed)

        elif self.switching_strategy == ModelSwitchingStrategy.PRIORITY:
            # Based on current task priority
            model_priorities = {}
            for model_id in loaded_models:
                min_priority = min(
                    allocation.priority for allocation in self.resource_allocations.values()
                    if allocation.model_id == model_id
                ) if any(allocation.model_id == model_id for allocation in self.resource_allocations.values()) else 10
                model_priorities[model_id] = min_priority

            return sorted(loaded_models, key=lambda x: model_priorities[x], reverse=True)

        elif self.switching_strategy == ModelSwitchingStrategy.SPECIALIZATION:
            # Keep models that are best at current tasks
            current_task_types = set()
            for allocation in self.resource_allocations.values():
                # This would need task type information from the allocation
                current_task_types.add("general")  # Placeholder

            candidates = []
            for model_id in loaded_models:
                has_specialization = any(
                    score > 0.7 for score in self.model_resources[model_id].specialization_scores.values()
                    if any(task in score for task in current_task_types)
                )
                if not has_specialization:
                    candidates.append(model_id)

            return candidates if candidates else loaded_models

        else:  # HYBRID
            # Combine multiple factors
            scores = {}
            current_time = time.time()

            for model_id in loaded_models:
                resource = self.model_resources[model_id]

                # Time since last used (higher is better to keep)
                time_factor = (current_time - resource.last_used) / self.max_idle_time

                # Load count (higher is better to keep)
                load_factor = resource.load_count / max(1, max(r.load_count for r in self.model_resources.values()))

                # Current tasks (higher is better to keep)
                task_factor = resource.current_tasks / max(1, max(r.current_tasks for r in self.model_resources.values()))

                # Error count (lower is better)
                error_factor = 1 - (resource.error_count / max(1, max(r.error_count for r in self.model_resources.values())))

                # Combined score (lower is better for unloading)
                scores[model_id] = (time_factor * 0.3 + load_factor * 0.2 +
                                  task_factor * 0.4 + error_factor * 0.1)

            return sorted(loaded_models, key=lambda x: scores[x], reverse=True)

    async def update_model_performance(self, model_id: str, response_time: float, success: bool):
        """Update model performance metrics"""
        if model_id not in self.model_resources:
            return

        resource = self.model_resources[model_id]

        # Update average response time
        if resource.total_tasks_processed > 0:
            resource.average_response_time = (
                (resource.average_response_time * (resource.total_tasks_processed - 1) + response_time) /
                resource.total_tasks_processed
            )

        # Update error count
        if not success:
            resource.error_count += 1

        # Store in history
        self.model_performance_history[model_id].append({
            "timestamp": time.time(),
            "response_time": response_time,
            "success": success
        })

        # Keep only recent history
        if len(self.model_performance_history[model_id]) > 100:
            self.model_performance_history[model_id] = self.model_performance_history[model_id][-50:]

    async def _cleanup_loop(self):
        """Periodic cleanup of idle models and expired allocations"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_idle_resources(self):
        """Clean up idle models and expired allocations"""
        current_time = time.time()
        async with self._lock:
            # Clean up expired allocations
            expired_tasks = [
                task_id for task_id, allocation in self.resource_allocations.items()
                if allocation.expires_at < current_time
            ]

            for task_id in expired_tasks:
                await self.deallocate_model(task_id)

            # Unload idle models
            memory_usage_ratio = self.memory_monitor.get_vram_usage() / self.memory_monitor.total_vram_gb

            for model_id, resource in self.model_resources.items():
                if (resource.status == ModelStatus.LOADED and
                    resource.current_tasks == 0 and
                    (current_time - resource.last_used) > self.max_idle_time and
                    memory_usage_ratio > 0.5):  # Only unload if memory is somewhat used

                    await self.llm_loader.unload_model(model_id)
                    resource.status = ModelStatus.UNLOADED
                    logger.info(f"Unloaded idle model {model_id}")

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        return {
            "loaded_models": list(self.llm_loader.models.keys()),
            "model_resources": {
                model_id: {
                    "status": resource.status.value,
                    "vram_usage_gb": resource.vram_usage_gb,
                    "current_tasks": resource.current_tasks,
                    "total_tasks_processed": resource.total_tasks_processed,
                    "average_response_time": resource.average_response_time,
                    "error_count": resource.error_count,
                    "last_used": resource.last_used
                }
                for model_id, resource in self.model_resources.items()
            },
            "active_allocations": len(self.resource_allocations),
            "memory_info": self.llm_loader.get_memory_info(),
            "switching_strategy": self.switching_strategy.value
        }

    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the best model for a specific task type"""
        best_model = None
        best_score = -1

        for model_id, resource in self.model_resources.items():
            if resource.status != ModelStatus.LOADED:
                continue

            # Get specialization score
            specialization_score = resource.specialization_scores.get(task_type, 0.3)

            # Get performance score (inverse of response time)
            performance_score = 1.0 / max(0.1, resource.average_response_time)

            # Get availability score (inverse of current tasks)
            availability_score = 1.0 / max(1, resource.current_tasks)

            # Combined score
            combined_score = (specialization_score * 0.4 +
                            performance_score * 0.3 +
                            availability_score * 0.3)

            if combined_score > best_score:
                best_score = combined_score
                best_model = model_id

        return best_model

    def set_switching_strategy(self, strategy: ModelSwitchingStrategy):
        """Set the model switching strategy"""
        self.switching_strategy = strategy
        logger.info(f"Switching strategy changed to {strategy.value}")