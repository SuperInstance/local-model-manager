import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from .llm_loader import LLMLoader, ModelInstance

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ModelTask:
    task_id: str
    model_id: str
    task_type: str  # "generation", "analysis", "code", "creative"
    priority: TaskPriority
    prompt: str
    params: Dict[str, Any]
    callback: Optional[Callable] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class TaskResult:
    task_id: str
    model_id: str
    result: str
    tokens_generated: int
    time_taken: float
    success: bool
    error_message: Optional[str] = None

class ParallelModelManager:
    def __init__(self, llm_loader: LLMLoader):
        self.llm_loader = llm_loader
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.model_specializations = self._load_model_specializations()
        self.max_parallel_tasks = 3
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.task_results: Dict[str, TaskResult] = {}
        self.model_locks: Dict[str, asyncio.Lock] = {}
        self._shutdown = False
        self.processor_task = None

    def _load_model_specializations(self) -> Dict[str, List[str]]:
        """Load model specializations from config"""
        specializations = {
            "phi-3.5-mini": ["code", "reasoning", "technical", "analysis"],
            "llama-3.2-3b": ["general", "conversation", "analysis", "summarization"],
            "gemma-2-2b": ["creative", "writing", "summarization", "brainstorming"]
        }
        return specializations

    def _get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Select the best model for a given task type"""
        best_model = None
        best_score = -1

        for model_id, specializations in self.model_specializations.items():
            if task_type in specializations:
                # Check if model is loaded or can be loaded
                if model_id in self.llm_loader.models or self.llm_loader.get_model(model_id):
                    score = len(specializations) - specializations.index(task_type)
                    if score > best_score:
                        best_score = score
                        best_model = model_id

        return best_model

    async def submit_task(self, task: ModelTask) -> str:
        """Submit a task for processing"""
        # Assign model if not specified
        if not task.model_id:
            task.model_id = self._get_best_model_for_task(task.task_type)
            if not task.model_id:
                logger.error(f"No suitable model found for task type: {task.task_type}")
                return ""

        # Ensure model is loaded
        if task.model_id not in self.llm_loader.models:
            logger.info(f"Loading model {task.model_id} for task {task.task_id}")
            if not await self.llm_loader.load_model(task.model_id):
                logger.error(f"Failed to load model {task.model_id}")
                return ""

        # Create model lock if not exists
        if task.model_id not in self.model_locks:
            self.model_locks[task.model_id] = asyncio.Lock()

        # Add to priority queue (lower number = higher priority)
        priority = -task.priority.value  # Negative for max-heap behavior
        await self.task_queue.put((priority, task.created_at, task))

        logger.info(f"Task {task.task_id} submitted for model {task.model_id}")
        return task.task_id

    async def start_processor(self):
        """Start the task processor"""
        if self.processor_task and not self.processor_task.done():
            return

        self.processor_task = asyncio.create_task(self._process_tasks())
        logger.info("Task processor started")

    async def stop_processor(self):
        """Stop the task processor"""
        self._shutdown = True
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        self.executor.shutdown(wait=True)
        logger.info("Task processor stopped")

    async def _process_tasks(self):
        """Process tasks from the queue"""
        while not self._shutdown:
            try:
                # Get next task (with timeout to allow shutdown)
                priority, created_at, task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )

                # Start processing task
                processing_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks[task.task_id] = processing_task

                # Remove from running tasks when done
                processing_task.add_done_callback(
                    lambda t, task_id=task.task_id: self._cleanup_task(task_id)
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task processor: {e}")

    async def _execute_task(self, task: ModelTask) -> TaskResult:
        """Execute a single task"""
        start_time = time.time()
        logger.info(f"Executing task {task.task_id} with model {task.model_id}")

        try:
            # Get model lock
            async with self.model_locks[task.model_id]:
                model_instance = self.llm_loader.get_model(task.model_id)
                if not model_instance:
                    raise ValueError(f"Model {task.model_id} not available")

                # Execute generation in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._generate_response,
                    model_instance,
                    task
                )

            time_taken = time.time() - start_time
            task_result = TaskResult(
                task_id=task.task_id,
                model_id=task.model_id,
                result=result["text"],
                tokens_generated=result.get("tokens_used", 0),
                time_taken=time_taken,
                success=True
            )

            # Store result
            self.task_results[task.task_id] = task_result

            # Call callback if provided
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(task_result)
                    else:
                        task.callback(task_result)
                except Exception as e:
                    logger.error(f"Callback error for task {task.task_id}: {e}")

            logger.info(f"Task {task.task_id} completed in {time_taken:.2f}s")
            return task_result

        except Exception as e:
            time_taken = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Task {task.task_id} failed: {error_msg}")

            task_result = TaskResult(
                task_id=task.task_id,
                model_id=task.model_id,
                result="",
                tokens_generated=0,
                time_taken=time_taken,
                success=False,
                error_message=error_msg
            )

            self.task_results[task.task_id] = task_result
            return task_result

    def _generate_response(self, model, task: ModelTask) -> Dict[str, Any]:
        """Generate response using the model (runs in thread pool)"""
        try:
            # Prepare generation parameters
            model_config = self.llm_loader.models[task.model_id].config
            params = {
                "max_tokens": task.params.get("max_tokens", model_config.max_tokens),
                "temperature": task.params.get("temperature", model_config.temperature),
                "top_p": task.params.get("top_p", model_config.top_p),
                "repeat_penalty": task.params.get("repeat_penalty", model_config.repeat_penalty),
                "stop": task.params.get("stop", []),
                "echo": False
            }

            # Generate response
            response = model(
                prompt=task.prompt,
                **params
            )

            if "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0]["text"].strip()
                tokens_used = response.get("usage", {}).get("completion_tokens", 0)
            else:
                text = str(response).strip()
                tokens_used = 0

            return {
                "text": text,
                "tokens_used": tokens_used
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def _cleanup_task(self, task_id: str):
        """Clean up completed task"""
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.task_results:
            result = self.task_results[task_id]
            return {
                "task_id": task_id,
                "status": "completed",
                "success": result.success,
                "model_id": result.model_id,
                "time_taken": result.time_taken,
                "tokens_generated": result.tokens_generated,
                "error_message": result.error_message
            }
        elif task_id in self.running_tasks:
            return {
                "task_id": task_id,
                "status": "running"
            }
        else:
            return None

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Wait for a task to complete"""
        if task_id in self.running_tasks:
            try:
                await asyncio.wait_for(self.running_tasks[task_id], timeout=timeout)
            except asyncio.TimeoutError:
                return None

        return self.task_results.get(task_id)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and processing status"""
        return {
            "queued_tasks": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.task_results),
            "loaded_models": list(self.llm_loader.models.keys()),
            "max_parallel_tasks": self.max_parallel_tasks
        }

    async def submit_multiple_tasks(self, tasks: List[ModelTask]) -> List[str]:
        """Submit multiple tasks in parallel"""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            if task_id:
                task_ids.append(task_id)
        return task_ids

    async def wait_for_all_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for multiple tasks to complete"""
        if not task_ids:
            return []

        # Create a list of wait tasks
        wait_tasks = []
        for task_id in task_ids:
            wait_tasks.append(self.wait_for_task(task_id, timeout))

        # Wait for all tasks
        results = await asyncio.gather(*wait_tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, TaskResult):
                valid_results.append(result)
            else:
                logger.error(f"Task wait error: {result}")

        return valid_results

    def create_task(self,
                   prompt: str,
                   task_type: str = "general",
                   model_id: str = "",
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   **params) -> ModelTask:
        """Create a new task"""
        import uuid
        task_id = str(uuid.uuid4())[:8]

        return ModelTask(
            task_id=task_id,
            model_id=model_id,
            task_type=task_type,
            priority=priority,
            prompt=prompt,
            params=params
        )