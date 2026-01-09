import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..core.llm_loader import LLMLoader
from ..core.parallel_manager import ParallelModelManager, ModelTask, TaskPriority
from ..core.resource_manager import ResourceManager, ModelSwitchingStrategy
from ..monitoring.gpu_monitor import GPUMonitor
from ..monitoring.memory_optimizer import MemoryOptimizer

logger = logging.getLogger(__name__)

# Pydantic models for API
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate text from")
    model_id: Optional[str] = Field(None, description="Specific model to use (auto-selected if not provided)")
    task_type: str = Field("general", description="Type of task: code, creative, analysis, general")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Generation temperature")
    top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
    priority: str = Field("medium", description="Task priority: low, medium, high, critical")
    stream: bool = Field(False, description="Enable streaming response")

class GenerationResponse(BaseModel):
    task_id: str
    model_id: str
    text: str
    tokens_generated: int
    time_taken: float
    success: bool
    error_message: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str  # queued, running, completed, failed
    model_id: Optional[str] = None
    time_elapsed: float
    estimated_completion: Optional[float] = None
    result: Optional[GenerationResponse] = None

class ModelInfo(BaseModel):
    model_id: str
    name: str
    status: str
    vram_usage_gb: float
    current_tasks: int
    total_tasks_processed: int
    average_response_time: float
    specialization: str

class SystemStatus(BaseModel):
    loaded_models: List[str]
    total_vram_gb: float
    used_vram_gb: float
    available_vram_gb: float
    gpu_temperature_c: float
    gpu_utilization_percent: float
    queued_tasks: int
    running_tasks: int
    completed_tasks: int

class ModelLoadRequest(BaseModel):
    model_id: str
    force_reload: bool = False

class ModelSwitchRequest(BaseModel):
    from_model: str
    to_model: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Local Model Manager API")

    # Initialize components
    import os
    config_path = os.environ.get(
        "LOCAL_MODEL_CONFIG",
        Path.home() / ".local-model-manager" / "configs" / "model_configs.yaml"
    )
    config_path = str(config_path)

    llm_loader = LLMLoader(config_path)
    gpu_monitor = GPUMonitor()
    memory_optimizer = MemoryOptimizer(gpu_monitor)
    resource_manager = ResourceManager(llm_loader)
    parallel_manager = ParallelModelManager(llm_loader)

    # Start services
    await gpu_monitor.start_monitoring()
    await resource_manager.start()
    await parallel_manager.start_processor()

    # Store in app state
    app.state.llm_loader = llm_loader
    app.state.gpu_monitor = gpu_monitor
    app.state.memory_optimizer = memory_optimizer
    app.state.resource_manager = resource_manager
    app.state.parallel_manager = parallel_manager

    # Load initial models
    await _load_initial_models(llm_loader)

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Local Model Manager API")
    await parallel_manager.stop_processor()
    await resource_manager.stop()
    await gpu_monitor.stop_monitoring()
    await llm_loader.shutdown()
    logger.info("API shutdown complete")

async def _load_initial_models(llm_loader: LLMLoader):
    """Load initial set of models"""
    initial_models = ["phi-3.5-mini", "llama-3.2-3b"]  # Load 2 models initially

    for model_id in initial_models:
        try:
            success = await llm_loader.load_model(model_id)
            if success:
                logger.info(f"Loaded initial model: {model_id}")
            else:
                logger.warning(f"Failed to load initial model: {model_id}")
        except Exception as e:
            logger.error(f"Error loading initial model {model_id}: {e}")

# Create FastAPI app
app = FastAPI(
    title="Local Model Manager API",
    description="Local model inference API optimized for edge AI deployment",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def get_priority_from_string(priority_str: str) -> TaskPriority:
    """Convert string to TaskPriority enum"""
    priority_map = {
        "low": TaskPriority.LOW,
        "medium": TaskPriority.MEDIUM,
        "high": TaskPriority.HIGH,
        "critical": TaskPriority.CRITICAL
    }
    return priority_map.get(priority_str.lower(), TaskPriority.MEDIUM)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Local Model Manager API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/status"
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status"""
    llm_loader = app.state.llm_loader
    gpu_monitor = app.state.gpu_monitor
    parallel_manager = app.state.parallel_manager

    # Get memory info
    memory_info = llm_loader.get_memory_info()
    queue_status = parallel_manager.get_queue_status()
    current_snapshot = gpu_monitor.get_current_snapshot()

    return SystemStatus(
        loaded_models=llm_loader.list_loaded_models(),
        total_vram_gb=memory_info["total_vram_gb"],
        used_vram_gb=memory_info["used_vram_gb"],
        available_vram_gb=memory_info["available_vram_gb"],
        gpu_temperature_c=current_snapshot.temperature_c if current_snapshot else 0,
        gpu_utilization_percent=current_snapshot.utilization_percent if current_snapshot else 0,
        queued_tasks=queue_status["queued_tasks"],
        running_tasks=queue_status["running_tasks"],
        completed_tasks=queue_status["completed_tasks"]
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all models and their status"""
    llm_loader = app.state.llm_loader
    resource_manager = app.state.resource_manager

    models = []
    for model_id, config in llm_loader.downloader.configs.items():
        resource_info = resource_manager.model_resources.get(model_id)

        model_info = ModelInfo(
            model_id=model_id,
            name=config.name,
            status=resource_info.status.value if resource_info else "unknown",
            vram_usage_gb=config.estimated_vram_gb,
            current_tasks=resource_info.current_tasks if resource_info else 0,
            total_tasks_processed=resource_info.total_tasks_processed if resource_info else 0,
            average_response_time=resource_info.average_response_time if resource_info else 0,
            specialization=config.specialization
        )
        models.append(model_info)

    return models

@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model into memory"""
    llm_loader = app.state.llm_loader

    success = await llm_loader.load_model(request.model_id, request.force_reload)

    if success:
        return {"message": f"Model {request.model_id} loaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model {request.model_id}")

@app.post("/models/unload/{model_id}")
async def unload_model(model_id: str):
    """Unload a model from memory"""
    llm_loader = app.state.llm_loader

    success = await llm_loader.unload_model(model_id)

    if success:
        return {"message": f"Model {model_id} unloaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to unload model {model_id}")

@app.post("/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch from one model to another"""
    llm_loader = app.state.llm_loader

    success = await llm_loader.switch_model(request.from_model, request.to_model)

    if success:
        return {"message": f"Switched from {request.from_model} to {request.to_model}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to switch models")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text using local models"""
    parallel_manager = app.state.parallel_manager
    resource_manager = app.state.resource_manager

    # Create task
    task = parallel_manager.create_task(
        prompt=request.prompt,
        task_type=request.task_type,
        model_id=request.model_id or "",
        priority=get_priority_from_string(request.priority),
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    if request.stream:
        # Streaming implementation would go here
        raise HTTPException(status_code=501, detail="Streaming not implemented yet")

    # Submit task
    task_id = await parallel_manager.submit_task(task)

    if not task_id:
        raise HTTPException(status_code=500, detail="Failed to submit task")

    # Wait for completion
    result = await parallel_manager.wait_for_task(task_id, timeout=120)  # 2 minute timeout

    if not result:
        raise HTTPException(status_code=408, detail="Task timed out")

    # Update resource manager
    await resource_manager.update_model_performance(
        result.model_id, result.time_taken, result.success
    )

    # Update GPU monitor
    if result.success:
        app.state.gpu_monitor.track_model_memory(
            result.model_id, 1.0, result.time_taken  # Simplified tracking
        )

    return GenerationResponse(
        task_id=result.task_id,
        model_id=result.model_id,
        text=result.result,
        tokens_generated=result.tokens_generated,
        time_taken=result.time_taken,
        success=result.success,
        error_message=result.error_message
    )

@app.post("/generate/async")
async def generate_text_async(request: GenerationRequest):
    """Submit async generation task"""
    parallel_manager = app.state.parallel_manager

    # Create task
    task = parallel_manager.create_task(
        prompt=request.prompt,
        task_type=request.task_type,
        model_id=request.model_id or "",
        priority=get_priority_from_string(request.priority),
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    # Submit task
    task_id = await parallel_manager.submit_task(task)

    if not task_id:
        raise HTTPException(status_code=500, detail="Failed to submit task")

    return {"task_id": task_id, "status": "queued"}

@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    parallel_manager = app.state.parallel_manager

    status = await parallel_manager.get_task_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatus(**status)

@app.get("/tasks")
async def list_tasks(status: Optional[str] = None):
    """List all tasks (optionally filtered by status)"""
    parallel_manager = app.state.parallel_manager

    # This would need implementation in parallel_manager
    queue_status = parallel_manager.get_queue_status()

    return {
        "queue_status": queue_status,
        "filter": status
    }

@app.post("/memory/optimize")
async def optimize_memory(aggressive: bool = False):
    """Optimize GPU memory"""
    memory_optimizer = app.state.memory_optimizer

    result = await memory_optimizer.optimize_memory(aggressive=aggressive)

    return {
        "success": result.success,
        "memory_freed_gb": result.memory_freed_gb,
        "optimization_time_s": result.optimization_time_s,
        "optimizations_applied": result.optimizations_applied,
        "recommendations": result.recommendations
    }

@app.get("/memory/stats")
async def get_memory_stats():
    """Get detailed memory statistics"""
    gpu_monitor = app.state.gpu_monitor
    memory_optimizer = app.state.memory_optimizer

    memory_stats = gpu_monitor.get_memory_stats()
    optimization_stats = memory_optimizer.get_optimization_stats()

    return {
        "memory_stats": memory_stats,
        "optimization_stats": optimization_stats
    }

@app.post("/models/switching-strategy")
async def set_switching_strategy(strategy: str):
    """Set model switching strategy"""
    resource_manager = app.state.resource_manager

    try:
        strategy_enum = ModelSwitchingStrategy(strategy.lower())
        resource_manager.set_switching_strategy(strategy_enum)
        return {"message": f"Switching strategy set to {strategy}"}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        llm_loader = app.state.llm_loader
        gpu_monitor = app.state.gpu_monitor

        # Basic health checks
        memory_info = llm_loader.get_memory_info()
        current_snapshot = gpu_monitor.get_current_snapshot()

        health_status = "healthy"
        issues = []

        if memory_info["total_vram_gb"] == 0:
            health_status = "degraded"
            issues.append("GPU not detected")

        if current_snapshot and current_snapshot.temperature_c > 85:
            health_status = "degraded"
            issues.append(f"High GPU temperature: {current_snapshot.temperature_c:.1f}°C")

        return {
            "status": health_status,
            "issues": issues,
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "issues": [str(e)],
            "timestamp": time.time()
        }

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()