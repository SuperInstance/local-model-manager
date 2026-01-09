import gc
import threading
import asyncio
import logging
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
import psutil
import GPUtil
from dataclasses import dataclass
from llama_cpp import Llama
import numpy as np

from .model_manager import ModelConfig, ModelDownloader

logger = logging.getLogger(__name__)

@dataclass
class ModelInstance:
    id: str
    model: Llama
    config: ModelConfig
    vram_usage: float
    last_used: float
    is_loaded: bool
    lock: threading.Lock

class GPUMemoryMonitor:
    def __init__(self, safety_margin_gb: float = 1.0):
        self.safety_margin_gb = safety_margin_gb
        self.total_vram_gb = self._get_total_vram()

    def _get_total_vram(self) -> float:
        """Get total GPU VRAM in GB"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryTotal / 1024  # Convert MB to GB
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}")
        return 0.0

    def get_available_vram(self) -> float:
        """Get available GPU VRAM in GB"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                used = gpus[0].memoryUsed / 1024  # Convert MB to GB
                return max(0, self.total_vram_gb - used - self.safety_margin_gb)
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
        return 0.0

    def get_vram_usage(self) -> float:
        """Get current GPU VRAM usage in GB"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed / 1024  # Convert MB to GB
        except Exception as e:
            logger.warning(f"Could not get GPU memory usage: {e}")
        return 0.0

class LLMLoader:
    def __init__(self, config_path: str):
        self.downloader = ModelDownloader(config_path)
        self.config_path = config_path
        self.models: Dict[str, ModelInstance] = {}
        self.memory_monitor = GPUMemoryMonitor()
        self.max_concurrent_models = 3
        self.model_lock = asyncio.Lock()
        self._load_system_config()

    def _load_system_config(self):
        """Load system configuration"""
        import yaml
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        system_config = config_data.get('system', {})
        self.max_concurrent_models = system_config.get('max_concurrent_models', 3)
        self.memory_monitor.safety_margin_gb = system_config.get('safety_margin_gb', 1.0)

    async def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """Load a model into memory"""
        async with self.model_lock:
            return await self._load_model_internal(model_id, force_reload)

    async def _load_model_internal(self, model_id: str, force_reload: bool = False) -> bool:
        """Internal model loading without lock"""
        # Check if model is already loaded
        if model_id in self.models and not force_reload:
            logger.info(f"Model {model_id} is already loaded")
            return True

        # Check if we have too many models loaded
        if len(self.models) >= self.max_concurrent_models:
            logger.info("Maximum concurrent models reached, unloading oldest model")
            await self.unload_oldest_model()

        # Get model configuration and path
        config = self.downloader.configs.get(model_id)
        if not config:
            logger.error(f"Model {model_id} not found in configuration")
            return False

        model_path = self.downloader.get_model_path(model_id)
        if not model_path:
            logger.error(f"Model file not found for {model_id}")
            return False

        # Check available memory
        available_vram = self.memory_monitor.get_available_vram()
        if available_vram < config.estimated_vram_gb:
            logger.warning(f"Insufficient VRAM for {model_id}: "
                         f"Need {config.estimated_vram_gb}GB, Available {available_vram:.2f}GB")

            # Try to unload other models to make space
            if not await self._make_space_for_model(config.estimated_vram_gb):
                logger.error(f"Could not make space for model {model_id}")
                return False

        logger.info(f"Loading model {model_id} from {model_path}")

        try:
            # Create llama-cpp model instance
            model = Llama(
                model_path=str(model_path),
                n_ctx=config.context_size,
                n_gpu_layers=config.gpu_layers,
                n_batch=config.batch_size,
                f16_kv=True,
                use_mmap=True,
                embedding=False,
                verbose=False
            )

            # Create model instance
            model_instance = ModelInstance(
                id=model_id,
                model=model,
                config=config,
                vram_usage=config.estimated_vram_gb,
                last_used=asyncio.get_event_loop().time(),
                is_loaded=True,
                lock=threading.Lock()
            )

            self.models[model_id] = model_instance
            logger.info(f"Successfully loaded model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return False

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id not in self.models:
            logger.warning(f"Model {model_id} is not loaded")
            return True

        model_instance = self.models[model_id]

        try:
            with model_instance.lock:
                # Force garbage collection
                del model_instance.model
                model_instance.is_loaded = False

            del self.models[model_id]
            gc.collect()

            logger.info(f"Successfully unloaded model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {str(e)}")
            return False

    async def unload_oldest_model(self) -> bool:
        """Unload the least recently used model"""
        if not self.models:
            return True

        # Find oldest model
        oldest_model_id = min(self.models.keys(),
                            key=lambda k: self.models[k].last_used)

        logger.info(f"Unloading oldest model: {oldest_model_id}")
        return await self.unload_model(oldest_model_id)

    async def _make_space_for_model(self, required_vram_gb: float) -> bool:
        """Try to make space for a model by unloading others"""
        available_vram = self.memory_monitor.get_available_vram()

        # Sort models by last used time (oldest first)
        sorted_models = sorted(self.models.items(),
                             key=lambda x: x[1].last_used)

        for model_id, model_instance in sorted_models:
            if available_vram >= required_vram_gb:
                return True

            logger.info(f"Unloading model {model_id} to free memory")
            if await self.unload_model(model_id):
                available_vram = self.memory_monitor.get_available_vram()
                await asyncio.sleep(1)  # Give GPU time to free memory

        return available_vram >= required_vram_gb

    async def switch_model(self, from_model: str, to_model: str) -> bool:
        """Switch from one model to another"""
        if to_model not in self.models:
            # Load new model first
            if not await self.load_model(to_model):
                return False
        else:
            # Update last used time
            self.models[to_model].last_used = asyncio.get_event_loop().time()

        # Unload old model if different
        if from_model != to_model and from_model in self.models:
            await self.unload_model(from_model)

        return True

    def get_model(self, model_id: str) -> Optional[Llama]:
        """Get a loaded model instance"""
        if model_id in self.models:
            model_instance = self.models[model_id]
            if model_instance.is_loaded:
                model_instance.last_used = asyncio.get_event_loop().time()
                return model_instance.model

        return None

    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models"""
        return [model_id for model_id, instance in self.models.items()
                if instance.is_loaded]

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        total_vram = self.memory_monitor.total_vram_gb
        used_vram = self.memory_monitor.get_vram_usage()
        available_vram = self.memory_monitor.get_available_vram()

        model_memory = sum(instance.vram_usage for instance in self.models.values())

        return {
            "total_vram_gb": total_vram,
            "used_vram_gb": used_vram,
            "available_vram_gb": available_vram,
            "model_memory_gb": model_memory,
            "loaded_models": len(self.models),
            "max_concurrent_models": self.max_concurrent_models,
            "loaded_model_ids": self.list_loaded_models()
        }

    async def shutdown(self):
        """Cleanup and unload all models"""
        logger.info("Shutting down LLM loader")

        for model_id in list(self.models.keys()):
            await self.unload_model(model_id)

        gc.collect()
        logger.info("All models unloaded")