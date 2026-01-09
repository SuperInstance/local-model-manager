"""
Local Model Manager - A standalone package for managing local LLM inference.

This package provides tools for running multiple local language models
with intelligent memory management and parallel execution capabilities.
"""

__version__ = "1.0.0"
__author__ = "Local Model Manager Team"

from .core.model_manager import ModelConfig, ModelDownloader
from .core.llm_loader import LLMLoader
from .core.parallel_manager import ParallelModelManager, ModelTask, TaskPriority

__all__ = [
    "ModelConfig",
    "ModelDownloader",
    "LLMLoader",
    "ParallelModelManager",
    "ModelTask",
    "TaskPriority",
]
