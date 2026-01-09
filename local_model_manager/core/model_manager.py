import os
import yaml
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import requests
import aiohttp
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    repo_id: str
    gguf_file: str
    context_size: int
    gpu_layers: int
    batch_size: int
    max_tokens: int
    temperature: float
    top_p: float
    repeat_penalty: float
    estimated_vram_gb: float
    specialization: str

class ModelDownloader:
    def __init__(self, config_path: str, models_dir: str = None, cache_dir: str = None):
        self.config_path = Path(config_path)

        # Use provided directories or defaults
        if models_dir is None:
            models_dir = Path.home() / ".local-model-manager" / "models"
        if cache_dir is None:
            cache_dir = Path.home() / ".local-model-manager" / "cache"

        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, ModelConfig]:
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        models = {}
        for model_id, model_data in config_data['models'].items():
            models[model_id] = ModelConfig(**model_data)

        return models

    async def download_model(self, model_id: str, force_download: bool = False) -> bool:
        """Download a model from HuggingFace Hub"""
        if model_id not in self.configs:
            logger.error(f"Model {model_id} not found in configuration")
            return False

        config = self.configs[model_id]
        model_dir = self.models_dir / model_id
        model_path = model_dir / config.gguf_file

        # Check if model already exists
        if model_path.exists() and not force_download:
            logger.info(f"Model {model_id} already exists at {model_path}")
            return True

        # Create directory
        model_dir.mkdir(exist_ok=True)

        # Construct download URL
        gguf_repo_id = config.repo_id.replace("-instruct", "-GGUF").replace("-IT", "-GGUF")
        url = f"https://huggingface.co/TheBloke/{gguf_repo_id}/resolve/main/{config.gguf_file}"

        logger.info(f"Downloading {model_id} from {url}")

        try:
            await self._download_file(url, model_path)
            logger.info(f"Successfully downloaded {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {str(e)}")
            return False

    async def _download_file(self, url: str, destination: Path) -> None:
        """Download file with progress bar and resume capability"""
        resume_byte_pos = 0

        # Check for partial download
        if destination.exists():
            resume_byte_pos = destination.stat().st_size

        headers = {}
        if resume_byte_pos > 0:
            headers['Range'] = f'bytes={resume_byte_pos}-'

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos

                mode = 'ab' if resume_byte_pos > 0 else 'wb'

                with open(destination, mode) as file, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=destination.name,
                    initial=resume_byte_pos,
                    ascii=True
                ) as pbar:
                    async for chunk in response.content.iter_chunked(8192):
                        file.write(chunk)
                        pbar.update(len(chunk))

    async def verify_model(self, model_id: str) -> bool:
        """Verify model file integrity"""
        if model_id not in self.configs:
            return False

        config = self.configs[model_id]
        model_path = self.models_dir / model_id / config.gguf_file

        if not model_path.exists():
            return False

        # Basic size check
        if model_path.stat().st_size < 100_000_000:  # Less than 100MB
            logger.warning(f"Model file seems too small: {model_path}")
            return False

        logger.info(f"Model {model_id} verification passed")
        return True

    async def download_all_models(self) -> Dict[str, bool]:
        """Download all configured models"""
        results = {}

        for model_id in self.configs.keys():
            logger.info(f"Starting download for {model_id}")
            results[model_id] = await self.download_model(model_id)

        return results

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get local path for a model"""
        if model_id not in self.configs:
            return None

        config = self.configs[model_id]
        model_path = self.models_dir / model_id / config.gguf_file

        if model_path.exists():
            return model_path

        return None

    def list_available_models(self) -> List[str]:
        """List all available models"""
        available = []
        for model_id in self.configs.keys():
            if self.get_model_path(model_id):
                available.append(model_id)
        return available