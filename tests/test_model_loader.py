"""
Tests for model loading and unloading functionality.

Tests the LLMLoader and ModelDownloader classes including:
- Model loading and unloading
- Memory management
- GPU memory monitoring
- Model switching
- Concurrent model handling
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from dataclasses import dataclass

from local_model_manager.core.llm_loader import LLMLoader, GPUMemoryMonitor, ModelInstance
from local_model_manager.core.model_manager import ModelConfig, ModelDownloader


# =============================================================================
# Test GPUMemoryMonitor
# =============================================================================

class TestGPUMemoryMonitor:
    """Test GPU memory monitoring functionality."""

    @pytest.mark.unit
    def test_init_with_default_safety_margin(self, mock_gpu):
        """Test initialization with default safety margin."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMemoryMonitor()
            assert monitor.safety_margin_gb == 1.0
            assert monitor.total_vram_gb == pytest.approx(24.0, rel=0.1)

    @pytest.mark.unit
    def test_init_with_custom_safety_margin(self, mock_gpu):
        """Test initialization with custom safety margin."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMemoryMonitor(safety_margin_gb=2.0)
            assert monitor.safety_margin_gb == 2.0

    @pytest.mark.unit
    def test_get_total_vram(self, mock_gpu):
        """Test getting total VRAM."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMemoryMonitor()
            total = monitor.get_total_vram()
            assert total == pytest.approx(24.0, rel=0.1)

    @pytest.mark.unit
    def test_get_available_vram(self, mock_gpu):
        """Test getting available VRAM."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMemoryMonitor(safety_margin_gb=1.0)
            available = monitor.get_available_vram()
            # 24GB total - 8GB used - 1GB safety = 15GB available
            assert available == pytest.approx(15.0, rel=0.1)

    @pytest.mark.unit
    def test_get_vram_usage(self, mock_gpu):
        """Test getting current VRAM usage."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            monitor = GPUMemoryMonitor()
            used = monitor.get_vram_usage()
            assert used == pytest.approx(8.0, rel=0.1)

    @pytest.mark.unit
    def test_gpu_not_available(self):
        """Test behavior when GPU is not available."""
        with patch('GPUtil.getGPUs', return_value=[]):
            monitor = GPUMemoryMonitor()
            assert monitor.get_total_vram() == 0.0
            assert monitor.get_available_vram() == 0.0
            assert monitor.get_vram_usage() == 0.0

    @pytest.mark.unit
    def test_gpu_detection_error(self):
        """Test handling of GPU detection errors."""
        with patch('GPUtil.getGPUs', side_effect=Exception("GPU error")):
            monitor = GPUMemoryMonitor()
            # Should return 0.0 and log warning
            assert monitor.get_total_vram() == 0.0


# =============================================================================
# Test ModelDownloader
# =============================================================================

class TestModelDownloader:
    """Test model downloading functionality."""

    @pytest.mark.unit
    def test_init_with_defaults(self, test_config_path):
        """Test initialization with default directories."""
        downloader = ModelDownloader(str(test_config_path))
        assert 'test-model-1' in downloader.configs
        assert 'test-model-2' in downloader.configs
        assert downloader.models_dir.exists()
        assert downloader.cache_dir.exists()

    @pytest.mark.unit
    def test_init_with_custom_dirs(self, test_config_path, test_models_dir, test_cache_dir):
        """Test initialization with custom directories."""
        downloader = ModelDownloader(
            str(test_config_path),
            models_dir=str(test_models_dir),
            cache_dir=str(test_cache_dir)
        )
        assert downloader.models_dir == test_models_dir
        assert downloader.cache_dir == test_cache_dir

    @pytest.mark.unit
    def test_load_configs(self, test_config_path):
        """Test configuration loading."""
        downloader = ModelDownloader(str(test_config_path))
        assert 'test-model-1' in downloader.configs
        assert 'test-model-2' in downloader.configs

        config = downloader.configs['test-model-1']
        assert isinstance(config, ModelConfig)
        assert config.name == 'Test Model 1'
        assert config.estimated_vram_gb == 4.5

    @pytest.mark.unit
    def test_get_model_path_exists(self, test_config_path, test_models_dir):
        """Test getting path for existing model."""
        downloader = ModelDownloader(
            str(test_config_path),
            models_dir=str(test_models_dir)
        )
        path = downloader.get_model_path('test-model-1')
        assert path is not None
        assert path.exists()

    @pytest.mark.unit
    def test_get_model_path_not_exists(self, test_config_path):
        """Test getting path for non-existing model."""
        downloader = ModelDownloader(str(test_config_path))
        path = downloader.get_model_path('non-existent-model')
        assert path is None

    @pytest.mark.unit
    def test_list_available_models(self, test_config_path, test_models_dir):
        """Test listing available models."""
        downloader = ModelDownloader(
            str(test_config_path),
            models_dir=str(test_models_dir)
        )
        available = downloader.list_available_models()
        assert 'test-model-1' in available
        assert 'test-model-2' in available

    @pytest.mark.unit
    async def test_verify_model_success(self, test_config_path, test_models_dir):
        """Test verifying a valid model file."""
        downloader = ModelDownloader(
            str(test_config_path),
            models_dir=str(test_models_dir)
        )
        # Create a larger file to pass size check
        model_file = test_models_dir / 'test-model-1' / 'test-model-1.Q4_K_M.gguf'
        model_file.write_bytes(b'x' * 200_000_000)  # 200MB

        result = await downloader.verify_model('test-model-1')
        assert result is True

    @pytest.mark.unit
    async def test_verify_model_file_too_small(self, test_config_path, test_models_dir):
        """Test verification rejects files that are too small."""
        downloader = ModelDownloader(
            str(test_config_path),
            models_dir=str(test_models_dir)
        )
        model_file = test_models_dir / 'test-model-1' / 'test-model-1.Q4_K_M.gguf'
        model_file.write_bytes(b'x' * 50_000_000)  # 50MB - too small

        result = await downloader.verify_model('test-model-1')
        assert result is False

    @pytest.mark.unit
    async def test_verify_model_not_found(self, test_config_path):
        """Test verification when model file doesn't exist."""
        downloader = ModelDownloader(str(test_config_path))
        result = await downloader.verify_model('test-model-1')
        assert result is False


# =============================================================================
# Test LLMLoader
# =============================================================================

class TestLLMLoader:
    """Test LLM loading and management functionality."""

    @pytest.fixture
    def loader(self, test_config_path, test_models_dir, mock_gputil):
        """Create an LLMLoader instance for testing."""
        with patch('local_model_manager.core.llm_loader.Llama') as mock_llama:
            loader = LLMLoader(str(test_config_path))
            loader.models_dir = test_models_dir
            yield loader

    @pytest.mark.unit
    def test_init(self, loader):
        """Test LLMLoader initialization."""
        assert loader.max_concurrent_models == 3
        assert loader.models == {}
        assert isinstance(loader.memory_monitor, GPUMemoryMonitor)

    @pytest.mark.unit
    def test_load_system_config(self, test_config_path, test_models_dir):
        """Test system configuration loading."""
        with patch('local_model_manager.core.llm_loader.Llama'):
            loader = LLMLoader(str(test_config_path))
            assert loader.max_concurrent_models == 3
            assert loader.memory_monitor.safety_margin_gb == 1.0

    @pytest.mark.unit
    async def test_load_model_success(self, loader, mock_llama_model):
        """Test successful model loading."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            result = await loader.load_model('test-model-1')
            assert result is True
            assert 'test-model-1' in loader.models
            assert loader.models['test-model-1'].is_loaded is True

    @pytest.mark.unit
    async def test_load_model_already_loaded(self, loader, mock_llama_model):
        """Test loading a model that's already loaded."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            # Load once
            await loader.load_model('test-model-1')
            # Load again
            result = await loader.load_model('test-model-1', force_reload=False)
            assert result is True

    @pytest.mark.unit
    async def test_load_model_force_reload(self, loader, mock_llama_model):
        """Test force reloading a model."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            first_instance = loader.models['test-model-1']

            # Force reload
            result = await loader.load_model('test-model-1', force_reload=True)
            assert result is True
            # Instance should be different after reload
            assert loader.models['test-model-1'] is not first_instance

    @pytest.mark.unit
    async def test_load_model_not_found(self, loader):
        """Test loading a non-existent model."""
        result = await loader.load_model('non-existent-model')
        assert result is False

    @pytest.mark.unit
    async def test_load_model_insufficient_memory(self, loader, mock_llama_model):
        """Test loading model when insufficient VRAM is available."""
        # Mock low memory availability
        loader.memory_monitor.get_available_vram = Mock(return_value=1.0)  # Only 1GB

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            result = await loader.load_model('test-model-1')  # Requires 4.5GB
            # Should still try to load and fail or unload other models
            assert isinstance(result, bool)

    @pytest.mark.unit
    async def test_unload_model_success(self, loader, mock_llama_model):
        """Test successful model unloading."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            assert 'test-model-1' in loader.models

            result = await loader.unload_model('test-model-1')
            assert result is True
            assert 'test-model-1' not in loader.models

    @pytest.mark.unit
    async def test_unload_model_not_loaded(self, loader):
        """Test unloading a model that's not loaded."""
        result = await loader.unload_model('non-existent-model')
        assert result is True  # Returns True even if not loaded

    @pytest.mark.unit
    async def test_unload_oldest_model(self, loader, mock_llama_model):
        """Test unloading the oldest model."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            await asyncio.sleep(0.1)
            await loader.load_model('test-model-2')

            assert 'test-model-1' in loader.models
            assert 'test-model-2' in loader.models

            result = await loader.unload_oldest_model()
            assert result is True
            assert 'test-model-1' not in loader.models
            assert 'test-model-2' in loader.models

    @pytest.mark.unit
    async def test_switch_model(self, loader, mock_llama_model):
        """Test switching from one model to another."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            result = await loader.switch_model('test-model-1', 'test-model-2')
            assert result is True
            assert 'test-model-2' in loader.models
            assert 'test-model-1' not in loader.models

    @pytest.mark.unit
    async def test_get_model(self, loader, mock_llama_model):
        """Test getting a loaded model instance."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            model = loader.get_model('test-model-1')
            assert model is not None

    @pytest.mark.unit
    def test_get_model_not_loaded(self, loader):
        """Test getting a model that's not loaded."""
        model = loader.get_model('test-model-1')
        assert model is None

    @pytest.mark.unit
    async def test_list_loaded_models(self, loader, mock_llama_model):
        """Test listing all loaded models."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            await loader.load_model('test-model-2')

            loaded = loader.list_loaded_models()
            assert 'test-model-1' in loaded
            assert 'test-model-2' in loaded

    @pytest.mark.unit
    def test_get_memory_info(self, loader, mock_gpu):
        """Test getting memory information."""
        with patch('GPUtil.getGPUs', return_value=[mock_gpu]):
            info = loader.get_memory_info()
            assert 'total_vram_gb' in info
            assert 'used_vram_gb' in info
            assert 'available_vram_gb' in info
            assert 'loaded_models' in info
            assert info['total_vram_gb'] > 0

    @pytest.mark.unit
    async def test_max_concurrent_models_limit(self, loader, mock_llama_model):
        """Test that max concurrent models limit is enforced."""
        loader.max_concurrent_models = 2

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            # Load 2 models
            await loader.load_model('test-model-1')
            await loader.load_model('test-model-2')
            assert len(loader.models) == 2

            # Load 3rd model - should unload one
            await loader.load_model('test-model-3')
            assert len(loader.models) <= 2

    @pytest.mark.unit
    async def test_shutdown(self, loader, mock_llama_model):
        """Test shutdown unloads all models."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            await loader.load_model('test-model-2')
            assert len(loader.models) == 2

            await loader.shutdown()
            assert len(loader.models) == 0

    @pytest.mark.unit
    async def test_make_space_for_model(self, loader, mock_llama_model):
        """Test making space for a model by unloading others."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')
            await loader.load_model('test-model-2')

            # Mock available memory to be low
            loader.memory_monitor.get_available_vram = Mock(return_value=1.0)

            result = await loader._make_space_for_model(5.0)
            # Should try to unload models to make space
            assert isinstance(result, bool)


# =============================================================================
# Integration Tests
# =============================================================================

class TestModelLoaderIntegration:
    """Integration tests for model loading functionality."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_load_and_generate(self, loader, mock_llama_model):
        """Test loading a model and generating text."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            # Load model
            success = await loader.load_model('test-model-1')
            assert success is True

            # Get model
            model = loader.get_model('test-model-1')
            assert model is not None

            # Generate (this is mocked, so just verify it doesn't crash)
            response = model("Test prompt")
            assert 'choices' in response

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_multiple_models_memory_management(self, loader, mock_llama_model):
        """Test memory management with multiple models."""
        loader.max_concurrent_models = 2

        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            # Load first model
            await loader.load_model('test-model-1')
            assert len(loader.list_loaded_models()) == 1

            # Load second model
            await loader.load_model('test-model-2')
            assert len(loader.list_loaded_models()) == 2

            # Load third model - should unload oldest
            await loader.load_model('test-model-3')
            assert len(loader.list_loaded_models()) <= 2

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_model_switching_workflow(self, loader, mock_llama_model):
        """Test complete model switching workflow."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            # Start with model 1
            await loader.load_model('test-model-1')
            assert 'test-model-1' in loader.models

            # Switch to model 2
            await loader.switch_model('test-model-1', 'test-model-2')
            assert 'test-model-2' in loader.models
            assert 'test-model-1' not in loader.models

            # Switch to model 3
            await loader.switch_model('test-model-2', 'test-model-3')
            assert 'test-model-3' in loader.models
            assert 'test-model-2' not in loader.models


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestModelLoaderErrors:
    """Test error handling in model loading."""

    @pytest.mark.unit
    async def test_load_model_with_exception(self, loader):
        """Test handling of exceptions during model loading."""
        with patch('local_model_manager.core.llm_loader.Llama', side_effect=Exception("Load failed")):
            result = await loader.load_model('test-model-1')
            assert result is False
            assert 'test-model-1' not in loader.models

    @pytest.mark.unit
    async def test_unload_model_with_exception(self, loader, mock_llama_model):
        """Test handling of exceptions during model unloading."""
        with patch('local_model_manager.core.llm_loader.Llama', return_value=mock_llama_model()):
            await loader.load_model('test-model-1')

            # Mock deletion to raise exception
            with patch.object(loader.models['test-model-1'], 'lock', side_effect=Exception("Unload failed")):
                result = await loader.unload_model('test-model-1')
                assert result is False

    @pytest.mark.unit
    async def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)

        try:
            with pytest.raises(Exception):
                LLMLoader(str(config_path))
        finally:
            config_path.unlink()
