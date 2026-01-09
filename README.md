# Local Model Manager

A robust local model management system optimized for edge AI deployment, capable of running multiple small models in parallel with automatic memory management and intelligent model switching.

## Features

### Core Capabilities

- **Parallel Model Execution**: Run 2-3 models simultaneously (Phi-3.5-mini, Llama-3.2-3B, Gemma-2-2B)
- **Intelligent Memory Management**: Automatic GPU memory monitoring and optimization
- **Smart Model Switching**: Multiple strategies (LRU, LFU, priority-based, hybrid)
- **Resource Allocation**: Dynamic resource allocation based on task requirements
- **Real-time Monitoring**: GPU temperature, memory usage, and performance tracking

### Supported Models

- **Phi-3.5-mini**: Optimized for code and reasoning tasks (~2.1GB VRAM)
- **Llama-3.2-3B**: General purpose and conversation (~2.4GB VRAM)
- **Gemma-2-2B**: Creative writing and summarization (~1.8GB VRAM)

### System Features

- **RESTful API**: FastAPI-based inference server with comprehensive endpoints
- **Async Client**: Python client library for easy integration
- **Auto-optimization**: Memory cleanup and GPU optimization
- **Health Monitoring**: System health checks and performance metrics
- **Batch Processing**: Parallel generation for multiple prompts

## Quick Start

### Prerequisites

- NVIDIA GPU with 6GB+ VRAM (RTX 4050 optimized)
- Python 3.10+
- CUDA-compatible drivers

### Installation

1. **Install the package**:
```bash
pip install local-model-manager
```

2. **Download models**:
```bash
python -m local_model_manager.scripts.download_models
```

3. **Start the server**:
```bash
local-model-server
```

4. **Test the system**:
```bash
python examples/demo.py
```

### Development Installation

```bash
git clone https://github.com/yourusername/local-model-manager.git
cd local-model-manager
pip install -e ".[dev]"
```

## API Usage

### Python Client

```python
import asyncio
from local_model_manager.api.client import LocalModelClient

async def main():
    async with LocalModelClient("http://localhost:8000") as client:
        # Generate text
        response = await client.generate_text_with_wait(
            prompt="Explain quantum computing",
            task_type="analysis"
        )
        print(response["text"])

        # Code generation
        code = await client.code_generation(
            "Write a Python function to find prime numbers"
        )
        print(code)

        # Check system status
        status = await client.get_system_status()
        print(f"GPU Memory: {status['used_vram_gb']:.2f}GB")

asyncio.run(main())
```

### REST API

```bash
# Generate text
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "task_type": "general",
    "max_tokens": 100
  }'

# Check system status
curl "http://localhost:8000/status"

# List models
curl "http://localhost:8000/models"
```

## Architecture

### Core Components

1. **Model Manager** (`local_model_manager/core/model_manager.py`)
   - Handles model downloads and verification
   - Manages GGUF model files from HuggingFace

2. **LLM Loader** (`local_model_manager/core/llm_loader.py`)
   - Loads models with llama-cpp-python
   - Manages GPU memory allocation
   - Handles model unloading and switching

3. **Parallel Manager** (`local_model_manager/core/parallel_manager.py`)
   - Task queue with priority support
   - Parallel execution across multiple models
   - Model selection based on task type

4. **Resource Manager** (`local_model_manager/core/resource_manager.py`)
   - Memory usage tracking
   - Model switching strategies
   - Performance optimization

5. **GPU Monitor** (`local_model_manager/monitoring/gpu_monitor.py`)
   - Real-time GPU monitoring
   - Temperature and memory tracking
   - Alert system for threshold breaches

6. **Memory Optimizer** (`local_model_manager/monitoring/memory_optimizer.py`)
   - Automatic memory cleanup
   - GPU memory compaction
   - Performance recommendations

### API Server (`local_model_manager/api/server.py`)

- FastAPI-based REST interface
- Async endpoints for all operations
- Comprehensive health and status endpoints

### Client Library (`local_model_manager/api/client.py`)

- Python async client
- Convenience methods for different task types
- Batch processing support

## Configuration

### Model Configuration

Configuration is managed through `~/.local-model-manager/configs/model_configs.yaml`:

```yaml
models:
  phi-3.5-mini:
    name: "Phi-3.5-mini"
    gguf_file: "phi-3.5-mini.Q4_K_M.gguf"
    gpu_layers: 33  # Optimized for RTX 4050
    estimated_vram_gb: 2.1
    specialization: "code, reasoning, technical"

system:
  max_total_vram_gb: 6.0
  safety_margin_gb: 1.0
  max_concurrent_models: 3
```

### Environment Variables

- `LOCAL_MODEL_CONFIG`: Path to model configuration file
- `LOCAL_MODEL_MODELS_DIR`: Directory for model storage
- `LOCAL_MODEL_CACHE_DIR`: Directory for cache storage

## Performance Optimization

### Memory Management

- **Automatic Unloading**: Models are automatically unloaded when not in use
- **Smart Switching**: Multiple strategies for optimal model selection
- **Memory Monitoring**: Real-time tracking with alerts
- **Garbage Collection**: Periodic cleanup of unused resources

### GPU Optimization

- **Layer Offloading**: Configurable GPU layers for each model
- **Batch Processing**: Optimized batch sizes for throughput
- **Memory Mapping**: Efficient memory access patterns
- **Temperature Monitoring**: Thermal management

### Model Specialization

- **Task-Based Selection**: Automatic model selection based on task type
- **Performance Tracking**: Learning from historical performance
- **Resource Allocation**: Dynamic resource distribution

## Monitoring and Maintenance

### Health Checks

```bash
# System health
curl "http://localhost:8000/health"

# Detailed status
curl "http://localhost:8000/status"

# Memory statistics
curl "http://localhost:8000/memory/stats"
```

### Memory Optimization

```bash
# Manual optimization
curl -X POST "http://localhost:8000/memory/optimize"

# Aggressive cleanup
curl -X POST "http://localhost:8000/memory/optimize?aggressive=true"
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `gpu_layers` in model config
   - Enable aggressive memory optimization
   - Check for memory leaks

2. **Slow Generation**
   - Verify GPU is being used (check memory usage)
   - Adjust batch sizes and context lengths
   - Monitor GPU temperature

3. **Model Loading Failures**
   - Verify model files are downloaded correctly
   - Check file permissions
   - Ensure sufficient VRAM

### Debug Mode

```bash
# Start with debug logging
local-model-server --log-level debug
```

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate text (synchronous) |
| `/generate/async` | POST | Submit async generation task |
| `/tasks/{task_id}` | GET | Get task status |
| `/models` | GET | List all models |
| `/models/load` | POST | Load a model |
| `/models/unload/{id}` | POST | Unload a model |
| `/status` | GET | System status |
| `/health` | GET | Health check |
| `/memory/optimize` | POST | Optimize memory |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Model quantization by [TheBloke](https://huggingface.co/TheBloke)
- Inspired by the need for efficient edge AI deployment

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for detailed error messages
3. Run the system test to diagnose problems
4. Check GPU memory and temperature usage
5. Open an issue on GitHub

---

**Note**: This system is specifically optimized for RTX 4050 6GB VRAM but should work with other NVIDIA GPUs with 6GB+ VRAM. Adjust model configurations for different hardware specifications.
