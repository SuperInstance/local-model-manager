#!/usr/bin/env python3
"""
Demonstration script for Local Model Manager system
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_model_manager.api.client import LocalModelClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_basic_usage(client: LocalModelClient):
    """Demonstrate basic usage"""
    logger.info("=== Basic Usage Demo ===")

    # Check system status
    status = await client.get_system_status()
    logger.info(f"System Status:")
    logger.info(f"  Loaded models: {status['loaded_models']}")
    logger.info(f"  GPU Memory: {status['used_vram_gb']:.2f}/{status['total_vram_gb']:.2f}GB")
    logger.info(f"  GPU Temperature: {status['gpu_temperature_c']:.1f}°C")

    # Simple generation
    logger.info("\n1. Simple text generation:")
    response = await client.generate_text_with_wait(
        prompt="What is the future of artificial intelligence?",
        task_type="general",
        max_tokens=150
    )
    logger.info(f"Response: {response['text'][:200]}...")

    # Code generation
    logger.info("\n2. Code generation:")
    code = await client.code_generation(
        "Write a Python function to calculate fibonacci numbers"
    )
    logger.info(f"Generated code:\n{code}")

    # Creative writing
    logger.info("\n3. Creative writing:")
    creative = await client.creative_writing(
        "Write a short story about a robot discovering music"
    )
    logger.info(f"Creative writing:\n{creative[:300]}...")

async def demo_parallel_processing(client: LocalModelClient):
    """Demonstrate parallel processing capabilities"""
    logger.info("\n=== Parallel Processing Demo ===")

    # Different types of tasks in parallel
    tasks = [
        ("Explain photosynthesis", "general"),
        ("Write a function to sort a list in Python", "code"),
        ("Describe a peaceful mountain scene", "creative"),
        ("Analyze the economic impact of renewable energy", "analysis"),
        ("What are the main causes of climate change?", "general")
    ]

    prompts = [task[0] for task in tasks]
    task_types = [task[1] for task in tasks]

    logger.info(f"Running {len(prompts)} different tasks in parallel...")
    start_time = time.time()

    results = await client.batch_generate(
        prompts=prompts,
        max_tokens=100
    )

    end_time = time.time()
    successful = sum(1 for r in results if not isinstance(r, Exception))

    logger.info(f"Completed {successful}/{len(prompts)} tasks in {end_time - start_time:.2f} seconds")

    for i, (prompt, task_type) in enumerate(tasks):
        if not isinstance(results[i], Exception) and results[i].get("success"):
            logger.info(f"✓ {task_type}: {results[i]['text'][:100]}...")
        else:
            logger.error(f"✗ {task_type}: Failed")

async def demo_memory_management(client: LocalModelClient):
    """Demonstrate memory management features"""
    logger.info("\n=== Memory Management Demo ===")

    # Get initial memory state
    initial_stats = await client.get_memory_stats()
    initial_memory = initial_stats["memory_stats"]["current"]["used_vram_gb"]
    logger.info(f"Initial GPU memory: {initial_memory:.2f}GB")

    # Run multiple generations to use memory
    logger.info("Running multiple generations to stress memory...")
    for i in range(5):
        await client.generate_text_with_wait(
            prompt=f"Generate a detailed explanation of topic {i+1}",
            max_tokens=200
        )

    # Check memory after stress
    stress_stats = await client.get_memory_stats()
    stress_memory = stress_stats["memory_stats"]["current"]["used_vram_gb"]
    logger.info(f"Memory after stress test: {stress_memory:.2f}GB (+{stress_memory - initial_memory:.2f}GB)")

    # Optimize memory
    logger.info("Running memory optimization...")
    opt_result = await client.optimize_memory(aggressive=False)
    logger.info(f"Optimization freed {opt_result['memory_freed_gb']:.3f}GB")
    logger.info(f"Optimizations applied: {', '.join(opt_result['optimizations_applied'])}")

    # Check final memory
    final_stats = await client.get_memory_stats()
    final_memory = final_stats["memory_stats"]["current"]["used_vram_gb"]
    logger.info(f"Final GPU memory: {final_memory:.2f}GB")

async def demo_model_switching(client: LocalModelClient):
    """Demonstrate model switching capabilities"""
    logger.info("\n=== Model Switching Demo ===")

    # List available models
    models = await client.list_models()
    logger.info("Available models:")
    for model in models:
        logger.info(f"  - {model['model_id']}: {model['status']} ({model['specialization']})")

    # Generate with different specializations
    tasks = [
        ("Write a Python class for a binary tree", "code"),
        ("Compose a haiku about programming", "creative"),
        ("Explain the concept of machine learning", "general")
    ]

    for prompt, expected_type in tasks:
        logger.info(f"\nTask: {expected_type}")
        logger.info(f"Prompt: {prompt}")

        response = await client.generate_text_with_wait(
            prompt=prompt,
            task_type=expected_type,
            max_tokens=150
        )

        logger.info(f"Used model: {response['model_id']}")
        logger.info(f"Response: {response['text'][:150]}...")
        logger.info(f"Time: {response['time_taken']:.2f}s, Tokens: {response['tokens_generated']}")

async def demo_monitoring(client: LocalModelClient):
    """Demonstrate monitoring capabilities"""
    logger.info("\n=== Monitoring Demo ===")

    # Monitor memory for a short period
    logger.info("Monitoring system for 60 seconds...")
    monitor_task = asyncio.create_task(
        monitor_system_for_duration(client, 60)
    )

    # Run some tasks while monitoring
    await asyncio.sleep(5)
    await client.generate_text_with_wait("Test prompt 1", max_tokens=50)
    await asyncio.sleep(10)
    await client.generate_text_with_wait("Test prompt 2", max_tokens=100)
    await asyncio.sleep(15)
    await client.optimize_memory()

    await monitor_task

async def monitor_system_for_duration(client: LocalModelClient, duration_seconds: int):
    """Monitor system for specified duration"""
    start_time = time.time()
    readings = []

    while time.time() - start_time < duration_seconds:
        try:
            stats = await client.get_memory_stats()
            current = stats["memory_stats"]["current"]
            readings.append({
                "timestamp": time.time(),
                "memory_gb": current["used_vram_gb"],
                "temperature": current["temperature_c"],
                "utilization": current["utilization_percent"]
            })

            if len(readings) % 5 == 0:  # Log every 5th reading
                logger.info(f"Memory: {current['used_vram_gb']:.2f}GB, "
                          f"Temp: {current['temperature_c']:.1f}°C, "
                          f"Util: {current['utilization_percent']:.1f}%")

        except Exception as e:
            logger.error(f"Monitoring error: {e}")

        await asyncio.sleep(5)

    # Summary
    if readings:
        avg_memory = sum(r["memory_gb"] for r in readings) / len(readings)
        max_memory = max(r["memory_gb"] for r in readings)
        avg_temp = sum(r["temperature"] for r in readings) / len(readings)
        max_temp = max(r["temperature"] for r in readings)

        logger.info(f"Monitoring Summary:")
        logger.info(f"  Memory - Avg: {avg_memory:.2f}GB, Max: {max_memory:.2f}GB")
        logger.info(f"  Temperature - Avg: {avg_temp:.1f}°C, Max: {max_temp:.1f}°C")
        logger.info(f"  Total readings: {len(readings)}")

async def main():
    """Main demonstration function"""
    logger.info("🚀 Local Model Manager Demo")
    logger.info("=" * 50)

    async with LocalModelClient("http://localhost:8000") as client:
        try:
            # Health check
            health = await client.health_check()
            if health["status"] != "healthy":
                logger.warning(f"System health: {health['status']}")
                if health.get("issues"):
                    logger.warning(f"Issues: {health['issues']}")
            else:
                logger.info("✓ System health check passed")

            # Run demos
            await demo_basic_usage(client)
            await demo_parallel_processing(client)
            await demo_memory_management(client)
            await demo_model_switching(client)
            await demo_monitoring(client)

            logger.info("\n🎉 Demo completed successfully!")
            logger.info("The Local Model Manager system is ready for use.")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            logger.info("Make sure the server is running with: python scripts/start_server.py")
            return False

    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)