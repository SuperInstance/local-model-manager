#!/usr/bin/env python3
"""
Test script for Local Model Manager system
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

async def test_health(client: LocalModelClient):
    """Test health endpoint"""
    logger.info("Testing health check...")
    try:
        health = await client.health_check()
        logger.info(f"Health status: {health['status']}")
        if health.get('issues'):
            logger.warning(f"Health issues: {health['issues']}")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

async def test_system_status(client: LocalModelClient):
    """Test system status"""
    logger.info("Testing system status...")
    try:
        status = await client.get_system_status()
        logger.info(f"Loaded models: {status['loaded_models']}")
        logger.info(f"GPU memory: {status['used_vram_gb']:.2f}/{status['total_vram_gb']:.2f}GB")
        logger.info(f"GPU temp: {status['gpu_temperature_c']:.1f}°C")
        return True
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return False

async def test_model_list(client: LocalModelClient):
    """Test model listing"""
    logger.info("Testing model listing...")
    try:
        models = await client.list_models()
        logger.info(f"Available models: {len(models)}")
        for model in models:
            logger.info(f"  - {model['model_id']}: {model['status']} ({model['specialization']})")
        return True
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        return False

async def test_generation(client: LuciddreamerClient):
    """Test text generation"""
    logger.info("Testing text generation...")
    test_prompts = [
        ("What is artificial intelligence?", "general"),
        ("Write a Python function to sort a list", "code"),
        ("Write a short poem about the ocean", "creative"),
        ("Analyze the benefits of renewable energy", "analysis")
    ]

    for prompt, task_type in test_prompts:
        try:
            logger.info(f"Testing {task_type}: {prompt[:50]}...")
            start_time = time.time()
            response = await client.generate_text_with_wait(
                prompt=prompt,
                task_type=task_type,
                max_tokens=100,
                timeout=60
            )
            end_time = time.time()

            logger.info(f"  Response: {response['text'][:100]}...")
            logger.info(f"  Tokens: {response['tokens_generated']}, Time: {response['time_taken']:.2f}s")
            logger.info(f"  Total time: {end_time - start_time:.2f}s")

        except Exception as e:
            logger.error(f"Generation failed for {task_type}: {e}")
            return False

    return True

async def test_parallel_generation(client: LuciddreamerClient):
    """Test parallel generation"""
    logger.info("Testing parallel generation...")
    prompts = [
        "Explain machine learning",
        "Write a haiku about technology",
        "What is the capital of France?",
        "Describe a sunset"
    ]

    try:
        logger.info(f"Running {len(prompts)} generations in parallel...")
        start_time = time.time()

        results = await client.batch_generate(
            prompts=prompts,
            task_type="general",
            max_tokens=50
        )

        end_time = time.time()
        successful = sum(1 for r in results if not isinstance(r, Exception) and r.get("success", False))

        logger.info(f"Parallel generation completed in {end_time - start_time:.2f}s")
        logger.info(f"Successful: {successful}/{len(prompts)}")

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"  Prompt {i}: Failed - {result}")
            else:
                logger.info(f"  Prompt {i}: Success - {result['tokens_generated']} tokens")

        return successful > 0

    except Exception as e:
        logger.error(f"Parallel generation failed: {e}")
        return False

async def test_memory_optimization(client: LuciddreamerClient):
    """Test memory optimization"""
    logger.info("Testing memory optimization...")
    try:
        # Get initial memory stats
        initial_stats = await client.get_memory_stats()
        initial_memory = initial_stats["memory_stats"]["current"]["used_vram_gb"]
        logger.info(f"Initial GPU memory: {initial_memory:.2f}GB")

        # Run optimization
        opt_result = await client.optimize_memory(aggressive=False)
        logger.info(f"Optimization freed: {opt_result['memory_freed_gb']:.3f}GB")
        logger.info(f"Optimizations: {', '.join(opt_result['optimizations_applied'])}")

        # Get final memory stats
        final_stats = await client.get_memory_stats()
        final_memory = final_stats["memory_stats"]["current"]["used_vram_gb"]
        logger.info(f"Final GPU memory: {final_memory:.2f}GB")

        return True

    except Exception as e:
        logger.error(f"Memory optimization test failed: {e}")
        return False

async def test_model_switching(client: LuciddreamerClient):
    """Test model switching"""
    logger.info("Testing model switching...")
    try:
        models = await client.list_models()
        loaded_models = [m["model_id"] for m in models if m["status"] == "loaded"]

        if len(loaded_models) < 2:
            logger.warning("Need at least 2 loaded models for switching test")
            return True

        # Switch between models
        from_model = loaded_models[0]
        to_model = loaded_models[1]

        logger.info(f"Switching from {from_model} to {to_model}")
        result = await client.switch_model(from_model, to_model)
        logger.info(f"Switch result: {result['message']}")

        return True

    except Exception as e:
        logger.error(f"Model switching test failed: {e}")
        return False

async def run_all_tests(base_url: str = "http://localhost:8000"):
    """Run all tests"""
    logger.info("Starting Luciddreamer Local Models System Test")
    logger.info("=" * 60)

    async with LuciddreamerClient(base_url) as client:
        tests = [
            ("Health Check", test_health),
            ("System Status", test_system_status),
            ("Model List", test_model_list),
            ("Text Generation", test_generation),
            ("Parallel Generation", test_parallel_generation),
            ("Memory Optimization", test_memory_optimization),
            ("Model Switching", test_model_switching)
        ]

        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                result = await test_func(client)
                results[test_name] = result
                status = "✓ PASSED" if result else "✗ FAILED"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name}: ✗ ERROR - {e}")
                results[test_name] = False

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status}")

        logger.info(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All tests passed! System is ready to use.")
        else:
            logger.warning("⚠️  Some tests failed. Check the logs for details.")

        return passed == total

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Luciddreamer Local Models system")
    parser.add_argument("--url", default="http://localhost:8000", help="API server URL")
    args = parser.parse_args()

    success = asyncio.run(run_all_tests(args.url))
    sys.exit(0 if success else 1)