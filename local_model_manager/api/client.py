import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
import json
import time

logger = logging.getLogger(__name__)

class LocalModelClient:
    """Client for interacting with Local Model Manager API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use 'async with' context.")

        url = f"{self.base_url}{endpoint}"

        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")

                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            raise

    # System endpoints
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return await self._request("GET", "/status")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return await self._request("GET", "/health")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return await self._request("GET", "/memory/stats")

    async def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """Optimize memory"""
        params = {"aggressive": aggressive} if aggressive else {}
        return await self._request("POST", "/memory/optimize", params=params)

    # Model endpoints
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models"""
        response = await self._request("GET", "/models")
        return response

    async def load_model(self, model_id: str, force_reload: bool = False) -> Dict[str, Any]:
        """Load a model"""
        data = {"model_id": model_id, "force_reload": force_reload}
        return await self._request("POST", "/models/load", json=data)

    async def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload a model"""
        return await self._request("POST", f"/models/unload/{model_id}")

    async def switch_model(self, from_model: str, to_model: str) -> Dict[str, Any]:
        """Switch models"""
        data = {"from_model": from_model, "to_model": to_model}
        return await self._request("POST", "/models/switch", json=data)

    async def set_switching_strategy(self, strategy: str) -> Dict[str, Any]:
        """Set model switching strategy"""
        data = {"strategy": strategy}
        return await self._request("POST", "/models/switching-strategy", json=data)

    # Generation endpoints
    async def generate_text(self,
                           prompt: str,
                           model_id: Optional[str] = None,
                           task_type: str = "general",
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           priority: str = "medium") -> Dict[str, Any]:
        """Generate text (synchronous)"""
        data = {
            "prompt": prompt,
            "model_id": model_id,
            "task_type": task_type,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "priority": priority,
            "stream": False
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        return await self._request("POST", "/generate", json=data)

    async def generate_text_async(self,
                                 prompt: str,
                                 model_id: Optional[str] = None,
                                 task_type: str = "general",
                                 max_tokens: Optional[int] = None,
                                 temperature: Optional[float] = None,
                                 top_p: Optional[float] = None,
                                 priority: str = "medium") -> str:
        """Submit async generation task"""
        data = {
            "prompt": prompt,
            "model_id": model_id,
            "task_type": task_type,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "priority": priority
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        response = await self._request("POST", "/generate/async", json=data)
        return response["task_id"]

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status"""
        return await self._request("GET", f"/tasks/{task_id}")

    async def wait_for_task(self,
                           task_id: str,
                           timeout: Optional[float] = None,
                           poll_interval: float = 1.0) -> Dict[str, Any]:
        """Wait for task completion"""
        start_time = time.time()

        while True:
            status = await self.get_task_status(task_id)

            if status["status"] in ["completed", "failed"]:
                return status

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

            await asyncio.sleep(poll_interval)

    async def generate_text_with_wait(self,
                                     prompt: str,
                                     model_id: Optional[str] = None,
                                     task_type: str = "general",
                                     max_tokens: Optional[int] = None,
                                     temperature: Optional[float] = None,
                                     top_p: Optional[float] = None,
                                     priority: str = "medium",
                                     timeout: Optional[float] = 120) -> Dict[str, Any]:
        """Generate text and wait for completion"""
        task_id = await self.generate_text_async(
            prompt=prompt,
            model_id=model_id,
            task_type=task_type,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            priority=priority
        )

        final_status = await self.wait_for_task(task_id, timeout=timeout)

        if final_status["status"] == "completed":
            return final_status["result"]
        else:
            raise Exception(f"Task failed: {final_status.get('error_message', 'Unknown error')}")

    # Batch operations
    async def batch_generate(self,
                           prompts: List[str],
                           model_id: Optional[str] = None,
                           task_type: str = "general",
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           priority: str = "medium") -> List[Dict[str, Any]]:
        """Generate text for multiple prompts"""
        tasks = []

        for prompt in prompts:
            task = self.generate_text_with_wait(
                prompt=prompt,
                model_id=model_id,
                task_type=task_type,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                priority=priority
            )
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    # Helper methods for common use cases
    async def code_generation(self, prompt: str, model_id: Optional[str] = None) -> str:
        """Generate code using specialized model"""
        response = await self.generate_text_with_wait(
            prompt=prompt,
            model_id=model_id,
            task_type="code",
            temperature=0.2,
            top_p=0.9,
            priority="high"
        )
        return response["text"]

    async def creative_writing(self, prompt: str, model_id: Optional[str] = None) -> str:
        """Creative writing using specialized model"""
        response = await self.generate_text_with_wait(
            prompt=prompt,
            model_id=model_id,
            task_type="creative",
            temperature=0.9,
            top_p=0.95,
            priority="medium"
        )
        return response["text"]

    async def analysis(self, prompt: str, model_id: Optional[str] = None) -> str:
        """Analysis using specialized model"""
        response = await self.generate_text_with_wait(
            prompt=prompt,
            model_id=model_id,
            task_type="analysis",
            temperature=0.5,
            top_p=0.9,
            priority="medium"
        )
        return response["text"]

    async def conversation(self, prompt: str, model_id: Optional[str] = None) -> str:
        """General conversation"""
        response = await self.generate_text_with_wait(
            prompt=prompt,
            model_id=model_id,
            task_type="general",
            temperature=0.7,
            top_p=0.9,
            priority="medium"
        )
        return response["text"]

    # Monitoring and utilities
    async def monitor_memory(self, duration_minutes: int = 10) -> AsyncGenerator[Dict[str, Any], None]:
        """Monitor memory usage over time"""
        end_time = time.time() + (duration_minutes * 60)

        while time.time() < end_time:
            try:
                stats = await self.get_memory_stats()
                yield {
                    "timestamp": time.time(),
                    "stats": stats
                }
            except Exception as e:
                logger.error(f"Error monitoring memory: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def auto_optimize_if_needed(self, threshold: float = 0.85) -> bool:
        """Auto-optimize memory if usage exceeds threshold"""
        try:
            stats = await self.get_memory_stats()
            memory_stats = stats["memory_stats"]["current"]
            usage_ratio = memory_stats["usage_percent"] / 100

            if usage_ratio > threshold:
                logger.info(f"Memory usage {usage_ratio:.1%} exceeds threshold {threshold:.1%}, optimizing...")
                result = await self.optimize_memory(aggressive=False)
                return result["memory_freed_gb"] > 0.1  # Return True if significant memory was freed

            return False

        except Exception as e:
            logger.error(f"Error in auto-optimization: {e}")
            return False

    async def benchmark_model(self,
                             model_id: str,
                             test_prompts: List[str],
                             max_tokens: int = 100) -> Dict[str, Any]:
        """Benchmark a model with test prompts"""
        results = []

        # Ensure model is loaded
        await self.load_model(model_id)

        for i, prompt in enumerate(test_prompts):
            try:
                start_time = time.time()
                response = await self.generate_text_with_wait(
                    prompt=prompt,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    priority="high"
                )
                end_time = time.time()

                results.append({
                    "prompt_index": i,
                    "prompt_length": len(prompt),
                    "response_length": len(response["text"]),
                    "tokens_generated": response["tokens_generated"],
                    "time_taken": response["time_taken"],
                    "success": True
                })

            except Exception as e:
                results.append({
                    "prompt_index": i,
                    "error": str(e),
                    "success": False
                })

        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            avg_time = sum(r["time_taken"] for r in successful_results) / len(successful_results)
            total_tokens = sum(r["tokens_generated"] for r in successful_results)
            tokens_per_second = total_tokens / sum(r["time_taken"] for r in successful_results)
        else:
            avg_time = 0
            tokens_per_second = 0

        return {
            "model_id": model_id,
            "total_prompts": len(test_prompts),
            "successful_prompts": len(successful_results),
            "success_rate": len(successful_results) / len(test_prompts),
            "average_time_per_prompt": avg_time,
            "tokens_per_second": tokens_per_second,
            "detailed_results": results
        }

# Convenience functions for quick usage
async def quick_generate(prompt: str,
                        task_type: str = "general",
                        base_url: str = "http://localhost:8000") -> str:
    """Quick generation with minimal setup"""
    async with LocalModelClient(base_url) as client:
        response = await client.generate_text_with_wait(
            prompt=prompt,
            task_type=task_type
        )
        return response["text"]

async def quick_status(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Quick status check"""
    async with LocalModelClient(base_url) as client:
        return await client.get_system_status()