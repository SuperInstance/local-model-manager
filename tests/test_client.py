"""
Tests for async client functionality.

Tests the LocalModelClient class including:
- Connection management
- System status queries
- Model management
- Text generation
- Task monitoring
- Batch operations
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path

# Import after setting path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_model_manager.api.client import LocalModelClient


# =============================================================================
# Test Client Initialization
# =============================================================================

class TestClientInit:
    """Test client initialization."""

    @pytest.mark.unit
    def test_init_default_url(self):
        """Test initialization with default URL."""
        client = LocalModelClient()
        assert client.base_url == "http://localhost:8000"
        assert client.session is None

    @pytest.mark.unit
    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        client = LocalModelClient("http://example.com:8080")
        assert client.base_url == "http://example.com:8080"

    @pytest.mark.unit
    def test_context_manager(self):
        """Test using client as context manager."""
        async def test():
            async with LocalModelClient() as client:
                assert client.session is not None
            assert client.session.closed  # Should be closed after context

        # Run the test
        asyncio.run(test())


# =============================================================================
# Test System Endpoints
# =============================================================================

class TestSystemEndpoints:
    """Test client system endpoint methods."""

    @pytest.fixture
    async def client(self):
        """Create client with mocked session."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            # Mock request method
            async def mock_request(method, endpoint, **kwargs):
                response = MagicMock()
                response.status = 200

                async def json():
                    if "status" in endpoint:
                        return {"loaded_models": ["model-1"]}
                    elif "health" in endpoint:
                        return {"status": "healthy"}
                    elif "memory" in endpoint and "stats" in endpoint:
                        return {"memory_stats": {}}
                    return {}

                response.json = json
                return response

            session.request = mock_request
            session.close = AsyncMock()

            mock_session_class.return_value = session

            client = LocalModelClient()
            await client.__aenter__()
            yield client
            await client.__aexit__(None, None, None)

    @pytest.mark.unit
    async def test_get_system_status(self, client):
        """Test getting system status."""
        status = await client.get_system_status()
        assert "loaded_models" in status

    @pytest.mark.unit
    async def test_health_check(self, client):
        """Test health check."""
        health = await client.health_check()
        assert "status" in health

    @pytest.mark.unit
    async def test_get_memory_stats(self, client):
        """Test getting memory stats."""
        stats = await client.get_memory_stats()
        assert "memory_stats" in stats


# =============================================================================
# Test Model Management
# =============================================================================

class TestModelManagementClient:
    """Test client model management methods."""

    @pytest.fixture
    async def client(self):
        """Create client with mocked session."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            async def mock_request(method, endpoint, **kwargs):
                response = MagicMock()
                response.status = 200

                async def json():
                    if "models" in endpoint and method == "GET":
                        return [
                            {"model_id": "model-1", "name": "Model 1"},
                            {"model_id": "model-2", "name": "Model 2"}
                        ]
                    return {"message": "success"}

                response.json = json
                return response

            session.request = mock_request
            session.close = AsyncMock()

            mock_session_class.return_value = session

            client = LocalModelClient()
            await client.__aenter__()
            yield client
            await client.__aexit__(None, None, None)

    @pytest.mark.unit
    async def test_list_models(self, client):
        """Test listing models."""
        models = await client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    @pytest.mark.unit
    async def test_load_model(self, client):
        """Test loading a model."""
        result = await client.load_model("model-1")
        assert "message" in result

    @pytest.mark.unit
    async def test_unload_model(self, client):
        """Test unloading a model."""
        result = await client.unload_model("model-1")
        assert "message" in result

    @pytest.mark.unit
    async def test_switch_model(self, client):
        """Test switching models."""
        result = await client.switch_model("model-1", "model-2")
        assert "message" in result


# =============================================================================
# Test Text Generation
# =============================================================================

class TestTextGenerationClient:
    """Test client text generation methods."""

    @pytest.fixture
    async def client(self):
        """Create client with mocked session."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            async def mock_request(method, endpoint, **kwargs):
                response = MagicMock()
                response.status = 200

                async def json():
                    if "generate" in endpoint and "async" not in endpoint:
                        return {
                            "task_id": "test-001",
                            "text": "Generated text",
                            "tokens_generated": 50,
                            "time_taken": 1.5,
                            "success": True
                        }
                    elif "async" in endpoint:
                        return {"task_id": "test-001", "status": "queued"}
                    return {}

                response.json = json
                return response

            session.request = mock_request
            session.close = AsyncMock()

            mock_session_class.return_value = session

            client = LocalModelClient()
            await client.__aenter__()
            yield client
            await client.__aexit__(None, None, None)

    @pytest.mark.unit
    async def test_generate_text(self, client):
        """Test text generation."""
        result = await client.generate_text("Write hello world")
        assert "text" in result
        assert result["success"] is True

    @pytest.mark.unit
    async def test_generate_text_async(self, client):
        """Test async text generation submission."""
        task_id = await client.generate_text_async("Write hello world")
        assert isinstance(task_id, str)

    @pytest.mark.unit
    async def test_code_generation(self, client):
        """Test code generation helper."""
        # Mock the generate_text_with_wait method
        client.generate_text_with_wait = AsyncMock(return_value={"text": "code here"})
        result = await client.code_generation("Write a function")
        assert isinstance(result, str)

    @pytest.mark.unit
    async def test_creative_writing(self, client):
        """Test creative writing helper."""
        client.generate_text_with_wait = AsyncMock(return_value={"text": "story here"})
        result = await client.creative_writing("Write a story")
        assert isinstance(result, str)

    @pytest.mark.unit
    async def test_analysis(self, client):
        """Test analysis helper."""
        client.generate_text_with_wait = AsyncMock(return_value={"text": "analysis here"})
        result = await client.analysis("Analyze this")
        assert isinstance(result, str)

    @pytest.mark.unit
    async def test_conversation(self, client):
        """Test conversation helper."""
        client.generate_text_with_wait = AsyncMock(return_value={"text": "response here"})
        result = await client.conversation("Hello")
        assert isinstance(result, str)


# =============================================================================
# Test Task Monitoring
# =============================================================================

class TestTaskMonitoringClient:
    """Test client task monitoring methods."""

    @pytest.fixture
    async def client(self):
        """Create client with mocked session."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            call_count = [0]

            async def mock_request(method, endpoint, **kwargs):
                response = MagicMock()
                response.status = 200

                async def json():
                    if "tasks" in endpoint:
                        # First call returns running, subsequent calls return completed
                        call_count[0] += 1
                        if call_count[0] == 1:
                            return {"task_id": "test-001", "status": "running"}
                        else:
                            return {
                                "task_id": "test-001",
                                "status": "completed",
                                "result": {"text": "done"}
                            }
                    return {}

                response.json = json
                return response

            session.request = mock_request
            session.close = AsyncMock()

            mock_session_class.return_value = session

            client = LocalModelClient()
            await client.__aenter__()
            yield client
            await client.__aexit__(None, None, None)

    @pytest.mark.unit
    async def test_get_task_status(self, client):
        """Test getting task status."""
        status = await client.get_task_status("test-001")
        assert "task_id" in status
        assert "status" in status

    @pytest.mark.unit
    async def test_wait_for_task(self, client):
        """Test waiting for task completion."""
        result = await client.wait_for_task("test-001", timeout=5.0, poll_interval=0.1)
        assert "status" in result
        assert result["status"] == "completed"


# =============================================================================
# Test Batch Operations
# =============================================================================

class TestBatchOperationsClient:
    """Test client batch operation methods."""

    @pytest.fixture
    async def client(self):
        """Create client with mocked session."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            client_instance = LocalModelClient()

            # Mock generate_text_with_wait
            async def mock_generate(prompt, **kwargs):
                return {"text": f"Response to: {prompt[:20]}"}

            client_instance.generate_text_with_wait = mock_generate

            await client_instance.__aenter__()
            yield client_instance
            await client_instance.__aexit__(None, None, None)

    @pytest.mark.unit
    async def test_batch_generate(self):
        """Test batch generation."""
        with patch('aiohttp.ClientSession'):
            client = LocalModelClient()
            await client.__aenter__()

            # Mock the method
            client.generate_text_with_wait = AsyncMock(
                side_effect=[
                    {"text": "response 1"},
                    {"text": "response 2"},
                    {"text": "response 3"}
                ]
            )

            prompts = ["prompt 1", "prompt 2", "prompt 3"]
            results = await client.batch_generate(prompts)

            assert len(results) == 3

            await client.__aexit__(None, None, None)


# =============================================================================
# Test Error Handling
# =============================================================================

class TestClientErrors:
    """Test client error handling."""

    @pytest.mark.unit
    async def test_request_without_session(self):
        """Test that request fails without session."""
        client = LocalModelClient()

        with pytest.raises(RuntimeError, match="session not initialized"):
            await client._request("GET", "/status")

    @pytest.mark.unit
    async def test_api_error_response(self):
        """Test handling API error responses."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            async def mock_request(method, endpoint, **kwargs):
                response = MagicMock()
                response.status = 500

                async def text():
                    return "Internal Server Error"

                response.text = text
                return response

            session.request = mock_request

            mock_session_class.return_value = session

            client = LocalModelClient()
            await client.__aenter__()

            with pytest.raises(Exception):
                await client._request("GET", "/status")

            await client.__aexit__(None, None, None)


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test client convenience functions."""

    @pytest.mark.unit
    async def test_quick_generate(self):
        """Test quick_generate convenience function."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            from local_model_manager.api.client import quick_generate

            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            async def mock_request(method, endpoint, **kwargs):
                response = MagicMock()
                response.status = 200
                response.json = AsyncMock(return_value={"text": "quick response"})
                return response

            session.request = mock_request

            mock_session_class.return_value = session

            result = await quick_generate("Test prompt")
            assert isinstance(result, str)

    @pytest.mark.unit
    async def test_quick_status(self):
        """Test quick_status convenience function."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            from local_model_manager.api.client import quick_status

            session = MagicMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock()

            async def mock_request(method, endpoint, **kwargs):
                response = MagicMock()
                response.status = 200
                response.json = AsyncMock(return_value={"loaded_models": ["model-1"]})
                return response

            session.request = mock_request

            mock_session_class.return_value = session

            status = await quick_status()
            assert "loaded_models" in status
