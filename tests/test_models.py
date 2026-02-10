import unittest
from unittest import mock

import aiohttp

from agent.exceptions import OllamaConnectionError
from agent.models import OllamaClient


class TestOllamaClient(unittest.IsolatedAsyncioTestCase):
    async def test_with_retry_succeeds_after_failures(self):
        client = OllamaClient(max_retries=3)
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise aiohttp.ClientError("boom")
            return "ok"

        with mock.patch("agent.models.asyncio.sleep", new=mock.AsyncMock()) as sleep_mock:
            result = await client._with_retry("test", operation)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    async def test_with_retry_raises_after_exhausted(self):
        client = OllamaClient(max_retries=2)

        async def operation():
            raise aiohttp.ClientError("boom")

        with mock.patch("agent.models.asyncio.sleep", new=mock.AsyncMock()):
            with self.assertRaises(OllamaConnectionError):
                await client._with_retry("test", operation)

    def test_filter_missing_models_prefix_match(self):
        available = ["llama3.2:latest", "phi3:mini"]
        missing = OllamaClient.filter_missing_models(["llama3.2", "codellama"], available)
        self.assertEqual(missing, ["codellama"])
