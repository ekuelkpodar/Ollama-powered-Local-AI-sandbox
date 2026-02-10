"""OllamaClient - Direct HTTP communication with the Ollama REST API."""

import asyncio
import json
from typing import AsyncIterator, Awaitable, Callable, Iterable, TypeVar

import aiohttp
from agent.exceptions import OllamaConnectionError, OllamaModelError


T = TypeVar("T")


class OllamaClient:
    """Direct async HTTP client for the Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max_retries

    async def health_check(self) -> bool:
        """Check if Ollama is running. GET /api/tags"""
        try:
            async def _request() -> bool:
                async with aiohttp.ClientSession(timeout=self._timeout()) as session:
                    async with session.get(
                        f"{self.base_url}/api/tags",
                    ) as resp:
                        return resp.status == 200

            return await self._with_retry("health check", _request)
        except OllamaConnectionError:
            return False

    async def list_models(self) -> list[dict]:
        """List available local models. GET /api/tags"""
        async def _request() -> list[dict]:
            async with aiohttp.ClientSession(timeout=self._timeout()) as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise OllamaConnectionError(
                            f"Failed to list models (HTTP {resp.status}): {body}"
                        )
                    data = await resp.json()
                    return data.get("models", [])

        return await self._with_retry("list models", _request)

    async def get_missing_models(self, required_models: Iterable[str]) -> list[str]:
        """Return a list of required model names that are not available."""
        models = await self.list_models()
        names = [m.get("name", "") for m in models]
        return self.filter_missing_models(required_models, names)

    @staticmethod
    def filter_missing_models(
        required_models: Iterable[str],
        available_models: Iterable[str],
    ) -> list[str]:
        """Filter required models against a list of available model names."""
        available = [m for m in available_models if m]
        missing: list[str] = []
        for required in required_models:
            if not required:
                continue
            if not OllamaClient._model_available(required, available):
                missing.append(required)
        return missing

    @staticmethod
    def _model_available(required: str, available: Iterable[str]) -> bool:
        """Check whether a model name is available, accounting for tags."""
        if required in available:
            return True
        tag_prefix = f"{required}:"
        return any(name.startswith(tag_prefix) for name in available)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = True,
        temperature: float = 0.7,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        """
        Send a chat completion request. POST /api/chat
        Yields text chunks when streaming.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                **(options or {}),
            },
        }

        attempt = 0
        delay = 1.0
        while True:
            received_any = False
            try:
                async with aiohttp.ClientSession(timeout=self._timeout()) as session:
                    async with session.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    ) as resp:
                        if resp.status == 404:
                            raise OllamaModelError(
                                f"Model '{model}' not found. Pull it with: ollama pull {model}"
                            )
                        if resp.status != 200:
                            body = await resp.text()
                            raise OllamaConnectionError(
                                f"Ollama chat failed (HTTP {resp.status}): {body}"
                            )

                        if stream:
                            async for line in resp.content:
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if "message" in data and "content" in data["message"]:
                                    received_any = True
                                    yield data["message"]["content"]
                                if data.get("done", False):
                                    return
                        else:
                            data = await resp.json()
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                attempt += 1
                if received_any or attempt >= self.max_retries:
                    raise OllamaConnectionError(
                        self._connection_error_message("chat", e)
                    ) from e
                await asyncio.sleep(delay)
                delay *= 2

    async def chat_full(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        options: dict | None = None,
    ) -> str:
        """Non-streaming chat convenience. Returns the full response text."""
        chunks = []
        async for chunk in self.chat(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            options=options,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.3,
        options: dict | None = None,
    ) -> str:
        """
        One-shot text generation. POST /api/generate
        Used for utility tasks (summarization, etc.).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                **(options or {}),
            },
        }

        async def _request() -> str:
            async with aiohttp.ClientSession(timeout=self._timeout()) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as resp:
                    if resp.status == 404:
                        raise OllamaModelError(
                            f"Model '{model}' not found. Pull it with: ollama pull {model}"
                        )
                    if resp.status != 200:
                        body = await resp.text()
                        raise OllamaConnectionError(
                            f"Ollama generate failed (HTTP {resp.status}): {body}"
                        )
                    data = await resp.json()
                    return data.get("response", "")

        return await self._with_retry("generate", _request)

    def _timeout(self) -> aiohttp.ClientTimeout:
        """Build a client timeout configuration from settings."""
        return aiohttp.ClientTimeout(
            total=None,
            connect=self.connect_timeout,
            sock_connect=self.connect_timeout,
            sock_read=self.read_timeout,
        )

    async def _with_retry(self, operation: str, func: Callable[[], Awaitable[T]]) -> T:
        """Run an async operation with exponential backoff retries."""
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await func()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(delay)
                delay *= 2

        raise OllamaConnectionError(self._connection_error_message(operation, last_error))

    def _connection_error_message(self, operation: str, error: Exception | None) -> str:
        """Create a user-friendly connection error message."""
        details = f"{error}" if error else "unknown error"
        return (
            f"Cannot connect to Ollama at {self.base_url} during {operation} "
            f"(after {self.max_retries} attempt(s)): {details}"
        )
