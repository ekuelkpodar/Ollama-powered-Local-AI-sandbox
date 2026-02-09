"""OllamaClient - Direct HTTP communication with the Ollama REST API."""

import json
import aiohttp
from typing import AsyncIterator
from agent.exceptions import OllamaConnectionError, OllamaModelError


class OllamaClient:
    """Direct async HTTP client for the Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def health_check(self) -> bool:
        """Check if Ollama is running. GET /"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False

    async def list_models(self) -> list[dict]:
        """List available local models. GET /api/tags"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        raise OllamaConnectionError(f"Failed to list models: HTTP {resp.status}")
                    data = await resp.json()
                    return data.get("models", [])
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")

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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300, sock_read=120),
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
                                yield data["message"]["content"]
                            if data.get("done", False):
                                return
                    else:
                        data = await resp.json()
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]

        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")

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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
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

        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")
