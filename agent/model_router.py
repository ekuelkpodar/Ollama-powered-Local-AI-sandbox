"""Model router for selecting task-appropriate Ollama models."""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable

from agent.config import AgentConfig
from agent.models import OllamaClient


class ModelRouter:
    """Route tasks to specialized models based on heuristic intent."""

    def __init__(self, config: AgentConfig, client: OllamaClient):
        self.config = config
        self._client = client
        self._available_models: list[str] | None = None
        self._logger = self._build_logger(config.log_dir)

        routes = config.model_router.routes
        self._default_model = routes.get("default") or config.chat_model.model_name

    async def ensure_models_loaded(self) -> None:
        """Ensure available model list is loaded."""
        if not self.config.model_router.enabled:
            return
        if self._available_models is not None:
            return
        await self.refresh_models()

    async def refresh_models(self) -> None:
        """Refresh available models from Ollama."""
        if not self.config.model_router.enabled:
            return
        try:
            models = await self._client.list_models()
            self._available_models = [m.get("name", "") for m in models if m.get("name")]
        except Exception as exc:
            self._available_models = None
            self._logger.warning("Model router failed to refresh models: %s", exc)

    def set_available_models(self, models: Iterable[str]) -> None:
        """Set available models (useful for tests)."""
        self._available_models = [m for m in models if m]

    def select_model(self, messages: list[dict], last_tool_name: str | None = None) -> str:
        """Select the best model for the current task."""
        if not self.config.model_router.enabled:
            return self.config.chat_model.model_name

        route_key = self._route_from_tool(last_tool_name)
        if route_key is None:
            route_key = self._route_from_messages(messages)

        model = self._model_for_route(route_key)
        if self._is_available(model):
            return model

        fallback = self._default_model
        if self._is_available(fallback):
            return fallback

        return self.config.chat_model.model_name

    def _route_from_tool(self, tool_name: str | None) -> str | None:
        if not tool_name:
            return None
        affinity = self.config.model_router.tool_affinity
        return affinity.get(tool_name)

    def _route_from_messages(self, messages: list[dict]) -> str:
        text = self._last_user_message(messages)
        if not text:
            return "default"

        if self._looks_like_code(text):
            return "coding"
        if self._looks_like_summary(text):
            return "summarization"
        return "reasoning"

    def _model_for_route(self, route_key: str | None) -> str:
        routes = self.config.model_router.routes
        if not route_key:
            return self._default_model
        return routes.get(route_key, self._default_model)

    def _is_available(self, model: str) -> bool:
        if not model:
            return False
        if self._available_models is None:
            return True
        missing = OllamaClient.filter_missing_models([model], self._available_models)
        return not missing

    @staticmethod
    def _last_user_message(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        if "```" in text:
            return True
        code_keywords = [
            r"\bdef\b", r"\bclass\b", r"\bimport\b", r"\bfunction\b",
            r"\bconst\b", r"\blet\b", r"\bvar\b", r"\breturn\b",
            r"\basync\b", r"\bawait\b", r"\bpublic\b", r"\bstatic\b",
        ]
        if any(re.search(pattern, text) for pattern in code_keywords):
            return True
        file_exts = r"\.(py|js|ts|java|go|rs|cpp|c|cs|rb|php|sh)\b"
        return re.search(file_exts, text) is not None

    @staticmethod
    def _looks_like_summary(text: str) -> bool:
        summary_keywords = [
            r"\bsummarize\b", r"\bsummary\b", r"\btl;dr\b", r"\bcondense\b",
            r"\bbrief\b", r"\boverview\b", r"\bhigh-level\b", r"\bkey points\b",
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in summary_keywords)

    def _build_logger(self, log_dir: str) -> logging.Logger:
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f"model_router.{id(self)}")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_dir, "model_router.log")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
