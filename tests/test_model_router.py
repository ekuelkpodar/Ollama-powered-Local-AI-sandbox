import tempfile
import unittest
from pathlib import Path

from agent.config import AgentConfig, ModelConfig, ModelRouterConfig
from agent.model_router import ModelRouter
from agent.models import OllamaClient


class TestModelRouter(unittest.TestCase):
    def _make_config(self, log_dir: Path, enabled: bool = True) -> AgentConfig:
        router = ModelRouterConfig(
            enabled=enabled,
            routes={
                "reasoning": "base-model",
                "coding": "code-model",
                "summarization": "sum-model",
                "default": "base-model",
            },
            tool_affinity={"code_execution": "coding"},
        )
        return AgentConfig(
            chat_model=ModelConfig(model_name="base-model"),
            model_router=router,
            log_dir=str(log_dir),
        )

    def test_selects_coding_for_code_like_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(Path(tmpdir))
            router = ModelRouter(config, OllamaClient())
            router.set_available_models(["base-model", "code-model"])

            messages = [{"role": "user", "content": "Please fix this:\n```python\nprint('hi')\n```"}]
            model = router.select_model(messages)
            self.assertEqual(model, "code-model")

    def test_selects_summary_for_summary_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(Path(tmpdir))
            router = ModelRouter(config, OllamaClient())
            router.set_available_models(["base-model", "sum-model"])

            messages = [{"role": "user", "content": "Summarize the following text."}]
            model = router.select_model(messages)
            self.assertEqual(model, "sum-model")

    def test_tool_affinity_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(Path(tmpdir))
            router = ModelRouter(config, OllamaClient())
            router.set_available_models(["base-model", "code-model"])

            messages = [{"role": "user", "content": "Summarize this."}]
            model = router.select_model(messages, last_tool_name="code_execution")
            self.assertEqual(model, "code-model")

    def test_missing_model_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(Path(tmpdir))
            router = ModelRouter(config, OllamaClient())
            router.set_available_models(["base-model"])

            messages = [{"role": "user", "content": "Summarize this."}]
            model = router.select_model(messages)
            self.assertEqual(model, "base-model")

    def test_disabled_router_uses_chat_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(Path(tmpdir), enabled=False)
            router = ModelRouter(config, OllamaClient())

            messages = [{"role": "user", "content": "Summarize this."}]
            model = router.select_model(messages)
            self.assertEqual(model, "base-model")
