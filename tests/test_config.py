import json
import tempfile
import unittest
from pathlib import Path

from agent.config import load_config
from agent.exceptions import ConfigError


class TestConfig(unittest.TestCase):
    def test_load_config_ollama_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data_dir = Path(tmpdir) / "data"
            config_path.write_text(json.dumps({"data_dir": str(data_dir)}))

            config = load_config(str(config_path))

            self.assertEqual(config.ollama.connect_timeout, 5.0)
            self.assertEqual(config.ollama.read_timeout, 120.0)
            self.assertEqual(config.ollama.max_retries, 3)
            self.assertTrue(config.ollama.health_check_on_start)
            self.assertFalse(config.session.persist)
            self.assertEqual(config.session.storage_path, "./data/sessions.db")
            self.assertTrue(config.session.auto_save)
            self.assertEqual(config.session.max_sessions, 100)
            self.assertFalse(config.model_router.enabled)
            self.assertEqual(config.model_router.routes.get("default"), "llama3.2")
            self.assertEqual(config.tool_execution.default_timeout, 30.0)
            self.assertEqual(config.tool_execution.timeouts, {})
            self.assertTrue(config.tool_execution.cache_enabled)
            self.assertFalse(config.memory.importance_scoring)
            self.assertIsNone(config.memory.ttl_days)
            self.assertFalse(config.memory.compaction_enabled)
            self.assertEqual(config.memory.compaction_threshold, 1000)
            self.assertEqual(config.memory.namespaces, ["default"])
            self.assertEqual(config.memory.embedding_provider, "ollama")
            self.assertEqual(config.rag.chunk_strategy, "fixed")
            self.assertEqual(config.rag.chunk_size, 500)
            self.assertEqual(config.rag.chunk_overlap, 50)
            self.assertFalse(config.rag.rerank_enabled)
            self.assertFalse(config.rag.hybrid_search)
            self.assertFalse(config.mcp.enabled)
            self.assertEqual(config.mcp.servers, [])
            self.assertEqual(config.extensions.enabled_map, {})
            self.assertFalse(config.auth.enabled)
            self.assertEqual(config.auth.username, "admin")
            self.assertEqual(config.auth.password_hash, "")
            self.assertEqual(config.auth.api_key, "")
            self.assertFalse(config.telemetry.enabled)
            self.assertEqual(
                config.telemetry.log_dir,
                str(data_dir / "metrics"),
            )
            self.assertFalse(config.telemetry.otel_enabled)
            self.assertIsNone(config.telemetry.otel_endpoint)
            self.assertEqual(config.telemetry.otel_service_name, "local-ollama-agents")

    def test_invalid_ollama_timeout_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data_dir = Path(tmpdir) / "data"
            config_path.write_text(json.dumps({
                "data_dir": str(data_dir),
                "ollama": {"connect_timeout": -1},
            }))

            with self.assertRaises(ConfigError):
                load_config(str(config_path))

    def test_env_overrides_ollama_base_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            data_dir = Path(tmpdir) / "data"
            config_path.write_text(json.dumps({"data_dir": str(data_dir)}))

            import os
            original = os.environ.get("OLLAMA_BASE_URL")
            os.environ["OLLAMA_BASE_URL"] = "http://ollama:11434"
            try:
                config = load_config(str(config_path))
                self.assertEqual(config.chat_model.base_url, "http://ollama:11434")
                self.assertEqual(config.utility_model.base_url, "http://ollama:11434")
            finally:
                if original is None:
                    os.environ.pop("OLLAMA_BASE_URL", None)
                else:
                    os.environ["OLLAMA_BASE_URL"] = original
