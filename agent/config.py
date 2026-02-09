"""Configuration loading and validation."""

import json
import os
from dataclasses import dataclass, field
from agent.exceptions import ConfigError


@dataclass
class ModelConfig:
    """Configuration for a single Ollama model."""
    model_name: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    ctx_length: int = 8192
    options: dict = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    chat_model: ModelConfig = field(default_factory=ModelConfig)
    utility_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        temperature=0.3, ctx_length=4096
    ))
    embedding_model: str = "all-MiniLM-L6-v2"
    prompt_profile: str = "default"
    max_monologue_iterations: int = 25
    memory_recall_enabled: bool = True
    memory_recall_threshold: float = 0.6
    memory_recall_count: int = 5
    data_dir: str = "data"
    knowledge_dir: str = "data/knowledge"
    memory_dir: str = "data/memory"
    log_dir: str = "data/logs"


def load_config(config_path: str = "config.json") -> AgentConfig:
    """Load configuration from JSON file with defaults."""
    if not os.path.exists(config_path):
        return AgentConfig()

    try:
        with open(config_path, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ConfigError(f"Failed to load config from {config_path}: {e}")

    chat_raw = raw.get("chat_model", {})
    chat_model = ModelConfig(
        model_name=chat_raw.get("model_name", "llama3.2"),
        base_url=chat_raw.get("base_url", "http://localhost:11434"),
        temperature=chat_raw.get("temperature", 0.7),
        ctx_length=chat_raw.get("ctx_length", 8192),
        options=chat_raw.get("options", {}),
    )

    util_raw = raw.get("utility_model", {})
    utility_model = ModelConfig(
        model_name=util_raw.get("model_name", "llama3.2"),
        base_url=util_raw.get("base_url", "http://localhost:11434"),
        temperature=util_raw.get("temperature", 0.3),
        ctx_length=util_raw.get("ctx_length", 4096),
        options=util_raw.get("options", {}),
    )

    # Ensure data directories exist
    data_dir = raw.get("data_dir", "data")
    knowledge_dir = raw.get("knowledge_dir", os.path.join(data_dir, "knowledge"))
    memory_dir = raw.get("memory_dir", os.path.join(data_dir, "memory"))
    log_dir = raw.get("log_dir", os.path.join(data_dir, "logs"))

    for d in [data_dir, knowledge_dir, memory_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    return AgentConfig(
        chat_model=chat_model,
        utility_model=utility_model,
        embedding_model=raw.get("embedding_model", "all-MiniLM-L6-v2"),
        prompt_profile=raw.get("prompt_profile", "default"),
        max_monologue_iterations=raw.get("max_monologue_iterations", 25),
        memory_recall_enabled=raw.get("memory_recall_enabled", True),
        memory_recall_threshold=raw.get("memory_recall_threshold", 0.6),
        memory_recall_count=raw.get("memory_recall_count", 5),
        data_dir=data_dir,
        knowledge_dir=knowledge_dir,
        memory_dir=memory_dir,
        log_dir=log_dir,
    )
