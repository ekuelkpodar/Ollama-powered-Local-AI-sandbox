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
class OllamaSettings:
    """Configuration for Ollama connectivity and retries."""
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    max_retries: int = 3
    health_check_on_start: bool = True


@dataclass
class SessionConfig:
    """Configuration for session persistence."""
    persist: bool = False
    storage_path: str = "./data/sessions.db"
    auto_save: bool = True
    max_sessions: int = 100


@dataclass
class ModelRouterConfig:
    """Configuration for routing tasks to specialized models."""
    enabled: bool = False
    routes: dict[str, str] = field(default_factory=lambda: {
        "reasoning": "llama3.2",
        "coding": "codellama",
        "summarization": "phi3:mini",
        "default": "llama3.2",
    })
    tool_affinity: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolExecutionConfig:
    """Configuration for tool execution behavior."""
    default_timeout: float = 30.0
    timeouts: dict[str, float] = field(default_factory=dict)
    cache_enabled: bool = True


@dataclass
class MemoryConfig:
    """Configuration for memory system behavior."""
    importance_scoring: bool = False
    ttl_days: int | None = None
    compaction_enabled: bool = False
    compaction_threshold: int = 1000
    namespaces: list[str] = field(default_factory=lambda: ["default"])
    embedding_provider: str = "ollama"  # "local" or "ollama"
    ollama_embedding_model: str = "nomic-embed-text"


@dataclass
class RagConfig:
    """Configuration for RAG chunking and retrieval."""
    chunk_strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 50
    rerank_enabled: bool = False
    hybrid_search: bool = False
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    semantic_threshold: float = 0.75


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    transport: str  # "stdio" or "sse"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    send_url: str | None = None


@dataclass
class MCPConfig:
    """Configuration for MCP tool bridging."""
    enabled: bool = False
    servers: list[MCPServerConfig] = field(default_factory=list)


@dataclass
class ExtensionsConfig:
    """Configuration for extension toggles."""
    enabled_map: dict[str, bool] = field(default_factory=dict)


@dataclass
class AuthConfig:
    """Configuration for web authentication."""
    enabled: bool = False
    username: str = "admin"
    password_hash: str = ""
    api_key: str = ""


@dataclass
class TelemetryConfig:
    """Configuration for telemetry and metrics logging."""
    enabled: bool = False
    log_dir: str = "./data/metrics"
    otel_enabled: bool = False
    otel_endpoint: str | None = None
    otel_service_name: str = "local-ollama-agents"


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    chat_model: ModelConfig = field(default_factory=ModelConfig)
    utility_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        temperature=0.3, ctx_length=4096
    ))
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    session: SessionConfig = field(default_factory=SessionConfig)
    model_router: ModelRouterConfig = field(default_factory=ModelRouterConfig)
    tool_execution: ToolExecutionConfig = field(default_factory=ToolExecutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
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

    data_dir = raw.get("data_dir", "data")

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
    env_base_url = os.getenv("OLLAMA_BASE_URL")
    if env_base_url:
        chat_model.base_url = env_base_url
        utility_model.base_url = env_base_url

    ollama_raw = raw.get("ollama", {})
    ollama = _load_ollama_settings(ollama_raw)

    session_raw = raw.get("session", {})
    session = _load_session_settings(session_raw)

    router_raw = raw.get("model_router", {})
    model_router = _load_model_router_settings(router_raw)

    tool_exec_raw = raw.get("tool_execution", {})
    tool_execution = _load_tool_execution_settings(tool_exec_raw)

    memory_raw = raw.get("memory", {})
    memory = _load_memory_settings(memory_raw)

    rag_raw = raw.get("rag", {})
    rag = _load_rag_settings(rag_raw)

    mcp_raw = raw.get("mcp", {})
    mcp = _load_mcp_settings(mcp_raw)

    extensions_raw = raw.get("extensions", {})
    extensions = _load_extensions_settings(extensions_raw)

    auth_raw = raw.get("auth", {})
    auth = _load_auth_settings(auth_raw)

    telemetry_raw = raw.get("telemetry", {})
    telemetry = _load_telemetry_settings(telemetry_raw, data_dir)

    # Ensure data directories exist
    knowledge_dir = raw.get("knowledge_dir", os.path.join(data_dir, "knowledge"))
    memory_dir = raw.get("memory_dir", os.path.join(data_dir, "memory"))
    log_dir = raw.get("log_dir", os.path.join(data_dir, "logs"))
    telemetry_dir = telemetry.log_dir or os.path.join(data_dir, "metrics")

    for d in [data_dir, knowledge_dir, memory_dir, log_dir, telemetry_dir]:
        os.makedirs(d, exist_ok=True)

    return AgentConfig(
        chat_model=chat_model,
        utility_model=utility_model,
        ollama=ollama,
        session=session,
        model_router=model_router,
        tool_execution=tool_execution,
        memory=memory,
        rag=rag,
        mcp=mcp,
        extensions=extensions,
        auth=auth,
        telemetry=telemetry,
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


def _load_ollama_settings(raw: dict) -> OllamaSettings:
    """Parse and validate Ollama settings from config."""
    connect_timeout = _coerce_float(raw.get("connect_timeout", 5.0), "ollama.connect_timeout", 0.1)
    read_timeout = _coerce_float(raw.get("read_timeout", 120.0), "ollama.read_timeout", 0.1)
    max_retries = _coerce_int(raw.get("max_retries", 3), "ollama.max_retries", 1)

    health_check_on_start = raw.get("health_check_on_start", True)
    if not isinstance(health_check_on_start, bool):
        raise ConfigError("ollama.health_check_on_start must be a boolean")

    return OllamaSettings(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=max_retries,
        health_check_on_start=health_check_on_start,
    )


def _load_session_settings(raw: dict) -> SessionConfig:
    """Parse and validate session persistence settings."""
    persist = raw.get("persist", False)
    if not isinstance(persist, bool):
        raise ConfigError("session.persist must be a boolean")

    auto_save = raw.get("auto_save", True)
    if not isinstance(auto_save, bool):
        raise ConfigError("session.auto_save must be a boolean")

    storage_path = raw.get("storage_path", "./data/sessions.db")
    if not isinstance(storage_path, str) or not storage_path.strip():
        raise ConfigError("session.storage_path must be a non-empty string")

    max_sessions = _coerce_int(raw.get("max_sessions", 100), "session.max_sessions", 1)

    return SessionConfig(
        persist=persist,
        storage_path=storage_path,
        auto_save=auto_save,
        max_sessions=max_sessions,
    )


def _load_model_router_settings(raw: dict) -> ModelRouterConfig:
    """Parse and validate model router settings."""
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("model_router.enabled must be a boolean")

    routes = raw.get("routes", {})
    if routes is None:
        routes = {}
    if not isinstance(routes, dict):
        raise ConfigError("model_router.routes must be an object")
    cleaned_routes: dict[str, str] = {}
    for key, value in routes.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ConfigError("model_router.routes must map strings to strings")
        cleaned_routes[key] = value

    tool_affinity = raw.get("tool_affinity", {})
    if tool_affinity is None:
        tool_affinity = {}
    if not isinstance(tool_affinity, dict):
        raise ConfigError("model_router.tool_affinity must be an object")
    cleaned_affinity: dict[str, str] = {}
    for key, value in tool_affinity.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ConfigError("model_router.tool_affinity must map strings to strings")
        cleaned_affinity[key] = value

    return ModelRouterConfig(
        enabled=enabled,
        routes=cleaned_routes or ModelRouterConfig().routes,
        tool_affinity=cleaned_affinity,
    )


def _load_tool_execution_settings(raw: dict) -> ToolExecutionConfig:
    """Parse and validate tool execution settings."""
    default_timeout = _coerce_float(
        raw.get("default_timeout", 30.0),
        "tool_execution.default_timeout",
        0.1,
    )

    timeouts_raw = raw.get("timeouts", {})
    if timeouts_raw is None:
        timeouts_raw = {}
    if not isinstance(timeouts_raw, dict):
        raise ConfigError("tool_execution.timeouts must be an object")

    timeouts: dict[str, float] = {}
    for key, value in timeouts_raw.items():
        if not isinstance(key, str):
            raise ConfigError("tool_execution.timeouts keys must be strings")
        timeouts[key] = _coerce_float(value, f"tool_execution.timeouts.{key}", 0.1)

    cache_enabled = raw.get("cache_enabled", True)
    if not isinstance(cache_enabled, bool):
        raise ConfigError("tool_execution.cache_enabled must be a boolean")

    return ToolExecutionConfig(
        default_timeout=default_timeout,
        timeouts=timeouts,
        cache_enabled=cache_enabled,
    )


def _load_memory_settings(raw: dict) -> MemoryConfig:
    """Parse and validate memory settings."""
    importance_scoring = raw.get("importance_scoring", False)
    if not isinstance(importance_scoring, bool):
        raise ConfigError("memory.importance_scoring must be a boolean")

    ttl_raw = raw.get("ttl_days", None)
    ttl_days = None
    if ttl_raw is not None:
        ttl_days = _coerce_int(ttl_raw, "memory.ttl_days", 1)

    compaction_enabled = raw.get("compaction_enabled", False)
    if not isinstance(compaction_enabled, bool):
        raise ConfigError("memory.compaction_enabled must be a boolean")

    compaction_threshold = _coerce_int(
        raw.get("compaction_threshold", 1000),
        "memory.compaction_threshold",
        2,
    )

    namespaces_raw = raw.get("namespaces", ["default"])
    if not isinstance(namespaces_raw, list) or not namespaces_raw:
        raise ConfigError("memory.namespaces must be a non-empty list")
    namespaces = []
    for ns in namespaces_raw:
        if not isinstance(ns, str) or not ns.strip():
            raise ConfigError("memory.namespaces must contain non-empty strings")
        namespaces.append(ns.strip())

    embedding_provider = raw.get("embedding_provider", "ollama")
    if embedding_provider not in ("local", "ollama"):
        raise ConfigError("memory.embedding_provider must be 'local' or 'ollama'")

    ollama_embedding_model = raw.get("ollama_embedding_model", "nomic-embed-text")
    if not isinstance(ollama_embedding_model, str) or not ollama_embedding_model.strip():
        raise ConfigError("memory.ollama_embedding_model must be a non-empty string")

    return MemoryConfig(
        importance_scoring=importance_scoring,
        ttl_days=ttl_days,
        compaction_enabled=compaction_enabled,
        compaction_threshold=compaction_threshold,
        namespaces=namespaces,
        embedding_provider=embedding_provider,
        ollama_embedding_model=ollama_embedding_model,
    )


def _load_rag_settings(raw: dict) -> RagConfig:
    """Parse and validate RAG settings."""
    chunk_strategy = raw.get("chunk_strategy", "fixed")
    if chunk_strategy not in ("fixed", "sentence", "semantic"):
        raise ConfigError("rag.chunk_strategy must be 'fixed', 'sentence', or 'semantic'")

    chunk_size = _coerce_int(raw.get("chunk_size", 500), "rag.chunk_size", 1)
    chunk_overlap = _coerce_int(raw.get("chunk_overlap", 50), "rag.chunk_overlap", 0)
    if chunk_overlap >= chunk_size:
        raise ConfigError("rag.chunk_overlap must be less than rag.chunk_size")

    rerank_enabled = raw.get("rerank_enabled", False)
    if not isinstance(rerank_enabled, bool):
        raise ConfigError("rag.rerank_enabled must be a boolean")

    hybrid_search = raw.get("hybrid_search", False)
    if not isinstance(hybrid_search, bool):
        raise ConfigError("rag.hybrid_search must be a boolean")

    bm25_weight = _coerce_float(raw.get("bm25_weight", 0.3), "rag.bm25_weight", 0.0)
    vector_weight = _coerce_float(raw.get("vector_weight", 0.7), "rag.vector_weight", 0.0)

    semantic_threshold = _coerce_float(
        raw.get("semantic_threshold", 0.75),
        "rag.semantic_threshold",
        0.0,
    )

    return RagConfig(
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rerank_enabled=rerank_enabled,
        hybrid_search=hybrid_search,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        semantic_threshold=semantic_threshold,
    )


def _load_mcp_settings(raw: dict) -> MCPConfig:
    """Parse and validate MCP settings."""
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("mcp.enabled must be a boolean")

    servers_raw = raw.get("servers", [])
    if servers_raw is None:
        servers_raw = []
    if not isinstance(servers_raw, list):
        raise ConfigError("mcp.servers must be a list")

    servers: list[MCPServerConfig] = []
    for idx, server in enumerate(servers_raw):
        if not isinstance(server, dict):
            raise ConfigError("mcp.servers entries must be objects")
        name = server.get("name")
        transport = server.get("transport")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"mcp.servers[{idx}].name must be a non-empty string")
        if transport not in ("stdio", "sse"):
            raise ConfigError(f"mcp.servers[{idx}].transport must be 'stdio' or 'sse'")

        command = server.get("command")
        args = server.get("args", [])
        url = server.get("url")
        send_url = server.get("send_url")

        if transport == "stdio":
            if not isinstance(command, str) or not command.strip():
                raise ConfigError(f"mcp.servers[{idx}].command must be a non-empty string for stdio")
            if args is None:
                args = []
            if not isinstance(args, list):
                raise ConfigError(f"mcp.servers[{idx}].args must be a list")
        else:
            if not isinstance(url, str) or not url.strip():
                raise ConfigError(f"mcp.servers[{idx}].url must be a non-empty string for sse")
            if send_url is not None and (not isinstance(send_url, str) or not send_url.strip()):
                raise ConfigError(f"mcp.servers[{idx}].send_url must be a string if provided")

        servers.append(
            MCPServerConfig(
                name=name.strip(),
                transport=transport,
                command=command.strip() if isinstance(command, str) else None,
                args=[str(a) for a in args] if isinstance(args, list) else [],
                url=url.strip() if isinstance(url, str) else None,
                send_url=send_url.strip() if isinstance(send_url, str) else None,
            )
        )

    return MCPConfig(enabled=enabled, servers=servers)


def _load_extensions_settings(raw: dict) -> ExtensionsConfig:
    """Parse and validate extension toggle settings."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("extensions must be an object mapping extension name to boolean")
    enabled_map: dict[str, bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError("extensions keys must be non-empty strings")
        if not isinstance(value, bool):
            raise ConfigError(f"extensions.{key} must be a boolean")
        enabled_map[key.strip()] = value
    return ExtensionsConfig(enabled_map=enabled_map)


def _load_auth_settings(raw: dict) -> AuthConfig:
    """Parse and validate auth settings."""
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("auth.enabled must be a boolean")

    username = raw.get("username", "admin")
    if not isinstance(username, str) or not username.strip():
        raise ConfigError("auth.username must be a non-empty string")

    password_hash = raw.get("password_hash", "")
    if not isinstance(password_hash, str):
        raise ConfigError("auth.password_hash must be a string")

    api_key = raw.get("api_key", "")
    if not isinstance(api_key, str):
        raise ConfigError("auth.api_key must be a string")

    if enabled and not password_hash.strip() and not api_key.strip():
        raise ConfigError("auth.enabled requires password_hash or api_key")

    return AuthConfig(
        enabled=enabled,
        username=username.strip(),
        password_hash=password_hash.strip(),
        api_key=api_key.strip(),
    )


def _load_telemetry_settings(raw: dict, data_dir: str) -> TelemetryConfig:
    """Parse and validate telemetry settings."""
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("telemetry.enabled must be a boolean")

    log_dir = raw.get("log_dir", os.path.join(data_dir, "metrics"))
    if not isinstance(log_dir, str) or not log_dir.strip():
        raise ConfigError("telemetry.log_dir must be a non-empty string")

    otel_enabled = raw.get("otel_enabled", False)
    if not isinstance(otel_enabled, bool):
        raise ConfigError("telemetry.otel_enabled must be a boolean")

    otel_endpoint = raw.get("otel_endpoint")
    if otel_endpoint is not None and (not isinstance(otel_endpoint, str) or not otel_endpoint.strip()):
        raise ConfigError("telemetry.otel_endpoint must be a non-empty string if provided")

    otel_service_name = raw.get("otel_service_name", "local-ollama-agents")
    if not isinstance(otel_service_name, str) or not otel_service_name.strip():
        raise ConfigError("telemetry.otel_service_name must be a non-empty string")

    return TelemetryConfig(
        enabled=enabled,
        log_dir=log_dir,
        otel_enabled=otel_enabled,
        otel_endpoint=otel_endpoint.strip() if isinstance(otel_endpoint, str) else None,
        otel_service_name=otel_service_name.strip(),
    )


def _coerce_float(value: object, name: str, min_value: float) -> float:
    """Coerce config value to float with basic validation."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number")

    if value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}")
    return value


def _coerce_int(value: object, name: str, min_value: int) -> int:
    """Coerce config value to int with basic validation."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be an integer")

    if value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}")
    return value
