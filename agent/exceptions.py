"""Custom exceptions for Local Ollama Agents."""


class OllamaConnectionError(Exception):
    """Raised when unable to connect to the Ollama server."""
    pass


class OllamaModelError(Exception):
    """Raised when the requested model is not available."""
    pass


class ToolExecutionError(Exception):
    """Raised when a tool fails during execution."""
    pass


class ToolNotFoundError(Exception):
    """Raised when a requested tool does not exist."""
    pass


class PromptTemplateError(Exception):
    """Raised when a prompt template fails to render."""
    pass


class MemoryError(Exception):
    """Raised when a memory operation fails."""
    pass


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class MaxIterationsError(Exception):
    """Raised when the monologue loop exceeds max iterations."""
    pass
