"""Response and ToolCall dataclasses."""

from dataclasses import dataclass, field


@dataclass
class Response:
    """Result from a tool execution or agent monologue."""
    message: str
    break_loop: bool = False


@dataclass
class ToolCall:
    """Parsed tool invocation from LLM output."""
    name: str
    args: dict = field(default_factory=dict)
