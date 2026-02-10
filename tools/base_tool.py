"""Abstract base class for all tools."""

from abc import ABC, abstractmethod
from agent.response import Response


class Tool(ABC):
    """Base class for all agent tools. Subclass this to create new tools."""

    name: str = ""
    description: str = ""
    arg_schema: dict[str, type | tuple[type, ...]] = {}
    required_args: list[str] = []
    timeout_seconds: float | None = None
    cacheable: bool = True
    parallel_safe: bool = True

    def __init__(self, agent):
        self.agent = agent

    async def before_execution(self, **kwargs):
        """Hook called before execute. Override to validate/modify args."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Response:
        """Execute the tool. Must be implemented by subclasses."""
        ...

    async def after_execution(self, response: Response) -> Response:
        """Hook called after execute. Override to modify response."""
        return response

    def should_cache(self, **kwargs) -> bool:
        """Whether this tool call should be cached."""
        return self.cacheable

    def is_parallel_safe(self, **kwargs) -> bool:
        """Whether this tool call can be executed in parallel."""
        return self.parallel_safe

    def get_prompt_description(self) -> str:
        """Return the prompt description for this tool. Loaded from prompts/ if available."""
        engine = getattr(self.agent, "_prompt_engine", None)
        if engine:
            prompt = engine.get_tool_prompt(self.name)
            if prompt:
                return prompt
        return f"### {self.name}\n{self.description}\n"
