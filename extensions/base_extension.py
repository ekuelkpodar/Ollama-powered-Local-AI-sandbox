"""Abstract base class for all extensions (lifecycle hooks)."""

from abc import ABC


class Extension(ABC):
    """
    Base class for extensions. Override the hook methods you need.
    Extensions are called in registration order at each lifecycle point.
    """

    name: str = ""
    enabled: bool = True

    def __init__(self, config):
        self.config = config

    async def on_message_loop_start(self, agent, **kwargs):
        pass

    async def on_message_loop_prompts_before(self, agent, **kwargs):
        pass

    async def on_before_llm_call(self, agent, messages, **kwargs):
        """Return modified messages list, or None to keep original."""
        return None

    async def on_after_llm_call(self, agent, response, **kwargs):
        pass

    async def on_tool_execute_before(self, agent, tool_call, **kwargs):
        pass

    async def on_tool_execute_after(self, agent, tool_call, response, **kwargs):
        pass

    async def on_monologue_end(self, agent, final_response, **kwargs):
        pass
