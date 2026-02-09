"""AgentContext — session container shared across all agents in a hierarchy."""

from __future__ import annotations
import uuid
from typing import Callable, TYPE_CHECKING
from agent.config import AgentConfig

if TYPE_CHECKING:
    from agent.agent import Agent


class AgentContext:
    """
    A conversational session. One per user interaction.
    Shared across all agents in the hierarchy (Agent 0 + subordinates).
    """

    def __init__(self, config: AgentConfig):
        self.id: str = uuid.uuid4().hex[:12]
        self.config = config
        self.agents: dict[int, "Agent"] = {}
        self.extension_manager = None  # Set externally after extensions loaded
        self.on_stream: Callable[["Agent", str], None] | None = None
        self.data: dict = {}  # Shared data store (sessions, etc.)
        self.log: list[dict] = []

    def create_agent(
        self,
        agent_id: int = 0,
        parent: "Agent | None" = None,
        system_prompt: str = "",
    ) -> "Agent":
        """Create and register an agent in this context."""
        from agent.agent import Agent

        agent = Agent(
            agent_id=agent_id,
            context=self,
            config=self.config,
            parent=parent,
            system_prompt_override=system_prompt,
        )
        self.agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: int) -> "Agent | None":
        return self.agents.get(agent_id)
