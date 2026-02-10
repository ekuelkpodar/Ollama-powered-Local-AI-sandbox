"""AgentContext — session container shared across all agents in a hierarchy."""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Callable, TYPE_CHECKING
from agent.config import AgentConfig
from agent.session_store import SessionStore
from agent.telemetry import Telemetry

if TYPE_CHECKING:
    from agent.agent import Agent


class AgentContext:
    """
    A conversational session. One per user interaction.
    Shared across all agents in the hierarchy (Agent 0 + subordinates).
    """

    def __init__(self, config: AgentConfig, session_id: str | None = None):
        self.id: str = session_id or uuid.uuid4().hex[:12]
        self.config = config
        self.agents: dict[int, "Agent"] = {}
        self.extension_manager = None  # Set externally after extensions loaded
        self.on_stream: Callable[["Agent", str], None] | None = None
        self.data: dict = {}  # Shared data store (sessions, etc.)
        self.log: list[dict] = []
        self.session_store: SessionStore | None = None
        self.telemetry: Telemetry = Telemetry(config.telemetry, self.id)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.data["session_created_at"] = now
        self.data["session_updated_at"] = now
        self.data["tool_calls"] = []

        if self.config.session.persist:
            self.session_store = SessionStore(
                db_path=self.config.session.storage_path,
                max_sessions=self.config.session.max_sessions,
            )
            self.session_store.ensure_session(self.id, self.config.chat_model.model_name)

            if session_id:
                session_data = self.session_store.get_session(session_id)
                if session_data:
                    self.data["session_history"] = session_data["messages"]
                    self.data["session_tool_calls"] = session_data["tool_calls"]
                    self.data["tool_calls"] = session_data["tool_calls"]
                    if session_data.get("title"):
                        self.data["session_title"] = session_data.get("title")
                    if session_data.get("created_at"):
                        self.data["session_created_at"] = session_data.get("created_at")
                    if session_data.get("updated_at"):
                        self.data["session_updated_at"] = session_data.get("updated_at")

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
        if agent_id == 0 and "session_history" in self.data:
            stored_history = self.data.get("session_history", [])
            agent.history = [
                {"role": msg.get("role", "system"), "content": msg.get("content", "")}
                for msg in stored_history
            ]
        self.agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: int) -> "Agent | None":
        return self.agents.get(agent_id)
