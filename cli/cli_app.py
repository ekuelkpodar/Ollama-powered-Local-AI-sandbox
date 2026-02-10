"""Interactive CLI for Local Ollama Agents."""

import asyncio
import sys
from agent.config import AgentConfig
from agent.agent_context import AgentContext
from memory.memory_manager import MemoryManager
from extensions.extension_manager import ExtensionManager


# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"

AGENT_COLORS = [GREEN, CYAN, MAGENTA, YELLOW]


def agent_color(agent_id: int) -> str:
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


class CLIApp:
    """Interactive REPL with streaming output."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.context: AgentContext | None = None
        self._current_agent_id: int = -1

    async def run(self):
        """Main REPL loop."""
        self._print_banner()

        if not await self._preflight_ollama():
            return

        print()
        self._new_session()

        while True:
            try:
                user_input = input(f"{BOLD}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}Goodbye!{RESET}")
                break

            if not user_input:
                continue

            # Commands
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                print(f"{DIM}Goodbye!{RESET}")
                break
            if user_input.lower() in ("reset", "/reset", "/new"):
                self._new_session()
                print(f"{DIM}[Session reset]{RESET}")
                continue
            if user_input.lower() in ("sessions", "/sessions"):
                self._list_sessions()
                continue
            if user_input.lower().startswith(("/resume", "resume")):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"{YELLOW}[Usage] /resume <session_id>{RESET}")
                    continue
                if self._resume_session(parts[1].strip()):
                    print(f"{DIM}[Resumed session {parts[1].strip()}]{RESET}")
                continue
            if user_input.lower().startswith(("/delete", "delete")):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"{YELLOW}[Usage] /delete <session_id>{RESET}")
                    continue
                if self._delete_session(parts[1].strip()):
                    print(f"{DIM}[Deleted session {parts[1].strip()}]{RESET}")
                continue
            if user_input.lower() in ("help", "/help"):
                self._print_help()
                continue

            # Run agent
            self._current_agent_id = -1
            agent = self.context.get_agent(0)
            print()

            try:
                result = await agent.monologue(user_input)
            except Exception as e:
                print(f"\n{RED}[Error: {e}]{RESET}")
                continue

            # Print final response (if not already streamed as part of the response tool)
            print(f"\n{BOLD}{GREEN}Agent 0:{RESET} {result}")
            print()

    def _stream_handler(self, agent, chunk: str):
        """Stream callback — print chunks with agent ID prefix."""
        if agent.agent_id != self._current_agent_id:
            self._current_agent_id = agent.agent_id
            color = agent_color(agent.agent_id)
            prefix = f"\n{color}{DIM}[Agent {agent.agent_id}]{RESET} "
            sys.stdout.write(prefix)
        sys.stdout.write(f"{DIM}{chunk}{RESET}")
        sys.stdout.flush()

    def _new_session(self, session_id: str | None = None):
        """Create a fresh session."""
        self.context = AgentContext(self.config, session_id=session_id)
        self.context.on_stream = self._stream_handler

        # Initialize memory manager
        mm = MemoryManager(self.config, telemetry=self.context.telemetry)
        self.context.data["memory_manager"] = mm

        # Initialize extensions
        ext_mgr = ExtensionManager(self.config)
        ext_mgr.discover_extensions()
        self.context.extension_manager = ext_mgr

        self.context.create_agent(agent_id=0)

    def _print_banner(self):
        print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════╗
║      Local Ollama Agents v0.1.0      ║
║   Multi-Agent AI Framework (Local)   ║
╚══════════════════════════════════════╝{RESET}
{DIM}Chat model: {self.config.chat_model.model_name}
Utility model: {self.config.utility_model.model_name}
Ollama: {self.config.chat_model.base_url}{RESET}
""")

    def _print_help(self):
        print(f"""
{BOLD}Commands:{RESET}
  {CYAN}/reset{RESET}  — Start a new session
  {CYAN}/sessions{RESET} — List saved sessions
  {CYAN}/resume{RESET} <id> — Resume a saved session
  {CYAN}/delete{RESET} <id> — Delete a saved session
  {CYAN}/help{RESET}   — Show this help
  {CYAN}/exit{RESET}   — Quit

{BOLD}How it works:{RESET}
  Agent 0 receives your message and enters a monologue loop.
  It reasons, uses tools (code execution, memory, etc.), and
  can delegate to subordinate agents. The loop ends when it
  calls the 'response' tool with its final answer.
""")

    def _list_sessions(self):
        """List persisted sessions."""
        store = self._get_session_store()
        if store is None:
            print(f"{YELLOW}[Session persistence is disabled]{RESET}")
            return
        sessions = store.list_sessions()
        if not sessions:
            print(f"{DIM}[No saved sessions]{RESET}")
            return

        print(f"{DIM}Saved sessions:{RESET}")
        for s in sessions:
            title = s.get("title") or "Untitled"
            print(
                f"{DIM}- {s['session_id']} | {title} | msgs: {s['message_count']} | "
                f"updated: {s['updated_at']} | model: {s['model']}{RESET}"
            )

    def _resume_session(self, session_id: str) -> bool:
        """Resume a saved session by ID."""
        store = self._get_session_store()
        if store is None:
            print(f"{YELLOW}[Session persistence is disabled]{RESET}")
            return False
        session_data = store.get_session(session_id)
        if not session_data:
            print(f"{RED}[Error] Session '{session_id}' not found{RESET}")
            return False
        self._new_session(session_id=session_id)
        return True

    def _delete_session(self, session_id: str) -> bool:
        """Delete a saved session by ID."""
        store = self._get_session_store()
        if store is None:
            print(f"{YELLOW}[Session persistence is disabled]{RESET}")
            return False
        deleted = store.delete_session(session_id)
        if not deleted:
            print(f"{RED}[Error] Session '{session_id}' not found{RESET}")
            return False
        if self.context and self.context.id == session_id:
            self._new_session()
        return True

    def _get_session_store(self):
        if not self.config.session.persist:
            return None
        from agent.session_store import SessionStore
        return SessionStore(
            db_path=self.config.session.storage_path,
            max_sessions=self.config.session.max_sessions,
        )

    async def _preflight_ollama(self) -> bool:
        """Check Ollama connectivity and model availability before starting."""
        from agent.models import OllamaClient

        base_url_map: dict[str, list[str]] = {}

        def _add_model(base_url: str, model_name: str) -> None:
            models = base_url_map.setdefault(base_url, [])
            if model_name not in models:
                models.append(model_name)

        _add_model(self.config.chat_model.base_url, self.config.chat_model.model_name)
        _add_model(self.config.utility_model.base_url, self.config.utility_model.model_name)

        for base_url, required_models in base_url_map.items():
            client = OllamaClient(
                base_url=base_url,
                connect_timeout=self.config.ollama.connect_timeout,
                read_timeout=self.config.ollama.read_timeout,
                max_retries=self.config.ollama.max_retries,
            )

            if self.config.ollama.health_check_on_start:
                healthy = await client.health_check()
                if not healthy:
                    print(f"{RED}[Error] Cannot connect to Ollama at {base_url}{RESET}")
                    print(f"{DIM}Make sure Ollama is running: ollama serve{RESET}")
                    return False

            try:
                models = await client.list_models()
            except Exception as e:
                print(f"{RED}[Error] {e}{RESET}")
                return False

            model_names = [m.get("name", "?") for m in models]
            label = f"Available models" if base_url == self.config.chat_model.base_url else f"Available models ({base_url})"
            print(f"{DIM}{label}: {', '.join(model_names) if model_names else 'none'}{RESET}")

            missing = OllamaClient.filter_missing_models(required_models, model_names)
            if missing:
                missing_list = ", ".join(missing)
                print(f"{RED}[Error] Missing models at {base_url}: {missing_list}{RESET}")
                print(f"{DIM}Pull with: ollama pull <model>{RESET}")
                return False

        return True
