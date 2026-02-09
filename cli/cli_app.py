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

        # Check Ollama connectivity
        from agent.models import OllamaClient
        client = OllamaClient(self.config.chat_model.base_url)
        if not await client.health_check():
            print(f"{RED}[Error] Cannot connect to Ollama at {self.config.chat_model.base_url}")
            print(f"Make sure Ollama is running: ollama serve{RESET}")
            return

        models = await client.list_models()
        model_names = [m.get("name", "?") for m in models]
        print(f"{DIM}Available models: {', '.join(model_names) if model_names else 'none'}{RESET}")

        chat_model = self.config.chat_model.model_name
        if not any(chat_model in m for m in model_names):
            print(f"{YELLOW}[Warning] Chat model '{chat_model}' not found locally.")
            print(f"Pull it with: ollama pull {chat_model}{RESET}")

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

    def _new_session(self):
        """Create a fresh session."""
        self.context = AgentContext(self.config)
        self.context.on_stream = self._stream_handler

        # Initialize memory manager
        mm = MemoryManager(self.config)
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
  {CYAN}/help{RESET}   — Show this help
  {CYAN}/exit{RESET}   — Quit

{BOLD}How it works:{RESET}
  Agent 0 receives your message and enters a monologue loop.
  It reasons, uses tools (code execution, memory, etc.), and
  can delegate to subordinate agents. The loop ends when it
  calls the 'response' tool with its final answer.
""")
