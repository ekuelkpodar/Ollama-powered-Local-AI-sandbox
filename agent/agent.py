"""Agent class — the core actor with the monologue loop."""

from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agent.config import AgentConfig
from agent.models import OllamaClient
from agent.response import Response, ToolCall
from tools.tool_registry import ToolRegistry
from prompts.template_engine import PromptTemplateEngine

if TYPE_CHECKING:
    from agent.agent_context import AgentContext


class Agent:
    """
    Core agent with the monologue loop.
    Reasons, calls tools, delegates to subordinates, and delivers responses.
    """

    def __init__(
        self,
        agent_id: int,
        context: "AgentContext",
        config: AgentConfig,
        parent: "Agent | None" = None,
        system_prompt_override: str = "",
    ):
        self.agent_id = agent_id
        self.context = context
        self.config = config
        self.parent = parent
        self.system_prompt_override = system_prompt_override
        self.history: list[dict] = []
        self.data: dict = {}

        # Ollama client
        self._client = OllamaClient(base_url=config.chat_model.base_url)

        # Prompt template engine
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
        self._prompt_engine = PromptTemplateEngine(prompts_dir, config.prompt_profile)

        # Tool registry
        self.tool_registry = ToolRegistry(self)
        self.tool_registry.discover_tools()

    # ── Monologue loop ───────────────────────────────────────────────

    async def monologue(self, user_message: str) -> str:
        """
        The core infinite loop. This drives all agent behavior:
        reason → tool extraction → tool execution → loop or break.
        """
        self.append_message("user", user_message)

        # Fire hook: message_loop_start
        await self._dispatch_hook("message_loop_start", agent=self)

        iteration = 0
        max_iter = self.config.max_monologue_iterations
        no_tool_count = 0

        while iteration < max_iter:
            iteration += 1

            # 1. Fire hook: before prompts assembly (memory recall injects here)
            await self._dispatch_hook(
                "message_loop_prompts_before", agent=self, iteration=iteration
            )

            # 2. Build system prompt
            system_prompt = self._build_system_prompt()

            # 3. Assemble messages for LLM
            messages = [{"role": "system", "content": system_prompt}] + self.history

            # 4. Fire hook: before_llm_call
            modified = await self._dispatch_hook(
                "before_llm_call", agent=self, messages=messages
            )
            if modified is not None:
                messages = modified

            # 5. Call Ollama (streaming)
            full_response = ""
            try:
                async for chunk in self._client.chat(
                    model=self.config.chat_model.model_name,
                    messages=messages,
                    stream=True,
                    temperature=self.config.chat_model.temperature,
                    options=self.config.chat_model.options,
                ):
                    full_response += chunk
                    if self.context.on_stream:
                        self.context.on_stream(self, chunk)
            except Exception as e:
                error_msg = f"[LLM Error: {e}]"
                self.append_message("assistant", error_msg)
                return error_msg

            # 6. Fire hook: after_llm_call
            await self._dispatch_hook(
                "after_llm_call", agent=self, response=full_response
            )

            # 7. Append assistant response to history
            self.append_message("assistant", full_response)

            # 8. Extract tool calls from response
            tool_calls = self._extract_tools(full_response)

            if not tool_calls:
                no_tool_count += 1
                # If agent hasn't used tools for 3 iterations, nudge it
                if no_tool_count >= 3:
                    self.append_message(
                        "system",
                        "You have been reasoning without using any tools. "
                        "Please use the `response` tool to deliver your answer to the user, "
                        "or use another tool to take action."
                    )
                    no_tool_count = 0
                continue

            no_tool_count = 0

            # 9. Execute each tool call (typically one per iteration)
            for tc in tool_calls:
                # Fire hook: tool_execute_before
                await self._dispatch_hook(
                    "tool_execute_before", agent=self, tool_call=tc
                )

                response = await self._execute_tool(tc)

                # Fire hook: tool_execute_after
                await self._dispatch_hook(
                    "tool_execute_after", agent=self, tool_call=tc, response=response
                )

                # Append tool result to history
                self.append_message(
                    "system",
                    f"[Tool '{tc.name}' result]:\n{response.message}"
                )

                # If tool signals break (response, task_done)
                if response.break_loop:
                    await self._dispatch_hook(
                        "monologue_end", agent=self, final_response=response.message
                    )
                    return response.message

        # Safety: max iterations reached
        fallback = "[Agent reached maximum iterations without producing a final response]"
        await self._dispatch_hook("monologue_end", agent=self, final_response=fallback)
        return fallback

    # ── System prompt assembly ───────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt from templates + tool descriptions."""
        if self.system_prompt_override:
            # Subordinate with custom prompt — still add tool descriptions
            tool_desc = self.tool_registry.get_tool_descriptions()
            return (
                self.system_prompt_override
                + "\n\n## Available Tools\n\n"
                + tool_desc
                + "\n\nUse tools by including a JSON block:\n"
                + '```json\n{"tool_name": "...", "tool_args": {...}}\n```\n'
                + "Use the `task_done` tool when you have completed your task."
            )

        variables = {
            "agent_id": str(self.agent_id),
            "current_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "tool_descriptions": self.tool_registry.get_tool_descriptions(),
            "memory_context": self.data.get("memory_context", ""),
        }

        try:
            return self._prompt_engine.render("agent.system.main.md", variables)
        except Exception as e:
            # Fallback if templates fail
            return (
                f"You are Agent {self.agent_id}. "
                f"Available tools:\n{variables['tool_descriptions']}\n"
                "Use the `response` tool to deliver your final answer."
            )

    # ── Tool extraction from LLM output ──────────────────────────────

    def _extract_tools(self, text: str) -> list[ToolCall]:
        """
        Extract tool call JSON from the LLM's text output.
        Multi-strategy: fenced code blocks, then bare JSON objects.
        Returns at most one tool call per iteration.
        """
        # Strategy 1: fenced code blocks ```json ... ```
        fenced = re.findall(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        for block in fenced:
            tc = self._try_parse_tool_json(block)
            if tc:
                return [tc]

        # Strategy 2: bare JSON with tool_name key
        bare = re.findall(r'\{[^{}]*"tool_name"[^{}]*\}', text, re.DOTALL)
        for block in bare:
            tc = self._try_parse_tool_json(block)
            if tc:
                return [tc]

        # Strategy 3: nested JSON (tool_args may contain braces)
        nested = re.findall(r'\{"tool_name".*?\}\s*\}', text, re.DOTALL)
        for block in nested:
            tc = self._try_parse_tool_json(block)
            if tc:
                return [tc]

        return []

    def _try_parse_tool_json(self, text: str) -> ToolCall | None:
        """Attempt to parse a JSON string into a ToolCall. Lenient parsing."""
        text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try fixing common issues: trailing commas, single quotes
            cleaned = re.sub(r',\s*}', '}', text)
            cleaned = re.sub(r",\s*]", ']', cleaned)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                return None

        if not isinstance(data, dict) or "tool_name" not in data:
            return None

        name = str(data["tool_name"]).strip()
        args = data.get("tool_args", {})
        if not isinstance(args, dict):
            args = {}

        return ToolCall(name=name, args=args)

    # ── Tool execution ───────────────────────────────────────────────

    async def _execute_tool(self, tool_call: ToolCall) -> Response:
        """Look up and execute a tool."""
        tool = self.tool_registry.get_tool(tool_call.name)
        if tool is None:
            return Response(
                message=f"[Error: Unknown tool '{tool_call.name}'. "
                f"Available tools: {', '.join(self.tool_registry.tool_names)}]"
            )

        try:
            await tool.before_execution(**tool_call.args)
            response = await tool.execute(**tool_call.args)
            response = await tool.after_execution(response)
            return response
        except Exception as e:
            return Response(message=f"[Tool '{tool_call.name}' error: {e}]")

    # ── Extension hook dispatch ──────────────────────────────────────

    async def _dispatch_hook(self, hook_name: str, **kwargs):
        """Dispatch a lifecycle hook to the extension manager if available."""
        if self.context.extension_manager:
            return await self.context.extension_manager.dispatch(hook_name, **kwargs)
        return None

    # ── History management ───────────────────────────────────────────

    def append_message(self, role: str, content: str):
        """Add a message to conversation history."""
        # Map our roles to Ollama-compatible roles
        ollama_role = role if role in ("user", "assistant", "system") else "system"
        self.history.append({"role": ollama_role, "content": content})
