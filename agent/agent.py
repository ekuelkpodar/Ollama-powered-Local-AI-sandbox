"""Agent class — the core actor with the monologue loop."""

from __future__ import annotations
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agent.config import AgentConfig
from agent.models import OllamaClient
from agent.exceptions import OllamaConnectionError, OllamaModelError
from agent.output_parser import OutputParser
from agent.model_router import ModelRouter
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
        self._client = OllamaClient(
            base_url=config.chat_model.base_url,
            connect_timeout=config.ollama.connect_timeout,
            read_timeout=config.ollama.read_timeout,
            max_retries=config.ollama.max_retries,
        )
        self._model_router = ModelRouter(config, self._client)

        # Prompt template engine
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
        self._prompt_engine = PromptTemplateEngine(prompts_dir, config.prompt_profile)

        # Tool registry
        self.tool_registry = ToolRegistry(self)
        self.tool_registry.discover_tools()

        # Output parser
        self._output_parser = OutputParser(log_dir=config.log_dir)

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
            iteration_start = time.monotonic()

            # 1. Fire hook: before prompts assembly (memory recall injects here)
            await self._dispatch_hook(
                "message_loop_prompts_before", agent=self, iteration=iteration
            )

            # 2. Build system prompt
            system_prompt = self._build_system_prompt()

            # 3. Assemble messages for LLM
            messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m.get("role", "system"), "content": m.get("content", "")}
                for m in self.history
            ]

            # 4. Fire hook: before_llm_call
            modified = await self._dispatch_hook(
                "before_llm_call", agent=self, messages=messages
            )
            if modified is not None:
                messages = modified

            # 5. Call Ollama (streaming)
            full_response = ""
            llm_start = time.monotonic()
            model_name = self.config.chat_model.model_name
            try:
                await self._model_router.ensure_models_loaded()
                model_name = self._model_router.select_model(
                    messages,
                    last_tool_name=self.data.get("last_tool_name"),
                )
                async for chunk in self._client.chat(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    temperature=self.config.chat_model.temperature,
                    options=self.config.chat_model.options,
                ):
                    full_response += chunk
                    if self.context.on_stream:
                        self.context.on_stream(self, chunk)
            except OllamaModelError as e:
                error_msg = f"[LLM Model Error: {e}]"
                llm_latency = (time.monotonic() - llm_start) * 1000
                prompt_tokens = self._estimate_prompt_tokens(messages)
                self._record_llm_metrics(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    latency_ms=llm_latency,
                    error=error_msg,
                )
                self.append_message("assistant", error_msg)
                self._record_iteration(iteration, "llm_error", iteration_start)
                return error_msg
            except OllamaConnectionError as e:
                error_msg = f"[LLM Connection Error: {e}]"
                llm_latency = (time.monotonic() - llm_start) * 1000
                prompt_tokens = self._estimate_prompt_tokens(messages)
                self._record_llm_metrics(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    latency_ms=llm_latency,
                    error=error_msg,
                )
                self.append_message("assistant", error_msg)
                self._record_iteration(iteration, "llm_error", iteration_start)
                return error_msg
            except Exception as e:
                error_msg = f"[LLM Error: {e}]"
                llm_latency = (time.monotonic() - llm_start) * 1000
                prompt_tokens = self._estimate_prompt_tokens(messages)
                self._record_llm_metrics(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    latency_ms=llm_latency,
                    error=error_msg,
                )
                self.append_message("assistant", error_msg)
                self._record_iteration(iteration, "llm_error", iteration_start)
                return error_msg

            # 6. Fire hook: after_llm_call
            await self._dispatch_hook(
                "after_llm_call", agent=self, response=full_response
            )

            llm_latency = (time.monotonic() - llm_start) * 1000
            prompt_tokens = self._estimate_prompt_tokens(messages)
            completion_tokens = self._estimate_tokens(full_response)
            self._record_llm_metrics(
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=llm_latency,
            )

            # Persist token counts
            self._persist_token_counts(
                messages,
                full_response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            # 7. Append assistant response to history
            self.append_message("assistant", full_response)

            # 8. Extract tool calls from response
            tool_calls = self._extract_tools(full_response)

            if not tool_calls:
                no_tool_count += 1
                # If agent hasn't used tools for 5 iterations, nudge it with a structured reminder
                if no_tool_count >= 5:
                    self.append_message(
                        "system",
                        "Reminder: You must use a tool call JSON to proceed. "
                        "Example:\n"
                        '```json\n{"tool_name":"response","tool_args":{"text":"<final answer>"}}\n```\n'
                        "Use the `response` tool to deliver your final answer."
                    )
                    no_tool_count = 0
                self._record_iteration(iteration, "no_tool", iteration_start)
                continue

            no_tool_count = 0

            # 9. Execute tool calls (sequentially or in parallel)
            results = await self._execute_tool_calls(tool_calls)

            final_response = None
            final_tool_name = None
            for tc, response, _cached in results:
                # Persist tool call
                self._persist_tool_call(tc, response)
                self.data["last_tool_name"] = tc.name

                # Append tool result to history
                self.append_message(
                    "system",
                    f"[Tool '{tc.name}' result]:\n{response.message}"
                )

                # If tool signals break (response, task_done)
                if response.break_loop and final_response is None:
                    final_response = response.message
                    final_tool_name = tc.name

            if final_response is not None:
                await self._dispatch_hook(
                    "monologue_end", agent=self, final_response=final_response
                )
                self.context.telemetry.finalize(final_tool_name or "response")
                self._record_iteration(iteration, "final", iteration_start)
                return final_response

            self._record_iteration(
                iteration,
                f"tool:{','.join(tc.name for tc in tool_calls)}",
                iteration_start,
            )

        # Safety: max iterations reached
        fallback = "[Agent reached maximum iterations without producing a final response]"
        await self._dispatch_hook("monologue_end", agent=self, final_response=fallback)
        self.context.telemetry.finalize("max_iterations")
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
        Returns zero or more tool calls per iteration.
        """
        return self._output_parser.extract_tool_calls(
            raw_text=text,
            registered_tools=self.tool_registry.tool_names,
            tool_schemas=self.tool_registry.get_tool_schemas(),
        )

    # ── Tool execution ───────────────────────────────────────────────

    async def _execute_tool(self, tool_call: ToolCall) -> Response:
        """Look up and execute a tool."""
        tool = self.tool_registry.get_tool(tool_call.name)
        if tool is None:
            return Response(
                message=f"[Error: Unknown tool '{tool_call.name}'. "
                f"Available tools: {', '.join(self.tool_registry.tool_names)}]"
            )
        response, _cached = await self._execute_tool_with_instance(tool_call, tool)
        return response

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[tuple[ToolCall, Response, bool]]:
        """Execute tool calls sequentially or in parallel when safe."""
        if not tool_calls:
            return []

        prepared: list[tuple[ToolCall, object | None]] = []
        for tc in tool_calls:
            tool = self.tool_registry.get_tool(tc.name)
            prepared.append((tc, tool))

        all_parallel_safe = all(
            tool is not None and tool.is_parallel_safe(**tc.args)
            for tc, tool in prepared
        )

        if len(tool_calls) == 1 or not all_parallel_safe:
            results: list[tuple[ToolCall, Response, bool]] = []
            for tc, tool in prepared:
                results.append(await self._run_tool_call(tc, tool))
            return results

        tasks = [
            asyncio.create_task(self._run_tool_call(tc, tool))
            for tc, tool in prepared
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def _run_tool_call(
        self,
        tool_call: ToolCall,
        tool,
    ) -> tuple[ToolCall, Response, bool]:
        """Run a tool call with hooks and caching."""
        if tool is None:
            response = Response(
                message=f"[Error: Unknown tool '{tool_call.name}'. "
                f"Available tools: {', '.join(self.tool_registry.tool_names)}]"
            )
            return tool_call, response, False

        await self._dispatch_hook(
            "tool_execute_before", agent=self, tool_call=tool_call
        )

        start_time = time.monotonic()
        response, cached = await self._execute_tool_with_instance(tool_call, tool)
        duration_ms = (time.monotonic() - start_time) * 1000
        self._record_tool_metrics(tool_call, response, cached, duration_ms)

        await self._dispatch_hook(
            "tool_execute_after",
            agent=self,
            tool_call=tool_call,
            response=response,
            cached=cached,
        )
        return tool_call, response, cached

    async def _execute_tool_with_instance(
        self,
        tool_call: ToolCall,
        tool,
    ) -> tuple[Response, bool]:
        """Execute a tool with timeout and caching."""
        cache = self._get_tool_cache()
        cache_key = self._tool_cache_key(tool_call)
        cache_enabled = self.config.tool_execution.cache_enabled
        can_cache = cache_enabled and tool.should_cache(**tool_call.args)

        if can_cache and cache_key in cache:
            cached = cache[cache_key]
            return Response(
                message=cached["message"],
                break_loop=cached["break_loop"],
            ), True

        timeout = self._get_tool_timeout(tool_call.name, tool)

        async def _run():
            await tool.before_execution(**tool_call.args)
            response = await tool.execute(**tool_call.args)
            response = await tool.after_execution(response)
            return response

        try:
            response = await asyncio.wait_for(_run(), timeout=timeout)
        except asyncio.TimeoutError:
            response = Response(
                message=f"[Tool '{tool_call.name}' timed out after {timeout:.1f}s]"
            )
        except Exception as e:
            response = Response(message=f"[Tool '{tool_call.name}' error: {e}]")

        if can_cache:
            cache[cache_key] = {
                "message": response.message,
                "break_loop": response.break_loop,
            }

        return response, False

    def _get_tool_timeout(self, tool_name: str, tool) -> float:
        """Resolve the tool execution timeout in seconds."""
        timeouts = self.config.tool_execution.timeouts
        if tool_name in timeouts:
            return timeouts[tool_name]
        if tool.timeout_seconds is not None:
            return tool.timeout_seconds
        return self.config.tool_execution.default_timeout

    def _get_tool_cache(self) -> dict:
        """Get or create the per-session tool cache."""
        return self.context.data.setdefault("tool_cache", {})

    def _tool_cache_key(self, tool_call: ToolCall) -> str:
        """Build a stable cache key for a tool call."""
        try:
            args_blob = json.dumps(tool_call.args, sort_keys=True, default=str)
        except Exception:
            args_blob = str(tool_call.args)
        return f"{tool_call.name}:{args_blob}"

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
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.history.append({
            "role": ollama_role,
            "content": content,
            "created_at": timestamp,
        })
        self.context.data["session_updated_at"] = timestamp
        if ollama_role == "user" and not self.context.data.get("session_title"):
            title = self._derive_session_title(content)
            if title:
                self.context.data["session_title"] = title

        if self.context.session_store and self.config.session.auto_save:
            try:
                self.context.session_store.save_message(
                    session_id=self.context.id,
                    role=ollama_role,
                    content=content,
                    created_at=timestamp,
                )
                if ollama_role == "user" and self.context.data.get("session_title"):
                    self.context.session_store.set_title(
                        session_id=self.context.id,
                        title=self.context.data["session_title"],
                    )
            except Exception:
                # Never break the loop on persistence errors
                pass

    def _persist_tool_call(self, tool_call: ToolCall, response: Response) -> None:
        """Persist tool call details if session storage is enabled."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.context.data.setdefault("tool_calls", []).append({
            "tool_name": tool_call.name,
            "args": tool_call.args,
            "result": response.message,
            "created_at": timestamp,
        })
        if not self.context.session_store or not self.config.session.auto_save:
            return
        try:
            self.context.session_store.save_tool_call(
                session_id=self.context.id,
                tool_name=tool_call.name,
                args=tool_call.args,
                result=response.message,
                created_at=timestamp,
            )
        except Exception:
            # Never break the loop on persistence errors
            pass

    def _persist_token_counts(
        self,
        messages: list[dict],
        response: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Persist approximate token counts if session storage is enabled."""
        if not self.context.session_store or not self.config.session.auto_save:
            return
        try:
            if prompt_tokens is None:
                prompt_tokens = sum(
                    self._estimate_tokens(m.get("content", "")) for m in messages
                )
            if completion_tokens is None:
                completion_tokens = self._estimate_tokens(response)
            self.context.session_store.increment_token_counts(
                session_id=self.context.id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception:
            pass

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for a piece of text."""
        try:
            import tiktoken

            encoder = tiktoken.get_encoding("cl100k_base")
            return len(encoder.encode(text))
        except Exception:
            return len(text.split())

    @staticmethod
    def _derive_session_title(content: str, max_len: int = 60) -> str:
        cleaned = " ".join(content.strip().split())
        if not cleaned:
            return ""
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[:max_len].rstrip() + "..."

    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for a list of chat messages."""
        return sum(self._estimate_tokens(m.get("content", "")) for m in messages)

    def _record_llm_metrics(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        telemetry = self.context.telemetry
        if telemetry:
            telemetry.record_llm_call(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                error=error,
            )

    def _record_iteration(self, iteration: int, decision: str, start_time: float) -> None:
        telemetry = self.context.telemetry
        if telemetry:
            duration_ms = (time.monotonic() - start_time) * 1000
            telemetry.record_iteration(
                iteration=iteration,
                decision=decision,
                duration_ms=duration_ms,
            )

    def _record_tool_metrics(
        self,
        tool_call: ToolCall,
        response: Response,
        cached: bool,
        duration_ms: float,
    ) -> None:
        telemetry = self.context.telemetry
        if not telemetry:
            return
        summary = self._summarize_text(response.message)
        error = None
        lower_msg = response.message.lower()
        if lower_msg.startswith("[error") or ("error" in lower_msg and lower_msg.startswith("[tool")):
            error = summary
        telemetry.record_tool_call(
            tool_name=tool_call.name,
            args=tool_call.args,
            duration_ms=duration_ms,
            cached=cached,
            result_summary=summary,
            error=error,
        )

    @staticmethod
    def _summarize_text(text: str, limit: int = 200) -> str:
        """Summarize text for telemetry payloads."""
        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return clean[:limit].rstrip() + "..."
