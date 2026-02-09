"""Output logger extension — logs all agent activity to JSON files."""

import json
import os
from datetime import datetime, timezone

from extensions.base_extension import Extension


class OutputLoggerExtension(Extension):
    name = "output_logger"
    enabled = True

    def _log_path(self, agent) -> str:
        log_dir = self.config.log_dir
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"session_{agent.context.id}.jsonl")

    def _write_entry(self, agent, event: str, data: dict):
        path = self._log_path(agent)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent.agent_id,
            "event": event,
            **data,
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def on_message_loop_start(self, agent, **kwargs):
        self._write_entry(agent, "loop_start", {})

    async def on_after_llm_call(self, agent, response, **kwargs):
        self._write_entry(agent, "llm_response", {
            "response_length": len(response),
            "response_preview": response[:500],
        })

    async def on_tool_execute_after(self, agent, tool_call, response, **kwargs):
        self._write_entry(agent, "tool_result", {
            "tool": tool_call.name,
            "args": tool_call.args,
            "result_preview": response.message[:500],
            "break_loop": response.break_loop,
        })

    async def on_monologue_end(self, agent, final_response, **kwargs):
        self._write_entry(agent, "monologue_end", {
            "response_preview": final_response[:500],
        })
