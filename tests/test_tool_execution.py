import asyncio
import tempfile
import unittest
from pathlib import Path

from agent.agent import Agent
from agent.agent_context import AgentContext
from agent.config import AgentConfig, ToolExecutionConfig
from agent.response import Response, ToolCall
from tools.base_tool import Tool


class CounterTool(Tool):
    name = "counter"
    description = "Counts executions."
    calls = 0

    async def execute(self, **kwargs) -> Response:
        CounterTool.calls += 1
        return Response(message="ok")


class SleepTool(Tool):
    name = "sleep"
    description = "Sleeps briefly."

    async def execute(self, **kwargs) -> Response:
        await asyncio.sleep(0.05)
        return Response(message="slept")


class ParallelGateTool(Tool):
    name = "parallel_gate"
    description = "Waits until two calls start."
    cacheable = False
    started = 0
    event: asyncio.Event | None = None

    async def execute(self, **kwargs) -> Response:
        if ParallelGateTool.event is None:
            ParallelGateTool.event = asyncio.Event()
        ParallelGateTool.started += 1
        if ParallelGateTool.started >= 2:
            ParallelGateTool.event.set()
        await ParallelGateTool.event.wait()
        return Response(message="ok")


class TestToolExecution(unittest.IsolatedAsyncioTestCase):
    def _make_agent(self) -> Agent:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(
                log_dir=str(Path(tmpdir) / "logs"),
                tool_execution=ToolExecutionConfig(default_timeout=1.0, cache_enabled=True),
            )
            context = AgentContext(config)
            agent = context.create_agent(agent_id=0)
            agent.tool_registry._tool_classes["counter"] = CounterTool
            agent.tool_registry._tool_classes["sleep"] = SleepTool
            agent.tool_registry._tool_classes["parallel_gate"] = ParallelGateTool
            return agent

    async def test_tool_result_cached(self):
        agent = self._make_agent()
        CounterTool.calls = 0

        tc = ToolCall(name="counter", args={"x": 1})
        await agent._execute_tool_calls([tc])
        await agent._execute_tool_calls([tc])

        self.assertEqual(CounterTool.calls, 1)

    async def test_tool_timeout(self):
        agent = self._make_agent()
        agent.config.tool_execution.default_timeout = 0.01

        tc = ToolCall(name="sleep", args={})
        results = await agent._execute_tool_calls([tc])
        _, response, _ = results[0]
        self.assertIn("timed out", response.message)

    async def test_parallel_execution(self):
        agent = self._make_agent()
        ParallelGateTool.started = 0
        ParallelGateTool.event = asyncio.Event()

        calls = [
            ToolCall(name="parallel_gate", args={}),
            ToolCall(name="parallel_gate", args={}),
        ]
        results = await agent._execute_tool_calls(calls)
        messages = [response.message for _, response, _ in results]
        self.assertEqual(messages, ["ok", "ok"])
