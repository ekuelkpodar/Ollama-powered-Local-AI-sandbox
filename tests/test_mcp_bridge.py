import tempfile
import unittest
from pathlib import Path

from agent.config import AgentConfig, MCPConfig, MCPServerConfig
from tools.mcp_bridge import MCPBridge, MCPToolSpec


class FakeClient:
    def __init__(self, server, tools):
        self.server = server
        self.tools = tools
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return self.tools

    async def call_tool(self, tool_name, args):
        return {"ok": True, "tool": tool_name, "args": args}


class TestMCPBridge(unittest.IsolatedAsyncioTestCase):
    async def test_discover_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(
                log_dir=str(Path(tmpdir) / "logs"),
                mcp=MCPConfig(
                    enabled=True,
                    servers=[
                        MCPServerConfig(name="alpha", transport="stdio", command="x"),
                        MCPServerConfig(name="beta", transport="stdio", command="y"),
                    ],
                ),
            )

            tools_alpha = [
                MCPToolSpec(
                    name="read_file",
                    description="Read file",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                    required=["path"],
                )
            ]
            tools_beta = [
                MCPToolSpec(
                    name="read_file",
                    description="Read file v2",
                    input_schema={"type": "object", "properties": {}},
                    required=[],
                )
            ]

            def factory(server):
                if server.name == "alpha":
                    return FakeClient(server, tools_alpha)
                return FakeClient(server, tools_beta)

            bridge = MCPBridge(config, client_factory=factory)
            tool_classes = await bridge.discover_tools()

            self.assertIn("read_file", tool_classes)
            self.assertIn("beta.read_file", tool_classes)
            self.assertEqual(tool_classes["read_file"].required_args, ["path"])

    async def test_call_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(
                log_dir=str(Path(tmpdir) / "logs"),
                mcp=MCPConfig(
                    enabled=True,
                    servers=[MCPServerConfig(name="alpha", transport="stdio", command="x")],
                ),
            )
            tools_alpha = [
                MCPToolSpec(
                    name="echo",
                    description="Echo",
                    input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                    required=["text"],
                )
            ]

            bridge = MCPBridge(config, client_factory=lambda server: FakeClient(server, tools_alpha))
            result = await bridge.call_tool("alpha", "echo", {"text": "hi"})
            self.assertEqual(result["ok"], True)
