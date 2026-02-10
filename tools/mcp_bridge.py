"""MCP bridge for discovering and executing MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import aiohttp

from agent.config import AgentConfig, MCPServerConfig
from agent.response import Response
from tools.base_tool import Tool


@dataclass
class MCPToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    required: list[str]


class MCPBridge:
    """Manages MCP server clients and tool registration."""

    def __init__(
        self,
        config: AgentConfig,
        client_factory: Callable[[MCPServerConfig], "MCPClient"] | None = None,
    ):
        self.config = config
        self._clients: dict[str, MCPClient] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._client_factory = client_factory or self._default_client_factory
        self._logger = self._build_logger(config.log_dir)

    async def discover_tools(self) -> dict[str, type[Tool]]:
        """Discover tools across all configured MCP servers."""
        tools: dict[str, type[Tool]] = {}
        if not self.config.mcp.enabled:
            return tools

        for server in self.config.mcp.servers:
            try:
                client = self._get_client(server)
                await client.initialize()
                tool_list = await client.list_tools()
                for spec in tool_list:
                    tool_name = self._unique_tool_name(spec.name, server.name, tools)
                    tool_cls = self._build_tool_class(tool_name, server.name, spec)
                    tools[tool_name] = tool_cls
            except Exception as exc:
                self._logger.warning("MCP server '%s' discovery failed: %s", server.name, exc)
        return tools

    def discover_tools_sync(self) -> dict[str, type[Tool]]:
        """Synchronous wrapper for tool discovery."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result: dict[str, type[Tool]] = {}
            error: Exception | None = None

            def _runner():
                nonlocal result, error
                try:
                    result = asyncio.run(self.discover_tools())
                except Exception as exc:
                    error = exc

            import threading

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            if error:
                raise error
            return result

        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(self.discover_tools())
        finally:
            new_loop.close()

    async def call_tool(self, server_name: str, tool_name: str, args: dict[str, Any]) -> Any:
        """Call a tool on the given MCP server."""
        server = self._find_server(server_name)
        if server is None:
            raise RuntimeError(f"MCP server '{server_name}' not configured")
        client = self._get_client(server)
        await client.initialize()
        return await client.call_tool(tool_name, args)

    def _get_client(self, server: MCPServerConfig) -> "MCPClient":
        if server.name not in self._clients:
            self._clients[server.name] = self._client_factory(server)
        return self._clients[server.name]

    def _build_tool_class(self, tool_name: str, server_name: str, spec: MCPToolSpec) -> type[Tool]:
        self._tool_map[tool_name] = (server_name, spec.name)
        arg_schema, required = self._schema_to_args(spec.input_schema, spec.required)
        description = f"[MCP:{server_name}] {spec.description}".strip()

        bridge = self

        class MCPTool(Tool):
            cacheable = False
            parallel_safe = False

            async def execute(self, **kwargs) -> Response:
                try:
                    result = await bridge.call_tool(server_name, spec.name, kwargs)
                    return Response(message=bridge._format_result(result))
                except Exception as exc:
                    return Response(message=f"[MCP tool '{tool_name}' error: {exc}]")

        MCPTool.name = tool_name
        MCPTool.description = description
        MCPTool.arg_schema = arg_schema
        MCPTool.required_args = required

        return MCPTool

    def _unique_tool_name(self, tool_name: str, server_name: str, existing: dict[str, type[Tool]]) -> str:
        if tool_name not in existing:
            return tool_name
        prefixed = f"{server_name}.{tool_name}"
        if prefixed not in existing:
            return prefixed
        idx = 2
        while True:
            candidate = f"{prefixed}.{idx}"
            if candidate not in existing:
                return candidate
            idx += 1

    def _schema_to_args(self, schema: dict[str, Any], required: list[str] | None) -> tuple[dict, list[str]]:
        if not isinstance(schema, dict):
            return {}, list(required or [])
        properties = schema.get("properties", {})
        arg_schema: dict[str, type | tuple[type, ...]] = {}
        for prop, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            arg_schema[prop] = self._map_json_type(spec.get("type"))
        return arg_schema, list(required or schema.get("required", []) or [])

    @staticmethod
    def _map_json_type(json_type: str | None) -> type | tuple[type, ...]:
        return {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": type(None),
        }.get(json_type or "", str)

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)

    def _find_server(self, name: str) -> MCPServerConfig | None:
        for server in self.config.mcp.servers:
            if server.name == name:
                return server
        return None

    def _default_client_factory(self, server: MCPServerConfig) -> "MCPClient":
        if server.transport == "stdio":
            return MCPStdioClient(server, self._logger)
        return MCPSseClient(server, self._logger)

    def _build_logger(self, log_dir: str) -> logging.Logger:
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f"mcp_bridge.{id(self)}")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_dir, "mcp_bridge.log")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger


class MCPClient:
    """Base MCP client interface."""

    def __init__(self, server: MCPServerConfig, logger: logging.Logger):
        self.server = server
        self._logger = logger
        self._initialized = False
        self._request_id = 0

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "local-ollama-agents", "version": "0.1.0"},
                "capabilities": {},
            },
        )
        await self._notify("initialized", {})
        self._initialized = True

    async def list_tools(self) -> list[MCPToolSpec]:
        response = await self._request("tools/list", {})
        tools = response.get("tools", []) if isinstance(response, dict) else []
        specs: list[MCPToolSpec] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            specs.append(
                MCPToolSpec(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}) or {},
                    required=tool.get("inputSchema", {}).get("required", []) if isinstance(tool.get("inputSchema"), dict) else [],
                )
            )
        return [spec for spec in specs if spec.name]

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        response = await self._request(
            "tools/call",
            {"name": tool_name, "arguments": args},
        )
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        return response

    async def close(self) -> None:
        return None

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id


class MCPStdioClient(MCPClient):
    """MCP client over stdio transport."""

    def __init__(self, server: MCPServerConfig, logger: logging.Logger):
        super().__init__(server, logger)
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _ensure_process(self) -> None:
        if self._proc and self._proc.returncode is None:
            return
        if not self.server.command:
            raise RuntimeError(f"MCP stdio server '{self.server.name}' missing command")
        self._proc = await asyncio.create_subprocess_exec(
            self.server.command,
            *self.server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        async with self._lock:
            await self._ensure_process()
            if not self._proc or not self._proc.stdin or not self._proc.stdout:
                raise RuntimeError("MCP stdio process not available")
            req_id = self._next_id()
            payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await self._proc.stdin.drain()
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP stdio server closed connection")
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if data.get("id") != req_id:
                    continue
                if "error" in data:
                    raise RuntimeError(data["error"])
                return data.get("result", {})

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._ensure_process()
        if not self._proc or not self._proc.stdin:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()


class MCPSseClient(MCPClient):
    """MCP client over SSE transport."""

    def __init__(self, server: MCPServerConfig, logger: logging.Logger):
        super().__init__(server, logger)
        self._session: aiohttp.ClientSession | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._listen_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._listen_task is None:
            self._listen_task = asyncio.create_task(self._listen())

    def _send_url(self) -> str:
        if self.server.send_url:
            return self.server.send_url
        if not self.server.url:
            raise RuntimeError(f"MCP SSE server '{self.server.name}' missing url")
        if self.server.url.endswith("/sse"):
            return self.server.url[:-4] + "/message"
        return self.server.url.rstrip("/") + "/message"

    async def _listen(self) -> None:
        if not self.server.url:
            return
        await self._ensure_session()
        assert self._session is not None
        try:
            async with self._session.get(self.server.url) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    req_id = data.get("id")
                    future = self._pending.pop(req_id, None)
                    if future and not future.done():
                        if "error" in data:
                            future.set_exception(RuntimeError(data["error"]))
                        else:
                            future.set_result(data.get("result", {}))
        except Exception as exc:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
            self._pending.clear()

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        await self._ensure_session()
        assert self._session is not None
        req_id = self._next_id()
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future
        async with self._lock:
            await self._session.post(self._send_url(), json=payload)
        return await future

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._ensure_session()
        assert self._session is not None
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        async with self._lock:
            await self._session.post(self._send_url(), json=payload)

    async def close(self) -> None:
        if self._listen_task:
            self._listen_task.cancel()
        if self._session:
            await self._session.close()
