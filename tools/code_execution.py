"""Code execution tool — runs Python, Shell, or Node.js in persistent subprocess sessions."""

import asyncio
import os
import uuid
from tools.base_tool import Tool
from agent.response import Response


# Sentinel marker for detecting end of output
_END_MARKER = f"__LOA_END_{uuid.uuid4().hex[:8]}__"

RUNTIME_COMMANDS = {
    "python": ["python3", "-u", "-i"],
    "shell": ["bash", "--norc", "--noprofile"],
    "node": ["node", "-i"],
}


class PersistentSession:
    """A long-lived subprocess session for a single runtime."""

    def __init__(self, runtime: str):
        self.runtime = runtime
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the subprocess."""
        cmd = RUNTIME_COMMANDS.get(self.runtime)
        if not cmd:
            raise ValueError(f"Unknown runtime: {self.runtime}")

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    async def run(self, code: str, timeout: float = 30.0) -> str:
        """Execute code and capture output until the sentinel marker."""
        async with self._lock:
            if self.process is None or self.process.returncode is not None:
                await self.start()

            # Construct the command with sentinel marker
            if self.runtime == "python":
                sentinel_cmd = f"\nprint('{_END_MARKER}')\n"
            elif self.runtime == "shell":
                sentinel_cmd = f"\necho '{_END_MARKER}'\n"
            elif self.runtime == "node":
                sentinel_cmd = f"\nconsole.log('{_END_MARKER}')\n"
            else:
                sentinel_cmd = f"\necho '{_END_MARKER}'\n"

            full_input = code + sentinel_cmd
            self.process.stdin.write(full_input.encode())
            await self.process.stdin.drain()

            # Read output until sentinel marker or timeout
            output_lines = []
            try:
                while True:
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(), timeout=timeout
                    )
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    if _END_MARKER in decoded:
                        break
                    output_lines.append(decoded)
            except asyncio.TimeoutError:
                output_lines.append(f"\n[Execution timed out after {timeout}s]")

            output = "".join(output_lines).strip()

            # Clean up interactive prompt artifacts
            for prefix in [">>> ", "... ", "> ", "$ "]:
                output = output.replace(prefix, "")

            # Truncate very long output
            max_len = 50000
            if len(output) > max_len:
                output = output[:max_len] + f"\n\n[Output truncated at {max_len} characters]"

            return output if output else "[No output]"

    async def close(self):
        """Terminate the subprocess."""
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()


class CodeExecutionTool(Tool):
    name = "code_execution"
    description = "Execute code in a persistent terminal session."

    async def execute(self, **kwargs) -> Response:
        runtime = kwargs.get("runtime", "python")
        code = kwargs.get("code", "")

        if not code.strip():
            return Response(message="[Error: No code provided]")

        if runtime not in RUNTIME_COMMANDS:
            return Response(
                message=f"[Error: Unknown runtime '{runtime}'. Use: python, shell, or node]"
            )

        # Get or create persistent session from context
        sessions = self.agent.context.data.setdefault("sessions", {})
        if runtime not in sessions:
            sessions[runtime] = PersistentSession(runtime)

        session = sessions[runtime]
        try:
            output = await session.run(code)
            return Response(message=output)
        except Exception as e:
            return Response(message=f"[Execution error: {e}]")
