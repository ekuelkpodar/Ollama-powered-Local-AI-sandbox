"""Call subordinate tool — delegates tasks to subordinate agents."""

from tools.base_tool import Tool
from agent.response import Response


class CallSubordinateTool(Tool):
    name = "call_subordinate"
    description = "Delegate a task to a subordinate agent."

    async def execute(self, **kwargs) -> Response:
        task = kwargs.get("task", "")
        system_prompt = kwargs.get("system_prompt", "")

        if not task.strip():
            return Response(message="[Error: No task provided for subordinate]")

        # Create subordinate agent with next ID
        next_id = max(self.agent.context.agents.keys(), default=-1) + 1
        subordinate = self.agent.context.create_agent(
            agent_id=next_id,
            parent=self.agent,
            system_prompt=system_prompt,
        )

        # Stream callback: prefix subordinate output with its ID
        if self.agent.context.on_stream:
            original_stream = self.agent.context.on_stream

            def sub_stream(agent, chunk):
                original_stream(agent, chunk)

            self.agent.context.on_stream = sub_stream

        try:
            result = await subordinate.monologue(task)
            return Response(message=f"Subordinate Agent {next_id} result:\n{result}")
        except Exception as e:
            return Response(message=f"[Subordinate Agent {next_id} failed: {e}]")
