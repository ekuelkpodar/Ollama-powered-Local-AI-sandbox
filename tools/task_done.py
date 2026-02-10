"""Task done tool — subordinate agent signals completion to its superior."""

from tools.base_tool import Tool
from agent.response import Response


class TaskDoneTool(Tool):
    name = "task_done"
    description = "Signal task completion and return results to the superior agent."
    arg_schema = {"text": str}
    required_args = ["text"]
    cacheable = False

    async def execute(self, **kwargs) -> Response:
        text = kwargs.get("text", "")
        return Response(message=text, break_loop=True)
