"""Response tool — delivers the final answer and breaks the monologue loop."""

from tools.base_tool import Tool
from agent.response import Response


class ResponseTool(Tool):
    name = "response"
    description = "Deliver your final answer to the user."

    async def execute(self, **kwargs) -> Response:
        text = kwargs.get("text", "")
        return Response(message=text, break_loop=True)
