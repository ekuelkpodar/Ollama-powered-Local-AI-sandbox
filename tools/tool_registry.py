"""Tool discovery and dispatch registry."""

import os
import importlib
import inspect
from tools.base_tool import Tool


class ToolRegistry:
    """Discovers and manages tools from the filesystem."""

    def __init__(self, agent):
        self.agent = agent
        self._tool_classes: dict[str, type[Tool]] = {}

    def discover_tools(self, tools_dir: str = None):
        """Scan the tools/ directory and register all Tool subclasses."""
        if tools_dir is None:
            tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

        for filename in sorted(os.listdir(tools_dir)):
            if not filename.endswith(".py") or filename.startswith("_") or filename in (
                "base_tool.py", "tool_registry.py", "__init__.py"
            ):
                continue

            module_name = f"tools.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Tool) and obj is not Tool and obj.name:
                        self._tool_classes[obj.name] = obj
            except Exception as e:
                print(f"[ToolRegistry] Failed to load {module_name}: {e}")

    def get_tool(self, name: str) -> Tool | None:
        """Instantiate and return a tool by name."""
        cls = self._tool_classes.get(name)
        if cls:
            return cls(self.agent)
        return None

    def get_tool_descriptions(self) -> str:
        """Generate markdown listing of all tools for the system prompt."""
        descriptions = []
        for name in sorted(self._tool_classes):
            tool = self._tool_classes[name](self.agent)
            descriptions.append(tool.get_prompt_description())
        return "\n".join(descriptions)

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._tool_classes.keys())
