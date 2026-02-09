"""Prompt template engine with {{variable}} substitution and {{include:file}} directives."""

import os
import re
from agent.exceptions import PromptTemplateError


class PromptTemplateEngine:
    """Renders markdown prompt templates with variable substitution and includes."""

    def __init__(self, prompts_dir: str, profile: str = "default"):
        self.base_dir = os.path.join(prompts_dir, profile)
        if not os.path.isdir(self.base_dir):
            raise PromptTemplateError(f"Prompts directory not found: {self.base_dir}")

    def render(self, template_name: str, variables: dict) -> str:
        """Load and render a template file with variable substitution."""
        path = os.path.join(self.base_dir, template_name)
        if not os.path.exists(path):
            raise PromptTemplateError(f"Template not found: {path}")

        with open(path, "r") as f:
            template = f.read()

        # Resolve {{include:filename.md}} directives (recursive)
        template = self._resolve_includes(template, variables, depth=0)

        # Substitute {{variable_name}} placeholders
        for key, value in variables.items():
            template = template.replace("{{" + key + "}}", str(value))

        return template

    def _resolve_includes(self, template: str, variables: dict, depth: int) -> str:
        """Recursively resolve {{include:filename.md}} directives."""
        if depth > 10:
            raise PromptTemplateError("Include depth exceeded 10 — possible circular reference")

        pattern = r"\{\{include:([^}]+)\}\}"

        def replacer(match):
            included_name = match.group(1).strip()
            included_path = os.path.join(self.base_dir, included_name)
            if not os.path.exists(included_path):
                return f"[Missing template: {included_name}]"
            with open(included_path, "r") as f:
                content = f.read()
            return self._resolve_includes(content, variables, depth + 1)

        return re.sub(pattern, replacer, template)

    def render_string(self, template_str: str, variables: dict) -> str:
        """Render a template string (not from file) with variable substitution."""
        for key, value in variables.items():
            template_str = template_str.replace("{{" + key + "}}", str(value))
        return template_str

    def list_templates(self, pattern: str = "*.md") -> list[str]:
        """List all template files matching a glob pattern."""
        import glob
        return sorted(glob.glob(os.path.join(self.base_dir, pattern)))

    def get_tool_prompt(self, tool_name: str) -> str:
        """Load a tool-specific prompt template if it exists."""
        filename = f"agent.tool.{tool_name}.md"
        path = os.path.join(self.base_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        return ""
