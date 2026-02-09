"""Extension manager — discovers and dispatches lifecycle hooks."""

import os
import importlib
import inspect
from extensions.base_extension import Extension


class ExtensionManager:
    """Discovers extension modules and dispatches hook calls."""

    def __init__(self, config):
        self.config = config
        self.extensions: list[Extension] = []

    def discover_extensions(self, extensions_dir: str | None = None):
        """Scan the builtin extensions directory and register them."""
        if extensions_dir is None:
            extensions_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "builtin"
            )

        if not os.path.isdir(extensions_dir):
            return

        for filename in sorted(os.listdir(extensions_dir)):
            if not filename.endswith(".py") or filename.startswith("_"):
                if filename == "__init__.py":
                    continue
                continue

            module_name = f"extensions.builtin.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Extension) and obj is not Extension and obj.name:
                        ext = obj(self.config)
                        if ext.enabled:
                            self.extensions.append(ext)
            except Exception as e:
                print(f"[ExtensionManager] Failed to load {module_name}: {e}")

    async def dispatch(self, hook_name: str, **kwargs):
        """
        Call the matching hook on all enabled extensions.
        For hooks that return data (before_llm_call), return the first non-None result.
        """
        method_name = f"on_{hook_name}"
        result = None

        for ext in self.extensions:
            if not ext.enabled:
                continue
            method = getattr(ext, method_name, None)
            if method is None:
                continue
            try:
                ret = await method(**kwargs)
                if ret is not None:
                    result = ret
            except Exception as e:
                print(f"[Extension '{ext.name}' hook '{hook_name}' error]: {e}")

        return result
