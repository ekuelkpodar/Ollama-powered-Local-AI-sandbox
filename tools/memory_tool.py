"""Memory tool — search, save, delete, and forget memories via FAISS."""

from tools.base_tool import Tool
from agent.response import Response
from memory.memory_manager import MemoryManager


class MemoryTool(Tool):
    name = "memory"
    description = "Search, save, or delete memories in the persistent vector database."
    arg_schema = {
        "action": str,
        "text": str,
        "area": str,
        "namespace": str,
    }
    required_args: list[str] = []
    cacheable = True
    parallel_safe = True

    def should_cache(self, **kwargs) -> bool:
        action = kwargs.get("action", "search")
        return action in ("search", "query", "recall")

    def is_parallel_safe(self, **kwargs) -> bool:
        return self.should_cache(**kwargs)

    async def execute(self, **kwargs) -> Response:
        action = kwargs.get("action", "search")
        text = kwargs.get("text", "")
        area = kwargs.get("area", "main")
        namespace = kwargs.get("namespace") or self._default_namespace()

        mm = self._get_memory_manager()
        if mm is None:
            return Response(message="[Memory system not initialized]")

        if action == "save":
            if not text.strip():
                return Response(message="[Error: No text provided to save]")
            await mm.save(content=text, area=area, namespace=namespace)
            return Response(message=f"Saved to memory (namespace: {namespace}, area: {area})")

        elif action in ("search", "query", "recall"):
            if not text.strip():
                return Response(message="[Error: No search query provided]")
            results = await mm.search(
                query=text,
                area=area if area != "all" else None,
                k=self.agent.config.memory_recall_count,
                threshold=self.agent.config.memory_recall_threshold,
                namespace=namespace,
            )
            if not results:
                return Response(message="No relevant memories found.")
            lines = []
            for r in results:
                score = r.get("score", 0)
                importance = r.get("importance_score")
                content = r.get("content", "")
                r_area = r.get("area", "?")
                r_namespace = r.get("namespace", namespace)
                if importance is not None:
                    lines.append(
                        f"- [{r_namespace}/{r_area}] (score: {score:.2f}, "
                        f"importance: {importance:.2f}) {content}"
                    )
                else:
                    lines.append(f"- [{r_namespace}/{r_area}] (score: {score:.2f}) {content}")
            return Response(message="Memories found:\n" + "\n".join(lines))

        elif action in ("delete", "remove"):
            if not text.strip():
                return Response(message="[Error: No deletion query provided]")
            count = await mm.delete(
                query=text,
                area=area if area != "all" else None,
                namespace=namespace,
            )
            return Response(message=f"Deleted {count} memories in namespace '{namespace}'.")

        elif action == "forget":
            await mm.forget(area=area, namespace=namespace)
            return Response(message=f"Cleared all memories in namespace '{namespace}', area: {area}")

        else:
            return Response(
                message=f"[Error: Unknown memory action '{action}'. "
                "Use: save, search, delete, or forget]"
            )

    def _get_memory_manager(self) -> MemoryManager | None:
        """Get the memory manager from context."""
        return self.agent.context.data.get("memory_manager")

    def _default_namespace(self) -> str:
        namespaces = self.agent.config.memory.namespaces
        return namespaces[0] if namespaces else "default"
