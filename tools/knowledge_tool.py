"""Knowledge tool — import documents into the FAISS knowledge base."""

from tools.base_tool import Tool
from agent.response import Response
from memory.memory_manager import MemoryManager
from memory.knowledge_import import KnowledgeImporter


class KnowledgeTool(Tool):
    name = "knowledge"
    description = "Import documents into the knowledge base for semantic search."

    async def execute(self, **kwargs) -> Response:
        action = kwargs.get("action", "import")
        directory = kwargs.get("directory", None)

        mm = self.agent.context.data.get("memory_manager")
        if mm is None:
            return Response(message="[Memory system not initialized]")

        if action == "import":
            importer = KnowledgeImporter(mm)
            stats = await importer.import_directory(directory)
            parts = [f"Knowledge import complete:"]
            parts.append(f"  Imported: {stats['imported']} files")
            parts.append(f"  Skipped (unchanged): {stats['skipped']} files")
            if stats["errors"]:
                parts.append(f"  Errors: {len(stats['errors'])}")
                for err in stats["errors"][:5]:
                    parts.append(f"    - {err}")
            return Response(message="\n".join(parts))

        elif action == "status":
            stats = await mm.get_stats()
            knowledge_count = stats.get("areas", {}).get("knowledge", 0)
            return Response(
                message=f"Knowledge base: {knowledge_count} chunks "
                f"(total memories: {stats['total']})"
            )

        return Response(
            message=f"[Error: Unknown action '{action}'. Use: import, status]"
        )
