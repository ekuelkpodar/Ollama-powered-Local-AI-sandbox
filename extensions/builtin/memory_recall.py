"""Memory recall extension — auto-injects relevant memories before each LLM call."""

from extensions.base_extension import Extension


class MemoryRecallExtension(Extension):
    name = "memory_recall"
    enabled = True

    async def on_message_loop_prompts_before(self, agent, **kwargs):
        """Search FAISS for relevant memories and inject into agent data."""
        if not self.config.memory_recall_enabled:
            return

        mm = agent.context.data.get("memory_manager")
        if mm is None:
            return

        # Get the last user message as the search query
        query = ""
        for msg in reversed(agent.history):
            if msg["role"] == "user":
                query = msg["content"]
                break

        if not query:
            return

        try:
            results = await mm.search(
                query=query,
                k=self.config.memory_recall_count,
                threshold=self.config.memory_recall_threshold,
            )

            if results:
                lines = ["## Recalled Memories\n"]
                for r in results:
                    area = r.get("area", "?")
                    content = r.get("content", "")
                    score = r.get("score", 0)
                    lines.append(f"- [{area}] (relevance: {score:.2f}) {content}")
                agent.data["memory_context"] = "\n".join(lines)
            else:
                agent.data["memory_context"] = ""
        except Exception:
            agent.data["memory_context"] = ""
