"""Message trimmer extension — summarizes old messages when history gets too long."""

from extensions.base_extension import Extension


class MessageTrimmerExtension(Extension):
    name = "message_trimmer"
    enabled = True

    # Rough estimate: ~4 chars per token
    CHARS_PER_TOKEN = 4

    async def on_before_llm_call(self, agent, messages, **kwargs):
        """Trim history if it exceeds ~50% of the model's context window."""
        max_ctx = agent.config.chat_model.ctx_length
        history_budget = int(max_ctx * 0.5)  # 50% of context for history
        budget_chars = history_budget * self.CHARS_PER_TOKEN

        # Calculate total history size (skip system prompt at index 0)
        history_msgs = messages[1:]  # everything after system prompt
        total_chars = sum(len(m.get("content", "")) for m in history_msgs)

        if total_chars <= budget_chars:
            return None  # No trimming needed

        # Keep the system prompt and the most recent messages
        system_msg = messages[0]
        keep_recent = 6  # Keep last 6 messages untouched

        if len(history_msgs) <= keep_recent:
            return None  # Not enough to trim

        old_msgs = history_msgs[:-keep_recent]
        recent_msgs = history_msgs[-keep_recent:]

        # Summarize old messages into a compact form
        summary_parts = []
        for msg in old_msgs:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            # Truncate each old message
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"[{role}]: {content}")

        summary = (
            "[Earlier conversation summary]\n"
            + "\n".join(summary_parts[-10:])  # Keep last 10 summaries max
        )

        trimmed = [
            system_msg,
            {"role": "system", "content": summary},
        ] + recent_msgs

        return trimmed
