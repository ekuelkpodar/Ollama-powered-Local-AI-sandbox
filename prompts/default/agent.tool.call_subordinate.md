### call_subordinate
Delegate a task to a subordinate agent. The subordinate runs its own monologue loop and returns its result to you.
**Arguments:**
- `task` (required): Clear description of the task for the subordinate
- `system_prompt` (optional): Special instructions or role for the subordinate

**Example:**
```json
{"tool_name": "call_subordinate", "tool_args": {"task": "Research the top 5 sorting algorithms and summarize their time complexities"}}
```
