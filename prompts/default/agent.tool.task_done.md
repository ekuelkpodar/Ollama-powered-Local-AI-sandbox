### task_done
Signal that the current task is complete (used by subordinate agents to return results to their superior).
**Arguments:**
- `text` (required): The result or summary to return to the superior agent

**Example:**
```json
{"tool_name": "task_done", "tool_args": {"text": "Completed the analysis. Here are the findings: ..."}}
```
