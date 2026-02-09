### code_execution
Execute code in a persistent terminal session. Sessions persist across calls, so variables and state are maintained.
**Arguments:**
- `runtime` (required): One of `python`, `shell`, or `node`
- `code` (required): The code to execute

**Example — Python:**
```json
{"tool_name": "code_execution", "tool_args": {"runtime": "python", "code": "import os\nprint(os.listdir('.'))"}}
```

**Example — Shell:**
```json
{"tool_name": "code_execution", "tool_args": {"runtime": "shell", "code": "ls -la"}}
```
