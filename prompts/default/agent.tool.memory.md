### memory
Search, save, or delete memories in the persistent vector database.
**Arguments:**
- `action` (required): One of `save`, `search`, `delete`, `forget`
- `text` (required): The content to save, search query, or deletion query
- `area` (optional): Memory area — `main`, `fragments`, `solutions`, or `knowledge`. Default: `main`

**Example — Save:**
```json
{"tool_name": "memory", "tool_args": {"action": "save", "text": "User prefers Python over JavaScript", "area": "main"}}
```

**Example — Search:**
```json
{"tool_name": "memory", "tool_args": {"action": "search", "text": "user preferences"}}
```

**Example — Forget all in area:**
```json
{"tool_name": "memory", "tool_args": {"action": "forget", "area": "fragments"}}
```
