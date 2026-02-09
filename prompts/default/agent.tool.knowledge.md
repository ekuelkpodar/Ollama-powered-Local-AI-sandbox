### knowledge
Import documents into the knowledge base for semantic search.
**Arguments:**
- `action` (required): One of `import`, `status`
- `directory` (optional): Path to directory containing documents. Default: configured knowledge directory.

**Example:**
```json
{"tool_name": "knowledge", "tool_args": {"action": "import"}}
```
