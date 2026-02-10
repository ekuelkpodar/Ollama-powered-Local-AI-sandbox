# Local Ollama Agents

A local multi-agent AI framework powered entirely by Ollama — monologue loop, multi-agent delegation, FAISS memory, extensible tools — running 100% locally with no cloud APIs.

## Features

- **Monologue Loop** — agents reason, use tools, observe results, and loop until they have an answer
- **Multi-Agent Hierarchy** — Agent 0 delegates to subordinate agents for complex tasks
- **Code Execution** — persistent Python, Shell, and Node.js sessions
- **FAISS Memory** — vector-based persistent memory with semantic search
- **Knowledge Import** — RAG pipeline for .txt, .md, .pdf, .csv, .html, .json documents
- **Extension System** — lifecycle hooks for memory recall, history trimming, logging
- **Web UI** — Flask + Alpine.js with real-time SSE streaming
- **CLI Mode** — interactive REPL with colored streaming output
- **Fully Local** — direct HTTP to Ollama, local sentence-transformers embeddings

## Architecture

### System Overview

```mermaid
graph TB
    User([fa:fa-user User])

    subgraph Interfaces["Interfaces"]
        CLI["CLI REPL<br/><i>run_cli.py</i>"]
        WebUI["Web UI<br/><i>Flask + Alpine.js</i>"]
    end

    subgraph Core["Core Agent Engine"]
        AC["AgentContext<br/><i>Session container</i>"]
        A0["Agent 0<br/><i>Top-level agent</i>"]
        ML["Monologue Loop<br/><i>Reason → Tool → Loop</i>"]
        TP["Tool Parser<br/><i>JSON extraction</i>"]
    end

    subgraph LLM["LLM Layer"]
        OC["OllamaClient<br/><i>Direct HTTP aiohttp</i>"]
        Ollama["Ollama Server<br/><i>localhost:11434</i>"]
    end

    subgraph Tools["Tool System"]
        TR["Tool Registry<br/><i>Auto-discovery</i>"]
        CE["code_execution<br/><i>Python / Shell / Node</i>"]
        CS["call_subordinate<br/><i>Multi-agent delegation</i>"]
        MT["memory<br/><i>Save / Search / Delete</i>"]
        KT["knowledge<br/><i>Document import</i>"]
        RT["response<br/><i>Final answer</i>"]
        TD["task_done<br/><i>Subordinate completion</i>"]
    end

    subgraph Memory["Memory System"]
        MM["MemoryManager<br/><i>High-level API</i>"]
        FAISS["FAISS IndexFlatIP<br/><i>Vector similarity</i>"]
        EMB["EmbeddingEngine<br/><i>all-MiniLM-L6-v2</i>"]
        KI["KnowledgeImporter<br/><i>RAG pipeline</i>"]
    end

    subgraph Extensions["Extension System"]
        EM["ExtensionManager<br/><i>Hook dispatch</i>"]
        MR["memory_recall<br/><i>Auto-inject memories</i>"]
        MTR["message_trimmer<br/><i>History compression</i>"]
        OL["output_logger<br/><i>JSON logging</i>"]
    end

    subgraph Prompts["Prompt System"]
        PE["TemplateEngine<br/><i>{{variable}} substitution</i>"]
        MD["Markdown Templates<br/><i>prompts/default/*.md</i>"]
    end

    User -->|message| CLI
    User -->|message| WebUI
    CLI --> AC
    WebUI -->|SSE streaming| AC
    AC --> A0
    A0 --> ML
    ML --> PE
    PE --> MD
    ML -->|stream| OC
    OC -->|HTTP POST /api/chat| Ollama
    Ollama -->|chunks| OC
    OC -->|text| TP
    TP -->|tool call| TR
    TR --> CE
    TR --> CS
    TR --> MT
    TR --> KT
    TR --> RT
    TR --> TD
    CS -->|spawn| A0
    MT --> MM
    KT --> KI
    KI --> MM
    MM --> EMB
    MM --> FAISS
    ML -.->|hooks| EM
    EM --> MR
    EM --> MTR
    EM --> OL
    MR --> MM
    RT -->|break_loop=true| ML

    style Core fill:#1a1b26,stroke:#7aa2f7,color:#c0caf5
    style LLM fill:#1a1b26,stroke:#9ece6a,color:#c0caf5
    style Tools fill:#1a1b26,stroke:#e0af68,color:#c0caf5
    style Memory fill:#1a1b26,stroke:#bb9af7,color:#c0caf5
    style Extensions fill:#1a1b26,stroke:#f7768e,color:#c0caf5
    style Prompts fill:#1a1b26,stroke:#7dcfff,color:#c0caf5
    style Interfaces fill:#1a1b26,stroke:#73daca,color:#c0caf5
```

### Monologue Loop (Core Algorithm)

```mermaid
flowchart TD
    START([User Message]) --> HISTORY[Append to history]
    HISTORY --> HOOK1{{"hook: message_loop_start"}}
    HOOK1 --> RECALL{{"hook: memory_recall<br/><i>FAISS search → inject context</i>"}}
    RECALL --> BUILD[Build system prompt<br/><i>Templates + tools + memory</i>]
    BUILD --> HOOK2{{"hook: before_llm_call<br/><i>Trim history if needed</i>"}}
    HOOK2 --> LLM["Call Ollama /api/chat<br/><i>Stream response chunks</i>"]
    LLM --> PARSE{Extract tool JSON<br/>from response}

    PARSE -->|No tool found| COUNT{3 iterations<br/>without tools?}
    COUNT -->|No| BUILD
    COUNT -->|Yes| NUDGE[Inject hint:<br/>'Use the response tool'] --> BUILD

    PARSE -->|Tool found| EXEC["Execute tool"]
    EXEC --> CHECK{break_loop?}
    CHECK -->|No| RESULT[Append tool result<br/>to history] --> BUILD
    CHECK -->|Yes| DONE([Return final response])

    ITER{Max iterations?} -.->|Yes| FALLBACK([Safety fallback])
    BUILD -.-> ITER

    style START fill:#7aa2f7,stroke:#7aa2f7,color:#1a1b26
    style DONE fill:#9ece6a,stroke:#9ece6a,color:#1a1b26
    style LLM fill:#e0af68,stroke:#e0af68,color:#1a1b26
    style EXEC fill:#bb9af7,stroke:#bb9af7,color:#1a1b26
    style FALLBACK fill:#f7768e,stroke:#f7768e,color:#1a1b26
```

### Multi-Agent Delegation

```mermaid
sequenceDiagram
    participant U as User
    participant A0 as Agent 0
    participant O as Ollama
    participant A1 as Agent 1 (Subordinate)

    U->>A0: Send task
    loop Monologue Loop
        A0->>O: Chat request
        O-->>A0: Stream response
        A0->>A0: Extract tool call
    end
    Note over A0: Decides to delegate
    A0->>A1: call_subordinate(task)
    loop Subordinate Monologue Loop
        A1->>O: Chat request
        O-->>A1: Stream response
        A1->>A1: Execute tools
    end
    A1-->>A0: task_done(result)
    A0->>A0: Continue reasoning with result
    A0-->>U: response(final answer)
```

### Project Structure

```mermaid
graph LR
    subgraph Root["Project Root"]
        RC["run_cli.py"]
        RW["run_web.py"]
        CF["config.json"]
    end

    subgraph agent/["agent/"]
        AG["agent.py<br/><i>Agent + monologue loop</i>"]
        ACT["agent_context.py<br/><i>Session container</i>"]
        MD["models.py<br/><i>OllamaClient</i>"]
        CO["config.py<br/><i>Configuration</i>"]
        RS["response.py<br/><i>Dataclasses</i>"]
        EX["exceptions.py"]
    end

    subgraph tools/["tools/"]
        BT["base_tool.py"]
        TRG["tool_registry.py"]
        CE2["code_execution.py"]
        CS2["call_subordinate.py"]
        MT2["memory_tool.py"]
        KT2["knowledge_tool.py"]
        RT2["response_tool.py"]
        TD2["task_done.py"]
    end

    subgraph memory/["memory/"]
        EM2["embeddings.py<br/><i>sentence-transformers</i>"]
        FS["faiss_store.py<br/><i>FAISS wrapper</i>"]
        MM2["memory_manager.py"]
        KI2["knowledge_import.py<br/><i>RAG pipeline</i>"]
    end

    subgraph prompts/["prompts/"]
        TE["template_engine.py"]
        DF["default/<br/><i>8 markdown templates</i>"]
    end

    subgraph extensions/["extensions/"]
        BEX["base_extension.py"]
        EMG["extension_manager.py"]
        BIN["builtin/<br/><i>3 hook extensions</i>"]
    end

    subgraph web/["web/"]
        AP["app.py<br/><i>Flask factory</i>"]
        RTS["routes/<br/><i>5 API modules</i>"]
        ST["static/<br/><i>HTML + CSS + JS</i>"]
    end

    subgraph cli/["cli/"]
        CA["cli_app.py<br/><i>REPL interface</i>"]
    end

    RC --> CA
    RW --> AP

    style Root fill:#24283b,stroke:#7aa2f7,color:#c0caf5
    style agent/ fill:#24283b,stroke:#7aa2f7,color:#c0caf5
    style tools/ fill:#24283b,stroke:#e0af68,color:#c0caf5
    style memory/ fill:#24283b,stroke:#bb9af7,color:#c0caf5
    style prompts/ fill:#24283b,stroke:#7dcfff,color:#c0caf5
    style extensions/ fill:#24283b,stroke:#f7768e,color:#c0caf5
    style web/ fill:#24283b,stroke:#73daca,color:#c0caf5
    style cli/ fill:#24283b,stroke:#73daca,color:#c0caf5
```

## Quick Start

### 1. Install Ollama

```bash
# macOS
brew install ollama

# or download from https://ollama.com
```

### 2. Pull a model

```bash
ollama pull llama3.2
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

**CLI mode:**
```bash
python run_cli.py
```

**Web UI:**
```bash
python run_web.py
# Open http://localhost:5000
```

### Docker (Optional)

Run the full stack with Ollama + agent in containers:

```bash
docker compose up --build
```

The agent will be available at `http://localhost:5000`. The Ollama service is exposed at `http://localhost:11434`.

## Configuration

Edit `config.json` to change models, temperatures, context lengths, and memory settings:

```json
{
  "chat_model": {
    "model_name": "llama3.2",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "ctx_length": 8192
  },
  "utility_model": {
    "model_name": "llama3.2",
    "temperature": 0.3,
    "ctx_length": 4096
  },
  "embedding_model": "all-MiniLM-L6-v2",
  "max_monologue_iterations": 25,
  "memory_recall_enabled": true,
  "memory_recall_threshold": 0.6,
  "memory_recall_count": 5
}
```

## How It Works

1. You send a message to Agent 0
2. Agent enters the **monologue loop**
3. System prompt is assembled from markdown templates + tool descriptions + recalled memories
4. Ollama generates a streaming response
5. Tool calls are extracted from JSON blocks in the response
6. Tools execute (code, memory search, subordinate delegation, etc.)
7. Results feed back into the loop
8. Loop breaks when the agent calls the `response` tool with its final answer

## Tools

| Tool | Description |
|------|-------------|
| `response` | Deliver final answer to user |
| `code_execution` | Run Python/Shell/Node.js in persistent sessions |
| `call_subordinate` | Delegate tasks to subordinate agents |
| `memory` | Save, search, delete, forget memories |
| `knowledge` | Import documents into the knowledge base |
| `task_done` | Subordinate signals completion |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (direct HTTP) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (faiss-cpu, IndexFlatIP) |
| Backend | Python 3.11+, Flask |
| Frontend | Alpine.js, vanilla CSS |
| Streaming | Server-Sent Events (SSE) |

## License

MIT
