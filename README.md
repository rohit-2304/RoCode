<!-- 
  PLACEHOLDER: docs/assets/cli_screenshot.png
  Recommended Screenshot: "Streaming Prose"
  Capture a shot showing Markdown rendering while streaming (with bold/code blocks), representing a standard user interaction.
-->
![RoCode Terminal Screenshot](docs/assets/cli_screenshot.png)
# RoCode

> A search-first AI coding agent built on a hand-rolled ReAct loop with a 4-stage context budget pipeline. Explores codebases, manages its own token window, and generates architecture reports.

---

## 🎬 Demo

<!-- 
  PLACEHOLDER: docs/assets/rocode_demo.gif
  Recommended Screenshot: "Demo GIF"
  Capture a looping GIF showing RoCode exploring a repository, executing search tools, and streaming the output.
-->
![RoCode Terminal Demo](docs/assets/explore.gif)
*RoCode exploring a repository: search-first tool calls, streaming markdown output, and live token stats.*

<!-- PLACEHOLDER: Demo Video — replace the URL below with your YouTube / Loom / Vimeo link -->
> 🎥 **Demo Video**: [Watch on YouTube](https://www.youtube.com/watch?v=Sa8fIVYj2fQ)

<!-- 
  PLACEHOLDER: docs/assets/report_preview.png and docs/assets/mermaid_graph.png
  Recommended Screenshot: "Architecture Report"
  Capture a view of the generated `architecture_report.html` (the payoff) and a zoomed-in shot of the Mermaid dependency graph.
-->
| Interactive Architecture Report | Dependency Graph |
| :---: | :---: |
| ![Architecture Report HTML](docs/assets/report_preview.gif) | ![Mermaid Dependency Graph](docs/assets/mermaid_graph.png) |

---

## The Problem

Naive LLM tooling dumps raw source files into the prompt. This fails fast:

- **Token explosion** — a moderately sized repo hits the context ceiling within a few files.
- **Attention dilution** — LLMs lose precision on facts buried deep in a long context, leading to hallucinated function signatures and file paths.
- **Cost blowout** — ingesting redundant content drives up prompt token counts and latency.

RoCode is built around the opposite principle: **never read what you haven't searched for first**.

---

## ReAct Loop

The agent runs a standard **Reason → Act → Observe** loop, implemented from scratch in [`agent.py`](agent.py).

```
┌────────────────────────────────────────────────────────────┐
│                      run_agent()                           │
│                                                            │
│  ┌──────────┐    tool_calls     ┌─────────────────────┐   │
│  │  LLM     │ ───────────────▶  │  Tools.execute_tool │   │
│  │  call()  │                   │  (sandboxed)        │   │
│  └──────────┘ ◀───────────────  └─────────────────────┘   │
│       │         tool results                               │
│       │                                                    │
│       ▼  no tool_calls in response                         │
│    [ break — return to user ]                              │
│                                                            │
│  Max turns: MAX_TURNS = 20                                 │
└────────────────────────────────────────────────────────────┘
```

Each iteration of the loop:

1. **Check token budget** — if `prompt_tokens` from the last call exceed `TOKEN_LIMIT` (4 000), trigger context compaction before the next LLM call (see [Context Management](#-context-management) below).
2. **Call the LLM** — `provider.call(conversation_history, TOOLS, provider_state)` → returns a normalised `assistant` message dict and a usage object.
3. **Stream prose** — any `content` field in the response is rendered word-by-word as Rich Markdown.
4. **Parse tool calls** — the `tool_calls` array is extracted; if empty, the loop exits.
5. **Execute tools** — each tool call runs through `Tools.execute_tool()`, which is sandboxed to the workspace root. Results are appended as `role: tool` messages.
6. **Repeat** — updated history feeds back into the next LLM call.

The loop handles malformed tool arguments (bad JSON → empty dict), space-leaked tool names (e.g. `"read_file path/to/foo"` → recovered), and 429 rate-limit errors (exponential backoff, up to 3 retries).

---

## Tool Calling

RoCode uses the **OpenAI function-calling format** for tool definitions — a list of `{type: "function", function: {name, description, parameters}}` objects passed as the `tools` argument on every LLM call. This is converted to Gemini's `function_declarations` format internally by `GeminiProvider`.

### Available Tools

| Tool | Description | Notes |
|---|---|---|
| `read_file` | Read a file ≤ 150 lines in full | Head+tail truncation for larger files |
| `read_file_part` | Read an arbitrary line range with 1-based offset | Default limit: 300 lines |
| `write_file` | Write/overwrite a file | Creates parent dirs automatically |
| `edit_file` | Exact-string replace (single occurrence) | Fails on 0 or >1 matches |
| `list_directory` | List directory contents | Skips `node_modules`, `.venv`, etc. |
| `search_files` | Case-insensitive substring search across all files | Capped at 50 results; skips secrets |
| `run_bash` | Execute a shell command | Requires interactive Y/N approval; 60s timeout |
| `run_python` | Execute Python in a subprocess sandbox | AST-validated; 10s timeout |

All file-access tools resolve paths through `resolve_workspace_path()`, which rejects anything outside the repository root (path traversal guard).

### `run_python` Safety Sandbox

Before any Python code runs, it is passed through an **AST validator** (`validator.py`). The `SafetyValidator` walks the AST and blocks:

- **Dangerous imports**: `os`, `subprocess`, `sys`, `socket`, `pathlib`, `io`, `importlib`, `ctypes`, `multiprocessing`, `threading`, `asyncio`, `pickle`, `marshal`, `shelve`
- **Dangerous builtins**: `exec`, `eval`, `compile`, `open`, `input`, `__import__`, `getattr`, `setattr`
- **Dangerous attributes**: `__code__`, `__globals__`, `__builtins__`, `__subclasses__`

Validated code is written to a `tempfile` and run by `subprocess` with a stripped environment (`PATH` + `HOME=/tmp` only).

---

## Context Management

Context management is the core engineering challenge in a long-running agentic loop. Tool outputs — especially file reads and search results — can easily consume 80%+ of a context window. RoCode uses a **4-stage pipeline** to keep the prompt within budget across many turns.

### Stage 1 — Tool Output Truncation (at read time)

`read_file` hard-caps at **150 lines**. For files that exceed this limit it returns the first 75 lines and last 75 lines with an omission note in between:

```
[file head — 75 lines]
.... [N lines omitted - use read_file_part(path, offset, limit) ...]
[file tail — 75 lines]
```

This prevents any single file read from dominating the prompt. The agent is instructed to use `read_file_part` with explicit offsets for large files.

Similarly, `search_files` caps results at **50 matches** and `run_bash` stdout is truncated at **10,000 characters**.

### Stage 2 — Per-Turn Token Tracking

Every LLM call returns a usage object. `get_prompt_tokens(usage)` normalises across provider formats:

```python
getattr(usage, "prompt_tokens", None)       # OpenAI / Groq
or getattr(usage, "prompt_token_count", None) # Gemini
```

This value is stored as `last_usage` and checked at the top of the next iteration against `TOKEN_LIMIT = 4000`.

### Stage 3 — Stale Payload Pruning

Before compaction fires, `prune_old_tool_outputs(conversation_history, keep_recent=3)` scans the history and **truncates stale tool result messages in-place**:

```python
# Any tool result older than the last 3 is trimmed to 87 chars:
msg["content"] = f"{content[:87]}....truncated"
```

The corresponding assistant `tool_calls` arguments for those stale calls are also trimmed to 80 characters. This is lossless from a reasoning perspective — the agent already acted on those results — but recovers significant token headroom without touching recent context.

### Stage 4 — LLM-Powered Context Compaction

When `prompt_tokens > TOKEN_LIMIT`, `compact_context()` fires:

1. **Splits history** into `system_msg` (index 0), `recent_context` (last 4 turns), and `older_context` (everything in between).
2. **Serialises** `older_context` to a plain prose string via `serialize_for_summary()` — role-prefixed lines, tool results truncated to 400 chars, all provider-specific fields stripped.
3. **Summarises** the prose via a fast, cheap model (`llama-3.1-8b-instant` on Groq, or OpenAI as fallback) with the prompt:

   > *"Summarize the key findings, decisions, and file locations discovered in this conversation so far. Be concise — this summary replaces the raw history."*

4. **Rebuilds history** as `[system_msg, compaction_note, ...recent_context]`.
5. **Prunes provider state** — any `provider_state_id` keys no longer referenced by the new history are deleted to prevent memory leaks of Gemini raw `Content` objects.

If summarisation fails (e.g. missing Groq key), the older context is dropped silently and the agent continues with only recent turns.

```
Before compaction (12 turns):
  system | user | assistant | tool | ... (8 older turns) ... | user | assistant | tool | user | assistant | tool

After compaction:
  system | [compaction_note: summarised older turns] | user | assistant | tool | user | assistant | tool
```

---

## LLM Provider Abstraction

`llm.py` defines an abstract `LLMProvider` with two methods:

```python
def call(history, tools_schema, provider_state) -> (assistant_message, usage)
def call_structured(history, schema, prompt, provider_state) -> dict
```

Both return a **normalised assistant message dict**:

```python
{
    "role": "assistant",
    "content": "...",                 # prose text
    "tool_calls": [...],              # OpenAI-format list, optional
    "provider": "gemini",             # GeminiProvider only
    "provider_state_id": "gemini_..."  # GeminiProvider only
}
```

### GeminiProvider

Uses the native `google-genai` SDK. The key challenge with Gemini thinking models is that **thought parts are non-serialisable SDK objects** — re-serialising them loses the thought structure and breaks subsequent API calls.

Solution: raw `Content` objects and thought parts are stored **out-of-band** in `provider_state`, keyed by a UUID:

```python
provider_state["gemini_<uuid>"] = {
    "gemini_raw_content": candidate.content,   # raw SDK object
    "gemini_thought_parts": [...],
}
assistant_message["provider_state_id"] = "gemini_<uuid>"
```

On the next turn, `_format_history()` retrieves the raw `Content` by key and passes it verbatim to the API, bypassing re-serialisation entirely.

Tool schema conversion: OpenAI's `{type: "function", function: {name, parameters}}` is converted to Gemini's `[{function_declarations: [{name, description, parameters}]}]` by `_convert_tools()`.

### OpenAICompatProvider

Wraps the `openai` SDK. Supports Groq, OpenRouter, and OpenAI by swapping `base_url` and `api_key`. Uses `tool_choice="auto"` and `temperature=0`.

`call_structured()` uses `response_format: {type: "json_schema", json_schema: {name, schema}}` for constrained JSON output.

### Factory

```python
get_provider("gemini", "gemini-3.1-flash-lite")  # → GeminiProvider
get_provider("groq", "llama-3.3-70b-versatile")   # → OpenAICompatProvider (Groq base_url)
get_provider("openrouter", "...")                  # → OpenAICompatProvider (OpenRouter base_url)
get_provider("openai", "...")                      # → OpenAICompatProvider (default)
```

---

## Explore Command & Report Generation

Typing `explore` substitutes the user message with a structured `EXPLORE_PROMPT` that instructs the agent to map the full architecture — stack, entry points, module boundaries, and request flows.

After the ReAct loop completes, a **second LLM call** is made via `call_structured_llm()` (not the ReAct loop) to extract a JSON object conforming to `ARCHITECTURE_SCHEMA`:

```
conversation_history (full explore context)
  + EXTRACTION_PROMPT (final user turn)
  → call_structured() with response_mime_type=application/json (Gemini)
    or response_format: json_schema (OpenAI)
  → validated against ARCHITECTURE_SCHEMA
  → rendered to .rocode/ARCHITECTURE.md and .rocode/architecture_report.html
  → rendered to AGENTS.md in the repo root
```

`validate_report_data()` enforces required keys and types before anything is rendered, so the renderers never receive malformed data.

### AGENTS.md Generation

After every `explore`, RoCode writes an **`AGENTS.md`** to the **root of the explored repository**. This file is a structured guide for downstream AI agents — Claude, Codex, Antigravity, etc. — covering the repo's overview, tech stack, entry points, module map, request flows, and extension patterns. It is generated from the same structured data as the architecture report, with no additional LLM call.

```
explore
  → ReAct loop
  → call_structured_llm() → arch_data dict
  → generate_architecture_report() → .rocode/ARCHITECTURE.md + .rocode/architecture_report.html
  → generate_agents_md()           → AGENTS.md (repo root)
```

Re-running `explore` overwrites all three files with fresh data.

### Report Rendering

`report.py` contains pure rendering functions:

- `render_module_diagram(modules)` — Mermaid `graph TD`, capped at 15 nodes (highest degree kept) to avoid diagram overflow.
- `render_flow_diagram(flow)` — Mermaid `sequenceDiagram`, participants collected in first-seen order.
- `render_markdown()` — assembles the full `.md` report.
- `render_html()` — standalone dark-mode HTML with CDN `mermaid.js@11` and no external CSS dependencies.

---

## Initial Context (Zero-LLM Bootstrap)

Before the first LLM call, `build_initial_context(root)` assembles a context block locally — no API call needed:

| Signal | Source |
|---|---|
| Directory tree (depth 3) | `os.walk` with `IGNORED_DIRS` filter |
| Stack detection | Presence of `package.json`, `requirements.txt`, `go.mod`, `Cargo.toml`, `Dockerfile`, etc. |
| File extension breakdown | Count of top-6 extensions across all non-ignored files |
| Entry point candidates | Presence of `main.py`, `app.py`, `server.js`, `index.ts`, `main.go`, etc. |
| Git metadata | `git branch --show-current` + `git log -1 --oneline` |
| README excerpt | Path noted if present |

This is prepended to the system prompt so the LLM starts with structural awareness without reading a single source file.

---

## Quickstart

**Prerequisites**: Python 3.10+, an API key for Google Gemini or Groq.

```bash
git clone https://github.com/rohit/RoCode.git
cd RoCode
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```bash
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key        # also used for context compaction
OPENROUTER_API_KEY=...            # optional
OPENAI_API_KEY=...                # optional
```

Run on any local repository:
```bash
python agent.py /path/to/target/repo
```

---

## CLI Commands

| Command | Effect |
|---|---|
| `explore` | Full architectural discovery → writes `.rocode/ARCHITECTURE.md`, `.rocode/architecture_report.html`, `AGENTS.md` |
| `stats` / `tokens` | Per-turn token breakdown, tool usage table, compaction event log |
| `clear` | Reset conversation history and provider state |
| `quit` / `exit` | Session summary panel and exit |

### Example Session

```text
rocode › What handles database migrations in this repo?

  → search_files  pattern="migration"
  ✓ found 4 matches
  → read_file_part  path="alembic/env.py"  offset=1  limit=40
  ✓ 40 lines

Database migrations are managed by Alembic in `alembic/env.py`.

  2 tool calls · 1,420 tokens · 3s · $0.00 (free tier)
```

---

## Module Map

| File | Responsibility |
|---|---|
| [`agent.py`](agent.py) | ReAct loop, CLI chat, explore trigger, provider lifecycle |
| [`llm.py`](llm.py) | `LLMProvider` abstract base, `GeminiProvider`, `OpenAICompatProvider`, factory |
| [`tools.py`](tools.py) | Tool schema definitions (`TOOLS`), `Tools` class, workspace path guard |
| [`context_mgm.py`](context_mgm.py) | 4-stage context pipeline, initial context builder, stack detection |
| [`prompts.py`](prompts.py) | `SYSTEM_PROMPT`, `EXPLORE_PROMPT`, `EXTRACTION_PROMPT`, `ARCHITECTURE_SCHEMA` |
| [`report.py`](report.py) | Mermaid renderers, markdown/HTML report, `AGENTS.md` renderer |
| [`telemetry.py`](telemetry.py) | `TaskMetrics`, `SessionMetrics` dataclasses, token/timing aggregation |
| [`ui.py`](ui.py) | Rich terminal output (one-way: `agent.py` → `ui.py`, no callbacks) |
| [`executor.py`](executor.py) | Python sandbox subprocess runner |
| [`validator.py`](validator.py) | AST-based safety validator for `run_python` |

---

## Design Decisions & Tradeoffs

**Search-first over full ingestion** — Multiple tool round-trips add latency per turn, but the token savings across a session are substantial. On a 10,000-line codebase, reading only the files actually relevant to a question uses ~5–10% of the tokens a full-ingest approach would consume.

**Lossy stale-payload pruning over perfect history** — Verbatim tool outputs from 8 turns ago are not needed for current reasoning. Trimming them to 87 characters recovers meaningful headroom. The agent already incorporated those results into its reasoning; losing the raw bytes doesn't degrade answer quality.

**Out-of-band Gemini provider state** — Serialising Gemini `Content` objects and re-constructing them from dicts loses thought parts, breaking thinking models. Storing the raw SDK objects by reference avoids the round-trip cost entirely. The tradeoff is a parallel state dict that must be pruned alongside history.

**Compaction via a fast/cheap secondary model** — Using `llama-3.1-8b-instant` (Groq free tier) for summarisation keeps compaction latency low (~1–2s) without burning quota on the primary reasoning model.

**JSON Schema enforcement before rendering** — `validate_report_data()` runs before any renderer is called. A bad schema extraction fails loudly at the validation step rather than silently producing a malformed report.

---

## Roadmap

- [ ] **Multi-File Atomic Refactoring**: Coordinated edits with automated rollback on failure.
- [ ] **Docker Sandbox Integration**: `run_bash` inside an isolated container, no `input()` gate needed.
- [ ] **Vector-Hybrid Indexing**: Combine lexical `search_files` with local embeddings for repos >50,000 files.
- [ ] **CI/CD Architecture Guard**: Run `explore` in GitHub Actions and diff the output to detect structural drift.

---

## Tech Stack

- **Core**: Python 3.10+
- **Terminal UI**: [Rich](https://github.com/Textualize/rich)
- **LLM SDKs**: `google-genai` (Gemini), `openai` (Groq & OpenAI-compatible endpoints)
- **Environment**: `python-dotenv`, `pyfiglet`
- **Diagram Engine**: [Mermaid.js](https://mermaid.js.org/) (client-side, CDN)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
