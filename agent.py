from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from typing import List, Any
import json
import time

from llm import get_provider, LLMProvider
from tools import TOOLS, Tools, resolve_workspace_path
from context_mgm import compact_context, prune_old_tool_outputs, build_initial_context
from telemetry import SessionMetrics, TaskMetrics

load_dotenv()
CLIENT = "gemini"
_provider: LLMProvider | None = None

#llama-3.3-70b-versatile
#qwen/qwen3.6-27b
#openai/gpt-oss-120b

#gemini-3.5-flash-lite
#gemini-3.1-flash-lite
MODEL = "gemini-3.1-flash-lite"
MAX_TURNS = 20
TOKEN_LIMIT = 4000
VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


SYSTEM_PROMPT = """You are RoCode, an AI engineer specializing in understanding unfamiliar 
codebases. Your job is to build an accurate mental model of a repository — 
its structure, stack, architecture, and request flow — before answering 
questions or making suggestions.

EXPLORATION
Explore before concluding: gather evidence from multiple files before 
explaining anything. Never read files at random — inspect structure, find 
entry points and config files, then search_files() for relevant symbols. 
Only read_file a whole file if it's small; otherwise use read_file_part() 
on the section you need. Before every tool call, know why you're calling 
it (e.g. "I need to locate auth code" → search_files("auth"), not "I'll 
read every file"). Never hallucinate — if something isn't verifiable from 
the repo, say so explicitly.

BUILDING THE MENTAL MODEL
As you explore, continuously infer and track:
- Stack: check package.json/pyproject.toml/requirements.txt/go.mod/Dockerfile 
  etc. to identify language, framework (FastAPI, Django, React, Express...), 
  and major dependencies.
- Entry points: where execution starts (main(), app factory, server bootstrap).
- Structure: major modules/directories and what each owns.
- Dependencies: which modules import/call which — build this incrementally 
  as a mental graph, don't assume it from folder names alone.
- Request flow: for backend/API repos, trace an example request from entry 
  point → routing → handler → business logic → persistence, citing the 
  actual files and functions involved at each step.

FILE SUGGESTIONS
Before opening a file, briefly state why it's likely relevant. When 
answering a question, list the 2-4 most relevant files as a suggestion 
before diving in, so the reasoning is visible.

ANSWERING QUESTIONS
Locate relevant files first (search before reading), then answer with 
evidence, citing file paths and line numbers. For "how do I add X" 
questions, find an existing analogous pattern in the repo and describe it 
rather than inventing a new one.

TOOLS
Use ONLY the tools provided, with exact registered names. All arguments go 
in the arguments JSON — never inside the function name. When you learn 
something durable about the repo (stack, structure, key file locations), 
call update_context_md to save it for future sessions.

OUTPUT
When asked to explain architecture or request flow, structure your answer 
as: Summary (1-2 sentences) → Evidence (files/functions, with paths) → 
Flow or relationships (if applicable). Keep answers grounded in what you 
actually read, not general framework knowledge.
"""

EXPLORE_PROMPT = """Give me a full architectural overview of this repository. Explore 
systematically: identify the stack and entry points, map the major 
modules and what each owns, and trace how a typical request or 
execution flows through the codebase from entry point to persistence 
(if applicable). Note any important patterns, conventions, or things 
that stand out. Structure your answer as: Overview, Tech Stack, Key 
Modules, Request/Execution Flow, Notable Observations."""

def validate_workspace_root(path: str) -> Path:
    """Validate and resolve the repository root. Call once at startup."""
    root = Path(path).resolve()

    if not root.exists():
        raise ValueError(f"Repository path does not exist: {path}")
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {path}")

    # guard against pointing the agent at something dangerously broad
    dangerous_roots = {Path.home(), Path("/"), Path.home().parent}
    if root in dangerous_roots:
        raise ValueError(
            f"Refusing to use '{root}' as workspace root — too broad. "
            f"Point at a specific project directory."
        )

    return root

def get_prompt_tokens(usage: Any) -> int:
    return (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "prompt_token_count", None)
        or 0
    )

def get_active_provider() -> LLMProvider:
    """Lazily construct the provider once; raise immediately if deps are missing."""
    global _provider
    if _provider is None:
        _provider = get_provider(CLIENT, MODEL)
    return _provider

def run_agent( user_message: str, tools: Tools, conversation_history: list = None, metrics: SessionMetrics = None, provider_state: dict[str, dict[str, Any]] | None = None, ) -> tuple[list[dict[str, Any]], SessionMetrics, dict[str, dict[str, Any]]]:
    root = tools.root

    if metrics is None:
        metrics = SessionMetrics()
    if provider_state is None:
        provider_state = {}

    task_metrics = TaskMetrics()
    task_metrics.start()

    if conversation_history is None:
        initial_context = build_initial_context(root)
        system_content = SYSTEM_PROMPT + "\n\n" + initial_context
        conversation_history = [{
            "role": "system",
            "content": system_content
        }]

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    turns = 0
    last_usage = None

    while turns < MAX_TURNS:
        if (
            turns != 0
            and last_usage is not None
            and get_prompt_tokens(last_usage) > TOKEN_LIMIT
        ):
            print("Compressing Context Window.....")
            try:
                conversation_history = compact_context(
                    conversation_history,
                    provider_state=provider_state,
                )
            except Exception as compact_err:
                print(f"[Warning] Context compaction failed ({compact_err}); continuing with full history.")

        try:
            assistant_message, last_usage = get_active_provider().call(
                conversation_history,
                TOOLS,
                provider_state,
            )
            assistant_content = assistant_message.get("content", "")
            print(assistant_content)

            task_metrics.update_llm(last_usage)

        except Exception as e:
            print(repr(e))
            task_metrics.update_error("llm")
            break

        tool_calls = []

        for tc in assistant_message.get("tool_calls", []):
            function = tc.get("function", {})
            tool_calls.append({
                "id": tc.get("id"),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            })

        conversation_history.append(assistant_message)

        # if no tool calls end the ReAct loop
        if not tool_calls:
            print("-" * 10 + "x" + "-" * 10)
            task_metrics.finish()
            task_metrics.display()
            metrics.merge(task_metrics)
            return conversation_history, metrics, provider_state

        # if tool calls, execute them one by one
        for tool_call in tool_calls:

            tool_name = tool_call["name"]

            # check if tool all is valid
            if tool_name not in VALID_TOOL_NAMES:
                if (" " in tool_name and tool_name.split(" ", 1)[0] in VALID_TOOL_NAMES):
                    recovered_name, leaked_args = tool_name.split(" ", 1)
                    tool_name = recovered_name
                    tool_call["arguments"] = (
                        tool_call["arguments"] or leaked_args
                    )
                else:
                    print(f"Skipping malformed tool call: {tool_name!r}")
                    continue

            try:
                tool_args = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            print(f"\nUsing tool: {tool_name}")

            tool_started_at = time.perf_counter()

            try:
                result = tools.execute_tool(tool_name, tool_args)
            except Exception as e:
                result = f"Tool execution failed: {e}"

            tool_duration = time.perf_counter() - tool_started_at

            success = not result.startswith((
                "Error",
                "Tool execution failed",
                "Execution failed",
                "Command failed",
                "User Denied Permission",
            ))

            task_metrics.update_tool(
                tool_name,
                success=success,
                duration=tool_duration,
            )

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": result,
            })

        turns += 1

    task_metrics.finish()
    task_metrics.display()
    metrics.merge(task_metrics)

    return conversation_history, metrics, provider_state

def main(root : Path):
    """Main chat loop."""
    print("=" * 60)
    print("RoCode Phase 1: Minimum Viable Coding Agent")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60)
    print()
    tools = Tools(root)
    metrics = SessionMetrics()

    initial_context = build_initial_context(root)

    system_content = SYSTEM_PROMPT + "\n\n" + initial_context
    conversation_history = [{"role": "system", "content": system_content}]
    provider_state = {}
    print(f"Loaded repository: {root}")
    print("Ask a question, or type 'explore' for a full architecture overview.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_history = None  # triggers system msg + initial context rebuild in run_agent
            provider_state = {}
            print("Conversation cleared.\n")
            continue
        if user_input.lower() == 'explore':
            user_input = EXPLORE_PROMPT

        print("\nAgent: ", end="", flush=True)
        conversation_history, metrics, provider_state = run_agent(
            user_input,
            tools,
            conversation_history,
            metrics,
            provider_state,
        )
        print()
    metrics.display_metrics_total()

if __name__ == "__main__":
    repo_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        root = validate_workspace_root(repo_arg)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    main(root)
