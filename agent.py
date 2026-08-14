from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from typing import List, Any
import json
import time

from llm import get_provider, LLMProvider
from tools import TOOLS, Tools, resolve_workspace_path
from context_mgm import compact_context, prune_old_tool_outputs, build_initial_context, detect_stack
from telemetry import SessionMetrics, TaskMetrics
from prompts import SYSTEM_PROMPT, EXPLORE_PROMPT, EXTRACTION_PROMPT, ARCHITECTURE_SCHEMA
from report import generate_architecture_report
from ui import RoCodeUI

load_dotenv()
CLIENT = "gemini"
_provider: LLMProvider | None = None
_ui: RoCodeUI | None = None

def get_ui() -> RoCodeUI:
    global _ui
    if _ui is None:
        _ui = RoCodeUI()
    return _ui

#llama-3.3-70b-versatile
#qwen/qwen3.6-27b
#openai/gpt-oss-120b

#gemini-3.5-flash-lite
#gemini-3.1-flash-lite
MODEL = "gemini-3.1-flash-lite"
MAX_TURNS = 20
TOKEN_LIMIT = 4000
VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

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

def set_active_provider(provider_name: str, model_name: str) -> None:
    """Explicitly set the active provider and model for the session."""
    global CLIENT, MODEL, _provider
    CLIENT = provider_name
    MODEL = model_name
    _provider = get_provider(provider_name, model_name)

def call_structured_llm( history: list[dict], schema: dict, prompt: str, provider_state: dict | None = None, ) -> dict:
    """Delegate a structured JSON extraction call to the active provider."""
    return get_active_provider().call_structured(history, schema, prompt, provider_state)

def run_agent( user_message: str, tools: Tools, conversation_history: list = None, metrics: SessionMetrics = None, provider_state: dict[str, dict[str, Any]] | None = None, ) -> tuple[list[dict[str, Any]], SessionMetrics, dict[str, dict[str, Any]]]:
    ui = get_ui()
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
        conversation_history = [{"role": "system", "content": system_content}]

    conversation_history.append({"role": "user", "content": user_message})

    turns = 0
    last_usage = None
    spinner_text = "Thinking..."

    with ui.agent_turn() as ui:
        while turns < MAX_TURNS:

            # ── context compaction ──────────────────────────────────────
            if turns != 0 and last_usage is not None and get_prompt_tokens(last_usage) > TOKEN_LIMIT:
                ui.print_compact_context()
                try:
                    conversation_history = compact_context(
                        conversation_history,
                        provider_state=provider_state,
                    )
                    metrics.add_event(
                        f"Turn {metrics.completed_tasks + 1}: Compacted context window (tokens > {TOKEN_LIMIT})"
                    )
                except Exception as compact_err:
                    ui.print_compaction_error(compact_err)

            # ── LLM call with spinner + 429 retry ──────────────────────
            llm_error: Exception | None = None
            for attempt in range(3):
                try:
                    with ui.thinking(spinner_text):
                        assistant_message, last_usage = get_active_provider().call(
                            conversation_history,
                            TOOLS,
                            provider_state,
                        )
                    llm_error = None
                    break
                except Exception as exc:
                    from ui import _is_rate_limit
                    if _is_rate_limit(exc) and attempt < 2:
                        wait = 10.0 * (2 ** attempt)
                        ui.print_rate_limit_retry(wait, attempt + 1)
                        time.sleep(wait)
                        spinner_text = f"Retrying ({attempt + 2}/3)..."
                        llm_error = exc
                    else:
                        llm_error = exc
                        break

            if llm_error is not None:
                ui.print_llm_error(llm_error)
                task_metrics.update_error("llm")
                break

            task_metrics.update_llm(last_usage)
            ui.update_stats(task_metrics)

            # ── prose output ────────────────────────────────────────────
            assistant_content = assistant_message.get("content", "")
            if assistant_content:
                ui.print_prose(assistant_content)

            # ── parse tool calls ────────────────────────────────────────
            tool_calls = []
            for tc in assistant_message.get("tool_calls", []):
                function = tc.get("function", {})
                tool_calls.append({
                    "id":        tc.get("id"),
                    "name":      function.get("name", ""),
                    "arguments": function.get("arguments", "{}"),
                })

            conversation_history.append(assistant_message)

            # ── no tool calls → ReAct loop ends ─────────────────────────
            if not tool_calls:
                break

            # ── execute tool calls ──────────────────────────────────────
            for tool_call in tool_calls:
                tool_name = tool_call["name"]

                # recover from space-leaked args (e.g. "run_bash ls -la")
                if tool_name not in VALID_TOOL_NAMES:
                    if " " in tool_name and tool_name.split(" ", 1)[0] in VALID_TOOL_NAMES:
                        recovered, leaked = tool_name.split(" ", 1)
                        tool_name = recovered
                        tool_call["arguments"] = tool_call["arguments"] or leaked
                    else:
                        ui.print_invalid_tool(tool_name)
                        continue

                try:
                    tool_args = json.loads(tool_call["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                ui.print_tool_call(tool_name, tool_call["arguments"])

                t0 = time.perf_counter()
                try:
                    result = tools.execute_tool(tool_name, tool_args)
                except Exception as exc:
                    result = f"Tool execution failed: {exc}"
                duration = time.perf_counter() - t0

                success = not result.startswith((
                    "Error", "Tool execution failed",
                    "Execution failed", "Command failed", "User Denied Permission",
                ))

                ui.print_tool_result(tool_name, result)
                task_metrics.update_tool(tool_name, success=success, duration=duration)
                ui.update_stats(task_metrics)

                conversation_history.append({
                    "role":         "tool",
                    "tool_call_id": tool_call["id"],
                    "name":         tool_name,
                    "content":      result,
                })

            spinner_text = "Deciding next step..."
            turns += 1

        else:
            # while condition exhausted — MAX_TURNS reached
            ui.print_max_turns(MAX_TURNS)

    task_metrics.finish()
    metrics.merge(task_metrics, prompt=user_message)
    ui.print_task_footer(task_metrics)
    return conversation_history, metrics, provider_state

def main(root: Path):
    """Main chat loop."""
    ui = get_ui()
    ui.startup_banner()
    tools = Tools(root)
    metrics = SessionMetrics()

    initial_context = build_initial_context(root)
    stack_str = detect_stack(root)

    # ── interactive model selection ─────────────────────────────────────────
    chosen_client, chosen_model = ui.select_model()
    set_active_provider(chosen_client, chosen_model)

    system_content = SYSTEM_PROMPT + "\n\n" + initial_context
    conversation_history = [{"role": "system", "content": system_content}]
    provider_state = {}

    # ── startup banner ────────────────────────────────────────────────────
    ui.print_banner(root.name, stack_str, root, model_name=MODEL)

    while True:
        try:
            user_input = ui.prompt()
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            ui.console.print("[dim]Goodbye![/]")
            break

        if user_input.lower() in ("stats", "tokens"):
            ui.print_detailed_stats(metrics, MODEL)
            continue

        if user_input.lower() == "clear":
            conversation_history = None
            provider_state = {}
            ui.console.print("[dim]  Conversation cleared.[/]\n")
            continue

        is_explore = user_input.lower() == "explore"
        if is_explore:
            user_input = EXPLORE_PROMPT

        t_start = time.perf_counter()
        conversation_history, metrics, provider_state = run_agent(
            user_input,
            tools,
            conversation_history,
            metrics,
            provider_state,
        )

        if is_explore:
            elapsed = time.perf_counter() - t_start
            try:
                stats = {
                    "model":             MODEL,
                    "tool_calls":        metrics.tool_calls,
                    "total_tokens":      metrics.total_tokens,
                    "prompt_tokens":     metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "task_time":         elapsed,
                }
                md_path, html_path = generate_architecture_report(
                    conversation_history=conversation_history,
                    repo_root=root,
                    repo_name=root.name,
                    stats=stats,
                    call_fn=call_structured_llm,
                    schema=ARCHITECTURE_SCHEMA,
                    extraction_prompt=EXTRACTION_PROMPT,
                    provider_state=provider_state,
                )
                ui.print_explore_summary(
                    md_path=md_path,
                    html_path=html_path,
                    task_calls=metrics.tool_calls,
                    total_tokens=metrics.total_tokens,
                    elapsed=elapsed,
                )
            except Exception as e:
                ui.print_explore_error(e)

        ui.console.print()

    ui.print_session_summary(metrics, MODEL)


if __name__ == "__main__":
    repo_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        root = validate_workspace_root(repo_arg)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    main(root)
