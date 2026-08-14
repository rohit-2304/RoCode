"""
ui.py — Terminal presentation layer for RoCode.

Provides RoCodeUI: a Rich-based display class that owns every piece of
output that the agent produces.  agent.py calls into this module; nothing
in here calls back into agent.py (no circular imports).

Colour palette (one colour per semantic category, used consistently):
  Tool calls      cyan
  Tool results    dim cyan  (success) / red (failure)
  Prose           default terminal colour
  Errors          bold red
  Warnings        yellow
  Milestone panels green border
  Stats           dim
  Accent / brand  bold blue
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import pyfiglet

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

# ---------------------------------------------------------------------------
# Tool-arg formatting helpers
# ---------------------------------------------------------------------------

# Key argument to display for each tool (first match wins for list entries)
_TOOL_KEY_ARG: dict[str, str | list[str]] = {
    "read_file":        "path",
    "read_file_part":   "path",
    "write_file":       "path",
    "edit_file":        "path",
    "list_directory":   "path",
    "search_files":     ["pattern", "path"],
    "run_bash":         "command",
    "run_python":       "code",
    "update_context_md": "content",
}

_MAX_ARG_LEN = 55


def _format_tool_args(tool_name: str, args: dict) -> str:
    """Return a compact 'key="value"' string for the most important arg(s)."""
    primary = _TOOL_KEY_ARG.get(tool_name)
    if primary is None:
        if not args:
            return ""
        k, v = next(iter(args.items()))
        val = str(v)
        if len(val) > _MAX_ARG_LEN:
            val = val[:_MAX_ARG_LEN - 3] + "..."
        return f'{k}="{val}"'

    keys = primary if isinstance(primary, list) else [primary]
    parts: list[str] = []
    remaining = _MAX_ARG_LEN
    for key in keys:
        if key not in args:
            continue
        val = str(args[key]).strip().replace("\n", "↵")
        if len(val) > remaining:
            val = val[:remaining - 3] + "..."
        parts.append(f'{key}="{val}"')
        remaining -= len(val) + len(key) + 4
        if remaining <= 0:
            break
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Result-summary helpers
# ---------------------------------------------------------------------------

_ERROR_PREFIXES = (
    "Error",
    "error",
    "Tool execution failed",
    "Execution failed",
    "Command failed",
    "User Denied",
)


def _is_error_output(output: str) -> bool:
    return any(output.lstrip().startswith(p) for p in _ERROR_PREFIXES)


def _summarize_result(tool_name: str, output: str) -> str:  # noqa: C901
    """Collapse a raw tool output string to a one-line human summary."""
    stripped = output.strip()
    if not stripped:
        return "no output"
    if _is_error_output(stripped):
        first = stripped.splitlines()[0][:80]
        return first

    if tool_name in ("read_file", "read_file_part"):
        lines = stripped.count("\n") + 1
        return f"{lines:,} lines"

    if tool_name == "list_directory":
        items = len([l for l in stripped.splitlines() if l.strip()])
        return f"{items} items"

    if tool_name == "search_files":
        if "no matches" in stripped.lower() or not stripped:
            return "no matches"
        # count lines that look like file hits
        hits = len([l for l in stripped.splitlines() if l.strip()])
        return f"found {hits} match{'es' if hits != 1 else ''}"

    if tool_name == "write_file":
        return "written"

    if tool_name == "edit_file":
        if "successfully" in stripped.lower():
            return "edited"
        return "applied"

    if tool_name == "run_bash":
        lines = stripped.splitlines()
        if not lines:
            return "completed (no output)"
        last = lines[-1][:70]
        n = len(lines)
        return f"{n} line{'s' if n != 1 else ''} — {last}"

    if tool_name == "run_python":
        n = len(stripped.splitlines())
        return f"{n} line{'s' if n != 1 else ''} of output"

    if tool_name == "update_context_md":
        return "context updated"

    # Generic fallback: first non-empty line
    first = stripped.splitlines()[0][:80]
    return first


# ---------------------------------------------------------------------------
# 429 / rate-limit detection
# ---------------------------------------------------------------------------

def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("429", "rate_limit", "rate limit", "ratelimit", "too many requests"))


# ---------------------------------------------------------------------------
# File counter (for banner — ignores hidden/vendor dirs)
# ---------------------------------------------------------------------------

def _count_project_files(root: Path) -> int:
    _SKIP_DIRS = {
        ".git", ".venv", "venv", "env", "node_modules",
        "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build",
        ".rocode",
    }
    count = 0
    try:
        for p in root.rglob("*"):
            if p.is_file() and not any(part in _SKIP_DIRS for part in p.parts):
                count += 1
                if count >= 50_000:  # cap walk
                    break
    except PermissionError:
        pass
    return count


# ---------------------------------------------------------------------------
# RoCodeUI
# ---------------------------------------------------------------------------

class RoCodeUI:
    """All terminal output for RoCode, driven by Rich."""

    # ── palette ──────────────────────────────────────────────────────────
    _C_TOOL    = "cyan"
    _C_RESULT  = "dim cyan"
    _C_OK      = "dim green"
    _C_ERR     = "bold red"
    _C_WARN    = "yellow"
    _C_STATS   = "dim"
    _C_ACCENT  = "bold blue"
    _C_MILE    = "green"         

    def __init__(self) -> None:
        self.console = Console(highlight=False)
        self._live: Live | None = None
        self._current_metrics: Any = None
        self._turn_start: float = 0.0

    def startup_banner(self)-> None:
        raw_ascii = pyfiglet.figlet_format("RoCode", font="ansi_shadow")
        self.console.print()
        self.console.print(raw_ascii, style=f"bold cyan ")

    def select_model(self) -> tuple[str, str]:
        """Interactive model selection at startup. Returns (provider, model)."""
        options = [
            ("gemini", "gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite"),
            ("gemini", "gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite"),
            ("groq",   "llama-3.3-70b-versatile", "Groq · Llama 3.3 70B Versatile"),
            ("groq",   "qwen/qwen3.6-27b",        "Groq · Qwen 3.6 27B"),
            ("groq",   "openai/gpt-oss-120b",     "Groq · GPT-OSS 120B"),
        ]

        tbl = Table(box=None, show_header=True, header_style="bold cyan", padding=(0, 1))
        tbl.add_column("#", justify="right", style="bold yellow")
        tbl.add_column("Provider", style="cyan")
        tbl.add_column("Model Name", style="white")

        for idx, (prov, model, label) in enumerate(options, 1):
            default_tag = " [dim green](default)[/]" if idx == 1 else ""
            tbl.add_row(str(idx), prov.upper(), f"{label}{default_tag}")

        #self.console.print()
        self.console.print(Panel(
            tbl,
            title="[bold blue]  Select LLM Model  [/]",
            border_style="blue",
            padding=(0, 1),
            expand=False,
        ))

        try:
            choice = self.console.input("[bold blue]Select model [1-5] (default: 1):[/] ").strip()
            if not choice:
                selected_idx = 1
            else:
                selected_idx = int(choice)
                if selected_idx < 1 or selected_idx > len(options):
                    selected_idx = 1
        except Exception:
            selected_idx = 1

        chosen_prov, chosen_model, chosen_label = options[selected_idx - 1]
        self.console.print(f"  [dim green]✓ Active model:[/] [bold]{chosen_label}[/]\n")
        return chosen_prov, chosen_model

    # ── banner ────────────────────────────────────────────────────────────

    def print_banner(self, repo_name: str, stack_str: str, root: Path, model_name: str = "") -> None:
        """One-time startup panel — printed before any LLM call."""
        file_count = _count_project_files(root)

        tbl = Table(box=None, show_header=False, padding=(0, 1), expand=False)
        tbl.add_column(style=self._C_STATS, min_width=12)
        tbl.add_column()
        tbl.add_row("Repository", f"[bold]{repo_name}[/]")
        tbl.add_row("Stack",      stack_str or "—")
        tbl.add_row("Files",      f"{file_count:,}")
        if model_name:
            tbl.add_row("Model",      f"[cyan]{model_name}[/]")

        self.console.print(Panel(
            tbl,
            title="[bold blue]  RoCode  [/]",
            border_style="blue",
            padding=(0, 1),
            expand=False,
        ))
        self.console.print(
            "[dim]  Commands: [bold]explore[/] · [bold]stats[/] · [bold]clear[/] · [bold]quit[/][/]\n"
        )

    # ── prompt ────────────────────────────────────────────────────────────

    def prompt(self) -> str:
        """Styled interactive input prompt."""
        return self.console.input("[bold blue]rocode[/] [bold white]›[/] ")

    # ── live context (spans entire run_agent call) ────────────────────────

    @contextmanager
    def agent_turn(self) -> Iterator["RoCodeUI"]:
        """
        Open a Rich Live context that:
        - shows a spinner during LLM calls (via .thinking())
        - shows a compact stats footer the rest of the time
        - lets prose/tool lines print above the footer via console.print()
        """
        self._turn_start = time.perf_counter()
        initial = self._render_stats()

        with Live(
            initial,
            console=self.console,
            refresh_per_second=12,
            transient=True,
            vertical_overflow="crop",
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

    @contextmanager
    def thinking(self, text: str) -> Iterator[None]:
        """Show spinner while the enclosed block runs, then restore stats."""
        if self._live is not None:
            self._live.update(Spinner("dots", text=f"[{self._C_ACCENT}]{text}[/]"))
            try:
                yield
            finally:
                self._live.update(self._render_stats())
        else:
            # Fallback when called outside agent_turn (e.g. in tests)
            with self.console.status(f"[{self._C_ACCENT}]{text}[/]"):
                yield

    def update_stats(self, task_metrics: Any) -> None:
        """Refresh the live stats footer with new metrics."""
        self._current_metrics = task_metrics
        if self._live is not None:
            self._live.update(self._render_stats())

    def _render_stats(self) -> Text:
        m = self._current_metrics
        elapsed = time.perf_counter() - self._turn_start

        t = Text(overflow="ellipsis")
        t.append("  ")
        if m is not None:
            t.append(f"{m.tool_calls}", style=self._C_TOOL)
            t.append(" tool calls · ", style=self._C_STATS)
            t.append(f"{m.total_tokens:,}", style=self._C_STATS)
            t.append(" tokens · ", style=self._C_STATS)
        t.append(f"{elapsed:.1f}s", style=self._C_STATS)
        return t

    # ── prose output ─────────────────────────────────────────────────────

    def print_prose(self, text: str, delay: float = 0.012) -> None:
        """Assistant reasoning / explanation text, streamed word-by-word with Rich Markdown."""
        stripped = text.strip()
        if not stripped:
            return

        tokens = re.split(r"(\s+)", text)
        accumulated = ""

        if self._live is not None:
            for tok in tokens:
                accumulated += tok
                self._live.update(Markdown(accumulated))
                time.sleep(delay)
            self.console.print(Markdown(text))
            self._live.update(self._render_stats())
        else:
            with Live(console=self.console, refresh_per_second=30) as live:
                for tok in tokens:
                    accumulated += tok
                    live.update(Markdown(accumulated))
                    time.sleep(delay)
            self.console.print(Markdown(text))

    # ── tool call / result lines ──────────────────────────────────────────

    def print_tool_call(self, tool_name: str, args_json: str) -> None:
        """'→ tool_name  key="value"' line printed when a tool call is parsed."""
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except Exception:
            args = {}
        compact = _format_tool_args(tool_name, args)

        t = Text()
        t.append("  → ", style=self._C_STATS)
        t.append(tool_name, style=f"bold {self._C_TOOL}")
        if compact:
            t.append(f"  {compact}", style=self._C_STATS)
        self.console.print(t)

    def print_tool_result(self, tool_name: str, output: str) -> None:
        """'✓ / ✗  one-line summary' printed immediately after a tool returns."""
        summary = _summarize_result(tool_name, output)
        if _is_error_output(output):
            t = Text()
            t.append("  ✗ ", style=self._C_ERR)
            t.append(summary[:100], style=f"dim {self._C_ERR}")
        else:
            t = Text()
            t.append("  ✓ ", style=self._C_OK)
            t.append(summary, style=self._C_RESULT)
        self.console.print(t)

    def print_invalid_tool(self, raw_name: str) -> None:
        """Called when the model emits an unrecognised tool name."""
        self.console.print(
            f"  [{self._C_WARN}]⚠  Skipping unknown tool:[/] [{self._C_ERR}]{raw_name!r}[/]"
        )

    # ── error / retry / system messages ──────────────────────────────────

    def print_llm_error(self, exc: Exception) -> None:
        """Non-retryable error from the LLM call."""
        short = str(exc).splitlines()[0][:120]
        self.console.print(f"  [{self._C_ERR}]✗  LLM error:[/] [{self._C_ERR}dim]{short}[/]")

    def print_rate_limit_retry(self, wait_secs: float, attempt: int) -> None:
        self.console.print(
            f"  [{self._C_WARN}]⚠  Rate limit reached — "
            f"waiting {wait_secs:.0f}s before retry {attempt}…[/]"
        )

    def print_max_turns(self, n: int) -> None:
        self.console.print(
            f"\n[{self._C_WARN}]⚠  Stopped after {n} turns without reaching a conclusion.[/]"
        )

    def print_compact_context(self) -> None:
        self.console.print(f"  [{self._C_STATS}]⟳  Compressing context window…[/]")

    def print_compaction_error(self, err: Exception) -> None:
        short = str(err).splitlines()[0][:80]
        self.console.print(
            f"  [{self._C_WARN}]⚠  Compaction failed ({short}); continuing with full history.[/]"
        )

    # ── explore completion summary ────────────────────────────────────────

    def print_explore_summary(
        self,
        md_path: Path,
        html_path: Path,
        agents_path: Path,
        task_calls: int,
        total_tokens: int,
        elapsed: float,
    ) -> None:
        """Distinct bordered milestone panel shown after explore finishes."""
        tbl = Table(box=None, show_header=False, padding=(0, 1), expand=False)
        tbl.add_column(style=self._C_STATS, min_width=14)
        tbl.add_column()
        tbl.add_row("Tool calls",  str(task_calls))
        tbl.add_row("Tokens",      f"{total_tokens:,}")
        tbl.add_row("Elapsed",     f"{elapsed:.1f}s")
        tbl.add_row("Markdown",    f"[bold]{md_path}[/]")
        tbl.add_row("HTML",        f"[bold]{html_path}[/]")
        tbl.add_row("AGENTS.md",   f"[bold]{agents_path}[/]")

        self.console.print()
        self.console.print(Panel(
            tbl,
            title=f"[bold {self._C_MILE}]  ✓  Explore Complete  [/]",
            border_style=self._C_MILE,
            padding=(0, 1),
            expand=False,
        ))

    def print_explore_error(self, err: Exception) -> None:
        short = str(err).splitlines()[0][:120]
        self.console.print(
            f"\n[{self._C_ERR}]✗  Report generation failed:[/] {short}"
        )

    # ── telemetry & stats displays (brief / summary / detailed) ────────────

    def print_task_footer(self, task_metrics: Any) -> None:
        """Brief line — shown after every user turn completes.
        Format: 12 tool calls · 8,240 tokens · 34s · $0.00 (free tier)
        """
        elapsed = int(round(getattr(task_metrics, "task_time", 0.0)))
        tool_calls = getattr(task_metrics, "tool_calls", 0)
        total_tokens = getattr(task_metrics, "total_tokens", 0)

        t = Text()
        t.append("  ")
        t.append(f"{tool_calls} tool calls", style=self._C_STATS)
        t.append(" · ", style=self._C_STATS)
        t.append(f"{total_tokens:,} tokens", style=self._C_STATS)
        t.append(" · ", style=self._C_STATS)
        t.append(f"{elapsed}s", style=self._C_STATS)
        t.append(" · ", style=self._C_STATS)
        t.append("$0.00 (free tier)", style=self._C_STATS)
        self.console.print(t)

    def print_session_summary(self, metrics: Any, model: str) -> None:
        """Session summary — shown on session end (quit/exit) or explicit trigger.
        Same milestone panel styling as explore completion.
        """
        completed = getattr(metrics, "completed_tasks", 0)
        if completed == 0:
            return

        tool_usage = getattr(metrics, "tool_usage", {})
        tool_parts = [
            f"{name}: {count}"
            for name, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)
        ]
        tools_breakdown = ", ".join(tool_parts) if tool_parts else "none"

        total_tokens = getattr(metrics, "total_tokens", 0)
        prompt_tokens = getattr(metrics, "prompt_tokens", 0)
        completion_tokens = getattr(metrics, "completion_tokens", 0)
        task_time = getattr(metrics, "task_time", 0.0)
        tool_calls = getattr(metrics, "tool_calls", 0)

        tokens_str = f"{total_tokens:,} ({prompt_tokens:,} in / {completion_tokens:,} out)"

        tbl = Table(box=None, show_header=False, padding=(0, 1), expand=False)
        tbl.add_column(style=self._C_STATS, min_width=16)
        tbl.add_column()
        tbl.add_row("Model", f"[bold]{model}[/]")
        tbl.add_row("Tasks / Turns", str(completed))
        tbl.add_row("Total Tool Calls", f"{tool_calls} ([dim]{tools_breakdown}[/])")
        tbl.add_row("Total Tokens", tokens_str)
        tbl.add_row("Total Time", f"{task_time:.1f}s")
        tbl.add_row("Estimated Cost", "$0.00 (free tier)")

        self.console.print()
        self.console.print(Panel(
            tbl,
            title=f"[bold {self._C_MILE}]  ✓  Session Summary  [/]",
            border_style=self._C_MILE,
            padding=(0, 1),
            expand=False,
        ))

    def print_detailed_stats(self, metrics: Any, model: str) -> None:
        """Detailed view — on demand via 'stats' or 'tokens' command."""
        self.console.print()
        self.console.print(f"[{self._C_ACCENT}]═══ RoCode Diagnostics & Telemetry ═══[/]")

        history = getattr(metrics, "task_history", [])
        if not history:
            self.console.print("  [dim]No completed tasks in this session yet.[/]\n")
            return

        # 1. Per-turn breakdown table
        tbl = Table(
            title="Per-Turn Breakdown",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            expand=False,
        )
        tbl.add_column("Turn", justify="right", style="dim")
        tbl.add_column("Prompt / Request", style="white")
        tbl.add_column("Tool Calls", justify="right", style="cyan")
        tbl.add_column("Prompt Tok", justify="right", style="dim")
        tbl.add_column("Comp Tok", justify="right", style="dim")
        tbl.add_column("Total Tok", justify="right", style="bold")
        tbl.add_column("Time", justify="right", style="dim green")

        for rec in history:
            tbl.add_row(
                str(rec["turn"]),
                rec["prompt"],
                str(rec["tool_calls"]),
                f"{rec['prompt_tokens']:,}",
                f"{rec['completion_tokens']:,}",
                f"{rec['total_tokens']:,}",
                f"{rec['task_time']:.1f}s",
            )
        self.console.print(tbl)

        # 2. Tool breakdown table if tool usage exists
        tool_usage = getattr(metrics, "tool_usage", {})
        tool_time_by_name = getattr(metrics, "tool_time_by_name", {})
        if tool_usage:
            t_tbl = Table(
                title="Tool Usage Breakdown",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
                expand=False,
            )
            t_tbl.add_column("Tool Name", style="cyan")
            t_tbl.add_column("Calls", justify="right", style="bold")
            t_tbl.add_column("Total Time", justify="right", style="dim green")
            t_tbl.add_column("Avg Time", justify="right", style="dim")

            for name, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
                tot_t = tool_time_by_name.get(name, 0.0)
                avg_t = tot_t / count if count > 0 else 0.0
                t_tbl.add_row(name, str(count), f"{tot_t:.2f}s", f"{avg_t:.2f}s")
            self.console.print(t_tbl)

        # 3. Context & Optimization event log
        events = getattr(metrics, "events_log", [])
        if events:
            self.console.print("\n[bold yellow]Context & Optimization Events[/]")
            for evt in events:
                self.console.print(f"  • [dim]{evt}[/]")
        else:
            self.console.print("\n[dim]Context Log: Context within budget (0 compaction events firing)[/]")

        # 4. Rate-limit headroom & metadata
        self.console.print(
            f"\n[dim]Model: {model} · Cost: $0.00 (free tier) · Rate-Limit Headroom: Active / Optimal[/]\n"
        )
