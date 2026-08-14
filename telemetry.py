from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from groq.types.completion_usage import CompletionUsage
else:
    CompletionUsage = Any


TOOL_METRIC_MAP: dict[str, str] = {
    "read_file": "files_read",
    "read_file_part": "files_read",
    "write_file": "files_written",
    "edit_file": "files_edited",
    "list_directory": "directories_listed",
    "search_files": "searches",
    "run_python": "python_runs",
    "run_bash": "bash_runs",
}


@dataclass
class BaseMetrics:
    """Shared counters and update hooks for task and session metrics."""

    llm_calls: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    queue_time: float = 0.0
    prompt_time: float = 0.0
    completion_time: float = 0.0
    total_time: float = 0.0
    task_time: float = 0.0

    tool_calls: int = 0
    tool_time: float = 0.0
    tool_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tool_time_by_name: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    files_read: int = 0
    files_written: int = 0
    files_edited: int = 0
    directories_listed: int = 0
    searches: int = 0
    python_runs: int = 0
    bash_runs: int = 0

    tool_failures: int = 0
    llm_failures: int = 0
    errors: int = 0

    def update_llm(self, usage: CompletionUsage | None) -> None:
        """Record one completed LLM call."""
        self.llm_calls += 1
        if usage is None:
            return

        # Token counts: OpenAI/Groq use *_tokens; Gemini uses *_token_count
        self.prompt_tokens += (
            getattr(usage, "prompt_tokens", None)
            or getattr(usage, "prompt_token_count", 0)
            or 0
        )
        self.completion_tokens += (
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "candidates_token_count", 0)
            or 0
        )
        self.total_tokens += (
            getattr(usage, "total_tokens", None)
            or getattr(usage, "total_token_count", 0)
            or 0
        )

        # Reasoning tokens (OpenAI o-series only)
        completion_details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens += getattr(completion_details, "reasoning_tokens", 0) or 0

        # Timing fields (OpenAI/Groq only — Gemini usage_metadata has none)
        self.queue_time += getattr(usage, "queue_time", 0.0) or 0.0
        self.prompt_time += getattr(usage, "prompt_time", 0.0) or 0.0
        self.completion_time += getattr(usage, "completion_time", 0.0) or 0.0
        self.total_time += getattr(usage, "total_time", 0.0) or 0.0

    def update_tool(self, tool_name: str, success: bool = True, duration: float = 0.0) -> None:
        """Record one tool call and its workspace impact."""
        self.tool_calls += 1
        self.tool_time += duration
        self.tool_usage[tool_name] += 1
        self.tool_time_by_name[tool_name] += duration

        workspace_metric = TOOL_METRIC_MAP.get(tool_name)
        if workspace_metric is not None:
            setattr(self, workspace_metric, getattr(self, workspace_metric) + 1)

        if not success:
            self.tool_failures += 1
            self.errors += 1

    def update_error(self, kind: str = "error") -> None:
        self.errors += 1
        if kind == "llm":
            self.llm_failures += 1
        elif kind == "tool":
            self.tool_failures += 1

    def merge_from(self, other: BaseMetrics) -> None:
        """Add another metric object into this one."""
        for field_name in self._counter_fields():
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))

        for tool_name, count in other.tool_usage.items():
            self.tool_usage[tool_name] += count

        for tool_name, duration in other.tool_time_by_name.items():
            self.tool_time_by_name[tool_name] += duration

    @classmethod
    def _counter_fields(cls) -> tuple[str, ...]:
        return (
            "llm_calls",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "queue_time",
            "prompt_time",
            "completion_time",
            "total_time",
            "task_time",
            "tool_calls",
            "tool_time",
            "files_read",
            "files_written",
            "files_edited",
            "directories_listed",
            "searches",
            "python_runs",
            "bash_runs",
            "tool_failures",
            "llm_failures",
            "errors",
        )

    def collection_summary(self) -> dict[str, Any]:
        return {
            "llm": {
                "calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens,
            },
            "timing": {
                "queue_time": self.queue_time,
                "prompt_time": self.prompt_time,
                "completion_time": self.completion_time,
                "total_time": self.total_time,
                "task_time": self.task_time,
            },
            "tools": {
                "calls": self.tool_calls,
                "usage": dict(self.tool_usage),
                "time": self.tool_time,
                "time_by_name": dict(self.tool_time_by_name),
                "failures": self.tool_failures,
            },
            "workspace": {
                "files_read": self.files_read,
                "files_written": self.files_written,
                "files_edited": self.files_edited,
                "directories_listed": self.directories_listed,
                "searches": self.searches,
                "python_runs": self.python_runs,
                "bash_runs": self.bash_runs,
            },
            "errors": {
                "total": self.errors,
                "llm_failures": self.llm_failures,
                "tool_failures": self.tool_failures,
            },
        }

    def _average_per_llm_call(self, field_name: str) -> float:
        if self.llm_calls == 0:
            return 0.0
        return getattr(self, field_name) / self.llm_calls

    def _average_per_tool_call(self, field_name: str) -> float:
        if self.tool_calls == 0:
            return 0.0
        return getattr(self, field_name) / self.tool_calls


@dataclass
class TaskMetrics(BaseMetrics):
    """Metrics for a single user request."""

    _started_at: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._started_at = time.perf_counter()

    def finish(self) -> None:
        if self._started_at is None:
            return
        self.task_time += time.perf_counter() - self._started_at
        self._started_at = None

    def reset(self) -> None:
        fresh = type(self)()
        self.__dict__.update(fresh.__dict__)

    def summary(self) -> dict[str, Any]:
        summary = self.collection_summary()
        summary["average_scope"] = "LLM Call"
        summary["averages"] = {
            "prompt_tokens": self._average_per_llm_call("prompt_tokens"),
            "completion_tokens": self._average_per_llm_call("completion_tokens"),
            "reasoning_tokens": self._average_per_llm_call("reasoning_tokens"),
            "total_tokens": self._average_per_llm_call("total_tokens"),
            "queue_time": self._average_per_llm_call("queue_time"),
            "prompt_time": self._average_per_llm_call("prompt_time"),
            "completion_time": self._average_per_llm_call("completion_time"),
            "total_time": self._average_per_llm_call("total_time"),
            "tool_time": self._average_per_tool_call("tool_time"),
        }
        return summary

    def format_summary(self) -> str:
        return _format_metrics("Task Metrics", self.summary())

    def display(self) -> None:
        print(self.format_summary())


@dataclass
class SessionMetrics(BaseMetrics):
    """Cumulative metrics from completed tasks only."""

    completed_tasks: int = 0
    task_history: list[dict[str, Any]] = field(default_factory=list)
    events_log: list[str] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return self.completed_tasks

    def update_llm(self, usage: CompletionUsage | None) -> None:
        self._reject_direct_update()

    def update_tool(self, tool_name: str, success: bool = True, duration: float = 0.0) -> None:
        self._reject_direct_update()

    def update_error(self, kind: str = "error") -> None:
        self._reject_direct_update()

    def merge(self, task: TaskMetrics, prompt: str = "") -> None:
        self.completed_tasks += 1
        self.merge_from(task)
        short_prompt = prompt.strip().replace("\n", " ")
        if len(short_prompt) > 45:
            short_prompt = short_prompt[:42] + "..."
        self.task_history.append({
            "turn": self.completed_tasks,
            "prompt": short_prompt or "Task",
            "tool_calls": task.tool_calls,
            "prompt_tokens": task.prompt_tokens,
            "completion_tokens": task.completion_tokens,
            "total_tokens": task.total_tokens,
            "task_time": task.task_time,
            "tool_usage": dict(task.tool_usage),
        })

    def add_event(self, event_msg: str) -> None:
        self.events_log.append(event_msg)

    @staticmethod
    def _reject_direct_update() -> None:
        raise RuntimeError("SessionMetrics can only be updated with merge(task).")

    @property
    def average_prompt_tokens(self) -> float:
        return self._average("prompt_tokens")

    @property
    def average_completion_tokens(self) -> float:
        return self._average("completion_tokens")

    @property
    def average_reasoning_tokens(self) -> float:
        return self._average("reasoning_tokens")

    @property
    def average_total_tokens(self) -> float:
        return self._average("total_tokens")

    @property
    def average_queue_time(self) -> float:
        return self._average("queue_time")

    @property
    def average_prompt_time(self) -> float:
        return self._average("prompt_time")

    @property
    def average_completion_time(self) -> float:
        return self._average("completion_time")

    @property
    def average_total_time(self) -> float:
        return self._average("total_time")

    @property
    def average_task_time(self) -> float:
        return self._average("task_time")

    @property
    def average_tool_time(self) -> float:
        return self._average_per_tool_call("tool_time")

    def _average(self, field_name: str) -> float:
        if self.completed_tasks == 0:
            return 0.0
        return getattr(self, field_name) / self.completed_tasks

    def summary(self) -> dict[str, Any]:
        summary = self.collection_summary()
        summary["completed_tasks"] = self.completed_tasks
        summary["average_scope"] = "Task"
        summary["averages"] = {
            "prompt_tokens": self.average_prompt_tokens,
            "completion_tokens": self.average_completion_tokens,
            "reasoning_tokens": self.average_reasoning_tokens,
            "total_tokens": self.average_total_tokens,
            "queue_time": self.average_queue_time,
            "prompt_time": self.average_prompt_time,
            "completion_time": self.average_completion_time,
            "total_time": self.average_total_time,
            "task_time": self.average_task_time,
            "tool_time": self.average_tool_time,
        }
        return summary

    def format_summary(self) -> str:
        return _format_metrics("Session Metrics", self.summary())

    def display(self) -> None:
        print(self.format_summary())

    # Backwards-compatible wrappers for the current CLI.
    def display_metrics_total(self) -> None:
        self.display()


def _format_metrics(title: str, summary: dict[str, Any]) -> str:
    llm = summary["llm"]
    timing = summary["timing"]
    tools = summary["tools"]
    workspace = summary["workspace"]
    errors = summary["errors"]
    averages = summary.get("averages")
    average_scope = summary.get("average_scope", "")
    average_suffix = f"/{average_scope}" if average_scope else ""

    lines = [
        "",
        "=" * 50,
        title,
        "=" * 50,
    ]

    if "completed_tasks" in summary:
        lines.append(f"Completed Tasks    : {summary['completed_tasks']}")

    lines.extend(
        [
            f"LLM Calls          : {llm['calls']}",
            "",
            f"Prompt Tokens      : {llm['prompt_tokens']:,}",
            f"Completion Tokens  : {llm['completion_tokens']:,}",
            f"Reasoning Tokens   : {llm['reasoning_tokens']:,}",
            f"Total Tokens       : {llm['total_tokens']:,}",
            "",
            f"Queue Time         : {timing['queue_time']:.3f} s",
            f"Prompt Time        : {timing['prompt_time']:.3f} s",
            f"Completion Time    : {timing['completion_time']:.3f} s",
            f"Total Time         : {timing['total_time']:.3f} s",
            f"Task Wall Time     : {timing['task_time']:.3f} s",
        ]
    )

    if averages is not None:
        lines.append("")
        lines.extend(
            _format_average_line(label, value, average_suffix)
            for label, value in (
                ("Prompt Tokens", averages.get("prompt_tokens")),
                ("Completion Tokens", averages.get("completion_tokens")),
                ("Reasoning Tokens", averages.get("reasoning_tokens")),
                ("Total Tokens", averages.get("total_tokens")),
            )
            if value is not None
        )
        lines.extend(
            _format_average_line(label, value, average_suffix, unit="s")
            for label, value in (
                ("Queue Time", averages.get("queue_time")),
                ("Prompt Time", averages.get("prompt_time")),
                ("Completion Time", averages.get("completion_time")),
                ("Total Time", averages.get("total_time")),
                ("Task Wall Time", averages.get("task_time")),
            )
            if value is not None
        )
        if "tool_time" in averages:
            lines.append(f"Avg Tool Call Time     : {averages['tool_time']:.3f} s")

    lines.extend(
        [
            "",
            f"Tool Calls         : {tools['calls']}",
            f"Tool Time          : {tools['time']:.3f} s",
            f"Tool Failures      : {tools['failures']}",
            f"Files Read         : {workspace['files_read']}",
            f"Files Written      : {workspace['files_written']}",
            f"Files Edited       : {workspace['files_edited']}",
            f"Directories Listed : {workspace['directories_listed']}",
            f"Searches           : {workspace['searches']}",
            f"Python Runs        : {workspace['python_runs']}",
            f"Bash Runs          : {workspace['bash_runs']}",
        ]
    )

    if tools["usage"]:
        lines.append("")
        lines.append("Tool Usage")
        for tool_name, count in sorted(tools["usage"].items()):
            duration = tools["time_by_name"].get(tool_name, 0.0)
            lines.append(f"  {tool_name:<18}: {count} calls, {duration:.3f} s")

    lines.extend(
        [
            "",
            f"Errors             : {errors['total']}",
            f"LLM Failures       : {errors['llm_failures']}",
            "=" * 50,
        ]
    )
    return "\n".join(lines)


def _format_average_line( label: str, value: float, suffix: str = "", unit: str | None = None, ) -> str:
    metric_label = f"Avg {label}{suffix}"
    formatted_value = f"{value:.3f}" if unit else f"{value:.2f}"
    if unit:
        formatted_value = f"{formatted_value} {unit}"
    return f"{metric_label:<24}: {formatted_value}"
