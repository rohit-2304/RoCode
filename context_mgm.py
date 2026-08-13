import os
from pathlib import Path
from collections import Counter
import subprocess

IGNORED_DIRS = {
    "node_modules", ".venv", "venv", "__pycache__", ".git",
    "dist", "build", ".next", "target", ".mypy_cache",
    ".pytest_cache", "egg-info", ".tox", "site-packages",
}
BLOCKED_FILENAMES = {".env", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {".pem", ".key"}

def build_initial_context(root : Path) -> str:
    tree = get_directory_tree(root, max_depth=3)
    stack = detect_stack(root)
    file_summary = get_file_extension_summary(root)
    entry_points = detect_entry_points(root)
    readme = get_readme_excerpt(root)
    git_info = get_git_info(root)

    parts = [
        f"## Repository: {root.name}",
        f"Stack signals: {stack}",
        f"File summary: {file_summary}",
        f"{readme}"
    ]
    if entry_points:
        parts.append(f"Possible entry points: {entry_points}")
    if git_info:
        parts.append(f"Git: {git_info}")

    parts.append(f"\n## Directory structure (depth-limited)\n{tree}")

    return "\n".join(parts)


def compact_context( conversation_history: list, keep_recent: int = 4, provider_state: dict | None = None, summary_provider: str = "groq", summary_model: str = "llama-3.1-8b-instant", ) -> list:
    if len(conversation_history) <= keep_recent + 1:
        return conversation_history

    system_msg = conversation_history[0]
    recent_context = conversation_history[-keep_recent:]
    older_context = conversation_history[1:-keep_recent]

    # serialize_for_summary now returns a plain string — safe for any provider
    summary_text_body = serialize_for_summary(older_context)

    summary_prompt = (
        "Summarize the key findings, decisions, and file locations discovered "
        "in this conversation so far. Be concise — this summary replaces the "
        "raw history, so keep anything the agent would need to continue the task.\n\n"
        + summary_text_body
    )

    summary_text = "[Summarization unavailable — raw history pruned.]"
    try:
        from openai import OpenAI
        base_url = None
        api_key = ""
        if summary_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "")
            base_url = "https://api.groq.com/openai/v1"
        elif summary_provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            base_url = "https://openrouter.ai/api/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        summary_client = OpenAI(**kwargs)
        summary_response = summary_client.chat.completions.create(
            model=summary_model,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0,
        )
        summary_text = summary_response.choices[0].message.content
    except Exception as e:
        print(f"[Warning] Context summarization failed ({e}); older context will be dropped without a summary.")

    compaction_note = {
        "role": "system",
        "content": f"[Earlier conversation summarized: {summary_text}]",
    }

    compacted_history = [system_msg, compaction_note] + recent_context
    prune_provider_state(provider_state, compacted_history)
    return compacted_history

def serialize_for_summary(history: list[dict]) -> str:
    """Flatten conversation history to a human-readable prose string.

    Returns a plain string (not a list) so it can be embedded safely in any
    provider's user message without hitting role-validation or unknown-field
    errors.  Strips all provider-specific fields (provider_state_id, etc.).
    """
    lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""

        if role == "assistant":
            if content:
                lines.append(f"[Assistant]: {content}")
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                lines.append(
                    f"[Tool Call]: {fn.get('name', '?')} "
                    f"args={fn.get('arguments', '{}')}"
                )
        elif role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "?"
            # Truncate large tool outputs — the summarizer doesn't need the full blob
            snippet = content[:400] + ("…" if len(content) > 400 else "")
            lines.append(f"[Tool Result: {name}]: {snippet}")
        elif role == "user":
            lines.append(f"[User]: {content}")
        # system messages inside older_context are rare but handled gracefully
        elif role == "system":
            lines.append(f"[System]: {content}")

    return "\n".join(lines)

def prune_provider_state(provider_state: dict | None, conversation_history: list[dict]) -> None:
    if provider_state is None:
        return

    live_state_ids = {
        msg["provider_state_id"]
        for msg in conversation_history
        if msg.get("provider_state_id")
    }

    for state_id in list(provider_state):
        if state_id not in live_state_ids:
            del provider_state[state_id]

def prune_old_tool_outputs(conversation_history: list, keep_recent: int = 3) -> list:
    tool_indices = [i for i, m in enumerate(conversation_history) if m["role"] == "tool"]
    stale = set(tool_indices[:-keep_recent]) if len(tool_indices) > keep_recent else set()

    for i, msg in enumerate(conversation_history):
        if msg["role"] == "tool" and i in stale:
            content = msg["content"]
            if len(content) > 100:
                msg["content"] = f"{content[:87]}....truncated"

        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            # is the NEXT tool result stale? if so, shrink this call too
            next_tool_idx = i + 1
            if next_tool_idx < len(conversation_history) and conversation_history[next_tool_idx]["role"] == "tool":
                if next_tool_idx in stale:
                    for tc in msg["tool_calls"]:
                        args = tc["function"]["arguments"]
                        if len(args) > 80:
                            tc["function"]["arguments"] = args[:80] + "...[truncated]"

    return conversation_history


def should_ignore(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts) or path.name.endswith(".egg-info")

def get_git_info(root: Path) -> str:
    if not (root / ".git").exists():
        return ""
    try:
        branch = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        last_commit = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--oneline"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return f"branch '{branch}', last commit: {last_commit}"
    except Exception:
        return ""

def get_file_extension_summary(root: Path, sample_limit: int = 5000) -> str:
    counter = Counter()
    scanned = 0
    for path in root.rglob("*"):
        if scanned >= sample_limit:
            break
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        scanned += 1
        ext = path.suffix or "(no ext)"
        counter[ext] += 1

    top = counter.most_common(6)
    total = sum(counter.values())
    breakdown = ", ".join(f"{ext} ({count})" for ext, count in top)
    return f"{total} files — {breakdown}"

def detect_stack(root: Path) -> str:
    signals = []
    checks = {
        "package.json": "Node.js",
        "requirements.txt": "Python (pip)",
        "pyproject.toml": "Python (poetry/pep517)",
        "go.mod": "Go",
        "Cargo.toml": "Rust",
        "pom.xml": "Java (Maven)",
        "build.gradle": "Java/Kotlin (Gradle)",
        "Gemfile": "Ruby",
        "Dockerfile": "Containerized",
        "docker-compose.yml": "Docker Compose",
    }
    for filename, label in checks.items():
        if (root / filename).exists():
            signals.append(f"{label} ({filename})")
    return "; ".join(signals) if signals else "No standard manifest files detected"


def get_directory_tree(root: Path, max_depth: int = 3) -> str:
    lines = []

    def walk(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(
                [p for p in path.iterdir() if p.name not in IGNORED_DIRS
                 and not should_ignore(p)],
                key=lambda p: (p.is_file(), p.name.lower())
            )
        except PermissionError:
            return

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir() and depth < max_depth:
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(entry, prefix + extension, depth + 1)

    lines.append(f"{root.name}/")
    walk(root, "", 1)
    return "\n".join(lines)

def detect_entry_points(root: Path) -> str:
    candidates = [
        "main.py", "app.py", "server.py", "manage.py", "wsgi.py", "asgi.py",
        "index.js", "index.ts", "server.js", "app.js",
        "main.go", "cmd/main.go",
    ]
    found = [c for c in candidates if (root / c).exists()]
    # also check one level deep in common src dirs
    for subdir in ("src", "app", "cmd"):
        sub = root / subdir
        if sub.exists():
            for c in candidates:
                if (sub / c).exists():
                    found.append(f"{subdir}/{c}")
    return ", ".join(found) if found else ""

def get_readme_excerpt(root: Path) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme_path = root / name
        if readme_path.exists():
            return f"README.md exists at {str(readme_path)}"
    return "README.md doesn't exists"
