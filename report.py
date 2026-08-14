"""
report.py — Architecture report renderer for RoCode.

Public API:
    render_module_diagram(modules)          -> mermaid graph TD string
    render_flow_diagram(flow)               -> mermaid sequenceDiagram string
    render_markdown(data, repo_name, stats) -> full markdown report string
    render_html(data, repo_name, stats)     -> standalone HTML report string
    validate_report_data(data)              -> raises ValueError on bad data
    generate_architecture_report(...)       -> writes files, returns (md_path, html_path)
    render_agents_md(data, repo_name)       -> AGENTS.md string for downstream agents
    generate_agents_md(data, repo_root, repo_name) -> writes AGENTS.md, returns path
"""

from __future__ import annotations

import html as html_mod
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Return a mermaid-safe identifier (alphanumeric + underscore, no leading digit)."""
    safe = re.sub(r"[^a-zA-Z0-9]", "_", name.strip())
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_unknown"


def _mermaid_body(fenced: str) -> str:
    """Strip ```mermaid / ``` fences and return the inner diagram body."""
    lines = fenced.splitlines()
    inner = []
    in_block = False
    for line in lines:
        if line.strip().startswith("```mermaid"):
            in_block = True
            continue
        if in_block and line.strip() == "```":
            break
        if in_block:
            inner.append(line)
    return "\n".join(inner)


# ---------------------------------------------------------------------------
# render_module_diagram
# ---------------------------------------------------------------------------

def render_module_diagram(modules: list[dict]) -> str:
    """Return a ```mermaid graph TD block for the module dependency graph.

    - Caps at 15 nodes (highest in+out degree kept).
    - Skips edges to undefined modules.
    - Node IDs are mermaid-safe; labels use ["..."] syntax when needed.
    """
    if not modules:
        return ""

    module_names = {m["name"] for m in modules}
    omitted = 0

    if len(modules) > 15:
        degree: dict[str, int] = {m["name"]: 0 for m in modules}
        for m in modules:
            for dep in m.get("depends_on", []):
                if dep in module_names:
                    degree[m["name"]] += 1   # out-degree
                    degree[dep] += 1          # in-degree
        ranked = sorted(modules, key=lambda m: degree[m["name"]], reverse=True)
        omitted = len(modules) - 15
        modules = ranked[:15]
        module_names = {m["name"] for m in modules}

    lines = ["```mermaid", "graph TD"]

    for m in modules:
        raw = m["name"]
        safe_id = _sanitize(raw)
        if safe_id != raw:
            lines.append(f'    {safe_id}["{raw}"]')
        else:
            lines.append(f"    {safe_id}")

    for m in modules:
        src = _sanitize(m["name"])
        for dep in m.get("depends_on", []):
            if dep in module_names:
                dst = _sanitize(dep)
                lines.append(f"    {src} --> {dst}")

    lines.append("```")

    result = "\n".join(lines)
    if omitted:
        result += f"\n\n_{omitted} additional modules omitted for clarity._"
    return result


# ---------------------------------------------------------------------------
# render_flow_diagram
# ---------------------------------------------------------------------------

def render_flow_diagram(flow: dict) -> str:
    """Return a ```mermaid sequenceDiagram block for one request flow.

    Skips rendering if fewer than 2 steps (would produce a broken diagram).
    """
    steps = flow.get("steps", [])
    if len(steps) < 2:
        return ""

    # Collect participants in first-seen order
    participants: list[tuple[str, str]] = []   # (safe_id, display_name)
    seen: set[str] = set()
    for step in steps:
        for key in ("from", "to"):
            raw = step.get(key, "").strip()
            safe = _sanitize(raw)
            if safe and safe not in seen:
                participants.append((safe, raw))
                seen.add(safe)

    lines = ["```mermaid", "sequenceDiagram"]

    for safe_id, display in participants:
        if safe_id != display:
            lines.append(f"    participant {safe_id} as {display}")
        else:
            lines.append(f"    participant {safe_id}")

    for step in steps:
        frm = _sanitize(step.get("from", ""))
        to  = _sanitize(step.get("to", ""))
        action = step.get("action", "")
        file_  = step.get("file", "")
        lines.append(f"    {frm}->>{to}: {action}")
        if file_:
            lines.append(f"    Note right of {to}: {file_}")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# validate_report_data
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = (
    "overview",
    "tech_stack",
    "modules",
    "request_flows",
    "entry_points",
    "how_to_extend",
    "notable_observations",
)


def validate_report_data(data: dict) -> None:
    """Raise ValueError with a clear message if required keys are missing or wrong type."""
    if not isinstance(data, dict):
        raise ValueError(f"Report data must be a dict, got {type(data).__name__}")

    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Report data missing required keys: {missing}")

    type_checks = {
        "overview": str,
        "tech_stack": list,
        "entry_points": list,
        "modules": list,
        "request_flows": list,
        "notable_observations": list,
    }
    for key, expected in type_checks.items():
        if not isinstance(data[key], expected):
            raise ValueError(
                f"Report data['{key}'] must be {expected.__name__}, "
                f"got {type(data[key]).__name__}"
            )


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

def render_markdown(data: dict, repo_name: str, stats: dict) -> str:
    """Assemble and return a complete markdown architecture report."""
    validate_report_data(data)

    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    model     = stats.get("model", "unknown")
    tool_n    = stats.get("tool_calls", 0)
    tokens    = stats.get("total_tokens", 0)
    elapsed   = stats.get("task_time", 0.0)
    stats_line = f"Model: `{model}` · {tool_n} tool calls · {tokens:,} tokens · {elapsed:.1f}s"

    sections: list[str] = []

    # Title
    sections.append(f"# Architecture Report: {repo_name}")
    sections.append(f"_{ts}_ | {stats_line}")

    # Overview
    sections.append("## Overview")
    sections.append(data["overview"])

    # Tech stack
    tech = data.get("tech_stack", [])
    if tech:
        sections.append("## Tech Stack")
        rows = ["| Layer | Technology |", "| ----- | ---------- |"]
        for item in tech:
            rows.append(f"| {item.get('layer', '')} | {item.get('technology', '')} |")
        sections.append("\n".join(rows))

    # Module dependency diagram
    modules = data.get("modules", [])
    if modules:
        sections.append("## Module Dependency Graph")
        diagram = render_module_diagram(modules)
        if diagram:
            sections.append(diagram)

    # Module breakdown table
    if modules:
        sections.append("## Module Breakdown")
        rows = ["| Module | Path | Responsibility |", "| ------ | ---- | -------------- |"]
        for m in modules:
            rows.append(
                f"| {m.get('name', '')} "
                f"| `{m.get('path', '')}` "
                f"| {m.get('responsibility', '')} |"
            )
        sections.append("\n".join(rows))

    # Request / execution flows
    flows = data.get("request_flows", [])
    if flows:
        sections.append("## Request / Execution Flows")
        for flow in flows:
            name = flow.get("flow_name", "Flow")
            diagram = render_flow_diagram(flow)
            if diagram:
                sections.append(f"### {name}")
                sections.append(diagram)

    # Entry points
    entry_points = data.get("entry_points", [])
    if entry_points:
        sections.append("## Entry Points")
        for ep in entry_points:
            f = ep.get("file", "")
            d = ep.get("description", "")
            sections.append(f"- **`{f}`** — {d}")

    # How to extend
    how = data.get("how_to_extend", "")
    if how:
        sections.append("## How to Extend")
        sections.append(how)

    # Notable observations
    obs = data.get("notable_observations", [])
    if obs:
        sections.append("## Notable Observations")
        for o in obs:
            sections.append(f"- {o}")

    # Footer
    sections.append("---")
    sections.append(f"_Generated by RoCode · {stats_line}_")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Architecture Report: {repo_name}</title>
<style>
  :root {{
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --accent:   #58a6ff;
    --accent2:  #3fb950;
    --danger:   #f85149;
    --font:     -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    --mono:     ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --radius:   8px;
    --max-w:    960px;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.7;
    padding: 2rem 1rem 4rem;
  }}
  .container {{
    max-width: var(--max-w);
    margin: 0 auto;
  }}

  /* ── Header ── */
  .report-header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }}
  .report-header h1 {{
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: .25rem;
  }}
  .stats-line {{
    font-size: .85rem;
    color: var(--muted);
    font-family: var(--mono);
  }}

  /* ── Sections ── */
  section {{
    margin-bottom: 2.5rem;
  }}
  h2 {{
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: .4rem;
    margin-bottom: 1rem;
  }}
  h3 {{
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin: 1.25rem 0 .5rem;
  }}
  p {{
    margin-bottom: .75rem;
  }}
  code {{
    font-family: var(--mono);
    font-size: .875em;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: .1em .35em;
  }}

  /* ── Tables ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: .75rem 0 1rem;
    font-size: .9rem;
  }}
  th {{
    background: var(--surface);
    color: var(--accent);
    font-weight: 600;
    text-align: left;
    padding: .55rem .75rem;
    border: 1px solid var(--border);
  }}
  td {{
    padding: .5rem .75rem;
    border: 1px solid var(--border);
    vertical-align: top;
  }}
  tr:nth-child(even) td {{
    background: rgba(255,255,255,.025);
  }}

  /* ── Lists ── */
  ul {{
    list-style: none;
    padding: 0;
  }}
  ul li {{
    padding: .3rem 0 .3rem 1.1rem;
    position: relative;
  }}
  ul li::before {{
    content: "›";
    position: absolute;
    left: 0;
    color: var(--accent);
    font-weight: bold;
  }}

  /* ── Mermaid diagrams ── */
  .diagram-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    margin: .75rem 0 1.25rem;
    overflow-x: auto;
  }}
  .mermaid {{
    display: flex;
    justify-content: center;
  }}

  /* ── How-to-extend block ── */
  .extend-block {{
    background: var(--surface);
    border-left: 3px solid var(--accent2);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.25rem;
    white-space: pre-wrap;
    font-size: .9rem;
  }}

  /* ── Footer ── */
  footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: .8rem;
    color: var(--muted);
    font-family: var(--mono);
    text-align: center;
  }}
</style>
</head>
<body>
<div class="container">

  <header class="report-header">
    <h1>Architecture Report: {repo_name}</h1>
    <div class="stats-line">{timestamp} &nbsp;|&nbsp; {stats_line}</div>
  </header>

{body}

  <footer>Generated by RoCode &nbsp;·&nbsp; {stats_line}</footer>
</div>

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'loose' }});
</script>
</body>
</html>
"""


def _h(text: str) -> str:
    """HTML-escape a string."""
    return html_mod.escape(str(text))


def render_html(data: dict, repo_name: str, stats: dict) -> str:
    """Return a standalone dark-theme HTML architecture report with mermaid.js."""
    validate_report_data(data)

    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    model     = stats.get("model", "unknown")
    tool_n    = stats.get("tool_calls", 0)
    tokens    = stats.get("total_tokens", 0)
    elapsed   = stats.get("task_time", 0.0)
    stats_line = f"Model: {model} · {tool_n} tool calls · {tokens:,} tokens · {elapsed:.1f}s"

    body_parts: list[str] = []

    def section(content: str) -> None:
        body_parts.append(f"  <section>\n{content}\n  </section>")

    def h2(title: str) -> str:
        return f"    <h2>{_h(title)}</h2>"

    def h3(title: str) -> str:
        return f"    <h3>{_h(title)}</h3>"

    def para(text: str) -> str:
        return f"    <p>{_h(text)}</p>"

    def mermaid_div(fenced_diagram: str) -> str:
        body = _mermaid_body(fenced_diagram)
        return (
            '    <div class="diagram-wrap">\n'
            '      <div class="mermaid">\n'
            f"{body}\n"
            '      </div>\n'
            '    </div>'
        )

    # Overview
    section(h2("Overview") + "\n" + para(data["overview"]))

    # Tech stack table
    tech = data.get("tech_stack", [])
    if tech:
        rows = ["    <table>", "      <thead><tr><th>Layer</th><th>Technology</th></tr></thead>", "      <tbody>"]
        for item in tech:
            rows.append(f"        <tr><td>{_h(item.get('layer',''))}</td><td>{_h(item.get('technology',''))}</td></tr>")
        rows.append("      </tbody></table>")
        section(h2("Tech Stack") + "\n" + "\n".join(rows))

    # Module dependency diagram
    modules = data.get("modules", [])
    if modules:
        diagram = render_module_diagram(modules)
        if diagram:
            parts = [h2("Module Dependency Graph")]
            parts.append(mermaid_div(diagram))
            # omitted note
            if "additional modules omitted" in diagram:
                note_line = [l for l in diagram.splitlines() if "additional modules omitted" in l]
                if note_line:
                    parts.append(f'    <p class="stats-line">{_h(note_line[0].strip("_"))}</p>')
            section("\n".join(parts))

    # Module breakdown table
    if modules:
        rows = [
            "    <table>",
            "      <thead><tr><th>Module</th><th>Path</th><th>Responsibility</th></tr></thead>",
            "      <tbody>",
        ]
        for m in modules:
            rows.append(
                f"        <tr>"
                f"<td>{_h(m.get('name',''))}</td>"
                f"<td><code>{_h(m.get('path',''))}</code></td>"
                f"<td>{_h(m.get('responsibility',''))}</td>"
                f"</tr>"
            )
        rows.append("      </tbody></table>")
        section(h2("Module Breakdown") + "\n" + "\n".join(rows))

    # Request flows
    flows = data.get("request_flows", [])
    if flows:
        flow_parts = [h2("Request / Execution Flows")]
        for flow in flows:
            name = flow.get("flow_name", "Flow")
            diagram = render_flow_diagram(flow)
            if diagram:
                flow_parts.append(h3(name))
                flow_parts.append(mermaid_div(diagram))
        if len(flow_parts) > 1:
            section("\n".join(flow_parts))

    # Entry points
    entry_points = data.get("entry_points", [])
    if entry_points:
        items = ["    <ul>"]
        for ep in entry_points:
            f = ep.get("file", "")
            d = ep.get("description", "")
            items.append(f"      <li><code>{_h(f)}</code> — {_h(d)}</li>")
        items.append("    </ul>")
        section(h2("Entry Points") + "\n" + "\n".join(items))

    # How to extend
    how = data.get("how_to_extend", "")
    if how:
        block = f'    <div class="extend-block">{_h(how)}</div>'
        section(h2("How to Extend") + "\n" + block)

    # Notable observations
    obs = data.get("notable_observations", [])
    if obs:
        items = ["    <ul>"]
        for o in obs:
            items.append(f"      <li>{_h(o)}</li>")
        items.append("    </ul>")
        section(h2("Notable Observations") + "\n" + "\n".join(items))

    body_html = "\n\n".join(body_parts)
    return _HTML_TEMPLATE.format(
        repo_name=_h(repo_name),
        timestamp=_h(ts),
        stats_line=_h(stats_line),
        body=body_html,
    )


# ---------------------------------------------------------------------------
# generate_architecture_report
# ---------------------------------------------------------------------------

def generate_architecture_report(
    conversation_history: list,
    repo_root: Path,
    repo_name: str,
    stats: dict,
    call_fn: Callable,
    schema: dict,
    extraction_prompt: str,
    provider_state: dict | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, Path, dict]:
    """Orchestrate structured extraction → validation → render → write files.

    Args:
        conversation_history: Full history from the explore ReAct loop.
        repo_root:            Repository root Path (used for writing output).
        repo_name:            Human-readable repo name for report titles.
        stats:                Dict with model, tool_calls, total_tokens, task_time.
        call_fn:              Callable matching call_structured_llm's signature.
        schema:               JSON Schema dict for the structured extraction.
        extraction_prompt:    Prompt appended as final user turn for extraction.
        provider_state:       Gemini provider state (optional).
        out_dir:              Output directory; defaults to repo_root/.rocode/

    Returns:
        (md_path, html_path, data) — absolute Paths of the written files and
        the validated architecture data dict (re-usable for AGENTS.md generation
        without a second LLM call).

    Raises:
        ValueError:  If the LLM response fails schema validation.
        RuntimeError: If the LLM call itself fails.
    """
    if out_dir is None:
        out_dir = repo_root / ".rocode"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract structured data
    try:
        data = call_fn(conversation_history, schema, extraction_prompt, provider_state)
    except Exception as exc:
        raise RuntimeError(f"Structured extraction failed: {exc}") from exc

    # 2. Validate — don't pass bad data into renderers
    validate_report_data(data)

    # 3. Render
    md_str   = render_markdown(data, repo_name, stats)
    html_str = render_html(data, repo_name, stats)

    # 4. Write files
    md_path   = out_dir / "ARCHITECTURE.md"
    html_path = out_dir / "architecture_report.html"
    md_path.write_text(md_str, encoding="utf-8")
    html_path.write_text(html_str, encoding="utf-8")

    return md_path, html_path, data


# ---------------------------------------------------------------------------
# render_agents_md / generate_agents_md
# ---------------------------------------------------------------------------

def render_agents_md(data: dict, repo_name: str) -> str:
    """Render a concise AGENTS.md guide for downstream AI agents.

    Produced from the same structured *data* dict used by render_markdown().
    Written to the *repo root* (not .rocode/) so any agent that opens the
    repository immediately finds it alongside README.md.
    """
    validate_report_data(data)

    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append("#" * level + " " + text)

    def blank() -> None:
        lines.append("")

    # ── Header ──────────────────────────────────────────────────────────────
    h(1, f"AGENTS.md — {repo_name}")
    blank()
    lines.append(
        "> Auto-generated by **RoCode** during `explore`. "
        "This file gives AI coding agents (Claude, Codex, Antigravity, etc.) "
        "an accurate map of this repository so they can work without "
        "re-exploring from scratch. Re-run `explore` to refresh."
    )
    blank()
    lines.append("---")
    blank()

    # ── Overview ────────────────────────────────────────────────────────────
    h(2, "Overview")
    blank()
    lines.append(data["overview"])
    blank()

    # ── Tech Stack ──────────────────────────────────────────────────────────
    tech = data.get("tech_stack", [])
    if tech:
        h(2, "Tech Stack")
        blank()
        lines.append("| Layer | Technology |")
        lines.append("| ----- | ---------- |")
        for item in tech:
            lines.append(f"| {item.get('layer', '')} | {item.get('technology', '')} |")
        blank()

    # ── Entry Points ────────────────────────────────────────────────────────
    entry_points = data.get("entry_points", [])
    if entry_points:
        h(2, "Entry Points")
        blank()
        for ep in entry_points:
            f = ep.get("file", "")
            d = ep.get("description", "")
            lines.append(f"- **`{f}`** — {d}")
        blank()

    # ── Module Map ──────────────────────────────────────────────────────────
    modules = data.get("modules", [])
    if modules:
        h(2, "Key Modules")
        blank()
        lines.append("| Module | Path | Responsibility | Depends On |")
        lines.append("| ------ | ---- | -------------- | ---------- |")
        for m in modules:
            deps = ", ".join(m.get("depends_on", [])) or "—"
            lines.append(
                f"| {m.get('name', '')} "
                f"| `{m.get('path', '')}` "
                f"| {m.get('responsibility', '')} "
                f"| {deps} |"
            )
        blank()

    # ── Request / Execution Flows ────────────────────────────────────────────
    flows = data.get("request_flows", [])
    if flows:
        h(2, "Request / Execution Flows")
        blank()
        for flow in flows:
            name = flow.get("flow_name", "Flow")
            h(3, name)
            blank()
            steps = flow.get("steps", [])
            if steps:
                lines.append("| Step | From | To | Action | File |")
                lines.append("| ---- | ---- | -- | ------ | ---- |")
                for i, step in enumerate(steps, 1):
                    file_ = step.get("file", "—")
                    lines.append(
                        f"| {i} "
                        f"| {step.get('from', '')} "
                        f"| {step.get('to', '')} "
                        f"| {step.get('action', '')} "
                        f"| `{file_}` |"
                    )
                blank()

    # ── How to Extend ────────────────────────────────────────────────────────
    how = data.get("how_to_extend", "")
    if how:
        h(2, "How to Extend")
        blank()
        lines.append(how)
        blank()

    # ── Notable Observations ─────────────────────────────────────────────────
    obs = data.get("notable_observations", [])
    if obs:
        h(2, "Notable Observations")
        blank()
        for o in obs:
            lines.append(f"- {o}")
        blank()

    # ── Footer ───────────────────────────────────────────────────────────────
    lines.append("---")
    blank()
    lines.append("_Generated by [RoCode](https://github.com/rohit/RoCode) · re-run `explore` to refresh._")

    return "\n".join(lines)


def generate_agents_md(
    data: dict,
    repo_root: Path,
    repo_name: str,
) -> Path:
    """Render and write AGENTS.md to the repository root.

    Args:
        data:      Validated architecture data dict (same as used by render_markdown).
        repo_root: Root directory of the explored repository.
        repo_name: Human-readable name used in the report header.

    Returns:
        Absolute Path of the written AGENTS.md file.
    """
    content = render_agents_md(data, repo_name)
    agents_path = repo_root / "AGENTS.md"
    agents_path.write_text(content, encoding="utf-8")
    return agents_path
