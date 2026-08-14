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

EXTRACTION_PROMPT = """Based on everything you explored in this repository, produce a 
complete architecture summary as JSON matching the given schema.

Rules:
- Only include modules, flows, and observations you actually verified through the 
  tools — do not invent file paths, function names, or dependencies you didn't see.
- For request_flows, pick the 1-2 most important flows in the application (e.g. the 
  primary user action), not every possible path. Trace them concretely, citing files.
- For modules, group by top-level directory or logical unit if the repo is large — 
  aim for 6-15 modules, not one per file.
- notable_observations should be specific and evidence-based (e.g. "no test files 
  found for src/auth/" not "code quality could be improved").
- If something can't be verified from what you explored, omit it rather than guessing.
"""

ARCHITECTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {
            "type": "string",
            "description": "2-4 sentence plain-language summary of what this project does and why it exists."
        },
        "tech_stack": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "layer": {"type": "string", "description": "e.g. Backend, Frontend, Database, Infra"},
                    "technology": {"type": "string"}
                },
                "required": ["layer", "technology"]
            }
        },
        "modules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string"},
                    "responsibility": {"type": "string", "description": "One sentence, what this module owns."},
                    "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Names of other modules this one calls or imports from."}
                },
                "required": ["name", "path", "responsibility", "depends_on"]
            }
        },
        "request_flows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "flow_name": {"type": "string", "description": "e.g. 'User login', 'Create resource'"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "action": {"type": "string"},
                                "file": {"type": "string"}
                            },
                            "required": ["from", "to", "action"]
                        }
                    }
                },
                "required": ["flow_name", "steps"]
            }
        },
        "entry_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["file", "description"]
            }
        },
        "how_to_extend": {
            "type": "string",
            "description": "Numbered steps for adding a new endpoint/feature, based on an existing pattern found in the repo."
        },
        "notable_observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific, concrete observations — missing tests, hardcoded config, tight coupling, etc. Not generic praise."
        }
    },
    "required": ["overview", "tech_stack", "modules", "request_flows", "entry_points", "how_to_extend", "notable_observations"]
}
