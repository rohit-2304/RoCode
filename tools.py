import os
import subprocess
from pathlib import Path
import fnmatch
from executor import execute_code

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files recursively for a text pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root directory to search."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for."
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional filename glob pattern such as '*.py' or '*.md'."
                    }
                },
                "required": [
                    "path",
                    "pattern"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read an entire text file. Use only for small files (<300 lines).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_part",
            "description": (
                "Read a portion of a text file with line numbers. "
                "Useful for viewing large files in chunks. "
                "If offset is omitted, reading starts from the beginning. "
                "If limit is omitted, a default maximum number of lines is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based starting line number. Optional."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Optional."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace exactly one occurrence of a string in a file. "
                "Fails if the text is not found or appears multiple times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit."
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to replace."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text."
                    }
                },
                "required": [
                    "path",
                    "old_string",
                    "new_string"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path. Creates the file if it doesn't exist, or overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file."
                    },
                    "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files and directories in the given directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (defaults to current directory)",
                        "default": "."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": """Execute a bash command and return the output.
                            The command runs with a 60 second timeout.
                            Be careful with commands that modify the system.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to be executed."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": """Execute Python code in a sandboxed environment.
                            The code runs in isolation with limited permissions.
                            Some imports and functions are blocked for security.
                            There is a 10 second timeout.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }
]


WORKSPACE = Path.cwd().resolve()

def resolve_workspace_path(path : str) -> Path:
    """Resolve a path and reject anything outside the current workspace"""
    resolved = (WORKSPACE/path).resolve()
    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise ValueError(f"Path escapes workspace : {path}")
    return resolved

def read_file(path : str) -> str:
    """Read a file and return its content"""
    try:
        safe_path = resolve_workspace_path(path)
        with safe_path.open('r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

MAX_LINES = 500

def read_file_part(path: str, offset: int = None, limit: int = None) -> str:
    with open(path, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    start = (offset - 1) if offset else 0
    end = min(start + (limit or MAX_LINES), total)

    # Add line numbers
    result = '\n'.join(f"{i:4} | {line.rstrip()}"
                       for i, line in enumerate(lines[start:end], start + 1))

    if end < total:
        result += f"\n\n[Showing lines {start+1}-{end} of {total} total]"
        result += f"\nUse read_file with offset={end+1} to see more."

    return result

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def edit_file(path: str, old_string: str, new_string: str) -> str:
    with open(path, 'r') as f:
        content = f.read()

    if old_string not in content:
        return f"Error: Could not find the specified text in {path}"

    if content.count(old_string) > 1:
        return f"Error: Found {content.count(old_string)} occurrences. Be more specific."

    new_content = content.replace(old_string, new_string, 1)

    with open(path, 'w') as f:
        f.write(new_content)

    return f"Successfully edited {path}"

def list_directory(path : str) -> str:
    try:
        safe_path = resolve_workspace_path(path)
        entries = []
        for entry in sorted(safe_path.iterdir()):
            if entry.is_dir():
                entries.append(f"[DIR] {entry.name}/")
            else:
                entries.append(f"[FILE] {entry.name}")
        if not entries:
            return f"Directory is empty: {path}"
        return "\n".join(entries)
    
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except NotADirectoryError:
        return f"Error: Not a directory: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def run_bash(command: str) -> str:
    approval = input(f"Agent wants to execute the following comand :\n{command}\nAllow? (Y/N) :")
    if(approval == "N"):
        return f"User Denied Permission to run the command : {command}"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd()
        )

        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr

        if len(output) > 10000:
            output = output[:10000] + "\n... (output truncated)"

        if result.returncode == 0:
            return output if output else "(no output)"
        else:
            return f"Command failed (exit code {result.returncode}):\n{output}"
        
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"

def run_python(code: str) -> str:
    """Execute Python code in the sandbox."""
    success, output = execute_code(code)
    if success:
        return f"Execution successful:\n{output}"
    else:
        return f"Execution failed:\n{output}"


def search_files(path: str, pattern: str, file_pattern: str = None) -> str:
    results = []
    for file_path in Path(path).rglob("*"):
        if not file_path.is_file():
            continue

        # Skip noise
        if any(part in ['node_modules', '__pycache__', '.git', 'venv']
               for part in file_path.parts):
            continue

        # Filter by file pattern if specified
        if file_pattern and not fnmatch.fnmatch(file_path.name, file_pattern):
            continue

        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    if pattern.lower() in line.lower():
                        display = line.rstrip()[:200]  # Truncate long lines
                        results.append(f"{file_path}:{i}: {display}")
                        if len(results) >= 50:
                            return '\n'.join(results) + "\n... (limited to 50 results)"
        except (UnicodeDecodeError, PermissionError):
            continue

    return '\n'.join(results) if results else f"No matches for '{pattern}'"


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result."""
    try:
        if tool_name == "read_file":
            return read_file(tool_input["path"])
        elif tool_name == "list_directory":
            return list_directory(tool_input.get("path", "."))
        elif tool_name == "write_file":
            return write_file(tool_input["path"], tool_input["content"])
        elif tool_name == "run_python":
            return run_python(tool_input["code"])
        elif tool_name == "run_bash":
            return run_bash(tool_input["command"])
        else:
            return f"Error: Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing {tool_name}: {e}"