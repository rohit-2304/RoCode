import os
import subprocess
from pathlib import Path
import fnmatch
from executor import execute_code

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a small-to-medium file (under ~150 lines) in full. For larger files, use read_file_part or search_files instead — read_file will truncate and waste context on large files.",
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
            "name": "search_files",
            "description": (
                "Search for a text pattern across files in a directory (substring, "
                "case-insensitive, not regex). Use this FIRST to locate code instead "
                "of reading files one by one. Skips node_modules/.venv/build dirs and "
                "secrets. Capped at 50 results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to search."},
                    "pattern": {"type": "string", "description": "Text to search for."},
                    "file_pattern": {"type": "string", "description": "Optional glob to narrow files, e.g. '*.py'."}
                },
                "required": ["path", "pattern"]
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

READ_FILE_MAX_LINES = 150
MAX_LINES = 300

IGNORED_DIRS = {
    "node_modules", ".venv", "venv", "__pycache__", ".git",
    "dist", "build", ".next", "target", ".mypy_cache",
    ".pytest_cache", "egg-info", ".tox", "site-packages",
}
BLOCKED_FILENAMES = {".env", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {".pem", ".key"}

def resolve_workspace_path(path : str, root: str) -> Path:
    """Resolve a path and reject anything outside the root workspace"""
    WORKSPACE = Path(root).resolve()
    resolved = (WORKSPACE/path).resolve()
    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise ValueError(f"Path escapes workspace : {path}")
    return resolved

def should_ignore(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts) or path.name.endswith(".egg-info")



class Tools:
    def __init__(self, root:Path):
        self.root = root

    def read_file(self, path : str) -> str:
        """Read a file and return its content"""
        try:
            safe_path = resolve_workspace_path(path, self.root)
            print(f"Reading file {path}")
            with safe_path.open('r') as f:
                lines = f.readlines()

            if(len(lines) <= READ_FILE_MAX_LINES):
                return "".join(lines)

            head = lines[:READ_FILE_MAX_LINES // 2]
            tail = lines[-READ_FILE_MAX_LINES // 2: ]
            omitted = len(lines) - READ_FILE_MAX_LINES
            
            return (
                "".join(head)
                + f"\n.... [{omitted}] lines omitted - use read_file_part(path, offset, limit)"
                + f"or search_files/grep to see more] ...\n\n"
                + "".join(tail)
            )
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"

    def read_file_part(self, path: str, offset: int = None, limit: int = None) -> str:
        print(f"Reading file {path} from line:{offset}")
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

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        try:
            # Create parent directories if they don't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            print(f"Wrting to file {path}")
            return f"Successfully wrote to {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
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

    def list_directory(self, path : str) -> str:
        try:
            safe_path = resolve_workspace_path(path, self.root)
            print(f"Listing directory {path}")
            if should_ignore(safe_path):
                return "This directory should be ignored"
            entries = []
            for entry in sorted(safe_path.iterdir()):
                if should_ignore(entry):
                    continue
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


    def run_bash(self, command: str) -> str:
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

            if len(output) > 1000:
                output = output[:10000] + "\n... (output truncated)"

            if result.returncode == 0:
                return output if output else "(no output)"
            else:
                return f"Command failed (exit code {result.returncode}):\n{output}"
            
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60 seconds"

    def run_python(self, code: str) -> str:
        """Execute Python code in the sandbox."""
        success, output = execute_code(code)
        if success:
            return f"Execution successful:\n{output}"
        else:
            return f"Execution failed:\n{output}"


    def search_files(self, path: str, pattern: str, file_pattern: str = None) -> str:
        safe_path = resolve_workspace_path(path, self.root)
        print(f"Searching for pattern : {pattern} in {path}")
        results = []
        files_scanned = 0
        MAX_FILES_SCANNED = 2000
        for file_path in Path(safe_path).rglob("*"):
            
            if not file_path.is_file():
                continue
            if file_path.name in BLOCKED_FILENAMES or file_path.suffix in BLOCKED_SUFFIXES:
                continue
            
            # Skip noise
            skip = False
            for part in file_path.parts:
                if part in IGNORED_DIRS:
                    skip = True
                    break
            if skip:
                continue
            files_scanned += 1
            if files_scanned > MAX_FILES_SCANNED:
                results.append("... (scan limit reached, narrow your search with file_pattern)")
                break
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


    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return its result."""
        try:
            if tool_name == "read_file":
                return self.read_file(tool_input["path"])
            elif tool_name == "list_directory":
                return self.list_directory(tool_input.get("path", "."))
            elif tool_name == "write_file":
                return self.write_file(tool_input["path"], tool_input["content"])
            elif tool_name == "run_python":
                return self.run_python(tool_input["code"])
            elif tool_name == "run_bash":
                return self.run_bash(tool_input["command"])
            elif tool_name == "search_files":
                return self.search_files(
                    tool_input["path"],
                    tool_input["pattern"],
                    tool_input.get("file_pattern"),
                )

            elif tool_name == "read_file_part":
                return self.read_file_part(
                    tool_input["path"],
                    tool_input.get("offset"),
                    tool_input.get("limit"),
                )

            elif tool_name == "edit_file":
                return self.edit_file(
                    tool_input["path"],
                    tool_input["old_string"],
                    tool_input["new_string"],
                )
            else:
                return f"Error: Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"