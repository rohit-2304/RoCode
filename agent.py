from dotenv import load_dotenv
import os
from pathlib import Path
from groq import Groq
from typing import List, Any
import json
import time

from tools import TOOLS, execute_tool
from telemetry import SessionMetrics, TaskMetrics

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
MODEL = "qwen/qwen3.6-27b"
MAX_TURNS = 15
SYSTEM_PROMPT = """You are a helpful coding assistant that can read, write, and execute code.

list_directory(path): List files and directories.
- search_files(path, pattern, file_pattern): Search files for text.
- read_file(path): Read an entire file.
- read_file_part(path, offset, limit): Read part of a large file.
- edit_file(path, old_string, new_string): Replace one unique text occurrence.
- write_file(path, content): Create or overwrite a file.
- run_bash(command): Execute a shell command.
- run_python(code): Execute Python code in a sandbox.

Tool usage guidelines:
- Use list_directory before exploring an unfamiliar directory.
- Prefer search_files to locate symbols, functions, classes, or text.
- Prefer read_file_part for large files and read_file only for reasonably sized files.
- Read files before editing them.
- Use edit_file for modifying existing files and write_file for creating or replacing files.
- Use run_python to test Python code or perform calculations.
- Use run_bash for shell commands, builds, tests, git, and other command-line tasks.
- Do not guess file contents when a tool can retrieve them.
- Call tools only when necessary. If the answer is already known from the conversation, respond directly.

When working on coding tasks:
1. Read existing files to understand the context
2. Write or modify code as needed
3. Use run_python to test Python code, or run_bash for shell commands
4. Iterate if there are errors

The Python sandbox has some restrictions:
- No file I/O (use read_file/write_file tools instead)
- No network access
- No dangerous imports (os, subprocess, etc.)
- 10 second timeout

For shell commands, use run_bash. It is subject to permission from the human. It has a 60 second timeout.

Always test your code before considering the task complete."""

def run_agent(user_message: str, conversation_history: list = None, metrics : SessionMetrics = None) -> tuple[list[dict[str, Any]], SessionMetrics] :
    """Run the agent with a user message, streaming the response.
    
    This implements the ReAct (Reason, Act, Observe) loop:
    1. Send message to the LLM (streaming)
    2. If the LLM wants to use a tool, execute it and continue
    3. Repeat until the LLM gives a final response
    """
    if metrics is None:
        metrics = SessionMetrics()
    task_metrics = TaskMetrics()
    task_metrics.start()

    if conversation_history is None:
        conversation_history = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    turns = 0

    # React Loop - keep going until model stops using tools
    while turns < MAX_TURNS:
        last_usage = None
        try: 
            stream = client.chat.completions.create(
                model=MODEL,
                messages=conversation_history,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
                stream=True
            )

            assistant_content = ""
            tool_calls ={}

            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    last_usage = chunk.usage
                delta = chunk.choices[0].delta

                # stream text content
                if delta.content:
                    print(delta.content, end="", flush=True)
                    assistant_content += delta.content

                # collect tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index

                        if idx not in tool_calls:
                            tool_calls[idx]={
                                "id": tc.id,
                                "name": "",
                                "arguments": ""
                            }

                        if tc.function:
                            if tc.function.name:
                                tool_calls[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[idx]["arguments"] += tc.function.arguments
        except Exception as e:
            print(repr(e))
            task_metrics.update_error("llm")
            break
        
        
        print()
        task_metrics.update_llm(last_usage)
        tool_calls_list = []
        for tool_call in tool_calls.values():
            tool_calls_list.append({
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": tool_call["arguments"]
                }
            })


        # add response to conversation history
        assistant_message = {
            "role": "assistant",
            "content": assistant_content,
        }

        if tool_calls_list:
            assistant_message["tool_calls"] = tool_calls_list

        conversation_history.append(assistant_message)


        if not tool_calls:
            print("-"*10 + "x" + "-"*10)
            task_metrics.finish()
            task_metrics.display()
            metrics.merge(task_metrics)
            return conversation_history, metrics

        # execute every requested tool
        for tool_call in tool_calls.values():
            tool_name = tool_call["name"] 
            try:
                tool_args = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            print(f"\n Using tool : {tool_name}")

            tool_started_at = time.perf_counter()
            try:
                result = execute_tool(tool_name, tool_args)
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
            task_metrics.update_tool(tool_name, success=success, duration=tool_duration)

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })
        turns+=1
    task_metrics.finish()
    task_metrics.display()
    metrics.merge(task_metrics)
    return conversation_history, metrics

def main():
    """Main chat loop."""
    print("=" * 60)
    print("RoCode Phase 1: Minimum Viable Coding Agent")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'clear' to reset conversation")
    print("=" * 60)
    print()

    conversation_history = []
    metrics = SessionMetrics()
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
            conversation_history = []
            print("Conversation cleared.\n")
            continue

        print("\nAgent: ", end="", flush=True)
        run_agent(user_input, conversation_history, metrics)
        print()
    metrics.display_metrics_total()

if __name__ == "__main__":
    main()
