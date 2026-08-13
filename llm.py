import os
import json
from abc import ABC, abstractmethod
from uuid import uuid4
from typing import Any


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Each subclass owns:
    - Client initialisation (fail-fast dependency checks at __init__ time)
    - History formatting for its own API
    - Response parsing into a normalised assistant message dict

    The normalised dict always has:
        role          str  "assistant"
        content       str  text content (may be "")
        tool_calls    list optional, OpenAI-format tool call list
        provider      str  optional, set to "gemini" by GeminiProvider
        provider_state_id str optional, key into caller's provider_state dict
    """

    @abstractmethod
    def call( self, history: list[dict], tools_schema: list, provider_state: dict, ) -> tuple[dict, Any]:
        """Call the LLM.

        Returns:
            (normalised_assistant_message, usage_object)
        """


# ---------------------------------------------------------------------------
# OpenAI-compatible provider  (OpenAI / Groq / OpenRouter)
# ---------------------------------------------------------------------------

class OpenAICompatProvider(LLMProvider):
    """Thin wrapper around the OpenAI SDK for any OpenAI-compatible endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAICompatProvider requires the openai package. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def call( self, history: list[dict], tools_schema: list, provider_state: dict, ) -> tuple[dict, Any]:  # provider_state unused — kept for interface uniformity
        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0,
        )
        usage = response.usage
        message = response.choices[0].message
        return self._parse_response(message), usage

    def _parse_response(self, message) -> dict:
        assistant_message: dict = {
            "role": "assistant",
            "content": message.content or "",
        }
        if getattr(message, "tool_calls", None):
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return assistant_message


# ---------------------------------------------------------------------------
# Gemini native provider
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Gemini via the native google-genai SDK.

    Thought parts and raw Content objects are stored in provider_state (keyed
    by provider_state_id) so they can be passed back verbatim on the next turn.
    This is required for thinking models where re-serialising the content would
    lose the thought structure.
    """

    def __init__(self, model: str):
        try:
            from google import genai
            self._genai = genai
        except ImportError as exc:
            raise RuntimeError(
                "GeminiProvider requires the google-genai package. "
                "Install it with: pip install google-genai"
            ) from exc

        self.model = model
        self.client = self._genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call( self, history: list[dict], tools_schema: list, provider_state: dict, ) -> tuple[dict, Any]:
        system_instruction, contents = self._format_history(history, provider_state)
        gemini_tools = self._convert_tools(tools_schema)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config={
                "system_instruction": system_instruction,
                "tools": gemini_tools,
            },
        )
        assistant_message = self._parse_response(response, provider_state)
        usage = getattr(response, "usage_metadata", None)
        return assistant_message, usage

    # ------------------------------------------------------------------
    # History formatting
    # ------------------------------------------------------------------

    def _format_history( self, history: list[dict], provider_state: dict, ) -> tuple[str, list]:
        """Convert the normalised history list into Gemini's (system, contents) format."""
        system_instruction = ""
        contents = []

        for msg in history:
            role = msg["role"]

            if role == "system":
                system_instruction += msg["content"]

            elif role == "assistant" and msg.get("provider") == "gemini":
                # Try to reuse the original Content object (preserves thought parts)
                state_entry = provider_state.get(msg.get("provider_state_id"))
                raw_content = state_entry.get("gemini_raw_content") if state_entry else None
                if raw_content is not None:
                    contents.append(raw_content)
                else:
                    # Fallback: reconstruct from normalised fields
                    contents.append(self._build_model_content(msg))

            elif role == "assistant":
                # Non-Gemini assistant turn (e.g. after provider switch or compaction)
                contents.append(self._build_model_content(msg))

            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

            elif role == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": msg.get("name") or msg.get("tool_call_id", ""),
                            "response": {"result": msg["content"]},
                        }
                    }],
                })

        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools_schema: list) -> list:
        """Convert OpenAI-format tools to Gemini function_declarations format.

        OpenAI:  [{"type": "function", "function": {"name": ..., "parameters": ...}}]
        Gemini:  [{"function_declarations": [{"name": ..., "parameters": ...}]}]
        """
        if not tools_schema:
            return []
        declarations = []
        for tool in tools_schema:
            if tool.get("type") == "function":
                fn = tool["function"]
                decl: dict = {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                }
                if "parameters" in fn:
                    decl["parameters"] = fn["parameters"]
                declarations.append(decl)
        return [{"function_declarations": declarations}]

    @staticmethod
    def _build_model_content(msg: dict) -> dict:
        """Reconstruct a Gemini model Content dict from a normalised assistant message."""
        parts = []
        if msg.get("content"):
            parts.append({"text": msg["content"]})
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            parts.append({
                "function_call": {
                    "name": fn.get("name", ""),
                    "args": args,
                }
            })
        return {"role": "model", "parts": parts or [{"text": ""}]}

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response, provider_state: dict) -> dict:
        candidate = response.candidates[0]
        text = ""
        tool_calls: list = []
        thought_parts: list = []

        for part in candidate.content.parts:
            if getattr(part, "thought", False):
                thought_parts.append(part)
            elif hasattr(part, "text") and part.text:
                text += part.text
            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args)),
                    },
                })

        assistant_message: dict = {
            "role": "assistant",
            "content": text,
            "provider": "gemini",
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        # Store raw content and thought parts out-of-band so they stay off
        # the serialisable history but can be retrieved on the next turn.
        provider_state_id = f"gemini_{uuid4().hex}"
        provider_state[provider_state_id] = {
            "gemini_thought_parts": thought_parts,
            "gemini_raw_content": candidate.content,
        }
        assistant_message["provider_state_id"] = provider_state_id

        return assistant_message


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(provider_name: str, model: str) -> LLMProvider:
    """Return the correct LLMProvider for *provider_name*.

    Supported names: "gemini", "groq", "openrouter", "openai" (default).
    Raises RuntimeError immediately if the required package is missing.
    """
    name = provider_name.lower()

    if name == "gemini":
        return GeminiProvider(model)

    if name == "groq":
        return OpenAICompatProvider(
            model=model,
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )

    if name == "openrouter":
        return OpenAICompatProvider(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )

    # Default — plain OpenAI
    return OpenAICompatProvider(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
