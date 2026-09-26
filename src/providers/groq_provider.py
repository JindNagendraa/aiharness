"""Groq provider using its official chat completions SDK."""
from __future__ import annotations

import json
from src.providers import ProviderError
from src.providers.actions import ACTION_NAMES, ACTION_TOOLS
from src.providers.base import safe_error_message


class Provider:
    def __init__(self, config):
        self.config = config
        self.actual_model = ""

    def generate(self, prompt: str, messages: list[dict[str, str]], *, json_mode: bool = False, use_tools: bool = True) -> str:
        try:
            from groq import Groq

            client = Groq(api_key=self.config.api_key)
            body = {
                "model": self.config.model,
                "messages": messages,
            }
            if use_tools:
                body.update({
                    "tools": ACTION_TOOLS,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                })
            elif json_mode:
                body["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**body)
            self.actual_model = str(getattr(response, "model", "") or "")
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                call = tool_calls[0]
                function = getattr(call, "function", None)
                name = getattr(function, "name", None)
                raw_arguments = getattr(function, "arguments", "")
                if name not in ACTION_NAMES:
                    raise ProviderError("Groq returned an unsupported harness action")
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except (TypeError, ValueError):
                    raise ProviderError("Groq returned malformed action arguments") from None
                if not isinstance(arguments, dict):
                    raise ProviderError("Groq returned malformed action arguments")
                return json.dumps({"action": name, "arguments": arguments})
            return getattr(message, "content", None) or ""
        except Exception as exc:
            raise ProviderError(f"Groq failed: {safe_error_message(exc, self.config.api_key)}") from None
