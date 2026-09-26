"""Groq provider using its official chat completions SDK."""
from __future__ import annotations

from src.providers import ProviderError
from src.providers.base import safe_error_message


class Provider:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt: str, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            from groq import Groq

            client = Groq(api_key=self.config.api_key)
            body = {"model": self.config.model, "messages": messages}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**body)
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise ProviderError(f"Groq failed: {safe_error_message(exc, self.config.api_key)}") from None
