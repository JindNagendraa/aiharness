"""OpenRouter adapter using its Chat Completions API."""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from src.providers import ProviderError
from src.providers.base import safe_error_message


class Provider:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt: str, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            body = {"model": self.config.model, "messages": messages}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(
                f"{self.config.base_url or 'https://openrouter.ai/api/v1'}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
            return payload["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read())
            except (ValueError, OSError):
                body = None
            error = _HTTPProviderError(exc.code, body, f"HTTP {exc.code}")
            raise ProviderError(f"OpenRouter failed: {safe_error_message(error, self.config.api_key)}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"OpenRouter failed: {safe_error_message(exc, self.config.api_key)}") from None
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"OpenRouter failed: {safe_error_message(exc, self.config.api_key)}") from None
        except Exception as exc:
            raise ProviderError(f"OpenRouter failed: {safe_error_message(exc, self.config.api_key)}") from None


class _HTTPProviderError(Exception):
    def __init__(self, status_code: int, body: object, message: str):
        self.status_code = status_code
        self.body = body
        super().__init__(message)
