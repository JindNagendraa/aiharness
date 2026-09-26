"""OpenRouter adapter using its Chat Completions API."""
from __future__ import annotations
import json
import ssl
import urllib.error
import urllib.request
from src.providers import ProviderError
from src.providers.actions import ACTION_NAMES, ACTION_TOOLS
from src.providers.base import safe_error_message

def _verified_ssl_context() -> ssl.SSLContext:
    """Build a normal certificate-verifying context with certifi's CA bundle."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class Provider:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt: str, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            # The harness consumes discrete actions. Native function tools give
            # OpenRouter's free-model router an explicit capability to select
            # tool-capable routes. Some upstream models reject JSON mode and
            # function calling together, so the adapter translates tool calls
            # to the established action JSON contract instead.
            body = {
                "model": self.config.model,
                "messages": messages,
                "tools": ACTION_TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            }
            request = urllib.request.Request(
                f"{self.config.base_url or 'https://openrouter.ai/api/v1'}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120, context=_verified_ssl_context()) as response:
                payload = json.loads(response.read())
            choice = payload["choices"][0]
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                call = tool_calls[0]
                function = call.get("function") or {}
                name = function.get("name")
                if name not in ACTION_NAMES:
                    raise ProviderError("OpenRouter returned an unsupported harness action")
                raw_arguments = function.get("arguments", "")
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except (TypeError, ValueError):
                    raise ProviderError("OpenRouter returned malformed action arguments") from None
                if not isinstance(arguments, dict):
                    raise ProviderError("OpenRouter returned malformed action arguments")
                return json.dumps({"action": name, "arguments": arguments})

            content = message.get("content")
            if isinstance(content, str) and content:
                return content

            # Report only response shape/metadata, never raw content, headers,
            # or request credentials. This distinguishes empty output, token
            # limits, and unexpected tool responses safely.
            metadata = [
                f"returned_model={payload.get('model', '<unknown>')}",
                f"finish_reason={choice.get('finish_reason', '<unknown>')}",
                f"message_fields={','.join(sorted(message.keys())) or '<none>'}",
            ]
            usage = payload.get("usage")
            if isinstance(usage, dict):
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if isinstance(usage.get(field), (int, float)):
                        metadata.append(f"{field}={usage[field]}")
            raise ProviderError("OpenRouter returned no content or valid action (" + ", ".join(metadata) + ")")
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
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"OpenRouter failed: {safe_error_message(exc, self.config.api_key)}") from None


class _HTTPProviderError(Exception):
    def __init__(self, status_code: int, body: object, message: str):
        self.status_code = status_code
        self.body = body
        super().__init__(message)
