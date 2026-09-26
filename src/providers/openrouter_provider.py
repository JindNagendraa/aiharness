"""OpenRouter adapter using its Chat Completions API."""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request

from src.providers import ProviderError
from src.providers.actions import ACTION_NAMES, ACTION_TOOLS
from src.providers.base import safe_error_message, sanitize_text


_ACTION_PROTOCOL = (
    "Use the supplied native function tools for every turn. Call exactly one listed tool. "
    "Do not return commentary, channel markers, markdown, or a natural-language answer."
)


def _request_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prioritize native actions for OpenRouter without mutating caller messages."""
    result = [dict(message) for message in messages]
    for message in result:
        if message.get("role") == "system":
            content = re.sub(
                r"Return exactly one JSON object:.*?Actions:",
                "Available actions:",
                str(message.get("content", "")),
                count=1,
            )
            message["content"] = f"{content}\n\n{_ACTION_PROTOCOL}".strip()
            break
    else:
        result.insert(0, {"role": "system", "content": _ACTION_PROTOCOL})
    return result


def _validated_action_json(value: object) -> str | None:
    """Accept only the Agent's exact JSON action envelope from text responses."""
    if not isinstance(value, str) or not value:
        return None
    try:
        decision = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(decision, dict):
        raise ProviderError("OpenRouter returned malformed structured action JSON: expected an object")
    action = decision.get("action")
    arguments = decision.get("arguments", {})
    if not isinstance(action, str) or action not in ACTION_NAMES:
        raise ProviderError("OpenRouter returned malformed structured action JSON: unsupported action")
    if not isinstance(arguments, dict):
        raise ProviderError("OpenRouter returned malformed structured action JSON: arguments must be an object")
    return value


def _response_diagnostic(model: object, choice: dict, message: dict) -> str:
    content = message.get("content")
    channel = None
    if isinstance(content, str):
        match = re.search(r"<\|channel\|>([A-Za-z0-9_-]{1,40})", content)
        channel = match.group(1) if match else None
    pieces = [
        f"returned_model={model or '<unknown>'}",
        f"finish_reason={choice.get('finish_reason', '<unknown>')}",
        f"content_type={'text' if isinstance(content, str) else type(content).__name__}",
    ]
    if channel:
        pieces.append(f"channel={channel}")
    if message.get("refusal"):
        pieces.append("refusal=true")
    return ", ".join(pieces)


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
        self.actual_model = ""

    def generate(self, prompt: str, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            # The Agent consumes one structured action per turn. Requiring a
            # native tool call prevents commentary-only responses from being
            # mistaken for valid model output.
            body = {
                "messages": _request_messages(messages),
                "tools": ACTION_TOOLS,
                "tool_choice": "required",
                "parallel_tool_calls": False,
            }
            # Evaluator credentials may omit a model; let the configured
            # OpenRouter endpoint select it instead of inventing a model slug.
            if self.config.model:
                body["model"] = self.config.model
            request = urllib.request.Request(
                f"{self.config.base_url or 'https://openrouter.ai/api/v1'}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120, context=_verified_ssl_context()) as response:
                payload = json.loads(response.read())
            self.actual_model = str(payload.get("model") or "")
            choice = payload["choices"][0]
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if len(tool_calls) != 1:
                    raise ProviderError("OpenRouter returned multiple actions; exactly one action is allowed")
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

            # Some OpenRouter-compatible routes return the action envelope as
            # message content rather than a function call. Accept only strict
            # JSON with a known action; never extract commands from prose or
            # Harmony channel text.
            action_json = _validated_action_json(message.get("content"))
            if action_json:
                return action_json

            metadata = sanitize_text(
                _response_diagnostic(payload.get("model"), choice, message),
                (self.config.api_key,),
            )
            usage = payload.get("usage")
            if isinstance(usage, dict):
                token_metadata = [
                    f"{field}={usage[field]}"
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if isinstance(usage.get(field), (int, float))
                ]
                if token_metadata:
                    metadata += ", " + ", ".join(token_metadata)
            raise ProviderError("OpenRouter did not return a valid action/tool call (" + metadata + ")")
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
