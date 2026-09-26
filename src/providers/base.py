"""Shared provider contract and message conversion helpers."""
from __future__ import annotations
import re
from typing import Protocol

class ModelProvider(Protocol):
    def generate(self, prompt: str, messages: list[dict[str,str]], *, json_mode: bool = False) -> str: ...

def messages_to_prompt(messages: list[dict[str,str]], *, json_mode: bool = False) -> str:
    rendered="\n\n".join(f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages)
    if json_mode: rendered += "\n\nReturn only a valid JSON object."
    return rendered

def sanitize_text(text: object, secrets: tuple[str, ...] = ()) -> str:
    """Redact common credential formats and known secrets from diagnostics."""
    safe = str(text)
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[REDACTED]")
    patterns = (
        (r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]"),
        (r"(?i)((?:api[_ -]?key|access[_ -]?token|secret)\s*[\"']?\s*[:=]\s*[\"']?)[^\s,;\"'}]+", r"\1[REDACTED]"),
        (r"(?i)(authorization\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]"),
        (r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]"),
        (r"\bAIza[A-Za-z0-9_-]{20,}\b", "[REDACTED]"),
    )
    for pattern, replacement in patterns:
        safe = re.sub(pattern, replacement, safe)
    return safe

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+"),
    re.compile(r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[a-zA-Z0-9_-]{24,}\b"),
)

def safe_error_message(exc: Exception, api_key: str = "") -> str:
    """Extract useful API diagnostics while redacting credential-shaped data."""
    status = getattr(exc, "status_code", None)
    sdk_status = getattr(exc, "status", None)
    code = getattr(exc, "code", None)
    if status is None:
        # Google Gen AI exposes HTTP status in `code` and the API status enum
        # (for example RESOURCE_EXHAUSTED) in `status`.
        if isinstance(code, int): status = code
        elif isinstance(sdk_status, int): status = sdk_status
    if code is None and sdk_status is not None: code = sdk_status
    body = getattr(exc, "body", None)
    error_type = type(exc).__name__
    pieces = [error_type]
    if status is not None:
        try:
            status_number=int(status)
            pieces.append(f"HTTP {status_number}")
            if status_number in (401,403): pieces.append("authentication error")
            elif status_number==429: pieces.append("rate limit")
            elif status_number>=500: pieces.append("provider server error")
        except (TypeError, ValueError): pass
    if isinstance(body, dict):
        data = body.get("error", body)
        if isinstance(data, dict):
            code = data.get("code", code)
            error_type = data.get("type", error_type)
            message = data.get("message")
        else: message = None
    else:
        message = getattr(exc, "message", None) or str(exc)
    if sdk_status and not isinstance(sdk_status, int) and str(sdk_status) not in pieces:
        pieces.append(str(sdk_status))
    if error_type and error_type not in pieces: pieces.append(str(error_type))
    if code is not None: pieces.append(f"code={code}")
    if message:
        safe = sanitize_text(message, (api_key,))
        safe = safe[:500]
        if safe and safe not in pieces: pieces.append(safe)
    return ": ".join(pieces)
