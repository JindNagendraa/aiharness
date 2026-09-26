"""Provider configuration and provider-agnostic model router."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from src.providers import ProviderError, create_provider, configured_providers
from src.providers.base import safe_error_message

class ModelError(RuntimeError):
    """Configuration or provider error with credentials removed."""

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    model: str
    base_url: str | None = None

    @classmethod
    def from_env(cls, name: str) -> "ProviderConfig":
        keyname = {"openrouter":"OPENROUTER_API_KEY", "groq":"GROQ_API_KEY"}.get(name)
        modelname = {"openrouter":"OPENROUTER_MODEL", "groq":"GROQ_MODEL"}.get(name)
        if not keyname: raise ModelError(f"Unknown provider: {name}")
        key = os.environ.get(keyname, "").strip()
        model = os.environ.get(modelname or "", "").strip()
        # Official evaluator credentials take precedence over local provider settings.
        if name == "openrouter" and os.environ.get("AI_API_KEY", "").strip():
            key = os.environ["AI_API_KEY"].strip()
            model = os.environ.get("AI_MODEL", "").strip() or model
            return cls(name, key, model, os.environ.get("AI_BASE_URL", "").strip() or "https://openrouter.ai/api/v1")
        if name == "openrouter":
            return cls(name, key, model, "https://openrouter.ai/api/v1")
        return cls(name, key, model)

class TextModel:
    """Legacy-compatible facade: messages in, text out, with provider fallback."""
    def __init__(self, provider: str | None = None, *, fallback: list[str] | None = None):
        self.primary = (provider or os.environ.get("PRIMARY_PROVIDER", "").strip() or ("openrouter" if os.environ.get("AI_API_KEY") else "openrouter")).lower()
        self.fallback = fallback
        self.active_provider = ""
        self.active_model = ""
        self.last_model = ""
        self.available = configured_providers()

    def _order(self) -> list[str]:
        # The evaluator's generic key intentionally maps to OpenRouter. Keep
        # that compatibility, while treating an explicitly empty fallback
        # variable as "do not fall back" instead of silently trying every
        # configured provider.
        evaluator_key = bool(os.environ.get("AI_API_KEY", "").strip())
        requested = "openrouter" if evaluator_key else (
            os.environ.get("PRIMARY_PROVIDER", "").strip().lower() or self.primary
        )
        if requested not in {"openrouter", "groq"}:
            raise ModelError(f"Invalid PRIMARY_PROVIDER: {requested}")
        if self.fallback is not None:
            fallbacks = [str(p).strip().lower() for p in self.fallback if str(p).strip()]
        elif "FALLBACK_PROVIDERS" in os.environ:
            fallbacks = [p.strip().lower() for p in os.environ["FALLBACK_PROVIDERS"].split(",") if p.strip()]
        else:
            # Preserve the historical default: other configured providers are
            # fallbacks only when the setting is omitted entirely.
            fallbacks = [p for p in self.available if p != requested]
        invalid = [p for p in fallbacks if p not in {"openrouter", "groq"}]
        if invalid: raise ModelError(f"Invalid fallback provider: {invalid[0]}")
        return [p for p in dict.fromkeys([requested] + fallbacks) if p in self.available]

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        from src.providers.base import messages_to_prompt
        errors = []
        for name in self._order():
            config = None
            try:
                config = ProviderConfig.from_env(name)
                if not config.api_key: continue
                if not config.model:
                    evaluator_default = name == "openrouter" and bool(os.environ.get("AI_API_KEY", "").strip())
                    if not evaluator_default:
                        raise ModelError(f"{name} model is not configured; set {name.upper()}_MODEL")
                self.last_model = config.model
                provider = create_provider(config)
                value = provider.generate(messages_to_prompt(messages, json_mode=json_mode), messages, json_mode=json_mode)
                self.active_provider = name
                self.active_model = getattr(provider, "actual_model", "") or config.model
                return value
            except Exception as exc:
                if isinstance(exc, (ProviderError, ModelError)):
                    detail = str(exc)
                else:
                    detail = f"{name} failed: {safe_error_message(exc, config.api_key if config else '')}"
                for secret_name in ("AI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
                    secret = os.environ.get(secret_name, "").strip()
                    if secret: detail = detail.replace(secret, "[REDACTED]")
                if config and config.api_key: detail = detail.replace(config.api_key, "[REDACTED]")
                detail = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", detail)
                errors.append(detail)
        if errors:
            detail = "; ".join(errors)
        else:
            requested = os.environ.get("PRIMARY_PROVIDER", self.primary).strip().lower()
            if requested not in {"openrouter", "groq"}: requested = "openrouter"
            model_hint = " (AI_MODEL is optional when the evaluator endpoint selects its own model)" if requested == "openrouter" and os.environ.get("AI_API_KEY", "").strip() else " and the corresponding model setting"
            detail = f"no supported provider is configured; set {requested.upper()}_API_KEY{model_hint}, or configure the other provider for fallback (evaluator AI_API_KEY maps to OpenRouter)"
        raise ModelError(f"No configured provider succeeded ({detail})")

class ModelConfig(ProviderConfig):
    """Backward-compatible evaluator configuration (`AI_API_KEY`/`AI_MODEL`)."""

    @classmethod
    def from_env(cls) -> "ModelConfig":
        key = os.environ.get("AI_API_KEY", "").strip()
        if not key:
            raise ModelError("AI_API_KEY is not set")
        model = os.environ.get("AI_MODEL", "").strip()
        if not model:
            raise ModelError("AI_MODEL is not set; configure the organizer-prescribed model")
        return cls("openrouter", key, model, os.environ.get("AI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"))
