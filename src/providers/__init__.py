"""Provider implementations loaded lazily, so imports need no credentials."""
from __future__ import annotations

class ProviderError(RuntimeError):
    """Provider failures are sanitized before reaching callers."""

def configured_providers() -> list[str]:
    import os
    result=[]
    pairs=(("openrouter","OPENROUTER_API_KEY"),("groq","GROQ_API_KEY"))
    for name,key in pairs:
        # The evaluator's generic credentials use the OpenRouter-compatible adapter.
        if os.environ.get(key,"").strip() or (name=="openrouter" and os.environ.get("AI_API_KEY","").strip()): result.append(name)
    return result

def create_provider(config):
    module_name={"openrouter":"openrouter_provider","groq":"groq_provider"}.get(config.name)
    if not module_name: raise ProviderError("Unsupported provider")
    module=__import__(f"src.providers.{module_name}",fromlist=["Provider"])
    return module.Provider(config)

__all__=["ProviderError","configured_providers","create_provider"]
