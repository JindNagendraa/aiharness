"""Minimal line-oriented terminal interface for the coding harness."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class HarnessTUI:
    """Run the existing Agent behind an ordinary terminal prompt."""

    def __init__(self, repo: str | Path = ".", verify_command: str = "python -m pytest -q",
                 *, agent_factory=None, model_factory=None,
                 input_fn: Callable[[str], str] | None = None,
                 output_fn: Callable[[str], object] | None = None):
        self.repo = Path(repo).expanduser().resolve()
        self.verify_command = verify_command
        self.agent_factory = agent_factory
        self.model_factory = model_factory
        self.input_fn = input_fn
        self.output_fn = output_fn

    @staticmethod
    def _secrets() -> tuple[str, ...]:
        return tuple(os.environ.get(name, "").strip() for name in (
            "AI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"
        ))

    @classmethod
    def _sanitize(cls, value: object) -> str:
        from src.providers.base import sanitize_text
        return sanitize_text(value, cls._secrets())

    @staticmethod
    def _initial_provider_model() -> tuple[str, str]:
        # Resolve display values through the same configuration objects used
        # by the Agent's TextModel. This keeps the header aligned with
        # PRIMARY_PROVIDER and provider-specific model settings, while
        # ProviderConfig retains AI_API_KEY/AI_MODEL evaluator compatibility
        # for OpenRouter requests.
        from src.model import ProviderConfig, TextModel

        model = TextModel()
        provider = model.primary
        configured = ProviderConfig.from_env(provider)
        return provider, configured.model or "not selected yet"

    def _print(self, message: str = "") -> None:
        emit = self.output_fn or print
        emit(self._sanitize(message))

    def _event(self, event: dict[str, str]) -> None:
        event_type = event.get("type", "working")
        message = self._sanitize(event.get("message", ""))
        if event_type == "provider":
            provider = self._sanitize(event.get("provider", "unknown"))
            model = self._sanitize(event.get("model", "not selected"))
            self._print(f"Provider: {provider}\nModel: {model}")
        elif event_type in {"verify", "verify_result"}:
            self._print(f"[VERIFICATION] {message}")
        elif event_type in {"recovery", "retry"}:
            self._print(f"[RECOVERY] {message}")
        elif event_type in {"planning", "context", "tool", "edit", "observation", "model"}:
            if message:
                self._print(message)

    def _run_one(self, task: str) -> None:
        from src.agent import Agent
        from src.model import TextModel

        model_factory = self.model_factory or TextModel
        agent_factory = self.agent_factory or Agent
        model = model_factory()
        self._print("[RUNNING]\nAgent is working...")
        agent = agent_factory(
            self.repo,
            model,
            verify_command=self.verify_command,
            event_callback=self._event,
        )
        result = agent.run(task)

        verification = result.get("verification") or {}
        if result.get("success"):
            self._print("[SUCCESS]\nTask completed successfully.")
        else:
            self._print("[FAILED]\nTask failed.")
        summary = result.get("summary")
        if summary:
            self._print(f"Summary: {summary}")
        if verification:
            status = verification.get("status") or (
                "passed" if verification.get("passed") else "failed"
            )
            self._print(f"Verification: {status}")
            for label in ("stdout", "stderr"):
                output = verification.get(label)
                if output:
                    self._print(f"Verification {label}:\n{str(output)[:2000]}")
        error = result.get("error")
        if error:
            self._print(f"Error: {error}")
        self._print("")

    def run(self) -> None:
        input_fn = self.input_fn or input
        provider, model = self._initial_provider_model()
        self._print("FORGEAI / AI CODING HARNESS")
        self._print(f"Repository: {self.repo}")
        self._print(f"Provider: {provider}")
        self._print(f"Model: {model}")
        self._print("\nEnter coding task:")

        while True:
            try:
                task = input_fn("> ")
            except EOFError:
                self._print("\nExiting.")
                return
            except KeyboardInterrupt:
                self._print("\nInterrupted. Exiting.")
                return

            task = task.strip()
            if not task:
                continue
            try:
                self._run_one(task)
            except KeyboardInterrupt:
                # Agent execution is synchronous, so there is no worker left
                # running when control returns here.
                self._print("\n[INTERRUPTED]\nAgent run interrupted. Exiting.")
                return
            except Exception as exc:
                self._print(f"[FAILED]\n{self._sanitize(exc)}")
                self._print("")
            self._print("Enter coding task:")


def run_tui(repo: str | Path = ".", verify_command: str = "python -m pytest -q", *,
            agent_factory=None, model_factory=None,
            input_fn: Callable[[str], str] | None = None,
            output_fn: Callable[[str], object] | None = None) -> None:
    """Start the line-oriented interface using normal terminal input semantics."""
    HarnessTUI(
        repo,
        verify_command,
        agent_factory=agent_factory,
        model_factory=model_factory,
        input_fn=input_fn,
        output_fn=output_fn,
    ).run()
