"""Bounded failure recovery policy."""
from dataclasses import dataclass

@dataclass
class Recovery:
    max_retries: int = 2
    attempts: int = 0
    def on_failure(self, reason: str) -> dict[str, object]:
        self.attempts += 1
        return {"retry": self.attempts <= self.max_retries, "attempt": self.attempts, "reason": reason[:1000]}
