"""Bounded failure recovery policy."""
from dataclasses import dataclass

@dataclass
class Recovery:
    max_retries: int = 2
    attempts: int = 0
    last_reason: str = ""
    def on_failure(self, reason: str) -> dict[str, object]:
        evidence = " ".join(reason.split())[:1000]
        repeated = bool(evidence) and evidence == self.last_reason
        self.attempts += 1
        retry = self.attempts <= self.max_retries and not repeated
        self.last_reason = evidence
        detail = f"Repeated failure evidence: {evidence}" if repeated else evidence
        return {"retry": retry, "attempt": self.attempts, "reason": detail, "repeated": repeated}
