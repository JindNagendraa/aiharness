"""Structured initial plan and plan revisions."""
from dataclasses import dataclass, field

@dataclass
class Plan:
    steps: list[str] = field(default_factory=list)
    current: int = 0
    @classmethod
    def initial(cls, task: str) -> "Plan": return cls(["Inspect repository structure and relevant code", f"Implement: {task}", "Run focused verification and review the diff"])
    @property
    def done(self) -> bool: return self.current >= len(self.steps)
    def advance(self) -> None: self.current = min(self.current+1,len(self.steps))
    def revise(self, remaining: list[str]) -> None: self.steps = self.steps[:self.current] + remaining
