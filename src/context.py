"""Bounded working memory for a harness run."""
from dataclasses import dataclass, field

@dataclass
class RunContext:
    task: str
    max_chars: int = 14000
    files: dict[str, str] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def add_file(self, path: str, content: str) -> None:
        self.files[path] = content[:5000]; self._trim()

    def add_event(self, event: str) -> None:
        self.events.append(event[:1500]); self._trim()

    def _trim(self) -> None:
        # Retain the newest event; older history is expendable first.
        while self.size > self.max_chars and len(self.events) > 1:
            self.events.pop(0)
        while self.size > self.max_chars and self.files:
            self.files.pop(next(iter(self.files)))
        # A single oversized task/event/file must itself be bounded.
        if self.size > self.max_chars:
            excess = self.size - self.max_chars
            if self.events:
                self.events[-1] = self.events[-1][excess:]
            elif self.task:
                self.task = self.task[excess:]

    @property
    def size(self) -> int:
        return len(self.task) + sum(len(k)+len(v) for k,v in self.files.items()) + sum(map(len,self.events))

    def prompt_view(self) -> str:
        header = f"Task: {self.task}\n\nRelevant files:\n"
        event_header = "\n\nRecent actions/results:\n"
        event_text = "\n".join(self.events[-8:])
        room = max(0, self.max_chars - len(header) - len(event_header) - len(event_text))
        file_parts=[]
        for path, content in reversed(list(self.files.items())):
            section=f"--- {path} ---\n{content}"
            separator="\n\n" if file_parts else ""
            if len(section) + len(separator) > room:
                if not file_parts and room:
                    file_parts.append(section[-room:])
                break
            file_parts.append(section)
            room -= len(section) + len(separator)
        files="\n\n".join(reversed(file_parts))
        base=header+files+event_header+event_text
        # For tiny limits, retain the newest event as the most useful signal.
        if len(base) > self.max_chars:
            newest=self.events[-1] if self.events else self.task
            return newest[-self.max_chars:]
        return base
