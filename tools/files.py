"""Repository-scoped file operations."""
from pathlib import Path

class FileToolError(ValueError): pass

class FileTools:
    def __init__(self, root: str | Path): self.root = Path(root).resolve()
    def resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents: raise FileToolError("Path escapes repository root")
        return candidate
    def list_files(self, limit: int = 300) -> list[str]:
        ignored = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}; out = []
        for path in self.root.rglob("*"):
            if ignored.intersection(path.parts): continue
            if path.is_file():
                out.append(path.relative_to(self.root).as_posix())
                if len(out) >= limit: break
        return out
    def read_file(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_file(): raise FileToolError("File does not exist")
        return target.read_text(errors="replace")
    def write_file(self, path: str, content: str) -> None:
        target = self.resolve(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content)
    create_file = write_file
