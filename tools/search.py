"""Repository search that skips generated and VCS directories."""
from pathlib import Path
SKIP = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", "dist", "build"}

class SearchTools:
    def __init__(self, root: str | Path): self.root = Path(root).resolve()
    def filenames(self, query: str, limit: int = 50) -> list[str]:
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file() and not SKIP.intersection(p.parts) and query.lower() in p.name.lower()][:limit]
    def text(self, query: str, limit: int = 30) -> list[dict[str, object]]:
        results = []
        for path in self.root.rglob("*"):
            if not path.is_file() or SKIP.intersection(path.parts): continue
            try:
                for number,line in enumerate(path.read_text(errors="ignore").splitlines(),1):
                    if query.lower() in line.lower():
                        results.append({"path":path.relative_to(self.root).as_posix(),"line":number,"snippet":line[:300]})
                        if len(results)>=limit:return results
            except OSError: continue
        return results
