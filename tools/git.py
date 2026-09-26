"""Read-only Git inspection; the harness never pushes or commits."""
import subprocess
from pathlib import Path

class GitTools:
    def __init__(self, root: str | Path): self.root=Path(root).resolve()
    def _run(self,*args: str) -> str:
        p=subprocess.run(["git",*args],cwd=self.root,capture_output=True,text=True,timeout=15)
        return p.stdout if p.returncode==0 else p.stderr
    def status(self) -> str: return self._run("status","--short","--branch")
    def diff(self) -> str: return self._run("diff","--")
