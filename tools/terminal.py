"""Timed shell command execution scoped to the repository."""
import re, subprocess
from dataclasses import dataclass
from pathlib import Path
BLOCKED = re.compile(r"(^|[;&|\s])(?:sudo|rm|mkfs|shutdown|reboot|dd|curl|wget|chmod|chown)\b|>\s*/dev/",re.I)

@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

class TerminalTool:
    def __init__(self, root: str | Path, timeout: int = 120): self.root,self.timeout=Path(root).resolve(),timeout
    def run(self, command: str) -> CommandResult:
        if BLOCKED.search(command): return CommandResult(command,"","Command blocked by safety policy",126)
        try:
            p=subprocess.run(command,shell=True,cwd=self.root,capture_output=True,text=True,timeout=self.timeout)
            return CommandResult(command,p.stdout[-12000:],p.stderr[-12000:],p.returncode)
        except subprocess.TimeoutExpired as exc: return CommandResult(command,str(exc.stdout or "")[-12000:],str(exc.stderr or "")[-12000:],124,True)
