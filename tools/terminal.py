'''Timed shell command execution scoped to the repository.'''
import os, re, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
BLOCKED = re.compile(r"(^|[;&|\s])(?:sudo|rm|mkfs|shutdown|reboot|dd|curl|wget|chmod|chown)\b|>\s*/dev/", re.I)

@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

class TerminalTool:
    def __init__(self, root: str | Path, timeout: int = 120):
        self.root, self.timeout = Path(root).resolve(), timeout

    def run(self, command: str) -> CommandResult:
        # Block unsafe commands early.
        if BLOCKED.search(command):
            return CommandResult(command, "", "Command blocked by safety policy", 126)
        try:
            env = os.environ.copy()
            # Determine the appropriate python/bin directory.
            repository_venv_bin = self.root / ".venv" / "bin"
            # Path to the interpreter currently running the harness.
            interpreter_bin = Path(os.path.abspath(sys.executable)).parent
            # Prefer repository virtualenv if it exists; otherwise fall back to the harness interpreter.
            command_bin = repository_venv_bin if (repository_venv_bin / "python").is_file() else interpreter_bin
            # Ensure the selected bin directory is at the front of PATH.
            env["PATH"] = os.pathsep.join((str(command_bin), env.get("PATH", "")))

            # If the command starts with "python", rewrite it to use the repository python binary explicitly.
            stripped = command.lstrip()
            if stripped.startswith("python"):
                # Determine the path to the python executable we want to use.
                python_path = repository_venv_bin / "python" if (repository_venv_bin / "python").is_file() else Path(sys.executable)
                # Preserve any arguments after the "python" token.
                rest = stripped[len("python"):].lstrip()
                command = f"{python_path} {rest}" if rest else str(python_path)

            p = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return CommandResult(command, p.stdout[-12000:], p.stderr[-12000:], p.returncode)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command,
                str(exc.stdout or "")[-12000:],
                str(exc.stderr or "")[-12000:],
                124,
                True,
            )
