"""Verification command selection and structured results."""
from dataclasses import dataclass
from tools.terminal import TerminalTool

@dataclass
class VerificationResult:
    command: str
    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

class Verifier:
    def __init__(self, terminal: TerminalTool): self.terminal=terminal
    def verify(self, command: str = "python -m pytest -q") -> VerificationResult:
        result=self.terminal.run(command)
        return VerificationResult(command,result.exit_code==0,result.stdout,result.stderr,result.exit_code,result.timed_out)
