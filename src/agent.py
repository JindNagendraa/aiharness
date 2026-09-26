"""Autonomous coding loop with a restricted JSON tool protocol."""
from __future__ import annotations
import json
import os
from pathlib import Path
from src.context import RunContext
from src.planner import Plan
from src.recovery import Recovery
from src.verifier import Verifier
from src.providers.base import sanitize_text
from tools.files import FileTools
from tools.git import GitTools
from tools.search import SearchTools
from tools.terminal import TerminalTool

SYSTEM = '''You are a text-only software engineering agent. Use ONLY the listed actions. Return exactly one JSON object: {"action":"...","arguments":{...},"done":false}. Actions: list_files{}, read_file{path}, search_text{query}, search_names{query}, write_file{path,content}, run_command{command}, git_status{}, finish{summary}. Never request arbitrary code execution, secrets, or destructive operations. Inspect before editing. Make focused changes and verify them. Set done true only when implementation is complete; the harness then runs verification.'''

class Agent:
    def __init__(self, root: str | Path, model, *, max_steps: int = 24, verify_command: str = "python -m pytest -q"):
        self.root=Path(root).resolve(); self.model=model; self.max_steps=max_steps
        self.files=FileTools(self.root); self.search=SearchTools(self.root); self.terminal=TerminalTool(self.root)
        self.git=GitTools(self.root); self.verifier=Verifier(self.terminal); self.verify_command=verify_command
        self.state="ready"; self.plan: Plan | None=None; self.recovery=Recovery()

    def _result(self, success: bool, summary: str, *, error: str | None = None,
                verification: dict[str, object] | None = None, steps: int = 0) -> dict[str, object]:
        """Build the stable result shape consumed by the CLI and callers."""
        result: dict[str, object] = {
            "success": success,
            "summary": summary,
            "status": self.state,
            "state": self.state,  # compatibility with earlier CLI/tests
            "steps": steps,
        }
        if error is not None:
            result["error"] = error
        if verification is not None:
            result["verification"] = verification
        return result

    def _action(self, action: str, args: dict) -> str:
        if action=="list_files": return json.dumps(self.files.list_files())
        if action=="read_file":
            content=self.files.read_file(str(args["path"])); self.context.add_file(str(args["path"]),content); return content[:5000]
        if action=="search_text": return json.dumps(self.search.text(str(args["query"])))
        if action=="search_names": return json.dumps(self.search.filenames(str(args["query"])))
        if action=="write_file": self.files.write_file(str(args["path"]),str(args["content"])); return "File written"
        if action=="run_command":
            result=self.terminal.run(str(args["command"])); return json.dumps({"stdout":result.stdout,"stderr":result.stderr,"exit_code":result.exit_code,"timed_out":result.timed_out})
        if action=="git_status": return self.git.status()
        if action=="finish": return str(args.get("summary","Task complete"))
        raise ValueError(f"Unsupported action: {action}")

    @staticmethod
    def _parse_decision(raw: str) -> dict:
        try:
            decision=json.loads(raw)
        except json.JSONDecodeError as exc:
            secrets=tuple(os.environ.get(name, "").strip() for name in (
                "AI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"
            ))
            excerpt=sanitize_text(raw, secrets)[:240] or "<empty response>"
            raise ValueError(
                f"Model returned invalid JSON at line {exc.lineno}, column {exc.colno}; "
                f"sanitized response excerpt: {excerpt}"
            ) from None
        if not isinstance(decision,dict) or not isinstance(decision.get("arguments",{}),dict):
            raise ValueError("Expected a JSON action object with an arguments object")
        return decision

    def run(self, task: str) -> dict[str, object]:
        if not task.strip(): raise ValueError("Task cannot be empty")
        self.state="planning"; self.plan=Plan.initial(task); self.context=RunContext(task)
        self.context.add_event("Plan: " + " -> ".join(self.plan.steps)); self.state="working"
        messages=[{"role":"system","content":SYSTEM}]
        last_summary=""
        for step in range(self.max_steps):
            user=self.context.prompt_view()+"\n\nCurrent plan step: "+(self.plan.steps[self.plan.current] if not self.plan.done else "Finish and verify")
            try:
                raw=self.model.generate(messages+[ {"role":"user","content":user}],json_mode=True)
                decision=self._parse_decision(raw)
                action=str(decision.get("action","")); result=self._action(action,decision.get("arguments",{}))
                self.context.add_event(f"{action}: {result}")
                messages.extend([{"role":"assistant","content":raw},{"role":"user","content":"Observed result:\n"+result[:5000]}])
                if action=="finish" or decision.get("done") is True:
                    last_summary=result; break
            except Exception as exc:
                info=self.recovery.on_failure(str(exc)); self.context.add_event("Failure: "+str(info))
                if not info["retry"]:
                    self.state="failed"
                    return self._result(False, "Agent stopped after repeated failures.", error=str(exc), steps=step+1)
                messages.append({"role":"user","content":f"Action failed: {type(exc).__name__}: {exc}. Return exactly one valid JSON action object using the required schema, then retry."})
        else:
            self.state="failed"
            return self._result(False, "Agent reached the maximum action limit.", error="Step limit reached", steps=self.max_steps)
        self.state="verifying"; verification=self.verifier.verify(self.verify_command)
        self.context.add_event(f"Verification {verification.command}: exit={verification.exit_code}")
        if not verification.passed:
            recovery=self.recovery.on_failure(verification.stderr or verification.stdout)
            if recovery["retry"]:
                self.state="working"
                messages.append({"role":"user","content":f"Verification failed (exit {verification.exit_code}). Diagnose and fix based on output:\n{verification.stdout}\n{verification.stderr}"})
                # One bounded repair cycle; each turn remains subject to max_steps.
                for _ in range(min(6,self.max_steps)):
                    try:
                        raw=self.model.generate(messages+[ {"role":"user","content":self.context.prompt_view()}],json_mode=True); decision=self._parse_decision(raw)
                        result=self._action(str(decision.get("action","")),decision.get("arguments",{})); self.context.add_event(result)
                        messages.extend([{"role":"assistant","content":raw},{"role":"user","content":"Observed result: "+result[:4000]}])
                        if decision.get("action")=="finish" or decision.get("done") is True: break
                    except Exception as exc:
                        self.context.add_event("Recovery action failed: "+str(exc)); break
                self.state="verifying"; verification=self.verifier.verify(self.verify_command)
        self.state="complete" if verification.passed else "failed"
        verification_data={"command":verification.command,"passed":verification.passed,"exit_code":verification.exit_code,"stdout":verification.stdout,"stderr":verification.stderr}
        error=None if verification.passed else f"Verification failed with exit code {verification.exit_code}"
        summary=last_summary or ("Task completed and verification passed." if verification.passed else "Task did not pass verification.")
        return self._result(verification.passed, summary, error=error, verification=verification_data, steps=step+1)
