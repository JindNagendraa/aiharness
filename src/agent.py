"""Autonomous coding loop with a restricted JSON tool protocol."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Callable
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
    _RECENT_MODEL_MESSAGES = 4

    def __init__(self, root: str | Path, model, *, max_steps: int = 24, verify_command: str = "python -m pytest -q",
                 event_callback: Callable[[dict[str, str]], None] | None = None):
        self.root=Path(root).resolve(); self.model=model; self.max_steps=max_steps
        self.files=FileTools(self.root); self.search=SearchTools(self.root); self.terminal=TerminalTool(self.root)
        self.git=GitTools(self.root); self.verifier=Verifier(self.terminal); self.verify_command=verify_command
        self.state="ready"; self.plan: Plan | None=None; self.recovery=Recovery()
        self.event_callback=event_callback

    def _emit(self, event_type: str, message: str, **details: str) -> None:
        """Publish sanitized execution events without coupling Agent logic to a UI."""
        if self.event_callback is not None:
            try:
                secrets=self._secret_values()
                event={"type": event_type, "message": sanitize_text(message,secrets)}
                event.update({key:sanitize_text(value,secrets) for key,value in details.items()})
                self.event_callback(event)
            except Exception:
                # Observability must never change the coding or recovery flow.
                pass

    @staticmethod
    def _secret_values() -> tuple[str, ...]:
        return tuple(os.environ.get(name, "").strip() for name in (
            "AI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"
        ))

    def _log_recovery(self, info: dict[str, object]) -> None:
        self._emit("recovery", "Analyzing failure")
        print("[RECOVERY]", flush=True)
        print("Analyzing failure...", flush=True)
        reason=sanitize_text(info.get("reason", ""),self._secret_values())[:1000]
        if reason:
            print(f"Failure details: {reason}", flush=True)
        if info.get("retry"):
            self._emit("retry", f"Retry {info.get('attempt', '?')}: applying a repair")
            print("Applying fix...", flush=True)
        else:
            if info.get("repeated"):
                self._emit("recovery", "Repeated failure evidence; stopping recovery")
            else:
                self._emit("failed", "Recovery retry limit reached")
            print("Retry limit reached; stopping recovery.", flush=True)

    def _verification_output(self, verification) -> str:
        output="\n".join(part for part in (verification.stdout,verification.stderr) if part).strip()
        return sanitize_text(output,self._secret_values())[:1600]

    def _has_existing_verification_state(self) -> bool:
        """Only baseline-check repositories with recognizable test state."""
        paths=self.files.list_files()
        test_files=[]
        for path in paths:
            name=Path(path).name.lower()
            if (name.startswith("test_") or name.endswith("_test.py") or
                    re.search(r"\.(?:test|spec)\.[a-z0-9]+$",name) or name.endswith("_test.go")):
                test_files.append(path)
        command=self.verify_command.lower()
        if "pytest" in command:
            return any(path.lower().endswith((".py",)) for path in test_files)
        if re.search(r"\bmake\s+(?:\w+\s+)*test\b",command):
            makefile=self.root/"Makefile"
            return makefile.is_file() and bool(re.search(r"(?m)^test\s*:",makefile.read_text(errors="replace")))
        return bool(test_files)

    def _verify_with_logging(self, label: str):
        self._emit("verify", "Running verification" if "BASELINE" not in label else "Running baseline verification")
        print(label, flush=True)
        print("Running verification...", flush=True)
        verification=self.verifier.verify(self.verify_command)
        if verification.passed:
            self._emit("verify_result", "Verification PASSED")
            print("[PASS]", flush=True)
            print("All tests passed." if "pytest" in self.verify_command else "Verification passed.", flush=True)
        else:
            result_state = "INCONCLUSIVE" if verification.timed_out else "FAILED"
            self._emit("verify_result", f"Verification {result_state}")
            print("[INCONCLUSIVE]" if verification.timed_out else "[FAIL]", flush=True)
            details=self._verification_output(verification)
            label="Tests failed" if "pytest" in self.verify_command else "Verification failed"
            if details:
                print(f"{label}:\n{details}", flush=True)
            else:
                print(f"{label} with exit code {verification.exit_code}.", flush=True)
        return verification

    def _result(self, success: bool, summary: str, *, error: str | None = None,
                verification: dict[str, object] | None = None, steps: int = 0) -> dict[str, object]:
        """Build the stable result shape consumed by the CLI and callers."""
        result: dict[str, object] = {
            "success": success,
            "summary": summary,
            "status": self.state,
            "state": self.state,  # compatibility with earlier CLI/tests
            "steps": steps,
            "recovery_attempts": self.recovery.attempts,
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

    @classmethod
    def _bounded_model_messages(cls, messages: list[dict[str, str]], current_user: dict[str, str]) -> list[dict[str, str]]:
        """Bound transcript growth while retaining instructions and current context.

        RunContext already supplies the task, recent action/verification events,
        and bounded file contents on every turn. Earlier chat exchanges are
        redundant once they have been recorded there, so send only the most
        recent two exchanges in addition to the system rules and current view.
        """
        if messages and messages[0].get("role") == "system":
            system = [messages[0]]
            history = messages[1:]
        else:
            system = []
            history = messages
        return system + history[-cls._RECENT_MODEL_MESSAGES:] + [current_user]

    def run(self, task: str) -> dict[str, object]:
        if not task.strip(): raise ValueError("Task cannot be empty")
        self._emit("planning", "Understanding task")
        self.state="planning"; self.plan=Plan.initial(task); self.context=RunContext(task)
        self.context.add_event("Plan: " + " -> ".join(self.plan.steps)); self.state="working"
        self._emit("context", "Inspecting repository structure and available tests")
        messages=[{"role":"system","content":SYSTEM}]
        last_summary=""

        has_verification = self._has_existing_verification_state()
        self._emit("context", "Repository inspection complete")
        if has_verification:
            self._emit("verify", "Running baseline verification")
            print("[BASELINE VERIFICATION]",flush=True)
            baseline=self.verifier.verify(self.verify_command)
            if baseline.passed:
                self._emit("verify_result", "Baseline verification PASSED")
                print("[PASS]",flush=True)
                print("Existing verification state passes.",flush=True)
            else:
                baseline_state="INCONCLUSIVE" if baseline.timed_out else "FAILED"
                self._emit("verify_result", "Baseline verification " + baseline_state)
                print("[INCONCLUSIVE]" if baseline.timed_out else "[FAIL]",flush=True)
                details=self._verification_output(baseline)
                if details:
                    print(("Tests failed:\n" if "pytest" in self.verify_command else "Verification failed:\n")+details,flush=True)
                else:
                    print(f"Verification failed with exit code {baseline.exit_code}.",flush=True)
                safe_output=details or f"Verification exited with code {baseline.exit_code}."
                self.context.add_event("Baseline verification failed: "+safe_output)
                recovery=self.recovery.on_failure(safe_output)
                self._log_recovery(recovery)
                if recovery["retry"]:
                    messages.append({"role":"user","content":
                        "Baseline verification failed before this task. Diagnose and repair the existing failure "
                        "while also completing the requested task. Sanitized verification output:\n"+safe_output
                    })

        for step in range(self.max_steps):
            user=self.context.prompt_view()+"\n\nCurrent plan step: "+(self.plan.steps[self.plan.current] if not self.plan.done else "Finish and verify")
            try:
                self._emit("model", "Requesting a structured action")
                current_user={"role":"user","content":user}
                raw=self.model.generate(self._bounded_model_messages(messages,current_user),json_mode=True)
                provider=getattr(self.model,"active_provider","")
                if provider:
                    actual_model=getattr(self.model,"active_model","") or getattr(self.model,"last_model","")
                    self._emit("provider", "Provider response received", provider=str(provider), model=str(actual_model or "selected by endpoint"))
                decision=self._parse_decision(raw)
                action=str(decision.get("action",""))
                action_events={"list_files":("tool","Listing repository files"),"read_file":("tool","Reading a repository file"),
                    "search_text":("tool","Searching repository text"),"search_names":("tool","Searching repository filenames"),
                    "write_file":("edit","Updating a repository file"),"run_command":("tool","Executing a repository command"),
                    "git_status":("tool","Reading Git status"),"finish":("working","Preparing task summary")}
                event = action_events.get(action)
                if event:
                    self._emit(*event)
                result=self._action(action,decision.get("arguments",{}))
                self.context.add_event(f"{action}: {result}")
                self._emit("observation", f"{action} completed")
                messages.extend([{"role":"assistant","content":raw},{"role":"user","content":"Observed result:\n"+result[:5000]}])
                if action=="finish" or decision.get("done") is True:
                    last_summary=result; break
            except Exception as exc:
                safe_error=sanitize_text(exc,self._secret_values())
                info=self.recovery.on_failure(safe_error); self.context.add_event("Failure: "+str(info))
                self._log_recovery(info)
                if not info["retry"]:
                    self.state="failed"
                    self._emit("failed", "Agent stopped after repeated failures")
                    return self._result(False, "Agent stopped after repeated failures.", error=safe_error, steps=step+1)
                messages.append({"role":"user","content":f"Action failed: {type(exc).__name__}: {safe_error}. Return exactly one valid JSON action object using the required schema, then retry."})
        else:
            self.state="failed"
            self._emit("failed", "Agent reached the maximum action limit")
            return self._result(False, "Agent reached the maximum action limit.", error="Step limit reached", steps=self.max_steps)
        self.state="verifying"; verification=self._verify_with_logging("[VERIFICATION ATTEMPT]")
        self.context.add_event(f"Verification {verification.command}: exit={verification.exit_code}")
        if not verification.passed:
            recovery=self.recovery.on_failure(self._verification_output(verification))
            self._log_recovery(recovery)
            if recovery["retry"]:
                self.state="working"
                safe_output=self._verification_output(verification)
                messages.append({"role":"user","content":f"Verification failed (exit {verification.exit_code}). Diagnose and fix based on sanitized output:\n{safe_output}"})
                # One bounded repair cycle; each turn remains subject to max_steps.
                for _ in range(min(6,self.max_steps)):
                    try:
                        self._emit("model", "Requesting a verification repair action")
                        current_user={"role":"user","content":self.context.prompt_view()}
                        raw=self.model.generate(self._bounded_model_messages(messages,current_user),json_mode=True)
                        provider=getattr(self.model,"active_provider","")
                        if provider:
                            actual_model=getattr(self.model,"active_model","") or getattr(self.model,"last_model","")
                            self._emit("provider", "Provider response received", provider=str(provider), model=str(actual_model or "selected by endpoint"))
                        decision=self._parse_decision(raw)
                        action=str(decision.get("action",""))
                        action_events={"read_file":("tool","Reading a repository file"),"search_text":("tool","Searching repository text"),
                            "write_file":("edit","Updating a repository file"),"run_command":("tool","Executing a repository command"),
                            "list_files":("tool","Listing repository files"),"search_names":("tool","Searching repository filenames"),
                            "git_status":("tool","Reading Git status")}
                        if action in action_events: self._emit(*action_events[action])
                        result=self._action(action,decision.get("arguments",{})); self.context.add_event(result)
                        self._emit("observation", f"{action} completed")
                        messages.extend([{"role":"assistant","content":raw},{"role":"user","content":"Observed result: "+result[:4000]}])
                        if decision.get("action")=="finish" or decision.get("done") is True: break
                    except Exception as exc:
                        safe_error=sanitize_text(exc,self._secret_values())
                        self.context.add_event("Recovery action failed: "+safe_error)
                        self._emit("recovery", "Repair action failed: "+safe_error)
                        break
                self.state="verifying"; verification=self._verify_with_logging("[VERIFICATION ATTEMPT]")
        self.state="complete" if verification.passed else "failed"
        verification_data={"command":verification.command,"passed":verification.passed,
                           "status":"PASSED" if verification.passed else "INCONCLUSIVE" if verification.timed_out else "FAILED",
                           "timed_out":verification.timed_out,"exit_code":verification.exit_code,
                           "stdout":sanitize_text(verification.stdout,self._secret_values()),
                           "stderr":sanitize_text(verification.stderr,self._secret_values())}
        error=None if verification.passed else (
            f"Verification inconclusive: command timed out (exit code {verification.exit_code})"
            if verification.timed_out else f"Verification failed with exit code {verification.exit_code}"
        )
        summary=last_summary or ("Task completed and verification passed." if verification.passed else "Task did not pass verification.")
        self._emit("success" if verification.passed else "failed", "Task completed" if verification.passed else "Task did not pass verification")
        return self._result(verification.passed, summary, error=error, verification=verification_data, steps=step+1)
