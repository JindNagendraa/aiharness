import json
from pathlib import Path
import subprocess
import pytest
from src.agent import Agent
from src.context import RunContext
from src.model import ModelConfig, ModelError
from src.planner import Plan
from src.recovery import Recovery
from src.verifier import Verifier
from tools.files import FileTools, FileToolError
from tools.search import SearchTools
from tools.terminal import TerminalTool

class FakeModel:
    def __init__(self, responses): self.responses=iter(responses)
    def generate(self, messages, json_mode=False): return json.dumps(next(self.responses))

def test_model_config_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("AI_API_KEY",raising=False); monkeypatch.delenv("AI_MODEL",raising=False)
    with pytest.raises(ModelError,match="AI_API_KEY"): ModelConfig.from_env()
    monkeypatch.setenv("AI_API_KEY","example");
    with pytest.raises(ModelError,match="AI_MODEL"): ModelConfig.from_env()

def test_model_config_reads_provider_settings(monkeypatch):
    monkeypatch.setenv("AI_API_KEY","not-a-real-key"); monkeypatch.setenv("AI_MODEL","organizer-model"); monkeypatch.setenv("AI_BASE_URL","https://example.invalid/v1/")
    config=ModelConfig.from_env()
    assert (config.model,config.base_url)==("organizer-model","https://example.invalid/v1")

def test_files_scope_and_io(tmp_path):
    files=FileTools(tmp_path); files.write_file("sub/a.txt","hello")
    assert files.read_file("sub/a.txt")=="hello" and "sub/a.txt" in files.list_files()
    with pytest.raises(FileToolError): files.write_file("../escape.txt","no")
    with pytest.raises(FileToolError): files.read_file("missing")

def test_symlink_cannot_escape_root(tmp_path):
    outside=tmp_path.parent/"outside-test.txt"; outside.write_text("secret")
    (tmp_path/"link").symlink_to(outside)
    with pytest.raises(FileToolError): FileTools(tmp_path).read_file("link")

def test_search_returns_path_and_snippet(tmp_path):
    (tmp_path/"a.py").write_text("alpha\nneedle found\n")
    assert SearchTools(tmp_path).filenames("a")==["a.py"]
    assert SearchTools(tmp_path).text("needle")==[{"path":"a.py","line":2,"snippet":"needle found"}]

def test_terminal_captures_output_and_blocks_dangerous(tmp_path):
    terminal=TerminalTool(tmp_path,timeout=2)
    result=terminal.run("python -c 'print(42)'")
    assert result.exit_code==0 and "42" in result.stdout
    assert terminal.run("rm -rf anything").exit_code==126

def test_terminal_prefers_repository_virtualenv_path(tmp_path,monkeypatch):
    import os
    import sys
    venv_bin=tmp_path/".venv"/"bin"
    venv_bin.mkdir(parents=True)
    (venv_bin/"python").symlink_to(sys.executable)
    monkeypatch.setenv("PATH","/usr/bin")
    result=TerminalTool(tmp_path,timeout=2).run("python -c 'import sys; print(sys.executable)'")
    assert result.exit_code==0
    assert str(venv_bin) in result.stdout

def test_terminal_uses_harness_interpreter_when_target_has_no_venv(tmp_path,monkeypatch):
    import os
    import sys
    monkeypatch.setenv("PATH","/usr/bin")
    result=TerminalTool(tmp_path,timeout=2).run("python -c 'import sys; print(sys.executable)'")
    assert result.exit_code==0
    assert sys.executable in result.stdout

def test_context_bounded():
    ctx=RunContext("task",max_chars=100); ctx.add_file("a","x"*60); ctx.add_event("old"*20); ctx.add_event("new")
    assert ctx.size<=100 and "new" in ctx.prompt_view()

def test_plan_and_recovery():
    plan=Plan.initial("fix issue"); assert len(plan.steps)==3
    plan.advance(); assert plan.current==1
    recovery=Recovery(max_retries=1)
    first=recovery.on_failure("x")
    second=recovery.on_failure("x")
    assert first["retry"] is True
    assert second["retry"] is False and second["repeated"] is True

def test_agent_publishes_real_execution_events_and_ignores_callback_errors(tmp_path):
    events=[]
    def callback(event):
        events.append(event)
        raise RuntimeError("display unavailable")
    result=Agent(
        tmp_path,
        FakeModel([{"action":"list_files","arguments":{}},{"action":"finish","arguments":{"summary":"done"}}]),
        verify_command="true",
        event_callback=callback,
    ).run("inspect project")
    event_types=[event["type"] for event in events]
    assert result["success"] is True
    assert "planning" in event_types and "context" in event_types
    assert "tool" in event_types and "verify" in event_types
    assert "verify_result" in event_types and "success" in event_types

def test_verifier_reports_exit_status(tmp_path):
    result=Verifier(TerminalTool(tmp_path)).verify("python -c 'print(1)'")
    assert result.passed and result.exit_code==0

def test_agent_state_transitions_and_verifies(tmp_path):
    model=FakeModel([{"action":"list_files","arguments":{}},{"action":"finish","arguments":{"summary":"done"}}])
    result=Agent(tmp_path,model,verify_command="python -c 'print(\"verified\")'").run("inspect project")
    assert result["state"]=="complete" and result["verification"]["passed"]



def test_agent_rejects_missing_required_action_argument(tmp_path):
    model = FakeModel([])
    agent = Agent(
        tmp_path,
        model,
        verify_command="python -c 'print(1)'",
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="missing required argument.*query",
    ):
        agent._action("search_names", {})


def test_agent_fails_when_verification_fails(tmp_path):
    model=FakeModel([{"action":"finish","arguments":{}}])
    result=Agent(tmp_path,model,verify_command="python -c 'raise SystemExit(3)'").run("task")
    assert result["state"]=="failed" and result["verification"]["exit_code"]==3
    assert result["success"] is False and result["summary"] and result["error"]

def test_agent_logs_verification_recovery_and_pass(tmp_path,capsys):
    from src.verifier import VerificationResult
    class SequenceVerifier:
        def __init__(self): self.calls=0
        def verify(self,command):
            self.calls+=1
            if self.calls==1:
                return VerificationResult(command,False,"FAILED test_example", "",1)
            return VerificationResult(command,True,"1 passed", "",0)
    model=FakeModel([
        {"action":"finish","arguments":{"summary":"implemented"}},
        {"action":"finish","arguments":{"summary":"repaired"}},
    ])
    agent=Agent(tmp_path,model,verify_command="python -m pytest -q")
    agent.verifier=SequenceVerifier()
    result=agent.run("make tests pass")
    output=capsys.readouterr().out
    assert result["success"] is True
    assert agent.verifier.calls==2
    assert output.count("[VERIFICATION ATTEMPT]")==2 and "Running verification..." in output
    assert "[FAIL]" in output and "Tests failed:" in output and "FAILED test_example" in output
    assert "[RECOVERY]" in output and "Analyzing failure..." in output and "Applying fix..." in output
    assert "[PASS]" in output and "All tests passed." in output
    assert output.index("[VERIFICATION ATTEMPT]") < output.index("[FAIL]") < output.index("[RECOVERY]") < output.rindex("[VERIFICATION ATTEMPT]") < output.rindex("[PASS]")

def test_agent_logs_when_recovery_retry_limit_is_reached(tmp_path,capsys):
    class AlwaysFailVerifier:
        def verify(self,command):
            from src.verifier import VerificationResult
            return VerificationResult(command,False,"failure output", "",1)
    agent=Agent(tmp_path,FakeModel([{"action":"finish","arguments":{}}]),verify_command="pytest")
    agent.verifier=AlwaysFailVerifier()
    agent.recovery=Recovery(max_retries=0)
    result=agent.run("task")
    output=capsys.readouterr().out
    assert result["success"] is False
    assert "[VERIFICATION ATTEMPT]" in output and "[FAIL]" in output
    assert "[RECOVERY]" in output and "Retry limit reached; stopping recovery." in output
    assert "Applying fix..." not in output

def test_timed_out_verification_is_inconclusive_not_failed(tmp_path,capsys):
    from src.verifier import VerificationResult
    class TimedOutVerifier:
        def verify(self,command):
            return VerificationResult(command,False,"partial output", "",124,True)
    agent=Agent(tmp_path,FakeModel([{"action":"finish","arguments":{}}]),verify_command="pytest")
    agent.verifier=TimedOutVerifier()
    agent.recovery=Recovery(max_retries=0)
    result=agent.run("task")
    output=capsys.readouterr().out
    assert result["success"] is False
    assert result["verification"]["status"] == "INCONCLUSIVE"
    assert "Verification inconclusive" in result["error"]
    assert "[INCONCLUSIVE]" in output
    assert "[FAIL]" not in output

def test_baseline_failure_is_repaired_before_post_task_verification(tmp_path,capsys):
    from src.verifier import VerificationResult
    (tmp_path/"tests").mkdir()
    (tmp_path/"tests"/"test_calculator.py").write_text("# existing failing test suite\n")
    class SequenceVerifier:
        def __init__(self): self.calls=0
        def verify(self,command):
            self.calls+=1
            if self.calls==1:
                return VerificationResult(command,False,"FAILED test_calculator.py::test_add", "",1)
            return VerificationResult(command,True,"1 passed", "",0)
    class RecordingModel:
        def __init__(self):
            self.calls=[]
            self.responses=iter([
                {"action":"write_file","arguments":{"path":"calculator.py","content":"def add(a, b): return a + b\\n"}},
                {"action":"finish","arguments":{"summary":"repaired and completed"}},
            ])
        def generate(self,messages,json_mode=False):
            self.calls.append(messages)
            return json.dumps(next(self.responses))
    model=RecordingModel()
    agent=Agent(tmp_path,model,verify_command="python -m pytest -q")
    agent.verifier=SequenceVerifier()
    result=agent.run("repair the existing test failure")
    output=capsys.readouterr().out
    assert result["success"] is True
    assert agent.verifier.calls==2
    assert (tmp_path/"calculator.py").exists()
    assert "[BASELINE VERIFICATION]" in output
    assert output.index("[BASELINE VERIFICATION]") < output.index("[FAIL]") < output.index("[RECOVERY]")
    assert "[VERIFICATION ATTEMPT]" in output and "[PASS]" in output
    assert any("Baseline verification failed before this task" in message["content"]
               for message in model.calls[0] if message.get("role")=="user")

def test_new_project_skips_baseline_and_runs_normal_task(tmp_path,capsys):
    from src.verifier import VerificationResult
    class CountingVerifier:
        def __init__(self): self.calls=0
        def verify(self,command):
            self.calls+=1
            return VerificationResult(command,True,"ok", "",0)
    agent=Agent(tmp_path,FakeModel([{"action":"finish","arguments":{"summary":"created project"}}]),verify_command="python -m pytest -q")
    agent.verifier=CountingVerifier()
    result=agent.run("create a new project")
    output=capsys.readouterr().out
    assert result["success"] is True
    assert agent.verifier.calls==1
    assert "[BASELINE VERIFICATION]" not in output
    assert "[VERIFICATION ATTEMPT]" in output

def test_baseline_failure_retry_limit_does_not_block_new_task(tmp_path,capsys):
    from src.verifier import VerificationResult
    (tmp_path/"tests").mkdir()
    (tmp_path/"tests"/"test_existing.py").write_text("# test state\n")
    class SequenceVerifier:
        def __init__(self): self.calls=0
        def verify(self,command):
            self.calls+=1
            if self.calls==1:
                return VerificationResult(command,False,"baseline failure", "",1)
            return VerificationResult(command,True,"passed", "",0)
    model=FakeModel([
        {"action":"write_file","arguments":{"path":"new.py","content":"value = 1\\n"}},
        {"action":"finish","arguments":{"summary":"new task completed"}},
    ])
    agent=Agent(tmp_path,model,verify_command="python -m pytest -q")
    agent.verifier=SequenceVerifier()
    agent.recovery=Recovery(max_retries=0)
    result=agent.run("create new.py")
    output=capsys.readouterr().out
    assert result["success"] is True
    assert agent.verifier.calls==2
    assert (tmp_path/"new.py").exists()
    assert "Retry limit reached; stopping recovery." in output

def test_make_run_forwards_repo_and_defaults_to_current_directory():
    root=Path(__file__).resolve().parents[1]
    selected=subprocess.run(
        ["make","-n","run","REPO=recovery_lab","TASK=demo"],cwd=root,
        capture_output=True,text=True,check=True,
    ).stdout
    default=subprocess.run(
        ["make","-n","run","TASK=demo"],cwd=root,
        capture_output=True,text=True,check=True,
    ).stdout
    assert "-m src.main --repo \"recovery_lab\" \"demo\"" in selected
    assert "-m src.main --repo \".\" \"demo\"" in default

def test_selected_project_baseline_fails_then_recovers_and_passes(tmp_path,capsys):
    project=tmp_path/"recovery_lab"
    tests=project/"tests"
    tests.mkdir(parents=True)
    # Keep the broken source a different size so Python invalidates the .pyc
    # produced by baseline verification after the model repairs it.
    (project/"calculator.py").write_text("def divide(a, b):\n    return a * b  # intentionally broken\n")
    (tests/"test_calculator.py").write_text(
        "from calculator import divide\n\n"
        "def test_divide():\n    assert divide(10, 2) == 5\n"
    )
    model=FakeModel([
        {"action":"write_file","arguments":{"path":"calculator.py","content":"def divide(a, b):\n    return a / b\n"}},
        {"action":"finish","arguments":{"summary":"repaired calculator"}},
    ])
    agent=Agent(project,model,verify_command="python -m pytest -q")
    result=agent.run("repair calculator division")
    output=capsys.readouterr().out
    assert agent.root==project.resolve()
    assert result["success"] is True
    assert "[BASELINE VERIFICATION]" in output
    assert output.index("[BASELINE VERIFICATION]") < output.index("[FAIL]")
    assert output.index("[FAIL]") < output.index("[RECOVERY]")
    assert output.index("[RECOVERY]") < output.index("[VERIFICATION ATTEMPT]")
    assert output.index("[VERIFICATION ATTEMPT]") < output.rindex("[PASS]")
    assert "assert 20 == 5" in output
    assert "All tests passed." in output
    assert "return a / b" in (project/"calculator.py").read_text()

def test_agent_result_contract_has_summary_on_early_failure(tmp_path):
    # A broken model/action used to return an error-only result and trigger
    # KeyError in main.py when it indexed result["summary"].
    class BrokenModel:
        def generate(self, messages, json_mode=False): raise RuntimeError("model unavailable")
    result=Agent(tmp_path,BrokenModel(),max_steps=1).run("task")
    assert {"success","summary","status","state","error"}.issubset(result)
    assert result["success"] is False
    assert result["summary"]

def test_agent_bounds_chat_history_but_keeps_system_current_context_and_recent_actions():
    system = {"role": "system", "content": "Keep the restricted action protocol."}
    history = [
        system,
        {"role": "user", "content": "Baseline failed: repair it."},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old observation"},
        {"role": "assistant", "content": "recent action one"},
        {"role": "user", "content": "recent observation one"},
        {"role": "assistant", "content": "recent action two"},
        {"role": "user", "content": "recent observation two"},
    ]
    current = {"role": "user", "content": "Task: original task\nBaseline verification failed details"}

    bounded = Agent._bounded_model_messages(history, current)

    assert len(bounded) == Agent._RECENT_MODEL_MESSAGES + 2
    assert bounded[0] == system
    assert bounded[-1] == current
    assert [message["content"] for message in bounded[1:-1]] == [
        "recent action one", "recent observation one", "recent action two", "recent observation two"
    ]
    assert "Baseline verification failed" in bounded[-1]["content"]

def test_agent_sends_bounded_history_on_each_model_turn(tmp_path):
    class CapturingModel:
        active_provider = ""

        def __init__(self):
            self.calls = []
            self.responses = iter(
                [{"action": "search_names", "arguments": {"query": f"marker-{i}"}}
                 for i in range(7)]
                + [{"action": "finish", "arguments": {"summary": "done"}}]
            )

        def generate(self, messages, json_mode=False):
            self.calls.append(messages)
            return json.dumps(next(self.responses))

    model = CapturingModel()
    result = Agent(tmp_path, model, verify_command="true", max_steps=10).run("inspect many steps")

    assert result["success"] is True
    assert len(model.calls) == 8
    assert max(len(messages) for messages in model.calls) <= Agent._RECENT_MODEL_MESSAGES + 2
    assert all(messages[0]["role"] == "system" for messages in model.calls)
    assert all("Task: inspect many steps" in messages[-1]["content"] for messages in model.calls)
    assert "search_names: []" in model.calls[-1][-1]["content"]

def test_agent_retries_malformed_json_with_specific_feedback(tmp_path):
    class RecordingModel:
        def __init__(self):
            self.responses=iter(["not JSON at all", '{"action":"finish","arguments":{"summary":"recovered"}}'])
            self.calls=[]
        def generate(self,messages,json_mode=False):
            self.calls.append(messages)
            return next(self.responses)
    model=RecordingModel()
    result=Agent(tmp_path,model,verify_command="true").run("task")
    assert result["success"] is True
    assert result["summary"]=="recovered"
    assert len(model.calls)==2
    retry_prompt=model.calls[1][-1]["content"]
    assert "invalid JSON" in retry_prompt
    assert "not JSON at all" in retry_prompt

def test_agent_never_executes_command_embedded_in_harmony_text(tmp_path):
    from src.recovery import Recovery
    harmony=("<|start|>assistant<|channel|>commentary\n"
             "assistant to=run_command\n{\"command\":\"touch should_not_exist\"}\n<|end|>")
    class HarmonyModel:
        def generate(self,messages,json_mode=False): return harmony
    agent=Agent(tmp_path,HarmonyModel(),verify_command="true")
    agent.recovery=Recovery(max_retries=0)
    result=agent.run("inspect this task")
    assert result["success"] is False
    assert "invalid JSON" in result["error"]
    assert not (tmp_path/"should_not_exist").exists()

def test_malformed_json_diagnostic_redacts_provider_key(tmp_path,monkeypatch):
    secret="mock-openrouter-credential-not-real-123"
    monkeypatch.setenv("OPENROUTER_API_KEY",secret)
    class MalformedModel:
        def generate(self,messages,json_mode=False): return f"not json {secret}"
    result=Agent(tmp_path,MalformedModel(),max_steps=4).run("task")
    assert result["success"] is False
    assert "invalid JSON" in result["error"]
    assert "[REDACTED]" in result["error"]
    assert secret not in result["error"]
