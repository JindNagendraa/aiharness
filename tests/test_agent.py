import json
from pathlib import Path
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

def test_context_bounded():
    ctx=RunContext("task",max_chars=100); ctx.add_file("a","x"*60); ctx.add_event("old"*20); ctx.add_event("new")
    assert ctx.size<=100 and "new" in ctx.prompt_view()

def test_plan_and_recovery():
    plan=Plan.initial("fix issue"); assert len(plan.steps)==3
    plan.advance(); assert plan.current==1
    recovery=Recovery(max_retries=1)
    assert recovery.on_failure("x")["retry"] is True
    assert recovery.on_failure("x")["retry"] is False

def test_verifier_reports_exit_status(tmp_path):
    result=Verifier(TerminalTool(tmp_path)).verify("python -c 'print(1)'")
    assert result.passed and result.exit_code==0

def test_agent_state_transitions_and_verifies(tmp_path):
    model=FakeModel([{"action":"list_files","arguments":{}},{"action":"finish","arguments":{"summary":"done"}}])
    result=Agent(tmp_path,model,verify_command="python -c 'print(\"verified\")'").run("inspect project")
    assert result["state"]=="complete" and result["verification"]["passed"]

def test_agent_fails_when_verification_fails(tmp_path):
    model=FakeModel([{"action":"finish","arguments":{}}])
    result=Agent(tmp_path,model,verify_command="python -c 'raise SystemExit(3)'").run("task")
    assert result["state"]=="failed" and result["verification"]["exit_code"]==3
    assert result["success"] is False and result["summary"] and result["error"]

def test_agent_result_contract_has_summary_on_early_failure(tmp_path):
    # A broken model/action used to return an error-only result and trigger
    # KeyError in main.py when it indexed result["summary"].
    class BrokenModel:
        def generate(self, messages, json_mode=False): raise RuntimeError("model unavailable")
    result=Agent(tmp_path,BrokenModel(),max_steps=1).run("task")
    assert {"success","summary","status","state","error"}.issubset(result)
    assert result["success"] is False
    assert result["summary"]

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

def test_malformed_json_diagnostic_redacts_provider_key(tmp_path,monkeypatch):
    secret="sk-or-v1-mocked-secret-credential"
    monkeypatch.setenv("OPENROUTER_API_KEY",secret)
    class MalformedModel:
        def generate(self,messages,json_mode=False): return f"not json {secret}"
    result=Agent(tmp_path,MalformedModel(),max_steps=4).run("task")
    assert result["success"] is False
    assert "invalid JSON" in result["error"]
    assert "[REDACTED]" in result["error"]
    assert secret not in result["error"]
