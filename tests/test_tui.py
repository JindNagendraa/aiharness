import subprocess
from pathlib import Path

from src import main
from src.tui import HarnessTUI, run_tui


class FakeModel:
    active_provider = "groq"
    active_model = "mock-model"


class FakeAgent:
    tasks = []
    results = []
    error = None

    def __init__(self, repo, model, *, verify_command, event_callback=None):
        self.repo = repo
        self.model = model
        self.verify_command = verify_command
        self.event_callback = event_callback

    def run(self, task):
        type(self).tasks.append(task)
        if self.event_callback:
            self.event_callback({"type": "planning", "message": "Understanding task"})
            self.event_callback({"type": "provider", "provider": "groq", "model": "mock-model"})
            self.event_callback({"type": "verify", "message": "Running verification"})
        if type(self).error is not None:
            error, type(self).error = type(self).error, None
            raise error
        if type(self).results:
            return type(self).results.pop(0)
        return {
            "success": True,
            "status": "complete",
            "summary": "Implemented and verified the change.",
            "steps": 2,
            "verification": {"command": self.verify_command, "passed": True, "stdout": "2 passed"},
        }


def setup_fake_agent(monkeypatch):
    monkeypatch.setattr(FakeAgent, "tasks", [])
    monkeypatch.setattr(FakeAgent, "results", [])
    monkeypatch.setattr(FakeAgent, "error", None)


def scripted_input(tasks):
    pending = iter(tasks)
    prompts = []

    def read(prompt):
        prompts.append(prompt)
        try:
            return next(pending)
        except StopIteration:
            raise EOFError

    return read, prompts


def test_normal_terminal_text_preserves_c_r_q_and_punctuation(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    tasks = [
        'Create a file called hello.py that prints "Hello from ForgeAI"',
        "create r q c",
        'Create a file called test.py and print "c r q"',
    ]
    read, prompts = scripted_input(tasks)
    output = []

    run_tui(tmp_path, "pytest -q", agent_factory=FakeAgent, model_factory=FakeModel,
            input_fn=read, output_fn=output.append)

    assert FakeAgent.tasks == tasks
    assert prompts[:len(tasks)] == ["> "] * len(tasks)
    assert any("[SUCCESS]" in line for line in output)
    assert any("Verification: passed" in line for line in output)
    assert any("Provider: groq" in line for line in output)
    assert any("Model: mock-model" in line for line in output)


def test_runs_again_after_success(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    output = []
    read, _ = scripted_input(["first task", "second task"])

    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=read, output_fn=output.append).run()

    assert FakeAgent.tasks == ["first task", "second task"]
    assert sum("[RUNNING]" in line for line in output) == 2
    assert sum("[SUCCESS]" in line for line in output) == 2


def test_failure_returns_to_prompt_and_next_task_can_run(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    FakeAgent.error = RuntimeError("first task failed")
    read, _ = scripted_input(["failing task", "next task"])
    output = []

    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=read, output_fn=output.append).run()

    assert FakeAgent.tasks == ["failing task", "next task"]
    assert any("first task failed" in line for line in output)
    assert any("[SUCCESS]" in line for line in output)


def test_failed_verification_is_reported_without_skipping_agent(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    FakeAgent.results = [{
        "success": False,
        "status": "failed",
        "summary": "Tests failed.",
        "error": "assertion failed",
        "verification": {"passed": False, "status": "failed", "stderr": "assert False"},
    }]
    read, _ = scripted_input(["broken task"])
    output = []

    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=read, output_fn=output.append).run()

    assert FakeAgent.tasks == ["broken task"]
    assert any("[FAILED]" in line for line in output)
    assert any("Verification: failed" in line for line in output)
    assert any("assert False" in line for line in output)


def test_ctrl_c_exits_cleanly_when_idle(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)

    def interrupt(_prompt):
        raise KeyboardInterrupt

    output = []
    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=interrupt, output_fn=output.append).run()
    assert FakeAgent.tasks == []
    assert any("Interrupted. Exiting." in line for line in output)


def test_ctrl_c_during_agent_run_exits_without_background_worker(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    FakeAgent.error = KeyboardInterrupt()
    read, _ = scripted_input(["interrupt active run"])
    output = []

    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=read, output_fn=output.append).run()

    assert FakeAgent.tasks == ["interrupt active run"]
    assert any("[INTERRUPTED]" in line for line in output)
    assert not any(thread.name.startswith("Thread-") for thread in __import__("threading").enumerate())


def test_ctrl_d_exits(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    read, _ = scripted_input([])
    output = []
    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=read, output_fn=output.append).run()
    assert any("Exiting." in line for line in output)


def test_task_errors_redact_configured_secret(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    secret = "SYNTHETIC_TEST_SECRET_DO_NOT_USE"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    FakeAgent.error = RuntimeError(f"authorization: Bearer {secret}")
    read, _ = scripted_input(["task"])
    output = []
    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=read, output_fn=output.append).run()
    rendered = "\n".join(output)
    assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_provider_model_header_uses_environment_without_secrets(tmp_path, monkeypatch):
    setup_fake_agent(monkeypatch)
    monkeypatch.setenv("PRIMARY_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "configured-model")
    read, _ = scripted_input([])
    output = []
    HarnessTUI(tmp_path, agent_factory=FakeAgent, model_factory=FakeModel,
               input_fn=read, output_fn=output.append).run()
    assert f"Repository: {tmp_path.resolve()}" in output
    assert "Provider: groq" in output
    assert "Model: configured-model" in output


def test_cli_tui_and_direct_task_paths_remain_compatible(monkeypatch, tmp_path):
    launched = []
    monkeypatch.setattr("src.tui.run_tui", lambda repo, verify: launched.append((repo, verify)))
    monkeypatch.setattr("sys.argv", ["src.main", "--repo", str(tmp_path)])
    assert main.main() == 0
    assert launched == [(str(tmp_path), "python -m pytest -q")]

    class CliAgent:
        def __init__(self, repo, model, *, verify_command, event_callback=None):
            assert repo == tmp_path
            assert verify_command == "python -m pytest -q"

        def run(self, task):
            assert task == "task from evaluator"
            return {"success": True, "status": "complete", "summary": "done",
                    "verification": {"passed": True}}

    class CliModel:
        active_provider = "mock"

    monkeypatch.setattr("src.main.Agent", CliAgent)
    monkeypatch.setattr("src.main.TextModel", CliModel)
    monkeypatch.setattr("sys.argv", ["src.main", "--repo", str(tmp_path), "task from evaluator"])
    assert main.main() == 0
    assert len(launched) == 1


def test_make_run_keeps_tui_default_and_task_repo_routes():
    root = Path(__file__).resolve().parents[1]
    interactive = subprocess.run(["make", "-n", "run"], cwd=root, capture_output=True,
                                 text=True, check=True).stdout
    selected = subprocess.run(["make", "-n", "run", "REPO=leetcode_lab"], cwd=root,
                              capture_output=True, text=True, check=True).stdout
    task = subprocess.run(["make", "-n", "run", "TASK=demo", "REPO=leetcode_lab"], cwd=root,
                          capture_output=True, text=True, check=True).stdout
    assert "--repo \".\" --tui" in interactive
    assert "--repo \"leetcode_lab\" --tui" in selected
    assert "--repo \"leetcode_lab\" \"demo\"" in task
