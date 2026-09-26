# ForgeAI AI Coding Harness

ForgeAI is a text-only coding harness for focused engineering tasks in an existing repository. It uses a structured action protocol, scoped repository tools, verification, and bounded recovery. The official user and execution interface is the terminal UI (TUI), launched with `make run`.

## Architecture

```text
make run → TUI → Agent → TextModel → OpenRouter / Groq
                    ↓
             Repository tools → Verification → bounded Recovery → TUI result
```

The TUI wraps the real `src.agent.Agent`; it does not duplicate planning, provider selection, tool execution, verification, or recovery.

## TUI

The TUI shows the selected repository, active provider and model (when returned by the provider), harness state, Agent activity, verification output, recovery attempts, and the result returned by the Agent. The Agent runs in a background thread so the interface can continue to update. Statuses include `READY`, `PLANNING`, `WORKING`, `VERIFYING`, `RECOVERING`, `SUCCESS`, and `FAILED`.

Controls:

- Enter: run the entered task
- `c`: clear after a submitted task (Ctrl+U clears while editing)
- `r`: rerun the previous task
- `q`: quit when no run is active and the task is not being edited (Ctrl+U clears a draft first)
- Backspace: edit the current task

## Agent and tools

The Agent plans a bounded sequence of actions and accepts only structured JSON decisions. Tools in `tools/` provide repository-scoped file access, search, timed terminal commands, and read-only Git status. File tools prevent paths and symlinks from escaping the selected repository. Terminal commands have a timeout and block common destructive operations.

## Verification and recovery

The harness runs the configured verification command after the Agent finishes. Repositories with recognizable tests receive a baseline verification first. Verification failures are reported to the Agent for a bounded repair attempt, subject to existing retry and action limits. The TUI shows verification output and recovery activity emitted by the real Agent.

## Provider configuration

Provider credentials are read from environment variables and are never stored in source files. Configure either provider or both to enable fallback:

```sh
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
export OPENROUTER_MODEL="MODEL_AVAILABLE_TO_YOUR_ACCOUNT"
export GROQ_API_KEY="YOUR_GROQ_KEY" # optional fallback
export GROQ_MODEL="MODEL_AVAILABLE_TO_YOUR_ACCOUNT" # optional fallback
export PRIMARY_PROVIDER=openrouter
export FALLBACK_PROVIDERS=groq
```

The evaluation environment may supply only `AI_API_KEY`; this routes through OpenRouter. `AI_MODEL` and `AI_BASE_URL` are also supported when the evaluator specifies them. If the evaluator endpoint chooses a model when `model` is omitted, the harness leaves that choice to the endpoint rather than guessing a model name. An actual task run still requires working evaluator credentials and an endpoint that supports the harness's native tool definitions. The harness does not make provider calls during setup or tests.

## Setup

Python 3.10+ and `make` are required. From the repository root:

```sh
export AI_API_KEY="PROVIDED_API_KEY"
make setup
```

`make setup` creates/uses `.venv` and installs the Python dependencies. The evaluator provides the value for `AI_API_KEY`; export that value in your shell. No source changes or extra setup files are required. Keep credentials in environment variables, not tracked files.

## Running

The primary workflow launches the TUI:

```sh
make run
```

The default repository is `.`. To target a different repository:

```sh
make run REPO=leetcode_lab
```

For automation or a pre-supplied task, the existing noninteractive CLI remains available through Make:

```sh
make run TASK="Fix the bug in this repository" REPO=.
```

The CLI also accepts `--verify 'command'` when invoked directly with `python -m src.main`. Verification defaults to `python -m pytest -q`.

## Testing

```sh
make test
python -m pytest -q
```

Tests use mocked providers and do not make real model API calls.

## Security

The harness never hard-codes provider credentials. Provider errors and Agent diagnostics redact known credentials. File operations stay inside the chosen repository. The terminal command guard is a safety measure, not a complete sandbox; only run the harness on repositories and tasks you trust. No automatic commit or push is performed.
