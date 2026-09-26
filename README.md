# AI Coding-Agent Harness

A text-only Python coding agent that inspects a target repository, plans work, uses a small JSON action protocol to read/search/edit files and run commands, then verifies the result. It does not execute model-generated Python.

## Architecture

`src/main.py` parses the CLI; `src/agent.py` coordinates a bounded action loop; `src/model.py` routes through a provider-agnostic interface in `src/providers/`; `context.py`, `planner.py`, `recovery.py`, and `verifier.py` manage working memory, plan state, bounded recovery, and verification. `tools/` contains repository-scoped file/search operations, timed command execution, and read-only Git inspection.

## Requirements and setup

Python 3.10+ and `make` are required. From this directory run:

```sh
python3 -m venv .venv
source .venv/bin/activate
make setup
```

Set credentials and organizer model configuration in the environment. Never put a real key in source or commit it:

```sh
export OPENROUTER_API_KEY="YOUR_KEY"
export OPENROUTER_MODEL="YOUR_MODEL"
export GROQ_API_KEY="YOUR_KEY" # optional
export GROQ_MODEL="YOUR_MODEL" # choose a model available to your account
export PRIMARY_PROVIDER=openrouter
export FALLBACK_PROVIDERS=groq
# Or reverse the order:
export PRIMARY_PROVIDER=groq
export FALLBACK_PROVIDERS=openrouter
# Hackathon evaluator settings route through the OpenRouter adapter:
export AI_API_KEY="<PROVIDED_API_KEY>"
export AI_MODEL="<ORGANIZER_PRESCRIBED_MODEL>"
export AI_BASE_URL="<EVALUATOR_API_BASE_URL>" # optional
```

`.env.example` contains empty placeholders. The harness reads environment variables directly; `.env` is ignored. Never commit API keys. Evaluator `AI_API_KEY` and `AI_MODEL` take precedence and route through the OpenRouter adapter; set `AI_BASE_URL` if the evaluator supplies a compatible endpoint other than OpenRouter.

## Run

Run from the harness root, targeting the current directory:

```sh
make run TASK='Fix the bug in this repository'
```

Or target another repository:

```sh
.venv/bin/python -m src.main --repo /path/to/repository 'Fix the bug'
```

Without a positional task the CLI prompts interactively. Verification defaults to `python -m pytest -q`; override with `--verify 'command'`.

## Tests

```sh
make test
```

## Tools and workflow

The agent starts with a simple inspect/implement/verify plan, sends bounded context to the selected model, validates each response as a JSON action, executes only allowlisted actions, records results, and verifies before reporting completion. OpenRouter uses its Chat Completions HTTP API; Groq uses its official Python SDK. Both require an explicitly configured model. `PRIMARY_PROVIDER` chooses the first attempt; `FALLBACK_PROVIDERS` sets ordered alternatives. Each provider is attempted at most once per generation call. Errors are sanitized.

File paths are resolved under the selected repository root. The terminal has a timeout and rejects common destructive commands. Git support is read-only; it never commits or pushes. Add a tool by implementing a narrow class under `tools/`, adding a named action dispatch in `src/agent.py`, and testing the behavior.

## Security and limitations

API credentials are passed only to provider SDKs and omitted from error output. Commands run with the current user's permissions, so the command blocklist is a guardrail rather than a complete sandbox. Run against repositories you trust. JSON actions are requested in the prompt and validated by the agent. Reviewers do not currently receive diffs because doing so transmits repository source to external services.

## Evaluation workflow

From a clean checkout, set provider credentials or evaluator `AI_API_KEY` and `AI_MODEL`, then run `make setup`, `make test`, and `make run TASK='...'`. `make clean` removes the virtual environment and generated Python/pytest caches.
