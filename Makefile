.PHONY: setup run test clean
PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
REPO ?= .

setup:
	@if ! test -x $(BIN)/python; then $(PYTHON) -m venv $(VENV); fi
	$(BIN)/pip install -r requirements.txt

run:
	@if [ -n "$(strip $(TASK))" ]; then \
		PYTHONPATH=. $(BIN)/python -m src.main --repo "$(REPO)" "$(TASK)"; \
	else \
		PYTHONPATH=. $(BIN)/python -m src.main --repo "$(REPO)" --tui; \
	fi

test:
	PYTHONPATH=. $(BIN)/python -m pytest -q tests

clean:
	rm -rf $(VENV)
	find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
