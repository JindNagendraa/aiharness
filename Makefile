.PHONY: setup run test clean
PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

setup:
	@if ! test -x $(BIN)/python; then $(PYTHON) -m venv $(VENV); fi
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

run:
	PYTHONPATH=. $(BIN)/python -m src.main $(TASK)

test:
	PYTHONPATH=. $(BIN)/python -m pytest -q

clean:
	rm -rf $(VENV) .pytest_cache src/__pycache__ tools/__pycache__ tests/__pycache__
