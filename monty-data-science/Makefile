.DEFAULT_GOAL := help
ENV_FILE ?= .env
DATASET ?= evals/datasets/small.yaml
PYTHON := uv run python

.PHONY: help install eval eval-online eval-all eval-judge eval-pytest live lint format reset-prompt \
	seed demo demo-start demo-stop demo-status demo-logs

# Background trace-traffic controls (the continuous-improvement demo). The
# pattern matches the venv console-script process; `[.]venv` is written with a
# bracket so the pgrep/pkill lines don't match themselves.
DEMO_LOG ?= demo.log
DEMO_MATCH := [.]venv/bin/demo

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

# ==============================================================================
# Continuous-improvement demo (see DEMO.md)
# ==============================================================================

seed: ## Reset the managed variable to the wrong-schema "before" state (+ fresh optimizer window)
	uv run seed-demo-variable

demo: ## Generate trace traffic in the FOREGROUND (Ctrl-C to stop). Pass args via DEMO_ARGS="--workers 5"
	uv run demo $(DEMO_ARGS)

demo-start: ## Start trace traffic in the BACKGROUND (logs to demo.log)
	@if pgrep -f "$(DEMO_MATCH)" >/dev/null 2>&1; then \
		echo "Demo traffic already running — 'make demo-status'. Use 'make demo-stop' to restart."; \
	else \
		nohup uv run demo $(DEMO_ARGS) > $(DEMO_LOG) 2>&1 & \
		sleep 2; \
		echo "Started demo traffic in the background -> $(DEMO_LOG)"; \
		echo "  watch:  make demo-logs     status: make demo-status     stop: make demo-stop"; \
	fi

demo-stop: ## Stop background trace traffic
	@pkill -INT -f "$(DEMO_MATCH)" 2>/dev/null && sleep 2 || true
	@pkill -f "$(DEMO_MATCH)" 2>/dev/null || true
	@echo "Stopped demo traffic."

demo-status: ## Show whether background traffic is running + recent runs
	@if pgrep -f "$(DEMO_MATCH)" >/dev/null 2>&1; then \
		echo "RUNNING (pid $$(pgrep -f '$(DEMO_MATCH)' | tr '\n' ' '))"; \
		printf "  completed runs: "; grep -cE '#[0-9]+ (ok|FAIL)' $(DEMO_LOG) 2>/dev/null || true; \
		grep -E '#[0-9]+ (ok|FAIL)' $(DEMO_LOG) 2>/dev/null | tail -3 || true; \
	else \
		echo "NOT running. Start with 'make demo-start'."; \
	fi

demo-logs: ## Follow the background traffic log
	@touch $(DEMO_LOG); tail -f $(DEMO_LOG)

eval: ## Run offline eval on small dataset (override with DATASET=evals/datasets/medium.yaml)
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --dataset $(DATASET) --mode offline

eval-medium: ## Run offline eval on medium dataset
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --dataset evals/datasets/medium.yaml --mode offline

eval-large: ## Run offline eval on large dataset
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --dataset evals/datasets/large.yaml --mode offline

eval-online: ## Run online cases via batch eval
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --dataset $(DATASET) --mode online

eval-all: ## Run all cases regardless of mode
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --dataset $(DATASET) --all

eval-fast: ## Run offline eval without the LLM judge (cheap smoke test)
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --dataset $(DATASET) --mode offline --no-judge

eval-pytest: ## Run evals via pytest
	ENV_FILE=$(ENV_FILE) uv run pytest evals/ -v

live: ## Run online cases through agent with live evaluators
	ENV_FILE=$(ENV_FILE) $(PYTHON) -m monty_data_science --env $(ENV_FILE)

reset-prompt: ## Bump the managed system_prompt variable forward (preserves history)
	ENV_FILE=$(ENV_FILE) $(PYTHON) -m monty_data_science.reset_prompt --env $(ENV_FILE)

wipe-prompt: ## Delete the managed variable server-side and reseed at v1 (drops all history)
	ENV_FILE=$(ENV_FILE) $(PYTHON) -m monty_data_science.reset_prompt --env $(ENV_FILE) --wipe

verify-spans: ## Query Logfire to confirm recent eval spans + managed-variable resolutions landed
	ENV_FILE=$(ENV_FILE) $(PYTHON) -m monty_data_science.verify_spans --env $(ENV_FILE)

lint: ## Run linter
	uv run ruff check src/ evals/
	uv run ruff format --check src/ evals/

typecheck: ## Run ty type checker
	uv run ty check

check: lint typecheck ## Run lint + typecheck

format: ## Auto-format code
	uv run ruff check --fix src/ evals/
	uv run ruff format src/ evals/
