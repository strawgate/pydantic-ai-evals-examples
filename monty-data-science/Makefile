.DEFAULT_GOAL := help
ENV_FILE ?= .env
DATASET ?= evals/datasets/small.yaml
PYTHON := uv run python

.PHONY: help install eval eval-online eval-all eval-judge eval-pytest live lint format reset-prompt

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

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
