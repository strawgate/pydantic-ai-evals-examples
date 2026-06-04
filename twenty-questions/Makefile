.DEFAULT_GOAL := help
ENV_FILE ?= .env
PYTHON := uv run python

.PHONY: help install eval eval-online eval-all eval-judge live lint format

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

eval: ## Run offline dataset eval (training cases)
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --mode offline

eval-online: ## Run online cases via batch eval
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --mode online

eval-all: ## Run all cases regardless of mode
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --all

eval-judge: ## Run offline eval with LLM judge
	ENV_FILE=$(ENV_FILE) $(PYTHON) evals/run_eval.py --env $(ENV_FILE) --mode offline --judge

eval-pytest: ## Run evals via pytest
	ENV_FILE=$(ENV_FILE) uv run pytest evals/ -v

live: ## Run online cases with live evaluators
	ENV_FILE=$(ENV_FILE) $(PYTHON) -m twenty_questions --env $(ENV_FILE)

lint: ## Run linter
	uv run ruff check src/ evals/
	uv run ruff format --check src/ evals/

format: ## Auto-format code
	uv run ruff check --fix src/ evals/
	uv run ruff format src/ evals/
