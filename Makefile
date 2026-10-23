.PHONY: help setup format lint train-eps-greedy train-eps-annealing train-boltzmann train-max-boltzmann train-all clean

PYTHON := uv run python

help:
	@echo "Available commands:"
	@echo "  make setup                - Install project dependencies via uv"
	@echo "  make format               - Auto-format code and fix linting errors"
	@echo "  make lint                 - Check code for linting and formatting issues"
	@echo "  make train-eps-greedy     - Train Exp 1: Epsilon-Greedy strategy"
	@echo "  make train-eps-annealing  - Train Exp 2: Epsilon Annealing strategy"
	@echo "  make train-boltzmann      - Train Exp 3: Boltzmann exploration"
	@echo "  make train-max-boltzmann  - Train Exp 4: Max-Boltzmann exploration"
	@echo "  make train-all            - Run all four training experiments sequentially"
	@echo "  make clean                - Remove cache, temporary files, and local wandb logs"

setup:
	uv sync --all-extras

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

train-eps-greedy:
	@echo "Running Experiment 1: Epsilon-Greedy..."
	$(PYTHON) -m src.train --strategy eps_greedy --start_epsilon 0.1

train-eps-annealing:
	@echo "Running Experiment 2: Epsilon Annealing..."
	$(PYTHON) -m src.train --strategy eps_annealing --start_epsilon 1.0 --final_epsilon 0.1 --anneal_steps 30000

train-boltzmann:
	@echo "Running Experiment 3: Boltzmann Exploration..."
	$(PYTHON) -m src.train --strategy boltzmann --start_temperature 1.5

train-max-boltzmann:
	@echo "Running Experiment 4: Max-Boltzmann Exploration..."
	$(PYTHON) -m src.train --strategy max_boltzmann --start_epsilon 0.5 --start_temperature 1.0

train-all: train-eps-greedy train-eps-annealing train-boltzmann train-max-boltzmann

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache wandb
	find . -type d -name "__pycache__" -exec rm -rf {} +