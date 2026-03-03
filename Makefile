.PHONY: help setup format

help:
	@echo "Available targets:"
	@printf "  %-12s %s\n" "help:" "Print available make targets"
	@printf "  %-12s %s\n" "setup:" "Install all dependencies using uv in virtual environment"
	@printf "  %-12s %s\n" "format:" "Format code using ruff"

setup:
	uv sync --all-extras
	uv run pre-commit install

format:
	uv run ruff format src
	uv run ruff check --fix
