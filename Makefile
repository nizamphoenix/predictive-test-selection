SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := ${shell basename $(ROOT_DIR)}
UV_VERSION := $(shell grep 'uv =' mise.toml | cut -d '"' -f 2)
PYTHON_VERSION := $(shell grep 'python =' mise.toml | cut -d '"' -f 2)

.PHONY: sync-py-versions
sync-py-versions:
	@echo "Syncing Python versions across pyproject.toml and .python-version files..."
	@echo "$(PYTHON_VERSION)" > .python-version
	@sed 's/requires-python = .*/requires-python = ">=$(shell echo $(PYTHON_VERSION) | cut -d. -f1,2)"/' pyproject.toml > pyproject.toml.tmp
	@mv pyproject.toml.tmp pyproject.toml
	@echo "✅ Versions synced: Python $(PYTHON_VERSION)"

.PHONY: install-pre-commit-hooks
install-pre-commit-hooks:
	@echo "Installing pre-commit hooks..."
	@uv run pre-commit install --hook-type pre-commit --hook-type pre-push
	@echo "✅pre-commit hooks installed successfully"

.PHONY: install
install:
	@echo "Installing all dependencies..."
	@uv sync --all-groups
	@echo "✅all dependencies installed successfully"

.PHONY: install-prod
install-prod:
	@echo "Installing production dependencies..."
	@uv sync --no-dev --no-editable
	@echo "✅production dependencies installed successfully"

.PHONY: install-main-and-dev-deps
install-main-and-dev-deps:
	@echo "Installing main and dev dependencies..."
	@uv sync
	@echo "✅main and dev dependencies installed successfully"

.PHONY: install-only-main-deps
install-only-main-deps:
	@echo "Installing main dependencies..."
	@uv sync --no-dev
	@echo "✅main dependencies installed successfully"

.PHONY: setup-local-env
setup-local-env: install-pre-commit-hooks install sync-py-versions
	@echo "✅Local environment set up successfully"

.PHONY: add-main-deps
add-main-deps:
	@echo "Adding main dependency..."
	@read -p "Enter package name: " pkg; \
	uv add $$pkg

# Usage:
#   make add-group-deps                <- This will prompt you for both
#   make add-group-deps pkg=ruff       <- Prompts for group only
#   make add-group-deps pkg=ruff g=viz <- Runs instantly
.PHONY: add-group-deps
add-group-deps:
	@pkg=$(pkg); \
	group=$(g); \
	if [ -z "$$pkg" ]; then \
		read -p "Enter package name: " pkg; \
	fi; \
	if [ -z "$$group" ]; then \
		read -p "Enter group name (e.g., lint, viz, dev): " group; \
	fi; \
	echo "Adding $$pkg to group [$$group]..."; \
	uv add $$pkg --group $$group
	@echo "✅$$pkg added to group [$$group] successfully"

.PHONY: remove-main-deps
remove-main-deps:
	@echo "Removing main dependency..."
	@read -p "Enter package name: " pkg; \
	uv remove $$pkg
	@echo "✅$$pkg removed from main dependencies successfully"

.PHONY: remove-group-deps
remove-group-deps:
	@pkg=$(pkg); \
	group=$(g); \
	if [ -z "$$pkg" ]; then \
		read -p "Enter package name: " pkg; \
	fi; \
	if [ -z "$$group" ]; then \
		read -p "Enter group name (e.g., lint, viz, dev): " group; \
	fi; \
	echo "Removing $$pkg from group [$$group]..."; \
	uv remove $$pkg --group $$group
	@echo "✅$$pkg removed from group [$$group] successfully"

.PHONY: check-py-types
check-py-types:
	@echo "Checking Python types with mypy..."
	@uvx ty check
	@echo "✅Python type check passed successfully"

.PHONY: run-pre-commit
run-pre-commit:
	@echo "Running pre-commit hooks..."
	@uvx pre-commit run --all-files
	@echo "✅pre-commit hooks ran successfully"
