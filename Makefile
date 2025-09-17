.DEFAULT_GOAL := help

.PHONY: help
help: ##@ List available commands with their descriptions
	@printf "\nUsage: make <command>\n"
	@grep -F -h "##@" $(MAKEFILE_LIST) | grep -F -v grep -F | sed -e 's/\\$$//' | awk 'BEGIN {FS = ":*[[:space:]]*##@[[:space:]]*"}; \
	{ \
		if($$2 == "") \
			pass; \
		else if($$0 ~ /^#/) \
			printf "%s", $$2; \
		else if($$1 == "") \
			printf "     %-20s%s", "", $$2; \
		else \
			printf "    \033[34m%-20s\033[0m %s\n", $$1, $$2; \
	}'

.PHONY: install
install: ##@ Create the virtual environment and install the pre-commit hooks
	@echo "Creating virtual environment using uv..."
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ##@ Run code quality tools.
	@echo "Checking lock file consistency with 'pyproject.toml'..."
	@uv lock --locked
	@echo "Linting code..."
	@uv run pre-commit run -a
	@echo "Static type checking..."
	@uv run ty check

.PHONY: test
test: ##@ Test the code with pytest
	@echo "Testing code..."
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=html

.PHONY: tox
tox: ##@ Run tox to test the code with all supported Python versions
	@uv run tox

.PHONY: build
build: ##@ Build wheel file
	@make clean-build
	@echo "Creating wheel file..."
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ##@ Clean build artifacts
	@echo "Removing build artifacts..."
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: docs-test
docs-test: ##@ Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ##@ Build and serve the documentation
	@uv run mkdocs serve

.PHONY: clean
clean: ##@ Clean up the project
	@echo "Cleaning up the project..."
	@rm -rf `find . -name __pycache__`
	@rm -f `find . -type f -name '*.py[co]'`
	@rm -f `find . -type f -name '*~'`
	@rm -f `find . -type f -name '.*~'`
	@rm -f `find . -type f -name '*.log'`
	@rm -rf `find . -type d -name '.ipynb_checkpoints'`
	@rm -rf `find . -type d -name '*.egg-info'`
	@rm -rf .cache
	@rm -rf dist
	@rm -rf .mypy_cache
	@rm -rf .venv
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@rm -rf htmlcov
	@rm -f .coverage*
	@rm -rf target
	@rm -rf .tox

.PHONY: jupyterlab
jupyterlab: ##@ Spin up JupyterLab
	@uv run jupyter lab
