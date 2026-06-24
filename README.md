# template_repo

Template Repo

### Quickstart

- Install `mise`.

```bash
brew install mise
```

- Activate `mise` in your shell.

```bash
mise activate zsh
```

Or, add `eval "$(mise activate zsh)"` to your shell config (e.g. `~/.zshrc`) to
avoid having to run the activation command manually each time.

- Install prerequisite tools and pin versions managed by `mise`.

```bash
mise up
```

- Sync Python versions across `pyproject.toml` and `.python-version` files.

```bash
make sync-py-versions
```

- Setup Local Environment. We use [uv](https://docs.astral.sh/uv/) to manage our
  Python virtual environment and dependencies. To set up this repo and get
  started quickly, run the following,

```bash
make setup-local-env
```

This command will:

- Create a Python virtual environment (`.venv`) under the project directory.
- Install all required dependencies as specified in `pyproject.toml`.
- Set up pre-commit hooks defined in `.pre-commit-config.yaml` for code quality
  checks.

Thereafter, python commands can be run using `uv run python <script_name>.py` to
ensure they execute within `.venv`.

Other useful Makefile targets:

- `make add-group-deps` — adds dependency
- `make remove-group-deps` — removes dependency

Refer to the `Makefile` for more details.
