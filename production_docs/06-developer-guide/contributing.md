# Contributing Guide

This guide summarizes how to set up a development environment, follow project conventions, run checks locally, and align with the CI pipeline. For test commands and layout, see [Testing Guide](testing.md). The repository root [CONTRIBUTING.md](../../CONTRIBUTING.md) repeats branch naming, commit style, and pull request steps in short form.

## 1. Development Setup

1. **Clone the repository** (include submodules):

   ```bash
   git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
   cd Primus
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install pre-commit hooks** (recommended):

   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Optional — JAX / MaxText work:**

   ```bash
   pip install -r requirements-jax.txt
   ```

5. **Quick verification** (from the repository root, with Primus on your `PATH` or via the bundled launcher):

   ```bash
   ./primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096
   ```

## 2. Code Style

Configuration lives in `.pre-commit-config.yaml`. Hooks run automatically on `git commit` after `pre-commit install`.

| Tool | Version | Purpose |
|------|---------|---------|
| **black** | 24.8.0 | Python formatter, line length 110 |
| **isort** | 5.13.2 | Import sorting, profile `black` |
| **autoflake** | 2.3.1 | Removes unused imports and variables (see hook args for star imports and `__init__`) |
| **shellcheck** | 0.10.0.1 (shellcheck-py) | Shell script analysis |
| **pre-commit-hooks** | v4.0.1 | `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`, `check-merge-conflict` |

Manual one-off runs (repository root):

```bash
black --line-length=110 .
isort --profile black .
autoflake --remove-all-unused-imports --remove-unused-variables --expand-star-imports --ignore-init-module-imports --recursive --in-place .
```

CI runs autoflake, isort, and black in **check** mode (see [Testing Guide](testing.md) and `.github/workflows/ci.yaml`).

## 3. Branch Naming Convention

Format:

```text
<type>/<scope>/<short-description>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

**Scope (optional):** `engine`, `model`, `scheduler`, `docs`, `tests`, `config`, or another short area name.

**Examples:**

- `feat/model/implement-moe-routing`
- `fix/engine/init-error`

## 4. Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <description>
```

**Examples:**

- `feat(model): add MOE routing functionality`
- `fix(engine): resolve initialization error`

## 5. Testing Requirements

Before opening a pull request, run the following from the repository root:

- **Shell integration tests:**

  ```bash
  bash ./tests/runner/run_all_tests.sh
  ```

- **Python unit tests:**

  ```bash
  pytest tests/unit_tests/ --maxfail=1 -s
  ```

- **Backend / trainer tests** (GPU, datasets, and sometimes Hugging Face tokens): run the relevant file under `tests/trainer/` when your change touches that backend. See [Testing Guide](testing.md).

- **Pre-commit on all files:**

  ```bash
  pre-commit run --all-files
  ```

## 6. Pull Request Process

1. Fork the repository (unless you have write access and use a feature branch).
2. Create a branch that follows the naming convention above.
3. Implement changes and commit using the commit message convention.
4. Run tests and pre-commit locally.
5. Push and open a pull request with a clear description.
6. Reference related issues when applicable.
7. Request reviewers.
8. Address review feedback.
9. Ensure CI passes (lint and unit tests on the paths your PR triggers).

## 7. CI Pipeline

The workflow **`.github/workflows/ci.yaml`** defines how changes are validated.

**Triggers:** `workflow_dispatch`, pushes to `main`, tags matching `v*`, and pull requests.

**Jobs (high level):**

- **`code-lint`:** Ubuntu, Python 3.12 — autoflake (no changes allowed), isort check, black check.
- **`build-docker`:** Builds and pushes Docker images (depends on `code-lint`).
- **`run-unittest-torch`:** Self-hosted GPU runner — installs dependencies (including Primus-Turbo and AITER as defined in the workflow), runs `bash ./tests/runner/run_all_tests.sh`, `pytest` on `tests/unit_tests/` (with a few deselected tests), then Megatron and TorchTitan trainer tests with `DATA_PATH`, `MASTER_PORT`, `HSA_NO_SCRATCH_RECLAIM=1`, and `HF_TOKEN` where required.
- **`run-unittest-jax`:** JAX runner — installs `requirements-jax.txt`, runs shell tests and `python ./tests/run_unit_tests.py --jax` with CI-specific environment variables.

Lint checks mirror the pre-commit stack. Trainer jobs require GPU resources and shared secrets (for example `HF_TOKEN`) in the hosted environment.

For a focused description of local vs CI test commands, see [Testing Guide](testing.md).
