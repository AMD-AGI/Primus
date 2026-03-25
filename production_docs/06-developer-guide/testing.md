# Testing Guide

This guide describes where tests live, how to run them locally, and how they map to CI. For coding standards and PR workflow, see [Contributing Guide](contributing.md). The canonical CI definition is `.github/workflows/ci.yaml`.

## 1. Test Organization

Layout (simplified from the repository root):

```text
Primus/
├── runner/
│   └── lib/
│       └── common.sh              # Shared logging/helpers sourced by the shell test runner
├── tests/
│   ├── runner/                    # Shell integration tests
│   │   ├── run_all_tests.sh       # Master shell test runner
│   │   ├── lib/                   # test_common.sh, test_config.sh, test_validation.sh
│   │   ├── helpers/               # Hook and env tests
│   │   ├── test_primus_cli.sh
│   │   ├── test_primus_cli_direct.sh
│   │   ├── test_primus_cli_container.sh
│   │   └── test_primus_cli_slurm.sh
│   ├── unit_tests/                # Python unit tests (pytest)
│   │   ├── backends/
│   │   ├── cli/
│   │   ├── config/
│   │   └── patches/
│   ├── trainer/                   # Integration tests (typically need GPU + data)
│   │   ├── test_megatron_trainer.py
│   │   ├── test_torchtitan_trainer.py
│   │   └── test_maxtext_trainer.py
│   └── run_unit_tests.py          # Optional orchestrator (walks tests/, see below)
```

`tests/runner/run_all_tests.sh` sources shared helpers from **`runner/lib/common.sh`** at the repository root (not under `tests/runner/`).

## 2. Running Tests

**Shell integration tests** (CLI behavior, config loading, hooks, environment):

```bash
bash ./tests/runner/run_all_tests.sh
```

**Python unit tests:**

```bash
pytest tests/unit_tests/ --maxfail=1 -s
```

**Trainer integration tests** (GPU and data; may require Hugging Face access):

```bash
# Megatron
DATA_PATH=<path> pytest tests/trainer/test_megatron_trainer.py -s

# TorchTitan
DATA_PATH=<path> pytest tests/trainer/test_torchtitan_trainer.py -s

# MaxText (JAX) — often run via the orchestrator in CI
python ./tests/run_unit_tests.py --jax
```

`tests/run_unit_tests.py` walks **`tests/`** and runs every `test_*.py` it finds, except for **`tests/trainer/test_maxtext_trainer.py`** in the default mode (that file is only selected when **`--jax`** is set). That means the default orchestrator run includes **`tests/unit_tests/`** and **`tests/trainer/`** (and any other matching tests), which is broader than `pytest tests/unit_tests/` alone.

**Orchestrator (default — all discovered tests except MaxText trainer):**

```bash
python ./tests/run_unit_tests.py
```

**Orchestrator (JAX / MaxText trainer only):**

```bash
python ./tests/run_unit_tests.py --jax
```

## 3. Test Types

- **Shell tests:** Exercise runner scripts, CLI wiring, configuration loading, hook execution, and environment setup. Implemented as bash scripts under `tests/runner/` and orchestrated by `run_all_tests.sh`.
- **Unit tests:** Cover configuration parsing, preset loading, CLI behavior, patch registration, adapters, and other library logic under `tests/unit_tests/`.
- **Trainer tests:** End-to-end training against real backends; require AMD GPUs and appropriate data paths (and sometimes tokens). See `.github/workflows/ci.yaml` for CI values such as `DATA_PATH`, `MASTER_PORT`, and `HSA_NO_SCRATCH_RECLAIM`.

## 4. Writing New Tests

- **Pytest:** Add files named `test_*.py` under `tests/unit_tests/`, following existing patterns and reusing fixtures from `conftest.py` where present.
- **Shell:** Add scripts under `tests/runner/` or extend `tests/runner/run_all_tests.sh` to invoke new suites, consistent with existing `test_primus_cli*.sh` scripts.
- **Backends:** Prefer `tests/unit_tests/backends/<backend>/` for adapter-focused tests.

## 5. CI Pipeline Details

From `.github/workflows/ci.yaml`:

- **`code-lint`:** Python 3.12 on GitHub-hosted runners. Runs autoflake (must produce no output diff), isort `--check-only`, and black `--check` with line length 110.
- **`run-unittest-torch`:** Self-hosted GPU runner. Installs `requirements.txt`, runs `bash ./tests/runner/run_all_tests.sh`, then `pytest tests/unit_tests/` with specific tests deselected (see the workflow file). Trainer steps set `MASTER_PORT`, `DATA_PATH`, `HSA_NO_SCRATCH_RECLAIM=1`, and `HF_TOKEN` for Megatron and TorchTitan trainer tests.
- **`run-unittest-jax`:** JAX runner. Installs `requirements-jax.txt`, runs the same shell test script, then `python ./tests/run_unit_tests.py --jax` with CI environment variables (for example `JAX_SKIP_UT=1` and `DATA_PATH` as defined in the workflow).

The **`build-docker`** job builds images after lint passes; unit test jobs depend on **`code-lint`**, not on **`build-docker`**.

## 6. Pre-commit Hooks

Install once per clone:

```bash
pip install pre-commit
pre-commit install
```

Run manually on the whole tree:

```bash
pre-commit run --all-files
```

Hooks include: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`, `check-merge-conflict`, `isort`, `autoflake`, `black`, and `shellcheck` (as configured in `.pre-commit-config.yaml`). These align with the **`code-lint`** job in CI; see [Contributing Guide](contributing.md) for manual equivalents.
