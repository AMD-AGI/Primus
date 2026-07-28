# Primus Tests

This directory contains the test suite for Primus.

## Test Structure

```
tests/
├── runner/                    # Shell integration tests for primus-cli
│   ├── run_all_tests.sh       # Master test runner
│   ├── lib/                   # Library function tests
│   ├── helpers/               # Hook and environment tests
│   └── test_primus_cli*.sh    # CLI mode tests
├── unit_tests/                # Python unit tests (pytest)
│   ├── agents/                # Tuning-agent tests
│   ├── backends/              # Backend-specific tests (megatron, torchtitan, maxtext, ...)
│   ├── ci/                    # CI helper tests
│   ├── cli/                   # CLI tests
│   ├── core/                  # Core library tests (config/, patches/, backend/, launcher/, projection/, pipeline_parallel/, runtime/, trainer/, utils/)
│   ├── megatron/              # Megatron-specific unit tests
│   ├── modules/               # Module/trainer tests
│   └── tools/                 # Tooling tests
├── trainer/                   # Integration tests (require GPU)
│   ├── test_megatron_trainer.py
│   ├── test_torchtitan_trainer.py
│   └── test_maxtext_trainer.py
├── scripts/                   # CI unit/integration launch scripts and UT patches
├── utils.py                   # Shared test utilities
├── conftest.py                # Shared pytest fixtures
└── run_unit_tests.py          # Python test orchestrator (walks tests/)
```

> **Note:** `config/` and `patches/` live under `unit_tests/core/` (i.e. `tests/unit_tests/core/config/` and `tests/unit_tests/core/patches/`), not directly under `unit_tests/`.

## Running Tests

```bash
# Shell integration tests
bash ./tests/runner/run_all_tests.sh

# Python unit tests
pytest tests/unit_tests/ --maxfail=1 -s

# All tests via orchestrator
python ./tests/run_unit_tests.py          # Torch backends
python ./tests/run_unit_tests.py --jax    # JAX/MaxText backend
```

For comprehensive testing documentation, see the [Testing Guide](../docs/06-developer-guide/testing.md).
