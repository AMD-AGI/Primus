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
│   ├── backends/              # Backend-specific tests
│   ├── cli/                   # CLI tests
│   ├── config/                # Configuration system tests
│   └── patches/               # Patch system tests
├── trainer/                   # Integration tests (require GPU)
│   ├── test_megatron_trainer.py
│   ├── test_torchtitan_trainer.py
│   └── test_maxtext_trainer.py
└── run_unit_tests.py          # Python test orchestrator
```

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

For comprehensive testing documentation, see the [Testing Guide](../production_docs/06-developer-guide/testing.md).
