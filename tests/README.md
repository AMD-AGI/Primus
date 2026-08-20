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

## Test Tiers

Every trainer E2E case is a full training launch, so the suites hold **one model
per architecture and per feature path**. Anything beyond that is either deleted
or deferred to the weekend:

- **Deleted.** A model that only scales the dims of an existing test earns no
  coverage. Its recipe is still schema-checked by
  `unit_tests/configs/test_example_configs.py`, which loads every yaml under
  `examples/**/configs/`.
- **`@pytest.mark.weekly`.** Extended coverage worth having but not worth a PR's
  wall clock, and new cases during burn-in. Deselected on PRs and pushes; run by
  the weekend scheduled build (Saturday 18:00 UTC), which you can also trigger by
  hand via the `Primus-CI-TAS` workflow with `full_tests=true`. Promote a case to
  the per-PR tier by deleting the marker once a weekend run has shown it green.
- **Per-PR.** Everything else.

```bash
# What PR CI runs
pytest tests/trainer/test_megatron_trainer.py -m "not weekly" -s

# The weekend tier: full matrix, plus the slow unit-test shape gates
pytest tests/trainer/test_megatron_trainer.py -s
pytest tests/unit_tests/ --run-slow -s
```

Before removing a model, check it is not the last user of a feature path. The
difference often lives in the recipe yaml rather than in the test body, so two
cases whose `extra_args` look identical can still cover different code — diff the
recipes before concluding they are isomorphic.

Cases hidden by `--deselect` in `ci.yaml` (broken on the current toolchain) and
by `JAX_SKIP_UT=1` (most MaxText models) stay hidden in **both** tiers; they are
not part of this mechanism.

For comprehensive testing documentation, see the [Testing Guide](../docs/06-developer-guide/testing.md).
