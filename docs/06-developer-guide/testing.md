# Testing guide

This guide describes where tests live, how to run them locally, and how they map to CI. For coding standards and PR workflow, see [Contributing Guide](contributing.md). The canonical CI definition is `.github/workflows/ci.yaml`.

## 1. Test organization

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
│   │   ├── agents/                # Tuning-agent tests
│   │   ├── backends/
│   │   ├── ci/
│   │   ├── cli/
│   │   ├── core/                  # config, backend, launcher, patches, projection, pipeline_parallel, runtime, trainer, utils
│   │   ├── megatron/              # Megatron-specific unit tests
│   │   ├── modules/
│   │   └── tools/
│   ├── trainer/                   # Integration tests (typically need GPU + data)
│   │   ├── test_megatron_trainer.py
│   │   ├── test_torchtitan_trainer.py
│   │   └── test_maxtext_trainer.py
│   ├── scripts/                   # CI unit/integration launch scripts and UT patches
│   ├── utils.py                   # Shared test utilities
│   └── run_unit_tests.py          # Optional orchestrator (walks tests/, see below)
```

`tests/runner/run_all_tests.sh` sources shared helpers from **`runner/lib/common.sh`** at the repository root (not under `tests/runner/`).

## 2. Running tests

**Shell integration tests** (CLI behavior, config loading, hooks, environment):

```bash
bash ./tests/runner/run_all_tests.sh
```

**Python unit tests:**

```bash
pytest tests/unit_tests/ --maxfail=1 -s
```

**Trainer integration tests** (GPU and data; might require Hugging Face access):

```bash
# Megatron
DATA_PATH=<path> pytest tests/trainer/test_megatron_trainer.py -s

# TorchTitan
DATA_PATH=<path> pytest tests/trainer/test_torchtitan_trainer.py -s

# MaxText (JAX) — often run via the orchestrator in CI
python ./tests/run_unit_tests.py --jax
```

`tests/run_unit_tests.py` walks **`tests/`** and runs every `test_*.py` it finds, except for **`tests/trainer/test_maxtext_trainer.py`** in the default mode (that file is only selected when **`--jax`** is set). That means the default orchestrator run includes **`tests/unit_tests/`** and **`tests/trainer/`** (and any other matching tests), which is broader than `pytest tests/unit_tests/` alone.

**Orchestrator (default—all discovered tests except MaxText trainer):**

```bash
python ./tests/run_unit_tests.py
```

**Orchestrator (JAX / MaxText trainer only):**

```bash
python ./tests/run_unit_tests.py --jax
```

## 3. Test types

- **Shell tests:** Exercise runner scripts, CLI wiring, configuration loading, hook execution, and environment setup. Implemented as bash scripts under `tests/runner/` and orchestrated by `run_all_tests.sh`.
- **Unit tests:** Cover configuration parsing, preset loading, CLI behavior, patch registration, adapters, and other library logic under `tests/unit_tests/`.
- **Trainer tests:** End-to-end training against real backends; require AMD GPUs and appropriate data paths (and sometimes tokens). See `.github/workflows/ci.yaml` for CI values such as `DATA_PATH`, `MASTER_PORT`, and `HSA_NO_SCRATCH_RECLAIM`.

### Test tiers (slim on PRs, full on weekends)

Each trainer E2E case is a full training launch, so the suites hold one model per
architecture and per feature path. A model that only scales the dims of an
existing test is **deleted**, not kept in a slower tier — its recipe stays
schema-checked by the example-config smoke test below. What remains is split by
the **`@pytest.mark.weekly`** marker (registered in `tests/conftest.py`):

| Tier | Trigger | Model E2E | Unit tests |
| --- | --- | --- | --- |
| slim | pull request, push to `main`/tags | `-m "not weekly"` | default (slow gates skipped) |
| full | `schedule` cron (Sat 18:00 UTC), or `workflow_dispatch` with `full_tests=true` | no filter | `--run-slow` |

The workflow derives every test step's arguments from a single `PRIMUS_CI_FULL`
workflow-level env var, so there is one switch to flip.

Two kinds of case belong in the weekly tier: extended coverage that is not worth
a PR's wall clock — secondary architectures, extra precisions, variants of a
feature whose primary path stays per-PR — and new E2E cases during burn-in,
promoted to the per-PR tier by deleting the marker once a weekend run has shown
them green. Each marked case carries a comment saying what still covers its path
on PRs.

Reproduce either tier locally:

```bash
pytest tests/trainer/test_megatron_trainer.py -m "not weekly" -s   # what PR CI runs
pytest tests/trainer/test_megatron_trainer.py -s                   # full matrix
```

Two things are deliberately outside this mechanism and stay hidden in **both**
tiers: the cases `--deselect`ed in `ci.yaml` because they are broken on the
current toolchain, and the MaxText models hidden by `JAX_SKIP_UT=1`.

Do not delete the last remaining test of an architecture or of a feature path.
Two cases with identical `extra_args` are not necessarily isomorphic: the flag
that makes one unique often lives in its recipe yaml, so compare those too.

### Example-config smoke test

`tests/unit_tests/configs/test_example_configs.py` loads **every** yaml under
`examples/**/configs/` through the real config stack — env interpolation,
`extends:` merge, module/model preset merge, experiment overrides — and asserts
each resolves to a well-formed experiment whose declared `framework` matches its
backend directory. It is CPU-only (`load_primus_config` does not import
megatron/torchtitan/jax) and runs in seconds.

This is what makes deleting an isomorphic E2E safe: the recipe stops being
*trained*, but a renamed model preset, a broken `extends:` chain or a `${VAR}`
without a default still fails CI, naming the exact yaml.

## 4. Writing new tests

- **Pytest:** Add files named `test_*.py` under `tests/unit_tests/`, following existing patterns and reusing fixtures from `conftest.py` where present.
- **Shell:** Add scripts under `tests/runner/` or extend `tests/runner/run_all_tests.sh` to invoke new suites, consistent with existing `test_primus_cli*.sh` scripts.
- **Backends:** Prefer `tests/unit_tests/backends/<backend>/` for adapter-focused tests.

## 5. CI pipeline details

From `.github/workflows/ci.yaml`:

- **`code-lint`:** Python 3.12 on GitHub-hosted runners. Runs `pre-commit run --all-files --show-diff-on-failure`, so checks follow `.pre-commit-config.yaml`.
- **`dependency-review`:** Runs `actions/dependency-review-action` on pull requests to flag dependency changes.
- **`run-unittest-torch`:** Self-hosted GPU runner. Installs `requirements.txt`, runs `bash ./tests/runner/run_all_tests.sh`, then `pytest tests/unit_tests/` under coverage (`--cov=primus --cov-report=term-missing`) with specific tests `--deselect`ed (currently some `megatron/cco` TP-overlap and `megatron/transformer/moe` dispatcher cases—see the workflow file). Trainer steps set `MASTER_PORT`, `DATA_PATH`, `HSA_NO_SCRATCH_RECLAIM=1`, and `HF_TOKEN` for Megatron and TorchTitan trainer tests. A follow-up **coverage** step combines unit and E2E coverage.
- **`run-unittest-jax`:** JAX runner. Installs `requirements-jax.txt`, runs the same shell test script, then `python ./tests/run_unit_tests.py --jax` with CI environment variables (for example `JAX_SKIP_UT=1` and `DATA_PATH` as defined in the workflow).

The torch job picks its tier from `PRIMUS_CI_FULL` and prints the resolved tier in the job summary. See [Test tiers](#test-tiers-slim-on-prs-full-on-weekends).

The **`build-docker`** job builds images after lint passes; unit test jobs depend on **`code-lint`**, not on **`build-docker`**.

## 6. Pre-commit hooks

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
