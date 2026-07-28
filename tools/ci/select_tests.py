###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Map a PR's changed files (stdin, one per line) to the E2E trainer suites to
run (or "all"). A single classify() decides each path's blast radius.
Conventions over hard-coded tables:
  - E2E suites are auto-discovered from tests/trainer/test_<name>_trainer.py;
  - a backend is named by its dir (primus/backends/<X> or examples/<X>).
Fail-safe is the only invariant: anything global, a non-backend source change,
or a backend without a trainer expands to everything -- over-select, never
under-select.

    git diff --name-only "$BASE" HEAD | python tools/ci/select_tests.py

Unit tests are deliberately NOT selected/narrowed here: the whole suite only
takes ~5 minutes (vs. each E2E suite's tens of minutes), so narrowing risked
under-selection for little wall-clock gain. ci.yaml always runs it in full.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The only hard-coded list: changes whose blast radius is the whole repo.
# Scoped to paths *outside* primus/, tests/unit_tests/, and examples/ --
# anything under those trees that isn't a recognized backend already falls
# back to "component" below, which select_e2e() treats the same as "global"
# anyway, so listing e.g. primus/core/launcher/ here would be redundant.
# Being absent here only ever falls back to other fail-safe paths, never "skip".
GLOBAL_TRIGGERS = (
    ".github/",
    "tools/",
    "runner/",  # the launcher drives all training
    "pyproject.toml",
    "tests/utils.py",
    "tests/conftest.py",
    "tests/unit_tests/conftest.py",
    "tests/run_unit_tests.py",
)

_TRAINER_RE = re.compile(r"test_(.+)_trainer")
_BACKEND_RE = re.compile(r"(?:primus/backends|examples)/([^/]+)/")


def discover_e2e_suites():
    return {
        m.group(1)
        for p in (ROOT / "tests/trainer").glob("test_*_trainer.py")
        if (m := _TRAINER_RE.fullmatch(p.stem))
    }


def _is_global(path):
    if "/" not in path and path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(path == t or path.startswith(t) for t in GLOBAL_TRIGGERS)


def classify(path):
    """('global', None) | ('backend', name) | ('component', None) | ('ignore', None)."""
    if _is_global(path):
        return ("global", None)
    backend = _BACKEND_RE.match(path)  # primus/backends/<X>/ or examples/<X>/
    if backend:
        return ("backend", backend.group(1))
    if path.startswith("tests/trainer/"):
        m = _TRAINER_RE.search(path)
        return ("backend", m.group(1)) if m else ("ignore", None)
    # A bare examples/<file> (no backend subdir, so it didn't match _BACKEND_RE
    # above) is shared launcher plumbing, not per-backend -- e.g. maxtext's E2E
    # shells out to examples/run_pretrain.sh directly.
    if path.startswith("primus/") or path.startswith("tests/unit_tests/") or path.startswith("examples/"):
        return ("component", None)  # any other source/unit-test/launcher change
    return ("ignore", None)  # docs, README, ... outside the source/test trees


def select_e2e(files, suites=None):
    suites = discover_e2e_suites() if suites is None else set(suites)
    files = [f.strip() for f in files if f.strip()]
    if not files:
        return sorted(suites)
    selected = set()
    for path in files:
        kind, val = classify(path)
        if kind == "global":
            return sorted(suites)
        if kind == "backend":
            if val in suites:
                selected.add(val)  # backend has a trainer -> run its suite
            else:
                return sorted(suites)  # no trainer (bridge/hummingbirdxt/TE) -> all
        elif kind == "component":
            return sorted(suites)  # non-backend source change -> all training
        # ignore -> skip
    return sorted(selected)


def main():
    files = sys.stdin.read().splitlines()
    suites = discover_e2e_suites()
    e2e = select_e2e(files, suites)
    print("all" if suites and set(e2e) == suites else " ".join(e2e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
