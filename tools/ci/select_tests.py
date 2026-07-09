###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Map a PR's changed files (stdin, one per line) to the tests to run.

Default prints the minimal unit-test paths; --e2e prints the E2E trainer suites
(or "all"). A single classify() decides each path's blast radius and both
selections build on it. Conventions over hard-coded tables:
  - unit dirs mirror the source tree (primus/<x> -> tests/unit_tests/<x>),
    resolved by walking up to the nearest existing dir;
  - E2E suites are auto-discovered from tests/trainer/test_<name>_trainer.py;
  - a backend is named by its dir (primus/backends/<X> or examples/<X>).
Fail-safe is the only invariant: anything global, unlocatable, or a backend
without a trainer expands to everything -- over-select, never under-select.

    git diff --name-only "$BASE" HEAD | python tools/ci/select_tests.py [--e2e]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FULL = "tests/unit_tests/"

# The only hard-coded list: changes whose blast radius is the whole repo. Being
# absent here only ever falls back to other fail-safe paths, never to "skip".
GLOBAL_TRIGGERS = (
    ".github/",
    "tools/",
    "runner/",  # the launcher drives all training
    "pyproject.toml",
    "primus/__init__.py",
    "primus/core/launcher/",
    "primus/core/utils/",
    "primus/core/config/",
    "tests/utils.py",
    "tests/conftest.py",
    "tests/unit_tests/conftest.py",
    "tests/run_unit_tests.py",
)

# megatron's GPU-operator tests aren't path-isomorphic to the backend source.
_BACKEND_EXTRA_UNIT = {"megatron": ("tests/unit_tests/megatron/",)}

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


def _nearest_unit_dir(rel):
    # rel is source-relative (under primus/ or tests/unit_tests/); walk up to the
    # nearest existing tests/unit_tests/<...> dir, or None if none exists.
    parts = rel.split("/")[:-1]
    while parts:
        cand = "tests/unit_tests/" + "/".join(parts) + "/"
        if (ROOT / cand).is_dir():
            return cand
        parts.pop()
    return None


def classify(path):
    """('global', None) | ('backend', name) | ('component', unit_dir|None) | ('ignore', None)."""
    if _is_global(path):
        return ("global", None)
    backend = _BACKEND_RE.match(path)  # primus/backends/<X>/ or examples/<X>/
    if backend:
        return ("backend", backend.group(1))
    if path.startswith("tests/trainer/"):
        m = _TRAINER_RE.search(path)
        return ("backend", m.group(1)) if m else ("ignore", None)
    for root in ("primus/", "tests/unit_tests/"):
        if path.startswith(root):
            if not path.endswith(".py"):
                return ("global", None)  # non-.py here (configs, fixtures) -> fail-safe
            return ("component", _nearest_unit_dir(path[len(root) :]))
    return ("ignore", None)  # docs, README, ... outside the source/test trees


def select_targets(files):
    files = [f.strip() for f in files if f.strip()]
    if not files:
        return [FULL]
    targets = []

    def add(d):
        if d and d not in targets:
            targets.append(d)

    for path in files:
        kind, val = classify(path)
        if kind == "global":
            return [FULL]
        if kind == "backend":
            base = f"tests/unit_tests/backends/{val}/"
            if not (ROOT / base).is_dir():
                return [FULL]  # backend without a unit dir (e.g. transformer_engine) -> safe
            add(base)
            for extra in _BACKEND_EXTRA_UNIT.get(val, ()):
                add(extra)
        elif kind == "component":
            if val is None:
                return [FULL]  # couldn't localize a unit dir -> safe
            add(val)
        # ignore -> skip
    return targets or [FULL]


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
    if "--e2e" in sys.argv[1:]:
        suites = discover_e2e_suites()
        e2e = select_e2e(files, suites)
        print("all" if suites and set(e2e) == suites else " ".join(e2e))
    else:
        print(" ".join(select_targets(files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
