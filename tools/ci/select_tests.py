###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Map a PR's changed files (stdin, one per line) to the unit-test paths to run.

Fail-safe by design: shared / unknown / CI / packaging changes expand to the
full suite, so it can only over-select, never under-select. Otherwise prints the
union of the matched components' test dirs (a changed test file -> its own dir).

    git diff --name-only "$BASE" HEAD | python tools/ci/select_tests.py
"""

import sys

FULL = "tests/unit_tests/"

# Changed paths matching any of these -> run the whole suite.
FULL_TRIGGERS = (
    ".github/",
    "tools/",
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

# Source-prefix -> test paths. Checked in order; first match wins, so more
# specific prefixes (projection, megatron) must precede their generic parents.
COMPONENT_MAP = (
    ("primus/core/projection/", ("tests/unit_tests/core/projection/",)),
    ("primus/backends/megatron/", ("tests/unit_tests/backends/megatron/", "tests/unit_tests/megatron/")),
    ("primus/agents/", ("tests/unit_tests/agents/",)),
    ("primus/cli/", ("tests/unit_tests/cli/",)),
    ("primus_cli.py", ("tests/unit_tests/cli/",)),
    ("runner/", ("tests/unit_tests/cli/",)),
    ("primus/core/", ("tests/unit_tests/core/",)),
    ("primus/backends/", ("tests/unit_tests/backends/",)),
)


def _is_full_trigger(path):
    # Root-level requirements*.txt (dependency change) -> full suite.
    if "/" not in path and path.startswith("requirements") and path.endswith(".txt"):
        return True
    return any(path == trig or path.startswith(trig) for trig in FULL_TRIGGERS)


def select_targets(files):
    files = [f.strip() for f in files if f.strip()]
    if not files:
        return [FULL]

    targets = []

    def add(path):
        if path not in targets:
            targets.append(path)

    for path in files:
        if _is_full_trigger(path):
            return [FULL]
        # A changed test file: run its own directory.
        if path.startswith("tests/unit_tests/") and path.endswith(".py"):
            add(path.rsplit("/", 1)[0] + "/")
            continue
        matched = False
        for prefix, tests in COMPONENT_MAP:
            if path.startswith(prefix):
                for test_path in tests:
                    add(test_path)
                matched = True
                break
        if not matched and path.startswith("primus/"):
            return [FULL]  # unknown source area -> be safe
        # else: non-source path (docs, README, ...) -> ignore

    return targets or [FULL]


def main():
    print(" ".join(select_targets(sys.stdin.read().splitlines())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
