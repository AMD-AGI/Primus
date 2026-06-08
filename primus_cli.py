#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Console-script entry point for the ``primus-cli`` command.

This module is intentionally dependency-free (standard library only) and does
**not** import the heavy ``primus`` package. That keeps ``primus-cli`` usable on
hosts that do not have the full training stack installed -- for example running
``primus-cli container ...`` or ``primus-cli slurm ...`` from a bare login node
where ``import primus`` (and its transitive deps) would otherwise fail.

The real implementation is the ``runner/primus-cli`` bash toolkit, which is
shipped as package data under ``primus/runner/`` inside the wheel.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path


def _candidate_runner_clis() -> list[Path]:
    """Return possible locations of the ``runner/primus-cli`` bash launcher."""
    candidates: list[Path] = []

    # 1. Installed wheel layout: <site-packages>/primus/runner/primus-cli
    #    ``find_spec`` locates the package without executing primus/__init__.py.
    try:
        spec = importlib.util.find_spec("primus")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        locations = list(spec.submodule_search_locations or [])
        if spec.origin and spec.origin != "namespace":
            locations.append(str(Path(spec.origin).parent))
        for loc in locations:
            candidates.append(Path(loc) / "runner" / "primus-cli")

    # 2. Source-tree fallback: this file lives next to runner/ at the repo root.
    here = Path(__file__).resolve().parent
    candidates.append(here / "runner" / "primus-cli")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _find_runner_cli() -> Path:
    for candidate in _candidate_runner_clis():
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(c) for c in _candidate_runner_clis())
    raise FileNotFoundError("Could not locate the 'runner/primus-cli' launcher. Searched:\n  " + searched)


def _find_bash() -> str:
    """Locate a usable bash interpreter."""
    override = os.environ.get("PRIMUS_BASH")
    if override:
        resolved = shutil.which(override) or (override if Path(override).is_file() else None)
        if resolved:
            return resolved

    found = shutil.which("bash")
    if found:
        return found

    for path in ("/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash"):
        if Path(path).is_file():
            return path

    raise FileNotFoundError(
        "Could not find 'bash'. primus-cli requires bash to run. "
        "Install bash or set PRIMUS_BASH to its path."
    )


def main() -> "int | None":
    try:
        runner_cli = _find_runner_cli()
        bash = _find_bash()
    except FileNotFoundError as exc:
        print(f"[primus-cli] {exc}", file=sys.stderr)
        raise SystemExit(1)

    argv = [bash, str(runner_cli), *sys.argv[1:]]

    # Replace the current process on POSIX so signals and exit codes propagate.
    if hasattr(os, "execv"):
        try:
            os.execv(bash, argv)
        except OSError:
            pass

    # Fallback (e.g. Windows): spawn a child and forward its exit code.
    import subprocess

    raise SystemExit(subprocess.call(argv))


if __name__ == "__main__":
    main()
