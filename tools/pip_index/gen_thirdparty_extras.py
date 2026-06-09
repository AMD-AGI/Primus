#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inject third_party submodules as pip extras into pyproject.toml.

For each pip-installable submodule under ``third_party/``, this emits an
optional-dependency (extra) that installs the **exact pinned commit** via a
PEP 508 direct reference (``<dist> @ git+<url>@<commit>``). Run it in CI right
before ``python -m build`` so the wheel metadata advertises
``pip install primus[<extra>]``.

Commits are read dynamically from the current checkout's submodule pins (the
gitlink in the index, no submodule checkout needed), so each branch / release
produces extras that point at *its own* pins.

Notes:
- ``HummingbirdXT`` has no pyproject/setup.py (not a Python distribution), so it
  is intentionally excluded -- a ``git+`` install would fail.
- ``Megatron-Bridge`` vendors ``Megatron-LM`` as a nested submodule; pip does
  not fetch submodules for git references, so its extra also pulls the
  ``megatron`` extra to provide megatron-core.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import tomlkit

REPO_ROOT = Path(__file__).resolve().parents[2]

# submodule path -> (extra name, PEP 508 distribution name)
# Only submodules that expose installable project metadata are included.
EXTRA_MAP: dict[str, tuple[str, str]] = {
    "third_party/Megatron-LM": ("megatron", "megatron-core"),
    "third_party/torchtitan": ("torchtitan", "torchtitan"),
    "third_party/maxtext": ("maxtext", "maxtext"),
    "third_party/Emerging-Optimizers": ("emerging-optimizers", "emerging-optimizers"),
    "third_party/Megatron-Bridge": ("megatron-bridge", "megatron-bridge"),
}

# Extras whose project nests another submodule we already expose. pip does not
# fetch git submodules, so reuse the sibling extra to install the nested dep.
NESTED_REUSE: dict[str, list[str]] = {
    "megatron-bridge": ["megatron"],  # Megatron-Bridge vendors Megatron-LM
}


def read_submodule_urls(gitmodules: Path) -> dict[str, str]:
    """Parse .gitmodules into {path: url}."""
    urls: dict[str, str] = {}
    if not gitmodules.exists():
        return urls
    current_path: str | None = None
    for raw in gitmodules.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("path"):
            current_path = line.split("=", 1)[1].strip()
        elif line.startswith("url") and current_path is not None:
            urls[current_path] = line.split("=", 1)[1].strip()
            current_path = None
    return urls


def submodule_commit(repo_root: Path, path: str) -> str | None:
    """Read a submodule's pinned commit from the index (gitlink, mode 160000)."""
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", path],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("160000"):
            return line.split()[1]
    return None


def build_extras(repo_root: Path) -> dict[str, list[str]]:
    urls = read_submodule_urls(repo_root / ".gitmodules")
    extras: dict[str, list[str]] = {}
    for path, (extra, dist) in EXTRA_MAP.items():
        url = urls.get(path)
        commit = submodule_commit(repo_root, path)
        if not url or not commit:
            print(f"[extras] skip {path}: url={url!r} commit={commit!r}", file=sys.stderr)
            continue
        deps = [f"primus[{reuse}]" for reuse in NESTED_REUSE.get(extra, [])]
        deps.append(f"{dist} @ git+{url}@{commit}")
        extras[extra] = deps
    return extras


def inject(pyproject: Path, extras: dict[str, list[str]]) -> None:
    doc = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    project = doc["project"]
    opt = project.get("optional-dependencies")
    if opt is None:
        opt = tomlkit.table()
        project["optional-dependencies"] = opt
    for extra, deps in extras.items():
        arr = tomlkit.array()
        arr.multiline(True)
        for dep in deps:
            arr.append(dep)
        opt[extra] = arr
    pyproject.write_text(tomlkit.dumps(doc), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject third_party submodules as pip extras.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root (default: auto)")
    parser.add_argument(
        "--pyproject", default=None, help="pyproject.toml path (default: <repo-root>/pyproject.toml)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print extras without writing pyproject.toml")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    pyproject = Path(args.pyproject).resolve() if args.pyproject else repo_root / "pyproject.toml"

    extras = build_extras(repo_root)
    if not extras:
        print("[extras] no installable third_party submodules found; nothing to inject.", file=sys.stderr)
        return 0

    print("[extras] generated third_party extras:")
    for extra, deps in extras.items():
        print(f"  [{extra}]")
        for dep in deps:
            print(f"      {dep}")

    if args.dry_run:
        print("[extras] --dry-run: pyproject.toml not modified.")
        return 0

    inject(pyproject, extras)
    print(f"[extras] injected {len(extras)} extra(s) into {pyproject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
