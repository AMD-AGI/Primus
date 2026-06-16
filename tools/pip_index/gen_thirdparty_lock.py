#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Generate ``primus/_thirdparty.lock`` from the tracked submodule pins.

Writes a JSON manifest of every ``third_party`` submodule (name, path, url, and
pinned commit) into the primus package. The installed wheel ships this lock so
``primus-cli deps sync`` can clone the exact sources later via
``git clone --recurse-submodules`` + ``git checkout <commit>`` -- giving full
source trees (Makefiles, .cpp helpers, nested submodules) that ``git+`` wheel
installs cannot provide.

Commits are read from the current checkout's gitlinks (the index, mode 160000),
so each branch/release produces a lock pointing at *its own* pins. Run this in
CI right before ``python -m build``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_submodules(gitmodules: Path) -> list[tuple[str, str]]:
    """Parse .gitmodules into an ordered list of (path, url)."""
    subs: list[tuple[str, str]] = []
    if not gitmodules.exists():
        return subs
    current_path: str | None = None
    for raw in gitmodules.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("path"):
            current_path = line.split("=", 1)[1].strip()
        elif line.startswith("url") and current_path is not None:
            subs.append((current_path, line.split("=", 1)[1].strip()))
            current_path = None
    return subs


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


def build_entries(repo_root: Path) -> list[dict]:
    entries: list[dict] = []
    for path, url in read_submodules(repo_root / ".gitmodules"):
        commit = submodule_commit(repo_root, path)
        if not commit:
            print(f"[lock] skip {path}: no pinned commit found", file=sys.stderr)
            continue
        entries.append({"name": Path(path).name, "path": path, "url": url, "commit": commit})
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate primus/_thirdparty.lock from submodule pins.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root (default: auto)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <repo-root>/primus/_thirdparty.lock)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the lock without writing it")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve() if args.output else repo_root / "primus" / "_thirdparty.lock"

    entries = build_entries(repo_root)
    payload = json.dumps({"third_party": entries}, indent=2) + "\n"

    print("[lock] third_party pins:")
    for entry in entries:
        print(f"  {entry['name']:<22} {entry['commit'][:12]}  {entry['url']}")

    if args.dry_run:
        print("[lock] --dry-run: not writing.\n" + payload)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    # Force LF (newline="\n") so the lock is byte-identical regardless of the OS
    # that runs this generator. The committed lock is LF (see .gitattributes), and
    # CI regenerates it on Linux to diff against the commit; a CRLF write on Windows
    # would otherwise make that drift check fail spuriously.
    output.write_text(payload, encoding="utf-8", newline="\n")
    print(f"[lock] wrote {len(entries)} entrie(s) to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
