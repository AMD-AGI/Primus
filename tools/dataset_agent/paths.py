# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Decide whether a tool call may write a path. The phase machine enforces this."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from tools.dataset_agent.spec import PathPolicy

_WRITE_VERBS = {
    "rm",
    "mv",
    "cp",
    "tee",
    "sed",
    "perl",
    "truncate",
    "install",
    "mkdir",
    "touch",
    "chmod",
    "chown",
    "ln",
    "dd",
}
_GIT_BLOCKED = {"commit", "push", "reset", "clean", "rebase", "merge"}
_REDIRECT = re.compile(r">>?\s*([^\s;&|]+)")


def repo_relative(repo: Path, raw: str) -> str | None:
    """Return a posix path relative to ``repo``, or None when ``raw`` is not a path."""
    text = raw.strip().strip("'\"")
    if not text or text.startswith("-"):
        return None
    candidate = Path(text)
    if not candidate.is_absolute() and "/" not in text and not candidate.suffix:
        return None
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        return candidate.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None


def write_allowed(policy: PathPolicy, relative: str) -> bool:
    if relative in policy.write_files:
        return True
    for root in policy.write_roots:
        prefix = root.strip("/")
        if relative == prefix or relative.startswith(prefix + "/"):
            return True
    return False


def edit_deny_reason(repo: Path, policy: PathPolicy, raw_path: str) -> str:
    relative = repo_relative(repo, raw_path)
    if relative is None:
        return f"Refusing to write outside the Primus repo: {raw_path}"
    if write_allowed(policy, relative):
        return ""
    return f"This phase cannot write `{relative}`."


def bash_deny_reason(repo: Path, policy: PathPolicy, command: str) -> str:
    if not policy.allow_bash:
        return "This phase cannot run shell commands."
    for match in _REDIRECT.finditer(command):
        target = match.group(1)
        if target in {"&1", "&2"}:
            continue
        relative = repo_relative(repo, target)
        if relative is not None and not write_allowed(policy, relative):
            return f"This phase cannot redirect output to `{relative}`."
    for segment in re.split(r"(?:&&|\|\||[;&\n|])", command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            words = shlex.split(segment)
        except ValueError:
            words = segment.split()
        index = 0
        while index < len(words) and "=" in words[index] and not words[index].startswith("="):
            index += 1
        if index >= len(words):
            continue
        verb = os.path.basename(words[index])
        args = words[index + 1 :]
        if verb == "git" and any(arg in _GIT_BLOCKED for arg in args):
            return "Git history changes stay with the operator. This phase cannot commit, push, or reset."
        if verb not in _WRITE_VERBS and not (
            verb in {"sed", "perl"} and any(arg.startswith("-i") for arg in args)
        ):
            continue
        for arg in args:
            if arg.startswith("-"):
                continue
            relative = repo_relative(repo, arg)
            if relative is None:
                bare = repo_relative(repo, str(Path(arg)))
                if bare is None and verb in {"rm", "mv", "cp", "tee"}:
                    # A bare filename written in the repo cwd.
                    relative = arg
                else:
                    relative = bare
            if relative and not write_allowed(policy, relative):
                return f"`{verb}` would modify `{relative}`, which this phase cannot write."
    return ""
