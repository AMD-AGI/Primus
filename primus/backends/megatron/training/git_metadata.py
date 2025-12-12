from __future__ import annotations

import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import Dict, MutableMapping, Optional, Union

PathLike = Union[str, Path]
_SECRET_PATTERN = re.compile(r"(TOKEN|SECRET|KEY)", re.IGNORECASE)

# Git submodule status markers (prefix characters in `git submodule status` output):
#   - : submodule not initialized
#   + : currently checked-out commit differs from the one recorded in the superproject
#   U : submodule has merge conflicts
_SUBMODULE_STATUS_MARKERS = "-+U"

# Maximum directory depth to search upward when looking for a .git directory
_MAX_GIT_ROOT_SEARCH_DEPTH = 10


# ---------- env collector ----------


def get_env_variables():
    """
    Filter environment variables for secrets and return as dict.
    """
    env_vars = {}
    for k, v in os.environ.items():
        # Skip anything that looks secret-ish
        if _SECRET_PATTERN.search(k) is None:
            env_vars[k] = v
    return env_vars


def format_env_variables() -> str:
    """
    Format env vars as 'KEY=VALUE' lines.
    """
    return "\n".join(f"{k}={v}" for k, v in sorted(get_env_variables().items()))


# ---------- git helpers ----------


def find_git_repository_root(start: Path, max_depth: int = _MAX_GIT_ROOT_SEARCH_DEPTH) -> Optional[Path]:
    """
    Search upward from `start` to find the root of a git repository.

    Walks up the directory tree until a `.git` directory is found, the filesystem
    root is reached, or the maximum number of parent directories (`max_depth`)
    has been checked.

    Args:
        start (Path): The directory to start searching from.
        max_depth (int): The maximum number of parent directories to check.

    Returns:
        Optional[Path]: The path to the git repository root if found within
        `max_depth` levels, otherwise None.
    """
    current = start
    for _ in range(max_depth):
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    """Run a git command in `cwd` and return stripped stdout or None on error."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    out = out.strip()
    return out or None


def _collect_repo_git_metadata(
    meta: MutableMapping[str, str],
    label: str,
    repo_path: Path,
) -> None:
    """
    Collect git metadata for a single repo into `meta` under the given label.

    Example keys:
      git/primus_commit
      git/primus_branch
      git/primus_remote
      git/primus_dirty
      git/primus/submodule/third_party--Megatron-LM_commit
    """
    commit = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    if not commit:
        return  # not a git repo (or git failed)

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    remote = _run_git(["config", "--get", "remote.origin.url"], cwd=repo_path)
    status = _run_git(["status", "--porcelain"], cwd=repo_path)

    base = f"git/{label}"
    meta[f"{base}_commit"] = commit
    if branch:
        meta[f"{base}_branch"] = branch
    if remote:
        meta[f"{base}_remote"] = remote
    if status is not None:
        meta[f"{base}_dirty"] = "true" if status else "false"

    # Submodules (recursive)
    submods = _run_git(["submodule", "status", "--recursive"], cwd=repo_path)
    if not submods:
        return

    for line in submods.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        raw_commit = parts[0]  # e.g. "-8f3e2bf..."
        path = parts[1]  # e.g. "third_party/Megatron-LM"

        commit_hash = raw_commit[1:] if raw_commit[0] in _SUBMODULE_STATUS_MARKERS else raw_commit
        ref = None
        if len(parts) >= 3 and parts[2].startswith("("):
            ref = parts[2].strip("()")

        key_prefix = f"{base}/submodule/{path.replace('/', '--')}"
        meta[f"{key_prefix}_commit"] = commit_hash
        if ref:
            meta[f"{key_prefix}_ref"] = ref


# ---------- main git collector ----------


def collect_git_metadata(
    primus_root: PathLike | None = None,
    workspace_root: PathLike | None = None,
) -> Dict[str, str]:
    """
    Collect git metadata for:
      - Primus repo (label `primus`)
      - all other git repos directly under workspace_root
        (labelled by directory name, lowercased with '-' -> '_';
        collisions are resolved by appending '_1', '_2', etc.)
      - plus all their submodules (recursively)

    Any directory that is missing or not a git repo is silently skipped.
    """
    # 1) Locate Primus root (starting from this file's path or cwd)
    if primus_root is None:
        primus_root = find_git_repository_root(Path(__file__).parent)
        if primus_root is None:
            warnings.warn(
                f"No git repository found searching upward from {Path(__file__).parent}. "
                "Git metadata collection may be incomplete.",
                stacklevel=2,
            )
            primus_root = Path(__file__).parent
    primus_root = Path(primus_root)

    meta: Dict[str, str] = {}

    # 2) Primus itself
    _collect_repo_git_metadata(meta, "primus", primus_root)

    # 3) Workspace: siblings under /workspace (or parent of Primus root)
    ws_root = Path(workspace_root) if workspace_root is not None else primus_root.parent
    seen_labels: Dict[str, int] = {}
    if ws_root.is_dir():
        for child in ws_root.iterdir():
            if not child.is_dir():
                continue
            if child == primus_root:
                continue
            if not (child / ".git").exists():
                continue

            # Normalize label; detect and resolve collisions by appending a numeric suffix.
            # e.g., 'my-repo', 'My_Repo', 'my_repo' all normalize to 'my_repo',
            # so the first occurrence is 'my_repo', the second becomes 'my_repo_1', the third 'my_repo_2', etc.
            base_label = child.name.lower().replace("-", "_")
            if base_label in seen_labels:
                label = f"{base_label}_{seen_labels[base_label]}"
                seen_labels[base_label] += 1
            else:
                label = base_label
                seen_labels[base_label] = 1
            _collect_repo_git_metadata(meta, label, child)

    return meta
