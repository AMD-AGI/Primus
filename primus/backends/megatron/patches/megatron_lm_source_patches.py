###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-LM source-tree patches (applied in-place to ``third_party/Megatron-LM``).

Why not Python monkey-patching?
    Several of these behaviours (notably the extra ``torch.cuda.synchronize()``
    calls inserted throughout ``megatron.training.training.get_model`` and the
    module-top-level ``_GENERAL_GEMM_SUPPORTS_WORKSPACE`` introspection) live
    inside control-flow and module-import side effects that cannot be reached
    by replacing the outer function reference at runtime. Re-implementing
    ``get_model`` just to inject four one-line synchronize calls is both
    brittle and a maintenance hazard across Megatron versions. Applying the
    source patch is 1:1 with the behaviour previously shipped in the
    MLPerf-training Dockerfile and keeps the diff auditable.

When are the patches applied?
    At *module import time* -- specifically when
    ``primus.backends.megatron.patches`` is imported (which happens eagerly
    from ``primus.backends.megatron.megatron_adapter`` before any
    ``megatron.*`` module is imported). This ensures the patches land on
    disk before Python caches any Megatron module.

Idempotency
    Each patch is tried with ``git apply --reverse --check`` first to detect
    the already-applied case. Otherwise ``git apply --check`` is used to
    verify it applies cleanly before actually applying it, so a silent
    corruption of the source tree is not possible.

Opt-out
    Set ``PRIMUS_SKIP_MEGATRON_LM_SOURCE_PATCHES=1`` in the environment.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple

# Use a plain logger here rather than ``primus.modules.module_utils.log_rank_0``
# because this module runs at import time, potentially before torch.distributed
# has been initialised, and ``log_rank_0`` dispatches off dist rank.
import logging

_logger = logging.getLogger("primus.megatron_lm_source_patches")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Patch manifest -- order matters only for hunks that touch the same file.
# Every patch here was previously shipped in
# ``mlperf-training/small_llm_moe_pretraining/primus/patches/`` and applied by
# the MLPerf-training Dockerfile. They are now owned by Primus.
# -----------------------------------------------------------------------------
_PATCH_FILES: Tuple[str, ...] = (
    "megatron_validation_consumed_samples.patch",
    "megatron_cuda_sync_model_init.patch",
    "megatron_te_general_gemm_workspace_compat.patch",
    "megatron_te_bshd_layout.patch",
    "megatron_moe_skip_identity_sort.patch",
)

_PATCH_DIR = Path(__file__).parent / "megatron_lm_source"


def _find_megatron_root() -> Path:
    """Locate the Megatron-LM source tree shipped as a Primus third_party.

    Search order:
      1. ``PRIMUS_MEGATRON_LM_PATH`` env var -- explicit override.
      2. ``<primus-repo>/third_party/Megatron-LM`` -- the standard layout.
    """
    override = os.environ.get("PRIMUS_MEGATRON_LM_PATH")
    if override:
        p = Path(override).resolve()
        if (p / "megatron").is_dir():
            return p
        raise RuntimeError(
            f"PRIMUS_MEGATRON_LM_PATH={override!r} does not look like a "
            "Megatron-LM checkout (missing ``megatron/`` directory)."
        )

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "third_party" / "Megatron-LM"
        if candidate.is_dir() and (candidate / "megatron").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate third_party/Megatron-LM. Set PRIMUS_MEGATRON_LM_PATH "
        "to the Megatron-LM checkout explicitly."
    )


def _git_apply(
    patch_path: Path, repo: Path, *, reverse: bool = False, check: bool = False
) -> Tuple[int, str]:
    cmd: List[str] = ["git", "apply"]
    if reverse:
        cmd.append("--reverse")
    if check:
        cmd.append("--check")
    cmd.append(str(patch_path))
    completed = subprocess.run(
        cmd, cwd=str(repo), capture_output=True, text=True, check=False
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def _apply_one(patch_path: Path, repo: Path) -> str:
    """Apply ``patch_path`` to ``repo``. Returns a short status string."""
    # Already applied? (reverse-check succeeds)
    rc_rev_chk, _ = _git_apply(patch_path, repo, reverse=True, check=True)
    if rc_rev_chk == 0:
        return "already-applied"

    rc_chk, chk_msg = _git_apply(patch_path, repo, check=True)
    if rc_chk != 0:
        raise RuntimeError(
            f"Patch does not apply cleanly to {repo}: {patch_path.name}\n"
            f"git apply --check output:\n{chk_msg}"
        )

    rc, msg = _git_apply(patch_path, repo)
    if rc != 0:
        raise RuntimeError(
            f"git apply failed for {patch_path.name} after --check passed:\n{msg}"
        )
    return "applied"


def apply_megatron_lm_source_patches() -> None:
    """Apply the bundled Megatron-LM source patches. Idempotent."""
    if os.environ.get("PRIMUS_SKIP_MEGATRON_LM_SOURCE_PATCHES", "0") == "1":
        _logger.info("Skipped (PRIMUS_SKIP_MEGATRON_LM_SOURCE_PATCHES=1)")
        return

    try:
        repo = _find_megatron_root()
    except RuntimeError as e:
        _logger.info(f"{e}. Skipping.")
        return

    if not (repo / "megatron" / "training" / "training.py").is_file():
        _logger.info(
            f"{repo} does not look like a valid Megatron-LM source tree. Skipping."
        )
        return

    _logger.info(f"Target: {repo}")

    for name in _PATCH_FILES:
        patch_path = _PATCH_DIR / name
        if not patch_path.is_file():
            raise FileNotFoundError(f"Missing patch file: {patch_path}")
        status = _apply_one(patch_path, repo)
        _logger.info(f"  {status:<16s} {name}")


# ----- Apply at import time ---------------------------------------------------
# This module is imported eagerly by primus.backends.megatron.patches.__init__
# via its ``walk_packages`` auto-import. That chain is triggered by
# ``primus.backends.megatron.megatron_adapter`` before any ``megatron.*``
# module has been touched, so patches land on disk in time to take effect.
apply_megatron_lm_source_patches()
