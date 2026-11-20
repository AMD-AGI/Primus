###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build gsplat the way the WorldMirror hook used to.

gsplat v1.5.3 is cloned into a temporary directory, the ROCm patch is applied
there, GLM is fetched into that copy, and pip installs the copy. The Primus
tree is not modified.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROCM_PATCH = Path(__file__).with_name("gsplat_rocm.patch")
_GSPLAT_URL = "https://github.com/nerfstudio-project/gsplat.git"
_GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"  # v1.5.3
_GLM_URL = "https://github.com/g-truc/glm.git"
_GLM_COMMIT = "6f14f4792a0cde5d0cf2c910506724d61cb95834"


def _drop_failed_gsplat_import() -> None:
    for name in list(sys.modules):
        if name == "gsplat" or name.startswith("gsplat."):
            sys.modules.pop(name, None)


def gsplat_imports() -> bool:
    try:
        import gsplat  # noqa: F401
    except ModuleNotFoundError:
        _drop_failed_gsplat_import()
        return False
    return True


def _apply_rocm_patch(staged: Path) -> None:
    """Apply the ROCm gsplat fixes to the temporary build tree."""
    setup = staged / "setup.py"
    if not setup.is_file() or not (staged / "gsplat" / "cuda" / "include" / "Utils.cuh").is_file():
        return
    if "gsplat_glm" in setup.read_text(encoding="utf-8"):
        return
    subprocess.check_call(
        ["patch", "-p1", "--forward", "--batch", "-i", str(_ROCM_PATCH)],
        cwd=staged,
    )


def _ensure_glm(staged: Path) -> None:
    """Fetch GLM into the build tree. The gsplat checkout does not vendor it."""
    glm_root = staged / "gsplat" / "cuda" / "csrc" / "third_party" / "glm"
    if (glm_root / "glm" / "glm.hpp").is_file():
        return
    if not (staged / "gsplat" / "cuda" / "csrc").is_dir():
        return
    if glm_root.exists():
        shutil.rmtree(glm_root)
    glm_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout", _GLM_URL, str(glm_root)])
    subprocess.check_call(["git", "fetch", "--depth", "1", "origin", _GLM_COMMIT], cwd=glm_root)
    subprocess.check_call(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=glm_root)


def _stage_gsplat_source(source: Path) -> Path:
    """Copy gsplat out of its git checkout before building it.

    Setuptools lists package files with git. A checkout the container user
    does not own fails that listing. A copy under the temp directory is not
    inside the repository, so the build uses the filesystem instead of git.
    """
    destination = Path(tempfile.mkdtemp(prefix="gsplat-build-"))

    def _ignore(directory: str, names: list[str]) -> set[str]:
        skipped = {".git", "build", "hip", "__pycache__"}
        return {name for name in names if name in skipped or name.endswith(".egg-info")}

    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=_ignore)
    return destination


def _fetch_gsplat_source() -> Path:
    """Clone gsplat v1.5.3 and return a copy that is not a git checkout."""
    checkout = Path(tempfile.mkdtemp(prefix="gsplat-src-"))
    subprocess.check_call(
        ["git", "clone", "--filter=blob:none", "--no-checkout", _GSPLAT_URL, str(checkout)]
    )
    subprocess.check_call(["git", "fetch", "--depth", "1", "origin", _GSPLAT_COMMIT], cwd=checkout)
    subprocess.check_call(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
    staged = _stage_gsplat_source(checkout)
    shutil.rmtree(checkout)
    return staged


def _git_config_for_install(env: dict) -> None:
    """Point this install at a throwaway git config that trusts every directory."""
    config = tempfile.NamedTemporaryFile("w", prefix="gsplat-gitconfig-", delete=False)
    config.write("[safe]\n\tdirectory = *\n")
    config.close()
    env["GIT_CONFIG_GLOBAL"] = config.name
    env["GIT_CONFIG_SYSTEM"] = os.devnull


def install_gsplat() -> None:
    """Build gsplat v1.5.3 when it is not already installed."""
    if gsplat_imports():
        return

    from primus.core.utils.module_utils import log_rank_0

    staged = _fetch_gsplat_source()
    log_rank_0("[Primus:Vision] applying ROCm gsplat fixes and fetching GLM")
    _apply_rocm_patch(staged)
    _ensure_glm(staged)
    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "16")
    _git_config_for_install(env)
    log_rank_0(f"[Primus:Vision] building gsplat from {staged}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", str(staged)],
        cwd=staged,
        env=env,
    )
    importlib.invalidate_caches()
    _drop_failed_gsplat_import()
    if not gsplat_imports():
        raise RuntimeError(f"gsplat install finished, but import still fails: {staged}")
