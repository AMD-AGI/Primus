###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Emerging-Optimizers commit is written in three places, which must agree.

The Megatron hook installs ``EO_COMMIT`` for Muon runs, ``primus/_thirdparty.lock``
records the pin and the ``third_party/Emerging-Optimizers`` gitlink is what a
submodule checkout gets.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "runner/helpers/hooks/train/pretrain/megatron/01_install_emerging_optimizers.sh"
SUBMODULE = "third_party/Emerging-Optimizers"


def _hook_default() -> str:
    match = re.search(
        r'^EO_COMMIT="\$\{PRIMUS_EMERGING_OPTIMIZERS_COMMIT:-([0-9a-f]{40})\}"', HOOK.read_text(), re.M
    )
    assert match, f"no EO_COMMIT default in {HOOK.name}"
    return match.group(1)


def _lock_commit() -> str:
    entries = json.loads((REPO_ROOT / "primus/_thirdparty.lock").read_text())["third_party"]
    (entry,) = [e for e in entries if e["path"] == SUBMODULE]
    return entry["commit"]


def test_hook_default_matches_the_lock():
    assert _hook_default() == _lock_commit()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_lock_matches_the_submodule_pointer():
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(REPO_ROOT), "ls-tree", "HEAD", SUBMODULE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split()[2] == _lock_commit()
