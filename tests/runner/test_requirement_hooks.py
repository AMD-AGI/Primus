###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEGATRON_REQUIREMENTS_HOOK = (
    ROOT / "runner" / "helpers" / "hooks" / "train" / "pretrain" / "megatron" / "00_install_requirements.sh"
)


def test_primus_skip_pip_makes_megatron_hook_clean_noop(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pip_marker = tmp_path / "pip-was-called"
    fake_pip = fake_bin / "pip"
    fake_pip.write_text(
        f"#!/bin/bash\ntouch '{pip_marker}'\nexit 99\n",
        encoding="utf-8",
    )
    fake_pip.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PRIMUS_SKIP_PIP": "1",
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(MEGATRON_REQUIREMENTS_HOOK), "--data_path", str(tmp_path / "data")],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "PRIMUS_SKIP_PIP=1" in result.stdout
    assert not pip_marker.exists()
    assert not (tmp_path / "data").exists()
