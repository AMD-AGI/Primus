###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREPARE_EXPERIMENT = ROOT / "runner" / "helpers" / "hooks" / "train" / "pretrain" / "prepare_experiment.sh"
# A backend without shell hooks, so only its prepare.py is dispatched.
BACKEND = "specforge"


def dispatched_data_path(tmp_path, data_path_env=None, *extra_args):
    """Run the dispatcher from tmp_path and return the --data_path it hands
    to the backend's prepare.py."""
    backend_dir = PREPARE_EXPERIMENT.parent / BACKEND
    assert not list(backend_dir.glob("*.sh")), f"{backend_dir} now has shell hooks; pick another backend"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    recorded = tmp_path / "prepare_args"
    # Inline `python3 -c` snippets (framework detection) still run for real;
    # only the backend prepare.py call is recorded.
    (fake_bin / "python3").write_text(
        "#!/bin/bash\n"
        f'if [[ "$1" == "-c" ]]; then exec "{sys.executable}" "$@"; fi\n'
        f"printf '%s\\n' \"$@\" > '{recorded}'\n",
        encoding="utf-8",
    )
    (fake_bin / "python3").chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.pop("DATA_PATH", None)
    if data_path_env is not None:
        env["DATA_PATH"] = data_path_env
    result = subprocess.run(
        [
            "/bin/bash",
            str(PREPARE_EXPERIMENT),
            "train",
            "pretrain",
            "--config",
            str(tmp_path / "exp.yaml"),
            "--backend",
            BACKEND,
            "--patch_args",
            str(tmp_path / "patch_args.txt"),
            *extra_args,
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert recorded.exists(), f"{BACKEND}/prepare.py was not dispatched:\n{result.stdout}\n{result.stderr}"
    args = recorded.read_text(encoding="utf-8").splitlines()
    assert args[0].endswith(f"{BACKEND}/prepare.py"), args
    return args[args.index("--data_path") + 1]


@pytest.mark.parametrize(
    "data_path_env, extra_args, expected",
    [
        (None, (), "./data/"),
        ("/shared/data", (), "/shared/data"),
        ("rel/data", (), "{tmp_path}/rel/data"),
        ("/shared/data", ("--data_path", "/cli/data"), "/cli/data"),
    ],
)
def test_prepare_experiment_data_path_precedence(tmp_path, data_path_env, extra_args, expected):
    expected = expected.format(tmp_path=tmp_path)
    assert dispatched_data_path(tmp_path, data_path_env, *extra_args) == expected
