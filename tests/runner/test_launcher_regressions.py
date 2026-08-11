###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SLURM_ENTRY = ROOT / "runner" / "primus-cli-slurm-entry.sh"
SHARED_LAUNCHER = ROOT / "runner" / "helpers" / "launch" / "slurm_pretrain.sh"


def run_slurm_entry(extra_env):
    env = os.environ.copy()
    for key in (
        "GPUS_PER_NODE",
        "SLURM_GPUS_ON_NODE",
    ):
        env.pop(key, None)
    env.update(
        {
            "PRIMUS_RUNNER_DIR": str(ROOT / "runner"),
            "SLURM_JOB_ID": "123",
            "SLURM_NNODES": "1",
            "SLURM_NODEID": "0",
            "SLURM_NODELIST": "testnode",
            **extra_env,
        }
    )
    return subprocess.run(
        ["/bin/bash", str(SLURM_ENTRY), "--dry-run", "--", "direct", "--", "train"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


@pytest.mark.parametrize(
    ("extra_env", "expected"),
    [
        ({"SLURM_GPUS_ON_NODE": "6"}, "GPUS_PER_NODE=6"),
        ({"SLURM_GPUS_ON_NODE": "6", "GPUS_PER_NODE": "3"}, "GPUS_PER_NODE=3"),
        ({}, "GPUS_PER_NODE=8"),
    ],
)
def test_slurm_gpu_count_precedence(extra_env, expected):
    output = run_slurm_entry(extra_env)
    assert expected in output


def test_shared_launcher_forwards_uep_and_maps_clean(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text('#!/bin/bash\nprintf "%s\\n" "$*"\n', encoding="utf-8")
    fake_bash.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CLEAN_DOCKER_CONTAINER": "1",
            "DATA_PATH": str(tmp_path / "data"),
            "EXP": str(tmp_path / "exp.yaml"),
            "LOG_DIR": str(tmp_path / "logs"),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REBUILD_UEP": "1",
            "USING_UEP": "1",
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(SHARED_LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    command = result.stdout
    assert "container --clean --image" in command
    assert "--env USING_UEP --env REBUILD_UEP" in command
    assert "-- --env DATA_PATH" not in command
    assert "--env TOKENIZED_TRAIN_DATA_PATH --env TOKENIZED_EVAL_DATA_PATH -- train pretrain" in command
