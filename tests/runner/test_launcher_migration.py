###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_ODC = ROOT / "primus" / "core" / "odc" / "rocshmem_runtime" / "scripts" / "run_odc.sh"
SLURM_CLI = ROOT / "runner" / "primus-cli-slurm.sh"


@pytest.mark.parametrize("child_status", [0, 7])
def test_run_odc_propagates_primus_cli_status_and_keeps_log(tmp_path, child_status):
    fake_root = tmp_path / "primus"
    fake_runner = fake_root / "runner"
    fake_runner.mkdir(parents=True)
    fake_cli = fake_runner / "primus-cli"
    fake_cli.write_text(
        """#!/bin/bash
set -u
log_file=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--log_file" ]]; then
    log_file="$2"
    shift 2
  else
    shift
  fi
done
printf 'fake primus-cli status=%s\\n' "$FAKE_PRIMUS_CLI_STATUS" > "$log_file"
exit "$FAKE_PRIMUS_CLI_STATUS"
""",
        encoding="utf-8",
    )

    log_dir = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_PRIMUS_CLI_STATUS": str(child_status),
            "HOME": str(tmp_path),
            "PRIMUS_PACK_CACHE_DIR": str(tmp_path / "pack-cache"),
            "PRIMUS_ROOT": str(fake_root),
            "TRAIN_LOG_DIR": str(log_dir),
            "TRAIN_LOG_TS": "unit-test",
            "TRITON_CACHE_DIR": str(tmp_path / "triton-cache"),
        }
    )
    result = subprocess.run(
        [
            "/bin/bash",
            str(RUN_ODC),
            "rocshmem",
            "nopad",
            "exp.yaml",
            "status-test",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    expected_log = log_dir / "runlog_status-test_unit-test.log"
    assert result.returncode == child_status
    assert expected_log.read_text(encoding="utf-8") == f"fake primus-cli status={child_status}\n"
    if child_status == 0:
        assert f"[run_odc] DONE exit=0 log={expected_log}" in result.stdout
    else:
        assert f"[run_odc] FAILED exit={child_status} log={expected_log}" in result.stdout


def _slurm_command(tmp_path: Path, node_args: list[str]) -> list[str]:
    config = tmp_path / "primus.yaml"
    config.write_text(
        """slurm:
  nodes: 1
  time: "00:10:00"
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    result = subprocess.run(
        [
            "/bin/bash",
            str(SLURM_CLI),
            "--config",
            str(config),
            "--dry-run",
            "srun",
            *node_args,
            "--",
            "direct",
            "--",
            "train",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    marker = "[DRY RUN] Would execute: "
    command_line = next(line.split(marker, 1)[1] for line in result.stdout.splitlines() if marker in line)
    return shlex.split(command_line)


def _node_values(command: list[str]) -> list[str]:
    values = []
    for index, token in enumerate(command):
        if token in ("-N", "--nodes"):
            values.append(command[index + 1])
        elif token.startswith("-N") and token != "-N":
            values.append(token[2:])
        elif token.startswith("--nodes="):
            values.append(token.split("=", 1)[1])
    return values


@pytest.mark.parametrize(
    "node_args",
    [
        ["-N", "2"],
        ["-N2"],
        ["--nodes", "2"],
        ["--nodes=2"],
    ],
)
def test_slurm_node_cli_forms_override_default_once(tmp_path, node_args):
    assert _node_values(_slurm_command(tmp_path, node_args)) == ["2"]


def test_slurm_nodes_default_is_kept_when_cli_omits_nodes(tmp_path):
    assert _node_values(_slurm_command(tmp_path, [])) == ["1"]
