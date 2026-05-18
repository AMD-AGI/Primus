###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Integration tests for tools/preflight_bisect/bisect.py. These require a live
# Slurm allocation and skip automatically when the required environment is
# missing.
#
# Required env vars
# -----------------
# Both tests require a Slurm nodelist. If BISECT_NODELIST is unset, the tests
# use SLURM_NODELIST from the current allocation.
#
# Running from inside a Slurm allocation
# --------------------------
#   cd ~/Primus
#   export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
#   python3 -m pytest tests/unit_tests/tools/test_preflight_bisect_slurm.py -v
# More complex example
# --------------------------
#   export NCCL_SOCKET_IFNAME=tw-eth0
#   export GLOO_SOCKET_IFNAME=tw-eth0
#   export NCCL_IB_HCA="rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1"
#   export BISECT_PREFLIGHT_ENV="USING_AINIC=1 NCCL_IB_GID_INDEX=3 NCCL_CROSS_NIC=0 NCCL_PXN_DISABLE=0"
#
#   python3 -m pytest \
#       tests/unit_tests/tools/test_preflight_bisect_slurm.py \
#       -v \
#       --basetemp=/tmp/preflight-bisect-pytest
#
# Inspect pytest artifacts:
#   ls -R /tmp/preflight-bisect-pytest
#   cat /tmp/preflight-bisect-pytest/test_bisect_all_nodes_pass*/trial-*.log
#   cat /tmp/preflight-bisect-pytest/test_bisect_all_nodes_pass*/summary.txt
#
# Defaults
# --------
# - BISECT_NODELIST defaults to SLURM_NODELIST.
# - BISECT_PARTITION defaults to SLURM_JOB_PARTITION; otherwise --partition is omitted.
# - BISECT_BAD_NODE defaults to the last host in the resolved nodelist.
#
# Optional tuning:
#   BISECT_TRIAL_TIMEOUT, BISECT_SLURM_TIME, BISECT_PREFLIGHT_ENV
# Keep comma-containing values such as NCCL_IB_HCA as normal exported env vars;
# BISECT_PREFLIGHT_ENV is passed through srun --export=ALL,...
#
###############################################################################

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
BISECT_PY = REPO_ROOT / "tools" / "preflight_bisect" / "bisect.py"
FAKE_RUNNER = REPO_ROOT / "tools" / "preflight_bisect" / "fake_runner.sh"
REAL_RUNNER = REPO_ROOT / "runner" / "primus-cli"


def _require_env(*names: str) -> dict[str, str]:
    """Return a dict of the requested env var values, skipping the test if any are missing."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"Required env var(s) not set: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def _resolve_nodelist() -> str:
    nodelist = os.environ.get("BISECT_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        pytest.skip("Set BISECT_NODELIST or run from inside a Slurm allocation with SLURM_NODELIST set")
    return nodelist


def _resolve_partition() -> str:
    return os.environ.get("BISECT_PARTITION") or os.environ.get("SLURM_JOB_PARTITION") or ""


def _resolve_hosts(nodelist: str) -> list[str]:
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("scontrol not found; run this test on a Slurm login/head node")
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"scontrol failed to resolve nodelist {nodelist!r}: {exc.stderr.strip()}")

    hosts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not hosts:
        pytest.skip(f"scontrol produced no hostnames for nodelist {nodelist!r}")
    return hosts


def _resolve_bad_node(nodelist: str) -> str:
    return os.environ.get("BISECT_BAD_NODE") or _resolve_hosts(nodelist)[-1]


def _resolve_bisect_env() -> dict[str, str]:
    env = {"BISECT_NODELIST": _resolve_nodelist()}
    partition = _resolve_partition()
    if partition:
        env["BISECT_PARTITION"] = partition
    return env


def _run_bisect(extra_args: list[str], env: dict[str, str], tmp_path: Path, timeout: int) -> str:
    """Invoke bisect.py as a subprocess and return the contents of summary.txt."""
    preflight_env_args: list[str] = []
    for kv in os.environ.get("BISECT_PREFLIGHT_ENV", "").split():
        preflight_env_args += ["--preflight-env", kv]

    cmd = [
        sys.executable,
        str(BISECT_PY),
        "--nodelist", env["BISECT_NODELIST"],
        "--output-dir", str(tmp_path),
    ]
    if env.get("BISECT_PARTITION"):
        cmd.extend(["--partition", env["BISECT_PARTITION"]])
    cmd.extend([*preflight_env_args, *extra_args])

    result = subprocess.run(
        cmd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    summary_path = tmp_path / "summary.txt"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""

    if result.returncode != 0:
        pytest.fail(
            f"bisect.py exited {result.returncode} and wrote no summary.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    return summary_text


def test_bisect_all_nodes_pass(tmp_path):
    """Run bisect.py with the real preflight runner against a known-healthy nodeset.

    Requires VENV_ACTIVATE and either BISECT_NODELIST or SLURM_NODELIST.
    All nodes are expected to pass the preflight perf-test, so bisect should
    report SUSPECT_NODES: (none).
    """
    env = {**_resolve_bisect_env(), **_require_env("VENV_ACTIVATE")}
    trial_timeout = int(os.environ.get("BISECT_TRIAL_TIMEOUT", "600"))
    slurm_time = os.environ.get("BISECT_SLURM_TIME", "00:15:00")

    # subprocess timeout: give a comfortable margin above the per-trial timeout
    # to account for bisect recursion and Slurm scheduling overhead.
    # Each level of bisection can run up to 2 concurrent trials, and a nodeset
    # of N nodes has at most log2(N)+1 levels, so multiply generously.
    subprocess_timeout = trial_timeout * 8

    summary_text = _run_bisect(
        extra_args=[
            "--trial-timeout-sec", str(trial_timeout),
            "--slurm-time", slurm_time,
            "--runner", str(REAL_RUNNER),
        ],
        env=env,
        tmp_path=tmp_path,
        timeout=subprocess_timeout,
    )

    assert "SUSPECT_NODES: (none)" in summary_text, (
        f"Expected no suspect nodes for a healthy nodeset, but got:\n{summary_text}"
    )


def test_bisect_identifies_bad_node(tmp_path):
    """Run bisect.py with fake_runner.sh seeding one bad node.

    Requires either BISECT_NODELIST or SLURM_NODELIST.
    BISECT_BAD_NODE can override the default bad node, which is the last host in
    the resolved nodelist.
    bisect.py is expected to identify exactly that node as the sole suspect.
    """
    env = _resolve_bisect_env()
    bad_node = _resolve_bad_node(env["BISECT_NODELIST"])
    env["BISECT_BAD_NODE"] = bad_node
    trial_timeout = int(os.environ.get("BISECT_TRIAL_TIMEOUT", "30"))
    slurm_time = os.environ.get("BISECT_SLURM_TIME", "00:02:00")

    subprocess_timeout = trial_timeout * 8

    summary_text = _run_bisect(
        extra_args=[
            "--trial-timeout-sec", str(trial_timeout),
            "--slurm-time", slurm_time,
            "--runner", str(FAKE_RUNNER),
            "--preflight-env", f"BAD_NODE={bad_node}",
            "--max-concurrent-trials", "2",
        ],
        env=env,
        tmp_path=tmp_path,
        timeout=subprocess_timeout,
    )

    assert f"SUSPECT_NODES: {bad_node}" in summary_text, (
        f"Expected '{bad_node}' to be identified as the sole suspect, but got:\n{summary_text}"
    )
