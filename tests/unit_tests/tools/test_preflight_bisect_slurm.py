###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Integration tests for tools/preflight_bisect/bisect.py that require a live
# Slurm cluster. These tests are opt-in: they are skipped automatically unless
# the required environment variables are set.
#
# Required env vars
# -----------------
# Both tests:
#   BISECT_NODELIST   Slurm nodelist expression, e.g. "chi2867,chi2879"
#   BISECT_PARTITION  Slurm partition, e.g. "mi355x"
#
# Test 1 (test_bisect_all_nodes_pass) additionally requires:
#   VENV_ACTIVATE     Path to the venv activate script used by run_preflight_direct.sh
#
# Test 2 (test_bisect_identifies_bad_node) additionally requires:
#   BISECT_BAD_NODE   Hostname of the node that should appear faulty, e.g. "chi2879"
#
# Optional tuning vars
# --------------------
#   BISECT_TRIAL_TIMEOUT   Timeout per trial in seconds
#                          (default: 600 for test 1, 30 for test 2)
#   BISECT_SLURM_TIME      srun -t limit per trial
#                          (default: 00:15:00 for test 1, 00:02:00 for test 2)
#   BISECT_PREFLIGHT_ENV   Space-separated KEY=VALUE pairs forwarded to bisect.py
#                          as repeated --preflight-env args, e.g.
#                          "NCCL_DEBUG=INFO NCCL_IB_GID_INDEX=3"
#
# Example usage
# -------------
# Test 2 only (fast, no venv needed):
#   BISECT_NODELIST="chi2867,chi2879" \
#   BISECT_PARTITION="mi355x" \
#   BISECT_BAD_NODE="chi2879" \
#   pytest tests/unit_tests/tools/test_preflight_bisect_slurm.py::test_bisect_identifies_bad_node -v
#
# Both tests:
#   BISECT_NODELIST="chi2867,chi2879" \
#   BISECT_PARTITION="mi355x" \
#   BISECT_BAD_NODE="chi2879" \
#   VENV_ACTIVATE="/path/to/venv/bin/activate" \
#   pytest tests/unit_tests/tools/test_preflight_bisect_slurm.py -v
###############################################################################

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
BISECT_PY = REPO_ROOT / "tools" / "preflight_bisect" / "bisect.py"
FAKE_RUNNER = REPO_ROOT / "tools" / "preflight_bisect" / "fake_runner.sh"
REAL_RUNNER = REPO_ROOT / "runner" / "run_preflight_direct.sh"


def _require_env(*names: str) -> dict[str, str]:
    """Return a dict of the requested env var values, skipping the test if any are missing."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"Required env var(s) not set: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def _run_bisect(extra_args: list[str], env: dict[str, str], tmp_path: Path, timeout: int) -> str:
    """Invoke bisect.py as a subprocess and return the contents of summary.txt."""
    preflight_env_args: list[str] = []
    for kv in os.environ.get("BISECT_PREFLIGHT_ENV", "").split():
        preflight_env_args += ["--preflight-env", kv]

    cmd = [
        sys.executable,
        str(BISECT_PY),
        "--nodelist", env["BISECT_NODELIST"],
        "--partition", env["BISECT_PARTITION"],
        "--output-dir", str(tmp_path),
        *preflight_env_args,
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    summary_path = tmp_path / "summary.txt"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""

    if result.returncode != 0 and not summary_text:
        pytest.fail(
            f"bisect.py exited {result.returncode} and wrote no summary.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    return summary_text


def test_bisect_all_nodes_pass(tmp_path):
    """Run bisect.py with the real preflight runner against a known-healthy nodeset.

    Requires BISECT_NODELIST, BISECT_PARTITION, and VENV_ACTIVATE to be set.
    All nodes are expected to pass the preflight perf-test, so bisect should
    report SUSPECT_NODES: (none).
    """
    env = _require_env("BISECT_NODELIST", "BISECT_PARTITION", "VENV_ACTIVATE")
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

    Requires BISECT_NODELIST, BISECT_PARTITION, and BISECT_BAD_NODE to be set.
    BISECT_BAD_NODE must be one of the nodes in BISECT_NODELIST.
    bisect.py is expected to identify exactly that node as the sole suspect.
    """
    env = _require_env("BISECT_NODELIST", "BISECT_PARTITION", "BISECT_BAD_NODE")
    bad_node = env["BISECT_BAD_NODE"]
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
