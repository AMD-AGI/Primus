###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for the node-smoke package.

Each test is a deliberate guard against a behaviour that has either
broken in production (history items in `docs/node-smoke.md`) or is part
of the operator-facing contract (CLI flags, JSON schema, report section
order). They are pure-Python: no GPU, no subprocess fanout, no real
amd-smi/rocm-smi/lsof dependency.

Run directly:

    pytest primus/tools/preflight/node_smoke/tests/test_node_smoke.py -v
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from primus.tools.preflight.node_smoke.aggregator.report import write_smoke_report
from primus.tools.preflight.node_smoke.aggregator.summarizers import (
    _busy_gpu_rows,
    _clock_summary,
    _stack_drift_rows,
)
from primus.tools.preflight.node_smoke.collectors.gpu_processes import _parse_lsof_pcn
from primus.tools.preflight.node_smoke.collectors.rocm_smi import (
    _parse_rocm_smi_ras_info_text,
)
from primus.tools.preflight.node_smoke.logging_utils import _short_name
from primus.tools.preflight.node_smoke.orchestrator import _node_status_from
from primus.tools.preflight.node_smoke.shell_utils import _parse_size_with_unit

# ---------------------------------------------------------------------------
# A. Pure-helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        # Plain ints + decimals (no unit -> bytes)
        ("12345", 12345),
        ("", None),
        # SI / IEC mix, both with and without space
        ("256 MB", 256 * (1 << 20)),
        ("256MB", 256 * (1 << 20)),
        ("12.5 GiB", int(12.5 * (1 << 30))),
        ("12.5GiB", int(12.5 * (1 << 30))),
        # The "unlimited" sentinel
        ("  -1  ", -1),
        # Things that MUST NOT silently become byte-counts. `500 MHz` was
        # the historical bug -- frequency masquerading as bytes.
        ("500 MHz", None),
        ("12 GB extra", None),
        ("not-a-number", None),
    ],
)
def test_parse_size_with_unit(s, expected):
    """A.1 -- regex + unit table; unrecognized units MUST return None."""
    assert _parse_size_with_unit(s) == expected


def test_short_name_strips_fqdn():
    """A.2 -- FQDN normalisation (history item: SLURM-ready txt outputs)."""
    assert _short_name("tus1-p3-g25.cluster.example.com") == "tus1-p3-g25"
    assert _short_name("tus1-p3-g25") == "tus1-p3-g25"
    assert _short_name("") == ""


def test_parse_rocm_smi_ras_sums_per_gpu_and_skips_status_only_rows():
    """A.3 -- ECC text parser: one GPU block per `GPU[N]: RAS INFO` header,
    summing the last two int columns of each block row, ignoring rows that
    have only Status (e.g. ATHUB UNAVAILABLE)."""
    sample = (
        "GPU[0]:         RAS INFO\n"
        "        Block       Status    Correctable Error  Uncorrectable Error\n"
        "          UMC        ENABLED                  3                    1\n"
        "         SDMA        ENABLED                  0                    0\n"
        "       ATHUB     UNAVAILABLE\n"
        "GPU[1]:         RAS INFO\n"
        "          UMC        ENABLED                  0                    0\n"
    )
    out = _parse_rocm_smi_ras_info_text(sample)
    assert out == [
        {"gpu": 0, "ecc_correctable_total": 3, "ecc_uncorrectable_total": 1},
        {"gpu": 1, "ecc_correctable_total": 0, "ecc_uncorrectable_total": 0},
    ]


def test_parse_lsof_pcn_handles_multi_open_pid():
    """A.4 -- lsof -Fpcn field-prefix parser: same PID across multiple open
    files collapses to a single annotated record per PID."""
    text = "p123\ncpython\nf3\nf4\np123\ncpython\np456\nctrain.py\n"
    rows = _parse_lsof_pcn(
        text,
        lambda pid, name, hbm: {"pid": pid, "name": name, "hbm_bytes": hbm},
    )
    assert sorted(r["pid"] for r in rows) == [123, 456]
    assert {r["pid"]: r["name"] for r in rows} == {
        123: "python",
        456: "train.py",
    }


# ---------------------------------------------------------------------------
# B. Aggregator-summarizer regression guards
# ---------------------------------------------------------------------------


def test_stack_drift_does_not_crash_on_heterogeneous_dict_vs_none():
    """B.1 -- guards against the production crash captured in
    docs/node-smoke.md history item 6: a fingerprint key that is None on
    one node and a dict on another USED to crash Counter() with
    `TypeError: unhashable type: 'dict'`. After the fix the key must
    simply be excluded from scalar drift (since it is not a scalar on
    any node)."""
    nodes = [
        {
            "host": "a",
            "tier1": {"fingerprint": {"kernel": "5.15", "rocm": "6.2", "nic_fw": None}},
        },
        {
            "host": "b",
            "tier1": {"fingerprint": {"kernel": "5.15", "rocm": "6.2", "nic_fw": {"rdma0": "20.0"}}},
        },
    ]
    # Must return without raising.
    rows = _stack_drift_rows(nodes)
    # And nic_fw must NOT appear in scalar drift (it's not a scalar
    # on any node).
    assert all(r["key"] != "nic_fw" for r in rows)


def test_clock_summary_spread_and_warn():
    """B.2 -- spread = max - min, warn flag fires above threshold,
    nodes with no active time daemon are listed."""
    nodes = [
        {"host": "a", "tier1": {"clock": {"wall_time_unix": 1000.0, "any_active": True}}},
        {"host": "b", "tier1": {"clock": {"wall_time_unix": 1042.5, "any_active": True}}},
        {"host": "c", "tier1": {"clock": {"wall_time_unix": 1010.0, "any_active": False}}},
    ]
    s = _clock_summary(nodes, skew_warn_sec=30.0)
    assert s["spread_sec"] == 42.5
    assert s["spread_warn"] is True  # 42.5 > 30
    # node_rank defaults to "?" when missing from the input dict.
    assert ("?", "c") in s["no_daemon_hosts"]
    assert s["earliest_host"] == "a"
    assert s["latest_host"] == "b"


def test_busy_gpu_rows_filters_to_is_foreign_only():
    """B.3 -- only processes flagged is_foreign make it into the
    Busy-GPU table; HBM is converted from bytes to GiB."""
    nodes = [
        {
            "host": "a",
            "node_rank": 0,
            "tier1": {
                "gpu_processes": {
                    "ok": True,
                    "per_gpu": [
                        {
                            "gpu": 0,
                            "processes": [
                                {
                                    "pid": 1,
                                    "name": "self",
                                    "hbm_bytes": 1 << 30,
                                    "is_self": True,
                                    "is_allowed": False,
                                    "is_foreign": False,
                                },
                                {
                                    "pid": 2,
                                    "name": "agent",
                                    "hbm_bytes": 0,
                                    "is_self": False,
                                    "is_allowed": True,
                                    "is_foreign": False,
                                },
                                {
                                    "pid": 3,
                                    "name": "leak",
                                    "hbm_bytes": 4 * (1 << 30),
                                    "is_self": False,
                                    "is_allowed": False,
                                    "is_foreign": True,
                                },
                            ],
                        }
                    ],
                }
            },
        }
    ]
    rows = _busy_gpu_rows(nodes)
    assert [r["pid"] for r in rows] == [3]
    assert rows[0]["hbm_gib"] == 4.0
    assert rows[0]["name"] == "leak"


# ---------------------------------------------------------------------------
# C. Orchestrator decision-logic tests
# ---------------------------------------------------------------------------


def test_node_status_empty_means_pass():
    """C.1 -- no per-GPU results, no Tier 1 issues, no Tier 2 issues
    -> empty reasons list -> node PASS."""
    assert _node_status_from([], {}, {}) == []


def test_node_status_nonzero_ecc_fails():
    """C.2 -- any non-zero uncorrectable ECC count is an unconditional
    node FAIL. The amd-smi schema is unstable so we trust only ints."""
    tier1 = {"gpu_low_level": {"per_gpu": [{"gpu": 3, "ecc_uncorrectable_total": 7}]}}
    reasons = _node_status_from([], tier1, {})
    assert any("gpu3" in r and "uncorrectable" in r for r in reasons)


def test_node_status_allow_foreign_procs_downgrades():
    """C.3 -- foreign processes hard-fail the node by default, but
    --allow-foreign-procs downgrades to silent inclusion in the JSON."""
    tier1 = {
        "gpu_processes": {
            "ok": True,
            "foreign_count": 1,
            "per_gpu": [
                {
                    "gpu": 0,
                    "processes": [
                        {
                            "pid": 99,
                            "name": "leak",
                            "hbm_bytes": 1 << 30,
                            "is_foreign": True,
                        }
                    ],
                }
            ],
        }
    }
    # Default (allow_foreign_procs=False) -> FAIL.
    assert _node_status_from([], tier1, {}, allow_foreign_procs=False)
    # Operator opted in -> PASS.
    assert _node_status_from([], tier1, {}, allow_foreign_procs=True) == []


def test_node_status_require_tools_missing_amd_smi_fails():
    """C.4 -- --require-tools promotes a missing CLI tool to a hard
    node FAIL; satisfied requirements remain silent."""
    tier1 = {
        "tooling_inventory": {
            "tools": {
                "amd-smi": {"present": False, "path": None},
                "rocm-smi": {"present": True, "path": "/usr/bin/rocm-smi"},
                "lsof": {"present": True, "path": "/usr/bin/lsof"},
            }
        }
    }
    # amd-smi required but absent -> FAIL.
    reasons = _node_status_from([], tier1, {}, required_tools=["amd-smi", "rocm-smi"])
    assert any("amd-smi" in r for r in reasons)
    # Only rocm-smi required and present -> no reason added.
    assert _node_status_from([], tier1, {}, required_tools=["rocm-smi"]) == []


# ---------------------------------------------------------------------------
# D. CLI / report parity tests
# ---------------------------------------------------------------------------


# Section ORDER + headings are part of the operator-facing contract --
# slack bots and CI scripts grep for these. Update this list ONLY when
# you intentionally change the contract.
EXPECTED_SECTIONS = [
    "## Stack drift across cluster",
    "## NIC firmware drift across cluster",
    "## NIC / RDMA roll-call issues",
    "## NIC port-count summary",
    "## Host limits issues",
    "## GPU visibility issues",
    "## GPU low-level outliers (PCIe link / HBM)",
    "## XGMI link issues",
    "## Cluster clock + time daemons",
    "## Tooling self-latency (`rocm-smi --version`)",
    "## Tooling availability",
    "## Busy GPUs / leaked processes",
    "## GPU pre-touch HBM usage outliers",
    "## GPU compute-activity outliers",
    # Tier 2 perf summary intentionally omitted -- only renders when at
    # least one node ran Tier 2; this fixture has none.
    "## Failing nodes -- full reasons",  # only when failing is non-empty
]


def test_report_section_order_stable(tmp_path):
    """D.1 -- report section order + headings are stable. This catches
    anyone who reorders the _write_<section> calls in
    aggregator.report.write_smoke_report."""
    nodes = [
        {
            "host": "a",
            "node_rank": 0,
            "status": "FAIL",
            "duration_sec": 1.0,
            "fail_reasons": ["xgmi: 1 non-XGMI pair"],
            "tier1": {},
            "tier2": {},
        }
    ]
    out = tmp_path / "report.md"
    write_smoke_report(
        str(out),
        nodes=nodes,
        passing=[],
        failing=nodes,
        expected=1,
        clock_skew_warn_sec=30.0,
        rocm_smi_warn_sec=1.0,
        hbm_busy_threshold_gib=2.0,
        gpu_activity_warn_pct=20.0,
    )
    seen = [line for line in out.read_text().splitlines() if line.startswith("## ")]
    assert seen == EXPECTED_SECTIONS


def test_module_help_exits_zero():
    """D.2 -- `python -m primus.tools.preflight.node_smoke --help` must
    exit 0. Doubles as a smoke test that the package imports cleanly
    under `-m` (i.e. __main__.py + __init__.py + cli.py are all
    consistent)."""
    cp = subprocess.run(
        [sys.executable, "-m", "primus.tools.preflight.node_smoke", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert cp.returncode == 0, cp.stderr
    assert "node-local preflight smoke test" in cp.stdout.lower()
