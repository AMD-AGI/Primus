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
    _pretouch_hbm_rows,
    _stack_drift_rows,
)
from primus.tools.preflight.node_smoke.collectors.gpu_processes import (
    _flatten_amd_smi_process_json,
    _parse_lsof_pcn,
)
from primus.tools.preflight.node_smoke.collectors.nics import (
    _parse_nic_selector,
    _resolve_selector,
    _selector_matches,
)
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
    "## NIC excluded ports (informational)",
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


# ---------------------------------------------------------------------------
# E. amd-smi schema-drift detection + HBM threshold boundary guards
# ---------------------------------------------------------------------------
#
# These tests lock in two operator-facing contracts:
#
#   1. _flatten_amd_smi_process_json must return (parsed, drift) where
#      `drift` is True only when at least one record looked like a process
#      dict structurally but had no usable pid/process_id. This lets the
#      caller fall through to text/rocm-smi/lsof on a future amd-smi
#      schema rename instead of silently reporting the node clean.
#
#   2. The aggregator's pre-touch HBM threshold uses `>=` (inclusive
#      boundary). A GPU sitting at exactly the threshold MUST appear in
#      the rendered `smoke_report.md` outliers section; the operator-
#      facing help / docs / report wording is aligned to "at least".
#      (We test the aggregator side because that's the artifact
#      operators read; the runner-side `>=` in per_gpu.py is the
#      one-line counterpart, kept honest by argparse help + code review.)


def _annotate_passthrough(pid, name, hbm):
    """Minimal annotate() shim for the schema-drift tests."""
    return {"pid": pid, "name": name, "hbm_bytes": hbm}


def test_flatten_amd_smi_top_level_unknown_returns_empty_no_drift():
    """E.1 -- a doc that is neither Shape A/A'/B (no `process_list`, no
    pid-bearing items) returns ([], False). False on `drift` is what
    distinguishes "schema we do not speak yet at the top level" from
    "schema we do speak but per-process fields drifted"; the caller
    falls through in either case but the message differs."""
    parsed, drift = _flatten_amd_smi_process_json({"unexpected_key": 1}, _annotate_passthrough)
    assert parsed == []
    assert drift is False
    parsed, drift = _flatten_amd_smi_process_json([{"some_other_field": 42}], _annotate_passthrough)
    assert parsed == []
    assert drift is False


def test_flatten_amd_smi_shape_a_empty_process_list_no_drift():
    """E.2 -- Shape A with empty process_list is the clean-node case.
    Buckets are pre-registered (so the caller can tell schema-matched
    from schema-not-matched), processes lists are empty, drift is False
    -- the caller trusts the result and skips the fallback chain."""
    doc = [
        {"gpu": 0, "process_list": []},
        {"gpu": 1, "process_list": []},
    ]
    parsed, drift = _flatten_amd_smi_process_json(doc, _annotate_passthrough)
    assert parsed == [
        {"gpu": 0, "processes": []},
        {"gpu": 1, "processes": []},
    ]
    assert drift is False


def test_flatten_amd_smi_shape_a_renamed_pid_field_raises_drift():
    """E.3 -- the future-schema-drift smoking gun: a Shape A entry whose
    process_list items are dicts (so the structural shape matches) but
    use a renamed key (`proc_pid`) instead of `pid` / `process_id`.
    The bucket is registered, processes is empty, but drift=True so the
    caller falls through to amd-smi text / rocm-smi / lsof rather than
    silently reporting the GPU clean."""
    doc = [
        {
            "gpu": 0,
            "process_list": [
                {"proc_pid": 1234, "name": "python"},
            ],
        }
    ]
    parsed, drift = _flatten_amd_smi_process_json(doc, _annotate_passthrough)
    assert parsed == [{"gpu": 0, "processes": []}]
    assert drift is True


def test_pretouch_hbm_rows_includes_gpu_at_exact_threshold():
    """E.4 -- the aggregator's `_pretouch_hbm_rows` uses `>= threshold`,
    which is what populates the `GPU pre-touch HBM usage outliers`
    section in `smoke_report.md`. Lock in the inclusive boundary at the
    exact value so the wording fix in `aggregator/report.py` and the
    runtime stay aligned: a GPU sitting at threshold MUST be listed,
    a GPU below MUST NOT, a GPU above MUST."""
    nodes = [
        {
            "host": "host-a",
            "node_rank": 0,
            "tier1": {
                "per_gpu": [
                    {"gpu": 0, "details": {"hbm_pre_touch_used_gib": 2.0}},  # ==
                    {"gpu": 1, "details": {"hbm_pre_touch_used_gib": 1.99}},  # <
                    {"gpu": 2, "details": {"hbm_pre_touch_used_gib": 2.01}},  # >
                ],
            },
        },
    ]
    rows = _pretouch_hbm_rows(nodes, threshold_gib=2.0)
    assert sorted(r["gpu"] for r in rows) == [0, 2]


# ---------------------------------------------------------------------------
# F. NIC training-NIC selector (NCCL_IB_HCA-style allowlist + heuristic
#    fallback + precedence chain)
#
# These guard the operator-facing contract introduced when multi-role
# clusters (front-end / storage RoCE NICs co-resident with the training
# NICs) started showing up. The fail mode they protect against: 56 of 57
# real production nodes were reported as FAIL because two unplugged
# front-end ports showed `state=DOWN phys_state=Disabled`, even though
# `NCCL_IB_HCA` explicitly listed only the 8 healthy back-end NICs.
# ---------------------------------------------------------------------------


def _install_synthetic_ib_tree(monkeypatch, base):
    """Redirect `_collect_nic_status`'s sysfs reads at a tmp tree.

    The collector hard-codes ``/sys/class/infiniband`` so we monkey-patch
    its ``os.path.isdir`` / ``os.listdir`` / ``_read_text`` symbols to
    rewrite that prefix to ``base``. We DELIBERATELY rewrite at the
    string level (not via a chroot-style abstraction) because the
    rewrite has to also fire for the recursive sub-paths the collector
    constructs (``<base>/<dev>/ports/<port>/state`` etc.) -- the rewrite
    must therefore intercept every path-shaped call inside the
    collector, not just the entry-point listdir.
    """
    import os

    real_listdir = os.listdir
    real_isdir = os.path.isdir

    def _isdir(p: str) -> bool:
        if p == "/sys/class/infiniband":
            return True
        return real_isdir(p.replace("/sys/class/infiniband", str(base)))

    def _listdir(p: str):
        return real_listdir(p.replace("/sys/class/infiniband", str(base)))

    def _read_text(path: str) -> str:
        try:
            with open(path.replace("/sys/class/infiniband", str(base)), "r") as fh:
                return fh.read().strip()
        except Exception:
            return ""

    monkeypatch.setattr(
        "primus.tools.preflight.node_smoke.collectors.nics.os.path.isdir", _isdir
    )
    monkeypatch.setattr(
        "primus.tools.preflight.node_smoke.collectors.nics.os.listdir", _listdir
    )
    monkeypatch.setattr(
        "primus.tools.preflight.node_smoke.collectors.nics._read_text", _read_text
    )


def test_parse_nic_selector_allowlist_with_ports():
    """F.1 -- baseline NCCL_IB_HCA syntax: comma-separated `device:port`."""
    sel = _parse_nic_selector("rocep158s0:1,rocep190s0:1")
    assert sel["mode"] == "allowlist"
    assert sel["entries"] == [
        ("rocep158s0", False, 1),
        ("rocep190s0", False, 1),
    ]


def test_parse_nic_selector_denylist_prefix():
    """F.2 -- a leading `^` flips the whole selector to denylist."""
    sel = _parse_nic_selector("^roceo12399,roceo12409")
    assert sel["mode"] == "denylist"
    # The ^ is stripped from the entries; only the global mode changes.
    assert [e[0] for e in sel["entries"]] == ["roceo12399", "roceo12409"]


def test_parse_nic_selector_exact_prefix():
    """F.3 -- `=name` forces exact (not prefix) device-name matching."""
    sel = _parse_nic_selector("=mlx5,mlx5_other")
    assert sel["entries"][0] == ("mlx5", True, None)  # exact-match flag set
    assert sel["entries"][1] == ("mlx5_other", False, None)


def test_parse_nic_selector_no_port_matches_any_port():
    """F.4 -- entries without `:port` accept any port on the device."""
    sel = _parse_nic_selector("mlx5_0")
    assert sel["entries"] == [("mlx5_0", False, None)]
    assert _selector_matches(sel, "mlx5_0", 1) is True
    assert _selector_matches(sel, "mlx5_0", 2) is True


def test_parse_nic_selector_empty_is_passthrough():
    """F.5 -- empty / whitespace-only input means 'no selector', so the
    caller falls through to the next precedence layer (env / heuristic)
    rather than treating it as 'allowlist nothing' (which would silently
    exclude every port)."""
    assert _parse_nic_selector("")["mode"] == "passthrough"
    assert _parse_nic_selector("   ")["mode"] == "passthrough"
    assert _parse_nic_selector("^")["mode"] == "passthrough"  # ^ but no entries
    assert _parse_nic_selector(",, ,")["mode"] == "passthrough"


def test_selector_matches_prefix_matches_mlx5_to_mlx5_0():
    """F.6 -- NCCL prefix semantics: `mlx5` allowlist entry matches every
    device that starts with `mlx5` (the historical NCCL behavior). This
    is why operators must use `=mlx5` if they want exact-only matching."""
    sel = _parse_nic_selector("mlx5")
    assert _selector_matches(sel, "mlx5", 1) is True
    assert _selector_matches(sel, "mlx5_0", 1) is True
    assert _selector_matches(sel, "mlx5_bond_0", 1) is True
    assert _selector_matches(sel, "rocep158s0", 1) is False


def test_selector_matches_denylist_inverts():
    """F.7 -- denylist semantics: ports NOT in the list pass; ports in
    the list are rejected. This is the more ergonomic form for clusters
    with many training NICs and a small handful of front-end ports."""
    sel = _parse_nic_selector("^roceo12399,roceo12409")
    # Listed -> excluded.
    assert _selector_matches(sel, "roceo12399", 1) is False
    assert _selector_matches(sel, "roceo12409", 1) is False
    # Not listed -> included.
    assert _selector_matches(sel, "rocep158s0", 1) is True


def test_selector_matches_port_filter_is_per_port():
    """F.8 -- `:port` suffix narrows the match to that specific port on
    the device; the same device on a different port is NOT matched."""
    sel = _parse_nic_selector("mlx5_0:2")
    assert _selector_matches(sel, "mlx5_0", 2) is True
    assert _selector_matches(sel, "mlx5_0", 1) is False


def test_resolve_selector_cli_beats_env():
    """F.9 -- precedence: when both the CLI flag and NCCL_IB_HCA env are
    set, the CLI wins. Operators must be able to override their shell
    env without touching it."""
    sel = _resolve_selector(
        allowlist_arg="mlx5_0:1",
        env={"NCCL_IB_HCA": "rocep158s0:1"},
    )
    assert sel["source"] == "cli"
    assert sel["entries"][0][0] == "mlx5_0"


def test_resolve_selector_env_used_when_cli_absent():
    """F.10 -- precedence: env is consulted only when the CLI flag is
    unset. This is the most common case in production (operator sets
    NCCL_IB_HCA once in their job script; smoke picks it up
    transparently)."""
    sel = _resolve_selector(
        allowlist_arg=None,
        env={"NCCL_IB_HCA": "rocep158s0:1"},
    )
    assert sel["source"] == "env"


def test_resolve_selector_heuristic_when_both_absent():
    """F.11 -- precedence fallback: when neither CLI nor env is set,
    return a `heuristic` marker. The caller (`_collect_nic_status`)
    handles this by auto-excluding ports whose `phys_state` is in
    {Disabled, Sleep}."""
    sel = _resolve_selector(allowlist_arg=None, env={})
    assert sel["source"] == "heuristic"
    # Document the heuristic's exact admin-down set so a future widening
    # (e.g. to also exclude `Polling`) has to update this test
    # deliberately. `Polling` MUST NOT be in here -- it means
    # "actively looking for a link partner" which is a real failure on
    # a port intended to be used.
    assert sel["admin_down_phys_states"] == ["disabled", "sleep"]


def test_resolve_selector_empty_string_falls_through_to_env():
    """F.12 -- an empty `--rdma-nic-allowlist ''` must not silently
    block out everything; it should behave the same as not passing the
    flag at all and fall through to the env. Defensive against
    shell-quoting accidents (`--rdma-nic-allowlist "$VAR"` when
    `$VAR` is unset)."""
    sel = _resolve_selector(
        allowlist_arg="",
        env={"NCCL_IB_HCA": "rocep158s0:1"},
    )
    assert sel["source"] == "env"


def test_resolve_selector_blank_env_falls_through_to_heuristic():
    """F.13 -- same defensive behavior for an empty NCCL_IB_HCA env
    (some clusters export it unset / empty by accident). MUST fall
    through to the heuristic rather than allowlist-nothing."""
    sel = _resolve_selector(allowlist_arg=None, env={"NCCL_IB_HCA": ""})
    assert sel["source"] == "heuristic"


def test_collect_nic_status_env_allowlist_excludes_disabled_frontend_ports(
    tmp_path, monkeypatch
):
    """F.14 -- end-to-end against a synthetic /sys/class/infiniband
    mirroring the production failure mode: 12 IB devices (8 training +
    2 storage ACTIVE + 2 frontend Disabled). With NCCL_IB_HCA listing
    only the 8 training NICs, the collector must:
      * include exactly the 8 listed devices,
      * exclude the other 4 with a source=`env` info_issue each,
      * emit ZERO hard issues,
      * tag the Disabled ports with their state in the info_issue
        (so an operator who reads excluded_ports can still see they
        were unhealthy, just not relevant).
    """
    from primus.tools.preflight.node_smoke.collectors.nics import _collect_nic_status

    # Synthetic sysfs tree.
    base = tmp_path / "ib"
    devices = [
        # (name, state, phys_state, rate_str)
        ("rocep158s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep190s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep206s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep222s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep28s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep62s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep79s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep96s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        # Storage / control-plane: ACTIVE but NOT in NCCL_IB_HCA.
        ("rocep159s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("rocep29s0", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        # Front-end: Disabled, NOT in NCCL_IB_HCA -- the user's bug.
        ("roceo12399", "1: DOWN", "3: Disabled", ""),
        ("roceo12409", "1: DOWN", "3: Disabled", ""),
    ]
    import os

    for name, state, phys, rate in devices:
        port_dir = base / name / "ports" / "1"
        port_dir.mkdir(parents=True)
        (port_dir / "state").write_text(state)
        (port_dir / "phys_state").write_text(phys)
        (port_dir / "rate").write_text(rate)
        (port_dir / "link_layer").write_text("Ethernet")
        # One valid RoCE v2 GID on every ACTIVE port so the GID rule
        # doesn't accidentally fail the test (which would mask the
        # selector behavior we're trying to verify).
        gids = port_dir / "gids"
        gids.mkdir()
        (gids / "0").write_text("fe80:0000:0000:0000:0000:0000:0000:0001")
        types = port_dir / "gid_attrs" / "types"
        types.mkdir(parents=True)
        (types / "0").write_text("IB/RoCE v2")

    _install_synthetic_ib_tree(monkeypatch, base)

    monkeypatch.setenv(
        "NCCL_IB_HCA",
        "rocep158s0:1,rocep190s0:1,rocep206s0:1,rocep222s0:1,"
        "rocep28s0:1,rocep62s0:1,rocep79s0:1,rocep96s0:1",
    )

    out = _collect_nic_status(expected_count=None)

    # 12 ports total, 8 included, 4 excluded -- the headline assertion.
    assert len(out["ports"]) == 12
    assert len(out["included_ports"]) == 8
    assert len(out["excluded_ports"]) == 4
    # The 2 Disabled frontend ports and the 2 not-in-HCA storage ports.
    assert set(out["excluded_ports"]) == {
        "rocep159s0:1",
        "rocep29s0:1",
        "roceo12399:1",
        "roceo12409:1",
    }
    # Selector metadata records the source so the operator can verify
    # which precedence layer fired.
    assert out["selector"]["source"] == "env"
    # Zero hard issues -> node would PASS.
    assert out["issues"] == []
    # Every excluded port produced an info_issue mentioning the env
    # source. The two Disabled ports MUST also carry their state info
    # so operators investigating "are my frontend NICs still down?" can
    # see it without opening every per-port record.
    info = "\n".join(out["info_issues"])
    assert "NCCL_IB_HCA" in info
    assert "phys_state=Disabled" in info  # frontend ports
    # ACTIVE-but-not-in-HCA ports do NOT need the state suffix (they're
    # healthy, just reserved for sockets/storage).


def test_collect_nic_status_heuristic_only_excludes_disabled_phys_state(
    tmp_path, monkeypatch
):
    """F.15 -- with no env and no CLI selector, the heuristic must
    auto-exclude `phys_state=Disabled` ports but MUST keep `phys_state=
    Polling` (cable unplugged on an intended-up port) and `phys_state=
    LinkUp with state=INIT` (driver/SM didn't finish bringup) in the
    included set so they hard-fail. The whole point of choosing Disabled
    as the exclusion signal is that it's the only phys_state that
    unambiguously means "admin-down, not used"."""
    from primus.tools.preflight.node_smoke.collectors.nics import _collect_nic_status

    base = tmp_path / "ib"
    devices = [
        ("training_ok", "4: ACTIVE", "5: LinkUp", "400 Gb/sec"),
        ("frontend_disabled", "1: DOWN", "3: Disabled", ""),
        ("training_cable_pulled", "1: DOWN", "2: Polling", ""),
        ("training_unconfigured", "2: INIT", "5: LinkUp", "400 Gb/sec"),
    ]
    for name, state, phys, rate in devices:
        port_dir = base / name / "ports" / "1"
        port_dir.mkdir(parents=True)
        (port_dir / "state").write_text(state)
        (port_dir / "phys_state").write_text(phys)
        (port_dir / "rate").write_text(rate)
        (port_dir / "link_layer").write_text("Ethernet")
        gids = port_dir / "gids"
        gids.mkdir()
        (gids / "0").write_text("fe80:0000:0000:0000:0000:0000:0000:0001")
        types = port_dir / "gid_attrs" / "types"
        types.mkdir(parents=True)
        (types / "0").write_text("IB/RoCE v2")

    _install_synthetic_ib_tree(monkeypatch, base)
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)

    out = _collect_nic_status(expected_count=None)

    assert out["selector"]["source"] == "heuristic"
    # Only the Disabled port is excluded by the heuristic.
    assert out["excluded_ports"] == ["frontend_disabled:1"]
    # The 3 remaining are included; the 2 broken-but-included ones produce
    # hard issues.
    assert set(out["included_ports"]) == {
        "training_ok:1",
        "training_cable_pulled:1",
        "training_unconfigured:1",
    }
    issues_text = " ".join(out["issues"])
    assert "training_cable_pulled" in issues_text  # state=DOWN
    assert "training_unconfigured" in issues_text  # state=INIT
    assert "training_ok" not in issues_text


def test_collect_nic_status_empty_set_guard_when_everything_excluded(
    tmp_path, monkeypatch
):
    """F.16 -- defense in depth: if the selector ends up excluding every
    discovered port (e.g. operator typo'd --rdma-nic-allowlist, or the
    whole RoCE card got admin-disabled), the node MUST still hard-fail.
    A node with zero training NICs cannot participate in inter-node
    training and silently passing it would be the worst possible
    regression introduced by the selector feature."""
    from primus.tools.preflight.node_smoke.collectors.nics import _collect_nic_status

    base = tmp_path / "ib"
    port_dir = base / "training_ok" / "ports" / "1"
    port_dir.mkdir(parents=True)
    (port_dir / "state").write_text("4: ACTIVE")
    (port_dir / "phys_state").write_text("5: LinkUp")
    (port_dir / "rate").write_text("400 Gb/sec")
    (port_dir / "link_layer").write_text("Ethernet")

    _install_synthetic_ib_tree(monkeypatch, base)
    # CLI allowlist that matches NOTHING (typo or operator mistake).
    out = _collect_nic_status(expected_count=None, allowlist="this_device_does_not_exist:1")

    assert out["included_ports"] == []
    assert out["excluded_ports"] == ["training_ok:1"]
    # The empty-set guard fires -- node FAIL.
    assert any(
        "no included RDMA NIC ports" in issue for issue in out["issues"]
    ), out["issues"]


def test_collect_nic_status_expected_count_compares_included(
    tmp_path, monkeypatch
):
    """F.17 -- the behavior change for --expected-rdma-nics: it must
    compare against the *included* count, not the total /sys/class/
    infiniband count. Otherwise `--expected-rdma-nics 8` would fail on
    the very clusters this feature exists to help (12 devices, 8
    training)."""
    from primus.tools.preflight.node_smoke.collectors.nics import _collect_nic_status

    base = tmp_path / "ib"
    # 8 training + 2 frontend Disabled = 10 total ports, 8 training NICs.
    for i in range(8):
        d = base / f"trainnic{i}" / "ports" / "1"
        d.mkdir(parents=True)
        (d / "state").write_text("4: ACTIVE")
        (d / "phys_state").write_text("5: LinkUp")
        (d / "rate").write_text("400 Gb/sec")
        (d / "link_layer").write_text("Ethernet")
        gids = d / "gids"
        gids.mkdir()
        (gids / "0").write_text("fe80:0000:0000:0000:0000:0000:0000:0001")
        types = d / "gid_attrs" / "types"
        types.mkdir(parents=True)
        (types / "0").write_text("IB/RoCE v2")
    for i in range(2):
        d = base / f"frontnic{i}" / "ports" / "1"
        d.mkdir(parents=True)
        (d / "state").write_text("1: DOWN")
        (d / "phys_state").write_text("3: Disabled")
        (d / "rate").write_text("")
        (d / "link_layer").write_text("Ethernet")

    _install_synthetic_ib_tree(monkeypatch, base)
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)

    # With --expected-rdma-nics 8 on the heuristic path: the 2 Disabled
    # ports get auto-excluded, leaving 8 included = matches expected.
    out = _collect_nic_status(expected_count=8)
    assert out["issues"] == [], out["issues"]
    # And the inverse: --expected-rdma-nics 10 (treating the total
    # `/sys/class/infiniband` count as the expected) MUST fail, because
    # the comparison happens against the included set.
    out = _collect_nic_status(expected_count=10)
    assert any("!= expected 10" in i for i in out["issues"])
