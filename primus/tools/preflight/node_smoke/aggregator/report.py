###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Cluster smoke-report markdown writer.

The single entry point :func:`write_smoke_report` composes the report by
calling small per-section ``_write_<section>`` helpers in the exact
order the original monolithic block produced. Behaviour is preserved
verbatim, including:

* every section header
* the per-section ``try/except`` wrapping pattern
* the two intentional EXCEPTIONS to that pattern -- ``Tier 2 perf
  summary`` and ``Failing nodes -- full reasons`` -- which deliberately
  do NOT have their own ``try/except`` in the original (so a regression
  in either bubbles up rather than being swallowed). Per refactor
  guardrails these stay un-wrapped.
"""

from __future__ import annotations

from typing import IO, Any, Dict, List

from ..logging_utils import _warn
from .summarizers import (
    _busy_gpu_rows,
    _clock_summary,
    _gpu_activity_rows,
    _gpu_low_level_outlier_rows,
    _host_limits_issue_rows,
    _nic_fw_drift_rows,
    _nic_issue_rows,
    _pretouch_hbm_rows,
    _stack_drift_rows,
    _tooling_inventory_rows,
    _tooling_latency_rows,
    _xgmi_issue_rows,
)


def _write_header(
    f: IO[str],
    nodes: List[Dict[str, Any]],
    passing: List[Dict[str, Any]],
    failing: List[Dict[str, Any]],
    expected: Any,
) -> None:
    """Write the report title, summary counts, and per-node status table."""
    f.write("# Node-Local Smoke Test Report\n\n")
    f.write(
        f"- **Expected nodes**: `{expected if expected is not None else 'unknown'}`\n"
    )
    f.write(f"- **Reported nodes**: `{len(nodes)}`\n")
    f.write(f"- **PASS**: `{len(passing)}`  **FAIL**: `{len(failing)}`\n\n")
    f.write("| Node | Hostname | Status | Duration | Top fail reason |\n")
    f.write("|------|----------|--------|----------|-----------------|\n")
    for n in nodes:
        reasons = n.get("fail_reasons") or []
        top = (reasons[0] if reasons else "").replace("|", "/")
        if len(top) > 120:
            top = top[:117] + "..."
        f.write(
            f"| {n.get('node_rank', '?')} | {n.get('host', '?')} | "
            f"{n.get('status', '?')} | {n.get('duration_sec', 0)}s | {top} |\n"
        )


def _write_stack_drift(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- A. Stack drift across cluster -----
    # Empty section when every node reports the same value for every
    # scalar fingerprint key. We always print the section header so the
    # operator can see at a glance that the check ran.
    # Each helper is wrapped so a single bug in one section can never
    # truncate the whole report; the failure is recorded inline so the
    # operator still sees something for that section.
    f.write("\n## Stack drift across cluster\n\n")
    try:
        drift = _stack_drift_rows(nodes)
        if not drift:
            f.write("*All nodes match.*\n")
        else:
            f.write("| Key | Majority (count/total) | Outlier nodes |\n")
            f.write("|------|-------------------------|----------------|\n")
            for row in drift:
                outliers = "; ".join(
                    f"`{h}` = `{v}`" for h, v in row["outliers"]
                )
                f.write(
                    f"| `{row['key']}` | `{row['majority']}` "
                    f"({row['count']}/{row['total']}) | {outliers} |\n"
                )
    except Exception as e:
        f.write(f"*Stack-drift section failed to render: {e}*\n")
        _warn(f"stack-drift render failed: {e}")


def _write_nic_fw_drift(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- A.2 NIC firmware drift across cluster -----
    f.write("\n## NIC firmware drift across cluster\n\n")
    try:
        nic_drift = _nic_fw_drift_rows(nodes)
        if not nic_drift:
            f.write("*All NIC firmwares match (or no NICs reported).*\n")
        else:
            f.write("| NIC | Majority FW (count/total) | Outlier nodes |\n")
            f.write("|-----|---------------------------|----------------|\n")
            for row in nic_drift:
                outliers = "; ".join(
                    f"`{h}` = `{v}`" for h, v in row["outliers"]
                )
                f.write(
                    f"| `{row['device']}` | `{row['majority']}` "
                    f"({row['count']}/{row['total']}) | {outliers} |\n"
                )
    except Exception as e:
        f.write(f"*NIC firmware drift section failed to render: {e}*\n")
        _warn(f"nic-fw-drift render failed: {e}")


def _write_nic_issues(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- B. NIC / RDMA roll-call issues -----
    f.write("\n## NIC / RDMA roll-call issues\n\n")
    try:
        nic_issues = _nic_issue_rows(nodes)
        if not nic_issues:
            f.write("*No NIC issues.*\n")
        else:
            f.write("| Node | Hostname | Issue |\n")
            f.write("|------|----------|-------|\n")
            for row in nic_issues:
                msg = str(row["issue"]).replace("|", "/")
                if len(msg) > 160:
                    msg = msg[:157] + "..."
                f.write(
                    f"| {row['node_rank']} | {row['host']} | {msg} |\n"
                )
    except Exception as e:
        f.write(f"*NIC issues section failed to render: {e}*\n")
        _warn(f"nic-issues render failed: {e}")


def _write_nic_port_count(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- B.2 NIC port-count summary (helps spot "node X has fewer
    # NICs than the cluster") -- always rendered, even when no per-port
    # issue tripped. We flag any node whose port count differs from the
    # cluster majority so operators can act on partial-degradation cases
    # like 7/8 ports without having to set --expected-rdma-nics.
    f.write("\n## NIC port-count summary\n\n")
    try:
        from collections import Counter

        counts = []
        for n in nodes:
            nic = (n.get("tier1") or {}).get("nics") or {}
            counts.append((
                n.get("node_rank", "?"),
                n.get("host", "?"),
                len(nic.get("ports") or []),
            ))
        if not counts:
            f.write("*No NIC data reported.*\n")
        else:
            cnt = Counter(c for *_, c in counts)
            majority_count, _ = cnt.most_common(1)[0]
            anomalies = [
                (nr, h, c) for nr, h, c in counts if c != majority_count
            ]
            f.write(
                f"Cluster-majority port count: **{majority_count}** "
                f"(seen on {cnt[majority_count]}/{len(counts)} nodes).\n\n"
            )
            if not anomalies:
                f.write("*Every node reports the majority count.*\n")
            else:
                f.write("| Node | Hostname | Ports found |\n")
                f.write("|------|----------|-------------|\n")
                for nr, h, c in anomalies:
                    f.write(f"| {nr} | {h} | {c} |\n")
    except Exception as e:
        f.write(f"*NIC port-count summary failed to render: {e}*\n")
        _warn(f"nic-port-count render failed: {e}")


def _write_host_limits(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- C. Host limits issues -----
    f.write("\n## Host limits issues\n\n")
    try:
        limits_issues = _host_limits_issue_rows(nodes)
        if not limits_issues:
            f.write("*No host-limit issues.*\n")
        else:
            f.write("| Node | Hostname | Issue |\n")
            f.write("|------|----------|-------|\n")
            for row in limits_issues:
                msg = str(row["issue"]).replace("|", "/")
                if len(msg) > 200:
                    msg = msg[:197] + "..."
                f.write(
                    f"| {row['node_rank']} | {row['host']} | {msg} |\n"
                )
    except Exception as e:
        f.write(f"*Host limits section failed to render: {e}*\n")
        _warn(f"host-limits render failed: {e}")


def _write_gpu_visibility(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- GPU visibility issues (no GPUs / amd-smi vs torch mismatch) -----
    # Independent guard -- doesn't rely on the reused gpu_info collector
    # emitting a level=fail finding, which has been known to silently
    # downgrade to warn when collect_gpu_info() raises.
    f.write("\n## GPU visibility issues\n\n")
    try:
        vis_rows: List[Dict[str, Any]] = []
        for n in nodes:
            vis = (n.get("tier1") or {}).get("gpu_visibility") or {}
            for issue in vis.get("fail_reasons", []) or []:
                vis_rows.append({
                    "node_rank": n.get("node_rank", "?"),
                    "host": n.get("host", "?"),
                    "torch": vis.get("torch_visible"),
                    "amd_smi": vis.get("amd_smi_visible"),
                    "expected": vis.get("expected_gpus"),
                    "issue": issue,
                })
        if not vis_rows:
            f.write("*Every node resolved expected_gpus >= 1 and torch + "
                    "amd-smi agree on the GPU count.*\n")
        else:
            f.write(
                "Nodes where the GPU is invisible to torch, or where "
                "amd-smi sees more GPUs than torch (stale ROCm / wedged "
                "amdgpu driver). These are hard fails independent of "
                "every other collector.\n\n"
            )
            f.write(
                "| Node | Hostname | expected | torch | amd-smi | Issue |\n"
            )
            f.write(
                "|------|----------|----------|-------|---------|-------|\n"
            )
            for row in vis_rows:
                msg = str(row["issue"]).replace("|", "/")
                if len(msg) > 200:
                    msg = msg[:197] + "..."
                f.write(
                    f"| {row['node_rank']} | {row['host']} | "
                    f"{row['expected']} | {row['torch']} | "
                    f"{row['amd_smi']} | {msg} |\n"
                )
    except Exception as e:
        f.write(f"*GPU visibility section failed to render: {e}*\n")
        _warn(f"gpu-visibility render failed: {e}")


def _write_gpu_low_level(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- D-1: GPU low-level outliers (PCIe link, HBM total) -----
    f.write("\n## GPU low-level outliers (PCIe link / HBM)\n\n")
    try:
        gpu_outliers = _gpu_low_level_outlier_rows(nodes)
        if not gpu_outliers:
            f.write("*All GPUs match the cluster majority on PCIe link "
                    "and HBM total.*\n")
        else:
            f.write(
                "Per-GPU values that differ from the cluster majority. A "
                "GPU sitting at half PCIe width / half HBM is almost "
                "always a hardware fault on that single device.\n\n"
            )
            f.write(
                "| Metric | Cluster majority (count/total) | "
                "Outliers (`host:gpu` = value) |\n"
            )
            f.write(
                "|--------|---------------------------------|"
                "-------------------------------|\n"
            )
            for row in gpu_outliers:
                out_str = "; ".join(
                    f"`{h}:{g}` = `{v}`" for h, g, v in row["outliers"]
                )
                f.write(
                    f"| {row['label']} | `{row['majority']}` "
                    f"({row['count']}/{row['total']}) | {out_str} |\n"
                )
    except Exception as e:
        f.write(f"*GPU low-level section failed to render: {e}*\n")
        _warn(f"gpu-low-level render failed: {e}")


def _write_xgmi(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- D-2: XGMI link issues -----
    f.write("\n## XGMI link issues\n\n")
    try:
        xgmi_issues = _xgmi_issue_rows(nodes)
        if not xgmi_issues:
            f.write("*All GPU pairs report XGMI on every node "
                    "(or amd-smi topology was unavailable).*\n")
        else:
            f.write(
                "Any non-XGMI GPU pair is a hard fail -- intra-node "
                "collectives silently fall back to PCIe and lose 5-10x "
                "of the bandwidth NCCL/RCCL expects.\n\n"
            )
            f.write("| Node | Hostname | Issue |\n")
            f.write("|------|----------|-------|\n")
            for row in xgmi_issues:
                msg = str(row["summary"]).replace("|", "/")
                if len(msg) > 200:
                    msg = msg[:197] + "..."
                f.write(
                    f"| {row['node_rank']} | {row['host']} | {msg} |\n"
                )
    except Exception as e:
        f.write(f"*XGMI section failed to render: {e}*\n")
        _warn(f"xgmi render failed: {e}")


def _write_clock(
    f: IO[str], nodes: List[Dict[str, Any]], skew_warn_sec: float,
) -> None:
    # ----- E: cluster wall-clock spread + time-daemon roll-call -----
    f.write("\n## Cluster clock + time daemons\n\n")
    try:
        clk = _clock_summary(nodes, skew_warn_sec=skew_warn_sec)
        spread = clk["spread_sec"]
        if spread is None:
            f.write("*Not enough nodes reported a wall-clock timestamp.*\n")
        else:
            marker = " (**warn** -- exceeds " \
                     f"{clk['spread_warn_sec']}s)" if clk["spread_warn"] else ""
            f.write(
                f"- Wall-clock spread across {clk['n_nodes_with_time']} "
                f"nodes: **{spread}s**{marker}.\n"
            )
            f.write(
                f"- Earliest: `{clk['earliest_host']}`, "
                f"latest: `{clk['latest_host']}`.\n"
            )
            f.write(
                "- (Spread is an upper bound on real clock skew -- it "
                "also includes srun launch jitter.)\n"
            )
        if clk["no_daemon_hosts"]:
            f.write("\n**Nodes with no active time-sync daemon "
                    "(chronyd / ntpd / systemd-timesyncd):**\n\n")
            f.write("| Node | Hostname |\n")
            f.write("|------|----------|\n")
            for nr, h in clk["no_daemon_hosts"]:
                f.write(f"| {nr} | {h} |\n")
        else:
            f.write("\n*Every node has at least one active time-sync "
                    "daemon.*\n")
    except Exception as e:
        f.write(f"*Clock section failed to render: {e}*\n")
        _warn(f"clock render failed: {e}")


def _write_tooling_latency(
    f: IO[str], nodes: List[Dict[str, Any]], rocm_smi_warn_sec: float,
) -> None:
    # ----- F-partial: rocm-smi self-latency -----
    f.write("\n## Tooling self-latency (`rocm-smi --version`)\n\n")
    try:
        tool_rows = _tooling_latency_rows(
            nodes, warn_sec=float(rocm_smi_warn_sec),
        )
        if not tool_rows:
            f.write(
                "*No nodes exceeded the warn threshold "
                f"({rocm_smi_warn_sec}s) and no timeouts.*\n"
            )
        else:
            f.write(
                "Slow `rocm-smi --version` calls historically precede a "
                "wedged amdgpu driver. Hitting the hard timeout is a "
                "node FAIL; slow-but-completed calls are warn-only.\n\n"
            )
            f.write("| Node | Hostname | Latency (s) | Flag |\n")
            f.write("|------|----------|-------------|------|\n")
            for r in tool_rows:
                lat = r.get("latency_sec")
                lat_s = (
                    f"{lat:.2f}" if isinstance(lat, (int, float)) else "?"
                )
                f.write(
                    f"| {r['node_rank']} | {r['host']} | "
                    f"{lat_s} | {r['flag']} |\n"
                )
    except Exception as e:
        f.write(f"*Tooling section failed to render: {e}*\n")
        _warn(f"tooling render failed: {e}")


def _write_tooling_availability(
    f: IO[str], nodes: List[Dict[str, Any]],
) -> None:
    # ----- Tooling availability (always-on; loud counterweight to
    # the silent skips that happen when amd-smi / rocm-smi / lsof
    # are missing from PATH) -----
    f.write("\n## Tooling availability\n\n")
    try:
        inv = _tooling_inventory_rows(nodes)
        tracked = inv["tracked"]
        mc = inv["missing_counts"]
        if not inv["any_missing"]:
            f.write(
                "*Every tracked tool (`"
                + "`, `".join(tracked)
                + "`) was present in PATH on every node.*\n"
            )
        else:
            summary_bits = []
            for t in tracked:
                if mc[t]:
                    summary_bits.append(
                        f"`{t}` missing on **{mc[t]}** node(s)"
                    )
            # Compute cluster-wide uncovered-checks summary so the
            # operator immediately knows whether the missing tools
            # actually leave a coverage hole or whether the rocm-smi
            # / lsof fallbacks are picking up the slack.
            uncovered_counts: Dict[str, int] = {}
            for n in nodes:
                uc = ((n.get("tier1") or {})
                      .get("tooling_inventory") or {}).get("uncovered") or []
                for c in uc:
                    uncovered_counts[c] = uncovered_counts.get(c, 0) + 1
            if uncovered_counts:
                uc_bits = [
                    f"`{c}` on **{n}** node(s)"
                    for c, n in sorted(uncovered_counts.items())
                ]
                coverage_line = (
                    "Checks with NO working tool (truly silent-skipped): "
                    + "; ".join(uc_bits) + "."
                )
            else:
                coverage_line = (
                    "Every check is still covered via the rocm-smi "
                    "or lsof fallback on every node -- no checks are "
                    "silently skipped."
                )
            f.write(
                "Several Tier 1 checks (ECC, XGMI, foreign-process, "
                "GPU activity, wedged-driver) prefer `amd-smi` but "
                "fall back to `rocm-smi` (and `lsof` for foreign-"
                "process) when amd-smi is missing. "
                + "; ".join(summary_bits) + ". " + coverage_line
                + " Add `--require-tools amd-smi,rocm-smi` to `run` to "
                "promote a missing tool to a node FAIL anyway.\n\n"
            )
            # Per-node table -- only the nodes that ARE missing something,
            # so a healthy cluster doesn't get a giant N-row table.
            f.write(
                "| Node | Hostname | "
                + " | ".join(tracked)
                + " |\n"
            )
            f.write(
                "|------|----------| "
                + " | ".join("---" for _ in tracked)
                + " |\n"
            )
            for r in inv["rows"]:
                if all(r.get(t) for t in tracked):
                    continue
                cells = []
                for t in tracked:
                    cells.append("OK" if r.get(t) else "**MISSING**")
                f.write(
                    f"| {r['node_rank']} | {r['host']} | "
                    + " | ".join(cells)
                    + " |\n"
                )
    except Exception as e:
        f.write(f"*Tooling availability section failed to render: {e}*\n")
        _warn(f"tooling-availability render failed: {e}")


def _write_busy_gpus(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    # ----- G: Busy GPUs / leaked processes -----
    f.write("\n## Busy GPUs / leaked processes\n\n")
    try:
        busy_rows = _busy_gpu_rows(nodes)
        if not busy_rows:
            f.write(
                "*No foreign processes detected on any GPU "
                "(or `amd-smi process` was unavailable on every node).*\n"
            )
        else:
            f.write(
                "Foreign PIDs found holding GPUs at smoke start. The most "
                "common cause is leaked Python ranks from a previous "
                "training job (look for `python` / `torchrun` / `train.py`). "
                "Clean up with `pkill -9 -f train.py` (or similar) on the "
                "listed nodes BEFORE launching the next job.\n\n"
            )
            f.write("| Node | Hostname | GPU | PID | Process | HBM held (GiB) |\n")
            f.write("|------|----------|-----|-----|---------|----------------|\n")
            for r in busy_rows:
                name = str(r.get("name", "")).replace("|", "/")[:40]
                hbm = r.get("hbm_gib")
                hbm_s = f"{hbm}" if hbm is not None else "?"
                f.write(
                    f"| {r['node_rank']} | {r['host']} | "
                    f"{r['gpu']} | {r['pid']} | `{name}` | {hbm_s} |\n"
                )
    except Exception as e:
        f.write(f"*Busy-GPU section failed to render: {e}*\n")
        _warn(f"busy-gpu render failed: {e}")


def _write_pretouch_hbm(
    f: IO[str], nodes: List[Dict[str, Any]], hbm_busy_threshold_gib: float,
) -> None:
    # ----- G: Pre-touch HBM-used outliers -----
    f.write("\n## GPU pre-touch HBM usage outliers\n\n")
    try:
        threshold = float(hbm_busy_threshold_gib)
        pt_rows = _pretouch_hbm_rows(nodes, threshold_gib=threshold)
        if not pt_rows:
            f.write(
                f"*No GPU exceeded the pre-touch HBM threshold "
                f"({threshold} GiB) -- every GPU started clean.*\n"
            )
        else:
            f.write(
                f"GPUs with more than **{threshold} GiB** of HBM already "
                f"in use BEFORE smoke touched the device. This number is "
                "not polluted by our own caching allocator (it's measured "
                "before any allocation), so it directly reflects foreign "
                "or leaked occupancy.\n\n"
            )
            f.write("| Node | Hostname | GPU | HBM used pre-touch (GiB) |\n")
            f.write("|------|----------|-----|---------------------------|\n")
            for r in pt_rows:
                f.write(
                    f"| {r['node_rank']} | {r['host']} | "
                    f"{r['gpu']} | {r['used_gib']} |\n"
                )
    except Exception as e:
        f.write(f"*Pre-touch HBM section failed to render: {e}*\n")
        _warn(f"pretouch-hbm render failed: {e}")


def _write_gpu_activity(
    f: IO[str], nodes: List[Dict[str, Any]], gpu_activity_warn_pct: float,
) -> None:
    # ----- G: GPU compute activity outliers -----
    f.write("\n## GPU compute-activity outliers\n\n")
    try:
        warn_pct = float(gpu_activity_warn_pct)
        act_rows = _gpu_activity_rows(nodes, warn_pct=warn_pct)
        if not act_rows:
            f.write(
                f"*No GPU exceeded `gfx_activity_pct >= {warn_pct}%` at "
                "smoke start (or amd-smi did not report activity).*\n"
            )
        else:
            f.write(
                f"GPUs reporting **>= {warn_pct}%** compute activity at "
                "smoke start. Short bursts are normal; sustained "
                "non-trivial activity across multiple GPUs strongly "
                "suggests a leaked rank still running compute. Warn-only "
                "(does not by itself fail the node).\n\n"
            )
            f.write("| Node | Hostname | GPU | Activity % |\n")
            f.write("|------|----------|-----|------------|\n")
            for r in act_rows:
                f.write(
                    f"| {r['node_rank']} | {r['host']} | "
                    f"{r['gpu']} | {r['activity_pct']} |\n"
                )
    except Exception as e:
        f.write(f"*Activity section failed to render: {e}*\n")
        _warn(f"activity render failed: {e}")


def _write_tier2_perf_summary(f: IO[str], nodes: List[Dict[str, Any]]) -> None:
    """Write the Tier 2 perf summary section.

    INTENTIONALLY UN-WRAPPED in try/except (matching the original
    monolithic block): a regression in this loop should bubble up rather
    than be silently swallowed. Only emitted when at least one node
    actually ran Tier 2.
    """
    # Tier 2 perf summary -- only emitted when at least one node ran Tier 2.
    # Surfaces per-node GEMM TFLOPS / HBM GB/s (min/median/max across the
    # node's GPUs) plus the local RCCL all-reduce GB/s, so outliers across
    # the cluster are visible without opening every per-node JSON.
    perf_rows: List[str] = []
    any_tier2 = False
    for n in nodes:
        t2 = n.get("tier2") or {}
        per_gpu = (n.get("tier1") or {}).get("per_gpu") or []
        gemm = [
            p.get("details", {}).get("gemm_tflops")
            for p in per_gpu
            if isinstance(p.get("details", {}).get("gemm_tflops"), (int, float))
        ]
        hbm = [
            p.get("details", {}).get("hbm_gbs")
            for p in per_gpu
            if isinstance(p.get("details", {}).get("hbm_gbs"), (int, float))
        ]
        rccl_gbs = (t2.get("rccl") or {}).get("gbs")
        if not gemm and not hbm and rccl_gbs is None:
            perf_rows.append(
                f"| {n.get('node_rank', '?')} | {n.get('host', '?')} |  |  |  |"
            )
            continue
        any_tier2 = True

        def _fmt_stats(xs):
            if not xs:
                return ""
            xs_sorted = sorted(xs)
            med = xs_sorted[len(xs_sorted) // 2]
            return f"{min(xs):.1f} / {med:.1f} / {max(xs):.1f}"

        perf_rows.append(
            f"| {n.get('node_rank', '?')} | {n.get('host', '?')} | "
            f"{_fmt_stats(gemm)} | {_fmt_stats(hbm)} | "
            f"{rccl_gbs if rccl_gbs is not None else ''} |"
        )

    if any_tier2:
        f.write("\n## Tier 2 perf summary\n\n")
        f.write(
            "Per-node GEMM TFLOPS (8192^3 bf16) and HBM GB/s shown as "
            "`min / median / max` across the node's GPUs. RCCL GB/s is the "
            "node-local 8-GPU all-reduce algorithmic bandwidth at 64 MB.\n\n"
        )
        f.write(
            "| Node | Hostname | GEMM TFLOPS (min/med/max) | "
            "HBM GB/s (min/med/max) | Local RCCL GB/s |\n"
        )
        f.write(
            "|------|----------|----------------------------|"
            "------------------------|------------------|\n"
        )
        for r in perf_rows:
            f.write(r + "\n")


def _write_failing_reasons(
    f: IO[str], failing: List[Dict[str, Any]],
) -> None:
    """Write the per-node fail-reason dump for failing nodes.

    INTENTIONALLY UN-WRAPPED in try/except (matching the original
    monolithic block): if iterating fail_reasons raises, surface it
    instead of swallowing it.
    """
    if failing:
        f.write("\n## Failing nodes -- full reasons\n\n")
        for n in failing:
            f.write(f"### {n.get('host', '?')}\n\n")
            for r in n.get("fail_reasons") or []:
                f.write(f"- {r}\n")
            f.write("\n")


def write_smoke_report(
    report_path: str,
    *,
    nodes: List[Dict[str, Any]],
    passing: List[Dict[str, Any]],
    failing: List[Dict[str, Any]],
    expected: Any,
    clock_skew_warn_sec: float,
    rocm_smi_warn_sec: float,
    hbm_busy_threshold_gib: float,
    gpu_activity_warn_pct: float,
) -> None:
    """Write the cluster smoke report to ``report_path``.

    Section ORDER and HEADINGS are part of the operator-facing contract
    -- many CI/CD pipelines and slack bots scrape ``smoke_report.md`` for
    specific ``##`` headings. Do NOT reorder or rename without a deliberate
    behavior change. The order matches the original monolithic writer
    (header, A, A.2, B, B.2, C, GPU visibility, D-1, D-2, E, F-partial,
    Tooling availability, G x3, Tier 2 perf summary [conditional],
    Failing nodes [conditional]).
    """
    with open(report_path, "w", encoding="utf-8") as f:
        _write_header(f, nodes, passing, failing, expected)
        _write_stack_drift(f, nodes)
        _write_nic_fw_drift(f, nodes)
        _write_nic_issues(f, nodes)
        _write_nic_port_count(f, nodes)
        _write_host_limits(f, nodes)
        _write_gpu_visibility(f, nodes)
        _write_gpu_low_level(f, nodes)
        _write_xgmi(f, nodes)
        _write_clock(f, nodes, skew_warn_sec=clock_skew_warn_sec)
        _write_tooling_latency(f, nodes, rocm_smi_warn_sec=rocm_smi_warn_sec)
        _write_tooling_availability(f, nodes)
        _write_busy_gpus(f, nodes)
        _write_pretouch_hbm(
            f, nodes, hbm_busy_threshold_gib=hbm_busy_threshold_gib,
        )
        _write_gpu_activity(
            f, nodes, gpu_activity_warn_pct=gpu_activity_warn_pct,
        )
        _write_tier2_perf_summary(f, nodes)
        _write_failing_reasons(f, failing)
