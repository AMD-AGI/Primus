###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pure data-shaping helpers used by the cluster-level report writer.

Every function takes the loaded per-node JSON list (``nodes``) and
returns plain rows or summary dicts. None of them format markdown -- the
report writer in :mod:`.report` is solely responsible for layout.
Each helper is best-effort: missing data degrades to an empty list rather
than raising.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..collectors.tooling import _TRACKED_TOOLS

# ---------------------------------------------------------------------------
# Aggregator helpers -- A. stack/NIC drift, B. NIC issues, C. host limits
# ---------------------------------------------------------------------------


def _stack_drift_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For every *scalar* fingerprint key, find the cluster-majority value and
    list the nodes that disagree.

    Returns one row per key that has at least one outlier. Keys missing from
    every node, or where every node reported the same value, are omitted so
    a healthy cluster produces an empty list.
    """
    from collections import Counter

    # Only collect keys that at least ONE node reported as a scalar. We
    # ignore None here so a key that happens to be None on one node and a
    # dict on another (e.g. nic_fw on a node without an IB stack) doesn't
    # leak into the scalar-drift loop and crash Counter() with an unhashable
    # value.
    keys: set = set()
    for n in nodes:
        fp = ((n.get("tier1") or {}).get("fingerprint") or {}) or {}
        for k, v in fp.items():
            if isinstance(v, (str, int, float)):
                keys.add(k)

    rows: List[Dict[str, Any]] = []
    for k in sorted(keys):
        per_host: List[tuple] = []
        for n in nodes:
            fp = ((n.get("tier1") or {}).get("fingerprint") or {}) or {}
            v = fp.get(k)
            # Defense in depth: skip non-scalar values per-host too, in case
            # different nodes disagree on the type for the same key.
            if not isinstance(v, (str, int, float)):
                continue
            per_host.append((n.get("host", "?"), v))
        if not per_host:
            continue
        c = Counter(v for _, v in per_host)
        majority, count = c.most_common(1)[0]
        outliers = [(h, v) for h, v in per_host if v != majority]
        if not outliers:
            continue
        rows.append(
            {
                "key": k,
                "majority": majority,
                "count": count,
                "total": len(per_host),
                "outliers": outliers,
            }
        )
    return rows


def _nic_fw_drift_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-IB-device firmware drift across the cluster (e.g. rdma0 mismatch)."""
    from collections import Counter

    all_devs: set = set()
    for n in nodes:
        fp = ((n.get("tier1") or {}).get("fingerprint") or {}) or {}
        all_devs.update((fp.get("nic_fw") or {}).keys())

    rows: List[Dict[str, Any]] = []
    for dev in sorted(all_devs):
        per_host: List[tuple] = []
        for n in nodes:
            fp = ((n.get("tier1") or {}).get("fingerprint") or {}) or {}
            v = (fp.get("nic_fw") or {}).get(dev)
            if v is None:
                continue
            per_host.append((n.get("host", "?"), v))
        if not per_host:
            continue
        c = Counter(v for _, v in per_host)
        majority, count = c.most_common(1)[0]
        outliers = [(h, v) for h, v in per_host if v != majority]
        if not outliers:
            continue
        rows.append(
            {
                "device": dev,
                "majority": majority,
                "count": count,
                "total": len(per_host),
                "outliers": outliers,
            }
        )
    return rows


def _nic_issue_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-node NIC roll-call issues (port DOWN / no GIDs / count mismatch)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        nic = (n.get("tier1") or {}).get("nics") or {}
        for issue in nic.get("issues", []) or []:
            rows.append(
                {
                    "node_rank": n.get("node_rank", "?"),
                    "host": n.get("host", "?"),
                    "issue": issue,
                }
            )
    return rows


def _host_limits_issue_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-node host-limit hard violations (ulimit -l / /dev/shm too low)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        hl = (n.get("tier1") or {}).get("host_limits") or {}
        for issue in hl.get("fail_reasons", []) or []:
            rows.append(
                {
                    "node_rank": n.get("node_rank", "?"),
                    "host": n.get("host", "?"),
                    "issue": issue,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Aggregator helpers -- D-1 / D-2 / E / F
# ---------------------------------------------------------------------------


def _gpu_low_level_outlier_rows(
    nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find per-GPU outliers in PCIe link + HBM total across the cluster.

    For each scalar metric the cluster has a strong majority value (e.g.
    16 lanes, 32 GT/s, 191 GiB HBM). A single GPU below the majority on
    any of these is almost always a hardware issue -- a cold-soldered
    socket, a degraded PCIe link, or HBM that the firmware refused to
    bring online. We surface every such (host, gpu, metric, value)
    tuple, with the cluster majority for context.

    Power cap and ECC counters from amd-smi are intentionally NOT included
    here; they have their own narrower checks (ECC = hard fail in
    ``_node_status_from``; power cap = informational only because cluster
    operators sometimes set per-rack caps deliberately).
    """
    from collections import Counter

    fields = (
        ("pcie_link_width", "PCIe width (lanes)"),
        ("pcie_link_speed_gts", "PCIe speed (GT/s)"),
        ("hbm_total_gib", "HBM total (GiB)"),
    )
    rows: List[Dict[str, Any]] = []
    for key, label in fields:
        per_gpu: List[tuple] = []  # (host, gpu_idx, value)
        for n in nodes:
            for p in (n.get("tier1") or {}).get("per_gpu") or []:
                low = (p.get("details") or {}).get("low_level") or {}
                v = low.get(key)
                if isinstance(v, (int, float)):
                    per_gpu.append((n.get("host", "?"), p.get("gpu", "?"), v))
        if not per_gpu:
            continue
        c = Counter(v for _, _, v in per_gpu)
        majority, count = c.most_common(1)[0]
        outliers = [(h, g, v) for h, g, v in per_gpu if v != majority]
        if not outliers:
            continue
        rows.append(
            {
                "key": key,
                "label": label,
                "majority": majority,
                "count": count,
                "total": len(per_gpu),
                "outliers": outliers,
            }
        )
    return rows


def _xgmi_issue_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-node XGMI link issues (any non-XGMI GPU pair)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        xg = (n.get("tier1") or {}).get("xgmi") or {}
        if not xg.get("ok"):
            err = xg.get("error")
            if err:
                rows.append(
                    {
                        "node_rank": n.get("node_rank", "?"),
                        "host": n.get("host", "?"),
                        "summary": f"could not collect topology: {err}",
                    }
                )
            continue
        bad = xg.get("non_xgmi_pairs") or []
        if not bad:
            continue
        # Show up to 6 sample pairs to keep the table readable; the full
        # matrix lives in the per-node JSON.
        sample = ", ".join(f"({i},{j})={t}" for i, j, t in bad[:6])
        suffix = "" if len(bad) <= 6 else f" (+{len(bad) - 6} more)"
        rows.append(
            {
                "node_rank": n.get("node_rank", "?"),
                "host": n.get("host", "?"),
                "summary": f"{len(bad)} non-XGMI pair(s): {sample}{suffix}",
            }
        )
    return rows


def _clock_summary(
    nodes: List[Dict[str, Any]],
    skew_warn_sec: float,
) -> Dict[str, Any]:
    """Compute wall-clock spread + per-node time-daemon health."""
    times: List[tuple] = []  # (host, wall_time_unix)
    no_daemon_hosts: List[tuple] = []  # (node_rank, host)
    for n in nodes:
        clk = (n.get("tier1") or {}).get("clock") or {}
        wt = clk.get("wall_time_unix")
        if isinstance(wt, (int, float)):
            times.append((n.get("host", "?"), float(wt)))
        if clk and not clk.get("any_active", True):
            no_daemon_hosts.append((n.get("node_rank", "?"), n.get("host", "?")))

    spread_sec = None
    earliest_h = latest_h = None
    if len(times) >= 2:
        earliest_h, earliest = min(times, key=lambda x: x[1])
        latest_h, latest = max(times, key=lambda x: x[1])
        spread_sec = round(latest - earliest, 3)
    return {
        "n_nodes_with_time": len(times),
        "spread_sec": spread_sec,
        "spread_warn_sec": skew_warn_sec,
        "spread_warn": (spread_sec is not None and spread_sec > skew_warn_sec),
        "earliest_host": earliest_h,
        "latest_host": latest_h,
        "no_daemon_hosts": no_daemon_hosts,
    }


def _tooling_latency_rows(
    nodes: List[Dict[str, Any]],
    warn_sec: float,
) -> List[Dict[str, Any]]:
    """Per-node `rocm-smi --version` self-latency outliers (timed-out + slow)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        t = (n.get("tier1") or {}).get("tooling") or {}
        lat = t.get("latency_sec")
        timed_out = bool(t.get("timed_out"))
        if t.get("error") and lat is None:
            # Tool missing -- not interesting for a slow-tool report.
            continue
        flag = ""
        if timed_out:
            flag = "TIMEOUT"
        elif isinstance(lat, (int, float)) and lat > warn_sec:
            flag = f">{warn_sec}s"
        if not flag:
            continue
        rows.append(
            {
                "node_rank": n.get("node_rank", "?"),
                "host": n.get("host", "?"),
                "latency_sec": lat,
                "flag": flag,
                "timeout_sec": t.get("timeout_sec"),
            }
        )
    return rows


def _tooling_inventory_rows(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-node tooling-presence rows + a count of nodes missing each tool.

    Returns::

        {
            "rows": [
                {"node_rank": 0, "host": "tus1-p3-g25",
                 "amd-smi": True, "rocm-smi": True, "lsof": True},
                ...
            ],
            "missing_counts": {"amd-smi": 0, "rocm-smi": 1, "lsof": 0},
            "any_missing": True,
            "tracked": ["amd-smi", "rocm-smi", "lsof"],
        }

    Always-on: even when every tool is present everywhere, we still emit
    a (small, reassuring) summary so the operator can see at a glance
    that the toolchain was healthy on every node.
    """
    tracked = list(_TRACKED_TOOLS)
    rows: List[Dict[str, Any]] = []
    missing_counts: Dict[str, int] = {t: 0 for t in tracked}
    for n in nodes:
        inv = (n.get("tier1") or {}).get("tooling_inventory") or {}
        tools = inv.get("tools") or {}
        row: Dict[str, Any] = {
            "node_rank": n.get("node_rank", "?"),
            "host": n.get("host", "?"),
        }
        for t in tracked:
            present = bool((tools.get(t) or {}).get("present"))
            row[t] = present
            if not present:
                missing_counts[t] += 1
        rows.append(row)
    return {
        "rows": rows,
        "missing_counts": missing_counts,
        "any_missing": any(c > 0 for c in missing_counts.values()),
        "tracked": tracked,
    }


def _busy_gpu_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-node foreign GPU process listing (for the "Busy GPUs" section)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        gp = (n.get("tier1") or {}).get("gpu_processes") or {}
        if not gp.get("ok"):
            continue
        per_gpu = gp.get("per_gpu") or []
        for g in per_gpu:
            for p in g.get("processes") or []:
                if not p.get("is_foreign"):
                    continue
                hbm_b = p.get("hbm_bytes")
                rows.append(
                    {
                        "node_rank": n.get("node_rank", "?"),
                        "host": n.get("host", "?"),
                        "gpu": g.get("gpu", "?"),
                        "pid": p.get("pid"),
                        "name": p.get("name") or "",
                        "hbm_gib": (
                            round(hbm_b / (1 << 30), 2) if isinstance(hbm_b, int) and hbm_b > 0 else None
                        ),
                    }
                )
    return rows


def _pretouch_hbm_rows(
    nodes: List[Dict[str, Any]],
    threshold_gib: float,
) -> List[Dict[str, Any]]:
    """Per-GPU pre-touch HBM-used outliers (above threshold)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        per_gpu = (n.get("tier1") or {}).get("per_gpu") or []
        for p in per_gpu:
            d = p.get("details") or {}
            used_gib = d.get("hbm_pre_touch_used_gib")
            if not isinstance(used_gib, (int, float)):
                continue
            if used_gib >= threshold_gib:
                rows.append(
                    {
                        "node_rank": n.get("node_rank", "?"),
                        "host": n.get("host", "?"),
                        "gpu": p.get("gpu", "?"),
                        "used_gib": round(float(used_gib), 2),
                    }
                )
    return rows


def _gpu_activity_rows(
    nodes: List[Dict[str, Any]],
    warn_pct: float,
) -> List[Dict[str, Any]]:
    """Per-GPU compute-activity outliers (above warn threshold)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        amd = (n.get("tier1") or {}).get("gpu_low_level") or {}
        for rec in amd.get("per_gpu") or []:
            pct = rec.get("gfx_activity_pct")
            if not isinstance(pct, (int, float)):
                continue
            if float(pct) >= warn_pct:
                rows.append(
                    {
                        "node_rank": n.get("node_rank", "?"),
                        "host": n.get("host", "?"),
                        "gpu": rec.get("gpu", "?"),
                        "activity_pct": round(float(pct), 1),
                    }
                )
    return rows
