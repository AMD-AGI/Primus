###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI wiring + the three subcommand entry points (`run`, `aggregate`,
`_per_gpu`).

The argparse layout is part of the operator-facing contract: flag names,
defaults, help strings, and abbreviation behaviour are preserved exactly.
``allow_abbrev=False`` on the ``run`` subparser keeps an old ``--tier2``
flag from a stale script silently matching the new ``--tier2-perf``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .aggregator.report import write_smoke_report
from .collectors.clock import _collect_clock_state
from .collectors.dmesg import _collect_dmesg_errors
from .collectors.fingerprint import _collect_node_fingerprint
from .collectors.gpu_low_level import _collect_amd_smi_metrics
from .collectors.gpu_processes import _collect_gpu_processes
from .collectors.host_limits import _collect_host_limits
from .collectors.nics import _collect_nic_status
from .collectors.reused_info import _collect_reused_info
from .collectors.rocm_smi import _collect_rocm_smi_self_latency
from .collectors.tooling import _collect_tooling_inventory
from .collectors.xgmi import _collect_xgmi_topology
from .logging_utils import _log, _short_name, _this_host_short, _warn
from .orchestrator import _clean_dump_path, _node_status_from, _spawn_per_gpu
from .per_gpu import _per_gpu_body
from .rccl_local import _run_local_rccl
from .types import GPUResult, NodeResult


def _cmd_per_gpu(ns: argparse.Namespace) -> int:
    """Internal subcommand: run all per-GPU checks for a single GPU index."""
    result = _per_gpu_body(
        gpu=int(ns.gpu),
        tier2_perf=bool(ns.tier2_perf),
        gemm_tflops_min=float(ns.gemm_tflops_min),
        hbm_gbs_min=float(ns.hbm_gbs_min),
        hbm_busy_threshold_bytes=int(float(ns.hbm_busy_threshold_gib) * (1 << 30)),
    )
    # Single JSON line on stdout; nothing else.
    print(json.dumps(result), flush=True)
    return 0 if result.get("status") == "PASS" else 1


def _cmd_run(ns: argparse.Namespace) -> int:
    """Per-node entry: orchestrate Tier 1 + optional Tier 2 + write JSON."""
    # Always store the short hostname so consumers (passing_nodes.txt /
    # failing_nodes.txt and SLURM tools that read them) get a name they
    # can use directly.
    host = _this_host_short()
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", "0")))

    # Rank-0 only: wipe stale JSONs / aggregator outputs from a previous
    # run so re-runs on a different (smaller) nodelist cannot inherit
    # ghost PASS verdicts from removed nodes. This MUST run before any
    # rank's _spawn_per_gpu loop completes (and thus before any rank
    # writes its JSON) -- which is why it lives at the top of _cmd_run
    # on rank 0 only, not in the wrapper.
    if node_rank == 0 and not ns.no_clean_dump_path:
        removed = _clean_dump_path(ns.dump_path)
        if removed:
            _log(
                f"cleaned {len(removed)} stale file(s) from {ns.dump_path} "
                f"(per-node JSONs + aggregator outputs)"
            )

    expected_gpus = ns.expected_gpus
    if expected_gpus is None:
        expected_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("GPUS_PER_NODE", "0")) or 0)
        if expected_gpus <= 0:
            try:
                import torch  # type: ignore

                expected_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except Exception:
                expected_gpus = 0
    expected_gpus = max(0, int(expected_gpus))

    # GPU visibility guard. We capture each independent source (the
    # --expected-gpus flag, env vars, torch, and -- below -- amd-smi) so
    # the JSON tells the operator *why* we resolved to N. The hard-fail
    # rules live here, decoupled from any other collector, because we have
    # seen `_collect_reused_info()` downgrade the "No GPUs detected" fail
    # to a warn when collect_gpu_info() raises -- which would otherwise
    # let a CPU-only or stale-GPU node PASS smoke silently.
    torch_visible = 0
    torch_is_available = False
    try:
        import torch  # type: ignore

        torch_is_available = bool(torch.cuda.is_available())
        torch_visible = int(torch.cuda.device_count()) if torch_is_available else 0
    except Exception:
        pass
    gpu_visibility: Dict[str, Any] = {
        "expected_gpus": expected_gpus,
        "explicit_expected_gpus": ns.expected_gpus,
        "torch_visible": torch_visible,
        "torch_is_available": torch_is_available,
        "env_local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0),
        "env_gpus_per_node": int(os.environ.get("GPUS_PER_NODE", "0") or 0),
        "amd_smi_visible": None,  # filled in after _collect_amd_smi_metrics
        "fail_reasons": [],
    }
    if expected_gpus < 1:
        msg = (
            f"expected_gpus={expected_gpus}: no per-GPU sanity tests will "
            f"run (torch_is_available={torch_is_available}, "
            f"torch_visible={torch_visible}, "
            f"LOCAL_WORLD_SIZE={gpu_visibility['env_local_world_size']}, "
            f"GPUS_PER_NODE={gpu_visibility['env_gpus_per_node']})"
        )
        gpu_visibility["fail_reasons"].append(msg)
        _warn(msg)

    _log(
        f"start node-smoke: node_rank={node_rank} expected_gpus={expected_gpus} "
        f"tier2_perf={ns.tier2_perf}"
    )
    if ns.tier2_perf and expected_gpus < 2:
        # Tier 2 also includes a node-local RCCL all-reduce, which needs at
        # least 2 GPUs. Surface the skip up front instead of silently doing
        # only GEMM/HBM and giving the operator false coverage confidence.
        _warn(
            f"--tier2-perf requested but expected_gpus={expected_gpus} < 2; "
            "the node-local RCCL all-reduce phase will be skipped. "
            "Per-GPU GEMM and HBM checks will still run."
        )

    # G: enumerate processes currently holding each GPU BEFORE we spawn
    # any per-GPU subprocess. Anything we see here is, by definition, not
    # us -- it's a leaked rank from a previous job, a foreign tenant, or
    # an in-band monitoring agent. The aggregator + _node_status_from
    # turn this into a hard FAIL unless the operator opted out via
    # --allow-foreign-procs (or whitelisted the agent name).
    allowed_proc_names = [s for s in (getattr(ns, "allowed_procs", "") or "").split(",") if s.strip()]
    tier1_extra_pre: Dict[str, Any] = {}
    tier1_extra_pre["gpu_processes"] = _collect_gpu_processes(
        self_pid=os.getpid(),
        allowed_proc_names=allowed_proc_names,
    )
    gp = tier1_extra_pre["gpu_processes"]
    if gp.get("ok"):
        _log(
            f"gpu_processes ({gp.get('tool')}): "
            f"{gp.get('foreign_count', 0)} foreign PID(s) across "
            f"{len(gp.get('per_gpu') or [])} GPU bucket(s)"
        )
        if gp.get("foreign_count", 0) > 0:
            for g in gp.get("per_gpu") or []:
                for p in g.get("processes") or []:
                    if p.get("is_foreign"):
                        hbm = p.get("hbm_bytes")
                        hbm_s = (
                            f"{round(hbm / (1 << 30), 2)} GiB" if isinstance(hbm, int) and hbm > 0 else "?"
                        )
                        _warn(
                            f"foreign process on gpu{g.get('gpu')}: "
                            f"pid={p.get('pid')} name={p.get('name')!r} "
                            f"hbm={hbm_s}"
                        )
    else:
        _warn(
            f"gpu_processes: enumeration unavailable "
            f"({gp.get('error') or gp.get('json_error') or gp.get('text_error') or '?'})"
        )

    t0 = time.time()
    per_gpu: List[GPUResult] = []
    for i in range(expected_gpus):
        r = _spawn_per_gpu(
            i,
            timeout_sec=ns.per_gpu_timeout_sec,
            tier2_perf=bool(ns.tier2_perf),
            gemm_tflops_min=ns.gemm_tflops_min,
            hbm_gbs_min=ns.hbm_gbs_min,
            hbm_busy_threshold_gib=float(ns.hbm_busy_threshold_gib),
        )
        per_gpu.append(r)
        _log(
            f"gpu{i}: {r.status} ({r.duration_sec:.1f}s)"
            + (f" -- {r.reason}" if r.reason else "")
            + (f" -- {r.details}" if r.details else "")
        )

    # Tier 1 reused info collectors
    tier1_extra: Dict[str, Any] = {}
    tier1_extra.update(tier1_extra_pre)  # carry forward gpu_processes
    tier1_extra.update(_collect_reused_info())
    if not ns.skip_dmesg:
        tier1_extra["dmesg"] = _collect_dmesg_errors(window_minutes=ns.dmesg_minutes)
    else:
        tier1_extra["dmesg"] = {"ok": True, "matches": [], "error": "skipped"}

    # A/B/C: software-stack fingerprint, NIC roll-call, host limits.
    # All three are pure data-collection (millisecond-scale sysfs reads); the
    # heavy cluster-level drift detection happens at aggregation time.
    tier1_extra["fingerprint"] = _collect_node_fingerprint()
    tier1_extra["nics"] = _collect_nic_status(expected_count=ns.expected_rdma_nics)
    tier1_extra["host_limits"] = _collect_host_limits(
        ulimit_l_min_gb=ns.ulimit_l_min_gb,
        shm_min_gb=ns.shm_min_gb,
    )

    # D-1 heavy: per-GPU ECC / throttle / clocks / power via amd-smi (one
    # node-level call, results indexed by gpu).
    # D-2: XGMI link matrix via amd-smi topology (one node-level call).
    # E:   wall-time + time-daemon active states.
    # F-partial: rocm-smi --version self-latency with a hard timeout to
    # catch drivers that are starting to wedge.
    # Tooling inventory FIRST so we can warn loudly before running the
    # collectors that depend on each tool. Several downstream collectors
    # (gpu_low_level, xgmi, gpu_processes, tooling) silently no-op when
    # their tool is missing, which can let a broken node pass smoke
    # unnoticed -- this section is the operator-visible counterweight.
    tier1_extra["tooling_inventory"] = _collect_tooling_inventory()
    inv = tier1_extra["tooling_inventory"]
    inv_missing = inv["missing"]
    inv_uncovered = inv["uncovered"]
    if inv_missing:
        # First line: which tools are missing. Loud regardless of whether
        # a fallback covers everything.
        _warn(f"tooling: {len(inv_missing)} tracked tool(s) NOT in PATH: " f"{', '.join(inv_missing)}.")
        if inv_uncovered:
            # Second line: which checks have NO working tool at all. This
            # is the actually-dangerous case (a check that will silently
            # no-op no matter which tool we try).
            _warn(
                f"tooling: {len(inv_uncovered)} check(s) have NO working "
                f"tool and will be silently skipped: "
                + ", ".join(inv_uncovered)
                + ". Use --require-tools to promote missing tools to a "
                "node FAIL."
            )
        else:
            # Reassuring line: every check is covered via fallback.
            _warn(
                "tooling: every check is still covered via fallback "
                "(rocm-smi / lsof) -- no checks will be silently skipped. "
                "Use --require-tools to promote missing tools to a node "
                "FAIL anyway if your environment requires them."
            )

    tier1_extra["gpu_low_level"] = _collect_amd_smi_metrics()
    tier1_extra["xgmi"] = _collect_xgmi_topology()
    tier1_extra["clock"] = _collect_clock_state()
    tier1_extra["tooling"] = _collect_rocm_smi_self_latency(timeout_sec=float(ns.rocm_smi_timeout_sec))

    # Visibility cross-check: if amd-smi successfully enumerated GPUs but
    # torch couldn't see them, that's a high-signal sign of a stale ROCm
    # install / wedged amdgpu driver -- exactly the case where a "smoke
    # test" is supposed to pull the node out of rotation. We only treat
    # the JSON path as authoritative for counting (the text fallback
    # cannot be reliably parsed for a count).
    amd_low = tier1_extra["gpu_low_level"]
    if amd_low.get("ok") and amd_low.get("tool") == "amd-smi metric --json":
        per = amd_low.get("per_gpu") or []
        n_amd = len(per) if isinstance(per, list) else 0
        gpu_visibility["amd_smi_visible"] = n_amd
        if n_amd > 0 and torch_visible < n_amd:
            mismatch = (
                f"gpu_visibility_mismatch: amd-smi sees {n_amd} GPU(s) "
                f"but torch.cuda.device_count()={torch_visible} "
                f"(torch_is_available={torch_is_available}); ROCm install "
                f"or amdgpu driver may be broken on this node"
            )
            gpu_visibility["fail_reasons"].append(mismatch)
            _warn(mismatch)
    tier1_extra["gpu_visibility"] = gpu_visibility
    xg = tier1_extra["xgmi"]
    if xg.get("ok"):
        bad = xg.get("non_xgmi_pairs") or []
        _log(f"xgmi: {xg.get('n_gpus', 0)}x{xg.get('n_gpus', 0)} matrix, " f"{len(bad)} non-XGMI pair(s)")
    elif xg.get("error"):
        _warn(f"xgmi: {xg.get('error')}")
    tool = tier1_extra["tooling"]
    if tool.get("ok"):
        _log(f"rocm-smi --version: {tool.get('latency_sec')}s")
    elif tool.get("timed_out"):
        _warn(f"rocm-smi --version timed out after {tool.get('timeout_sec')}s " "-- driver may be wedging")
    elif tool.get("error"):
        _warn(f"tooling: {tool.get('error')}")
    nic_summary = tier1_extra["nics"]
    _log(
        f"nics: {len(nic_summary.get('ports', []))} port(s) found, "
        f"{len(nic_summary.get('issues', []))} issue(s)"
    )
    if tier1_extra["host_limits"].get("fail_reasons"):
        for r in tier1_extra["host_limits"]["fail_reasons"]:
            _warn(f"host_limits: {r}")

    # Tier 2 local RCCL all-reduce. Gated on a single flag now (--tier2-perf)
    # so users cannot accidentally end up with only the per-GPU half running.
    tier2_extra: Dict[str, Any] = {}
    if ns.tier2_perf and expected_gpus > 1:
        _log(f"tier2 local RCCL all-reduce: {expected_gpus} ranks, {ns.rccl_size_mb}MB")
        rccl = _run_local_rccl(
            local_world_size=expected_gpus,
            size_mb=ns.rccl_size_mb,
            timeout_sec=ns.rccl_timeout_sec,
        )
        if rccl.get("status") == "PASS" and rccl.get("gbs") is not None:
            if float(rccl["gbs"]) < ns.rccl_gbs_min:
                rccl = {
                    "status": "FAIL",
                    "gbs": rccl["gbs"],
                    "error": (f"local RCCL {rccl['gbs']} GB/s < threshold {ns.rccl_gbs_min}"),
                }
        tier2_extra["rccl"] = rccl
        _log(f"tier2 RCCL: {rccl}")

    required_tools = [s.strip() for s in (getattr(ns, "require_tools", "") or "").split(",") if s.strip()]
    fail_reasons = _node_status_from(
        per_gpu,
        tier1_extra,
        tier2_extra,
        allow_foreign_procs=bool(ns.allow_foreign_procs),
        required_tools=required_tools,
    )
    status = "PASS" if not fail_reasons else "FAIL"

    node_result = NodeResult(
        host=host,
        node_rank=node_rank,
        status=status,
        duration_sec=round(time.time() - t0, 3),
        fail_reasons=fail_reasons,
        tier1={
            "per_gpu": [asdict(r) for r in per_gpu],
            **tier1_extra,
        },
        tier2=tier2_extra,
    )

    smoke_dir = os.path.join(ns.dump_path, "smoke")
    os.makedirs(smoke_dir, exist_ok=True)
    out_path = os.path.join(smoke_dir, f"{host}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(node_result), f, indent=2, default=str)
    _log(f"wrote {out_path} status={status} duration={node_result.duration_sec}s")
    if fail_reasons:
        for r in fail_reasons[:5]:
            _warn(r)

    # Per-node `run` exits 0 whenever the smoke test ran to completion --
    # the node verdict (PASS/FAIL) is in the JSON, in failing_nodes.txt,
    # and in the aggregator's exit code. Conflating "this node is broken"
    # with "this tool crashed" makes srun output look like the smoke test
    # itself is failing, when in fact it's correctly DOING ITS JOB of
    # identifying broken nodes. The aggregator (rank 0) is the single
    # source of truth for the CI-friendly cluster-health exit signal.
    # Tool failures (couldn't import, couldn't write JSON, etc.) still
    # propagate as non-zero via Python's default exception handling.
    return 0


def _cmd_aggregate(ns: argparse.Namespace) -> int:
    """Read all per-node JSONs from ``<dump>/smoke/`` and emit summary outputs."""
    smoke_dir = os.path.join(ns.dump_path, "smoke")
    os.makedirs(smoke_dir, exist_ok=True)

    expected = int(ns.expected_nodes) if ns.expected_nodes is not None else None
    deadline = time.time() + max(0, int(ns.wait_timeout_sec))
    found_paths: List[str] = []
    while True:
        found_paths = sorted(os.path.join(smoke_dir, p) for p in os.listdir(smoke_dir) if p.endswith(".json"))
        if expected is None or len(found_paths) >= expected:
            break
        if time.time() >= deadline:
            break
        time.sleep(1)

    nodes: List[Dict[str, Any]] = []
    for p in found_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                nodes.append(json.load(f))
        except Exception as e:
            nodes.append(
                {
                    "host": os.path.basename(p).rsplit(".json", 1)[0],
                    "status": "FAIL",
                    "fail_reasons": [f"failed to parse {p}: {e}"],
                    "duration_sec": 0,
                    "node_rank": -1,
                }
            )

    # Normalize every loaded ``host`` to its short form so legacy JSON files
    # that hold an FQDN (older runs of node_smoke) still produce SLURM-ready
    # passing/failing lists.
    for n in nodes:
        n["host"] = _short_name(str(n.get("host", "")))

    # Optional: an explicit expected hostname list (one per line). When
    # provided, we name missing nodes by their real short hostname instead
    # of synthetic ``<missing-N>`` placeholders, so the failing nodes list
    # is directly usable with ``srun --exclude=``.
    expected_hosts_short: List[str] = []
    nodelist_file = getattr(ns, "expected_nodelist_file", None)
    if nodelist_file:
        try:
            with open(nodelist_file, "r", encoding="utf-8") as f:
                expected_hosts_short = [_short_name(line.strip()) for line in f if line.strip()]
            _log(f"loaded {len(expected_hosts_short)} expected hostnames from " f"{nodelist_file}")
        except Exception as e:
            _warn(f"failed to read --expected-nodelist-file {nodelist_file}: {e}")

    seen_hosts_short = {n.get("host", "") for n in nodes}

    if expected_hosts_short:
        # An explicit list always wins over --expected-nodes for both the
        # count and (more importantly) the identity of missing nodes.
        if expected is None or expected != len(expected_hosts_short):
            expected = len(expected_hosts_short)
        missing_hosts = sorted(set(expected_hosts_short) - seen_hosts_short)
        for h in missing_hosts:
            nodes.append(
                {
                    "host": h,
                    "status": "FAIL",
                    "fail_reasons": [
                        f"no JSON received within {ns.wait_timeout_sec}s "
                        f"(expected hostname '{h}' from --expected-nodelist-file)"
                    ],
                    "duration_sec": 0,
                    "node_rank": -1,
                }
            )
    elif expected is not None and len(seen_hosts_short) < expected:
        # Fallback: we know the count but not the identities -> emit
        # synthetic placeholders. These intentionally do NOT land in
        # passing/failing txt files (see _is_real_host below).
        for i in range(expected - len(seen_hosts_short)):
            nodes.append(
                {
                    "host": f"<missing-{i}>",
                    "status": "FAIL",
                    "fail_reasons": [
                        f"no JSON received within {ns.wait_timeout_sec}s "
                        f"(expected_nodes={expected}, "
                        f"found={len(seen_hosts_short)})"
                    ],
                    "duration_sec": 0,
                    "node_rank": -1,
                }
            )

    # Sort by node_rank if present, otherwise by hostname.
    def _key(n: Dict[str, Any]):
        nr = n.get("node_rank", 0)
        return (
            int(nr) if isinstance(nr, (int, str)) and str(nr).lstrip("-").isdigit() else 1 << 30,
            str(n.get("host", "")),
        )

    nodes.sort(key=_key)

    passing = [n for n in nodes if n.get("status") == "PASS"]
    failing = [n for n in nodes if n.get("status") != "PASS"]

    report_path = os.path.join(ns.dump_path, "smoke_report.md")
    pass_path = os.path.join(ns.dump_path, "passing_nodes.txt")
    fail_path = os.path.join(ns.dump_path, "failing_nodes.txt")

    write_smoke_report(
        report_path,
        nodes=nodes,
        passing=passing,
        failing=failing,
        expected=expected,
        clock_skew_warn_sec=float(ns.clock_skew_warn_sec),
        rocm_smi_warn_sec=float(ns.rocm_smi_warn_sec),
        hbm_busy_threshold_gib=float(getattr(ns, "hbm_busy_threshold_gib", 2.0)),
        gpu_activity_warn_pct=float(getattr(ns, "gpu_activity_warn_pct", 20.0)),
    )

    # Only write REAL hostnames to the txt files so they can be piped directly
    # into `srun --nodelist=` / `srun --exclude=`. Synthetic "<missing-N>"
    # placeholders for nodes that never reported are surfaced in the markdown
    # report instead.
    def _is_real_host(h: str) -> bool:
        return bool(h) and not (h.startswith("<missing-") and h.endswith(">"))

    with open(pass_path, "w", encoding="utf-8") as f:
        for n in passing:
            h = str(n.get("host", ""))
            if _is_real_host(h):
                f.write(h + "\n")
    with open(fail_path, "w", encoding="utf-8") as f:
        for n in failing:
            h = str(n.get("host", ""))
            if _is_real_host(h):
                f.write(h + "\n")

    _log(
        f"aggregate: {len(passing)}/{len(nodes)} PASS  "
        f"report={report_path}  passing={pass_path}  failing={fail_path}"
    )
    return 0 if not failing and (expected is None or len(nodes) == expected) else 1


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m primus.tools.preflight.node_smoke",
        description=(
            "Node-local preflight smoke test. Each node runs independently "
            "(no global rendezvous) and writes a per-node JSON verdict."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run ----
    # allow_abbrev=False so abbreviated forms (e.g. an old --tier2 left in
    # a script) do NOT silently match the new --tier2-perf as a prefix.
    # We want them to error out loudly so behavior never changes silently
    # underneath an unsuspecting caller.
    pr = sub.add_parser(
        "run",
        help="Run per-node smoke test on this node.",
        allow_abbrev=False,
    )
    pr.add_argument(
        "--dump-path", default="output/preflight", help="Directory under which smoke/<host>.json is written."
    )
    pr.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="Expected GPU count on this node (default: LOCAL_WORLD_SIZE/GPUS_PER_NODE/torch.cuda.device_count()).",
    )
    pr.add_argument(
        "--per-gpu-timeout-sec", type=int, default=15, help="Hard timeout for each per-GPU subprocess."
    )
    pr.add_argument(
        "--tier2-perf",
        action="store_true",
        help="Enable Tier 2 perf sanity: per-GPU GEMM TFLOPS, "
        "HBM bandwidth, AND node-local RCCL all-reduce. "
        "All three are fast (< 30 s/node total).",
    )
    pr.add_argument(
        "--gemm-tflops-min",
        type=float,
        default=600.0,
        help="FAIL if Tier 2 GEMM TFLOPS is below this. Default: 600 (MI300X-class).",
    )
    pr.add_argument(
        "--hbm-gbs-min",
        type=float,
        default=2000.0,
        help="FAIL if Tier 2 HBM GB/s is below this. Default: 2000.",
    )
    pr.add_argument(
        "--rccl-size-mb", type=int, default=64, help="Tensor size for local RCCL all-reduce (MB)."
    )
    pr.add_argument(
        "--rccl-gbs-min",
        type=float,
        default=100.0,
        help="FAIL if local RCCL GB/s is below this. Default: 100.",
    )
    pr.add_argument(
        "--rccl-timeout-sec", type=int, default=30, help="Hard timeout for the local RCCL all-reduce phase."
    )
    pr.add_argument(
        "--skip-dmesg", action="store_true", help="Skip the dmesg recent-error scan (e.g. inside containers)."
    )
    pr.add_argument("--dmesg-minutes", type=int, default=15, help="Window for dmesg --since (minutes).")
    # NIC / RDMA roll-call (B). expected_count=None means "report only";
    # set this to e.g. 8 to make a missing or down NIC port a node FAIL.
    pr.add_argument(
        "--expected-rdma-nics",
        type=int,
        default=None,
        help="Expected RDMA NIC port count. If set, a count " "mismatch becomes a node FAIL.",
    )
    # Host-limits hard thresholds (C). Set to 0 to disable a check.
    pr.add_argument(
        "--ulimit-l-min-gb",
        type=float,
        default=32.0,
        help="FAIL the node if RLIMIT_MEMLOCK is finite and below "
        "this many GiB (RDMA pin will fail). 0 disables.",
    )
    pr.add_argument(
        "--shm-min-gb",
        type=float,
        default=8.0,
        help="FAIL the node if /dev/shm is below this many GiB " "(NCCL shared-mem may fail). 0 disables.",
    )
    # F-partial: rocm-smi self-latency. Hitting this timeout is treated as a
    # hard fail because a wedging amdgpu driver typically makes rocm-smi
    # hang for 30-60 s before the GPU itself stops responding.
    pr.add_argument(
        "--rocm-smi-timeout-sec",
        type=float,
        default=5.0,
        help="Hard timeout for `rocm-smi --version`. Hitting it is " "a node FAIL (driver likely wedging).",
    )
    # G: foreign / leaked process detection. Hard-fail by default; the
    # operator can opt out for partitions that legitimately co-tenant the
    # GPU, or whitelist known in-band agents by name.
    pr.add_argument(
        "--hbm-busy-threshold-gib",
        type=float,
        default=2.0,
        help="FAIL the node if any GPU has more than this much "
        "HBM in use BEFORE we touch the device (i.e. someone "
        "else is holding it). Default: 2.0 GiB.",
    )
    pr.add_argument(
        "--allow-foreign-procs",
        action="store_true",
        help="Do NOT FAIL the node when foreign processes are "
        "found holding a GPU. They will still be reported.",
    )
    pr.add_argument(
        "--allowed-procs",
        type=str,
        default="gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter",
        help="Comma-separated process names that are OK to find "
        "holding the GPU. The default whitelists the AMD "
        "system agents (`gpuagent`, `rocm-smi-daemon`, "
        "`amd-smi`) and a common observability sidecar "
        "(`dcgm-exporter`) so they don't fail every node. "
        "Set to an empty string to disable the whitelist.",
    )
    pr.add_argument(
        "--gpu-activity-warn-pct",
        type=float,
        default=20.0,
        help="Aggregator warns (does NOT fail) if amd-smi reports "
        "any GPU's gfx_activity_pct above this when smoke "
        "starts. Default: 20.",
    )
    # Tooling availability. Default empty = warn-only (the WARN already
    # fires before the per-GPU subprocesses run, and the aggregator
    # always renders a "Tooling availability" section). Strict
    # environments can pass --require-tools amd-smi,rocm-smi to promote
    # a missing tool to a hard node FAIL.
    pr.add_argument(
        "--require-tools",
        type=str,
        default="",
        help="Comma-separated CLI tool names that MUST be "
        "present in PATH for the node to PASS. Anything "
        "missing becomes a hard node FAIL. Tracked tools: "
        "amd-smi, rocm-smi, lsof. Default: warn-only.",
    )
    pr.add_argument(
        "--no-clean-dump-path",
        action="store_true",
        help="Do NOT auto-wipe stale per-node JSONs and aggregator "
        "outputs from --dump-path on rank 0 at startup. "
        "Default behavior is to clean so re-runs on a "
        "different nodelist don't inherit ghost PASS "
        "verdicts from removed nodes.",
    )
    pr.set_defaults(func=_cmd_run)

    # ---- aggregate ----
    pa = sub.add_parser("aggregate", help="Aggregate per-node JSONs into report + passing/failing lists.")
    pa.add_argument("--dump-path", default="output/preflight", help="Same as `run --dump-path`.")
    pa.add_argument(
        "--expected-nodes",
        type=int,
        default=None,
        help="Number of nodes expected to report. Missing nodes are FAIL.",
    )
    pa.add_argument(
        "--wait-timeout-sec",
        type=int,
        default=60,
        help="How long to wait for all expected JSONs to land before aggregating anyway.",
    )
    pa.add_argument(
        "--rocm-smi-warn-sec",
        type=float,
        default=1.0,
        help="Flag (warn-only) any node where `rocm-smi --version` " "took longer than this many seconds.",
    )
    pa.add_argument(
        "--clock-skew-warn-sec",
        type=float,
        default=30.0,
        help="Warn (info-only) when wall-clock spread across nodes "
        "exceeds this many seconds. Includes srun launch "
        "jitter so the default is loose.",
    )
    # Mirror the run-side thresholds so the report can label its sections
    # using the same numbers each node's `run` was configured with.
    pa.add_argument(
        "--hbm-busy-threshold-gib",
        type=float,
        default=2.0,
        help="Pre-touch HBM-used threshold (GiB) used by the " "'GPU pre-touch HBM usage outliers' section.",
    )
    pa.add_argument(
        "--gpu-activity-warn-pct",
        type=float,
        default=20.0,
        help="GPU activity %% threshold used by the " "'GPU compute-activity outliers' section.",
    )
    pa.add_argument(
        "--expected-nodelist-file",
        type=str,
        default=None,
        help="Optional file with one expected (short) hostname per line. "
        "When provided, missing nodes are reported with their real "
        "hostname instead of synthetic <missing-N> placeholders, and "
        "are written to failing_nodes.txt directly. The runner script "
        "auto-populates this from `scontrol show hostnames` under SLURM.",
    )
    pa.set_defaults(func=_cmd_aggregate)

    # ---- _per_gpu (internal) ----
    pg = sub.add_parser(
        "_per_gpu", help="(internal) Run smoke checks for a single GPU index. Spawned by `run`."
    )
    pg.add_argument("gpu", type=int)
    pg.add_argument("--tier2-perf", action="store_true")
    pg.add_argument("--gemm-tflops-min", type=float, default=600.0)
    pg.add_argument("--hbm-gbs-min", type=float, default=2000.0)
    pg.add_argument("--hbm-busy-threshold-gib", type=float, default=2.0)
    pg.set_defaults(func=_cmd_per_gpu)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))
