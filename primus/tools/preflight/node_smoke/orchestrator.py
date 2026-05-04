###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-node orchestration helpers.

* :func:`_spawn_per_gpu`    -- launch the ``_per_gpu`` subcommand for one
                               GPU index with a hard timeout.
* :func:`_node_status_from` -- compute the list of fail reasons for the
                               whole node from collected per-GPU + tier1
                               + tier2 state.
* :func:`_clean_dump_path`  -- wipe stale artifacts from a previous run on
                               rank 0 before any rank writes its current
                               JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from .types import GPUResult


def _spawn_per_gpu(
    gpu: int,
    *,
    timeout_sec: int,
    tier2_perf: bool,
    gemm_tflops_min: float,
    hbm_gbs_min: float,
    hbm_busy_threshold_gib: float,
) -> GPUResult:
    """Spawn ``python -m primus.tools.preflight.node_smoke _per_gpu <gpu> ...``
    with a hard timeout so a stuck driver call cannot wedge the parent."""
    cmd = [
        sys.executable,
        "-m",
        "primus.tools.preflight.node_smoke",
        "_per_gpu",
        str(gpu),
        "--gemm-tflops-min",
        str(gemm_tflops_min),
        "--hbm-gbs-min",
        str(hbm_gbs_min),
        "--hbm-busy-threshold-gib",
        str(hbm_busy_threshold_gib),
    ]
    if tier2_perf:
        cmd.append("--tier2-perf")

    t0 = time.time()
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GPUResult(
            gpu=gpu,
            status="TIMEOUT",
            reason=f"per-gpu subprocess hit hard timeout {timeout_sec}s",
            duration_sec=round(time.time() - t0, 3),
        )
    except Exception as e:
        return GPUResult(
            gpu=gpu,
            status="FAIL",
            reason=f"failed to spawn per-gpu subprocess: {e}",
            duration_sec=round(time.time() - t0, 3),
        )

    # The subprocess prints exactly one JSON line on stdout for the result.
    raw = (cp.stdout or "").strip().splitlines()
    if not raw:
        return GPUResult(
            gpu=gpu,
            status="FAIL",
            reason=(
                f"per-gpu subprocess produced no JSON (rc={cp.returncode}, "
                f"stderr={cp.stderr.strip()[:200]})"
            ),
            duration_sec=round(time.time() - t0, 3),
        )
    try:
        data = json.loads(raw[-1])
    except Exception as e:
        return GPUResult(
            gpu=gpu,
            status="FAIL",
            reason=f"per-gpu JSON parse failed: {e}; raw={raw[-1][:200]}",
            duration_sec=round(time.time() - t0, 3),
        )

    return GPUResult(
        gpu=int(data.get("gpu", gpu)),
        status=str(data.get("status", "FAIL")),
        reason=str(data.get("reason", "")),
        duration_sec=float(data.get("duration_sec", time.time() - t0)),
        details=dict(data.get("details", {})),
    )


def _node_status_from(
    per_gpu: List[GPUResult],
    tier1_extra: Dict[str, Any],
    tier2_extra: Dict[str, Any],
    *,
    allow_foreign_procs: bool = False,
    required_tools: Optional[List[str]] = None,
) -> List[str]:
    """Compute a list of ``fail_reasons`` for the node from collected results.

    Empty list -> node PASS. Any non-empty result -> node FAIL.

    ``allow_foreign_procs`` downgrades the foreign-process FAIL to a
    silent inclusion in the JSON (still surfaced by the aggregator's
    "Busy GPUs" section, just not a hard fail).

    ``required_tools`` is the operator-supplied list of CLI tools that
    MUST be present on this node (e.g. ``["amd-smi", "rocm-smi"]``).
    Anything in this list that is not in the tooling-inventory becomes
    a hard FAIL. Empty / None means "warn only" (the default).
    """
    reasons: List[str] = []

    # Self-contained GPU visibility guard. Decoupled from any other
    # collector so a wrapped/downgraded "No GPUs detected" finding can
    # never silently turn a CPU-only or stale-GPU node into a PASS.
    vis = tier1_extra.get("gpu_visibility") or {}
    for r in vis.get("fail_reasons", []) or []:
        reasons.append(f"gpu_visibility: {r}")

    for r in per_gpu:
        if r.status != "PASS":
            reasons.append(f"gpu{r.gpu}: {r.status}: {r.reason}")

    for section_name in ("gpu_info", "host_info", "network_info"):
        for f in tier1_extra.get(section_name, []):
            if f.get("level") == "fail":
                reasons.append(f"{section_name}: {f.get('message', '<no message>')}")

    dmesg = tier1_extra.get("dmesg") or {}
    if dmesg.get("matches"):
        first = dmesg["matches"][0]
        reasons.append(
            f"dmesg ({len(dmesg['matches'])} match(es), e.g.): {first[:200]}"
        )

    # B. NIC / RDMA roll-call -- every issue here is a hard fail because each
    # one (port DOWN, missing RoCE v2 GID, wrong NIC count) silently breaks
    # inter-node training the moment the first global collective runs.
    for issue in (tier1_extra.get("nics") or {}).get("issues", []) or []:
        reasons.append(f"nic: {issue}")

    # C. Host limits -- only the entries the collector flagged as hard
    # (ulimit -l below threshold, /dev/shm too small) become node FAIL.
    for issue in (tier1_extra.get("host_limits") or {}).get("fail_reasons", []) or []:
        reasons.append(f"host_limits: {issue}")

    rccl = tier2_extra.get("rccl") or {}
    if rccl and rccl.get("status") not in (None, "PASS"):
        reasons.append(f"rccl: {rccl.get('status')}: {rccl.get('error', '')}")

    # D-1 heavy: any per-GPU uncorrectable ECC count is a hard fail. The
    # amd-smi schema isn't stable across releases so we trust only the
    # values our flattener was able to coerce to int. Throttle reasons stay
    # informational (the schema is too vendor-specific to fail on).
    amd = tier1_extra.get("gpu_low_level") or {}
    for rec in amd.get("per_gpu", []) or []:
        ue = rec.get("ecc_uncorrectable_total")
        if isinstance(ue, int) and ue > 0:
            reasons.append(
                f"gpu{rec.get('gpu', '?')}: ECC uncorrectable count = {ue}"
            )

    # D-2: any non-XGMI GPU pair is a hard fail -- intra-node collectives
    # silently fall back to PCIe and lose 5-10x bandwidth.
    xg = tier1_extra.get("xgmi") or {}
    bad = xg.get("non_xgmi_pairs") or []
    if bad:
        sample = ", ".join(f"({i},{j})={t}" for i, j, t in bad[:3])
        reasons.append(
            f"xgmi: {len(bad)} non-XGMI GPU pair(s) detected, e.g. {sample}"
        )

    # F-partial: rocm-smi --version that timed out -> driver is wedging.
    # Slow-but-completed calls are surfaced by the aggregator only.
    tool = tier1_extra.get("tooling") or {}
    if tool.get("timed_out"):
        reasons.append(
            f"tooling: rocm-smi --version did not return within "
            f"{tool.get('timeout_sec', '?')}s -- driver may be wedging"
        )

    # Tooling availability. Missing CLI tools (amd-smi, rocm-smi, lsof)
    # cause silent skips of several Tier 1 checks. Operators in strict
    # environments can pass --require-tools to convert "missing" into a
    # node FAIL so the node is pulled from rotation until the toolchain
    # is fixed. Default (empty list) is warn-only (the WARN already fires
    # in _cmd_run before the per-GPU subprocesses run).
    if required_tools:
        inv = (tier1_extra.get("tooling_inventory") or {}).get("tools") or {}
        missing_required = [
            t for t in required_tools
            if not (inv.get(t) or {}).get("present")
        ]
        if missing_required:
            reasons.append(
                f"tooling_inventory: required tool(s) NOT in PATH: "
                f"{', '.join(missing_required)} -- silent-skips several "
                f"Tier 1 checks (ECC, XGMI, foreign-process, wedged-driver). "
                f"Pass --require-tools '' to disable this fail."
            )

    # G: foreign processes holding the GPU. Hard-fail by default because
    # this is the single most common cause of training failing to launch
    # on an otherwise-healthy node (leaked Python ranks from a previous
    # job, a profiler that never detached, a foreign tenant on a shared
    # partition). The operator can downgrade with --allow-foreign-procs
    # if their workflow legitimately co-tenants the GPU.
    gp = tier1_extra.get("gpu_processes") or {}
    if not allow_foreign_procs and gp.get("foreign_count", 0) > 0:
        examples: List[str] = []
        for g in gp.get("per_gpu") or []:
            for p in g.get("processes") or []:
                if not p.get("is_foreign"):
                    continue
                hbm = p.get("hbm_bytes")
                hbm_s = (
                    f" hbm={round(hbm / (1 << 30), 2)}GiB"
                    if isinstance(hbm, int) and hbm > 0
                    else ""
                )
                examples.append(
                    f"gpu{g.get('gpu')}: pid={p.get('pid')} "
                    f"name={p.get('name')!r}{hbm_s}"
                )
                if len(examples) >= 3:
                    break
            if len(examples) >= 3:
                break
        reasons.append(
            f"gpu_processes: {gp.get('foreign_count', 0)} foreign process(es) "
            f"holding GPU(s) (e.g. " + "; ".join(examples) + ") -- likely "
            f"leaked rank(s) from a previous job. Clean up with "
            f"`pkill -9 -f train.py` (or similar) or pass --allow-foreign-procs."
        )

    return reasons


def _clean_dump_path(dump_path: str) -> List[str]:
    """Wipe stale per-node JSONs and aggregator outputs from a previous run.

    Without this, a re-run on a different (smaller) nodelist would leave
    JSONs from removed nodes in ``<dump>/smoke/`` and the aggregator would
    happily count them as PASS, contaminating the report. We clean only
    artifacts that ``run`` and ``aggregate`` produce (per-node JSONs and
    the four top-level outputs); anything else under ``--dump-path`` is
    left untouched.

    Race safety: this is called only on rank 0 in ``_cmd_run``, BEFORE any
    rank can have written its current-run JSON (each rank's per-GPU
    subprocess loop + collector phase takes seconds; rank 0's cleanup
    finishes in milliseconds). Other ranks never delete anything.

    Returns the list of files actually removed (for logging).
    """
    removed: List[str] = []
    smoke_dir = os.path.join(dump_path, "smoke")
    if os.path.isdir(smoke_dir):
        for name in os.listdir(smoke_dir):
            if name.endswith(".json"):
                p = os.path.join(smoke_dir, name)
                try:
                    os.remove(p)
                    removed.append(p)
                except OSError:
                    pass
    for name in (
        "smoke_report.md",
        "passing_nodes.txt",
        "failing_nodes.txt",
        "expected_nodes.txt",
    ):
        p = os.path.join(dump_path, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed.append(p)
            except OSError:
                pass
    return removed
