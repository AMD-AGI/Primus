###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Node-local preflight smoke test.

Goal: quickly identify which nodes have problems before launching a real
training job. Each node tests itself in parallel with no global rendezvous,
writes a per-node JSON verdict, and an aggregator (rank 0) reads all JSONs
and emits PASS/FAIL lists usable by SLURM ``--exclude=`` / ``--nodelist=``.

Subcommands
-----------

* ``run`` -- per-node entry. Runs Tier 1 (per-GPU sanity + reused
  host/gpu/network info collectors + a small dmesg scan) and, when
  ``--tier2`` is set, Tier 2 perf sanity (GEMM TFLOPS, HBM bandwidth, and
  optional local RCCL all-reduce). Writes ``<dump>/smoke/<host>.json``.

* ``aggregate`` -- read all per-node JSONs and emit
  ``<dump>/smoke_report.md``, ``<dump>/passing_nodes.txt``, and
  ``<dump>/failing_nodes.txt``. Exits non-zero if any node FAILs or is
  missing.

* ``_per_gpu`` -- internal subcommand spawned by ``run`` to test a single
  GPU in an isolated subprocess with a hard timeout. Not for direct use.

Why per-GPU subprocesses?
-------------------------

A stuck ``torch.cuda.set_device(i)`` cannot be aborted reliably with
``signal.alarm`` because the call may be inside a non-interruptible driver
syscall. By running each per-GPU test in its own subprocess we can SIGKILL
it on timeout without affecting the rest of the node's checks.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


@dataclass
class GPUResult:
    """Result of all checks for a single GPU on this node."""

    gpu: int
    status: str  # "PASS" | "FAIL" | "TIMEOUT"
    reason: str = ""
    duration_sec: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    """Whole-node verdict written to ``<dump>/smoke/<host>.json``."""

    host: str
    node_rank: int
    status: str  # "PASS" | "FAIL"
    duration_sec: float
    fail_reasons: List[str]
    tier1: Dict[str, Any]
    tier2: Dict[str, Any]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}][node-smoke][{socket.gethostname()}] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(
        f"[{_ts()}][node-smoke][{socket.gethostname()}] WARN: {msg}",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Per-GPU subprocess body (Tier 1 + optional Tier 2 perf)
# ---------------------------------------------------------------------------


def _per_gpu_body(
    gpu: int,
    *,
    tier2: bool,
    gemm_tflops_min: float,
    hbm_gbs_min: float,
) -> Dict[str, Any]:
    """Run all per-GPU tests for a single GPU and return a dict result.

    Tier 1 (always): set_device, allocate 256 MB, tiny GEMM 2048x2048 bf16
    with finite-value check.

    Tier 2 (when ``tier2`` is True): GEMM 8192x8192 bf16 TFLOPS measurement
    against ``gemm_tflops_min``, and HBM device-to-device copy bandwidth
    against ``hbm_gbs_min``. Each metric below threshold yields FAIL.
    """
    t0 = time.time()
    details: Dict[str, Any] = {}

    try:
        import torch  # type: ignore
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"torch import failed: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    if not torch.cuda.is_available():
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": "torch.cuda.is_available() is False",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }
    if gpu >= torch.cuda.device_count():
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": (
                f"gpu index {gpu} >= visible device_count {torch.cuda.device_count()}"
            ),
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- set_device ---
    try:
        torch.cuda.set_device(gpu)
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"set_device({gpu}) raised: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- 256 MB tensor alloc + simple write + sync ---
    try:
        nbytes_alloc = 256 * 1024 * 1024
        n_elem = nbytes_alloc // 2  # bf16
        x = torch.empty(n_elem, dtype=torch.bfloat16, device=f"cuda:{gpu}")
        x.fill_(1.0)
        torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"256MB bf16 alloc/fill/sync failed: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- tiny GEMM 2048x2048 bf16, finite-value check ---
    try:
        m = n = k = 2048
        a = torch.randn((m, k), dtype=torch.bfloat16, device=f"cuda:{gpu}")
        b = torch.randn((k, n), dtype=torch.bfloat16, device=f"cuda:{gpu}")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        if not torch.isfinite(c).all().item():
            return {
                "gpu": gpu,
                "status": "FAIL",
                "reason": "tiny GEMM produced non-finite values (possible HW corruption)",
                "duration_sec": round(time.time() - t0, 3),
                "details": details,
            }
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"tiny GEMM 2048x2048 failed: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- Tier 2 perf sanity (optional) ---
    if tier2:
        # GEMM TFLOPS. Warmup/iter counts mirror the preflight `--quick` preset
        # (`square_gemm.py` with WARMUP=5, ITERATION=20) so smoke and preflight
        # report comparable steady-state numbers.
        try:
            tflops = _measure_gemm_tflops(gpu, size=8192, warmup=5, iters=20)
            details["gemm_tflops"] = round(tflops, 2)
            if tflops < gemm_tflops_min:
                return {
                    "gpu": gpu,
                    "status": "FAIL",
                    "reason": (
                        f"GEMM TFLOPS {tflops:.0f} < threshold {gemm_tflops_min:.0f}"
                    ),
                    "duration_sec": round(time.time() - t0, 3),
                    "details": details,
                }
        except Exception as e:
            return {
                "gpu": gpu,
                "status": "FAIL",
                "reason": f"GEMM TFLOPS measurement failed: {e}",
                "duration_sec": round(time.time() - t0, 3),
                "details": details,
            }

        # HBM device-to-device copy bandwidth. HBM is fast enough that we
        # need a healthy number of timed iterations for stable timing.
        try:
            gbs = _measure_hbm_gbs(gpu, size_bytes=512 * 1024 * 1024, warmup=10, iters=20)
            details["hbm_gbs"] = round(gbs, 1)
            if gbs < hbm_gbs_min:
                return {
                    "gpu": gpu,
                    "status": "FAIL",
                    "reason": f"HBM GB/s {gbs:.0f} < threshold {hbm_gbs_min:.0f}",
                    "duration_sec": round(time.time() - t0, 3),
                    "details": details,
                }
        except Exception as e:
            return {
                "gpu": gpu,
                "status": "FAIL",
                "reason": f"HBM bandwidth measurement failed: {e}",
                "duration_sec": round(time.time() - t0, 3),
                "details": details,
            }

    return {
        "gpu": gpu,
        "status": "PASS",
        "reason": "",
        "duration_sec": round(time.time() - t0, 3),
        "details": details,
    }


def _measure_gemm_tflops(gpu: int, *, size: int, warmup: int, iters: int) -> float:
    """Measure GEMM TFLOPS for square ``size x size`` bf16 matmul on ``cuda:gpu``."""
    import torch  # type: ignore

    torch.cuda.set_device(gpu)
    a = torch.randn((size, size), dtype=torch.bfloat16, device=f"cuda:{gpu}")
    b = torch.randn((size, size), dtype=torch.bfloat16, device=f"cuda:{gpu}")
    for _ in range(warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    end.record()
    end.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0 / iters
    flops = 2.0 * size * size * size
    return (flops / elapsed_s) / 1e12


def _measure_hbm_gbs(
    gpu: int, *, size_bytes: int, warmup: int, iters: int
) -> float:
    """Measure local HBM bandwidth via device-to-device ``copy_``.

    Each iteration: 1 read of ``src`` + 1 write to ``dst`` = ``2 * size_bytes``
    of HBM traffic. Uses ``torch.cuda.Event`` for accurate GPU-side timing.
    """
    import torch  # type: ignore

    torch.cuda.set_device(gpu)
    n = size_bytes // 2  # bf16 = 2 bytes/element
    src = torch.empty(n, dtype=torch.bfloat16, device=f"cuda:{gpu}")
    dst = torch.empty_like(src)
    src.fill_(1.0)
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    end.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0 / iters
    return (2.0 * size_bytes / elapsed_s) / 1e9


# ---------------------------------------------------------------------------
# System checks: dmesg recent errors + reused info collectors
# ---------------------------------------------------------------------------


_DMESG_PATTERNS = (
    "xid",
    "hardware error",
    "gpu reset",
    "hung_task",
    "hung task",
    "page allocation failure",
    "soft lockup",
    "amdgpu.*error",
    "mce: ",
)


def _collect_dmesg_errors(window_minutes: int = 15) -> Dict[str, Any]:
    """Best-effort grep of recent dmesg lines for known-bad patterns.

    Returns a dict with ``ok`` (bool), ``matches`` (list of matched lines, capped),
    and ``error`` (str) when dmesg cannot be read.
    """
    out: Dict[str, Any] = {"ok": True, "matches": [], "error": None}
    try:
        # ``--since`` requires recent util-linux; fall back to the last 2000 lines.
        try:
            cp = subprocess.run(
                ["dmesg", "--since", f"-{window_minutes}min"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            if cp.returncode != 0:
                raise RuntimeError(cp.stderr.strip() or f"rc={cp.returncode}")
            text = cp.stdout
        except Exception:
            cp = subprocess.run(
                ["dmesg"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            if cp.returncode != 0:
                out["ok"] = False
                out["error"] = (cp.stderr or "").strip() or f"rc={cp.returncode}"
                return out
            text = "\n".join(cp.stdout.splitlines()[-2000:])

        matches: List[str] = []
        lower_patterns = [p.lower() for p in _DMESG_PATTERNS]
        for line in text.splitlines():
            ll = line.lower()
            if any(p in ll for p in lower_patterns):
                matches.append(line)
                if len(matches) >= 50:
                    break
        out["matches"] = matches
        return out
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
        return out


def _findings_to_dicts(findings: List[Any]) -> List[Dict[str, Any]]:
    """Normalize Finding dataclasses (different modules each define their own)
    into plain dicts."""
    return [
        {
            "level": getattr(f, "level", "info"),
            "message": getattr(f, "message", str(f)),
            "details": getattr(f, "details", {}),
        }
        for f in findings
    ]


def _collect_reused_info() -> Dict[str, Any]:
    """Run the existing host/gpu/network info collectors. They already work
    without a global PG and produce ``Finding`` objects (level='fail' counts
    as a node failure)."""
    section: Dict[str, Any] = {"gpu_info": [], "host_info": [], "network_info": []}
    try:
        from primus.tools.preflight.gpu.info import collect_gpu_info

        section["gpu_info"] = _findings_to_dicts(collect_gpu_info())
    except Exception as e:
        section["gpu_info"] = [
            {"level": "warn", "message": "collect_gpu_info raised", "details": {"error": str(e)}}
        ]
    try:
        from primus.tools.preflight.host.info import collect_host_info

        section["host_info"] = _findings_to_dicts(collect_host_info())
    except Exception as e:
        section["host_info"] = [
            {"level": "warn", "message": "collect_host_info raised", "details": {"error": str(e)}}
        ]
    try:
        from primus.tools.preflight.network.info import collect_network_info

        # expect_distributed=False so we don't WARN about a missing world PG.
        section["network_info"] = _findings_to_dicts(
            collect_network_info(expect_distributed=False)
        )
    except Exception as e:
        section["network_info"] = [
            {"level": "warn", "message": "collect_network_info raised", "details": {"error": str(e)}}
        ]
    return section


# ---------------------------------------------------------------------------
# Tier 2 -- node-local RCCL all-reduce (optional)
# ---------------------------------------------------------------------------


def _rccl_worker(
    local_rank: int,
    world_size: int,
    port: int,
    size_mb: int,
    out_path: str,
) -> None:
    """Subprocess body for ``torch.multiprocessing.spawn``.

    Runs ``warmup`` warmup + ``iters`` timed all-reduces of ``size_mb`` MB on a
    local-only NCCL/RCCL process group bound to ``tcp://127.0.0.1:port``.
    Local rank 0 writes the resulting GB/s to ``out_path``.

    Iteration counts are intentionally aligned with the preflight `--quick`
    preset (`intra_node_comm.py` with WARMUP=5, ITERATION=20) so smoke and
    preflight report comparable steady-state bandwidth.
    """
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore

    warmup = 5
    iters = 20

    try:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=world_size,
            rank=local_rank,
        )
        nbytes = size_mb * 1024 * 1024
        n_elem = nbytes // 2  # bf16
        t = torch.ones(n_elem, dtype=torch.bfloat16, device=f"cuda:{local_rank}")

        for _ in range(warmup):
            dist.all_reduce(t)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dist.all_reduce(t)
        end.record()
        end.synchronize()

        elapsed_s = start.elapsed_time(end) / 1000.0 / iters
        # NCCL all-reduce effective bandwidth: 2*S*(P-1)/P bytes per rank.
        comm_bytes = 2.0 * nbytes * (world_size - 1) / world_size
        gbs = comm_bytes / elapsed_s / 1e9

        if local_rank == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"status": "PASS", "gbs": round(gbs, 1)}, f)
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        if local_rank == 0:
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"status": "FAIL", "error": str(e)}, f)
            except Exception:
                pass


def _run_local_rccl(
    *, local_world_size: int, size_mb: int, timeout_sec: int
) -> Dict[str, Any]:
    """Spawn local-only RCCL workers to measure intra-node all-reduce bandwidth.

    Returns ``{"status": "PASS"|"FAIL"|"TIMEOUT", ...}``.
    """
    import tempfile

    if local_world_size <= 1:
        return {"status": "PASS", "gbs": None, "skipped": "local_world_size<=1"}

    try:
        import torch  # type: ignore
        import torch.multiprocessing as mp  # type: ignore
    except Exception as e:
        return {"status": "FAIL", "error": f"torch import failed: {e}"}

    # Pick a free local TCP port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    with tempfile.NamedTemporaryFile(
        prefix="node_smoke_rccl_", suffix=".json", delete=False
    ) as tf:
        out_path = tf.name

    ctx = mp.get_context("spawn")
    procs: List[Any] = []
    try:
        for r in range(local_world_size):
            p = ctx.Process(
                target=_rccl_worker,
                args=(r, local_world_size, port, size_mb, out_path),
            )
            p.start()
            procs.append(p)

        deadline = time.time() + timeout_sec
        for p in procs:
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=remaining)
            if p.is_alive():
                # One worker stuck -> kill all and report TIMEOUT.
                for q in procs:
                    if q.is_alive():
                        q.terminate()
                for q in procs:
                    q.join(timeout=5)
                    if q.is_alive():
                        q.kill()
                return {
                    "status": "TIMEOUT",
                    "error": f"local RCCL all-reduce did not finish in {timeout_sec}s",
                }

        if not os.path.exists(out_path):
            return {"status": "FAIL", "error": "no result file produced"}
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-node orchestration
# ---------------------------------------------------------------------------


def _spawn_per_gpu(
    gpu: int,
    *,
    timeout_sec: int,
    tier2: bool,
    gemm_tflops_min: float,
    hbm_gbs_min: float,
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
    ]
    if tier2:
        cmd.append("--tier2")

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
) -> List[str]:
    """Compute a list of ``fail_reasons`` for the node from collected results.

    Empty list -> node PASS. Any non-empty result -> node FAIL.
    """
    reasons: List[str] = []
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

    rccl = tier2_extra.get("rccl") or {}
    if rccl and rccl.get("status") not in (None, "PASS"):
        reasons.append(f"rccl: {rccl.get('status')}: {rccl.get('error', '')}")

    return reasons


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def _cmd_per_gpu(ns: argparse.Namespace) -> int:
    """Internal subcommand: run all per-GPU checks for a single GPU index."""
    result = _per_gpu_body(
        gpu=int(ns.gpu),
        tier2=bool(ns.tier2),
        gemm_tflops_min=float(ns.gemm_tflops_min),
        hbm_gbs_min=float(ns.hbm_gbs_min),
    )
    # Single JSON line on stdout; nothing else.
    print(json.dumps(result), flush=True)
    return 0 if result.get("status") == "PASS" else 1


def _cmd_run(ns: argparse.Namespace) -> int:
    """Per-node entry: orchestrate Tier 1 + optional Tier 2 + write JSON."""
    host = socket.gethostname()
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", "0")))
    expected_gpus = ns.expected_gpus
    if expected_gpus is None:
        expected_gpus = int(
            os.environ.get(
                "LOCAL_WORLD_SIZE", os.environ.get("GPUS_PER_NODE", "0")
            )
            or 0
        )
        if expected_gpus <= 0:
            try:
                import torch  # type: ignore

                expected_gpus = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                )
            except Exception:
                expected_gpus = 0
    expected_gpus = max(0, int(expected_gpus))

    _log(
        f"start node-smoke: node_rank={node_rank} expected_gpus={expected_gpus} "
        f"tier2={ns.tier2} tier2_rccl={ns.tier2_rccl}"
    )

    t0 = time.time()
    per_gpu: List[GPUResult] = []
    for i in range(expected_gpus):
        r = _spawn_per_gpu(
            i,
            timeout_sec=ns.per_gpu_timeout_sec,
            tier2=bool(ns.tier2),
            gemm_tflops_min=ns.gemm_tflops_min,
            hbm_gbs_min=ns.hbm_gbs_min,
        )
        per_gpu.append(r)
        _log(
            f"gpu{i}: {r.status} ({r.duration_sec:.1f}s)"
            + (f" -- {r.reason}" if r.reason else "")
            + (f" -- {r.details}" if r.details else "")
        )

    # Tier 1 reused info collectors
    tier1_extra: Dict[str, Any] = {}
    tier1_extra.update(_collect_reused_info())
    if not ns.skip_dmesg:
        tier1_extra["dmesg"] = _collect_dmesg_errors(window_minutes=ns.dmesg_minutes)
    else:
        tier1_extra["dmesg"] = {"ok": True, "matches": [], "error": "skipped"}

    # Tier 2 local RCCL all-reduce
    tier2_extra: Dict[str, Any] = {}
    if ns.tier2 and ns.tier2_rccl and expected_gpus > 1:
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
                    "error": (
                        f"local RCCL {rccl['gbs']} GB/s < threshold {ns.rccl_gbs_min}"
                    ),
                }
        tier2_extra["rccl"] = rccl
        _log(f"tier2 RCCL: {rccl}")

    fail_reasons = _node_status_from(per_gpu, tier1_extra, tier2_extra)
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

    return 0 if status == "PASS" else 1


def _cmd_aggregate(ns: argparse.Namespace) -> int:
    """Read all per-node JSONs from ``<dump>/smoke/`` and emit summary outputs."""
    smoke_dir = os.path.join(ns.dump_path, "smoke")
    os.makedirs(smoke_dir, exist_ok=True)

    expected = int(ns.expected_nodes) if ns.expected_nodes is not None else None
    deadline = time.time() + max(0, int(ns.wait_timeout_sec))
    found_paths: List[str] = []
    while True:
        found_paths = sorted(
            os.path.join(smoke_dir, p)
            for p in os.listdir(smoke_dir)
            if p.endswith(".json")
        )
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

    seen_hosts = {n.get("host", "") for n in nodes}
    if expected is not None and len(seen_hosts) < expected:
        # We don't necessarily know the missing hostnames; emit a marker entry.
        for i in range(expected - len(seen_hosts)):
            nodes.append(
                {
                    "host": f"<missing-{i}>",
                    "status": "FAIL",
                    "fail_reasons": [
                        f"no JSON received within {ns.wait_timeout_sec}s "
                        f"(expected_nodes={expected}, found={len(seen_hosts)})"
                    ],
                    "duration_sec": 0,
                    "node_rank": -1,
                }
            )

    # Sort by node_rank if present, otherwise by hostname.
    def _key(n: Dict[str, Any]):
        nr = n.get("node_rank", 0)
        return (int(nr) if isinstance(nr, (int, str)) and str(nr).lstrip("-").isdigit() else 1 << 30,
                str(n.get("host", "")))

    nodes.sort(key=_key)

    passing = [n for n in nodes if n.get("status") == "PASS"]
    failing = [n for n in nodes if n.get("status") != "PASS"]

    report_path = os.path.join(ns.dump_path, "smoke_report.md")
    pass_path = os.path.join(ns.dump_path, "passing_nodes.txt")
    fail_path = os.path.join(ns.dump_path, "failing_nodes.txt")

    with open(report_path, "w", encoding="utf-8") as f:
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

        if failing:
            f.write("\n## Failing nodes -- full reasons\n\n")
            for n in failing:
                f.write(f"### {n.get('host', '?')}\n\n")
                for r in n.get("fail_reasons") or []:
                    f.write(f"- {r}\n")
                f.write("\n")

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
    pr = sub.add_parser("run", help="Run per-node smoke test on this node.")
    pr.add_argument("--dump-path", default="output/preflight",
                    help="Directory under which smoke/<host>.json is written.")
    pr.add_argument("--expected-gpus", type=int, default=None,
                    help="Expected GPU count on this node (default: LOCAL_WORLD_SIZE/GPUS_PER_NODE/torch.cuda.device_count()).")
    pr.add_argument("--per-gpu-timeout-sec", type=int, default=15,
                    help="Hard timeout for each per-GPU subprocess.")
    pr.add_argument("--tier2", action="store_true",
                    help="Enable Tier 2 perf sanity (GEMM TFLOPS, HBM bandwidth).")
    pr.add_argument("--tier2-rccl", action="store_true",
                    help="Enable Tier 2 local RCCL all-reduce (implies --tier2 in spirit).")
    pr.add_argument("--gemm-tflops-min", type=float, default=600.0,
                    help="FAIL if Tier 2 GEMM TFLOPS is below this. Default: 600 (MI300X-class).")
    pr.add_argument("--hbm-gbs-min", type=float, default=2000.0,
                    help="FAIL if Tier 2 HBM GB/s is below this. Default: 2000.")
    pr.add_argument("--rccl-size-mb", type=int, default=64,
                    help="Tensor size for local RCCL all-reduce (MB).")
    pr.add_argument("--rccl-gbs-min", type=float, default=100.0,
                    help="FAIL if local RCCL GB/s is below this. Default: 100.")
    pr.add_argument("--rccl-timeout-sec", type=int, default=30,
                    help="Hard timeout for the local RCCL all-reduce phase.")
    pr.add_argument("--skip-dmesg", action="store_true",
                    help="Skip the dmesg recent-error scan (e.g. inside containers).")
    pr.add_argument("--dmesg-minutes", type=int, default=15,
                    help="Window for dmesg --since (minutes).")
    pr.set_defaults(func=_cmd_run)

    # ---- aggregate ----
    pa = sub.add_parser("aggregate",
                        help="Aggregate per-node JSONs into report + passing/failing lists.")
    pa.add_argument("--dump-path", default="output/preflight",
                    help="Same as `run --dump-path`.")
    pa.add_argument("--expected-nodes", type=int, default=None,
                    help="Number of nodes expected to report. Missing nodes are FAIL.")
    pa.add_argument("--wait-timeout-sec", type=int, default=60,
                    help="How long to wait for all expected JSONs to land before aggregating anyway.")
    pa.set_defaults(func=_cmd_aggregate)

    # ---- _per_gpu (internal) ----
    pg = sub.add_parser("_per_gpu",
                        help="(internal) Run smoke checks for a single GPU index. Spawned by `run`.")
    pg.add_argument("gpu", type=int)
    pg.add_argument("--tier2", action="store_true")
    pg.add_argument("--gemm-tflops-min", type=float, default=600.0)
    pg.add_argument("--hbm-gbs-min", type=float, default=2000.0)
    pg.set_defaults(func=_cmd_per_gpu)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
