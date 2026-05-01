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


def _short_name(h: str) -> str:
    """Return the leading short-hostname segment.

    SLURM tools (`scontrol show hostnames`, `srun --nodelist=`,
    `srun --exclude=`) all operate on short hostnames, so we normalize
    everywhere so the produced ``passing_nodes.txt`` / ``failing_nodes.txt``
    can be piped straight into them. ``socket.gethostname()`` returns the
    FQDN on some clusters, hence this helper.
    """
    if not h:
        return h
    return h.split(".", 1)[0]


def _this_host_short() -> str:
    """This node's short hostname (first segment of socket.gethostname())."""
    return _short_name(socket.gethostname())


def _log(msg: str) -> None:
    print(f"[{_ts()}][node-smoke][{_this_host_short()}] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(
        f"[{_ts()}][node-smoke][{_this_host_short()}] WARN: {msg}",
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

    # --- D-1 light: PCIe link + HBM (sysfs + torch only, fast & no shell-out) ---
    # Captured into details.low_level so the aggregator can flag drift across
    # the cluster (e.g. a single GPU running at Gen3 x8 because the slot
    # needs reseating, or a GPU that exposes only half of its HBM).
    low: Dict[str, Any] = {}
    try:
        props = torch.cuda.get_device_properties(gpu)
        bdf = getattr(props, "pci_bus_id", None) or None
        if bdf:
            low["pci_bdf"] = bdf
            sysdir = f"/sys/bus/pci/devices/{bdf.lower()}"
            speed = _read_text(f"{sysdir}/current_link_speed")
            width = _read_text(f"{sysdir}/current_link_width")
            low["pcie_link_speed_raw"] = speed or None
            low["pcie_link_width"] = int(width) if width.isdigit() else None
            # speed is e.g. "32.0 GT/s PCIe" -> 32.0
            try:
                low["pcie_link_speed_gts"] = float(speed.split()[0]) if speed else None
            except Exception:
                low["pcie_link_speed_gts"] = None
        try:
            free_b, total_b = torch.cuda.mem_get_info(gpu)
            low["hbm_total_bytes"] = int(total_b)
            low["hbm_free_bytes"] = int(free_b)
            low["hbm_total_gib"] = round(total_b / (1 << 30), 2)
        except Exception:
            low["hbm_total_bytes"] = int(getattr(props, "total_memory", 0)) or None
    except Exception as e:
        low["error"] = f"low_level capture failed: {e}"
    if low:
        details["low_level"] = low

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


def _read_text(path: str, default: str = "") -> str:
    """Best-effort read of a small sysfs/proc text file. Never raises."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return default


def _parse_os_release_pretty() -> Optional[str]:
    """Return PRETTY_NAME from /etc/os-release, or None."""
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return v
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tier 1 -- A. Software-stack fingerprint (drift detection happens at aggregate)
# ---------------------------------------------------------------------------


def _collect_node_fingerprint() -> Dict[str, Any]:
    """Collect a deterministic, hashable fingerprint of the software stack
    on this node so the aggregator can detect drift across the cluster.

    Every value is best-effort: missing tools / files become ``None`` rather
    than raising. The aggregator skips ``None`` values when computing the
    cluster majority for a given key.
    """
    fp: Dict[str, Any] = {}

    # Kernel + OS
    try:
        fp["kernel"] = os.uname().release
    except Exception:
        fp["kernel"] = None
    fp["os_release"] = _parse_os_release_pretty()
    fp["python"] = sys.version.split()[0]

    # ROCm / HIP / amdgpu
    fp["rocm"] = _read_text("/opt/rocm/.info/version") or None
    fp["amdgpu_driver"] = _read_text("/sys/module/amdgpu/version") or None

    # PyTorch + (R)CCL
    try:
        import torch  # type: ignore

        fp["torch"] = getattr(torch, "__version__", None)
        fp["torch_hip"] = getattr(getattr(torch, "version", None), "hip", None)
        try:
            v = torch.cuda.nccl.version()  # type: ignore[attr-defined]
            if isinstance(v, tuple):
                fp["rccl"] = ".".join(str(x) for x in v)
            else:
                fp["rccl"] = str(v)
        except Exception:
            fp["rccl"] = None

        # Locate librccl.so under torch's lib dir for a stable per-node path.
        try:
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            for n in sorted(os.listdir(torch_lib)):
                if n.startswith("librccl.so"):
                    fp["rccl_path"] = os.path.join(torch_lib, n)
                    break
        except Exception:
            pass
    except Exception:
        fp["torch"] = None
        fp["torch_hip"] = None
        fp["rccl"] = None

    # Per-IB-device firmware + HCA model fingerprints. Both are critical for
    # detecting "1 of N nodes flashed differently" silent regressions.
    nic_fw: Dict[str, str] = {}
    nic_hca: Dict[str, str] = {}
    ib_root = "/sys/class/infiniband"
    if os.path.isdir(ib_root):
        try:
            for dev in sorted(os.listdir(ib_root)):
                fw = _read_text(os.path.join(ib_root, dev, "fw_ver"))
                if fw:
                    nic_fw[dev] = fw
                hca = _read_text(os.path.join(ib_root, dev, "hca_type"))
                if hca:
                    nic_hca[dev] = hca
        except Exception:
            pass
    fp["nic_fw"] = nic_fw or None
    fp["nic_hca"] = nic_hca or None

    return fp


# ---------------------------------------------------------------------------
# Tier 1 -- B. NIC / RDMA roll-call (per-port state + GIDs from sysfs)
# ---------------------------------------------------------------------------


def _collect_nic_status(expected_count: Optional[int]) -> Dict[str, Any]:
    """Inventory every RDMA port on this node and flag the ones that would
    silently break inter-node training.

    Reads everything from ``/sys/class/infiniband`` so we don't depend on
    ``ibv_devinfo`` / ``ibstat`` being present in the container. Per port
    we capture:

    * link state (``state``: ``ACTIVE``/``DOWN``/``INIT``) and physical
      state (``phys_state``: ``LinkUp``/``Polling``/...);
    * link rate (Gb/s);
    * netdev + MTU (so the aggregator can detect MTU drift, which silently
      tanks RoCE all-reduce throughput);
    * GID counts -- total non-zero GIDs and the subset configured as
      ``RoCE v2`` (an empty RoCE v2 set is a frequent cause of training
      jobs hanging at the first inter-node collective).

    Issues are pushed into ``out["issues"]`` (each a short string). Hard
    issues (port not Active / no RoCE v2 GIDs / wrong NIC count) are
    treated as node FAIL by ``_node_status_from``.
    """
    out: Dict[str, Any] = {
        "expected_count": expected_count,
        "ports": [],
        "issues": [],
    }
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        # Container may not expose the IB stack; report and let the operator
        # decide. We only mark this as a hard issue when the user explicitly
        # asked for a positive expected_count.
        msg = f"{base} missing -- no RDMA stack visible"
        if expected_count and expected_count > 0:
            out["issues"].append(msg)
        else:
            out["info"] = msg
        return out

    try:
        devs = sorted(os.listdir(base))
    except Exception as e:
        out["issues"].append(f"failed to list {base}: {e}")
        return out

    for dev in devs:
        port_dir = os.path.join(base, dev, "ports")
        if not os.path.isdir(port_dir):
            continue
        try:
            ports = sorted(os.listdir(port_dir))
        except Exception:
            continue
        for port_str in ports:
            try:
                port = int(port_str)
            except ValueError:
                continue
            p = os.path.join(port_dir, port_str)

            # Sysfs values look like "4: ACTIVE" / "5: LinkUp" / "400 Gb/sec (4X NDR)"
            state_raw = _read_text(os.path.join(p, "state"))
            phys_raw = _read_text(os.path.join(p, "phys_state"))
            rate_raw = _read_text(os.path.join(p, "rate"))
            state = state_raw.split(":", 1)[-1].strip() if state_raw else ""
            phys = phys_raw.split(":", 1)[-1].strip() if phys_raw else ""
            rate_gbps: Optional[int] = None
            try:
                rate_gbps = int(rate_raw.split()[0])
            except Exception:
                pass

            # GID inventory. A GID is "all-zero" until configured.
            gid_count = 0
            rocev2_count = 0
            gids_dir = os.path.join(p, "gids")
            types_dir = os.path.join(p, "gid_attrs", "types")
            valid_gid_indices: List[int] = []
            if os.path.isdir(gids_dir):
                try:
                    for gn in sorted(os.listdir(gids_dir), key=lambda s: int(s) if s.isdigit() else 0):
                        if not gn.isdigit():
                            continue
                        g = _read_text(os.path.join(gids_dir, gn))
                        if g and g != "0000:0000:0000:0000:0000:0000:0000:0000":
                            gid_count += 1
                            valid_gid_indices.append(int(gn))
                except Exception:
                    pass
            if os.path.isdir(types_dir):
                for idx in valid_gid_indices:
                    t = _read_text(os.path.join(types_dir, str(idx)))
                    if "RoCE v2" in t or "RoCEv2" in t:
                        rocev2_count += 1

            # Linked netdev + MTU.
            ifname: Optional[str] = None
            mtu: Optional[int] = None
            net_dir = os.path.join(base, dev, "device", "net")
            if os.path.isdir(net_dir):
                try:
                    nets = sorted(os.listdir(net_dir))
                    if nets:
                        ifname = nets[0]
                        mtu_raw = _read_text(f"/sys/class/net/{ifname}/mtu")
                        try:
                            mtu = int(mtu_raw)
                        except Exception:
                            mtu = None
                except Exception:
                    pass

            out["ports"].append({
                "device": dev,
                "port": port,
                "state": state or None,
                "phys_state": phys or None,
                "rate_gbps": rate_gbps,
                "ifname": ifname,
                "mtu": mtu,
                "gid_count": gid_count,
                "rocev2_gid_count": rocev2_count,
            })

            # Per-port hard issues -> node FAIL.
            if state and state.upper() != "ACTIVE":
                out["issues"].append(f"{dev}:{port} state={state} (expected ACTIVE)")
            if phys and phys.upper() != "LINKUP":
                out["issues"].append(f"{dev}:{port} phys_state={phys} (expected LinkUp)")
            if state.upper() == "ACTIVE" and rocev2_count == 0:
                out["issues"].append(f"{dev}:{port} no RoCE v2 GIDs configured")

    if expected_count is not None and len(out["ports"]) != expected_count:
        out["issues"].append(
            f"RDMA NIC port count {len(out['ports'])} != expected {expected_count}"
        )

    return out


# ---------------------------------------------------------------------------
# Tier 1 -- C. Host limits (ulimit -l, /dev/shm, NUMA, CPU governor)
# ---------------------------------------------------------------------------


def _collect_host_limits(*, ulimit_l_min_gb: float, shm_min_gb: float) -> Dict[str, Any]:
    """Capture training-relevant kernel/process limits and tunables and
    return hard-failure reasons for the ones that block training under load.

    Hard fail today (cause node FAIL):

    * ``ulimit -l`` (RLIMIT_MEMLOCK) is not unlimited and below
      ``ulimit_l_min_gb`` -- RDMA pin failures look like NCCL hangs.
    * ``/dev/shm`` total size below ``shm_min_gb`` -- NCCL shared-memory
      transport falls back or fails.

    Soft (collected for drift detection only):

    * NUMA node count, CPU count, CPU governor, kernel/OS version. The
      aggregator flags drift across the cluster but does not FAIL nodes
      individually for these.
    """
    out: Dict[str, Any] = {}

    # Resource limits.
    try:
        import resource  # type: ignore

        soft_l, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        out["memlock_soft_bytes"] = (
            -1 if soft_l == resource.RLIM_INFINITY else int(soft_l)
        )
        soft_n, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        out["nofile_soft"] = int(soft_n)
        soft_p, _ = resource.getrlimit(resource.RLIMIT_NPROC)
        out["nproc_soft"] = -1 if soft_p == resource.RLIM_INFINITY else int(soft_p)
    except Exception as e:
        out["resource_error"] = str(e)

    # /dev/shm size + free.
    try:
        st = os.statvfs("/dev/shm")
        out["shm_size_bytes"] = int(st.f_blocks) * int(st.f_frsize)
        out["shm_avail_bytes"] = int(st.f_bavail) * int(st.f_frsize)
    except Exception as e:
        out["shm_error"] = str(e)

    # NUMA topology.
    try:
        nodes = [
            n for n in os.listdir("/sys/devices/system/node")
            if n.startswith("node") and n[4:].isdigit()
        ]
        out["numa_nodes"] = len(nodes)
    except Exception:
        out["numa_nodes"] = None

    # CPU count + governor.
    try:
        out["cpu_count"] = os.cpu_count()
    except Exception:
        out["cpu_count"] = None
    out["cpu_governor"] = (
        _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or None
    )

    # Hard checks.
    fail_reasons: List[str] = []
    memlock = out.get("memlock_soft_bytes")
    if memlock is not None and memlock != -1 and ulimit_l_min_gb > 0:
        if memlock < ulimit_l_min_gb * (1 << 30):
            fail_reasons.append(
                f"ulimit -l (memlock) = {memlock // (1 << 20)} MiB; "
                f"required: unlimited or >= {ulimit_l_min_gb} GiB. "
                "RDMA pin will fail under load."
            )
    shm = out.get("shm_size_bytes")
    if shm is not None and shm_min_gb > 0:
        if shm < shm_min_gb * (1 << 30):
            fail_reasons.append(
                f"/dev/shm size = {shm / (1 << 30):.2f} GiB; "
                f"required: >= {shm_min_gb} GiB. NCCL shared-mem may fail."
            )
    out["fail_reasons"] = fail_reasons

    return out


# ---------------------------------------------------------------------------
# Tier 1 -- D-1 heavy: per-GPU low-level via amd-smi (ECC, throttle, clocks,
# power cap). Runs ONCE per node (not per per-GPU subprocess) so the smoke
# step doesn't pay an amd-smi startup tax 8x. Best-effort: missing amd-smi
# or unparseable output degrades to {"ok": False, ...} without raising.
# ---------------------------------------------------------------------------


def _which(prog: str) -> Optional[str]:
    """Tiny shutil.which() replacement that doesn't pull in shutil at import."""
    for d in (os.environ.get("PATH") or "").split(os.pathsep):
        p = os.path.join(d, prog)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _collect_amd_smi_metrics() -> Dict[str, Any]:
    """Best-effort capture of per-GPU low-level metrics via ``amd-smi``.

    We try ``amd-smi metric --json`` first (newer builds emit valid JSON);
    if that fails we fall back to text output and surface the raw text under
    ``raw`` so an operator can still grep it. The on-disk shape is:

        {
            "ok": bool,
            "tool": "amd-smi metric --json" | "amd-smi metric" | None,
            "per_gpu": [ {gpu, gfx_clock_mhz, hbm_used_bytes,
                          power_avg_w, power_cap_w, temp_edge_c,
                          ecc_uncorrectable_total, ecc_correctable_total,
                          throttle_status_raw, ...}, ... ],
            "error": "..."  (only when ok is False)
        }

    Hard-fail semantics live in ``_node_status_from``: any non-zero
    uncorrectable ECC count, or any throttle reason that contains
    ``thermal``/``power``/``current``, becomes a node FAIL.
    """
    out: Dict[str, Any] = {"ok": False, "tool": None, "per_gpu": []}
    if _which("amd-smi") is None:
        out["error"] = "amd-smi not found in PATH"
        return out

    # Try JSON first.
    try:
        cp = subprocess.run(
            ["amd-smi", "metric", "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15, check=False,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            try:
                doc = json.loads(cp.stdout)
                out["ok"] = True
                out["tool"] = "amd-smi metric --json"
                out["per_gpu"] = _flatten_amd_smi_metric_json(doc)
                return out
            except Exception as e:
                out["json_parse_error"] = str(e)
        else:
            out["json_rc"] = cp.returncode
            out["json_stderr"] = (cp.stderr or "").strip()[:200]
    except subprocess.TimeoutExpired:
        out["json_error"] = "amd-smi metric --json timed out"
    except Exception as e:
        out["json_error"] = str(e)

    # Fallback: capture the raw text output for operator grepping. We don't
    # try to parse the human-readable text -- per-GPU outliers will still
    # show up via the sysfs/torch-side details we capture in _per_gpu_body.
    try:
        cp = subprocess.run(
            ["amd-smi", "metric"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15, check=False,
        )
        if cp.returncode == 0:
            out["ok"] = True
            out["tool"] = "amd-smi metric"
            out["raw"] = cp.stdout[:8000]  # cap to keep JSON small
            return out
        out["error"] = (cp.stderr or "").strip()[:200] or f"rc={cp.returncode}"
    except subprocess.TimeoutExpired:
        out["error"] = "amd-smi metric timed out"
    except Exception as e:
        out["error"] = str(e)
    return out


def _flatten_amd_smi_metric_json(doc: Any) -> List[Dict[str, Any]]:
    """Pull the fields we care about out of `amd-smi metric --json` output.

    The exact schema varies between amd-smi releases. We touch only the
    most-stable nesting -- a top-level list of per-GPU dicts, each with
    sub-blocks like ``power``, ``clock``, ``temperature``, ``ecc``,
    ``throttle_status`` -- and tolerate missing fields silently.
    """
    out: List[Dict[str, Any]] = []
    items = doc if isinstance(doc, list) else (doc.get("gpus", []) if isinstance(doc, dict) else [])
    for i, g in enumerate(items):
        if not isinstance(g, dict):
            continue
        rec: Dict[str, Any] = {"gpu": i}
        # gpu id may be in g["gpu"] or g["device_id"] depending on schema
        if isinstance(g.get("gpu"), int):
            rec["gpu"] = g["gpu"]
        # power
        power = g.get("power") or {}
        if isinstance(power, dict):
            for k_src, k_dst in (
                ("average_socket_power", "power_avg_w"),
                ("current_socket_power", "power_avg_w"),
                ("socket_power", "power_avg_w"),
                ("power_cap", "power_cap_w"),
                ("power_limit", "power_cap_w"),
            ):
                v = power.get(k_src)
                if isinstance(v, (int, float)) and rec.get(k_dst) is None:
                    rec[k_dst] = v
        # clocks (gfx clock most useful)
        clk = g.get("clock") or g.get("clocks") or {}
        if isinstance(clk, dict):
            gfx = clk.get("gfx") or clk.get("gfx_0") or clk.get("gfx_clock") or {}
            if isinstance(gfx, dict):
                for k in ("clk", "current", "value", "frequency"):
                    if isinstance(gfx.get(k), (int, float)):
                        rec["gfx_clock_mhz"] = gfx[k]
                        break
            elif isinstance(gfx, (int, float)):
                rec["gfx_clock_mhz"] = gfx
        # temperature
        temp = g.get("temperature") or {}
        if isinstance(temp, dict):
            for k in ("edge", "current", "value"):
                v = temp.get(k)
                if isinstance(v, (int, float)):
                    rec["temp_edge_c"] = v
                    break
        # ECC
        ecc = g.get("ecc") or g.get("ecc_count") or {}
        if isinstance(ecc, dict):
            ue = ecc.get("uncorrectable") or ecc.get("uncorrectable_total") or ecc.get("ue") or 0
            ce = ecc.get("correctable") or ecc.get("correctable_total") or ecc.get("ce") or 0
            try:
                rec["ecc_uncorrectable_total"] = int(ue)
                rec["ecc_correctable_total"] = int(ce)
            except Exception:
                pass
        # Throttle
        thr = g.get("throttle_status") or g.get("throttle") or {}
        if isinstance(thr, dict):
            rec["throttle_status_raw"] = thr
        elif isinstance(thr, (str, list)):
            rec["throttle_status_raw"] = thr
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Tier 1 -- D-2: XGMI topology matrix via `amd-smi topology` (text parser)
# ---------------------------------------------------------------------------


def _collect_xgmi_topology() -> Dict[str, Any]:
    """Parse ``amd-smi topology`` and return a square link-type matrix.

    ``amd-smi topology`` emits several BDF-labelled sub-tables (ACCESS,
    WEIGHT, HOPS, LINK TYPE, NUMA BW, ...). We pick the ``LINK TYPE TABLE``
    sub-section, which contains values like ``SELF`` (diagonal) and
    ``XGMI`` / ``PCIE`` / ``PIX`` / ``SOC`` etc. Off-diagonal cells that
    aren't ``XGMI`` are recorded as ``non_xgmi_pairs`` and treated as a
    hard fail by ``_node_status_from`` -- the moment a single GPU pair
    falls back to PCIe inside a node, intra-node collectives lose 5-10x
    of the bandwidth NCCL/RCCL expects.

    The on-disk shape:

        {
            "ok": bool,
            "tool": "amd-smi topology" | None,
            "bdfs": ["0000:05:00.0", ...],
            "matrix": [["SELF","XGMI",...], ["XGMI","SELF",...], ...],
            "n_gpus": int,
            "non_xgmi_pairs": [(i, j, link_type), ...],
            "error": "..."
        }
    """
    out: Dict[str, Any] = {
        "ok": False, "tool": None, "bdfs": [], "matrix": [],
        "non_xgmi_pairs": [],
    }
    if _which("amd-smi") is None:
        out["error"] = "amd-smi not found in PATH"
        return out
    try:
        cp = subprocess.run(
            ["amd-smi", "topology"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15, check=False,
        )
    except subprocess.TimeoutExpired:
        out["error"] = "amd-smi topology timed out"
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
    if cp.returncode != 0:
        out["error"] = (cp.stderr or "").strip()[:200] or f"rc={cp.returncode}"
        return out

    text = cp.stdout

    # Parse: find the `LINK TYPE TABLE:` section, then the BDF header row,
    # then the per-BDF data rows. Stop at the next section header (any all-
    # caps label ending in `TABLE:`) or end of text.
    import re
    bdf_re = re.compile(r"\b([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)\b")
    section_header_re = re.compile(r"^\s*[A-Z][A-Z0-9 -]+TABLE:\s*$")

    lines = text.splitlines()
    try:
        idx = next(
            i for i, l in enumerate(lines)
            if l.strip().upper() == "LINK TYPE TABLE:"
        )
    except StopIteration:
        out["error"] = "no `LINK TYPE TABLE:` section in `amd-smi topology` output"
        out["raw"] = text[:4000]
        return out

    # Header row is the next non-empty line after the label, and contains
    # no leading BDF -- only column BDFs.
    header_bdfs: List[str] = []
    data_start = None
    for j in range(idx + 1, len(lines)):
        l = lines[j].rstrip()
        if not l.strip():
            continue
        if section_header_re.match(l):
            break
        toks = bdf_re.findall(l)
        if not toks:
            continue
        # The header line has only column BDFs (no leading row label), and
        # the first non-whitespace char position lines up with the columns.
        # Heuristic: header has BDFs but no other tokens that look like
        # link-type values (XGMI/PCIE/SELF/...). Data rows always have
        # exactly one leading BDF followed by N value tokens.
        non_bdf_toks = [
            t for t in l.split() if not bdf_re.fullmatch(t)
        ]
        if not non_bdf_toks:
            header_bdfs = toks
            data_start = j + 1
            break

    if not header_bdfs or data_start is None:
        out["error"] = "could not find header row inside LINK TYPE TABLE"
        out["raw"] = text[:4000]
        return out

    n = len(header_bdfs)
    bdf_to_idx = {b: i for i, b in enumerate(header_bdfs)}
    matrix: List[List[str]] = [[""] * n for _ in range(n)]
    seen_rows = 0
    for j in range(data_start, len(lines)):
        l = lines[j].rstrip()
        if not l.strip():
            continue
        if section_header_re.match(l):
            break
        toks = l.split()
        # First token must be a BDF, the remaining N tokens are the row.
        if not bdf_re.fullmatch(toks[0]):
            continue
        row_bdf = toks[0]
        cells = toks[1:]
        if row_bdf not in bdf_to_idx:
            continue
        row_idx = bdf_to_idx[row_bdf]
        for k, cell in enumerate(cells[:n]):
            matrix[row_idx][k] = cell
        seen_rows += 1

    if seen_rows == 0:
        out["error"] = "no BDF-labelled rows found inside LINK TYPE TABLE"
        out["raw"] = text[:4000]
        return out

    healthy_diag = {"SELF", "X", "-", "0"}
    healthy_link = {"XGMI"}
    non_xgmi: List[Any] = []
    for i, row in enumerate(matrix):
        for j_idx, cell in enumerate(row):
            cu = cell.strip().upper()
            if i == j_idx:
                # Diagonal: must be SELF (or empty if the row was missing).
                if cu and cu not in healthy_diag and cu not in healthy_link:
                    non_xgmi.append((i, j_idx, cell))
                continue
            if not cu:
                # Missing cell -> can't certify XGMI -> flag.
                non_xgmi.append((i, j_idx, "<missing>"))
                continue
            if cu in healthy_link or cu in healthy_diag:
                continue
            non_xgmi.append((i, j_idx, cell))

    out["ok"] = True
    out["tool"] = "amd-smi topology"
    out["bdfs"] = header_bdfs
    out["matrix"] = matrix
    out["n_gpus"] = n
    out["non_xgmi_pairs"] = non_xgmi
    return out


# ---------------------------------------------------------------------------
# Tier 1 -- E: clock state (wall time + time-daemon active states)
# ---------------------------------------------------------------------------


def _systemctl_is_active(unit: str) -> Optional[str]:
    """Return ``systemctl is-active <unit>`` ('active'/'inactive'/'failed'/...)
    or None if systemctl is missing / errors. Always best-effort."""
    if _which("systemctl") is None:
        return None
    try:
        cp = subprocess.run(
            ["systemctl", "is-active", unit],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=3, check=False,
        )
        # systemctl returns non-zero for inactive/failed -- that's fine,
        # we just want the textual state.
        return (cp.stdout or "").strip() or "unknown"
    except Exception:
        return None


def _collect_clock_state() -> Dict[str, Any]:
    """Capture this node's wall time and time-daemon health.

    Wall time is captured early so the aggregator can compute a
    cluster-wide spread. Note this includes srun launch jitter, so the
    spread is an *upper bound* on the real clock skew. The aggregator
    uses loose thresholds (warn at 30 s, no hard fail) for the spread,
    and reserves the hard fail for "no time-sync daemon active".
    """
    out: Dict[str, Any] = {
        "wall_time_unix": time.time(),
        "monotonic": time.monotonic(),
        "daemons": {},
    }
    for unit in ("chronyd", "ntp", "ntpd", "systemd-timesyncd"):
        out["daemons"][unit] = _systemctl_is_active(unit)
    active = [u for u, s in out["daemons"].items() if s == "active"]
    out["any_active"] = bool(active)
    out["active_units"] = active
    return out


# ---------------------------------------------------------------------------
# Tier 1 -- F-partial: rocm-smi self-latency
# ---------------------------------------------------------------------------


def _collect_rocm_smi_self_latency(*, timeout_sec: float) -> Dict[str, Any]:
    """Time a single ``rocm-smi --version`` call against ``timeout_sec``.

    A wedged amdgpu driver makes ``rocm-smi`` calls take 30-60 s before
    failing outright -- usually minutes before the GPU itself stops
    responding. Catching this in preflight gives operators a chance to
    drain the node before a real training job starts hanging on it.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "tool": None,
        "latency_sec": None,
        "timeout_sec": float(timeout_sec),
    }
    binpath = _which("rocm-smi")
    if binpath is None:
        out["error"] = "rocm-smi not found in PATH"
        return out
    out["tool"] = binpath
    t0 = time.monotonic()
    try:
        cp = subprocess.run(
            [binpath, "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_sec, check=False,
        )
        out["latency_sec"] = round(time.monotonic() - t0, 3)
        out["rc"] = cp.returncode
        out["ok"] = (cp.returncode == 0)
        if cp.returncode != 0:
            out["error"] = (cp.stderr or cp.stdout or "").strip()[:200]
    except subprocess.TimeoutExpired:
        out["latency_sec"] = round(time.monotonic() - t0, 3)
        out["timed_out"] = True
        out["error"] = (
            f"rocm-smi --version did not finish in {timeout_sec}s -- driver may be wedging"
        )
    except Exception as e:
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
    # Always store the short hostname so consumers (passing_nodes.txt /
    # failing_nodes.txt and SLURM tools that read them) get a name they
    # can use directly.
    host = _this_host_short()
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
    tier1_extra["gpu_low_level"] = _collect_amd_smi_metrics()
    tier1_extra["xgmi"] = _collect_xgmi_topology()
    tier1_extra["clock"] = _collect_clock_state()
    tier1_extra["tooling"] = _collect_rocm_smi_self_latency(
        timeout_sec=float(ns.rocm_smi_timeout_sec)
    )

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
        _log(
            f"xgmi: {xg.get('n_gpus', 0)}x{xg.get('n_gpus', 0)} matrix, "
            f"{len(bad)} non-XGMI pair(s)"
        )
    elif xg.get("error"):
        _warn(f"xgmi: {xg.get('error')}")
    tool = tier1_extra["tooling"]
    if tool.get("ok"):
        _log(f"rocm-smi --version: {tool.get('latency_sec')}s")
    elif tool.get("timed_out"):
        _warn(
            f"rocm-smi --version timed out after {tool.get('timeout_sec')}s "
            "-- driver may be wedging"
        )
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
        rows.append({
            "key": k, "majority": majority, "count": count,
            "total": len(per_host), "outliers": outliers,
        })
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
        rows.append({
            "device": dev, "majority": majority, "count": count,
            "total": len(per_host), "outliers": outliers,
        })
    return rows


def _nic_issue_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-node NIC roll-call issues (port DOWN / no GIDs / count mismatch)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        nic = (n.get("tier1") or {}).get("nics") or {}
        for issue in nic.get("issues", []) or []:
            rows.append({
                "node_rank": n.get("node_rank", "?"),
                "host": n.get("host", "?"),
                "issue": issue,
            })
    return rows


def _host_limits_issue_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-node host-limit hard violations (ulimit -l / /dev/shm too low)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        hl = (n.get("tier1") or {}).get("host_limits") or {}
        for issue in hl.get("fail_reasons", []) or []:
            rows.append({
                "node_rank": n.get("node_rank", "?"),
                "host": n.get("host", "?"),
                "issue": issue,
            })
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
        rows.append({
            "key": key,
            "label": label,
            "majority": majority,
            "count": count,
            "total": len(per_gpu),
            "outliers": outliers,
        })
    return rows


def _xgmi_issue_rows(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-node XGMI link issues (any non-XGMI GPU pair)."""
    rows: List[Dict[str, Any]] = []
    for n in nodes:
        xg = (n.get("tier1") or {}).get("xgmi") or {}
        if not xg.get("ok"):
            err = xg.get("error")
            if err:
                rows.append({
                    "node_rank": n.get("node_rank", "?"),
                    "host": n.get("host", "?"),
                    "summary": f"could not collect topology: {err}",
                })
            continue
        bad = xg.get("non_xgmi_pairs") or []
        if not bad:
            continue
        # Show up to 6 sample pairs to keep the table readable; the full
        # matrix lives in the per-node JSON.
        sample = ", ".join(f"({i},{j})={t}" for i, j, t in bad[:6])
        suffix = "" if len(bad) <= 6 else f" (+{len(bad) - 6} more)"
        rows.append({
            "node_rank": n.get("node_rank", "?"),
            "host": n.get("host", "?"),
            "summary": f"{len(bad)} non-XGMI pair(s): {sample}{suffix}",
        })
    return rows


def _clock_summary(
    nodes: List[Dict[str, Any]], skew_warn_sec: float,
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
            no_daemon_hosts.append(
                (n.get("node_rank", "?"), n.get("host", "?"))
            )

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
    nodes: List[Dict[str, Any]], warn_sec: float,
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
        rows.append({
            "node_rank": n.get("node_rank", "?"),
            "host": n.get("host", "?"),
            "latency_sec": lat,
            "flag": flag,
            "timeout_sec": t.get("timeout_sec"),
        })
    return rows


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
                expected_hosts_short = [
                    _short_name(line.strip())
                    for line in f
                    if line.strip()
                ]
            _log(
                f"loaded {len(expected_hosts_short)} expected hostnames from "
                f"{nodelist_file}"
            )
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
            nodes.append({
                "host": h,
                "status": "FAIL",
                "fail_reasons": [
                    f"no JSON received within {ns.wait_timeout_sec}s "
                    f"(expected hostname '{h}' from --expected-nodelist-file)"
                ],
                "duration_sec": 0,
                "node_rank": -1,
            })
    elif expected is not None and len(seen_hosts_short) < expected:
        # Fallback: we know the count but not the identities -> emit
        # synthetic placeholders. These intentionally do NOT land in
        # passing/failing txt files (see _is_real_host below).
        for i in range(expected - len(seen_hosts_short)):
            nodes.append({
                "host": f"<missing-{i}>",
                "status": "FAIL",
                "fail_reasons": [
                    f"no JSON received within {ns.wait_timeout_sec}s "
                    f"(expected_nodes={expected}, "
                    f"found={len(seen_hosts_short)})"
                ],
                "duration_sec": 0,
                "node_rank": -1,
            })

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

        # ----- E: cluster wall-clock spread + time-daemon roll-call -----
        f.write("\n## Cluster clock + time daemons\n\n")
        try:
            clk = _clock_summary(nodes, skew_warn_sec=ns.clock_skew_warn_sec)
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

        # ----- F-partial: rocm-smi self-latency -----
        f.write("\n## Tooling self-latency (`rocm-smi --version`)\n\n")
        try:
            tool_rows = _tooling_latency_rows(
                nodes, warn_sec=float(ns.rocm_smi_warn_sec),
            )
            if not tool_rows:
                f.write(
                    "*No nodes exceeded the warn threshold "
                    f"({ns.rocm_smi_warn_sec}s) and no timeouts.*\n"
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
    # NIC / RDMA roll-call (B). expected_count=None means "report only";
    # set this to e.g. 8 to make a missing or down NIC port a node FAIL.
    pr.add_argument("--expected-rdma-nics", type=int, default=None,
                    help="Expected RDMA NIC port count. If set, a count "
                         "mismatch becomes a node FAIL.")
    # Host-limits hard thresholds (C). Set to 0 to disable a check.
    pr.add_argument("--ulimit-l-min-gb", type=float, default=32.0,
                    help="FAIL the node if RLIMIT_MEMLOCK is finite and below "
                         "this many GiB (RDMA pin will fail). 0 disables.")
    pr.add_argument("--shm-min-gb", type=float, default=8.0,
                    help="FAIL the node if /dev/shm is below this many GiB "
                         "(NCCL shared-mem may fail). 0 disables.")
    # F-partial: rocm-smi self-latency. Hitting this timeout is treated as a
    # hard fail because a wedging amdgpu driver typically makes rocm-smi
    # hang for 30-60 s before the GPU itself stops responding.
    pr.add_argument("--rocm-smi-timeout-sec", type=float, default=5.0,
                    help="Hard timeout for `rocm-smi --version`. Hitting it is "
                         "a node FAIL (driver likely wedging).")
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
    pa.add_argument("--rocm-smi-warn-sec", type=float, default=1.0,
                    help="Flag (warn-only) any node where `rocm-smi --version` "
                         "took longer than this many seconds.")
    pa.add_argument("--clock-skew-warn-sec", type=float, default=30.0,
                    help="Warn (info-only) when wall-clock spread across nodes "
                         "exceeds this many seconds. Includes srun launch "
                         "jitter so the default is loose.")
    pa.add_argument("--expected-nodelist-file", type=str, default=None,
                    help="Optional file with one expected (short) hostname per line. "
                         "When provided, missing nodes are reported with their real "
                         "hostname instead of synthetic <missing-N> placeholders, and "
                         "are written to failing_nodes.txt directly. The runner script "
                         "auto-populates this from `scontrol show hostnames` under SLURM.")
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
