###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
GPU check orchestration and summary functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .gpu_basic import run_gpu_basic_checks
from .gpu_perf import run_gpu_full_checks
from .gpu_topology import run_gpu_standard_checks


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: Dict[str, Any]


def run_gpu_checks() -> List[Finding]:
    """Run all GPU checks (basic + standard + full)."""
    out: List[Finding] = []

    # basic (may FAIL)
    basic = run_gpu_basic_checks()
    for f in basic["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    # standard (WARN by default)
    std = run_gpu_standard_checks()
    for f in std["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    # full (WARN only)
    full = run_gpu_full_checks()
    for f in full["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    return out


def _status_from_counts(fail_count: int, warn_count: int) -> str:
    if fail_count > 0:
        return "FAIL"
    if warn_count > 0:
        return "WARN"
    return "OK"


def _find_first_finding_details(findings: List[Dict[str, Any]], message: str) -> Optional[Dict[str, Any]]:
    for x in findings:
        if x.get("message") == message:
            d = x.get("details")
            return d if isinstance(d, dict) else None
    return None


def host_gpu_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize GPU-related fields for a host from per-rank records.
    """
    # Prefer any record that includes the "GPU identity" payload.
    identity_details: Optional[Dict[str, Any]] = None
    enum_details: Optional[Dict[str, Any]] = None
    occupied = False
    host_fail = 0
    host_warn = 0
    ranks: List[int] = []
    warn_msgs: set[str] = set()
    std_warn_msgs: set[str] = set()
    numa_imbalance: Optional[bool] = None
    topo_pcie_hint: Optional[bool] = None
    nccl_ib_hca_set: Optional[bool] = None
    gemm_results: List[Dict[str, Any]] = []  # Collect from all ranks
    mem_alloc: Optional[Dict[str, Any]] = None

    for r in records:
        host_fail += int(r.get("fail_count", 0) or 0)
        host_warn += int(r.get("warn_count", 0) or 0)
        if r.get("rank") is not None:
            try:
                ranks.append(int(r["rank"]))
            except Exception:
                pass

        rf = r.get("findings", [])
        if isinstance(rf, list):
            if identity_details is None:
                identity_details = _find_first_finding_details(rf, "GPU identity")
            if enum_details is None:
                enum_details = _find_first_finding_details(rf, "GPU enumeration")
            # Occupancy fail (best-effort)
            for x in rf:
                if "occupied by other processes" in str(x.get("message", "")).lower():
                    occupied = True
                if x.get("level") == "warn" and x.get("message"):
                    msg = str(x.get("message"))
                    warn_msgs.add(msg)
                    # Heuristic classification: "standard" warnings (topology/NUMA/env)
                    if (
                        "NUMA" in msg
                        or "topology" in msg.lower()
                        or msg.startswith("NCCL_IB_HCA")
                        or msg.startswith("NCCL_SOCKET_IFNAME")
                        or msg.startswith("NCCL_IB_GID_INDEX")
                    ):
                        std_warn_msgs.add(msg)

                # Standard-level signals (best-effort parsing)
                if x.get("message") == "GPU↔NUMA mapping":
                    d = x.get("details", {})
                    if isinstance(d, dict) and isinstance(d.get("imbalance"), bool):
                        numa_imbalance = d.get("imbalance")
                if x.get("message") == "GPU topology (amd-smi topo)":
                    d = x.get("details", {})
                    if isinstance(d, dict):
                        out = str(d.get("out", ""))
                        if out:
                            # Heuristic: presence of PCIE hints potential fallback.
                            topo_pcie_hint = ("PCIE" in out.upper()) or ("PCIe" in out)
                msg = str(x.get("message", ""))
                if "Topology indicates PCIe paths" in msg:
                    topo_pcie_hint = True
                if msg.startswith("NCCL_IB_HCA"):
                    d = x.get("details", {})
                    if isinstance(d, dict):
                        v = d.get("NCCL_IB_HCA", "")
                        if v is not None:
                            nccl_ib_hca_set = bool(str(v))

                # Full-level signals (best-effort parsing)
                if x.get("message") == "Single-GPU GEMM sanity":
                    d = x.get("details", {})
                    if isinstance(d, dict) and "tflops" in d:
                        gemm_results.append(d)
                if mem_alloc is None and x.get("message") == "Memory alloc/free sanity":
                    d = x.get("details", {})
                    if isinstance(d, dict):
                        mem_alloc = d

    devices = []
    if identity_details and isinstance(identity_details.get("devices"), list):
        devices = identity_details["devices"]

    gpu_count = len(devices)
    gpu_types = sorted({str(d.get("name")) for d in devices if d.get("name")})
    archs = sorted({str(d.get("arch")) for d in devices if d.get("arch")})
    totals = [d.get("total_memory_gb") for d in devices if d.get("total_memory_gb") is not None]
    frees = [d.get("free_memory_gb") for d in devices if d.get("free_memory_gb") is not None]

    # Version info (best-effort, may be None)
    amdgpu_version = ""
    rocm_version = ""
    if enum_details:
        av = enum_details.get("amdgpu_version")
        rv = enum_details.get("rocm_version")
        amdgpu_version = "" if av in (None, "") else str(av)
        rocm_version = "" if rv in (None, "") else str(rv)

    gpu_type_arch = ""
    if gpu_types or archs:
        gpu_type_arch = f"{','.join(gpu_types)}/{','.join(archs)}" if gpu_types else f"/{','.join(archs)}"

    warn_summary = "; ".join(sorted(warn_msgs))
    std_warn_summary = "; ".join(sorted(std_warn_msgs))

    # Aggregate GEMM results: min/max/avg TFLOPS
    gemm_agg: Dict[str, Any] = {}
    if gemm_results:
        tflops_list = [g["tflops"] for g in gemm_results if "tflops" in g]
        ms_list = [g["ms"] for g in gemm_results if "ms" in g]
        if tflops_list:
            gemm_agg["tflops_min"] = round(min(tflops_list), 2)
            gemm_agg["tflops_max"] = round(max(tflops_list), 2)
            gemm_agg["tflops_avg"] = round(sum(tflops_list) / len(tflops_list), 2)
            gemm_agg["num_ranks"] = len(tflops_list)
        if ms_list:
            gemm_agg["ms_min"] = round(min(ms_list), 3)
            gemm_agg["ms_max"] = round(max(ms_list), 3)
            gemm_agg["ms_avg"] = round(sum(ms_list) / len(ms_list), 3)
        # Keep shape from first result
        first = gemm_results[0]
        gemm_agg["m"] = first.get("m")
        gemm_agg["n"] = first.get("n")
        gemm_agg["k"] = first.get("k")

    return {
        "ranks": sorted(set(ranks)),
        "status": _status_from_counts(host_fail, host_warn),
        "gpu_count": gpu_count,
        "gpu_type_arch": gpu_type_arch,
        "total_memory_gb": (min(totals), max(totals)) if totals else None,
        "min_free_gb": min(frees) if frees else None,
        "occupied": occupied,
        "amdgpu_version": amdgpu_version,
        "rocm_version": rocm_version,
        "numa_imbalance": numa_imbalance,
        "topo_pcie_hint": topo_pcie_hint,
        "nccl_ib_hca_set": nccl_ib_hca_set,
        "warn_summary": warn_summary,
        "std_warn_summary": std_warn_summary,
        "gemm": gemm_agg,
        "mem_alloc": mem_alloc,
    }


def write_gpu_report(f: Any, by_host: Dict[str, List[Dict[str, Any]]]) -> None:
    """Write GPU report sections to file handle."""
    # GPU Devices table
    f.write("## GPU Devices\n\n")
    f.write(
        "| host | ranks | status | gpu_type/arch | gpu_count | total_mem_gb (min-max) | min_free_gb | occupied | amdgpu_version | rocm_version |\n"
    )
    f.write("|---|---|---|---|---:|---|---:|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_gpu_summary(by_host[h])
        total_mem = ""
        if s["total_memory_gb"] is not None:
            mn, mx = s["total_memory_gb"]
            total_mem = f"{mn}-{mx}"
        min_free = "" if s["min_free_gb"] is None else str(s["min_free_gb"])
        occ = "yes" if s["occupied"] else "no"
        ranks_str = ",".join(str(x) for x in s["ranks"]) if s["ranks"] else ""
        f.write(
            f"| {h} | {ranks_str} | {s['status']} | {s['gpu_type_arch']} | {s['gpu_count']} | "
            f"{total_mem} | {min_free} | {occ} | {s['amdgpu_version']} | {s['rocm_version']} |\n"
        )
    f.write("\n")

    # GPU Topology & Configuration table
    f.write("## GPU Topology & Configuration\n\n")
    f.write(
        "| host | ranks | status | numa_imbalance | topo_pcie_hint | NCCL_IB_HCA_set | warn_summary |\n"
    )
    f.write("|---|---|---|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_gpu_summary(by_host[h])
        ranks_str = ",".join(str(x) for x in s["ranks"]) if s["ranks"] else ""
        numa = "" if s["numa_imbalance"] is None else ("yes" if s["numa_imbalance"] else "no")
        topo = "" if s["topo_pcie_hint"] is None else ("yes" if s["topo_pcie_hint"] else "no")
        ib = "" if s["nccl_ib_hca_set"] is None else ("yes" if s["nccl_ib_hca_set"] else "no")
        warn_summary = s["std_warn_summary"] or s["warn_summary"]
        f.write(f"| {h} | {ranks_str} | {s['status']} | {numa} | {topo} | {ib} | {warn_summary} |\n")
    f.write("\n")

    # GPU Performance Sanity table
    f.write("## GPU Performance Sanity\n\n")
    # Extract gemm shape from first host for description
    gemm_shape_desc = ""
    for h in sorted(by_host.keys()):
        s = host_gpu_summary(by_host[h])
        if isinstance(s.get("gemm"), dict):
            gd = s["gemm"]
            m, n, k = gd.get("m", ""), gd.get("n", ""), gd.get("k", "")
            if m and n and k:
                gemm_shape_desc = f"GEMM shape: {m}×{n}×{k}"
                break
    if gemm_shape_desc:
        f.write(f"{gemm_shape_desc}\n\n")
    f.write("| host | num_ranks | status | tflops (min/max/avg) | ms (min/max/avg) | alloc_bytes |\n")
    f.write("|---|---:|---|---|---|---:|\n")
    for h in sorted(by_host.keys()):
        s = host_gpu_summary(by_host[h])
        num_ranks = ""
        tflops_str = ""
        ms_str = ""
        if isinstance(s.get("gemm"), dict):
            gd = s["gemm"]
            num_ranks = str(gd.get("num_ranks", ""))
            tmin = gd.get("tflops_min", "")
            tmax = gd.get("tflops_max", "")
            tavg = gd.get("tflops_avg", "")
            if tmin != "" and tmax != "" and tavg != "":
                tflops_str = f"{tmin}/{tmax}/{tavg}"
            mmin = gd.get("ms_min", "")
            mmax = gd.get("ms_max", "")
            mavg = gd.get("ms_avg", "")
            if mmin != "" and mmax != "" and mavg != "":
                ms_str = f"{mmin}/{mmax}/{mavg}"
        alloc_b = ""
        if isinstance(s.get("mem_alloc"), dict):
            alloc_b = str(s["mem_alloc"].get("bytes_approx", ""))
        f.write(f"| {h} | {num_ranks} | {s['status']} | {tflops_str} | {ms_str} | {alloc_b} |\n")
    f.write("\n")
