###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
GPU info collection + report writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .gpu_basic import run_gpu_basic_checks
from .gpu_perf import run_gpu_full_checks
from .gpu_topology import run_gpu_standard_checks


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: Dict[str, Any]


def collect_gpu_info() -> List[Finding]:
    """Collect all GPU info (basic + topology + perf sanity)."""
    out: List[Finding] = []

    gb = run_gpu_basic_checks()
    for f in gb["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    gs = run_gpu_standard_checks()
    for f in gs["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    gf = run_gpu_full_checks()
    for f in gf["findings"]:
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


def _min_max_int(vals: List[int]) -> Optional[Tuple[int, int]]:
    if not vals:
        return None
    return min(vals), max(vals)


def host_gpu_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a host-level GPU summary from per-rank records.
    """
    host_fail = 0
    host_warn = 0
    ranks: List[int] = []

    gpu_type_arch: str = ""
    gpu_count: int = 0
    totals: List[int] = []
    frees: List[int] = []
    occupied = False
    amdgpu_version: str = ""
    rocm_version: str = ""

    numa_imbalance: Optional[bool] = None
    topo_pcie_hint: Optional[bool] = None
    nccl_ib_hca_set: Optional[bool] = None

    warn_summary = ""
    std_warn_summary = ""

    gemm_agg: Optional[Dict[str, Any]] = None
    mem_alloc: Optional[Dict[str, Any]] = None

    # Collect GEMM per-rank values for aggregation
    gemm_tflops: List[float] = []
    gemm_ms: List[float] = []
    gemm_shape: Optional[Tuple[Any, Any, Any]] = None

    for r in records:
        host_fail += int(r.get("fail_count", 0) or 0)
        host_warn += int(r.get("warn_count", 0) or 0)
        if r.get("rank") is not None:
            try:
                ranks.append(int(r["rank"]))
            except Exception:
                pass

        rf = r.get("findings", [])
        if not isinstance(rf, list):
            continue

        # Basic GPU metrics
        d = _find_first_finding_details(rf, "GPU inventory")
        if d and isinstance(d.get("gpu"), dict):
            g = d["gpu"]
            if not gpu_type_arch:
                gpu_type_arch = str(g.get("gpu_type_arch") or "")
            gpu_count = int(g.get("gpu_count") or gpu_count or 0)
            if g.get("total_memory_gb") is not None:
                totals.append(int(g["total_memory_gb"]))
            if g.get("free_memory_gb") is not None:
                frees.append(int(g["free_memory_gb"]))
            if bool(g.get("occupied")):
                occupied = True
            if not amdgpu_version:
                amdgpu_version = str(g.get("amdgpu_version") or "")
            if not rocm_version:
                rocm_version = str(g.get("rocm_version") or "")

        # Standard topology/config signals
        d = _find_first_finding_details(rf, "GPU topology hints")
        if d and isinstance(d.get("topology"), dict):
            topo = d["topology"]
            if numa_imbalance is None and topo.get("numa_imbalance") is not None:
                numa_imbalance = bool(topo.get("numa_imbalance"))
            if topo_pcie_hint is None and topo.get("topo_pcie_hint") is not None:
                topo_pcie_hint = bool(topo.get("topo_pcie_hint"))
            if nccl_ib_hca_set is None and topo.get("nccl_ib_hca_set") is not None:
                nccl_ib_hca_set = bool(topo.get("nccl_ib_hca_set"))

        # Warnings summary strings (best-effort)
        d = _find_first_finding_details(rf, "GPU warning summary")
        if d and isinstance(d.get("warn_summary"), str):
            warn_summary = d["warn_summary"]
        d = _find_first_finding_details(rf, "GPU standard warning summary")
        if d and isinstance(d.get("std_warn_summary"), str):
            std_warn_summary = d["std_warn_summary"]

        # Perf sanity: GEMM per-rank numbers (aggregate)
        d = _find_first_finding_details(rf, "GEMM perf sanity")
        if d and isinstance(d.get("gemm"), dict):
            gd = d["gemm"]
            m, n, k = gd.get("m"), gd.get("n"), gd.get("k")
            if gemm_shape is None and m and n and k:
                gemm_shape = (m, n, k)
            if gd.get("tflops") is not None:
                try:
                    gemm_tflops.append(float(gd["tflops"]))
                except Exception:
                    pass
            if gd.get("ms") is not None:
                try:
                    gemm_ms.append(float(gd["ms"]))
                except Exception:
                    pass

        d = _find_first_finding_details(rf, "Memory alloc sanity")
        if d and isinstance(d.get("mem_alloc"), dict):
            mem_alloc = d["mem_alloc"]

    # Build GEMM aggregate summary for reporting
    if gemm_tflops or gemm_ms:
        gemm_agg = {
            "num_ranks": len(set(ranks)),
        }
        if gemm_shape is not None:
            m, n, k = gemm_shape
            gemm_agg.update({"m": m, "n": n, "k": k})
        if gemm_tflops:
            gemm_agg.update(
                {
                    "tflops_min": round(min(gemm_tflops), 2),
                    "tflops_max": round(max(gemm_tflops), 2),
                    "tflops_avg": round(sum(gemm_tflops) / len(gemm_tflops), 2),
                }
            )
        if gemm_ms:
            gemm_agg.update(
                {
                    "ms_min": round(min(gemm_ms), 3),
                    "ms_max": round(max(gemm_ms), 3),
                    "ms_avg": round(sum(gemm_ms) / len(gemm_ms), 3),
                }
            )

    return {
        "ranks": sorted(set(ranks)),
        "status": _status_from_counts(host_fail, host_warn),
        "gpu_type_arch": gpu_type_arch,
        "gpu_count": gpu_count,
        "total_memory_gb": _min_max_int(totals),
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
    # Section header
    f.write("---\n\n")
    f.write("# GPU Info\n\n")

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
    f.write("| host | ranks | status | numa_imbalance | topo_pcie_hint | NCCL_IB_HCA_set | warn_summary |\n")
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

    # GPU Performance Sanity
    f.write("## GPU Performance Sanity\n\n")
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
