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


def _min_max_int(vals: List[float]) -> Optional[Tuple[float, float]]:
    if not vals:
        return None
    return round(min(vals), 2), round(max(vals), 2)


def host_gpu_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a host-level GPU summary from per-rank records.
    """
    ranks: List[int] = []

    gpu_type_arch: str = ""
    gpu_count: int = 0
    totals: List[float] = []
    frees: List[float] = []
    occupied = False
    amdgpu_version: str = ""
    rocm_version: str = ""

    numa_imbalance: Optional[bool] = None
    topo_pcie_hint: Optional[bool] = None
    nccl_ib_hca_set: Optional[bool] = None

    warn_msgs: List[str] = []

    gemm_agg: Optional[Dict[str, Any]] = None
    mem_alloc: Optional[Dict[str, Any]] = None

    # Collect GEMM per-rank values for aggregation
    gemm_tflops: List[float] = []
    gemm_ms: List[float] = []
    gemm_shape: Optional[Tuple[Any, Any, Any]] = None

    for r in records:
        if r.get("rank") is not None:
            try:
                ranks.append(int(r["rank"]))
            except Exception:
                pass

        rf = r.get("findings", [])
        if not isinstance(rf, list):
            continue

        # GPU enumeration (versions + count)
        d = _find_first_finding_details(rf, "GPU enumeration")
        if d:
            if not amdgpu_version:
                amdgpu_version = str(d.get("amdgpu_version") or "")
            if not rocm_version:
                rocm_version = str(d.get("rocm_version") or "")
            if gpu_count <= 0 and d.get("device_count") is not None:
                try:
                    gpu_count = int(d.get("device_count") or 0)
                except Exception:
                    pass

        # Per-device identity & memory
        d = _find_first_finding_details(rf, "GPU identity")
        if d and isinstance(d.get("devices"), list):
            devs = [x for x in d["devices"] if isinstance(x, dict)]
            if devs:
                if gpu_count <= 0:
                    gpu_count = len(devs)
                # Prefer consistent name/arch display from first device.
                if not gpu_type_arch:
                    name = str(devs[0].get("name") or "")
                    arch = devs[0].get("arch")
                    gpu_type_arch = f"{name}/{arch}" if arch else name
                for x in devs:
                    tm = x.get("total_memory_gb")
                    fm = x.get("free_memory_gb")
                    if tm is not None:
                        try:
                            totals.append(float(tm))
                        except Exception:
                            pass
                    if fm is not None:
                        try:
                            frees.append(float(fm))
                        except Exception:
                            pass

        # Occupancy: treat any foreign processes as occupied.
        d = _find_first_finding_details(rf, "GPU occupied by other processes (amd-smi)")
        if d and isinstance(d.get("processes"), list) and len(d.get("processes", [])) > 0:
            occupied = True
        d = _find_first_finding_details(rf, "GPU occupancy (amd-smi)")
        if d and isinstance(d.get("processes"), list) and len(d.get("processes", [])) > 0:
            occupied = True

        # Standard topology/config signals
        d = _find_first_finding_details(rf, "GPU↔NUMA mapping")
        if d and numa_imbalance is None and d.get("imbalance") is not None:
            numa_imbalance = bool(d.get("imbalance"))

        # Heuristic: if we saw the PCIe topology warning, set the hint.
        for fx in rf:
            if (
                isinstance(fx, dict)
                and fx.get("level") == "warn"
                and fx.get("message") == "Topology indicates PCIe paths; XGMI may be absent"
            ):
                topo_pcie_hint = True

        # NCCL_IB_HCA env (best-effort)
        for fx in rf:
            if not isinstance(fx, dict):
                continue
            msg = str(fx.get("message") or "")
            if "NCCL_IB_HCA" in msg:
                details = fx.get("details", {})
                if isinstance(details, dict):
                    val = str(details.get("NCCL_IB_HCA") or "")
                    nccl_ib_hca_set = bool(val)

        # Warnings summary (GPU-related) for the row
        for fx in rf:
            if isinstance(fx, dict) and fx.get("level") == "warn":
                m = str(fx.get("message") or "")
                if m:
                    warn_msgs.append(m)

        # Perf sanity: GEMM per-rank numbers (aggregate)
        d = _find_first_finding_details(rf, "Single-GPU GEMM sanity")
        if d:
            m, n, k = d.get("m"), d.get("n"), d.get("k")
            if gemm_shape is None and m and n and k:
                gemm_shape = (m, n, k)
            if d.get("tflops") is not None:
                try:
                    gemm_tflops.append(float(d["tflops"]))
                except Exception:
                    pass
            if d.get("ms") is not None:
                try:
                    gemm_ms.append(float(d["ms"]))
                except Exception:
                    pass

        d = _find_first_finding_details(rf, "Memory alloc/free sanity")
        if d and d.get("bytes_approx") is not None:
            mem_alloc = {"bytes_approx": d.get("bytes_approx")}

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

    warn_summary = "; ".join(sorted(set(warn_msgs)))
    status = "OK"
    if any("fail" == str(f.get("level")) for r in records for f in (r.get("findings", []) if isinstance(r.get("findings", []), list) else [])):  # type: ignore
        status = "FAIL"
    elif warn_summary:
        status = "WARN"
    return {
        "ranks": sorted(set(ranks)),
        "status": status,
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
        "std_warn_summary": "",
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
