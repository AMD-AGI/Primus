###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Lightweight preflight checks (fast sanity checks).

CLI (benchmark-like suites):
  primus-cli preflight gpu [options]
  primus-cli preflight network [options]
  primus-cli preflight all [options]
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from primus.tools.preflight.gpu.gpu_basic import run_gpu_basic_checks
from primus.tools.preflight.gpu.gpu_perf import run_gpu_full_checks
from primus.tools.preflight.gpu.gpu_topology import run_gpu_standard_checks
from primus.tools.preflight.network.network_basic import run_network_basic_checks
from primus.tools.preflight.network.network_full import run_network_full_checks
from primus.tools.preflight.network.network_standard import run_network_standard_checks
from primus.tools.utils import gather_records, get_rank_world


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: Dict[str, Any]


def _split_ifnames(value: str) -> List[str]:
    # NCCL_SOCKET_IFNAME often supports comma-separated list and "^" exclusions.
    if not value:
        return []
    out: List[str] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw or raw.startswith("^"):
            continue
        out.append(raw)
    return out


def _gpu_findings() -> List[Finding]:
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


def _network_findings() -> List[Finding]:
    # Deprecated: network checks are now implemented in primus.tools.preflight.network.*
    return [Finding("warn", "Legacy network checks invoked", {"note": "Deprecated function."})]


def _apply_expectations(args: Any, findings: List[Finding]) -> List[Finding]:
    expect_count: Optional[int] = getattr(args, "expect_gpu_count", None)
    expect_arch: Optional[str] = getattr(args, "expect_arch", None)
    expect_mem: Optional[int] = getattr(args, "expect_memory", None)
    if expect_count is None and expect_arch is None and expect_mem is None:
        return findings

    try:
        import torch  # type: ignore
    except Exception:
        return findings
    if not torch.cuda.is_available():
        return findings

    count = int(torch.cuda.device_count())
    if expect_count is not None and count != expect_count:
        findings.append(
            Finding("fail", "GPU count expectation mismatch", {"expected": expect_count, "actual": count})
        )

    if expect_arch is not None or expect_mem is not None:
        per_dev: List[Dict[str, Any]] = []
        bad = False
        for i in range(count):
            d: Dict[str, Any] = {"index": i}
            try:
                p = torch.cuda.get_device_properties(i)
                name = getattr(p, "name", "")
                total_gb = int(round(getattr(p, "total_memory", 0) / (1024**3)))
                d["name"] = name
                d["total_memory_gb"] = total_gb

                if expect_arch is not None and expect_arch not in name:
                    d["arch_ok"] = False
                    bad = True
                else:
                    d["arch_ok"] = True

                if expect_mem is not None and total_gb < expect_mem:
                    d["memory_ok"] = False
                    bad = True
                else:
                    d["memory_ok"] = True
            except Exception as e:
                d["error"] = str(e)
                bad = True
            per_dev.append(d)

        if bad:
            findings.append(Finding("fail", "GPU expectation checks failed", {"devices": per_dev}))
        else:
            findings.append(Finding("info", "GPU expectation checks OK", {"devices": per_dev}))

    return findings


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


def _find_first_finding(findings: List[Dict[str, Any]], message: str) -> Optional[Dict[str, Any]]:
    for x in findings:
        if x.get("message") == message:
            return x if isinstance(x, dict) else None
    return None


def _host_gpu_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _host_network_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize network-related fields for a host from per-rank records.
    """
    host_fail = 0
    host_warn = 0
    ranks: List[int] = []
    env: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    intent: Dict[str, Any] = {}
    nics: Dict[str, Any] = {}
    ib: Dict[str, Any] = {}
    rccl: Dict[str, Any] = {}
    runtime: Dict[str, Any] = {}
    runtime_comm: Dict[str, Any] = {}
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
            if not summary:
                d = _find_first_finding_details(rf, "Network summary")
                if d and isinstance(d.get("summary"), dict):
                    summary = d["summary"]
            if not intent:
                d = _find_first_finding_details(rf, "Distributed intent")
                if d and isinstance(d.get("intent"), dict):
                    intent = d["intent"]
            if not env:
                d = _find_first_finding_details(rf, "Distributed env presence")
                if d and isinstance(d.get("env"), dict):
                    env = d["env"]
            if not nics:
                d = _find_first_finding_details(rf, "NIC and network path")
                if d and isinstance(d.get("nics"), dict):
                    nics = d["nics"]
            if not ib:
                d = _find_first_finding_details(rf, "InfiniBand / RDMA")
                if d and isinstance(d.get("ib"), dict):
                    ib = d["ib"]
            if not rccl:
                d = _find_first_finding_details(rf, "RCCL/NCCL snapshot")
                if d and isinstance(d.get("rccl"), dict):
                    rccl = d["rccl"]
            if not runtime:
                d = _find_first_finding_details(rf, "Runtime process group sanity")
                if d and isinstance(d.get("runtime"), dict):
                    runtime = d["runtime"]
            if not runtime_comm:
                d = _find_first_finding_details(rf, "Minimal communication sanity")
                if d and isinstance(d.get("runtime_comm"), dict):
                    runtime_comm = d["runtime_comm"]
    return {
        "ranks": sorted(set(ranks)),
        "status": _status_from_counts(host_fail, host_warn),
        "summary": summary,
        "intent": intent,
        "env": env,
        "nics": nics,
        "ib": ib,
        "rccl": rccl,
        "runtime": runtime,
        "runtime_comm": runtime_comm,
    }


def run_preflight_check(args: Any) -> int:
    """
    Run preflight checks based on args.

    Args:
        args: Namespace with optional fields:
            - check_gpu (bool): Run GPU checks
            - check_network (bool): Run network checks
            - dump_path (str): Output directory
            - report_file_name (str): Report base name
            - save_pdf (bool): Generate PDF report

    Return codes:
      0: success (or warnings without --fail-on-warn)
      2: failures
    """
    hostname = os.environ.get("HOSTNAME") or socket.gethostname()

    # Determine suite from --check-gpu / --check-network flags
    check_gpu = getattr(args, "check_gpu", False)
    check_network = getattr(args, "check_network", False)
    if not check_gpu and not check_network:
        # Neither specified → do both
        check_gpu = check_network = True

    if check_gpu and check_network:
        suite = "all"
    elif check_gpu:
        suite = "gpu"
    else:
        suite = "network"

    # Fixed defaults for simplified CLI
    fail_on_warn = False
    expect_ib = False
    comm_sanity = False

    dump_path = getattr(args, "dump_path", "output/preflight")
    report_file_name = getattr(args, "report_file_name", "preflight_report")
    save_pdf = bool(getattr(args, "save_pdf", True))

    findings: List[Finding] = []
    if suite in ("gpu", "all"):
        findings.extend(_gpu_findings())
        findings = _apply_expectations(args, findings)
    if suite in ("network", "all"):
        # Run all network checks (basic + standard + full)
        nb = run_network_basic_checks()
        for f in nb["findings"]:
            findings.append(Finding(level=f.level, message=f.message, details=f.details))
        ns = run_network_standard_checks(expect_ib=True if expect_ib else None)
        for f in ns["findings"]:
            findings.append(Finding(level=f.level, message=f.message, details=f.details))
        nf = run_network_full_checks(comm_sanity=comm_sanity)
        for f in nf["findings"]:
            findings.append(Finding(level=f.level, message=f.message, details=f.details))

    fail_count = sum(1 for x in findings if x.level == "fail")
    warn_count = sum(1 for x in findings if x.level == "warn")
    info_count = sum(1 for x in findings if x.level == "info")

    status = _status_from_counts(fail_count, warn_count)

    # Important for torchrun/container/srun:
    # - Standard/Full checks should not hard-fail by default.
    # - Any non-zero exit code from *any rank* makes torchrun error out.
    # Therefore:
    #   FAIL -> exit 2
    #   WARN -> exit 0 (unless --fail-on-warn)
    #   OK   -> exit 0
    rc = 0
    if fail_count > 0:
        rc = 2
    elif warn_count > 0 and fail_on_warn:
        rc = 2
    print(f"[Primus:Preflight] suite={suite} host={hostname} status={status}", file=sys.stderr)
    for f in findings:
        if f.level in ("warn", "fail"):
            print(f"[Primus:Preflight] {f.level.upper()}: {f.message}", file=sys.stderr)

    # Build per-rank record and gather to rank0.
    rank, world = get_rank_world()
    local_record: Dict[str, Any] = {
        "suite": suite,
        "status": status,
        "return_code": rc,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "info_count": info_count,
        "findings": [asdict(x) for x in findings],
    }
    gathered = gather_records(local_record, dst=0)

    # Rank0 writes aggregated report (Markdown + optional PDF) and optional JSON.
    if rank == 0 and gathered is not None:
        os.makedirs(dump_path, exist_ok=True)
        markdown_file = f"{dump_path}/{report_file_name}.md"
        pdf_file = f"{dump_path}/{report_file_name}.pdf"

        # Aggregate overall status across ranks (based on counts, not exit codes).
        overall_fail = sum(int(r.get("fail_count", 0) or 0) for r in gathered)
        overall_warn = sum(int(r.get("warn_count", 0) or 0) for r in gathered)
        overall_status = _status_from_counts(overall_fail, overall_warn)

        # Write markdown report
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(f"# Primus Preflight Report\n\n")
            f.write(f"- **Suite**: `{suite}`\n")
            f.write(f"- **World Size**: `{world}`\n")
            f.write(f"- **Generated**: `{now}`\n")
            f.write(f"- **Overall Status**: **{overall_status}**\n\n")

            # Group records by host
            by_host: Dict[str, List[Dict[str, Any]]] = {}
            for r in gathered:
                h = str(r.get("host", "unknown"))
                by_host.setdefault(h, []).append(r)

            if suite in ("gpu", "all"):
                f.write("## GPU Devices\n\n")
                f.write(
                    "| host | ranks | status | gpu_type/arch | gpu_count | total_mem_gb (min-max) | min_free_gb | occupied | amdgpu_version | rocm_version |\n"
                )
                f.write("|---|---|---|---|---:|---|---:|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_gpu_summary(by_host[h])
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

                f.write("## GPU Topology & Configuration\n\n")
                f.write(
                    "| host | ranks | status | numa_imbalance | topo_pcie_hint | NCCL_IB_HCA_set | warn_summary |\n"
                )
                f.write("|---|---|---|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_gpu_summary(by_host[h])
                    ranks_str = ",".join(str(x) for x in s["ranks"]) if s["ranks"] else ""
                    numa = "" if s["numa_imbalance"] is None else ("yes" if s["numa_imbalance"] else "no")
                    topo = "" if s["topo_pcie_hint"] is None else ("yes" if s["topo_pcie_hint"] else "no")
                    ib = "" if s["nccl_ib_hca_set"] is None else ("yes" if s["nccl_ib_hca_set"] else "no")
                    warn_summary = s["std_warn_summary"] or s["warn_summary"]
                    f.write(
                        f"| {h} | {ranks_str} | {s['status']} | {numa} | {topo} | {ib} | {warn_summary} |\n"
                    )
                f.write("\n")

                f.write("## GPU Performance Sanity\n\n")
                # Extract gemm shape from first host for description
                gemm_shape_desc = ""
                for h in sorted(by_host.keys()):
                    s = _host_gpu_summary(by_host[h])
                    if isinstance(s.get("gemm"), dict):
                        gd = s["gemm"]
                        m, n, k = gd.get("m", ""), gd.get("n", ""), gd.get("k", "")
                        if m and n and k:
                            gemm_shape_desc = f"GEMM shape: {m}×{n}×{k}"
                            break
                if gemm_shape_desc:
                    f.write(f"{gemm_shape_desc}\n\n")
                f.write(
                    "| host | num_ranks | status | tflops (min/max/avg) | ms (min/max/avg) | alloc_bytes |\n"
                )
                f.write("|---|---:|---|---|---|---:|\n")
                for h in sorted(by_host.keys()):
                    s = _host_gpu_summary(by_host[h])
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
            elif suite == "network":
                # Level-aware network reporting (basic/standard/full)
                f.write("## Network Status\n\n")
                f.write("| host | ranks | status | is_distributed | network_mode | has_fail | has_warn |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_network_summary(by_host[h])
                    ranks_str = ",".join(str(x) for x in s["ranks"]) if s["ranks"] else ""
                    summ = s.get("summary", {}) or {}
                    f.write(
                        f"| {h} | {ranks_str} | {s['status']} | {summ.get('is_distributed','')} | "
                        f"{summ.get('network_mode','')} | {summ.get('has_fail','')} | {summ.get('has_warn','')} |\n"
                    )
                f.write("\n")

                f.write("## Distributed Environment\n\n")
                f.write(
                    "| host | WORLD_SIZE | SLURM_NTASKS | OMPI_COMM_WORLD_SIZE | MASTER_ADDR | MASTER_PORT | RANK | LOCAL_RANK |\n"
                )
                f.write("|---|---:|---:|---:|---|---|---:|---:|\n")
                for h in sorted(by_host.keys()):
                    s = _host_network_summary(by_host[h])
                    intent = s.get("intent", {}) or {}
                    envp = s.get("env", {}) or {}
                    f.write(
                        f"| {h} | {intent.get('WORLD_SIZE','')} | {intent.get('SLURM_NTASKS','')} | "
                        f"{intent.get('OMPI_COMM_WORLD_SIZE','')} | {envp.get('MASTER_ADDR','')} | {envp.get('MASTER_PORT','')} | "
                        f"{envp.get('RANK','')} | {envp.get('LOCAL_RANK','')} |\n"
                    )
                f.write("\n")

                f.write("## Network Path\n\n")
                f.write(
                    "| host | NCCL_SOCKET_IFNAME | GLOO_SOCKET_IFNAME | ifname_valid | ifname_suspect |\n"
                )
                f.write("|---|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_network_summary(by_host[h])
                    nics = s.get("nics", {}) or {}
                    f.write(
                        f"| {h} | {nics.get('NCCL_SOCKET_IFNAME','')} | {nics.get('GLOO_SOCKET_IFNAME','')} | "
                        f"{nics.get('ifname_valid','')} | {nics.get('ifname_suspect','')} |\n"
                    )
                f.write("\n")

                f.write("## InfiniBand / RDMA\n\n")
                f.write("| host | expected_ib | has_ib | ib_status | NCCL_IB_DISABLE | ib_devices |\n")
                f.write("|---|---|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_network_summary(by_host[h])
                    ib = s.get("ib", {}) or {}
                    devs = ib.get("ib_devices", [])
                    devs_s = ",".join(devs) if isinstance(devs, list) else str(devs)
                    f.write(
                        f"| {h} | {ib.get('expected_ib','')} | {ib.get('has_ib','')} | {ib.get('ib_status','')} | "
                        f"{ib.get('NCCL_IB_DISABLE','')} | {devs_s} |\n"
                    )
                f.write("\n")

                f.write("## RCCL / NCCL Configuration\n\n")
                f.write("| host | NCCL_IB_HCA | NCCL_NET_GDR_LEVEL | NCCL_DEBUG |\n")
                f.write("|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_network_summary(by_host[h])
                    rccl = s.get("rccl", {}) or {}
                    f.write(
                        f"| {h} | {rccl.get('NCCL_IB_HCA','')} | {rccl.get('NCCL_NET_GDR_LEVEL','')} | {rccl.get('NCCL_DEBUG','')} |\n"
                    )
                f.write("\n")

                f.write("## Runtime Process Group\n\n")
                f.write("| host | pg_backend | pg_init_ok | pg_error |\n")
                f.write("|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_network_summary(by_host[h])
                    rt = s.get("runtime", {}) or {}
                    f.write(
                        f"| {h} | {rt.get('pg_backend','')} | {rt.get('pg_init_ok','')} | {rt.get('pg_error','')} |\n"
                    )
                f.write("\n")

                f.write("## (Optional) Minimal Communication Sanity\n\n")
                f.write("| host | allreduce_tested | allreduce_ok | allreduce_error |\n")
                f.write("|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = _host_network_summary(by_host[h])
                    runtime_comm_row = s.get("runtime_comm", {}) or {}
                    f.write(
                        f"| {h} | {runtime_comm_row.get('allreduce_tested','')} | {runtime_comm_row.get('allreduce_ok','')} | {runtime_comm_row.get('allreduce_error','')} |\n"
                    )
                f.write("\n")

        # Optional PDF (rank0 only, best-effort)
        if save_pdf:
            try:
                from primus.tools.preflight.utility import md_to_pdf

                md_to_pdf(markdown_file, pdf_file)
            except Exception as e:
                print(f"[Primus:Preflight] [rank0] WARN: PDF generation failed: {e}", file=sys.stderr)

    return rc
