###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Lightweight preflight checks (fast sanity checks).

CLI:
  primus-cli preflight                  # Run all checks (GPU + Network)
  primus-cli preflight --check-gpu      # Run GPU checks only
  primus-cli preflight --check-network  # Run network checks only
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from primus.tools.preflight.gpu.check import (
    Finding as GpuFinding,
    host_gpu_summary,
    run_gpu_checks,
)
from primus.tools.preflight.network.check import (
    Finding as NetworkFinding,
    host_network_summary,
    run_network_checks,
)
from primus.tools.utils import gather_records, get_rank_world


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: Dict[str, Any]


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

    # Determine checks from --check-gpu / --check-network flags
    check_gpu = getattr(args, "check_gpu", False)
    check_network = getattr(args, "check_network", False)
    if not check_gpu and not check_network:
        # Neither specified → do both
        check_gpu = check_network = True

    # Fixed defaults for simplified CLI
    fail_on_warn = False
    expect_ib = False
    comm_sanity = False

    dump_path = getattr(args, "dump_path", "output/preflight")
    report_file_name = getattr(args, "report_file_name", "preflight_report")
    save_pdf = bool(getattr(args, "save_pdf", True))

    findings: List[Finding] = []
    if check_gpu:
        for f in run_gpu_checks():
            findings.append(Finding(level=f.level, message=f.message, details=f.details))
        findings = _apply_expectations(args, findings)
    if check_network:
        for f in run_network_checks(expect_ib=expect_ib, comm_sanity=comm_sanity):
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
    checks = []
    if check_gpu:
        checks.append("gpu")
    if check_network:
        checks.append("network")
    print(f"[Primus:Preflight] checks={','.join(checks)} host={hostname} status={status}", file=sys.stderr)
    for f in findings:
        if f.level in ("warn", "fail"):
            print(f"[Primus:Preflight] {f.level.upper()}: {f.message}", file=sys.stderr)

    # Build per-rank record and gather to rank0.
    rank, world = get_rank_world()
    local_record: Dict[str, Any] = {
        "check_gpu": check_gpu,
        "check_network": check_network,
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
        checks_str = ", ".join(checks) if checks else "none"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(f"# Primus Preflight Report\n\n")
            f.write(f"- **Checks**: `{checks_str}`\n")
            f.write(f"- **World Size**: `{world}`\n")
            f.write(f"- **Generated**: `{now}`\n")
            f.write(f"- **Overall Status**: **{overall_status}**\n\n")

            # Group records by host
            by_host: Dict[str, List[Dict[str, Any]]] = {}
            for r in gathered:
                h = str(r.get("host", "unknown"))
                by_host.setdefault(h, []).append(r)

            if check_gpu:
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
                    f.write(
                        f"| {h} | {ranks_str} | {s['status']} | {numa} | {topo} | {ib} | {warn_summary} |\n"
                    )
                f.write("\n")

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
                f.write(
                    "| host | num_ranks | status | tflops (min/max/avg) | ms (min/max/avg) | alloc_bytes |\n"
                )
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
            if check_network:
                # Level-aware network reporting (basic/standard/full)
                f.write("## Network Status\n\n")
                f.write("| host | ranks | status | is_distributed | network_mode | has_fail | has_warn |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = host_network_summary(by_host[h])
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
                    s = host_network_summary(by_host[h])
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
                    s = host_network_summary(by_host[h])
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
                    s = host_network_summary(by_host[h])
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
                    s = host_network_summary(by_host[h])
                    rccl = s.get("rccl", {}) or {}
                    f.write(
                        f"| {h} | {rccl.get('NCCL_IB_HCA','')} | {rccl.get('NCCL_NET_GDR_LEVEL','')} | {rccl.get('NCCL_DEBUG','')} |\n"
                    )
                f.write("\n")

                f.write("## Runtime Process Group\n\n")
                f.write("| host | pg_backend | pg_init_ok | pg_error |\n")
                f.write("|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = host_network_summary(by_host[h])
                    rt = s.get("runtime", {}) or {}
                    f.write(
                        f"| {h} | {rt.get('pg_backend','')} | {rt.get('pg_init_ok','')} | {rt.get('pg_error','')} |\n"
                    )
                f.write("\n")

                f.write("## (Optional) Minimal Communication Sanity\n\n")
                f.write("| host | allreduce_tested | allreduce_ok | allreduce_error |\n")
                f.write("|---|---|---|---|\n")
                for h in sorted(by_host.keys()):
                    s = host_network_summary(by_host[h])
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
