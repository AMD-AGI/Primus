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
    run_gpu_checks,
    write_gpu_report,
)
from primus.tools.preflight.network.check import (
    Finding as NetworkFinding,
    run_network_checks,
    write_network_report,
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
        "host": hostname,
        "rank": rank,
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
                write_gpu_report(f, by_host)
            if check_network:
                write_network_report(f, by_host)

        # Optional PDF (rank0 only, best-effort)
        if save_pdf:
            try:
                from primus.tools.preflight.utility import md_to_pdf

                md_to_pdf(markdown_file, pdf_file)
            except Exception as e:
                print(f"[Primus:Preflight] [rank0] WARN: PDF generation failed: {e}", file=sys.stderr)

    return rc
