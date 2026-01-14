###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

from primus.tools.preflight.global_vars import LOCAL_RANK, RANK, set_hostnames
from primus.tools.preflight.gpu.info import collect_gpu_info, write_gpu_report
from primus.tools.preflight.host.info import collect_host_info, write_host_report
from primus.tools.preflight.inter_node_comm import run_inter_node_comm
from primus.tools.preflight.inter_node_comm_p2p import run_inter_node_comm_p2p
from primus.tools.preflight.inter_node_ring_p2p import run_inter_node_ring_p2p
from primus.tools.preflight.intra_node_comm import run_intra_node_comm
from primus.tools.preflight.network.info import (
    collect_network_info,
    write_network_report,
)
from primus.tools.preflight.square_gemm import run_square_gemm
from primus.tools.preflight.utility import (
    gather_hostnames,
    get_first_ib_unidirectional_bandwidth,
    log,
    md_to_pdf,
    remove_file,
)
from primus.tools.utils import gather_records, get_rank_world


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: Dict[str, Any]


def _status_from_counts(fail_count: int, warn_count: int) -> str:
    if fail_count > 0:
        return "FAIL"
    if warn_count > 0:
        return "WARN"
    return "OK"


def run_preflight_info(args: Any) -> int:
    """
    Run lightweight preflight info collection (host/gpu/network), aggregate across ranks,
    and write Markdown/PDF report on rank0.

    Args:
        args: Namespace with optional fields:
            - check_host (bool)
            - check_gpu (bool)
            - check_network (bool)
            - dump_path (str)
            - report_file_name (str)
            - save_pdf (bool)

    Return codes:
      0: success (WARN does not change rc)
      2: failures
    """
    hostname = os.environ.get("HOSTNAME") or socket.gethostname()

    # Determine sections from --host/--gpu/--network (compat: --check-*)
    check_host = getattr(args, "check_host", False)
    check_gpu = getattr(args, "check_gpu", False)
    check_network = getattr(args, "check_network", False)
    if not check_host and not check_gpu and not check_network:
        check_host = check_gpu = check_network = True

    dump_path = getattr(args, "dump_path", "output/preflight")
    report_file_name = getattr(args, "report_file_name", "preflight_report")
    save_pdf = bool(getattr(args, "save_pdf", True))

    findings: List[Finding] = []
    if check_host:
        for f in collect_host_info():
            findings.append(Finding(level=f.level, message=f.message, details=f.details))
    if check_gpu:
        for f in collect_gpu_info():
            findings.append(Finding(level=f.level, message=f.message, details=f.details))
    if check_network:
        for f in collect_network_info():
            findings.append(Finding(level=f.level, message=f.message, details=f.details))

    fail_count = sum(1 for x in findings if x.level == "fail")
    warn_count = sum(1 for x in findings if x.level == "warn")
    info_count = sum(1 for x in findings if x.level == "info")

    status = _status_from_counts(fail_count, warn_count)
    rc = 2 if fail_count > 0 else 0

    checks: List[str] = []
    if check_host:
        checks.append("host")
    if check_gpu:
        checks.append("gpu")
    if check_network:
        checks.append("network")
    print(f"[Primus:Preflight] checks={','.join(checks)} host={hostname} status={status}", file=sys.stderr)
    for f in findings:
        if f.level in ("warn", "fail"):
            print(f"[Primus:Preflight] {f.level.upper()}: {f.message}", file=sys.stderr)

    # Gather per-rank record → rank0 writes report.
    rank, world = get_rank_world()
    local_record: Dict[str, Any] = {
        "host": hostname,
        "rank": rank,
        "check_host": check_host,
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

    if rank == 0 and gathered is not None:
        os.makedirs(dump_path, exist_ok=True)
        markdown_file = f"{dump_path}/{report_file_name}.md"
        pdf_file = f"{dump_path}/{report_file_name}.pdf"

        overall_fail = sum(int(r.get("fail_count", 0) or 0) for r in gathered)
        overall_warn = sum(int(r.get("warn_count", 0) or 0) for r in gathered)
        overall_status = _status_from_counts(overall_fail, overall_warn)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checks_str = ", ".join(checks) if checks else "none"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write("# Primus Preflight Report\n\n")
            f.write(f"- **Checks**: `{checks_str}`\n")
            f.write(f"- **World Size**: `{world}`\n")
            f.write(f"- **Generated**: `{now}`\n")
            f.write(f"- **Overall Status**: **{overall_status}**\n\n")

            by_host: Dict[str, List[Dict[str, Any]]] = {}
            for r in gathered:
                h = str(r.get("host", "unknown"))
                by_host.setdefault(h, []).append(r)

            if check_host:
                write_host_report(f, by_host)
            if check_gpu:
                write_gpu_report(f, by_host)
            if check_network:
                write_network_report(f, by_host)

        if save_pdf:
            try:
                from primus.tools.preflight.utility import md_to_pdf as _md_to_pdf

                _md_to_pdf(markdown_file, pdf_file)
            except Exception as e:
                print(f"[Primus:Preflight] [rank0] WARN: PDF generation failed: {e}", file=sys.stderr)

    return rc


def run_preflight(args):
    """
    Preflight entry point with dispatch logic.

    - If any of --host/--gpu/--network is set → show only selected info sections
    - If no selection flags are set (plain `preflight`) → run ALL: info + perf tests
    - If --perf-test is set → run perf tests ONLY (skip info)
    """
    perf_test = getattr(args, "perf_test", False)

    # Unified distributed setup/teardown for both modes.
    # - Lightweight checks need it to aggregate multi-rank results and exit cleanly.
    # - Perf tests need it for NCCL collectives.
    from primus.tools.utils import finalize_distributed, init_distributed

    init_distributed()
    try:
        # If any selection flags are set, only run info collection/report.
        check_host = getattr(args, "check_host", False)
        check_gpu = getattr(args, "check_gpu", False)
        check_network = getattr(args, "check_network", False)
        any_selection = bool(check_host or check_gpu or check_network)

        # Perf-only mode
        if perf_test:
            any_selection = False  # ignore selection flags if perf-only is requested
        else:
            if any_selection:
                return run_preflight_info(args)

        # Plain `preflight` (no flags): run info first, then perf tests.
        info_rc = 0
        if not perf_test and not any_selection:
            info_rc = run_preflight_info(args)

        # Perf test mode (either perf-only, or plain preflight after info)
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.set_device(LOCAL_RANK)
        except Exception:
            pass

        # Perf tests rely on hostnames for reporting.
        set_hostnames(gather_hostnames())

        if RANK == 0:
            bw = get_first_ib_unidirectional_bandwidth()
            log("=======IB Bandwidth roofline (GB/s)=======")
            log(f"Bandwidth of first IB device of Node 0 : {bw:.2f} GB/s")
            args.ib_bw = bw

            if not os.path.isdir(args.dump_path):
                log(f"mkdir {args.dump_path}")
                os.makedirs(args.dump_path)

        # Avoid clobbering the lightweight preflight info report.
        # - Plain `preflight` runs both info+perf, so perf report uses a suffix.
        perf_suffix = "_perf"
        args.markdown_file = f"{args.dump_path}/{args.report_file_name}{perf_suffix}.md"
        args.pdf_file = f"{args.dump_path}/{args.report_file_name}{perf_suffix}.pdf"
        remove_file(args.markdown_file)

        # run tests
        run_square_gemm(args)
        run_intra_node_comm(args)
        run_inter_node_comm(args)
        run_inter_node_comm_p2p(args)
        run_inter_node_ring_p2p(args)

        if RANK == 0 and args.save_pdf:
            md_to_pdf(args.markdown_file, args.pdf_file)

        # If we ran info+perf, preserve failures from info (if any). Info returns 0 or 2.
        # Perf path itself is warn-only and returns 0 here.
        if not perf_test and not any_selection:
            return info_rc
        return 0
    finally:
        finalize_distributed()


def main():
    parser = argparse.ArgumentParser()
    from primus.tools.preflight.preflight_args import add_preflight_parser

    add_preflight_parser(parser)
    args = parser.parse_args()

    run_preflight(args)


if __name__ == "__main__":
    main()
