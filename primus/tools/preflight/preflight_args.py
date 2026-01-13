###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Argument helpers for the Primus preflight tool.

This mirrors the pattern used by `primus.tools.benchmark.*_bench_args`.
"""

import argparse


def add_preflight_perf_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register preflight *performance* arguments to the given CLI parser.

    This is the legacy (and still supported) mode:
      primus-cli preflight --dump-path ... --report-file-name ...
    """
    # Perf mode uses the same common preflight output/report flags as checks.
    add_preflight_common_check_parser(parser)
    return parser


def add_preflight_common_check_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register preflight *check (common)* arguments to the given parser.

    Example:
      primus-cli preflight gpu --level basic
    """
    # Output/report controls (common for both perf preflight and lightweight checks)
    parser.add_argument(
        "--dump-path",
        type=str,
        default="output/preflight",
        help="Directory to store preflight reports and intermediate artifacts "
        "(default: output/preflight).",
    )
    parser.add_argument(
        "--report-file-name",
        type=str,
        default="preflight_report",
        help="Base name for the generated report files (default: preflight_report).",
    )
    parser.add_argument(
        "--disable-pdf",
        dest="save_pdf",
        action="store_false",
        help="Disable generation of the PDF report (only Markdown will be produced).",
    )
    parser.add_argument(
        "--disable-plot",
        dest="plot",
        action="store_false",
        help="Disable generation of plots in the report.",
    )

    # Check controls
    parser.add_argument("--level", choices=["basic", "standard", "full"], default="basic")
    parser.add_argument("--fail-on-warn", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)

    # Optional: pass backend name if you want backend-specific checks
    parser.add_argument("--backend", type=str, default=None)

    return parser


def add_preflight_gpu_check_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register GPU-specific preflight check arguments.
    """
    add_preflight_common_check_parser(parser)

    parser.add_argument("--expect-gpu-count", type=int, default=None)
    parser.add_argument("--expect-arch", type=str, default=None)
    parser.add_argument(
        "--expect-memory",
        type=int,
        default=None,
        help="Expected GPU memory in GB",
    )

    parser.add_argument("--topology", action="store_true", help="Force topology checks")
    parser.add_argument("--perf-sanity", action="store_true", help="Force perf sanity checks")

    return parser


def add_preflight_network_check_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register network-specific preflight check arguments.

    Keep this intentionally minimal for now; network checks are mostly driven by
    environment variables (e.g., NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME).
    """
    add_preflight_common_check_parser(parser)
    parser.add_argument(
        "--expect-ib",
        action="store_true",
        help="Explicitly indicate InfiniBand/RDMA is expected (otherwise inferred).",
    )
    parser.add_argument(
        "--comm-sanity",
        action="store_true",
        help="(full only) Run a minimal allreduce sanity test (no perf measurement).",
    )
    return parser


def add_preflight_all_check_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register arguments for `preflight all` without duplicating common flags.
    """
    add_preflight_common_check_parser(parser)
    # GPU expectations/control flags are useful for "all" as well.
    parser.add_argument("--expect-gpu-count", type=int, default=None)
    parser.add_argument("--expect-arch", type=str, default=None)
    parser.add_argument(
        "--expect-memory",
        type=int,
        default=None,
        help="Expected GPU memory in GB",
    )
    parser.add_argument("--topology", action="store_true", help="Force topology checks")
    parser.add_argument("--perf-sanity", action="store_true", help="Force perf sanity checks")
    return parser


def add_preflight_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register preflight *performance* arguments to the given CLI parser.

    Note:
      The `primus-cli preflight` subcommand defines `gpu|network|all` suites in
      `primus/cli/subcommands/preflight.py` (benchmark-like).
    """
    return add_preflight_perf_parser(parser)


def build_parser() -> argparse.ArgumentParser:
    """
    Build a standalone preflight parser with subcommands.
    """
    p = argparse.ArgumentParser(prog="primus-cli preflight")
    # Standalone parser includes both perf args and suites.
    add_preflight_perf_parser(p)

    sub = p.add_subparsers(dest="suite", required=True)
    gpu = sub.add_parser("gpu", help="Run GPU preflight checks")
    add_preflight_gpu_check_parser(gpu)
    network = sub.add_parser("network", help="Run network preflight checks")
    add_preflight_network_check_parser(network)
    all_ = sub.add_parser("all", help="Run all preflight checks (GPU + network)")
    add_preflight_all_check_parser(all_)
    return p
