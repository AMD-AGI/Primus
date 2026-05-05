###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from primus.tools.preflight.global_vars import (
    LOCAL_RANK,
    LOCAL_WORLD_SIZE,
    RANK,
    WORLD_SIZE,
    set_hostnames,
    set_iteration,
    set_warmup,
)
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
from primus.tools.preflight.preflight_args import PERF_TEST_TOKENS
from primus.tools.preflight.square_gemm import run_square_gemm
from primus.tools.preflight.utility import (
    gather_hostnames,
    get_first_ib_unidirectional_bandwidth,
    log,
    md_to_pdf,
    remove_file,
)
from primus.tools.utils import gather_records, get_rank_world


def _parse_csv_int_list(value: Optional[str], name: str) -> List[int]:
    """Parse a CSV of positive ints. Empty / None -> []."""
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(f"{name}: '{tok}' is not an integer")
        if v <= 0:
            raise ValueError(f"{name}: values must be positive (got {v})")
        out.append(v)
    return out


def _parse_csv_inter_group_list(value: Optional[str], name: str) -> List[Any]:
    """Parse a CSV that may include the special token 'all'."""
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    out: List[Any] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "all":
            out.append("all")
            continue
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(f"{name}: '{tok}' is not an integer or 'all'")
        if v <= 0:
            raise ValueError(f"{name}: values must be positive (got {v})")
        out.append(v)
    return out


def _write_node_hostname_legend(markdown_file: str) -> None:
    """Write a Node -> Hostname legend at the top of the perf markdown report
    and mirror it to the console on rank 0.

    Uses the per-rank hostnames already gathered via `set_hostnames()`, picking
    `LOCAL_RANK == 0` of each node as the canonical hostname for that node.
    """
    from primus.tools.preflight.global_vars import get_hostnames

    hostnames = get_hostnames()
    if not hostnames:
        return
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE

    log("=======Nodes=======")
    log(f"{'Node':<6} Hostname")
    with open(markdown_file, "a", encoding="utf-8") as f:
        f.write("# Nodes\n\n")
        f.write("| Node | Hostname |\n")
        f.write("|------|----------|\n")
        for n in range(num_nodes):
            rank = n * LOCAL_WORLD_SIZE
            host = hostnames[rank] if rank < len(hostnames) else ""
            f.write(f"| {n} | {host} |\n")
            log(f"{n:<6} {host}")
        f.write("\n")
    log("")


def _parse_perf_tests(value: Optional[str]) -> Set[str]:
    """Parse --tests CSV into a set of canonical tokens.

    None / '' / 'all' -> every token. A non-empty value that yields zero valid
    tokens (e.g. ',,,') raises ValueError so users notice the typo instead of
    silently running no perf tests.
    """
    if value is None:
        return set(PERF_TEST_TOKENS)
    raw = str(value)
    s = raw.strip().lower()
    if not s or s == "all":
        return set(PERF_TEST_TOKENS)
    selected: Set[str] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "all":
            return set(PERF_TEST_TOKENS)
        if tok not in PERF_TEST_TOKENS:
            raise ValueError(
                f"--tests: unknown token '{tok}'. " f"Valid tokens: {', '.join(PERF_TEST_TOKENS)}, all"
            )
        selected.add(tok)
    if not selected:
        raise ValueError(
            f"--tests: no valid tokens parsed from {raw!r}. "
            f"Valid tokens: {', '.join(PERF_TEST_TOKENS)}, all"
        )
    return selected


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


def run_preflight_info(args: Any, expect_distributed: bool = True) -> int:
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
        expect_distributed: Whether the run is expected to be in a distributed
            (multi-rank) context. When True (default), the network portion of
            preflight assumes multiple ranks may participate and will emit
            warnings if it detects conditions that look like a misconfigured
            or partially initialized distributed environment. When False, the
            run is treated as local-only: distributed-related network warnings
            are suppressed, which is appropriate for single-node or
            non-distributed preflight invocations.

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
        for f in collect_network_info(expect_distributed=expect_distributed):
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


def _list_set_perf_tuning_knobs(args) -> List[str]:
    """Return CLI flag names of perf tuning knobs the user explicitly set.

    Used to warn that these knobs are inert in info-only mode (i.e. when none
    of --perf-test/--tests/--quick are set but at least one of
    --host/--gpu/--network is).
    """
    set_flags: List[str] = []
    if getattr(args, "comm_sizes_mb", None) is not None:
        set_flags.append("--comm-sizes-mb")
    if getattr(args, "intra_comm_sizes_mb", None) is not None:
        set_flags.append("--intra-comm-sizes-mb")
    if getattr(args, "inter_comm_sizes_mb", None) is not None:
        set_flags.append("--inter-comm-sizes-mb")
    if getattr(args, "intra_group_sizes", None) is not None:
        set_flags.append("--intra-group-sizes")
    if getattr(args, "inter_group_sizes", None) is not None:
        set_flags.append("--inter-group-sizes")
    if getattr(args, "ring_p2p_sizes_mb", None) is not None:
        set_flags.append("--ring-p2p-sizes-mb")
    if getattr(args, "plot", False):
        set_flags.append("--plot")
    if not getattr(args, "split_nodes_subgroup", True):
        set_flags.append("--no-split-nodes-subgroup")
    return set_flags


def run_preflight(args):
    """
    Preflight entry point with dispatch logic.

    Mode precedence (single rule):

      1. Any of --perf-test / --tests / --quick is set -> perf-only mode.
         If info selectors (--host/--gpu/--network) are also present, they
         are dropped with a WARN.
      2. Otherwise, any of --host/--gpu/--network is set -> info-only mode.
         Perf tuning knobs (e.g. --comm-sizes-mb), if set, are inert and a
         WARN is emitted.
      3. Otherwise (no flags) -> default: info AND all perf tests.
    """
    perf_test = bool(getattr(args, "perf_test", False))
    tests_set = getattr(args, "tests", None) is not None
    quick_set = bool(getattr(args, "quick", False))
    has_perf_intent = perf_test or tests_set or quick_set

    def _append_dist_init_failure(markdown_file: str, timeout_sec: int, err: Exception) -> None:
        try:
            with open(markdown_file, "a", encoding="utf-8") as f:
                f.write("---\n\n")
                f.write("# Distributed Init\n\n")
                f.write(
                    "Failed to initialize `torch.distributed` process group within the timeout. "
                    "This usually indicates a network / rendezvous configuration problem.\n\n"
                )
                f.write(f"- **timeout_sec**: `{timeout_sec}`\n")
                f.write(f"- **MASTER_ADDR**: `{os.environ.get('MASTER_ADDR', '')}`\n")
                f.write(f"- **MASTER_PORT**: `{os.environ.get('MASTER_PORT', '')}`\n")
                f.write(f"- **WORLD_SIZE**: `{os.environ.get('WORLD_SIZE', '')}`\n")
                f.write(f"- **error**: `{str(err)}`\n\n")
        except Exception as ee:
            print(
                f"[Primus:Preflight] [rank0] WARN: failed to append dist init failure to report: {ee}",
                file=sys.stderr,
            )

    # If any selection flags are set, only run info collection/report.
    # IMPORTANT: do NOT initialize torch.distributed for info-only mode; preflight must not hang
    # when networking/rendezvous is misconfigured.
    check_host = bool(getattr(args, "check_host", False))
    check_gpu = bool(getattr(args, "check_gpu", False))
    check_network = bool(getattr(args, "check_network", False))
    any_selection = bool(check_host or check_gpu or check_network)

    # Precedence: perf-mode wins over info selectors.
    info_dropped_warning: Optional[str] = None
    if has_perf_intent:
        if any_selection:
            dropped = [
                flag
                for present, flag in (
                    (check_host, "--host"),
                    (check_gpu, "--gpu"),
                    (check_network, "--network"),
                )
                if present
            ]
            info_dropped_warning = (
                f"info selectors {','.join(dropped)} were dropped because perf "
                f"mode (--perf-test/--tests/--quick) takes precedence. "
                f"Run them in a separate invocation if you want both reports."
            )
            print(f"[Primus:Preflight] WARN: {info_dropped_warning}", file=sys.stderr)
            check_host = check_gpu = check_network = False
            any_selection = False
            # Reflect the override on `args` so info-section helpers downstream
            # (e.g. run_preflight_info getattr) see consistent state.
            args.check_host = False
            args.check_gpu = False
            args.check_network = False
        # Auto-imply --perf-test so the rest of the dispatch treats this as perf-only.
        perf_test = True
        args.perf_test = True
    elif any_selection:
        # Info-only mode: tuning knobs are inert. Emit a single WARN listing them.
        inert = _list_set_perf_tuning_knobs(args)
        if inert:
            print(
                f"[Primus:Preflight] WARN: {','.join(inert)} have no effect in "
                f"info-only mode (no --perf-test/--tests/--quick).",
                file=sys.stderr,
            )

    # 1) Info-only mode: run without distributed init.
    if not perf_test and any_selection:
        # First, emit a local-only report immediately (so user gets output even if PG init hangs).
        local_rc = run_preflight_info(args, expect_distributed=False)

        # Then attempt to initialize distributed with a timeout, and if successful, re-run info
        # to produce an aggregated multi-node report.
        from datetime import timedelta

        from primus.tools.utils import finalize_distributed, init_distributed

        dist_timeout_sec = int(getattr(args, "dist_timeout_sec", 120) or 120)
        rank, world = get_rank_world()
        try:
            if world > 1:
                init_distributed(timeout=timedelta(seconds=dist_timeout_sec))
                try:
                    return run_preflight_info(args)
                finally:
                    finalize_distributed()
        except Exception as e:
            if rank == 0:
                dump_path = getattr(args, "dump_path", "output/preflight")
                report_file_name = getattr(args, "report_file_name", "preflight_report")
                os.makedirs(dump_path, exist_ok=True)
                markdown_file = f"{dump_path}/{report_file_name}.md"
                _append_dist_init_failure(markdown_file, dist_timeout_sec, e)
            print(f"[Primus:Preflight] ERROR: distributed init failed: {e}", file=sys.stderr)
            return 2

        # world==1 fallback
        return local_rc

    # 2) Plain `preflight` (no flags): run info FIRST (no dist init) so we always get a report.
    info_rc = 0
    if not perf_test and not any_selection:
        info_rc = run_preflight_info(args, expect_distributed=False)

    # 2.5) Resolve perf config NOW, before any distributed rendezvous, so that
    # invalid CLI input (typos, bad sizes/group-sizes for selected tests) fails
    # fast instead of after a 120s NCCL init. This is also where we apply the
    # `--quick` warmup/iteration overrides on every rank.
    try:
        perf_cfg = _resolve_perf_config(args)
    except ValueError as e:
        print(f"[Primus:Preflight] ERROR: invalid perf config: {e}", file=sys.stderr)
        return 2
    if perf_cfg["warmup"] is not None:
        set_warmup(perf_cfg["warmup"])
    if perf_cfg["iteration"] is not None:
        set_iteration(perf_cfg["iteration"])

    # 3) Perf tests (perf-only OR plain preflight after info): now attempt distributed init
    # with a timeout so we fail fast instead of hanging.
    from datetime import timedelta

    from primus.tools.utils import finalize_distributed, init_distributed

    dist_timeout_sec = int(getattr(args, "dist_timeout_sec", 120) or 120)
    try:
        init_distributed(timeout=timedelta(seconds=dist_timeout_sec))
    except Exception as e:
        # We already wrote the info report in plain preflight mode; append a clear failure note.
        rank, _world = get_rank_world()
        dump_path = getattr(args, "dump_path", "output/preflight")
        report_file_name = getattr(args, "report_file_name", "preflight_report")
        if rank == 0:
            try:
                os.makedirs(dump_path, exist_ok=True)
                markdown_file = f"{dump_path}/{report_file_name}.md"
                _append_dist_init_failure(markdown_file, dist_timeout_sec, e)
            except Exception as ee:
                print(f"[Primus:Preflight] [rank0] WARN: failed to write report: {ee}", file=sys.stderr)

        print(f"[Primus:Preflight] ERROR: distributed init failed: {e}", file=sys.stderr)
        return 2

    try:
        # Optional: if we are in plain preflight mode and dist init succeeded, re-run info
        # to produce an aggregated report (overwriting the earlier local-only one).
        if not perf_test and not any_selection:
            run_preflight_info(args)

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

        # Write a Node -> Hostname legend at the top of the perf report so
        # subsequent tables can use compact (and possibly truncated) hostname
        # representations without losing the host<->node mapping.
        if RANK == 0:
            if info_dropped_warning:
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"> Note: {info_dropped_warning}\n\n")
            _write_node_hostname_legend(args.markdown_file)

        if RANK == 0:
            log(
                f"[Primus:Preflight] perf tests selected: "
                f"{','.join(sorted(perf_cfg['enabled_tests'])) or '(none)'}"
            )

        # ------------------------------------------------------------------
        # Dispatch tests by token, with per-test wall-clock logging.
        # ------------------------------------------------------------------
        enabled = perf_cfg["enabled_tests"]
        intra_comms = {c for c in ("allreduce", "alltoall") if f"intra-{c}" in enabled}
        inter_comms = {c for c in ("allreduce", "alltoall") if f"inter-{c}" in enabled}

        def _timed(name: str, fn):
            t0 = time.time()
            fn()
            if RANK == 0:
                log(f"[Primus:Preflight] {name} done in {time.time() - t0:.1f}s")

        if "gemm" in enabled:
            _timed("gemm", lambda: run_square_gemm(args))

        if intra_comms:
            _timed(
                f"intra-comm ({','.join(sorted(intra_comms))})",
                lambda: run_intra_node_comm(
                    args,
                    enabled_comms=intra_comms,
                    sizes_mb=perf_cfg["intra_sizes_mb"],
                    group_sizes=perf_cfg["intra_group_sizes"],
                ),
            )

        if inter_comms:
            _timed(
                f"inter-comm ({','.join(sorted(inter_comms))})",
                lambda: run_inter_node_comm(
                    args,
                    enabled_comms=inter_comms,
                    sizes_mb=perf_cfg["inter_sizes_mb"],
                    group_sizes=perf_cfg["inter_group_sizes"],
                ),
            )

        if "inter-p2p" in enabled:
            _timed(
                "inter-p2p",
                lambda: run_inter_node_comm_p2p(args, sizes_mb=perf_cfg["inter_sizes_mb"]),
            )

        if "inter-ring-p2p" in enabled:
            _timed(
                "inter-ring-p2p",
                lambda: run_inter_node_ring_p2p(args, sizes_mb=perf_cfg["ring_sizes_mb"]),
            )

        if RANK == 0 and args.save_pdf:
            md_to_pdf(args.markdown_file, args.pdf_file)

        # If we ran info+perf, preserve failures from info (if any). Info returns 0 or 2.
        # Perf path itself is warn-only and returns 0 here.
        if not perf_test and not any_selection:
            return info_rc
        return 0
    finally:
        finalize_distributed()


# Built-in defaults for tuning knobs. These mirror the documented defaults in
# the parser; we keep argparse's `default=None` so that "user did not pass it"
# is detectable, and substitute these defaults when resolving the config.
_DEFAULT_COMM_SIZES_MB = "2,4,8,16,32,64,128,256,512,1024"
_DEFAULT_INTRA_GROUP_SIZES = "2,4,8"
_DEFAULT_INTER_GROUP_SIZES = "2,4,all"
_DEFAULT_RING_P2P_SIZES_MB = "10,20,40,80,160"


def _resolve_perf_config(args) -> Dict[str, Any]:
    """Resolve perf-test selection, sizes, group sizes, and quick-preset overrides.

    User-supplied values take precedence over `--quick` defaults. Validates and
    raises `ValueError` on bad input.

    This function is side-effect free: it returns `warmup` / `iteration` for
    the caller to apply (e.g. via `set_warmup` / `set_iteration`) instead of
    mutating module globals itself.

    Validation of intra-/inter-/ring-specific knobs is gated by which tests
    are actually selected, so e.g. `--tests gemm --intra-group-sizes 3` does
    NOT abort on a host with LOCAL_WORLD_SIZE=8.
    """
    # All perf-related CLI args have argparse default=None, so a non-None value
    # unambiguously means "the user passed this flag".
    tests_str = getattr(args, "tests", None)
    comm_sizes_str = getattr(args, "comm_sizes_mb", None)
    intra_comm_sizes_str = getattr(args, "intra_comm_sizes_mb", None)
    inter_comm_sizes_str = getattr(args, "inter_comm_sizes_mb", None)
    intra_group_sizes_str = getattr(args, "intra_group_sizes", None)
    inter_group_sizes_str = getattr(args, "inter_group_sizes", None)
    ring_sizes_str = getattr(args, "ring_p2p_sizes_mb", None)
    quick = bool(getattr(args, "quick", False))
    split_nodes_subgroup = bool(getattr(args, "split_nodes_subgroup", True))

    # Apply --quick defaults only where the user did not override (i.e. value is None).
    warmup: Optional[int] = None
    iteration: Optional[int] = None
    if quick:
        if tests_str is None:
            tests_str = "gemm,intra-allreduce,inter-allreduce"
        if comm_sizes_str is None:
            comm_sizes_str = "64,1024"
        if intra_group_sizes_str is None:
            intra_group_sizes_str = str(LOCAL_WORLD_SIZE)
        if inter_group_sizes_str is None:
            inter_group_sizes_str = "all"
        # Lower warmup / iterations for a fast go/no-go signal. Caller applies.
        warmup = 5
        iteration = 20

    # Substitute built-in defaults for any values still unset.
    if comm_sizes_str is None:
        comm_sizes_str = _DEFAULT_COMM_SIZES_MB
    if intra_group_sizes_str is None:
        intra_group_sizes_str = _DEFAULT_INTRA_GROUP_SIZES
    if inter_group_sizes_str is None:
        inter_group_sizes_str = _DEFAULT_INTER_GROUP_SIZES
    if ring_sizes_str is None:
        ring_sizes_str = _DEFAULT_RING_P2P_SIZES_MB

    enabled_tests = _parse_perf_tests(tests_str)

    # Honor the legacy --no-split-nodes-subgroup alias: drop subgroup sizes
    # AND skip the inter-p2p test (matching the prior behavior of the script).
    if not split_nodes_subgroup:
        inter_group_sizes_str = "all"
        enabled_tests.discard("inter-p2p")

    needs_intra = any(t in enabled_tests for t in ("intra-allreduce", "intra-alltoall"))
    needs_inter_coll = any(t in enabled_tests for t in ("inter-allreduce", "inter-alltoall"))
    needs_inter_p2p = "inter-p2p" in enabled_tests
    needs_inter = needs_inter_coll or needs_inter_p2p
    needs_ring = "inter-ring-p2p" in enabled_tests

    default_sizes = _parse_csv_int_list(comm_sizes_str, "--comm-sizes-mb")
    intra_sizes_override = _parse_csv_int_list(intra_comm_sizes_str, "--intra-comm-sizes-mb")
    inter_sizes_override = _parse_csv_int_list(inter_comm_sizes_str, "--inter-comm-sizes-mb")
    intra_sizes_mb = intra_sizes_override or default_sizes
    inter_sizes_mb = inter_sizes_override or default_sizes
    if needs_intra and not intra_sizes_mb:
        raise ValueError("--comm-sizes-mb / --intra-comm-sizes-mb yielded no sizes")
    if needs_inter and not inter_sizes_mb:
        raise ValueError("--comm-sizes-mb / --inter-comm-sizes-mb yielded no sizes")

    intra_group_sizes = _parse_csv_int_list(intra_group_sizes_str, "--intra-group-sizes")
    if needs_intra:
        if not intra_group_sizes:
            raise ValueError("--intra-group-sizes yielded no values")
        bad = [g for g in intra_group_sizes if LOCAL_WORLD_SIZE % g != 0]
        if bad:
            raise ValueError(f"--intra-group-sizes: {bad} do not divide LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE}")

    inter_group_sizes = _parse_csv_inter_group_list(inter_group_sizes_str, "--inter-group-sizes")
    if needs_inter and not inter_group_sizes:
        raise ValueError("--inter-group-sizes yielded no values")

    ring_sizes_mb = _parse_csv_int_list(ring_sizes_str, "--ring-p2p-sizes-mb")
    if needs_ring and not ring_sizes_mb:
        raise ValueError("--ring-p2p-sizes-mb yielded no sizes")

    return {
        "enabled_tests": enabled_tests,
        "intra_sizes_mb": intra_sizes_mb,
        "inter_sizes_mb": inter_sizes_mb,
        "intra_group_sizes": intra_group_sizes,
        "inter_group_sizes": inter_group_sizes,
        "ring_sizes_mb": ring_sizes_mb,
        "warmup": warmup,
        "iteration": iteration,
    }


def main():
    parser = argparse.ArgumentParser()
    from primus.tools.preflight.preflight_args import add_preflight_parser

    add_preflight_parser(parser)
    args = parser.parse_args()

    run_preflight(args)


if __name__ == "__main__":
    main()
