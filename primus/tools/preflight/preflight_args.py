###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Argument helpers for the Primus preflight tool.

This mirrors the pattern used by `primus.tools.benchmark.*_bench_args`.
"""

import argparse

# Canonical perf-test tokens accepted by --tests.
PERF_TEST_TOKENS = (
    "gemm",
    "intra-allreduce",
    "intra-alltoall",
    "inter-allreduce",
    "inter-alltoall",
    "inter-p2p",
    "inter-ring-p2p",
)

# Names of the "intent-bearing" perf flags. Setting any of these implies perf
# mode (no need to also pass --perf-test). When mixed with info selectors
# (--host/--gpu/--network), perf wins and the info selectors are dropped with
# a warning.
PERF_INTENT_FLAGS = ("--perf-test", "--tests", "--quick")


def add_preflight_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register arguments for `primus-cli preflight`.

    Mode precedence:
        1. Any of --perf-test / --tests / --quick is set -> perf mode wins.
           If info selectors (--host/--gpu/--network) are also set, they are
           dropped with a warning. Perf-only tuning knobs (--comm-sizes-mb,
           --intra-group-sizes, etc.) take effect.
        2. Otherwise, any of --host/--gpu/--network is set -> info-only mode.
           Perf-only tuning knobs, if set, are inert and a WARN is emitted.
        3. Otherwise (no flags) -> default: run info AND all perf tests.

    Usage:
        primus-cli preflight                          # Default: info + all perf
        primus-cli preflight --host                   # Host info only
        primus-cli preflight --gpu                    # GPU info only
        primus-cli preflight --network                # Network info only
        primus-cli preflight --gpu --network          # GPU + Network info
        primus-cli preflight --mori                   # MORI runtime build + smoke
        primus-cli preflight --perf-test              # Perf only, all tests
        primus-cli preflight --quick                  # Perf only, fast preset
        primus-cli preflight --tests gemm             # Perf only, GEMM only
        primus-cli preflight --tests gemm,inter-allreduce \\
            --comm-sizes-mb 64,1024 \\
            --inter-group-sizes all
    """
    # Check selection flags
    # Keep --check-* as compatibility aliases.
    parser.add_argument(
        "--host",
        "--check-host",
        dest="check_host",
        action="store_true",
        help="Show host info (CPU, memory, PCIe)",
    )
    parser.add_argument(
        "--gpu",
        "--check-gpu",
        dest="check_gpu",
        action="store_true",
        help="Show GPU info",
    )
    parser.add_argument(
        "--network",
        "--check-network",
        dest="check_network",
        action="store_true",
        help="Show network info",
    )

    # Performance test mode (full GEMM, intra/inter node comm tests)
    parser.add_argument(
        "--perf-test",
        action="store_true",
        help="Run perf tests ONLY (GEMM, intra/inter node communication). "
        "Skips the host/gpu/network info report. Implied by --tests/--quick.",
    )

    # Performance test specific options.
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (perf mode only).",
    )

    # Test selection (CSV). Tokens: gemm,intra-allreduce,intra-alltoall,
    # inter-allreduce,inter-alltoall,inter-p2p,inter-ring-p2p, or 'all'.
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Comma-separated list of perf tests to run. Tokens: "
        "gemm, intra-allreduce, intra-alltoall, inter-allreduce, inter-alltoall, "
        "inter-p2p, inter-ring-p2p, all. Implies --perf-test. "
        "When unset, runs every test.",
    )

    # Message size config (CSV in MB) for comm tests.
    parser.add_argument(
        "--comm-sizes-mb",
        type=str,
        default=None,
        help="Default message sizes (CSV in MB) used for intra-/inter-node "
        "allreduce, alltoall, and inter-node p2p tests when no specific override is given. "
        "Default: 2,4,8,16,32,64,128,256,512,1024.",
    )
    parser.add_argument(
        "--intra-comm-sizes-mb",
        type=str,
        default=None,
        help="Override message sizes (CSV in MB) for intra-node allreduce/alltoall. "
        "Falls back to --comm-sizes-mb when unset.",
    )
    parser.add_argument(
        "--inter-comm-sizes-mb",
        type=str,
        default=None,
        help="Override message sizes (CSV in MB) for inter-node allreduce/alltoall/p2p. "
        "Falls back to --comm-sizes-mb when unset.",
    )

    # Group size config.
    parser.add_argument(
        "--intra-group-sizes",
        type=str,
        default=None,
        help="Comma-separated list of intra-node GPU group sizes to test "
        "(each must divide LOCAL_WORLD_SIZE). Default: 2,4,8.",
    )
    parser.add_argument(
        "--inter-group-sizes",
        type=str,
        default=None,
        help="Comma-separated list of inter-node group sizes to test. Use 'all' for "
        "the full N-node group. Default: 2,4,all. Note: for inter-node alltoall "
        "each value is clamped to 16 (real-world MoE training rarely dispatches "
        "across more nodes); other inter-node tests use the requested sizes "
        "unchanged. See docs/preflight.md \u00a75.2 for details.",
    )

    # Inter-node ring p2p sizes.
    parser.add_argument(
        "--ring-p2p-sizes-mb",
        type=str,
        default=None,
        help="Message sizes (CSV in MB) for the inter-node ring P2P test. " "Default: 10,20,40,80,160.",
    )

    # Quick preset.
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast pre-launch preset. Implies --perf-test. Selects gemm + "
        "intra-allreduce + inter-allreduce, uses sizes 64,1024 MB, full "
        "intra-node group only, full N-node inter-node group only, and lowers "
        "warmup/iterations. User-supplied flags override.",
    )

    # Back-compat alias: kept so existing scripts keep working.
    # Internally it maps to --inter-group-sizes all and disables inter-p2p.
    parser.add_argument(
        "--no-split-nodes-subgroup",
        dest="split_nodes_subgroup",
        action="store_false",
        help="[Deprecated] Skip inter-node comm tests on node subgroups (2-node, 4-node). "
        "Equivalent to --inter-group-sizes all and dropping inter-p2p.",
    )

    # Distributed init timeout (prevents hangs when network/rendezvous is misconfigured)
    parser.add_argument(
        "--dist-timeout-sec",
        type=int,
        default=120,
        help="Timeout (seconds) for torch.distributed process group init. "
        "If init times out, preflight will write the info report and exit with failure.",
    )

    # Communicator cleanup delay (cross-rank sync across destroy -> setup;
    # the actual "Address already in use" defense is the inter-alltoall cap
    # in --inter-group-sizes / inter_node_comm.py, not this delay).
    parser.add_argument(
        "--comm-cleanup-delay-sec",
        type=float,
        default=2.0,
        help="Delay (seconds) inserted between destroying NCCL/RCCL process "
        "groups and creating new ones. Provides cross-rank synchronization "
        "across the destroy/setup transition and gives the kernel a moment "
        "to unlink closed-socket bookkeeping. Set to 0 to disable the sleep "
        "(barrier only). The actual 'Address already in use' defense at "
        "scale is the inter-node alltoall sub-group cap of 16 (see "
        "--inter-group-sizes); widening net.ipv4.ip_local_port_range "
        "(docs/preflight.md \u00a77.2) is the directly relevant OS knob if "
        "you ever need more headroom.",
    )

    # Report output options
    parser.add_argument(
        "--dump-path",
        type=str,
        default="output/preflight",
        help="Directory to store preflight reports (default: output/preflight).",
    )
    parser.add_argument(
        "--report-file-name",
        type=str,
        default=None,
        help=(
            "Base name for report files. When omitted, an auto-generated "
            "timestamped name of the form 'preflight-{NNODES}N-{YYYYMMDD-HHMMSS}' "
            "is used so each run writes to a fresh path and never overwrites "
            "or collides with stale leftovers."
        ),
    )
    parser.add_argument(
        "--disable-pdf",
        dest="save_pdf",
        action="store_false",
        help="Disable PDF report generation.",
    )

    # MORI preflight is an exclusive orchestration mode. It runs once per node
    # (not once per GPU), builds MORI inside a privileged runtime container,
    # and executes local plus optional cross-node all-gather correctness tests.
    mori = parser.add_argument_group("MORI runtime preflight")
    mori.add_argument(
        "--mori",
        action="store_true",
        help="Run MORI NIC diagnostics, runtime source build, and all-gather smoke. "
        "Cannot be combined with standard info/perf selectors.",
    )
    mori.add_argument(
        "--mori-base-image",
        default=("unifiedtrainingdockers.azurecr.io/utd/nightly:" "primus_the_rock_rocm7.15_20260728"),
        help="Base image pulled by MORI preflight.",
    )
    mori.add_argument(
        "--mori-repo",
        default="https://github.com/ROCm/mori.git",
        help="MORI git repository.",
    )
    mori.add_argument(
        "--mori-ref",
        default="12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4",
        help="Pinned MORI git revision to build.",
    )
    mori.add_argument(
        "--mori-max-jobs",
        type=int,
        default=32,
        help="Maximum parallel MORI build jobs.",
    )
    mori.add_argument(
        "--mori-smoke-numel",
        type=int,
        default=67108864,
        help="BF16 elements per rank in MORI all-gather smokes " "(default: 67108864 = 128 MiB).",
    )
    mori.add_argument(
        "--mori-keep-container",
        action="store_true",
        help="Keep the temporary build container after preflight.",
    )
    mori.add_argument(
        "--mori-log-dir",
        default=None,
        help="MORI phase-log directory. Defaults under --dump-path.",
    )
    mori.add_argument(
        "--mori-nodes",
        default=None,
        help="General multi-node target: comma-separated hosts, Slurm hostlist, "
        "or @file. Each node runs full diagnostics/build/local smoke before "
        "the N-node all-gather.",
    )
    mori.add_argument(
        "--mori-master-addr",
        default=None,
        help="Master IP for the N-node smoke (auto-detected when unset).",
    )
    mori.add_argument(
        "--mori-master-port",
        type=int,
        default=29610,
        help="torchrun rendezvous port for the N-node MORI smoke.",
    )
    mori.add_argument(
        "--mori-socket-ifname",
        default=None,
        help="Bootstrap interface for the N-node smoke.",
    )
    mori.add_argument(
        "--mori-gid-index",
        type=int,
        default=None,
        help="RCCL RoCEv2 GID index for the N-node smoke.",
    )
    return parser
