###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Bare-metal Cluster Sphere tools (no torchrun / torch.distributed).

See docs: ``docs/preflight.md`` — Slurm execution pipelines.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

from primus.tools.preflight.cluster_sphere import verbs_bw


def _cmd_env(args: argparse.Namespace) -> int:
    from primus.tools.preflight.cluster_sphere.env_recommender import collect_cluster_sphere_env_findings
    from primus.tools.preflight.cluster_sphere.report import emit_cluster_sphere_env_markdown

    hostname = os.environ.get("HOSTNAME") or socket.gethostname()
    findings = collect_cluster_sphere_env_findings()
    if args.markdown:
        sys.stdout.write(emit_cluster_sphere_env_markdown(hostname, findings))
        return 0

    for fin in findings:
        print(f"[{fin.level}] {fin.message}", file=sys.stderr)
        if fin.details:
            det = fin.details
            w = det.get("warnings")
            if isinstance(w, list):
                for line in w:
                    print(f"  - {line}", file=sys.stderr)
    return 0


def _cmd_verbs_server(args: argparse.Namespace) -> int:
    ib_dev = args.device or verbs_bw.first_ib_device_name()
    if not ib_dev:
        print("No RDMA device under /sys/class/infiniband; pass --device.", file=sys.stderr)
        return 2
    port = args.port if args.port is not None else verbs_bw.default_port()
    cmd = verbs_bw.ib_write_bw_server_cmd(ib_dev, port)
    print(f"Running (server): {' '.join(cmd)}", file=sys.stderr)
    if shutil.which("ib_write_bw") is None:
        print("ib_write_bw not found; install perftest.", file=sys.stderr)
        return 2
    os.execvp(cmd[0], cmd)


def _cmd_verbs_client(args: argparse.Namespace) -> int:
    ib_dev = args.device or verbs_bw.first_ib_device_name()
    if not ib_dev:
        print("No RDMA device; pass --device.", file=sys.stderr)
        return 2
    port = args.port if args.port is not None else verbs_bw.default_port()
    cmd = verbs_bw.ib_write_bw_client_cmd(ib_dev, args.server_ip, port)
    print(f"Running (client): {' '.join(cmd)}", file=sys.stderr)
    if shutil.which("ib_write_bw") is None:
        print("ib_write_bw not found; install perftest.", file=sys.stderr)
        return 2

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(text)
    peak = verbs_bw.parse_peak_gbps(text)
    if peak is not None:
        print(f"\nParsed peak (best-effort): {peak:.2f} Gb/sec", file=sys.stderr)
    return proc.returncode


def _cmd_verbs_pair(args: argparse.Namespace) -> int:
    """
    Single Slurm step: task 0 runs ib_write_bw server, task 1 sleeps then runs client.
    Requires: ``srun -N2 -n2`` (or equivalent) and ``SERVER_RDMA_IP`` for the server’s RDMA address.
    """
    raw = os.environ.get("SLURM_PROCID")
    if raw is None or raw == "":
        print(
            "verbs-pair must run under Slurm with exactly 2 tasks, e.g.\n"
            "  export SERVER_RDMA_IP=<server_host_RDMA_IP>\n"
            "  srun -N2 -n2 -t 00:30:00 env PYTHONPATH=... SERVER_RDMA_IP=... \\\n"
            "    python3 -m primus.tools.preflight.cluster_sphere verbs-pair -d mlx5_0",
            file=sys.stderr,
        )
        return 2

    try:
        procid = int(raw)
    except ValueError:
        print(f"Invalid SLURM_PROCID={raw!r}", file=sys.stderr)
        return 2

    if procid == 0:
        return _cmd_verbs_server(args)

    if procid == 1:
        delay = float(getattr(args, "client_delay", 15.0))
        if delay > 0:
            time.sleep(delay)
        ip = os.environ.get("SERVER_RDMA_IP", "").strip()
        if not ip:
            print(
                "Task 1: set SERVER_RDMA_IP to the server host’s RDMA-accessible address.",
                file=sys.stderr,
            )
            return 2
        args.server_ip = ip
        return _cmd_verbs_client(args)

    print(
        f"verbs-pair expects SLURM_PROCID 0 or 1 (two tasks); got {procid}.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster Sphere bare-metal helpers (no torchrun / no torch.distributed).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("env", help="Print RDMA / NCCL export recommendations for this host.")
    p_env.add_argument("--markdown", action="store_true", help="Emit Markdown section to stdout.")
    p_env.set_defaults(func=_cmd_env)

    p_srv = sub.add_parser("verbs-server", help="Run ib_write_bw server (replaces current process).")
    p_srv.add_argument("--device", "-d", help="RDMA device name (default: first in /sys/class/infiniband)")
    p_srv.add_argument(
        "--port",
        "-p",
        type=int,
        help=f"TCP port (default: env {verbs_bw.DEFAULT_PORT_ENV} or 2000)",
    )
    p_srv.set_defaults(func=_cmd_verbs_server)

    p_cli = sub.add_parser("verbs-client", help="Run ib_write_bw client toward verbs-server.")
    p_cli.add_argument("--server-ip", required=True, help="Server IP/hostname on the RDMA-accessible network")
    p_cli.add_argument("--device", "-d", help="RDMA device name (default: first IB device)")
    p_cli.add_argument(
        "--port",
        "-p",
        type=int,
        help=f"TCP port (must match server; default env {verbs_bw.DEFAULT_PORT_ENV} or 2000)",
    )
    p_cli.add_argument("--timeout", type=int, default=120, help="Run timeout (seconds)")
    p_cli.set_defaults(func=_cmd_verbs_client)

    p_pair = sub.add_parser(
        "verbs-pair",
        help="Slurm: run server on task 0 and client on task 1 (use srun -N2 -n2; set SERVER_RDMA_IP).",
    )
    p_pair.add_argument("--device", "-d", help="RDMA device name (default: first IB device)")
    p_pair.add_argument(
        "--port",
        "-p",
        type=int,
        help=f"TCP port (default: env {verbs_bw.DEFAULT_PORT_ENV} or 2000)",
    )
    p_pair.add_argument(
        "--client-delay",
        type=float,
        default=15.0,
        help="Seconds to wait on client task before connecting (default: 15)",
    )
    p_pair.add_argument("--timeout", type=int, default=120, help="Client ib_write_bw timeout (seconds)")
    p_pair.set_defaults(func=_cmd_verbs_pair)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
