"""Per-node entrypoint for the SLURM-mode preflight fan-out.

This module is invoked once per node by the head-node Pilot tool, via:

    srun --jobid=<id> -N <nnodes> --ntasks-per-node=1 \\
        python -m pilot.tools._preflight_node_entry \\
            --rdzv-endpoint <head>:<port> --rdzv-id pf_<id> --nnodes <nnodes>

Each instance:

  1. Reads ``$SLURM_NODEID`` to learn its own rank within the allocation.
  2. Counts locally visible GPUs (honoring HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES).
  3. Spawns ``torchrun --nnodes=<nnodes> --node-rank=$SLURM_NODEID
     --nproc-per-node=<n_gpus> --rdzv-* ...`` which in turn launches the
     ``pilot.tools._preflight_workers`` distributed measurement code on every
     local GPU.

This thin wrapper is necessary because:
  - ``srun`` runs *one task* per node (so SLURM env vars are clean), but
  - we need *one process per GPU* to drive AllReduce / AllToAll, which torchrun handles.

The wrapper is a CLI-only module; it has no public Python API. Exit code is
the exit code of the spawned torchrun (or 2 for argument errors, 3 for env errors).

Stdout: passes through torchrun's stdout (rank 0 will emit a single JSON line).
Stderr: passes through torchrun's stderr.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import NoReturn


def _eprint(*a: object) -> None:
    print(*a, file=sys.stderr)


def _detect_local_gpus() -> int:
    """Count GPUs visible to this node-entry process.

    Honors HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES. Falls back to
    rocm-smi / nvidia-smi enumeration. Returns 0 when no GPU is found.
    """
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if visible:
        return len([x for x in visible.split(",") if x.strip()])

    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0:
                return sum(1 for ln in r.stdout.splitlines() if "Device Name" in ln)
        except subprocess.TimeoutExpired:
            pass

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0:
                return sum(1 for ln in r.stdout.splitlines() if ln.startswith("GPU "))
        except subprocess.TimeoutExpired:
            pass

    return 0


def _resolve_node_rank() -> int:
    """Resolve this node's rank within the allocation.

    Priority:
      1. ``--node-rank`` CLI arg (already parsed by caller before this is called)
      2. ``$SLURM_NODEID`` (set by srun)
      3. ``$PMIX_RANK`` / ``$OMPI_COMM_WORLD_RANK`` (set by mpi-style launchers)
      4. fallback to 0 (assume single node)
    """
    for name in ("SLURM_NODEID", "PMIX_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(name)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def _main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="pilot.tools._preflight_node_entry",
        description="Per-node entry: bridges SLURM-allocated tasks into a torchrun process group.",
    )
    p.add_argument("--rdzv-endpoint", required=True,
                   help="<head_host>:<port> for c10d rendezvous.")
    p.add_argument("--rdzv-id", required=True,
                   help="Stable rendezvous id shared across all nodes.")
    p.add_argument("--nnodes", type=int, required=True,
                   help="Total number of nodes participating.")
    p.add_argument("--node-rank", type=int, default=None,
                   help="Optional override; defaults to $SLURM_NODEID.")
    p.add_argument("--nproc-per-node", type=int, default=None,
                   help="Optional override; defaults to count of locally visible GPUs.")
    p.add_argument("--ar-sizes-mb", default=None,
                   help="Comma-separated AllReduce sizes in MB. Forwarded as PILOT_AR_SIZES_MB.")
    p.add_argument("--a2a-sizes-mb", default=None,
                   help="Comma-separated AllToAll sizes in MB. Forwarded as PILOT_A2A_SIZES_MB.")
    args = p.parse_args(argv)

    node_rank = args.node_rank if args.node_rank is not None else _resolve_node_rank()
    if node_rank < 0 or node_rank >= args.nnodes:
        _eprint(
            f"[node_entry] resolved node_rank={node_rank} out of range "
            f"[0, {args.nnodes}); $SLURM_NODEID={os.environ.get('SLURM_NODEID')}"
        )
        return 2

    n_gpus = args.nproc_per_node if args.nproc_per_node is not None else _detect_local_gpus()
    if n_gpus < 1:
        _eprint(
            "[node_entry] no GPU visible on this node; "
            "check container GPU passthrough / HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES"
        )
        return 3

    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nnodes={args.nnodes}",
        f"--node-rank={node_rank}",
        f"--nproc-per-node={n_gpus}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint={args.rdzv_endpoint}",
        f"--rdzv-id={args.rdzv_id}",
        "-m", "pilot.tools._preflight_workers",
    ]

    env = os.environ.copy()
    if args.ar_sizes_mb:
        env["PILOT_AR_SIZES_MB"] = args.ar_sizes_mb
    if args.a2a_sizes_mb:
        env["PILOT_A2A_SIZES_MB"] = args.a2a_sizes_mb
    env.setdefault("OMP_NUM_THREADS", "8")

    _eprint(
        f"[node_entry rank={node_rank}/{args.nnodes}] launching torchrun "
        f"with nproc-per-node={n_gpus} rdzv={args.rdzv_endpoint} id={args.rdzv_id}"
    )

    try:
        r = subprocess.run(cmd, env=env)
    except FileNotFoundError as exc:
        _eprint(f"[node_entry] failed to spawn torchrun: {exc}")
        return 3
    except KeyboardInterrupt:
        _eprint("[node_entry] interrupted by user")
        return 130

    return r.returncode


def main() -> NoReturn:
    sys.exit(_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
