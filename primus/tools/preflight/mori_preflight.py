###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MORI per-node preflight and general N-node correctness orchestrator."""

from __future__ import annotations

import concurrent.futures
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRIMUS_ROOT = SCRIPT_DIR.parents[2]
FINGERPRINT_PREFIX = "[preflight] NODE_FINGERPRINT "


@dataclass
class NodeResult:
    node: str
    returncode: int
    fingerprint: str | None
    log_file: Path


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def resolve_nodes(spec: str | None) -> list[str]:
    if not spec:
        return [socket.gethostname().split(".")[0]]
    if spec.startswith("@"):
        return _dedupe(Path(spec[1:]).read_text().splitlines())
    if "," in spec:
        return _dedupe(spec.split(","))
    if shutil.which("scontrol"):
        result = subprocess.run(
            ["scontrol", "show", "hostnames", spec],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        nodes = _dedupe(result.stdout.splitlines())
        if result.returncode == 0 and nodes:
            return nodes
    return [spec]


def is_local(node: str) -> bool:
    names = {
        "localhost",
        "127.0.0.1",
        socket.gethostname(),
        socket.gethostname().split(".")[0],
        socket.getfqdn(),
    }
    return node in names


def preflight_container_name(node: str) -> str:
    short_node = node.split(".")[0]
    return f"primus_mori_preflight_{os.environ.get('USER', 'user')}_{short_node}"


def shell_join_env(env: dict[str, str], command: list[str]) -> str:
    assignments = [f"{name}={value}" for name, value in env.items()]
    return shlex.join(["env", *assignments, *command])


def local_preflight_command(
    args: Any,
    node: str,
    node_log_dir: Path,
    repo_root: Path,
) -> list[str]:
    env = {
        "BASE_IMAGE": args.mori_base_image,
        "MORI_REPO": args.mori_repo,
        "MORI_REF": args.mori_ref,
        "MAX_JOBS": str(args.mori_max_jobs),
        "SMOKE_NUMEL": str(args.mori_smoke_numel),
        # Reuse these containers for the N-node smoke and remove them after.
        "KEEP_CONTAINER": "1",
        "CONTAINER_NAME": preflight_container_name(node),
        "LOG_DIR": str(node_log_dir),
        "PRIMUS_ROOT": str(repo_root),
    }
    helper = str(repo_root / "primus" / "tools" / "preflight" / "mori_preflight.sh")
    command = ["bash", helper]
    if is_local(node):
        return ["env", *[f"{k}={v}" for k, v in env.items()], *command]

    remote_command = f"cd {shlex.quote(str(repo_root))} && {shell_join_env(env, command)}"
    return ["ssh", "-o", "BatchMode=yes", node, remote_command]


def run_node_preflight(
    args: Any,
    node: str,
    root_log_dir: Path,
    repo_root: Path,
) -> NodeResult:
    short_node = node.split(".")[0]
    node_log_dir = root_log_dir / "nodes" / short_node
    node_log_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = node_log_dir / "launcher.log"
    command = local_preflight_command(args, node, node_log_dir, repo_root)
    fingerprint = None

    with launcher_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{short_node}] {line}", end="", flush=True)
            if line.startswith(FINGERPRINT_PREFIX):
                fingerprint = line[len(FINGERPRINT_PREFIX) :].strip()
        returncode = process.wait()

    return NodeResult(node, returncode, fingerprint, launcher_log)


def remote_output(node: str, command: str) -> str:
    if is_local(node):
        argv = ["bash", "-lc", command]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", node, command]
    result = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{node}: command failed ({result.returncode}): " f"{result.stderr.strip() or command}"
        )
    return result.stdout.strip()


def probe_network(
    node: str,
    interface_override: str | None,
    gid_override: int | None,
    address_override: str | None,
) -> tuple[str, int, str]:
    if interface_override and gid_override is not None and address_override:
        return interface_override, gid_override, address_override

    command = f"""
interface={shlex.quote(interface_override or "")}
gid_index={shlex.quote("" if gid_override is None else str(gid_override))}
master_addr={shlex.quote(address_override or "")}

if [ -z "$interface" ]; then
    if ip -o -4 addr show dev fenic scope global >/dev/null 2>&1; then
        interface=fenic
    else
        for path in /sys/class/infiniband/*/device/net/*; do
            [ -e "$path" ] || continue
            candidate=${{path##*/}}
            if ip -o -4 addr show dev "$candidate" scope global >/dev/null 2>&1; then
                interface=$candidate
                break
            fi
        done
    fi
fi
[ -n "$interface" ] || {{ echo "no RDMA interface with IPv4" >&2; exit 1; }}

if [ -z "$gid_index" ]; then
    gid_index=0
    for dev in /sys/class/infiniband/*; do
        [ -d "$dev" ] || continue
        for path in "$dev"/ports/1/gids/*; do
            [ -f "$path" ] || continue
            idx=${{path##*/}}
            gid=$(cat "$path")
            type=$(cat "$dev/ports/1/gid_attrs/types/$idx" 2>/dev/null || true)
            case "$type:$gid" in
                "RoCE v2:"*":ffff:"*) gid_index=$idx; break 2 ;;
            esac
        done
    done
fi

if [ -z "$master_addr" ]; then
    master_addr=$(ip -o -4 addr show dev "$interface" scope global |
        awk 'NR==1{{split($4,a,"/"); print a[1]}}')
fi
[ -n "$master_addr" ] || {{ echo "no IPv4 address on $interface" >&2; exit 1; }}
printf '%s\\t%s\\t%s\\n' "$interface" "$gid_index" "$master_addr"
"""
    fields = remote_output(node, command).split("\t")
    if len(fields) != 3:
        raise RuntimeError(f"{node}: malformed network probe output: {fields}")
    return fields[0], int(fields[1]), fields[2]


def validate_fingerprints(results: list[NodeResult]) -> None:
    missing = [result.node for result in results if not result.fingerprint]
    if missing:
        raise RuntimeError(f"Missing MORI fingerprint from nodes: {', '.join(missing)}")
    grouped: dict[str, list[str]] = {}
    for result in results:
        assert result.fingerprint is not None
        grouped.setdefault(result.fingerprint, []).append(result.node)
    if len(grouped) != 1:
        detail = "; ".join(f"{nodes}: {fingerprint}" for fingerprint, nodes in grouped.items())
        raise RuntimeError(f"MORI node-stack mismatch: {detail}")


def run_multinode(args: Any, nodes: list[str], log_dir: Path) -> int:
    master_node = nodes[0]
    interface, gid_index, master_addr = probe_network(
        master_node,
        args.mori_socket_ifname,
        args.mori_gid_index,
        args.mori_master_addr,
    )
    multinode_log_dir = log_dir / "multinode"
    multinode_log_dir.mkdir(parents=True, exist_ok=True)

    def launch(node: str, rank: int) -> tuple[int, Path]:
        torchrun = shlex.join(
            [
                "torchrun",
                f"--nnodes={len(nodes)}",
                "--nproc_per_node=8",
                f"--node_rank={rank}",
                f"--master_addr={master_addr}",
                f"--master_port={args.mori_master_port}",
                "/src/primus/runner/helpers/mori/multinode_allgather_smoke.py",
                "--numel",
                str(args.mori_smoke_numel),
            ]
        )
        docker_command = [
            "docker",
            "exec",
            "-e",
            "PYTHONPATH=/src/primus",
            "-e",
            f"NCCL_SOCKET_IFNAME={interface}",
            "-e",
            f"NCCL_IB_GID_INDEX={gid_index}",
            preflight_container_name(node),
            "bash",
            "-lc",
            "export LD_LIBRARY_PATH=/opt/mori-host-libs:${LD_LIBRARY_PATH}; " f"exec {torchrun}",
        ]
        command = (
            docker_command
            if is_local(node)
            else ["ssh", "-o", "BatchMode=yes", node, shlex.join(docker_command)]
        )
        rank_log = multinode_log_dir / f"node-{rank}-{node}.log"
        with rank_log.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode, rank_log

    print(
        f"[MORI:Preflight] launching {len(nodes)} nodes via {master_node} "
        f"({master_addr}, {interface}, gid={gid_index})",
        flush=True,
    )
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = {}
        for rank, node in enumerate(nodes):
            futures[executor.submit(launch, node, rank)] = node
            if rank == 0:
                time.sleep(1)
        for future, node in futures.items():
            returncode, rank_log = future.result()
            if returncode:
                failures.append((node, rank_log))
    for node, rank_log in failures:
        print(f"[MORI:Preflight] FAIL {node}: {rank_log}", file=sys.stderr)
    return int(bool(failures))


def remove_preflight_containers(nodes: list[str]) -> None:
    def remove(node: str) -> None:
        name = preflight_container_name(node)
        remote_output(
            node,
            f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        list(executor.map(remove, nodes))


def run_orchestrator(args: Any, repo_root: Path, log_dir: Path) -> int:
    nodes = resolve_nodes(args.mori_nodes)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MORI:Preflight] nodes ({len(nodes)}): {', '.join(nodes)}")
    print(f"[MORI:Preflight] logs: {log_dir}")

    try:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
            future_map = {
                executor.submit(run_node_preflight, args, node, log_dir, repo_root): node for node in nodes
            }
            for future in concurrent.futures.as_completed(future_map):
                results.append(future.result())

        results.sort(key=lambda result: nodes.index(result.node))
        failed = [result for result in results if result.returncode != 0]
        if failed:
            for result in failed:
                print(
                    f"[MORI:Preflight] FAIL {result.node}: {result.log_file}",
                    file=sys.stderr,
                )
            return 1

        validate_fingerprints(results)
        print("[MORI:Preflight] all node fingerprints match", flush=True)
        if len(nodes) > 1:
            return run_multinode(args, nodes, log_dir)
        return 0
    finally:
        if not args.mori_keep_container:
            remove_preflight_containers(nodes)


def _validate_mori_mode(args: Any, extra_args: Sequence[str]) -> int | None:
    if extra_args:
        print(
            f"[Primus:Preflight] ERROR: unknown MORI arguments: {list(extra_args)}",
            file=sys.stderr,
        )
        return 2

    incompatible = []
    for attr, flag in (
        ("check_host", "--host"),
        ("check_gpu", "--gpu"),
        ("check_network", "--network"),
        ("perf_test", "--perf-test"),
        ("plot", "--plot"),
        ("tests", "--tests"),
        ("comm_sizes_mb", "--comm-sizes-mb"),
        ("intra_comm_sizes_mb", "--intra-comm-sizes-mb"),
        ("inter_comm_sizes_mb", "--inter-comm-sizes-mb"),
        ("intra_group_sizes", "--intra-group-sizes"),
        ("inter_group_sizes", "--inter-group-sizes"),
        ("ring_p2p_sizes_mb", "--ring-p2p-sizes-mb"),
        ("quick", "--quick"),
    ):
        if getattr(args, attr, None):
            incompatible.append(flag)
    if getattr(args, "split_nodes_subgroup", True) is False:
        incompatible.append("--no-split-nodes-subgroup")
    if incompatible:
        print(
            "[Primus:Preflight] ERROR: --mori cannot be combined with " + ", ".join(incompatible),
            file=sys.stderr,
        )
        return 2
    return None


def run_mori_preflight(args: Any, extra_args: Sequence[str] = ()) -> int:
    validation_rc = _validate_mori_mode(args, extra_args)
    if validation_rc is not None:
        return validation_rc

    repo_root = Path(
        os.environ.get(
            "PRIMUS_PATH",
            str(DEFAULT_PRIMUS_ROOT),
        )
    ).resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_host = socket.gethostname().split(".")[0]
    report_name = args.report_file_name or f"mori-preflight-{short_host}-{timestamp}"
    log_dir = Path(args.mori_log_dir or (Path(args.dump_path) / report_name)).resolve()

    print(f"[Primus:Preflight] MORI logs: {log_dir}", flush=True)
    return run_orchestrator(args, repo_root, log_dir)
