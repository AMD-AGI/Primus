###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Rank-aware Mooncake / SGLang / SpecForge supervision for online mode.

Capture ranks cannot ``execvp`` into ``specforge train``: they have to keep
Mooncake and the capture servers alive until every consumer finishes. Trainer
ranks wait for ``inference.ready`` then run ``--role consumer``.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from primus.backends.specforge.online_launch import (
    build_online_overrides,
    build_role_argv,
    consumer_done_paths,
    is_capture_rank,
    mooncake_env,
    mooncake_master_argv,
    nnodes,
    node_rank,
    online_settings,
    read_lines,
    read_status,
    ready_paths,
    resolve_head_ip,
    resolve_local_ip,
    server_gpu_group,
    server_urls,
    sglang_server_argv,
    sglang_url_path,
    trainer_node_rank,
    validate_online_identity,
    write_lines,
    write_status,
)
from primus.core.utils.module_utils import log_rank_0

# SpecForge producer is a 1-node job. Trainer NODE_RANK must be relative to
# the consumer group (0..trainer_nnodes), not the Slurm allocation, or
# SpecForge raises ``node_rank=N must be in [0, nnodes)``.
_SPECFORGE_RANK_KEYS = (
    "NODE_RANK",
    "NNODES",
    "SLURM_NODEID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NNODES",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_NPROCS",
    "GROUP_RANK",
    "ROLE_RANK",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "PET_NODE_RANK",
    "PET_NNODES",
)


def _merge_env(extra: dict[str, str], visible_devices: Optional[str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra)
    if visible_devices is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
        env["HIP_VISIBLE_DEVICES"] = visible_devices
    return env


def _specforge_child_env(
    extra: dict[str, str],
    visible_devices: Optional[str],
    *,
    specforge_node_rank: int = 0,
    specforge_nnodes: int = 1,
) -> dict[str, str]:
    env = _merge_env(extra, visible_devices)
    for key in _SPECFORGE_RANK_KEYS:
        env.pop(key, None)
    env["NODE_RANK"] = str(specforge_node_rank)
    env["NNODES"] = str(specforge_nnodes)
    return env


def _popen(argv: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab", buffering=0)
    return subprocess.Popen(
        argv,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _kill_group(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError, AttributeError):
        proc.terminate()
    deadline = time.time() + 10
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError, AttributeError):
            proc.kill()


def _wait_for_file(path: Path, timeout_s: int, peer: Optional[Path] = None, description: str = "") -> None:
    started = time.time()
    while not path.exists():
        if peer is not None and peer.exists():
            raise RuntimeError(
                f"[Primus:specforge] {description or path} aborted; peer status {read_status(peer)}"
            )
        if time.time() - started >= timeout_s:
            raise RuntimeError(f"[Primus:specforge] timed out waiting for {description or path}: {path}")
        time.sleep(2)


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 400
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _http_up(url: str, timeout: float = 1.0) -> bool:
    """True if an HTTP server answered, including 4xx/5xx.

    Mooncake's metadata GET of a dummy key returns 404; that still means the
    HTTP server is listening. ``urllib`` raises HTTPError (a URLError) for it,
    so ``_http_ok`` would treat a healthy master as down.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            int(response.status)
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def _tcp_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_mooncake(head_ip: str, settings: dict[str, Any], proc: subprocess.Popen, log_path: Path) -> None:
    metadata = f"http://{head_ip}:{settings['mooncake_http_port']}/metadata?key=specforge-health-check"
    started = time.time()
    while True:
        if _http_up(metadata) and _tcp_ok(head_ip, settings["mooncake_rpc_port"]):
            return
        if proc.poll() is not None:
            raise RuntimeError(f"[Primus:specforge] Mooncake exited; see {log_path}")
        if time.time() - started >= settings["start_timeout_s"]:
            raise RuntimeError("[Primus:specforge] Mooncake readiness timed out")
        time.sleep(1)


def _wait_sglang(
    advertise_ip: str, index: int, settings: dict[str, Any], proc: subprocess.Popen, log_path: Path
) -> None:
    url = f"http://{advertise_ip}:{int(settings['server_port']) + index}/health"
    started = time.time()
    while not _http_ok(url, timeout=2.0):
        if proc.poll() is not None:
            raise RuntimeError(f"[Primus:specforge] SGLang server {index} exited; see {log_path}")
        if time.time() - started >= settings["start_timeout_s"]:
            raise RuntimeError(f"[Primus:specforge] SGLang server {index} readiness timed out")
        time.sleep(5)


def _start_local_sglang(
    settings: dict[str, Any], head_ip: str, local_ip: str, log_root: Path, capture_rank: int
) -> list[subprocess.Popen]:
    servers: list[subprocess.Popen] = []
    for index in range(int(settings["server_count"])):
        gpus = server_gpu_group(settings["server_gpus"], index, int(settings["server_tp"]))
        server_env = _merge_env(mooncake_env(head_ip, settings, local_ip), gpus)
        log_path = log_root / f"sglang-server-{capture_rank}-{index}.log"
        proc = _popen(sglang_server_argv(index, settings), server_env, log_path)
        servers.append(proc)
        _wait_sglang(local_ip, index, settings, proc, log_path)
    write_lines(sglang_url_path(str(log_root), capture_rank), server_urls(local_ip, settings))
    log_rank_0(f"SGLang capture server(s) healthy on rank {capture_rank}")
    return servers


def _capture_stack_failed(
    head_ip: str,
    settings: dict[str, Any],
    master: Optional[subprocess.Popen],
    servers: list[subprocess.Popen],
) -> Optional[str]:
    """Return a reason if Mooncake or a local capture server process died.

    Do not probe SGLang ``/health`` here: once capture traffic starts the
    endpoint can time out on a live server (2p1c job 164088).
    """

    if master is not None and master.poll() is not None:
        return "Mooncake exited"
    for index, proc in enumerate(servers):
        if proc.poll() is not None:
            return f"SGLang server {index} exited"
    metadata = f"http://{head_ip}:{settings['mooncake_http_port']}/metadata?key=specforge-health-check"
    if not (_http_up(metadata) and _tcp_ok(head_ip, settings["mooncake_rpc_port"])):
        return "Mooncake stopped answering"
    return None


def _failed_consumer_status(done_files: list[Path]) -> Optional[str]:
    for path in done_files:
        if path.exists() and read_status(path) != "0":
            return read_status(path)
    return None


def _collect_server_urls(settings: dict[str, Any], timeout_s: int) -> list[str]:
    urls: list[str] = []
    for rank in range(int(settings["capture_nnodes"])):
        path = sglang_url_path(settings["run_root"], rank)
        _wait_for_file(path, timeout_s, description=f"SGLang URLs from capture rank {rank}")
        urls.extend(read_lines(path))
    return urls


def run_online(params: Any) -> None:
    from primus.backends.specforge.specforge_pretrain_trainer import (
        align_visible_devices,
        clear_partial_distributed_env,
    )

    rank = node_rank()
    nodes = nnodes()
    settings = online_settings(params)
    issues = validate_online_identity(settings, rank=rank, nodes=nodes)
    if issues:
        bullets = "\n".join(f"  - {item}" for item in issues)
        raise RuntimeError(f"[Primus:specforge] online identity failed:\n{bullets}")

    cleared = clear_partial_distributed_env()
    if cleared:
        log_rank_0(f"Cleared partial distributed env so SpecForge owns the launch: {cleared}")
    realigned = align_visible_devices()
    if realigned:
        old, new = realigned
        log_rank_0(f"Aligned HIP_VISIBLE_DEVICES with CUDA_VISIBLE_DEVICES: {old} -> {new}")

    if is_capture_rank(rank, settings):
        if rank == 0:
            _run_capture_node(params, settings)
        else:
            _run_capture_replica(params, settings, rank)
        return
    _run_trainer_node(params, settings, trainer_node_rank(rank, settings))


def _run_capture_node(params: Any, settings: dict[str, Any]) -> None:
    paths = ready_paths(settings["run_root"])
    head_ip = resolve_head_ip(interface=settings["bind_interface"])
    done_files = consumer_done_paths(settings["run_root"], int(settings["trainer_nnodes"]))

    if shutil.which("mooncake_master") is None:
        raise RuntimeError("[Primus:specforge] mooncake_master is not on PATH")

    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["head_ip"].exists():
        raise RuntimeError(
            f"[Primus:specforge] run root already has head.ip; choose a fresh RUN_ID: {paths['root']}"
        )

    result = 1
    master = None
    servers: list[subprocess.Popen] = []
    producer = None
    try:
        mc_env = os.environ.copy()
        mc_env.update(mooncake_env(head_ip, settings, head_ip))
        mc_env.setdefault("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(32 << 30))
        mc_env.setdefault("MOONCAKE_LOCAL_BUFFER_SIZE", str(1 << 30))
        master = _popen(mooncake_master_argv(settings), mc_env, paths["mooncake_log"])
        _wait_mooncake(head_ip, settings, master, paths["mooncake_log"])
        write_status(paths["head_ip"], head_ip)
        log_rank_0(f"Mooncake ready at {head_ip}:{settings['mooncake_rpc_port']}")

        servers = _start_local_sglang(settings, head_ip, head_ip, paths["root"], capture_rank=0)
        url_list = _collect_server_urls(settings, settings["start_timeout_s"])
        write_lines(paths["server_urls"], url_list)
        trainer_master = None
        if int(settings["trainer_nnodes"]) > 1:
            _wait_for_file(
                paths["trainer_ip"],
                settings["peer_timeout_s"],
                description="trainer master_addr",
            )
            trainer_master = read_status(paths["trainer_ip"])
        overrides = build_online_overrides(
            head_ip,
            settings,
            output_dir=getattr(params, "output_dir", None) or None,
            server_url_list=url_list,
            trainer_master_addr=trainer_master,
        )
        argv = build_role_argv(params, "producer", overrides)
        log_rank_0(f"SpecForge producer command: {' '.join(argv)}")
        write_status(paths["inference_ready"], head_ip)

        producer_env = _specforge_child_env(mooncake_env(head_ip, settings, head_ip), None)
        producer = _popen(argv, producer_env, paths["producer_log"])
        while producer.poll() is None:
            failed = _failed_consumer_status(done_files)
            if failed is not None:
                raise RuntimeError(f"[Primus:specforge] consumer failed with {failed}")
            stack = _capture_stack_failed(head_ip, settings, master, servers)
            if stack is not None:
                raise RuntimeError(f"[Primus:specforge] {stack} after producer start")
            time.sleep(2)
        if producer.returncode != 0:
            raise RuntimeError(f"[Primus:specforge] producer exited with status {producer.returncode}")
        producer = None

        for path in done_files:
            _wait_for_file(path, settings["peer_timeout_s"], description="consumer completion")
        failed = _failed_consumer_status(done_files)
        if failed is not None:
            raise RuntimeError(f"[Primus:specforge] consumer exited with status {failed}")
        result = 0
        log_rank_0("Online capture node finished; tearing down sidecars")
    finally:
        _kill_group(producer)
        for proc in servers:
            _kill_group(proc)
        _kill_group(master)
        write_status(paths["inference_done"], str(result))


def _run_capture_replica(_params: Any, settings: dict[str, Any], rank: int) -> None:
    paths = ready_paths(settings["run_root"])
    _wait_for_file(
        paths["head_ip"],
        settings["peer_timeout_s"],
        peer=paths["inference_done"],
        description="Mooncake ready (head.ip)",
    )
    head_ip = read_status(paths["head_ip"])
    local_ip = resolve_local_ip(interface=settings["bind_interface"])
    servers: list[subprocess.Popen] = []
    try:
        servers = _start_local_sglang(settings, head_ip, local_ip, paths["root"], capture_rank=rank)
        _wait_for_file(
            paths["inference_done"],
            settings["peer_timeout_s"],
            description="capture head teardown",
        )
        if read_status(paths["inference_done"]) != "0":
            raise RuntimeError(
                f"[Primus:specforge] capture head failed with {read_status(paths['inference_done'])}"
            )
    finally:
        for proc in servers:
            _kill_group(proc)


def _run_trainer_node(params: Any, settings: dict[str, Any], specforge_rank: int) -> None:
    paths = ready_paths(settings["run_root"])
    local_ip = resolve_local_ip(interface=settings["bind_interface"])
    trainer_nnodes = int(settings["trainer_nnodes"])
    if specforge_rank == 0:
        write_status(paths["trainer_ip"], local_ip)
    trainer_master = local_ip
    if trainer_nnodes > 1:
        _wait_for_file(paths["trainer_ip"], settings["peer_timeout_s"], description="trainer master_addr")
        trainer_master = read_status(paths["trainer_ip"])

    _wait_for_file(
        paths["inference_ready"],
        settings["peer_timeout_s"],
        peer=paths["inference_done"],
        description="inference readiness",
    )
    head_ip = read_status(paths["head_ip"] if paths["head_ip"].exists() else paths["inference_ready"])
    url_list = read_lines(paths["server_urls"]) if paths["server_urls"].exists() else None
    overrides = build_online_overrides(
        head_ip,
        settings,
        output_dir=getattr(params, "output_dir", None) or None,
        server_url_list=url_list,
        trainer_master_addr=trainer_master,
    )
    consumer_rank = specforge_rank if trainer_nnodes > 1 else None
    argv = build_role_argv(params, "consumer", overrides, node_rank=consumer_rank)
    log_rank_0(f"SpecForge consumer command: {' '.join(argv)}")

    consumer_dir = Path(settings["consumer_state_dir"])
    try:
        consumer_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"[Primus:specforge] consumer state already exists; choose a fresh "
            f"CONSUMER_STATE_DIR: {consumer_dir}"
        ) from exc

    result = 1
    consumer = None
    done_path = consumer_done_paths(settings["run_root"], trainer_nnodes)[specforge_rank]
    log_path = paths["consumer_log"]
    if trainer_nnodes > 1:
        log_path = paths["root"] / f"consumer-{specforge_rank}.log"
    try:
        env = _specforge_child_env(
            mooncake_env(head_ip, settings, local_ip),
            settings["trainer_gpus"],
            specforge_node_rank=specforge_rank,
            specforge_nnodes=trainer_nnodes,
        )
        consumer = _popen(argv, env, log_path)
        while consumer.poll() is None:
            if paths["inference_done"].exists() and read_status(paths["inference_done"]) != "0":
                raise RuntimeError(
                    f"[Primus:specforge] capture node failed with {read_status(paths['inference_done'])}"
                )
            time.sleep(2)
        result = int(consumer.returncode or 0)
        consumer = None
        if result != 0:
            raise RuntimeError(f"[Primus:specforge] consumer exited with status {result}")
        log_rank_0("Online trainer node finished")
    finally:
        _kill_group(consumer)
        write_status(done_path, str(result))
