###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Rank-aware Mooncake / SGLang / SpecForge supervision for online mode.

Rank 0 cannot ``execvp`` into ``specforge train``: it has to keep Mooncake and
the capture server alive until the consumer finishes. Rank 1 waits for
``inference.ready`` then runs ``--role consumer``.
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
    mooncake_env,
    mooncake_master_argv,
    nnodes,
    node_rank,
    online_settings,
    read_status,
    ready_paths,
    resolve_routable_ip,
    server_gpu_group,
    sglang_server_argv,
    validate_online_identity,
    write_status,
)
from primus.core.utils.module_utils import log_rank_0

# SpecForge producer/consumer are 1-node jobs. Primus NODE_RANK=1 on the
# trainer host must not leak in, or SpecForge raises
# ``node_rank=1 must be in [0, 1)``.
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


def _specforge_child_env(extra: dict[str, str], visible_devices: Optional[str]) -> dict[str, str]:
    env = _merge_env(extra, visible_devices)
    for key in _SPECFORGE_RANK_KEYS:
        env.pop(key, None)
    env["NODE_RANK"] = "0"
    env["NNODES"] = "1"
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
    head_ip: str, index: int, settings: dict[str, Any], proc: subprocess.Popen, log_path: Path
) -> None:
    url = f"http://{head_ip}:{int(settings['server_port']) + index}/health"
    started = time.time()
    while not _http_ok(url, timeout=2.0):
        if proc.poll() is not None:
            raise RuntimeError(f"[Primus:specforge] SGLang server {index} exited; see {log_path}")
        if time.time() - started >= settings["start_timeout_s"]:
            raise RuntimeError(f"[Primus:specforge] SGLang server {index} readiness timed out")
        time.sleep(5)


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

    if rank == 0:
        _run_capture_node(params, settings)
        return
    _run_trainer_node(params, settings)


def _run_capture_node(params: Any, settings: dict[str, Any]) -> None:
    paths = ready_paths(settings["run_root"])
    head_ip = resolve_routable_ip(interface=settings["bind_interface"])
    overrides = build_online_overrides(
        head_ip,
        settings,
        output_dir=getattr(params, "output_dir", None) or None,
    )
    extra = list(overrides)
    # Trainer-local output_dir on the shared run root wins over a leftover alias.
    argv = build_role_argv(params, "producer", extra)
    log_rank_0(f"SpecForge producer command: {' '.join(argv)}")

    if shutil.which("mooncake_master") is None:
        raise RuntimeError("[Primus:specforge] mooncake_master is not on PATH")

    try:
        paths["root"].mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"[Primus:specforge] run root already exists; choose a fresh RUN_ID: {paths['root']}"
        ) from exc

    write_status(paths["head_ip"], head_ip)
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
        log_rank_0(f"Mooncake ready at {head_ip}:{settings['mooncake_rpc_port']}")

        for index in range(int(settings["server_count"])):
            gpus = server_gpu_group(settings["server_gpus"], index, int(settings["server_tp"]))
            server_env = _merge_env(mooncake_env(head_ip, settings, head_ip), gpus)
            log_path = paths["root"] / f"sglang-server-{index}.log"
            proc = _popen(sglang_server_argv(index, settings), server_env, log_path)
            servers.append(proc)
            _wait_sglang(head_ip, index, settings, proc, log_path)
        log_rank_0("SGLang capture server(s) healthy")
        write_status(paths["inference_ready"], head_ip)

        producer_env = _specforge_child_env(mooncake_env(head_ip, settings, head_ip), None)
        producer = _popen(argv, producer_env, paths["producer_log"])
        while producer.poll() is None:
            if paths["consumer_done"].exists() and read_status(paths["consumer_done"]) != "0":
                raise RuntimeError(
                    f"[Primus:specforge] consumer failed with {read_status(paths['consumer_done'])}"
                )
            time.sleep(2)
        if producer.returncode != 0:
            raise RuntimeError(f"[Primus:specforge] producer exited with status {producer.returncode}")
        producer = None

        _wait_for_file(
            paths["consumer_done"],
            settings["peer_timeout_s"],
            description="consumer completion",
        )
        consumer_result = read_status(paths["consumer_done"])
        if consumer_result != "0":
            raise RuntimeError(f"[Primus:specforge] consumer exited with status {consumer_result}")
        result = 0
        log_rank_0("Online capture node finished; tearing down sidecars")
    finally:
        _kill_group(producer)
        for proc in servers:
            _kill_group(proc)
        _kill_group(master)
        write_status(paths["inference_done"], str(result))


def _run_trainer_node(params: Any, settings: dict[str, Any]) -> None:
    paths = ready_paths(settings["run_root"])
    _wait_for_file(
        paths["inference_ready"],
        settings["peer_timeout_s"],
        peer=paths["inference_done"],
        description="inference readiness",
    )
    head_ip = read_status(paths["head_ip"] if paths["head_ip"].exists() else paths["inference_ready"])
    local_ip = resolve_routable_ip(interface=settings["bind_interface"])
    overrides = build_online_overrides(
        head_ip,
        settings,
        output_dir=getattr(params, "output_dir", None) or None,
    )
    argv = build_role_argv(params, "consumer", overrides)
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
    try:
        env = _specforge_child_env(mooncake_env(head_ip, settings, local_ip), settings["trainer_gpus"])
        consumer = _popen(argv, env, paths["consumer_log"])
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
        write_status(paths["consumer_done"], str(result))
