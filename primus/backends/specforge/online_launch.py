###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Pure helpers for 2-node online SpecForge (Mooncake + SGLang + roles).

Cluster IPs stay out of git YAML. After Slurm allocates, Primus resolves a
routable HEAD_IP and renders Hydra overrides. Rank dispatch and sidecar
lifetime live in ``online_supervisor``.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from primus.backends.specforge.argument_builder import (
    build_specforge_argv,
    flatten_overrides,
)

DEFAULT_MOONCAKE_RPC_PORT = 35551
DEFAULT_MOONCAKE_HTTP_PORT = 35880
DEFAULT_MOONCAKE_METRICS_PORT = 35903
DEFAULT_SERVER_PORT = 30000
# Must match configs/qwen3.5-4b-dflash.json dflash_config.target_layer_ids.
DEFAULT_CAPTURE_LAYER_IDS = (1, 8, 15, 22, 29)
DEFAULT_LEASE_TTL_MS = 500
DEFAULT_PROTOCOL = "tcp"
LOOPBACK_PREFIXES = ("127.", "0.")


def node_rank(env: Optional[Mapping[str, str]] = None) -> int:
    environ = os.environ if env is None else env
    raw = environ.get("NODE_RANK") or environ.get("SLURM_NODEID") or environ.get("SLURM_PROCID") or "0"
    return int(str(raw).strip())


def nnodes(env: Optional[Mapping[str, str]] = None) -> int:
    environ = os.environ if env is None else env
    raw = environ.get("NNODES") or environ.get("SLURM_NNODES") or environ.get("SLURM_JOB_NUM_NODES") or "1"
    return int(str(raw).strip())


def parse_csv_devices(value: Any, default: str = "0") -> str:
    if value is None or str(value).strip() == "":
        return default
    text = str(value).replace(" ", "")
    return text


def parse_layer_ids(value: Any) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return DEFAULT_CAPTURE_LAYER_IDS
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    parts = re.split(r"[,\s]+", str(value).strip())
    return tuple(int(p) for p in parts if p)


def parse_extra_args(value: Any) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return str(value).split()


def online_settings(params: Any, env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Flatten ``specforge_online`` plus a few env aliases used at launch."""

    environ = os.environ if env is None else env
    raw = flatten_overrides(getattr(params, "specforge_online", None))
    run_id = raw.get("run_id") or environ.get("RUN_ID") or environ.get("DISAGG_STORE_ID")
    run_root = raw.get("run_root") or environ.get("RUN_ROOT") or environ.get("DISAGG_RUN_ROOT")
    consumer_state = (
        raw.get("consumer_state_dir")
        or environ.get("CONSUMER_STATE_DIR")
        or environ.get("DISAGG_CONSUMER_STATE_DIR")
    )
    server_count = int(raw.get("server_count") or 1)
    server_tp = int(raw.get("server_tp") or 1)
    server_gpus = parse_csv_devices(raw.get("server_gpus"), default="0")
    trainer_nproc = int(
        raw.get("trainer_nproc") or raw.get("nproc_per_node") or environ.get("NPROC_PER_NODE") or 1
    )
    trainer_gpus = parse_csv_devices(raw.get("trainer_gpus"), default="0")
    protocol = str(raw.get("mooncake_protocol") or environ.get("MOONCAKE_PROTOCOL") or DEFAULT_PROTOCOL)
    lease = raw.get("mooncake_lease_ttl_ms") or environ.get("MOONCAKE_DEFAULT_KV_LEASE_TTL")
    lease_ttl = int(lease) if lease not in (None, "", "null") else DEFAULT_LEASE_TTL_MS
    target = (
        raw.get("target_model_path")
        or flatten_overrides(getattr(params, "specforge_overrides", None)).get("model.target_model_path")
        or environ.get("TARGET_MODEL")
        or "Qwen/Qwen3.5-4B"
    )
    extra = parse_extra_args(raw.get("sglang_extra_args"))
    if not extra:
        extra = ["--attention-backend", "aiter", "--disable-radix-cache"]
    return {
        "run_id": str(run_id).strip() if run_id else "",
        "run_root": str(run_root).strip() if run_root else "",
        "consumer_state_dir": str(consumer_state).strip() if consumer_state else "",
        "server_count": server_count,
        "server_tp": server_tp,
        "server_gpus": server_gpus,
        "server_port": int(raw.get("server_port") or DEFAULT_SERVER_PORT),
        "server_mem_fraction": str(raw.get("server_mem_fraction") or "0.85"),
        "trainer_nproc": trainer_nproc,
        "trainer_gpus": trainer_gpus,
        "mooncake_protocol": protocol,
        "mooncake_rpc_port": int(raw.get("mooncake_rpc_port") or DEFAULT_MOONCAKE_RPC_PORT),
        "mooncake_http_port": int(raw.get("mooncake_http_port") or DEFAULT_MOONCAKE_HTTP_PORT),
        "mooncake_metrics_port": int(raw.get("mooncake_metrics_port") or DEFAULT_MOONCAKE_METRICS_PORT),
        "mooncake_lease_ttl_ms": lease_ttl,
        "capture_layer_ids": parse_layer_ids(raw.get("capture_layer_ids")),
        "target_model_path": str(target),
        "sglang_extra_args": extra,
        "bind_interface": str(raw.get("bind_interface") or environ.get("PRIMUS_SPECFORGE_BIND_IFACE") or ""),
        "start_timeout_s": int(raw.get("start_timeout_s") or environ.get("START_TIMEOUT_S") or 1800),
        "peer_timeout_s": int(raw.get("peer_timeout_s") or environ.get("PEER_TIMEOUT_S") or 1800),
    }


def validate_online_identity(settings: Mapping[str, Any], *, rank: int, nodes: int) -> list[str]:
    issues: list[str] = []
    if nodes != 2:
        issues.append(f"online mode expects NNODES=2 (one capture, one trainer); got {nodes}")
    if rank not in (0, 1):
        issues.append(f"online mode expects NODE_RANK 0 or 1; got {rank}")
    if not settings.get("run_id"):
        issues.append("set RUN_ID / specforge_online.run_id (letters, digits, '.', '_' or '-')")
    elif not re.fullmatch(r"[A-Za-z0-9._-]+", str(settings["run_id"])):
        issues.append(f"invalid RUN_ID: {settings['run_id']}")
    if not settings.get("run_root"):
        issues.append("set RUN_ROOT / specforge_online.run_root on shared storage")
    if not settings.get("consumer_state_dir"):
        issues.append("set CONSUMER_STATE_DIR / specforge_online.consumer_state_dir on trainer-local disk")
    expected_server = settings["server_count"] * settings["server_tp"]
    got_server = len([p for p in str(settings["server_gpus"]).split(",") if p])
    if got_server != expected_server:
        issues.append(
            f"server_gpus must contain server_count*server_tp={expected_server} devices; got {got_server}"
        )
    got_trainer = len([p for p in str(settings["trainer_gpus"]).split(",") if p])
    if got_trainer != settings["trainer_nproc"]:
        issues.append(
            f"trainer_gpus must contain trainer_nproc={settings['trainer_nproc']} devices; got {got_trainer}"
        )
    return issues


def _first_non_loopback(candidates: Sequence[str]) -> Optional[str]:
    for raw in candidates:
        ip = str(raw).strip()
        if not ip:
            continue
        if any(ip.startswith(prefix) for prefix in LOOPBACK_PREFIXES):
            continue
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
            return ip
    return None


def interface_ipv4(name: str) -> Optional[str]:
    """IPv4 of a Linux netdev, or None if missing (Windows/unit tests)."""

    if not name:
        return None

    if not Path(f"/sys/class/net/{name}").is_dir():
        return None
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "dev", name],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
    return match.group(1) if match else None


def hostname_ipv4s() -> list[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    seen: list[str] = []
    for info in infos:
        ip = info[4][0]
        if ip not in seen:
            seen.append(ip)
    return seen


def resolve_routable_ip(
    env: Optional[Mapping[str, str]] = None,
    *,
    interface: str = "",
    interface_ip: Optional[str] = None,
    hostname_ips: Optional[Sequence[str]] = None,
) -> str:
    """Routable IPv4 for Mooncake/SGLang advertise. Never a hostname."""

    environ = os.environ if env is None else env
    for key in ("PRIMUS_SPECFORGE_HEAD_IP", "HEAD_IP"):
        override = environ.get(key)
        if override and str(override).strip():
            return str(override).strip()
    iface = interface or environ.get("PRIMUS_SPECFORGE_BIND_IFACE") or ""
    if interface_ip is None:
        interface_ip = interface_ipv4(iface)
    if interface_ip:
        return interface_ip
    if hostname_ips is None:
        hostname_ips = hostname_ipv4s()
    picked = _first_non_loopback(hostname_ips)
    if picked:
        return picked
    raise RuntimeError(
        "[Primus:specforge] could not resolve a routable IPv4; "
        "set PRIMUS_SPECFORGE_HEAD_IP / HEAD_IP or PRIMUS_SPECFORGE_BIND_IFACE"
    )


def server_urls(head_ip: str, settings: Mapping[str, Any]) -> list[str]:
    port0 = int(settings["server_port"])
    return [f"http://{head_ip}:{port0 + i}" for i in range(int(settings["server_count"]))]


def hydra_quoted_list(values: Sequence[str]) -> str:
    inner = ",".join(f'"{v}"' for v in values)
    return f"[{inner}]"


def build_online_overrides(
    head_ip: str, settings: Mapping[str, Any], output_dir: Optional[str] = None
) -> list[str]:
    """Hydra overrides that rewrite loopback YAML to the allocated HEAD_IP."""

    run_root = settings["run_root"]
    out = output_dir or f"{run_root}/output"
    http_port = settings["mooncake_http_port"]
    rpc_port = settings["mooncake_rpc_port"]
    return [
        f"model.target_model_path={settings['target_model_path']}",
        f"run_id={settings['run_id']}",
        f"output_dir={out}",
        "deployment.trainer.nnodes=1",
        f"deployment.trainer.nproc_per_node={settings['trainer_nproc']}",
        f"deployment.disaggregated.control_dir={run_root}/control",
        f"deployment.disaggregated.consumer_state_dir={settings['consumer_state_dir']}",
        f"deployment.disaggregated.store_id={settings['run_id']}",
        f"deployment.disaggregated.server_urls={hydra_quoted_list(server_urls(head_ip, settings))}",
        f"deployment.disaggregated.mooncake_metadata_server=http://{head_ip}:{http_port}/metadata",
        f"deployment.disaggregated.mooncake_master_server_addr={head_ip}:{rpc_port}",
        f"deployment.disaggregated.mooncake_protocol={settings['mooncake_protocol']}",
        f"deployment.disaggregated.idle_timeout_s={settings['peer_timeout_s']}",
        f"deployment.disaggregated.peer_wait_timeout_s={settings['peer_timeout_s']}",
    ]


def build_role_argv(params: Any, role: str, extra_overrides: Sequence[str]) -> list[str]:
    """``specforge train --config ... --role {{producer,consumer}}`` plus Hydra overrides."""

    argv = build_specforge_argv(params, extra_overrides=list(extra_overrides), role=role)
    return argv


def mooncake_master_argv(settings: Mapping[str, Any]) -> list[str]:
    argv = [
        "mooncake_master",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        f"--rpc_port={settings['mooncake_rpc_port']}",
        f"--http_metadata_server_port={settings['mooncake_http_port']}",
        f"--metrics_port={settings['mooncake_metrics_port']}",
        f"--default_kv_lease_ttl={settings['mooncake_lease_ttl_ms']}",
    ]
    return argv


def server_gpu_group(server_gpus: str, index: int, server_tp: int) -> str:
    devices = [p for p in str(server_gpus).split(",") if p]
    start = index * server_tp
    group = devices[start : start + server_tp]
    if len(group) != server_tp:
        raise ValueError(f"server {index} needs {server_tp} GPUs from {server_gpus}")
    return ",".join(group)


def sglang_server_argv(index: int, settings: Mapping[str, Any]) -> list[str]:
    port = int(settings["server_port"]) + index
    argv = [
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        "0.0.0.0",
        "--model-path",
        str(settings["target_model_path"]),
        "--trust-remote-code",
        "--skip-tokenizer-init",
        "--tp-size",
        str(settings["server_tp"]),
        "--mem-fraction-static",
        str(settings["server_mem_fraction"]),
        "--chunked-prefill-size",
        "-1",
        "--enable-spec-capture",
        "--spec-capture-method",
        "dflash",
        "--spec-capture-aux-layer-ids",
        *[str(i) for i in settings["capture_layer_ids"]],
        "--port",
        str(port),
        *list(settings["sglang_extra_args"]),
    ]
    return argv


def ready_paths(run_root: str) -> dict[str, Path]:
    root = Path(run_root)
    return {
        "root": root,
        "head_ip": root / "head.ip",
        "inference_ready": root / "inference.ready",
        "inference_done": root / "inference.done",
        "consumer_done": root / "consumer.done",
        "mooncake_log": root / "mooncake.log",
        "producer_log": root / "producer.log",
        "consumer_log": root / "consumer.log",
    }


def write_status(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(f"{value}\n", encoding="utf-8")
    tmp.replace(path)


def read_status(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def mooncake_env(head_ip: str, settings: Mapping[str, Any], local_ip: str) -> dict[str, str]:
    """Process env for Mooncake clients (producer/consumer/SGLang)."""

    return {
        "MOONCAKE_MASTER_SERVER_ADDR": f"{head_ip}:{settings['mooncake_rpc_port']}",
        "MOONCAKE_METADATA_SERVER": f"http://{head_ip}:{settings['mooncake_http_port']}/metadata",
        "MOONCAKE_PROTOCOL": str(settings["mooncake_protocol"]),
        "MOONCAKE_LOCAL_HOSTNAME": local_ip,
        "MC_TCP_BIND_ADDRESS": local_ip,
        "DISAGG_CLIENT_SEGMENT_SIZE": "0",
    }
