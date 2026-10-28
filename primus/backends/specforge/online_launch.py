###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Pure helpers for online SpecForge (Mooncake + SGLang + roles).

Cluster IPs stay out of git YAML. After Slurm allocates, Primus resolves the
capture-rank-0 IP (``HEAD_IP`` only there) and each node's local Mooncake
advertise IP, then renders Hydra overrides. Rank dispatch and sidecar
lifetime live in ``online_supervisor``. Capture nodes are
``[0, capture_nnodes)``; trainer nodes follow.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from primus.backends.specforge.argument_builder import (
    build_specforge_argv,
    flatten_overrides,
    resolve_specforge_root,
)

DEFAULT_MOONCAKE_RPC_PORT = 35551
DEFAULT_MOONCAKE_HTTP_PORT = 35880
DEFAULT_MOONCAKE_METRICS_PORT = 35903
DEFAULT_SERVER_PORT = 30000
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


def expand_device_list(value: str, count: int) -> str:
    """Expand a single device id to ``count`` consecutive ids (``0`` → ``0..N-1``)."""

    devices = [p for p in str(value).split(",") if p]
    if count <= 1 or len(devices) != 1:
        return ",".join(devices) if devices else str(value)
    try:
        start = int(devices[0])
    except ValueError:
        return ",".join(devices)
    return ",".join(str(start + i) for i in range(int(count)))


def parse_extra_args(value: Any) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return str(value).split()


def _positive_int(value: Any, default: int) -> int:
    if value in (None, "", "null"):
        return default
    return int(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value in (None, "", "null"):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _specforge_config_file(params: Any, env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """SpecForge Hydra file. Relative paths resolve only against the SpecForge checkout."""

    raw = getattr(params, "specforge_config", None)
    if raw in (None, "", "null"):
        return None
    path = Path(str(raw))
    if path.is_absolute():
        if path.is_file():
            return path
        raise ValueError(f"specforge_config is not a file: {raw}")
    root = resolve_specforge_root(params, env=env)
    if root is not None and (root / path).is_file():
        return root / path
    raise ValueError(f"specforge_config is not a file: {raw}")


def _model_section(params: Any, env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """``model:`` from one SpecForge YAML file.

    Hydra ``defaults:`` composition is not applied. A file that only inherits
    ``model`` through ``defaults`` is an error. The mapping is read once per
    ``online_settings`` call.
    """

    path = _specforge_config_file(params, env)
    if path is None:
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read SpecForge config {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    if "defaults" in data and "model" not in data:
        raise ValueError(
            f"{path} uses Hydra defaults and has no model section; "
            "Primus does not compose Hydra config groups"
        )
    model = data.get("model", {})
    if model is None:
        return {}
    if not isinstance(model, dict):
        raise ValueError(f"{path} model must be a mapping")
    return model


def _model_scalar(model: Mapping[str, Any], key: str) -> Optional[str]:
    if key not in model:
        return None
    value = model[key]
    if value in (None, "", "null"):
        return None
    if isinstance(value, (dict, list)):
        raise ValueError(f"model.{key} must be a scalar")
    text = str(value).strip()
    if "${" in text:
        raise ValueError(
            f"model.{key} contains an unresolved interpolation ({text}); "
            "Primus does not compose the Hydra file"
        )
    return text


def _present(value: Any) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    return str(value)


def _path_in_specforge_checkout(raw: str, root: Optional[Path]) -> Optional[Path]:
    """Absolute paths as written. Relative paths only under the SpecForge checkout."""

    path = Path(raw)
    if path.is_absolute():
        return path if path.is_file() else None
    if root is None:
        return None
    candidate = root / path
    return candidate if candidate.is_file() else None


def _existing_draft_config(
    params: Any,
    env: Optional[Mapping[str, str]] = None,
    model: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Draft JSON named by ``model.draft_model_config``.

    An override wins over the Hydra file. Relative paths resolve only against
    the SpecForge checkout, never the process working directory. A name that
    is set but does not exist is an error.
    """

    if model is None:
        model = _model_section(params, env)
    overrides = flatten_overrides(getattr(params, "specforge_overrides", None))
    raw = _present(overrides.get("model.draft_model_config"))
    if raw is None:
        raw = _model_scalar(model, "draft_model_config")
    if raw is None:
        return None
    if "${" in raw:
        raise ValueError(
            f"model.draft_model_config contains an unresolved interpolation ({raw}); "
            "Primus does not compose the Hydra file"
        )
    root = resolve_specforge_root(params, env=env)
    found = _path_in_specforge_checkout(raw, root)
    if found is None:
        raise ValueError(f"model.draft_model_config is not a file: {raw}")
    return found


def _layer_id_tuple(value: Any) -> Optional[tuple[int, ...]]:
    if isinstance(value, bool) or not isinstance(value, (list, tuple)) or not value:
        return None
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        ids.append(item)
    return tuple(ids)


def _draft_layer_ids(data: Mapping[str, Any]) -> Optional[tuple[int, ...]]:
    """Layer ids embedded in the draft JSON, for whichever drafter wrote them."""

    for group, key in (
        ("dflash_config", "target_layer_ids"),
        ("eagle_config", "eagle_aux_hidden_state_layer_ids"),
    ):
        block = data.get(group)
        if not isinstance(block, dict):
            continue
        ids = _layer_id_tuple(block.get(key))
        if ids:
            return ids
    return None


def _is_eagle_draft(data: Mapping[str, Any]) -> bool:
    if isinstance(data.get("eagle_config"), dict):
        return True
    architectures = data.get("architectures")
    if not isinstance(architectures, (list, tuple)):
        return False
    return any("eagle" in str(name).lower() for name in architectures)


def _target_num_hidden_layers(target_model_path: Optional[str]) -> int:
    if not target_model_path:
        raise ValueError(
            "EAGLE draft JSON has no eagle_config.eagle_aux_hidden_state_layer_ids, "
            "and no target model path is set to derive them"
        )
    root = Path(target_model_path)
    config_path = root / "config.json" if root.is_dir() else root
    if config_path.name != "config.json" or not config_path.is_file():
        raise ValueError(
            "EAGLE draft JSON has no eagle_config.eagle_aux_hidden_state_layer_ids; "
            f"expected a target config.json under {target_model_path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"could not read target config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} is not a JSON object")
    text = config.get("text_config")
    source = text if isinstance(text, dict) and "num_hidden_layers" in text else config
    num_layers = source.get("num_hidden_layers")
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError(f"{config_path} has no positive num_hidden_layers")
    return num_layers


def _eagle_layers_from_target_depth(num_layers: int) -> tuple[int, ...]:
    """SpecForge's EAGLE default when the draft JSON omits the layer list."""

    layers = (1, num_layers // 2 - 1, num_layers - 4)
    if any(layer < 0 for layer in layers):
        raise ValueError(
            "EAGLE capture layers derived from the target depth are negative: "
            f"num_hidden_layers={num_layers} -> {layers}"
        )
    return layers


def layer_ids_from_draft_json(path: Path, *, target_model_path: Optional[str] = None) -> tuple[int, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"could not read draft JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    ids = _draft_layer_ids(data)
    if ids:
        return ids
    if _is_eagle_draft(data):
        return _eagle_layers_from_target_depth(_target_num_hidden_layers(target_model_path))
    raise ValueError(
        f"{path} does not define capture layer ids "
        "(dflash_config.target_layer_ids or eagle_config.eagle_aux_hidden_state_layer_ids)"
    )


def resolve_capture_layer_ids(
    params: Any,
    env: Optional[Mapping[str, str]] = None,
    target_model_path: Optional[str] = None,
    model: Optional[Mapping[str, Any]] = None,
) -> tuple[int, ...]:
    """Read capture layer ids from the recipe draft JSON.

    ``model.draft_model_config`` in the SpecForge Hydra file, or an override of
    that field, names the file. DFlash-family drafts list
    ``dflash_config.target_layer_ids``. EAGLE drafts list
    ``eagle_config.eagle_aux_hidden_state_layer_ids``, or, when that list is
    absent, use ``[1, num_hidden_layers // 2 - 1, num_hidden_layers - 4]`` from
    the target ``config.json``. Online settings and environment variables do
    not supply layer ids.
    """

    draft = _existing_draft_config(params, env=env, model=model)
    if draft is None:
        raise ValueError(
            "model.draft_model_config must point at an existing draft JSON; "
            "capture layer ids come from that file"
        )
    return layer_ids_from_draft_json(draft, target_model_path=target_model_path)


def online_settings(params: Any, env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Flatten ``specforge_online`` plus a few env aliases used at launch.

    SGLang extra args default to AITER + radix-cache off so recipes omit those knobs.
    ``target_model_path`` comes from ``specforge_online``, then
    ``specforge_overrides``, then ``TARGET_MODEL``, then the Hydra file.
    ``skip_tokenizer_init`` comes from ``specforge_online`` or ``SKIP_TOKENIZER_INIT``
    and defaults to true. The Hydra file is one YAML document: ``defaults:``
    groups are not composed.
    """

    environ = os.environ if env is None else env
    raw = flatten_overrides(getattr(params, "specforge_online", None))
    model = _model_section(params, environ)
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
    server_gpus = expand_device_list(server_gpus, server_count * server_tp)
    trainer_gpus = expand_device_list(trainer_gpus, trainer_nproc)
    capture_nnodes = _positive_int(raw.get("capture_nnodes") or environ.get("CAPTURE_NNODES"), 1)
    trainer_nnodes = _positive_int(raw.get("trainer_nnodes") or environ.get("TRAINER_NNODES"), 1)
    protocol = str(raw.get("mooncake_protocol") or environ.get("MOONCAKE_PROTOCOL") or DEFAULT_PROTOCOL)
    lease = raw.get("mooncake_lease_ttl_ms") or environ.get("MOONCAKE_DEFAULT_KV_LEASE_TTL")
    lease_ttl = int(lease) if lease not in (None, "", "null") else DEFAULT_LEASE_TTL_MS
    target = (
        _present(raw.get("target_model_path"))
        or _present(
            flatten_overrides(getattr(params, "specforge_overrides", None)).get("model.target_model_path")
        )
        or _present(environ.get("TARGET_MODEL"))
        or _model_scalar(model, "target_model_path")
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
        "capture_nnodes": capture_nnodes,
        "trainer_nnodes": trainer_nnodes,
        "mooncake_protocol": protocol,
        "mooncake_rpc_port": int(raw.get("mooncake_rpc_port") or DEFAULT_MOONCAKE_RPC_PORT),
        "mooncake_http_port": int(raw.get("mooncake_http_port") or DEFAULT_MOONCAKE_HTTP_PORT),
        "mooncake_metrics_port": int(raw.get("mooncake_metrics_port") or DEFAULT_MOONCAKE_METRICS_PORT),
        "mooncake_lease_ttl_ms": lease_ttl,
        "capture_layer_ids": resolve_capture_layer_ids(
            params, environ, target_model_path=str(target), model=model
        ),
        "target_model_path": str(target),
        "sglang_extra_args": extra,
        "skip_tokenizer_init": _as_bool(
            (
                raw.get("skip_tokenizer_init")
                if raw.get("skip_tokenizer_init") not in (None, "", "null")
                else environ.get("SKIP_TOKENIZER_INIT")
            ),
            True,
        ),
        "bind_interface": str(raw.get("bind_interface") or environ.get("PRIMUS_SPECFORGE_BIND_IFACE") or ""),
        "start_timeout_s": int(raw.get("start_timeout_s") or environ.get("START_TIMEOUT_S") or 1800),
        "peer_timeout_s": int(raw.get("peer_timeout_s") or environ.get("PEER_TIMEOUT_S") or 1800),
    }


def expected_nnodes(settings: Mapping[str, Any]) -> int:
    return int(settings["capture_nnodes"]) + int(settings["trainer_nnodes"])


def is_capture_rank(rank: int, settings: Mapping[str, Any]) -> bool:
    return 0 <= rank < int(settings["capture_nnodes"])


def trainer_node_rank(rank: int, settings: Mapping[str, Any]) -> int:
    return rank - int(settings["capture_nnodes"])


def validate_online_identity(settings: Mapping[str, Any], *, rank: int, nodes: int) -> list[str]:
    issues: list[str] = []
    capture_nnodes = int(settings.get("capture_nnodes") or 0)
    trainer_nnodes = int(settings.get("trainer_nnodes") or 0)
    if capture_nnodes < 1:
        issues.append("capture_nnodes must be >= 1")
    if trainer_nnodes < 1:
        issues.append("trainer_nnodes must be >= 1")
    want = expected_nnodes(settings)
    if capture_nnodes >= 1 and trainer_nnodes >= 1 and nodes != want:
        issues.append(
            f"online mode expects NNODES=capture_nnodes+trainer_nnodes={want} "
            f"(capture_nnodes={capture_nnodes}, trainer_nnodes={trainer_nnodes}); got {nodes}"
        )
    if rank < 0 or (want > 0 and rank >= want):
        issues.append(f"online mode expects NODE_RANK in [0, {want}); got {rank}")
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
            f"server_gpus must contain server_count*server_tp={expected_server} devices per capture node; "
            f"got {got_server}"
        )
    got_trainer = len([p for p in str(settings["trainer_gpus"]).split(",") if p])
    if got_trainer != settings["trainer_nproc"]:
        issues.append(
            f"trainer_gpus must contain trainer_nproc={settings['trainer_nproc']} devices per trainer node; "
            f"got {got_trainer}"
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
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        logging.getLogger(__name__).warning("timed out reading IPv4 for interface %s", name)
        return None
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


def _override_ip(environ: Mapping[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        raw = environ.get(key)
        if raw and str(raw).strip():
            return str(raw).strip()
    return None


def resolve_local_ip(
    env: Optional[Mapping[str, str]] = None,
    *,
    interface: str = "",
    interface_ip: Optional[str] = None,
    hostname_ips: Optional[Sequence[str]] = None,
) -> str:
    """This node's Mooncake/SGLang advertise IP. Never a hostname or HEAD_IP."""

    environ = os.environ if env is None else env
    override = _override_ip(environ, ("PRIMUS_SPECFORGE_LOCAL_IP",))
    if override:
        return override
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
        "[Primus:specforge] could not resolve a local IPv4; "
        "set PRIMUS_SPECFORGE_LOCAL_IP or PRIMUS_SPECFORGE_BIND_IFACE"
    )


def resolve_head_ip(
    env: Optional[Mapping[str, str]] = None,
    *,
    interface: str = "",
    interface_ip: Optional[str] = None,
    hostname_ips: Optional[Sequence[str]] = None,
) -> str:
    """Capture rank-0 Mooncake/SGLang address. ``HEAD_IP`` applies only here."""

    environ = os.environ if env is None else env
    override = _override_ip(environ, ("PRIMUS_SPECFORGE_HEAD_IP", "HEAD_IP"))
    if override:
        return override
    return resolve_local_ip(
        env=environ, interface=interface, interface_ip=interface_ip, hostname_ips=hostname_ips
    )


def resolve_routable_ip(
    env: Optional[Mapping[str, str]] = None,
    *,
    interface: str = "",
    interface_ip: Optional[str] = None,
    hostname_ips: Optional[Sequence[str]] = None,
) -> str:
    """Backward-compatible alias of ``resolve_head_ip``."""

    return resolve_head_ip(env=env, interface=interface, interface_ip=interface_ip, hostname_ips=hostname_ips)


def server_urls(head_ip: str, settings: Mapping[str, Any]) -> list[str]:
    port0 = int(settings["server_port"])
    return [f"http://{head_ip}:{port0 + i}" for i in range(int(settings["server_count"]))]


def sglang_url_path(run_root: str, capture_rank: int) -> Path:
    return Path(run_root) / f"sglang-urls.{capture_rank}"


def consumer_done_paths(run_root: str, trainer_nnodes: int) -> list[Path]:
    root = Path(run_root)
    if int(trainer_nnodes) <= 1:
        return [root / "consumer.done"]
    return [root / f"consumer.done.{i}" for i in range(int(trainer_nnodes))]


def write_lines(path: Path, values: Sequence[str]) -> None:
    write_status(path, "\n".join(str(v).strip() for v in values if str(v).strip()))


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hydra_quoted_list(values: Sequence[str]) -> str:
    inner = ",".join(f'"{v}"' for v in values)
    return f"[{inner}]"


def build_online_overrides(
    head_ip: str,
    settings: Mapping[str, Any],
    output_dir: Optional[str] = None,
    server_url_list: Optional[Sequence[str]] = None,
    trainer_master_addr: Optional[str] = None,
) -> list[str]:
    """Hydra overrides that rewrite loopback YAML to the allocated HEAD_IP."""

    run_root = settings["run_root"]
    out = output_dir or f"{run_root}/output"
    http_port = settings["mooncake_http_port"]
    rpc_port = settings["mooncake_rpc_port"]
    urls = list(server_url_list) if server_url_list is not None else server_urls(head_ip, settings)
    overrides = [
        f"model.target_model_path={settings['target_model_path']}",
        f"run_id={settings['run_id']}",
        f"output_dir={out}",
        f"deployment.trainer.nnodes={settings['trainer_nnodes']}",
        f"deployment.trainer.nproc_per_node={settings['trainer_nproc']}",
        f"deployment.disaggregated.control_dir={run_root}/control",
        f"deployment.disaggregated.consumer_state_dir={settings['consumer_state_dir']}",
        f"deployment.disaggregated.store_id={settings['run_id']}",
        f"deployment.disaggregated.server_urls={hydra_quoted_list(urls)}",
        f"deployment.disaggregated.mooncake_metadata_server=http://{head_ip}:{http_port}/metadata",
        f"deployment.disaggregated.mooncake_master_server_addr={head_ip}:{rpc_port}",
        f"deployment.disaggregated.mooncake_protocol={settings['mooncake_protocol']}",
        f"deployment.disaggregated.idle_timeout_s={settings['peer_timeout_s']}",
        f"deployment.disaggregated.peer_wait_timeout_s={settings['peer_timeout_s']}",
    ]
    master = trainer_master_addr or settings.get("trainer_master_addr")
    if int(settings["trainer_nnodes"]) > 1 and master:
        overrides.append(f"deployment.trainer.master_addr={master}")
    return overrides


def build_role_argv(
    params: Any,
    role: str,
    extra_overrides: Sequence[str],
    *,
    node_rank: Optional[int] = None,
) -> list[str]:
    """``specforge train --config ... --role {{producer,consumer}}`` plus Hydra overrides."""

    return build_specforge_argv(params, extra_overrides=list(extra_overrides), role=role, node_rank=node_rank)


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
    ]
    extra = [arg for arg in list(settings["sglang_extra_args"]) if arg != "--skip-tokenizer-init"]
    if _as_bool(settings.get("skip_tokenizer_init", True), True):
        argv.insert(argv.index("--trust-remote-code") + 1, "--skip-tokenizer-init")
    argv.extend(extra)
    return argv


def ready_paths(run_root: str) -> dict[str, Path]:
    root = Path(run_root)
    return {
        "root": root,
        "head_ip": root / "head.ip",
        "trainer_ip": root / "trainer.ip",
        "server_urls": root / "server.urls",
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


def wait_for_status(path: Path, timeout_s: int = 1800, interval_s: float = 2.0) -> str:
    """Block until ``path`` exists, then return its stripped contents."""

    started = time.time()
    while not path.exists():
        if time.time() - started >= timeout_s:
            raise RuntimeError(f"[Primus:specforge] timed out waiting for {path}")
        time.sleep(interval_s)
    return read_status(path)


def rendezvous_head_ip(
    shared_dir: str | Path,
    env: Optional[Mapping[str, str]] = None,
    timeout_s: int = 1800,
    *,
    rank: Optional[int] = None,
) -> str:
    """Rank 0 writes ``head.ip`` under ``shared_dir``; other ranks wait and read it.

    ``PRIMUS_SPECFORGE_HEAD_IP`` / ``HEAD_IP`` apply only on rank 0. Do not trust
    Slurm ``MASTER_ADDR``.
    """

    path = Path(shared_dir) / "head.ip"
    this_rank = node_rank(env) if rank is None else int(rank)
    if this_rank == 0:
        ip = resolve_head_ip(env=env)
        write_status(path, ip)
        return ip
    return wait_for_status(path, timeout_s=timeout_s)


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
