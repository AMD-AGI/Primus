#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Convert HuggingFace checkpoints for native Megatron SFT using Megatron-Bridge.

This hook runs before Megatron SFT training to prepare pretrained checkpoints.
It calls ``AutoBridge.import_ckpt()`` directly and then normalizes the produced
checkpoint layout so native Megatron-LM finetune loading can consume it.

``native_ckpt_convert`` (in ``sft_trainer`` / ``post_trainer`` overrides) accepts a
YAML bool; the number 1 or 0 (also written as text: "1" / "0" / "1.0" / "0.0"); or the
strings true/false, yes/no, on/off (any case). An empty string (e.g. an unset
``${VAR:}``) means unset. Any other value (2, 0.5, "y", ...) stops this hook with an
error (exit 1) before conversion.
"""

import argparse
import importlib.util
import math
import numbers
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# Add primus to path
PRIMUS_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PRIMUS_ROOT))

from primus.core.config.primus_config import get_module_config, load_primus_config


def _is_rank_0() -> bool:
    """Check if current process is rank 0."""
    return int(os.environ.get("RANK", os.environ.get("NODE_RANK", 0))) == 0


def log_info(msg: str):
    """Log info message (only on rank 0)."""
    if _is_rank_0():
        print(f"[INFO] {msg}")


def log_error(msg: str):
    """Log error message (all ranks)."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def log_success(msg: str):
    """Log success message (only on rank 0)."""
    if _is_rank_0():
        print(f"[OK] {msg}")


def _first_config_value(obj, *names: str) -> str | None:
    """Return the first non-empty attribute value found on an object."""
    if obj is None:
        return None

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value not in (None, ""):
                return value
    return None


def get_checkpoint_config(config_file: str) -> tuple[str | None, str | None]:
    """
    Extract HF source path and any already-configured checkpoint path.

    Returns:
        tuple: (hf_path, checkpoint_path)
            - hf_path: HuggingFace model path to convert
            - checkpoint_path: Existing Megatron checkpoint path (if configured)
    """
    cfg = load_primus_config(Path(config_file), None)

    hf_path = None
    checkpoint_path = None

    # Try different module names that might contain the paths
    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is not None:
            params = getattr(module, "params", None)

            # Old Megatron-Bridge SFT configs used hf_path, while native Megatron
            # configs typically reuse tokenizer_model as the HF source identifier.
            hf_path = _first_config_value(params, "hf_path", "tokenizer_model") or _first_config_value(
                module, "hf_path", "tokenizer_model"
            )

            # If the user already configured a Megatron-format checkpoint via either
            # pretrained_checkpoint or load, the conversion hook should be skipped.
            checkpoint_path = _first_config_value(
                params, "pretrained_checkpoint", "load"
            ) or _first_config_value(module, "pretrained_checkpoint", "load")

            break

    return hf_path, checkpoint_path


_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})
# the numbers 1 / 0 written as text, beyond "1" / "0" above (a closed set: no float parsing)
_NUMBER_STRINGS = {"1.0": True, "0.0": False}


def parse_native_enabled(value) -> bool | None:
    """``native_ckpt_convert`` value -> ``None`` (unset) / ``True`` / ``False``.

    Accepts a bool, the numbers 0 / 1, or a true/false-like string; anything else raises
    ValueError.

    * ``None`` -> None; a bool is returned as is.
    * Numbers: only 0 and 1 (``0``, ``1``, ``0.0``, ``1.0``). Any other number (``2``,
      ``-1``, ``1.5``, NaN, inf) raises ValueError.
    * Strings (e.g. from env / CLI overrides; ``bool("false")`` would be True), any case,
      spaces stripped: ``true`` / ``yes`` / ``on`` and ``false`` / ``no`` / ``off``, and the
      four spellings ``"1"``, ``"0"``, ``"1.0"``, ``"0.0"`` (a closed set, no float
      parsing: ``"1e0"``, ``"+1"``, ``"1.00"``, ``"-0"`` raise). An empty string (an unset
      ``${VAR:}`` substitution) means unset. Anything else (``"2"``, ``"y"``, ``"enabled"``)
      raises ValueError.
    """
    if value is None or isinstance(value, bool):
        return value
    bad = ValueError(
        f"native_ckpt_convert: cannot parse {value!r} as a boolean (use true / false, "
        f"one of {sorted(_TRUE_STRINGS)} or {sorted(_FALSE_STRINGS)}, or the number 1 / 0)"
    )
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return None
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        if text in _NUMBER_STRINGS:
            return _NUMBER_STRINGS[text]
        raise bad
    elif isinstance(value, numbers.Real):
        number = float(value)
    else:
        raise bad
    if number == 1.0:
        return True
    if number == 0.0:
        return False
    raise bad  # 2, -1, 1.5, NaN, inf


def read_native_opts(config_file: str) -> dict:
    """Read native (bridge-free) conversion options from the config.

    Returns a dict with ``enabled`` (tri-state: ``None`` when the config does not
    set ``native_ckpt_convert`` at all, else ``True`` / ``False``), plus optional
    ``dtype`` / ``out`` overrides and ``max_params`` (``native_ckpt_max_params``, Muse
    Spark; as set: ``plan_conversion`` validates it on every rank before any conversion).
    ``enabled=None`` lets the caller pick the default: native for every family with a
    native converter (Muse Spark included). ``True`` forces native; ``False`` forces the
    legacy Megatron-Bridge path (an error for the ``_NO_BRIDGE`` families).
    """
    cfg = load_primus_config(Path(config_file), None)
    opts = {"enabled": None, "dtype": None, "out": None, "max_params": None}

    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is None:
            continue
        params = getattr(module, "params", None)

        enabled = _first_config_value(params, "native_ckpt_convert")
        if enabled is None:
            enabled = _first_config_value(module, "native_ckpt_convert")
        opts["enabled"] = parse_native_enabled(enabled)
        opts["dtype"] = _first_config_value(params, "native_ckpt_dtype") or _first_config_value(
            module, "native_ckpt_dtype"
        )
        opts["out"] = _first_config_value(params, "native_ckpt_out") or _first_config_value(
            module, "native_ckpt_out"
        )
        # Muse Spark only: the single-process converter's parameter cap (None -> its default).
        # ``is None``, not ``or``: a set falsy value (0) must reach plan_conversion's check.
        max_params = _first_config_value(params, "native_ckpt_max_params")
        if max_params is None:
            max_params = _first_config_value(module, "native_ckpt_max_params")
        opts["max_params"] = max_params
        break

    return opts


def _ckpt_options():
    """``native_ckpt_options`` (stdlib only), loaded by file path: no torch / Megatron import."""
    path = PRIMUS_ROOT / "primus/backends/megatron/checkpoint/native_ckpt_options.py"
    spec = importlib.util.spec_from_file_location("_primus_native_ckpt_options", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_bridge_paths() -> tuple[Path, Path]:
    """Resolve Megatron-Bridge source paths for direct AutoBridge import."""
    bridge_root = os.environ.get("MEGATRON_BRIDGE_PATH")
    if bridge_root:
        bridge_root = Path(bridge_root)
        log_info(f"Using MEGATRON_BRIDGE_PATH: {bridge_root}")
    else:
        bridge_root = PRIMUS_ROOT / "third_party" / "Megatron-Bridge"

    bridge_path = bridge_root / "src"
    bridge_megatron_path = bridge_root / "3rdparty" / "Megatron-LM"
    return bridge_path, bridge_megatron_path


@contextmanager
def _prepend_sys_path(*paths: Path):
    """Temporarily prepend import paths needed by Megatron-Bridge."""
    original_sys_path = list(sys.path)
    for path in reversed([str(path) for path in paths if path]):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path


@contextmanager
def _unset_nvte_attention_env():
    """Neutralize the TE attention-backend env vars around AutoBridge model construction.

    This is a general posttrain fix, not a test-only workaround: this hook runs
    for every native-Megatron SFT run, in a separate subprocess against
    Megatron-Bridge's own bundled Megatron-LM that never sees Primus's
    before_train patches (including the ROCm-safe attention_backend one). So it
    hits *stock* megatron's ``auto`` validation: ``AutoBridge.import_ckpt`` builds
    a plain ``MCoreGPTModel`` whose ``_set_attention_backend()`` asserts the three
    ``NVTE_*_ATTN`` vars are unset-or-1, and the ROCm image's baked
    ``NVTE_FLASH_ATTN=0`` trips it.

    Checkpoint conversion only reshapes weights -- it computes no attention -- so
    the backend is irrelevant here; the only goal is to get past that assert.
    Rather than counteract one specific baked value, unset all three for the
    duration (stock ``auto`` then accepts them and picks defaults harmlessly) and
    restore whatever was there afterwards. This stays correct for any image: if a
    future image leaves them unset or sets them to 1, the pop/restore is a no-op;
    it never assumes a particular baked value. Mirrors the Flux/diffusion conftest
    fix.
    """
    names = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")
    saved = {name: os.environ.pop(name, None) for name in names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
# Native (bridge-free) conversion path
# ---------------------------------------------------------------------------
# DEFAULT conversion path for supported families (Qwen3, DeepSeek, Llama, Muse Spark; see
# choose_native): this hook builds the mcore model in-process and copies the HF safetensors
# weights onto it (Primus-managed converters), without importing ``megatron.bridge``. It
# writes a legacy ``torch`` checkpoint (Muse Spark: ``torch_dist``, through the
# NativeConverter template). The convert-time monkeypatches (ROCm
# fused-kernels no-op) are applied in THIS process via ``run_patches(phase=
# "convert")`` so they take effect for the single-process conversion; the
# Megatron-LM submodule stays pristine. Set ``native_ckpt_convert: false`` to
# force the legacy Megatron-Bridge path (or ``true`` to force native).

# HF architectures / model_type -> Primus native converter family.
_NATIVE_FAMILIES = {
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3forcausallm": "qwen3",
    "qwen3moeforcausallm": "qwen3",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseekv2forcausallm": "deepseek",
    "deepseekv3forcausallm": "deepseek",
    "deepseek": "deepseek",
    "llama": "llama",
    "llamaforcausallm": "llama",
    # Muse Spark: text decoder only (the converter skips the vision / audio towers).
    # NativeConverter subclass (native_loader_muse_spark.MuseSparkConverter): a torch_dist
    # checkpoint that Megatron reshards on load. muse_spark_text is the text_config type.
    "muse_spark": "muse_spark",
    "muse_spark_text": "muse_spark",
    "musesparkforconditionalgeneration": "muse_spark",
    "musesparkforcausallm": "muse_spark",
}


# Families without a Megatron-Bridge model: ``native_ckpt_convert: false`` stops with an error.
_NO_BRIDGE = frozenset({"muse_spark"})

# The one user-facing list of native families (every message that names them uses it).
NATIVE_SUPPORTED_DESC = "Qwen3 dense/MoE, DeepSeek-V2/V3, Llama 2/3/3.1, Muse Spark"


def bridge_unsupported_reason(family, hf_path="") -> str | None:
    """Error text when ``family`` cannot go to Megatron-Bridge, which has no model for it
    (``_NO_BRIDGE``), else None. Only call it when ``choose_native`` is False, i.e. when
    ``native_ckpt_convert`` is false (a native family with it unset converts natively).
    Megatron-Bridge's import_ckpt would only fail later with an opaque error, so the hook
    stops early with the fix."""
    if family not in _NO_BRIDGE:
        return None
    return (
        f"HF model at {hf_path} is a {family} checkpoint: Megatron-Bridge cannot convert it "
        "(native_ckpt_convert is false). Unset native_ckpt_convert (or set it to true) for the "
        "native converter, or pass a converted Megatron checkpoint."
    )


def choose_native(enabled, family) -> bool:
    """``native_ckpt_convert`` true / false forces the path; unset -> native for every
    family with a native converter (Muse Spark included)."""
    if enabled is not None:
        return enabled
    return family is not None


# Released Hub ids of the _NO_BRIDGE families (exact, case-insensitive). Without an
# explicit ``native_ckpt_convert: true`` a Hub id is matched to such a family only by this
# exact list: not by the node's HF cache (nodes can differ, and every node routes on its
# own), not by a substring of the repo name (a fine-tune / adapter / GGUF repo with
# "muse-spark" in its name is not a Muse Spark checkpoint), and with no network lookup.
# Each entry: (family, nominal parameter count from the model name). The native converter
# is single-process (the whole model plus a state dict in host RAM) and refuses a model
# above ``native_ckpt_max_params`` (default 60B); plan_conversion checks the count on every
# rank before anything is downloaded, so the default route does not end in that error
# after rank 0 fetched the configs.
_RELEASED_HUB_IDS = {"meta-models/muse-spark-open-816b-a42b": ("muse_spark", 816 * 10**9)}
_NO_BRIDGE_HUB_IDS = {hub_id: family for hub_id, (family, _) in _RELEASED_HUB_IDS.items()}


def _is_hub_id(hf_path: str) -> bool:
    """True for ``org/name`` that is not a local path (what ``from_pretrained`` downloads)."""
    if not hf_path or os.path.exists(hf_path) or os.path.isabs(hf_path):
        return False
    parts = hf_path.split("/")
    return len(parts) == 2 and all(parts) and not hf_path.startswith(".")


def _cached_hub_config_file(hub_id: str) -> str | None:
    """``config.json`` of ``hub_id`` in the local HF cache, else None. Local files only:
    never a network call (``huggingface_hub.try_to_load_from_cache``); never raises."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(hub_id, "config.json")
    except Exception:  # noqa: BLE001 - not installed / bad id: no cached config
        return None
    return path if isinstance(path, str) and os.path.isfile(path) else None


def _read_hf_config(hf_path: str) -> dict | None:
    """``config.json`` of a local HF dir, or of a Hub id from the local HF cache only (no
    network), else None. Stdlib + optional huggingface_hub: never raises."""
    import json

    cfg_file = os.path.join(hf_path, "config.json")
    if not os.path.isfile(cfg_file):
        cfg_file = _cached_hub_config_file(hf_path) if _is_hub_id(hf_path) else None
        if cfg_file is None:
            return None
    try:
        with open(cfg_file) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _family_of_config(cfg: dict) -> str | None:
    for arch in cfg.get("architectures", []) or []:
        fam = _NATIVE_FAMILIES.get(str(arch).lower())
        if fam:
            return fam
    fam = _NATIVE_FAMILIES.get(str(cfg.get("model_type", "")).lower())
    if fam:
        return fam
    # Only Muse Spark is recognised through its text_config: other multimodal wrappers
    # (LLaVA, Qwen3-VL, ...) keep falling through to Megatron-Bridge, because the
    # text-only native loaders do not know their key prefixes.
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict) and str(text_cfg.get("model_type", "")).lower() == "muse_spark_text":
        return "muse_spark"
    return None


def _hub_config_file(hub_id: str) -> str | None:
    """``config.json`` of ``hub_id``: the local HF cache, else a download of that one file
    (``hf_hub_download``). Only for an explicit ``native_ckpt_convert: true``. Never raises."""
    path = _cached_hub_config_file(hub_id)
    if path is not None:
        return path
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(hub_id, "config.json")
    except Exception as e:  # noqa: BLE001 - offline / no such repo: no config
        log_info(f"could not fetch config.json of {hub_id}: {e}")
        return None
    return path if isinstance(path, str) and os.path.isfile(path) else None


def _detect_native_family(hf_path: str, fetch: bool = False) -> str | None:
    """Return 'qwen3' | 'deepseek' | 'llama' | 'muse_spark' | None from the HF config.json.

    ``hf_path`` is a local HF dir or a Hub id (``org/name``, e.g. the Muse preset's
    ``tokenizer_model``). Every node makes this decision on its own, so for a Hub id it
    must not depend on what one node has in its HF cache:

    * ``fetch=False`` (``native_ckpt_convert`` unset): only the exact released ids in
      ``_NO_BRIDGE_HUB_IDS`` map to a family; every other Hub id returns None (the
      Megatron-Bridge path). No cache and no network access.
    * ``fetch=True`` (``native_ckpt_convert: true``): the Hub id's own config.json decides
      (the HF cache, else a download of that file), the same file on every node. Only a
      ``_NO_BRIDGE`` family is taken from a Hub config; if the file cannot be read the
      exact list applies, and an unknown id stops that node with an error (no hang).
    """
    if not _is_hub_id(hf_path):
        cfg = _read_hf_config(hf_path)
        return _family_of_config(cfg) if cfg is not None else None
    if not fetch:
        return _NO_BRIDGE_HUB_IDS.get(hf_path.lower())
    cfg_file = _hub_config_file(hf_path)
    cfg = _read_hf_config(os.path.dirname(cfg_file)) if cfg_file else None
    if cfg is not None:
        fam = _family_of_config(cfg)
        return fam if fam in _NO_BRIDGE else None  # other families: local dirs only
    return _NO_BRIDGE_HUB_IDS.get(hf_path.lower())


def _native_dtype_from_config(hf_path: str, override: str | None) -> str:
    if override:
        return override
    cfg = _read_hf_config(hf_path)
    if cfg is not None:
        td = str(cfg.get("torch_dtype") or (cfg.get("text_config") or {}).get("torch_dtype") or "").lower()
        if "bfloat16" in td:
            return "bf16"
        if "float16" in td:
            return "fp16"
        if "float32" in td:
            return "fp32"
    return "bf16"


def _local_hf_dir(hf_path: str, family: str, opts: dict) -> str:
    """The local HF dir to convert: ``hf_path`` itself, or for a Hub id its snapshot in the
    HF cache (downloaded when missing). For Muse Spark the size cap is checked on the
    config files first, so a model above ``native_ckpt_max_params`` is refused before its
    weights are downloaded."""
    if not _is_hub_id(hf_path):
        return hf_path
    from huggingface_hub import snapshot_download

    log_info(f"{hf_path} is a Hugging Face Hub id: using its snapshot in the HF cache")
    if family == "muse_spark":
        from primus.backends.megatron.checkpoint import native_loader_muse_spark as muse

        cfg_dir = snapshot_download(hf_path, allow_patterns=["*.json"])
        cfg = _read_hf_config(cfg_dir) or {}
        muse.check_convert_size(muse.derive_megatron_fields(cfg), cfg, max_params=opts.get("max_params"))
    return snapshot_download(hf_path)


def native_convert_checkpoint(hf_path: str, megatron_path: str, opts: dict):
    """Convert a HF checkpoint to a Megatron checkpoint, bridge-free: a legacy ``torch``
    checkpoint for llama / qwen3 / deepseek, a ``torch_dist`` one for Muse Spark.

    Applies the ``phase="convert"`` patches in this process, then runs the
    Primus-managed native converter for the detected family, all single-process.
    """
    family = opts.get("family") or _detect_native_family(hf_path)
    if family is None:
        raise ValueError(
            f"native_ckpt_convert requested but no native converter for HF model at {hf_path} "
            f"(supported: {NATIVE_SUPPORTED_DESC})"
        )

    # Ensure the Megatron-LM source tree is importable in this standalone hook
    # process (run_pretrain.sh's PYTHONPATH does not include it), and register +
    # apply the convert-phase patches BEFORE any Megatron model build.
    from primus.backends.megatron.checkpoint import native_convert_common as common

    common.ensure_megatron_on_path()
    hf_path = _local_hf_dir(hf_path, family, opts)
    applied = common.apply_convert_patches(module_name="checkpoint_convert")
    log_info(f"Applied {applied} convert-phase patch(es)")

    dtype = _native_dtype_from_config(hf_path, opts.get("dtype"))
    tp = int(opts.get("tensor_parallel_size", 1))
    pp = int(opts.get("pipeline_parallel_size", 1))
    ep = int(opts.get("expert_parallel_size", 1))

    log_info(f"Native ({family}) conversion: dtype={dtype} TP={tp} PP={pp} EP={ep}")
    log_info(f"  Source: {hf_path}")
    log_info(f"  Target: {megatron_path}")

    if family == "qwen3":
        from primus.backends.megatron.checkpoint import native_loader_qwen3 as conv
    elif family == "llama":
        from primus.backends.megatron.checkpoint import native_loader_llama as conv
    elif family == "muse_spark":
        from primus.backends.megatron.checkpoint import (
            native_loader_muse_spark_text as conv,
        )
    else:
        from primus.backends.megatron.checkpoint import native_loader_deepseek as conv

    extra = {}
    if family == "muse_spark" and opts.get("max_params") is not None:
        extra["max_params"] = opts["max_params"]  # native_ckpt_max_params
    conv.convert(
        hf_path,
        megatron_path,
        dtype=dtype,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        **extra,
    )

    # Hard evidence for the bridge-free guarantee.
    assert "megatron.bridge" not in sys.modules, "megatron.bridge was imported during native conversion"
    log_success("Native (bridge-free) checkpoint conversion completed")


def convert_checkpoint(hf_path: str, megatron_path: str):
    """
    Convert HuggingFace checkpoint to Megatron torch_dist format.
    """
    bridge_path, bridge_megatron_path = _resolve_bridge_paths()
    log_info(f"Megatron-Bridge path: {bridge_path}")
    log_info(f"Using Megatron-LM from: {bridge_megatron_path}")

    log_info(f"Converting HF -> Megatron checkpoint...")
    log_info(f"  Source: {hf_path}")
    log_info(f"  Target: {megatron_path}")

    with _prepend_sys_path(bridge_path, bridge_megatron_path), _unset_nvte_attention_env():
        from megatron.bridge import AutoBridge

        # Convert using AutoBridge - creates torch_dist format checkpoint
        AutoBridge.import_ckpt(
            hf_model_id=hf_path,
            megatron_path=megatron_path,
            trust_remote_code=True,
        )

    log_success("Checkpoint conversion completed")


TRACKER_FILE = "latest_checkpointed_iteration.txt"  # Megatron's "a checkpoint is here" file


def _tracker_file(megatron_path: Path) -> Path:
    return megatron_path / TRACKER_FILE


def _checkpoint_ready(megatron_path: Path) -> bool:
    """A finished conversion output: both converters write Megatron's tracker file last
    (Megatron-Bridge: fixed up to ``release`` by rank 0), so a dir without it is partial
    (a killed or failed run) and is converted again, not used. The skip check of both
    paths and the waiters' ``ready`` callback."""
    return megatron_path.is_dir() and _tracker_file(megatron_path).exists()


# How the other node ranks wait for rank 0's conversion (wait_for_conversion):
#
# * Rank 0 holds ``<checkpoint>.converting.lock`` for the whole conversion and touches it
#   every HEARTBEAT_INTERVAL_S seconds (a heartbeat thread). A waiter that sees the lock
#   unchanged for $PRIMUS_CKPT_CONVERT_HEARTBEAT_TIMEOUT seconds (its own clock: no clock
#   skew between nodes) stops: rank 0 was killed (SIGKILL, OOM) and can write no marker.
# * When no lock appears and nothing else happens for START_TIMEOUT seconds, rank 0 did
#   not start a conversion: the waiter stops after START_TIMEOUT (600 s).
# * $PRIMUS_CKPT_CONVERT_TIMEOUT (default 4 h, for every family) bounds the whole wait
#   while rank 0 is alive. The heartbeat, not the timeout, detects a dead rank 0, so the
#   timeout does not need to fit the checkpoint size.
CONVERT_TIMEOUT_ENV = "PRIMUS_CKPT_CONVERT_TIMEOUT"
DEFAULT_CONVERT_TIMEOUT = 4 * 3600
START_TIMEOUT = 600
HEARTBEAT_TIMEOUT_ENV = "PRIMUS_CKPT_CONVERT_HEARTBEAT_TIMEOUT"
DEFAULT_HEARTBEAT_TIMEOUT = 300.0
HEARTBEAT_INTERVAL_S = 30
# A waiting rank fails at once when rank 0 left a ``.failed`` marker of *this* run (its
# conversion raised). The run is identified by a token that every node of one launch
# shares: $PRIMUS_CKPT_CONVERT_RUN_ID, else $SLURM_JOB_ID + $SLURM_STEP_ID (+
# $SLURM_RESTART_COUNT: each srun step and each requeue is a new run). Without one
# (torchrun / k8s launches) the token is $MASTER_ADDR:$MASTER_PORT, which a later launch
# may reuse; see wait_for_conversion for how a marker is then judged. Rank 0 writes the
# token into the ``.failed`` marker and into the ``.done`` file.
RUN_ID_ENV = "PRIMUS_CKPT_CONVERT_RUN_ID"
# A ``.failed`` marker that was already there when the wait started counts as this run's
# only after the lock file has been absent, without a break, for this many seconds (> 0).
MARKER_GRACE_ENV = "PRIMUS_CKPT_CONVERT_MARKER_GRACE"
DEFAULT_MARKER_GRACE = 120.0
# wait_for_conversion polls every POLL_INTERVAL_S seconds, every POLL_INTERVAL_LOCKED_S
# while rank 0's lock file is there (a conversion takes minutes to hours)
POLL_INTERVAL_S = 2
POLL_INTERVAL_LOCKED_S = 5


def _positive_seconds(env: str, default: float) -> float:
    """``$env`` as a finite number of seconds > 0, else ``default``; ValueError otherwise."""
    raw = os.environ.get(env, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{env}={raw!r} is not a finite positive number of seconds")
    return value


def convert_timeout(default: float = DEFAULT_CONVERT_TIMEOUT) -> float:
    """The wait timeout in seconds: ``$PRIMUS_CKPT_CONVERT_TIMEOUT``, else ``default`` (4 h).

    The value must be a finite number > 0; anything else (0, -1, inf, nan, abc) raises
    ValueError. ``plan_conversion()`` calls this on every rank before any conversion starts.
    """
    return _positive_seconds(CONVERT_TIMEOUT_ENV, default)


def heartbeat_timeout() -> float:
    """``$PRIMUS_CKPT_CONVERT_HEARTBEAT_TIMEOUT`` seconds (finite, > HEARTBEAT_INTERVAL_S = 30),
    else 300."""
    value = _positive_seconds(HEARTBEAT_TIMEOUT_ENV, DEFAULT_HEARTBEAT_TIMEOUT)
    if value <= HEARTBEAT_INTERVAL_S:
        raise ValueError(
            f"{HEARTBEAT_TIMEOUT_ENV}={value:g} must be above the {HEARTBEAT_INTERVAL_S} s heartbeat interval"
        )
    return value


def marker_grace() -> float:
    """``$PRIMUS_CKPT_CONVERT_MARKER_GRACE`` seconds (finite, > 0), else 120."""
    return _positive_seconds(MARKER_GRACE_ENV, DEFAULT_MARKER_GRACE)


def failed_marker(done_file: Path) -> Path:
    """``<checkpoint>.failed``, next to ``<checkpoint>.done``."""
    return done_file.with_suffix(".failed")


def conversion_run_id() -> str | None:
    """The token shared by every node of this launch (see RUN_ID_ENV), or None if unknown."""
    token, _strong = _run_token()
    return token


def _run_token() -> tuple[str | None, bool]:
    """``(token, strong)``: a strong token ($PRIMUS_CKPT_CONVERT_RUN_ID, Slurm) names one
    launch; the $MASTER_ADDR:$MASTER_PORT fallback may repeat across launches."""
    explicit = os.environ.get(RUN_ID_ENV, "").strip()
    if explicit:
        return explicit, True
    job = os.environ.get("SLURM_JOB_ID", "").strip()
    if job:
        # the step id: every srun step of one allocation (retries, sweeps) is a new run
        # (without one, e.g. a batch script with no srun, the token may repeat: weak)
        step = os.environ.get("SLURM_STEP_ID", "").strip()
        restart = os.environ.get("SLURM_RESTART_COUNT", "").strip() or "0"
        return f"slurm:{job}:{step or 'batch'}:{restart}", bool(step)
    addr = os.environ.get("MASTER_ADDR", "").strip()
    port = os.environ.get("MASTER_PORT", "").strip()
    if addr and port:
        return f"master:{addr}:{port}", False
    return None, False


def _file_state(path: Path):
    """``(inode, mtime_ns, size)`` of the file, or None when it does not exist."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _file_token(path: Path) -> str | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return text or None


def _write_token(path: Path, token: str | None):
    """Write ``token`` to ``path`` by write-then-rename (a reader never sees half of it)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{token or ''}\n")
    os.replace(tmp, path)


def wait_for_conversion(done_file: Path, lock_file: Path, timeout: float | None = None, ready=None):
    """Wait for rank 0 to complete checkpoint conversion (``timeout`` None -> convert_timeout()).

    Returns when rank 0's ``.done`` file of this run is there. Raises RuntimeError when
    rank 0 left a ``.failed`` marker of this run, or when its lock file stopped changing
    for ``heartbeat_timeout()`` seconds (rank 0 was killed). Raises TimeoutError when no
    lock appears for ``START_TIMEOUT`` seconds, or after ``timeout`` seconds.

    A marker (``.done`` or ``.failed``) that holds another launch's token is never this
    run's, also when it appeared or changed during the wait (two launches that share the
    checkpoint dir). An empty marker or one with this run's token is judged as below.

    A ``.done`` file is of this run when it appeared or changed after this wait started.
    One that was already there counts only while no lock file is there and ``ready()`` (the
    checkpoint is complete) is true; without ``ready`` it must hold this run's strong token.

    A ``.failed`` marker is of this run when:

    * it appeared or changed after this wait started (a snapshot, no clock), or
    * it was already there, holds no other launch's token, and the lock file has been
      absent, without a break, for ``marker_grace()`` seconds *now* (the lock's current
      state, not an earlier poll's). Rank 0 unlinks an old marker before it creates the
      lock file, so only a rank 0 that starts more than the grace period after this rank
      is mistaken for a failure (raise the grace in that case).

    """
    if timeout is None:
        timeout = convert_timeout()
    failed = failed_marker(done_file)
    run_id, strong = _run_token()
    grace = marker_grace()
    hb_timeout = heartbeat_timeout()
    done_at_start = _file_state(done_file)
    failed_at_start = _file_state(failed)
    start = time.monotonic()
    deadline = start + timeout
    start_deadline = start + min(START_TIMEOUT, timeout)
    lock_state = _file_state(lock_file)
    lock_changed_at = start  # the last time this rank saw the lock file change (or go)
    lock_seen = lock_state is not None

    def other_launch(path):
        """The marker holds another launch's token (never this run's, changed or not)."""
        token = _file_token(path)
        return run_id is not None and token not in (None, run_id)

    def done_is_ours():
        state = _file_state(done_file)
        if state is None:
            return False
        if other_launch(done_file):
            return False
        if state != done_at_start:
            return True
        # there before the wait: maybe an earlier run's, which rank 0 has not removed yet.
        # Only a complete checkpoint with no conversion running counts.
        if lock_state is not None:
            return False
        if ready is not None:
            return bool(ready())
        return strong and _file_token(done_file) == run_id

    def failed_is_ours(now):
        state = _file_state(failed)
        if state is None:
            return False
        if other_launch(failed):
            return False
        if state != failed_at_start:
            return True
        # there before the wait: this run's only when no lock file is there now and none
        # has been for the grace period
        return lock_state is None and now - lock_changed_at >= grace

    while True:
        now = time.monotonic()
        state = _file_state(lock_file)
        if state != lock_state:
            lock_state, lock_changed_at = state, now
            lock_seen = lock_seen or state is not None
        if done_is_ours():
            return
        if failed_is_ours(now):
            raise RuntimeError(f"rank 0 failed to convert the checkpoint (see its log; marker: {failed})")
        if lock_state is not None and now - lock_changed_at >= hb_timeout:
            raise RuntimeError(
                f"rank 0's lock file {lock_file} has not changed for {now - lock_changed_at:.0f} s: "
                f"rank 0 stopped (killed / out of memory?) without a failure marker. Remove the lock "
                f"file and relaunch; {HEARTBEAT_TIMEOUT_ENV} sets this limit"
            )
        if now >= deadline:
            break
        if not lock_seen and now >= start_deadline:
            raise TimeoutError(
                f"rank 0 did not start the checkpoint conversion within {START_TIMEOUT} s "
                f"(no {lock_file}, no {done_file})"
            )
        time.sleep(POLL_INTERVAL_LOCKED_S if lock_state is not None else POLL_INTERVAL_S)

    raise TimeoutError(
        f"Timeout ({timeout:.0f} s) waiting for checkpoint conversion; set {CONVERT_TIMEOUT_ENV} to wait longer"
    )


@contextmanager
def _heartbeat(lock_file: Path, interval: float = HEARTBEAT_INTERVAL_S):
    """Touch ``lock_file`` every ``interval`` seconds while the body runs (a daemon thread)."""
    stop = threading.Event()

    def beat():
        while not stop.wait(interval):
            try:
                os.utime(lock_file)
            except OSError:
                pass  # the waiters then see a stale lock and stop: the safe side

    thread = threading.Thread(target=beat, name="ckpt-convert-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


@contextmanager
def _rank0_conversion(done_file: Path, lock_file: Path):
    """Rank 0's conversion: drops a stale ``.done`` file and ``.failed`` marker, holds the
    lock file (touched by a heartbeat thread), writes ``.done`` (holding
    ``conversion_run_id()``) when the body returns and a new ``.failed`` marker (holding the
    same token) when the body or that ``.done`` write raises, so the waiting ranks of this run go on or stop at once. The
    body must not create ``.done`` itself."""
    failed = failed_marker(done_file)
    done_file.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    _write_token(lock_file, conversion_run_id())  # a new inode: waiters see a change
    keep_lock = False
    try:
        with _heartbeat(lock_file):
            yield
        # inside the try: a failed write (ENOSPC / EROFS / EACCES) is a failed conversion
        # for the waiting ranks too (a .failed marker, or the kept lock going stale), not
        # a removed lock with no marker that they would wait on for the whole timeout
        _write_token(done_file, conversion_run_id())
    except BaseException:
        try:
            _write_token(failed, conversion_run_id())
        except OSError as e:
            # e.g. ENOSPC / EROFS / EACCES on the checkpoint dir: the lock is kept (its
            # heartbeat has stopped), so the waiting ranks stop when it goes stale after
            # PRIMUS_CKPT_CONVERT_HEARTBEAT_TIMEOUT instead of waiting the whole timeout
            keep_lock = True
            log_error(
                f"Could not write the failure marker {failed} ({type(e).__name__}: {e}); "
                f"the lock {lock_file} is kept, so the other ranks stop after "
                f"{HEARTBEAT_TIMEOUT_ENV}; remove it before a relaunch"
            )
        raise
    finally:
        if not keep_lock:
            lock_file.unlink(missing_ok=True)


def fix_common_pt_for_megatron_lm(checkpoint_dir: Path):
    """
    Fix common.pt to include 'args' for Megatron-LM compatibility.

    Megatron-LM expects 'args' in common.pt's state_dict for loading torch_dist
    checkpoints. HuggingFace converted checkpoints are always TP=1, PP=1.
    """
    from types import SimpleNamespace

    import torch

    common_pt = checkpoint_dir / "common.pt"

    log_info(f"  3. Adding 'args' to common.pt for Megatron-LM compatibility")

    # Load existing common.pt
    state_dict = torch.load(common_pt, map_location="cpu")

    # Check if args already exists
    if "args" in state_dict:
        log_info("     'args' already exists in common.pt, skipping")
        return

    # Create args namespace with default values for HuggingFace converted checkpoints
    # HF models are single-device, so TP=1, PP=1
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        no_save_rng=True,
        no_save_optim=True,
        ckpt_fully_parallel_save=False,
    )

    # Add args to state_dict and save
    state_dict["args"] = args
    torch.save(state_dict, common_pt)
    log_success("     Successfully added 'args' to common.pt")


def _native_main(hf_path: str, native_opts: dict, family: str):
    """Bridge-free native conversion branch of main().

    Mirrors the Bridge path's skip-if-exists / rank-0-converts-others-wait
    semantics, but produces a Megatron checkpoint via the Primus-managed in-process
    converter (legacy ``torch`` for llama / qwen3 / deepseek, ``torch_dist`` for Muse Spark) and emits ``pretrained_checkpoint`` + ``finetune`` back
    to the runner. ``family`` is the one ``plan_conversion`` detected.
    """
    native_opts["family"] = family

    # Output path: <out>/iter_0000001/ (mp_rank_00/... for a legacy torch checkpoint,
    # torch_dist shard files for Muse Spark), plus latest_checkpointed_iteration.txt.
    if native_opts.get("out"):
        megatron_path = Path(native_opts["out"])
    else:
        data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
        megatron_path = data_path / "megatron_checkpoints" / f"{Path(hf_path).name}_native"

    log_info(f"HF Model: {hf_path}  (native family: {family})")
    log_info(f"Megatron Path: {megatron_path}")

    if _checkpoint_ready(megatron_path):
        log_info(f"Native Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        print(f"extra.pretrained_checkpoint={megatron_path}")
        print(f"extra.finetune=true")
        return

    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        log_info("Converting HF checkpoint to Megatron format (native, bridge-free)...")
        megatron_path.parent.mkdir(parents=True, exist_ok=True)
        with _rank0_conversion(done_file, lock_file):
            native_convert_checkpoint(hf_path, str(megatron_path), native_opts)
            log_success(f"Checkpoint prepared at {megatron_path}")
    else:
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete native conversion...")
        # a waiter that reached the hook after rank 0 finished sees an unchanged .done:
        # the same check as rank 0's skip-if-exists shows the checkpoint is complete
        wait_for_conversion(
            done_file,
            lock_file,
            ready=lambda: _checkpoint_ready(megatron_path),  # the same check as rank 0's skip
        )
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


def _hub_id_too_big_reason(hf_path: str, max_params) -> str | None:
    """The error for a released Hub id (``_RELEASED_HUB_IDS``) above the native converter's
    cap (``max_params``: the parsed ``native_ckpt_max_params``, None -> the default), else
    None. Stdlib only, no download: the same answer on every rank."""
    n = _RELEASED_HUB_IDS.get(hf_path.lower(), (None, None))[1] if _is_hub_id(hf_path) else None
    if n is None:
        return None
    opts = _ckpt_options()
    limit = opts.parse_max_params(max_params)
    if n <= limit:
        return None
    return (
        f"{hf_path} has ~{n / 1e9:.0f}B params, above the native converter's single-process limit "
        f"of {limit:,} params ({opts.MAX_PARAMS_KEY}); the converter holds the whole model and a "
        "state dict in host RAM (about 4 bytes per param: ~"
        f"{n * 4 / 1e12:.1f} TB here). Primus ships no distributed converter. Supported routes: "
        f"(1) on a host with that much RAM, set {opts.MAX_PARAMS_KEY} to at least {int(n)} and "
        "this hook converts as usual; (2) set pretrained_checkpoint to a Megatron checkpoint "
        "made elsewhere (any TP / PP / EP; the hook then skips conversion); (3) point hf_path "
        "at a smaller (e.g. proxy) checkpoint."
    )


def plan_conversion(config_file: str):
    """The hook's whole routing decision: ``(action, detail, hf_path, native_opts, family)``.

    It writes nothing and starts no converter. Its one side effect is network access: for a
    Hub id with ``native_ckpt_convert: true``, ``_detect_native_family`` may download the
    id's ``config.json`` (``hf_hub_download``) into the HF cache. Without that opt-in the
    routing is offline (local dirs and the exact ``_NO_BRIDGE_HUB_IDS`` list only).

    ``action`` is one of:

    * ``"pretrained"``: a Megatron checkpoint is configured; no conversion.
    * ``"no_hf_path"``: nothing to convert.
    * ``"error"``: the hook stops with exit code 1 before any converter runs; ``detail``
      is the message (a bad ``native_ckpt_convert``, ``native_ckpt_max_params``,
      ``PRIMUS_CKPT_CONVERT_TIMEOUT``, ``PRIMUS_CKPT_CONVERT_MARKER_GRACE`` or
      ``PRIMUS_CKPT_CONVERT_HEARTBEAT_TIMEOUT``,
      ``native_ckpt_convert: true`` on a family without a native converter, or
      ``native_ckpt_convert: false`` on a family without a Megatron-Bridge model, such as
      Muse Spark, or a released Hub id above the native converter's size cap).
    * ``"native"``: the native (bridge-free) converter runs.
    * ``"bridge"``: the Megatron-Bridge converter runs; ``detail`` is why.

    ``main()`` acts on it, and ``00_install_requirements.sh`` calls it to decide whether
    the Megatron-Bridge dependencies are needed, so the two cannot disagree.
    """
    hf_path, pretrained_checkpoint = get_checkpoint_config(config_file)
    if pretrained_checkpoint:
        return "pretrained", str(pretrained_checkpoint), hf_path, None, None
    if not hf_path:
        return "no_hf_path", "", hf_path, None, None
    # choose_native decides: ``native_ckpt_convert`` true / false forces the path;
    # unset -> native for every family with a native converter; false on a _NO_BRIDGE
    # family (no Megatron-Bridge model) stops with bridge_unsupported_reason's error.
    try:
        native_opts = read_native_opts(config_file)
        # every rank: a bad PRIMUS_CKPT_CONVERT_TIMEOUT / PRIMUS_CKPT_CONVERT_MARKER_GRACE /
        # PRIMUS_CKPT_CONVERT_HEARTBEAT_TIMEOUT or native_ckpt_max_params stops before conversion
        convert_timeout()
        marker_grace()
        heartbeat_timeout()
        if native_opts.get("max_params") is not None:
            native_opts["max_params"] = _ckpt_options().parse_max_params(native_opts["max_params"])
    except ValueError as e:
        return "error", str(e), hf_path, None, None
    # a Hub id's config.json is read only on an explicit opt-in (the same file on every node)
    family = _detect_native_family(hf_path, fetch=native_opts["enabled"] is True)
    if choose_native(native_opts["enabled"], family):
        if family is None:
            return (
                "error",
                "native_ckpt_convert=true but no native converter matches the HF "
                f"model at {hf_path} (supported: {NATIVE_SUPPORTED_DESC}). "
                "Set native_ckpt_convert: false to use the Megatron-Bridge path.",
                hf_path,
                native_opts,
                family,
            )
        too_big = _hub_id_too_big_reason(hf_path, native_opts.get("max_params"))
        if too_big:
            return "error", too_big, hf_path, native_opts, family
        return "native", family, hf_path, native_opts, family
    block = bridge_unsupported_reason(family, hf_path)
    if block:
        return "error", block, hf_path, native_opts, family
    if family is None:
        reason = f"no native converter for this model (native: {NATIVE_SUPPORTED_DESC})"
    else:
        reason = "native_ckpt_convert: false"
    return "bridge", reason, hf_path, native_opts, family


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Megatron format")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args, _ = parser.parse_known_args()

    # Get config file path
    config_file = args.config
    if not os.path.isabs(config_file):
        config_file = str(PRIMUS_ROOT / config_file)

    log_info("Preparing Megatron SFT checkpoint...")

    action, detail, hf_path, native_opts, family = plan_conversion(config_file)
    if action == "pretrained":
        log_info(f"Pretrained checkpoint already configured: {detail}, skipping conversion")
        return
    if action == "no_hf_path":
        log_info("No hf_path found in config, assuming checkpoint already exists")
        return
    if action == "error":
        log_error(detail)
        sys.exit(1)
    if action == "native":
        _native_main(hf_path, native_opts, family)
        return
    log_info(f"Using Megatron-Bridge conversion path ({detail})")

    # Set paths
    data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
    megatron_path = data_path / "megatron_checkpoints" / Path(hf_path).name

    log_info(f"HF Model: {hf_path}")
    log_info(f"Megatron Path: {megatron_path}")

    # Check if a complete Megatron checkpoint already exists (a partial dir is converted again)
    if _checkpoint_ready(megatron_path):
        log_info(f"Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        print(f"extra.pretrained_checkpoint={megatron_path}")
        print(f"extra.finetune=true")
        return

    # Convert checkpoint (only on rank 0, others wait)
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        # Rank 0: perform the conversion
        log_info("Converting HF checkpoint to Megatron format using Megatron-Bridge...")
        megatron_path.parent.mkdir(parents=True, exist_ok=True)

        with _rank0_conversion(done_file, lock_file):
            convert_checkpoint(hf_path, str(megatron_path))

            # Fix metadata and directory structure for converted checkpoints
            # Megatron-Bridge creates iter_0000000 directory with iteration=0 metadata
            # But Megatron-LM requires:
            #   - iteration > 0 OR metadata = "release"
            #   - If metadata = "release", checkpoint must be in "release/" directory

            metadata_file = _tracker_file(megatron_path)
            iter_dir = megatron_path / "iter_0000000"
            release_dir = megatron_path / "release"

            if metadata_file.exists() and iter_dir.exists():
                with open(metadata_file, "r") as f:
                    content = f.read().strip()

                if content == "0":
                    log_info("Fixing HuggingFace converted checkpoint structure:")

                    # Step 1: Update metadata file
                    log_info("  1. Changing metadata from '0' to 'release'")
                    with open(metadata_file, "w") as f:
                        f.write("release")

                    # Step 2: Rename directory to match
                    if not release_dir.exists():
                        log_info("  2. Renaming 'iter_0000000' -> 'release'")
                        iter_dir.rename(release_dir)

                    log_success("Checkpoint structure fixed for Megatron-LM compatibility")

            # Step 3: Add 'args' to common.pt for Megatron-LM compatibility
            # Megatron-Bridge saves config to run_config.yaml, but Megatron-LM expects 'args' in common.pt
            fix_common_pt_for_megatron_lm(release_dir if release_dir.exists() else iter_dir)

            log_success(f"Checkpoint prepared at {megatron_path}")
    else:
        # Other ranks: wait for rank 0 to complete
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete checkpoint conversion...")
        # complete: Megatron-Bridge writes the tracker file last (fixed up above by rank 0)
        wait_for_conversion(
            done_file,
            lock_file,
            ready=lambda: _checkpoint_ready(megatron_path),  # the same check as rank 0's skip
        )
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    # Output the checkpoint path for the main training process
    # Use pretrained_checkpoint + finetune=true (same as Megatron-Bridge workflow)
    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


if __name__ == "__main__":
    main()
