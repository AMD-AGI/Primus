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
"""

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Add primus to path
PRIMUS_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PRIMUS_ROOT))

from primus.core.config.primus_config import get_module_config, load_primus_config
from primus.core.utils.arg_utils import (
    as_int,
    is_unset,
    parse_cli_overrides,
    parse_switch,
)


def _is_rank_0() -> bool:
    """Check if current process is rank 0."""
    return int(os.environ.get("RANK", os.environ.get("NODE_RANK", 0))) == 0


def _node_rank() -> int:
    """This node's rank for conversion coordination (rank 0 converts, the others wait):
    ``NODE_RANK``, else ``RANK``, else 0. ``_is_rank_0`` (log output) keeps the base order."""
    return int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))


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


# keys whose value is a path: a bare ``--key`` (parsed as True) does not name a checkpoint or source
_PATH_KEYS = ("load", "pretrained_checkpoint", "hf_path", "tokenizer_model", "native_ckpt_out")


def _cli_values(cli_overrides: list[str] | None) -> dict:
    """The trainer's command-line overrides (``--key value`` / ``key=value``, the tokens after
    ``--config``) as a dict with ``_`` keys (``--native-ckpt-out`` is ``native_ckpt_out``), parsed
    as the trainer parses them (``parse_cli_overrides``: a bare ``--flag`` is true; for a path key
    in ``_PATH_KEYS`` a bare flag is dropped, so it neither counts as a configured checkpoint nor
    clears the YAML value). The hook
    target ``<group> <name>`` (e.g. ``train posttrain``) that execute_hooks.sh passes first is not
    an override: at most these two leading tokens are dropped, a third non-option token is kept
    (``parse_cli_overrides`` warns about it). An unset value (``--load null``) is kept: it clears
    the YAML value of that key, as in training."""
    tokens = [str(t) for t in (cli_overrides or [])]
    for _ in range(2):
        if tokens and not tokens[0].startswith("-") and "=" not in tokens[0]:
            tokens.pop(0)
    values = {str(k).replace("-", "_"): v for k, v in parse_cli_overrides(tokens, type_mode="modern").items()}
    for key in _PATH_KEYS:
        if values.get(key) is True:  # a bare ``--load``: no path given; ignored, the YAML value applies
            log_info(f"Checkpoint conversion: --{key} without a value is not a path; it is ignored")
            del values[key]
    return values


def _module_value_reader(config_file: str, cli_overrides: list[str] | None = None):
    """``value(*keys)`` for the first SFT module (``sft_trainer`` / ``post_trainer``) of the config,
    with the trainer's command-line overrides applied. None if the config has neither module.

    Lookup order: the module ``params`` (each key in order), then the module itself (each key in
    order); the first set value wins. Example: ``value("hf_path", "tokenizer_model")`` returns
    ``params.hf_path``, else ``params.tokenizer_model``, else ``module.hf_path``, else
    ``module.tokenizer_model``.

    The same rules apply to every model family. Unset (``is_unset``): None, "", and the strings
    null / ~ in any case (``none`` is a value, e.g. a path named ``none``). A command-line
    override replaces the ``params`` value of the same key, as the trainer applies overrides to
    the module params; an unset override (``--load null``) clears it, so the lookup goes on to the
    next key (it does not fall back to the YAML value of that key)."""
    cfg = load_primus_config(Path(config_file), None)
    cli = _cli_values(cli_overrides)

    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is None:
            continue
        params = getattr(module, "params", None)

        def value(*keys):
            for key in keys:
                v = cli[key] if key in cli else getattr(params, key, None)
                if not is_unset(v):
                    return v
            for key in keys:
                if key in cli:  # the command line set (or cleared) this key
                    continue
                v = getattr(module, key, None)
                if not is_unset(v):
                    return v
            return None

        return value
    return None


def _checkpoint_paths(value) -> tuple[str | None, str | None]:
    """(HF source, configured Megatron checkpoint) from a ``_module_value_reader`` reader."""
    if value is None:
        return None, None
    # Old Megatron-Bridge SFT configs used hf_path, while native Megatron configs typically reuse
    # tokenizer_model as the HF source identifier. A configured Megatron-format checkpoint
    # (pretrained_checkpoint or load) skips the conversion hook.
    return value("hf_path", "tokenizer_model"), value("pretrained_checkpoint", "load")


_NATIVE_OPT_KEYS = {
    "dtype": "native_ckpt_dtype",
    "out": "native_ckpt_out",
    "wait_timeout": "native_ckpt_wait_timeout",
    "model_type": "model_type",
    "tensor_model_parallel_size": "tensor_model_parallel_size",
    "pipeline_model_parallel_size": "pipeline_model_parallel_size",
    "expert_model_parallel_size": "expert_model_parallel_size",
    "make_vocab_size_divisible_by": "make_vocab_size_divisible_by",
    "olmo3_rope_layout": "olmo3_rope_layout",
}


def _native_opts(value) -> dict:
    """Native conversion options from a ``_module_value_reader`` reader (see read_native_opts)."""
    opts = {k: None for k in ("enabled", *_NATIVE_OPT_KEYS)}
    if value is None:
        return opts
    enabled = value("native_ckpt_convert")
    if enabled is not None:
        opts["enabled"] = parse_switch(enabled, "native_ckpt_convert")
    for key, config_key in _NATIVE_OPT_KEYS.items():
        opts[key] = value(config_key)
    return opts


def get_checkpoint_config(
    config_file: str, cli_overrides: list[str] | None = None
) -> tuple[str | None, str | None]:
    """
    Extract HF source path and any already-configured checkpoint path.

    Public API for external callers (with :func:`read_native_opts`); ``main`` reads both at once
    through :func:`read_hook_settings`, which uses the same reader.

    ``cli_overrides``: the trainer's command-line overrides; they replace the module params of the
    same key (see _module_value_reader). None: the YAML only.

    Returns:
        tuple: (hf_path, checkpoint_path)
            - hf_path: HuggingFace model path to convert
            - checkpoint_path: Existing Megatron checkpoint path (if configured)
    """
    return _checkpoint_paths(_module_value_reader(config_file, cli_overrides))


def read_native_opts(config_file: str, cli_overrides: list[str] | None = None) -> dict:
    """Read native (bridge-free) conversion options from the config (same reading as
    get_checkpoint_config, for every family).

    Returns a dict with ``enabled`` (tri-state: ``None`` when the config does not set
    ``native_ckpt_convert``, else ``True`` / ``False`` from ``parse_switch``: true / false, 1 / 0,
    yes / no, on / off; any other value raises ValueError), plus ``dtype`` / ``out`` overrides.
    ``None`` lets the caller pick the default (native for supported families); ``True`` forces
    native; ``False`` forces the Megatron-Bridge path. ``wait_timeout`` (``native_ckpt_wait_timeout``)
    is read only by the OLMo-3 flow (how long the other nodes wait for rank 0); the other families
    and the Bridge path keep their fixed wait.
    ``model_type`` (``olmo3``) recognises an OLMo-3 run; the training layout and
    ``olmo3_rope_layout`` are read only by the OLMo-3 flow
    (primus/backends/megatron/checkpoint/native_flow_olmo3.py).
    """
    return _native_opts(_module_value_reader(config_file, cli_overrides))


def read_hook_settings(config_file: str, cli_overrides: list[str] | None = None):
    """(hf_path, pretrained_checkpoint, native_opts) with one reading of the config and the
    command-line overrides. main() and 00_install_requirements.sh both call it with the arguments
    execute_hooks.sh forwards, so they see the same settings. ValueError for a value the hook
    cannot read (e.g. ``native_ckpt_convert: maybe``)."""
    value = _module_value_reader(config_file, cli_overrides)
    hf_path, pretrained_checkpoint = _checkpoint_paths(value)
    native_opts = _native_opts(value)
    return hf_path, pretrained_checkpoint, native_opts


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
# DEFAULT conversion path for supported families (Qwen3, DeepSeek, Llama, OLMo-3): this
# hook builds the mcore model in-process and copies HF safetensors weights onto
# it (Primus-managed converters), producing a legacy ``torch`` checkpoint WITHOUT
# ever importing ``megatron.bridge``. The convert-time monkeypatches (ROCm
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
    "olmo3": "olmo3",
    "olmo3forcausallm": "olmo3",
}


def _detect_native_family(hf_path: str) -> str | None:
    """Return 'qwen3' | 'deepseek' | 'llama' | 'olmo3' | None from the HF config.json."""
    cfg_file = os.path.join(hf_path, "config.json")
    if not os.path.isfile(cfg_file):
        return None
    with open(cfg_file) as f:
        cfg = json.load(f)

    for arch in cfg.get("architectures", []) or []:
        fam = _NATIVE_FAMILIES.get(str(arch).lower())
        if fam:
            return fam
    mt = str(cfg.get("model_type", "")).lower()
    return _NATIVE_FAMILIES.get(mt)


def _native_dtype_from_config(hf_path: str, override: str | None) -> str:
    if override:
        return override
    cfg_file = os.path.join(hf_path, "config.json")
    if os.path.isfile(cfg_file):
        with open(cfg_file) as f:
            td = str(json.load(f).get("torch_dtype", "")).lower()
        if "bfloat16" in td:
            return "bf16"
        if "float16" in td:
            return "fp16"
        if "float32" in td:
            return "fp32"
    return "bf16"


def native_convert_checkpoint(hf_path: str, megatron_path: str, opts: dict):
    """Convert a HF checkpoint to a legacy Megatron torch ckpt, bridge-free.

    Applies the ``phase="convert"`` patches in this process, then runs the
    Primus-managed native converter for the detected family, all single-process.
    """
    family = opts.get("family") or _detect_native_family(hf_path)
    if family is None:
        raise ValueError(
            f"native_ckpt_convert requested but no native converter for HF model at {hf_path} "
            "(supported: Qwen3 dense/MoE, DeepSeek-V2/V3, Llama 2/3/3.1, OLMo-3)"
        )

    # Ensure the Megatron-LM source tree is importable in this standalone hook
    # process (run_pretrain.sh's PYTHONPATH does not include it), and register +
    # apply the convert-phase patches BEFORE any Megatron model build.
    from primus.backends.megatron.checkpoint import native_convert_common as common

    common.ensure_megatron_on_path()
    applied = common.apply_convert_patches(module_name="checkpoint_convert")
    log_info(f"Applied {applied} convert-phase patch(es)")

    dtype = _native_dtype_from_config(hf_path, opts.get("dtype"))
    # The native converters write a legacy ``torch`` checkpoint (ckpt_format torch) for one rank,
    # TP=PP=EP=1, for every family. A legacy checkpoint is not resharded at load. OLMo-3 refuses
    # another training layout before any download (native_flow_olmo3.single_shard_conflict); the
    # other families get a warning (_warn_single_shard_layout). The layout keys in ``opts`` are read only by
    # those checks, never as the conversion layout.
    tp = pp = ep = 1

    log_info(f"Native ({family}) conversion: dtype={dtype} TP={tp} PP={pp} EP={ep}")
    log_info(f"  Source: {hf_path}")
    log_info(f"  Target: {megatron_path}")

    if family == "qwen3":
        from primus.backends.megatron.checkpoint import native_loader_qwen3 as conv
    elif family == "llama":
        from primus.backends.megatron.checkpoint import native_loader_llama as conv
    elif family == "olmo3":
        from primus.backends.megatron.checkpoint import native_loader_olmo3 as conv
    else:
        from primus.backends.megatron.checkpoint import native_loader_deepseek as conv

    extra = {}
    if family == "olmo3":
        from primus.backends.megatron.checkpoint import native_flow_olmo3

        extra = native_flow_olmo3.converter_options(opts)
    if extra:
        log_info(f"  Converter options: {extra}")

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


def wait_for_conversion(done_file: Path, lock_file: Path, timeout: int = 600):
    """Wait for rank 0 to complete checkpoint conversion (Qwen3 / Llama / DeepSeek and the
    Megatron-Bridge path). The OLMo-3 flow has its own wait with
    ``native_ckpt_wait_timeout`` (native_flow_olmo3.wait_for_checkpoint)."""
    elapsed = 0
    while not done_file.exists() and elapsed < timeout:
        if not lock_file.exists() and not done_file.exists():
            time.sleep(2)
        else:
            time.sleep(5)
        elapsed += 5

    if not done_file.exists():
        raise TimeoutError("Timeout waiting for checkpoint conversion")


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


def _olmo3_native_flow(hf_path: str, megatron_path: Path, native_opts: dict):
    """OLMo-3: run native_flow_olmo3.run_native_flow; a refusal or failure exits 1 with the reason."""
    from types import SimpleNamespace

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    node_rank = _node_rank()
    log = SimpleNamespace(info=log_info, error=log_error, success=log_success)
    try:
        flow.run_native_flow(
            hf_path,
            megatron_path,
            native_opts,
            native_convert_checkpoint,
            _native_dtype_from_config(hf_path, native_opts.get("dtype")),
            node_rank,
            log,
        )
    except flow.ConversionFailed as e:
        log_error(f"[RANK {node_rank}] {e}")
        sys.exit(1)


# The model name (the last part of the HF id) starts with "olmo3", "olmo-3" or "olmo_3" followed by
# a separator or the end: Olmo-3-1025-7B matches; OLMo-30B, olmo3x, olmo_32b, olmo-35 and
# llama-olmo3-distill do not.
_OLMO3_NAME = re.compile(r"^olmo[-_]?3(?![0-9a-z])", re.IGNORECASE)


def _trains_olmo3(native_opts: dict) -> bool:
    """The training config builds OLMo-3 (``model_type: olmo3``)."""
    return str(native_opts.get("model_type") or "").strip().lower() == "olmo3"


def _is_olmo3_run(hf_path, native_family, native_opts: dict) -> bool:
    """True when the run is OLMo-3: the training config's ``model_type`` is ``olmo3`` (the model
    built is OLMo-3, so only an OLMo-3 checkpoint can load; a checkpoint of another family is then
    refused by :func:`_olmo3_bridge_refusal`), or the HF config.json says so (``native_family``),
    or, when there is no local config.json (a hub id), the HF id names OLMo-3
    (``allenai/Olmo-3-1025-7B``; the name must start with ``olmo3`` / ``olmo-3`` / ``olmo_3``
    followed by a separator or the end, so ``OLMo-30B``, ``olmo3x`` or ``llama-olmo3-distill`` do
    not count). The preset file name is not used."""
    if _trains_olmo3(native_opts):
        return True
    if native_family is not None:
        return native_family == "olmo3"
    return bool(hf_path) and _OLMO3_NAME.search(Path(str(hf_path)).name) is not None


def _default_native_out(hf_path) -> Path:
    """The native checkpoint directory when ``native_ckpt_out`` is not set:
    ``<DATA_PATH>/megatron_checkpoints/<name>_native`` (``<name>``: the last part of ``hf_path``)."""
    data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
    return data_path / "megatron_checkpoints" / f"{Path(str(hf_path)).name}_native"


def _resolve_hub_snapshot(hf_path, native_opts: dict):
    """A complete local snapshot (config.json and all safetensors weights) of the OLMo-3 hub id
    ``hf_path`` (``org/name``), else None. A local path is not a hub id (None).

    Rank 0 takes it from the Hugging Face cache or downloads it; the other nodes wait for it in the
    shared cache and stop when rank 0 cannot get it (native_flow_olmo3.resolve_hub_snapshot)."""
    from types import SimpleNamespace

    from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

    if not flow.is_hub_id(hf_path):
        return None
    out = native_opts.get("out") or _default_native_out(hf_path)
    node_rank = _node_rank()
    log = SimpleNamespace(info=log_info, error=log_error, success=log_success)
    return flow.resolve_hub_snapshot(str(hf_path), out, native_opts, node_rank, log)


def _olmo3_refusal_kind(native_family, native_opts: dict) -> str | None:
    """For an OLMo-3 run: why it cannot convert, or None. The one rule that conversion_plan (plan
    "refuse") and _olmo3_bridge_refusal (the message) both use:

    * ``"disabled"``: ``native_ckpt_convert`` is false (the Bridge path has no OLMo-3);
    * ``"foreign"``: the HF config.json is of another family;
    * ``"no_config"``: no local config.json (a hub id not resolved to a snapshot, or a directory
      without config.json); conversion_plan still says "native" here, because _main() first
      resolves a hub id to its snapshot."""
    if native_opts.get("enabled") is False:
        return "disabled"
    if native_family is None:
        return "no_config"
    return "foreign" if native_family != "olmo3" else None


def _olmo3_bridge_refusal(hf_path, native_family, native_opts: dict) -> str | None:
    """The error when an OLMo-3 run (:func:`_is_olmo3_run`) cannot convert
    (:func:`_olmo3_refusal_kind`), else None. Other families are not affected."""
    if not _is_olmo3_run(hf_path, native_family, native_opts):
        return None
    kind = _olmo3_refusal_kind(native_family, native_opts)
    if kind == "disabled":  # the run set native_ckpt_convert to false
        return (
            f"OLMo-3 checkpoint conversion needs the native converter, but native_ckpt_convert is "
            f"false; the Megatron-Bridge path has no OLMo-3. Remove native_ckpt_convert: false (or set "
            f"it to true), or set pretrained_checkpoint to a converted checkpoint. HF source: {hf_path}"
        )
    if kind == "foreign":
        return (
            f"model_type: olmo3 builds an OLMo-3 model, but the HF checkpoint {hf_path} is "
            f"{native_family} (config.json); it cannot load into OLMo-3. Point hf_path (else "
            "tokenizer_model) at an OLMo-3 checkpoint, or set model_type to the checkpoint's family."
        )
    if kind == "no_config":
        from primus.backends.megatron.checkpoint import native_flow_olmo3 as flow

        why = (
            "it is a hub id that is not in the Hugging Face cache and could not be downloaded"
            if flow.is_hub_id(hf_path)
            else "it has no config.json"
        )
        return (
            f"OLMo-3 checkpoint conversion needs a local HF directory with config.json, got "
            f"{hf_path!r} (hf_path, else tokenizer_model): {why}. Download the model and set hf_path "
            "to its directory, or set pretrained_checkpoint to a converted checkpoint."
        )
    return None


def conversion_plan(hf_path, pretrained_checkpoint, native_opts: dict, native_family=None) -> str:
    """What the hook does with these settings, without converting or downloading anything:

    * ``"skip"``: a Megatron checkpoint is configured, or there is no HF source;
    * ``"native"``: the bridge-free converter (a supported family by default, or forced by
      ``native_ckpt_convert: true``; an OLMo-3 run given as a hub id, which the hook resolves to
      a local snapshot first);
    * ``"bridge"``: the Megatron-Bridge path (``native_ckpt_convert: false``, or an unsupported
      family; the install hook then installs its dependencies);
    * ``"refuse"``: an OLMo-3 run with ``native_ckpt_convert: false`` (the Bridge path has no
      OLMo-3), or a ``model_type: olmo3`` run given a checkpoint of another family.

    ``native_family``: ``_detect_native_family(hf_path)`` if the caller has it. Used by main() and
    by 00_install_requirements.sh, so both hooks decide the same way from the same settings."""
    if pretrained_checkpoint or not hf_path:
        return "skip"
    if native_family is None:
        native_family = _detect_native_family(hf_path)
    if _is_olmo3_run(hf_path, native_family, native_opts):
        return (
            "refuse"
            if _olmo3_refusal_kind(native_family, native_opts) in ("disabled", "foreign")
            else "native"
        )
    if native_opts.get("enabled") is True:
        return "native"
    if native_opts.get("enabled") is False or native_family is None:
        return "bridge"
    return "native"


_LAYOUT_KEYS = ("tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size")


def _warn_single_shard_layout(family: str, native_opts: dict):
    """Other families (not OLMo-3): the native converter writes a single-shard legacy ``torch``
    checkpoint (TP=PP=EP=1). A run that trains at TP / PP / EP > 1
    gets a warning (not a refusal, so their behaviour does not change); OLMo-3 refuses it up front."""
    # arg_utils only (not the OLMo-3 flow module), so these families do not depend on it. An
    # unreadable value only warns here: the training run reads it again and refuses it there.
    try:
        layout = {
            k: 1 if is_unset(native_opts.get(k)) else as_int(native_opts.get(k), k) for k in _LAYOUT_KEYS
        }
    except ValueError as e:
        layout = {"unreadable": str(e)}
    if any(v != 1 for v in layout.values()):
        log_info(
            f"[WARN] Native ({family}) conversion writes a single-shard legacy torch checkpoint "
            f"(TP=PP=EP=1), which is not resharded at load, but this run trains with {layout}. "
            "Convert once and reshard it to the training layout (docs/04-technical-guides/"
            "native-sft-lora.md), or set native_ckpt_convert: false for the Megatron-Bridge path."
        )


def _native_main(hf_path: str, native_opts: dict, family: str | None = None):
    """Bridge-free native conversion branch of main().

    Mirrors the Bridge path's skip-if-exists / rank-0-converts-others-wait
    semantics, but produces a legacy ``torch`` checkpoint via the Primus-managed
    in-process converter and emits ``pretrained_checkpoint`` + ``finetune`` back
    to the runner.

    OLMo-3 uses its own flow (primus/backends/megatron/checkpoint/native_flow_olmo3.py): the
    checkpoint carries a record of its conversion settings and source, which a reused checkpoint
    must match, and the other nodes stop when rank 0 stops.
    """
    family = family or _detect_native_family(hf_path)
    if family is None:
        log_error(f"native_ckpt_convert: true but no native converter matches HF model at {hf_path}")
        sys.exit(1)
    native_opts["family"] = family

    # Output path (legacy torch layout: <out>/iter_0000001/mp_rank_00/...).
    megatron_path = Path(native_opts["out"]) if native_opts.get("out") else _default_native_out(hf_path)

    log_info(f"HF Model: {hf_path}  (native family: {family})")
    log_info(f"Megatron Path: {megatron_path}")

    if family == "olmo3":
        _olmo3_native_flow(hf_path, megatron_path, native_opts)
        print(f"extra.pretrained_checkpoint={megatron_path}")
        print("extra.finetune=true")
        return

    _warn_single_shard_layout(family, native_opts)
    if megatron_path.exists() and (megatron_path / "latest_checkpointed_iteration.txt").exists():
        log_info(f"Native Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        print(f"extra.pretrained_checkpoint={megatron_path}")
        print(f"extra.finetune=true")
        return

    node_rank = _node_rank()
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        log_info("Converting HF checkpoint to Megatron format (native, bridge-free)...")
        megatron_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file.touch()
        try:
            native_convert_checkpoint(hf_path, str(megatron_path), native_opts)
            done_file.touch()
            log_success(f"Checkpoint prepared at {megatron_path}")
        finally:
            lock_file.unlink(missing_ok=True)
    else:
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete native conversion...")
        wait_for_conversion(done_file, lock_file)
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Megatron format")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args, cli_overrides = parser.parse_known_args()

    # Get config file path
    config_file = args.config
    if not os.path.isabs(config_file):
        config_file = str(PRIMUS_ROOT / config_file)

    log_info("Preparing Megatron SFT checkpoint...")

    # Read the settings once (the YAML with the command-line overrides applied, the same reading
    # for every family). Settings errors (a ValueError here, or from single_shard_conflict in
    # _main) are logged as one line and exit 1; errors during the conversion keep their traceback.
    try:
        hf_path, pretrained_checkpoint, native_opts = read_hook_settings(config_file, cli_overrides)
    except ValueError as e:  # a config value the hook cannot read (e.g. native_ckpt_convert: maybe)
        log_error(f"Checkpoint conversion settings: {e}")
        sys.exit(1)
    _main(hf_path, pretrained_checkpoint, native_opts)


def _exit_on_olmo3_layout_conflict(native_opts: dict):
    """Exit 1 with one line when an OLMo-3 run trains with TP / PP / EP > 1 (the native checkpoint
    is a single-shard legacy ``torch`` one): checked before any download or conversion."""
    from primus.backends.megatron.checkpoint import native_flow_olmo3

    try:
        conflict = native_flow_olmo3.single_shard_conflict(native_opts)
    except ValueError as e:  # e.g. tensor_model_parallel_size: two
        conflict = f"Checkpoint conversion settings: {e}"
    if conflict:
        log_error(conflict)
        sys.exit(1)


def _main(hf_path, pretrained_checkpoint, native_opts: dict):
    # If pretrained_checkpoint is already configured, skip conversion
    if pretrained_checkpoint:
        log_info(f"Pretrained checkpoint already configured: {pretrained_checkpoint}, skipping conversion")
        return

    # If no hf_path, nothing to convert
    if not hf_path:
        log_info("No hf_path found in config, assuming checkpoint already exists")
        return

    # ---- Choose conversion backend (conversion_plan) ----------------------
    # Native (bridge-free) conversion is the DEFAULT for supported families
    # (Qwen3, DeepSeek, Llama, OLMo-3). ``native_ckpt_convert: false`` forces the legacy
    # Megatron-Bridge path; ``native_ckpt_convert: true`` forces native. OLMo-3 has no Bridge path:
    # it is converted natively, and ``native_ckpt_convert: false`` (or a checkpoint of another
    # family) is refused.
    native_family = _detect_native_family(hf_path)
    plan = conversion_plan(hf_path, None, native_opts, native_family)
    is_olmo3 = _is_olmo3_run(hf_path, native_family, native_opts)
    if is_olmo3 and plan != "refuse":
        # the cheap OLMo-3 settings checks run before any hub download or conversion work
        _exit_on_olmo3_layout_conflict(native_opts)
        if native_family is None:
            # an OLMo-3 hub id (e.g. the preset's tokenizer_model): use its local snapshot. Rank 0
            # downloads it; the other nodes wait for it in the Hugging Face cache, which must be
            # shared.
            snapshot = _resolve_hub_snapshot(hf_path, native_opts)
            if snapshot is not None:
                log_info(f"OLMo-3 HF source {hf_path} -> local snapshot {snapshot}")
                if not native_opts.get("out"):  # name the output after the model, not the snapshot hash
                    native_opts["out"] = str(_default_native_out(hf_path))
                hf_path, native_family = snapshot, _detect_native_family(snapshot)
                plan = conversion_plan(hf_path, None, native_opts, native_family)  # on the snapshot

    # an OLMo-3 run that is refused (plan "refuse"), or has no local HF directory: one clear error
    # (the message is chosen by _olmo3_bridge_refusal), before the lock file and any conversion work
    refusal = _olmo3_bridge_refusal(hf_path, native_family, native_opts)  # None unless OLMo-3
    if refusal:
        log_error(refusal)
        sys.exit(1)

    if plan == "native":
        if native_family is None:
            log_error(
                "native_ckpt_convert=true but no native converter matches the HF "
                f"model at {hf_path} (supported: Qwen3, DeepSeek-V2/V3, Llama, OLMo-3). "
                "Set native_ckpt_convert: false to use the Megatron-Bridge path."
            )
            sys.exit(1)
        _native_main(hf_path, native_opts, native_family)
        return

    log_info("Using Megatron-Bridge conversion path (native disabled or unsupported family)")

    # Set paths
    data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
    megatron_path = data_path / "megatron_checkpoints" / Path(hf_path).name

    log_info(f"HF Model: {hf_path}")
    log_info(f"Megatron Path: {megatron_path}")

    # Check if Megatron checkpoint already exists
    if megatron_path.exists():
        log_info(f"Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        print(f"extra.pretrained_checkpoint={megatron_path}")
        print(f"extra.finetune=true")
        return

    # Convert checkpoint (only on rank 0, others wait)
    node_rank = _node_rank()
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        # Rank 0: perform the conversion
        log_info("Converting HF checkpoint to Megatron format using Megatron-Bridge...")
        megatron_path.parent.mkdir(parents=True, exist_ok=True)

        # Create lock file
        lock_file.touch()

        try:
            convert_checkpoint(hf_path, str(megatron_path))

            # Fix metadata and directory structure for converted checkpoints
            # Megatron-Bridge creates iter_0000000 directory with iteration=0 metadata
            # But Megatron-LM requires:
            #   - iteration > 0 OR metadata = "release"
            #   - If metadata = "release", checkpoint must be in "release/" directory

            metadata_file = megatron_path / "latest_checkpointed_iteration.txt"
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

            done_file.touch()
            log_success(f"Checkpoint prepared at {megatron_path}")
        finally:
            lock_file.unlink(missing_ok=True)
    else:
        # Other ranks: wait for rank 0 to complete
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete checkpoint conversion...")
        wait_for_conversion(done_file, lock_file)
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    # Output the checkpoint path for the main training process
    # Use pretrained_checkpoint + finetune=true (same as Megatron-Bridge workflow)
    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


if __name__ == "__main__":
    main()
