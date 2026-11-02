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
import importlib.util
import json
import os
import sys
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


def read_native_opts(config_file: str) -> dict:
    """Read native (bridge-free) conversion options from the config.

    Returns a dict with ``enabled`` (tri-state: ``None`` when the config does not
    set ``native_ckpt_convert`` at all, else ``True`` / ``False``), optional
    ``dtype`` / ``out`` overrides, and the training config's ``use_rope_scaling``
    / ``rope_scaling_factor`` (``_native_main`` logs when HF's llama3 RoPE
    scaling replaces them). ``None`` lets the caller pick the default
    (native for supported families); ``True`` forces native; ``False`` forces the
    legacy Megatron-Bridge path.
    """
    cfg = load_primus_config(Path(config_file), None)
    opts = {
        "enabled": None,
        "dtype": None,
        "out": None,
        "use_rope_scaling": None,
        "rope_scaling_factor": None,
    }

    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is None:
            continue
        params = getattr(module, "params", None)

        enabled = _first_config_value(params, "native_ckpt_convert")
        if enabled is None:
            enabled = _first_config_value(module, "native_ckpt_convert")
        if enabled is not None:
            opts["enabled"] = bool(enabled)
        opts["dtype"] = _first_config_value(params, "native_ckpt_dtype") or _first_config_value(
            module, "native_ckpt_dtype"
        )
        opts["out"] = _first_config_value(params, "native_ckpt_out") or _first_config_value(
            module, "native_ckpt_out"
        )
        for key in ("use_rope_scaling", "rope_scaling_factor"):
            value = _first_config_value(params, key)
            opts[key] = value if value is not None else _first_config_value(module, key)
        break

    return opts


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
# DEFAULT conversion path for supported configs (Qwen3, DeepSeek, Llama, and
# Mistral with full causal attention): this
# hook builds the mcore model in-process and copies HF safetensors weights onto
# it (Primus-managed converters), producing a legacy ``torch`` checkpoint WITHOUT
# ever importing ``megatron.bridge``. The convert-time monkeypatches (ROCm
# fused-kernels no-op) are applied in THIS process via ``run_patches(phase=
# "convert")`` so they take effect for the single-process conversion; the
# Megatron-LM submodule stays pristine. Set ``native_ckpt_convert: false`` to
# force the legacy Megatron-Bridge path (or ``true`` to force native).


def _llama_config_rules():
    """``primus/backends/megatron/checkpoint/llama_config_rules.py``, loaded by path.

    It is standard library only. Loading it by path (not as
    ``primus.backends.megatron...``) does not import that package, torch or
    Megatron, so this routing check stays a pure config.json check.
    """
    name = "_primus_llama_config_rules"
    if name not in sys.modules:
        path = PRIMUS_ROOT / "primus/backends/megatron/checkpoint/llama_config_rules.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


_RULES = _llama_config_rules()

# HF architectures / model_type -> Primus native converter family. Llama and dense
# Mistral (e.g. Mistral-7B-v0.2/v0.3, the Llama weight layout) use the Llama
# converter; which of their configs it supports is decided by llama_config_rules,
# which the converter itself uses.
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
    **dict.fromkeys(_RULES.LLAMA_NAMES + _RULES.MISTRAL_NAMES, "llama"),
}

_SUPPORTED = "Qwen3 dense/MoE, DeepSeek-V2/V3, Llama 2/3/3.x, Mistral with full causal attention"


def _native_support(cfg: dict) -> tuple[str | None, str | None]:
    """Return ``(family, None)`` for a config a native converter supports, else
    ``(None, reason)``. family is 'qwen3' | 'deepseek' | 'llama'."""
    try:
        names = _RULES.config_names(cfg)
    except ValueError as e:
        return None, str(e)
    # architectures first; a model_type of another converter family is a conflict
    from_arch = {_NATIVE_FAMILIES[n] for n in names[:-1] if n in _NATIVE_FAMILIES}
    from_type = _NATIVE_FAMILIES.get(names[-1])
    if len(from_arch) > 1 or (from_arch and from_type and from_type not in from_arch):
        return None, f"architectures/model_type {names} name different model families"
    family = from_arch.pop() if from_arch else from_type
    if family is None:
        return None, f"no native converter for architectures/model_type {names} (supported: {_SUPPORTED})"
    if family == "llama":
        reason = _RULES.unsupported_reason(cfg)
        if reason:
            return None, f"the native Llama/Mistral converter does not support {reason}"
    return family, None


def _read_hf_config(hf_path: str) -> tuple[dict | None, str | None]:
    """``(config, None)`` for the parsed HF ``config.json`` object, else ``(None, reason)``."""
    cfg_file = os.path.join(hf_path, "config.json")
    if not os.path.isfile(cfg_file):
        return None, f"no config.json in {hf_path}"
    try:
        with open(cfg_file) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"cannot read {cfg_file}: {e}"
    if not isinstance(cfg, dict):
        return None, f"{cfg_file} is not a JSON object"
    return cfg, None


def _native_support_for_path(hf_path: str) -> tuple[str | None, str | None]:
    cfg, reason = _read_hf_config(hf_path)
    return (None, reason) if cfg is None else _native_support(cfg)


def _llama3_rope_args(hf_cfg: dict, native_opts: dict) -> tuple[dict, str | None]:
    """The training arguments that apply the HF config's llama3 RoPE scaling,
    and an info message when they change what the training config sets.

    The converter copies weights only; the RoPE scaling is a training option.
    Megatron takes ``use_rope_scaling`` and ``rope_scaling_factor`` from the
    training config and fixes the other llama3 parameters
    (``MEGATRON_LLAMA3_ROPE_FIXED``; a config with other values is not
    converted natively). Used on both conversion paths, so a checkpoint trains
    with the same RoPE whichever converter made it. ``({}, None)`` for a config
    without llama3 scaling."""
    params = _RULES.llama3_rope_scaling(hf_cfg)
    if params is None:
        return {}, None
    problem = _RULES.llama3_rope_problem(params)
    factor = params.get("factor")
    if problem and problem.startswith("factor "):
        # only on the Megatron-Bridge path (the native rules reject this config)
        return {}, f"the HF config's llama3 RoPE scaling is not applied: {problem}"
    factor = float(factor)
    args = {"use_rope_scaling": "true", "rope_scaling_factor": repr(factor)}
    notes = []
    if not (native_opts.get("use_rope_scaling") and native_opts.get("rope_scaling_factor") == factor):
        notes.append(
            f"the HF config has llama3 RoPE scaling (factor {factor}); the training config has "
            f"use_rope_scaling {native_opts.get('use_rope_scaling')!r}, rope_scaling_factor "
            f"{native_opts.get('rope_scaling_factor')!r}: training uses HF's "
            f"(use_rope_scaling true, rope_scaling_factor {factor})"
        )
    if problem:
        # only on the Megatron-Bridge path (the native rules reject this config)
        notes.append(f"Megatron cannot reproduce the rest of the HF llama3 RoPE scaling: {problem}")
    return args, "; ".join(notes) or None


def _emit_checkpoint_args(megatron_path: Path, extra: dict):
    """Print the ``extra.*`` lines the runner turns into training arguments."""
    for name, value in extra.items():
        print(f"extra.{name}={value}")
    print(f"extra.pretrained_checkpoint={megatron_path}")
    print("extra.finetune=true")


def _native_dtype_from_config(hf_path: str, override: str | None, hf_cfg: dict | None = None) -> str:
    """``override``, else the HF config's dtype (``hf_cfg`` when the caller already
    read it), else bf16."""
    if override:
        return override
    cfg = hf_cfg if hf_cfg is not None else _read_hf_config(hf_path)[0] or {}
    # transformers 5.x writes "dtype", 4.x "torch_dtype"
    td = str(cfg.get("dtype") or cfg.get("torch_dtype") or "").lower()
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
    family = opts["family"]  # _native_main sets it from main()'s _native_support

    # Ensure the Megatron-LM source tree is importable in this standalone hook
    # process (run_pretrain.sh's PYTHONPATH does not include it), and register +
    # apply the convert-phase patches BEFORE any Megatron model build.
    from primus.backends.megatron.checkpoint import native_convert_common as common

    common.ensure_megatron_on_path()
    applied = common.apply_convert_patches(module_name="checkpoint_convert")
    log_info(f"Applied {applied} convert-phase patch(es)")

    dtype = _native_dtype_from_config(hf_path, opts.get("dtype"), opts.get("hf_config"))
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
    else:
        from primus.backends.megatron.checkpoint import native_loader_deepseek as conv

    conv.convert(
        hf_path,
        megatron_path,
        dtype=dtype,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
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
    """Wait for rank 0 to complete checkpoint conversion."""
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


def _native_main(hf_path: str, native_opts: dict, family: str, hf_cfg: dict | None = None):
    """Bridge-free native conversion branch of main().

    Mirrors the Bridge path's skip-if-exists / rank-0-converts-others-wait
    semantics, but produces a legacy ``torch`` checkpoint via the Primus-managed
    in-process converter and emits ``pretrained_checkpoint`` + ``finetune`` back
    to the runner. ``family`` is the converter family main() detected from
    ``hf_cfg``, the parsed HF config.json (read here when not given).
    """
    if hf_cfg is None:
        hf_cfg = _read_hf_config(hf_path)[0] or {}
    native_opts["family"] = family
    native_opts["hf_config"] = hf_cfg
    rope_args, rope_note = _llama3_rope_args(hf_cfg, native_opts)
    if rope_note:
        log_info(rope_note)

    # Output path (legacy torch layout: <out>/iter_0000001/mp_rank_00/...).
    if native_opts.get("out"):
        megatron_path = Path(native_opts["out"])
    else:
        data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
        megatron_path = data_path / "megatron_checkpoints" / f"{Path(hf_path).name}_native"

    log_info(f"HF Model: {hf_path}  (native family: {family})")
    log_info(f"Megatron Path: {megatron_path}")

    if megatron_path.exists() and (megatron_path / "latest_checkpointed_iteration.txt").exists():
        log_info(f"Native Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        _emit_checkpoint_args(megatron_path, rope_args)
        return

    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
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

    _emit_checkpoint_args(megatron_path, rope_args)


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Megatron format")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args, _ = parser.parse_known_args()

    # Get config file path
    config_file = args.config
    if not os.path.isabs(config_file):
        config_file = str(PRIMUS_ROOT / config_file)

    log_info("Preparing Megatron SFT checkpoint...")

    # Extract hf_path and pretrained_checkpoint from config
    hf_path, pretrained_checkpoint = get_checkpoint_config(config_file)

    # If pretrained_checkpoint is already configured, skip conversion
    if pretrained_checkpoint:
        log_info(f"Pretrained checkpoint already configured: {pretrained_checkpoint}, skipping conversion")
        return

    # If no hf_path, nothing to convert
    if not hf_path:
        log_info("No hf_path found in config, assuming checkpoint already exists")
        return

    # ---- Choose conversion backend --------------------------------------
    # Native (bridge-free) conversion is the DEFAULT for supported configs
    # (Qwen3, DeepSeek, Llama, Mistral with full causal attention; the Llama
    # converter rejects some configs, see _native_support). ``native_ckpt_convert:
    # false`` forces the legacy Megatron-Bridge path (no config check);
    # ``native_ckpt_convert: true`` forces native.
    native_opts = read_native_opts(config_file)
    enabled = native_opts["enabled"]
    hf_cfg, read_reason = _read_hf_config(hf_path)
    if enabled is False:
        native_reason = "native_ckpt_convert: false"  # no config check
    else:
        native_reason = read_reason
        native_family = None
        if hf_cfg is not None:
            native_family, native_reason = _native_support(hf_cfg)
        if native_family is not None:
            _native_main(hf_path, native_opts, native_family, hf_cfg)
            return
        if enabled is True:
            log_error(
                f"native_ckpt_convert: true but HF model at {hf_path}: {native_reason}. "
                "Set native_ckpt_convert: false (or leave it unset) to use the Megatron-Bridge path."
            )
            sys.exit(1)

    log_info(f"Using Megatron-Bridge conversion path ({native_reason})")
    rope_args, rope_note = _llama3_rope_args(hf_cfg or {}, native_opts)
    if rope_note:
        log_info(rope_note)

    # Set paths
    data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
    megatron_path = data_path / "megatron_checkpoints" / Path(hf_path).name

    log_info(f"HF Model: {hf_path}")
    log_info(f"Megatron Path: {megatron_path}")

    # Check if Megatron checkpoint already exists
    if megatron_path.exists():
        log_info(f"Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        _emit_checkpoint_args(megatron_path, rope_args)
        return

    # Convert checkpoint (only on rank 0, others wait)
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
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
    _emit_checkpoint_args(megatron_path, rope_args)


if __name__ == "__main__":
    main()
