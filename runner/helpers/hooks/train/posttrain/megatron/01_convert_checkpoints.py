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
    set ``native_ckpt_convert`` at all, else ``True`` / ``False``), plus optional
    ``dtype`` / ``out`` overrides. ``None`` lets the caller pick the default
    (native for supported families); ``True`` forces native; ``False`` forces the
    legacy Megatron-Bridge path.
    """
    cfg = load_primus_config(Path(config_file), None)
    opts = {"enabled": None, "dtype": None, "out": None}

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
# DEFAULT conversion path for supported families (Qwen3, DeepSeek, Llama): this
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
}


def _detect_native_family(hf_path: str) -> str | None:
    """Return 'qwen3' | 'deepseek' | 'llama' | None from the HF config.json."""
    import json

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
    import json

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
            "(supported: Qwen3 dense/MoE, DeepSeek-V2/V3, Llama 2/3/3.1)"
        )

    # Ensure the Megatron-LM source tree is importable in this standalone hook
    # process (run_pretrain.sh's PYTHONPATH does not include it), and register +
    # apply the convert-phase patches BEFORE any Megatron model build.
    from primus.backends.megatron.checkpoint import native_convert_common as common

    common.ensure_megatron_on_path()
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

    log_info(f"Converting HF → Megatron checkpoint...")
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


def _native_main(hf_path: str, native_opts: dict):
    """Bridge-free native conversion branch of main().

    Mirrors the Bridge path's skip-if-exists / rank-0-converts-others-wait
    semantics, but produces a legacy ``torch`` checkpoint via the Primus-managed
    in-process converter and emits ``pretrained_checkpoint`` + ``finetune`` back
    to the runner.
    """
    family = _detect_native_family(hf_path)
    if family is None:
        log_error(
            f"native_ckpt_convert: true but no native converter matches HF model at {hf_path}"
        )
        sys.exit(1)
    native_opts["family"] = family

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
        print(f"extra.pretrained_checkpoint={megatron_path}")
        print(f"extra.finetune=true")
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

    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


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
    # Native (bridge-free) conversion is the DEFAULT for supported families
    # (Qwen3, DeepSeek, Llama). ``native_ckpt_convert: false`` forces the legacy
    # Megatron-Bridge path; ``native_ckpt_convert: true`` forces native.
    native_opts = read_native_opts(config_file)
    native_family = _detect_native_family(hf_path)
    if native_opts["enabled"] is True:
        use_native = True
    elif native_opts["enabled"] is False:
        use_native = False
    else:  # not set -> default to native when the family is supported
        use_native = native_family is not None

    if use_native:
        if native_family is None:
            log_error(
                "native_ckpt_convert=true but no native converter matches the HF "
                f"model at {hf_path} (supported: Qwen3, DeepSeek-V2/V3, Llama). "
                "Set native_ckpt_convert: false to use the Megatron-Bridge path."
            )
            sys.exit(1)
        _native_main(hf_path, native_opts)
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
    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


if __name__ == "__main__":
    main()
