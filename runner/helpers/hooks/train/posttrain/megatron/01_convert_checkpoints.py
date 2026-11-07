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
    ``dtype`` / ``out`` / ``ckpt_format`` overrides. ``None`` lets the caller pick
    the default (native for supported families); ``True`` forces native; ``False``
    forces the legacy Megatron-Bridge path. ``ckpt_format`` (``native_ckpt_format``)
    is ``torch_dist`` or ``torch``; ``None`` follows the experiment's ``ckpt_format``.
    """
    cfg = load_primus_config(Path(config_file), None)
    opts = {"enabled": None, "dtype": None, "out": None, "ckpt_format": None}

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
        opts["ckpt_format"] = _first_config_value(params, "native_ckpt_format") or _first_config_value(
            module, "native_ckpt_format"
        )
        break

    return opts


def read_experiment_checkpoint_settings(config_file: str) -> dict:
    """Read the experiment's parallel layout and checkpoint format.

    Returns ``{"tp", "pp", "ep", "ckpt_format"}``: the tensor / pipeline /
    expert parallel sizes (default 1) and ``ckpt_format`` (default
    ``torch_dist``, Megatron's default) the fine-tune runs with.
    """
    cfg = load_primus_config(Path(config_file), None)
    names = {
        "tensor_model_parallel_size": "tp",
        "pipeline_model_parallel_size": "pp",
        "expert_model_parallel_size": "ep",
        "ckpt_format": "ckpt_format",
    }
    settings = {"tp": 1, "pp": 1, "ep": 1, "ckpt_format": "torch_dist"}

    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is None:
            continue
        params = getattr(module, "params", None)
        for name, key in names.items():
            value = _first_config_value(params, name)
            if value is None:
                value = _first_config_value(module, name)
            if value is not None:
                settings[key] = str(value) if key == "ckpt_format" else int(value)
        break

    return settings


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
# DEFAULT conversion path for the registered families (``native_registry``): this
# hook builds the mcore model in-process and copies HF safetensors weights onto
# it (Primus-managed converters), producing a ``torch_dist`` checkpoint that
# Megatron reshards to the experiment's TP / PP / EP on load (or a legacy
# TP1/PP1 ``torch`` checkpoint for an experiment with ``ckpt_format: torch``)
# WITHOUT ever importing ``megatron.bridge``. The convert-time monkeypatches (ROCm
# fused-kernels no-op) are applied in THIS process via ``run_patches(phase=
# "convert")`` so they take effect for the single-process conversion; the
# Megatron-LM submodule stays pristine. Set ``native_ckpt_convert: false`` to
# force the legacy Megatron-Bridge path (or ``true`` to force native).
#
# Families, their HF names, converter modules and options are registered in
# ``primus.backends.megatron.checkpoint.native_registry`` (built-ins) and in one
# module per added family under ``.../checkpoint/native_families/``; nothing
# here is family-specific.


def _registry():
    from primus.backends.megatron.checkpoint import native_registry

    return native_registry


def _detect_native_family(hf_path: str) -> str | None:
    """The registered native family of the HF model at ``hf_path`` (its name), or None."""
    family = _registry().detect_family(hf_path)
    return family.name if family is not None else None


def _native_dtype_from_config(hf_path: str, override: str | None, family: str | None = None) -> str:
    """``override``, else the HF config's dtype (``bf16`` / ``fp16`` / ``fp32``; ``bf16`` if unknown)."""
    registry = _registry()
    fam = registry.get_family(family) if family else None
    return registry.dtype_from_hf_config(hf_path, override, fam)


def native_convert_checkpoint(hf_path: str, megatron_path: str, opts: dict):
    """Convert a HF checkpoint to a Megatron checkpoint, bridge-free.

    ``opts["ckpt_format"]`` selects ``torch_dist`` (default) or legacy ``torch``.
    ``opts["convert_kwargs"]`` (the family's options and its ``prepare`` plan)
    are passed on to the family's ``convert()``.

    Applies the ``phase="convert"`` patches in this process, then runs the
    Primus-managed native converter for the detected family, all single-process.
    """
    registry = _registry()
    family_name = opts.get("family") or _detect_native_family(hf_path)
    if family_name is None:
        raise ValueError(f"native_ckpt_convert requested but {registry.unknown_model_text(hf_path)}")
    family = registry.get_family(family_name)

    # Ensure the Megatron-LM source tree is importable in this standalone hook
    # process (run_pretrain.sh's PYTHONPATH does not include it), and register +
    # apply the convert-phase patches BEFORE any Megatron model build.
    from primus.backends.megatron.checkpoint import native_convert_common as common

    common.ensure_megatron_on_path()
    applied = common.apply_convert_patches(module_name="checkpoint_convert")
    log_info(f"Applied {applied} convert-phase patch(es)")

    dtype = _native_dtype_from_config(hf_path, opts.get("dtype"), family.name)
    tp = int(opts.get("tensor_parallel_size", 1))
    pp = int(opts.get("pipeline_parallel_size", 1))
    ep = int(opts.get("expert_parallel_size", 1))
    ckpt_format = common.check_ckpt_format(opts.get("ckpt_format") or common.DEFAULT_CKPT_FORMAT)
    extra = dict(opts.get("convert_kwargs") or {})

    log_info(f"Native ({family.name}) conversion: dtype={dtype} TP={tp} PP={pp} EP={ep} format={ckpt_format}")
    if extra:
        log_info(f"  {family.label} options: {extra}")
    log_info(f"  Source: {hf_path}")
    log_info(f"  Target: {megatron_path}")

    conv = family.converter_module()
    conv.convert(
        hf_path,
        megatron_path,
        dtype=dtype,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        ckpt_format=ckpt_format,
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


def _emit_native_checkpoint_args(megatron_path: Path):
    """Point training at the converted checkpoint."""
    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


def _resolve_native_ckpt_format(requested: str | None, experiment: dict) -> str:
    """Pick the format the native converter writes, or exit with the reason.

    The experiment's ``ckpt_format`` must be one the converter writes
    (``CKPT_FORMATS``). A run in another format (``torch_dcp``, ``fsdp_dtensor``)
    does not load the converted checkpoint; with ``auto_detect_ckpt_format`` a
    ``torch_dcp`` run even starts from random weights without an error. Unset
    (``requested is None``), the converter writes the experiment's format
    (``torch_dist`` by default; ``torch`` for experiments that ask for it, as
    before). Then two layouts are refused, because Megatron cannot load them:

    * ``torch`` at any TP / PP / EP other than 1: a legacy checkpoint is not
      resharded on load.
    * ``torch_dist`` for an experiment with ``ckpt_format: torch``: Megatron
      builds the state dict it loads into from the experiment's ``ckpt_format``.
    """
    from primus.backends.megatron.checkpoint import native_convert_common as common

    run_format = experiment.get("ckpt_format", "torch_dist")
    if run_format not in common.CKPT_FORMATS:
        log_error(
            f"ckpt_format: {run_format} is not a format the native converter writes {common.CKPT_FORMATS}. "
            "Set ckpt_format: torch_dist (Megatron reshards it on load) and fine-tune without FSDP "
            "(use_torch_fsdp2, use_megatron_fsdp): an FSDP run cannot load a converted checkpoint."
        )
        sys.exit(1)
    ckpt_format = run_format if requested is None else requested
    if ckpt_format not in common.CKPT_FORMATS:
        log_error(f"native_ckpt_format: {ckpt_format!r} is not supported; use one of {common.CKPT_FORMATS}")
        sys.exit(1)

    layout = {k: int(experiment.get(k, 1)) for k in ("tp", "pp", "ep")}
    if ckpt_format == "torch" and any(v != 1 for v in layout.values()):
        log_error(
            "A legacy torch checkpoint (TP1/PP1/EP1) cannot be loaded at "
            f"TP={layout['tp']} PP={layout['pp']} EP={layout['ep']}. Set ckpt_format: torch_dist "
            "(and native_ckpt_format: torch_dist if set): Megatron reshards a torch_dist checkpoint on load."
        )
        sys.exit(1)
    if ckpt_format == "torch_dist" and run_format == "torch":
        log_error(
            "native_ckpt_format: torch_dist needs the experiment to load torch_dist checkpoints; "
            "set ckpt_format: torch_dist (or native_ckpt_format: torch at TP1/PP1/EP1)."
        )
        sys.exit(1)
    return ckpt_format


def _config_reader(config_file: str | None):
    """``value(name)``: ``name`` from the experiment's ``sft_trainer`` / ``post_trainer``
    module, its params first (``None`` when unset, or when there is no config file).
    An explicit ``0`` / ``False`` is a value. The config is loaded on the first read,
    so a family without options or a ``prepare`` step does not load it again."""
    if not config_file:
        return lambda name: None
    loaded = []

    def value(name: str):
        if not loaded:
            cfg = load_primus_config(Path(config_file), None)
            loaded.append(
                next(
                    (
                        m
                        for m in (get_module_config(cfg, n) for n in ("sft_trainer", "post_trainer"))
                        if m is not None
                    ),
                    None,
                )
            )
        module = loaded[0]
        params = getattr(module, "params", None)
        found = _first_config_value(params, name)
        return found if found is not None else _first_config_value(module, name)

    return value


def _prepare_native_family(
    hf_path: str, family: str, native_opts: dict, experiment: dict, config_file: str | None, argv
):
    """Run the registered family's option reading and ``prepare`` step.

    Sets ``native_opts["convert_kwargs"]`` (the family's options, then the plan's
    keywords) and returns ``(experiment, plan)``: the experiment settings to check
    the format against (the plan may replace them) and the family's
    ``ConvertPlan``. A ``NativeConvertRefused`` from the family exits with its
    reason.
    """
    registry = _registry()
    fam = registry.get_family(family)
    config_value = _config_reader(config_file)
    options = registry.read_options(fam, config_value)
    plan = None
    prepare = fam.hook("prepare")
    if prepare is not None:
        ctx = registry.ConvertContext(
            hf_path=hf_path,
            hf_config=registry.read_config(hf_path) or {},
            family=fam,
            experiment=dict(experiment),
            options=dict(options),
            argv=list(argv or []),
            config_value=config_value,
        )
        try:
            plan = prepare(ctx)
        except registry.NativeConvertRefused as e:
            log_error(str(e))
            sys.exit(1)
    plan = plan or registry.ConvertPlan()
    native_opts["convert_kwargs"] = {**options, **plan.convert_kwargs}
    return (plan.experiment if plan.experiment is not None else experiment), plan


def _native_main(
    hf_path: str,
    native_opts: dict,
    experiment: dict | None = None,
    *,
    config_file: str | None = None,
    argv: list | None = None,
):
    """Bridge-free native conversion branch of main().

    Mirrors the Bridge path's skip-if-exists / rank-0-converts-others-wait
    semantics, but produces the checkpoint via the Primus-managed in-process
    converter and emits ``pretrained_checkpoint`` + ``finetune`` back to the
    runner.

    ``experiment`` holds the fine-tune's parallel layout and ``ckpt_format``
    (``read_experiment_checkpoint_settings``). The converter writes
    ``torch_dist`` for a ``torch_dist`` experiment, which Megatron reshards on
    load to any TP / PP / EP; see ``_resolve_native_ckpt_format``. The format is
    part of the default output path and is checked against an existing
    checkpoint before it is reused, so a checkpoint of one format is never
    handed to a run that needs the other.

    The family's registration (``native_registry``) adds its options, and its
    ``prepare`` step (``config_file`` and ``argv``, the hook's command line, are
    passed to it) may add converter keywords, a suffix of the default output
    path, a check of an existing checkpoint before it is reused, and a step
    before the conversion. Its ``wait`` replaces how other nodes wait.
    """
    from primus.backends.megatron.checkpoint import native_convert_common as common

    family = _detect_native_family(hf_path)
    if family is None:
        log_error(f"native_ckpt_convert: true but {_registry().unknown_model_text(hf_path)}")
        sys.exit(1)
    native_opts["family"] = family
    experiment, plan = _prepare_native_family(
        hf_path, family, native_opts, dict(experiment or {}), config_file, argv
    )

    ckpt_format = _resolve_native_ckpt_format(native_opts.get("ckpt_format"), experiment)
    native_opts["ckpt_format"] = ckpt_format

    # Output path. torch_dist: <out>/release/ (metadata.json, .metadata, *.distcp);
    # torch: <out>/iter_0000001/mp_rank_00/model_optim_rng.pt.
    if native_opts.get("out"):
        megatron_path = Path(native_opts["out"])
    else:
        data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
        megatron_path = data_path / "megatron_checkpoints" / f"{Path(hf_path).name}_native_{ckpt_format}"
        if plan.path_suffix:
            megatron_path = megatron_path.with_name(megatron_path.name + plan.path_suffix)

    log_info(f"HF Model: {hf_path}  (native family: {family}, format: {ckpt_format})")
    log_info(f"Megatron Path: {megatron_path}")

    if (megatron_path / common.TRACKER_FILENAME).exists():
        existing = common.saved_ckpt_format(str(megatron_path))
        if existing != ckpt_format:
            log_error(
                f"{megatron_path} holds a {existing or 'missing or incomplete'} checkpoint, not "
                f"{ckpt_format}. Remove it, or set native_ckpt_out to another directory."
            )
            sys.exit(1)
        if plan.check_existing is not None:
            reason = plan.check_existing(megatron_path)
            if reason:
                log_error(reason)
                sys.exit(1)
        log_info(f"Native {ckpt_format} checkpoint already exists at {megatron_path}, skipping conversion")
        _emit_native_checkpoint_args(megatron_path)
        return

    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        log_info("Converting HF checkpoint to Megatron format (native, bridge-free)...")
        megatron_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file.touch()
        try:
            if plan.before_convert is not None:
                plan.before_convert(megatron_path)
            native_convert_checkpoint(hf_path, str(megatron_path), native_opts)
            done_file.touch()
            log_success(f"Checkpoint prepared at {megatron_path}")
        finally:
            lock_file.unlink(missing_ok=True)
    else:
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete native conversion...")
        wait = _registry().get_family(family).hook("wait")
        if wait is not None:
            wait(megatron_path, done_file, lock_file, node_rank)
        else:
            wait_for_conversion(done_file, lock_file)
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    _emit_native_checkpoint_args(megatron_path)


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Megatron format")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args, unknown_args = parser.parse_known_args()

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
    # Native (bridge-free) conversion is the DEFAULT for the registered families
    # (native_registry). ``native_ckpt_convert: false`` forces the legacy
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
                f"native_ckpt_convert=true but {_registry().unknown_model_text(hf_path)}. "
                "Set native_ckpt_convert: false to use the Megatron-Bridge path."
            )
            sys.exit(1)
        _native_main(
            hf_path,
            native_opts,
            read_experiment_checkpoint_settings(config_file),
            config_file=config_file,
            argv=unknown_args,
        )
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
