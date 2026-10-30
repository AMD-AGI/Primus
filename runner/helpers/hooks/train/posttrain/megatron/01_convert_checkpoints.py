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
import functools
import importlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Add primus to path
PRIMUS_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PRIMUS_ROOT))

from primus.core.config.primus_config import get_module_config, load_primus_config
from primus.core.utils.hf_config import HFConfigError, read_hf_config


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


def _trainer_module(config_file: str):
    """The experiment's ``sft_trainer`` (else ``post_trainer``) module config, or None.

    The one place the hook loads and walks the experiment config: ``get_checkpoint_config``,
    ``read_native_opts`` and ``_preset_family`` all read the same module. The parsed config
    is cached per path, so ``main()`` and ``bridge_deps_needed()`` parse the YAML once.
    """
    cfg = _load_experiment_config(str(config_file))
    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is not None:
            return module
    return None


@functools.lru_cache(maxsize=None)
def _load_experiment_config(config_file: str):
    return load_primus_config(Path(config_file), None)


def _module_value(module, *names: str):
    """First non-empty value of ``names`` on the module's params, else on the module."""
    if module is None:
        return None
    value = _first_config_value(getattr(module, "params", None), *names)
    # Fall back only when params has no value: an explicit false (native_ckpt_convert: false)
    # is a value and must be kept.
    return value if value is not None else _first_config_value(module, *names)


def get_checkpoint_config(config_file: str) -> tuple[str | None, str | None]:
    """
    Extract HF source path and any already-configured checkpoint path.

    Returns:
        tuple: (hf_path, checkpoint_path)
            - hf_path: HuggingFace model path to convert
            - checkpoint_path: Existing Megatron checkpoint path (if configured)
    """
    module = _trainer_module(config_file)
    # Old Megatron-Bridge SFT configs used hf_path, while native Megatron
    # configs typically reuse tokenizer_model as the HF source identifier.
    hf_path = _module_value(module, "hf_path", "tokenizer_model")
    # If the user already configured a Megatron-format checkpoint via either
    # pretrained_checkpoint or load, the conversion hook should be skipped.
    checkpoint_path = _module_value(module, "pretrained_checkpoint", "load")
    return hf_path, checkpoint_path


_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


def _tri_state(value, name: str):
    """None (unset, null, ``""`` or ``"null"``/``"none"``), True or False. Strings from an
    env substitution (``${VAR:false}``) are parsed, so ``"false"`` and ``"0"`` are False;
    any other value stops the hook with the reason rather than being guessed."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("", "null", "none", "~"):
            return None
        if v in _TRUE_STRINGS:
            return True
        if v in _FALSE_STRINGS:
            return False
    log_error(f"{name} must be true, false or unset, got {value!r}")
    sys.exit(1)


def read_native_opts(config_file: str) -> dict:
    """Read native (bridge-free) conversion options from the config.

    Returns a dict with ``enabled`` (tri-state: ``None`` when the config does not
    set ``native_ckpt_convert`` at all, else ``True`` / ``False``), plus optional
    ``dtype`` / ``out`` overrides. ``None`` lets the caller pick the default
    (native for supported families); ``True`` forces native; ``False`` forces the
    legacy Megatron-Bridge path.
    """
    module = _trainer_module(config_file)
    return {
        "enabled": _tri_state(_module_value(module, "native_ckpt_convert"), "native_ckpt_convert"),
        "dtype": _module_value(module, "native_ckpt_dtype"),
        "out": _module_value(module, "native_ckpt_out"),
    }


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
# DEFAULT conversion path for supported families (Qwen3, DeepSeek, Llama, Granite): this
# hook builds the mcore model in-process and copies HF safetensors weights onto
# it (Primus-managed converters), producing a legacy ``torch`` checkpoint WITHOUT
# ever importing ``megatron.bridge``. The convert-time monkeypatches (ROCm
# fused-kernels no-op) are applied in THIS process via ``run_patches(phase=
# "convert")`` so they take effect for the single-process conversion; the
# Megatron-LM submodule stays pristine. Set ``native_ckpt_convert: false`` to
# force the legacy Megatron-Bridge path (or ``true`` to force native); families in
# ``_NO_BRIDGE_FAMILIES`` (Granite) have no Bridge path and refuse ``false``.

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
    "granite": "granite",
    "graniteforcausallm": "granite",
}

# Families the pinned Megatron-Bridge cannot convert (it has no bridge for their HF
# architecture, so AutoBridge.import_ckpt refuses them). For these the native converter is
# the only working path: a config the native converter refuses is an error, and
# ``native_ckpt_convert: false`` (which forces the Bridge path) is refused early with a
# clear message instead of failing later inside AutoBridge.
_NO_BRIDGE_FAMILIES = frozenset({"granite"})


def _load_hf_config(hf_path: str) -> dict | None:
    """Return ``hf_path``/config.json as a dict, or None if there is no such file (a Hub
    repo id, or a directory without config.json). Raise ``HFConfigError`` if the file exists
    but is unreadable, not JSON, or not a JSON object. Only local files are read; the
    reader is the one the native converters use (``primus.core.utils.hf_config``)."""
    try:
        return read_hf_config(hf_path)
    except FileNotFoundError:
        return None


def _preset_family(config_file: str) -> str | None:
    """The native family of the experiment's model preset (its ``model_type``, e.g.
    ``granite`` from ``granite_4_1_3b.yaml``), or None. Read from the experiment config,
    so it does not depend on what ``hf_path`` points at."""
    model_type = _module_value(_trainer_module(config_file), "model_type")
    return _NATIVE_FAMILIES.get(str(model_type).lower()) if model_type else None


def _family_of(cfg: dict) -> str | None:
    """The native family named by an HF config (architectures first, then model_type)."""
    for arch in cfg.get("architectures", []) or []:
        fam = _NATIVE_FAMILIES.get(str(arch).lower())
        if fam:
            return fam
    return _NATIVE_FAMILIES.get(str(cfg.get("model_type", "")).lower())


# Native family -> converter module in primus.backends.megatron.checkpoint.
_NATIVE_CONVERTERS = {
    "qwen3": "native_loader_qwen3",
    "llama": "native_loader_llama",
    "granite": "native_loader_granite",
    "deepseek": "native_loader_deepseek",
}


# Display names for messages.
_FAMILY_NAMES = {"qwen3": "Qwen3", "llama": "Llama", "granite": "Granite", "deepseek": "DeepSeek"}

# Every family _family_of can return has a converter (so the lookup above cannot fail).
assert set(_NATIVE_FAMILIES.values()) <= set(_NATIVE_CONVERTERS), "native family without a converter"

_UNREAD = object()


def _native_decision(hf_path: str, cfg=_UNREAD) -> tuple[str | None, str, str | None]:
    """Return ``(family, reason, detected)`` for the HF checkpoint at ``hf_path``.

    ``cfg`` is the already-loaded ``_load_hf_config(hf_path)`` (main() passes it, and passes
    it on to the dtype lookup, so config.json is read once); when it is not given it is read here.
    ``detected`` is the family named by config.json ('qwen3' | 'deepseek' | 'llama' |
    'granite'), or None. ``family`` is ``detected`` when its native converter accepts the
    config, else None; ``reason`` says why (shown in the log and in errors). A missing or
    malformed config.json gives ``(None, reason, None)``; main() refuses a malformed one
    before calling this.
    A converter module that defines ``check_supported(cfg)`` decides (today Granite's): a
    config it refuses (RoPE scaling, biases, wrapper models, ...) gets ``family=None`` and
    the converter's message as the reason. An ImportError of the converter is not caught:
    the hook process imports primus, so it is a real bug, and hiding it would silently
    switch converters.
    """
    if cfg is _UNREAD:
        try:
            cfg = _load_hf_config(hf_path)
        except HFConfigError as exc:
            return None, str(exc), None
    if cfg is None:
        return None, f"no config.json in {hf_path} (not a local HF checkpoint directory)", None

    fam = _family_of(cfg)
    if fam is None:
        return (
            None,
            (
                f"architectures={cfg.get('architectures')!r} model_type={cfg.get('model_type')!r} "
                "is not a native family (supported: Qwen3, DeepSeek-V2/V3, Llama, Granite dense)"
            ),
            None,
        )
    # A converter module may define check_supported(cfg); it decides which configs of its
    # family it converts (Granite refuses RoPE scaling, biases, wrapper models, ...).
    conv = importlib.import_module(f"primus.backends.megatron.checkpoint.{_NATIVE_CONVERTERS[fam]}")
    check_supported = getattr(conv, "check_supported", None)
    if check_supported is not None:
        try:
            check_supported(cfg)
        except (ValueError, NotImplementedError) as exc:
            name = _FAMILY_NAMES.get(fam, fam)
            return None, f"{name} checkpoint not supported by the native {name} converter: {exc}", fam
    return fam, f"native {fam} converter", fam


def _route(hf_cfg: dict | None, preset_family: str | None, enabled):
    """The one conversion-path decision, shared by main() and bridge_deps_needed().

    Inputs are the local ``hf_path``/config.json (``hf_cfg``; None for a Hub repo id or a
    directory without config.json), the native family of the model preset's ``model_type``
    and ``native_ckpt_convert``. Nothing else is read (no HF cache, no network), so the same
    experiment config and checkpoint give the same route, and the same dependencies, on
    every node and in every run.

    Returns ``(route, family)``. ``family`` is the family whose rules decided: a family
    without a Bridge converter (Granite) when the preset or config.json names one, else the
    native family named by config.json (or None). ``route`` is one of
      * ``"refuse_forced_off"``: ``native_ckpt_convert: false`` for a family without a
        Bridge converter, named by the preset or by config.json;
      * ``"download"``: the preset is such a family and hf_path is not a local checkpoint
        (a Hub repo id, e.g. the preset's tokenizer_model): main() downloads it and routes
        again with the downloaded config.json;
      * ``"refuse_mismatch"``: the preset and config.json disagree and one of them is a family
        without a Bridge converter. Both directions are refused the same way: any
        checkpoint that is not ``family`` (Llama, Mistral, granitemoehybrid, ...) for the
        Granite preset, and a Granite checkpoint for a preset whose model_type is not
        granite (that model would be built without Granite's multipliers);
      * ``"native"``: the native converter (main() still runs its support check, which
        refuses e.g. a Granite config with RoPE scaling with the converter's reason);
      * ``"bridge"``: the Megatron-Bridge path (its dependencies are needed).
    A preset of a family without a Bridge converter never routes to the Bridge. Otherwise a
    local config.json naming a native family routes native. Pure: reads no files and
    imports nothing.
    """
    named = _family_of(hf_cfg) if hf_cfg is not None else None
    no_bridge = next((f for f in (preset_family, named) if f in _NO_BRIDGE_FAMILIES), None)
    if no_bridge is not None:
        if enabled is False:
            return "refuse_forced_off", no_bridge
        if hf_cfg is None and preset_family == no_bridge:
            return "download", no_bridge
        if named != preset_family:
            return "refuse_mismatch", no_bridge
        return "native", no_bridge
    if enabled is True or (enabled is None and named is not None):
        return "native", named
    return "bridge", named if enabled is False else None


def _mismatch_message(hf_path: str, hf_cfg: dict, preset_family: str | None, family: str) -> str:
    """The refusal of a preset / checkpoint mismatch; ``family`` is the side without a
    Bridge converter (``_route``), and the advice names only that side's fix."""
    name = _FAMILY_NAMES.get(family, family)
    ckpt = f"architectures={hf_cfg.get('architectures')!r} model_type={hf_cfg.get('model_type')!r}"
    if preset_family == family:
        return (
            f"the model preset's model_type is {family}, but config.json at {hf_path} is not a "
            f"{name} checkpoint ({ckpt}). The {name} preset converts only {name} checkpoints: "
            f"point hf_path at one, or use the model preset of this checkpoint."
        )
    preset = preset_family or "not a native family"
    return (
        f"config.json at {hf_path} is a {name} checkpoint ({ckpt}), but the model preset's "
        f"model_type is {preset}, so the model would be built without {name}'s settings. "
        f"Use a {name} model preset (model_type: {family}"
        + (", e.g. granite_4_1_3b.yaml)." if family == "granite" else ").")
    )


def bridge_deps_needed(config_file: str, hf_path: str | None) -> bool:
    """True when main() takes the Megatron-Bridge path, so ``00_install_requirements.sh``
    must install the Bridge dependencies (transformers==4.57.6, modelopt, ...).

    It reads the same inputs as main() (the experiment config and the local config.json)
    and calls the same ``_route``, so the two cannot disagree. It does not import the
    native converters (torch, Megatron, the patch registry).
    """
    if not hf_path:
        return False
    try:
        hf_cfg = _load_hf_config(hf_path)
    except HFConfigError:
        return False  # main() stops on a malformed config.json
    route, _ = _route(hf_cfg, _preset_family(config_file), read_native_opts(config_file)["enabled"])
    return route == "bridge"


_DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"]


def _download_hf_checkpoint(repo_id: str, family: str) -> str:
    """Download the HF checkpoint ``repo_id`` (config, weights, tokenizer) and return its
    local snapshot directory. Called on node rank 0 only (the other nodes wait for the
    converted checkpoint). Stops with a clear error when the download fails (no network
    and not cached, unknown repo, ...)."""
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id, allow_patterns=_DOWNLOAD_PATTERNS)
    except Exception as exc:  # noqa: BLE001 - any failure ends the hook with its reason
        log_error(
            f"hf_path {repo_id} is not a local HF checkpoint directory and downloading it "
            f"failed: {type(exc).__name__}: {exc}. The pinned Megatron-Bridge has no {family} "
            f"bridge, so the native converter needs a local checkpoint: download it "
            f"(huggingface-cli download {repo_id} --local-dir <dir>) and set hf_path to that "
            f"directory."
        )
        sys.exit(1)
    log_info(f"Downloaded {repo_id} to {local}")
    return local


def _native_dtype_from_config(hf_path: str, override: str | None, cfg=_UNREAD) -> str:
    """The conversion dtype: ``override``, else config.json's torch_dtype. ``cfg`` is the
    already-loaded config.json (main() passes it down, so the file is read once); a direct
    caller that does not pass it gets it read here."""
    if override:
        return override
    if cfg is _UNREAD:
        cfg = _load_hf_config(hf_path)  # hf_path is a local directory (native_convert_checkpoint)
    if cfg is not None:
        td = str(cfg.get("torch_dtype", "")).lower()
        if "bfloat16" in td:
            return "bf16"
        if "float16" in td:
            return "fp16"
        if "float32" in td:
            return "fp32"
    return "bf16"


def native_convert_checkpoint(
    hf_path: str, megatron_path: str, opts: dict, decision: tuple | None = None, hf_cfg=_UNREAD
):
    """Convert a HF checkpoint (a local directory) to a legacy Megatron torch ckpt, bridge-free.

    ``hf_path`` must be a local HF checkpoint directory (config.json + safetensors); a
    Hub repo id is refused with that reason by the support check.

    Applies the ``phase="convert"`` patches in this process, then runs the
    Primus-managed native converter for the detected family, all single-process.
    ``decision`` is ``_native_decision(hf_path)`` when the caller (``_native_main``) has
    it; a direct caller passes none and it is computed here, so the support check always
    runs.
    """
    family, reason, _ = decision if decision is not None else _native_decision(hf_path, hf_cfg)
    if family is None:
        raise ValueError(
            f"native_ckpt_convert requested but no native converter for HF model at {hf_path}: "
            f"{reason} (supported: Qwen3 dense/MoE, DeepSeek-V2/V3, Llama 2/3/3.1, Granite dense)"
        )
    conv_module = _NATIVE_CONVERTERS[family]  # every _NATIVE_FAMILIES value has one (checked at import)

    # Ensure the Megatron-LM source tree is importable in this standalone hook
    # process (run_pretrain.sh's PYTHONPATH does not include it), and register +
    # apply the convert-phase patches BEFORE any Megatron model build.
    from primus.backends.megatron.checkpoint import native_convert_common as common

    common.ensure_megatron_on_path()
    applied = common.apply_convert_patches(module_name="checkpoint_convert")
    log_info(f"Applied {applied} convert-phase patch(es)")

    dtype = _native_dtype_from_config(hf_path, opts.get("dtype"), hf_cfg)
    tp = int(opts.get("tensor_parallel_size", 1))
    pp = int(opts.get("pipeline_parallel_size", 1))
    ep = int(opts.get("expert_parallel_size", 1))

    log_info(f"Native ({family}) conversion: dtype={dtype} TP={tp} PP={pp} EP={ep}")
    log_info(f"  Source: {hf_path}")
    log_info(f"  Target: {megatron_path}")

    conv = importlib.import_module(f"primus.backends.megatron.checkpoint.{conv_module}")

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


# How long the other nodes wait for node rank 0's conversion (seconds). Downloading the
# checkpoint first (a Hub repo id) takes longer, so that route waits longer by default.
# PRIMUS_CKPT_CONVERT_TIMEOUT overrides both.
_WAIT_TIMEOUT_S = 600
_WAIT_TIMEOUT_DOWNLOAD_S = 7200


def _wait_timeout(downloading: bool = False) -> int:
    env = os.environ.get("PRIMUS_CKPT_CONVERT_TIMEOUT")
    if env:
        return int(env)
    return _WAIT_TIMEOUT_DOWNLOAD_S if downloading else _WAIT_TIMEOUT_S


def wait_for_conversion(
    done_file: Path,
    lock_file: Path,
    timeout: int | None = None,
    failed_file: Path | None = None,
    checkpoint_dir: Path | None = None,
):
    """Wait for rank 0 to complete checkpoint conversion (``timeout`` seconds of wall time;
    default ``_wait_timeout()``).

    Markers left by an earlier run are not trusted. Only the shared file system's own state
    is compared, never clocks of different nodes:
      * ``done_file`` counts when it was written during this wait, or (with
        ``checkpoint_dir``) when it is not older than the checkpoint's
        ``latest_checkpointed_iteration.txt``. A ``.done`` whose checkpoint was deleted, or
        that is older than a checkpoint rank 0 is writing again, does not count.
      * ``failed_file`` (rank 0's reason) stops the wait only when it was written during
        this wait. A marker from an earlier failed run is ignored; rank 0 deletes it when it
        retries.
    """
    if timeout is None:
        timeout = _wait_timeout()
    done_at_start = _marker_sig(done_file)
    failed_at_start = _marker_sig(failed_file) if failed_file is not None else None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _done(done_file, done_at_start, checkpoint_dir):
            return
        if failed_file is not None:
            sig = _marker_sig(failed_file)
            if sig is not None and sig != failed_at_start:
                raise RuntimeError(
                    f"node rank 0 could not prepare the checkpoint: {_read_marker(failed_file)}"
                )
        time.sleep(5 if lock_file.exists() else 2)
    if _done(done_file, done_at_start, checkpoint_dir):
        return
    raise TimeoutError(
        f"Timeout waiting for checkpoint conversion ({timeout} s; set "
        f"PRIMUS_CKPT_CONVERT_TIMEOUT to wait longer)"
    )


def _marker_sig(path: Path):
    """Identity of a marker file as the shared file system reports it, or None."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _read_marker(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return "see node rank 0's log"


def _done(done_file: Path, done_at_start, checkpoint_dir: Path | None) -> bool:
    sig = _marker_sig(done_file)
    if sig is None:
        return False
    if sig != done_at_start:
        return True  # written during this wait
    if checkpoint_dir is None:
        return False
    latest = _marker_sig(Path(checkpoint_dir) / "latest_checkpointed_iteration.txt")
    return latest is not None and sig[1] >= latest[1]


def _start_attempt(megatron_path: Path):
    """Rank 0, before it (re)builds ``megatron_path``: drop the markers of an earlier run
    so no waiting node trusts them."""
    megatron_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".done", ".failed"):
        Path(f"{megatron_path}{suffix}").unlink(missing_ok=True)


def _mark_failed(megatron_path: Path, exc: BaseException):
    """Rank 0: tell the waiting nodes at once (rather than after their timeout)."""
    reason = "see node rank 0's log" if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
    try:
        megatron_path.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{megatron_path}.failed").write_text(reason + "\n")
    except OSError as err:
        log_error(f"could not write {megatron_path}.failed: {err}")


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


def _data_ckpt_path(name: str) -> Path:
    """``<DATA_PATH>/megatron_checkpoints/<name>`` (DATA_PATH defaults to ``<primus>/data``)."""
    data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
    return data_path / "megatron_checkpoints" / name


def _native_out_path(native_opts: dict, name: str) -> Path:
    """Where the native converter writes: ``native_ckpt_out``, else ``_data_ckpt_path(name)``."""
    if native_opts.get("out"):
        return Path(native_opts["out"])
    return _data_ckpt_path(name)


def _converted(megatron_path: Path) -> bool:
    return (megatron_path / "latest_checkpointed_iteration.txt").exists()


def _emit_checkpoint(megatron_path: Path):
    """The runner protocol: the converted checkpoint is loaded as a pretrained checkpoint
    and training runs as a fine-tune. Every route prints it through this function."""
    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


def _native_main(
    hf_path: str,
    native_opts: dict,
    decision: tuple | None = None,
    hf_cfg=_UNREAD,
    megatron_name: str | None = None,
):
    """Bridge-free native conversion branch of main().

    Mirrors the Bridge path's skip-if-exists / rank-0-converts-others-wait
    semantics, but produces a legacy ``torch`` checkpoint via the Primus-managed
    in-process converter and emits ``pretrained_checkpoint`` + ``finetune`` back
    to the runner. ``megatron_name`` names the output directory when ``hf_path`` is a
    downloaded snapshot (whose directory name is a commit hash).
    """
    # main() passes its decision; a direct caller gets a fresh one, so the converter's
    # support check always runs (e.g. it refuses Granite with RoPE scaling).
    if decision is None:
        decision = _native_decision(hf_path, hf_cfg)
    family, reason = decision[0], decision[1]
    if family is None:
        log_error(f"native conversion requested but {reason} (HF model at {hf_path})")
        sys.exit(1)
    # Output path (legacy torch layout: <out>/iter_0000001/mp_rank_00/...).
    megatron_path = _native_out_path(native_opts, megatron_name or f"{Path(hf_path).name}_native")

    log_info(f"HF Model: {hf_path}  (native family: {family})")
    log_info(f"Megatron Path: {megatron_path}")

    if _converted(megatron_path):
        log_info(f"Native Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        _emit_checkpoint(megatron_path)
        return

    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        log_info("Converting HF checkpoint to Megatron format (native, bridge-free)...")
        _start_attempt(megatron_path)
        lock_file.touch()
        try:
            native_convert_checkpoint(hf_path, str(megatron_path), native_opts, decision, hf_cfg)
            done_file.touch()
            log_success(f"Checkpoint prepared at {megatron_path}")
        except BaseException as exc:
            _mark_failed(megatron_path, exc)
            raise
        finally:
            lock_file.unlink(missing_ok=True)
    else:
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete native conversion...")
        wait_for_conversion(
            done_file, lock_file, failed_file=Path(f"{megatron_path}.failed"), checkpoint_dir=megatron_path
        )
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    _emit_checkpoint(megatron_path)


def _run_native(hf_path: str, hf_cfg, preset_family, native_opts: dict, megatron_name: str | None = None):
    """main()'s ``native`` route: the converter's support check, then _native_main."""
    decision = _native_decision(hf_path, hf_cfg)
    native_family, reason, detected = decision
    if native_family is None:
        no_bridge = next((f for f in (detected, preset_family) if f in _NO_BRIDGE_FAMILIES), None)
        if no_bridge is not None:
            hint = (
                f"The pinned Megatron-Bridge has no {no_bridge} bridge either, so this "
                "checkpoint cannot be converted; change the checkpoint config or extend "
                "the native converter."
            )
        else:
            hint = "Set native_ckpt_convert: false to use the Megatron-Bridge path."
        log_error(f"the native converter cannot be used: {reason} (HF model at {hf_path}). {hint}")
        sys.exit(1)
    _native_main(hf_path, native_opts, decision, hf_cfg=hf_cfg, megatron_name=megatron_name)


def _download_and_convert(repo_id: str, preset_family, native_opts: dict, family: str, megatron_name: str):
    """Node rank 0's ``download`` route: download ``repo_id``, route again with its
    config.json, and convert natively. Any other outcome stops with its reason (never the
    Bridge path: the preset's family has no Bridge converter)."""
    hf_path = _download_hf_checkpoint(repo_id, family)
    try:
        hf_cfg = _load_hf_config(hf_path)
    except HFConfigError as exc:
        log_error(f"{exc} (HF model at {hf_path})")
        sys.exit(1)
    if hf_cfg is None:
        log_error(
            f"{repo_id} was downloaded to {hf_path} but has no config.json, so it is not a "
            f"HF checkpoint the native {family} converter can read. Set hf_path to a "
            f"{family} checkpoint (a Hub repo id or a local directory with config.json)."
        )
        sys.exit(1)
    route, family = _route(hf_cfg, preset_family, native_opts["enabled"])
    if route == "refuse_mismatch":
        log_error(_mismatch_message(hf_path, hf_cfg, preset_family, family))
        sys.exit(1)
    if route != "native":  # refuse_forced_off is decided before any download
        log_error(f"unexpected route {route!r} for the downloaded checkpoint {hf_path}")
        sys.exit(1)
    _run_native(hf_path, hf_cfg, preset_family, native_opts, megatron_name)


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
    # (Qwen3, DeepSeek, Llama, Granite). ``native_ckpt_convert: false`` forces the legacy
    # Megatron-Bridge path; ``native_ckpt_convert: true`` forces native.
    # The path is chosen by _route(), the same function bridge_deps_needed() (used by
    # 00_install_requirements.sh) calls, so the install step and the conversion agree.
    # main() reads config.json once and passes it on (_native_main ->
    # native_convert_checkpoint -> _native_dtype_from_config). Only a local HF directory
    # is converted natively. A Hub repo id (the tokenizer_model fallback) uses the
    # Megatron-Bridge path, except for families without a Bridge converter (Granite): those
    # are downloaded and converted natively. The family comes from the model preset's
    # model_type and from the local config.json only (never from the HF cache), so every
    # node decides the same.
    native_opts = read_native_opts(config_file)
    preset_family = _preset_family(config_file)
    try:
        hf_cfg = _load_hf_config(hf_path)
    except HFConfigError as exc:
        # Neither converter can use it: stop here with the reason, not in a traceback.
        log_error(f"{exc} (HF model at {hf_path})")
        sys.exit(1)
    route, family = _route(hf_cfg, preset_family, native_opts["enabled"])

    if route == "download":
        # The native converter reads a local directory. Only node rank 0 downloads (the HF
        # cache is reused; HF_HUB_OFFLINE=1 makes this use the cache only) and converts;
        # the other nodes wait for its result and never touch the network. A checkpoint
        # already converted by an earlier run is used without any download.
        megatron_name = f"{Path(hf_path).name}_native"
        megatron_path = _native_out_path(native_opts, megatron_name)
        if _converted(megatron_path):
            log_info(f"Native Megatron checkpoint already exists at {megatron_path}, skipping download")
            _emit_checkpoint(megatron_path)
            return
        node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
        done_file = Path(f"{megatron_path}.done")
        Path(f"{megatron_path}.failed")
        if node_rank != 0:
            log_info(f"[RANK {node_rank}] Waiting for rank 0 to download and convert {hf_path}...")
            wait_for_conversion(
                Path(f"{megatron_path}.done"),
                Path(f"{megatron_path}.converting.lock"),
                _wait_timeout(downloading=True),
                failed_file=Path(f"{megatron_path}.failed"),
                checkpoint_dir=megatron_path,
            )
            log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")
            _emit_checkpoint(megatron_path)
            return
        _start_attempt(megatron_path)
        try:
            _download_and_convert(hf_path, preset_family, native_opts, family, megatron_name)
        except BaseException as exc:
            _mark_failed(megatron_path, exc)
            raise
        return
    if route == "refuse_mismatch":
        log_error(_mismatch_message(hf_path, hf_cfg, preset_family, family))
        sys.exit(1)
    if route == "refuse_forced_off":
        # The Bridge path cannot convert this family: fail now with the reason,
        # not later with an opaque AutoBridge error.
        log_error(
            f"native_ckpt_convert: false forces the Megatron-Bridge path, but the pinned "
            f"Megatron-Bridge has no {family} bridge (HF model at {hf_path}). "
            f"Remove native_ckpt_convert or set it to true to use the native "
            f"{family} converter."
        )
        sys.exit(1)

    if route == "native":
        _run_native(hf_path, hf_cfg, preset_family, native_opts)
        return

    # route == "bridge"
    if native_opts["enabled"] is False:
        reason = "native_ckpt_convert: false"
    elif hf_cfg is None:
        reason = f"no config.json in {hf_path} (not a local HF checkpoint directory)"
    else:
        reason = (
            f"architectures={hf_cfg.get('architectures')!r} model_type={hf_cfg.get('model_type')!r} "
            "is not a native family"
        )

    log_info(f"Using Megatron-Bridge conversion path ({reason})")

    # Set paths
    megatron_path = _data_ckpt_path(Path(hf_path).name)

    log_info(f"HF Model: {hf_path}")
    log_info(f"Megatron Path: {megatron_path}")

    # Check if Megatron checkpoint already exists
    if megatron_path.exists():
        log_info(f"Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        _emit_checkpoint(megatron_path)
        return

    # Convert checkpoint (only on rank 0, others wait)
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        # Rank 0: perform the conversion
        log_info("Converting HF checkpoint to Megatron format using Megatron-Bridge...")
        _start_attempt(megatron_path)

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
        except BaseException as exc:
            _mark_failed(megatron_path, exc)
            raise
        finally:
            lock_file.unlink(missing_ok=True)
    else:
        # Other ranks: wait for rank 0 to complete
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete checkpoint conversion...")
        wait_for_conversion(
            done_file, lock_file, failed_file=Path(f"{megatron_path}.failed"), checkpoint_dir=megatron_path
        )
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    # Output the checkpoint path for the main training process
    _emit_checkpoint(megatron_path)


if __name__ == "__main__":
    main()
