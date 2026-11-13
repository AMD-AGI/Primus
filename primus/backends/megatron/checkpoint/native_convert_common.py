###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared infrastructure for Primus' native (bridge-free) HF -> Megatron-Core
checkpoint converters.

Design (identical for every family, mirrors the proven self-contained DeepSeek
converter):

    1. Read the HF ``config.json`` and derive Megatron args (per-family).
    2. ``set_global_variables(build_tokenizer=True)`` so ``padded_vocab_size`` is
       computed from the *tokenizer's real vocab*, EXACTLY as Primus native SFT
       computes it -> the converted embedding / output_layer heights line up for
       a clean ``load_state_dict`` at SFT time.
    3. Build the mcore ``GPTModel`` on CPU with the SAME builder Primus training
       uses (``get_model_provider`` -> ``model_provider(gpt_builder)``), so the
       saved parameter names are precisely the keys native SFT expects.
    4. Copy HF safetensors weights straight onto those parameters (a pure rename
       + a couple of fused layouts), validate the mapping is complete and exact
       (0 unexpected, 0 non-``_extra_state`` missing, 0 unconsumed HF tensors),
       and save a legacy ``torch`` checkpoint.

Bridge-free guarantee: this module (and its callers) import ONLY
``megatron.core`` / ``megatron.training``. ``megatron.bridge`` is never imported;
a hard guard asserts it never leaked into ``sys.modules``.
"""

import errno
import json
import os
import shutil
import sys
import time

import torch

# q / k / v -> linear_qkv: the shared row layout (re-exported for the converters)
from primus.backends.megatron.core.models.common.qkv_layout import (  # noqa: F401
    fuse_qkv,
)


# ---------------------------------------------------------------------------
# Megatron-LM source tree location
# ---------------------------------------------------------------------------
def megatron_lm_root() -> str:
    """Absolute path of the in-tree Megatron-LM submodule (``third_party/Megatron-LM``).

    Resolved from this file's location first (``primus/backends/megatron/
    checkpoint/`` -> repo root), then from ``$PRIMUS_PATH``. The converter needs
    the Megatron-LM *root* (not ``tools/checkpoint``) on ``sys.path`` so that
    ``import megatron`` and the top-level ``model_provider`` / ``gpt_builders``
    helper modules are importable in the standalone hook process.
    """
    from pathlib import Path

    # primus/backends/megatron/checkpoint/<this file> -> parents[4] == repo root
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "third_party" / "Megatron-LM"
    if candidate.exists():
        return str(candidate)

    primus_path = os.getenv("PRIMUS_PATH")
    if primus_path:
        candidate = Path(primus_path) / "third_party" / "Megatron-LM"
        if candidate.exists():
            return str(candidate)

    # Last resort: return the file-relative guess even if it does not exist so
    # the caller gets a clear ImportError pointing at the expected location.
    return str(repo_root / "third_party" / "Megatron-LM")


def ensure_megatron_on_path() -> str:
    """Prepend the Megatron-LM root to ``sys.path`` (idempotent). Returns the path."""
    root = megatron_lm_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def assert_bridge_free(stage: str) -> None:
    """Hard guard: fail loudly if ``megatron.bridge`` ever gets imported."""
    assert (
        "megatron.bridge" not in sys.modules
    ), f"megatron.bridge leaked into sys.modules ({stage}); native conversion must stay bridge-free"


def neutralize_nvte_attention_env() -> None:
    """Drop baked ``NVTE_*_ATTN`` env vars for the duration of this process.

    Building a TE mcore model runs ``_set_attention_backend()`` which asserts the
    three ``NVTE_{FLASH,FUSED,UNFUSED}_ATTN`` vars are unset-or-1; the ROCm image
    bakes ``NVTE_FLASH_ATTN=0`` and trips it. Conversion computes no attention, so
    the backend is irrelevant -- unset all three (this hook process is
    short-lived and exits after the conversion, so no restore is needed).
    """
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# Convert-phase patches
# ---------------------------------------------------------------------------
def apply_convert_patches(module_name: str | None = None, platform: str | None = None) -> int:
    """Register + run all ``phase="convert"`` Megatron patches in THIS process.

    Importing the ``checkpoint_convert_patches`` module triggers its
    ``@register_patch`` side effect; ``run_patches`` then applies the ones whose
    ``condition`` matches. Must be called BEFORE any Megatron model build so the
    monkeypatches are in effect. Returns the number of patches applied.
    """
    # Register the convert-phase patch(es). Import the single module rather than
    # the whole patches package to avoid pulling heavy TE/training imports into
    # the lightweight conversion process; run_patches only reads the registry.
    import primus.backends.megatron.patches.checkpoint_convert_patches  # noqa: F401
    from primus.core.patches import run_patches

    return run_patches(
        backend="megatron",
        phase="convert",
        module_name=module_name,
        platform=platform,
    )


# ---------------------------------------------------------------------------
# Direct safetensors reader (no ``transformers`` model class / trust_remote_code)
# ---------------------------------------------------------------------------
class SafetensorsStore:
    """Lazily fetch tensors by name from a HF checkpoint dir.

    Handles a single ``model.safetensors`` and a sharded checkpoint with a
    ``model.safetensors.index.json`` weight map; falls back to legacy torch
    ``pytorch_model.bin`` pickles only if no safetensors are present. File
    handles are cached so a shard is not re-opened per tensor.
    """

    def __init__(self, hf_dir: str):
        self.hf_dir = hf_dir
        self._handles: dict = {}
        self._name_to_file = None  # sharded weight map
        self._single = None  # single-file safetensors path
        self._bin_cache = None  # fully-loaded legacy .bin dict

        index = os.path.join(hf_dir, "model.safetensors.index.json")
        single = os.path.join(hf_dir, "model.safetensors")
        have_st = False
        try:
            from safetensors import safe_open  # noqa: F401

            have_st = True
        except ImportError:
            have_st = False

        if have_st and os.path.isfile(index):
            with open(index) as f:
                self._name_to_file = {
                    k: os.path.join(hf_dir, v) for k, v in json.load(f)["weight_map"].items()
                }
        elif have_st and os.path.isfile(single):
            self._single = single
        else:
            self._bin_cache = {}
            bin_index = os.path.join(hf_dir, "pytorch_model.bin.index.json")
            if os.path.isfile(bin_index):
                with open(bin_index) as f:
                    weight_map = json.load(f)["weight_map"]
                for fp in {os.path.join(hf_dir, v) for v in weight_map.values()}:
                    self._bin_cache.update(torch.load(fp, map_location="cpu", weights_only=False))
            else:
                self._bin_cache.update(
                    torch.load(
                        os.path.join(hf_dir, "pytorch_model.bin"),
                        map_location="cpu",
                        weights_only=False,
                    )
                )

    def _handle(self, filepath: str):
        h = self._handles.get(filepath)
        if h is None:
            from safetensors import safe_open

            h = safe_open(filepath, framework="pt", device="cpu")
            self._handles[filepath] = h
        return h

    def has(self, name: str) -> bool:
        if self._bin_cache is not None:
            return name in self._bin_cache
        if self._single is not None:
            return name in self._handle(self._single).keys()
        return name in self._name_to_file

    def get(self, name: str, dtype) -> torch.Tensor:
        if self._bin_cache is not None:
            t = self._bin_cache[name]
        elif self._single is not None:
            t = self._handle(self._single).get_tensor(name)
        else:
            t = self._handle(self._name_to_file[name]).get_tensor(name)
        return t.to(dtype)

    def all_keys(self) -> set:
        if self._bin_cache is not None:
            return set(self._bin_cache.keys())
        if self._single is not None:
            return set(self._handle(self._single).keys())
        return set(self._name_to_file.keys())


# ---------------------------------------------------------------------------
# Vocab padding (replicate Megatron's pad-with-last-row / trim behaviour)
# ---------------------------------------------------------------------------
def pad_vocab(t: torch.Tensor, padded_vocab_size: int) -> torch.Tensor:
    """Pad (or trim) the vocab dim (dim 0) to ``padded_vocab_size``.

    HF over-pads its embedding table (e.g. Qwen3 config vocab_size 151936) while
    Megatron sizes the embedding from the tokenizer's real vocab rounded up to a
    multiple of ``make_vocab_size_divisible_by`` (e.g. 151680). Trimming to the
    first ``padded_vocab_size`` rows keeps every real token and drops only HF pad
    rows; padding (rare) replicates the last row like Megatron does.
    """
    v = t.shape[0]
    if v == padded_vocab_size:
        return t
    if v > padded_vocab_size:
        return t[:padded_vocab_size].contiguous()
    pad = t[-1:].expand(padded_vocab_size - v, -1)
    return torch.cat([t, pad], dim=0).contiguous()


# ---------------------------------------------------------------------------
# Single-process Megatron init (real trivial gloo groups, TP=PP=EP=1)
# ---------------------------------------------------------------------------
def init_megatron_single_process(margs, master_port: int, log=print):
    """Initialise Megatron globals + a trivial single-rank gloo world.

    ``build_tokenizer=True`` so ``padded_vocab_size`` is derived from the HF
    tokenizer's real vocab, identical to how Primus native SFT sizes the model.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    neutralize_nvte_attention_env()

    from megatron.core import mpu
    from megatron.training.global_vars import get_args, set_global_variables

    set_global_variables(margs, build_tokenizer=True)
    margs = get_args()
    log(f"padded_vocab_size (from tokenizer) = {margs.padded_vocab_size}")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=margs.tensor_model_parallel_size,
        pipeline_model_parallel_size=margs.pipeline_model_parallel_size,
        expert_model_parallel_size=margs.expert_model_parallel_size,
    )
    return margs


def build_mcore_gpt_model(margs, model_type: str | None = None, *, family: str | None = None):
    """Build the mcore ``GPTModel`` via Primus' own model-provider resolver.

    Using ``get_model_provider(model_type)`` (the same shim native SFT uses) means
    the produced parameter names match exactly what SFT will build, and keeps us
    off the ``tools/checkpoint`` directory layout. ``model_type`` is the type
    training builds the model with: given explicitly, else the ``model_type`` the
    registered native ``family`` declares (``native_registry``), else ``"gpt"``.
    """
    from primus.core.utils.import_utils import get_model_provider

    if model_type is None:
        from primus.backends.megatron.checkpoint import native_registry

        model_type = native_registry.model_type_for(family)
    model_provider = get_model_provider(model_type=model_type)
    model = model_provider(pre_process=True, post_process=True)
    return model.to(margs.params_dtype)


# ---------------------------------------------------------------------------
# Load mapped state dict, validate, and save in the requested checkpoint format
# ---------------------------------------------------------------------------
def load_validate_save(model, sd, consumed, store, margs, save_dir, log):
    """Apply ``sd`` to ``model`` (strict=False), assert an exact mapping, save.

    The checkpoint is written in ``margs.ckpt_format`` (see ``CKPT_FORMATS``), a
    ``torch_dist`` one straight into ``release/`` (``set_aside_interrupted_release``
    makes way for it first).

    Raises ``RuntimeError`` if the mapping is not complete/exact so the caller
    (conversion hook) fails loudly instead of writing a silently-wrong ckpt, and
    before writing anything if ``save_dir`` has less free space than the checkpoint
    needs (``check_free_space``). A write that fails anyway is re-raised as a
    ``RuntimeError`` that names the directory, and its free space when the write
    ran out of it.
    """
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_non_extra = [k for k in missing if "_extra_state" not in k]
    log(f"load_state_dict: {len(sd)} tensors applied")
    log(f"  unexpected keys: {len(unexpected)}")
    log(f"  missing keys total: {len(missing)}  (non-_extra_state: {len(missing_non_extra)})")
    if unexpected:
        log(f"  first unexpected: {unexpected[:8]}")
    if missing_non_extra:
        log(f"  first missing(non-extra): {missing_non_extra[:8]}")

    hf_keys = store.all_keys()
    unconsumed = sorted(hf_keys - consumed)
    log(f"HF tensors: {len(hf_keys)} total, {len(consumed)} consumed, {len(unconsumed)} unconsumed")
    if unconsumed:
        log(f"  first unconsumed HF keys: {unconsumed[:8]}")

    ok = (len(unexpected) == 0) and (len(missing_non_extra) == 0) and (len(unconsumed) == 0)
    if not ok:
        raise RuntimeError(
            "native conversion mapping incomplete/incorrect "
            f"(unexpected={len(unexpected)}, missing_non_extra={len(missing_non_extra)}, "
            f"unconsumed={len(unconsumed)}); refusing to save."
        )
    log("mapping is COMPLETE and EXACT (0 unexpected, 0 non-_extra_state missing, 0 unconsumed).")

    from megatron.training.checkpointing import save_checkpoint

    ckpt_format = check_ckpt_format(margs.ckpt_format)
    os.makedirs(save_dir, exist_ok=True)
    release = ckpt_format == "torch_dist"
    if release:
        set_aside_interrupted_release(save_dir, log)
    check_free_space(model, save_dir, log)
    log(f"saving {ckpt_format} checkpoint to {save_dir} ...")
    # torch's DCP CheckpointException derives from BaseException, not Exception
    try:
        save_checkpoint(
            margs.iteration,
            [model],
            None,
            None,
            num_floating_point_operations_so_far=0,
            release=release,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        message = f"writing the {ckpt_format} checkpoint to {save_dir} failed ({e!r})"
        if _ran_out_of_space(e):
            message += (
                f"; most likely the disk filled up ({_free_space_text(save_dir)}). "
                "Set native_ckpt_out (or DATA_PATH) to a volume with room."
            )
        raise RuntimeError(message) from e
    log(f"DONE. Converted {ckpt_format} checkpoint at: {save_dir}")


# How a write that runs out of space is reported: torch.save loses the OSError and raises
# "[enforce fail at inline_container.cc] unexpected pos <n> vs <m>" (see check_free_space);
# newer torch and other writers raise the OSError.
_OUT_OF_SPACE_TEXT = ("unexpected pos", "No space left on device", "Disk quota exceeded")


def _ran_out_of_space(exc: BaseException) -> bool:
    """Whether ``exc``, or an exception it was raised from, is a write that ran out of space.

    A DCP ``CheckpointException`` includes the errors of its ranks in its text.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        if any(text in str(exc) for text in _OUT_OF_SPACE_TEXT):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _free_space_text(path: str) -> str:
    try:
        return f"{_gib(shutil.disk_usage(path).free)} free there now"
    except OSError as e:
        return f"its free space could not be read: {e}"


# Per-tensor overhead of the files a save writes: each tensor is its own small zip
# archive (torch.save: pickle, version records, 64-byte aligned data) in torch_dist,
# plus the checkpoint metadata. Measured at ~1.6 KiB per tensor; 4 KiB leaves room.
_SAVE_OVERHEAD_PER_TENSOR = 4096


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def checkpoint_bytes(model) -> int:
    """Bytes a save of ``model``'s weights writes, in either format: every tensor of its
    state dict at its own size (``torch_dist`` writes each tensor's data; legacy ``torch``
    each storage once), plus a small per-tensor file overhead."""
    total, seen = 0, set()
    for t in model.state_dict().values():
        if not torch.is_tensor(t):
            continue
        total += _SAVE_OVERHEAD_PER_TENSOR
        key = (t.data_ptr(), tuple(t.shape), t.dtype)
        if key in seen:  # the same tensor under two names (tied weights)
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total


def check_free_space(model, save_dir: str, log=print) -> None:
    """Refuse a save that cannot fit in ``save_dir``, before anything is written.

    A full disk otherwise surfaces only as a torn checkpoint: torch reports the failed
    write of a tensor's data as ``[enforce fail at inline_container.cc] unexpected pos
    <n> vs <m>`` (the ``OSError`` behind it is lost), after minutes of writing, and a
    retry fails the same way. The converted checkpoint of a large model is hundreds of
    GiB (bf16), more than a node's local ``/tmp`` often has. The refusal names the
    interrupted saves set aside in ``save_dir`` (``set_aside_interrupted_release``),
    which take space there.
    """
    need = checkpoint_bytes(model)
    os.makedirs(save_dir, exist_ok=True)
    free = shutil.disk_usage(save_dir).free
    log(f"checkpoint needs ~{_gib(need)}; {_gib(free)} free in {save_dir}")
    if free < need:
        raise RuntimeError(
            f"not enough disk space for the converted checkpoint in {save_dir}: needs ~{_gib(need)}, "
            f"{_gib(free)} free.{_set_aside_text(save_dir)} Set native_ckpt_out (or DATA_PATH) to a "
            "volume with room."
        )


def _set_aside_text(save_dir: str) -> str:
    aside = sorted(n for n in os.listdir(save_dir) if n.startswith("release.incomplete-"))
    if not aside:
        return ""
    size = sum(_dir_bytes(os.path.join(save_dir, n)) for n in aside)
    return (
        f" Interrupted saves set aside there use {_gib(size)} ({', '.join(aside)}); "
        "deleting them frees that space."
    )


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # removed or unreadable while walking; the total only goes in a message
                continue
    return total


def read_hf_config(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Output checkpoint format
# ---------------------------------------------------------------------------
# ``torch_dist``: Megatron distributed checkpoint (``megatron.core.dist_checkpointing``)
#     in the ``release/`` layout. Each tensor is stored with its global shape, so
#     Megatron reshards it on load and the same converted checkpoint fine-tunes
#     at any tensor / pipeline / expert parallel size. Default.
# ``torch``: legacy per-rank checkpoint (``iter_0000001/mp_rank_00/``). Megatron
#     does not reshard it, so it loads only at TP = PP = EP = 1.
CKPT_FORMATS = ("torch_dist", "torch")
DEFAULT_CKPT_FORMAT = "torch_dist"

TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def check_ckpt_format(ckpt_format: str) -> str:
    """Return ``ckpt_format`` if the native converter can write it, else raise ``ValueError``."""
    if ckpt_format not in CKPT_FORMATS:
        raise ValueError(
            f"unsupported native checkpoint format {ckpt_format!r}; expected one of {CKPT_FORMATS}"
        )
    return ckpt_format


def set_aside_interrupted_release(save_dir: str, log=print) -> None:
    """Make way for a ``torch_dist`` save into ``<save_dir>/release``.

    A converted model is saved as Megatron's ``release`` checkpoint: it is a
    starting point, not a training step. Megatron's loader finds ``release``
    through the tracker file, and Primus' LoRA pre-wrap loader looks for a
    ``torch_dist`` checkpoint in ``<path>/release`` (the layout the
    Megatron-Bridge conversion path produces).

    Megatron writes the tracker file after the checkpoint, so a ``release/``
    without one is what an interrupted save leaves: it is moved to
    ``release.incomplete-<unix time>`` and the conversion runs again. A
    ``release/`` with a tracker file beside it is refused, whatever the file
    holds (``release``, an iteration, or nothing Megatron can read): only a
    missing tracker shows an interrupted save.
    """
    release_dir = os.path.join(save_dir, "release")
    if not os.path.exists(release_dir):
        return
    tracker = os.path.join(save_dir, TRACKER_FILENAME)
    if os.path.exists(tracker):
        with open(tracker, errors="replace") as f:
            holds = f.read().strip()[:40]
        raise RuntimeError(
            f"{save_dir} already has a 'release' directory, and a tracker file holding {holds!r}; "
            "remove both or convert into an empty directory."
        )
    aside = f"{release_dir}.incomplete-{int(time.time())}"
    os.rename(release_dir, aside)
    log(f"WARNING: {release_dir} has no tracker file (an interrupted save); moved it to {aside}")


def tracked_ckpt_dir(save_dir: str) -> str | None:
    """The checkpoint directory the tracker file in ``save_dir`` names.

    ``<save_dir>/release`` or ``<save_dir>/iter_<7 digits>``, whether or not it
    exists; ``None`` when there is no tracker file or it holds neither.
    """
    tracker = os.path.join(save_dir, TRACKER_FILENAME)
    if not os.path.isfile(tracker):
        return None
    with open(tracker) as f:
        content = f.read().strip()
    if content == "release":
        return os.path.join(save_dir, "release")
    try:
        return os.path.join(save_dir, f"iter_{int(content):07d}")
    except ValueError:
        return None


def saved_ckpt_format(save_dir: str) -> str | None:
    """Format of the checkpoint the tracker file in ``save_dir`` points at.

    Returns ``"torch_dist"`` (the directory has dist-checkpointing
    ``metadata.json``), ``"torch"`` (it has ``mp_rank_*`` shards), or ``None``
    when there is no tracker file or the checkpoint it names is missing or
    incomplete.
    """
    ckpt_dir = tracked_ckpt_dir(save_dir)
    if ckpt_dir is None or not os.path.isdir(ckpt_dir):
        return None
    if os.path.isfile(os.path.join(ckpt_dir, "metadata.json")):
        return "torch_dist"
    if any(name.startswith("mp_rank_") for name in os.listdir(ckpt_dir)):
        return "torch"
    return None
