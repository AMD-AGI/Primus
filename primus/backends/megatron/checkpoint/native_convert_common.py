###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Author: Hu, Botao
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
``megatron.core`` / ``megatron.training`` / ``megatron.legacy``. ``megatron.bridge``
is never imported; a hard guard asserts it never leaked into ``sys.modules``.

The ROCm-only ``megatron.legacy.fused_kernels.load`` no-op that the legacy CPU
init path needs is supplied as a Primus ``phase="convert"`` monkeypatch
(``primus.backends.megatron.patches.checkpoint_convert_patches``), applied
in-process before conversion -- NOT by editing the submodule.
"""

import json
import os
import sys

import torch


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
    ``condition`` matches (e.g. the ROCm fused-kernels no-op). Must be called
    BEFORE any Megatron model build / ``fused_kernels.load`` so the monkeypatch
    is in effect. Returns the number of patches applied.
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
    from megatron.legacy import fused_kernels
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
    # ROCm: legacy fused kernels are CUDA-only and their build shells out to
    # nvcc; conversion never runs them. Neutralised by the phase="convert"
    # Primus patch (checkpoint_convert_patches) applied earlier this process.
    fused_kernels.load(margs)
    return margs


def build_mcore_gpt_model(margs):
    """Build the mcore ``GPTModel`` via Primus' own model-provider resolver.

    Using ``get_model_provider('gpt')`` (the same shim native SFT uses) means the
    produced parameter names match exactly what SFT will build, and keeps us off
    the ``tools/checkpoint`` directory layout.
    """
    from primus.core.utils.import_utils import get_model_provider

    model_provider = get_model_provider(model_type="gpt")
    model = model_provider(pre_process=True, post_process=True)
    return model.to(margs.params_dtype)


# ---------------------------------------------------------------------------
# Load mapped state dict, validate, and save a legacy torch checkpoint
# ---------------------------------------------------------------------------
def load_validate_save(model, sd, consumed, store, margs, save_dir, log):
    """Apply ``sd`` to ``model`` (strict=False), assert an exact mapping, save.

    Raises ``RuntimeError`` if the mapping is not complete/exact so the caller
    (conversion hook) fails loudly instead of writing a silently-wrong ckpt.
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

    os.makedirs(save_dir, exist_ok=True)
    log(f"saving legacy torch checkpoint to {save_dir} ...")
    save_checkpoint(
        margs.iteration,
        [model],
        None,
        None,
        num_floating_point_operations_so_far=0,
    )
    log(f"DONE. Converted checkpoint at: {save_dir}")


def read_hf_config(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        return json.load(f)
