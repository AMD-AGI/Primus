###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free converter: Mistral Large 4 (``mistral_large4_text``) consolidated
block-FP8 checkpoint -> Megatron-Core (Primus's ``MistralLarge4Model``).

Same three entry points as ``native_loader_llama.py`` / ``native_loader_deepseek.py``
(``derive_megatron_fields``, ``build_margs``, ``build_mcore_state_dict``) plus
``build_model`` (the model is Primus's own, not ``GPTModel``; the l3 harness and ``convert``
build it through it) and ``convert`` (the posttrain
hook's in-process entry, through ``native_convert_common.load_validate_save``).

Checkpoint specifics:

* Mistral native names in ``consolidated-*.safetensors`` (``config.json`` names the index in
  ``transformers_weights``), not HF names; ``ConsolidatedStore`` reads that index.
* Most linears are FP8 e4m3 with 128x128 block scales in ``<name>.qscale_weight``:
  ``W = fp8 * scale`` per block (``dequant_block_fp8``); the scales are consumed with their
  weight. ``kv_a`` / ``kv_b`` / indexer ``wk`` / ``weights_proj`` / norms / router / head are
  bf16.
* Layer kinds follow the config's per-layer lists: NoPE layers (``kv_a`` 512 rows,
  ``kv_b`` 64*(128+128) rows), KV-sharing consumers (no ``kv_a`` / ``kv_a_norm``), shared
  indexer layers (no indexer).
* Experts: ``fc1 = cat[w1 (gate), w3 (up)]``, ``fc2 = w2``, one ``SequentialMLP`` expert each
  (the converter builds without grouped GEMM; torch_dist keys are the same, so training loads
  them into Primus-Turbo grouped GEMM).
* Vocab: padded at conversion to a multiple of ``make_vocab_size_divisible_by`` x the
  fine-tune's TP (``tensor_parallel_size``), as training pads it.
* The vision tower, projector and DSpark MTP head (``vision_encoder.*``, ``patch_merger.*``,
  ``pre_mm_projector_norm.*``, ``vision_language_adapter.*``, ``mtp_dspark.*``) are not
  part of this text model: they are recognised and skipped explicitly.

Memory: tensors are read in fp32 and copied into the model's own tensors (no second copy of the
weights), and the torch_dist save writes the model's tensors without cloning them
(``save_without_copies``). The routed experts of a layer are read by EXPERT_WORKERS threads at once, so up to that
many experts' fp32 temporaries (w1, w3, w2 and the fc1 concatenation) are alive beside the model. The conversion runs in one process (TP = PP = EP = 1) and saves
a ``torch_dist`` checkpoint, which Megatron reshards to the training layout when it loads. The
model is held on the host in params_dtype: the 16-layer cut needs ~420 GiB in bf16, the full
1T-parameter text model ~2 TiB, so convert it on a host with at least that much RAM.
"""

import argparse
import errno
import json
import logging
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.checkpoint.native_registry import dtype_name

# the Megatron-free layout module: this loader imports before Megatron is on sys.path
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    DEFAULT_FIRST_K_DENSE_REPLACE,
    MODEL_TYPE,
    hf_layer_lists,
    moe_layer_freq_list,
    text_config,
)

# ---------------------------------------------------------------------------
# Save with retry of a failed write (used by convert())
# ---------------------------------------------------------------------------
SAVE_ATTEMPTS = 3


def _write_failure(e) -> bool:
    """True for the intermittent writer failures a retry can fix: torch's ``CheckpointException``
    or Megatron's async-writer ``RuntimeError('Worker failure ...')``, unless the error is a full
    disk (ENOSPC), which a rewrite of the whole checkpoint would hit again."""
    from torch.distributed.checkpoint.api import CheckpointException

    text = repr(e)
    if "No space left on device" in text or getattr(e, "errno", None) == errno.ENOSPC:
        return False
    return isinstance(e, CheckpointException) or (isinstance(e, RuntimeError) and "Worker failure" in str(e))


def _free_gib(path) -> str:
    """Free space at ``path`` in GiB for the log, or ``?`` when it cannot be read."""
    try:
        return f"{shutil.disk_usage(path).free / 2**30:.1f}"
    except OSError:
        return "?"


def _save_with_retry(model, sd, consumed, store, margs, save_dir, log):
    """``load_validate_save`` with up to SAVE_ATTEMPTS tries of the write only.

    ``load_validate_save`` writes a ``torch_dist`` checkpoint straight into ``release/`` (a
    ``torch`` one into ``iter_<n>``). A failure before the write starts (the mapping check, a
    refused target) is raised at once. A failure after it started (the directory the save writes
    was created by this attempt: torch's ``CheckpointException`` or Megatron's
    ``RuntimeError('Worker failure ...')`` from the async writer, e.g. the intermittent
    "[enforce fail at inline_container.cc:672] unexpected pos") is retried after removing only
    what that save wrote (that directory and the tracker file); after the last attempt the partial
    directory is removed too, so the next conversion starts clean. Any other error after the
    write started (no space left on device, a validation error) is raised at once, after removing
    that directory, as another full write would fail the same way. Nothing else in ``save_dir``
    is touched. An earlier, complete conversion (a tracker file) is refused; an interrupted
    ``release/`` without a tracker file is set aside first (``set_aside_interrupted_release``,
    as ``load_validate_save`` would), so every ``release/`` the attempts see is their own.
    """
    # CheckpointException derives from BaseException, not Exception: caught by name
    from torch.distributed.checkpoint.api import CheckpointException

    if margs.ckpt_format == "torch_dist":
        written = os.path.join(save_dir, "release")
    else:
        written = os.path.join(save_dir, f"iter_{int(margs.iteration):07d}")
    tracker = os.path.join(save_dir, common.TRACKER_FILENAME)
    earlier = common.tracked_ckpt_dir(save_dir)
    if earlier is not None or (margs.ckpt_format != "torch_dist" and os.path.exists(written)):
        raise RuntimeError(f"{earlier or written} exists: refusing to overwrite an earlier conversion")
    if margs.ckpt_format == "torch_dist":
        # a leftover tracker-less release/ is moved aside before the first attempt: a failure of
        # the first write is then retried and its partial release/ removed like any other's
        common.set_aside_interrupted_release(save_dir, log)
    for attempt in range(1, SAVE_ATTEMPTS + 1):
        existed = os.path.exists(written)
        try:
            common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
            return
        except (Exception, CheckpointException) as e:  # noqa: BLE001 - only a started write is retried
            if existed or not os.path.exists(written):
                raise  # the write never started (mapping error or refused target): nothing of ours
            if not _write_failure(e):
                # a deterministic error (no space, validation): another full write would fail too
                shutil.rmtree(written, ignore_errors=True)
                if os.path.isfile(tracker):
                    os.remove(tracker)
                raise
            log(
                f"{margs.ckpt_format} write failed (attempt {attempt} of {SAVE_ATTEMPTS}): {e!r}; "
                f"{_free_gib(save_dir)} GiB free in {save_dir}"
            )
            shutil.rmtree(written, ignore_errors=True)
            if os.path.isfile(tracker):
                os.remove(tracker)
            if attempt == SAVE_ATTEMPTS:
                raise


# release components that are not part of the text decoder (not converted)
NOT_TEXT_PREFIXES = (
    "vision_encoder.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "vision_language_adapter.",
    "mtp_dspark.",
)
FP8_BLOCK = 128
# threads that read and dequantise one MoE layer's experts at once (build_mcore_state_dict)
EXPERT_WORKERS = 16
# progress log cadence of build_mcore_state_dict (layers)
LOG_EVERY_LAYERS = 8


# ---------------------------------------------------------------------------
# Consolidated (Mistral-format) safetensors store with block-FP8 dequantisation
# ---------------------------------------------------------------------------
def dequant_block_fp8(w: torch.Tensor, scale: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """``W = fp8 * scale`` per ``block x block`` tile (fp32 result); ``scale`` [ceil(r/b), ceil(c/b)].

    When both dims are multiples of ``block`` (every quantised weight of the release) the scale is
    broadcast over a ``[r/b, b, c/b, b]`` view: one pass, no expanded copy of the scale. The input
    is never modified."""
    rows, cols = w.shape
    expected = (-(-rows // block), -(-cols // block))
    if tuple(scale.shape) != expected:
        # a scale on another block grid would dequantise to wrong weights without an error
        raise ValueError(
            f"block-FP8 scale shape {tuple(scale.shape)} does not match weight {(rows, cols)} with "
            f"{block}x{block} blocks (expected {expected})"
        )
    if rows % block == 0 and cols % block == 0:
        # out of place: the input may be a store's cached tensor (w.float() is w for fp32)
        out = w.float().view(rows // block, block, cols // block, block) * scale.float()[:, None, :, None]
        return out.view(rows, cols)
    s = scale.float().repeat_interleave(block, dim=0)[:rows].repeat_interleave(block, dim=1)[:, :cols]
    return w.float() * s


class ConsolidatedStore(common.SafetensorsStore):
    """``SafetensorsStore`` over the index config.json names in ``transformers_weights``
    (``consolidated.safetensors.index.json``; the base store reads only
    ``model.safetensors.index.json``, so the constructor sets the weight map itself and every
    read goes through the base ``get``). ``read_weight`` (the converter's one read path, for
    this store and the base one) dequantises FP8 weights with their ``qscale_weight``."""

    def __init__(self, hf_dir: str):
        cfg = common.read_hf_config(hf_dir)
        name = cfg.get("transformers_weights") or "consolidated.safetensors.index.json"
        index = os.path.join(hf_dir, name)
        if not os.path.isfile(index):
            # the converter maps Mistral-native names (layers.N.attention.wq.weight, ...); an
            # HF-layout copy (model.layers.N.self_attn...) has none of them, so it is refused here
            # instead of failing on the first missing key
            how = "config.json's transformers_weights" if cfg.get("transformers_weights") else "the default"
            raise FileNotFoundError(
                f"Mistral Large 4 converter: {index} ({how}) is missing. It reads the release's Mistral "
                "consolidated layout (consolidated-*.safetensors and their index); an HF-layout copy "
                "(model.safetensors.index.json, HF tensor names) is not supported."
            )
        self.hf_dir = hf_dir
        self._local = threading.local()  # per-thread safe_open handles (see _handle)
        self._single = None
        self._bin_cache = None
        with open(index) as f:
            self._name_to_file = {k: os.path.join(hf_dir, v) for k, v in json.load(f)["weight_map"].items()}

    def _handle(self, filepath: str):
        """One ``safe_open`` handle per file and per thread (the converter reads experts in a
        thread pool)."""
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        h = handles.get(filepath)
        if h is None:
            from safetensors import safe_open

            h = handles[filepath] = safe_open(filepath, framework="pt", device="cpu")
        return h


def _stored_is_fp8(store, name: str) -> bool:
    """Whether ``name`` is stored as a float8 tensor (read from the safetensors header, no data).
    A store that is not a ``SafetensorsStore`` (a test's dict store) says it with ``is_fp8``."""
    if not isinstance(store, common.SafetensorsStore):
        return bool(getattr(store, "is_fp8", lambda n: False)(name))
    if store._bin_cache is not None:
        return "float8" in str(store._bin_cache[name].dtype)
    path = store._single if store._single is not None else store._name_to_file[name]
    return str(store._handle(path).get_slice(name).get_dtype()).upper().startswith("F8")


def read_weight(store, name: str, dtype, consumed: set) -> torch.Tensor:
    """``name`` in ``dtype``, dequantised (in fp32) with its ``<name minus .weight>.qscale_weight``
    block scales. An FP8 tensor without that scale is refused (a missing shard or a renamed key would
    otherwise be cast without scaling: wrong weights, no error)."""
    consumed.add(name)
    scale_name = name[: -len(".weight")] + ".qscale_weight" if name.endswith(".weight") else None
    has = getattr(store, "has", None) or (lambda n: n in store.all_keys())
    has_scale = scale_name is not None and has(scale_name)
    if not has_scale and _stored_is_fp8(store, name):
        raise RuntimeError(f"{name} is stored in FP8 but has no block scale {scale_name!r} in the checkpoint")
    t = store.get(name, torch.float32)
    if has_scale:
        consumed.add(scale_name)
        t = dequant_block_fp8(t, store.get(scale_name, torch.float32))
    return t.to(dtype)


# ---------------------------------------------------------------------------
# config.json -> Megatron fields
# ---------------------------------------------------------------------------
def _rope_theta(t: dict, rope: dict) -> float:
    """RoPE base: ``rope_theta`` in the rope dict (``rope_parameters``, transformers 5) or at the top
    of the text config (the older layout beside ``rope_scaling``). If both are set they must agree;
    if neither is set it is an error (never a silent default)."""
    in_rope, top = rope.get("rope_theta"), t.get("rope_theta")
    if in_rope is None and top is None:
        raise ValueError(
            "Mistral Large 4 config has no rope_theta (neither in rope_parameters nor top level)"
        )
    if in_rope is not None and top is not None and float(in_rope) != float(top):
        raise ValueError(f"Mistral Large 4 config: rope_theta {in_rope} in the rope dict != top-level {top}")
    return float(in_rope if in_rope is not None else top)


def yarn_attention_factor(rope: dict) -> float:
    """The cos / sin scale transformers' YaRN applies (``_compute_yarn_parameters``):
    ``attention_factor`` when the config gives it, else ``get_mscale(factor, mscale) /
    get_mscale(factor, mscale_all_dim)`` when both mscales are given, else ``get_mscale(factor)``
    (``0.1 * ln(factor) + 1`` for ``factor > 1``)."""

    def get_mscale(scale, mscale=1.0):
        return 1.0 if scale <= 1 else 0.1 * float(mscale) * math.log(scale) + 1.0

    if rope.get("attention_factor") is not None:
        return float(rope["attention_factor"])
    factor = float(rope["factor"])
    if rope.get("mscale") and rope.get("mscale_all_dim"):
        return get_mscale(factor, rope["mscale"]) / get_mscale(factor, rope["mscale_all_dim"])
    return get_mscale(factor)


def _check_yarn_scaling(rope: dict) -> None:
    """The model applies YaRN frequencies with no cos / sin scale and a softmax scale of
    ``qk_head_dim ** -0.5`` (the release: ``attention_factor`` 1.0, no ``mscale_all_dim``). A config
    whose YaRN scale resolves to anything else (a missing ``attention_factor`` included: transformers
    then uses ``0.1 ln(factor) + 1``, ~1.76 for 2048) is refused instead of converting to a model
    with different attention logits."""
    factor = yarn_attention_factor(rope)
    if abs(factor - 1.0) > 1e-12 or rope.get("mscale_all_dim"):
        raise ValueError(
            f"Mistral Large 4 converter expects YaRN attention_factor 1.0 and no mscale_all_dim; this config "
            f"gives attention_factor {factor:.6g} (rope parameters {dict(rope)}), which the model does not apply"
        )


def _check_routing(t: dict) -> None:
    """The converter and model implement the release's routing: softmax over all experts, top-k,
    renormalised, no expert bias, one expert group. Refuse a config that asks for other routing
    instead of loading it with different semantics."""
    want = {"n_group": 1, "topk_group": 1}
    for key, value in want.items():
        if t.get(key) is not None and int(t[key]) != value:
            raise ValueError(
                f"Mistral Large 4 converter: {key}={t[key]} (group-limited routing) is not supported"
            )
    scoring = t.get("scoring_func", "softmax")
    if scoring != "softmax":
        raise ValueError(
            f"Mistral Large 4 converter: router scoring_func {scoring!r} is not supported (softmax)"
        )
    if t.get("norm_topk_prob") is False:
        raise ValueError(
            "Mistral Large 4 converter: norm_topk_prob false is not supported (renormalised top-k)"
        )
    if t.get("topk_method") not in (None, "greedy"):
        raise ValueError(f"Mistral Large 4 converter: topk_method {t.get('topk_method')!r} is not supported")


def derive_megatron_fields(hf_config):
    t = text_config(hf_config)
    n = int(t["num_hidden_layers"])
    rope = t.get("rope_parameters") or t.get("rope_scaling")
    if not rope:
        raise ValueError(
            "Mistral Large 4 converter expects YaRN rope parameters (rope_parameters / rope_scaling)"
        )
    rope_type = rope.get("rope_type", rope.get("type"))  # legacy rope_scaling dicts use "type"
    if rope_type != "yarn":
        raise ValueError(f"Mistral Large 4 converter expects YaRN rope, got {rope_type!r}")
    missing = [k for k in ("factor", "original_max_position_embeddings") if k not in rope]
    if missing:
        raise ValueError(f"Mistral Large 4 converter: YaRN rope parameters lack {missing}")
    _check_routing(t)
    if t.get("tie_word_embeddings", hf_config.get("tie_word_embeddings", False)):
        raise ValueError(
            "Mistral Large 4 converter: tied embeddings are not supported (the release has output.weight)"
        )
    _check_yarn_scaling(rope)
    first_k = t.get("first_k_dense_replace")
    first_k = DEFAULT_FIRST_K_DENSE_REPLACE if first_k is None else int(first_k)
    # omitted per-layer lists take the HF default (as the training-side patch does)
    lists, defaulted = hf_layer_lists(t)
    if defaulted:
        logging.getLogger(__name__).warning(
            "Mistral Large 4 converter: config.json has no %s; using the HF default (no NoPE layers, no sharing)",
            defaulted,
        )
    n_shared = int(t.get("n_shared_experts", 0) or 0)
    return dict(
        num_layers=n,
        hidden_size=int(t["hidden_size"]),
        ffn_hidden_size=int(t["intermediate_size"]),
        num_attention_heads=int(t["num_attention_heads"]),
        kv_channels=int(t.get("head_dim") or t["qk_rope_head_dim"]),
        norm_epsilon=float(t["rms_norm_eps"]),
        vocab_size=int(t["vocab_size"]),
        untie_embeddings_and_output_weights=True,  # tied configs are refused above
        multi_latent_attention=True,
        q_lora_rank=None,
        kv_lora_rank=int(t["kv_lora_rank"]),
        qk_head_dim=int(t["qk_nope_head_dim"]),
        qk_pos_emb_head_dim=int(t["qk_rope_head_dim"]),
        v_head_dim=int(t["v_head_dim"]),
        qk_layernorm=False,
        rope_type="yarn",
        rotary_base=_rope_theta(t, rope),
        rotary_scaling_factor=float(rope["factor"]),
        original_max_position_embeddings=int(rope["original_max_position_embeddings"]),
        beta_fast=float(rope.get("beta_fast", 32)),
        beta_slow=float(rope.get("beta_slow", 1)),
        mscale=1.0,
        mscale_all_dim=0.0,
        num_experts=int(t["n_routed_experts"]),
        moe_ffn_hidden_size=int(t["moe_intermediate_size"]),
        moe_shared_expert_intermediate_size=int(t["moe_intermediate_size"]) * n_shared if n_shared else None,
        moe_router_topk=int(t["num_experts_per_tok"]),
        moe_layer_freq=moe_layer_freq_list(first_k, n, int(t["n_routed_experts"])),
        moe_router_topk_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
        first_k_dense_replace=first_k,
        **lists,
        index_n_heads=int(t["index_n_heads"]),
        index_head_dim=int(t["index_head_dim"]),
        index_topk=int(t["index_topk"]),
        final_logit_softcapping=t.get("final_logit_softcapping"),
        _num_shared_experts=n_shared,
    )


# extra (non-Megatron-argparse) fields the model config reads off margs
MODEL_FIELDS = (
    "first_k_dense_replace",
    "no_rope_layers",
    "indexer_types",
    "kv_source_layers",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "final_logit_softcapping",
    "beta_fast",
    "beta_slow",
)
ARG_FIELDS = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "kv_channels",
    "norm_epsilon",
    "untie_embeddings_and_output_weights",
    "multi_latent_attention",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_head_dim",
    "qk_pos_emb_head_dim",
    "v_head_dim",
    "qk_layernorm",
    "rope_type",
    "rotary_base",
    "rotary_scaling_factor",
    "original_max_position_embeddings",
    "mscale",
    "mscale_all_dim",
    "num_experts",
    "moe_ffn_hidden_size",
    "moe_shared_expert_intermediate_size",
    "moe_router_topk",
    "moe_layer_freq",
    "moe_router_topk_scaling_factor",
)


def _check_layout(args):
    """The converter maps full (unsharded) tensors with global layer and expert indices in one
    process (TP = PP = EP = 1) whatever layout the fine-tune uses: a ``torch_dist`` checkpoint is
    resharded by Megatron to the training layout when it loads (AMD-AGI/Primus#1400), so the
    requested TP / PP / EP are accepted (TP sets only the vocab padding, see ``build_margs``). The legacy ``torch`` format loads only at
    TP = PP = EP = 1 (the posttrain hook refuses it earlier for other layouts)."""
    sizes = {
        "tensor_parallel_size": int(getattr(args, "tensor_parallel_size", 1) or 1),
        "pipeline_parallel_size": int(getattr(args, "pipeline_parallel_size", 1) or 1),
        "expert_parallel_size": int(getattr(args, "expert_parallel_size", 1) or 1),
    }
    fmt = getattr(args, "ckpt_format", common.DEFAULT_CKPT_FORMAT)
    if fmt == "torch" and any(v != 1 for v in sizes.values()):
        raise ValueError(
            f"a legacy torch checkpoint loads only at TP = PP = EP = 1, got {sizes}; use "
            "ckpt_format torch_dist, which Megatron reshards to the training layout when it loads"
        )


def hf_dtype(cfg: dict):
    """The release's dtype string (registry ``dtype_from_config`` hook): its ``config.json`` has a
    top-level ``dtype`` (or one in ``text_config``), not ``torch_dtype``."""
    return cfg.get("dtype") or text_config(cfg).get("dtype")


def vocab_divisor(args) -> int:
    """``make_vocab_size_divisible_by`` x the fine-tune's TP: Megatron pads the vocab to a multiple
    of that product, so the conversion (at TP 1) pads it as training will."""
    return int(args.make_vocab_size_divisible_by) * int(getattr(args, "tensor_parallel_size", 1) or 1)


def conversion_dtype(raw) -> str:
    """``"bf16"`` / ``"fp16"`` / ``"fp32"`` for the dtype to convert in: those names, or a torch / HF
    name (``bfloat16``, ``torch.float16``, ...) as ``native_registry.dtype_name`` reads it. Raises
    ``ValueError`` for anything else (a typo must not build the 1T model in fp32 silently)."""
    text = str(raw or "").strip().lower()
    if text in ("bf16", "fp16", "fp32"):
        return text
    name = dtype_name(text)
    if name is None:
        raise ValueError(f"Mistral Large 4 converter: unknown dtype {raw!r}; expected bf16, fp16 or fp32")
    return name


def build_margs(fields, args):
    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])
    for k in ARG_FIELDS:
        setattr(m, k, fields[k])
    for k in MODEL_FIELDS:  # per-layer lists cut to num_layers by the config
        setattr(m, k, fields[k])
    m.model_type = MODEL_TYPE

    m.vocab_size = None  # padded from the tokenizer, as Primus SFT does
    # Megatron pads the vocab to a multiple of make_vocab_size_divisible_by x TP; the conversion
    # runs at TP 1, so the training TP goes into the divisor: the embedding / head rows match a
    # fine-tune at that TP. Another training TP pads to the same rows only when the vocab is
    # already a multiple of both divisors (the release's 204800 is a multiple of 128 x 8); otherwise
    # convert again for that TP (the torch_dist load fails on the row count, it does not pad)
    m.make_vocab_size_divisible_by = vocab_divisor(args)
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False
    m.normalization = "RMSNorm"
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = False
    m.position_embedding_type = "rope"
    m.apply_rope_fusion = False
    m.attention_softmax_in_fp32 = True
    m.moe_router_score_function = "softmax"
    m.moe_router_pre_softmax = False
    m.moe_router_enable_expert_bias = False
    m.moe_router_dtype = "fp32"
    m.moe_router_load_balancing_type = "none"
    m.moe_aux_loss_coeff = 0.0
    # SequentialMLP experts (one linear per expert): the same weights and checkpoint keys as the
    # grouped GEMM training uses (torch_dist reshards between them); TE's grouped GEMM keeps
    # per-device stream handles and fails ("invalid resource handle") when a check spreads the
    # model's layers over several GPUs in one process
    m.moe_grouped_gemm = False
    # allgather: the conversion runs at EP 1, where the dispatcher changes neither the weights nor the
    # checkpoint keys. Megatron's alltoall dispatcher copies its token counts to the host on one
    # class-level side stream (made on the first device), which races when a check spreads the
    # layers over several GPUs in one process ("split_with_sizes expects split_sizes to sum ... got
    # [0, 0, ...]"); allgather reads them with a blocking copy on each layer's own device
    m.moe_token_dispatcher_type = "allgather"
    m.moe_shared_expert_overlap = False
    m.moe_permute_fusion = False

    m.seq_length = args.seq_length
    m.max_position_embeddings = args.seq_length
    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1
    _check_layout(args)  # converted at TP = PP = EP = 1; torch_dist reshards on load
    m.tensor_model_parallel_size = 1
    m.pipeline_model_parallel_size = 1
    m.expert_model_parallel_size = 1
    m.expert_tensor_parallel_size = 1
    m.context_parallel_size = 1
    m.sequence_parallel = False
    m.world_size = 1
    m.rank = 0
    dtype = conversion_dtype(args.dtype)
    if dtype == "bf16":
        m.bf16 = True
        m.params_dtype = torch.bfloat16
    elif dtype == "fp16":
        m.fp16 = True
        m.params_dtype = torch.float16
    else:
        m.params_dtype = torch.float32
    m.use_cpu_initialization = True
    m.perform_initialization = False
    m.transformer_impl = "transformer_engine"
    m.gradient_accumulation_fusion = False
    m.async_tensor_model_parallel_allreduce = False
    m.bias_gelu_fusion = False
    m.bias_swiglu_fusion = False
    m.masked_softmax_fusion = False
    m.bias_dropout_fusion = False
    m.apply_query_key_layer_scaling = False

    m.save = args.save_dir
    # torch_dist (default): Megatron reshards it on load, so the one-process conversion
    # (TP = PP = EP = 1) loads into any training layout; legacy torch loads only at TP = PP = EP = 1
    m.ckpt_format = getattr(args, "ckpt_format", common.DEFAULT_CKPT_FORMAT)
    m.ckpt_fully_parallel_save = False
    m.no_save_optim = True
    m.no_save_rng = True
    m.no_load_optim = True
    m.no_load_rng = True
    m.save_interval = 1
    m.async_save = False
    m.ckpt_assume_constant_structure = False
    m.iteration = 1

    m = validate_args(m)
    m.iteration = 1
    return m


def build_model(margs):
    """The Primus ``MistralLarge4Model``, built as training builds it (the family's model_type through
    ``native_convert_common.build_mcore_gpt_model``). ``convert()`` uses it, and so can any caller
    that loads the released weights into the model for a parity check."""
    return common.build_mcore_gpt_model(margs, family=MODEL_TYPE)


# ---------------------------------------------------------------------------
# Mistral consolidated -> mcore mapping
# ---------------------------------------------------------------------------
def build_mcore_state_dict(model, store, margs, fields, log):
    """``(sd, consumed)``: the model's state dict filled from the release.

    Every tensor is read as fp32 (FP8 weights dequantised with their block scales) and copied
    straight into the model's own tensor of that name (``own[k].copy_``), so the cast happens once,
    to the dtype the model keeps that tensor in: params_dtype for most, fp32 for the fp32-kept
    LeRoPE ``alpha`` and indexer ``weights_proj`` / SSMax (their values are never rounded through
    bf16; the attention SSMax ``s`` / ``b`` are trainable and stored in params_dtype). The returned
    ``sd`` holds the model's own tensors (no second copy of the weights); host memory is the model
    plus the fp32 temporaries in flight: one tensor, or up to EXPERT_WORKERS experts of a MoE layer.
    """
    own = model.state_dict()
    padded_vocab = own["embedding.word_embeddings.weight"].shape[0]
    n = int(margs.num_layers)
    cfg = model.config
    consumed = set()

    class _Into(dict):
        """``sd[k] = t`` copies ``t`` into the model's tensor ``k``. A key the model does not have,
        or a shape that differs, is raised at once (before the remaining layers are read)."""

        def __setitem__(self, k, t):
            if k not in own or not torch.is_tensor(own[k]):
                raise RuntimeError(f"{k}: the model has no such tensor (mapping error)")
            if own[k].shape != t.shape:
                raise RuntimeError(
                    f"{k}: shape {tuple(t.shape)} does not match the model's {tuple(own[k].shape)}"
                )
            with torch.no_grad():
                own[k].copy_(t)
            dict.__setitem__(self, k, own[k])

    sd = _Into()

    def get(name):
        """The release tensor ``name`` in fp32 (dequantised when it is FP8)."""
        return read_weight(store, name, torch.float32, consumed)

    sd["embedding.word_embeddings.weight"] = common.pad_vocab(get("tok_embeddings.weight"), padded_vocab)
    sd["decoder.final_layernorm.weight"] = get("norm.weight")
    sd["output_layer.weight"] = common.pad_vocab(get("output.weight"), padded_vocab)
    sd["decoder.rotary.alpha"] = get("lerope.alpha")

    # one pool reads every MoE layer's experts (created once, not per layer)
    expert_ids = range(int(margs.num_experts))
    has_shared = bool(fields.get("_num_shared_experts"))
    with ThreadPoolExecutor(max_workers=EXPERT_WORKERS) as pool:
        for L in range(n):
            p, h = f"decoder.layers.{L}.", f"layers.{L}."
            kv_src, idx_src, is_moe = (
                cfg.kv_source(L),
                cfg.index_source_layer(L),
                bool(margs.moe_layer_freq[L]),
            )
            a, ha = p + "self_attention.", h + "attention."
            sd[p + "input_layernorm.weight"] = get(h + "attention_norm.weight")
            sd[p + "pre_mlp_layernorm.weight"] = get(h + "ffn_norm.weight")
            sd[a + "linear_q_proj.weight"] = get(ha + "wq.weight")
            if kv_src == L:
                sd[a + "linear_kv_down_proj.weight"] = get(ha + "wkv_a_with_mqa.weight")
                sd[a + "kv_layernorm.weight"] = get(ha + "kv_a_norm.weight")
            sd[a + "linear_kv_up_proj.weight"] = get(ha + "wkv_b.weight")
            sd[a + "linear_proj.weight"] = get(ha + "wo.weight")
            sd[a + "ssmax_s"] = get(ha + "ssmax_scaling.s")
            sd[a + "ssmax_b"] = get(ha + "ssmax_scaling.b")
            if idx_src == L:
                i, hi = a + "indexer.", ha + "indexer."
                for mine, theirs in (
                    ("q_norm_weight", "q_input_norm.weight"),
                    ("q_norm_bias", "q_input_norm.bias"),
                    ("wq_b", "wq_b.weight"),
                    ("wk", "wk.weight"),
                    ("k_norm_weight", "k_norm.weight"),
                    ("k_norm_bias", "k_norm.bias"),
                    ("weights_proj", "weights_proj.weight"),
                    ("ssmax_s", "ssmax_scaling.s"),
                    ("ssmax_b", "ssmax_scaling.b"),
                ):
                    sd[i + mine] = get(hi + theirs)
            if not is_moe:
                sd[p + "mlp.linear_fc1.weight"] = torch.cat(
                    [get(h + "feed_forward.w1.weight"), get(h + "feed_forward.w3.weight")], dim=0
                )
                sd[p + "mlp.linear_fc2.weight"] = get(h + "feed_forward.w2.weight")
            else:
                sd[p + "mlp.router.weight"] = get(h + "gate.weight")

                def expert(e, p=p, h=h):
                    he = h + f"experts.{e}."
                    fc1 = torch.cat([get(he + "w1.weight"), get(he + "w3.weight")], dim=0)
                    fc2 = get(he + "w2.weight")
                    sd[p + f"mlp.experts.local_experts.{e}.linear_fc1.weight"] = fc1  # SequentialMLP
                    sd[p + f"mlp.experts.local_experts.{e}.linear_fc2.weight"] = fc2

                # the experts are ~97% of the bytes: read and dequantised in parallel (each writes its
                # own model tensors; torch's CPU ops and the file reads release the GIL)
                list(pool.map(expert, expert_ids))
                if has_shared:
                    sd[p + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                        [get(h + "shared_experts.w1.weight"), get(h + "shared_experts.w3.weight")], dim=0
                    )
                    sd[p + "mlp.shared_experts.linear_fc2.weight"] = get(h + "shared_experts.w2.weight")
            if L % LOG_EVERY_LAYERS == 0 or L == n - 1:
                log(
                    f"  mapped layer {L:2d} (rope={cfg.uses_rope(L)} kv_src={kv_src} "
                    f"index_src={idx_src} {'MoE' if is_moe else 'dense'})"
                )

    # release tensors outside the text decoder, and layers past a depth-reduced model's num_layers
    skipped = [k for k in store.all_keys() if k.startswith(NOT_TEXT_PREFIXES) or _layer_of(k) >= n]
    consumed.update(skipped)
    if skipped:
        log(
            f"  not part of this text model (vision / projector / DSpark MTP, or layers >= {n}): {len(skipped)} tensors"
        )
    return dict(sd), consumed


def _layer_of(name: str) -> int:
    if not name.startswith("layers."):
        return -1
    return int(name.split(".", 2)[1])


# ---------------------------------------------------------------------------
# Save without a second copy of the weights
# ---------------------------------------------------------------------------
def storage_exact_alias(t: torch.Tensor) -> torch.Tensor:
    """``t`` itself, or, for a contiguous CPU view into a larger storage, a tensor over a storage that
    is exactly its bytes and shares their memory (no copy).

    Megatron's torch_dist writer (``FileSystemWriterAsync.prepare_write_data``) clones every CPU
    tensor whose storage is larger than the tensor before it writes anything. The gated ``linear_fc1``
    weights are saved as two such views (``apply_swiglu_sharded_factory`` chunks gate | up), so a
    save of this model would hold ~2/3 of every expert's bytes twice in host memory. A slice of an ``UntypedStorage`` shares the
    memory of the storage it is sliced from, so the alias costs nothing and the writer writes it
    as it is."""
    if t.device.type != "cpu" or not t.is_contiguous() or t.numel() == 0:
        return t
    storage = t.untyped_storage()
    nbytes = t.numel() * t.element_size()
    if storage.nbytes() == nbytes:
        return t
    start = t.storage_offset() * t.element_size()
    alias = torch.empty(0, dtype=t.dtype)
    alias.set_(storage[start : start + nbytes], 0, t.shape, t.stride())
    return alias


def save_without_copies(model):
    """Make ``model.sharded_state_dict`` (this instance only: the conversion's save) return
    ShardedTensors whose data the torch_dist writer does not clone (``storage_exact_alias``).
    Factories (the SwiGLU split) are built here, as ``dist_checkpointing.save`` would build them;
    the model's own parameters are untouched, so a retried save sees the same weights."""
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, apply_factories

    original = model.sharded_state_dict

    def sharded_state_dict(*args, **kwargs):
        sharded = original(*args, **kwargs)
        apply_factories(sharded)

        def alias(x):
            if isinstance(x, ShardedTensor) and x.data is not None:
                x.data = storage_exact_alias(x.data)
            return x

        dict_list_map_inplace(alias, sharded)
        return sharded

    model.sharded_state_dict = sharded_state_dict
    return model


# ---------------------------------------------------------------------------
# In-process entry point (the posttrain checkpoint hook)
# ---------------------------------------------------------------------------
def check_vocab(padded_vocab_size, fields, hf_dir):
    """The tokenizer-padded vocabulary must hold every row of the release's embedding and head."""
    if int(padded_vocab_size) < int(fields["vocab_size"]):
        raise ValueError(
            f"tokenizer at {hf_dir} gives padded vocab {padded_vocab_size}, fewer rows than the "
            f"checkpoint's vocab_size {fields['vocab_size']}"
        )


def convert(
    hf_dir,
    save_dir,
    *,
    dtype="bf16",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    expert_parallel_size=1,
    ckpt_format=common.DEFAULT_CKPT_FORMAT,
    seq_length=4096,
    make_vocab_size_divisible_by=128,
    master_port=29562,
    log=None,
):
    """Convert the Mistral Large 4 release to a Megatron checkpoint in one process (TP = PP = EP = 1).

    ``tensor_parallel_size`` / ``pipeline_parallel_size`` / ``expert_parallel_size`` are the
    fine-tune's layout: any layout is accepted for ``torch_dist`` (the checkpoint is written
    unsharded and Megatron reshards it when it loads); legacy ``torch`` needs all three at 1.
    ``ckpt_format`` is ``"torch_dist"`` (default; Megatron reshards it to the training layout when
    it loads it) or ``"torch"`` (legacy, loads only at TP = PP = EP = 1); see
    ``native_convert_common.CKPT_FORMATS``.
    """
    if log is None:

        def log(*a):
            print("[ml4-convert]", *a, flush=True)

    dtype = conversion_dtype(dtype)  # refuse an unknown dtype before any work
    common.ensure_megatron_on_path()
    common.assert_bridge_free("entry")
    args = SimpleNamespace(
        hf_dir=hf_dir,
        save_dir=save_dir,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        ckpt_format=common.check_ckpt_format(ckpt_format),
        seq_length=seq_length,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
    )
    fields = derive_megatron_fields(common.read_hf_config(hf_dir))
    store = ConsolidatedStore(hf_dir)  # refuses a snapshot without the consolidated index before the build
    log(
        f"arch: L={fields['num_layers']} h={fields['hidden_size']} experts={fields['num_experts']} "
        f"topk={fields['moe_router_topk']} kv_lora={fields['kv_lora_rank']}"
    )
    margs = build_margs(fields, args)
    margs = common.init_megatron_single_process(margs, master_port, log=log)
    check_vocab(margs.padded_vocab_size, fields, hf_dir)
    common.assert_bridge_free("post-init")
    # the training builder (model_type mistral_large4) through Primus's model provider
    model = build_model(margs)
    sd, consumed = build_mcore_state_dict(model, store, margs, fields, log)
    if margs.ckpt_format == "torch_dist":
        save_without_copies(model)  # one copy of the weights on the host, also while saving
    _save_with_retry(model, sd, consumed, store, margs, save_dir, log)
    common.assert_bridge_free("post-save")
    return save_dir
