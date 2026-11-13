###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HF ``muse_spark`` (Muse Spark text decoder) -> Megatron-Core converter.

Single-process build-model-and-copy path (:class:`MuseSparkConverter`). The weights are
copied into the built model one decoder layer at a time (:func:`stream_into_model`), so host
RAM holds one copy of the model plus one layer, never a second full state dict (the
torch_dist save copies no weights, see :func:`unsplit_expert_fc1_save`); :func:`check_convert_size` refuses a
conversion the host's memory cannot hold, before the build. The model is Primus' own :class:`MuseSparkModel`
(``model_type: muse_spark``), so :func:`build_model` builds it through the
``muse_spark`` model provider.

The text decoder (``<prefix>*`` + ``lm_head``) is mapped. The vision tower, adapter and
projection and the audio tower are not ported: their tensors (``SKIPPED_MODULES``, by
prefix) are logged and left out of the exact-mapping check (``MappedStore``); every other
tensor must be consumed.

:class:`MuseSparkConverter` uses the shared helpers of ``native_convert_common`` (single-rank
init, safetensors store) and saves a ``torch_dist`` checkpoint, which Megatron reshards on load (any TP / PP / EP). The
build is single-rank (TP = PP = EP = 1, else ValueError), uses the training
layout (grouped-GEMM expert keys ``linear_fc{1,2}.weight<e>``, as ``moe_grouped_gemm:
true`` in the presets) and the training vocab padding (``vocab_size_with_padding``:
a multiple of 64 (the model rule; ``make_vocab_size_divisible_by`` is not read), independent of TP:
202048 rows, the same as the muse_spark_vocab training patch).

The text prefix is detected from the checkpoint keys (``detect_hf_prefix``):
``model.language_model.`` for ``MuseSparkForConditionalGeneration``, ``model.`` for a
text-only ``MuseSparkForCausalLM``.

Mapping (HF ``<prefix>layers.N.`` -> mcore ``decoder.layers.N.``):

==============================================  ==========================================
``input_layernorm.weight``                      ``input_layernorm.weight`` (centered)
``self_attn.{q,gate,k,v}_proj.weight``          ``self_attention.linear_qkv.weight``
                                                (per KV group: q heads, gate heads, k, v)
``self_attn.o_proj.weight``                     ``self_attention.linear_proj.weight``
``post_attention_residual_gate.gate``           ``self_attn_bda.gate``
``pre_feedforward_layernorm.weight``            ``pre_mlp_layernorm.weight`` (centered)
``mlp.gate.weight``                             ``mlp.router.weight``
``mlp.gate.e_score_correction_bias``            ``mlp.router.expert_bias``
``mlp.pre_expert_proj.weight``                  ``mlp.fc1_latent_proj.weight``
``mlp.pre_expert_norm.weight``                  ``mlp.pre_expert_norm.weight``
``mlp.experts.gate_up_proj[e]``                 expert ``e`` ``linear_fc1`` ([gate; up])
``mlp.experts.down_proj[e]``                    expert ``e`` ``linear_fc2``
``mlp.experts.post_expert_norm.weight``         ``mlp.post_expert_norm.weight``
``mlp.post_expert_proj.weight``                 ``mlp.fc2_latent_proj.weight``
``post_feedforward_layernorm.weight``           ``mlp_bda.post_norm.weight``
``post_feedforward_residual_gate.gate``         ``mlp_bda.gate``
==============================================  ==========================================

The post-attention norm, QK-norm, ``o_norm`` and the embedding norm have no weights.
"""

import argparse
import contextlib
import copy
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.checkpoint import native_registry as registry
from primus.backends.megatron.checkpoint.native_loader_qwen3 import (
    _detect_expert_key_style,
)
from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    HF_OUTPUT_MULTIPLIER,
    MAKE_VOCAB_SIZE_DIVISIBLE_BY,
    normalize_layer_pattern,
    vocab_size_with_padding,
)
from primus.backends.megatron.core.models.muse_spark.memory_check import (
    DEFAULT_VOCAB_SIZE,
    GiB,
    gpus_per_node,
    param_parts,
    param_split,
    required_state_bytes_or_raise,
    skip_requested,
)
from primus.backends.megatron.core.models.muse_spark.number_utils import (
    require_positive_int,
)
from primus.core.utils.arg_utils import parse_cli_overrides

# the iteration stored inside the converted checkpoint (build_margs sets margs.iteration to
# it); the torch_dist save goes to <save_dir>/release, not to an iter_* directory
CONVERT_ITERATION = 1

# HF text-decoder key prefixes, most specific first: the conditional-generation wrapper
# (text decoder with vision / audio towers) and a text-only MuseSparkForCausalLM.
HF_PREFIXES = ("model.language_model.", "model.")
# Same values as the MuseSparkTransformerConfig field defaults (kept import-free here:
# derive_megatron_fields runs on the CPU without Megatron on the path).
_CFG_DEFAULTS = {
    "qk_scale_factor": 5.4730064863875,
    "final_logit_softcapping": 20.0,
    "post_norm_eps": 1e-8,
    "residual_gate_temperature": 0.3,
    # HF config defaults (Megatron args, not MuseSparkTransformerConfig fields)
    "rope_theta": 500000.0,
    "max_position_embeddings": 131072,
}
# Megatron keys that must stay fp32 in a bf16 checkpoint (HF _keep_in_fp32_modules_strict:
# the residual gates, the router weight and its score-correction bias). convert() runs
# check_fp32_tensors on the built model's state dict: stream_into_model copies (copy_) the
# HF weights into the model's own tensors and keeps their dtype, so that is the saved dtype.
FP32_SUFFIXES = ("_bda.gate", "mlp.router.weight", "mlp.router.expert_bias")


# the converter's log prefix (MuseSparkConverter.family = "muse-spark")
LOG_TAG = "[muse-spark-convert]"


def check_fp32_tensors(sd, num_layers):
    """Raise when an ``FP32_SUFFIXES`` tensor is not fp32, or one is missing for a layer.

    Run it on ``model.state_dict()`` after ``model.to(params_dtype)``: a bf16 router weight,
    expert bias or gate there would be saved as bf16 (``stream_into_model``'s ``copy_`` casts
    the fp32 source into it silently)."""
    found = 0
    for k, v in sd.items():
        if k.endswith(FP32_SUFFIXES):
            if v.dtype != torch.float32:
                raise RuntimeError(f"{k} must stay fp32 (HF keeps it fp32), got {v.dtype}")
            found += 1
    expected = 4 * num_layers  # self_attn_bda.gate, mlp_bda.gate, router weight, expert_bias
    if found != expected:
        raise RuntimeError(
            f"expected {expected} fp32 tensors ({FP32_SUFFIXES}) in the model state dict, found {found}"
        )


def text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def cut_to_layers(cfg, num_layers):
    """``cfg`` (an HF config dict) cut to its first ``num_layers`` decoder layers; ``cfg``
    itself when ``num_layers`` is None or the full depth.

    Layer i of the cut is the released layer i, so its weights are the released tensors
    (a fine-tune with ``--num_layers 4`` trains such a cut). ``num_hidden_layers`` and
    every per-layer list of the text config (``layer_types``, ``layer_rope_theta``,
    ``mlp_layer_types``, ...: a list with one entry per layer) are cut; nothing else changes.
    """
    if num_layers is None:
        return cfg
    t = text_config(cfg)
    total = int(t["num_hidden_layers"])
    n = int(num_layers)
    if not 1 <= n <= total:
        raise ValueError(f"num_layers={num_layers!r}: the checkpoint has {total} layers (use 1..{total})")
    if n == total:
        return cfg
    cut = {k: (v[:n] if isinstance(v, list) and len(v) == total else v) for k, v in t.items()}
    cut["num_hidden_layers"] = n
    return dict(cfg, text_config=cut) if "text_config" in cfg else cut


_LAYER_KEY = re.compile(r"layers\.(\d+)\.")
# HF text_config.layer_types values: a sliding-window layer or a full (causal) layer
_LAYER_TYPES = frozenset({"sliding_attention", "full_attention"})


def kept_by_depth_cut(name, prefix, num_layers):
    """Whether a depth cut to ``num_layers`` keeps the tensor ``name``: False (dropped) for a
    text-decoder layer tensor at or past ``num_layers`` (``<prefix>layers.N.``); True (kept)
    for every other tensor (layers below ``num_layers``, embeddings, final norm, head, audio /
    vision towers) and for every tensor when ``num_layers`` is None."""
    if num_layers is None or not name.startswith(prefix + "layers."):
        return True
    m = _LAYER_KEY.match(name, len(prefix))
    return m is None or int(m.group(1)) < int(num_layers)


# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
def derive_megatron_fields(cfg):
    """Translate a Muse Spark HF config dict (top level or ``text_config``) into Megatron arg values."""
    t = text_config(cfg)
    rope = t.get("rope_parameters") or {}
    n = t["num_hidden_layers"]
    layer_types = t.get("layer_types") or ["full_attention"] * n
    rotary_base = rope.get("rope_theta", t.get("rope_theta", _CFG_DEFAULTS["rope_theta"]))
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type != "default" or t.get("rope_scaling"):
        raise ValueError(
            f"Muse Spark: only default RoPE is supported, got rope_type={rope_type!r} "
            f"rope_scaling={t.get('rope_scaling')!r}"
        )
    rope_theta = t.get("layer_rope_theta") or [rotary_base] * n
    if len(rope_theta) != n:
        raise ValueError(f"layer_rope_theta has {len(rope_theta)} entries for {n} layers")
    base_theta = float(rotary_base)
    other = sorted({float(th) for th in rope_theta if th and float(th) != base_theta})
    if other:
        # Megatron has one rotary_base; a layer is either RoPE with it or NoPE (theta 0).
        raise ValueError(f"layer_rope_theta has values {other} other than 0 and rope_theta={rotary_base}")
    if len(layer_types) != n:
        raise ValueError(f"layer_types has {len(layer_types)} entries for {n} layers")
    unknown = sorted(set(layer_types) - _LAYER_TYPES)
    if unknown:
        raise ValueError(f"layer_types has unsupported values {unknown} (supported: {sorted(_LAYER_TYPES)})")
    sliding_window = t.get("sliding_window")
    if "sliding_attention" in layer_types and (
        isinstance(sliding_window, bool) or not isinstance(sliding_window, int) or sliding_window <= 0
    ):
        # window_size None with sliding layers would run them as full attention, unlike HF
        raise ValueError(
            "Muse Spark: layer_types has sliding_attention layers, so sliding_window must be a "
            f"positive integer; got sliding_window={sliding_window!r}"
        )
    mlp_types = t.get("mlp_layer_types") or ["sparse"] * n
    if len(mlp_types) != n:
        raise ValueError(f"mlp_layer_types has {len(mlp_types)} entries for {n} layers")
    if not all(x == "sparse" for x in mlp_types):
        raise ValueError("dense Muse Spark layers are not supported (mlp_layer_types must all be 'sparse')")
    if t.get("num_shared_experts"):
        raise ValueError("Muse Spark shared experts are not supported (num_shared_experts must be 0)")
    if t.get("attention_bias") or t.get("mlp_bias"):
        # the key mapping has no q / gate / k / v / o (or MLP) bias entries: reject up front,
        # before a full model is built, rather than fail late on unconsumed HF tensors
        raise ValueError(
            "Muse Spark: attention_bias / mlp_bias checkpoints are not supported (the released "
            f"model has no linear biases); got attention_bias={t.get('attention_bias')!r} "
            f"mlp_bias={t.get('mlp_bias')!r}"
        )
    output_multiplier = t.get("output_multiplier", HF_OUTPUT_MULTIPLIER)
    return dict(
        hidden_size=t["hidden_size"],
        num_attention_heads=t["num_attention_heads"],
        num_layers=n,
        max_position_embeddings=t.get("max_position_embeddings", _CFG_DEFAULTS["max_position_embeddings"]),
        norm_epsilon=t["rms_norm_eps"],
        rotary_base=rotary_base,
        untie_embeddings_and_output_weights=not tied_word_embeddings(cfg),
        ffn_hidden_size=t.get("intermediate_size", t["moe_intermediate_size"]),
        num_query_groups=t["num_key_value_heads"],
        kv_channels=t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
        add_qkv_bias=False,  # attention_bias is rejected above
        qk_layernorm=True,
        num_experts=t["num_local_experts"],
        moe_router_topk=t["num_experts_per_tok"],
        moe_ffn_hidden_size=t["moe_intermediate_size"],
        moe_latent_size=t["moe_hidden_size"],
        moe_router_topk_scaling_factor=t.get("routed_scaling_factor", 1.0),
        # per-layer patterns (1-indexed in Megatron, listed here 0-indexed by layer)
        window_size=((int(sliding_window), 0) if sliding_window else None),  # HF count; config -> TE
        window_attn_skip_freq=[1 if x == "sliding_attention" else 0 for x in layer_types],
        no_rope_freq=[1 if not th else 0 for th in rope_theta],
        # Muse Spark fields (MuseSparkTransformerConfig)
        post_norm_eps=t.get("post_norm_eps", _CFG_DEFAULTS["post_norm_eps"]),
        residual_gate_temperature=t.get(
            "residual_gate_temperature", _CFG_DEFAULTS["residual_gate_temperature"]
        ),
        # Defaults are MuseSparkTransformerConfig's (the released values), not 1.0.
        qk_scale_factor=t.get("qk_scale_factor", _CFG_DEFAULTS["qk_scale_factor"]),
        # HF has one ``output_multiplier`` and applies it twice: to the router logits
        # (MuseSparkTopKRouter.gating) and to the LM-head logits (before the soft-cap).
        moe_router_output_multiplier=output_multiplier,
        output_logit_multiplier=output_multiplier,
        final_logit_softcapping=t.get("final_logit_softcapping", _CFG_DEFAULTS["final_logit_softcapping"]),
    )


MUSE_FIELDS = (
    "moe_latent_size",
    "window_size",
    "window_attn_skip_freq",
    "no_rope_freq",
    "post_norm_eps",
    "residual_gate_temperature",
    "qk_scale_factor",
    "moe_router_output_multiplier",
    "output_logit_multiplier",
    "final_logit_softcapping",
)


def _per_layer(value, num_layers):
    """Cut a per-layer pattern list down to ``num_layers`` (depth-reduced builds); shared parser."""
    if isinstance(value, (list, str)):
        return normalize_layer_pattern(value, num_layers)
    return value


def tied_word_embeddings(cfg):
    """``tie_word_embeddings`` from the loaded HF config: ``text_config``'s, else the top
    level's (a composite ``*ForConditionalGeneration`` config may keep it there), else False
    (the released Muse Spark configs set it false; the converter then requires
    ``lm_head.weight``, so a missing head fails the exact-mapping check instead of loading
    wrongly)."""
    cfg = cfg or {}
    for source in (text_config(cfg), cfg):
        if source.get("tie_word_embeddings") is not None:
            return bool(source["tie_word_embeddings"])
    return False


def checkpoint_vocab(cfg):
    """The checkpoint's ``vocab_size`` from the loaded HF config: ``text_config.vocab_size``,
    else the top-level ``vocab_size`` (the lookup :func:`build_margs`'s error names), else
    None."""
    cfg = cfg or {}
    # text_config.vocab_size, else the top level (a composite config may keep it there)
    v = text_config(cfg).get("vocab_size") or cfg.get("vocab_size")
    return int(v) if v else None


def estimate_num_params(f, vocab_size=DEFAULT_VOCAB_SIZE):
    """Muse Spark text-decoder parameter count from derive_megatron_fields output.

    The same count as the training memory check (``memory_check.param_split``), so the
    converter cap and the memory check agree on a model. Muse Spark always has the
    attention output gate.
    """
    return sum(param_split(_param_args(f), vocab_size))


def _param_args(f):
    """The args namespace of the parameter count (Muse Spark always has the output gate)."""
    return SimpleNamespace(**f, attention_output_gate=True, mtp_num_layers=0)


# The sequence length of the single-process build: it sizes only the rotary tables; the
# saved weights do not depend on it.
DEFAULT_CONVERT_SEQ_LENGTH = 4096

# The converter holds one copy of the model in host RAM: the weights are mapped into the built
# model one decoder layer at a time (:func:`stream_into_model`), never as a second full state
# dict, and the save copies nothing (:func:`unsplit_expert_fc1_save`).
# :func:`check_convert_size` refuses a conversion that does not fit in the host's available
# memory (:func:`conversion_memory_plan`), before anything is built.
# The config key the posttrain hook passes to ``convert(max_params=...)``: an optional cap on
# the estimated parameter count (unset: no cap; the memory check alone decides).
MAX_PARAMS_KEY = "native_ckpt_max_params"
# headroom over the plan: Python / TE objects, the safetensors reader, allocator slack
MEMORY_MARGIN = 1.05
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
# cgroup v1 reports "no limit" as a huge page-aligned value (~2**63): a limit at or above this
# is not a limit
_CGROUP_V1_UNLIMITED = 1 << 60


def conversion_memory_plan(fields, vocab_size=DEFAULT_VOCAB_SIZE, dtype="bf16"):
    """Host bytes a conversion needs at its peak, as a dict:

    * ``params``: the estimated text-decoder parameter count;
    * ``model``: the built model (every text-decoder parameter once, in ``dtype``; the fp32
      islands are counted at ``dtype``: they are a few MB);
    * ``layer``: one decoder layer as read from the HF files (the mapping chunk);
    * ``total``: their sum times ``MEMORY_MARGIN``.

    The save adds no copy: :func:`unsplit_expert_fc1_save` keeps the torch_dist writer from
    cloning the expert weights."""
    width = _DTYPE_BYTES[dtype]  # an unsupported dtype is refused by check_convert_args
    n = estimate_num_params(fields, vocab_size)  # the same count as the converter cap
    parts = param_parts(_param_args(fields), vocab_size)
    per_layer = parts["layer_dense"] + parts["layer_expert"]
    plan = {"params": n, "model": n * width, "layer": per_layer * width}
    plan["total"] = int((plan["model"] + plan["layer"]) * MEMORY_MARGIN)
    return plan


def _cgroup_dirs(proc_cgroup, root):
    """Directories to check for a memory limit, from this process's own cgroup up to the
    root, read from ``/proc/self/cgroup``: the v2 entry (``0::<path>``) under ``root``, and
    the v1 ``memory`` controller's path under ``<root>/memory``. A cgroup namespace gives
    ``/``; a job cgroup without one (Slurm / systemd) gives its full path."""
    dirs = []
    for line in proc_cgroup.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hier, controllers, path = parts
        if hier == "0" and controllers == "":
            base = root
        elif "memory" in controllers.split(","):
            base = os.path.join(root, "memory")
        else:
            continue
        path = path.strip().strip("/")
        while True:
            d = os.path.join(base, path) if path else base
            if d not in dirs:
                dirs.append(d)
            if not path:
                break
            path = os.path.dirname(path)
    for d in (root, os.path.join(root, "memory")):  # no /proc/self/cgroup: the roots
        if d not in dirs:
            dirs.append(d)
    return dirs


def _cgroup_free(d):
    """Free bytes under the memory limit set in cgroup directory ``d`` (limit - usage + its
    inactive page cache), or None when ``d`` sets no limit (v2 ``memory.max``, v1
    ``memory.limit_in_bytes``)."""
    for limit_name, usage_name in (
        ("memory.max", "memory.current"),
        ("memory.limit_in_bytes", "memory.usage_in_bytes"),
    ):
        limit = _read_text(os.path.join(d, limit_name), "")
        usage = _read_text(os.path.join(d, usage_name), "")
        if limit.isdigit() and usage.isdigit() and int(limit) < _CGROUP_V1_UNLIMITED:
            # the cgroup's usage counts the page cache; its inactive part is reclaimable
            stat = _read_text(os.path.join(d, "memory.stat"), "")
            cache = 0
            for line in stat.splitlines():
                key, _, value = line.partition(" ")
                if key in ("inactive_file", "total_inactive_file") and value.strip().isdigit():
                    cache = int(value)
            return max(0, int(limit) - int(usage) + cache)
    return None


def host_memory_available(root="/sys/fs/cgroup", proc="/proc"):
    """Bytes this process can still allocate on the host: ``MemAvailable`` from
    ``/proc/meminfo``, lowered to the smallest free memory under a cgroup memory limit on
    this process's cgroup path (from ``/proc/self/cgroup``, its own cgroup up to the root;
    v2 ``memory.max`` - ``memory.current``, v1 ``limit_in_bytes`` - ``usage_in_bytes``).
    A Slurm / systemd job cgroup sets its limit on that path, not on the root. None when
    nothing can be read (not Linux)."""
    found = []
    meminfo = _read_text(os.path.join(proc, "meminfo"), "")
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            found.append(int(line.split()[1]) * 1024)
    for d in _cgroup_dirs(_read_text(os.path.join(proc, "self", "cgroup"), ""), root):
        free = _cgroup_free(d)
        if free is not None:
            found.append(free)
    return min(found) if found else None


def check_convert_size(fields, cfg=None, max_params=None, dtype="bf16", available=None):
    """Refuse a conversion that cannot fit (only ``convert`` calls this); return the
    parameter estimate.

    * ``max_params`` (the ``convert`` argument; the posttrain hook passes the config key
      ``native_ckpt_max_params``): an optional cap on the estimated parameter count, a
      positive int or a string of one (``"60e9"``, ``"820_000_000_000"``); None or blank:
      no cap.
    * host memory: :func:`conversion_memory_plan` against ``available`` (default
      :func:`host_memory_available`; not checked when that is unknown). The released
      model (822B parameters) needs ~1.5 TiB (1.6 TB) in bf16 (one model copy plus one layer).
    """
    vocab = checkpoint_vocab(cfg) or DEFAULT_VOCAB_SIZE
    plan = conversion_memory_plan(fields, vocab, dtype)
    n = plan["params"]
    if max_params is not None and str(max_params).strip():
        limit = require_positive_int(
            max_params, f"{MAX_PARAMS_KEY}={max_params!r} must be a positive integer parameter count"
        )
        if n > limit:
            raise ValueError(
                f"Muse Spark native converter: ~{n:,} params (~{n / 1e9:.3g}B) is above "
                f"{MAX_PARAMS_KEY}={limit:,}; raise or unset it"
            )
    if available is None:
        available = host_memory_available()
    if available is not None and plan["total"] > available:
        raise ValueError(
            f"Muse Spark native converter: ~{n / 1e9:.3g}B params need ~{common._gib(plan['total'])} of host "
            f"memory in {dtype} (the model {common._gib(plan['model'])}, one layer {common._gib(plan['layer'])}, "
            f"x{MEMORY_MARGIN}); "
            f"{common._gib(available)} is available. Convert on a host with more memory and set "
            "pretrained_checkpoint to the result."
        )
    return n


# the save dtypes the converter accepts; any other value is an error, not a silent fp32
SUPPORTED_DTYPES = tuple(_DTYPE_BYTES)


# the converter's save: torch_dist, synchronous, weights only (build_margs and
# prepare_torch_dist_save both apply it)
_TORCH_DIST_SAVE = {
    "ckpt_format": "torch_dist",
    "async_save": False,
    "no_save_optim": True,
    "no_save_rng": True,
}


def _hf_cfg_of(args, hf_cfg=None):
    """``hf_cfg`` when given, else ``<args.hf_dir>/config.json`` (None without an hf_dir)."""
    if hf_cfg is None and getattr(args, "hf_dir", None):
        hf_cfg = common.read_hf_config(args.hf_dir)
    return hf_cfg


def check_convert_args(args, hf_cfg):
    """The import-free checks of :func:`build_margs` (``convert`` runs them before it claims
    the output directory): TP = PP = EP = 1, a supported dtype, and a vocab_size in the HF
    config. Returns the vocab_size."""
    tp, pp, ep = (
        int(getattr(args, k, 1) or 1)
        for k in ("tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size")
    )
    if (tp, pp, ep) != (1, 1, 1):
        # iter_mcore_state_dict (used by stream_into_model) maps full (unsharded) tensors;
        # it does not split them.
        raise ValueError(
            f"Muse Spark native converter builds with TP=PP=EP=1 only (got {tp}/{pp}/{ep}). Its "
            "torch_dist checkpoint loads into any TP / PP / EP (Megatron reshards on load), so "
            "convert with 1/1/1."
        )

    # checked before any Megatron import: fail fast, never a silent fp32
    if args.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Muse Spark native conversion: dtype={args.dtype!r} is not supported "
            f"(use one of {', '.join(SUPPORTED_DTYPES)})"
        )

    # checked before any Megatron import too
    vocab_size = checkpoint_vocab(hf_cfg)
    if not vocab_size:
        # Without it Megatron would size the head from the tokenizer (and its own 128*TP
        # padding): a checkpoint whose rows differ from training's, with no error.
        src = "the HF config" if hf_cfg is not None else "no HF config (args.hf_dir is unset)"
        raise ValueError(
            f"Muse Spark native conversion: {src} has no vocab_size (text_config.vocab_size or the "
            "top level); the checkpoint's vocab sizes the embedding and output layer, so it is required"
        )
    return vocab_size


def build_margs(cfg_fields, args, hf_cfg=None):
    """Megatron args for a single-process build.

    ``hf_cfg`` is the loaded HF config.json (``convert()`` passes the one it read). Callers
    of the two-argument form (e.g. the logprob-parity harness) get it read from
    ``args.hf_dir``, so the checkpoint's vocab_size (not the tokenizer's) sizes the head
    on every path.
    """
    hf_cfg = _hf_cfg_of(args, hf_cfg)
    vocab_size = check_convert_args(args, hf_cfg)
    tp = pp = ep = 1  # check_convert_args accepts only TP = PP = EP = 1

    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    m.model_type = "muse_spark"
    m.hidden_size = cfg_fields["hidden_size"]
    m.num_attention_heads = cfg_fields["num_attention_heads"]
    m.num_layers = cfg_fields["num_layers"]
    m.norm_epsilon = cfg_fields["norm_epsilon"]
    m.ffn_hidden_size = cfg_fields["ffn_hidden_size"]
    m.untie_embeddings_and_output_weights = cfg_fields["untie_embeddings_and_output_weights"]
    m.max_position_embeddings = max(cfg_fields["max_position_embeddings"], args.seq_length)
    m.seq_length = args.seq_length

    # attention: GQA, gated output, weightless QK-norm (spec), sliding window + NoPE pattern
    m.group_query_attention = True
    m.num_query_groups = cfg_fields["num_query_groups"]
    m.kv_channels = cfg_fields["kv_channels"]
    m.qk_layernorm = cfg_fields["qk_layernorm"]
    m.attention_output_gate = True
    m.add_qkv_bias = cfg_fields["add_qkv_bias"]
    m.position_embedding_type = "rope"
    m.rotary_base = cfg_fields["rotary_base"]
    m.rotary_percent = 1.0
    m.rotary_interleaved = False
    m.apply_rope_fusion = False

    m.normalization = "RMSNorm"
    # HF has no dropout (and the embedding norm runs after Megatron's embedding dropout)
    m.hidden_dropout = 0.0
    m.attention_dropout = 0.0
    m.swiglu = True
    m.add_bias_linear = False

    # MoE: sigmoid router with selection bias, latent experts
    m.num_experts = cfg_fields["num_experts"]
    m.moe_router_topk = cfg_fields["moe_router_topk"]
    m.moe_ffn_hidden_size = cfg_fields["moe_ffn_hidden_size"]
    m.moe_router_score_function = "sigmoid"
    m.moe_router_enable_expert_bias = True
    m.moe_router_pre_softmax = False
    m.moe_router_topk_scaling_factor = cfg_fields["moe_router_topk_scaling_factor"]
    m.moe_router_load_balancing_type = "none"
    m.moe_aux_loss_coeff = 0.0
    m.moe_router_dtype = "fp32"
    m.moe_token_dispatcher_type = "alltoall"
    # convert() saves with grouped-GEMM experts (the training layout, moe_grouped_gemm: true).
    # Without the arg (e.g. the logprob-parity harness), SequentialMLP experts keep exact fp32 math.
    m.moe_grouped_gemm = bool(getattr(args, "moe_grouped_gemm", False))
    m.moe_permute_fusion = False

    # the Muse fields (per-layer patterns expanded, scalars as is), built once: set here and set again after validate_args
    muse_fields = {k: _per_layer(cfg_fields.get(k), m.num_layers) for k in MUSE_FIELDS}
    for k, v in muse_fields.items():
        setattr(m, k, v)

    # The checkpoint's vocab_size (text tower). padded_vocab_size is set here, so
    # build_tokenizer (init_megatron_single_process) keeps it: the training run sets the
    # same rows from the preset's vocab_size (the muse_spark_vocab patch), and the
    # torch_dist save reshards to any TP / PP / EP on load.
    m.vocab_size = vocab_size
    # The model's multiple (64), as the muse_spark_vocab training patch: the args'
    # make_vocab_size_divisible_by (the module config's 128) is not read, so the converted
    # rows (202048) are the rows training builds at any TP.
    m.make_vocab_size_divisible_by = MAKE_VOCAB_SIZE_DIVISIBLE_BY
    m.padded_vocab_size = vocab_size_with_padding(m.vocab_size, m.make_vocab_size_divisible_by, tp)
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False

    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1

    # the validated sizes (all 1, checked above), not the raw args again
    m.tensor_model_parallel_size = tp
    m.pipeline_model_parallel_size = pp
    m.expert_model_parallel_size = ep
    m.expert_tensor_parallel_size = 1
    m.context_parallel_size = 1
    m.sequence_parallel = False
    m.world_size = 1  # one rank: TP = PP = EP = 1
    m.rank = 0
    if args.dtype == "bf16":
        m.bf16 = True
        m.params_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        m.fp16 = True
        m.params_dtype = torch.float16
    else:  # "fp32"
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
    vars(m).update(_TORCH_DIST_SAVE)  # load_validate_save's save follows ckpt_format
    m.no_load_optim = True
    m.no_load_rng = True
    m.save_interval = 1
    m.ckpt_assume_constant_structure = False
    m.iteration = CONVERT_ITERATION

    m = validate_args(m)
    # validate_args may rebuild list-valued fields from strings; restore ours.
    for k, v in muse_fields.items():
        setattr(m, k, v)
    m.iteration = CONVERT_ITERATION
    return m


def build_model(margs):
    """Build :class:`MuseSparkModel` with the shared ``native_convert_common.build_mcore_gpt_model``
    and the model_type this family registers (``muse_spark``). Kept as the converter's
    ``build_model(margs)`` entry point (the logprob-parity harness calls it)."""
    model = common.build_mcore_gpt_model(margs, family="muse_spark")
    # .to(dtype) casts every param; the router weight and the residual gates stay fp32
    # (MuseSparkTopKRouter._apply restores the router; the gates restore themselves too).
    return model


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping
# ---------------------------------------------------------------------------
def iter_mcore_state_dict(model, store, margs, cfg_fields, log):
    """The HF -> mcore mapping in chunks: yields ``(chunk, consumed)`` pairs (``chunk``: a
    dict of mcore tensors; ``consumed``: the shared set of HF names read so far), the
    embedding / final norm / head first, then one per decoder layer.

    Each chunk holds only its own tensors, so :func:`stream_into_model` copies a chunk into
    the built model and frees it before it reads the next (host RAM: one model copy plus one layer).
    ``consumed`` is a set shared by every chunk: the HF names read so far."""
    dtype = margs.params_dtype
    padded_vocab = model.embedding.word_embeddings.weight.shape[0]
    nh, ng, hd, hidden = (
        margs.num_attention_heads,
        margs.num_query_groups,
        margs.kv_channels,
        margs.hidden_size,
    )
    if nh % ng:
        raise ValueError(f"num_attention_heads={nh} is not a multiple of num_query_groups={ng}")
    num_experts = cfg_fields["num_experts"]
    style = _detect_expert_key_style(model, margs.num_layers)  # the qwen3 loader's detector
    if style is None:
        raise RuntimeError("could not detect the MoE expert key layout on the built model")

    prefix = getattr(store, "prefix", None) or detect_hf_prefix(store.all_keys())
    if style == "grouped":
        fc1_key, fc2_key = "{p}mlp.experts.linear_fc1.weight{e}", "{p}mlp.experts.linear_fc2.weight{e}"
    else:
        fc1_key = "{p}mlp.experts.local_experts.{e}.linear_fc1.weight"
        fc2_key = "{p}mlp.experts.local_experts.{e}.linear_fc2.weight"
    consumed = set()
    q_rows = (nh // ng) * hd  # query (and output-gate) rows per KV group

    def get(name, keep_fp32=False):
        consumed.add(name)
        return store.get(name, torch.float32 if keep_fp32 else dtype)

    def gated_qkv(h):
        q = get(h + "self_attn.q_proj.weight").reshape(ng, q_rows, hidden)
        g = get(h + "self_attn.gate_proj.weight").reshape(ng, q_rows, hidden)
        k = get(h + "self_attn.k_proj.weight").reshape(ng, hd, hidden)
        v = get(h + "self_attn.v_proj.weight").reshape(ng, hd, hidden)
        return torch.cat([q, g, k, v], dim=1).reshape(-1, hidden).contiguous()

    sd = {}
    sd["embedding.word_embeddings.weight"] = common.pad_vocab(
        get(prefix + "embed_tokens.weight"), padded_vocab
    )
    sd["decoder.final_layernorm.weight"] = get(prefix + "norm.weight")
    if margs.untie_embeddings_and_output_weights:
        sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)
    yield sd, consumed

    for L in range(margs.num_layers):
        t0 = time.monotonic()
        sd = {}
        p = f"decoder.layers.{L}."
        h = f"{prefix}layers.{L}."
        sd[p + "input_layernorm.weight"] = get(h + "input_layernorm.weight")
        sd[p + "self_attention.linear_qkv.weight"] = gated_qkv(h)
        sd[p + "self_attention.linear_proj.weight"] = get(h + "self_attn.o_proj.weight")
        sd[p + "self_attn_bda.gate"] = get(h + "post_attention_residual_gate.gate", keep_fp32=True)
        sd[p + "pre_mlp_layernorm.weight"] = get(h + "pre_feedforward_layernorm.weight")
        sd[p + "mlp.router.weight"] = get(h + "mlp.gate.weight", keep_fp32=True)
        sd[p + "mlp.router.expert_bias"] = get(h + "mlp.gate.e_score_correction_bias", keep_fp32=True)
        sd[p + "mlp.fc1_latent_proj.weight"] = get(h + "mlp.pre_expert_proj.weight")
        sd[p + "mlp.pre_expert_norm.weight"] = get(h + "mlp.pre_expert_norm.weight")
        sd[p + "mlp.post_expert_norm.weight"] = get(h + "mlp.experts.post_expert_norm.weight")
        sd[p + "mlp.fc2_latent_proj.weight"] = get(h + "mlp.post_expert_proj.weight")
        sd[p + "mlp_bda.post_norm.weight"] = get(h + "post_feedforward_layernorm.weight")
        sd[p + "mlp_bda.gate"] = get(h + "post_feedforward_residual_gate.gate", keep_fp32=True)

        gate_up = get(h + "mlp.experts.gate_up_proj")  # [E, 2I, latent]: [gate; up]
        down = get(h + "mlp.experts.down_proj")  # [E, latent, I]
        if gate_up.shape[0] != num_experts or down.shape[0] != num_experts:
            raise ValueError(
                f"{h}mlp.experts: gate_up_proj / down_proj have {gate_up.shape[0]} / {down.shape[0]} "
                f"experts, the config has {num_experts}"
            )
        for e in range(num_experts):
            # views of gate_up / down (no copy): the chunk holds one layer's experts once
            sd[fc1_key.format(p=p, e=e)] = gate_up[e]
            sd[fc2_key.format(p=p, e=e)] = down[e]
        del gate_up, down
        yield sd, consumed
        log(f"layer {L} mapped ({style} experts) in {time.monotonic() - t0:.1f} s")


def build_mcore_state_dict(model, store, margs, cfg_fields, log):
    """The whole mapping as one state dict (the logprob-parity harness and small models):
    every chunk of :func:`iter_mcore_state_dict`. Returns ``(sd, consumed)``."""
    sd = {}
    # iter_mcore_state_dict always yields the embedding / head chunk, so ``consumed`` is bound
    for chunk, consumed in iter_mcore_state_dict(model, store, margs, cfg_fields, log):
        sd.update(chunk)
    return sd, consumed


def stream_into_model(model, store, margs, cfg_fields, log):
    """Copy the mapping into ``model``'s own tensors chunk by chunk (one decoder layer at a
    time) and free each chunk: host RAM holds one model copy plus one layer, not a second
    full state dict (the released model, 822B parameters, is ~1.5 TiB (1.6 TB) in bf16).

    Raises on a key the model does not have or a shape that differs. Returns
    ``(sd, consumed)``: ``sd`` maps every assigned key to the model's own tensor (no copy),
    so ``native_convert_common.load_validate_save`` still checks the mapping is exact
    (a model key never assigned is reported missing) and saves."""
    own = model.state_dict(keep_vars=True)
    assigned = set()
    with torch.no_grad():
        for chunk, consumed in iter_mcore_state_dict(model, store, margs, cfg_fields, log):
            unexpected = sorted(k for k in chunk if k not in own)
            if unexpected:
                raise RuntimeError(
                    f"native conversion mapping incorrect: {len(unexpected)} keys the model does not "
                    f"have (first: {unexpected[:8]}); refusing to save."
                )
            for k, v in chunk.items():
                dst = own[k]
                if tuple(dst.shape) != tuple(v.shape):
                    raise RuntimeError(f"{k}: HF tensor shape {tuple(v.shape)} != model {tuple(dst.shape)}")
                dst.copy_(v)  # casts into the model's dtype (the fp32 islands stay fp32)
                assigned.add(k)
            del chunk
    return {k: own[k].detach() for k in assigned}, consumed


def detect_hf_prefix(keys):
    """The text-decoder key prefix of an HF Muse Spark checkpoint (one of ``HF_PREFIXES``),
    found from ``<prefix>embed_tokens.weight``; raise on an unknown layout."""
    keys = set(keys)
    for prefix in HF_PREFIXES:
        if f"{prefix}embed_tokens.weight" in keys:
            return prefix
    found = sorted({k.split(".", 1)[0] + "." for k in keys if "." in k})
    found += sorted({k[: k.index("embed_tokens.")] or "<none>" for k in keys if "embed_tokens." in k})
    raise ValueError(
        f"unsupported Muse Spark checkpoint layout: no embed_tokens.weight under any of {HF_PREFIXES}; "
        f"found key prefixes {found}"
    )


# The HF modules this port does not map (the vision tower, adapter and projection, and the
# audio tower; not ported), next to the text prefix ``model.language_model.`` (the released
# checkpoint's names). Only these are set aside; any other tensor the mapping does not
# consume fails the exact-mapping check.
SKIPPED_MODULES = ("vision_tower.", "vision_adapter.", "vision_projection.", "audio_tower.")


def out_of_scope_prefixes(text_prefix):
    """The HF key prefixes set aside from the exact-mapping check (see ``SKIPPED_MODULES``);
    none for the text-only ``model.`` layout (it has no towers)."""
    if "language_model." not in text_prefix:
        return ()
    return tuple(text_prefix.replace("language_model.", m, 1) for m in SKIPPED_MODULES)


# safetensors dtype names -> torch dtypes (the parallel reader handles these; any other
# dtype falls back to the store's own reader)
_ST_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
READ_CHUNK_BYTES = 64 << 20  # one pread per chunk: large sequential reads suit Lustre / NFS
READ_THREADS = 16  # concurrent chunk reads of one tensor


class ParallelSafetensorsReader:
    """Read one tensor of a ``.safetensors`` file with concurrent ``os.preadv`` calls of
    ``READ_CHUNK_BYTES`` each, straight into the tensor's buffer.

    ``safe_open(...).get_tensor`` reads through an mmap, one page fault at a time from one
    thread, which is slow on a parallel file system; many large concurrent reads use its
    bandwidth. The bytes are the file's, so the
    tensor is the same as ``get_tensor``'s (``test_parallel_reader_matches_safe_open``)."""

    def __init__(self, threads=READ_THREADS, chunk=READ_CHUNK_BYTES):
        self.threads, self.chunk = max(1, int(threads)), max(1, int(chunk))
        self._headers = {}  # path -> (byte offset of the data, header dict)
        self._pool = None

    def _header(self, path):
        h = self._headers.get(path)
        if h is None:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                h = (8 + n, json.loads(f.read(n)))
            self._headers[path] = h
        return h

    def read(self, path, name):
        """The tensor ``name`` of the safetensors file ``path``, or None when its dtype is
        not in ``_ST_DTYPES`` (the caller then uses the store's reader)."""
        start, header = self._header(path)
        meta = header[name]
        dtype = _ST_DTYPES.get(meta["dtype"])
        if dtype is None:
            return None
        begin, end = meta["data_offsets"]
        n = end - begin
        buf = torch.empty(n, dtype=torch.uint8)
        if n:
            view = memoryview(buf.numpy())
            fd = os.open(path, os.O_RDONLY)
            try:

                def read_part(off):
                    stop, pos = min(off + self.chunk, n), off
                    while pos < stop:
                        got = os.preadv(fd, [view[pos:stop]], start + begin + pos)
                        if got <= 0:
                            raise OSError(f"{path}: short read of {name} at byte {pos} of {n}")
                        pos += got

                parts = range(0, n, self.chunk)
                if len(parts) == 1:
                    read_part(0)
                else:
                    if self._pool is None:
                        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="muse-spark-read")
                    # wait for every chunk before the fd closes (a failed chunk must not
                    # close it under reads still in flight); then raise the first error
                    futures = [self._pool.submit(read_part, off) for off in parts]
                    wait(futures)
                    for fut in futures:
                        fut.result()
            finally:
                os.close(fd)
        return buf.view(dtype).reshape(meta["shape"])

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


def _tensor_file(store, name):
    """The safetensors file that holds ``name`` in a ``native_convert_common.SafetensorsStore``
    (sharded or single file), or None (a legacy ``.bin`` checkpoint)."""
    files = getattr(store, "_name_to_file", None)
    if files is not None:
        return files.get(name)
    return getattr(store, "_single", None)


class MappedStore:
    """A SafetensorsStore view whose ``all_keys`` holds the tensors the exact-mapping check
    covers: every tensor except the modules that are not ported (:func:`out_of_scope_prefixes`:
    the vision tower, adapter and projection, and the audio tower) and the layers a depth
    cut drops.

    The set-aside tensors are logged with their count and module; any other tensor (an MTP
    head, a projector, an unexpected prefix) stays in ``all_keys``, so a mapping that does
    not consume it fails the check instead of dropping it.
    """

    def __init__(self, store, log=print, num_layers=None):
        self._store = store
        self._reader = ParallelSafetensorsReader()
        keys = store.all_keys()
        self.prefix = detect_hf_prefix(keys)
        aside = out_of_scope_prefixes(self.prefix)
        # a depth cut (num_layers) maps the first num_layers decoder layers only
        # (dropped: the decoder-layer tensors at or past num_layers, logged on their own)
        dropped = {k for k in keys if not kept_by_depth_cut(k, self.prefix, num_layers)}
        self.skipped = sorted(k for k in keys if k not in dropped and k.startswith(aside))
        self.dropped = sorted(dropped)
        self._mapped_keys = frozenset(keys) - dropped - set(self.skipped)
        for module in aside:
            n = sum(k.startswith(module) for k in self.skipped)
            if n:
                log(f"not mapped (not ported, skipped): {n} tensors under {module}*")
        if self.dropped:
            log(f"not mapped (layers >= {num_layers}, dropped by the depth cut): {len(self.dropped)} tensors")

    def all_keys(self):
        # the key set is fixed: computed once in __init__ (a fresh set, callers may mutate it)
        return set(self._mapped_keys)

    def close(self):
        """Stop the reader's threads (the store stays usable: a later read restarts them)."""
        self._reader.close()

    def get(self, name, dtype):
        """``SafetensorsStore.get``, read by :class:`ParallelSafetensorsReader` when the
        tensor is in a safetensors file (the same bytes; else the store's own reader)."""
        path = _tensor_file(self._store, name)
        t = self._reader.read(path, name) if path is not None else None
        if t is None:
            return self._store.get(name, dtype)
        return t.to(dtype)

    def __getattr__(self, name):
        return getattr(self._store, name)


@contextlib.contextmanager
def unsplit_expert_fc1_save(model):
    """While the block runs, the MoE experts write their gated ``linear_fc1`` weights as one
    shard per expert instead of Megatron's SwiGLU pair of half shards.

    The half shards are two views of one storage, and the torch_dist writer copies every
    such view before it writes (for the released model, about the size of the experts on
    top of the model). One shard per expert has the same key, global
    shape and expert offset, so the global tensor on disk is the same and a training run
    (which reads it in halves, at any TP / EP) loads it as before. Only the experts' config
    is replaced (a copy with ``gated_linear_unit`` False, read by their
    ``sharded_state_dict`` only; no forward runs in the block), then restored."""
    from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

    changed = []
    experts = [m for m in model.modules() if isinstance(m, (TEGroupedMLP, SequentialMLP))]
    for module in (sub for e in experts for sub in e.modules()):  # SequentialMLP: its MLPs too
        cfg = getattr(module, "config", None)
        if getattr(cfg, "gated_linear_unit", False):
            unsplit = copy.copy(cfg)
            unsplit.gated_linear_unit = False
            changed.append((module, cfg))
            module.config = unsplit
    try:
        yield len(changed)
    finally:
        for module, cfg in changed:
            module.config = cfg


def prepare_torch_dist_save(margs, save_dir):
    """Set ``margs`` so Megatron's ``save_checkpoint`` (called by
    ``native_convert_common.load_validate_save``) writes a torch_dist checkpoint into
    ``<save_dir>/release`` (no optimizer; ``margs.iteration`` is stored inside the
    checkpoint) and a tracker naming ``release`` (the hook's skip-if-exists check reads it).
    ``save_dir`` is checked again (:func:`refuse_existing_output`; ``convert`` claimed it
    with :func:`claim_output`).

    The files hold global shapes and offsets, so Megatron reshards them for the training
    TP / PP / EP on load."""
    # the same settings as build_margs (one dict): set again here because validate_args and
    # init_megatron_single_process run between the two and may derive save settings anew
    vars(margs).update(_TORCH_DIST_SAVE, use_dist_ckpt=True, ckpt_fully_parallel_save=False, save=save_dir)
    refuse_existing_output(save_dir)  # checked again: deletes nothing


# written into save_dir by the conversion, removed after a complete save: marks the
# directory as this converter's while it is written ("<host> <pid> <boot id> <pid namespace>
# <token>"); touched every MARKER_HEARTBEAT_S while the conversion runs
IN_PROGRESS_MARKER = ".muse_spark_conversion_in_progress"
MARKER_HEARTBEAT_S = 60
# a marker whose heartbeat has not moved for this long (on the file server's clock): the
# conversion that wrote it was killed. Used when its pid cannot be checked: another host,
# another boot, or another pid namespace (a container that shares the host name)
MARKER_STALE_S = 900
TRACKER = common.TRACKER_FILENAME  # the tracker file the shared save writes
_ITER_DIR = re.compile(r"iter_(\d{7})$")


def _read_text(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _process_identity():
    """(host, pid, boot id, pid namespace) of this process: a pid is checkable with
    ``os.kill`` only by a process with the same host, boot and pid namespace."""
    try:
        pidns = os.readlink("/proc/self/ns/pid")
    except OSError:
        pidns = "-"
    boot = _read_text("/proc/sys/kernel/random/boot_id", "-") or "-"
    return socket.gethostname(), os.getpid(), boot, pidns


_TOKEN = None


def _marker_text(identity=None):
    """This process's marker content: its identity (``identity``: a
    :func:`_process_identity` tuple already read) and a random token, so two markers of
    different conversions never have the same content."""
    global _TOKEN
    if _TOKEN is None:
        _TOKEN = os.urandom(8).hex()
    host, pid, boot, pidns = identity or _process_identity()
    return f"{host} {pid} {boot} {pidns} {_TOKEN}"


def _fs_now(save_dir):
    """The file server's current time: the mtime of a file written now in ``save_dir``, so
    a file's age is measured on the same clock as its mtime (no skew between nodes). None
    when the directory cannot be written: no local clock is used instead (the callers then
    treat the age as unknown)."""
    try:
        fd, probe = tempfile.mkstemp(prefix=".muse_spark_clock.", dir=save_dir)
    except OSError:
        return None
    try:
        return os.fstat(fd).st_mtime
    finally:
        os.close(fd)
        os.unlink(probe)


def _marker_owner(save_dir, text=None):
    """The in-progress marker's owner (``text``: its content if already read):

    * ``None``: no readable marker. An empty marker (left by a process killed between the
      create and the write) is judged by its age, like a marker of another host;
    * ``"self"``: this process;
    * ``"live"``: a heartbeat younger than ``MARKER_STALE_S`` on the file server's clock,
      and, when host, boot and pid namespace match this process's (both readable), a pid
      that still runs; or no file-server clock (the directory is not writable);
    * ``"dead"``: a pid of this host / boot / namespace that no longer runs, or a heartbeat
      that has not moved for ``MARKER_STALE_S`` (also for a live pid: a reused pid or a
      process that no longer converts). An unparsable pid is judged by the heartbeat."""
    path = os.path.join(save_dir, IN_PROGRESS_MARKER)
    if text is None:
        text = _read_text(path, None)
    if text is None:
        return None
    fields = text.split()
    identity = _process_identity()  # read once: the self check and the host / boot match
    if text == _marker_text(identity):
        return "self"
    host, _, boot, pidns = identity
    if (
        len(fields) >= 4
        and fields[0] == host
        and fields[2] == boot
        and fields[3] == pidns
        and "-" not in (boot, pidns)
    ):
        try:
            os.kill(int(fields[1]), 0)
        except ProcessLookupError:
            return "dead"
        except (PermissionError, ValueError):
            pass  # a running pid of another user, or no pid: the heartbeat decides
    now = _fs_now(save_dir)
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return None
    if now is None:
        return "live"  # no file-server clock (directory not writable): never judged dead
    return "live" if now - mtime < MARKER_STALE_S else "dead"


def _tracker_iteration(save_dir):
    """The iteration Megatron's tracker file in ``save_dir`` names (an int, or the string
    ``"release"`` / its raw text), or None without a tracker."""
    text = _read_text(os.path.join(save_dir, TRACKER), None)
    if text is None:
        return None
    return int(text) if text.isdigit() else text


def _output_state(save_dir):
    """(release dir, state, detail) of ``save_dir`` for this conversion. The torch_dist save
    (``native_convert_common.load_validate_save``) writes straight into ``<save_dir>/release``
    and then the tracker file naming ``release`` (the save's iteration is written into the
    checkpoint, not its path, so it plays no part here).

    * ``"running"``: another live conversion holds the marker (detail: its marker text);
    * ``"other"``: the directory holds another checkpoint: a tracker naming an iteration, or
      an ``iter_*`` directory (detail: what);
    * ``"finished"``: the tracker names ``release`` and ``release/`` exists;
    * ``None``: no ``release/`` (a new or empty directory);
    * ``"ours"``: ``release/`` without a tracker, and the marker of a killed conversion of this
      converter (see :func:`_marker_owner`);
    * ``"foreign"``: ``release/`` without a tracker and no such marker."""
    target = os.path.join(save_dir, "release")
    text = _read_text(os.path.join(save_dir, IN_PROGRESS_MARKER), None)  # read once
    owner = _marker_owner(save_dir, text)
    if owner == "live":
        return target, "running", text or "?"
    tracked = _tracker_iteration(save_dir)
    if tracked == "release" and os.path.isdir(target):
        return target, "finished", ""
    if tracked is not None:
        return target, "other", f"{TRACKER} names iteration {tracked!r}"
    try:
        entries = os.listdir(save_dir)
    except FileNotFoundError:
        entries = []
    others = sorted(e for e in entries if _ITER_DIR.match(e))
    if others:
        return target, "other", f"it has {', '.join(others[:3])}"
    if not os.path.lexists(target):
        return target, None, ""
    # "self": this process's own earlier attempt (it failed after the claim): ours as well
    return target, ("ours" if owner in ("dead", "self") else "foreign"), ""


def refuse_existing_output(save_dir):
    """Raise unless the converter can write ``release/`` into ``save_dir``; return
    ``(target, state)`` otherwise (``state`` None or ``"ours"``). ``convert`` calls it first,
    before the model build and the mapping (hours and TBs of host RAM at full size). It never
    deletes anything: a leftover of this converter's own killed run (``"ours"``) is accepted
    here and removed by :func:`claim_output` after every other check."""
    target, state, detail = _output_state(save_dir)
    if state == "finished":
        raise ValueError(f"{target} already exists: move it away or convert into another directory")
    if state == "other":
        raise ValueError(
            f"{save_dir} holds another checkpoint ({detail}): the conversion would replace its "
            f"{TRACKER}; convert into another directory"
        )
    if state == "running":
        raise OutputBusyError(
            f"{save_dir} is being converted by another conversion ({IN_PROGRESS_MARKER}: {detail}; on "
            f"another host it counts as running until its heartbeat has stopped for {MARKER_STALE_S} s): "
            "wait for it, or remove the marker if that conversion is gone"
        )
    if state == "foreign":
        raise ValueError(
            f"{target} exists without {TRACKER} and was not left by a killed conversion of this "
            f"converter (no {IN_PROGRESS_MARKER} of a dead process): remove it or convert into another "
            "directory"
        )
    return target, state


def _retire_dead_marker(path, judged):
    """Move the marker that :func:`_marker_owner` judged dead (content ``judged``) out of the
    way, atomically: rename it to a name of this process, then check that the renamed file
    is that same marker (another conversion may have retired it and created its own in
    between: the content has a per-process token, so a new marker never equals ``judged``).
    A different marker is put back with ``os.link`` (which never overwrites); if yet another
    marker is there by then, both stay, and this conversion is refused."""
    aside = f"{path}.dead.{socket.gethostname()}.{os.getpid()}"
    try:
        os.rename(path, aside)  # of several processes, one renames it; the others see it gone
    except FileNotFoundError:
        return
    if _read_text(aside, None) == judged:
        os.unlink(aside)
        return
    try:  # a live conversion's marker was moved: put it back
        os.link(aside, path)
    except FileExistsError:
        # a third marker is in place; keep the moved one beside it (not lost) for the user
        raise OutputBusyError(
            f"{os.path.dirname(path)} was claimed by other conversions just now; a marker was "
            f"left at {aside}"
        ) from None
    os.unlink(aside)
    raise OutputBusyError(f"{os.path.dirname(path)} was claimed by another conversion just now")


def claim_output(save_dir, log=print):
    """``convert``, after every refusal (existing output, arguments, size cap): check
    ``save_dir`` again (:func:`refuse_existing_output`), take the in-progress marker
    atomically (exclusive creation; a killed run's marker is retired first, compared by
    content with the one judged dead), and only then, holding the marker, remove this
    converter's own leftover of a killed run (``"ours"``). Of two conversions that pass the
    check together, one gets the marker and the other is refused, so no two write or remove
    the same directory. :func:`release_output` removes the marker after the save; a failed
    conversion leaves it, so its leftover is known as this converter's on the rerun."""
    target, state = refuse_existing_output(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, IN_PROGRESS_MARKER)
    judged = _read_text(path, None)
    # killed, or ours; an empty marker (killed between create and write) once it is stale
    if judged is not None and _marker_owner(save_dir, judged) in ("dead", "self"):
        _retire_dead_marker(path, judged)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise OutputBusyError(
            f"{save_dir} is being converted by another conversion ({IN_PROGRESS_MARKER} was created "
            "by it just now): wait for it, or convert into another directory"
        ) from None
    with os.fdopen(fd, "w") as f:
        f.write(_marker_text() + "\n")
    if state == "ours":  # the marker is ours now: no other conversion removes or writes here
        log(f"{target}: the unfinished output of a killed conversion of this converter; removing it")
        # a TB-scale removal on shared storage can outlast MARKER_STALE_S: keep the marker
        # live meanwhile, so waiting nodes and other hosts do not judge it dead
        with marker_heartbeat(save_dir):
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)


@contextlib.contextmanager
def marker_heartbeat(save_dir, every=None):
    """Touch the in-progress marker every ``every`` seconds (default ``MARKER_HEARTBEAT_S``)
    while the block runs (a daemon thread), so another host can tell a running conversion
    from a killed one."""
    every = MARKER_HEARTBEAT_S if every is None else every
    path = os.path.join(save_dir, IN_PROGRESS_MARKER)
    stop = threading.Event()

    def beat():
        while not stop.wait(every):
            try:
                os.utime(path)
            except OSError:
                pass  # gone or transient: the next beat retries

    thread = threading.Thread(target=beat, name="muse-spark-convert-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=every + 1)


# the hook's waiting nodes (wait_for_hook_conversion): poll interval, how long node 0 may take
# to start (its marker to appear), and how long a gone lock may precede the .done marker
WAIT_POLL_S = 5
WAIT_START_S = 1800
WAIT_LOCK_GONE_S = 60
WAIT_REPORT_S = 600  # a progress line every 10 min while the other nodes wait


class OutputBusyError(ValueError):
    """``save_dir`` is held by another running conversion (its in-progress marker). The
    refused conversion leaves that conversion's failure record alone."""


RUN_ID_PREFIX = "run: "  # the first line of a failure or start record: the launch id ("-": none)


def format_run_line():
    """The first line of a failure or start record: this launch's id (``-``: none)."""
    return f"{RUN_ID_PREFIX}{conversion_run_id() or '-'}\n"


def format_failure_record(error):
    """The failure record :func:`convert` writes: the launch id, then the error."""
    return f"{format_run_line()}{type(error).__name__}: {error}\n"


def parse_run_record(text):
    """(launch id or None, rest) of a failure or start record's ``text``."""
    text = text or ""
    if not text.startswith(RUN_ID_PREFIX):
        return None, text.strip()
    first, _, rest = text.partition("\n")
    value = first[len(RUN_ID_PREFIX) :].strip()
    return (None if value in ("", "-") else value), rest.strip()


def conversion_run_id():
    """An id every node of this launch shares and no other launch has, or None: the
    optional ``PRIMUS_CONVERT_RUN_ID`` environment variable, returned as
    ``PRIMUS_CONVERT_RUN_ID=<value>`` (None when unset or blank). The user sets one
    new value per launch on every node (for example the Slurm job and step ids); the
    launchers do not set it. Without it, a failure of node 0 before the other nodes start
    waiting is seen after ``WAIT_START_S``; one while they wait is seen at once."""
    explicit = os.environ.get("PRIMUS_CONVERT_RUN_ID", "").strip()
    return f"PRIMUS_CONVERT_RUN_ID={explicit}" if explicit else None


def _record_path(save_dir, suffix):
    """``<save_dir><suffix>``: a record next to the checkpoint directory (a trailing
    separator of ``save_dir`` is dropped, so the record is never inside it)."""
    return os.fspath(save_dir).rstrip(os.sep) + suffix


def failure_record_path(save_dir):
    """``<save_dir>.failed``: written by :func:`convert` when the conversion fails (next to
    the hook's ``<save_dir>.done`` / ``.converting.lock``)."""
    return _record_path(save_dir, ".failed")


def write_failure_record(save_dir, error):
    """Write ``error`` to :func:`failure_record_path` with this launch's id (best effort: the
    error is the one to raise, so an OSError here is ignored)."""
    failed = failure_record_path(save_dir)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(failed)), exist_ok=True)
        with open(failed, "w") as f:
            f.write(format_failure_record(error))
    except OSError as exc:
        print(
            f"{LOG_TAG} warning: could not write failure record {failed}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def start_record_path(save_dir):
    """``<save_dir>.started``: written by node 0's hook (:func:`hook_before_convert`) with this
    launch's id once it holds the lock, before anything of the conversion runs."""
    return _record_path(save_dir, ".started")


def hook_before_convert(save_dir):
    """The posttrain hook's step on node 0 just before the conversion: record that this
    launch's node 0 started (its id, :func:`conversion_run_id`), and remove a previous
    launch's failure record. A failure of node 0 after this point, also one before the
    converter runs (convert-phase patches, its import), ends with the hook's lock removed;
    a waiting node that never saw the lock still knows from this record that node 0 started,
    so the gone lock without a finished checkpoint ends its wait within ``WAIT_LOCK_GONE_S``."""
    save_dir = os.fspath(save_dir)
    try:
        os.unlink(failure_record_path(save_dir))
    except FileNotFoundError:
        pass
    path = start_record_path(save_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(format_run_line())


def _record_of_this_launch(path, run_id):
    """True when the record at ``path`` carries this launch's ``run_id`` and was written less
    than ``WAIT_START_S`` ago (file server clock; a reused id of an old launch is not taken)."""
    if run_id is None or not os.path.exists(path):
        return False
    return parse_run_record(_read_text(path, ""))[0] == run_id and _record_age(path) < WAIT_START_S


def _file_identity(path):
    """(inode, mtime_ns, size) of ``path``, or None when it does not exist: it changes when the
    file is rewritten or touched. Compared with itself only, so no clock is involved."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


def _record_age(path):
    """Seconds since ``path`` was last written, on the file server's clock (``_fs_now`` of
    its directory); infinite when it is gone or that clock cannot be read (the record then
    never counts as this launch's)."""
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return float("inf")
    now = _fs_now(os.path.dirname(os.path.abspath(path)))
    return float("inf") if now is None else now - mtime


def wait_for_hook_conversion(
    save_dir, done_file, lock_file, log=print, clock=time.monotonic, sleep=time.sleep
):
    """The posttrain hook's other nodes, for a Muse Spark conversion on node 0: return when
    node 0's conversion is finished; there is no fixed limit while it converts.

    No local wall clock is compared with a file time (nodes and the file server may
    disagree). The only file ages are those of a pre-existing record (below), measured on the
    file server's clock (:func:`_record_age`: an mtime against the mtime of a file written
    now); when that clock cannot be read the record is not taken as this launch's:

    * finished: ``done_file`` and the checkpoint's tracker (``save_dir/{TRACKER}``) exist and
      ``lock_file`` does not. Node 0 skips the conversion when the tracker exists and converts
      when it does not; Megatron writes the tracker last, and the hook writes ``done_file``
      before it removes the lock. So a previous run's ``done_file`` without a finished
      checkpoint never ends the wait, and a conversion that finished just before this wait
      began does.
    * node 0's in-progress marker counts as this run's once its identity (inode, mtime, size)
      differs from the one seen when the wait began: a killed run's leftover is ignored.
      Its heartbeat is measured on this node's monotonic clock: the time since the marker
      last changed.

    It raises at once when node 0's conversion writes its failure record
    (:func:`failure_record_path`, also for a refusal before the lock is ever seen, e.g. the size
    cap). A record that was there before the wait began counts at once when it carries this
    launch's :func:`conversion_run_id` and was written less than ``WAIT_START_S`` ago (node 0
    failed before this node started waiting), and otherwise only once it is rewritten (a
    previous run's record, also one with a reused id).
    It raises when the hook's ``lock_file`` is gone for ``WAIT_LOCK_GONE_S`` without a
    finished checkpoint (node 0's conversion failed: the hook removes the lock in a
    ``finally``); when this run's marker has not changed for ``MARKER_STALE_S`` (node 0 was
    killed); and when node 0 has not started (no lock, or a lock without this run's
    marker) within ``WAIT_START_S``.
    """
    marker = os.path.join(save_dir, IN_PROGRESS_MARKER)
    tracker = os.path.join(save_dir, TRACKER)
    start = clock()
    initial_marker = _file_identity(marker)  # a killed run's leftover, if any
    seen_marker, marker_changed_at = initial_marker, start
    ours = False  # node 0 wrote or touched the marker during this wait
    last_report = start
    lock_seen = gone_since = None

    failed = failure_record_path(save_dir)
    run_id = conversion_run_id()
    # a record already there when this wait begins: this run's (node 0 failed before this node
    # started waiting) when it carries this launch's run id; else a previous run's, which
    # counts only once node 0 rewrites it
    # A matching id is also checked for age (file server clock), in case the id was reused
    # (e.g. a constant PRIMUS_CONVERT_RUN_ID): a record older than WAIT_START_S was not
    # written by this launch's node 0, which starts within that time of the other nodes.
    initial_failed = _file_identity(failed)
    initial_identity_with_id = None  # a pre-existing record taken as this launch's
    if initial_failed is not None and _record_of_this_launch(failed, run_id):
        initial_identity_with_id, initial_failed = initial_failed, None
    if run_id is None:
        log(
            "no launch id (PRIMUS_CONVERT_RUN_ID): a failure of "
            f"node 0 before this node started waiting is seen only after {WAIT_START_S} s; set "
            "PRIMUS_CONVERT_RUN_ID to one new value per launch on every node to see it at once"
        )
    if not os.path.exists(lock_file) and _record_of_this_launch(start_record_path(save_dir), run_id):
        # node 0 of this launch took the lock and released it before this wait began: from
        # here it counts as a gone lock (finished, or failed before or in the converter)
        lock_seen = True
    while True:
        now = clock()
        locked = os.path.exists(lock_file)
        if not locked and os.path.exists(done_file) and os.path.exists(tracker):
            break
        current_failed = _file_identity(failed)
        if current_failed is not None and current_failed != initial_failed:
            message = parse_run_record(_read_text(failed, "?"))[1] or "?"
            reused = (
                ""
                if current_failed != initial_identity_with_id
                else (
                    " (a record written before this wait began, with this launch's id "
                    f"{run_id}: if PRIMUS_CONVERT_RUN_ID was not changed since an earlier failed launch, "
                    "set a new value per launch, or remove the record)"
                )
            )
            raise RuntimeError(
                f"Muse Spark conversion on node 0 failed: {message} ({failed}); see node 0's log{reused}"
            )
        if locked:
            lock_seen, gone_since = True, None
        elif lock_seen:
            gone_since = now if gone_since is None else gone_since
            if now - gone_since >= WAIT_LOCK_GONE_S:
                raise RuntimeError(
                    f"Muse Spark conversion on node 0 ended without a finished checkpoint ({done_file} "
                    f"and {tracker}; its lock {lock_file} is gone): it failed; see node 0's log"
                )
        current = _file_identity(marker)
        if current != seen_marker:
            seen_marker, marker_changed_at = current, now
            ours = ours or current is not None
        if ours and current is not None and now - marker_changed_at >= MARKER_STALE_S:
            raise RuntimeError(
                f"Muse Spark conversion on node 0 stopped: its marker {marker} has not changed for "
                f"{now - marker_changed_at:.0f} s (node 0 was killed?)"
            )
        if now - last_report >= WAIT_REPORT_S:  # a sign of life on this node during hours
            last_report = now
            state = "converting (heartbeat moving)" if ours else ("lock seen" if lock_seen else "not started")
            log(f"waiting for node 0's conversion: {now - start:.0f} s, {state}")
        if not ours and now - start >= WAIT_START_S:
            what = f"no {lock_file}" if not lock_seen else f"no new {marker}"
            raise TimeoutError(
                f"Muse Spark conversion on node 0 did not start within {WAIT_START_S} s ({what})"
            )
        sleep(WAIT_POLL_S)
    log(f"node 0's conversion finished ({done_file})")


def release_output(save_dir):
    """After a complete save: remove the in-progress marker."""
    try:
        os.unlink(os.path.join(save_dir, IN_PROGRESS_MARKER))
    except FileNotFoundError:
        pass


class MuseSparkConverter:
    """The Muse Spark text decoder; the vision and audio towers are not ported.

    Single-rank build (TP = PP = EP = 1) with grouped-GEMM experts and the training vocab
    padding; saved as torch_dist, so a run with any TP / PP / EP loads it (Megatron
    reshards on load).
    """

    family = "muse-spark"
    # the rendezvous port of the single-process build; distinct from the llama / qwen3
    # (29563) and deepseek (29561) loaders
    master_port = 29571

    def __init__(self, log=None, num_layers=None):
        self.log = log or (lambda *a: print(LOG_TAG, *a, flush=True))
        self.num_layers = num_layers  # a depth cut (cut_to_layers); None: every layer

    def open_store(self, hf_dir):
        """The HF checkpoint as the built text model consumes it (the vision and audio
        towers' tensors are skipped, see :class:`MappedStore`)."""
        return MappedStore(common.SafetensorsStore(hf_dir), self.log, num_layers=self.num_layers)

    def convert(
        self,
        hf_dir,
        save_dir,
        *,
        dtype="bf16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        seq_length=DEFAULT_CONVERT_SEQ_LENGTH,
        master_port=None,
        max_params=None,
        ckpt_format=common.DEFAULT_CKPT_FORMAT,
    ):
        if common.check_ckpt_format(ckpt_format) != "torch_dist":
            raise ValueError(
                f"Muse Spark: native_ckpt_format {ckpt_format!r} is not supported; the converter writes "
                "torch_dist (Megatron reshards it to the training TP / PP / EP on load)"
            )
        refuse_existing_output(save_dir)  # before the build: a rerun fails at once, deletes nothing
        common.ensure_megatron_on_path()
        args = SimpleNamespace(
            hf_dir=hf_dir,
            save_dir=save_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            seq_length=seq_length,
            moe_grouped_gemm=True,  # the training layout (presets: moe_grouped_gemm: true)
        )
        cfg = common.read_hf_config(hf_dir)
        fields = derive_megatron_fields(cut_to_layers(cfg, self.num_layers))
        check_convert_args(args, cfg)  # build_margs' import-free checks, before the claim
        check_convert_size(fields, cfg, max_params=max_params, dtype=dtype)
        # every refusal is above: only now remove a killed run's own leftover and mark the
        # directory as this conversion's (a second conversion into it is refused)
        claim_output(save_dir, log=self.log)
        with marker_heartbeat(save_dir):
            margs = build_margs(fields, args, cfg)
            margs = common.init_megatron_single_process(margs, master_port or self.master_port, log=self.log)
            t_build = time.monotonic()
            model = build_model(margs)
            self.log(f"model built in {time.monotonic() - t_build:.1f} s")
            # stream_into_model copies (copy_) into the model's own tensors, keeping their
            # dtypes, so these are the saved dtypes
            check_fp32_tensors(model.state_dict(), margs.num_layers)
            store = self.open_store(hf_dir)
            try:  # close the store (and its reader threads) on an error too
                # into the model's own tensors, one layer at a time: no second full state dict
                sd, consumed = stream_into_model(model, store, margs, fields, self.log)
                common.assert_bridge_free("native conversion")
                prepare_torch_dist_save(margs, save_dir)
                # native_convert_common.load_validate_save: its mapping check (exact mapping or
                # RuntimeError) and the save, which follows margs.ckpt_format (torch_dist)
                t_save = time.monotonic()
                with unsplit_expert_fc1_save(model):
                    common.load_validate_save(model, sd, consumed, store, margs, save_dir, self.log)
                self.log(f"checked and saved in {time.monotonic() - t_save:.1f} s")
            finally:
                store.close()
        release_output(save_dir)  # the save wrote the tracker: the output is finished
        return save_dir


def convert(
    hf_dir,
    save_dir,
    *,
    dtype="bf16",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    expert_parallel_size=1,
    seq_length=DEFAULT_CONVERT_SEQ_LENGTH,
    master_port=None,
    log=None,
    max_params=None,
    num_layers=None,
    ckpt_format=common.DEFAULT_CKPT_FORMAT,
):
    """Full conversion, called by the posttrain conversion hook as
    ``convert(hf_dir, save_dir, dtype=..., tensor_parallel_size=..., pipeline_parallel_size=...,
    expert_parallel_size=...)``: build, map, check the mapping is exact, save a torch_dist
    checkpoint (Megatron reshards it on load, so the build is TP=PP=EP=1). ``max_params``
    optionally caps the estimated text-decoder size (None: no cap; the hook passes
    ``native_ckpt_max_params``); the host-memory check (:func:`check_convert_size`) always
    runs. ``num_layers`` cuts the checkpoint to its first
    layers (``cut_to_layers``); None keeps all.

    Any failure of this conversion (including a refusal before the claim, e.g. the size cap
    or a finished checkpoint) is written to ``failure_record_path(save_dir)``
    (``<save_dir>.failed``) before it is raised, so the hook's waiting nodes
    (:func:`wait_for_hook_conversion`) stop at once instead of waiting for a lock they may
    never see; a previous run's record is removed first. A conversion refused because
    another one is running in ``save_dir`` (:class:`OutputBusyError`) touches neither: the
    record is the running conversion's."""
    failed = failure_record_path(save_dir)
    try:
        try:
            refuse_existing_output(save_dir)  # a busy directory: refused before the record is touched
        except OutputBusyError:
            raise
        except ValueError:
            pass  # another refusal: reported through the record below (convert raises it again)
        try:
            os.unlink(failed)
        except FileNotFoundError:
            pass
        return MuseSparkConverter(log=log, num_layers=num_layers).convert(
            hf_dir,
            save_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            seq_length=seq_length,
            master_port=master_port,
            max_params=max_params,
            ckpt_format=ckpt_format,
        )
    except OutputBusyError:
        raise  # another conversion claimed the directory: its record, not ours
    except BaseException as e:
        write_failure_record(save_dir, e)
        raise


# ---------------------------------------------------------------------------
# The posttrain hook's side of the Muse Spark family (native_families/muse_spark.py
# registers these with native_registry): dtype fallback, prepare step and the wait
# ---------------------------------------------------------------------------
def _text_and_wrapper_dtypes(cfg):
    """(text_config's raw dtype, the wrapper's raw dtype) of an HF config dict; each may be
    None. transformers >= 4.56 writes ``dtype``, older versions ``torch_dtype``."""
    text = cfg.get("text_config") or {}
    return (text.get("torch_dtype") or text.get("dtype"), cfg.get("torch_dtype") or cfg.get("dtype"))


def hook_dtype_from_config(cfg):
    """``NativeFamily.dtype_from_config``: the raw dtype the text decoder is stored in when
    the config has no top-level ``torch_dtype``: text_config's dtype first (the converter
    converts the text decoder), else the wrapper's ``dtype``."""
    text, wrapper = _text_and_wrapper_dtypes(cfg)
    return text or wrapper


def hook_prepare(ctx):
    """``NativeFamily.prepare``: refuse a config whose wrapper dtype (the top-level
    ``torch_dtype``, which the registry reads first) differs from text_config's dtype
    unless ``native_ckpt_dtype`` is set (the decoder would be converted in the wrapper's
    dtype); check that the fine-tune fits (:func:`check_training_fits`); then plan the
    conversion: the depth cut of :func:`planned_cut` (``convert(num_layers=...)``, its own
    output directory, a depth record checked before a reuse) and :func:`hook_before_convert`
    on node 0."""
    text, _ = _text_and_wrapper_dtypes(ctx.hf_config)
    # the registry reads the top-level torch_dtype before the family's fallback: only it can
    # override text_config's dtype
    top = ctx.hf_config.get("torch_dtype")
    if top and text and not ctx.config_value("native_ckpt_dtype"):
        if registry.dtype_name(top) != registry.dtype_name(text):
            raise registry.NativeConvertRefused(
                f"Muse Spark: config.json's torch_dtype ({top}) differs from text_config's dtype "
                f"({text}); set native_ckpt_dtype (e.g. {registry.dtype_name(text) or 'bf16'}) to "
                "choose the dtype the text decoder is converted in"
            )
    # a fine-tune shallower than the checkpoint converts its first layers (convert's
    # num_layers), into its own directory (path suffix) so a full conversion is not reused;
    # check_training_fits returns that cut (training args parsed once)
    layers = check_training_fits(ctx)
    return registry.ConvertPlan(
        convert_kwargs={} if layers is None else {"num_layers": layers},
        path_suffix="" if layers is None else f"_layers{layers}",
        check_existing=_cut_record_check(layers),
        before_convert=_before_convert_with_cut(layers),
    )


# The training args the fail-fast checks read (memory_check.state_bytes_per_gpu and the
# depth check), from the fine-tune's config and its command-line overrides.
_TRAINING_KEYS = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "group_query_attention",
    "num_query_groups",
    "kv_channels",
    "num_experts",
    "moe_ffn_hidden_size",
    "moe_latent_size",
    "swiglu",
    "vocab_size",
    "untie_embeddings_and_output_weights",
    "mtp_num_layers",
    "bf16",
    "fp16",
    "params_dtype",
    "main_grads_dtype",
    "accumulate_allreduce_grads_in_fp32",
    "grad_reduce_in_bf16",
    "optimizer",
    "optimizer_cpu_offload",
    "optimizer_offload_fraction",
    "use_precision_aware_optimizer",
    "use_distributed_optimizer",
    "use_megatron_fsdp",
    "use_torch_fsdp2",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "decoder_first_pipeline_num_layers",
    "decoder_last_pipeline_num_layers",
    "account_for_embedding_in_pipeline_split",
    "account_for_loss_in_pipeline_split",
    "muse_spark_skip_memory_check",
)


def training_args(ctx):
    """The fine-tune's training args (``_TRAINING_KEYS``) as a namespace: the command-line
    overrides (``ctx.argv``), else the experiment module's value, else None."""
    tokens = list(ctx.argv or [])
    first = next((i for i, t in enumerate(tokens) if t.startswith("--") or "=" in t), len(tokens))
    over = parse_cli_overrides(tokens[first:]) if first < len(tokens) else {}
    return SimpleNamespace(
        **{k: (over[k] if over.get(k) is not None else ctx.config_value(k)) for k in _TRAINING_KEYS},
        attention_output_gate=True,
    )


def _gpu_memory_bytes():
    """This node's GPU memory (bytes), or None without a visible GPU."""
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # noqa: BLE001 - no device query: the before_train check still runs
        return None
    return None


CUT_RECORD_SUFFIX = ".muse_spark_layers"  # <checkpoint>.muse_spark_layers: the converted depth


def planned_cut(ctx, a=None):
    """The depth to convert for the fine-tune of ``ctx``: None (every layer) when its
    ``num_layers`` is unset or equals the HF checkpoint's, else its ``num_layers`` (the
    checkpoint cut to its first layers, :func:`cut_to_layers`, as a
    fine-tune with ``--num_layers 4`` trains). A depth above the checkpoint's raises
    ``NativeConvertRefused``."""
    a = a or training_args(ctx)
    hf_layers = text_config(ctx.hf_config or {}).get("num_hidden_layers")
    if a.num_layers is None or hf_layers is None:
        return None
    n, total = int(a.num_layers), int(hf_layers)
    if n > total:
        raise registry.NativeConvertRefused(
            f"Muse Spark: the fine-tune builds num_layers={n}, but the HF checkpoint "
            f"{ctx.hf_path} has only {total} layers. Set --num_layers {total} or less (a smaller "
            "depth converts the checkpoint's first layers)."
        )
    if n < 1:
        raise registry.NativeConvertRefused(f"Muse Spark: num_layers={n} must be at least 1")
    return None if n == total else n


CUT_RECORD_ALL = "all"  # the depth record's value for a conversion of every layer


def _cut_record_value(layers):
    """The ``<checkpoint>.muse_spark_layers`` record for a cut of ``layers`` (None: all)."""
    return CUT_RECORD_ALL if layers is None else str(layers)


def cut_record_path(save_dir):
    """``<save_dir>.muse_spark_layers``: the converted depth, next to the checkpoint."""
    return _record_path(save_dir, CUT_RECORD_SUFFIX)


def _cut_record_check(layers):
    """``ConvertPlan.check_existing``: refuse to reuse a checkpoint converted at another
    depth (its ``<checkpoint>.muse_spark_layers`` record; ``all`` for every layer). A
    checkpoint without the record (converted elsewhere) is reused as before."""
    want = _cut_record_value(layers)

    def check(path):
        got = _read_text(cut_record_path(path), None)
        if got is None or got.strip() == want:
            return None
        return (
            f"{path} was converted with {got.strip()} decoder layer(s), but this fine-tune needs "
            f"{want}: remove it, or set native_ckpt_out to another directory"
        )

    return check


def _before_convert_with_cut(layers):
    want = _cut_record_value(layers)

    def before(save_dir):
        hook_before_convert(save_dir)
        path = cut_record_path(save_dir)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(want + "\n")

    return before


def check_training_fits(ctx, gpu_bytes=None, world_size=None, log=print):
    """Refuse, before the conversion, a fine-tune that cannot use the converted checkpoint:

    * its ``num_layers`` is above the HF checkpoint's (a smaller one is converted as a cut,
      :func:`planned_cut`);
    * its weights + grads + optimizer states do not fit the GPUs
      (:func:`memory_check.required_state_bytes_or_raise`, the before_train check, here
      run before hours of conversion and a ~1.6 TB (1.5 TiB) write), unless
      ``muse_spark_skip_memory_check`` is set. World size: ``NNODES`` x ``GPUS_PER_NODE``
      (the launcher sets both); no GPU visible: only the depth check runs.

    Returns the depth cut (:func:`planned_cut`, None for every layer), computed once from
    the same training args. Raises ``NativeConvertRefused`` with the reason."""
    a = training_args(ctx)
    cut = planned_cut(ctx, a)  # a depth above the checkpoint's is refused
    if a.num_layers is None or a.num_experts is None or skip_requested(a):
        return cut
    gpu_bytes = gpu_bytes if gpu_bytes is not None else _gpu_memory_bytes()
    if gpu_bytes is None:
        return cut
    n_gpus = gpus_per_node()
    if world_size is None:
        nodes = require_positive_int(os.environ.get("NNODES") or 1, "NNODES must be a positive integer")
        world_size = n_gpus * nodes
    try:
        need = required_state_bytes_or_raise(a, world_size, gpu_bytes, n_gpus_per_node=n_gpus)
    except ValueError as e:
        raise registry.NativeConvertRefused(f"{e} (checked before the conversion)") from e
    log(f"{LOG_TAG} the fine-tune fits: ~{need / GiB:.0f} GiB of model states per GPU")
    return cut


def hook_wait(megatron_path, done_file, lock_file, node_rank):
    """``NativeFamily.wait``: a node other than node 0 follows node 0's conversion through
    :func:`wait_for_hook_conversion` (hours at full size, longer than the hook's 600 s
    wait). Its messages print on this node (the hook's log_info prints on rank 0 only)."""

    def log_node(msg):
        print(f"[INFO] [NODE_RANK {node_rank}] {msg}", flush=True)

    log_node(f"Waiting for node 0's Muse Spark conversion into {megatron_path}")
    wait_for_hook_conversion(megatron_path, done_file, lock_file, log=log_node)
    log_node(f"Checkpoint ready at {megatron_path}")
