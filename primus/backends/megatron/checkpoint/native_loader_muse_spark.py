###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HF ``muse_spark`` (Muse Spark text decoder) -> Megatron-Core converter.

Single-process build-model-and-copy path (``native_converter.NativeConverter``). The model is Primus' own :class:`MuseSparkModel`
(``model_type: muse_spark``), so :func:`build_model` builds it through the
``muse_spark`` model provider.

The text decoder (``<prefix>*`` + ``lm_head``) and the waveform audio tower
(``model.audio_tower.*`` -> ``audio_tower.*``, same module names) are mapped. The vision
tower and its adapter are not ported yet: their tensors are reported and left out of the
exact-mapping check (``MappedStore``); every text and audio tensor must be consumed.

:class:`MuseSparkConverter` follows the :class:`NativeConverter` template and saves a
``torch_dist`` checkpoint, which Megatron reshards on load (any TP / PP / EP). The
build is single-rank (TP = PP = EP = 1, asserted), uses the training
layout (grouped-GEMM expert keys ``linear_fc{1,2}.weight<e>``, as ``moe_grouped_gemm:
true`` in the presets) and the training vocab padding (``vocab_size_with_padding``:
a multiple of ``make_vocab_size_divisible_by``, 64 in the presets, independent of TP:
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
from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.checkpoint.native_ckpt_options import (
    MAX_PARAMS_KEY,
    parse_max_params,
)
from primus.backends.megatron.checkpoint.native_converter import NativeConverter
from primus.backends.megatron.checkpoint.native_loader_qwen3 import (
    _detect_expert_key_style,
)
from primus.backends.megatron.core.models.muse_spark.audio_encoder import (
    audio_config_from_hf,
)
from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    MAKE_VOCAB_SIZE_DIVISIBLE_BY,
    normalize_layer_pattern,
    vocab_size_with_padding,
)
from primus.backends.megatron.core.models.muse_spark.memory_check import (
    DEFAULT_VOCAB_SIZE,
    param_split,
)

# HF text-decoder key prefixes, most specific first: the conditional-generation wrapper
# (text + vision / audio towers) and a text-only MuseSparkForCausalLM.
HF_PREFIXES = ("model.language_model.", "model.")
HF_PREFIX = HF_PREFIXES[0]
# Same values as the MuseSparkTransformerConfig field defaults (kept import-free here:
# derive_megatron_fields runs on the CPU without Megatron on the path).
_CFG_DEFAULTS = {
    "audio_token_id": 200038,
    "qk_scale_factor": 5.4730064863875,
    "moe_router_output_multiplier": 0.17677669529663687,
    "output_logit_multiplier": 0.17677669529663687,
    "final_logit_softcapping": 20.0,
    "post_norm_eps": 1e-8,
    "residual_gate_temperature": 0.3,
    # HF config defaults (Megatron args, not MuseSparkTransformerConfig fields)
    "rope_theta": 500000.0,
    "max_position_embeddings": 131072,
}
# Megatron keys that must stay fp32 in a bf16 checkpoint (HF _keep_in_fp32_modules_strict:
# the residual gates, the router weight and its score-correction bias). convert() runs
# check_fp32_tensors on the built model's state dict: load_state_dict copies into the
# model's own tensors and keeps their dtype, so that is the dtype that gets saved.
FP32_SUFFIXES = ("_bda.gate", "mlp.router.weight", "mlp.router.expert_bias")


def check_fp32_tensors(sd, num_layers):
    """Raise when an ``FP32_SUFFIXES`` tensor is not fp32, or one is missing for a layer.

    Run it on ``model.state_dict()`` after ``model.to(params_dtype)``: a bf16 router weight,
    expert bias or gate there would be saved as bf16 (``load_state_dict`` casts the fp32
    source into it silently)."""
    found = {}
    for k, v in sd.items():
        if k.endswith(FP32_SUFFIXES):
            if v.dtype != torch.float32:
                raise RuntimeError(f"{k} must stay fp32 (HF keeps it fp32), got {v.dtype}")
            found[k] = True
    expected = 4 * num_layers  # self_attn_bda.gate, mlp_bda.gate, router weight, expert_bias
    if len(found) != expected:
        raise RuntimeError(
            f"expected {expected} fp32 tensors ({FP32_SUFFIXES}) in the model state dict, found {len(found)}"
        )


def text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


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
    mlp_types = t.get("mlp_layer_types") or ["sparse"] * n
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
    return dict(
        hidden_size=t["hidden_size"],
        num_attention_heads=t["num_attention_heads"],
        num_layers=n,
        max_position_embeddings=t.get("max_position_embeddings", _CFG_DEFAULTS["max_position_embeddings"]),
        norm_epsilon=t["rms_norm_eps"],
        rotary_base=rotary_base,
        untie_embeddings_and_output_weights=not t.get("tie_word_embeddings", False),
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
        window_size=(
            (int(t["sliding_window"]), 0) if t.get("sliding_window") else None
        ),  # HF count; config -> TE
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
        moe_router_output_multiplier=t.get(
            "output_multiplier", _CFG_DEFAULTS["moe_router_output_multiplier"]
        ),
        output_logit_multiplier=t.get("output_multiplier", _CFG_DEFAULTS["output_logit_multiplier"]),
        final_logit_softcapping=t.get("final_logit_softcapping", _CFG_DEFAULTS["final_logit_softcapping"]),
        # the waveform audio encoder (top-level HF audio_config; None for a text-only config)
        muse_audio_config=audio_config_from_hf(cfg.get("audio_config") if "text_config" in cfg else None),
        audio_token_id=int(cfg.get("audio_token_id", _CFG_DEFAULTS["audio_token_id"])),
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


def checkpoint_vocab(cfg):
    """The checkpoint's ``vocab_size`` from the loaded HF config (``text_config``), else None."""
    v = text_config(cfg or {}).get("vocab_size")
    return int(v) if v else None


def estimate_num_params(f, vocab_size=DEFAULT_VOCAB_SIZE):
    """Muse Spark text-decoder parameter count from derive_megatron_fields output.

    The same count as the training memory check (``memory_check.param_split``), so the
    converter cap and the memory check agree on a model. Muse Spark always has the
    attention output gate.
    """
    a = SimpleNamespace(**f, attention_output_gate=True, mtp_num_layers=0)
    return sum(param_split(a, vocab_size))


def check_convert_size(fields, cfg=None, max_params=None):
    """Refuse a checkpoint conversion above the cap (only ``convert`` calls this).

    The cap is ``max_params`` (the ``native_ckpt_max_params`` config value, see
    :func:`parse_max_params`; None -> 60B params).

    ``convert`` builds the whole model on CPU in one process plus a full second state
    dict (a torch_dist checkpoint; Megatron reshards it on load). For the released 816B
    model that is TBs of host RAM. Returns the estimate.
    """
    vocab = checkpoint_vocab(cfg) or DEFAULT_VOCAB_SIZE
    n = estimate_num_params(fields, vocab)
    limit = parse_max_params(max_params)
    if n > limit:
        raise ValueError(
            f"Muse Spark native converter: ~{n:,} params (~{n / 1e9:.3g}B) is above the {limit:,} "
            "params single-process limit (it holds the model and a full state dict in host RAM), so the "
            "full 816B model has no converter route. It is for small / proxy checkpoints; the "
            f"{MAX_PARAMS_KEY} config key overrides the limit."
        )
    return n


# the save dtypes the converter accepts; any other value is an error, not a silent fp32
SUPPORTED_DTYPES = ("bf16", "fp16", "fp32")


def _hf_cfg_of(args, hf_cfg=None):
    """``hf_cfg`` when given, else ``<args.hf_dir>/config.json`` (None without an hf_dir)."""
    if hf_cfg is None and getattr(args, "hf_dir", None):
        hf_cfg = common.read_hf_config(args.hf_dir)
    return hf_cfg


def build_margs(cfg_fields, args, hf_cfg=None):
    """Megatron args for a single-process build.

    ``hf_cfg`` is the loaded HF config.json (``convert()`` passes the one it read). Callers
    of the two-argument form (e.g. the logprob-parity harness) get it read from
    ``args.hf_dir``, so the checkpoint's vocab_size (not the tokenizer's) sizes the head
    on every path.
    """
    hf_cfg = _hf_cfg_of(args, hf_cfg)

    tp, pp, ep = (
        int(getattr(args, k, 1) or 1)
        for k in ("tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size")
    )
    if (tp, pp, ep) != (1, 1, 1):
        # build_mcore_state_dict maps full (unsharded) tensors; it does not split them.
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
    m.qk_layernorm = True
    m.attention_output_gate = True
    m.add_qkv_bias = cfg_fields.get("add_qkv_bias", False)
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
    m.moe_router_topk_scaling_factor = cfg_fields.get("moe_router_topk_scaling_factor", 1.0)
    m.moe_router_load_balancing_type = "none"
    m.moe_aux_loss_coeff = 0.0
    m.moe_router_dtype = "fp32"
    m.moe_token_dispatcher_type = "alltoall"
    # convert() saves with grouped-GEMM experts (the training layout, moe_grouped_gemm: true).
    # Without the arg (the l3 parity harness), SequentialMLP experts keep exact fp32 math.
    m.moe_grouped_gemm = bool(getattr(args, "moe_grouped_gemm", False))
    m.moe_permute_fusion = False

    # the per-layer Muse fields, built once: set here and set again after validate_args
    per_layer = {k: _per_layer(cfg_fields.get(k), m.num_layers) for k in MUSE_FIELDS}
    for k, v in per_layer.items():
        setattr(m, k, v)
    # the audio tower (MuseSparkModel builds it when muse_audio_config is set)
    m.muse_audio_config = cfg_fields.get("muse_audio_config")
    m.audio_token_id = cfg_fields.get("audio_token_id", _CFG_DEFAULTS["audio_token_id"])

    # The checkpoint's vocab_size (text tower), not the tokenizer's: the tokenizer drops
    # added tokens, so a tokenizer-sized head would miss rows of the softmax.
    m.vocab_size = checkpoint_vocab(hf_cfg)
    m.make_vocab_size_divisible_by = int(
        getattr(args, "make_vocab_size_divisible_by", None) or MAKE_VOCAB_SIZE_DIVISIBLE_BY
    )
    if m.vocab_size:
        # The training padding (vocab_size_with_padding, shared with the muse_spark_vocab
        # build_args patch; it does not depend on TP), so a saved checkpoint loads into a
        # training run with the same make_vocab_size_divisible_by at any TP.
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
    m.world_size = tp * pp * ep
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
    m.ckpt_format = "torch_dist"  # NativeConverter.save writes torch_dist
    m.no_save_optim = True
    m.no_save_rng = True
    m.no_load_optim = True
    m.no_load_rng = True
    m.save_interval = 1
    m.async_save = False
    m.ckpt_assume_constant_structure = False
    m.iteration = 1

    m = validate_args(m)
    # validate_args may rebuild list-valued fields from strings; restore ours.
    for k, v in per_layer.items():
        setattr(m, k, v)
    m.iteration = 1
    return m


def build_model(margs):
    """Build :class:`MuseSparkModel` through the ``muse_spark`` model provider."""
    from primus.core.utils.import_utils import get_model_provider

    model_provider = get_model_provider(model_type="muse_spark")
    model = model_provider(pre_process=True, post_process=True)
    model = model.to(margs.params_dtype)
    # .to(dtype) casts every param; the router weight and the residual gates stay fp32
    # (MuseSparkTopKRouter._apply restores the router; the gates restore themselves too).
    return model


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping
# ---------------------------------------------------------------------------
def build_mcore_state_dict(model, store, margs, cfg_fields, log):
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
    sd, consumed = {}, set()
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

    sd["embedding.word_embeddings.weight"] = common.pad_vocab(
        get(prefix + "embed_tokens.weight"), padded_vocab
    )
    sd["decoder.final_layernorm.weight"] = get(prefix + "norm.weight")
    if margs.untie_embeddings_and_output_weights:
        sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)

    for L in range(margs.num_layers):
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
            sd[fc1_key.format(p=p, e=e)] = gate_up[e].contiguous()
            sd[fc2_key.format(p=p, e=e)] = down[e].contiguous()
        del gate_up, down
        log(f"layer {L} mapped ({style} experts)")

    if getattr(model, "audio_tower", None) is not None:
        sd.update(map_audio_tower(model.audio_tower, audio_prefix(prefix), get, log))
    return sd, consumed


def audio_prefix(text_prefix):
    """The HF audio-tower prefix next to the text prefix (``model.audio_tower.``)."""
    return (
        text_prefix.replace("language_model.", "audio_tower.", 1)
        if "language_model." in text_prefix
        else None
    )


def map_audio_tower(tower, prefix, get, log=print):
    """HF ``<prefix>*`` -> ``audio_tower.*``: the module names are the same (one to one)."""
    if not prefix:
        raise ValueError(
            "the model has an audio_tower, but the checkpoint is text-only (no audio tower prefix)"
        )
    out = {}
    for name in tower.state_dict():
        out["audio_tower." + name] = get(prefix + name)
    log(f"audio tower mapped: {len(out)} tensors from {prefix}*")
    return out


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


def is_text_key(name, prefix=HF_PREFIX):
    """True for the text decoder's HF tensors (the ones this converter maps)."""
    return name.startswith(prefix) or name.startswith("lm_head.")


class MappedStore:
    """A SafetensorsStore view whose ``all_keys`` holds only the tensors this port maps:
    the text decoder and (``with_audio``) the audio tower; not the vision tower.

    The exact-mapping check fails on HF tensors the mapping did not consume; the vision
    tower and its adapter are not ported yet (and the audio tower is not mapped when the
    built model has none), so those are logged and excluded from it.
    """

    def __init__(self, store, log=print, with_audio=True):
        self._store = store
        keys = store.all_keys()
        self.prefix = detect_hf_prefix(keys)
        audio = audio_prefix(self.prefix) if with_audio else None  # computed once, not per key
        self._mapped_keys = frozenset(
            k for k in keys if is_text_key(k, self.prefix) or (audio is not None and k.startswith(audio))
        )
        self.skipped = sorted(k for k in keys if k not in self._mapped_keys)
        if self.skipped:
            scope = "vision" if with_audio else "vision + audio"
            log(
                f"not mapped (out of scope: {scope} tower): {len(self.skipped)} tensors, e.g. {self.skipped[:4]}"
            )

    def all_keys(self):
        # the key set is fixed: computed once in __init__ (a fresh set, callers may mutate it)
        return set(self._mapped_keys)

    def __getattr__(self, name):
        return getattr(self._store, name)


class MuseSparkConverter(NativeConverter):
    """The Muse Spark text decoder on the :class:`NativeConverter` template.

    Single-rank build (TP = PP = EP = 1) with grouped-GEMM experts and the training vocab
    padding; saved as torch_dist, so a run with any TP / PP / EP loads it (Megatron
    reshards on load).
    """

    family = "muse-spark"
    with_audio = True  # set from the built model in check_model

    def derive_fields(self, cfg):
        return derive_megatron_fields(cfg)

    def check_config(self, fields, args, cfg):
        check_convert_size(fields, cfg, max_params=getattr(args, "max_params", None))

    def build_margs(self, fields, args, cfg):
        return build_margs(fields, args, cfg)

    def build_model(self, margs):
        return build_model(margs)

    def check_model(self, model, margs):
        # Checked on the model, not on the mapped dict (that one is fp32 by construction):
        # load_state_dict keeps the model's dtypes, so these are the saved dtypes.
        check_fp32_tensors(model.state_dict(), margs.num_layers)
        self.with_audio = getattr(model, "audio_tower", None) is not None

    def open_store(self, hf_dir):
        return MappedStore(common.SafetensorsStore(hf_dir), self.log, with_audio=self.with_audio)

    def build_state_dict(self, model, store, margs, fields):
        return build_mcore_state_dict(model, store, margs, fields, self.log)


def convert(
    hf_dir,
    save_dir,
    *,
    dtype="bf16",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    expert_parallel_size=1,
    seq_length=4096,
    make_vocab_size_divisible_by=MAKE_VOCAB_SIZE_DIVISIBLE_BY,
    master_port=None,
    log=None,
    max_params=None,
):
    """Full conversion (the posttrain conversion hook calls this): build, map, check the
    mapping is exact, save a torch_dist checkpoint. ``max_params`` is the
    ``native_ckpt_max_params`` config value (None -> 60B). ``master_port`` None uses
    ``NativeConverter.master_port``."""
    return MuseSparkConverter(log=log).convert(
        hf_dir,
        save_dir,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        seq_length=seq_length,
        master_port=master_port,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        moe_grouped_gemm=True,  # the training layout (presets: moe_grouped_gemm: true)
        max_params=max_params,
    )
