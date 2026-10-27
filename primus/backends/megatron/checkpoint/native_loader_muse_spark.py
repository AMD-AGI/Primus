###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HF ``muse_spark`` (Muse Spark text decoder) -> Megatron-Core converter.

Single-process build-model-and-copy path (see ``native_convert_common``), like
the Qwen3 converter. The model is Primus' own :class:`MuseSparkModel`
(``model_type: muse_spark``), so :func:`build_model` builds it through the
``muse_spark`` model provider.

Only the text decoder (``model.language_model.*`` + ``lm_head``) is mapped; the
vision / audio towers and their adapters are not part of this port.

Mapping (HF ``model.language_model.layers.N.`` -> mcore ``decoder.layers.N.``):

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

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common

HF_PREFIX = "model.language_model."
FP32_SUFFIXES = (".gate", "mlp.gate.weight", "mlp.gate.e_score_correction_bias")


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
    rope_theta = t.get("layer_rope_theta") or [rope.get("rope_theta", t.get("rope_theta", 500000.0))] * n
    mlp_types = t.get("mlp_layer_types") or ["sparse"] * n
    assert all(x == "sparse" for x in mlp_types), "dense Muse Spark layers are not supported"
    assert not t.get("num_shared_experts"), "Muse Spark shared experts are not supported"
    return dict(
        hidden_size=t["hidden_size"],
        num_attention_heads=t["num_attention_heads"],
        num_layers=n,
        max_position_embeddings=t.get("max_position_embeddings", 131072),
        norm_epsilon=t["rms_norm_eps"],
        rotary_base=rope.get("rope_theta", t.get("rope_theta", 500000.0)),
        untie_embeddings_and_output_weights=not t.get("tie_word_embeddings", False),
        ffn_hidden_size=t.get("intermediate_size", t["moe_intermediate_size"]),
        num_query_groups=t["num_key_value_heads"],
        kv_channels=t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
        add_qkv_bias=bool(t.get("attention_bias", False)),
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
        post_norm_eps=t.get("post_norm_eps", 1e-8),
        residual_gate_temperature=t.get("residual_gate_temperature", 0.3),
        qk_scale_factor=t.get("qk_scale_factor", 1.0),
        moe_router_output_multiplier=t.get("output_multiplier", 1.0),
        output_logit_multiplier=t.get("output_multiplier", 1.0),
        final_logit_softcapping=t.get("final_logit_softcapping"),
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
    """Cut a per-layer pattern list down to ``num_layers`` (depth-reduced builds)."""
    if isinstance(value, list):
        return value[:num_layers]
    return value


def _checkpoint_vocab(hf_dir, cfg_fields):
    import json
    import os

    v = cfg_fields.get("vocab_size") if isinstance(cfg_fields, dict) else None
    if not v and hf_dir and os.path.isfile(os.path.join(hf_dir, "config.json")):
        with open(os.path.join(hf_dir, "config.json")) as f:
            cfg = json.load(f)
        v = (cfg.get("text_config") or {}).get("vocab_size") or cfg.get("vocab_size")
    return int(v) if v else None


def build_margs(cfg_fields, args):
    import argparse

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
    # SequentialMLP experts keep exact fp32 math (grouped GEMM is a training-time choice).
    m.moe_grouped_gemm = bool(getattr(args, "moe_grouped_gemm", False))
    m.moe_permute_fusion = False

    for k in MUSE_FIELDS:
        setattr(m, k, _per_layer(cfg_fields.get(k), m.num_layers))

    # The checkpoint's vocab_size (text tower), not the tokenizer's: the tokenizer drops
    # added tokens, so a tokenizer-sized head would miss rows of the softmax.
    m.vocab_size = _checkpoint_vocab(args.hf_dir, cfg_fields)
    m.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    if m.vocab_size:
        # Keep every checkpoint row (202048 = 64 * 3157): Megatron would otherwise pad the
        # tokenizer's smaller vocab to a multiple of 128 and drop real lm_head rows.
        tp = max(int(args.tensor_parallel_size), 1)
        m.padded_vocab_size = -(-m.vocab_size // tp) * tp
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False

    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1

    m.tensor_model_parallel_size = args.tensor_parallel_size
    m.pipeline_model_parallel_size = args.pipeline_parallel_size
    m.expert_model_parallel_size = args.expert_parallel_size
    m.expert_tensor_parallel_size = 1
    m.context_parallel_size = 1
    m.sequence_parallel = False
    m.world_size = args.tensor_parallel_size * args.pipeline_parallel_size * args.expert_parallel_size
    m.rank = 0
    if args.dtype == "bf16":
        m.bf16 = True
        m.params_dtype = torch.bfloat16
    elif args.dtype == "fp16":
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
    m.ckpt_format = "torch"
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
    for k in MUSE_FIELDS:
        setattr(m, k, _per_layer(cfg_fields.get(k), m.num_layers))
    m.iteration = 1
    return m


def build_model(margs):
    """Build :class:`MuseSparkModel` through the ``muse_spark`` model provider."""
    from primus.core.utils.import_utils import get_model_provider

    model_provider = get_model_provider(model_type="muse_spark")
    model = model_provider(pre_process=True, post_process=True)
    return model.to(margs.params_dtype)


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping
# ---------------------------------------------------------------------------
def _expert_key_style(model, num_layers):
    keys = set(model.state_dict().keys())
    for L in range(num_layers):
        p = f"decoder.layers.{L}.mlp.experts."
        if (p + "linear_fc1.weight0") in keys:
            return "grouped"
        if (p + "local_experts.0.linear_fc1.weight") in keys:
            return "sequential"
    return None


def build_mcore_state_dict(model, store, margs, cfg_fields, log):
    dtype = margs.params_dtype
    padded_vocab = model.state_dict()["embedding.word_embeddings.weight"].shape[0]
    nh, ng, hd, hidden = (
        margs.num_attention_heads,
        margs.num_query_groups,
        margs.kv_channels,
        margs.hidden_size,
    )
    assert nh % ng == 0
    num_experts = cfg_fields["num_experts"]
    style = _expert_key_style(model, margs.num_layers)
    if style is None:
        raise RuntimeError("could not detect the MoE expert key layout on the built model")

    sd, consumed = {}, set()

    def get(name, keep_fp32=False):
        consumed.add(name)
        return store.get(name, torch.float32 if keep_fp32 else dtype)

    def gated_qkv(h):
        q = get(h + "self_attn.q_proj.weight").reshape(ng, (nh // ng) * hd, hidden)
        g = get(h + "self_attn.gate_proj.weight").reshape(ng, (nh // ng) * hd, hidden)
        k = get(h + "self_attn.k_proj.weight").reshape(ng, hd, hidden)
        v = get(h + "self_attn.v_proj.weight").reshape(ng, hd, hidden)
        return torch.cat([q, g, k, v], dim=1).reshape(-1, hidden).contiguous()

    sd["embedding.word_embeddings.weight"] = common.pad_vocab(
        get(HF_PREFIX + "embed_tokens.weight"), padded_vocab
    )
    sd["decoder.final_layernorm.weight"] = get(HF_PREFIX + "norm.weight")
    if margs.untie_embeddings_and_output_weights:
        sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)

    for L in range(margs.num_layers):
        p = f"decoder.layers.{L}."
        h = f"{HF_PREFIX}layers.{L}."
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
        assert gate_up.shape[0] == num_experts and down.shape[0] == num_experts
        for e in range(num_experts):
            if style == "grouped":
                sd[f"{p}mlp.experts.linear_fc1.weight{e}"] = gate_up[e].contiguous()
                sd[f"{p}mlp.experts.linear_fc2.weight{e}"] = down[e].contiguous()
            else:
                sd[f"{p}mlp.experts.local_experts.{e}.linear_fc1.weight"] = gate_up[e].contiguous()
                sd[f"{p}mlp.experts.local_experts.{e}.linear_fc2.weight"] = down[e].contiguous()
        del gate_up, down
        log(f"layer {L} mapped ({style} experts)")

    return sd, consumed


def convert(args, log=print):
    """Full conversion: build, map, validate exactly, save a torch checkpoint."""
    cfg = common.read_hf_config(args.hf_dir)
    fields = derive_megatron_fields(cfg)
    margs = build_margs(fields, args)
    margs = common.init_megatron_single_process(margs, getattr(args, "master_port", 29571), log=log)
    model = build_model(margs)
    store = common.SafetensorsStore(args.hf_dir)
    sd, consumed = build_mcore_state_dict(model, store, margs, fields, log)
    return common.load_validate_save(model, sd, consumed, store, margs, args.save_dir, log)
