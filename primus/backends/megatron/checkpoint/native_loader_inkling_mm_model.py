###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for Inkling
(``thinkingmachines/Inkling``, HF ``model_type: inkling_mm_model``).

Only the text backbone (``model.llm.*``) is mapped; the released audio / vision
encoder tensors and the MTP head (``model.audio.*``, ``model.visual.*``,
``model.mtp*``) are not part of the Megatron text model and are explicitly
ignored (consumed with a logged count).

The Megatron model is Primus's own ``InklingModel``
(``primus/backends/megatron/core/models/inkling``), built by :func:`build_model`.

Mapping (HF released name -> mcore name, per layer ``L``):

==============================================  ==============================================
``model.llm.embed.weight``                      ``embedding.word_embeddings.weight`` (padded)
``model.llm.embed_norm.weight``                 ``embedding.embed_norm.weight``
``model.llm.norm.weight``                       ``decoder.final_layernorm.weight``
``model.llm.unembed.weight``                    ``output_layer.weight`` (padded)
``layers.L.attn_norm.weight``                   ``decoder.layers.L.input_layernorm.weight``
``layers.L.mlp_norm.weight``                    ``decoder.layers.L.pre_mlp_layernorm.weight``
``layers.L.attn_sconv.weight``                  ``decoder.layers.L.attn_sconv.weight``
``layers.L.mlp_sconv.weight``                   ``decoder.layers.L.mlp_sconv.weight``
``layers.L.attn.wq_du/wk_dv/wv_dv/wr_du``       ``self_attention.linear_q/k/v/r.weight``
``layers.L.attn.wo_ud``                         ``self_attention.linear_proj.weight``
``layers.L.attn.q_norm/k_norm``                 ``self_attention.q_norm_weight/k_norm_weight``
``layers.L.attn.k_sconv/v_sconv``               ``self_attention.k_sconv/v_sconv.weight``
``layers.L.attn.rel_logits_proj.proj``          ``self_attention.rel_proj``
``layers.L.mlp.gate.weight/bias/global_scale``  ``mlp.router_weight/router_expert_bias/router_global_scale``
``layers.L.mlp.experts.w13_weight``             ``mlp.experts_fc1`` (de-interleaved: gate rows, then up)
``layers.L.mlp.experts.w2_weight``              ``mlp.experts_fc2``
``layers.L.mlp.shared_experts.shared_w13``      ``mlp.shared_fc1`` (de-interleaved like w13)
``layers.L.mlp.shared_experts.shared_w2``       ``mlp.shared_fc2``
==============================================  ==============================================

The released ``w13`` tensors store gate and up rows interleaved
(row ``2i`` = gate_i, row ``2i+1`` = up_i); transformers' ``Interleave(dim=1)``
conversion op de-interleaves them, which is reproduced here.
"""

from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common

LLM = "model.llm."


# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
def _text_config(cfg):
    return cfg.get("text_config", cfg)


def derive_megatron_fields(cfg):
    """Translate an Inkling HF config dict (top-level or text_config) into Megatron arg values."""
    t = _text_config(cfg)
    local_ids = list(t.get("local_layer_ids", list(range(0, t["num_hidden_layers"], 2))))
    period = 2
    assert local_ids == list(range(0, t["num_hidden_layers"], period)), (
        f"Inkling converter expects local layers at every {period}nd layer from 0, got {local_ids}")
    assert t.get("swa_head_dim", t["head_dim"]) == t["head_dim"]
    assert t.get("swa_num_attention_heads", t["num_attention_heads"]) == t["num_attention_heads"]
    assert t.get("dense_mlp_idx", 0) == 0, "Inkling converter supports all-MoE layers only"
    assert t.get("use_sconv", True) and t.get("use_embed_norm", True)
    assert t.get("gate_activation", "sigmoid") == "sigmoid" and t.get("norm_after_topk", True)
    assert t.get("final_logit_softcapping") is None
    moe_ffn = t.get("moe_intermediate_size", t["intermediate_size"])
    return dict(
        hidden_size=t["hidden_size"],
        num_attention_heads=t["num_attention_heads"],
        num_layers=t["num_hidden_layers"],
        max_position_embeddings=4096,
        norm_epsilon=t.get("rms_norm_eps", 1e-6),
        rotary_base=10000,
        untie_embeddings_and_output_weights=True,
        ffn_hidden_size=moe_ffn,
        num_query_groups=t["num_key_value_heads"],
        kv_channels=t["head_dim"],
        add_qkv_bias=False,
        vocab_size=t["vocab_size"],
        inkling_local_layer_period=period,
        inkling_sliding_window=t["sliding_window_size"],
        inkling_swa_num_query_groups=t["swa_num_key_value_heads"],
        inkling_rel_dim=t["d_rel"],
        inkling_rel_extent=t["rel_extent"],
        inkling_sconv_kernel_size=t.get("sconv_kernel_size", 4),
        inkling_num_routed_experts=t["n_routed_experts"],
        inkling_moe_router_topk=t["num_experts_per_tok"],
        inkling_num_shared_experts=t["n_shared_experts"],
        inkling_route_scale=float(t["route_scale"]),
        inkling_logits_mup_divisor=float(t.get("logits_mup_width_multiplier", 1.0)),
        inkling_unpadded_vocab_size=t.get("unpadded_vocab_size"),
        inkling_log_scaling_n_floor=t.get("log_scaling_n_floor"),
        inkling_log_scaling_alpha=float(t.get("log_scaling_alpha", 0.1)),
    )


INKLING_KEYS = (
    "inkling_local_layer_period", "inkling_sliding_window", "inkling_swa_num_query_groups", "inkling_rel_dim",
    "inkling_rel_extent", "inkling_sconv_kernel_size", "inkling_num_routed_experts", "inkling_moe_router_topk",
    "inkling_num_shared_experts", "inkling_route_scale", "inkling_logits_mup_divisor",
    "inkling_unpadded_vocab_size", "inkling_log_scaling_n_floor", "inkling_log_scaling_alpha",
)


def build_margs(cfg_fields, args):
    import argparse

    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    m.model_type = "inkling"
    m.hidden_size = cfg_fields["hidden_size"]
    m.num_attention_heads = cfg_fields["num_attention_heads"]
    m.num_layers = cfg_fields["num_layers"]
    m.norm_epsilon = cfg_fields["norm_epsilon"]
    # Megatron's ``--norm-epsilon`` flag stores into ``layernorm_epsilon`` (the field
    # core_transformer_config_from_args copies); set it, or the RMSNorms keep 1e-5.
    m.layernorm_epsilon = cfg_fields["norm_epsilon"]
    m.ffn_hidden_size = cfg_fields["ffn_hidden_size"]
    m.untie_embeddings_and_output_weights = True
    m.max_position_embeddings = max(cfg_fields["max_position_embeddings"], args.seq_length)
    m.seq_length = args.seq_length

    m.group_query_attention = True
    m.num_query_groups = cfg_fields["num_query_groups"]
    m.kv_channels = cfg_fields["kv_channels"]
    m.qk_layernorm = False  # Inkling's per-head q/k norms live inside InklingAttention

    m.normalization = "RMSNorm"
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = False
    # Inkling has no RoPE; the model never applies a rotary embedding.
    m.position_embedding_type = "rope"
    m.rotary_base = cfg_fields["rotary_base"]
    m.apply_rope_fusion = False
    m.attention_dropout = 0.0
    m.hidden_dropout = 0.0

    for k in INKLING_KEYS:
        setattr(m, k, cfg_fields[k])

    # vocab: the released embedding has 201024 rows (> tokenizer vocab); keep them all.
    m.vocab_size = None
    m.padded_vocab_size = cfg_fields["vocab_size"]
    m.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False

    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1

    m.tensor_model_parallel_size = args.tensor_parallel_size
    m.pipeline_model_parallel_size = args.pipeline_parallel_size
    m.expert_model_parallel_size = args.expert_parallel_size
    m.expert_tensor_parallel_size = args.tensor_parallel_size
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
    m.padded_vocab_size = cfg_fields["vocab_size"]
    m.iteration = 1
    return m


def build_model(margs):
    """Primus's InklingModel (eval mode), built from ``margs`` via the same builder training uses."""
    from primus.backends.megatron.core.models.inkling.inkling_builders import inkling_builder

    model = inkling_builder(margs, True, True)
    return model.to(margs.params_dtype).eval()


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping
# ---------------------------------------------------------------------------
def _deinterleave(w, dim):
    """transformers ``Interleave(dim)``: rows (2i, 2i+1) -> (i, n/2 + i), i.e. gate rows first, then up rows."""
    shape = list(w.shape)
    split = shape[:dim] + [shape[dim] // 2, 2] + shape[dim + 1:]
    return w.reshape(split).transpose(dim, dim + 1).reshape(shape).contiguous()


def build_mcore_state_dict(model, store, margs, cfg_fields, log):
    dtype = margs.params_dtype
    padded_vocab = model.state_dict()["embedding.word_embeddings.weight"].shape[0]
    sd = {}
    consumed = set()

    def get(name):
        consumed.add(name)
        return store.get(name, dtype)

    def get_f32(name):
        consumed.add(name)
        return store.get(name, torch.float32)

    sd["embedding.word_embeddings.weight"] = common.pad_vocab(get(LLM + "embed.weight"), padded_vocab)
    sd["embedding.embed_norm.weight"] = get(LLM + "embed_norm.weight")
    sd["decoder.final_layernorm.weight"] = get(LLM + "norm.weight")
    sd["output_layer.weight"] = common.pad_vocab(get(LLM + "unembed.weight"), padded_vocab)

    for L in range(margs.num_layers):
        p = f"decoder.layers.{L}."
        h = f"{LLM}layers.{L}."
        sd[p + "input_layernorm.weight"] = get(h + "attn_norm.weight")
        sd[p + "pre_mlp_layernorm.weight"] = get(h + "mlp_norm.weight")
        sd[p + "attn_sconv.weight"] = get(h + "attn_sconv.weight")
        sd[p + "mlp_sconv.weight"] = get(h + "mlp_sconv.weight")

        a = p + "self_attention."
        sd[a + "linear_q.weight"] = get(h + "attn.wq_du.weight")
        sd[a + "linear_k.weight"] = get(h + "attn.wk_dv.weight")
        sd[a + "linear_v.weight"] = get(h + "attn.wv_dv.weight")
        sd[a + "linear_r.weight"] = get(h + "attn.wr_du.weight")
        sd[a + "linear_proj.weight"] = get(h + "attn.wo_ud.weight")
        sd[a + "q_norm_weight"] = get(h + "attn.q_norm.weight")
        sd[a + "k_norm_weight"] = get(h + "attn.k_norm.weight")
        sd[a + "k_sconv.weight"] = get(h + "attn.k_sconv.weight")
        sd[a + "v_sconv.weight"] = get(h + "attn.v_sconv.weight")
        sd[a + "rel_proj"] = get(h + "attn.rel_logits_proj.proj")

        mm = p + "mlp."
        sd[mm + "router_weight"] = get(h + "mlp.gate.weight")
        sd[mm + "router_expert_bias"] = get_f32(h + "mlp.gate.bias")
        sd[mm + "router_global_scale"] = get(h + "mlp.gate.global_scale").reshape(1)
        sd[mm + "experts_fc1"] = _deinterleave(get(h + "mlp.experts.w13_weight"), 1)
        sd[mm + "experts_fc2"] = get(h + "mlp.experts.w2_weight")
        sd[mm + "shared_fc1"] = _deinterleave(get(h + "mlp.shared_experts.shared_w13_weight"), 1)
        sd[mm + "shared_fc2"] = get(h + "mlp.shared_experts.shared_w2_weight")
        log(f"  mapped layer {L:2d} ({'local' if L % cfg_fields['inkling_local_layer_period'] == 0 else 'global'})")

    # Non-text tensors (audio / vision encoders, MTP head): not part of the text model.
    ignored = sorted(k for k in store.all_keys() if not k.startswith(LLM))
    consumed.update(ignored)
    # Layers beyond a reduced num_layers (proxy runs) are intentionally not mapped.
    skipped = sorted(k for k in store.all_keys() if k.startswith(LLM + "layers.")
                     and int(k[len(LLM + "layers."):].split(".")[0]) >= margs.num_layers)
    consumed.update(skipped)
    log(f"  ignored {len(ignored)} non-text tensors (audio/vision/mtp), {len(skipped)} tensors of unused layers")
    return sd, consumed


# ---------------------------------------------------------------------------
# In-process entrypoint
# ---------------------------------------------------------------------------
def convert(hf_dir, save_dir, *, dtype="bf16", tensor_parallel_size=1, pipeline_parallel_size=1,
            expert_parallel_size=1, seq_length=4096, make_vocab_size_divisible_by=128, master_port=29563,
            log=None):
    """Convert an Inkling HF checkpoint to a legacy Megatron torch ckpt (single process, CPU)."""
    if log is None:

        def log(*a):
            print("[inkling-convert]", *a, flush=True)

    common.ensure_megatron_on_path()
    common.assert_bridge_free("entry")
    args = SimpleNamespace(hf_dir=hf_dir, save_dir=save_dir, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
                           pipeline_parallel_size=pipeline_parallel_size, expert_parallel_size=expert_parallel_size,
                           seq_length=seq_length, make_vocab_size_divisible_by=make_vocab_size_divisible_by)
    cfg_fields = derive_megatron_fields(common.read_hf_config(hf_dir))
    margs = build_margs(cfg_fields, args)
    margs = common.init_megatron_single_process(margs, master_port, log=log)
    common.assert_bridge_free("post-init")
    model = build_model(margs)
    store = common.SafetensorsStore(hf_dir)
    sd, consumed = build_mcore_state_dict(model, store, margs, cfg_fields, log)
    common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
    common.assert_bridge_free("post-save")
    return save_dir
