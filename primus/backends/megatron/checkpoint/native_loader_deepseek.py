###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for the DeepSeek-V2 /
V3 family (MLA + DeepSeekMoE).

Ported verbatim (mapping semantics unchanged) from the previously in-submodule
``tools/checkpoint/loader_deepseek.py``; only the process plumbing changed:

  * ``main()`` (argparse + one-shot script) -> importable ``convert(...)`` that
    the Primus conversion hook calls in-process, AFTER applying the
    ``phase="convert"`` patches.
  * ``from model_provider import model_provider`` / ``from gpt_builders import
    gpt_builder`` -> Primus' ``get_model_provider('gpt')`` (same underlying root
    modules, but resolved without depending on the ``tools/checkpoint`` layout).

Why a self-contained builder (not a ``convert.py --loader`` plugin): the
loader<->saver queue protocol has no representation for MLA (linear_q/kv
down/up projections + kv/q layernorms) or DeepSeekMoE shared experts. Building
the mcore model directly and copying weights onto it is both simpler and
provably exact (validated via ``load_state_dict`` missing/unexpected == 0).

The HF<->mcore name mapping is the one documented by the Megatron-Bridge reference
(``deepseek/common.py``, read for reference only, NOT imported). All projections
map by a pure rename (+ gate/up concat for SwiGLU fc1); no RoPE permutation is
required because mcore's default rotate_half matches HF DeepSeek's.
"""

import argparse
from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common


# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
def derive_megatron_fields(cfg):
    """Translate a DeepSeek-V2/V3 HF config dict into Megatron arg values."""
    num_layers = cfg["num_hidden_layers"]
    first_k_dense = cfg.get("first_k_dense_replace", 0)
    n_shared = cfg.get("n_shared_experts", 0) or 0
    moe_inter = cfg.get("moe_intermediate_size", cfg["intermediate_size"])

    # DeepSeek-V3 uses aux-loss-free routing with a correction bias
    # (topk_method == "noaux_tc" and scoring_func == "sigmoid"); V2 does not.
    topk_method = cfg.get("topk_method", "greedy")
    scoring_func = cfg.get("scoring_func", "softmax")
    enable_expert_bias = topk_method == "noaux_tc"

    rope_scaling = cfg.get("rope_scaling") or {}

    fields = dict(
        num_layers=num_layers,
        hidden_size=cfg["hidden_size"],
        ffn_hidden_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        kv_channels=cfg.get("v_head_dim", 128),
        norm_epsilon=cfg.get("rms_norm_eps", 1e-6),
        vocab_size=cfg["vocab_size"],
        untie_embeddings_and_output_weights=not cfg.get("tie_word_embeddings", False),
        # ---- MLA ----
        multi_latent_attention=True,
        q_lora_rank=cfg.get("q_lora_rank", None),
        kv_lora_rank=cfg["kv_lora_rank"],
        qk_head_dim=cfg["qk_nope_head_dim"],
        qk_pos_emb_head_dim=cfg["qk_rope_head_dim"],
        v_head_dim=cfg["v_head_dim"],
        qk_layernorm=True,
        rope_type="yarn" if rope_scaling.get("type") == "yarn" else "rope",
        rotary_base=int(cfg.get("rope_theta", 10000)),
        rotary_scaling_factor=float(rope_scaling.get("factor", 1.0)),
        mscale=float(rope_scaling.get("mscale", 1.0)),
        mscale_all_dim=float(rope_scaling.get("mscale_all_dim", 0.0)),
        original_max_position_embeddings=int(
            rope_scaling.get("original_max_position_embeddings", 4096)
        ),
        # ---- MoE ----
        num_experts=cfg["n_routed_experts"],
        moe_ffn_hidden_size=moe_inter,
        moe_shared_expert_intermediate_size=(moe_inter * n_shared) if n_shared else None,
        moe_router_topk=cfg["num_experts_per_tok"],
        moe_layer_freq=[0] * first_k_dense + [1] * (num_layers - first_k_dense),
        moe_router_score_function="sigmoid" if scoring_func == "sigmoid" else "softmax",
        moe_router_enable_expert_bias=enable_expert_bias,
        moe_router_topk_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
        moe_router_num_groups=cfg.get("n_group", None),
        moe_router_group_topk=cfg.get("topk_group", None),
        # meta
        _num_shared_experts=n_shared,
        _first_k_dense=first_k_dense,
        _mtp_num_layers=cfg.get("num_nextn_predict_layers", 0) or 0,
    )
    return fields


def build_margs(cfg_fields, args):
    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    # architecture / shapes
    for k in (
        "num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads",
        "kv_channels", "norm_epsilon", "untie_embeddings_and_output_weights",
        "multi_latent_attention", "q_lora_rank", "kv_lora_rank", "qk_head_dim",
        "qk_pos_emb_head_dim", "v_head_dim", "qk_layernorm", "rope_type",
        "rotary_base", "rotary_scaling_factor", "mscale", "mscale_all_dim",
        "original_max_position_embeddings", "num_experts", "moe_ffn_hidden_size",
        "moe_shared_expert_intermediate_size", "moe_router_topk", "moe_layer_freq",
        "moe_router_score_function", "moe_router_enable_expert_bias",
        "moe_router_topk_scaling_factor", "moe_router_num_groups",
        "moe_router_group_topk",
    ):
        setattr(m, k, cfg_fields[k])

    # IMPORTANT: leave vocab_size / padded_vocab_size UNSET so build_tokenizer()
    # drives padded_vocab_size from the tokenizer's real vocab, matching how
    # Primus native SFT sizes the embedding (avoids a load_state_dict mismatch).
    m.vocab_size = None
    m.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False
    m.normalization = "RMSNorm"
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = False
    m.position_embedding_type = "rope"
    m.apply_rope_fusion = False  # MLA does not support rope fusion
    m.attention_softmax_in_fp32 = True

    # MoE execution knobs (do not affect saved weight shapes)
    m.moe_grouped_gemm = True
    m.moe_token_dispatcher_type = "alltoall"
    m.moe_router_load_balancing_type = "seq_aux_loss"
    m.moe_router_pre_softmax = True
    m.moe_aux_loss_coeff = 0.001
    m.moe_shared_expert_overlap = False
    m.moe_permute_fusion = False

    # sizes required by the parser/validator
    m.seq_length = args.seq_length
    m.max_position_embeddings = args.seq_length
    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1

    # parallel / precision / init
    m.tensor_model_parallel_size = args.tensor_parallel_size
    m.pipeline_model_parallel_size = args.pipeline_parallel_size
    m.expert_model_parallel_size = args.expert_parallel_size
    m.expert_tensor_parallel_size = args.tensor_parallel_size
    m.context_parallel_size = 1
    m.sequence_parallel = False
    m.world_size = (
        args.tensor_parallel_size * args.pipeline_parallel_size * args.expert_parallel_size
    )
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

    # checkpoint saving
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
    m.iteration = 1
    return m


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping
# ---------------------------------------------------------------------------
def build_mcore_state_dict(model, store, margs, cfg_fields, log):
    dtype = margs.params_dtype
    padded_vocab = model.state_dict()["embedding.word_embeddings.weight"].shape[0]
    q_lora_rank = cfg_fields["q_lora_rank"]
    num_experts = cfg_fields["num_experts"]
    n_shared = cfg_fields["_num_shared_experts"]
    enable_bias = cfg_fields["moe_router_enable_expert_bias"]
    moe_layer_freq = cfg_fields["moe_layer_freq"]

    def cat_gate_up(gate, up):
        return torch.cat([gate, up], dim=0).contiguous()

    sd = {}
    consumed = set()

    def get(name):
        consumed.add(name)
        return store.get(name, dtype)

    # ---- non-layer ----
    sd["embedding.word_embeddings.weight"] = common.pad_vocab(
        get("model.embed_tokens.weight"), padded_vocab
    )
    sd["decoder.final_layernorm.weight"] = get("model.norm.weight")
    if margs.untie_embeddings_and_output_weights:
        sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)

    # ---- layers ----
    for L in range(margs.num_layers):
        p = f"decoder.layers.{L}."
        h = f"model.layers.{L}."

        sd[p + "input_layernorm.weight"] = get(h + "input_layernorm.weight")

        # ---- MLA attention ----
        if q_lora_rank is None:
            sd[p + "self_attention.linear_q_proj.weight"] = get(h + "self_attn.q_proj.weight")
        else:
            sd[p + "self_attention.linear_q_down_proj.weight"] = get(h + "self_attn.q_a_proj.weight")
            sd[p + "self_attention.linear_q_up_proj.weight"] = get(h + "self_attn.q_b_proj.weight")
            sd[p + "self_attention.linear_q_up_proj.layer_norm_weight"] = get(
                h + "self_attn.q_a_layernorm.weight"
            )
        sd[p + "self_attention.linear_kv_down_proj.weight"] = get(
            h + "self_attn.kv_a_proj_with_mqa.weight"
        )
        sd[p + "self_attention.linear_kv_up_proj.weight"] = get(h + "self_attn.kv_b_proj.weight")
        sd[p + "self_attention.linear_kv_up_proj.layer_norm_weight"] = get(
            h + "self_attn.kv_a_layernorm.weight"
        )
        sd[p + "self_attention.linear_proj.weight"] = get(h + "self_attn.o_proj.weight")

        # ---- MLP: dense or MoE ----
        is_moe = moe_layer_freq[L] == 1
        if not is_moe:
            sd[p + "mlp.linear_fc1.layer_norm_weight"] = get(h + "post_attention_layernorm.weight")
            sd[p + "mlp.linear_fc1.weight"] = cat_gate_up(
                get(h + "mlp.gate_proj.weight"), get(h + "mlp.up_proj.weight")
            )
            sd[p + "mlp.linear_fc2.weight"] = get(h + "mlp.down_proj.weight")
        else:
            sd[p + "pre_mlp_layernorm.weight"] = get(h + "post_attention_layernorm.weight")
            sd[p + "mlp.router.weight"] = get(h + "mlp.gate.weight")
            if enable_bias:
                sd[p + "mlp.router.expert_bias"] = get(h + "mlp.gate.e_score_correction_bias")
            for e in range(num_experts):
                sd[p + f"mlp.experts.linear_fc1.weight{e}"] = cat_gate_up(
                    get(h + f"mlp.experts.{e}.gate_proj.weight"),
                    get(h + f"mlp.experts.{e}.up_proj.weight"),
                )
                sd[p + f"mlp.experts.linear_fc2.weight{e}"] = get(
                    h + f"mlp.experts.{e}.down_proj.weight"
                )
            if n_shared:
                sd[p + "mlp.shared_experts.linear_fc1.weight"] = cat_gate_up(
                    get(h + "mlp.shared_experts.gate_proj.weight"),
                    get(h + "mlp.shared_experts.up_proj.weight"),
                )
                sd[p + "mlp.shared_experts.linear_fc2.weight"] = get(
                    h + "mlp.shared_experts.down_proj.weight"
                )
        if L % 5 == 0 or L == margs.num_layers - 1:
            log(f"  mapped layer {L:2d} ({'MoE' if is_moe else 'dense'})")

    return sd, consumed


# ---------------------------------------------------------------------------
# In-process entrypoint
# ---------------------------------------------------------------------------
def convert(
    hf_dir,
    save_dir,
    *,
    dtype="bf16",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    expert_parallel_size=1,
    seq_length=4096,
    make_vocab_size_divisible_by=128,
    master_port=29561,
    log=None,
):
    """Convert a DeepSeek-V2/V3 HF checkpoint to a legacy Megatron torch ckpt.

    Single process, CPU-only, ~1 min. Assumes the ``phase="convert"`` patches
    have already been applied in this process (the conversion hook does so).
    """
    if log is None:
        def log(*a):
            print("[ds-convert]", *a, flush=True)

    common.ensure_megatron_on_path()
    common.assert_bridge_free("entry")

    args = SimpleNamespace(
        hf_dir=hf_dir,
        save_dir=save_dir,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        seq_length=seq_length,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
    )

    log(f"reading HF config from {hf_dir}")
    cfg = common.read_hf_config(hf_dir)
    cfg_fields = derive_megatron_fields(cfg)
    log(
        f"arch: L={cfg_fields['num_layers']} h={cfg_fields['hidden_size']} "
        f"heads={cfg_fields['num_attention_heads']} q_lora={cfg_fields['q_lora_rank']} "
        f"kv_lora={cfg_fields['kv_lora_rank']} experts={cfg_fields['num_experts']} "
        f"shared={cfg_fields['_num_shared_experts']} topk={cfg_fields['moe_router_topk']} "
        f"first_k_dense={cfg_fields['_first_k_dense']} mtp={cfg_fields['_mtp_num_layers']} "
        f"expert_bias={cfg_fields['moe_router_enable_expert_bias']}"
    )
    if cfg_fields["_mtp_num_layers"]:
        log(
            f"WARNING: model declares {cfg_fields['_mtp_num_layers']} MTP layer(s); "
            "MTP weights are NOT converted by this script (out of scope for V2-Lite)."
        )

    margs = build_margs(cfg_fields, args)
    log(
        f"megatron: TP={margs.tensor_model_parallel_size} PP={margs.pipeline_model_parallel_size} "
        f"EP={margs.expert_model_parallel_size} dtype={margs.params_dtype} "
        f"rope_type={margs.rope_type}"
    )

    margs = common.init_megatron_single_process(margs, master_port, log=log)
    common.assert_bridge_free("post-init")

    log("building mcore GPTModel on CPU ...")
    model = common.build_mcore_gpt_model(margs)
    log("model built.")

    store = common.SafetensorsStore(hf_dir)
    log("mapping HF safetensors -> mcore parameters ...")
    sd, consumed = build_mcore_state_dict(model, store, margs, cfg_fields, log)

    common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
    common.assert_bridge_free("post-save")
    return save_dir
