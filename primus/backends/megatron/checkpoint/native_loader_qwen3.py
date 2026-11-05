###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for the Qwen3 family
(dense *and* MoE: GQA + QK-layernorm, and for Qwen3-MoE a per-layer router with
N SwiGLU experts and no shared expert).

Ported from the previously in-submodule ``tools/checkpoint/loader_qwen3.py``.
The weight-mapping *math* is unchanged (same interleaved fused-QKV layout, same
gate/up SwiGLU concat, same tokenizer-driven vocab padding); only the plumbing
moved from the ``convert.py`` loader<->saver queue protocol to a self-contained
single-process build-model-and-copy path (like the DeepSeek converter). This
removes the dependency on the ``tools/checkpoint`` ``schema_core`` / ``saver_base``
files entirely -- the QK-layernorm schema keys and per-expert placement that
those files were patched for are now expressed directly against the mcore model's
own parameters, and validated exactly via ``load_state_dict``.
"""

from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common


# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
def derive_megatron_fields(cfg):
    """Translate a Qwen3 / Qwen3-MoE HF config dict into Megatron arg values."""
    fields = dict(
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_layers=cfg["num_hidden_layers"],
        max_position_embeddings=cfg["max_position_embeddings"],
        norm_epsilon=cfg["rms_norm_eps"],
        rotary_base=cfg.get("rope_theta", 1000000.0),
        untie_embeddings_and_output_weights=not cfg.get("tie_word_embeddings", False),
        ffn_hidden_size=cfg["intermediate_size"],
        # GQA
        num_query_groups=cfg["num_key_value_heads"],
        # Qwen3 sets an explicit head_dim that is NOT necessarily hidden/num_heads.
        kv_channels=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        # Qwen3 (dense and MoE) always applies QK-layernorm.
        qk_layernorm=True,
    )
    # ---- MoE (qwen3_moe) ----
    if cfg.get("num_experts", None):
        fields.update(
            num_experts=cfg["num_experts"],
            moe_router_topk=cfg["num_experts_per_tok"],
            moe_ffn_hidden_size=cfg["moe_intermediate_size"],
            _norm_topk_prob=cfg.get("norm_topk_prob", True),
        )
    else:
        fields["num_experts"] = None
    return fields


def build_margs(cfg_fields, args):
    import argparse

    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    is_moe = bool(cfg_fields["num_experts"])

    # architecture / shapes
    m.hidden_size = cfg_fields["hidden_size"]
    m.num_attention_heads = cfg_fields["num_attention_heads"]
    m.num_layers = cfg_fields["num_layers"]
    m.norm_epsilon = cfg_fields["norm_epsilon"]
    m.ffn_hidden_size = cfg_fields["ffn_hidden_size"]
    m.untie_embeddings_and_output_weights = cfg_fields["untie_embeddings_and_output_weights"]
    m.max_position_embeddings = max(cfg_fields["max_position_embeddings"], args.seq_length)
    m.seq_length = args.seq_length

    # GQA + QK-layernorm
    m.group_query_attention = True
    m.num_query_groups = cfg_fields["num_query_groups"]
    m.kv_channels = cfg_fields["kv_channels"]
    m.qk_layernorm = True

    # norm / activation / bias
    m.normalization = "RMSNorm"
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = False  # Qwen3 uses qk-layernorm, NOT qkv bias (unlike Qwen2.5)
    m.position_embedding_type = "rope"
    m.rotary_base = cfg_fields["rotary_base"]
    m.apply_rope_fusion = False

    # ---- MoE ----
    if is_moe:
        m.num_experts = cfg_fields["num_experts"]
        m.moe_router_topk = cfg_fields["moe_router_topk"]
        m.moe_ffn_hidden_size = cfg_fields["moe_ffn_hidden_size"]
        m.moe_router_pre_softmax = False
        m.moe_router_load_balancing_type = "none"
        m.moe_token_dispatcher_type = "alltoall"
        # Keep experts un-fused (SequentialMLP) so the saved checkpoint uses
        # ``mlp.experts.local_experts.{i}.linear_fc{1,2}.weight`` keys -- the same
        # layout the verified reference conversion used. Grouped-GEMM is a
        # training-time perf choice, not a storage-format requirement; the actual
        # emitted key format is auto-detected from the built model below.
        m.moe_grouped_gemm = False
        if cfg_fields.get("_norm_topk_prob", True):
            m.moe_router_topk_scaling_factor = 1.0

    # vocab: leave unset so build_tokenizer() drives padded_vocab_size from the
    # tokenizer's real vocab (matches native SFT sizing -> clean load_state_dict).
    m.vocab_size = None
    m.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False

    # sizes / batching required by validator
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
def _detect_expert_key_style(model, num_layers):
    """Return 'grouped' | 'sequential' | None based on the built model's keys."""
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
    is_moe = bool(cfg_fields["num_experts"])
    num_experts = cfg_fields["num_experts"] if is_moe else 0

    nh = margs.num_attention_heads
    ng = margs.num_query_groups
    hdim = margs.kv_channels
    hidden = margs.hidden_size
    assert nh % ng == 0, f"num_heads ({nh}) must be divisible by num_query_groups ({ng})"

    expert_style = _detect_expert_key_style(model, margs.num_layers) if is_moe else None
    if is_moe and expert_style is None:
        raise RuntimeError("could not detect MoE expert key layout on the built model")

    sd = {}
    consumed = set()

    def get(name):
        consumed.add(name)
        return store.get(name, dtype)

    def cat_gate_up(gate, up):
        return torch.cat([gate, up], dim=0).contiguous()

    def fused_qkv(layer):
        q = get(f"model.layers.{layer}.self_attn.q_proj.weight")
        k = get(f"model.layers.{layer}.self_attn.k_proj.weight")
        v = get(f"model.layers.{layer}.self_attn.v_proj.weight")
        # Interleave into the mcore linear_qkv layout: for each of the ng query
        # groups, emit (nh/ng) query heads then 1 key head then 1 value head.
        return torch.cat(
            [
                q.reshape(ng, (nh // ng) * hdim, hidden),
                k.reshape(ng, hdim, hidden),
                v.reshape(ng, hdim, hidden),
            ],
            dim=1,
        ).reshape(-1, hidden)

    # ---- non-layer ----
    sd["embedding.word_embeddings.weight"] = common.pad_vocab(get("model.embed_tokens.weight"), padded_vocab)
    sd["decoder.final_layernorm.weight"] = get("model.norm.weight")
    if margs.untie_embeddings_and_output_weights:
        sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)
    elif store.has("lm_head.weight"):
        # Tied embeddings (e.g. Qwen3-0.6B): the model shares embedding/output
        # weights, so there is no output_layer.weight to fill. Some HF exports
        # still ship a redundant (tied) lm_head.weight tensor -- mark it consumed
        # so the exact-mapping check does not flag it (the mcore model uses the
        # shared embedding we already set).
        consumed.add("lm_head.weight")

    # ---- layers ----
    for L in range(margs.num_layers):
        p = f"decoder.layers.{L}."
        h = f"model.layers.{L}."

        # input layernorm is fused into the TE qkv linear (layer_norm_weight)
        sd[p + "self_attention.linear_qkv.layer_norm_weight"] = get(h + "input_layernorm.weight")
        sd[p + "self_attention.linear_qkv.weight"] = fused_qkv(L)
        sd[p + "self_attention.q_layernorm.weight"] = get(h + "self_attn.q_norm.weight")
        sd[p + "self_attention.k_layernorm.weight"] = get(h + "self_attn.k_norm.weight")
        sd[p + "self_attention.linear_proj.weight"] = get(h + "self_attn.o_proj.weight")

        if not is_moe:
            # post-attention layernorm is fused into the TE fc1 linear
            sd[p + "mlp.linear_fc1.layer_norm_weight"] = get(h + "post_attention_layernorm.weight")
            sd[p + "mlp.linear_fc1.weight"] = cat_gate_up(
                get(h + "mlp.gate_proj.weight"), get(h + "mlp.up_proj.weight")
            )
            sd[p + "mlp.linear_fc2.weight"] = get(h + "mlp.down_proj.weight")
        else:
            # MoE fc1 is not a TE-LayerNorm linear -> separate pre_mlp_layernorm
            sd[p + "pre_mlp_layernorm.weight"] = get(h + "post_attention_layernorm.weight")
            sd[p + "mlp.router.weight"] = get(h + "mlp.gate.weight")
            for e in range(num_experts):
                fc1 = cat_gate_up(
                    get(h + f"mlp.experts.{e}.gate_proj.weight"),
                    get(h + f"mlp.experts.{e}.up_proj.weight"),
                )
                fc2 = get(h + f"mlp.experts.{e}.down_proj.weight")
                if expert_style == "grouped":
                    sd[p + f"mlp.experts.linear_fc1.weight{e}"] = fc1
                    sd[p + f"mlp.experts.linear_fc2.weight{e}"] = fc2
                else:  # sequential
                    ep = p + f"mlp.experts.local_experts.{e}."
                    sd[ep + "linear_fc1.weight"] = fc1
                    sd[ep + "linear_fc2.weight"] = fc2

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
    master_port=29563,
    log=None,
):
    """Convert a Qwen3 (dense or MoE) HF checkpoint to a legacy Megatron torch ckpt.

    Single process, CPU-only. Assumes the ``phase="convert"`` patches have
    already been applied in this process (the conversion hook does so).
    """
    if log is None:

        def log(*a):
            print("[qwen3-convert]", *a, flush=True)

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
    is_moe = bool(cfg_fields["num_experts"])
    log(
        f"arch: L={cfg_fields['num_layers']} h={cfg_fields['hidden_size']} "
        f"heads={cfg_fields['num_attention_heads']} kv_groups={cfg_fields['num_query_groups']} "
        f"head_dim={cfg_fields['kv_channels']} ffn={cfg_fields['ffn_hidden_size']} "
        f"moe={'yes(E=%d,topk=%d)' % (cfg_fields['num_experts'], cfg_fields['moe_router_topk']) if is_moe else 'no'} "
        f"tie_emb={not cfg_fields['untie_embeddings_and_output_weights']}"
    )

    margs = build_margs(cfg_fields, args)
    log(
        f"megatron: TP={margs.tensor_model_parallel_size} PP={margs.pipeline_model_parallel_size} "
        f"EP={margs.expert_model_parallel_size} dtype={margs.params_dtype}"
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
