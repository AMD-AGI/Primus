###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for the Muse Spark
text tower (``model_type: muse_spark``).

Same plumbing as ``native_loader_qwen3.py``: derive Megatron args from the HF
``config.json``, build the model on CPU through ``get_model_provider
("muse_spark")``, copy the safetensors onto its parameters, require an exact
mapping and save. The HF names are those of transformers'
``MuseSparkForConditionalGeneration`` (``model.language_model.*`` +
``lm_head``); a text-only export rooted at ``model.*`` is accepted too.

Mapping notes:

  * ``linear_qkv`` packs, per KV group, the group's query heads, the matching
    rows of ``gate_proj`` (the attention output gate), then one key and one
    value head.
  * ``input_layernorm`` / ``pre_feedforward_layernorm`` / ``pre_expert_norm``
    (HF ``CenteredRMSNorm``) and ``post_feedforward_layernorm`` /
    ``post_expert_norm`` (``InputScaledRMSNorm``) already store the zero-
    centred weight Megatron uses, so they copy as is. The final ``norm`` is a
    plain scaled RMSNorm, so it becomes ``w - 1``.
  * The 3-D ``experts.gate_up_proj`` / ``experts.down_proj`` are split per
    expert; ``gate_up`` keeps gate first, which is Megatron's GLU order.
  * The vision and audio towers and the vision projector are not part of the
    text tower; their tensors are skipped and counted.

The model is built on one rank (TP=PP=EP=1). The default ``ckpt_format="torch_dist"``
writes a distributed checkpoint that Megatron reshards to the training TP / PP / EP
on load, which is what the multi-node PP x EP runs need; ``"torch"`` writes the
single-rank legacy layout of the other native converters.
"""

from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common

_SKIPPED_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.audio_tower.",
)


# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
def derive_megatron_fields(cfg):
    """Translate a Muse Spark HF config dict into Megatron arg values."""
    from primus.backends.megatron.core.models.muse_spark.muse_spark_ops import (
        global_attention_layer_mask,
    )

    tc = cfg.get("text_config", cfg)
    num_layers = tc.get("num_hidden_layers", 62)
    interval = 4

    is_global = global_attention_layer_mask(num_layers, interval)
    layer_types = tc.get("layer_types") or [
        "full_attention" if g else "sliding_attention" for g in is_global
    ]
    if layer_types != ["full_attention" if g else "sliding_attention" for g in is_global]:
        raise ValueError(
            "layer_types does not follow Muse Spark's pattern (full attention every "
            f"{interval}th layer counted back from the last): {layer_types}"
        )
    rope = tc.get("rope_parameters") or {"rope_type": "default", "rope_theta": 500000.0}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"only default RoPE is supported, got {rope}")
    rope_theta = rope["rope_theta"]
    layer_rope_theta = tc.get("layer_rope_theta")
    if layer_rope_theta is not None:
        expected = [0 if g else rope_theta for g in is_global]
        if [float(t) for t in layer_rope_theta] != [float(t) for t in expected]:
            raise ValueError(
                "layer_rope_theta must be 0 exactly on the full-attention layers and the "
                f"global theta elsewhere, got {layer_rope_theta}"
            )
    mlp_layer_types = tc.get("mlp_layer_types") or ["sparse"] * num_layers
    if any(t != "sparse" for t in mlp_layer_types):
        raise ValueError("dense MLP layers are not supported; every layer must be sparse")
    if tc.get("attention_bias", False):
        raise ValueError("attention_bias is not supported")
    if tc.get("hidden_act", "silu") != "silu":
        raise ValueError(f"hidden_act must be silu, got {tc.get('hidden_act')}")

    hidden = tc.get("hidden_size", 8192)
    heads = tc.get("num_attention_heads", 128)
    return dict(
        num_layers=num_layers,
        hidden_size=hidden,
        ffn_hidden_size=tc.get("intermediate_size", 16384),
        num_attention_heads=heads,
        num_query_groups=tc.get("num_key_value_heads", 16),
        kv_channels=tc.get("head_dim") or hidden // heads,
        vocab_size=tc.get("vocab_size", 202048),
        max_position_embeddings=tc.get("max_position_embeddings", 1048576),
        rms_norm_eps=tc.get("rms_norm_eps", 1e-5),
        post_norm_eps=tc.get("post_norm_eps", 1e-8),
        rotary_base=rope_theta,
        sliding_window=tc.get("sliding_window", 2048),
        global_attn_interval=interval,
        qk_scale_factor=tc.get("qk_scale_factor", 5.4730064863875),
        output_multiplier=tc.get("output_multiplier", 0.1767766952966369),
        final_logit_softcapping=tc.get("final_logit_softcapping", 20.0),
        residual_gate_temperature=tc.get("residual_gate_temperature", 0.3),
        num_experts=tc.get("num_local_experts", 256),
        moe_router_topk=tc.get("num_experts_per_tok", 8),
        moe_ffn_hidden_size=tc.get("moe_intermediate_size", 4096),
        moe_latent_size=tc.get("moe_hidden_size", 4096),
        routed_scaling_factor=tc.get("routed_scaling_factor", 1.0),
        untie_embeddings_and_output_weights=not tc.get("tie_word_embeddings", False),
    )


def build_margs(fields, args):
    import argparse
    import os

    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    # architecture / shapes
    m.num_layers = fields["num_layers"]
    m.hidden_size = fields["hidden_size"]
    m.ffn_hidden_size = fields["ffn_hidden_size"]
    m.num_attention_heads = fields["num_attention_heads"]
    m.group_query_attention = True
    m.num_query_groups = fields["num_query_groups"]
    m.kv_channels = fields["kv_channels"]
    m.untie_embeddings_and_output_weights = fields["untie_embeddings_and_output_weights"]
    m.max_position_embeddings = min(fields["max_position_embeddings"], max(args.seq_length, 4096))
    m.seq_length = min(args.seq_length, m.max_position_embeddings)

    # attention: weightless QK-norm (the query norm applies the scale),
    # sigmoid output gate, hybrid SWA / NoPE from the muse_* keys
    m.qk_layernorm = False
    m.qk_l2_norm = True
    m.attention_output_gate = True
    m.position_embedding_type = "rope"
    m.rotary_base = fields["rotary_base"]
    m.rotary_percent = 1.0
    m.rotary_interleaved = False
    m.use_rope_scaling = False
    m.apply_rope_fusion = False
    m.muse_global_attn_interval = fields["global_attn_interval"]
    m.muse_sliding_window = fields["sliding_window"]
    m.muse_qk_scale_factor = fields["qk_scale_factor"]

    # norms, residual gates, output
    m.normalization = "RMSNorm"
    m.layernorm_zero_centered_gamma = True
    m.layernorm_epsilon = fields["rms_norm_eps"]
    m.muse_post_norm_eps = fields["post_norm_eps"]
    m.muse_residual_gate_temperature = fields["residual_gate_temperature"]
    m.muse_residual_gate_eps = 1.0e-3
    m.muse_output_multiplier = fields["output_multiplier"]
    m.final_logit_softcapping = fields["final_logit_softcapping"]
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = False
    m.hidden_dropout = 0.0
    m.attention_dropout = 0.0

    # MoE: every layer, latent experts, sigmoid router with expert bias
    m.num_experts = fields["num_experts"]
    m.moe_layer_freq = 1
    m.moe_router_topk = fields["moe_router_topk"]
    m.moe_ffn_hidden_size = fields["moe_ffn_hidden_size"]
    m.moe_latent_size = fields["moe_latent_size"]
    m.moe_shared_expert_intermediate_size = None
    m.moe_router_score_function = "sigmoid"
    m.moe_router_pre_softmax = False
    m.moe_router_topk_scaling_factor = fields["routed_scaling_factor"]
    m.moe_router_enable_expert_bias = True
    m.moe_router_dtype = "fp32"
    m.moe_router_load_balancing_type = "none"
    m.moe_aux_loss_coeff = 0.0
    m.moe_token_dispatcher_type = "alltoall"
    m.moe_grouped_gemm = True
    m.moe_use_legacy_grouped_gemm = False
    m.moe_permute_fusion = False
    m.mtp_num_layers = None
    # Read by PrimusTopKRouter when the converted model runs a forward.
    m.router_logit_softcapping = None
    m.enable_primus_turbo = False
    m.moe_use_fused_router_with_aux_score = False
    m.moe_router_force_load_balancing = False

    # vocab: the HF tokenizer when the checkpoint ships one, else a
    # NullTokenizer sized so that padded_vocab_size covers the HF table.
    m.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    if os.path.isfile(os.path.join(args.hf_dir, "tokenizer.json")) or os.path.isfile(
        os.path.join(args.hf_dir, "tokenizer.model")
    ):
        m.vocab_size = None
        m.tokenizer_type = "HuggingFaceTokenizer"
        m.tokenizer_model = args.hf_dir
    else:
        m.tokenizer_type = "NullTokenizer"
        # NullTokenizer appends one EOD id.
        m.vocab_size = fields["vocab_size"] - 1
    m.trust_remote_code = False

    # sizes / batching required by validator
    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1

    # single rank, CPU build
    m.tensor_model_parallel_size = 1
    m.pipeline_model_parallel_size = 1
    m.expert_model_parallel_size = 1
    m.expert_tensor_parallel_size = 1
    m.context_parallel_size = 1
    m.sequence_parallel = False
    m.world_size = 1
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
    m.bias_gelu_fusion = False
    m.bias_swiglu_fusion = False
    m.masked_softmax_fusion = False
    m.bias_dropout_fusion = False
    m.apply_query_key_layer_scaling = False

    # checkpoint saving
    m.save = args.save_dir
    m.ckpt_format = args.ckpt_format
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
def _hf_text_prefix(store):
    for prefix in ("model.language_model.", "model."):
        if store.has(prefix + "embed_tokens.weight"):
            return prefix
    raise RuntimeError("no Muse Spark text tower found (expected model.language_model.embed_tokens.weight)")


def build_mcore_state_dict(model, store, margs, log):
    dtype = margs.params_dtype
    padded_vocab = model.state_dict()["embedding.word_embeddings.weight"].shape[0]

    nh = margs.num_attention_heads
    ng = margs.num_query_groups
    hdim = margs.kv_channels
    hidden = margs.hidden_size
    num_experts = margs.num_experts
    assert nh % ng == 0, f"num_heads ({nh}) must be divisible by num_query_groups ({ng})"

    sd = {}
    consumed = set()

    def get(name, as_dtype=dtype):
        consumed.add(name)
        return store.get(name, as_dtype)

    t = _hf_text_prefix(store)

    def fused_qkv_gate(h):
        q = get(h + "self_attn.q_proj.weight")
        gate = get(h + "self_attn.gate_proj.weight")
        k = get(h + "self_attn.k_proj.weight")
        v = get(h + "self_attn.v_proj.weight")
        return torch.cat(
            [
                q.reshape(ng, (nh // ng) * hdim, hidden),
                gate.reshape(ng, (nh // ng) * hdim, hidden),
                k.reshape(ng, hdim, hidden),
                v.reshape(ng, hdim, hidden),
            ],
            dim=1,
        ).reshape(-1, hidden)

    # ---- non-layer ----
    sd["embedding.word_embeddings.weight"] = common.pad_vocab(get(t + "embed_tokens.weight"), padded_vocab)
    # Plain scaled RMSNorm in HF; Megatron's final norm is zero-centred.
    sd["decoder.final_layernorm.weight"] = (get(t + "norm.weight", torch.float32) - 1.0).to(dtype)
    if margs.untie_embeddings_and_output_weights:
        sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)
    elif store.has("lm_head.weight"):
        consumed.add("lm_head.weight")

    # ---- layers ----
    for L in range(margs.num_layers):
        p = f"decoder.layers.{L}."
        h = f"{t}layers.{L}."

        sd[p + "self_attention.linear_qkv.layer_norm_weight"] = get(h + "input_layernorm.weight")
        sd[p + "self_attention.linear_qkv.weight"] = fused_qkv_gate(h)
        sd[p + "self_attention.linear_proj.weight"] = get(h + "self_attn.o_proj.weight")
        sd[p + "post_self_attn_residual_gate.gate"] = get(h + "post_attention_residual_gate.gate")

        sd[p + "pre_mlp_layernorm.weight"] = get(h + "pre_feedforward_layernorm.weight")
        sd[p + "mlp.router.weight"] = get(h + "mlp.gate.weight")
        sd[p + "mlp.router.expert_bias"] = get(h + "mlp.gate.e_score_correction_bias", torch.float32)
        sd[p + "mlp.fc1_latent_proj.weight"] = get(h + "mlp.pre_expert_proj.weight")
        sd[p + "mlp.pre_expert_norm.weight"] = get(h + "mlp.pre_expert_norm.weight")
        gate_up = get(h + "mlp.experts.gate_up_proj")
        down = get(h + "mlp.experts.down_proj")
        assert gate_up.shape[0] == num_experts and down.shape[0] == num_experts
        for e in range(num_experts):
            sd[p + f"mlp.experts.linear_fc1.weight{e}"] = gate_up[e].contiguous()
            sd[p + f"mlp.experts.linear_fc2.weight{e}"] = down[e].contiguous()
        sd[p + "mlp.post_expert_norm.weight"] = get(h + "mlp.experts.post_expert_norm.weight")
        sd[p + "mlp.fc2_latent_proj.weight"] = get(h + "mlp.post_expert_proj.weight")
        sd[p + "post_mlp_norm.weight"] = get(h + "post_feedforward_layernorm.weight")
        sd[p + "post_mlp_residual_gate.gate"] = get(h + "post_feedforward_residual_gate.gate")

        if L % 8 == 0 or L == margs.num_layers - 1:
            log(f"  mapped layer {L:2d}")

    skipped = {k for k in store.all_keys() if k.startswith(_SKIPPED_PREFIXES)}
    if skipped:
        log(f"skipping {len(skipped)} vision / audio / projector tensors (not part of the text tower)")
    consumed |= skipped
    return sd, consumed


# ---------------------------------------------------------------------------
# In-process entrypoint
# ---------------------------------------------------------------------------
def build_model(hf_dir, *, dtype="bf16", seq_length=4096, make_vocab_size_divisible_by=128,
                save_dir=None, ckpt_format="torch_dist", master_port=29563, log=print):
    """Init single-rank Megatron from the HF config and build the CPU model.

    Returns ``(model, margs)``. Split out of :func:`convert` so tests can map the
    weights and run the converted model in the same process.
    """
    common.ensure_megatron_on_path()
    common.assert_bridge_free("entry")

    args = SimpleNamespace(
        hf_dir=hf_dir,
        save_dir=save_dir,
        dtype=dtype,
        seq_length=seq_length,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        ckpt_format=ckpt_format,
    )
    cfg = common.read_hf_config(hf_dir)
    fields = derive_megatron_fields(cfg)
    log(
        f"arch: L={fields['num_layers']} h={fields['hidden_size']} heads={fields['num_attention_heads']} "
        f"kv_groups={fields['num_query_groups']} head_dim={fields['kv_channels']} "
        f"experts={fields['num_experts']} topk={fields['moe_router_topk']} "
        f"expert_ffn={fields['moe_ffn_hidden_size']} latent={fields['moe_latent_size']} "
        f"swa={fields['sliding_window']} vocab={fields['vocab_size']}"
    )
    margs = build_margs(fields, args)
    margs = common.init_megatron_single_process(margs, master_port, log=log)
    common.assert_bridge_free("post-init")

    from primus.core.utils.import_utils import get_model_provider

    model = get_model_provider(model_type="muse_spark")(pre_process=True, post_process=True)
    model = model.to(margs.params_dtype)
    # The router keeps expert_bias in fp32 during training; undo the blanket cast
    # so the HF fp32 e_score_correction_bias is stored without rounding.
    for module in model.modules():
        if getattr(module, "expert_bias", None) is not None:
            module.expert_bias.data = module.expert_bias.data.float()
    return model, margs


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
    ckpt_format="torch_dist",
    master_port=29563,
    log=None,
):
    """Convert a Muse Spark HF checkpoint to a Megatron checkpoint.

    Single process, CPU build. The full 816B model needs about 1.7 TB of host
    memory in bf16. Only ``torch_dist`` loads into a PP / EP training run.
    """
    if log is None:

        def log(*a):
            print("[muse-spark-convert]", *a, flush=True)

    if (tensor_parallel_size, pipeline_parallel_size, expert_parallel_size) != (1, 1, 1):
        raise ValueError(
            "the Muse Spark converter builds the model on one rank; keep TP = PP = EP = 1 and "
            "ckpt_format='torch_dist', and Megatron reshards it to the training TP / PP / EP on load"
        )

    log(f"reading HF config from {hf_dir}")
    model, margs = build_model(
        hf_dir,
        dtype=dtype,
        seq_length=seq_length,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        save_dir=save_dir,
        ckpt_format=ckpt_format,
        master_port=master_port,
        log=log,
    )
    log("model built.")

    store = common.SafetensorsStore(hf_dir)
    log("mapping HF safetensors -> mcore parameters ...")
    sd, consumed = build_mcore_state_dict(model, store, margs, log)

    common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
    common.assert_bridge_free("post-save")
    return save_dir
