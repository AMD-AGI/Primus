###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for OLMo-3 (``model_type: olmo3``,
e.g. allenai/Olmo-3-1025-7B).

OLMo-3 is not a Llama layout, so it is built by Primus' own model code
(``primus/backends/megatron/core/models/olmo3``, ``model_type: olmo3``):

  * post-sublayer RMSNorms (``post_attention_layernorm``, ``post_feedforward_layernorm``) and no
    input / pre-MLP norm, so the TE linears are plain (not layernorm-fused);
  * ``q_norm`` / ``k_norm`` over the full projection width (all heads), held by
    ``Olmo3SelfAttention.{q,k}_norm``;
  * YaRN RoPE on every layer; sliding window on 3 of every 4 layers (no weights involved).

Weight mapping (every HF tensor exactly once):

  model.embed_tokens.weight                         -> embedding.word_embeddings.weight (vocab-padded)
  lm_head.weight                                    -> output_layer.weight (vocab-padded)
  model.norm.weight                                 -> decoder.final_layernorm.weight
  model.layers.L.self_attn.{q,k,v}_proj.weight      -> decoder.layers.L.self_attention.linear_qkv.weight
                                                       (interleaved per GQA group, no RoPE permutation)
  model.layers.L.self_attn.o_proj.weight            -> decoder.layers.L.self_attention.linear_proj.weight
  model.layers.L.self_attn.{q,k}_norm.weight        -> decoder.layers.L.self_attention.{q,k}_norm.weight
  model.layers.L.post_attention_layernorm.weight    -> decoder.layers.L.post_attention_layernorm.weight
  model.layers.L.post_feedforward_layernorm.weight  -> decoder.layers.L.post_feedforward_layernorm.weight
  model.layers.L.mlp.{gate,up}_proj.weight          -> decoder.layers.L.mlp.linear_fc1.weight (concat)
  model.layers.L.mlp.down_proj.weight               -> decoder.layers.L.mlp.linear_fc2.weight
"""

from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common

MODEL_TYPE = "olmo3"


# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
def _rope_scaling(cfg):
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    if (rs.get("rope_type") or rs.get("type")) != "yarn":
        return None
    return rs


def _window_skip_freq(cfg, num_layers):
    """Megatron's window_attn_skip_freq from HF layer_types (1 = sliding, 0 = full)."""
    types = cfg.get("layer_types")
    if not types:
        return None
    pattern = [1 if t == "sliding_attention" else 0 for t in types][:num_layers]
    # The regular OLMo-3 pattern (full every 4th layer) is the integer form, which also fits a reduced depth.
    if all(v == (0 if (i + 1) % 4 == 0 else 1) for i, v in enumerate(pattern)):
        return 4
    return pattern


def derive_megatron_fields(cfg):
    """Translate an OLMo-3 HF config dict into Megatron arg values."""
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    rs = _rope_scaling(cfg)
    fields = dict(
        hidden_size=cfg["hidden_size"],
        num_attention_heads=num_heads,
        num_layers=cfg["num_hidden_layers"],
        max_position_embeddings=cfg["max_position_embeddings"],
        norm_epsilon=cfg.get("rms_norm_eps", 1e-6),
        rotary_base=cfg.get("rope_theta", 10000.0),
        untie_embeddings_and_output_weights=not cfg.get("tie_word_embeddings", False),
        ffn_hidden_size=cfg["intermediate_size"],
        num_query_groups=num_kv_heads,
        kv_channels=cfg.get("head_dim") or cfg["hidden_size"] // num_heads,
        add_qkv_bias=bool(cfg.get("attention_bias", False)),
        sliding_window=cfg.get("sliding_window"),
        layer_types=cfg.get("layer_types"),
        yarn=None,
    )
    if rs is not None:
        fields["yarn"] = dict(
            factor=float(rs["factor"]),
            original_max_position_embeddings=int(rs["original_max_position_embeddings"]),
            beta_fast=float(rs.get("beta_fast", 32.0)),
            beta_slow=float(rs.get("beta_slow", 1.0)),
            attention_factor=rs.get("attention_factor"),
            truncate=bool(rs.get("truncate", True)),
        )
    return fields


def build_margs(cfg_fields, args):
    import argparse

    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    m.model_type = MODEL_TYPE

    # architecture / shapes
    m.hidden_size = cfg_fields["hidden_size"]
    m.num_attention_heads = cfg_fields["num_attention_heads"]
    m.num_layers = cfg_fields["num_layers"]
    # TransformerConfig's field is `layernorm_epsilon` (flag --norm-epsilon); `norm_epsilon` is inert for GPT.
    m.layernorm_epsilon = cfg_fields["norm_epsilon"]
    m.norm_epsilon = cfg_fields["norm_epsilon"]
    m.ffn_hidden_size = cfg_fields["ffn_hidden_size"]
    m.untie_embeddings_and_output_weights = cfg_fields["untie_embeddings_and_output_weights"]
    m.max_position_embeddings = max(cfg_fields["max_position_embeddings"], args.seq_length)
    m.seq_length = args.seq_length

    m.group_query_attention = True
    m.num_query_groups = cfg_fields["num_query_groups"]
    m.kv_channels = cfg_fields["kv_channels"]
    # Full-width QK norm lives in Olmo3SelfAttention; Megatron's per-head qk_layernorm stays off.
    m.qk_layernorm = False

    m.normalization = "RMSNorm"
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = cfg_fields["add_qkv_bias"]
    m.position_embedding_type = "rope"
    m.rotary_base = cfg_fields["rotary_base"]
    m.apply_rope_fusion = False

    # YaRN (Olmo3TransformerConfig fields; the builder switches the rotary embedding to YaRN)
    y = cfg_fields.get("yarn")
    if y is not None:
        m.yarn_rotary_scaling_factor = y["factor"]
        m.yarn_original_max_position_embeddings = y["original_max_position_embeddings"]
        m.yarn_beta_fast = y["beta_fast"]
        m.yarn_beta_slow = y["beta_slow"]
        m.yarn_mscale = None
        m.yarn_mscale_all_dim = None
        m.yarn_correction_range_round_to_int = y["truncate"]
        if y.get("attention_factor") is not None:
            import math

            want = 0.1 * math.log(y["factor"]) + 1.0
            assert abs(float(y["attention_factor"]) - want) < 1e-6, (
                f"YaRN attention_factor {y['attention_factor']} != 0.1*ln(factor)+1 = {want}"
            )

    # Sliding window: window_size[0] in HF sliding_window units (Olmo3TransformerConfig converts for TE).
    if cfg_fields.get("sliding_window"):
        m.window_size = (int(cfg_fields["sliding_window"]), 0)
        m.window_attn_skip_freq = _window_skip_freq(
            {"layer_types": cfg_fields.get("layer_types")}, m.num_layers
        )

    m.num_experts = None

    m.vocab_size = None
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
    m.attention_dropout = 0.0
    m.hidden_dropout = 0.0

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


def build_model(margs):
    """The OLMo-3 model (Primus model code, ``model_type: olmo3``)."""
    from primus.core.utils.import_utils import get_model_provider

    model = get_model_provider(model_type=MODEL_TYPE)(pre_process=True, post_process=True)
    return model.to(margs.params_dtype)


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping
# ---------------------------------------------------------------------------
def build_mcore_state_dict(model, store, margs, cfg_fields, log):
    dtype = margs.params_dtype
    padded_vocab = model.state_dict()["embedding.word_embeddings.weight"].shape[0]

    nh = margs.num_attention_heads
    ng = margs.num_query_groups
    hdim = margs.kv_channels
    hidden = margs.hidden_size
    assert nh % ng == 0, f"num_heads ({nh}) must be divisible by num_query_groups ({ng})"

    sd = {}
    consumed = set()

    def get(name):
        assert name not in consumed, f"HF tensor {name} mapped twice"
        consumed.add(name)
        return store.get(name, dtype)

    def fused_qkv(h):
        q = get(h + "self_attn.q_proj.weight")
        k = get(h + "self_attn.k_proj.weight")
        v = get(h + "self_attn.v_proj.weight")
        return torch.cat(
            [
                q.reshape(ng, (nh // ng) * hdim, hidden),
                k.reshape(ng, hdim, hidden),
                v.reshape(ng, hdim, hidden),
            ],
            dim=1,
        ).reshape(-1, hidden)

    sd["embedding.word_embeddings.weight"] = common.pad_vocab(get("model.embed_tokens.weight"), padded_vocab)
    sd["decoder.final_layernorm.weight"] = get("model.norm.weight")
    if margs.untie_embeddings_and_output_weights:
        sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)
    elif store.has("lm_head.weight"):
        consumed.add("lm_head.weight")

    for L in range(margs.num_layers):
        p = f"decoder.layers.{L}."
        h = f"model.layers.{L}."
        sd[p + "self_attention.linear_qkv.weight"] = fused_qkv(h)
        # Full-width norms: [nh * hdim] in head order, the same order as the query heads above
        # (with ng < nh the query heads stay in HF order: group g holds heads g*nh/ng ...).
        sd[p + "self_attention.q_norm.weight"] = get(h + "self_attn.q_norm.weight")
        sd[p + "self_attention.k_norm.weight"] = get(h + "self_attn.k_norm.weight")
        sd[p + "self_attention.linear_proj.weight"] = get(h + "self_attn.o_proj.weight")
        sd[p + "post_attention_layernorm.weight"] = get(h + "post_attention_layernorm.weight")
        sd[p + "mlp.linear_fc1.weight"] = torch.cat(
            [get(h + "mlp.gate_proj.weight"), get(h + "mlp.up_proj.weight")], dim=0
        ).contiguous()
        sd[p + "mlp.linear_fc2.weight"] = get(h + "mlp.down_proj.weight")
        sd[p + "post_feedforward_layernorm.weight"] = get(h + "post_feedforward_layernorm.weight")
        if L % 8 == 0 or L == margs.num_layers - 1:
            log(f"  mapped layer {L:2d}")

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
    """Convert an OLMo-3 HF checkpoint to a legacy Megatron torch ckpt (single process, CPU)."""
    if log is None:

        def log(*a):
            print("[olmo3-convert]", *a, flush=True)

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
