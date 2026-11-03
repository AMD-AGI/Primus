###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native HF -> Megatron-Core loader for thinkingmachines/Inkling (``inkling_mm_model``).

The text backbone (``model.llm.*``) and the multi-token-prediction head (``model.mtp.*``, in
``mtp.safetensors``: transformers ignores it, so it is read here directly when the index does not
list it) are mapped; the audio / vision towers are out of scope and skipped. The model is Primus' ``InklingModel``
(``model_type: inkling``), built by :func:`build_model`.

Mapping (HF checkpoint key -> InklingModel parameter):

    model.llm.embed.weight                 -> embedding.word_embeddings.weight (vocab padded)
    model.llm.embed_norm.weight            -> embed_norm.weight
    model.llm.norm.weight                  -> decoder.final_layernorm.weight
    model.llm.unembed.weight               -> output_layer.weight (vocab padded)
    layers.L.attn_norm / mlp_norm          -> input_layernorm / pre_mlp_layernorm
    layers.L.attn.w{q_du,k_dv,v_dv,o_ud,r_du} -> self_attention.{q,k,v,o,r}_proj
    layers.L.attn.{q,k}_norm               -> self_attention.{q,k}_norm
    layers.L.attn.{k,v}_sconv [c,1,k]      -> self_attention.{k,v}_sconv.weight [c,k]
    layers.L.attn.rel_logits_proj.proj     -> self_attention.rel_proj
    layers.L.{attn,mlp}_sconv              -> {attn,mlp}_sconv.weight
    layers.L.mlp.gate.{weight,bias,global_scale} -> mlp.gate.{weight,expert_bias,global_scale}
    layers.L.mlp.experts.w13_weight        -> mlp.gate_up_proj (gate/up de-interleaved, as HF's Interleave op)
    layers.L.mlp.experts.w2_weight         -> mlp.down_proj
    layers.L.mlp.shared_experts.shared_w13_weight -> mlp.shared_gate_up_proj (de-interleaved)
    layers.L.mlp.shared_experts.shared_w2_weight  -> mlp.shared_down_proj

MTP depth D (HF ``model.mtp.layers.D.`` -> ``mtp.layers.D.``):

    embed_norm / hidden_norm               -> enorm / hnorm
    input_proj [h, 2h]                     -> eh_proj (as is: InklingMTPLayer concatenates
                                              hidden first, as HF / SGLang)
    transformer_block.*                    -> mtp_model_layer.* (the decoder-layer mapping above,
                                              for the attention, norms and short convs)
    transformer_block.mlp.w13_dn           -> mtp_model_layer.mlp.linear_fc1.weight (de-interleaved)
    transformer_block.mlp.w2_md            -> mtp_model_layer.mlp.linear_fc2.weight
    transformer_block.mlp.global_scale     -> mtp_model_layer.mlp.global_scale
"""

import dataclasses
import os
import re
from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    INKLING_DEFAULT_INIT_SEED,
    InklingTransformerConfig,
)
from primus.core.utils.import_utils import get_model_provider

PREFIX = "model.llm."
MTP_PREFIX = "model.mtp.layers."
MTP_FILE = "mtp.safetensors"
# A decoder-layer key; group 1 is the 0-based layer index.
_LAYER_KEY_RE = re.compile(re.escape(PREFIX) + r"layers\.(\d+)\.")
# Checkpoint parts this port does not build (the audio / vision towers; it ports the text
# backbone and the MTP depths). Their keys are
# marked consumed-as-skipped, by name, so the completeness check still catches any *llm* key
# that is left unmapped.
SKIPPED_PREFIXES = ("model.audio.", "model.visual.", "model.vision_tower.")
# MTP depth keys; group 1 is the depth.
_MTP_KEY_RE = re.compile(re.escape(MTP_PREFIX) + r"(\d+)\.")


def _text(cfg):
    return cfg.get("text_config", cfg)


# HF text_config keys the port needs; a missing one is an error (no silent fallback).
REQUIRED_KEYS = (
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "route_scale",
    "dense_mlp_idx",
    "d_rel",
    "rel_extent",
    "sliding_window_size",
    "local_layer_ids",
    "swa_num_attention_heads",
    "swa_num_key_value_heads",
    "swa_head_dim",
    "log_scaling_n_floor",
    "log_scaling_alpha",
    "sconv_kernel_size",
    "use_embed_norm",
    "logits_mup_width_multiplier",
    "unpadded_vocab_size",
    "dense_intermediate_size",
)
# Variant switches and the only values this port implements; any other stated value raises ValueError.
SUPPORTED_VALUES = {
    "dense_mlp_idx": (0,),
    "attention_bias": (False,),
    "tie_word_embeddings": (False,),
    "final_logit_softcapping": (None,),
    "gate_activation": ("sigmoid",),
    "use_sconv": (True,),
    "shared_expert_sink": (True,),
    "use_gate_bias": (True,),
    "use_global_scale": (True,),
    "mtp_chain_hidden_post_norm": (False,),
    "norm_after_topk": (True,),
    "q_bias": (False,),
    "o_bias": (False,),
}


def _mtp_fields(cfg):
    """MTP fields from the top-level ``mtp_config`` (HF ``InklingMMConfig``; SGLang canonicalizes
    the same keys onto ``text_config.mtp_*``, also accepted). No ``mtp_config``: no MTP."""
    t = _text(cfg)
    mc = cfg.get("mtp_config") or {}
    depths = mc.get("num_nextn_predict_layers", mc.get("n_layers", t.get("num_nextn_predict_layers")))
    chain = mc.get("chain_hidden_post_norm", t.get("mtp_chain_hidden_post_norm", False))
    if chain not in (False, None):
        raise ValueError("inkling converter: mtp chain_hidden_post_norm=True is not supported")
    depths = int(depths or 0)
    local_ids = mc.get("local_layer_ids", t.get("mtp_local_layer_ids")) or []
    local_ids = [int(i) for i in local_ids]
    if any(not 0 <= i < depths for i in local_ids):
        raise ValueError(f"inkling converter: mtp local_layer_ids {local_ids} outside [0, {depths})")
    extent = mc.get("local_extent", t.get("mtp_local_extent"))
    # SGLang refuses MTP sliding kv heads / head dim other than the trunk's sliding layers'
    for k in ("mtp_swa_num_key_value_heads", "mtp_swa_head_dim"):
        base = t[k.replace("mtp_", "", 1)]
        if t.get(k) not in (None, base):
            raise ValueError(f"inkling converter: {k}={t[k]} differs from the trunk's {base}")
    return dict(
        mtp_num_layers=depths or None,
        inkling_mtp_local_layer_ids=local_ids,
        inkling_mtp_local_extent=None if extent is None else int(extent),
        inkling_mtp_swa_num_attention_heads=t.get("mtp_swa_num_attention_heads"),
        inkling_mtp_chain_hidden_post_norm=False,
    )


def derive_megatron_fields(cfg):
    """Translate the Inkling HF config (``text_config``) into Megatron arg values.

    The standard fields describe the global (full-attention) layers; the sliding
    layers' shapes and every Inkling-only knob go in ``inkling_*``. Every needed key
    must be present (``REQUIRED_KEYS``) and every variant switch must hold a value the
    port implements (``SUPPORTED_VALUES``): anything else raises ``ValueError``.
    """
    t = _text(cfg)
    missing = [k for k in REQUIRED_KEYS if k not in t]
    if missing:
        raise ValueError(f"inkling converter: HF text_config lacks {missing}")
    # A variant key the config does not state takes the value HF's InklingConfig / modeling code
    # uses when it is absent (embed / unembed are separate tensors, the projections have no bias),
    # which is the first entry of SUPPORTED_VALUES; a stated value must be a supported one.
    unsupported = {k: t[k] for k, ok in SUPPORTED_VALUES.items() if k in t and t[k] not in ok}
    if unsupported:
        raise ValueError(f"inkling converter: unsupported Inkling variant {unsupported}")
    nl = t["num_hidden_layers"]
    local_ids = [int(i) for i in t["local_layer_ids"]]
    if any(not 0 <= i < nl for i in local_ids):
        raise ValueError(f"inkling converter: local_layer_ids {local_ids} outside [0, {nl})")
    moe_inter = t["intermediate_size"]
    # context length: HF's InklingConfig maps ``model_max_length`` to max_position_embeddings
    # (a key that is present but null counts as absent)
    max_pos = t.get("max_position_embeddings")
    if max_pos is None:
        max_pos = t.get("model_max_length")
    if max_pos is None:
        raise ValueError(
            "inkling converter: HF text_config states neither max_position_embeddings nor model_max_length"
        )
    if "rope_theta" in t or "rope_scaling" in t or "rope_parameters" in t:
        raise ValueError(
            "inkling converter: the config states RoPE settings, but Inkling has no rotary embedding"
        )
    return dict(
        hidden_size=t["hidden_size"],
        num_attention_heads=t["num_attention_heads"],
        num_layers=nl,
        max_position_embeddings=int(max_pos),
        norm_epsilon=t["rms_norm_eps"],
        untie_embeddings_and_output_weights=True,  # tie_word_embeddings=True is refused above
        ffn_hidden_size=moe_inter,
        num_query_groups=t["num_key_value_heads"],
        kv_channels=t["head_dim"],
        add_qkv_bias=False,  # attention_bias=True is refused above
        # ---- MoE (standard Megatron fields) ----
        num_experts=t["n_routed_experts"],
        moe_router_topk=t["num_experts_per_tok"],
        moe_ffn_hidden_size=moe_inter,
        # ---- Inkling-only ----
        inkling_num_shared_experts=t["n_shared_experts"],
        inkling_route_scale=float(t["route_scale"]),
        inkling_dense_mlp_idx=int(t["dense_mlp_idx"]),
        inkling_d_rel=t["d_rel"],
        inkling_rel_extent=t["rel_extent"],
        inkling_sliding_window=t["sliding_window_size"],
        inkling_local_layer_ids=local_ids,
        inkling_swa_num_attention_heads=t["swa_num_attention_heads"],
        inkling_swa_num_query_groups=t["swa_num_key_value_heads"],
        inkling_swa_kv_channels=t["swa_head_dim"],
        inkling_log_scaling_n_floor=t["log_scaling_n_floor"],
        inkling_log_scaling_alpha=float(t["log_scaling_alpha"]),
        inkling_sconv_kernel_size=t["sconv_kernel_size"],
        inkling_use_embed_norm=bool(t["use_embed_norm"]),
        inkling_logits_mup_width_multiplier=float(t["logits_mup_width_multiplier"]),
        inkling_unpadded_vocab_size=t["unpadded_vocab_size"],
        # ---- MTP (dense SwiGLU MLP in each depth) ----
        inkling_dense_intermediate_size=int(t["dense_intermediate_size"]),
        inkling_dense_use_global_scale=bool(t.get("use_global_scale", True)),
        **_mtp_fields(cfg),
    )


def _init_seed(args):
    """The init seed only seeds weights the checkpoint then overwrites; take the caller's (``seed``) if
    given (0 is a valid seed), else the config default."""
    seed = getattr(args, "seed", None)
    return INKLING_DEFAULT_INIT_SEED if seed is None else int(seed)


# Every inkling_* field of the config comes from the HF config except inkling_init_seed (the
# caller's seed), inkling_ignored_label_id (a training policy) and the kernel switches
# (inkling_sconv_impl, inkling_attention_impl: numerically close either way), all set by build_margs. The list
# is derived, so a new config field cannot be silently missed.
INKLING_RUNTIME_KEYS = (
    "inkling_init_seed",
    "inkling_ignored_label_id",
    "inkling_sconv_impl",
    "inkling_attention_impl",
)
INKLING_KEYS = [
    f.name
    for f in dataclasses.fields(InklingTransformerConfig)
    if f.name.startswith("inkling_") and f.name not in INKLING_RUNTIME_KEYS
]
_INKLING_DEFAULTS = {f.name: f.default for f in dataclasses.fields(InklingTransformerConfig)}


PARAMS_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _params_dtype(dtype):
    """The checkpoint dtype for ``--dtype``; anything else is refused (no silent fp32 fallback)."""
    if dtype not in PARAMS_DTYPES:
        choices = ", ".join(repr(k) for k in sorted(PARAMS_DTYPES))
        raise ValueError(f"inkling converter: dtype {dtype!r} is not one of {choices}")
    return PARAMS_DTYPES[dtype]


def build_margs(cfg_fields, args):
    """Megatron args for the single-process conversion.

    The converter writes one unsharded (TP = PP = EP = ETP = 1) legacy ``torch`` checkpoint
    (``mp_rank_00``). Megatron does *not* reshard a legacy checkpoint at load: train with
    TP = PP = EP = ETP = 1 (the stacked expert tensors are saved whole, not split per EP rank), or
    first re-save the checkpoint as ``torch_dist`` with a TP = PP = EP = ETP = 1 run and load that
    one with the training layout. Other conversion sizes are refused here.
    """
    import argparse

    sizes = (args.tensor_parallel_size, args.pipeline_parallel_size, args.expert_parallel_size)
    if sizes != (1, 1, 1):
        raise ValueError(
            "inkling converter: writes an unsharded legacy checkpoint only (TP=PP=EP=1), "
            f"got TP/PP/EP={sizes}. "
            "Megatron cannot reshard a legacy torch checkpoint at load: convert with TP=PP=EP=1 and train "
            "with TP=PP=EP=1, or re-save it as torch_dist from such a run before changing TP/PP/EP"
        )

    params_dtype = _params_dtype(args.dtype)  # refuse an unknown dtype before any other work

    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    m.hidden_size = cfg_fields["hidden_size"]
    m.num_attention_heads = cfg_fields["num_attention_heads"]
    m.num_layers = cfg_fields["num_layers"]
    m.norm_epsilon = cfg_fields["norm_epsilon"]
    m.layernorm_epsilon = cfg_fields["norm_epsilon"]  # upstream dest of --norm-epsilon
    m.ffn_hidden_size = cfg_fields["ffn_hidden_size"]
    m.untie_embeddings_and_output_weights = cfg_fields["untie_embeddings_and_output_weights"]
    m.max_position_embeddings = max(cfg_fields["max_position_embeddings"], args.seq_length)
    m.seq_length = args.seq_length
    m.group_query_attention = True
    m.num_query_groups = cfg_fields["num_query_groups"]
    m.kv_channels = cfg_fields["kv_channels"]
    m.qk_layernorm = False  # Inkling's per-head q/k RMSNorm lives in InklingAttention
    m.normalization = "RMSNorm"
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = cfg_fields["add_qkv_bias"]
    m.position_embedding_type = "none"  # Inkling has no rotary embedding (relative-position bias instead)
    m.apply_rope_fusion = False

    m.num_experts = cfg_fields["num_experts"]
    m.moe_router_topk = cfg_fields["moe_router_topk"]
    m.moe_ffn_hidden_size = cfg_fields["moe_ffn_hidden_size"]
    m.moe_grouped_gemm = False
    m.moe_router_load_balancing_type = "none"  # InklingMoE refuses aux losses
    m.moe_aux_loss_coeff = 0.0
    m.moe_router_score_function = "sigmoid"
    for k in INKLING_KEYS:
        setattr(m, k, cfg_fields[k])
    m.mtp_num_layers = cfg_fields["mtp_num_layers"]
    m.inkling_init_seed = _init_seed(args)
    m.inkling_ignored_label_id = None  # a real (HF) tokenizer: a label past the HF vocab is a bug
    # kernel switches: the config defaults (fused short conv when causal_conv1d imports, eager
    # attention); the caller's args may override them
    for k in ("inkling_sconv_impl", "inkling_attention_impl"):
        setattr(m, k, getattr(args, k, None) or _INKLING_DEFAULTS[k])

    m.vocab_size = None
    m.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False

    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1

    m.tensor_model_parallel_size = 1  # the only layout the converter writes (refused above otherwise)
    m.pipeline_model_parallel_size = 1
    m.expert_model_parallel_size = 1
    m.expert_tensor_parallel_size = 1
    m.context_parallel_size = 1
    m.sequence_parallel = False
    m.world_size = 1
    m.rank = 0
    m.params_dtype = params_dtype
    m.bf16 = args.dtype == "bf16"
    m.fp16 = args.dtype == "fp16"
    m.use_cpu_initialization = True
    m.perform_initialization = False
    m.transformer_impl = "local"
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
    """Build Primus' InklingModel (``model_type: inkling``) on the current device.

    ``build_model(margs)`` is the native-converter hook signature (callers that build a port's own
    model pass the Megatron args); the provider reads the same args from Megatron's globals, which
    ``init_megatron_single_process`` installed, so ``margs`` itself is not read here."""
    del margs

    provider = get_model_provider(model_type="inkling")
    return provider(pre_process=True, post_process=True)


def deinterleave(t, dim):
    """HF ``Interleave(dim)``: rows stored as (gate0, up0, gate1, up1, ...) -> (gate..., up...)."""
    n = t.shape[dim]
    shape = list(t.shape)
    t = t.reshape(*shape[:dim], n // 2, 2, *shape[dim + 1 :]).transpose(dim, dim + 1)
    return t.reshape(shape).contiguous()


def _place(sd: "_PlacingDict", msd, prefix=""):
    """Copy each mapped tensor under ``prefix`` straight into the model's own storage and hand back
    that storage (frees the checkpoint copies layer by layer; the caller's load_state_dict is then a
    self-copy, so no second full-size copy of the weights is held). Only the keys added since the
    last call are scanned (``sd`` is a ``_PlacingDict``), so placing layer by layer stays linear in
    the number of layers. A key the model does not have is left for load_state_dict to report; a
    key the model has with another shape is refused."""
    new = sd.take_new(prefix)
    with torch.no_grad():
        for k in new:
            v = sd[k]
            if k not in msd:
                continue
            if tuple(msd[k].shape) != tuple(v.shape):
                raise ValueError(
                    f"inkling converter: {k} has shape {tuple(v.shape)}, model expects "
                    f"{tuple(msd[k].shape)}"
                )
            msd[k].copy_(v)
            dict.__setitem__(sd, k, msd[k])


class _PlacingDict(dict):
    """The mapped state dict: remembers the keys set since ``_place`` last took them."""

    def __init__(self):
        super().__init__()
        self._new = {}

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self._new[k] = None

    def take_new(self, prefix=""):
        taken = [k for k in self._new if k.startswith(prefix)]
        for k in taken:
            del self._new[k]
        return taken


def _map_inkling_block(sd, get, h, p):
    """One Inkling block's attention, norms and short convs: HF ``h`` (``attn.*``, ``attn_norm``,
    ``mlp_norm``, ``attn_sconv``, ``mlp_sconv``) -> Megatron ``p``. Shared by the decoder layers
    and the MTP depths' ``transformer_block`` so the two mappings cannot drift. ``get(name,
    dtype=None)`` reads (and marks consumed) one HF tensor; the MLP is mapped by the caller (MoE
    in the decoder, dense in MTP)."""
    a, pa = h + "attn.", p + "self_attention."
    sd[p + "input_layernorm.weight"] = get(h + "attn_norm.weight")
    sd[p + "pre_mlp_layernorm.weight"] = get(h + "mlp_norm.weight")
    sd[p + "attn_sconv.weight"] = get(h + "attn_sconv.weight", torch.float32).squeeze(1)
    sd[p + "mlp_sconv.weight"] = get(h + "mlp_sconv.weight", torch.float32).squeeze(1)
    for src, dst in (
        ("wq_du", "q_proj"),
        ("wk_dv", "k_proj"),
        ("wv_dv", "v_proj"),
        ("wo_ud", "o_proj"),
        ("wr_du", "r_proj"),
    ):
        sd[pa + dst + ".weight"] = get(a + src + ".weight")
    sd[pa + "q_norm.weight"] = get(a + "q_norm.weight")
    sd[pa + "k_norm.weight"] = get(a + "k_norm.weight")
    sd[pa + "k_sconv.weight"] = get(a + "k_sconv.weight", torch.float32).squeeze(1)
    sd[pa + "v_sconv.weight"] = get(a + "v_sconv.weight", torch.float32).squeeze(1)
    sd[pa + "rel_proj"] = get(a + "rel_logits_proj.proj")


def build_mcore_state_dict(model, store, margs, cfg_fields, log, allow_truncated=True):
    """HF tensors -> the Megatron state dict of ``model`` (HF layer L -> ``decoder.layers.L``).

    The checkpoint's layer indices must be contiguous from 0. A model with more layers than the
    checkpoint is refused. A model with fewer layers (a truncated proxy, built by tests and the
    parity tooling) is mapped with a WARNING when ``allow_truncated``; ``convert`` passes
    ``allow_truncated=False``, so a conversion that would save fewer layers than the checkpoint
    (for example, a wrong ``num_hidden_layers``) fails."""
    dtype = margs.params_dtype
    msd = model.state_dict()
    padded_vocab = msd["embedding.word_embeddings.weight"].shape[0]
    sd, consumed = _PlacingDict(), set()  # _place scans only the keys added since its last call

    def get(name, dt=None):
        consumed.add(PREFIX + name)
        return store.get(PREFIX + name, dt or dtype)

    sd["embedding.word_embeddings.weight"] = common.pad_vocab(get("embed.weight"), padded_vocab)
    # Mapped by the use_embed_norm flag, read off the built model (InklingModel builds embed_norm
    # iff config.inkling_use_embed_norm), not by what the checkpoint happens to hold; a clear
    # error when the two disagree.
    has_embed_norm = store.has(PREFIX + "embed_norm.weight")
    if "embed_norm.weight" in msd:
        if not has_embed_norm:
            raise ValueError(
                f"the model uses embed_norm (use_embed_norm: true) but the checkpoint has no {PREFIX}embed_norm.weight"
            )
        sd["embed_norm.weight"] = get("embed_norm.weight")
    elif has_embed_norm:
        raise ValueError(
            f"the model has no embed_norm (use_embed_norm: false) but the checkpoint has {PREFIX}embed_norm.weight"
        )
    sd["decoder.final_layernorm.weight"] = get("norm.weight")
    sd["output_layer.weight"] = common.pad_vocab(get("unembed.weight"), padded_vocab)

    indices = sorted({int(m.group(1)) for m in map(_LAYER_KEY_RE.match, store.all_keys()) if m})
    hf_layers = len(indices)
    if indices != list(range(hf_layers)):
        gaps = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise ValueError(
            f"inkling converter: the checkpoint's layers are not contiguous (missing layer(s) {gaps}; "
            "a shard may be missing)"
        )
    if margs.num_layers > hf_layers:
        raise ValueError(
            f"inkling converter: the model builds {margs.num_layers} layers, the checkpoint has {hf_layers}"
        )
    if margs.num_layers < hf_layers:
        if not allow_truncated:
            raise ValueError(
                f"inkling converter: the model builds {margs.num_layers} layers but the checkpoint has "
                f"{hf_layers}; a conversion must cover every layer (check num_hidden_layers in config.json)"
            )
        log(
            f"inkling: WARNING: truncated proxy: building HF layers 0..{margs.num_layers - 1} of "
            f"{hf_layers} (no layer offset); layers {margs.num_layers}.. are not loaded"
        )
    # HF layer L -> decoder.layers.L: the model is always built whole (first layer is HF layer 0).
    for L in range(margs.num_layers):
        h, p = f"layers.{L}.", f"decoder.layers.{L}."
        _map_inkling_block(sd, get, h, p)

        g, pm = h + "mlp.", p + "mlp."
        sd[pm + "gate.weight"] = get(g + "gate.weight")
        sd[pm + "gate.global_scale"] = get(g + "gate.global_scale", torch.float32).reshape(1).to(dtype)
        sd[pm + "gate.expert_bias"] = get(g + "gate.bias", torch.float32)
        # EP = 1 (build_margs): all experts are local
        sd[pm + "gate_up_proj"] = deinterleave(get(g + "experts.w13_weight"), 1)
        sd[pm + "down_proj"] = get(g + "experts.w2_weight")
        sd[pm + "shared_gate_up_proj"] = deinterleave(get(g + "shared_experts.shared_w13_weight"), 1)
        sd[pm + "shared_down_proj"] = get(g + "shared_experts.shared_w2_weight")
        _place(sd, msd, p)

    _map_mtp(sd, msd, store, consumed, margs, log)
    _skip(store, consumed, margs, log)
    return _finish(sd, msd, consumed, cfg_fields, margs, log)


class MTPStore:
    """``store`` plus ``<hf_dir>/mtp.safetensors`` when the checkpoint index does not list it
    (transformers ignores the MTP file, so an index written by it may omit its keys). The MTP
    file's key set is read once (``safe_open.keys()`` builds a new list on each call)."""

    def __init__(self, store):
        self._store = store
        self._extra = None
        self._extra_keys = frozenset()
        path = os.path.join(getattr(store, "hf_dir", "") or "", MTP_FILE)
        if not any(k.startswith(MTP_PREFIX) for k in store.all_keys()) and os.path.isfile(path):
            from safetensors import safe_open

            self._extra = safe_open(path, framework="pt", device="cpu")
            self._extra_keys = frozenset(self._extra.keys())

    def __getattr__(self, name):
        return getattr(self._store, name)

    def all_keys(self):
        return set(self._store.all_keys()) | self._extra_keys

    def has(self, name):
        return name in self._extra_keys or self._store.has(name)

    def get(self, name, dtype=None):
        if name in self._extra_keys:
            t = self._extra.get_tensor(name)
            return t.to(dtype) if dtype is not None else t
        return self._store.get(name, dtype)


def _map_mtp(sd, msd, store, consumed, margs, log):
    """Map HF MTP depth D (``model.mtp.layers.D.``) to ``mtp.layers.D.`` (module docstring).

    The checkpoint must hold exactly the depths the model builds when it builds any; a model
    without MTP (``mtp_num_layers`` unset) skips the checkpoint's MTP keys with a log line."""
    dtype = margs.params_dtype
    depths = int(getattr(margs, "mtp_num_layers", None) or 0)
    if not depths:
        n = 0
        for k in store.all_keys():
            if k.startswith("model.mtp.") and k not in consumed:
                consumed.add(k)
                n += 1
        if n:
            log(f"inkling: skipped {n} MTP tensors (mtp_num_layers is not set)")
        return
    found = sorted({int(m.group(1)) for m in map(_MTP_KEY_RE.match, store.all_keys()) if m})
    if found != list(range(depths)):
        raise ValueError(
            f"inkling converter: the model builds MTP depths 0..{depths - 1}, the checkpoint has {found} "
            f"(is {MTP_FILE} present?)"
        )

    def get(name, dt=None):
        consumed.add(name)
        return store.get(name, dt or dtype)

    for d in range(depths):
        h, p = f"{MTP_PREFIX}{d}.", f"mtp.layers.{d}."
        sd[p + "enorm.weight"] = get(h + "embed_norm.weight")
        sd[p + "hnorm.weight"] = get(h + "hidden_norm.weight")
        sd[p + "eh_proj.weight"] = get(h + "input_proj.weight")
        b, pl = h + "transformer_block.", p + "mtp_model_layer."
        _map_inkling_block(sd, get, b, pl)
        g, pm = b + "mlp.", pl + "mlp."
        # HF's dense MLP stores w13 interleaved (gate rows even, up rows odd), as the experts
        sd[pm + "linear_fc1.weight"] = deinterleave(get(g + "w13_dn.weight"), 0)
        sd[pm + "linear_fc2.weight"] = get(g + "w2_md.weight")
        if pm + "global_scale" in msd or store.has(g + "global_scale"):
            sd[pm + "global_scale"] = get(g + "global_scale", torch.float32).reshape(1).to(dtype)
        _place(sd, msd, p)
    log(f"inkling: mapped {depths} MTP depths from model.mtp.layers.*")


def _skip(store, consumed, margs, log):
    """Mark the checkpoint keys this port intentionally does not map as consumed, and log them.

    Skipped: the audio / vision towers (``SKIPPED_PREFIXES``), and the
    backbone layers at or past ``margs.num_layers``. ``convert`` builds the full HF depth, so
    there are none there. A caller that builds fewer layers builds a *truncated proxy*: the
    model is the first ``margs.num_layers`` HF layers (``build_mcore_state_dict`` maps HF
    layer L to ``decoder.layers.L`` with no offset, and the sliding / global layer kinds use
    the same 0-based ids). This is not a pipeline stage: the model is always built whole
    (``pre_process`` and ``post_process``), so a loader that wants layers ``k..`` of the HF
    model must not use this function. Any other key stays unconsumed, and
    ``load_validate_save`` refuses to save.
    """
    skipped = {}
    for k in store.all_keys():
        if k in consumed:
            continue
        part = next((p for p in SKIPPED_PREFIXES if k.startswith(p)), None)
        if part is None:
            mt = _LAYER_KEY_RE.match(k)
            if mt and int(mt.group(1)) >= margs.num_layers:
                part = f"{PREFIX}layers.>={margs.num_layers}"
        if part is not None:
            skipped[part] = skipped.get(part, 0) + 1
            consumed.add(k)
    for part, n in sorted(skipped.items()):
        log(
            f"inkling: skipped {n} checkpoint tensors under {part} (audio / vision tower: not part of this port)"
        )


def _finish(sd, msd, consumed, cfg_fields, margs, log):
    _place(sd, msd)  # the tensors not placed per layer (embedding, final norm, output layer)
    if cfg_fields.get("inkling_unpadded_vocab_size") is None:
        padded = msd["output_layer.weight"].shape[0]
        log(f"inkling: unpadded_vocab_size is null; all {padded} padded logits stay unmasked")
    log(
        f"inkling: mapped {len(sd)} tensors; {len(consumed)} checkpoint keys consumed "
        f"(mapped + skipped); {margs.num_layers} decoder layers built"
    )
    return sd, consumed


def convert(
    hf_dir,
    save_dir,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    expert_parallel_size=1,
    dtype="bf16",
    seq_length=4096,
    make_vocab_size_divisible_by=128,
    master_port=29561,
    log=print,
    seed=INKLING_DEFAULT_INIT_SEED,
):
    common.assert_bridge_free("pre-init")
    args = SimpleNamespace(
        hf_dir=hf_dir,
        save_dir=save_dir,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        seq_length=seq_length,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        seed=seed,
    )
    cfg_fields = derive_megatron_fields(common.read_hf_config(hf_dir))
    margs = build_margs(cfg_fields, args)
    margs = common.init_megatron_single_process(margs, master_port, log=log)
    model = build_model(margs)
    store = MTPStore(common.SafetensorsStore(hf_dir))
    sd, consumed = build_mcore_state_dict(model, store, margs, cfg_fields, log, allow_truncated=False)
    common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
    common.assert_bridge_free("post-save")
    return save_dir
