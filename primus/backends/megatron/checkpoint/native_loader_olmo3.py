###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native, bridge-free HuggingFace -> Megatron-Core converter for OLMo-3
(HF ``model_type: olmo3``, e.g. ``allenai/Olmo-3-1025-7B``).

Layout differences from Llama / Qwen3 (see ``core/models/olmo3``):

* post-norm: ``post_attention_layernorm`` -> ``self_attention.post_attention_layernorm``
  and ``post_feedforward_layernorm`` -> ``mlp.post_feedforward_layernorm``;
  there is no input / pre-MLP norm, so ``linear_qkv`` / ``linear_fc1`` are plain
  (non-LayerNorm) TE linears;
* full-width QK-norm: ``self_attn.q_norm`` (width ``heads * head_dim``) / ``k_norm`` (width
  ``kv_heads * head_dim``) -> ``self_attention.q_norm`` / ``k_norm``;
* sliding window on 3 of 4 layers and YaRN RoPE on the full layers (no weights).
"""

import argparse
import math
from types import SimpleNamespace

import torch

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.checkpoint.native_flow_olmo3 import (
    check_single_shard_layout,
)
from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
    YARN_BETA_FAST_DEFAULT,
    YARN_BETA_SLOW_DEFAULT,
    check_olmo3_unsupported,
    olmo3_window_from_settings,
)

# ---------------------------------------------------------------------------
# Converter runtime (the build / save part is common.set_converter_runtime_args, shared by
# every native converter; the layout and dtype checks are OLMo-3 own)
# ---------------------------------------------------------------------------
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _apply_runtime_margs(m, args) -> None:
    """The converter's runtime Megatron args: 1 iteration of batch 1; TP = PP = EP = 1 in a 1-rank
    world (the converter writes a single-shard checkpoint); one dtype from :data:`DTYPES` (an
    unknown dtype raises; at most one of ``bf16`` / ``fp16`` is True); CPU build without init;
    fusions off; a legacy ``torch`` checkpoint saved to ``args.save_dir`` without optimizer / RNG
    state. The dtype and the layout are checked before ``m`` changes."""
    if args.dtype not in DTYPES:
        raise ValueError(f"converter dtype must be one of {sorted(DTYPES)}, got {args.dtype!r}")
    check_single_shard_layout(
        args.tensor_parallel_size, args.pipeline_parallel_size, args.expert_parallel_size
    )
    m.micro_batch_size = 1
    m.global_batch_size = 1
    m.train_iters = 1

    m.tensor_model_parallel_size = 1
    m.pipeline_model_parallel_size = 1
    m.expert_model_parallel_size = 1
    m.expert_tensor_parallel_size = 1
    m.context_parallel_size = 1
    m.sequence_parallel = False
    m.world_size = 1
    m.rank = 0
    m.bf16 = args.dtype == "bf16"
    m.fp16 = args.dtype == "fp16"
    m.params_dtype = DTYPES[args.dtype]
    common.set_converter_runtime_args(m, args.save_dir)


def _window_fields(cfg):
    """``(sliding_window, window_attn_skip_freq)`` from HF ``sliding_window`` + ``layer_types``.

    ``sliding_window`` is the HF W (keys incl. the current token); the caller stores it as
    ``olmo3_sliding_window`` and as Megatron's ``window_size = (W - 1, 0)``. ``window_attn_skip_freq``
    is an int ``k`` (every k-th, 1-indexed, layer is full attention) when the pattern is
    periodic over all layers, else an explicit 0/1 list (1 = sliding). OLMo-3 has sliding
    layers, so a config without an int ``sliding_window > 1``, without a ``layer_types``
    entry per layer, with unknown layer types, or with no sliding layer or no full_attention
    layer raises ``ValueError``.
    """
    n = cfg["num_hidden_layers"]
    types = cfg.get("layer_types")
    if not isinstance(types, (list, tuple)) or len(types) != n:
        got = len(types) if isinstance(types, (list, tuple)) else types
        raise ValueError(f"OLMo-3 config.json needs layer_types with one entry per layer ({n}), got {got!r}")
    unknown = sorted({t for t in types if t not in _LAYER_TYPES})
    if unknown:
        raise ValueError(f"OLMo-3 layer_types must be {_LAYER_TYPES}, got {unknown}")
    sliding = [t == "sliding_attention" for t in types]
    if not any(sliding):
        raise ValueError("OLMo-3 layer_types has no sliding_attention layer; not an OLMo-3 layout")
    if all(sliding):
        raise ValueError(
            "OLMo-3 layer_types has no full_attention layer; window_attn_skip_freq needs at least one "
            "full attention layer"
        )
    window = cfg.get("sliding_window")
    if not isinstance(window, int) or isinstance(window, bool) or window <= 1:
        raise ValueError(f"OLMo-3 has sliding layers but sliding_window is {window!r} (need an int > 1)")
    for k in range(2, n + 1):
        if all(s == ((i + 1) % k != 0) for i, s in enumerate(sliding)):
            return window, k
    return window, [1 if s else 0 for s in sliding]


_REQUIRED_KEYS = (
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
)
_LAYER_TYPES = ("full_attention", "sliding_attention")


def _check_hf_config(cfg):
    """Refuse configs this converter does not map (no biases, SwiGLU only, OLMo-3 only).

    Raises ``ValueError`` (not ``assert``: the checks must survive ``python -O``).
    """
    mt = cfg.get("model_type")
    if mt != "olmo3":
        raise ValueError(f"native_loader_olmo3 converts model_type 'olmo3', got {mt!r}")
    act = cfg.get("hidden_act", "silu")
    if act != "silu":
        raise ValueError(f"OLMo-3 converter supports hidden_act 'silu' (SwiGLU) only, got {act!r}")
    if cfg.get("attention_bias", False):
        raise ValueError("OLMo-3 converter maps no q/k/v/o biases; attention_bias must be false")
    if cfg.get("tie_word_embeddings", False):
        raise ValueError(
            "OLMo-3 converter supports untied embeddings only (tie_word_embeddings must be false); "
            "the OLMo-3 model refuses untie_embeddings_and_output_weights=false"
        )
    missing = [k for k in _REQUIRED_KEYS if cfg.get(k) is None]
    if missing:
        raise ValueError(f"OLMo-3 config.json is missing required keys: {', '.join(missing)}")
    nh = cfg["num_attention_heads"]
    ng = cfg.get("num_key_value_heads") or nh
    if nh % ng:
        raise ValueError(f"num_attention_heads ({nh}) must be divisible by num_key_value_heads ({ng})")


def _rope_type(params):
    t = params.get("rope_type", params.get("type")) or "default"
    if t not in ("default", "yarn"):
        raise ValueError(f"OLMo-3 converter supports rope_type 'default' or 'yarn', got {t!r}")
    return t


# Keys each supported rope_type may carry. Any other key (``mscale``, ``mscale_all_dim``,
# ``long_factor`` ...) changes the HF math in a way this port does not implement, so it is refused
# instead of dropped. ``truncate`` is an allowed YaRN key, but its value must be true:
# ``_yarn_params`` refuses ``truncate: false``.
_ROPE_KEYS = {
    "default": {"rope_type", "type", "rope_theta"},
    "yarn": {
        "rope_type",
        "type",
        "rope_theta",
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "attention_factor",
        "truncate",
    },
}


def _yarn_params(params, cfg):
    """Normalised YaRN values of one rope-parameter dict, with the HF defaults filled in.

    Refuses what Olmo3DualRotaryEmbedding does not implement: unknown keys (``mscale``,
    ``mscale_all_dim`` ...), ``truncate: false`` (Megatron rounds the correction range, as
    HF's default ``truncate: true``), ``factor < 1``, ``attention_factor <= 0``,
    ``beta_fast < beta_slow``, and, when ``original_max_position_embeddings`` is given, a
    ``factor`` that differs from ``max_position_embeddings / original_max_position_embeddings``.
    """
    t = _rope_type(params)
    extra = sorted(set(params) - _ROPE_KEYS[t])
    if extra:
        raise ValueError(f"OLMo-3 converter: unsupported {t!r} rope parameters {extra}")
    if t != "yarn":
        return None
    if params.get("truncate", True) is not True:
        raise ValueError(
            "OLMo-3 converter: YaRN truncate must be true (Megatron rounds the correction range)"
        )
    factor = params.get("factor")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor < 1.0:
        raise ValueError(f"OLMo-3 YaRN factor must be a number >= 1, got {factor!r}")
    factor = float(factor)
    max_pos = cfg["max_position_embeddings"]
    orig = params.get("original_max_position_embeddings")
    if orig is None:
        orig = max_pos
    elif not math.isclose(max_pos / float(orig), factor, rel_tol=1e-9):
        # also for factor 1.0: HF derives YaRN from this ratio, so 1.0 with orig < max_pos is not "no YaRN"
        raise ValueError(
            f"OLMo-3 YaRN factor {factor} != max_position_embeddings / original_max_position_embeddings "
            f"= {max_pos} / {orig}; refusing an inconsistent rope config"
        )
    beta_fast = float(params["beta_fast"] if params.get("beta_fast") is not None else YARN_BETA_FAST_DEFAULT)
    beta_slow = float(params["beta_slow"] if params.get("beta_slow") is not None else YARN_BETA_SLOW_DEFAULT)
    if beta_fast < beta_slow:
        raise ValueError(f"OLMo-3 YaRN beta_fast ({beta_fast}) must be >= beta_slow ({beta_slow})")
    af = params.get("attention_factor")
    if af is not None:
        if isinstance(af, bool) or not isinstance(af, (int, float)) or not af > 0:
            raise ValueError(f"OLMo-3 YaRN attention_factor must be > 0, got {af!r}")
        af = float(af)
    return dict(
        factor=factor,
        original_max_position_embeddings=int(orig),
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        attention_factor=af,
    )


def _rope_source(cfg):
    """The rope-parameter dict of an HF config: ``rope_parameters`` (transformers 5.x) or
    ``rope_scaling`` (4.57). A key that is present but null counts as absent. Both present: they
    must be equal (one dict written under both names), else ``ValueError``: one is not dropped in
    favour of the other. An empty dict is a value (no scaling), not "absent"."""
    present = {k: cfg[k] for k in ("rope_parameters", "rope_scaling") if cfg.get(k) is not None}
    for k, v in present.items():
        if not isinstance(v, dict):
            raise ValueError(f"OLMo-3 {k} must be a dict, got {type(v).__name__}")
    if len(present) == 2 and present["rope_parameters"] != present["rope_scaling"]:
        raise ValueError(
            "OLMo-3 config.json has both rope_parameters and rope_scaling and they differ "
            f"({present['rope_parameters']!r} vs {present['rope_scaling']!r}); refusing to pick one"
        )
    return next(iter(present.values()), {})


def _rope_config(cfg):
    """``(full-attention RoPE params, sliding-layer layout or None, rope_theta)`` from an HF config.

    Handles the flat ``rope_scaling`` / ``rope_parameters`` dict (transformers 4.57 checkpoints)
    and the per-layer-type form ``{"full_attention": {...}, "sliding_attention": {...}}``
    (transformers 5.x). In the per-type form the sliding layers' own ``rope_type`` gives the
    layout (``default`` -> ``per_type``, ``yarn`` -> ``yarn_all``); the flat form gives none.
    Unknown nesting or unsupported RoPE types raise ``ValueError``.
    """
    raw = _rope_source(cfg)
    layout = None
    if any(isinstance(v, dict) for v in raw.values()):
        unknown = set(raw) - set(_LAYER_TYPES)
        if unknown or "full_attention" not in raw:
            problems = []
            if unknown:
                problems.append(f"unknown keys {sorted(unknown)} (allowed: {list(_LAYER_TYPES)})")
            if "full_attention" not in raw:
                problems.append("the required 'full_attention' entry is missing")
            raise ValueError(f"OLMo-3 per-layer-type rope parameters ({sorted(raw)}): " + "; ".join(problems))
        full = raw["full_attention"]
        sliding = raw.get("sliding_attention")
        for name, entry in (("full_attention", full), ("sliding_attention", sliding)):
            if entry is not None and not isinstance(entry, dict):
                raise ValueError(
                    f"OLMo-3 rope parameters {name!r} must be a dict, got {type(entry).__name__}"
                )
        if full is None:
            raise ValueError("OLMo-3 rope parameters: 'full_attention' is null")
        if sliding is not None:
            st = _rope_type(sliding)
            if st == "yarn" and _rope_type(full) == "yarn":
                layout = "yarn_all"
            elif st == "default":
                layout = "per_type"
            else:
                raise ValueError("OLMo-3: YaRN on sliding layers needs YaRN on full layers too")
            if sliding.get("rope_theta", full.get("rope_theta")) != full.get("rope_theta"):
                raise ValueError("OLMo-3 converter needs one rope_theta for all layer types")
            # yarn_all builds one YaRN table (and attention factor) for every layer: the sliding
            # layers' YaRN values must be the full layers' ones, not silently replaced by them.
            ys, yf = _yarn_params(sliding, cfg), _yarn_params(full, cfg)
            if ys is not None and ys != yf:
                diff = sorted(k for k in ys if ys[k] != yf[k])
                raise ValueError(
                    f"OLMo-3 converter: sliding_attention YaRN parameters {diff} differ from "
                    "full_attention; yarn_all uses one YaRN table for all layers"
                )
    else:
        full = raw
    _rope_type(full)
    # transformers 5.x may write ``rope_theta: null`` at the top level and the value in the
    # rope parameters: None counts as missing; two different values are refused.
    top, nested = cfg.get("rope_theta"), full.get("rope_theta")
    if top is not None and nested is not None and float(top) != float(nested):
        raise ValueError(f"OLMo-3 config.json: rope_theta {top} differs from the rope parameters' {nested}")
    theta = top if top is not None else nested
    if theta is None:
        raise ValueError("OLMo-3 config.json has no rope_theta (top level or in the rope parameters)")
    return full, layout, float(theta)


def derive_megatron_fields(cfg, rope_layout=None, default_layout="auto"):
    """Translate an OLMo-3 HF config dict into Megatron arg values.

    RoPE layout (``olmo3_rope_layout``):

    * a config with per-layer-type rope parameters (transformers 5.x) declares its layout, and
      that layout is used; an explicit ``rope_layout`` that disagrees raises ``ValueError``;
    * a flat config (transformers 4.57) declares none: ``rope_layout`` when given, else
      ``default_layout``.

    ``default_layout="auto"`` follows the installed transformers and logs the resolved layout
    (a ``logging`` warning from :func:`resolve_olmo3_rope_layout`). It exists
    for parity tests that build the Primus model and compare it against the installed
    transformers' ``Olmo3ForCausalLM``. Everything that trains or saves passes a concrete
    layout: :func:`convert` refuses ``auto`` and uses ``per_type`` (the trained model) for a
    flat config, the model preset sets ``per_type``, and the OLMo-3 args patch refuses
    ``auto``. ``yarn_all`` on a config without YaRN is refused when asked for explicitly; when
    ``auto`` resolves to it, the layout is ``per_type`` (the same model). The result is always
    a concrete layout.
    """
    from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
        resolve_olmo3_rope_layout,
    )

    _check_hf_config(cfg)
    rope, cfg_layout, rotary_base = _rope_config(cfg)
    yarn = _yarn_params(rope, cfg)
    yarn_scales = yarn is not None and yarn["factor"] > 1.0
    if cfg_layout is not None:
        if rope_layout not in (None, "auto") and rope_layout != cfg_layout:
            raise ValueError(
                f"OLMo-3 config declares olmo3_rope_layout {cfg_layout!r} through its per-layer-type "
                f"rope parameters, but {rope_layout!r} was requested"
            )
        layout = cfg_layout
    else:
        explicit = rope_layout is not None and rope_layout != "auto"
        layout = resolve_olmo3_rope_layout(rope_layout if rope_layout is not None else default_layout)
        if layout == "yarn_all" and not yarn_scales:
            if explicit:
                raise ValueError("olmo3_rope_layout 'yarn_all' needs YaRN rope parameters (factor > 1)")
            layout = "per_type"
    sliding_window, skip_freq = _window_fields(cfg)
    nh = cfg["num_attention_heads"]
    return dict(
        hidden_size=cfg["hidden_size"],
        num_attention_heads=nh,
        num_layers=cfg["num_hidden_layers"],
        max_position_embeddings=cfg["max_position_embeddings"],
        norm_epsilon=cfg["rms_norm_eps"],
        rotary_base=rotary_base,
        untie_embeddings_and_output_weights=True,  # _check_hf_config: tie_word_embeddings is false
        ffn_hidden_size=cfg["intermediate_size"],
        num_query_groups=cfg.get("num_key_value_heads") or nh,
        kv_channels=cfg.get("head_dim") or cfg["hidden_size"] // nh,
        add_qkv_bias=False,  # _check_hf_config: attention_bias is false
        # Megatron / TE convention (W - 1, 0) by the shared window reader; the HF W is kept in
        # olmo3_sliding_window.
        window_size=olmo3_window_from_settings(None, sliding_window)[0],
        olmo3_sliding_window=sliding_window,
        olmo3_window_size_form="megatron",
        window_attn_skip_freq=skip_freq,
        olmo3_yarn_rotary_scaling_factor=yarn["factor"] if yarn else 1.0,
        olmo3_yarn_original_max_position_embeddings=(
            yarn["original_max_position_embeddings"] if yarn else int(cfg["max_position_embeddings"])
        ),
        olmo3_yarn_beta_fast=yarn["beta_fast"] if yarn else YARN_BETA_FAST_DEFAULT,
        olmo3_yarn_beta_slow=yarn["beta_slow"] if yarn else YARN_BETA_SLOW_DEFAULT,
        olmo3_yarn_attention_factor=yarn["attention_factor"] if yarn else None,
        olmo3_rope_layout=layout,
    )


# Upstream Megatron args (``--window-size``, ``--window-attn-skip-freq``): set before validate_args.
_UPSTREAM_WINDOW_FIELDS = ("window_size", "window_attn_skip_freq")
# Primus-only OLMo-3 fields (no argparse option): set after validate_args.
_OLMO3_FIELDS = (
    "olmo3_sliding_window",
    "olmo3_window_size_form",
    "olmo3_yarn_rotary_scaling_factor",
    "olmo3_yarn_original_max_position_embeddings",
    "olmo3_yarn_beta_fast",
    "olmo3_yarn_beta_slow",
    "olmo3_yarn_attention_factor",
    "olmo3_rope_layout",
)


def build_margs(cfg_fields, args):
    from megatron.training.arguments import add_megatron_arguments, validate_args

    p = argparse.ArgumentParser(allow_abbrev=False)
    p = add_megatron_arguments(p)
    m = p.parse_args([])

    m.model_type = "olmo3"
    m.hidden_size = cfg_fields["hidden_size"]
    m.num_attention_heads = cfg_fields["num_attention_heads"]
    m.num_layers = cfg_fields["num_layers"]
    m.norm_epsilon = cfg_fields["norm_epsilon"]
    m.layernorm_epsilon = cfg_fields["norm_epsilon"]
    m.ffn_hidden_size = cfg_fields["ffn_hidden_size"]
    m.untie_embeddings_and_output_weights = cfg_fields["untie_embeddings_and_output_weights"]
    # The YaRN factor is max_position_embeddings / original_max_position_embeddings: a longer
    # seq_length would need another factor, so it is refused, not silently raised.
    if int(args.seq_length) > int(cfg_fields["max_position_embeddings"]):
        raise ValueError(
            f"OLMo-3 converter: seq_length {args.seq_length} > max_position_embeddings "
            f"{cfg_fields['max_position_embeddings']} of the HF config (the YaRN factor is fitted to it)"
        )
    m.max_position_embeddings = cfg_fields["max_position_embeddings"]
    m.seq_length = args.seq_length

    m.group_query_attention = cfg_fields["num_query_groups"] != cfg_fields["num_attention_heads"]
    m.num_query_groups = cfg_fields["num_query_groups"]
    m.kv_channels = cfg_fields["kv_channels"]
    m.qk_layernorm = False  # OLMo-3 q/k norms are full-width, owned by Olmo3SelfAttention

    m.normalization = "RMSNorm"
    m.swiglu = True
    m.add_bias_linear = False
    m.add_qkv_bias = cfg_fields["add_qkv_bias"]
    m.position_embedding_type = "rope"
    m.rotary_base = cfg_fields["rotary_base"]
    m.apply_rope_fusion = False
    m.attention_softmax_in_fp32 = True
    m.hidden_dropout = 0.0
    m.attention_dropout = 0.0

    m.vocab_size = None
    m.make_vocab_size_divisible_by = args.make_vocab_size_divisible_by
    m.tokenizer_type = "HuggingFaceTokenizer"
    m.tokenizer_model = args.hf_dir
    m.trust_remote_code = False

    _apply_runtime_margs(m, args)
    for k in _UPSTREAM_WINDOW_FIELDS:  # derive_megatron_fields returns every key
        setattr(m, k, cfg_fields[k])

    m = common.set_save_iteration(validate_args(m))
    for k in _OLMO3_FIELDS:
        setattr(m, k, cfg_fields[k])
    # The olmo3_* fields are not argparse options, so they are set after validate_args. Check the
    # result with the same table the args patch, the config and the builder use.
    check_olmo3_unsupported(m)
    return m


def build_model(margs):
    """The OLMo-3 model (``get_model_provider('olmo3')``) through the shared builder.

    Kept as a module-level entry point: the parity tooling builds a family's own model through
    ``<converter>.build_model``; the plain GPT model would not have the OLMo-3 layers.
    """
    from primus.core.utils.import_utils import get_model_provider

    model = get_model_provider(model_type="olmo3")(pre_process=True, post_process=True)
    return model.to(margs.params_dtype)


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping
# ---------------------------------------------------------------------------
def build_mcore_state_dict(model, store, margs, _cfg_fields, log):
    """HF tensors -> mcore names. ``_cfg_fields`` is unused (kept for the shared converter API)."""
    dtype = margs.params_dtype
    # Read from the parameter, not model.state_dict(): state_dict() collects TE _extra_state, and
    # TE's get_extra_state synchronises the GPU (it fails on a host without one).
    padded_vocab = model.embedding.word_embeddings.weight.shape[0]

    nh = margs.num_attention_heads
    ng = margs.num_query_groups
    hdim = margs.kv_channels
    hidden = margs.hidden_size
    sd = {}
    consumed = set()

    def get(name):
        consumed.add(name)
        return store.get(name, dtype)

    def fused_qkv(layer):
        a = f"model.layers.{layer}.self_attn."
        return common.fuse_qkv(
            get(a + "q_proj.weight"), get(a + "k_proj.weight"), get(a + "v_proj.weight"), nh, ng, hdim, hidden
        )

    sd["embedding.word_embeddings.weight"] = common.pad_vocab(get("model.embed_tokens.weight"), padded_vocab)
    sd["decoder.final_layernorm.weight"] = get("model.norm.weight")
    # untied only: _check_hf_config refuses tie_word_embeddings
    sd["output_layer.weight"] = common.pad_vocab(get("lm_head.weight"), padded_vocab)

    for L in range(margs.num_layers):
        p = f"decoder.layers.{L}."
        h = f"model.layers.{L}."
        sd[p + "self_attention.linear_qkv.weight"] = fused_qkv(L)
        sd[p + "self_attention.q_norm.weight"] = get(h + "self_attn.q_norm.weight")
        sd[p + "self_attention.k_norm.weight"] = get(h + "self_attn.k_norm.weight")
        sd[p + "self_attention.linear_proj.weight"] = get(h + "self_attn.o_proj.weight")
        sd[p + "self_attention.post_attention_layernorm.weight"] = get(h + "post_attention_layernorm.weight")
        sd[p + "mlp.linear_fc1.weight"] = torch.cat(
            [get(h + "mlp.gate_proj.weight"), get(h + "mlp.up_proj.weight")], dim=0
        ).contiguous()
        sd[p + "mlp.linear_fc2.weight"] = get(h + "mlp.down_proj.weight")
        sd[p + "mlp.post_feedforward_layernorm.weight"] = get(h + "post_feedforward_layernorm.weight")
        if L % 5 == 0 or L == margs.num_layers - 1:
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
    rope_layout=None,
    log=None,
):
    """Convert an OLMo-3 HF checkpoint to a legacy Megatron torch ckpt (single process, CPU).

    Writes a single-shard checkpoint: TP = PP = EP = 1 (reshard a torch_dist
    checkpoint for other layouts). The saved args carry a concrete
    ``olmo3_rope_layout``, never ``auto`` (refused): the layout the config's
    per-layer-type rope parameters declare (transformers 5.x; a different
    ``rope_layout`` raises), else ``rope_layout`` if given, else ``per_type``
    (the trained model; flat 4.57-style configs).
    """
    if rope_layout == "auto":
        raise ValueError(
            "convert() writes a checkpoint: olmo3_rope_layout must be per_type or yarn_all, not auto"
        )
    # TP = PP = EP = 1 (also enforced in _apply_runtime_margs); checked first to fail fast.
    check_single_shard_layout(tensor_parallel_size, pipeline_parallel_size, expert_parallel_size)
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
    cfg = common.read_hf_config(hf_dir)
    cfg_fields = derive_megatron_fields(cfg, rope_layout=rope_layout, default_layout="per_type")
    log(f"arch: {cfg_fields}")
    margs = build_margs(cfg_fields, args)
    margs = common.init_megatron_single_process(margs, master_port, log=log)
    common.assert_bridge_free("post-init")
    model = build_model(margs)
    store = common.SafetensorsStore(hf_dir)
    sd, consumed = build_mcore_state_dict(model, store, margs, cfg_fields, log)
    common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
    common.assert_bridge_free("post-save")
    return save_dir
