###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 (text backbone, a.k.a. Kimi-Linear) transformer config.

Extends Megatron's :class:`MLATransformerConfig` with the fields the Kimi
K3 modules need and that upstream does not declare. The MLA geometry of
the full-attention layers (``q_lora_rank`` / ``kv_lora_rank`` /
``qk_head_dim`` / ``qk_pos_emb_head_dim`` / ``v_head_dim`` / ``rope_type``)
is inherited as-is from ``transformer_config.py``, and the KDA geometry
reuses upstream's linear-attention fields (``transformer_config.py``)
rather than introducing parallel names.

Two upstream behaviours are load-bearing here and worth stating outright.

``multi_latent_attention`` must stay **False**
    ``core_transformer_config_from_args`` overwrites the caller's
    ``config_class`` with plain ``MLATransformerConfig`` whenever
    ``args.multi_latent_attention`` is true
    (``megatron/training/arguments.py``). A YAML that turns the
    flag on therefore *silently* discards this class and every field
    below. Kimi K3 builds its attention modules from its own specs, so it
    does not need the flag; the family YAML leaves it at the
    ``language_model.yaml`` default of ``false`` and
    :func:`kimi_k3_builder` asserts the class survived.

New fields need no argparse registration
    ``train_runtime.py`` copies every YAML key onto ``backend_args``
    via ``merge_namespace``, and
    ``core_transformer_config_from_args(args, config_class=...)`` binds
    whatever the dataclass declares.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Union

from megatron.core.transformer.transformer_config import MLATransformerConfig

__all__ = ["KimiK3TransformerConfig", "normalize_linear_attention_freq"]

# The character whitelist Megatron's own pattern evaluator uses
# (arguments.py), widened by whitespace so YAML can breathe.
_PATTERN_CHARS = re.compile(r"[^,\d\[\]\(\)\+\*\s]")


def normalize_linear_attention_freq(
    value: Optional[Union[int, str, List[int]]],
    *,
    num_layers: int,
    field_name: str = "linear_attention_freq",
) -> Optional[List[int]]:
    """Normalize a per-layer attention pattern to ``list[int]`` of length ``num_layers``.

    ``1`` selects KDA (linear attention), ``0`` selects full attention
    (NoPE MLA) — the same polarity as upstream's ``linear_attention_freq``
    (``transformer_config.py``).

    Accepted input forms, mirroring
    ``_normalize_compress_ratios_field`` (``deepseek_v4_transformer_config.py``)
    plus the int-N semantics of ``get_linear_attention_pattern``
    (``experimental_attention_variant_module_specs.py``):

    * ``None`` — pattern not configured.
    * ``int N`` — one full-attention layer every ``N`` layers, i.e.
      ``(i + 1) % N == 0`` is full attention. Kimi K3's released pattern
      is **not** expressible this way; see the class docstring.
    * ``list`` / ``tuple`` — used directly, length-checked.
    * ``str`` — a Python list expression, evaluated the way Megatron's own
      ``la_freq_type`` does (``arguments.py``), so
      ``"([1]*3+[0])*22+[1]*3+[0]*2"`` works. Megatron's converter is
      unreachable from Primus: ``MegatronArgBuilder`` only applies
      argparse ``type=`` converters for *enum* args
      (``argument_builder.py``), so YAML values arrive raw and this
      is the only place the string form is resolved.

    The result is a ``list`` and not a tuple on purpose. Upstream tests the
    pattern with ``isinstance(..., list)`` in two places that would
    silently take the wrong branch on a tuple:
    ``transformer_block.py`` (which drives ``non_homogeneous_layers`` for
    checkpoint sharding) and
    ``experimental_attention_variant_module_specs.py``, whose ``else``
    raises ``ValueError``.
    """
    if value is None:
        return None

    parsed = value
    if isinstance(parsed, str):
        if _PATTERN_CHARS.search(parsed):
            raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
        try:
            parsed = eval(parsed, {"__builtins__": {}}, {})  # noqa: S307
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(f"{field_name} must be an int or a list expression, got {value!r}") from exc

    if isinstance(parsed, bool):
        raise TypeError(f"{field_name} must be int/list/str, got bool")

    if isinstance(parsed, int):
        if parsed <= 0:
            raise ValueError(f"{field_name} as an int must be positive, got {parsed}")
        return [0 if ((i + 1) % parsed == 0) else 1 for i in range(num_layers)]

    if isinstance(parsed, (list, tuple)):
        try:
            out = [int(x) for x in parsed]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} entries must be int-castable; got {parsed!r}") from exc
        if len(out) != num_layers:
            raise ValueError(
                f"{field_name} has length {len(out)}, expected num_layers={num_layers}. "
                "Unlike compress_ratios this field is not auto-padded: an off-by-one here "
                "would silently shift the whole KDA / full-attention interleave."
            )
        bad = sorted({x for x in out if x not in (0, 1)})
        if bad:
            raise ValueError(f"{field_name} entries must be 0 (full attention) or 1 (KDA); got {bad}")
        return out

    raise TypeError(f"{field_name} must be int/list/tuple/str, got {type(parsed).__name__}")


@dataclass
class KimiK3TransformerConfig(MLATransformerConfig):
    """Config for the Kimi K3 text backbone.

    The released 93-layer pattern is ``K K K F`` repeated with one
    irregularity: 0-indexed layers 91 *and* 92 are both full attention, so
    the tail is ``K K F F`` (``configuration_kimi_k3.py`` ships explicit
    ``kda_layers`` / ``full_attn_layers`` lists rather than a ratio for
    exactly this reason). ``linear_attention_freq`` therefore has to accept
    an explicit per-layer pattern; the int-N form cannot express it.
    """

    # ---- layer interleave -------------------------------------------------
    # 1 = KDA (linear attention), 0 = full attention (NoPE MLA). Redeclared
    # from transformer_config.py only to widen the type to the string
    # form; the semantics are upstream's.
    linear_attention_freq: Optional[Union[int, str, List[int]]] = None

    # ---- KDA (HF: linear_attn_config) -------------------------------------
    # The geometry lives in upstream's linear_* fields
    # (transformer_config.py):
    #   linear_conv_kernel_dim  <- short_conv_kernel_size (4)
    #   linear_key_head_dim     <- head_dim (128)
    #   linear_value_head_dim   <- head_dim (128)   # KDA has K == V dims
    #   linear_num_key_heads    <- num_heads (96)
    #   linear_num_value_heads  <- num_heads (96)   # KDA has no GQA
    # Only the knobs upstream has no equivalent for are declared here. The
    # defaults match the getattr fallbacks KimiDeltaAttention already uses
    # (kimi_delta_attention.py) so landing this config does not
    # change the module's behaviour.
    kda_gate_lower_bound: Optional[float] = -5.0
    kda_use_full_rank_gate: bool = True
    kda_backend: str = "eager"
    kda_chunk_size: int = 64

    # ---- unified KDA backend selector (the yaml-facing knob) --------------
    # ``use_kimi_k3_attention_backend`` is the SINGLE yaml key for choosing the
    # KDA implementation family, named after DeepSeek-V4's
    # ``use_v4_attention_backend`` (``deepseek_v4_attention.py``). When set it
    # supersedes BOTH lower-level knobs, so one key selects a coherent family:
    #   * ``kda_backend``            -- the chunk-kernel selector read by
    #                                   ``KimiDeltaAttention`` (eager |
    #                                   eager_recurrent | fla | flydsl), and
    #   * the ``K3P_KDA_CONV`` env   -- the depthwise-conv1d impl (fla
    #                                   ``causal_conv1d`` vs torch ``nn.Conv1d``).
    # "fla" -> fla chunk kernel + fla conv; any other value -> that chunk kernel
    # + the torch conv.
    #
    # Priority (highest first), resolved in ``__post_init__`` (chunk kernel) and
    # in ``KimiDeltaAttention`` (conv):
    #   1. ``use_kimi_k3_attention_backend`` (this field), when not None
    #   2. ``kda_backend`` (chunk kernel) + ``K3P_KDA_CONV`` env (conv) -- legacy
    #
    # Default is ``None``, NOT "fla", on purpose. ``kda_backend`` is pinned to
    # "eager" in ``kimi_k3_base.yaml`` and overridden to fla per-experiment via
    # ``${PRIMUS_KDA_BACKEND:fla}``. A non-None default here would either flip
    # every eager preset/test (base, debug, smoke, ``test_kimi_k3_yaml.py``)
    # to fla, or clobber the experiment-level ``kda_backend`` override -- i.e. it
    # would break "unset => behaviour unchanged". ``None`` means "defer to
    # ``kda_backend`` + ``K3P_KDA_CONV``", so adding this field changes nothing
    # until a yaml sets it. The production recommendation is "fla"; the
    # experiment yamls set it explicitly.
    use_kimi_k3_attention_backend: Optional[str] = None

    # ---- full-attention (MLA) extras --------------------------------------
    # NoPE, and it is real mechanism rather than a readable alias:
    # KimiK3MLASelfAttention reads this flag and, when set, replaces the
    # parent's qk_pos_emb_head_dim-wide frequency table with a zero-width
    # RotaryEmbedding(0) (kimi_k3_mla_attention.py). rot_dim then being 0 sends
    # the whole tensor down apply_rotary_pos_emb's t_pass branch, and the
    # closing torch.cat returns it bit-for-bit (rope_utils.py).
    #
    # It deliberately does NOT touch qk_pos_emb_head_dim. Zeroing that also
    # disables rope but changes the architecture: it deletes the 64
    # MQA-shared K dims (linear_kv_down_proj's width is kv_lora_rank +
    # qk_pos_emb_head_dim, multi_latent_attention.py, and those dims bypass
    # both the kv_lora_rank latent and kv_a_layernorm,
    # modeling_kimi_linear.py), narrows linear_q_up_proj from H*192 to
    # H*128, and changes softmax_scale from 192**-0.5 to 128**-0.5.
    mla_use_nope: bool = True
    # The sigmoid output gate is applied by the Kimi K3 attention subclass,
    # NOT by upstream's attention_output_gate: MLATransformerConfig.__post_init__
    # raises NotImplementedError for that flag (transformer_config.py).
    mla_use_output_gate: bool = True
    # Epsilon for MLA's two low-rank latent norms (HF: ``q_a_layernorm`` /
    # ``kv_a_layernorm``), which is NOT ``rms_norm_eps``.
    #
    # ``KimiRMSNorm.__init__`` defaults ``eps=1e-6``
    # (``modeling_kimi_linear.py``), and MLA constructs exactly these two
    # norms without passing one while every other norm in the released
    # model is given ``eps=config.rms_norm_eps`` = 1e-5. Both published
    # releases do it -- Kimi-Linear-48B's ``modeling_kimi.py`` too -- so it
    # is inherited from the DeepSeek-V3 code this was adapted from, where
    # ``rms_norm_eps`` happens to be 1e-6 and the two agree.
    # Megatron's ``MLASelfAttention`` passes ``config.layernorm_epsilon`` to
    # both, so without this field we run them at 1e-5.
    #
    # Measured on real Kimi-Linear-48B weights: at 1e-5 our MLA output
    # differs from the reference by rel_rms
    # 1.95e-03; at 1e-6 by 4.69e-07. The effect scales as
    # ``(eps_ours - eps_theirs) / (2 * mean(kv_compressed^2))``, so it grows as
    # the latent activations get smaller.
    #
    # ``None`` restores upstream's behaviour of using ``layernorm_epsilon``.
    mla_latent_layernorm_epsilon: Optional[float] = 1e-6

    # ---- attention residuals ----------------------------------------------
    # HF: attn_res_block_size (12). None / 0 disables the mechanism.
    attn_res_block_size: Optional[int] = None
    # Which mixer kernel AttentionResidualMixer resolves at construction:
    # "eager"  -- pure PyTorch, the numerical oracle, runs anywhere
    # "flydsl" -- one fused FlyDSL kernel per direction, gfx950 / CDNA4 only
    # Measured at the scaled shape the mixers are 16.8 % of a forward+backward
    # step, and the eager path moves ~1.07 GB for a 67 MB input because it
    # materialises six full-size intermediates; see
    # primus/backends/.../attn_res_kernels/__init__.py for the numbers.
    # Defaults to "eager" so the reference stays the default and the kernel is
    # an explicit opt-in, exactly as kda_backend does.
    attn_res_backend: str = "eager"

    # ---- Stable Latent MoE ------------------------------------------------
    # HF: routed_expert_hidden_size (3584 = hidden_size / 2). None keeps the
    # routed experts in model space, i.e. no latent bottleneck. ``__post_init__``
    # mirrors it onto upstream's ``moe_latent_size``, which is the field every
    # upstream consumer actually reads.
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False

    # ---- MoE load balancing: which bias-update rule ----------------------
    # "sign"     -- DeepSeek-V3's b <- b + rate * sign(violation), i.e. exactly
    #               what upstream's get_updated_expert_bias does
    #               (moe_utils.py). This is what phase 1 ran, the
    #               known-good baseline.
    # "quantile" -- Kimi K3's published rule, tech report §2.3.3 / Eq. 14:
    #               b_j = -quantile_{1-k/n}(s_{:,j} - tau), tau being the
    #               (k+1)-th largest *biased* score of each token. See
    #               moe/k3_quantile_balancing.py.
    # Default stays "sign" so selecting the faithful rule is an explicit,
    # reviewable act and nothing else in the tree changes behaviour.
    moe_router_bias_update_rule: str = "sign"

    # Quantile Balancing knobs. All of these exist because the report leaves
    # the corresponding detail unstated; each default is argued in
    # k3_quantile_balancing.py's module docstring.
    #
    # The binning is the big one: the report says only "a histogram of its
    # margins", and the sentence that would have described it is truncated in
    # every PDF extraction we have. sigmoid confines the raw score to (0, 1)
    # and a mean-centred bias keeps the cutoff near that interval, so margins
    # live in roughly (-1, 1); 1024 bins over that gives a 1.95e-3 resolution,
    # the same order as moe_router_bias_update_rate's 1e-3 step. Margins
    # outside the range are clamped into the end bins and counted.
    quantile_balancing_num_bins: int = 1024
    quantile_balancing_margin_min: float = -1.0
    quantile_balancing_margin_max: float = 1.0
    # Eq. 14's second line. Removing a common offset cannot change
    # argtopk(s + b), so this only stops the bias drifting.
    quantile_balancing_center_bias: bool = True
    # Cadence, in optimizer steps. The report's "QB derives the next bias from
    # a single forward pass", plus a quantile that "spans the full global
    # batch ... across ranks and accumulation steps", reads as every step.
    # >1 widens the sample window rather than discarding the extra samples.
    quantile_balancing_update_interval: int = 1
    # None reproduces the report: the bias is *set*, not smoothed. A value in
    # (0, 1) makes it b <- decay*b + (1-decay)*b_new, which is the reading the
    # report's "spans the full global batch" sentence rules out but which is
    # cheap to keep reachable at small batch sizes.
    quantile_balancing_ema_decay: Optional[float] = None
    # Which kernel computes the per-microbatch margin histogram:
    # "eager"  -- pure PyTorch, nine launches, the numerical oracle
    # "flydsl" -- one fused FlyDSL kernel behind torch.topk, gfx950 / CDNA4 only
    # Measured at 2.7 % of a forward+backward step at the scaled shape, in nine
    # launches over a 0.5 MB tensor, i.e. almost entirely launch and pass
    # overhead. Defaults to "eager", as every other backend selector does.
    quantile_balancing_backend: str = "eager"
    # Shape guard for that kernel. It is atomic-contention bound: measured 2.84x
    # of eager at 4096 tokens per microbatch, 0.99x at 16384 and 0.62x at 32768,
    # because torch.bincount privatises/sorts and the kernel does not. Above this
    # token count the flydsl backend runs the eager path and warns once, so
    # selecting the kernel can never make a large micro-batch slower. None keeps
    # the measured default (6144, the last rung that wins at BOTH 32 and 896
    # experts); 0 disables the guard.
    quantile_balancing_kernel_max_tokens: Optional[int] = None

    # ---- expert-load diagnostics ------------------------------------------
    # Opt-in output path for the expert-load-probe patch
    # (patches/kimi_k3_expert_load_probe_patches.py). When set to a filesystem
    # path the probe wraps reset_model_temporary_tensors once per optimizer step
    # and appends the all-reduced expert-load histogram -- entropy, max/min
    # ratio, dead-expert count -- to that path as JSONL. That histogram is the
    # quantity phase 2's Quantile-Balancing-vs-sign A/B compares, and Megatron
    # logs no such thing.
    #
    # None (the default everywhere -- no yaml in the tree sets it and no unit
    # test does either) leaves the patch's registration condition False, so
    # nothing is imported, wrapped or allocated: an unset run is byte-for-byte
    # unaffected. This is a diagnostic OUTPUT sink and NOT a model knob -- it
    # does not touch the computation, which is why it needs no __post_init__
    # handling and is read at the args layer by the patch rather than off this
    # config object.
    #
    # It is declared here, as a first-class config parameter, on purpose: the
    # path must arrive through the normal Primus config / CLI channel (a yaml
    # key or a --expert-load-probe-path override), NOT through an environment
    # variable. Mirrors how use_kimi_k3_attention_backend is declared above.
    expert_load_probe_path: Optional[str] = None

    # ---- Multi-Token Prediction --------------------------------------------
    # HF: num_nextn_predict_layers. The released config.json ships 0 and the
    # released modelling code has no MTP module, but tech-report Table 1 lists
    # "MTP Layers: 1 layer", so the release was stripped rather than trained
    # without one. This field is the HF-native name for upstream's
    # ``mtp_num_layers`` (transformer_config.py) and ``__post_init__``
    # mirrors the two, the same way ``routed_expert_hidden_size`` mirrors
    # ``moe_latent_size``.
    #
    # ``0`` is normalised to ``None``, because 0 is *not* "off" upstream:
    # ``mtp_on_this_rank`` tests ``config.mtp_num_layers is not None``
    # (multi_token_prediction.py) and so returns True on the last pipeline
    # stage, after which ``MultiTokenPredictionBlock.__init__`` asserts on an
    # empty layer list.
    num_nextn_predict_layers: Optional[int] = None
    # Which backbone layer the MTP layer mirrors. Report §4.1.4 says only
    # "an MTP layer that mirrors the structure of a backbone layer" and K3's
    # backbone is heterogeneous, so this is a resolved ambiguity, not a
    # reproduction:
    #   "mirror_last" -- copy the final backbone layer, which report §2.1
    #       guarantees is a Gated MLA layer ("An additional Gated MLA layer is
    #       placed at the end of the backbone, ensuring that the final layer
    #       always performs global attention"). The default.
    #   "mla" / "kda" -- force the attention variant explicitly.
    # The FFN always mirrors the final backbone layer's (MoE on any K3 shape
    # with experts), because ``first_k_dense_replace`` only makes the *first*
    # layers dense.
    mtp_layer_type: str = "mirror_last"

    # ---- situ activation --------------------------------------------------
    activation_situ_beta: Optional[float] = None  # 4.0, gate-branch soft clamp
    activation_situ_linear_beta: Optional[float] = None  # 25.0, up-branch soft clamp

    # ---- MoonViT-V2 vision tower (HF: config.json:vision_config) ----------
    # ``vt_num_hidden_layers`` is the master switch: None / 0 means text
    # backbone only, which is what every config in this tree was until the
    # vision work landed and is still the default. Set it to 27 for the
    # released tower. ``build_moonvit_configs`` turns this block into the
    # tower's own TransformerConfig; ``validate_moonvit_fields`` checks it.
    #
    # Field names are the HF ones with the ``vt_`` prefix the released
    # ``vision_config`` already uses, so the mapping needs no table.
    vt_num_hidden_layers: Optional[int] = None  # 27
    vt_hidden_size: int = 1024
    vt_intermediate_size: int = 4096
    vt_num_attention_heads: int = 12
    # The attention inner width, deliberately WIDER than vt_hidden_size:
    # 1536 = 12 heads x 128 against a 1024 residual stream. None falls back
    # to vt_hidden_size. Reaches the tower config as ``kv_channels``, which
    # upstream already treats as independent of hidden_size
    # (transformer_config.py).
    vt_qkv_hidden_size: Optional[int] = 1536
    vt_patch_size: int = 14
    vt_in_channels: int = 3
    vt_init_pos_emb_height: int = 64
    vt_init_pos_emb_width: int = 64
    vt_init_pos_emb_time: int = 4
    # ``bilinear`` is the released value. Note MoonVision3dPatchEmbed's own
    # default is ``bicubic`` (modeling_kimi_k3.py) while
    # VisionTowerConfig's fallback is ``bilinear``, so the two disagree and
    # only config.json settles it.
    vt_pos_emb_interpolation_mode: str = "bilinear"
    vt_merge_kernel_size: Union[List[int], tuple] = (2, 2)
    vt_rope_theta: float = 10000.0
    vt_rope_max_height: int = 512
    vt_rope_max_width: int = 512
    # The tower's RMSNorms are nn.RMSNorm(dim) with eps=None, for which ATen
    # substitutes double-precision epsilon, 2.22e-16 -- measured, see
    # moonvit_reference.moonvit_default_rmsnorm_eps. That is 9 orders of
    # magnitude under float32's own epsilon, so it is a no-op in every dtype
    # K3 trains in; what matters is that it is NOT Megatron's 1e-5 default,
    # which would be a 5e-6 relative error against the release.
    vt_layernorm_epsilon: float = 2.220446049250313e-16
    # eager | te | auto. ``eager`` is the fp32/CPU-capable block-diagonal
    # softmax and the parity oracle; ``te`` is TEDotProductAttention in thd
    # format, which dispatches to flash internally -- the released
    # production path.
    vt_attention_backend: str = "auto"
    # The token id vision features are spliced onto. HF's released value is
    # 163605 (``media_placeholder_token_id``); left None so a text-only run
    # cannot accidentally reserve a vocabulary slot.
    vt_media_placeholder_token_id: Optional[int] = None

    # ---- projector (HF: mm_projector) -------------------------------------
    mm_projector_type: str = "patchmergerv2"
    # The projector's per-patch input width. Defaults to vt_hidden_size in
    # __post_init__ because the merger concatenates the tower's own output.
    mm_hidden_size: Optional[int] = None
    projector_ln_eps: float = 1e-5

    # ---- compat aliases used by the Kimi K3 code paths --------------------
    # Neither ``vocab_size`` nor ``padded_vocab_size`` is an upstream
    # TransformerConfig field; they are declared for the same reason
    # DeepSeek-V4 declares them (deepseek_v4_transformer_config.py). Kimi K3
    # deliberately does NOT declare DeepSeek-V4's ``norm_epsilon`` alias: every
    # K3 norm reads ``config.layernorm_epsilon`` (never ``config.norm_epsilon``),
    # so K3 yaml sets ``layernorm_epsilon`` directly instead of shadowing it
    # with a second field.
    vocab_size: Optional[int] = None
    padded_vocab_size: Optional[int] = None
    position_embedding_type: str = "none"

    def __post_init__(self) -> None:
        self.linear_attention_freq = normalize_linear_attention_freq(
            self.linear_attention_freq,
            num_layers=int(self.num_layers),
            field_name="linear_attention_freq",
        )

        # Resolve the unified KDA backend selector (see the field's docstring).
        # When set, it wins over the legacy ``kda_backend`` field for the chunk
        # kernel; the conv1d half is resolved in ``KimiDeltaAttention``, which
        # reads this same field. Leaving it None reproduces the legacy behaviour
        # exactly, so this block is a no-op for every config that does not set it.
        if self.use_kimi_k3_attention_backend is not None:
            # Lazy import: kda_kernels pulls in torch via ._eager, so only reach
            # for it when the knob is actually in use (keeps a plain
            # KimiK3TransformerConfig() free of the dependency).
            from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
                KDA_BACKENDS,
            )

            selector = str(self.use_kimi_k3_attention_backend)
            if selector not in KDA_BACKENDS:
                raise ValueError(
                    "use_kimi_k3_attention_backend must be one of "
                    f"{list(KDA_BACKENDS)} (or null to defer to kda_backend); "
                    f"got {selector!r}."
                )
            self.use_kimi_k3_attention_backend = selector
            # The unified knob is authoritative for the chunk kernel.
            self.kda_backend = selector

        # A hybrid stack halves the output-layer init scaling
        # (transformer_config.py).
        self.is_hybrid_model = True

        if self.mla_use_nope:
            # Only the model-level knob is set here. rope_type stays "rope"
            # because multi_latent_attention.py validates it against
            # {"rope", "yarn"}, and qk_pos_emb_head_dim stays at the released
            # width because the attention module -- not the geometry -- is what
            # disables the rotation (see the field comment above).
            #
            # position_embedding_type is the field KimiK3Model reads to decide
            # whether to build any positional module at all; the *args* copy has
            # to stay "rope" or arguments.py raises before any config
            # exists.
            self.position_embedding_type = "none"

        # Upstream implements the routed-expert latent bottleneck itself, under
        # its own name for the same quantity: ``moe_latent_size`` (declared at
        # transformer_config.py, consumed at moe_layer.py, experts.py and
        # mlp.py). Map K3's HF-native name onto it here so the config the
        # *rest of the layer* sees is right too — ``StableLatentMoE`` maps it
        # defensively on its own shallow copy, which is correct for the MoE
        # module but leaves the outer config reading None, and that outer copy
        # is what decides the cudagraph pre-MLP-layernorm recompute
        # (transformer_layer.py).
        #
        # Same agreement rule as ``StableLatentMoE.resolve_latent_size``: either
        # name may be set, and if both are they must match. With the mapping
        # done here they always match, so ``_latent_config`` takes its
        # use-the-config-as-is fast path and makes no copy at all.
        if self.routed_expert_hidden_size is not None:
            latent = int(self.routed_expert_hidden_size)
            if latent <= 0:
                raise ValueError(f"routed_expert_hidden_size must be > 0 when set, got {latent}")
            if self.moe_latent_size is not None and int(self.moe_latent_size) != latent:
                raise ValueError(
                    f"routed_expert_hidden_size={latent} disagrees with "
                    f"moe_latent_size={self.moe_latent_size}. They name the same latent "
                    "width; set one, or set both to the same value."
                )
            self.routed_expert_hidden_size = latent
            self.moe_latent_size = latent

        rule = str(self.moe_router_bias_update_rule)
        if rule not in ("sign", "quantile"):
            raise ValueError(f"moe_router_bias_update_rule must be 'sign' or 'quantile', got {rule!r}")
        if rule == "quantile":
            if not self.moe_router_enable_expert_bias:
                raise ValueError(
                    "moe_router_bias_update_rule: quantile updates "
                    "e_score_correction_bias, so moe_router_enable_expert_bias must "
                    "be true. (transformer_config.py additionally requires "
                    "moe_router_score_function: sigmoid, which Kimi K3 uses.)"
                )
            if int(self.quantile_balancing_num_bins) < 2:
                raise ValueError(
                    f"quantile_balancing_num_bins must be >= 2, got {self.quantile_balancing_num_bins}"
                )
            if not float(self.quantile_balancing_margin_max) > float(self.quantile_balancing_margin_min):
                raise ValueError(
                    "quantile_balancing_margin_max must exceed "
                    "quantile_balancing_margin_min, got "
                    f"[{self.quantile_balancing_margin_min}, "
                    f"{self.quantile_balancing_margin_max}]"
                )
            if self.quantile_balancing_ema_decay is not None and not (
                0.0 <= float(self.quantile_balancing_ema_decay) < 1.0
            ):
                raise ValueError(
                    "quantile_balancing_ema_decay must be None or in [0, 1), got "
                    f"{self.quantile_balancing_ema_decay}"
                )

        self._resolve_mtp_fields()

        if self.padded_vocab_size is None and self.vocab_size is not None:
            self.padded_vocab_size = int(self.vocab_size)
        if self.vocab_size is None and self.padded_vocab_size is not None:
            self.vocab_size = int(self.padded_vocab_size)

        # Imported here rather than at module scope: kimi_k3_vision_config
        # imports TransformerConfig, and a top-level import would make this
        # dataclass module depend on the vision half for a text-only run.
        from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
            validate_moonvit_fields,
        )

        validate_moonvit_fields(self)

        super().__post_init__()

    @property
    def has_vision_tower(self) -> bool:
        """Whether this config carries a MoonViT-V2 tower."""
        return bool(self.vt_num_hidden_layers)

    # ---- Multi-Token Prediction -------------------------------------------
    def _resolve_mtp_fields(self) -> None:
        """Reconcile ``num_nextn_predict_layers`` with ``mtp_num_layers``.

        Same agreement rule as ``routed_expert_hidden_size`` /
        ``moe_latent_size``: either name may be set, and if both are they must
        match. After this runs the two always agree and ``mtp_num_layers`` is
        either ``None`` or ``>= 1`` -- never ``0``, for the reason given on the
        field.

        Note the args-layer half of this mapping lives in
        ``patches/kimi_k3_flops_patches.py::patch_k3_args_mtp_num_layers``.
        ``training.py`` reads ``args.mtp_num_layers`` and never the config,
        so a config-only mapping would leave the FLOPs report and the
        per-depth MTP loss logging blind -- exactly the same split we make
        for ``moe_latent_size``.
        """
        nextn = self.num_nextn_predict_layers
        mtp = self.mtp_num_layers

        if nextn is not None and int(nextn) < 0:
            raise ValueError(f"num_nextn_predict_layers must be >= 0 when set, got {nextn}")
        if mtp is not None and int(mtp) < 0:
            raise ValueError(f"mtp_num_layers must be >= 0 when set, got {mtp}")

        if nextn is not None and mtp is not None and int(nextn) != int(mtp):
            raise ValueError(
                f"num_nextn_predict_layers={nextn} disagrees with mtp_num_layers={mtp}. "
                "They name the same quantity -- the number of Multi-Token Prediction "
                "depths; set one, or set both to the same value."
            )

        depths = nextn if nextn is not None else mtp
        depths = None if depths is None or int(depths) == 0 else int(depths)
        self.num_nextn_predict_layers = depths
        self.mtp_num_layers = depths

        layer_type = str(self.mtp_layer_type)
        if layer_type not in ("mirror_last", "mla", "kda"):
            raise ValueError(
                "mtp_layer_type must be 'mirror_last', 'mla' or 'kda', got " f"{self.mtp_layer_type!r}"
            )
        self.mtp_layer_type = layer_type

        if depths is None:
            return

        # A zero weight builds the MTP layer, runs it, and then hands it no
        # gradient at all: ``process_mtp_loss`` folds
        # ``mtp_loss_scaling_factor / mtp_num_layers * loss`` into the tensor
        # ``MTPLossAutoScaler`` differentiates (multi_token_prediction.py), so
        # the factor multiplies straight through to every MTP parameter's
        # gradient. That is a silent misconfiguration, not a cheaper MTP, so
        # it is rejected.
        scale = self.mtp_loss_scaling_factor
        if scale is None or not float(scale) > 0.0:
            raise ValueError(
                f"mtp_loss_scaling_factor must be > 0 when MTP is enabled, got {scale!r}. "
                "A zero or None weight leaves every MTP parameter without a gradient."
            )

    @property
    def mtp_enabled(self) -> bool:
        """Whether Multi-Token Prediction is on.

        Reads the normalised ``mtp_num_layers``, so it is ``False`` for both
        ``None`` and a YAML-supplied ``0``.
        """
        return bool(self.mtp_num_layers)

    def mtp_layer_is_kda(self) -> bool:
        """Whether the MTP layer's attention is KDA.

        ``mirror_last`` resolves against the final backbone layer, which is
        what report §4.1.4's "mirrors the structure of a backbone layer"
        becomes once §2.1's "an additional Gated MLA layer is placed at the end
        of the backbone" is applied: the layer being mirrored is the one the
        report guarantees performs global attention.
        """
        if self.mtp_layer_type == "kda":
            return True
        if self.mtp_layer_type == "mla":
            return False
        return bool(self.is_kda_layer(int(self.num_layers) - 1))

    # ---- derived helpers --------------------------------------------------
    def is_kda_layer(self, layer_idx: int) -> bool:
        """Whether the 0-indexed ``layer_idx`` is a KDA layer.

        Mirrors ``KimiLinearConfig.is_kda_layer``
        (``configuration_kimi_k3.py``) but reads an already-0-indexed,
        already-normalized pattern, so the HF ``layer_idx + 1`` offset is
        gone.
        """
        if self.linear_attention_freq is None:
            return False
        return bool(self.linear_attention_freq[layer_idx])

    @property
    def kda_layer_indices(self) -> List[int]:
        """0-indexed KDA layers."""
        if self.linear_attention_freq is None:
            return []
        return [i for i, v in enumerate(self.linear_attention_freq) if v]

    @property
    def full_attention_layer_indices(self) -> List[int]:
        """0-indexed full-attention (NoPE MLA) layers."""
        if self.linear_attention_freq is None:
            return list(range(int(self.num_layers)))
        return [i for i, v in enumerate(self.linear_attention_freq) if not v]

    @property
    def attn_res_num_blocks_max(self) -> int:
        """Number of attention-residual block checkpoints the stack appends.

        A checkpoint is appended whenever ``layer_idx % attn_res_block_size == 0``
        (``modeling_kimi_linear.py``), which over ``num_layers`` layers is
        ``ceil(num_layers / attn_res_block_size)``. Returns 0 when the
        mechanism is disabled.
        """
        if not self.attn_res_block_size:
            return 0
        return -(-int(self.num_layers) // int(self.attn_res_block_size))
