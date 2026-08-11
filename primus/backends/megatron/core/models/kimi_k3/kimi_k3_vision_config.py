###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformer configs for the MoonViT-V2 tower and its projector.

The vision tower disagrees with the text backbone on almost every field a
:class:`TransformerConfig` carries -- 1024 against 7168 hidden, 4096 against
33792 FFN, 12 against 96 heads, non-gated GELU against gated ``situ``,
non-causal against causal -- so it needs its **own** config object. That is
also what Megatron's own VLM path does
(``pretrain_vlm.py`` builds ``get_vision_model_config(deepcopy(config))``).

Two configs come out of one :class:`KimiK3TransformerConfig`:

``KimiK3VisionTransformerConfig``
    the 27-layer tower. ``hidden_size`` 1024, ``kv_channels`` 128 with 12
    heads -- note ``kv_channels * num_attention_heads`` is 1536, **not**
    ``hidden_size``. Upstream supports that directly: ``kv_channels`` is an
    independent field (``transformer_config.py``) and
    ``Attention.__init__`` sizes the projections from it rather than from
    ``hidden_size // num_attention_heads``
    (``attention.py``).

``KimiK3VisionProjectorConfig``
    the ``patchmergerv2`` head: ``4096 -> 4096 -> 7168``. It is a separate
    config because Megatron's :class:`MultimodalProjector` reads
    ``config.ffn_hidden_size`` for the middle width and
    ``config.hidden_size`` for the output, and here those are 4096 and 7168
    while the tower's are 4096 and 1024.

Both inherit the *environment* -- dtype, TP size, initialisation policy,
recompute -- from the parent K3 config, and override only shape. A field
that is not copied is a field the vision tower is deliberately not given.

The epsilon trap
    The released tower's norms are ``nn.RMSNorm(hidden_dim)`` with **no**
    ``eps`` argument (``modeling_kimi_k3.py``), for which
    ATen substitutes double-precision epsilon, **2.22e-16** -- measured by
    :func:`moonvit_default_rmsnorm_eps`, not assumed. That is nine orders of
    magnitude below float32's own epsilon, so it is effectively zero in
    every dtype Kimi K3 trains in. Megatron's default ``layernorm_epsilon``
    is **1e-5**, which on unit-RMS activations is a 5e-6 relative error:
    invisible in bf16, and the difference between "matches the official
    implementation" and "nearly matches" in fp32. Only the *projector's*
    ``post_norm`` uses 1e-5, from ``projector_ln_eps`` -- the two
    epsilons in the released config are genuinely different numbers.

    ``vt_layernorm_epsilon`` therefore defaults to the measured value rather
    than to Megatron's. It stays a config field because an effectively-zero
    epsilon is a fidelity choice, not a training recommendation: a token
    whose activations are all exactly zero divides by zero. Nothing in a
    real tower produces that -- the patch embedding adds a positional term
    to every token -- but a from-scratch run that wants the guard can set it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig

if TYPE_CHECKING:  # pragma: no cover
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

__all__ = [
    "KimiK3VisionProjectorConfig",
    "KimiK3VisionTransformerConfig",
    "MOONVIT_INTERPOLATION_MODES",
    "MOONVIT_PROJECTOR_TYPES",
    "build_moonvit_configs",
    "gelu_tanh",
    "validate_moonvit_fields",
]

#: ``F.interpolate`` modes the resampler accepts. The released config picks
#: ``bilinear``; ``MoonVision3dPatchEmbed``'s own default is ``bicubic``
#: (``modeling_kimi_k3.py``) and ``VisionTowerConfig``'s fallback is
#: ``bilinear``, so the two disagree and only the explicit value in
#: ``config.json`` settles it.
MOONVIT_INTERPOLATION_MODES = ("nearest", "bilinear", "bicubic", "area")

#: ``mm_projector_type`` values that map onto something we build. The
#: released config is ``patchmergerv2``; the other three branches
#: (``identity`` / ``mlp`` / ``patchmerger``) are dead code in
#: this release and are rejected rather than half-implemented.
MOONVIT_PROJECTOR_TYPES = ("patchmergerv2",)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``PytorchGELUTanh`` (``modeling_kimi_k3.py``).

    A named module-level function rather than a ``functools.partial`` so it
    survives config ``repr``, pickling for the dataloader workers, and the
    identity comparisons Megatron makes against ``F.gelu`` when deciding
    whether a fusion applies.

    The tower uses the **tanh** approximation; the projector uses
    ``nn.GELU()``, the exact erf form. They differ by about 1e-3
    absolute near the knee and are not interchangeable.
    """
    return F.gelu(x, approximate="tanh")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class KimiK3VisionTransformerConfig(TransformerConfig):
    """The MoonViT-V2 tower's own transformer config."""

    patch_size: int = 14
    in_channels: int = 3
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    pos_emb_interpolation_mode: str = "bilinear"
    merge_kernel_size: Tuple[int, int] = (2, 2)
    rope_theta: float = 10000.0
    rope_max_height: int = 512
    rope_max_width: int = 512
    #: ``eager`` is the pure-PyTorch block-diagonal softmax and the only
    #: path that runs on CPU and in fp32; ``te`` routes to
    #: ``TEDotProductAttention`` in ``thd`` format. ``auto`` picks ``te``
    #: when TE is importable and a GPU is visible.
    attention_backend_name: str = "auto"

    @property
    def head_dim(self) -> int:
        return int(self.kv_channels)

    @property
    def qkv_hidden_size(self) -> int:
        """The attention inner width, ``1536`` at the released geometry."""
        return int(self.kv_channels) * int(self.num_attention_heads)


@dataclass
class KimiK3VisionProjectorConfig(TransformerConfig):
    """Config for the ``patchmergerv2`` head.

    ``hidden_size`` is the **text** width (the projector's output) and
    ``ffn_hidden_size`` is the merged vision width, so that upstream's
    :class:`MLP` -- which reads exactly those two fields -- lays out
    ``4096 -> 4096 -> 7168``.
    """

    projector_input_size: int = 4096
    projector_ln_eps: float = 1e-5
    merge_kernel_size: Tuple[int, int] = (2, 2)


# ---------------------------------------------------------------------------
# Validation + derivation
# ---------------------------------------------------------------------------


def validate_moonvit_fields(config: "KimiK3TransformerConfig") -> None:
    """Validate the ``vt_*`` / ``mm_*`` block of a Kimi K3 config.

    Called from ``KimiK3TransformerConfig.__post_init__`` so a bad vision
    geometry fails at config construction rather than at the first forward,
    in the same style as the ``moe_router_bias_update_rule`` and
    ``routed_expert_hidden_size`` checks alongside it.

    A ``None`` / 0 ``vt_num_hidden_layers`` means "no vision tower" and
    every other field is left alone, so a text-only config is unaffected.
    """
    if not config.vt_num_hidden_layers:
        return

    if int(config.vt_num_hidden_layers) < 1:
        raise ValueError(f"vt_num_hidden_layers must be >= 1 when set, got {config.vt_num_hidden_layers}")
    for name in ("vt_hidden_size", "vt_intermediate_size", "vt_num_attention_heads", "vt_patch_size"):
        value = int(getattr(config, name))
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")

    qkv = config.vt_qkv_hidden_size
    qkv = int(config.vt_hidden_size) if qkv is None else int(qkv)
    heads = int(config.vt_num_attention_heads)
    if qkv % heads:
        raise ValueError(
            f"vt_qkv_hidden_size={qkv} is not divisible by vt_num_attention_heads={heads}. "
            "Note this is the attention inner width and is deliberately allowed to differ "
            "from vt_hidden_size -- the released tower is 1536 against a 1024 hidden."
        )
    head_dim = qkv // heads
    if head_dim % 4:
        # Rope2DPosEmbRepeated.__init__ asserts it (modeling_kimi_k3.py):
        # the table interleaves an x and a y frequency per pair of channels,
        # so it consumes 4 channels per frequency.
        raise ValueError(
            f"the 2-D RoPE needs head_dim divisible by 4, got {head_dim} "
            f"(= vt_qkv_hidden_size {qkv} / vt_num_attention_heads {heads})"
        )

    if config.vt_pos_emb_interpolation_mode not in MOONVIT_INTERPOLATION_MODES:
        raise ValueError(
            f"vt_pos_emb_interpolation_mode must be one of {MOONVIT_INTERPOLATION_MODES}, "
            f"got {config.vt_pos_emb_interpolation_mode!r}"
        )
    for name in ("vt_init_pos_emb_height", "vt_init_pos_emb_width", "vt_init_pos_emb_time"):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(config, name)}")

    merge = tuple(int(v) for v in config.vt_merge_kernel_size)
    if len(merge) != 2 or any(v < 1 for v in merge):
        raise ValueError(f"vt_merge_kernel_size must be two positive ints, got {merge}")
    config.vt_merge_kernel_size = merge

    if config.mm_projector_type not in MOONVIT_PROJECTOR_TYPES:
        raise ValueError(
            f"mm_projector_type must be one of {MOONVIT_PROJECTOR_TYPES}, "
            f"got {config.mm_projector_type!r}"
        )

    if config.mm_hidden_size is None:
        config.mm_hidden_size = int(config.vt_hidden_size)
    elif int(config.mm_hidden_size) != int(config.vt_hidden_size):
        # The merger concatenates the tower's own outputs, so the projector's
        # per-patch input width is the tower's hidden size by construction.
        raise ValueError(
            f"mm_hidden_size={config.mm_hidden_size} must equal "
            f"vt_hidden_size={config.vt_hidden_size}; the projector consumes the "
            "tower's output directly."
        )

    if float(config.vt_layernorm_epsilon) < 0.0:
        raise ValueError(f"vt_layernorm_epsilon must be >= 0, got {config.vt_layernorm_epsilon}")
    if float(config.projector_ln_eps) <= 0.0:
        raise ValueError(f"projector_ln_eps must be > 0, got {config.projector_ln_eps}")

    backend = str(config.vt_attention_backend)
    if backend not in ("auto", "eager", "te"):
        raise ValueError(f"vt_attention_backend must be auto|eager|te, got {backend!r}")

    if config.vt_media_placeholder_token_id is not None:
        token = int(config.vt_media_placeholder_token_id)
        if token < 0:
            raise ValueError(f"vt_media_placeholder_token_id must be >= 0 when set, got {token}")


#: Environment fields the vision configs inherit verbatim from the text
#: config. Everything not listed here is a shape field and is set explicitly,
#: so adding a field to ``TransformerConfig`` cannot silently leak the text
#: backbone's value into the tower.
_INHERITED_ENV_FIELDS = (
    "params_dtype",
    "fp16",
    "bf16",
    "pipeline_dtype",
    "autocast_dtype",
    "tensor_model_parallel_size",
    "sequence_parallel",
    "context_parallel_size",
    "perform_initialization",
    "use_cpu_initialization",
    "init_method_std",
    "gradient_accumulation_fusion",
    "async_tensor_model_parallel_allreduce",
    "tp_comm_overlap",
    "deallocate_pipeline_outputs",
    "bias_activation_fusion",
    "masked_softmax_fusion",
    "persist_layer_norm",
    "attention_softmax_in_fp32",
    "clone_scatter_output_in_embedding",
    "gradient_reduce_div_fusion",
)


def _inherited_env(parent: TransformerConfig) -> dict:
    out = {}
    fields = {f.name for f in TransformerConfig.__dataclass_fields__.values()}
    for name in _INHERITED_ENV_FIELDS:
        if name in fields and hasattr(parent, name):
            out[name] = getattr(parent, name)
    return out


def build_moonvit_configs(
    config: "KimiK3TransformerConfig",
) -> Tuple[KimiK3VisionTransformerConfig, KimiK3VisionProjectorConfig]:
    """Derive the tower and projector configs from a Kimi K3 config.

    Raises ``ValueError`` if the parent config has no vision tower
    configured, so the caller cannot silently get an empty tower.
    """
    if not config.vt_num_hidden_layers:
        raise ValueError(
            "this Kimi K3 config has no vision tower (vt_num_hidden_layers is unset). "
            "Set it to 27 for the released MoonViT-V2 geometry."
        )

    env = _inherited_env(config)
    heads = int(config.vt_num_attention_heads)
    qkv = int(config.vt_qkv_hidden_size or config.vt_hidden_size)
    merge = tuple(int(v) for v in config.vt_merge_kernel_size)

    tower = KimiK3VisionTransformerConfig(
        num_layers=int(config.vt_num_hidden_layers),
        hidden_size=int(config.vt_hidden_size),
        ffn_hidden_size=int(config.vt_intermediate_size),
        num_attention_heads=heads,
        num_query_groups=heads,  # no GQA in the tower
        kv_channels=qkv // heads,
        normalization="RMSNorm",
        layernorm_epsilon=float(config.vt_layernorm_epsilon),
        gated_linear_unit=False,
        activation_func=gelu_tanh,
        add_bias_linear=False,
        add_qkv_bias=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        apply_rope_fusion=False,
        # The tower is not pipelined: MoonViT is 0.4 B against the backbone's
        # 2.78 T, and Megatron's own VLM path pins the vision encoder to one
        # stage for the same reason (pretrain_vlm.py).
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        num_moe_experts=None,
        patch_size=int(config.vt_patch_size),
        in_channels=int(config.vt_in_channels),
        init_pos_emb_height=int(config.vt_init_pos_emb_height),
        init_pos_emb_width=int(config.vt_init_pos_emb_width),
        init_pos_emb_time=int(config.vt_init_pos_emb_time),
        pos_emb_interpolation_mode=str(config.vt_pos_emb_interpolation_mode),
        merge_kernel_size=merge,
        rope_theta=float(config.vt_rope_theta),
        rope_max_height=int(config.vt_rope_max_height),
        rope_max_width=int(config.vt_rope_max_width),
        attention_backend_name=str(config.vt_attention_backend),
        **env,
    )

    merged_width = int(config.mm_hidden_size) * merge[0] * merge[1]
    projector = KimiK3VisionProjectorConfig(
        num_layers=1,  # unused: MultimodalProjector builds an MLP, not a block
        hidden_size=int(config.hidden_size),
        ffn_hidden_size=merged_width,
        num_attention_heads=1,  # unused, but TransformerConfig requires it
        gated_linear_unit=False,
        # The projector's activation is nn.GELU(), the exact erf form, not
        # the tower's tanh approximation (modeling_kimi_k3.py).
        activation_func=F.gelu,
        add_bias_linear=False,
        normalization="RMSNorm",
        layernorm_epsilon=float(config.projector_ln_eps),
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        num_moe_experts=None,
        projector_input_size=merged_width,
        projector_ln_eps=float(config.projector_ln_eps),
        merge_kernel_size=merge,
        **env,
    )
    return tower, projector
