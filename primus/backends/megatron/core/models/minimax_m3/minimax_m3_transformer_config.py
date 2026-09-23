###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3 transformer config (MSA = MiniMax Sparse Attention).

M3's text tower is a plain GQA + MoE decoder, so it reuses upstream
``GPTModel`` unchanged. The one thing upstream ``TransformerConfig`` cannot
express is MSA: an index branch scores each 128-token KV block, the top
``sparse_topk_blocks`` blocks are kept per GQA group, and the main branch then
runs exact attention over that subset only.

``MSATransformerConfig`` therefore extends ``TransformerConfig`` -- *not*
``MLATransformerConfig``: M3 has no latent KV, and Megatron's own sparse
attention path (``experimental_attention_variant: dsa``) is MLA-only.
``minimax_sparse_attention`` is the family switch, mirroring how
``multi_latent_attention`` selects ``MLATransformerConfig``; the selection
itself lives in ``primus/backends/megatron/patches/minimax_m3_config_patches.py``.

M3's other two deviations from a stock GQA+MoE decoder -- the ``swigluoai``
activation and ``use_gemma_norm`` -- need no new machinery: Megatron already
computes both, so this config carries them under their ``config.json`` names
and derives the upstream fields in ``__post_init__``.

Values and semantics come from https://huggingface.co/MiniMaxAI/MiniMax-M3
(config.json) and from the official implementation in transformers,
``models/minimax_m3_vl/modeling_minimax_m3_vl.py`` (``MiniMaxM3VLRMSNorm``,
``MiniMaxM3VLDenseMLP``, ``MiniMaxM3VLExperts``). Field names are kept verbatim
so the preset reads like the reference config.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from megatron.core.transformer.transformer_config import TransformerConfig

_LayerPattern = Optional[Union[int, str, List[int], Tuple[int, ...]]]

# Block-score reductions MSA defines. Only `max` is in the released config, and
# anything else would change what the index branch means, so the rest are
# rejected rather than silently treated as `max`.
_SUPPORTED_SCORE_TYPES = frozenset({"max"})

# Kernels that can compute the block-sparse attention. `flydsl` is declared so a
# preset can name it; MinimaxSparseAttention rejects it until it exists.
_SUPPORTED_MSA_BACKENDS = frozenset({"eager", "flydsl"})


def normalize_sparse_layer_pattern(
    value: _LayerPattern, num_layers: int, *, field_name: str
) -> Tuple[int, ...]:
    """Normalize a per-layer 0/1 pattern to a tuple of length ``num_layers``.

    Accepts the same spellings as ``moe_layer_freq`` -- an int N (every Nth
    layer), a list, or a Python list expression such as ``"([0]*3+[1]*57)"`` --
    and reuses upstream's ``moe_freq_type`` to evaluate the expression form.
    These keys are not in Megatron's argparse, so Primus merges them onto
    ``args`` as the raw YAML value and the string never gets parsed on the way
    in; this is where that happens.

    ``None`` means "every layer", which is what a config that turns MSA on
    without narrowing it to a subset of layers should get.
    """
    if value is None:
        return (1,) * num_layers

    # Imported lazily: megatron.training pulls in the whole training stack,
    # which the config dataclass itself does not need.
    from megatron.training.arguments import moe_freq_type

    pattern = moe_freq_type(value) if isinstance(value, (int, str)) else value

    if isinstance(pattern, int):
        return tuple(1 if (i % pattern == 0) else 0 for i in range(num_layers))

    if not isinstance(pattern, (list, tuple)):
        raise TypeError(f"{field_name} must be an int, a list, or a list expression; got {value!r}")

    if len(pattern) != num_layers:
        raise ValueError(
            f"Invalid length of {field_name}: {len(pattern)}, expected num_layers={num_layers}; "
            f"current pattern: {value!r}"
        )

    invalid = sorted({entry for entry in pattern if entry not in (0, 1)})
    if invalid:
        raise ValueError(f"{field_name} entries must be 0 or 1; got {invalid}")

    return tuple(int(entry) for entry in pattern)


@dataclass
class MSATransformerConfig(TransformerConfig):
    """Configuration object for MiniMax Sparse Attention (MSA) transformers.

    Field names mirror ``config.json``'s ``sparse_attention_config``; the
    defaults are M3's released values, so a preset only has to set
    ``minimax_sparse_attention: true`` plus whatever it wants to change.
    """

    minimax_sparse_attention: bool = True
    """Whether to use MiniMax Sparse Attention. Selects this config class."""

    sparse_attention_freq: _LayerPattern = None
    """Per-layer 0/1 pattern: 1 = MSA layer, 0 = dense attention. None = all layers."""

    sparse_num_index_heads: int = 4
    """Number of heads in the index branch that scores KV blocks."""

    sparse_index_dim: int = 128
    """Dimension per index-branch head."""

    sparse_block_size: int = 128
    """Number of KV tokens per scored block."""

    sparse_topk_blocks: int = 16
    """Blocks kept per GQA group, i.e. sparse_topk_blocks * sparse_block_size KV tokens."""

    sparse_score_type: str = "max"
    """How a block's token scores reduce to one block score."""

    sparse_init_block: int = 0
    """Leading blocks always attended, on top of the top-k selection."""

    sparse_local_block: int = 1
    """Trailing (most recent) blocks always attended, on top of the top-k selection."""

    sparse_disable_index_value: _LayerPattern = None
    """Per-layer 0/1 pattern for the index branch; M3 ships the same list as the freq."""

    # ---- Activation: config.json's `hidden_act: swigluoai` ----

    swiglu_alpha: float = 1.702
    """Gate steepness. Megatron's `quick_gelu` hardcodes 1.702, so only that value is expressible."""

    swiglu_limit: float = 7.0
    """Clamp bound on both GLU halves; mirrored into `activation_func_clamp_value`."""

    # ---- Normalization: config.json's `use_gemma_norm` ----

    use_gemma_norm: bool = True
    """RMSNorm weighted by (1 + w); mirrored into `layernorm_zero_centered_gamma`."""

    # ---- MSA runtime (Primus-side, not from config.json) ----

    msa_backend: str = "eager"
    """Which kernel computes the block-sparse attention: 'eager' or 'flydsl'."""

    sparse_indexer_loss_coeff: float = 1.0e-2
    """Weight of the indexer distillation loss. 0.0 leaves the indexer frozen: top-k is not
    differentiable, so this loss is the only gradient its projections ever receive."""

    def __post_init__(self):
        # Derive the upstream fields BEFORE super(), so TransformerConfig's own
        # validation (bias_activation_fusion vs glu_linear_offset, etc.) sees
        # the values this model actually runs with.
        if self.minimax_sparse_attention:
            self._derive_activation()
            self._derive_normalization()

        super().__post_init__()

        if not self.minimax_sparse_attention:
            return

        if self.multi_latent_attention:
            raise ValueError(
                "minimax_sparse_attention and multi_latent_attention are mutually exclusive: "
                "MSA runs on GQA, and core_transformer_config_from_args replaces the config "
                "class with MLATransformerConfig whenever multi_latent_attention is set."
            )

        if self.msa_backend not in _SUPPORTED_MSA_BACKENDS:
            raise ValueError(
                f"msa_backend={self.msa_backend!r} is not a known backend; "
                f"supported: {sorted(_SUPPORTED_MSA_BACKENDS)}"
            )

        # The eager backend builds a dense [b, h, sq, sk] mask over the whole key
        # range on every rank, and reads `hidden_states` for the indexer -- which
        # sequence parallel leaves sharded as [sq/tp, b, h] while q/k/v are full
        # length. Neither is handled yet, so refuse loudly instead of silently
        # attending the wrong keys.
        if self.context_parallel_size != 1:
            raise NotImplementedError(
                "MiniMax Sparse Attention does not support context parallelism yet; "
                f"got context_parallel_size={self.context_parallel_size}."
            )
        if self.sequence_parallel:
            raise NotImplementedError(
                "MiniMax Sparse Attention does not support sequence parallelism yet; "
                "set `sequence_parallel: false`."
            )

        # The indexer emits one block selection per GQA group, which is what the
        # reference implementation means by index_n_heads == num_key_value_heads.
        num_query_groups = self.num_query_groups or self.num_attention_heads
        if self.sparse_num_index_heads != num_query_groups:
            raise ValueError(
                f"sparse_num_index_heads ({self.sparse_num_index_heads}) must equal the number of "
                f"GQA groups ({num_query_groups}): the indexer produces one block selection per "
                "group."
            )

        if self.sparse_score_type not in _SUPPORTED_SCORE_TYPES:
            raise NotImplementedError(
                f"sparse_score_type={self.sparse_score_type!r} is not implemented; "
                f"supported: {sorted(_SUPPORTED_SCORE_TYPES)}"
            )

        for name in (
            "sparse_num_index_heads",
            "sparse_index_dim",
            "sparse_block_size",
            "sparse_topk_blocks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")

        for name in ("sparse_init_block", "sparse_local_block"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative; got {getattr(self, name)}")

        self.sparse_attention_freq = normalize_sparse_layer_pattern(
            self.sparse_attention_freq, self.num_layers, field_name="sparse_attention_freq"
        )
        self.sparse_disable_index_value = normalize_sparse_layer_pattern(
            self.sparse_disable_index_value,
            self.num_layers,
            field_name="sparse_disable_index_value",
        )

    def _derive_activation(self) -> None:
        """Map `swigluoai` onto Megatron's quick-GEGLU fields.

        The official implementation (``MiniMaxM3VLDenseMLP.forward`` and
        ``MiniMaxM3VLExperts._apply_gate``) is::

            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=swiglu_limit)
            up   = up.clamp(-swiglu_limit, swiglu_limit)
            out  = (up + 1.0) * gate * sigmoid(gate * swiglu_alpha)

        and Megatron's GLU path (``mlp.py`` / ``moe/experts.py``) is::

            out = activation_func(x_glu) * (x_linear + glu_linear_offset)

        with the same two clamps and the same ``chunk`` split (M3 does *not*
        interleave the halves the way GPT-OSS does -- see that class's own
        comment). With ``activation_func = quick_gelu`` these are the same
        function, so the ``+1`` is ``glu_linear_offset`` and the limit is
        ``activation_func_clamp_value``.

        The one value Megatron cannot carry is a non-default alpha: `quick_gelu`
        hardcodes 1.702. That happens to be M3's `swiglu_alpha`, but a preset
        that changes it would otherwise train a different activation than it
        asked for, so it is rejected.

        Note the import: Megatron defines `quick_gelu` twice, in
        ``core/activations.py`` and in ``core/fusions/fused_bias_geglu.py``.
        The identity checks in ``arguments.py``, ``transformer_config.py``,
        ``mlp.py`` and ``moe/experts.py`` all use the fusions one, so that is
        the object a config actually carries and the one to compare against --
        the two are the same maths but different objects, and `is` tells them
        apart.
        """
        from megatron.core.fusions.fused_bias_geglu import quick_gelu

        if self.activation_func is not quick_gelu:
            raise ValueError(
                "MiniMax-M3 uses the `swigluoai` activation, which Megatron spells "
                "`quick_geglu`. Set `quick_geglu: true` (and `swiglu: false`) in the preset; "
                f"got activation_func={getattr(self.activation_func, '__name__', self.activation_func)}."
            )

        if self.swiglu_alpha != 1.702:
            raise NotImplementedError(
                f"swiglu_alpha={self.swiglu_alpha} is not expressible: Megatron's quick_gelu "
                "hardcodes 1.702. Only M3's released value is supported."
            )

        self.gated_linear_unit = True
        self.glu_linear_offset = 1.0
        self.activation_func_clamp_value = self.swiglu_limit

        # The MoE path fuses quick_geglu (TEGroupedMLP -> weighted_bias_quick_geglu_impl,
        # which applies both clamps), but the dense MLP's fused branch only covers
        # gelu and swiglu: quick_gelu without a per-token scale falls through to
        # `raise ValueError("Only support fusion of gelu and swiglu")` in mlp.py,
        # at forward time. M3 has dense layers, so catch it here instead.
        if self.bias_activation_fusion:
            raise ValueError(
                "MiniMax-M3's swigluoai activation cannot use bias_activation_fusion: the dense "
                "MLP's fused branch handles only gelu and swiglu, and would raise at the first "
                "forward. Set `bias_gelu_fusion: false` in the preset (that is the flag "
                "core_transformer_config_from_args reads when swiglu is off)."
            )

    def _derive_normalization(self) -> None:
        """Map `use_gemma_norm` onto `layernorm_zero_centered_gamma`.

        ``MiniMaxM3VLRMSNorm`` normalises in fp32, scales by ``1.0 + weight``
        and initialises ``weight`` at zeros -- exactly TE's
        ``RMSNorm(zero_centered_gamma=True)``
        (``megatron/core/extensions/transformer_engine.py``). Primus's turbo and
        fused-residual RMSNorm paths read the same flag, so the fused kernels
        stay correct.

        The reference uses that one class for every norm, q/k norms included;
        Megatron's qk norm is ``TENorm`` reading the same flag, so setting it
        once covers them all.
        """
        if not self.use_gemma_norm:
            return

        if self.normalization != "RMSNorm":
            raise ValueError(f"use_gemma_norm requires normalization=RMSNorm; got {self.normalization}.")

        self.layernorm_zero_centered_gamma = True

    @property
    def sparse_layer_pattern(self) -> Tuple[int, ...]:
        """Per-layer 0/1 tuple: 1 where the layer runs MSA, 0 where it stays dense."""
        if not self.minimax_sparse_attention:
            return (0,) * self.num_layers
        return normalize_sparse_layer_pattern(
            self.sparse_attention_freq, self.num_layers, field_name="sparse_attention_freq"
        )
