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

Values come from https://huggingface.co/MiniMaxAI/MiniMax-M3 (config.json,
``text_config.sparse_attention_config``); the field names are kept verbatim so
the preset reads like the reference config.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from megatron.core.transformer.transformer_config import TransformerConfig

_LayerPattern = Optional[Union[int, str, List[int], Tuple[int, ...]]]

# Block-score reductions MSA defines. Only `max` is in the released config, and
# anything else would change what the index branch means, so the rest are
# rejected rather than silently treated as `max`.
_SUPPORTED_SCORE_TYPES = frozenset({"max"})


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

    def __post_init__(self):
        super().__post_init__()

        if not self.minimax_sparse_attention:
            return

        if self.multi_latent_attention:
            raise ValueError(
                "minimax_sparse_attention and multi_latent_attention are mutually exclusive: "
                "MSA runs on GQA, and core_transformer_config_from_args replaces the config "
                "class with MLATransformerConfig whenever multi_latent_attention is set."
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

    @property
    def sparse_layer_pattern(self) -> Tuple[int, ...]:
        """Per-layer 0/1 tuple: 1 where the layer runs MSA, 0 where it stays dense."""
        if not self.minimax_sparse_attention:
            return (0,) * self.num_layers
        return normalize_sparse_layer_pattern(
            self.sparse_attention_freq, self.num_layers, field_name="sparse_attention_freq"
        )
