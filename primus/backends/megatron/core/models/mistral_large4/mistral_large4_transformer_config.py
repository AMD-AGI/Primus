###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``mistral_large4_text``) transformer config.

An ``MLATransformerConfig`` (MLA without Q-LoRA, YaRN rope applied interleaved) with the
per-layer patterns of the released checkpoint:

* ``no_rope_layers`` (1 = rope, 0 = NoPE; HF convention): on a NoPE layer the MLA has
  ``qk_pos_emb_head_dim = 0`` and ``qk_head_dim = qk_head_dim + qk_pos_emb_head_dim``;
* ``indexer_types`` ("full" / "shared"): a shared layer reuses the DSA top-k indices of the
  previous full layer;
* ``kv_source_layers``: a layer whose source is another layer reuses that layer's normalized
  latent KV instead of computing its own;
* SSMax (learned per-head ``s * log(pos + 1) + b`` query scale), LeRoPE (learned
  per-frequency ``exp(alpha)`` on the rope inverse frequencies) and the final logit softcap.

Per-layer lists longer than ``num_layers`` are cut to their prefix, so a depth-reduced run
(``--num_layers 4``) keeps the released pattern of its first layers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Union

from megatron.core.transformer.transformer_config import MLATransformerConfig

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    DEFAULT_FIRST_K_DENSE_REPLACE,
    check_kv_sources,
    check_layer_list,
    kv_carrier_needed,
    kv_sharing_active,
    moe_layer_freq_list,
    normalize_layer_list,
)

__all__ = [
    "MistralLarge4TransformerConfig",
    "kv_carrier_needed",
    "kv_sharing_active",
    "moe_layer_freq_list",
    "normalize_layer_list",
]


@dataclass
class MistralLarge4TransformerConfig(MLATransformerConfig):
    """Config for the Mistral Large 4 text decoder."""

    # 1 = the layer applies rope, 0 = NoPE (HF ``no_rope_layers``).
    no_rope_layers: Optional[Union[str, List[int]]] = None
    # "full" or "shared" per layer (HF ``indexer_types``).
    indexer_types: Optional[Union[str, List[str]]] = None
    # The layer whose latent KV a layer uses (HF ``kv_source_layers``).
    kv_source_layers: Optional[Union[str, List[int]]] = None

    # DSA lightning indexer (Mistral variant: input LayerNorm, own SSMax, no Hadamard)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_norm_eps: float = 1e-6
    # accept sparse attention (sequence > index_topk) with an indexer that was never loaded (its
    # init values): code-path tests only, the selection is then random and does not train
    allow_untrained_indexer: bool = False

    # final logits: cap * tanh(logits / cap)
    final_logit_softcapping: Optional[float] = None

    # dense MLP on the first layers
    first_k_dense_replace: int = DEFAULT_FIRST_K_DENSE_REPLACE

    # the training sequence length (args.seq_length, set by the builder): with KV sharing across
    # pipeline stages a stage checks the received P2P tensor against it (decoder block
    # _split_input). None: not known (a config built without args), the check is skipped.
    pp_carrier_seq_length: Optional[int] = None

    def __post_init__(self) -> None:
        # always MLA (the builder may construct it from args with the flag off, see builders)
        self.multi_latent_attention = True
        # per-layer torch_dist keys (decoder.layers.<i>.*): the layers differ in shape (NoPE kv_a /
        # kv_b, KV-sharing consumers without kv_a, shared-indexer layers without an indexer, dense
        # layer 0), so they cannot be stacked along a layer axis
        self.hetereogenous_dist_checkpoint = True
        n = int(self.num_layers)
        self.no_rope_layers = normalize_layer_list(self.no_rope_layers, n, "no_rope_layers", int, 1)
        self.indexer_types = normalize_layer_list(self.indexer_types, n, "indexer_types", str, "full")
        self.kv_source_layers = normalize_layer_list(
            self.kv_source_layers, n, "kv_source_layers", int, lambda i: i
        )
        check_layer_list("no_rope_layers", self.no_rope_layers)  # 0 / 1 only (2 is not "rope")
        check_layer_list("indexer_types", self.indexer_types)
        if self.indexer_types and self.indexer_types[0] == "shared":
            raise ValueError("indexer_types[0] is 'shared': layer 0 has no earlier top-k to reuse")
        check_kv_sources(self.kv_source_layers, self.no_rope_layers)
        sources = sorted({s for i, s in enumerate(self.kv_source_layers) if s != i})
        # the one layer whose latent other layers reuse (kv_source_layer); checked just below
        self._kv_source_layer = sources[0] if sources else None
        if len(sources) > 1:
            # the decoder block (and its pipeline carrier) holds the latent of one source layer
            raise NotImplementedError(
                f"Mistral Large 4: KV sharing from more than one source layer ({sources}) is not supported; "
                "the released checkpoint shares only layer 35's latent"
            )
        if self.q_lora_rank is not None:
            raise ValueError("Mistral Large 4 has no Q-LoRA: q_lora_rank must be None")
        if self.mscale_all_dim:
            raise ValueError("Mistral Large 4 applies no YaRN softmax mscale: mscale_all_dim must be 0")
        super().__post_init__()

    # ---- per-layer helpers (0-based layer index) ----------------------------
    def uses_rope(self, layer_idx: int) -> bool:
        return bool(self.no_rope_layers[layer_idx])

    def index_source_layer(self, layer_idx: int) -> int:
        """The full layer whose top-k a layer uses (itself for a full layer)."""
        i = layer_idx
        while self.indexer_types[i] == "shared":
            i -= 1
        return i

    def kv_source(self, layer_idx: int) -> int:
        return int(self.kv_source_layers[layer_idx])

    def stores_kv(self, layer_idx: int) -> bool:
        return self.kv_source(layer_idx) == layer_idx and layer_idx in self.kv_source_layers[layer_idx + 1 :]

    @property
    def kv_source_layer(self) -> Optional[int]:
        """The one layer whose latent other layers reuse, or None (computed in ``__post_init__``)."""
        return self._kv_source_layer

    def layer_config(self, layer_idx: int) -> "MistralLarge4TransformerConfig":
        """This config with the MLA head split of layer ``layer_idx`` (NoPE: all dims content)."""
        if self.uses_rope(layer_idx) or not self.qk_pos_emb_head_dim:
            return self
        cfg = copy.copy(self)
        cfg.qk_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        cfg.qk_pos_emb_head_dim = 0
        # MLA builds its own (unused here) rope table from the config; with no rope dims a plain
        # rope table of size 0 is built instead of YaRN's, which cannot be sliced per CP rank
        cfg.rope_type = "rope"
        return cfg
