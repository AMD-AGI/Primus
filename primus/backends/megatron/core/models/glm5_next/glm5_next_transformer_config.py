###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``glm5_next``) transformer config.

GLM-5.3-Flash is a 45-layer hybrid:

* **KDA** (Kimi Delta Attention, linear attention) on every layer with
  ``idx % 4 != 3``; geometry reuses upstream's ``linear_*`` fields and the
  Kimi-K3 ``kda_*`` fields so :class:`KimiDeltaAttention` can be shared.
* **DSA** (NoPE absorbed MLA + kpool lightning indexer) on layers
  ``3, 7, ..., 43``. ``q_lora_rank`` / ``kv_lora_rank`` / ``qk_head_dim`` /
  ``v_head_dim`` describe the MLA geometry (``qk_rope_head_dim`` is 0, so
  there is no positional head at all); ``index_*`` describe the indexer.
* **mHC** (manifold-constrained hyper-connections) around every sublayer,
  ``hc_mult`` parallel residual streams, contracted by a plain mean before
  the final norm (no learned ``hc_head``).
* Dense SwiGLU FFN on the first ``first_k_dense_replace`` layers, then MoE
  (sigmoid router with expert bias, shared expert). Every MLP clamps its
  pre-activations at ``swiglu_limit`` via Megatron's
  ``activation_func_clamp_value``.

Like Kimi-K3, ``multi_latent_attention`` must stay false: the DSA layers are
built from this family's own specs, and ``core_transformer_config_from_args``
would otherwise replace this class with ``MLATransformerConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    normalize_linear_attention_freq,
)

__all__ = ["Glm5NextTransformerConfig"]

DSA_BACKENDS = ("triton", "torch")


@dataclass
class Glm5NextTransformerConfig(TransformerConfig):
    """Config for the GLM-5.3 (``glm5_next``) text backbone."""

    # ---- per-layer attention pattern: 1 = KDA, 0 = DSA ------------------
    linear_attention_freq: Optional[Union[int, str, List[int]]] = None

    # ---- KDA (shared with Kimi-K3's KimiDeltaAttention) -----------------
    kda_gate_lower_bound: Optional[float] = -5.0
    # GLM-5.3 uses the low-rank g_a_proj -> g_b_proj output gate.
    kda_use_full_rank_gate: bool = False
    kda_backend: str = "fla"
    kda_chunk_size: int = 64

    # ---- DSA: NoPE absorbed MLA -----------------------------------------
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    v_head_dim: int = 256

    # ---- DSA: kpool lightning indexer -----------------------------------
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    # Indexer k_norm is a LayerNorm (with bias); SGLang builds it with eps 1e-6.
    index_k_norm_eps: float = 1e-6
    # triton: Triton pooled-key / indexer-score / sparse-MLA kernels.
    # torch: pure PyTorch reference (materialises [s, s] scores; tests only).
    dsa_backend: str = "triton"
    # The released indexer is trained separately; keep its compress gate / ape
    # frozen like the Miles reference does.
    index_freeze_kpool_compress: bool = True

    # ---- mHC ------------------------------------------------------------
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    hc_post_mult: float = 2.0

    # ---- MoE / dense ----------------------------------------------------
    first_k_dense_replace: int = 3

    def __post_init__(self) -> None:
        self.linear_attention_freq = normalize_linear_attention_freq(
            self.linear_attention_freq,
            num_layers=int(self.num_layers),
            field_name="linear_attention_freq",
        )
        if self.dsa_backend not in DSA_BACKENDS:
            raise ValueError(f"dsa_backend must be one of {DSA_BACKENDS}; got {self.dsa_backend!r}")
        if int(self.index_topk) % int(self.index_kpool) != 0:
            raise ValueError(
                f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool})"
            )
        if int(self.hc_mult) < 1:
            raise ValueError(f"hc_mult must be >= 1, got {self.hc_mult}")
        # NoPE everywhere: DSA has qk_rope_head_dim == 0, KDA carries position
        # through its recurrence and causal short convolution.
        self.position_embedding_type = "none"
        self.is_hybrid_model = True
        super().__post_init__()

    def is_kda_layer(self, layer_idx: int) -> bool:
        if self.linear_attention_freq is None:
            return False
        return bool(self.linear_attention_freq[layer_idx])

    @property
    def dsa_layer_indices(self) -> List[int]:
        return [i for i in range(int(self.num_layers)) if not self.is_kda_layer(i)]
