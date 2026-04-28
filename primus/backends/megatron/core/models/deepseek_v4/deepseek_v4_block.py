###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 transformer block (Phase 3 scaffolding).

This is a transparent subclass of
:class:`megatron.core.transformer.transformer_block.TransformerBlock`. In
Phase 3 it just stashes the V4-specific config fields onto ``self`` so the
Phase-4 patches that swap in :class:`HyperConnection` /
:class:`HyperConnectionHead` can read them without re-walking ``config``.

Phase 4 will:

* Wrap each ``self.layers[i]`` with a per-layer :class:`HyperConnection`
  (``hc_mult`` parallel hidden streams; per-layer
  :func:`sinkhorn_normalize`).
* Insert a final :class:`HyperConnectionHead` that collapses the K streams
  back to one hidden state before the LM head.
* Dispatch attention by ``compress_ratios[layer_id]`` (Dense / HCA / CSA).

Phase 4 will also rewrite this class to be a standalone :class:`torch.nn.Module`
(no longer inheriting from ``TransformerBlock``) so the multi-stream HC
loop can be expressed cleanly. Phase 3 keeps the subclass form so the
end-to-end dispatch path stays runnable.
"""

from typing import List, Optional, Sequence

from megatron.core.transformer.transformer_block import TransformerBlock


class DeepseekV4TransformerBlock(TransformerBlock):
    """V4 decoder block.

    Phase 3 surface: identical to ``TransformerBlock`` except that V4 config
    fields are unpacked into attributes on ``self``. Phase 4 will override
    ``forward`` and ``__init__`` to add the K-stream HC machinery and
    per-layer attention dispatch.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        config = self.config

        # ---- mHC / HC fields (Phase 4) -----------------------------------
        self.hc_mult: int = int(getattr(config, "hc_mult", 1) or 1)
        self.hc_eps: float = float(getattr(config, "hc_eps", 1.0e-6) or 1.0e-6)
        self.hc_sinkhorn_iters: int = int(getattr(config, "hc_sinkhorn_iters", 20) or 20)

        # ---- per-layer attention dispatch (Phase 4) ----------------------
        compress_ratios: Optional[Sequence[int]] = getattr(config, "compress_ratios", None)
        if compress_ratios is None:
            compress_ratios = [0] * config.num_layers
        compress_ratios = list(compress_ratios)
        if len(compress_ratios) != config.num_layers:
            raise ValueError(
                "compress_ratios length " f"{len(compress_ratios)} != num_layers {config.num_layers}"
            )
        self.compress_ratios: List[int] = compress_ratios

        # ---- attention shared knobs (Phase 4) ----------------------------
        self.attn_sliding_window: int = int(getattr(config, "attn_sliding_window", 0) or 0)
        self.attn_sink_enabled: bool = bool(getattr(config, "attn_sink", False))
        self.q_lora_rank: Optional[int] = getattr(config, "q_lora_rank", None) or None

        # ---- Indexer (CSA, Phase 4) --------------------------------------
        self.index_topk: int = int(getattr(config, "index_topk", 512) or 512)
        self.index_head_dim: int = int(getattr(config, "index_head_dim", 128) or 128)
        self.index_n_heads: int = int(getattr(config, "index_n_heads", 64) or 64)


__all__ = ["DeepseekV4TransformerBlock"]
