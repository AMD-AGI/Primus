###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""V4-Flash / V4-Pro attention shape fixtures (plan-4 P24).

This module exposes the **single source-of-truth** shape tables every
plan-4 test parametrises over. The canonical config knobs come from:

* ``primus/configs/models/megatron/deepseek_v4_flash.yaml``
* ``primus/configs/models/megatron/deepseek_v4_pro.yaml``
* ``primus/configs/models/megatron/deepseek_v4_base.yaml``

Each variant exposes three sequence-length tiers:

* ``small`` (S=128) — fast CI tier; runs in milliseconds on CPU
* ``medium`` (S=512) — moderate tier; runs in seconds on a single GPU
* ``large`` (S=4096) — release tier; gated behind ``pytest.mark.slow``

Each shape is parametrised by the ``compress_ratio`` of the layer
under test (``0`` dense + SWA + sink, ``128`` HCA, ``4`` CSA). Pool
size ``P = S // compress_ratio`` for compressed branches; the indexer
top-K (``K``) follows the per-variant ``index_topk`` knob (clamped to
``P`` when ``P < K``, mirroring how the indexer behaves at small
sequences).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

SeqTier = str  # "small" | "medium" | "large"

_SEQ_TIERS: Tuple[SeqTier, ...] = ("small", "medium", "large")
_SEQ_LENGTHS = {
    "small": 128,
    "medium": 512,
    "large": 4096,
}


@dataclass(frozen=True)
class V4AttnShape:
    """Single-shape parametrisation for a V4 attention layer test.

    Attributes follow the V4 attention class:

    * ``B`` — batch size
    * ``S`` — sequence length (post-RoPE; raw token count)
    * ``H`` — number of attention heads (per-rank; tests run TP=1)
    * ``head_dim`` — per-head ``kv_channels``
    * ``q_pe_dim`` — partial-RoPE rotary head dim (channels with RoPE
      applied; the remaining ``head_dim - q_pe_dim`` channels are
      "NOPE")
    * ``attn_sliding_window`` — SWA window for dense / CSA local branch
    * ``sink`` — ``True`` enables the per-head ``[H]`` learned softmax
      sink (matches V4 yaml)
    * ``compress_ratio`` — ``0`` dense, ``4`` CSA, ``128`` HCA
    * ``P`` — compressed pool size (``S // compress_ratio`` when
      ``compress_ratio > 0``; ``0`` for dense)
    * ``K`` — indexer top-K (``min(index_topk, P)`` for CSA;
      meaningless / ``0`` for dense / HCA but kept on the dataclass
      for API uniformity)
    * ``variant`` — display name ("v4_flash" / "v4_pro") for test ids
    """

    variant: str
    B: int
    S: int
    H: int
    head_dim: int
    q_pe_dim: int
    attn_sliding_window: int
    sink: bool
    compress_ratio: int
    P: int
    K: int

    @property
    def Sk(self) -> int:
        """Effective key axis length per branch.

        * dense (cr=0): ``S`` local keys
        * HCA (cr=128): ``S + P`` (local + compressed pool)
        * CSA (cr=4): ``S`` local keys (the sparse top-K branch is
          handled separately via ``gathered``)
        """
        if self.compress_ratio == 128:
            return self.S + self.P
        return self.S

    def shape_id(self) -> str:
        """A short human-readable test-id for pytest parametrisation."""
        return (
            f"{self.variant}-cr{self.compress_ratio}-S{self.S}"
            f"-H{self.H}-D{self.head_dim}"
            f"-sink{int(self.sink)}-w{self.attn_sliding_window}"
        )


# ---------------------------------------------------------------------------
# V4-Flash and V4-Pro variant tables
# ---------------------------------------------------------------------------

# Source: deepseek_v4_flash.yaml + deepseek_v4_base.yaml
_V4_FLASH_DEFAULTS = dict(
    H=64,
    head_dim=512,
    q_pe_dim=64,
    attn_sliding_window=128,
    sink=True,
    index_topk=512,
)

# Source: deepseek_v4_pro.yaml + deepseek_v4_base.yaml
_V4_PRO_DEFAULTS = dict(
    H=128,
    head_dim=512,
    q_pe_dim=64,
    attn_sliding_window=128,
    sink=True,
    index_topk=1024,
)


def _make_shape(
    variant: str,
    defaults: dict,
    *,
    compress_ratio: int,
    seq: int,
    batch: int,
    sink: Optional[bool] = None,
) -> V4AttnShape:
    H = int(defaults["H"])
    head_dim = int(defaults["head_dim"])
    q_pe_dim = int(defaults["q_pe_dim"])
    attn_sliding_window = int(defaults["attn_sliding_window"])
    sink_default = bool(defaults["sink"])
    index_topk = int(defaults["index_topk"])

    if compress_ratio == 0:
        P = 0
        K = 0
    elif compress_ratio == 4:
        P = max(seq // compress_ratio, 1)
        K = min(index_topk, P)
    elif compress_ratio == 128:
        P = max(seq // compress_ratio, 1)
        K = 0  # not used by HCA
    else:
        raise ValueError(f"compress_ratio must be in {{0, 4, 128}}; got {compress_ratio}.")

    return V4AttnShape(
        variant=variant,
        B=int(batch),
        S=int(seq),
        H=H,
        head_dim=head_dim,
        q_pe_dim=q_pe_dim,
        attn_sliding_window=attn_sliding_window,
        sink=sink_default if sink is None else bool(sink),
        compress_ratio=int(compress_ratio),
        P=int(P),
        K=int(K),
    )


def v4_flash_shape(
    *,
    compress_ratio: int,
    seq_tier: SeqTier = "small",
    batch: int = 1,
    sink: Optional[bool] = None,
) -> V4AttnShape:
    """Build a single V4-Flash :class:`V4AttnShape`."""
    if seq_tier not in _SEQ_LENGTHS:
        raise ValueError(f"seq_tier must be one of {_SEQ_TIERS}; got {seq_tier!r}.")
    return _make_shape(
        "v4_flash",
        _V4_FLASH_DEFAULTS,
        compress_ratio=compress_ratio,
        seq=_SEQ_LENGTHS[seq_tier],
        batch=batch,
        sink=sink,
    )


def v4_pro_shape(
    *,
    compress_ratio: int,
    seq_tier: SeqTier = "small",
    batch: int = 1,
    sink: Optional[bool] = None,
) -> V4AttnShape:
    """Build a single V4-Pro :class:`V4AttnShape`."""
    if seq_tier not in _SEQ_LENGTHS:
        raise ValueError(f"seq_tier must be one of {_SEQ_TIERS}; got {seq_tier!r}.")
    return _make_shape(
        "v4_pro",
        _V4_PRO_DEFAULTS,
        compress_ratio=compress_ratio,
        seq=_SEQ_LENGTHS[seq_tier],
        batch=batch,
        sink=sink,
    )


def v4_attention_shape_grid(
    *,
    variants: Iterable[str] = ("v4_flash", "v4_pro"),
    compress_ratios: Iterable[int] = (0, 4, 128),
    seq_tiers: Iterable[SeqTier] = ("small",),
    batch: int = 1,
    sinks: Iterable[bool] = (True, False),
) -> list[V4AttnShape]:
    """Cartesian product of (variant × compress_ratio × seq_tier × sink).

    Returns a flat list of :class:`V4AttnShape` suitable for direct
    pytest parametrisation. Plan-4 tests typically pin to a small
    subset to keep CI green:

    * G22 (P24 refactor safety net): all three compress ratios at
      ``small`` seq tier with both sink modes.
    * G23 / G24 (P25 fwd / bwd): cr ∈ {0, 128}, seq ∈ {small, medium}.
    * G26 / G27 (P26 CSA fwd / bwd): cr == 4, seq ∈ {small, medium}.
    """
    out: list[V4AttnShape] = []
    builders = {
        "v4_flash": v4_flash_shape,
        "v4_pro": v4_pro_shape,
    }
    for variant in variants:
        if variant not in builders:
            raise ValueError(f"variant must be one of {tuple(builders)}; got {variant!r}.")
        builder = builders[variant]
        for cr in compress_ratios:
            for tier in seq_tiers:
                for sink in sinks:
                    out.append(
                        builder(
                            compress_ratio=cr,
                            seq_tier=tier,
                            batch=batch,
                            sink=sink,
                        )
                    )
    return out


__all__ = [
    "SeqTier",
    "V4AttnShape",
    "v4_attention_shape_grid",
    "v4_flash_shape",
    "v4_pro_shape",
]
