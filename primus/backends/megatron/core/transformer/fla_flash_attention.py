###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Drop-in `core_attention` replacement for Megatron MLA that calls
`flash_attn_func` directly, bypassing TransformerEngine's version check
and CK/SDPA fallback.

This module exists because on ROCm/MI300X Transformer-Engine's
`TEDotProductAttention` performs a strict version range check on the
installed `flash-attn` package (currently ``2.1.1 <= x <= 2.8.1``).  Newer
builds (e.g. 2.8.3 shipped in the production container) fail that check
and TE silently falls back to its Composable-Kernel backend, which on
small MLA dims (n_heads × head_dim = 16 × 64) loses ~30 ms per layer
versus calling flash-attn directly.

The wrapper matches FLA's reference path in
``fla/layers/mla.py`` -- a single call to
``flash_attn_func(q, k, v, causal=True, softmax_scale=…)`` with no other
overhead.

Activation
----------
The wrapper auto-enables whenever the installed ``flash_attn`` version is
outside Transformer-Engine's supported range (currently ``<= 2.8.1``).
Override with:

* ``PRIMUS_FLA_MLA_ATTN=1`` -- force-enable (use wrapper).
* ``PRIMUS_FLA_MLA_ATTN=0`` -- force-disable (use TE's
  ``TEDotProductAttention``, which on newer flash-attn drops to CK fallback).

When enabled, the hybrid GDN/KDA spec swaps
``core_attention=TEDotProductAttention`` for
``core_attention=FLAFlashAttention`` automatically (see
``primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs``).

Caveats
-------
* Only supports the unpacked training path (``packed_seq_params is None``).
  Varlen / packed sequences are not implemented; the wrapper will raise.
* Always uses causal masking (matches Megatron's MLA spec
  ``params={"attn_mask_type": AttnMaskType.causal}`` and FLA's MLA path).
* ``current_max_attn_logits`` is exposed as ``None`` so MLA's optional
  qk_clip path (disabled in our configs) keeps importing without error.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.enums import AttnMaskType

# TransformerEngine's pinned flash-attn version range.  When the installed
# `flash_attn` is newer (e.g. 2.8.3 in the production container), TE silently
# falls back to its Composable-Kernel backend, which is ~30 ms/MLA-block
# slower on MI300X.  We use the upper bound to decide whether the wrapper
# should auto-enable.
_TE_FLASH_ATTN_MAX_SUPPORTED = (2, 8, 1)


def _parse_version(ver: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for tok in ver.split("."):
        digits = "".join(c for c in tok if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]


def _flash_attn_exceeds_te_range() -> bool:
    """True if the installed flash-attn is newer than TE supports."""
    try:
        import flash_attn  # type: ignore
    except Exception:
        return False
    ver = getattr(flash_attn, "__version__", "0.0.0")
    return _parse_version(ver) > _TE_FLASH_ATTN_MAX_SUPPORTED


def is_enabled() -> bool:
    """Return True if MLA should use the direct flash-attn path.

    Precedence:
      1. ``PRIMUS_FLA_MLA_ATTN=0`` → force-disable (use TE).
      2. ``PRIMUS_FLA_MLA_ATTN=1`` → force-enable (use wrapper).
      3. Unset → auto-enable whenever the installed flash-attn version is
         outside TE's supported range (e.g. 2.8.3 > 2.8.1), since that's
         exactly the case where TE silently drops to its slower CK backend.
    """
    env = os.environ.get("PRIMUS_FLA_MLA_ATTN")
    if env is not None:
        return env == "1"
    return _flash_attn_exceeds_te_range()


# Lazy import of flash_attn so the module can be imported on hosts where
# flash-attn isn't installed (e.g. dev machines). The actual import only
# happens when the layer is constructed inside the training container.
_flash_attn_func = None

# One-shot banner so it is obvious from the training log whether the
# wrapper actually got instantiated.  Without this it can be ambiguous
# because the FLA path and TE+CK path produce near-identical loss/speed
# numbers on small attention dims.
_BANNER_PRINTED = False


def _load_flash_attn():
    global _flash_attn_func
    if _flash_attn_func is not None:
        return _flash_attn_func
    try:
        from flash_attn import flash_attn_func  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PRIMUS_FLA_MLA_ATTN=1 was set, but `flash_attn` is not "
            "importable in this Python.  Install it with "
            "`pip install flash-attn --no-build-isolation` and retry."
        ) from exc
    _flash_attn_func = flash_attn_func
    return _flash_attn_func


class FLAFlashAttention(MegatronModule):
    """`core_attention` plug-in that routes through `flash_attn_func`.

    Constructor signature is intentionally a superset of
    ``TEDotProductAttention.__init__`` so ``build_module(...)`` from
    ``MLASelfAttention`` can swap them without any other change.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Any = None,
        **_unused_kwargs: Any,
    ) -> None:
        super().__init__(config=config)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.softmax_scale = softmax_scale
        self.k_channels = k_channels
        self.v_channels = v_channels
        # Exposed for MLA's optional qk_clip code path (disabled by default
        # in our configs).  Keeping it on the instance silences AttributeError.
        self.current_max_attn_logits = None

        # Eager import so failures surface at model build time, not at
        # the first forward.
        _load_flash_attn()

        global _BANNER_PRINTED
        if not _BANNER_PRINTED:
            try:
                import flash_attn as _fa
                _ver = getattr(_fa, "__version__", "unknown")
            except Exception:
                _ver = "unknown"
            env = os.environ.get("PRIMUS_FLA_MLA_ATTN")
            if env == "1":
                _reason = "PRIMUS_FLA_MLA_ATTN=1"
            elif env is None:
                _reason = (
                    f"auto-enabled (flash_attn {_ver} > TE max "
                    f"{'.'.join(str(x) for x in _TE_FLASH_ATTN_MAX_SUPPORTED)})"
                )
            else:
                _reason = f"PRIMUS_FLA_MLA_ATTN={env}"
            _msg = (
                f"[PRIMUS_FLA_MLA_ATTN] FLAFlashAttention active "
                f"(layer={layer_number}, softmax_scale={softmax_scale}, "
                f"k_channels={k_channels}, v_channels={v_channels}, "
                f"flash_attn={_ver}, reason={_reason}). "
                f"This banner prints once per worker."
            )
            # Megatron silently consumes plain `print()` from rank-non-zero
            # workers (and sometimes from rank-0 once its logger is set up),
            # so emit to stderr -- which the run_pretrain.sh tee pipeline
            # still captures -- AND drop a marker file so activation is
            # provable even if all stdio gets eaten.
            print(_msg, file=sys.stderr, flush=True)
            try:
                rank = int(os.environ.get("RANK", "-1"))
                marker = f"/tmp/primus_fla_mla_attn_active.rank{rank}.txt"
                with open(marker, "w") as fh:
                    fh.write(_msg + "\n")
                    fh.write(f"pid={os.getpid()} ts={time.time()}\n")
            except Exception:
                pass
            _BANNER_PRINTED = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # ignored; causal=True
        attn_mask_type: Optional[AttnMaskType] = None,
        packed_seq_params: Any = None,
        **_unused_kwargs: Any,
    ) -> torch.Tensor:
        # Megatron's MLA always builds with attn_mask_type=causal.  The
        # `attention_mask` arg is unused for causal attention; we ignore it.
        if packed_seq_params is not None:
            raise NotImplementedError(
                "FLAFlashAttention does not yet support packed_seq_params; "
                "either disable PRIMUS_FLA_MLA_ATTN or run without "
                "sequence packing."
            )

        mask_type = attn_mask_type if attn_mask_type is not None else self.attn_mask_type
        if mask_type != AttnMaskType.causal:
            raise NotImplementedError(
                f"FLAFlashAttention only supports causal masking; got {mask_type}. "
                "Disable PRIMUS_FLA_MLA_ATTN if you need a different mask."
            )

        flash_attn_func = _load_flash_attn()

        # Megatron passes q/k/v as [s, b, h, d]; flash-attn expects [b, s, h, d].
        # The `.contiguous()` is required after `.transpose(0, 1)` because
        # flash-attn checks for contiguous last-dim-fastest layout.
        q = query.transpose(0, 1).contiguous()
        k = key.transpose(0, 1).contiguous()
        v = value.transpose(0, 1).contiguous()

        # FLA's MLA path:
        #   o = flash_attn_func(q, k, v, causal=True, softmax_scale=…)
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=True,
        )
        # out: [b, s, h, d_v]  ->  Megatron expects [s, b, h*d_v]
        out = out.transpose(0, 1).contiguous()
        s, b, h, d_v = out.shape
        return out.view(s, b, h * d_v)
