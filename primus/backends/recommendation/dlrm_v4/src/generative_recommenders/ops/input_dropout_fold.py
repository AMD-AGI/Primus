# pyre-strict
"""Fold the HSTU input dropout's backward into its consumer's dx epilogue.

The transducer applies ``F.dropout(posenc_out, p)`` before the STU stack. Its
backward is ``aten::native_dropout_backward`` (masked_scale_kernel), a full
[N, D] bf16 read + bool mask read + [N, D] write whose only input is the dx
written by the layer-0 input-LayerNorm backward (the residual-fused
``_weighted_layer_norm_bwd_dx_v2`` call). This module lets that kernel apply
``grad * mask * scale`` in its epilogue instead.

Forward is unchanged: ``aten::native_dropout`` is called with the same
arguments, so output, mask and Philox consumption are identical. The mask and
scale ride on the output tensor (a ``DropoutHandoff`` attribute). A consumer
that folds records the dx tensor it produced; this Function's backward then
passes that exact tensor through. If no consumer folded, the backward is the
stock ``native_dropout_backward`` (bitwise the old path). If a consumer folded
but the incoming gradient is not the tensor it produced (another consumer
contributed, so the gradient is a sum and part of it was already masked), the
backward raises instead of silently returning a wrong gradient.

Numerics: aten's masked_scale computes ``(float)mask * grad * (float)scale``
in fp32 and rounds to bf16; the epilogue does the same ops in the same order
on the bf16-rounded dx, so the result is bit-identical (including signed
zeros of dropped elements).

With DLRM_HSTU_FP8_DROPOUT_MASK_BITS (fused posenc+dropout path,
ops/triton/triton_position.py) the handoff's mask is bit-packed (uint8
[N, D / 8], pack_dropout_mask_bits) when the layer-0 LN fold can take it (the
producer checks triton_layer_norm.wln_bwd_mscale_bits_ok); the epilogue tests
the bit, and the native_dropout_backward fallback (no consumer folded) unpacks
it exactly, at the cost of extra elementwise passes.

Switch: DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD (gin default on, yambda_5b.gin item
(19)); absent or =0 keeps ``F.dropout``.
"""

import logging
import os
from typing import Optional, Tuple

import torch
from torch.fx._symbolic_trace import is_fx_tracing

_FOLD_ENABLED: bool = os.environ.get("DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD", "0") != "0"

DROPOUT_HANDOFF_ATTR = "_hstu_input_dropout_handoff"

_logger: logging.Logger = logging.getLogger(__name__)
_LOGGED: set = set()


def _log_once(key: str, msg: str) -> None:
    # One line per process per path (arming proof); never per step.
    if key not in _LOGGED:
        _LOGGED.add(key)
        _logger.info(msg)


def input_dropout_fold_enabled() -> bool:
    return _FOLD_ENABLED


def pack_dropout_mask_bits(mask: torch.Tensor) -> torch.Tensor:
    """bool [N, D] -> the packed layout of DLRM_HSTU_FP8_DROPOUT_MASK_BITS:
    uint8 [N, ceil(D / 8)], bit k of byte j of a row = column 8 * j + k (bits
    past column D - 1 are 0). Reference / test helper (torch ops)."""
    N, D = mask.shape
    pad = -D % 8
    m = mask.to(torch.uint8)
    if pad:
        m = torch.nn.functional.pad(m, (0, pad))
    shifts = torch.arange(8, device=mask.device, dtype=torch.uint8)
    return (m.reshape(N, -1, 8) << shifts).sum(-1, dtype=torch.uint8)


def unpack_dropout_mask_bits(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Exact inverse of pack_dropout_mask_bits: uint8 [N, ceil(D / 8)] -> bool
    [N, D]. Used by the fallbacks that need aten's bool mask (off the hot
    path)."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :D].to(torch.bool)


class DropoutHandoff:
    __slots__ = ("mask", "scale", "claimed", "folded_dx", "mask_bits", "shape")

    def __init__(
        self,
        mask: torch.Tensor,
        scale: float,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        # bool [N, D] (aten's mask), or with DLRM_HSTU_FP8_DROPOUT_MASK_BITS
        # (fused posenc+dropout path) uint8 [N, D / 8] packed bits
        # (pack_dropout_mask_bits layout); `shape` is the dropout output's.
        self.mask: torch.Tensor = mask
        # Same value autograd passes to native_dropout_backward (a double;
        # the kernel casts it to its fp32 opmath type).
        self.scale: float = scale
        self.claimed: bool = False
        self.folded_dx: Optional[torch.Tensor] = None
        self.mask_bits: bool = mask.dtype == torch.uint8
        if self.mask_bits:
            # [N, D] cannot be recovered from the packed [N, ceil(D / 8)]
            assert shape is not None and tuple(mask.shape) == (
                *tuple(shape)[:-1],
                -(-shape[-1] // 8),
            ), f"packed dropout mask {tuple(mask.shape)} needs its [N, D] shape, got {shape}"
        self.shape: Tuple[int, ...] = tuple(mask.shape if shape is None else shape)

    def claim(self) -> bool:
        """At most one consumer may fold; the first to claim wins."""
        if self.claimed:
            return False
        self.claimed = True
        return True

    def bool_mask(self) -> torch.Tensor:
        """aten's bool [N, D] mask (unpacked when the mask is bit-packed)."""
        if self.mask_bits:
            return unpack_dropout_mask_bits(self.mask, self.shape[-1])
        return self.mask


def take_dropout_handoff(x: torch.Tensor) -> Optional[DropoutHandoff]:
    """Claim the handoff carried by ``x`` (the dropout output), if any."""
    h = getattr(x, DROPOUT_HANDOFF_ATTR, None)
    if not isinstance(h, DropoutHandoff):
        return None
    if not h.claim():
        return None
    return h


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


def dropout_handoff_backward(h: DropoutHandoff, grad: torch.Tensor) -> torch.Tensor:
    """The dropout input's gradient for handoff ``h``: the folded dx passed
    through, or ``native_dropout_backward`` if no consumer folded. Shared by
    _InputDropoutHandoffFunction and the posenc+dropout fusion
    (ops/triton/triton_position.py, DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE)."""
    folded = h.folded_dx
    h.folded_dx = None
    if folded is None:
        _log_once(
            "fallback",
            "input-dropout mask/scale fold: no consumer folded -> "
            "native_dropout_backward (fallback)",
        )
        if h.mask_bits and "fallback_packed" not in _LOGGED:
            # exact, but about +3.4 ms / step at the production shape: the
            # producer's layout decision missed a wiring that does not fold
            _LOGGED.add("fallback_packed")
            _logger.warning(
                "DLRM_HSTU_FP8_DROPOUT_MASK_BITS: a bit-packed dropout mask was "
                "not folded and is unpacked for native_dropout_backward (slow "
                "path); set DLRM_HSTU_FP8_DROPOUT_MASK_BITS=0 for this wiring"
            )
        return torch.ops.aten.native_dropout_backward(grad, h.bool_mask(), h.scale)
    if not _same_tensor(grad, folded):
        raise RuntimeError(
            "input-dropout mask/scale fold: the dropout output's gradient is "
            "not the folded dx alone (another consumer contributed); set "
            "DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD=0"
        )
    _log_once(
        "folded",
        "armed OK: input-dropout mask/scale fold ACTIVE -- dropout backward "
        "applied in the layer-0 LayerNorm-backward epilogue "
        "(DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD)",
    )
    return grad


class _InputDropoutHandoffFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(ctx, x: torch.Tensor, p: float) -> torch.Tensor:  # pyre-ignore[2]
        out, mask = torch.ops.aten.native_dropout(x, p, True)
        handoff = DropoutHandoff(mask, 1.0 / (1.0 - p))
        ctx.handoff = handoff
        setattr(out, DROPOUT_HANDOFF_ATTR, handoff)
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, grad: torch.Tensor  # pyre-ignore[2]
    ) -> Tuple[torch.Tensor, None]:
        return dropout_handoff_backward(ctx.handoff, grad), None


def input_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """``F.dropout(x, p, training)``, carrying a mask/scale handoff when armed.

    Arms only for a CUDA training forward with 0 < p < 1 that records a graph;
    every other case is literally ``F.dropout``.
    """
    if (
        _FOLD_ENABLED
        and not torch.jit.is_scripting()
        and not is_fx_tracing()
        and training
        and 0.0 < p < 1.0
        and x.is_cuda
        and x.numel() > 0
        and x.requires_grad
        and torch.is_grad_enabled()
    ):
        return _InputDropoutHandoffFunction.apply(x, p)
    return torch.nn.functional.dropout(x, p=p, training=training)
