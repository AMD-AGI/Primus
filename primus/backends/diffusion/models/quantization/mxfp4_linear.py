###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""MXFP4 linear layers for the FLUX diffusion backend.

Replaces the TorchAO FP8 ``Float8Linear`` GEMMs with native MXFP4 (E2M1 data,
E8M0 per-32-element scales) on gfx950, with Fprop, Dgrad and Wgrad castable
independently so the three passes can be enabled one at a time.

Two deliberate departures from the Megatron MXFP4 spec provider in
``primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py``:

* That module quantizes through ``quantize_mxfp4_dual`` and passes a
  ``preshuffled`` kwarg to ``gemm_fp4_impl``. This module calls the single-axis
  quantizer and leaves preshuffle off, but still has to hand the quantizer a
  ``padding_align_size``: Primus-Turbo asserts it equals 128 for MXFP4.

* It stabilises with ``use_rht``, which in Primus-Turbo is a *randomised*
  Hadamard -- ``get_rht_matrix`` multiplies a fixed 32x32 Hadamard by a random
  sign vector. Randomised rotations are reported to leave Wgrad-quantized
  training non-convergent, while a *deterministic* rotation restores it, so the
  rotation here is applied in PyTorch ahead of the quantizer and ``use_rht``
  is left off.

Weights stay BF16 ``nn.Parameter``, so FSDP2 sharding and the optimizer are
untouched; quantization happens at each GEMM's use site.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from primus.backends.diffusion.utils.log import logger

FP4_DTYPE = torch.float4_e2m1fn_x2
MX_BLOCK_SIZE = 32

# Resolved on first module construction rather than at import: pulling in
# primus_turbo triggers an AITER JIT build, which FP8-only runs should not pay.
#
# These are populated eagerly by MXFP4Linear.__init__ and only *read* from the
# GEMM path. Dynamo rejects mutating a module-level container from inside a
# compiled region ("Mutating a variable not in the current scope"), so the
# populate step must never be reachable from a traced call.
_GEMM_FP4 = None
_GEMM_FP4_EAGER = None
_GRANULARITY = None
_BACKEND = None


def _resolve_fp4_dispatch() -> str:
    """Import-time Linear GEMM dispatch. ``torch.compile`` must not trace getenv.

    ``FLUX_FP4_DISPATCH`` selects the three measured CC12M recipes:

    * ``fusion`` — fused H16 pack + stock ``gemm_fp4_impl`` (66.06 samp/GPU/s)
    * ``host_dispatch`` — fused pack + eager ``gemm_fp4_impl`` (65.98)
    * ``mxfp4_mm`` — one ``primus_flux::mxfp4_mm`` custom op (65.63)

    Legacy flags still work if ``FLUX_FP4_DISPATCH`` is unset:
    ``FLUX_FP4_MXFP4_MM``, ``FLUX_FP4_HOST_DISPATCH``, ``FLUX_FP4_FUSED_H16_QUANT``.
    Empty means unfused Python H16 + C++ ``quantize_mxfp4`` (historical ~50.6).
    """
    explicit = os.getenv("FLUX_FP4_DISPATCH", "").strip().lower()
    if explicit in ("fusion", "host_dispatch", "mxfp4_mm"):
        return explicit
    if explicit:
        raise ValueError(
            f"FLUX_FP4_DISPATCH={explicit!r} is invalid; expected fusion, "
            "host_dispatch, mxfp4_mm, or empty"
        )
    if os.getenv("FLUX_FP4_MXFP4_MM", "0") == "1":
        return "mxfp4_mm"
    if os.getenv("FLUX_FP4_HOST_DISPATCH", "0") == "1":
        return "host_dispatch"
    if os.getenv("FLUX_FP4_FUSED_H16_QUANT", "0") == "1":
        return "fusion"
    return ""


_FP4_DISPATCH = _resolve_fp4_dispatch()
_FUSED_H16_QUANT = _FP4_DISPATCH in ("fusion", "host_dispatch", "mxfp4_mm")
_USE_MXFP4_MM = _FP4_DISPATCH == "mxfp4_mm"


def _init_turbo() -> None:
    global _GEMM_FP4, _GEMM_FP4_EAGER, _GRANULARITY, _BACKEND
    if _GEMM_FP4 is not None:
        return
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.core.low_precision import (
        ScalingGranularity,
        check_mxfp4_support,
    )
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl

    supported, reason = check_mxfp4_support()
    if not supported:
        raise RuntimeError(f"MXFP4 unsupported on this device: {reason}")

    # FlyDSL by default: same policy as FLUX FP8 (full_flydsl). Turbo's MXFP4
    # FlyDSL kernel tiles 256x256x256; every FLUX block GEMM dim is a multiple
    # of 256 so the backend can_handle all three passes. AITER / HIPBLASLT
    # remain selectable via FLUX_FP4_GEMM_BACKEND.
    name = os.getenv("FLUX_FP4_GEMM_BACKEND", "flydsl").strip().upper()
    try:
        backend = BackendType[name]
    except KeyError as exc:
        valid = ", ".join(b.name.lower() for b in BackendType)
        raise ValueError(
            f"FLUX_FP4_GEMM_BACKEND={name!r} is not a Primus-Turbo backend; expected one of: {valid}"
        ) from exc

    # PRIMUS_TURBO_GEMM_BACKEND outranks the per-call default_backend we pass,
    # and pinning FP4 to AITER through it also arms the preshuffle fast path.
    # That path expects pre-shuffled operands; ours are not shuffled (the
    # shuffle_* flags into quantize_mxfp4 are all False), and the mismatch is
    # silent -- measured relative GEMM error goes from 0.158 to 1.55, i.e. the
    # result is noise, with no error raised. Refuse to start rather than train
    # on garbage.
    env_pin = os.getenv("PRIMUS_TURBO_GEMM_BACKEND")
    if env_pin:
        raise RuntimeError(
            f"PRIMUS_TURBO_GEMM_BACKEND={env_pin!r} is set, which overrides the per-call "
            "backend and arms AITER's preshuffle fast path. This MXFP4 path does not "
            "preshuffle its operands, and the mismatch silently corrupts every GEMM "
            "(relative error 1.55 vs 0.158). Unset it and select the backend with "
            "FLUX_FP4_GEMM_BACKEND instead."
        )

    _GRANULARITY = ScalingGranularity.MX_BLOCKWISE.value
    _BACKEND = backend.value
    _GEMM_FP4 = gemm_fp4_impl
    try:
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import _gemm_fp4_impl_eager
    except ImportError:
        _gemm_fp4_impl_eager = gemm_fp4_impl
    _GEMM_FP4_EAGER = _gemm_fp4_impl_eager
    extra = f", dispatch={_FP4_DISPATCH}" if _FP4_DISPATCH else ""
    logger.info(f"MXFP4 GEMM backend: {backend.name} (preshuffle off){extra}")


# ---------------------------------------------------------------------------
# Deterministic Hadamard rotation
# ---------------------------------------------------------------------------

_HADAMARD_CACHE: dict[tuple, torch.Tensor] = {}


def hadamard_matrix(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sylvester Hadamard of order ``n``, scaled by 1/sqrt(n) to be orthogonal."""
    key = (n, dtype, str(device))
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        if n <= 0 or n & (n - 1):
            raise ValueError(f"Hadamard order must be a power of two, got {n}")
        h = torch.ones(1, 1, dtype=torch.float64, device=device)
        while h.shape[0] < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        cached = (h / math.sqrt(n)).to(dtype)
        _HADAMARD_CACHE[key] = cached
    return cached


def rotate_last_dim(x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
    """Apply the block-diagonal Hadamard ``h`` along ``x``'s last dim.

    Both operands of a GEMM are rotated along the dim they contract over, so
    the rotations cancel -- (XH)(WH)^T = X H H^T W^T = X W^T -- leaving the
    result unchanged while spreading per-channel outliers across each
    32-element micro-scaling block.

    ``h`` is passed in rather than looked up so nothing in the traced path
    touches a module-level cache.
    """
    if h is None:
        return x
    n = h.shape[0]
    *lead, d = x.shape
    if d % n:
        raise ValueError(f"last dim {d} is not divisible by Hadamard order {n}")
    return torch.matmul(x.reshape(*lead, d // n, n), h.to(x.dtype)).reshape(*lead, d)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAST_CHOICES = ("mxfp4", "bf16")


@dataclass(frozen=True)
class MXFP4LinearConfig:
    """Per-pass MXFP4 configuration.

    ``fprop``/``dgrad``/``wgrad`` each select ``"mxfp4"`` or ``"bf16"``, which
    is what makes the stage-wise ladder (Fprop, then +Dgrad, then +Wgrad)
    expressible as three configs over one implementation.
    """

    fprop: str = "mxfp4"
    dgrad: str = "mxfp4"
    wgrad: str = "mxfp4"
    # Deterministic Hadamard order; 0 disables. 16 is cheaper than 32 and
    # reported to stabilise equally well.
    hadamard: int = 16
    # 2D 32x32 block scaling for weights is transpose-invariant, but measured
    # ~8% *worse* than 1D rowwise on FLUX's shapes, so it is off by default.
    weight_2d_block: bool = False
    # Both off by default: stochastic rounding and Primus-Turbo's randomised
    # Hadamard are the two interventions reported not to rescue Wgrad.
    stochastic_rounding: bool = False
    randomized_hadamard: bool = False
    # Stash the Wgrad operand as MXFP4 (~0.53 B/element) rather than BF16.
    # The FP8 baseline saves ~18GB/GPU of activations here; BF16 would be ~36GB
    # and MXFP4 is ~10GB, so this keeps peak memory under the baseline.
    save_quantized: bool = True

    def __post_init__(self):
        for field in ("fprop", "dgrad", "wgrad"):
            value = getattr(self, field)
            if value not in CAST_CHOICES:
                raise ValueError(f"{field}={value!r} not in {CAST_CHOICES}")
        if self.hadamard and (self.hadamard & (self.hadamard - 1)):
            raise ValueError(f"hadamard={self.hadamard} is not a power of two")
        if self.hadamard and self.hadamard < MX_BLOCK_SIZE // 2:
            # Rotating over fewer than half a block leaves most of the block
            # unmixed, which defeats the point.
            logger.warning(f"hadamard={self.hadamard} is small vs MX block {MX_BLOCK_SIZE}")

    @property
    def any_mxfp4(self) -> bool:
        return "mxfp4" in (self.fprop, self.dgrad, self.wgrad)

    def describe(self) -> str:
        passes = ",".join(f"{name}={getattr(self, name)}" for name in ("fprop", "dgrad", "wgrad"))
        extras = [f"hadamard=H{self.hadamard}" if self.hadamard else "hadamard=off"]
        if self.weight_2d_block:
            extras.append("weight_2d")
        if self.stochastic_rounding:
            extras.append("sr")
        if self.randomized_hadamard:
            extras.append("rht")
        return f"{passes} {' '.join(extras)}"


def config_from_env() -> MXFP4LinearConfig | None:
    """Build a config from FLUX_FP4_* env vars, or None if FP4 is not requested.

    ``FLUX_FP4_PASSES`` is the ladder control: a comma list drawn from
    ``fprop,dgrad,wgrad``; anything omitted stays BF16.
    """
    passes = os.getenv("FLUX_FP4_PASSES", "").strip().lower()
    if not passes or passes in {"0", "off", "none"}:
        return None

    selected = {p.strip() for p in passes.split(",") if p.strip()}
    if passes in {"1", "all", "full"}:
        selected = {"fprop", "dgrad", "wgrad"}
    unknown = selected - {"fprop", "dgrad", "wgrad"}
    if unknown:
        raise ValueError(f"FLUX_FP4_PASSES has unknown entries: {sorted(unknown)}")

    def flag(name: str, default: str) -> bool:
        return os.getenv(name, default) == "1"

    return MXFP4LinearConfig(
        fprop="mxfp4" if "fprop" in selected else "bf16",
        dgrad="mxfp4" if "dgrad" in selected else "bf16",
        wgrad="mxfp4" if "wgrad" in selected else "bf16",
        hadamard=int(os.getenv("FLUX_FP4_HADAMARD", "16")),
        weight_2d_block=flag("FLUX_FP4_WEIGHT_2D", "0"),
        stochastic_rounding=flag("FLUX_FP4_SR", "0"),
        randomized_hadamard=flag("FLUX_FP4_RHT", "0"),
        save_quantized=flag("FLUX_FP4_SAVE_QUANTIZED", "1"),
    )


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


@torch.library.custom_op("primus_flux::quantize_mxfp4_rowwise", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_rowwise_op(
    x: torch.Tensor, use_2d_block: bool, use_sr: bool, use_rht: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4-quantize along ``x``'s last dim.

    Single-axis rather than ``quantize_mxfp4_dual``: each pass needs its operands
    rotated along *its own* contraction dim and one dual call can only carry one
    rotation, so the colwise half of a dual pack was always discarded -- half a
    packer paid for on every call. Row-only measured 14% cheaper overall and up
    to 35% on the smaller activations, and is bit-identical to the dual packer's
    row blob (same bytes, same dtypes, same resulting GEMM error).

    ``axis=1`` is the row direction; the kernel accepts only 0 or 1.

    Wrapped as a custom op because the raw C++ op has no meta kernel, which
    would force a graph break and cost the torch.compile speedup.
    """
    data, scale = torch.ops.primus_turbo_cpp_extension.quantize_mxfp4(
        x.contiguous(),
        FP4_DTYPE,
        1,  # axis: quantize along the last dim
        128,  # padding_align_size; the kernel asserts exactly 128 for MXFP4
        use_2d_block,
        use_sr,
        use_rht,
        False,  # shuffle_scale -- preshuffle is not used, see _init_turbo
        False,  # shuffle_out
    )
    return data, scale


@_quantize_mxfp4_rowwise_op.register_fake
def _(x, use_2d_block, use_sr, use_rht):
    m, n = x.shape
    # The kernel pads the quantized dim up to 128; every FLUX GEMM dim is
    # already a multiple of 128, so the unpadded shapes below are exact.
    data = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    return data.view(FP4_DTYPE), scale.view(torch.float8_e8m0fnu)


def _quantize_rowwise(
    x: torch.Tensor, use_2d: bool, use_sr: bool, use_rht: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % 128:
        raise ValueError(f"MXFP4 quantized dim must be a multiple of 128, got {x.shape[-1]}")
    return _quantize_mxfp4_rowwise_op(x, use_2d, use_sr, use_rht)


if _FP4_DISPATCH:
    logger.info(f"MXFP4 Linear dispatch: {_FP4_DISPATCH}")
elif os.getenv("FLUX_FP4_PASSES", "off") not in ("", "off"):
    logger.info("MXFP4 operand pack: Python H16 + C++ quantize_mxfp4")


def _fused_h16_quant_enabled(
    hadamard: torch.Tensor | None, use_2d: bool, use_sr: bool, use_rht: bool
) -> bool:
    """True when Linear should pack via FlyDSL ``flydsl_quant_mxfp4_h16``.

    Default off so an unflagged MXFP4 run stays Python H16 + C++ row-only.
    2D-block, SR, and randomized RHT have no fused kernel; those keep the
    original two-launch path. Flag is resolved at import so torch.compile
    does not trace ``os.getenv`` / logger (``sys._getframe``).
    """
    return bool(_FUSED_H16_QUANT and hadamard is not None and not use_2d and not use_sr and not use_rht)


@torch.library.custom_op("primus_flux::quantize_mxfp4_h16", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_h16_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """H16 + row-only MXFP4 in one FlyDSL launch. Custom-op so torch.compile
    does not trace into FlyDSL JIT.
    """
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import flydsl_quant_mxfp4_h16

    return flydsl_quant_mxfp4_h16(x.contiguous(), FP4_DTYPE)


@_quantize_mxfp4_h16_op.register_fake
def _(x):
    m, n = x.shape
    data = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    return data.view(FP4_DTYPE), scale.view(torch.float8_e8m0fnu)


def _quantize_operand(
    x: torch.Tensor,
    hadamard: torch.Tensor | None,
    use_2d: bool,
    use_sr: bool,
    use_rht: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % 128:
        raise ValueError(f"MXFP4 quantized dim must be a multiple of 128, got {x.shape[-1]}")
    if _fused_h16_quant_enabled(hadamard, use_2d, use_sr, use_rht):
        return _quantize_mxfp4_h16_op(x)
    return _quantize_mxfp4_rowwise_op(rotate_last_dim(x, hadamard), use_2d, use_sr, use_rht)


def _mxfp4_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """a[M,K] @ b[N,K]^T -> [M,N]; both operands quantized along K."""
    return _GEMM_FP4(
        a,
        a_scale,
        False,
        b,
        b_scale,
        True,
        out_dtype,
        False,
        granularity=_GRANULARITY,
        default_backend=_BACKEND,
    )


@torch.library.custom_op("primus_flux::mxfp4_mm", mutates_args=(), device_types="cuda")
def _mxfp4_mm_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused pack(A)+pack(B)+GEMM. Opaque to torch.compile.

    Uses FlyDSL ``flydsl_quant_mxfp4_h16`` and Turbo ``_gemm_fp4_impl_eager``.
    HIP-graph replay is not wired: copies wiped the micro win on large shapes.
    """
    _init_turbo()
    a_q, a_s = _quantize_mxfp4_h16_op(a)
    b_q, b_s = _quantize_mxfp4_h16_op(b)
    return _GEMM_FP4_EAGER(
        a_q,
        a_s,
        False,
        b_q,
        b_s,
        True,
        a.dtype,
        False,
        _GRANULARITY,
        _BACKEND,
        False,
    )


@_mxfp4_mm_op.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)


def _quantize_and_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    cfg: MXFP4LinearConfig,
    hadamard: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    a_sr: bool = False,
    b_sr: bool = False,
    b_2d: bool = False,
) -> torch.Tensor:
    """Rotate, quantize and multiply a[M,K] @ b[N,K]^T along the shared K."""
    if (
        _USE_MXFP4_MM
        and not a_sr
        and not b_sr
        and not b_2d
        and not cfg.randomized_hadamard
    ):
        return torch.ops.primus_flux.mxfp4_mm(a, b)
    a_q, a_s = _quantize_operand(a, hadamard, False, a_sr, cfg.randomized_hadamard)
    b_q, b_s = _quantize_operand(b, hadamard, b_2d, b_sr, cfg.randomized_hadamard)
    return _mxfp4_mm(a_q, a_s, b_q, b_s, out_dtype)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


class _MXFP4LinearFunction(torch.autograd.Function):
    """y = x @ w^T (+ bias) with each pass independently castable to MXFP4.

    Every pass is expressed as ``A[M,K] @ B[N,K]^T`` so one quantize-and-multiply
    helper serves all three. The transposes that normalise them to that form are
    also what lets each pass carry its own rotation: Fprop contracts over the
    input features, Dgrad over the output features and Wgrad over the tokens, so
    a rotation that cancels for one does nothing for the others.

    Weights are re-quantized in backward rather than stashed -- FSDP2 runs with
    ``reshard_after_forward=false``, so the BF16 parameter is still resident and
    holding a reference to it is free, whereas a saved FP4 copy would not be.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, hadamard, cfg):
        x_2d = x.reshape(-1, x.shape[-1])
        out_dtype = x.dtype

        if cfg.fprop == "mxfp4":
            out = _quantize_and_mm(x_2d, weight, cfg, hadamard, out_dtype, b_2d=cfg.weight_2d_block)
        else:
            out = torch.matmul(x_2d, weight.t())

        # Wgrad contracts over tokens, so its operand must be rotated along the
        # token dim -- a rotation the forward pass never forms. Build it here so
        # only the 4-bit copy has to live until backward.
        if cfg.wgrad == "mxfp4" and cfg.save_quantized:
            x_q, x_s = _quantize_operand(
                x_2d.t().contiguous(),
                hadamard,
                False,
                cfg.stochastic_rounding,
                cfg.randomized_hadamard,
            )
            ctx.save_for_backward(x_q.view(torch.uint8), x_s, weight, hadamard)
            ctx.x_is_quantized = True
        else:
            ctx.save_for_backward(x_2d, weight, hadamard)
            ctx.x_is_quantized = False

        ctx.cfg = cfg
        ctx.x_shape = tuple(x.shape)
        ctx.out_dtype = out_dtype
        ctx.has_bias = bias is not None

        if bias is not None:
            out = out + bias
        return out.reshape(*ctx.x_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        cfg: MXFP4LinearConfig = ctx.cfg
        out_dtype = ctx.out_dtype

        if ctx.x_is_quantized:
            x_q, x_s, weight, hadamard = ctx.saved_tensors
            x_q = x_q.view(FP4_DTYPE)
            x_2d = None
        else:
            x_2d, weight, hadamard = ctx.saved_tensors
            x_q = x_s = None

        grad_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()

        # Dgrad: dX = G @ W, contracting over the output features N.
        # A = G[M,N]; B = W^T[K,N]; A @ B^T -> [M,K].
        if cfg.dgrad == "mxfp4":
            grad_input = _quantize_and_mm(
                grad_2d,
                weight.t().contiguous(),
                cfg,
                hadamard,
                out_dtype,
                a_sr=cfg.stochastic_rounding,
                b_2d=cfg.weight_2d_block,
            )
        else:
            grad_input = torch.matmul(grad_2d, weight)

        # Wgrad: dW = G^T @ X, contracting over the tokens M.
        # A = G^T[N,M]; B = X^T[K,M]; A @ B^T -> [N,K].
        if cfg.wgrad == "mxfp4":
            g_q, g_s = _quantize_operand(
                grad_2d.t().contiguous(),
                hadamard,
                False,
                cfg.stochastic_rounding,
                cfg.randomized_hadamard,
            )
            if x_q is None:
                x_q, x_s = _quantize_operand(
                    x_2d.t().contiguous(),
                    hadamard,
                    False,
                    cfg.stochastic_rounding,
                    cfg.randomized_hadamard,
                )
            grad_weight = _mxfp4_mm(g_q, g_s, x_q, x_s, out_dtype)
        else:
            grad_weight = torch.matmul(grad_2d.t(), x_2d)

        grad_bias = grad_2d.sum(0) if ctx.has_bias else None
        return grad_input.reshape(ctx.x_shape), grad_weight, grad_bias, None, None


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class MXFP4Linear(nn.Linear):
    """``nn.Linear`` whose GEMMs run in MXFP4. Weights remain BF16."""

    def __init__(self, in_features, out_features, bias=True, config=None, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.config = config or MXFP4LinearConfig()
        for name, dim in (("in_features", in_features), ("out_features", out_features)):
            if dim % MX_BLOCK_SIZE:
                raise ValueError(f"MXFP4Linear needs {name} divisible by {MX_BLOCK_SIZE}, got {dim}")
            if self.config.hadamard and dim % self.config.hadamard:
                raise ValueError(
                    f"MXFP4Linear needs {name} divisible by the Hadamard order "
                    f"{self.config.hadamard}, got {dim}"
                )
        if self.config.any_mxfp4:
            _init_turbo()
        self.register_buffer("hadamard", None, persistent=False)
        self._build_hadamard(self.weight.device)

    def _build_hadamard(self, device: torch.device) -> None:
        """Materialise the rotation eagerly, as a non-persistent buffer.

        Building it on demand inside the GEMM would mutate a module-level cache
        from inside the compiled region, which Dynamo rejects outright
        ("Mutating a variable not in the current scope"). Non-persistent keeps
        it out of the checkpoint.
        """
        if not (self.config.any_mxfp4 and self.config.hadamard):
            return
        if torch.device(device).type == "meta":
            return
        self.hadamard = hadamard_matrix(self.config.hadamard, torch.float32, device)

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: MXFP4LinearConfig) -> "MXFP4Linear":
        # Built on meta so nn.Linear does not allocate a second full weight
        # tensor per module only for it to be discarded.
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            config=config,
            device="meta",
        )
        module.weight = linear.weight
        module.bias = linear.bias
        module._build_hadamard(linear.weight.device)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.any_mxfp4:
            return super().forward(x)
        return _MXFP4LinearFunction.apply(x, self.weight, self.bias, self.hadamard, self.config)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, mxfp4=({self.config.describe()})"


def convert_to_mxfp4_training(
    model: nn.Module,
    module_filter_fn: Callable[[nn.Module, str], bool],
    config: MXFP4LinearConfig,
    config_override_fn: Callable[[str], MXFP4LinearConfig | None] | None = None,
) -> list[str]:
    """Swap every ``nn.Linear`` accepted by ``module_filter_fn`` for MXFP4.

    Returns the FQNs that were converted. ``config_override_fn`` may return a
    per-module config, which is how layer-wise mixed precision (e.g. keeping
    QKV Wgrad out of FP4) is expressed.
    """
    converted: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            fqn = f"{prefix}{name}"
            if type(child) is nn.Linear and module_filter_fn(child, fqn):
                module_config = (config_override_fn(fqn) if config_override_fn else None) or config
                setattr(parent, name, MXFP4Linear.from_linear(child, module_config))
                converted.append(fqn)
            else:
                visit(child, f"{fqn}.")

    visit(model, "")
    return converted
