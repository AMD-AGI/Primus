###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
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

import inspect
import math
import os

_BF16_EVAL = os.getenv("FLUX_FP4_BF16_EVAL", "0") == "1"
import re
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from primus.backends.diffusion.utils.log import logger

FP4_DTYPE = torch.float4_e2m1fn_x2
MX_BLOCK_SIZE = 32

# Resolved on first module construction rather than at import: pulling in primus_turbo triggers an
# AITER JIT build, which FP8-only runs should not pay.
# These are populated eagerly by MXFP4Linear.__init__ and only *read* from the GEMM path. Dynamo
# rejects mutating a module-level container from inside a compiled region ("Mutating a variable
# not in the current scope"), so the populate step must never be reachable from a traced call.
_GEMM_FP4 = None
_GEMM_FP4_EAGER = None
_GEMM_FP4_BIAS = False  # does the installed gemm_fp4_impl take a fused bias? see _init_turbo
_GEMM_FP4_PACKED_KW = False  # P3 overlay's a_scale_packed=/b_scale_packed=
_GEMM_FP4_PRESHUFFLED = False  # upstream's preshuffled=True, same saving, different interface
_TURBO_LAYOUT_OK = None  # bound in _init_turbo; never imported from the traced backward
_GRANULARITY = None
_BACKEND = None


def _op_takes(op, name: str) -> bool:
    """Does this torch custom op declare an argument called ``name``?

    Read rather than probed by calling: the mismatch this guards against only surfaces while
    dynamo is tracing, where a schema error is not a catchable TypeError but a dead rank.
    ``_schema`` is a parsed FunctionSchema on an OpOverload but a plain string on the
    custom-op wrapper, so both shapes are handled before falling back to the signature.
    """
    schema = getattr(getattr(op, "default", op), "_schema", None)
    if hasattr(schema, "arguments"):
        return any(a.name == name for a in schema.arguments)
    if isinstance(schema, str):  # "(Tensor a, ..., bool preshuffled=False) -> Tensor"
        args = schema.split("->")[0].strip().strip("()")
        return name in {p.split("=")[0].split()[-1] for p in args.split(",") if p.split()}
    try:
        return name in inspect.signature(op).parameters
    except (TypeError, ValueError):
        return False


def _resolve_fp4_dispatch() -> str:
    """
Import-time Linear GEMM dispatch. ``torch.compile`` must not trace getenv.

* ``fusion`` — fused H16 pack + stock ``gemm_fp4_impl`` (66.06 samp/GPU/s) * ``host_dispatch`` — fused pack + eager ``gemm_fp4_impl`` (65.98) * ``mxfp4_mm`` — one ``primus_flux::mxfp4_mm`` custom op (65.63)

Legacy flags still work if ``FLUX_FP4_DISPATCH`` is unset: ``FLUX_FP4_MXFP4_MM``, ``FLUX_FP4_HOST_DISPATCH``, ``FLUX_FP4_FUSED_H16_QUANT``. Unset or empty selects the launch recipe default, fusion.
"""
    explicit = (os.getenv("FLUX_FP4_DISPATCH") or "fusion").strip().lower()
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


def _resolve_scale_rounding_mode() -> int:
    """
E8M0 rounding bias applied when a block's scale exponent is derived.

The FlyDSL packs add a constant to the fp32 bits before taking the exponent, so the mode picks where inside a power-of-two bin the scale rounds up: 0 is a quarter ULP, 1 is a half (round-to-nearest on the exponent), 2 is three eighths.

We ran mode 0 for every MXFP4 run to date, not by choice but because nothing ever passed this argument and the kernels default it. GPT-OSS-20B's converged MXFP4 recipe uses mode 2 (they call it UOS), alongside RHT, gradient SR and weight de-oscillation, and that is the only MXFP4 recipe in the building that converges on its own schedule.

Resolved at import, like the dispatch flag above, so torch.compile never traces os.getenv.
"""
    # `or "0"`, not a getenv default: the launcher forwards every name the config exports and
    # materialises the value from the live environment, so an unset knob arrives as "" and int("")
    # raises before anything useful has been logged.
    mode = int(os.getenv("FLUX_FP4_SCALE_ROUNDING") or "0")
    if mode not in (0, 1, 2):
        raise ValueError(f"FLUX_FP4_SCALE_ROUNDING must be 0, 1 or 2, got {mode}")
    return mode


_SCALE_ROUNDING_MODE = _resolve_scale_rounding_mode()
# Backward hands both FP4 GEMMs their scales already in the GEMM's packed layout, built by the
# grad dual-bias pack (`_quantize_mxfp4_h16_dual_bias_packed_op`), so neither GEMM launches its
# scale preshuffle.
_PACKED_SCALES = os.environ.get("FLUX_MXFP4_PACKED_SCALES", "1") == "1"


def _resolve_fp8_fprop_backend() -> str:
    """Import-time FP8 fprop GEMM backend. ``torch.compile`` must not trace getenv.

    Honors ``FLUX_FP4_FP8_GEMM_BACKEND``, else the stock Flux
    ``FLUX_FP8_GEMM_BACKEND`` (``selective_flydsl`` / ``full_flydsl`` -> FlyDSL).
    """
    explicit = os.getenv("FLUX_FP4_FP8_GEMM_BACKEND", "").strip().lower()
    if not explicit:
        explicit = os.getenv("FLUX_FP8_GEMM_BACKEND", "").strip().lower()
    if explicit in ("selective_flydsl", "full_flydsl", "flydsl"):
        return "flydsl"
    return "hipblaslt"


_FP8_FPROP_BACKEND = _resolve_fp8_fprop_backend()

# Collapse the activation's three HBM passes -- FP8 amax, FP8 row cast, MXFP4 col pack -- into the
# single-read `flydsl_quant_fp8_mxfp4_dual` kernel. Read at import for the same reason as above:
# torch.compile must not trace a getenv.
# Default OFF because it is the one thing here that changes numerics: the FP8 scale becomes the
# PREVIOUS step's, since a tensorwise scale otherwise has to be known before the first output byte
# can be written, which is exactly what forces the amax into its own pass.
_FP8_DUAL_PACK = os.getenv("FLUX_FP8_DUAL_PACK", "0") == "1"

# Fold the dual pack's delayed-scale arithmetic (reciprocal, amax reduce, clamp, `/ 448.0`) into
# `flydsl_quant_fp8_mxfp4_dual_delayed`: ~6 eager launches per Linear per step become two FlyDSL
# ones. Byte-identical to the eager path on all four outputs, so numerics do not move.
_FP8_DUAL_DELAYED = os.getenv("FLUX_FP8_DUAL_DELAYED", "0") == "1"

# FP8 all-gather: quantize each weight shard with ONE `flydsl_quant_fp8_mxfp4_dual` read instead
# of `quantize_fp8_tensorwise` (a scale kernel + the row cast) followed by a separate col-pack
# read. Byte-identical on all three payloads and scale_inv. Shards the kernel's R % 128, C % 256
# tiling does not cover keep the old path.
_FP8_AG_FUSED_QUANT = os.getenv("FLUX_FP8_AG_FUSED_QUANT", "0") == "1"
_FP8_AG_FUSED_QUANT_LOGGED = False


def _init_turbo() -> None:
    global _GEMM_FP4, _GEMM_FP4_EAGER, _GEMM_FP4_BIAS, _GRANULARITY, _BACKEND
    global _GEMM_FP4_PACKED_KW, _GEMM_FP4_PRESHUFFLED, _TURBO_LAYOUT_OK
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

    # FlyDSL by default: same policy as FLUX FP8 (full_flydsl). Turbo's MXFP4 FlyDSL kernel tiles
    # 256x256x256; every FLUX block GEMM dim is a multiple of 256 so the backend can_handle all three
    # passes. AITER / HIPBLASLT remain selectable via FLUX_FP4_GEMM_BACKEND.
    name = os.getenv("FLUX_FP4_GEMM_BACKEND", "flydsl").strip().upper()
    try:
        backend = BackendType[name]
    except KeyError as exc:
        valid = ", ".join(b.name.lower() for b in BackendType)
        raise ValueError(
            f"FLUX_FP4_GEMM_BACKEND={name!r} is not a Primus-Turbo backend; expected one of: {valid}"
        ) from exc

    # PRIMUS_TURBO_GEMM_BACKEND outranks the per-call default_backend we pass, and pinning FP4 to
    # AITER through it also arms the preshuffle fast path. That path expects pre-shuffled operands;
    # the result is noise, with no error raised. Refuse to start rather than train on garbage.
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
    # The fused `bias=` is the P3 overlay's, so it is absent whenever FLUX_MXFP4_P3_GEMM=0 or the
    # image's Turbo is newer than that overlay. Decide once here and let _mxfp4_mm fold the bias in on
    # the host when the operator cannot take it.
    _GEMM_FP4_BIAS = _op_takes(gemm_fp4_impl, "bias")
    # Packed scales reach the GEMM by one of two names. The P3 overlay takes them alongside the
    # canonical ones as `a_scale_packed=`; upstream's own equivalent, which landed after that
    # overlay was taken, takes them INSTEAD of the canonical ones with `preshuffled=True`. Same
    # saving either way -- the GEMM stops repacking scales on every launch.
    _GEMM_FP4_PACKED_KW = _op_takes(gemm_fp4_impl, "a_scale_packed")
    # Upstream's interface also needs its layout predicate, and both the predicate and the module it
    # lives in have to be resolved HERE. `warm_packed_scale_api` also primes upstream's own lazy CU
    # count, which was the specific mutation the second of those tripped on.
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        _turbo_packed_layout_agrees,
        warm_packed_scale_api,
    )

    # Reported by the operator AND confirmed by the warm-up, so that a build whose GEMM takes
    # `preshuffled=` but whose layout we cannot check never reaches the packed path at all.
    _GEMM_FP4_PRESHUFFLED = _op_takes(gemm_fp4_impl, "preshuffled") and warm_packed_scale_api() == "turbo"
    _TURBO_LAYOUT_OK = _turbo_packed_layout_agrees
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

CAST_CHOICES = ("mxfp4", "bf16", "fp8", "mxfp6", "a6w4")

# The six independently quantized tensors: two operands per GEMM, named by the
# role the tensor plays rather than by its orientation, because the same tensor is
# packed for two different passes and the two packs want different rounding.
#   Fprop  out  = act    @ weight^T
#   Dgrad  dX   = grad   @ weight
#   Wgrad  dW   = grad^T @ act
SR_OPERANDS = frozenset(
    {
        ("act", "fprop"),
        ("weight", "fprop"),
        ("grad", "dgrad"),
        ("weight", "dgrad"),
        ("grad", "wgrad"),
        ("act", "wgrad"),
    }
)

# Named points in that space, so an experiment names a published recipe instead of
# spelling out a tuple set that is easy to get subtly wrong.
SR_OPERAND_PRESETS = {
    # What Transformer Engine ships for NVFP4: stochastic rounding on the incoming
    # gradient only, round-to-nearest everywhere else.
    "grad": frozenset({("grad", "dgrad"), ("grad", "wgrad")}),
    # "FP4 All the Way" (arXiv 2505.19115) eq. 4-5, the full selective scheme from its six-way sweep:
    # the gradient in both backward GEMMs, plus Wgrad's activation, which is the one forward tensor
    # that helped.
    "paper": frozenset({("grad", "dgrad"), ("grad", "wgrad"), ("act", "wgrad")}),
}


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
    weight_2d_block: bool = False
    # Primus-Turbo's randomised Hadamard, reported not to rescue Wgrad.
    randomized_hadamard: bool = False
    # Stash the Wgrad operand as MXFP4 (~0.53 B/element) rather than BF16. The FP8 baseline saves
    # ~18GB/GPU of activations here; BF16 would be ~36GB and MXFP4 is ~10GB, so this keeps peak memory
    # under the baseline.
    save_quantized: bool = True
    # Passes that pack their operands with stochastic rounding, via the FlyDSL
    # packs' hardware SR convert. This replaces FLUX_FP4_SR, which never did
    # anything: the C++ quantize_mxfp4 takes use_sr and ignores it.
    # This is per-PASS, so it necessarily rounds BOTH operands of a pass the same
    # way, and that turns out to be the wrong granularity -- see sr_operands.
    sr_passes: frozenset = frozenset()
    # (tensor, pass) pairs that round stochastically, i.e. SR at the granularity the literature
    # actually prescribes it at.
    # Three GEMMs x two operands is six independently roundable tensors, and both published FP4
    # recipes ablate them one at a time and land on the same split: SR helps on the incoming gradient,
    # and HURTS on the forward tensors. NVFP4 (arXiv 2509.25149) states it directly -- "use stochastic
    # rounding for gradients and round-to-nearest-even for weights and activations", with "applying
    # stochastic rounding to the forward pass tensors is detrimental, as it amplifies quantization
    # error relative to nearest rounding" -- and ships exactly that in Transformer Engine, where SR is
    # enabled for gradients only. "FP4 All the Way" (arXiv 2505.19115) sweeps all six and keeps SR on
    # the gradient in both the Dgrad and Wgrad GEMMs plus the activation in Wgrad, with round-to-
    # nearest on both weight orientations and on Fprop's activation.
    # sr_passes cannot express any of that: dgrad,wgrad also rounds the weight stochastically, which
    # is one of the placements both papers call harmful.
    sr_operands: frozenset = frozenset()

    def __post_init__(self):
        for field in ("fprop", "dgrad", "wgrad"):
            value = getattr(self, field)
            if value not in CAST_CHOICES:
                raise ValueError(f"{field}={value!r} not in {CAST_CHOICES}")
        unknown = set(self.sr_passes) - {"fprop", "dgrad", "wgrad"}
        if unknown:
            raise ValueError(f"sr_passes has unknown passes: {sorted(unknown)}")
        unknown_ops = set(self.sr_operands) - SR_OPERANDS
        if unknown_ops:
            raise ValueError(f"sr_operands has unknown pairs: {sorted(unknown_ops)}")
        if self.hadamard and (self.hadamard & (self.hadamard - 1)):
            raise ValueError(f"hadamard={self.hadamard} is not a power of two")
        if self.hadamard and self.hadamard < MX_BLOCK_SIZE // 2:
            # Rotating over fewer than half a block leaves most of the block
            # unmixed, which defeats the point.
            logger.warning(f"hadamard={self.hadamard} is small vs MX block {MX_BLOCK_SIZE}")

    @property
    def any_mxfp4(self) -> bool:
        # mxfp6 counts, because it also routes through _MXFP4LinearFunction. Gating
        # on mxfp4 alone would make an mxfp6-fprop/bf16-backward config fall through
        # to nn.Linear and run the whole layer in BF16 while the log still disclosed
        # the requested recipe. The name is load-bearing: bf16_eval_guard.py matches
        # `if not self.config.any_mxfp4:` literally.
        return any(p in ("mxfp4", "mxfp6", "a6w4") for p in (self.fprop, self.dgrad, self.wgrad))

    def describe(self) -> str:
        passes = ",".join(f"{name}={getattr(self, name)}" for name in ("fprop", "dgrad", "wgrad"))
        extras = [f"hadamard=H{self.hadamard}" if self.hadamard else "hadamard=off"]
        if self.weight_2d_block:
            extras.append("weight_2d")
        if self.randomized_hadamard:
            extras.append("rht")
        if self.sr_passes:
            extras.append(f"sr_passes={'+'.join(sorted(self.sr_passes))}")
        if self.sr_operands:
            extras.append("sr_operands=" + "+".join(f"{t}@{p}" for t, p in sorted(self.sr_operands)))
        return f"{passes} {' '.join(extras)}"

    def sr(self, pass_name: str) -> bool:
        """Whether ``pass_name`` packs BOTH its operands with stochastic rounding."""
        return pass_name in self.sr_passes

    def sr_op(self, tensor: str, pass_name: str) -> bool:
        """
Whether ``tensor`` rounds stochastically when packed for ``pass_name``.

``tensor`` is one of act/weight/grad. Every pack site asks with its own role, so the six quantized tensors are controllable one at a time;
"""
        if (tensor, pass_name) not in SR_OPERANDS:
            raise ValueError(f"{tensor!r} is not an operand of {pass_name!r}")
        return pass_name in self.sr_passes or (tensor, pass_name) in self.sr_operands


def config_from_env() -> MXFP4LinearConfig | None:
    """
Build a config from FLUX_FP4_* env vars, or None if FP4 is not requested.

``FLUX_FP4_PASSES`` is the ladder control: a comma list drawn from ``fprop,dgrad,wgrad``. Omitted dgrad/wgrad stay BF16. Omitted fprop takes ``FLUX_FP4_FPROP_CAST``, which defaults to bf16 but should usually be mxfp6.

So mxfp6 wins on the axis that is stable.

fp8 used to sit ~1M samples better than that line predicted, which made it the anomaly and left mxfp6 looking like it had headroom to find;
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

    fprop_cast = os.getenv("FLUX_FP4_FPROP_CAST", "bf16").strip().lower()
    if fprop_cast not in CAST_CHOICES:
        raise ValueError(f"FLUX_FP4_FPROP_CAST={fprop_cast!r} not in {CAST_CHOICES}")

    def flag(name: str, default: str) -> bool:
        return os.getenv(name, default) == "1"

    # Same comma-list grammar as FLUX_FP4_PASSES, and intersected with it: asking
    # for SR on a pass that is not MXFP4 is a no-op, not an error, so that `all`
    # stays meaningful as the ladder changes underneath it.
    sr = os.getenv("FLUX_FP4_SR_PASSES", "").strip().lower()
    if sr in {"1", "all", "full"}:
        sr_set = {"fprop", "dgrad", "wgrad"}
    elif not sr or sr in {"0", "off", "none"}:
        sr_set = set()
    else:
        sr_set = {p.strip() for p in sr.split(",") if p.strip()}
        unknown = sr_set - {"fprop", "dgrad", "wgrad"}
        if unknown:
            raise ValueError(f"FLUX_FP4_SR_PASSES has unknown entries: {sorted(unknown)}")
    sr_set &= selected

    # Operand-level SR. Either a preset name (grad, paper) or a comma list of tensor@pass tokens drawn
    # from SR_OPERANDS. Intersected with the selected passes for the same reason sr_passes is: a pass
    # that is not MXFP4 has no pack to round, so naming it is a no-op rather than an error.
    sr_ops_env = os.getenv("FLUX_FP4_SR_OPERANDS", "").strip().lower()
    if not sr_ops_env or sr_ops_env in {"0", "off", "none"}:
        sr_ops = frozenset()
    elif sr_ops_env in SR_OPERAND_PRESETS:
        sr_ops = SR_OPERAND_PRESETS[sr_ops_env]
    else:
        sr_ops = set()
        for token in (t.strip() for t in sr_ops_env.split(",")):
            if not token:
                continue
            if "@" not in token:
                raise ValueError(
                    f"FLUX_FP4_SR_OPERANDS token {token!r} is not tensor@pass; "
                    f"presets are {sorted(SR_OPERAND_PRESETS)}"
                )
            tensor, _, pass_name = token.partition("@")
            if (tensor, pass_name) not in SR_OPERANDS:
                raise ValueError(
                    f"FLUX_FP4_SR_OPERANDS has unknown pair {token!r}; valid pairs are "
                    + ", ".join(f"{t}@{p}" for t, p in sorted(SR_OPERANDS))
                )
            sr_ops.add((tensor, pass_name))
        sr_ops = frozenset(sr_ops)
    sr_ops = frozenset((t, p) for t, p in sr_ops if p in selected)

    if flag("FLUX_FP4_SR", "0"):
        # Fail rather than ignore: this flag looked like the SR control for a long
        # time and never was one, so silently accepting it would hand back a
        # round-to-nearest run that the log claims is stochastic.
        raise ValueError(
            "FLUX_FP4_SR does nothing -- Primus-Turbo's quantize_mxfp4 accepts "
            "use_sr and ignores it. Use FLUX_FP4_SR_PASSES=fprop,dgrad,wgrad "
            "(or 'all'), which routes to the FlyDSL packs' hardware SR convert."
        )

    return MXFP4LinearConfig(
        fprop="mxfp4" if "fprop" in selected else fprop_cast,
        dgrad="mxfp4" if "dgrad" in selected else "bf16",
        wgrad="mxfp4" if "wgrad" in selected else "bf16",
        hadamard=int(os.getenv("FLUX_FP4_HADAMARD", "16")),
        weight_2d_block=flag("FLUX_FP4_WEIGHT_2D", "0"),
        randomized_hadamard=flag("FLUX_FP4_RHT", "0"),
        save_quantized=flag("FLUX_FP4_SAVE_QUANTIZED", "1"),
        sr_passes=frozenset(sr_set),
        sr_operands=sr_ops,
    )


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


@torch.library.custom_op("primus_flux::quantize_mxfp4_rowwise", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_rowwise_op(
    x: torch.Tensor, use_2d_block: bool, use_sr: bool, use_rht: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
MXFP4-quantize along ``x``'s last dim.

Single-axis rather than ``quantize_mxfp4_dual``: each pass needs its operands rotated along *its own* contraction dim and one dual call can only carry one rotation, so the colwise half of a dual pack was always discarded -- half a packer paid for on every call.

``axis=1`` is the row direction; the kernel accepts only 0 or 1.

Wrapped as a custom op because the raw C++ op has no meta kernel, which would force a graph break and cost the torch.compile speedup.
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
    if _SCALE_ROUNDING_MODE:
        # The C++ quantize_mxfp4 takes no scale rounding mode, so this path would hand back a mode-0 pack
        # while the log claims otherwise -- the same silent lie FLUX_FP4_SR used to tell. 2D-block and
        # randomized RHT are what route here (see _fused_h16_quant_enabled), so the fix is to drop one of
        # those rather than to accept a mixed-grid run.
        raise ValueError(
            f"FLUX_FP4_SCALE_ROUNDING={_SCALE_ROUNDING_MODE} needs the FlyDSL H16 packs, but "
            "this tensor fell back to the C++ quantize_mxfp4, which always rounds at mode 0. "
            "Turn off FLUX_FP4_WEIGHT_2D / FLUX_FP4_RHT, or set the mode back to 0."
        )
    return _quantize_mxfp4_rowwise_op(x, use_2d, use_sr, use_rht)


if _FP4_DISPATCH:
    logger.info(f"MXFP4 Linear dispatch: {_FP4_DISPATCH}")
elif os.getenv("FLUX_FP4_PASSES", "off") not in ("", "off"):
    logger.info("MXFP4 operand pack: Python H16 + C++ quantize_mxfp4")


def _fused_h16_quant_enabled(hadamard: torch.Tensor | None, use_2d: bool, use_rht: bool) -> bool:
    """True when Linear should pack via FlyDSL ``flydsl_quant_mxfp4_h16``.

    Default off so an unflagged MXFP4 run stays Python H16 + C++ row-only.
    2D-block and randomized RHT have no fused kernel; those keep the original
    two-launch path. Flag is resolved at import so torch.compile does not trace
    ``os.getenv`` / logger (``sys._getframe``).

    SR used to disqualify the fused path too, on the grounds that it had no fused
    kernel. The opposite is true: the FlyDSL packs are the ONLY ones that round
    stochastically (the C++ ``quantize_mxfp4`` takes ``use_sr`` and ignores it), so
    SR now travels as a per-operand argument into these kernels instead of routing
    around them.
    """
    return bool(_FUSED_H16_QUANT and hadamard is not None and not use_2d and not use_rht)


@torch.library.custom_op("primus_flux::quantize_mxfp4_h16", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_h16_op(x: torch.Tensor, sr: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """H16 + row-only MXFP4 in one FlyDSL launch. Custom-op so torch.compile
    does not trace into FlyDSL JIT.

    ``sr`` rounds stochastically instead of to nearest, in-kernel and free.
    """
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16,
    )

    return flydsl_quant_mxfp4_h16(
        x.contiguous(), FP4_DTYPE, scale_rounding_mode=_SCALE_ROUNDING_MODE, sr=bool(sr)
    )


@_quantize_mxfp4_h16_op.register_fake
def _(x, sr=False):
    m, n = x.shape
    data = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    return data.view(FP4_DTYPE), scale.view(torch.float8_e8m0fnu)


@torch.library.custom_op("primus_flux::quantize_mxfp4_h16_dual", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_h16_dual_op(
    x: torch.Tensor, row_sr: bool = False, col_sr: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
One read of ``x`` -> both the row pack (along ``x``'s own last dim) and the col pack (along ``x.t()``'s last dim) that two separate ``_quantize_mxfp4_h16_op`` calls -- the second fed ``x.t().contiguous()`` -- would otherwise independently produce. Bit-identical to those two calls; this only removes the materialized transpose copy that used to sit between them.

``row_sr``/``col_sr`` select stochastic rounding per orientation, which is why the two travel separately rather than as one flag: the row pack serves Fprop (from x) or Dgrad (from g) while the col pack serves Wgrad, so each side carries its own pass's setting.
"""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16_dual,
    )

    return flydsl_quant_mxfp4_h16_dual(
        x.contiguous(),
        FP4_DTYPE,
        scale_rounding_mode=_SCALE_ROUNDING_MODE,
        row_sr=bool(row_sr),
        col_sr=bool(col_sr),
    )


@_quantize_mxfp4_h16_dual_op.register_fake
def _(x, row_sr=False, col_sr=False):
    m, n = x.shape
    row_data = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    row_scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    col_data = torch.empty(n, m // 2, dtype=torch.uint8, device=x.device)
    col_scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    return (
        row_data.view(FP4_DTYPE),
        row_scale.view(torch.float8_e8m0fnu),
        col_data.view(FP4_DTYPE),
        col_scale.view(torch.float8_e8m0fnu),
    )


@torch.library.custom_op("primus_flux::quantize_mxfp4_h16_dual_bias", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_h16_dual_bias_op(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
``_quantize_mxfp4_h16_dual_op`` plus the bias-gradient column sum, fused into the dual kernel's col phase -- see ``flydsl_quant_mxfp4_h16_dual_bias``'s docstring in ``mxfp4_quant_kernel.py`` for the mechanism.

``_MXFP4LinearFunction.backward`` used to read ``grad_2d`` a THIRD time via ``grad_2d.sum(0)`` for the bias gradient, after this same tensor had already been read once for Dgrad's dual pack. Used only for ``grad_2d`` when the module has a bias AND the plain dual pack would already be fused (``fuse_g``) -- see that call site; a bias-less Linear keeps using the plain ``_quantize_mxfp4_h16_dual_op`` so it never pays for an unused column sum.
"""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16_dual_bias,
    )

    return flydsl_quant_mxfp4_h16_dual_bias(
        x.contiguous(), FP4_DTYPE, scale_rounding_mode=_SCALE_ROUNDING_MODE
    )


@_quantize_mxfp4_h16_dual_bias_op.register_fake
def _(x):
    m, n = x.shape
    row_data = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    row_scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    col_data = torch.empty(n, m // 2, dtype=torch.uint8, device=x.device)
    col_scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    bias = torch.empty(n, dtype=x.dtype, device=x.device)
    return (
        row_data.view(FP4_DTYPE),
        row_scale.view(torch.float8_e8m0fnu),
        col_data.view(FP4_DTYPE),
        col_scale.view(torch.float8_e8m0fnu),
        bias,
    )


@torch.library.custom_op(
    "primus_flux::quantize_mxfp4_h16_dual_bias_legs", mutates_args=(), device_types="cuda"
)
def _quantize_mxfp4_h16_dual_bias_legs_op(
    x: torch.Tensor, leg0: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
py`'s `_emit_dual_bias_body`) at the producer boundary ``leg0`` / ``x.shape[1] - leg0`` -- e.g. single.l1's backward ``grad_2d`` is itself ``cat(FA-bwd dqkv[*, 9216], dGELU(linear2_C)[*, 12288])`` one level up (`_legpack_g_split` below is the shape-exact gate this is used through).

``leg0`` is a plain Python int (a shape constant from the caller-side gate, not data-dependent), so it is safe as a non-tensor custom-op arg.
"""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16_dual_bias_legs,
    )

    leg_widths = (leg0, x.shape[1] - leg0)
    return flydsl_quant_mxfp4_h16_dual_bias_legs(
        x.contiguous(), leg_widths, FP4_DTYPE, scale_rounding_mode=_SCALE_ROUNDING_MODE
    )


@_quantize_mxfp4_h16_dual_bias_legs_op.register_fake
def _(x, leg0):
    del leg0
    m, n = x.shape
    row_data = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    row_scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    col_data = torch.empty(n, m // 2, dtype=torch.uint8, device=x.device)
    col_scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    bias = torch.empty(n, dtype=x.dtype, device=x.device)
    return (
        row_data.view(FP4_DTYPE),
        row_scale.view(torch.float8_e8m0fnu),
        col_data.view(FP4_DTYPE),
        col_scale.view(torch.float8_e8m0fnu),
        bias,
    )


@torch.library.custom_op(
    "primus_flux::quantize_mxfp4_h16_dual_bias_legs_dgelu", mutates_args=(), device_types="cuda"
)
def _quantize_mxfp4_h16_dual_bias_legs_dgelu_op(
    d_qkv: torch.Tensor, d_act: torch.Tensor, preact: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_quantize_mxfp4_h16_dual_bias_legs_op(cat(d_qkv, gelu_backward(d_act, preact,
    "tanh")), d_qkv.shape[1])`` without writing the bf16 cat.

    ``d_act`` and ``preact`` are column slices of wider tensors in training and are read
    in place through their row strides; only a non-unit column stride forces a copy.
    """
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16_dual_bias_legs_dgelu,
    )

    d_qkv, d_act, preact = (t if t.stride(1) == 1 else t.contiguous() for t in (d_qkv, d_act, preact))
    return flydsl_quant_mxfp4_h16_dual_bias_legs_dgelu(
        d_qkv, d_act, preact, FP4_DTYPE, scale_rounding_mode=_SCALE_ROUNDING_MODE
    )


@_quantize_mxfp4_h16_dual_bias_legs_dgelu_op.register_fake
def _(d_qkv, d_act, preact):
    del preact
    m = d_qkv.shape[0]
    n = d_qkv.shape[1] + d_act.shape[1]
    row_data = torch.empty(m, n // 2, dtype=torch.uint8, device=d_qkv.device)
    row_scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=d_qkv.device)
    col_data = torch.empty(n, m // 2, dtype=torch.uint8, device=d_qkv.device)
    col_scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=d_qkv.device)
    bias = torch.empty(n, dtype=d_qkv.dtype, device=d_qkv.device)
    return (
        row_data.view(FP4_DTYPE),
        row_scale.view(torch.float8_e8m0fnu),
        col_data.view(FP4_DTYPE),
        col_scale.view(torch.float8_e8m0fnu),
        bias,
    )


@torch.library.custom_op(
    "primus_flux::quantize_mxfp4_h16_dual_bias_packed", mutates_args=(), device_types="cuda"
)
def _quantize_mxfp4_h16_dual_bias_packed_op(
    x: torch.Tensor, leg0: int, bx_scale: torch.Tensor, bw_scale: torch.Tensor
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """The bias dual pack (``leg0 == 0``: monolithic, else the two-leg split) plus both
    backward GEMMs' scales in the FP4 GEMM's packed layout: row/col scale as the Dgrad/
    Wgrad A operands, ``bx_scale`` (activation col scale) / ``bw_scale`` (weight col
    scale) as the Wgrad/Dgrad B operands. See ``flydsl_quant_mxfp4_h16_dual_bias_packed``."""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16_dual_bias_packed,
    )

    legs = (x.shape[1],) if leg0 == 0 else (leg0, x.shape[1] - leg0)
    return flydsl_quant_mxfp4_h16_dual_bias_packed(
        x.contiguous(),
        legs,
        FP4_DTYPE,
        bx_scale,
        bw_scale,
        scale_rounding_mode=_SCALE_ROUNDING_MODE,
    )


@_quantize_mxfp4_h16_dual_bias_packed_op.register_fake
def _(x, leg0, bx_scale, bw_scale):
    del leg0
    m, n = x.shape
    e8 = torch.float8_e8m0fnu
    row_data = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    row_scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    col_data = torch.empty(n, m // 2, dtype=torch.uint8, device=x.device)
    col_scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    bias = torch.empty(n, dtype=x.dtype, device=x.device)
    return (
        row_data.view(FP4_DTYPE),
        row_scale.view(e8),
        col_data.view(FP4_DTYPE),
        col_scale.view(e8),
        bias,
        torch.empty_like(row_scale).view(e8),
        torch.empty_like(col_scale).view(e8),
        torch.empty(bx_scale.shape, dtype=e8, device=x.device),
        torch.empty(bw_scale.shape, dtype=e8, device=x.device),
    )


def _dual_bias_packed_ok(grad_2d, x_s, w_s):
    if not _PACKED_SCALES or x_s is None or w_s is None:
        return False
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        dual_bias_packed_eligible,
    )

    R, C = grad_2d.shape
    bdim = x_s.shape[0]
    return (
        tuple(x_s.shape) == (bdim, R // MX_BLOCK_SIZE)
        and tuple(w_s.shape) == (bdim, C // MX_BLOCK_SIZE)
        and dual_bias_packed_eligible(R, C, bdim)
    )


def _packed_gemm_flags(grad_2d, x_s):
    """
``(dgrad_ok, wgrad_ok)``: may each backward GEMM read the pack the quant just wrote?

Both backward GEMMs get a pack, but upstream's GEMM tiles the packed layout and picks the tile from the shape, so it reads ours for some shapes and not others -- in practice the Dgrad and not the Wgrad, whose K is the token count. The one that cannot keeps its canonical scales.

So nothing in this function may import, memoise, or allocate. The predicate is bound once in ``_init_turbo`` and only called from here; see ``warm_packed_scale_api`` for what that costs to arrange.
"""
    if not _GEMM_FP4_PRESHUFFLED:
        return True, True  # overlay interface: it takes the pack at any shape
    _turbo_packed_layout_agrees = _TURBO_LAYOUT_OK
    R, C = grad_2d.shape
    bdim = int(x_s.shape[0])
    return _turbo_packed_layout_agrees(R, bdim, C), _turbo_packed_layout_agrees(C, bdim, R)


# Gated residuals `x + gate * linear(h)` (FLUX_GATE_DGATE): backward hands the Linear dY instead
# of the bf16 `gate * dY`, and one pack applies the gate at its load and accumulates dgate from
# the same read. Off, `forward_gated` and the gated GELU-input path are never called.
_GATE_DGATE = os.getenv("FLUX_GATE_DGATE", "0") == "1"
# The fused kernel needs every 64-row tile inside one batch sample.
_GATE_DGATE_ROWS = 64


@torch.library.custom_op(
    "primus_flux::quantize_mxfp4_h16_dual_bias_gate", mutates_args=(), device_types="cuda"
)
def _quantize_mxfp4_h16_dual_bias_gate_op(
    dy: torch.Tensor,
    gate: torch.Tensor,
    y: torch.Tensor,
    y_bias: torch.Tensor,
    bx_scale: torch.Tensor,
    bw_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """``_quantize_mxfp4_h16_dual_bias_packed_op(gate * dy, 0, bx_scale, bw_scale)`` with the
    gate applied at the pack's bf16 load, plus ``dgate[b] = sum_l dy * (y + y_bias)`` from the
    same read. ``gate`` is ``[B, C]`` and the rows of ``dy`` and ``y`` are B samples of R / B
    tokens. See ``flydsl_quant_mxfp4_h16_dual_bias_gate``."""
    from primus_turbo.flydsl.quantization import mxfp4_quant_kernel as _mqk

    return _mqk.flydsl_quant_mxfp4_h16_dual_bias_gate(
        dy.contiguous(),
        gate if getattr(_mqk, "GATE_ROW_STRIDE_OK", False) else gate.contiguous(),
        y.contiguous(),
        y_bias.contiguous(),
        FP4_DTYPE,
        bx_scale,
        bw_scale,
    )


@_quantize_mxfp4_h16_dual_bias_gate_op.register_fake
def _(dy, gate, y, y_bias, bx_scale, bw_scale):
    del y, y_bias
    m, n = dy.shape
    e8 = torch.float8_e8m0fnu
    row_scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=dy.device)
    col_scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=dy.device)
    return (
        torch.empty(m, n // 2, dtype=torch.uint8, device=dy.device).view(FP4_DTYPE),
        row_scale.view(e8),
        torch.empty(n, m // 2, dtype=torch.uint8, device=dy.device).view(FP4_DTYPE),
        col_scale.view(e8),
        torch.empty(n, dtype=dy.dtype, device=dy.device),
        torch.empty_like(row_scale).view(e8),
        torch.empty_like(col_scale).view(e8),
        torch.empty(bx_scale.shape, dtype=e8, device=dy.device),
        torch.empty(bw_scale.shape, dtype=e8, device=dy.device),
        torch.empty(gate.shape, dtype=dy.dtype, device=dy.device),
    )


def _gate_dgate_ok(grad_2d, gate_2d, y_2d, y_bias, x_s, w_s):
    R, C = grad_2d.shape
    B = gate_2d.shape[0]
    return (
        B > 0
        and R % B == 0
        and (R // B) % _GATE_DGATE_ROWS == 0
        and _legpack_g_split(R, C) is None
        and all(t.dtype == torch.bfloat16 for t in (grad_2d, gate_2d, y_2d, y_bias))
        and _dual_bias_packed_ok(grad_2d, x_s, w_s)
    )


def _gate_dgate_unfused(grad_2d, gate_2d, y_2d, y_bias):
    """``(gate * dY, dgate)`` in torch, for a shape or format the fused pack does not take.
    dgate accumulates in fp32, as Inductor's reduction does."""
    B, C = gate_2d.shape
    dy = grad_2d.view(B, -1, C)
    dgate = (dy.float() * (y_2d.view(B, -1, C).float() + y_bias.float())).sum(1).to(grad_2d.dtype)
    return (dy * gate_2d.unsqueeze(1)).view(grad_2d.shape), dgate


@torch.library.custom_op(
    "primus_flux::quantize_mxfp4_h16_dual_bias_dgelu", mutates_args=(), device_types="cuda"
)
def _quantize_mxfp4_h16_dual_bias_dgelu_op(
    d_act: torch.Tensor, preact: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_quantize_mxfp4_h16_dual_bias_op(gelu_backward(d_act, preact, "tanh"))`` without
    writing the bf16 dGELU: the double blocks' MLP-up G."""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16_dual_bias_dgelu,
    )

    d_act, preact = (t if t.stride(1) == 1 else t.contiguous() for t in (d_act, preact))
    return flydsl_quant_mxfp4_h16_dual_bias_dgelu(
        d_act, preact, FP4_DTYPE, scale_rounding_mode=_SCALE_ROUNDING_MODE
    )


@_quantize_mxfp4_h16_dual_bias_dgelu_op.register_fake
def _(d_act, preact):
    del preact
    m, n = d_act.shape
    row_data = torch.empty(m, n // 2, dtype=torch.uint8, device=d_act.device)
    row_scale = torch.empty(m, n // MX_BLOCK_SIZE, dtype=torch.uint8, device=d_act.device)
    col_data = torch.empty(n, m // 2, dtype=torch.uint8, device=d_act.device)
    col_scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=d_act.device)
    bias = torch.empty(n, dtype=d_act.dtype, device=d_act.device)
    return (
        row_data.view(FP4_DTYPE),
        row_scale.view(torch.float8_e8m0fnu),
        col_data.view(FP4_DTYPE),
        col_scale.view(torch.float8_e8m0fnu),
        bias,
    )


def _legpack_g_split(R, C):
    """
only this one shape is validated as a keep so far.
"""
    if R == 16384 and C == 21504:
        return (9216, 12288)
    return None


@torch.library.custom_op("primus_flux::quantize_mxfp4_h16_col", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_h16_col_op(x: torch.Tensor, sr: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
Colwise-transpose-read H16 + MXFP4 pack in one FlyDSL launch: bit- identical to ``_quantize_mxfp4_h16_op(x.t().contiguous())`` (the row pack of ``x``'s transpose) without materializing that transpose copy first. Custom-op so torch.compile does not trace into FlyDSL JIT.

Weight's own row pack (Fprop) is untouched -- it keeps using ``_quantize_mxfp4_h16_op`` directly on the un-transposed weight.
"""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_mxfp4_h16_col,
    )

    return flydsl_quant_mxfp4_h16_col(
        x.contiguous(), FP4_DTYPE, scale_rounding_mode=_SCALE_ROUNDING_MODE, sr=bool(sr)
    )


@_quantize_mxfp4_h16_col_op.register_fake
def _(x, sr=False):
    m, n = x.shape  # x is [N, K] (e.g. weight), untransposed
    data = torch.empty(n, m // 2, dtype=torch.uint8, device=x.device)  # [K, N/2]
    scale = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)  # [K, N/32]
    return data.view(FP4_DTYPE), scale.view(torch.float8_e8m0fnu)


# Delayed scales, keyed by the address of the calling module's slot tensor.
# Both halves of that sentence are forced, and each rules out the obvious alternative.
# The values do NOT live in the module's tensor, they live here, because putting them there means
# mutating a tensor inside the op; a mutating custom op gets wrapped in an `auto_functionalized`
# node; and Inductor's `decompose_auto_functionalized` post-grad pass cannot remove that node for a
# buffer captured by a per-block compiled region, so it raises "auto_functionalized was not
# removed" during the MLPerf warmup compile, before step 1. That happens on BOTH functionalization
# paths -- turning `enable_auto_functionalized_v2` off only changes which of the two spellings
# appears in the assertion. A pure op with its state behind it cannot reach that pass at all.
# The key is a tensor ADDRESS and not an `int` slot, even though an int reads better, because
# Dynamo value-specialises int arguments. Every block's Linears would carry different slot numbers
# while sharing one `DoubleStreamBlock.forward` code object, so each block forced its own graph and
# the run died on "Dynamo recompile limit exceeded" instead. Tensors are guarded on metadata only,
# never on value, so a tensor key lets all blocks share a single graph.
# So the module's tensor is a pure identity token: never read, never written, only its address
# used. Writing to it from in here would be exactly the unannounced mutation functionalization
# assumes cannot happen.
# Plain Python state is safe only because a custom op is opaque to Dynamo and runs eagerly -- the
# same property that lets FlyDSL JIT live behind these ops. Values stay device-resident f32
# tensors, never floats, because reading one as a float would sync to the host once per Linear per
# step, which is the exact cost this fusion exists to remove.
_FP8_DUAL_SCALE: dict[int, torch.Tensor] = {}


def _scale_inv_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """``amax`` -> a 1-element f32 ``scale_inv``, matching `quantize_fp8`'s convention exactly.

    Shaped ``(1,)`` rather than left as the 0-dim tensor ``amax()`` returns, because the op's fake
    must agree with the real one on rank or the custom-op boundary rejects the call outright with
    "wrong number of dimensions".

    ``.float()`` before the divide is load-bearing, not defensive: ``amax()`` of a BF16 tensor is
    BF16, so dividing first would round the quotient to 8 mantissa bits -- a 1.8e-03 scale error
    that flips 485994 FP8 codes.
    """
    return (amax.float().clamp(min=1e-12) / 448.0).reshape(1)


@torch.library.custom_op("primus_flux::fp8_mxfp4_dual", mutates_args=(), device_types="cuda")
def _fp8_mxfp4_dual_op(
    x: torch.Tensor, slot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
One read of ``x`` -> (FP8 row pack, FP8 scale_inv, MXFP4 col pack, col scales).

Replaces three separate full-tensor reads that the FP8 fprop pays today, all of the same tensor: ``quantize_fp8``'s amax reduction, ``quantize_fp8``'s row cast, and ``_quantize_mxfp4_h16_col_op``'s Wgrad col pack.

``slot`` is the calling module's identity token; its address names this Linear's entry in `_FP8_DUAL_SCALE`, and its contents are never touched. The pack uses the scale found there and then replaces it with the value implied by the amax the same kernel reduced, so step N's amax sets step N+1's scale -- delayed scaling. The scale returned alongside the pack is the one that actually applied to it, which is what the GEMM needs for dequant.

The slot holds ``scale_inv`` (``amax/448``) and not the forward scale, even though the kernel wants the forward scale, because the CONVENTION has to match ``quantize_fp8`` exactly rather than approximately. ``quantize_fp8`` produces ``amax/448`` and quantizes with its reciprocal, so storing ``amax/448`` and inverting once here reproduces it bit for bit.

The bootstrap computes a real amax so step 0 is not quantized with a garbage scale. It costs one extra read on the first step of the run and nothing after.
"""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_fp8_mxfp4_dual,
    )
    from primus_turbo.pytorch.core.low_precision import float8_e4m3

    xc = x.contiguous()
    key = slot.data_ptr()
    used_inv = _FP8_DUAL_SCALE.get(key)
    if used_inv is None:
        used_inv = _scale_inv_from_amax(xc.abs().amax())
    if _FP8_DUAL_DELAYED:
        # The delayed kernel takes scale_inv directly and returns the NEXT step's scale_inv, so the
        # reciprocal on the way in and the _scale_inv_from_amax on the way out both disappear into
        # FlyDSL. Same slot convention, same value in the slot.
        from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
            flydsl_quant_fp8_mxfp4_dual_delayed,
        )

        row_q, col_q, col_s, _FP8_DUAL_SCALE[key] = flydsl_quant_fp8_mxfp4_dual_delayed(
            xc, FP4_DTYPE, float8_e4m3, used_inv
        )
        return row_q, used_inv, col_q, col_s
    row_q, col_q, col_s, amax_p = flydsl_quant_fp8_mxfp4_dual(xc, FP4_DTYPE, float8_e4m3, 1.0 / used_inv)
    # Rebind the dict entry rather than mutate the tensor, so the scale returned above keeps the
    # value the pack was made with and the op stays pure. Same stream, so this is ordered after
    # the kernel's read.
    _FP8_DUAL_SCALE[key] = _scale_inv_from_amax(amax_p.amax())
    return row_q, used_inv, col_q, col_s


@_fp8_mxfp4_dual_op.register_fake
def _(x, slot):
    m, n = x.shape
    row = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=x.device)
    used_inv = torch.empty(1, dtype=torch.float32, device=x.device)
    col = torch.empty(n, m // 2, dtype=torch.uint8, device=x.device)
    col_s = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    return row, used_inv, col.view(FP4_DTYPE), col_s.view(torch.float8_e8m0fnu)


# Forward GELU folded into the next Linear's FP8 dual pack (FLUX_FWD_GELU_PACK). The producing
# Linear hands on its pre-activation in place of gelu(pre-activation) -- a "lazy act": its value
# is the pre-activation, but the gradient flowing back into it is d(act), which that Linear's own
# backward turns into dGELU. So a lazy act may only be produced after `lazy_gelu_into` confirmed
# its one consumer packs it with the GELU applied (`_fp8_mxfp4_dual_legs_gelu_op`).
_FWD_GELU_PACK = os.getenv("FLUX_FWD_GELU_PACK", "0") == "1"
_FWD_GELU_PACK_CALLS = [0, 0]  # [with attn leg, gelu only]


@torch.library.custom_op("primus_flux::fp8_mxfp4_dual_legs_gelu", mutates_args=(), device_types="cuda")
def _fp8_mxfp4_dual_legs_gelu_op(
    attn: torch.Tensor | None, preact: torch.Tensor, slot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_fp8_mxfp4_dual_op(cat(attn, gelu(preact)), slot)` -- or of ``gelu(preact)`` when
    ``attn`` is None -- without writing the cat or the GELU output. Same slot, same delayed
    scale convention; ``preact`` may be a column-slice view."""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_fp8_mxfp4_dual_legs_delayed,
    )
    from primus_turbo.pytorch.core.low_precision import float8_e4m3

    if preact.stride(1) != 1:
        preact = preact.contiguous()
    if attn is not None and attn.stride(1) != 1:
        attn = attn.contiguous()
    xs = (preact,) if attn is None else (attn, preact)
    gelus = (True,) if attn is None else (False, True)
    _FWD_GELU_PACK_CALLS[attn is None] += 1
    if sum(_FWD_GELU_PACK_CALLS) == 1000:
        logger.info(
            f"FLUX_FWD_GELU_PACK: first 1000 fused packs: {_FWD_GELU_PACK_CALLS[0]} single-block, "
            f"{_FWD_GELU_PACK_CALLS[1]} double-block"
        )
    key = slot.data_ptr()
    used_inv = _FP8_DUAL_SCALE.get(key)
    if used_inv is None:
        amax = torch.nn.functional.gelu(preact, approximate="tanh").abs().amax()
        if attn is not None:
            amax = torch.maximum(amax, attn.abs().amax())
        used_inv = _scale_inv_from_amax(amax)
    row_q, col_q, col_s, _FP8_DUAL_SCALE[key] = flydsl_quant_fp8_mxfp4_dual_legs_delayed(
        xs, gelus, FP4_DTYPE, float8_e4m3, used_inv
    )
    return row_q, used_inv, col_q, col_s


@_fp8_mxfp4_dual_legs_gelu_op.register_fake
def _(attn, preact, slot):
    m = preact.shape[0]
    n = preact.shape[1] + (0 if attn is None else attn.shape[1])
    row = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=preact.device)
    used_inv = torch.empty(1, dtype=torch.float32, device=preact.device)
    col = torch.empty(n, m // 2, dtype=torch.uint8, device=preact.device)
    col_s = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=preact.device)
    return row, used_inv, col.view(FP4_DTYPE), col_s.view(torch.float8_e8m0fnu)


# Gated on out_features rather than the module's FQN so this needs no change to any file outside
# this one: 9216 = QKV (3 * hidden=3072), 21504 = single-block linear1 (7 * hidden=3072, FLUX's
# single-stream blocks fuse the MLP-up projection into the same Linear as QKV).
_FUSE_BIAS_OUT_FEATURES = frozenset((9216, 21504))
_FUSE_BIAS_EPILOGUE = os.getenv("FLUX_FP4_FUSE_BIAS_EPILOGUE", "1") == "1"

# The same fusion for the FP8 tensorwise fprop, on its own flag and defaulted OFF.
_FP8_FUSE_BIAS_EPILOGUE = os.getenv("FLUX_FP8_FUSE_BIAS_EPILOGUE", "0") == "1"

# The A6W4 GEMM's bias epilogue, on its own flag so it ablates apart from the MXFP4 one.
_A6W4_FUSE_BIAS_EPILOGUE = os.getenv("FLUX_A6W4_FUSE_BIAS_EPILOGUE", "1") == "1"


def _fuse_bias_eligible(cfg: MXFP4LinearConfig, out_features: int, any_width: bool = False) -> bool:
    """
True iff this Linear's `+bias` should move into the GEMM epilogue instead of running as a separate elementwise add. Requires a quantized Fprop (bf16 Fprop has no fused-GEMM epilogue to fuse into) and one of the opaque-consumer output widths, or ``any_width`` from a caller that saves the biased output itself (the GELU Function below, whose saved pre-activation would otherwise cost its own write).

MXFP6 qualifies as well as MXFP4: AITER's A6W6 asm GEMM shares the A4W4 kernarg ABI, bias epilogue included.

FP8 qualifies only under its own flag.
"""
    width_ok = any_width or out_features in _FUSE_BIAS_OUT_FEATURES
    if cfg.fprop == "fp8":
        return _FP8_FUSE_BIAS_EPILOGUE and width_ok
    if cfg.fprop == "a6w4":
        # Rounds like an eager separate add. Under torch.compile it is not byte-identical: without it
        # Inductor folds the add into the consumer (GELU, QK norm) in f32, unrounded. Opaque-consumer
        # widths only: elsewhere Inductor already folds the add for free.
        return _A6W4_FUSE_BIAS_EPILOGUE and out_features in _FUSE_BIAS_OUT_FEATURES
    return _FUSE_BIAS_EPILOGUE and cfg.fprop in ("mxfp4", "mxfp6") and width_ok


def _dual_pack_eligible(hadamard: torch.Tensor | None, cfg: MXFP4LinearConfig) -> bool:
    """True iff BOTH orientations ``_quantize_operand`` would independently
    produce for one logical 2D tensor -- row along its own last dim, col
    along its transpose's last dim -- resolve to the SAME
    ``_quantize_mxfp4_h16_op`` fused-H16 path, so collapsing them into one
    ``_quantize_mxfp4_h16_dual_op`` call changes nothing but the read count.

    Both orientations always pass ``use_2d=False`` (2D block scaling is only
    ever requested for the weight operand, never for X or G) and the same
    ``use_rht=cfg.randomized_hadamard`` (a cfg-level flag, not per-operand),
    so neither can differ between row and col.

    SR does differ between the two orientations, and that is fine rather than
    disqualifying: the dual kernel takes ``row_sr`` and ``col_sr`` separately, and
    its row/col split lines up exactly with the per-pass split -- row serves Fprop
    (from X) or Dgrad (from G), col serves Wgrad. So each orientation carries its
    own pass's setting inside one launch. This used to gate on SR and refuse to
    fuse, which had it backwards: the fused packs are the only ones that round
    stochastically at all.
    """
    return _fused_h16_quant_enabled(hadamard, False, cfg.randomized_hadamard)


def _fp8_dual_pack_eligible(
    t: torch.Tensor,
    hadamard: torch.Tensor | None,
    cfg: MXFP4LinearConfig,
    tensor: str,
    col_pass: str,
) -> bool:
    """
True iff ``tensor``'s FP8 row pack and MXFP4 col pack can come from one read.

Serves both operands of an FP8 fprop, which need the same pair of orientations: the activation wants an FP8 row pack for Fprop and an MXFP4 col pack for Wgrad, the weight wants an FP8 row pack for Fprop and an MXFP4 col pack for Dgrad. Hence ``col_pass``.

Builds on `_dual_pack_eligible`, which already asks the col-orientation question, and adds what is specific to mixing an FP8 row half in:

no SR on either half -- the fused kernel's col phase is the round-to-nearest one, copied verbatim from `_emit_dual_bias_body` so its output stays bit-exact with `flydsl_quant_mxfp4_h16_col`, and its row half is a plain FP8 cast. The current recipe has FLUX_FP4_SR=0 with no per-pass or per-operand overrides, so this is always true today; gating on it means a future SR run falls back to the separate packs rather than silently losing the stochastic rounding.

no 2D block scaling -- only ever requested for the weight, and it is a different scale layout than the 1D rowwise pack this kernel emits.

shape alignment -- `fp8_dual_eligible`'s R % 128 and C % 256, inherited from the validated H16 dual geometry. Every Flux fprop activation and weight clears it.
"""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import fp8_dual_eligible

    if tensor == "weight" and cfg.weight_2d_block:
        return False
    return (
        _dual_pack_eligible(hadamard, cfg)
        and not cfg.sr_op(tensor, col_pass)
        and not cfg.sr_op(tensor, "fprop")
        and fp8_dual_eligible(t.shape[0], t.shape[1])
    )


def _weight_colpack_eligible(hadamard: torch.Tensor | None, cfg: MXFP4LinearConfig) -> bool:
    """True iff weight's Dgrad col-pack -- today ``weight.t().contiguous()``
    fed to ``_quantize_operand(..., use_2d=cfg.weight_2d_block,
    use_rht=cfg.randomized_hadamard)`` at the Dgrad call site below -- would
    resolve to the fused-H16 path. Mirrors ``_dual_pack_eligible``'s pattern but
    for a single operand/orientation instead of a row+col pair, with the exact
    args that one call site always uses."""
    return _fused_h16_quant_enabled(hadamard, cfg.weight_2d_block, cfg.randomized_hadamard)


def _quantize_operand(
    x: torch.Tensor,
    hadamard: torch.Tensor | None,
    use_2d: bool,
    use_sr: bool,
    use_rht: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % 128:
        raise ValueError(f"MXFP4 quantized dim must be a multiple of 128, got {x.shape[-1]}")
    if _fused_h16_quant_enabled(hadamard, use_2d, use_rht):
        return _quantize_mxfp4_h16_op(x, use_sr)
    if use_sr:
        # Only the FlyDSL packs round stochastically; the C++ one accepts use_sr
        # and ignores it, so honouring the request here is impossible. Refuse
        # rather than return round-to-nearest data that looks like it complied.
        raise NotImplementedError(
            "stochastic rounding needs the fused FlyDSL pack "
            f"(use_2d={use_2d}, use_rht={use_rht}, hadamard={hadamard is not None}, "
            f"dispatch={_FP4_DISPATCH!r} all have to allow it)"
        )
    return _quantize_mxfp4_rowwise_op(rotate_last_dim(x, hadamard), use_2d, False, use_rht)


def _mxfp4_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    packed_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
a[M,K] @ b[N,K]^T -> [M,N]; both operands quantized along K.

``packed_scales``: ``(a_scale, b_scale)`` already in the FlyDSL GEMM's packed layout; the GEMM then skips its scale preshuffle (it falls back to the canonical scales on its own when the shape/config cannot use them).

``bias``: fp32 [N], fused into gemm_fp4_impl's FlyDSL epilogue when that backend and shape are eligible; gemm_fp4_impl falls back to a host-side add on its own if not (e.g. a non-FlyDSL backend, or a shape the tile autotune steers off the fused-store path), so this call is correct either way.
"""
    kw = {"granularity": _GRANULARITY, "default_backend": _BACKEND}
    host_bias = None
    if _GEMM_FP4_BIAS:
        kw["bias"] = bias
    else:
        host_bias = bias  # operator has no epilogue to fuse into; add it below
    if packed_scales is not None:
        if _GEMM_FP4_PACKED_KW:
            kw["a_scale_packed"], kw["b_scale_packed"] = packed_scales
        elif _GEMM_FP4_PRESHUFFLED:
            # Upstream takes the packed scales INSTEAD of the canonical ones, as flat int32,
            # and infers the layout from the tensor rather than from a flag -- so the canonical
            # arguments are the ones being replaced here, not supplemented. Whether this GEMM's
            # shape can read our layout was settled by `_packed_gemm_flags` in the caller's
            # eager body; nothing may ask it here, where this runs under tracing.
            a_scale, b_scale = (s.view(torch.int32).reshape(-1) for s in packed_scales)
            kw["preshuffled"] = True
    out = _GEMM_FP4(a, a_scale, False, b, b_scale, True, out_dtype, False, **kw)
    return out if host_bias is None else out + host_bias


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


@torch.library.custom_op("primus_flux::fp8_tensorwise_mm", mutates_args=(), device_types="cuda")
def _fp8_tensorwise_mm_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tensorwise FP8 fprop: A[M,K] @ B[N,K]^T -> [M,N].

    Uses Primus-Turbo ``quantize_fp8`` + ``gemm_fp8_impl`` (or FlyDSL) so this
    path matches Flux TorchAO tensorwise numerics without pulling TorchAO's full
    FP8 autograd into the module -- dgrad/wgrad stay MXFP4.
    """
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
    from primus_turbo.pytorch.ops.quantization import quantize_fp8

    a_fp8, a_si = quantize_fp8(a.contiguous(), float8_e4m3, ScalingGranularity.TENSORWISE)
    b_fp8, b_si = quantize_fp8(b.contiguous(), float8_e4m3, ScalingGranularity.TENSORWISE)
    return _fp8_tensorwise_gemm(a_fp8, a_si, b_fp8, b_si, a.dtype)


def _fp8_tensorwise_gemm(a_fp8, a_si, b_fp8, b_si, out_dtype):
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

    if _FP8_FPROP_BACKEND == "flydsl":
        from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
            gemm_fp8_tensorwise_flydsl_kernel,
        )

        return gemm_fp8_tensorwise_flydsl_kernel(a_fp8, a_si, b_fp8, b_si, False, True, out_dtype)
    return gemm_fp8_impl(
        a_fp8,
        a_si,
        False,
        b_fp8,
        b_si,
        True,
        out_dtype,
        False,
        granularity=ScalingGranularity.TENSORWISE.value,
        default_backend=BackendType.HIPBLASLT.value,
    )


@_fp8_tensorwise_mm_op.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)


@torch.library.custom_op("primus_flux::fp8_tensorwise_mm_prequant_w", mutates_args=(), device_types="cuda")
def _fp8_tensorwise_mm_prequant_w_op(
    a: torch.Tensor, b_fp8: torch.Tensor, b_scale_inv: torch.Tensor
) -> torch.Tensor:
    """``_fp8_tensorwise_mm_op`` with B already quantized (FP8 all-gather), as uint8 bytes."""
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
    from primus_turbo.pytorch.ops.quantization import quantize_fp8

    a_fp8, a_si = quantize_fp8(a.contiguous(), float8_e4m3, ScalingGranularity.TENSORWISE)
    return _fp8_tensorwise_gemm(a_fp8, a_si, b_fp8.view(float8_e4m3), b_scale_inv, a.dtype)


@_fp8_tensorwise_mm_prequant_w_op.register_fake
def _(a: torch.Tensor, b_fp8: torch.Tensor, b_scale_inv: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b_fp8.shape[0], device=a.device, dtype=a.dtype)


@torch.library.custom_op("primus_flux::fp8_tensorwise_mm_aq", mutates_args=(), device_types="cuda")
def _fp8_tensorwise_mm_aq_op(
    a_fp8: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    b_scale_inv: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """`_fp8_tensorwise_mm_op` with A -- and optionally B -- already quantized.

    Both operands of an FP8 fprop can come from `_fp8_mxfp4_dual_op`: the activation's row pack
    shares a read with its Wgrad col pack, and the weight's shares one with its Dgrad col pack.
    ``b_scale_inv`` is what distinguishes the two cases -- given, ``b`` is already FP8 and is
    used as-is; omitted, ``b`` is BF16 and is quantized here as before.

    Split out from `_fp8_tensorwise_mm_op` rather than made an option on it, so the unfused path
    stays byte-for-byte the op it has always been and an A/B of `FLUX_FP8_DUAL_PACK` moves one
    thing.

    ``bias`` is added to the rounded product -- in the FlyDSL epilogue, or as a separate add on
    the hipBLASLt path -- so either way the result is bit-identical to ``out + bias``. Only the
    caller's `_fuse_bias_eligible` decides when it is passed; see `_FP8_FUSE_BIAS_EPILOGUE`.
    """
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3
    from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
    from primus_turbo.pytorch.ops.quantization import quantize_fp8

    if b_scale_inv is None:
        b_fp8, b_si = quantize_fp8(b.contiguous(), float8_e4m3, ScalingGranularity.TENSORWISE)
    else:
        b_fp8, b_si = b, b_scale_inv
    if _FP8_FPROP_BACKEND == "flydsl":
        from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
            gemm_fp8_tensorwise_flydsl_kernel,
        )

        # Passing it unconditionally therefore breaks the unflagged path against the image's own kernel --
        # which it did, killing both FUSE_BIAS=0 arms of the first ladder batch with "unexpected keyword
        # argument 'bias'" while the three flagged arms ran fine.
        if bias is None:
            return gemm_fp8_tensorwise_flydsl_kernel(a_fp8, a_scale_inv, b_fp8, b_si, False, True, out_dtype)
        return gemm_fp8_tensorwise_flydsl_kernel(
            a_fp8, a_scale_inv, b_fp8, b_si, False, True, out_dtype, bias=bias
        )
    out = gemm_fp8_impl(
        a_fp8,
        a_scale_inv,
        False,
        b_fp8,
        b_si,
        True,
        out_dtype,
        False,
        granularity=ScalingGranularity.TENSORWISE.value,
        default_backend=BackendType.HIPBLASLT.value,
    )
    return out if bias is None else out.add_(bias)


@_fp8_tensorwise_mm_aq_op.register_fake
def _(a_fp8, a_scale_inv, b, out_dtype, b_scale_inv=None, bias=None):
    return torch.empty(a_fp8.shape[0], b.shape[0], device=a_fp8.device, dtype=out_dtype)


_ACT_STATS = os.getenv("FLUX_FP6_ACT_STATS", "0") == "1"
_ACT_STATS_EVERY = int(os.getenv("FLUX_FP6_ACT_STATS_EVERY", "200"))
_act_stats: dict = {}
_act_stats_calls = 0
try:
    with open("/results/fp6_act_stats_import.txt", "w") as _fh:
        _fh.write(
            "module imported; FLUX_FP6_ACT_STATS=%r -> _ACT_STATS=%s\n"
            % (os.getenv("FLUX_FP6_ACT_STATS"), _ACT_STATS)
        )
except OSError:
    pass


def _record_act_stats(x: torch.Tensor, w: torch.Tensor) -> None:
    """
Accumulate outlier statistics for real MXFP6 fprop operands. Diagnostic only.

Exists to settle one question. A rotation along K is algebraically free -- (A Q)(Q^T B^T) = A B^T -- and is the standard way to reduce quantization error, but it only pays by spreading outliers.

So the rotation question reduces to whether the real operands are heavy-tailed. What matters is the ratio of the group maximum to the group RMS *within each 32-element scale group along K*, since that is the granularity the format scales at: a group whose max dwarfs its RMS spends its 3 mantissa bits representing one large value and quantizes the rest coarsely. For Gaussian data that ratio is about 2.2; meaningfully above that means headroom a rotation could recover.
"""
    global _act_stats_calls
    _act_stats_calls += 1
    if _act_stats_calls == 1:
        print(
            "FLUX_FP6_ACT_STATS: first fprop seen, shapes x=%s w=%s" % (tuple(x.shape), tuple(w.shape)),
            flush=True,
        )
    if _act_stats_calls % _ACT_STATS_EVERY:
        return
    with torch.no_grad():
        for tag, t in (("act", x), ("weight", w)):
            K = t.shape[-1]
            if K % 32:
                continue
            g = t.detach().float().reshape(-1, 32)
            rms = g.pow(2).mean(dim=1).sqrt()
            mx = g.abs().max(dim=1).values
            ratio = mx / rms.clamp_min(1e-12)
            key = (tag, tuple(t.shape))
            prev = _act_stats.get(key, (0, 0.0, 0.0, 0.0))
            n = prev[0] + 1
            _act_stats[key] = (
                n,
                prev[1] + ratio.mean().item(),
                max(prev[2], ratio.max().item()),
                prev[3] + (g.pow(4).mean() / g.pow(2).mean().pow(2)).item(),
            )
    if _act_stats_calls % (_ACT_STATS_EVERY * 20) == 0:
        _dump_act_stats()


def _dump_act_stats() -> None:
    lines = ["FLUX_FP6_ACT_STATS (group-of-32 max/rms; Gaussian reference is ~2.2)"]
    for (tag, shape), (n, sratio, mxratio, skurt) in sorted(_act_stats.items()):
        lines.append(
            "  %-6s %-22s samples=%-4d mean_max/rms=%.3f  worst=%.3f  kurtosis=%.2f"
            % (tag, str(shape), n, sratio / n, mxratio, skurt / n)
        )
    text = "\n".join(lines)
    print(text, flush=True)
    # Also to a file: stdout from the training workers is not reliably captured, and two
    # runs of this diagnostic produced no output despite the overlay being installed and
    # all 228 Linears reporting fprop=mxfp6. /results is bind-mounted, so a file survives
    # whatever the launcher does with stdout.
    try:
        with open("/results/fp6_act_stats.txt", "w") as fh:
            fh.write(text + "\n")
    except OSError:
        pass


@torch.library.custom_op("primus_flux::mxfp6_mm", mutates_args=(), device_types="cuda")
def _mxfp6_mm_op(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """
MXFP6 fprop: A[M,K] @ B[N,K]^T -> [M,N], via AITER's A6W6 asm GEMM.

That closes the convergence gap MXFP4 cannot (total remaining FP4 headroom is 1.85 dB) for 1.25x the fprop GEMM.

No rotation is applied. H16 exists to keep MXFP4's 8 levels from being dominated by block outliers; at 32 levels the rotation is not what is binding, and skipping it also avoids paying for a pack this path cannot fuse.

Dgrad and Wgrad are untouched and stay MXFP4, which is free: the Wgrad arm tracked all-pass within 0.0007 over 32 evals. That also sidesteps the gradient_accumulation_fusion opt-out an FP6 wgrad would force, since the A6W6 entry point has no beta=1 accumulate epilogue to write main_grad in place.
"""
    from aiter.ops.gemm_op_a6w6 import gemm_a6w6, quant_mxfp6_gemm

    a_q, a_s = quant_mxfp6_gemm(a.contiguous())
    b_q, b_s = quant_mxfp6_gemm(b.contiguous())
    return gemm_a6w6(
        a_q,
        b_q,
        a_s,
        b_s,
        a.shape[0],
        b.shape[0],
        a.shape[1],
        dtype=a.dtype,
        bias=bias,
    )


@_mxfp6_mm_op.register_fake
def _(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)


@torch.library.custom_op("primus_flux::a6w4_mm", mutates_args=(), device_types="cuda")
def _a6w4_mm_op(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """A6W4 fprop: H16-rotated MXFP6 (E2M3) activations against H16-rotated MXFP4 (E2M1)
    weights, on FlyDSL's mixed-format preshuffle GEMM. Bit-identical to the fwddiag
    emulation that tracked all-MXFP6, i.e. MXFP4 weights are not what breaks MXFP4 fprop.
    `bias` is added in the epilogue with the rounding of a separate add."""
    from primus_turbo.pytorch.ops.flux_a6w4 import a6w4_mm

    return a6w4_mm(a, b, bias)


@_a6w4_mm_op.register_fake
def _(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)


@torch.library.custom_op("primus_flux::a6w4_mm_prequant_w", mutates_args=(), device_types="cuda")
def _a6w4_mm_prequant_w_op(
    a: torch.Tensor, wq: torch.Tensor, sb: torch.Tensor, n: int, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """`a6w4_mm` against a weight `_A6W4GatheredWeight` already holds as quant_w_a6w4 bytes."""
    from primus_turbo.pytorch.ops.flux_a6w4 import a6w4_mm_prequant_w

    return a6w4_mm_prequant_w(a, wq, sb, n, bias)


@_a6w4_mm_prequant_w_op.register_fake
def _(
    a: torch.Tensor, wq: torch.Tensor, sb: torch.Tensor, n: int, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.empty(a.shape[0], n, device=a.device, dtype=a.dtype)


# The A6W4 forward's activation feeds two packs: the FP6 fprop operand and, when Wgrad is MXFP4,
# the H16 col pack Wgrad saves. `a6w4_act_dual` makes both from one read of x, byte-identical to
# quant_act_a6w4 and `quantize_mxfp4_h16_col`. 0 puts back the two separate reads.
_A6W4_ACT_DUAL = os.getenv("FLUX_A6W4_ACT_DUAL", "1") == "1"


def _a6w4_act_dual_ok(x_2d: torch.Tensor, cfg: MXFP4LinearConfig, hadamard: torch.Tensor | None) -> bool:
    m, k = x_2d.shape
    return (
        _A6W4_ACT_DUAL
        and cfg.fprop == "a6w4"
        and cfg.wgrad == "mxfp4"
        and cfg.save_quantized
        and _dual_pack_eligible(hadamard, cfg)
        and not cfg.sr_op("act", "wgrad")
        and x_2d.dtype == torch.bfloat16
        and m % 256 == 0
        and k % 1024 == 0
    )


_A6W4_GELU_DUAL = os.getenv("FLUX_A6W4_GELU_DUAL", "1") == "1"
_A6W4_GELU_DUAL_SEEN: dict = {}


def _a6w4_gelu_dual_ok(m: int, n: int, k: int, cfg: MXFP4LinearConfig, bias, a_pack: int) -> bool:
    """Whether the fused epilogue can serve this Linear, so the caller keeps its own path.

    It does not ask whether the bias was already in the epilogue: the fused store puts it
    there itself, and `act_rounded` is what carries the difference that choice made."""
    if not (_A6W4_GELU_DUAL and _FWD_GELU_PACK and cfg.fprop == "a6w4" and bias is not None):
        return False
    from primus_turbo.pytorch.ops.flux_a6w4 import a6w4_gelu_dual_eligible

    return a6w4_gelu_dual_eligible(m, n, k, a_pack)


@torch.library.custom_op("primus_flux::a6w4_mm_gelu_dual_qa_w", mutates_args=(), device_types="cuda")
def _a6w4_mm_gelu_dual_qa_w_op(
    aq: torch.Tensor, sa: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, act_rounded: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`a6w4_mm_gelu_dual_qa` against a weight still in bf16, as `a6w4_mm_qa` takes it.

    ``act_rounded`` is what the consumer this replaces would have activated: the biased
    output it read back (1), or the bias-free one plus its own f32 bias add (0)."""
    from primus_turbo.pytorch.ops.flux_a6w4 import a6w4_mm_gelu_dual_qa_prequant_w, quantizers

    wq, sb = quantizers()[1](b.to(torch.bfloat16).contiguous())
    _log_gelu_dual(aq, b.shape[0], wq, "bf16 weight")
    return a6w4_mm_gelu_dual_qa_prequant_w(
        aq, sa, wq, sb, b.shape[0], bias, act_rounded=act_rounded
    )


def _log_gelu_dual(aq: torch.Tensor, n: int, wq: torch.Tensor, weight: str) -> None:
    key = (aq.shape[0], n, aq.shape[1], weight)
    _A6W4_GELU_DUAL_SEEN[key] = _A6W4_GELU_DUAL_SEEN.get(key, 0) + 1
    if _A6W4_GELU_DUAL_SEEN[key] == 1:
        logger.info(
            f"A6W4_GELU_DUAL fused: M={aq.shape[0]} N={n} K={wq.numel() * 2 // n} "
            f"aq row={aq.shape[1]} bytes, {weight}"
        )


@torch.library.custom_op(
    "primus_flux::a6w4_mm_gelu_dual_qa_prequant_w", mutates_args=(), device_types="cuda"
)
def _a6w4_mm_gelu_dual_qa_prequant_w_op(
    aq: torch.Tensor,
    sa: torch.Tensor,
    wq: torch.Tensor,
    sb: torch.Tensor,
    n: int,
    bias: torch.Tensor,
    act_rounded: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`a6w4_mm_gelu_dual_qa_w` on `_A6W4GatheredWeight`'s already-quantized row pack."""
    from primus_turbo.pytorch.ops.flux_a6w4 import a6w4_mm_gelu_dual_qa_prequant_w

    _log_gelu_dual(aq, n, wq, "gathered A6W4 weight")
    return a6w4_mm_gelu_dual_qa_prequant_w(aq, sa, wq, sb, n, bias, act_rounded=act_rounded)


@_a6w4_mm_gelu_dual_qa_prequant_w_op.register_fake
def _(aq, sa, wq, sb, n, bias, act_rounded):
    return _gelu_dual_fake(aq, n)


@_a6w4_mm_gelu_dual_qa_w_op.register_fake
def _(aq, sa, b, bias, act_rounded):
    return _gelu_dual_fake(aq, b.shape[0])


def _gelu_dual_fake(aq: torch.Tensor, n: int):
    m = aq.shape[0]
    pre = torch.empty(m, n, dtype=torch.bfloat16, device=aq.device)
    r_q = torch.empty(m, n, dtype=torch.uint8, device=aq.device)
    r_s = torch.empty(m * n // MX_BLOCK_SIZE, dtype=torch.uint8, device=aq.device)
    c_q = torch.empty(n, m // 2, dtype=torch.uint8, device=aq.device).view(FP4_DTYPE)
    c_s = torch.empty(n, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=aq.device).view(
        torch.float8_e8m0fnu
    )
    return pre, r_q, r_s, c_q, c_s


@torch.library.custom_op("primus_flux::a6w4_act_dual", mutates_args=(), device_types="cuda")
def _a6w4_act_dual_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(A6W4 fprop A, its shuffled scales, Wgrad col pack, its scales) from one read of x."""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_a6w4_act_dual,
    )

    return flydsl_quant_a6w4_act_dual(x.contiguous())


@_a6w4_act_dual_op.register_fake
def _(x):
    m, k = x.shape
    aq = torch.empty(m, k, dtype=torch.uint8, device=x.device)
    sa = torch.empty(m * k // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    co = torch.empty(k, m // 2, dtype=torch.uint8, device=x.device).view(FP4_DTYPE)
    cs = torch.empty(k, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device).view(torch.float8_e8m0fnu)
    return aq, sa, co, cs


# The A6W4 Linears behind an adaLN modulate (double qkv and fc1, single linear1) pack (1 + scale)
# * layer_norm(x) + shift straight from x: a stats-only Triton pass emits each row's rstd and
# -mean * rstd, and the dual pack applies the LN and the modulate in its tile load, so the
# modulated bf16 activation is never written or reread.
_A6W4_LNMOD_PACK = os.getenv("FLUX_A6W4_LNMOD_PACK", "0") == "1"
_LNMOD_L_ALIGN = 64


@torch.library.custom_op("primus_flux::a6w4_act_dual_lnmod", mutates_args=(), device_types="cuda")
def _a6w4_act_dual_lnmod_op(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_a6w4_act_dual_op((1 + scale) * layer_norm(x) + shift)` for x [B, L, K] and shift/scale
    [B, 1, K], plus the LN stats (rstd, nmr) [B*L] fp32 that backward needs."""
    from primus_turbo.flydsl.flux.a6w4_fwd_pack import _ln_stats
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_a6w4_act_dual_lnmod,
    )

    x = x.contiguous()
    b, l, k = x.shape
    rstd, nmr = _ln_stats(x)
    aq, sa, co, cs = flydsl_quant_a6w4_act_dual_lnmod(
        x.view(b * l, k),
        rstd,
        nmr,
        shift.reshape(b, k).contiguous(),
        scale.reshape(b, k).contiguous(),
    )
    return aq, sa, co, cs, rstd, nmr


@_a6w4_act_dual_lnmod_op.register_fake
def _(x, shift, scale):
    b, l, k = x.shape
    m = b * l
    aq = torch.empty(m, k, dtype=torch.uint8, device=x.device)
    sa = torch.empty(m * k // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device)
    co = torch.empty(k, m // 2, dtype=torch.uint8, device=x.device).view(FP4_DTYPE)
    cs = torch.empty(k, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=x.device).view(torch.float8_e8m0fnu)
    rstd = torch.empty(m, dtype=torch.float32, device=x.device)
    return aq, sa, co, cs, rstd, torch.empty_like(rstd)


def _a6w4_lnmod_eligible(linear: "MXFP4Linear", x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> bool:
    if not (_A6W4_LNMOD_PACK and isinstance(linear, MXFP4Linear) and linear.config.any_mxfp4 and linear.training):
        return False
    if x.dim() != 3 or x.shape[1] % _LNMOD_L_ALIGN:
        return False
    b, _, k = x.shape
    return (
        shift.shape == (b, 1, k)
        and scale.shape == (b, 1, k)
        and shift.dtype == scale.dtype == torch.bfloat16
        and _a6w4_act_dual_ok(x.reshape(-1, k), linear.config, linear.hadamard)
    )


def _lnmod_pre(x, shift, scale):
    aq, sa, co, cs, rstd, nmr = _a6w4_act_dual_lnmod_op(x, shift, scale)
    return ((aq, sa, co, cs), tuple(x.shape), x.dtype), rstd, nmr


def _lnmod_backward(d_y, x, scale, rstd, nmr):
    """d(x, shift, scale) of ``(1 + scale) * layer_norm(x) + shift`` (eps 1e-6, no affine), in f32
    from the stats the forward pack used."""
    b, l, k = x.shape
    rstd = rstd.view(b, l, 1)
    xn = x.float() * rstd + nmr.view(b, l, 1)
    d_y = d_y.reshape(b, l, k).float()
    d_shift = d_y.sum(1, keepdim=True)
    d_scale = (d_y * xn).sum(1, keepdim=True)
    g = d_y * (1 + scale.float())
    d_x = rstd * (g - g.mean(-1, keepdim=True) - xn * (g * xn).mean(-1, keepdim=True))
    return d_x.to(x.dtype), d_shift.to(scale.dtype), d_scale.to(scale.dtype)


# The single block's linear2 input cat(attn, gelu(preact)) packed from its two legs by
# `a6w4_act_dual_legs_gelu`, so the bf16 cat is never written. Byte-identical only while the
# producer's bias is in its GEMM epilogue: otherwise Inductor folds that add into the GELU in f32
# and the materialised preact (rounded to bf16) is not what the GELU saw. 0 keeps the cat.
_A6W4_GELU_PACK = os.getenv("FLUX_A6W4_GELU_PACK", "1") == "1"
# The double block's fc2 input gelu(fc1(x)) packed the same way. fc1's bias is not in its epilogue
# (width 12288), so fc1 hands over its bias-free GEMM output and the pack adds the bias in f32
# ahead of the GELU, as Inductor's fused gelu(out + bias) does. Byte-identical.
_A6W4_GELU_PACK_DOUBLE = os.getenv("FLUX_A6W4_GELU_PACK_DOUBLE", "0") == "1"
# Single linear2's two legs (attn, gelu(preact + bias)) packed in one launch instead of one per
# leg: flydsl_quant_a6w4_act_dual_legs_merged, byte-identical to flydsl_quant_a6w4_act_dual_legs.
# Needs the mounted mxfp4_quant_kernel_flux.py.
_A6W4_LEGS_ONE_LAUNCH = os.getenv("FLUX_A6W4_LEGS_ONE_LAUNCH", "0") == "1"
_A6W4_LEGS_BIAS_WIDTH_LOGGED = False


def _legs_bias_unfit(preact: torch.Tensor, bias: torch.Tensor | None) -> str:
    """
Why flydsl_quant_a6w4_act_dual_legs would reject this bias, or "" if it would not.

It asserts `b.shape == (w,) and b.dtype is bfloat16 and b.is_contiguous()` on the GELU leg, as a bare assert with no values. _a6w4_legs_gelu_eligible checks the width ahead of time, but only the width, and it reads the producer's parameter while this sees the tensor that actually arrives.
"""
    if bias is None:
        return ""
    if preact.dim() != 2:
        return f"preact {tuple(preact.shape)} is not 2-D; the legs pack indexes rows and columns"
    w = int(preact.shape[-1])
    if tuple(bias.shape) != (w,):
        return f"bias {tuple(bias.shape)} is not the GELU leg's width ({w},) of {tuple(preact.shape)}"
    if bias.dtype != torch.bfloat16:
        return f"bias dtype {bias.dtype} is not bfloat16"
    if not bias.is_contiguous():
        return f"bias of shape {tuple(bias.shape)} is not contiguous (strides {bias.stride()})"
    return ""


_A6W4_LEGS_UNFIT_LOGGED = False


@torch.library.custom_op("primus_flux::a6w4_act_dual_legs_gelu", mutates_args=(), device_types="cuda")
def _a6w4_act_dual_legs_gelu_op(
    attn: torch.Tensor | None, preact: torch.Tensor, preact_bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`a6w4_act_dual(cat(attn, gelu(preact + preact_bias, "tanh")))` without the cat, the bias
    add being in f32; ``attn`` and ``preact_bias`` are optional, ``preact`` may be a column-slice
    view."""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        flydsl_quant_a6w4_act_dual,
        flydsl_quant_a6w4_act_dual_legs,
    )

    if preact.stride(1) != 1:
        preact = preact.contiguous()
    unfit = _legs_bias_unfit(preact, preact_bias)
    if unfit:
        # Materialise what the legs kernel would have fused. Same arithmetic in the same order -- bias in
        # f32, tanh GELU, back to bf16 for the cat -- so this costs the write it exists to avoid and
        # nothing else.
        global _A6W4_LEGS_UNFIT_LOGGED
        if not _A6W4_LEGS_UNFIT_LOGGED:
            _A6W4_LEGS_UNFIT_LOGGED = True
            logger.info(f"A6W4_LEGS_GELU declined at call time: {unfit}; materialising the cat")
        if tuple(preact_bias.shape) != (int(preact.shape[-1]),):
            # Not a fusion question any more: nothing here can apply a bias of the wrong width, and adding it
            # under broadcasting would be silently wrong rather than slow.
            raise RuntimeError(
                f"a6w4 legs GELU got a bias {tuple(preact_bias.shape)} for a preact "
                f"{tuple(preact.shape)}; they are not the same Linear's. Set "
                f"FLUX_A6W4_GELU_PACK=0 to keep the materialised cat while this is fixed."
            )
        act = torch.nn.functional.gelu(preact.float() + preact_bias.float(), approximate="tanh").to(
            preact.dtype
        )
        x = act if attn is None else torch.cat((attn, act), dim=1)
        return flydsl_quant_a6w4_act_dual(x.contiguous())
    if attn is None:
        return flydsl_quant_a6w4_act_dual_legs((preact,), (True,), (preact_bias,))
    if attn.stride(1) != 1:
        attn = attn.contiguous()
    if _A6W4_LEGS_ONE_LAUNCH:
        from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
            flydsl_quant_a6w4_act_dual_legs_merged,
        )

        return flydsl_quant_a6w4_act_dual_legs_merged((attn, preact), (False, True), (None, preact_bias))
    return flydsl_quant_a6w4_act_dual_legs((attn, preact), (False, True), (None, preact_bias))


@_a6w4_act_dual_legs_gelu_op.register_fake
def _(attn, preact, preact_bias=None):
    m, k = preact.shape[0], (0 if attn is None else attn.shape[1]) + preact.shape[1]
    aq = torch.empty(m, k, dtype=torch.uint8, device=preact.device)
    sa = torch.empty(m * k // MX_BLOCK_SIZE, dtype=torch.uint8, device=preact.device)
    co = torch.empty(k, m // 2, dtype=torch.uint8, device=preact.device).view(FP4_DTYPE)
    cs = torch.empty(k, m // MX_BLOCK_SIZE, dtype=torch.uint8, device=preact.device).view(
        torch.float8_e8m0fnu
    )
    return aq, sa, co, cs


def _a6w4_legs_gelu_eligible(producer, consumer, rows: int, widths) -> bool:
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        a6w4_act_dual_legs_eligible,
    )

    cfg = consumer.config
    if len(widths) == 2:
        producer_ok = (
            _A6W4_GELU_PACK
            and producer.bias is not None
            and _fuse_bias_eligible(producer.config, producer.out_features)
        )
    else:
        producer_ok = (
            len(widths) == 1
            and _A6W4_GELU_PACK_DOUBLE
            and producer.config.fprop == "a6w4"
            and (producer.bias is None or producer.bias.dtype == torch.bfloat16)
        )
    # The pack asserts that a lazy bias is exactly the GELU leg's width -- in
    # mxfp4_quant_kernel.flydsl_quant_a6w4_act_dual_legs, `b.shape == (w,)`. Check that here, where
    # returning False falls back to the materialised cat, instead of letting it reach an assert inside
    # a custom op.
    lazy_bias = producer.lazy_gelu_bias()
    if lazy_bias is not None and tuple(lazy_bias.shape) != (int(widths[-1]),):
        global _A6W4_LEGS_BIAS_WIDTH_LOGGED
        if not _A6W4_LEGS_BIAS_WIDTH_LOGGED:
            _A6W4_LEGS_BIAS_WIDTH_LOGGED = True
            logger.info(
                f"A6W4_LEGS_GELU declined: producer lazy bias {tuple(lazy_bias.shape)} is not "
                f"the GELU leg's width {int(widths[-1])}, widths={tuple(int(w) for w in widths)}; "
                f"falling back to the materialised cat"
            )
        return False
    return (
        producer_ok
        and _A6W4_ACT_DUAL
        and consumer.training
        and cfg.fprop == "a6w4"
        and cfg.wgrad == "mxfp4"
        and cfg.save_quantized
        and _dual_pack_eligible(consumer.hadamard, cfg)
        and not cfg.sr_op("act", "wgrad")
        and a6w4_act_dual_legs_eligible(rows, widths)
    )


@torch.library.custom_op("primus_flux::a6w4_mm_qa", mutates_args=(), device_types="cuda")
def _a6w4_mm_qa_op(
    aq: torch.Tensor, sa: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """`a6w4_mm` on an activation `a6w4_act_dual` already quantized."""
    from primus_turbo.pytorch.ops.flux_a6w4 import a6w4_mm_qa

    return a6w4_mm_qa(aq, sa, b, bias)


@_a6w4_mm_qa_op.register_fake
def _(aq, sa, b, bias=None):
    return torch.empty(aq.shape[0], b.shape[0], device=aq.device, dtype=torch.bfloat16)


@torch.library.custom_op("primus_flux::a6w4_mm_qa_prequant_w", mutates_args=(), device_types="cuda")
def _a6w4_mm_qa_prequant_w_op(
    aq: torch.Tensor,
    sa: torch.Tensor,
    wq: torch.Tensor,
    sb: torch.Tensor,
    n: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from primus_turbo.pytorch.ops.flux_a6w4 import a6w4_mm_qa_prequant_w

    return a6w4_mm_qa_prequant_w(aq, sa, wq, sb, n, bias)


@_a6w4_mm_qa_prequant_w_op.register_fake
def _(aq, sa, wq, sb, n, bias=None):
    return torch.empty(aq.shape[0], n, device=aq.device, dtype=torch.bfloat16)


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
    a_quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
    b_quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
    bias: torch.Tensor | None = None,
    packed_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
Rotate, quantize and multiply a[M,K] @ b[N,K]^T along the shared K.

``packed_scales`` (with both operands pre-quantized): see ``_mxfp4_mm``.

``a_quantized``, if given, is an already-quantized ``(a_q, a_s)`` pair -- e.g. the row half of a fused dual pack the caller computed once because it also needed the col half for something else -- used instead of quantizing ``a`` here. Left as ``None`` this is exactly the previous behaviour. The opaque ``mxfp4_mm`` single-op fast path re-quantizes both operands from scratch, so it cannot consume a pre-quantized ``a``; skip it (fall through to the explicit-quantize route below) whenever ``a_quantized`` is supplied.

``b_quantized`` is the same idea for ``b``: e.g. weight's Dgrad col pack, computed directly from the UNTRANSPOSED weight via ``_quantize_mxfp4_h16_col_op`` instead of transposing first and packing that. When supplied, ``b``'s own value is never read (neither the ``mxfp4_mm`` fast path nor ``_quantize_operand(b,...)`` runs), so the caller may pass the untransposed tensor as ``b`` in that case -- exactly what the Dgrad call site below does.

``bias`` also disables the opaque ``mxfp4_mm`` fast path: that single-op custom op has no bias epilogue of its own, so a bias request always takes the explicit-quantize route below, which does.
"""
    if (
        _USE_MXFP4_MM
        and a_quantized is None
        and b_quantized is None
        and not a_sr
        and not b_sr
        and not b_2d
        and not cfg.randomized_hadamard
        and bias is None
    ):
        return torch.ops.primus_flux.mxfp4_mm(a, b)
    if a_quantized is not None:
        a_q, a_s = a_quantized
    else:
        a_q, a_s = _quantize_operand(a, hadamard, False, a_sr, cfg.randomized_hadamard)
    if b_quantized is not None:
        b_q, b_s = b_quantized
    else:
        b_q, b_s = _quantize_operand(b, hadamard, b_2d, b_sr, cfg.randomized_hadamard)
    if a_quantized is None or b_quantized is None:
        packed_scales = None
    return _mxfp4_mm(a_q, a_s, b_q, b_s, out_dtype, bias=bias, packed_scales=packed_scales)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


class _MXFP4LinearFunction(torch.autograd.Function):
    """
y = x @ w^T (+ bias) with each pass independently castable to MXFP4.

Every pass is expressed as ``A[M,K] @ B[N,K]^T`` so one quantize-and-multiply helper serves all three. The transposes that normalise them to that form are also what lets each pass carry its own rotation: Fprop contracts over the input features, Dgrad over the output features and Wgrad over the tokens, so a rotation that cancels for one does nothing for the others.

Weights are re-quantized in backward rather than stashed -- FSDP2 runs with ``reshard_after_forward=false``, so the BF16 parameter is still resident and holding a reference to it is free, whereas a saved FP4 copy would not be.

That FP4 col pack is ~4-8x smaller than a BF16 copy of the weight (0.5 B/elem data + ~0.03 B/elem E8M0 scale vs 2 B/elem), and it replaces a full second read+relaunch with zero extra reads -- so the "saved FP4 copy would not be free" caution above is about a BF16-sized stash, not this much smaller, already-being-computed FP4 one.
"""

    @staticmethod
    def forward(ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg):
        out, save_list = _MXFP4LinearFunction._forward(
            ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg
        )
        ctx.save_for_backward(*save_list)
        return out.reshape(*ctx.x_shape[:-1], weight.shape[0])

    @staticmethod
    def _forward(
        ctx,
        x,
        weight,
        bias,
        hadamard,
        act_slot,
        wgt_slot,
        cfg,
        bias_any_width=False,
        x_pre=None,
        return_unbiased=False,
        gelu_dual=False,
    ):
        """Everything ``forward`` does except ``save_for_backward``, so a caller can save
        extra tensors after these. Returns the 2D output, bias applied, and the save list.
        ``bias_any_width`` is `_fuse_bias_eligible`'s ``any_width``.

        ``x_pre`` is ``(x_fp8_dual, x_shape, x_dtype)`` from a caller that already packed an
        activation it never materialised; ``x`` is then None. Only the FP8 and A6W4 dual-pack
        paths consume it, so callers must have checked `_fp8_legs_gelu_eligible` or
        `_a6w4_legs_gelu_eligible`."""
        if x_pre is not None:
            x_2d = None
            out_dtype = x_pre[2]
        else:
            x_2d = x.reshape(-1, x.shape[-1])
            out_dtype = x.dtype

        # Fprop packs x_2d row-major (along K, its own last dim); the Wgrad-prep pack below packs x_2d.t()
        # (along tokens). Whenever both are needed and both would independently resolve to the fused-H16
        # path (see `_dual_pack_eligible`), get both from ONE `_quantize_mxfp4_h16_dual_op` read instead
        # of a `.t().contiguous()` transpose copy feeding a second, independent pack launch.
        fuse_x = (
            x_pre is None
            and cfg.fprop == "mxfp4"
            and cfg.wgrad == "mxfp4"
            and cfg.save_quantized
            and _dual_pack_eligible(hadamard, cfg)
        )
        # Row half feeds Fprop, col half feeds Wgrad, so each takes its own
        # stochastic-rounding setting. Both halves are the activation, and the two
        # consuming passes want different rounding for it -- "FP4 All the Way"
        # keeps Wgrad's activation on SR and Fprop's on round-to-nearest -- which
        # this split already expresses at no extra cost, since it is one launch.
        x_dual = (
            _quantize_mxfp4_h16_dual_op(x_2d, cfg.sr_op("act", "fprop"), cfg.sr_op("act", "wgrad"))
            if fuse_x
            else None
        )

        # The FP8 twin of `fuse_x` above: the same one-read-two-orientations trick, with the row
        # half cast to FP8 for Fprop instead of packed to MXFP4. `save_quantized` belongs in this
        # gate for exactly the reason it is in `fuse_x`'s -- without it there is no col pack for
        # the row cast to share a read with, and the fusion has nothing to fuse.
        # This is the branch that removes the FP8 fprop's three-reads-of-one-tensor: the amax
        # reduction and the row cast that `_fp8_tensorwise_mm_op` does internally, plus the col
        # pack the `elif` further down would launch on its own. See `_fp8_mxfp4_dual_op`.
        if x_pre is not None:
            x_fp8_dual = x_pre[0] if cfg.fprop == "fp8" else None
        else:
            fuse_x_fp8 = (
                _FP8_DUAL_PACK
                and cfg.fprop == "fp8"
                and cfg.wgrad == "mxfp4"
                and cfg.save_quantized
                and act_slot is not None
                and _fp8_dual_pack_eligible(x_2d, hadamard, cfg, "act", "wgrad")
            )
            x_fp8_dual = _fp8_mxfp4_dual_op(x_2d, act_slot) if fuse_x_fp8 else None

        # Both orientations always use the SAME (use_2d=cfg.weight_2d_block, use_sr=False,
        # use_rht=cfg.randomized_hadamard) args (`_weight_colpack_eligible`'s own docstring establishes
        # this), so whenever that resolves to the fused-H16 path AND both consuming passes are MXFP4, get
        # both packs from ONE `_quantize_mxfp4_h16_dual_op` read here (mirrors `fuse_x` above exactly) and
        # carry the col half to backward via `ctx.save_for_backward` (Rule 11 bucket K4 -- same-call data
        # flow, not an id()-keyed cache) instead of a second, independent read.
        fuse_w = cfg.fprop == "mxfp4" and cfg.dgrad == "mxfp4" and _weight_colpack_eligible(hadamard, cfg)
        # Row half feeds Fprop, col half feeds Dgrad, so each carries that pass's
        # stochastic-rounding setting -- the same row/col-to-pass mapping `fuse_x`
        # uses, just with Dgrad on the col side instead of Wgrad. This fusion is
        # also what makes weight SR cheap: both orientations now come out of one
        # read, so turning SR on costs nothing extra beyond the convert itself.
        w_dual = (
            _quantize_mxfp4_h16_dual_op(weight, cfg.sr_op("weight", "fprop"), cfg.sr_op("weight", "dgrad"))
            if fuse_w
            else None
        )

        # The FP8 twin of `fuse_w`, and the reason it is needed: `fuse_w` above requires `cfg.fprop ==
        # "mxfp4"`, so on an FP8 forward the weight gets NO fusion at all today and is read three times
        # per step -- amax, FP8 row cast, MXFP4 Dgrad col pack. The weight is the LARGER of the two
        # operands on FLUX's fprop shapes (21504x3072 against a 16384x3072 activation), so this half is
        # worth more than the activation half, not less.
        # Everything downstream of it already exists: backward reuses a saved weight col pack whenever
        # `ctx.w_is_dual`, and that reuse is orientation- and SR-correct for Dgrad.
        # Not when the weight arrived from the FP8 all-gather. `_FP8GatheredWeight` already holds both
        # halves this fusion would produce -- it intercepts the fprop GEMM into
        # `fp8_tensorwise_mm_prequant_w` using its own `_fp8`/`_scale_inv`, and answers the Dgrad col pack
        # from `_dgrad_col_pack()`. Packing again would redo, on the full matrix, work the gather already
        # did on a shard, so this is `world` times the cost for a bit-identical result. Without the guard
        # it does not merely waste time: the subclass raises NotImplementedError on `fp8_mxfp4_dual` and
        # Dynamo fails the graph, which is how turning both features on at once was found to abort at step
        # 0.
        fuse_w_fp8 = (
            _FP8_DUAL_PACK
            and cfg.fprop == "fp8"
            and cfg.dgrad == "mxfp4"
            and wgt_slot is not None
            and not isinstance(weight, _FP8GatheredWeight)
            and _fp8_dual_pack_eligible(weight, hadamard, cfg, "weight", "dgrad")
        )
        w_fp8_dual = _fp8_mxfp4_dual_op(weight, wgt_slot) if fuse_w_fp8 else None

        if x_pre is not None:
            x_a6w4_dual = x_pre[0] if cfg.fprop == "a6w4" else None
        else:
            x_a6w4_dual = _a6w4_act_dual_op(x_2d) if _a6w4_act_dual_ok(x_2d, cfg, hadamard) else None

        # Computed before the GEMM so the fused-bias request can be threaded straight into
        # _quantize_and_mm below.
        fuse_bias = bias is not None and _fuse_bias_eligible(cfg, weight.shape[0], bias_any_width)

        if cfg.fprop == "mxfp4":
            out = _quantize_and_mm(
                x_2d,
                weight,
                cfg,
                hadamard,
                out_dtype,
                a_sr=cfg.sr_op("act", "fprop"),
                b_sr=cfg.sr_op("weight", "fprop"),
                b_2d=cfg.weight_2d_block,
                a_quantized=x_dual[0:2] if x_dual is not None else None,
                b_quantized=w_dual[0:2] if w_dual is not None else None,
                bias=bias if fuse_bias else None,
            )
        elif cfg.fprop == "fp8":
            if x_fp8_dual is not None or w_fp8_dual is not None:
                # Either operand may be pre-quantized independently -- the two fusions are gated on different
                # passes (Wgrad for x, Dgrad for w) and one can be eligible alone.
                if x_fp8_dual is not None:
                    a_fp8, a_si = x_fp8_dual[0], x_fp8_dual[1]
                else:
                    from primus_turbo.pytorch.core.low_precision import (
                        ScalingGranularity,
                    )
                    from primus_turbo.pytorch.ops.quantization import quantize_fp8

                    a_fp8, a_si = quantize_fp8(
                        x_2d.contiguous(), torch.float8_e4m3fn, ScalingGranularity.TENSORWISE
                    )
                if w_fp8_dual is not None:
                    out = _fp8_tensorwise_mm_aq_op(
                        a_fp8,
                        a_si,
                        w_fp8_dual[0],
                        out_dtype,
                        w_fp8_dual[1],
                        bias if fuse_bias else None,
                    )
                else:
                    out = _fp8_tensorwise_mm_aq_op(
                        a_fp8, a_si, weight, out_dtype, None, bias if fuse_bias else None
                    )
            else:
                out = _fp8_tensorwise_mm_op(x_2d, weight)
                if fuse_bias:
                    out = out + bias
        elif cfg.fprop == "mxfp6":
            if _ACT_STATS:
                _record_act_stats(x_2d, weight)
            out = _mxfp6_mm_op(x_2d, weight, bias if fuse_bias else None)
        elif cfg.fprop == "a6w4":
            fuse_gelu = (
                gelu_dual
                and bias is not None
                and x_a6w4_dual is not None
                and _a6w4_gelu_dual_ok(
                    x_a6w4_dual[0].shape[0],
                    weight.shape[0],
                    weight.shape[1],
                    cfg,
                    bias,
                    0 if x_a6w4_dual[0].shape[-1] == weight.shape[1] else 1,
                )
            )
            if fuse_gelu:
                out, *packs = _a6w4_mm_gelu_dual_qa_w_op(
                    x_a6w4_dual[0], x_a6w4_dual[1], weight, bias, fuse_bias
                )
                ctx.gelu_dual_packs = tuple(packs)
                fuse_bias = True
            elif x_a6w4_dual is not None:
                out = _a6w4_mm_qa_op(x_a6w4_dual[0], x_a6w4_dual[1], weight, bias if fuse_bias else None)
            else:
                out = _a6w4_mm_op(x_2d, weight, bias if fuse_bias else None)
        else:
            out = torch.matmul(x_2d, weight.t())

        # Wgrad contracts over tokens, so its operand must be rotated along the
        # token dim -- a rotation the forward pass never forms. Build it here so
        # only the 4-bit copy has to live until backward.
        if x_dual is not None:
            x_q, x_s = x_dual[2], x_dual[3]
            save_list = [x_q.view(torch.uint8), x_s, weight, hadamard]
            ctx.x_is_quantized = True
        elif x_a6w4_dual is not None:
            x_q, x_s = x_a6w4_dual[2], x_a6w4_dual[3]
            save_list = [x_q.view(torch.uint8), x_s, weight, hadamard]
            ctx.x_is_quantized = True
        elif x_fp8_dual is not None:
            # Col half of the very read the FP8 row cast came from, so the standalone col-pack launch in the
            # branch below never happens. Bit-identical to what that branch would have produced --
            # `_fp8_dual_pack_eligible` refuses the fusion in any case where it would not be (SR on either
            # half, or a shape the kernel's tiling does not cover).
            x_q, x_s = x_fp8_dual[2], x_fp8_dual[3]
            save_list = [x_q.view(torch.uint8), x_s, weight, hadamard]
            ctx.x_is_quantized = True
        elif cfg.wgrad == "mxfp4" and cfg.save_quantized and _dual_pack_eligible(hadamard, cfg):
            # A non-MXFP4 Fprop (fp8, mxfp6) leaves no row pack for Wgrad's col pack to share a read with, but
            # the col packer still produces that col orientation from one coalesced read of `x_2d` -- bit-
            # identical to packing `x_2d.t().contiguous()`, without materializing the transpose.
            # `_dual_pack_eligible` is the right gate unchanged: it already asks the col-orientation question
            # for this operand with use_2d=False. Without this the fallback below pays a full activation-sized
            # HBM round trip per Linear per step purely because Fprop changed format.
            x_q, x_s = _quantize_mxfp4_h16_col_op(x_2d, cfg.sr_op("act", "wgrad"))
            save_list = [x_q.view(torch.uint8), x_s, weight, hadamard]
            ctx.x_is_quantized = True
        elif cfg.wgrad == "mxfp4" and cfg.save_quantized:
            x_q, x_s = _quantize_operand(
                x_2d.t().contiguous(),
                hadamard,
                False,
                cfg.sr_op("act", "wgrad"),
                cfg.randomized_hadamard,
            )
            save_list = [x_q.view(torch.uint8), x_s, weight, hadamard]
            ctx.x_is_quantized = True
        else:
            save_list = [x_2d, weight, hadamard]
            ctx.x_is_quantized = False

        # `.view(torch.uint8)` mirrors the x_q treatment above -- same reason (the FP4 view dtype survives
        # a plain attribute store but not every autograd-internal tensor op, so store it as the safe
        # view). The FP8 weight dual lands in the same fixed-position suffix as the MXFP4 one: its col
        # half is the same orientation, packed by the same code, for the same consumer. Only the row half
        # differs, and that was consumed by the GEMM above.
        _w_dual_any = w_dual if w_dual is not None else w_fp8_dual
        ctx.w_is_dual = _w_dual_any is not None
        if ctx.w_is_dual:
            w_col_q, w_col_s = _w_dual_any[2], _w_dual_any[3]
            save_list = save_list + [w_col_q.view(torch.uint8), w_col_s]

        ctx.cfg = cfg
        ctx.gelu_dual_packs = getattr(ctx, "gelu_dual_packs", None)
        ctx.x_shape = tuple(x.shape) if x_pre is None else tuple(x_pre[1])
        ctx.out_dtype = out_dtype
        ctx.has_bias = bias is not None

        unbiased = out
        if bias is not None and not fuse_bias:
            out = out + bias
        if return_unbiased:
            return out, save_list, unbiased
        return out, save_list

    @staticmethod
    def backward(ctx, grad_out):
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        return _MXFP4LinearFunction._backward(ctx, ctx.saved_tensors, grad_2d)

    @staticmethod
    def _backward(ctx, saved, grad_2d, g_packs=None, gate=None):
        """``backward`` on the tensors ``_forward`` saved. ``g_packs`` is G's
        (row, row_scale, col, col_scale, bias) when the caller already packed it; the
        bf16 ``grad_2d`` is then never read and may be None.

        ``gate`` is ``(gate [B, N], y [M, N], y_bias [N])`` for ``gate * (y + y_bias)``:
        ``grad_2d`` is then dY rather than G, and the gate's gradient is appended to the
        returned tuple."""
        cfg: MXFP4LinearConfig = ctx.cfg
        out_dtype = ctx.out_dtype

        if ctx.w_is_dual:
            w_col_q, w_col_s = saved[-2], saved[-1]
            saved = saved[:-2]
        else:
            w_col_q = w_col_s = None

        if ctx.x_is_quantized:
            x_q, x_s, weight, hadamard = saved
            x_q = x_q.view(FP4_DTYPE)
            x_2d = None
        else:
            x_2d, weight, hadamard = saved
            x_q = x_s = None

        # Dgrad packs grad_2d row-major (along N, its own last dim); Wgrad packs grad_2d.t() (along
        # tokens). Whenever both passes are MXFP4 and both would independently resolve to the fused-H16
        # path (see `_dual_pack_eligible`), get both from ONE `_quantize_mxfp4_h16_dual_op` read instead
        # of a transpose copy feeding a second pack launch.
        # When the module also has a bias, `grad_bias = grad_2d.sum(0)` below would be a THIRD full read
        # of this exact tensor -- the dual's col phase already holds every row of each column microblock
        # in registers, so folding a running sum into that phase makes the bias gradient a byproduct of
        # the pack read instead of another independent one. Gated on `ctx.has_bias` too so a bias-less
        # Linear never pays the fused kernel's (small) extra col-phase cost for a sum nobody wants.
        fuse_g = cfg.dgrad == "mxfp4" and cfg.wgrad == "mxfp4" and _dual_pack_eligible(hadamard, cfg)
        # The fused-bias pack kernels (`flydsl_quant_mxfp4_h16_dual_bias` and its legs variant) have no SR
        # path -- unlike the plain dual, they take no row_sr/col_sr. So when either consuming pass wants
        # stochastic rounding, skip the bias fusion and fall back to the plain dual plus an explicit
        # `grad_2d.sum(0)` below. That trades P6 back for correctness on biased modules only, and only
        # while SR is on; Threading row_sr/col_sr through the bias kernels would recover it.
        fuse_g_bias = (
            fuse_g and ctx.has_bias and not (cfg.sr_op("grad", "dgrad") or cfg.sr_op("grad", "wgrad"))
        )
        # Weight's OWN row pack (Fprop, above) is untouched -- only this second, transposed orientation is
        # replaced: whenever it would resolve to the fused-H16 path (see `_weight_colpack_eligible`), get
        # it directly from the UNTRANSPOSED weight via `_quantize_mxfp4_h16_col_op` (one coalesced read,
        # no transpose copy) instead of transpose-then-pack.
        # `ctx.w_is_dual` now has two sources and the guarantee holds for both. An MXFP4 fprop sets it
        # from `fuse_w`, which gates on `_weight_colpack_eligible` directly. An FP8 fprop sets it from
        # `fuse_w_fp8`, which gates on `_fp8_dual_pack_eligible(..., "weight", "dgrad")`; that requires
        # `_dual_pack_eligible` AND `not cfg.weight_2d_block`, and with 2D off those two predicates are
        # the same call -- `_fused_h16_quant_enabled(hadamard, False, cfg.randomized_hadamard)`. Either
        # way the col half was packed for Dgrad with `cfg.sr_op("weight", "dgrad")`, which is exactly what
        # this call site would ask for, so the reuse is SR-correct rather than SR-losing.
        if ctx.w_is_dual:
            w_col = (w_col_q.view(FP4_DTYPE), w_col_s)
        else:
            w_col = (
                _quantize_mxfp4_h16_col_op(weight, cfg.sr_op("weight", "dgrad"))
                if _weight_colpack_eligible(hadamard, cfg)
                else None
            )
        packed = None
        pk_dgrad = pk_wgrad = False
        grad_gate = None
        gate_fused = False
        if gate is not None:
            gate_2d, y_2d, y_bias = gate
            gate_fused = fuse_g_bias and _gate_dgate_ok(
                grad_2d, gate_2d, y_2d, y_bias, x_s, w_col[1] if w_col is not None else None
            )
            if not gate_fused:
                grad_2d, grad_gate = _gate_dgate_unfused(grad_2d, gate_2d, y_2d, y_bias)
        if g_packs is not None:
            g_dual = tuple(g_packs[:4])
            grad_bias = g_packs[4]
            fuse_g_bias = True
        elif gate_fused:
            g_out = _quantize_mxfp4_h16_dual_bias_gate_op(grad_2d, gate_2d, y_2d, y_bias, x_s, w_col[1])
            g_dual = tuple(g_out[:4])
            grad_bias = g_out[4]
            packed = list(g_out[5:9])
            pk_dgrad, pk_wgrad = _packed_gemm_flags(grad_2d, x_s)
            grad_gate = g_out[9]
        elif fuse_g_bias and _dual_bias_packed_ok(grad_2d, x_s, w_col[1] if w_col is not None else None):
            _legs = _legpack_g_split(*grad_2d.shape)
            g_row, g_row_scale, g_col, g_col_scale, grad_bias, *packed = (
                _quantize_mxfp4_h16_dual_bias_packed_op(grad_2d, _legs[0] if _legs else 0, x_s, w_col[1])
            )
            g_dual = (g_row, g_row_scale, g_col, g_col_scale)
            pk_dgrad, pk_wgrad = _packed_gemm_flags(grad_2d, x_s)
        elif fuse_g_bias:
            # Every other shape keeps the existing, unmodified monolithic call.
            _legs = _legpack_g_split(*grad_2d.shape)
            if _legs is not None:
                g_row, g_row_scale, g_col, g_col_scale, grad_bias = _quantize_mxfp4_h16_dual_bias_legs_op(
                    grad_2d, _legs[0]
                )
            else:
                g_row, g_row_scale, g_col, g_col_scale, grad_bias = _quantize_mxfp4_h16_dual_bias_op(grad_2d)
            g_dual = (g_row, g_row_scale, g_col, g_col_scale)
        else:
            # Row half feeds Dgrad, col half feeds Wgrad.
            g_dual = (
                _quantize_mxfp4_h16_dual_op(grad_2d, cfg.sr_op("grad", "dgrad"), cfg.sr_op("grad", "wgrad"))
                if fuse_g
                else None
            )
            grad_bias = None

        # Dgrad: dX = G @ W, contracting over the output features N.
        # A = G[M,N]; B = W^T[K,N]; A @ B^T -> [M,K].
        if cfg.dgrad == "mxfp4":
            grad_input = _quantize_and_mm(
                grad_2d,
                weight if w_col is not None else weight.t().contiguous(),
                cfg,
                hadamard,
                out_dtype,
                a_sr=cfg.sr_op("grad", "dgrad"),
                b_sr=cfg.sr_op("weight", "dgrad"),
                b_2d=cfg.weight_2d_block,
                a_quantized=g_dual[0:2] if g_dual is not None else None,
                b_quantized=w_col,
                packed_scales=(packed[0], packed[3]) if packed and pk_dgrad else None,
            )
        else:
            grad_input = torch.matmul(grad_2d, weight)

        # Wgrad: dW = G^T @ X, contracting over the tokens M.
        # A = G^T[N,M]; B = X^T[K,M]; A @ B^T -> [N,K].
        if cfg.wgrad == "mxfp4":
            if g_dual is not None:
                g_q, g_s = g_dual[2], g_dual[3]
            else:
                g_q, g_s = _quantize_operand(
                    grad_2d.t().contiguous(),
                    hadamard,
                    False,
                    cfg.sr_op("grad", "wgrad"),
                    cfg.randomized_hadamard,
                )
            if x_q is None:
                x_q, x_s = _quantize_operand(
                    x_2d.t().contiguous(),
                    hadamard,
                    False,
                    cfg.sr_op("act", "wgrad"),
                    cfg.randomized_hadamard,
                )
            grad_weight = _mxfp4_mm(
                g_q,
                g_s,
                x_q,
                x_s,
                out_dtype,
                packed_scales=(packed[1], packed[2]) if packed and pk_wgrad else None,
            )
        else:
            grad_weight = torch.matmul(grad_2d.t(), x_2d)

        if not fuse_g_bias:
            grad_bias = grad_2d.sum(0) if ctx.has_bias else None
        # x, weight, bias, hadamard, act_slot, wgt_slot, cfg
        grads = (grad_input.reshape(ctx.x_shape), grad_weight, grad_bias, None, None, None, None)
        return grads if gate is None else (*grads, grad_gate)


def _fp8_legs_gelu_eligible(module, rows: int, widths) -> bool:
    """True iff ``module`` (the consuming Linear) would take `_fp8_mxfp4_dual_op` on the
    materialised cat, with the delayed kernel, so the fused legs pack replaces exactly that."""
    from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
        fp8_dual_legs_eligible,
    )

    cfg = module.config
    return (
        _FWD_GELU_PACK
        and _FP8_DUAL_PACK
        and _FP8_DUAL_DELAYED
        and module.training
        and cfg.any_mxfp4
        and cfg.fprop == "fp8"
        and cfg.wgrad == "mxfp4"
        and cfg.save_quantized
        and module.fp8_act_slot is not None
        and _dual_pack_eligible(module.hadamard, cfg)
        and not cfg.sr_op("act", "wgrad")
        and not cfg.sr_op("act", "fprop")
        and fp8_dual_legs_eligible(rows, widths)
    )


class LazyAct(NamedTuple):
    """A lazy act plus the operands its producer's epilogue already built from it.

    ``dual`` is ``a6w4_act_dual(gelu(act))``, so a consumer holding one has nothing to read
    back; it is None when the producer could not fuse, and then ``act`` behaves exactly as
    the bare tensor `forward_gelu(lazy=True)` used to return. ``rebiased`` says the producer
    moved its bias into the epilogue to fuse, so ``act`` carries it and only the packs
    describe what the consumer wanted -- declining them is a wiring fault, not a fallback."""

    act: torch.Tensor
    dual: tuple[torch.Tensor, ...] | None = None
    rebiased: bool = False


class _MXFP4LinearGeluInFunction(torch.autograd.Function):
    """``_MXFP4LinearFunction(cat(attn, gelu(preact)))`` -- or of ``gelu(preact)`` when
    ``attn`` is None -- where ``preact`` is a lazy act. Fprop and the saved Wgrad col pack
    both come from `_fp8_mxfp4_dual_legs_gelu_op`, so the bf16 input is never built. Backward
    returns d(attn) and d(act) as column views of Dgrad's output; the producer of ``preact``
    applies dGELU."""

    @staticmethod
    def forward(
        ctx, attn, preact, weight, bias, hadamard, act_slot, wgt_slot, cfg, preact_bias=None, ready=None
    ):
        lead = preact.shape[:-1]
        a_2d = None if attn is None else attn.reshape(-1, attn.shape[-1])
        p_2d = preact.reshape(-1, preact.shape[-1])
        if ready is not None:
            packs = ready
        elif cfg.fprop == "a6w4":
            packs = _a6w4_act_dual_legs_gelu_op(a_2d, p_2d, preact_bias)
        else:
            packs = _fp8_mxfp4_dual_legs_gelu_op(a_2d, p_2d, act_slot)
        leg0 = 0 if attn is None else attn.shape[-1]
        x_shape = (*lead, leg0 + preact.shape[-1])
        out, save_list = _MXFP4LinearFunction._forward(
            ctx,
            None,
            weight,
            bias,
            hadamard,
            act_slot,
            wgt_slot,
            cfg,
            x_pre=(packs, x_shape, preact.dtype),
        )
        ctx.save_for_backward(*save_list)
        ctx.leg0 = leg0
        return out.reshape(*lead, weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grads = _MXFP4LinearFunction._backward(ctx, ctx.saved_tensors, grad_2d)
        d_x = grads[0]
        if ctx.leg0:
            d_attn, d_act = d_x[..., : ctx.leg0], d_x[..., ctx.leg0 :]
        else:
            d_attn, d_act = None, d_x
        return d_attn, d_act, grads[1], grads[2], None, None, None, None, None, None


def _gated_out(ctx, mm, gate, bias, lead, n):
    """``gate * (mm + bias)``, with the unbiased GEMM output, gate and bias saved for the
    fused backward. The add and multiply stay in the traced graph so Inductor fuses them
    into the residual add, as it does for ``gate * linear(h)``."""
    ctx.has_bias = True
    return gate * (mm.reshape(*lead, n) + bias)


def _gated_backward(ctx, grad_out):
    """`_MXFP4LinearFunction._backward` for a gated forward: returns its grads and dgate in
    the gate's own shape."""
    saved = ctx.saved_tensors
    mm, gate, bias = saved[-3:]
    grad_2d = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
    gate_2d = gate.reshape(gate.shape[0], gate.shape[-1])
    grads = _MXFP4LinearFunction._backward(ctx, saved[:-3], grad_2d, gate=(gate_2d, mm, bias))
    return grads[:-1], grads[-1].view(gate.shape)


class _MXFP4GatedLinearFunction(torch.autograd.Function):
    """``gate * _MXFP4LinearFunction(x)`` for a gated residual ``x + gate * linear(h)``, with
    ``gate`` ``[B, 1, N]`` (FLUX_GATE_DGATE). Backward receives dY instead of
    ``gate * dY``, so the gate multiply and dgate's re-read of dY fold into G's pack."""

    @staticmethod
    def forward(ctx, x, gate, weight, bias, hadamard, act_slot, wgt_slot, cfg):
        mm, save_list = _MXFP4LinearFunction._forward(ctx, x, weight, None, hadamard, act_slot, wgt_slot, cfg)
        ctx.save_for_backward(*save_list, mm, gate, bias)
        return _gated_out(ctx, mm, gate, bias, ctx.x_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        grads, grad_gate = _gated_backward(ctx, grad_out)
        return grads[0], grad_gate, grads[1], grads[2], None, None, None, None


class _MXFP4GatedLinearGeluInFunction(torch.autograd.Function):
    """``gate * _MXFP4LinearGeluInFunction(attn, preact)``: the gated form of the lazy-GELU
    consumer (single-block linear2, double-block MLP fc2)."""

    @staticmethod
    def forward(ctx, attn, preact, gate, weight, bias, hadamard, act_slot, wgt_slot, cfg, ready=None):
        lead = preact.shape[:-1]
        a_2d = None if attn is None else attn.reshape(-1, attn.shape[-1])
        p_2d = preact.reshape(-1, preact.shape[-1])
        # Dispatch on the fprop cast exactly as the non-gated twin does. This branch was absent,
        # so a gated lazy-GELU consumer on A6W4 silently took the FP8 pack -- a different element
        # format for the same activation. It could not be noticed while the gate flag also swapped
        # the whole overlay for a copy with no A6W4 code in it, because the two never ran together.
        if ready is not None:
            packs = ready
        elif cfg.fprop == "a6w4":
            packs = _a6w4_act_dual_legs_gelu_op(a_2d, p_2d, None)
        else:
            packs = _fp8_mxfp4_dual_legs_gelu_op(a_2d, p_2d, act_slot)
        leg0 = 0 if attn is None else attn.shape[-1]
        x_shape = (*lead, leg0 + preact.shape[-1])
        mm, save_list = _MXFP4LinearFunction._forward(
            ctx,
            None,
            weight,
            None,
            hadamard,
            act_slot,
            wgt_slot,
            cfg,
            x_pre=(packs, x_shape, preact.dtype),
        )
        ctx.save_for_backward(*save_list, mm, gate, bias)
        ctx.leg0 = leg0
        return _gated_out(ctx, mm, gate, bias, lead, weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        grads, grad_gate = _gated_backward(ctx, grad_out)
        d_x = grads[0]
        if ctx.leg0:
            d_attn, d_act = d_x[..., : ctx.leg0], d_x[..., ctx.leg0 :]
        else:
            d_attn, d_act = None, d_x
        return d_attn, d_act, grad_gate, grads[1], grads[2], None, None, None, None, None


class _MXFP4LinearGeluLegsFunction(torch.autograd.Function):
    """``_MXFP4LinearFunction`` whose output is split at ``leg0`` with tanh-GELU on the
    second leg: returns ``(y[..., :leg0], gelu(y[..., leg0:]))``. This is the single
    blocks' linear1, whose legs are qkv and the MLP pre-activation.

    Taking the GELU inside the Function hands backward the two leg gradients separately,
    so dGELU, the cat and G's MXFP4 dual-bias pack run as one kernel and the bf16 G
    ([16384, 21504] per call) is never written. The pre-activation saved here is the
    tensor autograd's own GELU would have saved.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg, leg0, lazy=False):
        outs, save_list = _MXFP4LinearGeluLegsFunction._fwd(
            ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg, leg0, lazy
        )
        ctx.save_for_backward(*save_list)
        return outs

    @staticmethod
    def _fwd(ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg, leg0, lazy, x_pre=None):
        out, save_list = _MXFP4LinearFunction._forward(
            ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg, x_pre=x_pre
        )
        preact = out[:, leg0:]
        ctx.leg0 = leg0
        ctx.fused_dgelu = (
            bias is not None
            and cfg.dgrad == "mxfp4"
            and cfg.wgrad == "mxfp4"
            and _dual_pack_eligible(hadamard, cfg)
            and not (cfg.sr_op("grad", "dgrad") or cfg.sr_op("grad", "wgrad"))
            and _legpack_g_split(*out.shape) == (leg0, out.shape[1] - leg0)
        )
        lead = ctx.x_shape[:-1]
        act = preact if lazy else torch.nn.functional.gelu(preact, approximate="tanh")
        outs = (out[:, :leg0].reshape(*lead, leg0), act.reshape(*lead, preact.shape[1]))
        return outs, [*save_list, preact]

    @staticmethod
    def backward(ctx, grad_qkv, grad_act):
        grads = _MXFP4LinearGeluLegsFunction._bwd(ctx, ctx.saved_tensors, grad_qkv, grad_act)
        return (*grads, None, None)

    @staticmethod
    def _bwd(ctx, saved, grad_qkv, grad_act):
        preact = saved[-1]
        d_qkv = grad_qkv.reshape(-1, ctx.leg0)
        d_act = grad_act.reshape(-1, preact.shape[1])
        if ctx.fused_dgelu:
            packs = _quantize_mxfp4_h16_dual_bias_legs_dgelu_op(d_qkv, d_act, preact)
            grads = _MXFP4LinearFunction._backward(ctx, saved[:-1], None, packs)
        else:
            d_pre = torch.ops.aten.gelu_backward(d_act, preact, approximate="tanh")
            grad_2d = torch.cat((d_qkv, d_pre), dim=1)
            grads = _MXFP4LinearFunction._backward(ctx, saved[:-1], grad_2d)
        return grads


class _MXFP4LinearGeluFunction(torch.autograd.Function):
    """``gelu(_MXFP4LinearFunction(x), approximate="tanh")``: the double blocks'
    img_mlp/txt_mlp first Linear plus its activation.

    Same reason as ``_MXFP4LinearGeluLegsFunction``: backward receives d_act rather than
    G, so dGELU and G's dual-bias pack are one kernel and the bf16 G ([8192, 12288] per
    call) is never written. The bias goes into the GEMM epilogue whatever the width, so
    the saved pre-activation is the GEMM's own output and saving it costs no extra write.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg, lazy=False, dual_out=False):
        act, save_list = _MXFP4LinearGeluFunction._fwd(
            ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg, lazy, dual_out=dual_out
        )
        ctx.save_for_backward(*save_list)
        packs, ctx.gelu_dual_packs = ctx.gelu_dual_packs, None
        if packs is None:
            return act
        u8 = tuple(p.view(torch.uint8) for p in packs)
        ctx.mark_non_differentiable(*u8)
        return (act, *u8)

    @staticmethod
    def _fwd(ctx, x, weight, bias, hadamard, act_slot, wgt_slot, cfg, lazy, x_pre=None, dual_out=False):
        preact, save_list, unbiased = _MXFP4LinearFunction._forward(
            ctx,
            x,
            weight,
            bias,
            hadamard,
            act_slot,
            wgt_slot,
            cfg,
            bias_any_width=True,
            x_pre=x_pre,
            return_unbiased=True,
            gelu_dual=lazy and dual_out,
        )
        ctx.fused_dgelu = (
            bias is not None
            and cfg.dgrad == "mxfp4"
            and cfg.wgrad == "mxfp4"
            and _dual_pack_eligible(hadamard, cfg)
            and not (cfg.sr_op("grad", "dgrad") or cfg.sr_op("grad", "wgrad"))
            and preact.shape[0] % 128 == 0
            and preact.shape[1] % 256 == 0
        )
        if ctx.gelu_dual_packs is not None:
            return preact.reshape(*ctx.x_shape[:-1], preact.shape[1]), [*save_list, preact]
        if lazy and cfg.fprop == "a6w4":
            # The A6W4 consumer adds the bias itself, in f32 ahead of its GELU, as Inductor's fused gelu(out +
            # bias) did; handing it the bf16-rounded preact would change the bytes.
            act = unbiased
        else:
            act = preact if lazy else torch.nn.functional.gelu(preact, approximate="tanh")
        return act.reshape(*ctx.x_shape[:-1], preact.shape[1]), [*save_list, preact]

    @staticmethod
    def backward(ctx, grad_act, *_packs):
        return (*_MXFP4LinearGeluFunction._bwd(ctx, ctx.saved_tensors, grad_act), None, None)

    @staticmethod
    def _bwd(ctx, saved, grad_act):
        preact = saved[-1]
        d_act = grad_act.reshape(-1, preact.shape[1])
        if ctx.fused_dgelu:
            packs = _quantize_mxfp4_h16_dual_bias_dgelu_op(d_act, preact)
            return _MXFP4LinearFunction._backward(ctx, saved[:-1], None, packs)
        grad_2d = torch.ops.aten.gelu_backward(d_act, preact, approximate="tanh")
        return _MXFP4LinearFunction._backward(ctx, saved[:-1], grad_2d)


class _MXFP4LinearLnModFunction(torch.autograd.Function):
    """One of the Functions above on ``(1 + scale) * layer_norm(x) + shift`` (eps 1e-6, no affine),
    with the LN and the modulate folded into the A6W4 activation pack (`_A6W4_LNMOD_PACK`).
    ``kind`` picks the Linear: "plain" (`_MXFP4LinearFunction`), "gelu" (`_MXFP4LinearGeluFunction`)
    or "legs" (`_MXFP4LinearGeluLegsFunction`, split at ``leg0``). Saves x and the row stats in place
    of the modulated activation; backward rebuilds the normalised x from them."""

    @staticmethod
    def forward(ctx, x, shift, scale, weight, bias, hadamard, act_slot, wgt_slot, cfg, kind, leg0, lazy):
        x_pre, rstd, nmr = _lnmod_pre(x, shift, scale)
        args = (ctx, None, weight, bias, hadamard, act_slot, wgt_slot, cfg)
        if kind == "legs":
            outs, save_list = _MXFP4LinearGeluLegsFunction._fwd(*args, leg0, lazy, x_pre=x_pre)
        elif kind == "gelu":
            outs, save_list = _MXFP4LinearGeluFunction._fwd(*args, lazy, x_pre=x_pre)
        else:
            out, save_list = _MXFP4LinearFunction._forward(*args, x_pre=x_pre)
            outs = out.reshape(*x.shape[:-1], weight.shape[0])
        ctx.kind = kind
        ctx.n_lin = len(save_list)
        ctx.save_for_backward(*save_list, x, scale, rstd, nmr)
        return outs

    @staticmethod
    def backward(ctx, *grad_outs):
        saved = ctx.saved_tensors
        lin, ln = saved[: ctx.n_lin], saved[ctx.n_lin :]
        if ctx.kind == "legs":
            grads = _MXFP4LinearGeluLegsFunction._bwd(ctx, lin, *grad_outs)
        elif ctx.kind == "gelu":
            grads = _MXFP4LinearGeluFunction._bwd(ctx, lin, *grad_outs)
        else:
            g = grad_outs[0]
            grads = _MXFP4LinearFunction._backward(ctx, lin, g.reshape(-1, g.shape[-1]).contiguous())
        # x, shift, scale, weight, bias, hadamard, act_slot, wgt_slot, cfg, kind, leg0, lazy
        return (*_lnmod_backward(grads[0], *ln), *grads[1:], None, None, None)


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
        self.register_buffer("fp8_act_slot", None, persistent=False)
        self.register_buffer("fp8_wgt_slot", None, persistent=False)
        self._build_hadamard(self.weight.device)
        self._build_fp8_slots(self.weight.device)

    def _build_fp8_slots(self, device: torch.device) -> None:
        """Identity tokens naming this Linear's two entries in `_FP8_DUAL_SCALE`.

        Two, not one: the activation and the weight have unrelated dynamic ranges and each needs
        its own running amax.

        Per module rather than keyed on shape, because two Linears can share an activation shape
        while seeing activations of completely different magnitude, and a shared scale would hand
        one module's range to the other.

        These carry no value -- only their addresses matter, and `_FP8_DUAL_SCALE` explains why
        the scales cannot live in them. They are tensors rather than ints because Dynamo
        value-specialises ints and that costs a graph per block; they are buffers rather than bare
        tensors so FSDP2 and `.to()` move them with the module, which keeps one address per module
        for the life of the run. A `.to()` that does rebind them just costs one re-bootstrap.
        """
        if not (_FP8_DUAL_PACK and self.config.fprop == "fp8"):
            return
        if torch.device(device).type == "meta":
            return
        self.fp8_act_slot = torch.empty(1, dtype=torch.float32, device=device)
        self.fp8_wgt_slot = torch.empty(1, dtype=torch.float32, device=device)

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
        module._build_fp8_slots(linear.weight.device)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.any_mxfp4 or (_BF16_EVAL and not self.training):
            return super().forward(x)
        return _MXFP4LinearFunction.apply(
            x, self.weight, self.bias, self.hadamard, self.fp8_act_slot, self.fp8_wgt_slot, self.config
        )

    def forward_split_gelu(
        self, x: torch.Tensor, leg0: int, lazy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``self(x)`` split at ``leg0``, with tanh-GELU on the second part.

        Called as a method, not through ``__call__``, which is safe only because FSDP2
        wraps whole blocks. Eval goes through ``self(x)`` so the BF16-eval guard in
        ``forward`` still applies.

        ``lazy`` returns the second part as a lazy act (see `_FWD_GELU_PACK`); only pass it
        after `lazy_gelu_into` said yes, and hand the result to the consumer's
        ``forward_gelu_input``.
        """
        if self.config.any_mxfp4 and self.training:
            return _MXFP4LinearGeluLegsFunction.apply(
                x,
                self.weight,
                self.bias,
                self.hadamard,
                self.fp8_act_slot,
                self.fp8_wgt_slot,
                self.config,
                leg0,
                lazy,
            )
        qkv, mlp = torch.split(self(x), [leg0, self.out_features - leg0], dim=-1)
        return qkv, torch.nn.functional.gelu(mlp, approximate="tanh")

    def forward_gelu(self, x: torch.Tensor, lazy: bool = False):
        """``gelu(self(x), approximate="tanh")``, same calling contract as
        ``forward_split_gelu``.

        Under ``lazy`` the result is a `LazyAct`, whose ``dual`` carries the consumer's
        operands when this GEMM's epilogue could build them."""
        if self.config.any_mxfp4 and self.training:
            out = _MXFP4LinearGeluFunction.apply(
                x,
                self.weight,
                self.bias,
                self.hadamard,
                self.fp8_act_slot,
                self.fp8_wgt_slot,
                self.config,
                lazy,
                lazy,
            )
            dual = None
            if isinstance(out, tuple):
                out, aq, sa, co, cs = out
                dual = (aq, sa, co.view(FP4_DTYPE), cs.view(torch.float8_e8m0fnu))
            act = out
            rebiased = dual is not None and not _fuse_bias_eligible(
                self.config, self.out_features, any_width=True
            )
            return LazyAct(act, dual, rebiased) if lazy else act
        return torch.nn.functional.gelu(self(x), approximate="tanh")

    def lnmod_ok(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> bool:
        """True iff `forward_lnmod` can take ``x`` unmodulated (see `_A6W4_LNMOD_PACK`)."""
        return _a6w4_lnmod_eligible(self, x, shift, scale)

    def forward_lnmod(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        kind: str = "plain",
        leg0: int = 0,
        lazy: bool = False,
    ):
        """``self``, `forward_gelu` (``kind="gelu"``) or `forward_split_gelu` (``kind="legs"``) on
        ``(1 + scale) * layer_norm(x) + shift``, with the LN and the modulate in the activation
        pack. Only after `lnmod_ok` said yes; same calling contract as ``forward_split_gelu``."""
        return _MXFP4LinearLnModFunction.apply(
            x,
            shift,
            scale,
            self.weight,
            self.bias,
            self.hadamard,
            self.fp8_act_slot,
            self.fp8_wgt_slot,
            self.config,
            kind,
            leg0,
            lazy,
        )

    def lazy_gelu_into(self, consumer: nn.Module, x: torch.Tensor, widths) -> bool:
        """True iff this Linear's GELU output can be handed lazily to ``consumer``, whose
        input is the cat of legs of these ``widths`` with GELU on the last one."""
        rows = x.numel() // x.shape[-1]
        return (
            self.config.any_mxfp4
            and self.training
            and isinstance(consumer, MXFP4Linear)
            and (
                (_fp8_legs_gelu_eligible(consumer, rows, widths) and self.config.fprop != "a6w4")
                or _a6w4_legs_gelu_eligible(self, consumer, rows, widths)
            )
        )

    def lazy_gelu_bias(self) -> torch.Tensor | None:
        """The bias ``forward_gelu(lazy=True)`` left out of its returned act, which the consumer's
        `forward_gelu_input` must add ahead of the GELU (A6W4 only)."""
        if (
            self.config.fprop == "a6w4"
            and self.bias is not None
            and not _fuse_bias_eligible(self.config, self.out_features, any_width=True)
        ):
            return self.bias.detach()
        return None

    def forward_gelu_input(
        self,
        attn: torch.Tensor | None,
        act: "torch.Tensor | LazyAct",
        act_bias: torch.Tensor | None = None,
        *,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``self(cat(attn, gelu(act + act_bias)))`` -- ``attn`` and ``act_bias`` optional -- for
        a lazy ``act`` (and its producer's `lazy_gelu_bias`) from a producer whose
        `lazy_gelu_into` returned True for this module. With ``gate``, ``gate * self(...)``
        (see `forward_gated`). ``gate`` is keyword-only: the third positional argument is
        ``act_bias``, and the residual gate is ``[B, 1, N]``, not the GELU leg's bias."""
        # The gated Function has no act_bias parameter, so it can only take a lazy act whose
        # producer kept its bias in the GEMM. A6W4 is the case that hands one over.
        ready, rebiased = None, False
        if isinstance(act, LazyAct):
            act, ready, rebiased = act.act, act.dual, act.rebiased
        if ready is not None and (attn is not None or act_bias is not None) and rebiased:
            raise RuntimeError(
                "a6w4 fused epilogue: the producer moved its bias into the GEMM for this "
                "consumer, so its act cannot be re-activated here. lazy_gelu_into must not "
                "offer the fused path to a consumer with another leg or a bias of its own."
            )
        if ready is not None and (attn is not None or act_bias is not None):
            ready = None
        if gate is not None and act_bias is None and self._gated_ok(act, gate):
            return _MXFP4GatedLinearGeluInFunction.apply(
                attn,
                act,
                gate,
                self.weight,
                self.bias,
                self.hadamard,
                self.fp8_act_slot,
                self.fp8_wgt_slot,
                self.config,
                ready,
            )
        out = _MXFP4LinearGeluInFunction.apply(
            attn,
            act,
            self.weight,
            self.bias,
            self.hadamard,
            self.fp8_act_slot,
            self.fp8_wgt_slot,
            self.config,
            act_bias,
            ready,
        )
        return out if gate is None else gate * out

    def _gated_ok(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """True iff ``gate * self(x)`` should take the gated Functions: a per-sample
        ``[B, 1, N]`` gate over ``[B, L, K]`` tokens, on a Linear whose bias stays outside the
        GEMM (the gated Function moves that add next to the gate multiply)."""
        return (
            _GATE_DGATE
            and self.training
            and self.config.any_mxfp4
            and self.bias is not None
            and not _fuse_bias_eligible(self.config, self.out_features)
            and x.dim() == 3
            and tuple(gate.shape) == (x.shape[0], 1, self.out_features)
            and gate.dtype == x.dtype
        )

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``gate * self(x)`` for a gated residual. Backward packs dY with the gate applied at
        the pack's load and returns dgate from the same read (`FLUX_GATE_DGATE`)."""
        if self._gated_ok(x, gate):
            return _MXFP4GatedLinearFunction.apply(
                x,
                gate,
                self.weight,
                self.bias,
                self.hadamard,
                self.fp8_act_slot,
                self.fp8_wgt_slot,
                self.config,
            )
        return gate * self(x)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, mxfp4=({self.config.describe()})"


# Single-block linear2 (23j) and both linears of each double-block MLP (24d step 2). Activation
# goes FP6 to FP4; backward stays MXFP4. Off restores A6W4 on all 228.
_A6W4_FWD_FP4_MLP = os.getenv("FLUX_A6W4_FWD_FP4_MLP", "1") == "1"
_A6W4_FWD_FP4_RE = re.compile(
    r"(?:^|\.)single_blocks\.\d+\.linear2$" r"|(?:^|\.)double_blocks\.\d+\.(?:img_mlp|txt_mlp)\.[02]$"
)


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
    fwd_fp4 = 0

    def visit(parent: nn.Module, prefix: str) -> None:
        nonlocal fwd_fp4
        for name, child in list(parent.named_children()):
            fqn = f"{prefix}{name}"
            if type(child) is nn.Linear and module_filter_fn(child, fqn):
                module_config = (config_override_fn(fqn) if config_override_fn else None) or config
                # 23j + the joint-MLP half of 24d. Single-block linear2 and both
                # linears of each double-block MLP run an MXFP4 forward. Attention
                # and single-block linear1 stay on the recipe cast. Their joint MLP
                # moved grad norm; this is that change anyway.
                if _A6W4_FWD_FP4_MLP and module_config.fprop == "a6w4" and _A6W4_FWD_FP4_RE.search(fqn):
                    module_config = replace(module_config, fprop="mxfp4")
                    fwd_fp4 += 1
                setattr(parent, name, MXFP4Linear.from_linear(child, module_config))
                converted.append(fqn)
            else:
                visit(child, f"{fqn}.")

    visit(model, "")
    if _A6W4_FWD_FP4_MLP:
        logger.info(
            f"FLUX_A6W4_FWD_FP4_MLP: {fwd_fp4} linears run an MXFP4 forward "
            f"(single linear2, double img_mlp and txt_mlp)"
        )

    if os.getenv("FLUX_FP8_ALL_GATHER", "0") == "1":
        wrapped = 0
        for module in model.modules():
            if isinstance(module, MXFP4Linear) and _fp8_all_gather_eligible(module):
                module.weight = nn.Parameter(
                    _FP8AllGatherWeight(module.weight.data), requires_grad=module.weight.requires_grad
                )
                wrapped += 1
        logger.info(f"FLUX_FP8_ALL_GATHER: {wrapped} MXFP4 linears gather FP8 fprop + MXFP4 dgrad weights")

    # The full-MXFP4 twin. Mutually exclusive with the FP8 gather by construction rather than by
    # check: the two eligibility gates require cfg.fprop == "fp8" and == "mxfp4", so at most one can
    # ever claim a given module. Logged either way, including the zero, because a silent 0 here is
    # exactly how an all-gather flag can look enabled and do nothing -- which is what
    # FLUX_FP8_ALL_GATHER=1 does on a full-MXFP4 recipe.
    # Defaults ON, and inert on anything that is not a full-MXFP4 forward, so it costs an FP8-forward
    # recipe exactly one isinstance check per module at startup.
    # Zero loss=nan, zero NaN evals, 228 linears wrapped -- every transformer Linear, 19 double blocks
    # x 8 plus 38 single blocks x 2. The packs are bit-exact against packing the gathered weight on
    # all five real FLUX [N, K] shapes at world 8 and 32, both orientations
    # (.tmp_cmp/mxfp4_ag_test.py).
    # One node structurally cannot pay for this; four nodes, where the gather crosses the network
    # instead of xGMI, should.
    if os.getenv("FLUX_FP4_ALL_GATHER", "1") == "1":
        wrapped = wrapped_a6w4 = 0
        a6w4_named: list[tuple[str, nn.Module]] = []
        for name, module in model.named_modules():
            if isinstance(module, MXFP4Linear) and _mxfp4_all_gather_eligible(module):
                module.weight = nn.Parameter(
                    _MXFP4AllGatherWeight(module.weight.data), requires_grad=module.weight.requires_grad
                )
                wrapped += 1
            elif isinstance(module, MXFP4Linear) and _a6w4_all_gather_eligible(module):
                module.weight = nn.Parameter(
                    _A6W4AllGatherWeight(module.weight.data), requires_grad=module.weight.requires_grad
                )
                a6w4_named.append((name, module))
                wrapped_a6w4 += 1
        if a6w4_named and _A6W4_AG_PIPELINE:
            # FSDP replaces these parameters with sharded DTensors after this
            # returns. The first pre-all-gather binds the live local tensors.
            _A6W4_NAMED[:] = a6w4_named
        logger.info(
            f"FLUX_FP4_ALL_GATHER: {wrapped} MXFP4 linears gather both MXFP4 packs "
            f"(1.0625 B/elem instead of 2.0)"
        )
        logger.info(
            f"FLUX_FP4_ALL_GATHER: {wrapped_a6w4} A6W4 linears gather the A6W4 weight + MXFP4 dgrad packs"
        )
    return converted


# --------------------------------------------------------------------------- FP8 weight all-
# gather under FSDP2 (FLUX_FP8_ALL_GATHER=1)
# ---------------------------------------------------------------------------
# The TorchAO flag of the same name only acts on Float8Linear, which this module replaces, so
# without this every rank gathers the bf16 weight and then quantizes the full matrix itself. Here
# each rank quantizes only its own shard before the gather: the FP8 tensorwise copy Fprop
# multiplies with, and the colwise H16 MXFP4 pack Dgrad multiplies with, 1.53 B/elem on the wire
# instead of 2.
# Both are bit-identical to quantizing the gathered bf16 weight. FP8 bytes depend only on the
# value and the scale, and the scale comes from the same kernel fed the global amax through
# `amax_partials`. The col pack's H16 rotation and 32-element scale blocks run along N, the dim
# FSDP shards on, so a shard's pack is exactly its slice of the full pack as long as the shard
# height is a multiple of 32.
# The linear itself is unchanged: the unsharded weight intercepts the two custom ops it already
# calls on `weight` and serves them from the gathered payloads.


def _fp8_all_gather_eligible(module: "MXFP4Linear") -> bool:
    cfg = module.config
    return (
        cfg.fprop == "fp8"
        and cfg.dgrad == "mxfp4"
        and _weight_colpack_eligible(module.hadamard, cfg)
        and not cfg.sr_op("weight", "dgrad")
        and module.weight.dtype == torch.float32
    )


_PRESERVE_SUBCLASS_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


def _wide_view(t: torch.Tensor) -> torch.Tensor:
    """Hand FSDP2 a byte payload typed like the module's other gather inputs (bf16).

    FSDP2 gives each module's all-gather one buffer dtype; if its inputs disagree it
    falls back to uint8 and copies everything into and out of that buffer a byte at a
    time (~25 ms/step here). Same-dtype copies only move bits, so bf16 is safe.
    """
    return t.view(torch.bfloat16) if t.shape[-1] % 2 == 0 else t


class _FP8AllGatherWeight(torch.Tensor):
    """Sharded fp32 parameter whose FSDP2 all-gather carries pre-quantized payloads.

    Every other op unwraps to the fp32 shard, so the optimizer and FSDP2's sharding
    see an ordinary tensor. ``_amax`` is the global abs-max of the bf16 weight, a
    1-element view into the buffer ``precompute_fp8_all_gather_amax`` fills after each
    optimizer step.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, amax: torch.Tensor | None = None):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, amax: torch.Tensor | None = None):
        self._tensor = tensor
        self._amax = amax
        # (payloads, scale_inv, param_dtype, ready event) from precompute_fp8_all_gather_amax; taken by
        # the next fsdp_pre_all_gather, which quantizes itself when it is absent.
        self._prequant = None
        # FLUX_FP8_AG_PREQUANT=gather: (graph entry, index of this shard's output in it).
        self._pq_slot = None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        from torch.utils import _pytree as pytree

        if func == torch.ops.aten.detach.default:
            return _FP8AllGatherWeight(args[0]._tensor, args[0]._amax)
        amax = None

        def unwrap(t):
            nonlocal amax
            amax = t._amax if amax is None else amax
            return t._tensor

        args, kwargs = pytree.tree_map_only(_FP8AllGatherWeight, unwrap, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _PRESERVE_SUBCLASS_OPS:
            return out
        return pytree.tree_map_only(torch.Tensor, lambda x: _FP8AllGatherWeight(x, amax), out)

    def __tensor_flatten__(self):
        return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        return _FP8AllGatherWeight(inner_tensors["_tensor"])

    def __repr__(self):
        return f"_FP8AllGatherWeight({self._tensor!r})"

    def _quantize(self, param_dtype: torch.dtype):
        from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
            flydsl_quant_fp8_mxfp4_dual,
            flydsl_quant_mxfp4_h16_col,
            fp8_dual_eligible,
        )
        from primus_turbo.pytorch.core.low_precision import float8_e4m3

        w = self._tensor.to(param_dtype).contiguous()
        if w.shape[0] % MX_BLOCK_SIZE:
            raise ValueError(
                f"FP8 all-gather needs the FSDP shard height {w.shape[0]} to be a multiple of "
                f"{MX_BLOCK_SIZE} for the col pack to be sliceable; set FLUX_FP8_ALL_GATHER=0."
            )
        amax = self._amax
        if amax is None or param_dtype != torch.bfloat16:
            amax = w.abs().amax().float().reshape(1)
            torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX)
        elif _FP8_AG_FUSED_QUANT and fp8_dual_eligible(*w.shape):
            # quantize_fp8_tensorwise's arithmetic: a tensor division, not `448.0 / t`, which is
            # reciprocal-then-multiply and differs by an ulp. Computed from the amax view here rather
            # than precomputed, so a FLUX_FP8_AG_PREQUANT graph reads the static amax on every replay.
            scale = torch.full_like(amax, 448.0) / amax.clamp(min=1e-12)
            fp8, col_q, col_s, _ = flydsl_quant_fp8_mxfp4_dual(w, FP4_DTYPE, float8_e4m3, scale)
            return tuple(_wide_view(t.view(torch.uint8)) for t in (fp8, col_q, col_s)), (1.0 / scale)[0]
        fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
            w, float8_e4m3, None, 1, 1, amax
        )
        col_q, col_s = flydsl_quant_mxfp4_h16_col(
            w, FP4_DTYPE, scale_rounding_mode=_SCALE_ROUNDING_MODE, sr=False
        )
        return tuple(_wide_view(t.view(torch.uint8)) for t in (fp8, col_q, col_s)), scale_inv

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        param_dtype = (mp_policy.param_dtype if mp_policy is not None else None) or self._tensor.dtype
        prequant, self._prequant = self._prequant, None
        if self._pq_slot is not None and param_dtype == torch.bfloat16:
            entry, k = self._pq_slot
            if entry[3]:
                entry[0].replay()
                entry[3] = False
            payloads, scale_inv = entry[2][k]
            prequant = (payloads, scale_inv, param_dtype, None)
        if prequant is not None and prequant[2] == param_dtype:
            payloads, scale_inv, _, ready = prequant
            if ready is not None:
                stream = torch.cuda.current_stream()
                stream.wait_event(ready)
                for t in (*payloads, scale_inv):
                    t.record_stream(stream)
            if _FP8_AG_PREQUANT_CHECK:
                # Byte compare: the payloads are FP8/FP4 bits viewed as bf16, where NaN != NaN.
                fresh, fresh_scale = self._quantize(param_dtype)
                for name, a, b in zip(
                    ("fp8", "col_q", "col_s", "scale_inv"), (*payloads, scale_inv), (*fresh, fresh_scale)
                ):
                    a, b = a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
                    diff = (a != b).sum().item()
                    if diff:
                        raise RuntimeError(
                            f"FLUX_FP8_AG_PREQUANT: precomputed {name} differs from a fresh quantize in "
                            f"{diff}/{a.numel()} bytes (shard {tuple(self.shape)})"
                        )
            return payloads, (scale_inv, mesh.size())
        payloads, scale_inv = self._quantize(param_dtype)
        return payloads, (scale_inv, mesh.size())

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        fp8, col_q, col_s = (t.view(torch.uint8) for t in all_gather_outputs)
        scale_inv, world = metadata
        if out is not None:
            target = out._local_tensor if hasattr(out, "_local_tensor") else out
            target._scale_inv = scale_inv
            return
        # FSDP2 frees and re-allocates the storages it returned; the uint8 views share them.
        kmajor = getattr(self, "_ag_kmajor", False)
        return _FP8GatheredWeight(fp8, scale_inv, col_q, col_s, world, param_dtype, kmajor), tuple(
            all_gather_outputs
        )


# Regroup the gathered Dgrad col pack in one opaque copy per payload instead of the inline
# view/transpose/reshape/dtype-view chain. Read at import: torch.compile must not trace a getenv.
_COLPACK_REGROUP_OP = os.getenv("FLUX_COLPACK_REGROUP_OP", "0") == "1"


@triton.jit
def _regroup2_kernel(
    q_src, q_dst, s_src, s_dst, K, W, XQ, XS, NQC, NSC, NQB, BR: tl.constexpr, BC: tl.constexpr
):
    # Destination row r = k * W + w is source row w * K + k; q and s tiles share one grid.
    pid = tl.program_id(0)
    if pid < NQB:
        src, dst, X, nc, p = q_src, q_dst, XQ, NQC, pid
    else:
        src, dst, X, nc, p = s_src, s_dst, XS, NSC, pid - NQB
    r = (p // nc) * BR + tl.arange(0, BR)
    c = (p % nc) * BC + tl.arange(0, BC)
    srow = (r % W) * K + r // W
    m = (r[:, None] < K * W) & (c[None, :] < X)
    v = tl.load(src + srow[:, None].to(tl.int64) * X + c[None, :], mask=m)
    tl.store(dst + r[:, None].to(tl.int64) * X + c[None, :], v, mask=m)


def _regroup_word(t: torch.Tensor) -> torch.dtype:
    return torch.int32 if t.shape[1] % 4 == 0 and t.storage_offset() % 4 == 0 else torch.uint8


@torch.library.custom_op("primus_flux::dgrad_col_regroup", mutates_args=())
def _dgrad_col_regroup_op(
    col_q: torch.Tensor, col_s: torch.Tensor, world: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gathered col pack ([world*K, N/world/2] q, [world*K, N/world/32] s, uint8) -> the
    Dgrad operand [K, N/2] FP4 and [K, N/32] e8m0, both payloads in one Triton launch.
    Bit-identical to the inline regroup in ``_dgrad_col_pack``; a custom op so the FP4 / e8m0
    reinterpretation stays a view rather than something Inductor may lower to a copy. Outputs
    never alias the inputs, even at ``world == 1``, as custom ops require."""
    k = col_q.shape[0] // world
    col_q, col_s = col_q.contiguous(), col_s.contiguous()
    word = torch.int32 if _regroup_word(col_q) == _regroup_word(col_s) == torch.int32 else torch.uint8
    q, s = col_q.view(word), col_s.view(word)
    out_q, out_s = torch.empty_like(q), torch.empty_like(s)
    br, bc = 16, 128
    nr = triton.cdiv(k * world, br)
    nqc, nsc = triton.cdiv(q.shape[1], bc), triton.cdiv(s.shape[1], bc)
    _regroup2_kernel[(nr * (nqc + nsc),)](
        q, out_q, s, out_s, k, world, q.shape[1], s.shape[1], nqc, nsc, nr * nqc, BR=br, BC=bc
    )
    return (
        out_q.view(torch.uint8).view(k, -1).view(FP4_DTYPE),
        out_s.view(torch.uint8).view(k, -1).view(torch.float8_e8m0fnu),
    )


@_dgrad_col_regroup_op.register_fake
def _(col_q, col_s, world):
    k = col_q.shape[0] // world
    q = torch.empty(k, world * col_q.shape[1], dtype=torch.uint8, device=col_q.device)
    s = torch.empty(k, world * col_s.shape[1], dtype=torch.uint8, device=col_s.device)
    return q.view(FP4_DTYPE), s.view(torch.float8_e8m0fnu)


# FLUX_AG_COLPACK_KMAJOR=1: FSDP2's all-gather copy-out writes each gathered col pack straight
# into the Dgrad operand's [K, N/2] / [K, N/32] order, so `_dgrad_col_pack` is a view and the
# regroup above (one more read and write of every col pack, 228 launches per step) goes away. The
# other payloads are copied as FSDP2's split_with_sizes_copy would. One launch per group for the
# 4-byte-aligned segments and one for the rest (the col scales' 3-21 byte rank rows).
# A segment is the [W, K, rb] block of one payload across the W ranks, written as [K, W, rb]:
# dst[(k * W + r) * rb + j] = src[r * PR + off + k * rb + j]. An ordinary payload is K = 1. Mode 0
# tiles it by dst rows of rb (long rows); mode 1 by K rows of W * rb (short rank rows).
_AG_KMAJOR = os.getenv("FLUX_AG_COLPACK_KMAJOR", "0") == "1"
_AG_KMAJOR_CHECK = int(os.getenv("FLUX_AG_COLPACK_KMAJOR_CHECK", "0"))
_AG_TAB = 8  # dst ptr, src off, rb, K, mode, first program, col blocks, -
_AG_TABLE_ARGS = os.getenv("FLUX_AG_TABLE_ARGS", "0") == "1"


@triton.jit
def _ag_copyout_kernel(
    src,
    tab,
    PR,
    NSEG,
    W: tl.constexpr,
    NSEG_P2: tl.constexpr,
    BC0: tl.constexpr,
    BR1: tl.constexpr,
    BC1: tl.constexpr,
):
    pid = tl.program_id(0)
    segs = tl.arange(0, NSEG_P2)
    starts = tl.load(tab + segs * 8 + 5, mask=segs < NSEG, other=1 << 62)
    s = tl.sum((starts <= pid).to(tl.int32)) - 1
    row = tab + s * 8
    dst = tl.load(row).to(tl.pointer_type(src.dtype.element_ty))
    off = tl.load(row + 1)
    rb = tl.load(row + 2)
    K = tl.load(row + 3)
    mode = tl.load(row + 4)
    p = pid - tl.load(row + 5)
    ncb = tl.load(row + 6)
    if mode == 0:
        d = p // ncb
        j = (p % ncb) * BC0 + tl.arange(0, BC0)
        m0 = j < rb
        v0 = tl.load(src + ((d % W) * PR + off + (d // W) * rb) + j, mask=m0)
        tl.store(dst + d * rb + j, v0, mask=m0)
    else:
        k = (p // ncb) * BR1 + tl.arange(0, BR1)
        c = (p % ncb) * BC1 + tl.arange(0, BC1)
        r = c // rb
        m1 = (k[:, None] < K) & (c[None, :] < W * rb)
        sidx = (r * PR + off + (c - r * rb))[None, :] + k[:, None] * rb
        v1 = tl.load(src + sidx, mask=m1)
        tl.store(dst + k[:, None] * (W * rb) + c[None, :], v1, mask=m1)


@triton.jit
def _ag_copyout_args_kernel(
    src,
    tab,
    PR,
    NSEG,
    dsts,
    W: tl.constexpr,
    NSEG_P2: tl.constexpr,
    BC0: tl.constexpr,
    BR1: tl.constexpr,
    BC1: tl.constexpr,
):
    pid = tl.program_id(0)
    segs = tl.arange(0, NSEG_P2)
    starts = tl.load(tab + segs * 8 + 5, mask=segs < NSEG, other=1 << 62)
    s = tl.sum((starts <= pid).to(tl.int32)) - 1
    row = tab + s * 8
    di = tl.load(row)
    addr = dsts[0].to(tl.int64)
    for i in tl.static_range(1, len(dsts)):
        addr = tl.where(di == i, dsts[i].to(tl.int64), addr)
    dst = addr.to(tl.pointer_type(src.dtype.element_ty))
    off = tl.load(row + 1)
    rb = tl.load(row + 2)
    K = tl.load(row + 3)
    mode = tl.load(row + 4)
    p = pid - tl.load(row + 5)
    ncb = tl.load(row + 6)
    if mode == 0:
        d = p // ncb
        j = (p % ncb) * BC0 + tl.arange(0, BC0)
        m0 = j < rb
        v0 = tl.load(src + ((d % W) * PR + off + (d // W) * rb) + j, mask=m0)
        tl.store(dst + d * rb + j, v0, mask=m0)
    else:
        k = (p // ncb) * BR1 + tl.arange(0, BR1)
        c = (p % ncb) * BC1 + tl.arange(0, BC1)
        r = c // rb
        m1 = (k[:, None] < K) & (c[None, :] < W * rb)
        sidx = (r * PR + off + (c - r * rb))[None, :] + k[:, None] * rb
        v1 = tl.load(src + sidx, mask=m1)
        tl.store(dst + k[:, None] * (W * rb) + c[None, :], v1, mask=m1)


_AG_LAUNCH = {4: (4096, 8, 512), 1: (4096, 16, 256)}  # word bytes -> BC0, BR1, BC1


def _ag_col_payloads(fsdp_param) -> tuple[int, ...]:
    inner = fsdp_param._sharded_local_tensor
    if isinstance(inner, _FP8AllGatherWeight):
        return (1, 2)
    if isinstance(inner, _MXFP4AllGatherWeight):
        return (2, 3)
    return ()


class _AGKMajorPlan:
    """One FSDP param group's copy-out as segments; the dst pointers are bound per call."""

    def __init__(self, fsdp_params, numels, dtypes, split_sizes, itemsize, world):
        self.world = world
        self.pr = sum(split_sizes) * itemsize
        self.segs = []  # (output index, off, rb, K) in bytes
        self.kmajor = []
        off = i = 0
        # Each payload's bytes come from ITS OWN dtype, not from the all-gather buffer's.
        # This group is mixed: the weight payload is bf16 while the Dgrad column packs are uint8 FP4 and
        # E8M0. Sizing everything by all_gather_output.element_size() undercounts a bf16 payload by 2x,
        # every later offset inherits the error, and the scatter then reads and writes the wrong regions
        # -- weights come out garbage and the loss is nan by step 6. The checker caught it as "output 0
        # (K=1, rb=1769472) is 113246208 bytes, the gathered segment is 56623104", a ratio of exactly 2.
        # It only bites when a payload dtype differs from the buffer's, which is why the flag can be clean
        # on one recipe or image and diverge on another.
        for fp, param_numels, param_dtypes in zip(fsdp_params, numels, dtypes):
            cols = _ag_col_payloads(fp)
            kdim = fp._orig_size[1] if len(fp._orig_size) == 2 else 0
            ok = (
                bool(cols)
                and fp.fsdp_placement.dim == 0
                and all(kdim and (param_numels[c] * param_dtypes[c].itemsize) % kdim == 0 for c in cols)
            )
            self.kmajor.append(ok)
            for c, n in enumerate(param_numels):
                b = n * param_dtypes[c].itemsize
                if ok and c in cols:
                    self.segs.append((i, off, b // kdim, kdim))
                else:
                    self.segs.append((i, off, b, 1))
                off += b
                i += 1
        self.tables = {}
        self.static = {}

    def _rows(self, ptrs):
        """Per launch word: (word, rows, programs, output index of each row). Column 0 is a
        placeholder; ptrs only decides which segments are 4-byte aligned."""
        out = []
        for word, (bc0, br1, bc1) in _AG_LAUNCH.items():
            rows, idx, prog = [], [], 0
            for i, off, rb, k in self.segs:
                fits = all(x % 4 == 0 for x in (off, rb, self.pr, ptrs[i]))
                if (word == 4) != fits:
                    continue
                o, rbw = off // word, rb // word
                if k == 1:
                    ncb = triton.cdiv(rbw, bc0)
                    rows.append([0, o, rbw, 1, 0, prog, ncb, 0])
                    prog += self.world * ncb
                else:
                    ncb = triton.cdiv(self.world * rbw, bc1)
                    rows.append([0, o, rbw, k, 1, prog, ncb, 0])
                    prog += triton.cdiv(k, br1) * ncb
                idx.append(i)
            if rows:
                out.append((word, rows, prog, idx))
        return out

    def _table(self, outs, device):
        key = tuple(o.data_ptr() for o in outs)
        hit = self.tables.get(key)
        if hit is not None:
            return hit
        launches = []
        for word, rows, prog, idx in self._rows(key):
            for r, i in zip(rows, idx):
                r[0] = key[i]
            host = torch.tensor(rows, dtype=torch.int64).pin_memory()
            launches.append((word, len(rows), prog, host, host.to(device, non_blocking=True)))
        if len(self.tables) >= 4:
            self.tables.pop(next(iter(self.tables)))
        self.tables[key] = launches
        return launches

    def _static_table(self, outs, device):
        ptrs = [o.data_ptr() for o in outs]
        align = tuple(p % 4 == 0 for p in ptrs)
        hit = self.static.get(align)
        if hit is None:
            hit = []
            for word, rows, prog, idx in self._rows(ptrs):
                for j, r in enumerate(rows):
                    r[0] = j
                hit.append((word, len(rows), prog, idx, torch.tensor(rows, dtype=torch.int64, device=device)))
            self.static[align] = hit
        return hit

    def copy(self, src: torch.Tensor, outs) -> None:
        src8 = src.view(torch.uint8)
        if _AG_TABLE_ARGS:
            for word, nseg, prog, idx, tab in self._static_table(outs, src.device):
                bc0, br1, bc1 = _AG_LAUNCH[word]
                s = src8.view(torch.int32) if word == 4 else src8
                _ag_copyout_args_kernel[(prog,)](
                    s,
                    tab,
                    self.pr // word,
                    nseg,
                    tuple(outs[i].data_ptr() for i in idx),
                    W=self.world,
                    NSEG_P2=triton.next_power_of_2(nseg),
                    BC0=bc0,
                    BR1=br1,
                    BC1=bc1,
                    num_warps=4,
                )
            return
        for word, nseg, prog, _host, tab in self._table(outs, src.device):
            bc0, br1, bc1 = _AG_LAUNCH[word]
            s = src8.view(torch.int32) if word == 4 else src8
            _ag_copyout_kernel[(prog,)](
                s,
                tab,
                self.pr // word,
                nseg,
                W=self.world,
                NSEG_P2=triton.next_power_of_2(nseg),
                BC0=bc0,
                BR1=br1,
                BC1=bc1,
                num_warps=4,
            )


_AG_PLANS: dict = {}
_AG_CHECKED: dict = {}


def _ag_check(all_gather_output, split_sizes, plan, outs, world) -> None:
    """
Compare the K-major copy-out against the bytes the stock path would have written.

The reference is sliced straight out of the gathered buffer rather than produced by torch.ops.fsdp.split_with_sizes_copy. That op requires every `out` tensor to share the source's dtype, and this all-gather carries MIXED dtypes -- bf16 weights beside uint8 FP4 and E8M0 packs -- so handing it `empty_like(o)` raises

Expected out tensor to have dtype unsigned char, but got c10::BFloat16 instead

Slicing bytes has no dtype constraint, and it tests the mapping more directly: the kernel's contract is dst[(k * W + r) * rb + j] = src[r * PR + off + k * rb + j], which is exactly a [W, k, rb] block read rank-major and written K-major.
"""
    src8 = all_gather_output.view(torch.uint8).view(world, -1)
    for (i, off, rb, k), got in zip(plan.segs, outs):
        want = src8[:, off : off + k * rb].reshape(world, k, rb)
        if k > 1:
            want = want.transpose(0, 1)
        want = want.reshape(-1)
        got8 = got.view(torch.uint8).reshape(-1)
        if got8.numel() != want.numel():
            raise RuntimeError(
                f"FLUX_AG_COLPACK_KMAJOR: output {i} (K={k}, rb={rb}) is {got8.numel()} bytes, "
                f"the gathered segment is {want.numel()}"
            )
        bad = (got8 != want).sum().item()
        if bad:
            raise RuntimeError(f"FLUX_AG_COLPACK_KMAJOR: output {i} (K={k}, rb={rb}) differs in {bad} bytes")


def _ag_kmajor_copy_out(all_gather_result, fsdp_params, group) -> None:
    from torch.distributed.device_mesh import _get_device_handle

    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        input_dtypes,
        input_numels,
        split_sizes,
    ) = all_gather_result
    world = group.size()
    key = (tuple(map(id, fsdp_params)), tuple(split_sizes), all_gather_output.dtype, world)
    plan = _AG_PLANS.get(key)
    if plan is None:
        plan = _AGKMajorPlan(
            fsdp_params,
            input_numels,
            input_dtypes,
            split_sizes,
            all_gather_output.element_size(),
            world,
        )
        if not any(plan.kmajor) or not all(fp.fsdp_placement.dim == 0 for fp in fsdp_params) or world == 1:
            plan = False
        _AG_PLANS[key] = plan
    for fp, ok in zip(fsdp_params, plan.kmajor if plan else [False] * len(fsdp_params)):
        inner = fp._sharded_local_tensor
        if getattr(inner, "_ag_kmajor", ok) != ok:
            raise RuntimeError("FLUX_AG_COLPACK_KMAJOR: a gathered weight changed col-pack layout")
        inner._ag_kmajor = ok
    if not plan:
        return _AG_COPY_OUT(all_gather_result, fsdp_params, group)
    device_handle = _get_device_handle(all_gather_output.device.type)
    if all_gather_event is not None:
        device_handle.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, torch.distributed.distributed_c10d.Work):
        all_gather_work.wait()
    outs = []
    for numels, dtypes, fp in zip(input_numels, input_dtypes, fsdp_params):
        fp.init_all_gather_outputs(numels, dtypes, world, all_gather_output.device)
        fp.alloc_all_gather_outputs()
        outs.extend(fp.all_gather_outputs)
    with torch.no_grad():
        plan.copy(all_gather_output, outs)
        n = _AG_CHECKED.get(key, 0)
        if n < _AG_KMAJOR_CHECK:
            _AG_CHECKED[key] = n + 1
            _ag_check(all_gather_output, split_sizes, plan, outs, world)


_AG_COPY_OUT = None


def _install_ag_kmajor_copy_out() -> None:
    global _AG_COPY_OUT
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group

    if _AG_COPY_OUT is None:
        _AG_COPY_OUT = _fsdp_param_group.foreach_all_gather_copy_out
        _fsdp_param_group.foreach_all_gather_copy_out = _ag_kmajor_copy_out
        logger.info("FLUX_AG_COLPACK_KMAJOR: FSDP2 copy-out writes the Dgrad col packs K-major")


if _AG_KMAJOR:
    _install_ag_kmajor_copy_out()


def _kmajor_col_pack(col_q, col_s, n, k):
    return col_q.view(k, n // 2).view(FP4_DTYPE), col_s.view(k, n // MX_BLOCK_SIZE).view(torch.float8_e8m0fnu)


class _FP8GatheredWeight(torch.Tensor):
    """Unsharded [N, K] weight held only as its gathered FP8 and col-pack payloads.

    ``_col_q`` / ``_col_s`` arrive as each rank's [K, N/world/2] and [K, N/world/32]
    stacked along dim 0; the Dgrad operand regroups them into [K, N/2] and [K, N/32].
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, fp8, scale_inv, col_q, col_s, world, dtype, kmajor=False):
        n = fp8.shape[0]
        k = fp8.shape[1]
        return torch.Tensor._make_wrapper_subclass(
            cls, (n, k), strides=(k, 1), dtype=dtype, device=fp8.device, requires_grad=False
        )

    def __init__(self, fp8, scale_inv, col_q, col_s, world, dtype, kmajor=False):
        self._fp8 = fp8
        self._scale_inv = scale_inv
        self._col_q = col_q
        self._col_s = col_s
        self._world = world
        self._kmajor = kmajor

    def _rewrap(self):
        return _FP8GatheredWeight(
            self._fp8, self._scale_inv, self._col_q, self._col_s, self._world, self.dtype, self._kmajor
        )

    def _dgrad_col_pack(self):
        n, k = self.shape
        w = self._world
        if self._kmajor:
            return _kmajor_col_pack(self._col_q, self._col_s, n, k)
        if _COLPACK_REGROUP_OP:
            return _dgrad_col_regroup_op(self._col_q, self._col_s, w)
        col_q = self._col_q.view(w, k, -1).transpose(0, 1).reshape(k, n // 2)
        col_s = self._col_s.view(w, k, -1).transpose(0, 1).reshape(k, n // MX_BLOCK_SIZE)
        return col_q.view(FP4_DTYPE), col_s.view(torch.float8_e8m0fnu)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return args[0]._rewrap()
        if func in (torch.ops.aten.as_strided.default, torch.ops.aten.view.default):
            self = args[0]
            if tuple(args[1]) != tuple(self.shape):
                raise NotImplementedError(f"_FP8GatheredWeight: {func} to {args[1]} from {tuple(self.shape)}")
            return self._rewrap()
        if func == torch.ops.primus_flux.fp8_tensorwise_mm.default:
            a, b = args
            return torch.ops.primus_flux.fp8_tensorwise_mm_prequant_w(a, b._fp8, b._scale_inv)
        if func == torch.ops.primus_flux.fp8_tensorwise_mm_aq.default:
            # The activation-side dual pack sends the fprop down `mm_aq` instead of
            # `fp8_tensorwise_mm`, so a gathered weight has to be answered here too or the two
            # features cannot be on together. `mm_aq` uses B as-is when given a scale, hence the
            # view that `mm_prequant_w` does for itself above -- `_fp8` is stored as bytes.
            from primus_turbo.pytorch.core.low_precision import float8_e4m3

            a_fp8, a_si, b, out_dtype = args[0], args[1], args[2], args[3]
            # Slot 5, past the b_scale_inv the caller cannot have supplied here (B is this subclass, so its
            # scale comes from `_scale_inv` below, not from the call).
            bias = args[5] if len(args) > 5 else kwargs.get("bias")
            return torch.ops.primus_flux.fp8_tensorwise_mm_aq(
                a_fp8, a_si, b._fp8.view(float8_e4m3), out_dtype, b._scale_inv, bias
            )
        if func == torch.ops.primus_flux.quantize_mxfp4_h16_col.default:
            x = args[0]
            sr = args[1] if len(args) > 1 else kwargs.get("sr", False)
            if sr:
                raise NotImplementedError("_FP8GatheredWeight carries a round-to-nearest col pack only")
            return x._dgrad_col_pack()
        raise NotImplementedError(f"_FP8GatheredWeight does not support {func}")

    def __tensor_flatten__(self):
        return ["_fp8", "_scale_inv", "_col_q", "_col_s"], (self._world, self.dtype, self._kmajor)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        return _FP8GatheredWeight(
            inner_tensors["_fp8"],
            inner_tensors["_scale_inv"],
            inner_tensors["_col_q"],
            inner_tensors["_col_s"],
            *ctx,
        )

    def __repr__(self):
        return f"_FP8GatheredWeight(shape={tuple(self.shape)}, world={self._world})"


# --------------------------------------------------------------------------- MXFP4 weight all-
# gather under FSDP2 (FLUX_FP4_ALL_GATHER=1)
# ---------------------------------------------------------------------------
# The FP8 gather above only fires when `cfg.fprop == "fp8"`, so a full-MXFP4 recipe
# (FLUX_FP4_PASSES=all) gathers the bf16 weight and then has every rank pack the whole matrix
# itself -- twice, since an MXFP4 forward needs both orientations. This is the same idea for that
# recipe: each rank packs only its own shard before the gather.
# It moves strictly more than the FP8 version and costs strictly less to set up:
# and there is NO amax to agree on. FP8 tensorwise needs one global scale, which is why
# `precompute_fp8_all_gather_amax` runs a foreach-norm plus an all-reduce every step; MXFP4 scales
# are per-32-element block and computed entirely inside the shard, so this path adds no collective
# of its own.
# Sliceability differs per orientation and only one of them needs a guard. For a [N, K] weight the
# row pack is [N, K/2] with scale blocks along K, so a shard's row pack IS its slice of the full
# one and the gather's dim-0 concat lands it already correct. The col pack is [K, N/2] with blocks
# along N -- the dim FSDP shards -- so it needs the shard height to be a multiple of 32 and a
# regroup after the gather, exactly like FP8's.
# Stochastic rounding is refused rather than supported: the pack is fixed at gather time, so one
# draw would have to serve both Fprop and Dgrad, which is not what `sr_op("weight", "fprop")` and
# `sr_op("weight", "dgrad")` mean. Same refusal as the FP8 path.


def _mxfp4_all_gather_eligible(module: "MXFP4Linear") -> bool:
    cfg = module.config
    return (
        cfg.fprop == "mxfp4"
        and cfg.dgrad == "mxfp4"
        and _weight_colpack_eligible(module.hadamard, cfg)
        and not cfg.sr_op("weight", "fprop")
        and not cfg.sr_op("weight", "dgrad")
        and module.weight.dtype == torch.float32
    )


class _MXFP4AllGatherWeight(torch.Tensor):
    """Sharded fp32 parameter whose FSDP2 all-gather carries both MXFP4 packs.

    Mirrors ``_FP8AllGatherWeight`` minus the amax: every op unwraps to the fp32 shard so
    the optimizer and FSDP2's sharding see an ordinary tensor.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        from torch.utils import _pytree as pytree

        if func == torch.ops.aten.detach.default:
            return cls(args[0]._tensor)
        args, kwargs = pytree.tree_map_only(_MXFP4AllGatherWeight, lambda t: t._tensor, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _PRESERVE_SUBCLASS_OPS:
            return out
        return pytree.tree_map_only(torch.Tensor, cls, out)

    def __tensor_flatten__(self):
        return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        return _MXFP4AllGatherWeight(inner_tensors["_tensor"])

    def __repr__(self):
        return f"{type(self).__name__}({self._tensor!r})"

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        param_dtype = (mp_policy.param_dtype if mp_policy is not None else None) or self._tensor.dtype
        w = self._tensor.to(param_dtype).contiguous()
        if w.shape[0] % MX_BLOCK_SIZE:
            raise ValueError(
                f"MXFP4 all-gather needs the FSDP shard height {w.shape[0]} to be a multiple of "
                f"{MX_BLOCK_SIZE} for the col pack to be sliceable; set FLUX_FP4_ALL_GATHER=0."
            )
        # One read of the shard for both orientations -- the same op the forward would have
        # called on the full matrix, so this is `world` times less packing work in total.
        row_q, row_s, col_q, col_s = _quantize_mxfp4_h16_dual_op(w, False, False)
        payloads = tuple(_wide_view(t.view(torch.uint8)) for t in (row_q, row_s, col_q, col_s))
        return payloads, (mesh.size(),)

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        (world,) = metadata
        if out is not None:
            return
        row_q, row_s, col_q, col_s = (t.view(torch.uint8) for t in all_gather_outputs)
        # FSDP2 frees and re-allocates the storages it returned; the uint8 views share them.
        return (
            _MXFP4GatheredWeight(
                row_q, row_s, col_q, col_s, world, param_dtype, getattr(self, "_ag_kmajor", False)
            ),
            tuple(all_gather_outputs),
        )


class _MXFP4GatheredWeight(torch.Tensor):
    """Unsharded [N, K] weight held only as its two gathered MXFP4 packs.

    ``_row_q`` / ``_row_s`` arrive already correct at [N, K/2] and [N, K/32], because the
    row pack's scale blocks run along K while FSDP concatenates along N. ``_col_q`` /
    ``_col_s`` arrive as each rank's [K, N/world/2] and [K, N/world/32] stacked along dim
    0 and are regrouped into [K, N/2] and [K, N/32] on demand.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, row_q, row_s, col_q, col_s, world, dtype, kmajor=False):
        n = row_q.shape[0]
        k = row_q.shape[1] * 2
        return torch.Tensor._make_wrapper_subclass(
            cls, (n, k), strides=(k, 1), dtype=dtype, device=row_q.device, requires_grad=False
        )

    def __init__(self, row_q, row_s, col_q, col_s, world, dtype, kmajor=False):
        self._row_q = row_q
        self._row_s = row_s
        self._col_q = col_q
        self._col_s = col_s
        self._world = world
        self._kmajor = kmajor

    def _rewrap(self):
        return type(self)(
            self._row_q, self._row_s, self._col_q, self._col_s, self._world, self.dtype, self._kmajor
        )

    def _fprop_row_pack(self):
        return self._row_q.view(FP4_DTYPE), self._row_s.view(torch.float8_e8m0fnu)

    def _dgrad_col_pack(self):
        n, k = self.shape
        w = self._world
        if self._kmajor:
            return _kmajor_col_pack(self._col_q, self._col_s, n, k)
        if _COLPACK_REGROUP_OP:
            return _dgrad_col_regroup_op(self._col_q, self._col_s, w)
        col_q = self._col_q.view(w, k, -1).transpose(0, 1).reshape(k, n // 2)
        col_s = self._col_s.view(w, k, -1).transpose(0, 1).reshape(k, n // MX_BLOCK_SIZE)
        return col_q.view(FP4_DTYPE), col_s.view(torch.float8_e8m0fnu)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return args[0]._rewrap()
        if func in (torch.ops.aten.as_strided.default, torch.ops.aten.view.default):
            self = args[0]
            if tuple(args[1]) != tuple(self.shape):
                raise NotImplementedError(
                    f"_MXFP4GatheredWeight: {func} to {args[1]} from {tuple(self.shape)}"
                )
            return self._rewrap()
        if func == torch.ops.primus_flux.quantize_mxfp4_h16_dual.default:
            # The forward's `fuse_w` path. Serving it from the gather is what makes this
            # feature worth having: without the interception every rank would re-pack the
            # full matrix that the shards already packed between them.
            self = args[0]
            row_sr = args[1] if len(args) > 1 else kwargs.get("row_sr", False)
            col_sr = args[2] if len(args) > 2 else kwargs.get("col_sr", False)
            if row_sr or col_sr:
                raise NotImplementedError(
                    "_MXFP4GatheredWeight carries round-to-nearest packs only; "
                    "_mxfp4_all_gather_eligible refuses weight SR for this reason"
                )
            return (*self._fprop_row_pack(), *self._dgrad_col_pack())
        if func == torch.ops.primus_flux.quantize_mxfp4_h16_col.default:
            # The unfused Dgrad path, for a module where `fuse_w` did not fire.
            self = args[0]
            sr = args[1] if len(args) > 1 else kwargs.get("sr", False)
            if sr:
                raise NotImplementedError("_MXFP4GatheredWeight carries a round-to-nearest col pack only")
            return self._dgrad_col_pack()
        raise NotImplementedError(f"_MXFP4GatheredWeight does not support {func}")

    def __tensor_flatten__(self):
        return ["_row_q", "_row_s", "_col_q", "_col_s"], (self._world, self.dtype, self._kmajor)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        return _MXFP4GatheredWeight(
            inner_tensors["_row_q"],
            inner_tensors["_row_s"],
            inner_tensors["_col_q"],
            inner_tensors["_col_s"],
            *ctx,
        )

    def __repr__(self):
        return f"{type(self).__name__}(shape={tuple(self.shape)}, world={self._world})"


# The A6W4 forward (cfg.fprop == "a6w4") under the same gather. Neither gate above admits it, so
# without this it gathers the bf16 weight and every rank quantizes the full matrix twice per step:
# quant_w_a6w4 for the fprop and the H16 col pack for Dgrad.
# The row payload is quant_w_a6w4's own output, per shard, so the fprop weight is the same
# kernel's bytes as before. Its layout is shuffle_weight_w4(16) / shuffle_scale_w4, whose
# outermost dims are 16- and 32-row groups, so with a shard height % 32 == 0 the dim-0 concat
# lands the full matrix already in the GEMM's layout. The col payload is the same
# `quantize_mxfp4_h16_col` the backward calls on the gathered weight today, and slices along N
# exactly as the MXFP4 gather's does.


def _a6w4_all_gather_eligible(module: "MXFP4Linear") -> bool:
    cfg = module.config
    n, k = module.weight.shape
    return (
        os.getenv("FLUX_A6W4_FP4_ALL_GATHER", "1") == "1"
        and cfg.fprop == "a6w4"
        and cfg.dgrad == "mxfp4"
        and _weight_colpack_eligible(module.hadamard, cfg)
        and not cfg.sr_op("weight", "dgrad")
        and module.weight.dtype == torch.float32
        and n % 256 == 0
        and k % 1024 == 0
    )


# Pack group i+1 while group i's all-gather is in flight. The quantize today runs on
# FSDP's copy-in stream immediately before that group's collective, so a double block's
# 1.60 ms gather cannot start until its own packs finish. The first group of a step still
# packs inline: nothing is in flight ahead of it. FLUX_A6W4_AG_PIPELINE=0 restores that.
_A6W4_AG_PIPELINE = os.getenv("FLUX_A6W4_AG_PIPELINE", "1") == "1"
_A6W4_NAMED: list = []
_A6W4_GROUPS: list[list] = []
_A6W4_BOUND = False
_A6W4_PASS = 0
_A6W4_KICKED: dict[int, int] = {}
_A6W4_SIDE = None
_A6W4_SIDE_ARMED = False
_A6W4_PIPE_LOGGED = False
_A6W4_MISS_LOGGED = False


class _A6W4AllGatherWeight(_MXFP4AllGatherWeight):
    """Sharded fp32 parameter whose FSDP2 all-gather carries the A6W4 fprop weight and the
    MXFP4 Dgrad col pack."""

    def __tensor_flatten__(self):
        return ["_tensor"], (getattr(self, "_ag_group", None), getattr(self, "_ag_index", None))

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        shard = _A6W4AllGatherWeight(inner_tensors["_tensor"])
        group, index = ctx if isinstance(ctx, tuple) else (None, None)
        shard._ag_group = group
        shard._ag_index = index
        shard._prequant = None
        if (
            group is not None
            and index is not None
            and 0 <= group < len(_A6W4_GROUPS)
            and 0 <= index < len(_A6W4_GROUPS[group])
        ):
            _A6W4_GROUPS[group][index] = shard
        return shard

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Do not call the parent implementation: its cls would be the parent,
        # and a preserved op would come back as an MXFP4 weight.
        from torch.utils import _pytree as pytree

        srcs = [t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, cls)]
        src = srcs[0] if len(srcs) == 1 else None
        if func == torch.ops.aten.detach.default:
            out = cls(args[0]._tensor)
        else:
            args, kwargs = pytree.tree_map_only(
                _MXFP4AllGatherWeight, lambda t: t._tensor, (args, kwargs or {})
            )
            out = func(*args, **kwargs)
            if func not in _PRESERVE_SUBCLASS_OPS:
                return out
            out = pytree.tree_map_only(torch.Tensor, cls, out)
        if src is None or getattr(src, "_ag_group", None) is None:
            return out

        def carry(dst):
            if not isinstance(dst, cls) or dst is src:
                return dst
            dst._ag_group = src._ag_group
            dst._ag_index = src._ag_index
            dst._prequant = None
            group, index = dst._ag_group, dst._ag_index
            if (
                index is not None
                and 0 <= group < len(_A6W4_GROUPS)
                and 0 <= index < len(_A6W4_GROUPS[group])
                and _A6W4_GROUPS[group][index] is src
            ):
                _A6W4_GROUPS[group][index] = dst
            return dst

        if isinstance(out, cls):
            return carry(out)
        if isinstance(out, list):
            return [carry(x) if isinstance(x, cls) else x for x in out]
        if isinstance(out, tuple):
            return tuple(carry(x) if isinstance(x, cls) else x for x in out)
        return out

    def _a6w4_pack(self, param_dtype: torch.dtype):
        from primus_turbo.pytorch.ops.flux_a6w4 import quantizers

        quant_w_a6w4 = quantizers()[1]
        w = self._tensor.to(param_dtype).contiguous()
        if w.shape[0] % MX_BLOCK_SIZE:
            raise ValueError(
                f"A6W4 all-gather needs the FSDP shard height {w.shape[0]} to be a multiple of "
                f"{MX_BLOCK_SIZE}; set FLUX_FP4_ALL_GATHER=0."
            )
        n, k = w.shape
        row_q, row_s = quant_w_a6w4(w.to(torch.bfloat16))
        col_q, col_s = _quantize_mxfp4_h16_col_op(w, False)
        # col_s is [K, n/32], odd-width at 32 ranks for N = 3072 / 9216 / 21504; two rows per
        # payload row keep it bf16-viewable so the group's gather stays one dtype.
        return tuple(
            _wide_view(t.view(torch.uint8))
            for t in (
                row_q.view(n, k // 2),
                row_s.view(n, k // MX_BLOCK_SIZE),
                col_q,
                col_s.view(torch.uint8).reshape(k // 2, -1),
            )
        )

    def _a6w4_take_or_pack(self, param_dtype: torch.dtype):
        cached = getattr(self, "_prequant", None)
        self._prequant = None
        if cached is not None and cached[0] == _A6W4_PASS and cached[1] == param_dtype:
            payloads, ready = cached[2], cached[3]
            if ready is not None:
                stream = torch.cuda.current_stream()
                stream.wait_event(ready)
                for t in payloads:
                    t.record_stream(stream)
            return payloads
        return self._a6w4_pack(param_dtype)

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        param_dtype = (mp_policy.param_dtype if mp_policy is not None else None) or self._tensor.dtype
        global _A6W4_PASS, _A6W4_SIDE_ARMED, _A6W4_MISS_LOGGED
        if _A6W4_AG_PIPELINE and not _A6W4_BOUND:
            _a6w4_bind_groups(_A6W4_NAMED)
        group = getattr(self, "_ag_group", None) if _A6W4_AG_PIPELINE else None
        if group is None:
            if _A6W4_AG_PIPELINE and not _A6W4_MISS_LOGGED:
                _A6W4_MISS_LOGGED = True
                logger.warning(
                    "FLUX_A6W4_AG_PIPELINE: the gathered shard has no group index; "
                    "packs stay on the copy-in stream"
                )
            return self._a6w4_pack(param_dtype), (mesh.size(),)
        if group == 0 and self._ag_index == 0:
            _A6W4_PASS += 1
            # The next kick waits on the default stream once. Later kicks in
            # this step only append to the side stream, so they do not wait
            # for compute queued after this forward began.
            _A6W4_SIDE_ARMED = False
        payloads = self._a6w4_take_or_pack(param_dtype)
        if self._ag_index == 0:
            _a6w4_kick(group + 1, param_dtype)
        return payloads, (mesh.size(),)

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        (world,) = metadata
        if out is not None:
            return
        row_q, row_s, col_q, col_s = (t.view(torch.uint8) for t in all_gather_outputs)
        col_s = col_s.view(world * row_q.shape[1] * 2, -1)
        return (
            _A6W4GatheredWeight(
                row_q, row_s, col_q, col_s, world, param_dtype, getattr(self, "_ag_kmajor", False)
            ),
            tuple(all_gather_outputs),
        )


class _A6W4GatheredWeight(_MXFP4GatheredWeight):
    """`_MXFP4GatheredWeight` whose row pack is quant_w_a6w4's preshuffled bytes. Serves the
    A6W4 fprop GEMM and the Dgrad col pack; the row pack is not an MXFP4 GEMM operand."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.primus_flux.a6w4_mm.default:
            a, self = args[0], args[1]
            bias = args[2] if len(args) > 2 else (kwargs or {}).get("bias")
            return torch.ops.primus_flux.a6w4_mm_prequant_w(a, self._row_q, self._row_s, self.shape[0], bias)
        if func == torch.ops.primus_flux.a6w4_mm_qa.default:
            aq, sa, self = args[0], args[1], args[2]
            bias = args[3] if len(args) > 3 else (kwargs or {}).get("bias")
            return torch.ops.primus_flux.a6w4_mm_qa_prequant_w(
                aq, sa, self._row_q, self._row_s, self.shape[0], bias
            )
        if func == torch.ops.primus_flux.a6w4_mm_gelu_dual_qa_w.default:
            aq, sa, self, bias, act_rounded = args
            return torch.ops.primus_flux.a6w4_mm_gelu_dual_qa_prequant_w(
                aq, sa, self._row_q, self._row_s, self.shape[0], bias, act_rounded
            )
        if func == torch.ops.primus_flux.quantize_mxfp4_h16_dual.default:
            raise NotImplementedError("_A6W4GatheredWeight's row pack is in the A6W4 GEMM layout")
        return super().__torch_dispatch__(func, types, args, kwargs)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        return _A6W4GatheredWeight(
            inner_tensors["_row_q"],
            inner_tensors["_row_s"],
            inner_tensors["_col_q"],
            inner_tensors["_col_s"],
            *ctx,
        )


def precompute_fp8_all_gather_amax(model: nn.Module) -> None:
    """Refresh every FP8-gathered weight's global amax: one fused norm, one all-reduce.

    max|bf16(w)| == bf16(max|w|) because rounding is monotonic, so the fp32 shard's
    inf-norm rounded to bf16 is exactly the amax of what the gather quantizes.
    """
    shards, groups = [], []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None) if isinstance(module, MXFP4Linear) else None
        local = getattr(weight, "_local_tensor", weight)
        if isinstance(local, _FP8AllGatherWeight):
            shards.append(local)
            groups.append(_prequant_group(name))
    if not shards:
        return
    global _FP8_AG_FUSED_QUANT_LOGGED
    if _FP8_AG_FUSED_QUANT and not _FP8_AG_FUSED_QUANT_LOGGED:
        from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
            fp8_dual_eligible,
        )

        _FP8_AG_FUSED_QUANT_LOGGED = True
        fused = sum(fp8_dual_eligible(*s._tensor.shape) for s in shards)
        logger.info(
            f"FLUX_FP8_AG_FUSED_QUANT: {fused}/{len(shards)} FP8-gathered shards take the fused quantize"
        )
    norms = torch._foreach_norm([s._tensor for s in shards], float("inf"))
    amax = torch.stack(norms).to(torch.bfloat16).float()
    torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX)
    if _FP8_AG_PREQUANT:
        _prequantize_fp8_all_gather(shards, groups, amax)
        return
    for i, shard in enumerate(shards):
        shard._amax = amax[i : i + 1]


# Quantize every FP8-gathered shard right after the optimizer, on a side stream, instead of inside
# the forward just before each block's all-gather, where the per-weight launches sit on the CPU-
# bound double blocks' critical chain. Same ops, same inputs, so bit-identical.
# The graph reads the shards in place and the amax from a static copy; its outputs are reused
# every step, which is safe because this runs after the optimizer, long after the previous
# forward's all-gathers have read them.
# FLUX_FP8_AG_PREQUANT=gather keeps the quantize where it always ran -- inside each block's all-
# gather, on FSDP's copy-in stream -- and only turns that block's ~16 launches into one graph
# replay, so the GPU schedule is the stock one minus the launch gaps. The replay is skipped when
# no optimizer step happened since the last one, because the shards and amax are unchanged.
_FP8_AG_PREQUANT = os.getenv("FLUX_FP8_AG_PREQUANT", "0") in ("1", "gather")
_PREQUANT_AT_GATHER = os.getenv("FLUX_FP8_AG_PREQUANT", "0") == "gather"
# Diagnostic: re-quantize in every forward and require the precomputed payload to match.
_FP8_AG_PREQUANT_CHECK = os.getenv("FLUX_FP8_AG_PREQUANT_CHECK", "0") == "1"
_FP8_AG_PREQUANT_GRAPH = os.getenv("FLUX_FP8_AG_PREQUANT_GRAPH", "1") == "1"
_PREQUANT_STREAM = None
_PREQUANT_CALLS = 0
_PREQUANT_AMAX = None
_PREQUANT_GRAPHS = None  # [[graph, [shard indices], [(payloads, scale_inv)], replay pending]]


def _prequant_group(name: str) -> str:
    """The FSDP unit a weight is gathered with: ``double_blocks.3`` / ``single_blocks.12``
    wherever it sits in the qualified name, else the module's parent."""
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i].endswith("blocks") and parts[i + 1].isdigit():
            return ".".join(parts[: i + 2])
    return ".".join(parts[:-1])


def _a6w4_held(module: nn.Module):
    weight = getattr(module, "weight", None)
    # After fully_shard the module parameter is a DTensor. The pre-all-gather
    # hook runs on its local tensor, which is the shard this rank packs.
    local = getattr(weight, "_local_tensor", None)
    if isinstance(local, _A6W4AllGatherWeight):
        return local
    return weight if isinstance(weight, _A6W4AllGatherWeight) else None


def _a6w4_bind_groups(named: list) -> None:
    """Record A6W4 shards in forward order, one list per FSDP unit."""
    global _A6W4_PIPE_LOGGED, _A6W4_BOUND
    _A6W4_BOUND = True
    buckets: dict[str, list] = {}
    order: list[str] = []
    missed = 0
    for name, module in named:
        held = _a6w4_held(module)
        if held is None:
            missed += 1
            continue
        key = _prequant_group(name)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        held._ag_group = len(order) - 1
        held._ag_index = len(buckets[key])
        held._prequant = None
        buckets[key].append(held)
    _A6W4_GROUPS[:] = [buckets[key] for key in order]
    if _A6W4_PIPE_LOGGED:
        return
    _A6W4_PIPE_LOGGED = True
    n = sum(len(group) for group in _A6W4_GROUPS)
    extra = f"; {missed} weights stayed inline" if missed else ""
    logger.info(
        f"FLUX_A6W4_AG_PIPELINE: {n} weights in {len(_A6W4_GROUPS)} groups; "
        f"group i+1 is packed while group i is gathered{extra}"
    )


def _a6w4_side() -> torch.cuda.Stream:
    global _A6W4_SIDE
    if _A6W4_SIDE is None:
        _A6W4_SIDE = torch.cuda.Stream()
    return _A6W4_SIDE


def _a6w4_kick(group_index: int, param_dtype: torch.dtype) -> None:
    """Pack one FSDP unit on a side stream. The collective of the unit now in
    the hook is queued after this returns, so the pack overlaps that gather."""
    if group_index < 0 or group_index >= len(_A6W4_GROUPS):
        return
    if _A6W4_KICKED.get(group_index) == _A6W4_PASS:
        return
    _A6W4_KICKED[group_index] = _A6W4_PASS
    global _A6W4_SIDE_ARMED
    side = _a6w4_side()
    # Arm once per step. FSDP waits the copy-in stream on the compute stream
    # at the root pre-forward, then reuses that stream for every group. Waiting
    # again here would also wait for GEMMs queued since that point, and the
    # pack would miss the gather it is supposed to overlap.
    if not _A6W4_SIDE_ARMED:
        side.wait_stream(torch.cuda.default_stream())
        _A6W4_SIDE_ARMED = True
    shards = _A6W4_GROUPS[group_index]
    with torch.cuda.stream(side):
        packed = [shard._a6w4_pack(param_dtype) for shard in shards]
        ready = side.record_event()
    for shard, payloads in zip(shards, packed):
        shard._prequant = (_A6W4_PASS, param_dtype, payloads, ready)


def _capture_prequant_graphs(shards: list, groups: list, stream) -> list:
    order = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    pool = torch.cuda.graph_pool_handle()
    graphs = []
    for idx in order.values():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
            outs = [shards[i]._quantize(torch.bfloat16) for i in idx]
        # [graph, shard indices, their (payloads, scale_inv), replay pending]
        graphs.append([graph, idx, outs, False])
    return graphs


def _prequantize_fp8_all_gather(shards: list, groups: list, amax: torch.Tensor) -> None:
    global _PREQUANT_STREAM, _PREQUANT_CALLS, _PREQUANT_AMAX, _PREQUANT_GRAPHS
    if _PREQUANT_STREAM is None:
        _PREQUANT_STREAM = torch.cuda.Stream()
        _PREQUANT_AMAX = torch.empty_like(amax)
    _PREQUANT_CALLS += 1
    if _PREQUANT_CALLS in (2, 3) and not _PREQUANT_AT_GATHER:
        # A payload the forward never took means FSDP gathered through another wrapper object.
        used = sum(s._prequant is None for s in shards)
        logger.info(f"FLUX_FP8_AG_PREQUANT: forward used {used}/{len(shards)} precomputed payloads")
    if _FP8_AG_PREQUANT_CHECK and _PREQUANT_CALLS > 1 and _PREQUANT_CALLS % 10 == 1:
        logger.info(f"FLUX_FP8_AG_PREQUANT_CHECK: payloads of the first {_PREQUANT_CALLS - 1} steps matched")
    stream = _PREQUANT_STREAM
    if _PREQUANT_AT_GATHER:
        # The gathers read the amax on FSDP's copy-in stream, which waits on this one first.
        _PREQUANT_AMAX.copy_(amax)
        for i, shard in enumerate(shards):
            shard._amax = _PREQUANT_AMAX[i : i + 1]
        # The first step quantizes eagerly inside the gathers, compiling every kernel.
        if _PREQUANT_GRAPHS is None and _PREQUANT_CALLS >= 2:
            _PREQUANT_GRAPHS = _capture_prequant_graphs(shards, groups, stream)
            for entry in _PREQUANT_GRAPHS:
                for k, i in enumerate(entry[1]):
                    shards[i]._pq_slot = (entry, k)
            logger.info(
                f"FLUX_FP8_AG_PREQUANT=gather: captured {len(_PREQUANT_GRAPHS)} graphs for {len(shards)} shards"
            )
        for entry in _PREQUANT_GRAPHS or ():
            entry[3] = True
        return
    stream.wait_stream(torch.cuda.current_stream())
    amax.record_stream(stream)
    with torch.cuda.stream(stream):
        _PREQUANT_AMAX.copy_(amax)
    for i, shard in enumerate(shards):
        shard._amax = _PREQUANT_AMAX[i : i + 1]
    # The first call runs eagerly so every kernel is compiled before capture.
    if _FP8_AG_PREQUANT_GRAPH and _PREQUANT_GRAPHS is None and _PREQUANT_CALLS >= 2:
        _PREQUANT_GRAPHS = _capture_prequant_graphs(shards, groups, stream)
        logger.info(f"FLUX_FP8_AG_PREQUANT: captured {len(_PREQUANT_GRAPHS)} graphs for {len(shards)} shards")
    if _PREQUANT_GRAPHS is None:
        with torch.cuda.stream(stream):
            for shard in shards:
                payloads, scale_inv = shard._quantize(torch.bfloat16)
                shard._prequant = (payloads, scale_inv, torch.bfloat16, stream.record_event())
        return
    with torch.cuda.stream(stream):
        for graph, idx, outs, _ in _PREQUANT_GRAPHS:
            graph.replay()
            ready = stream.record_event()
            for i, (payloads, scale_inv) in zip(idx, outs):
                shards[i]._prequant = (payloads, scale_inv, torch.bfloat16, ready)


# --------------------------------------------------------------------------- FSDP2 reduce-scatter
# copy-in off the compute stream (FLUX_FSDP_RS_COPY_IN=side)
# ---------------------------------------------------------------------------
# FSDP2 packs each group's unsharded grads into the reduce-scatter input with `_chunk_cat` on the
# compute stream, ~8 ms/step inside the GPU-bound backward. The same copy runs here on its own
# stream: it waits for the grads and the reduce-scatter stream waits for it. The bytes are
# chunk_cat's own.
# No record_stream: a recorded block is only reused once the GPU reaches an event taken when the
# CPU frees it, so with the CPU steps ahead every step's grads and inputs stay parked and the
# caching allocator grows until RCCL and HSA have no device memory left. The input is already
# kept by FSDP2 until the compute stream waits on its reduce-scatter. The grads are held here and
# released on the next copy-in, after the compute stream waits on the copy that read them.
# copy_in is not handed the reduce-scatter stream, so the foreach_reduce wrapper publishes it for
# the duration of the call.

_RS_STREAM = None
_RS_COPY_STREAM = None
_RS_COPY_CHECKS = 0
_RS_COPY_PENDING = None


def _foreach_reduce_publishing_stream(*args, **kwargs):
    global _RS_STREAM
    _RS_STREAM = args[3] if len(args) > 3 else kwargs["reduce_scatter_stream"]
    try:
        return _FOREACH_REDUCE_ORIG(*args, **kwargs)
    finally:
        _RS_STREAM = None


def _rs_copy_in_side_stream(unsharded_grads, reduce_scatter_input, world_size):
    global _RS_COPY_STREAM, _RS_COPY_CHECKS, _RS_COPY_PENDING
    if _RS_STREAM is None:
        return _RS_COPY_IN_ORIG(unsharded_grads, reduce_scatter_input, world_size)
    if _RS_COPY_STREAM is None:
        _RS_COPY_STREAM = torch.cuda.Stream()
        logger.info(
            f"FLUX_FSDP_RS_COPY_IN=side: reduce-scatter copy-in on its own stream (check {_RS_COPY_CHECK})"
        )
    copy_stream, current = _RS_COPY_STREAM, torch.cuda.current_stream()
    if _RS_COPY_PENDING is not None:
        current.wait_event(_RS_COPY_PENDING[0])
        _RS_COPY_PENDING = None
    copy_stream.wait_stream(current)
    with torch.cuda.stream(copy_stream):
        _RS_COPY_IN_ORIG(unsharded_grads, reduce_scatter_input, world_size)
        _RS_COPY_PENDING = (copy_stream.record_event(), list(unsharded_grads))
    _RS_STREAM.wait_stream(copy_stream)
    if _RS_COPY_CHECK and _RS_COPY_CHECKS < 64:
        _RS_COPY_CHECKS += 1
        ref = torch.empty_like(reduce_scatter_input)
        _RS_COPY_IN_ORIG(unsharded_grads, ref, world_size)
        current.wait_stream(copy_stream)
        if not torch.equal(ref.view(torch.uint8), reduce_scatter_input.view(torch.uint8)):
            raise RuntimeError("FLUX_FSDP_RS_COPY_IN=side: side-stream copy-in differs from chunk_cat")


_RS_COPY_CHECK = os.getenv("FLUX_FSDP_RS_COPY_CHECK", "0") == "1"
_RS_COPY_IN_ORIG = _FOREACH_REDUCE_ORIG = None
if os.getenv("FLUX_FSDP_RS_COPY_IN", "chunk_cat") == "side":
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_collectives
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as _fsdp_param_group

    _RS_COPY_IN_ORIG = _fsdp_collectives.foreach_reduce_scatter_copy_in
    _FOREACH_REDUCE_ORIG = _fsdp_param_group.foreach_reduce
    _fsdp_collectives.foreach_reduce_scatter_copy_in = _rs_copy_in_side_stream
    _fsdp_param_group.foreach_reduce = _foreach_reduce_publishing_stream


def precompute_float8_dynamic_scale_for_fsdp(model: nn.Module) -> None:
    """TorchAO's per-step FP8 all-gather scale refresh, plus this module's."""
    from torchao.float8 import (
        precompute_float8_dynamic_scale_for_fsdp as torchao_precompute,
    )

    torchao_precompute(model)
    precompute_fp8_all_gather_amax(model)

