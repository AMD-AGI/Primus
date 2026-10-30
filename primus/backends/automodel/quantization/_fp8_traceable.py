###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Compile-friendly tensorwise FP8 GEMMs for the Primus-Turbo FP8 linear swap.

WHY:
  Primus-Turbo's ``gemm_fp8`` builds ``QuantizedTensor`` wrapper subclasses inside
  its autograd forward and is ``torch._dynamo.disable``d because Dynamo cannot
  trace them, so every swapped linear is a graph break. On FLUX.1-schnell that was
  32 Dynamo breaks, 40 Inductor graphs and 436 device-to-device copies per step.
  The autograd functions here call the same kernels (``quantize_fp8_tensorwise``,
  ``transpose_2d``, ``gemm_fp8_impl``) but carry the quantized data and its inverse
  scale as plain tensors, so the GEMMs stay inside the compiled graph. Forward and
  all three gradients are bit-identical to ``Float8Linear``.

WHAT (settings in the ``primus_turbo:`` section, all off by default):
  fp8_linear_traceable: false            swap in TraceableFloat8Linear
  fp8_linear_delayed_scaling: false      reuse each operand's scale between refreshes
  fp8_linear_amax_warmup_steps: 2        forwards that always refresh the scales
  fp8_linear_amax_refresh_interval: 16   then refresh every n-th forward (0: never)
  fp8_linear_inductor_cast: false        scaled casts as plain ops Inductor can fuse
  fp8_linear_shared_input_scales: false  quantize an input shared by several linears once
  fp8_linear_gemm_backend: hipblaslt     default FP8 GEMM backend: hipblaslt|flydsl|ck|triton

  Delayed scaling keeps one fp32 scale per operand (input, weight, output
  gradient) in non-persistent buffers. A refresh forward measures the amax as the
  dynamic path does and stores its reciprocal; every other forward casts with the
  stored scale, which skips the amax pass. The schedule advances once per
  transformer forward, so with gradient accumulation it counts micro-batches.
  Buffers are not checkpointed: a resumed run re-warms.

  The Inductor cast writes a scaled cast as multiply, clamp, cast and transpose,
  which Inductor folds into the kernel that produced the activation instead of
  launching the extension's quantize and transpose. The bytes are identical.

  Shared input scales: linears that read the same tensor (Q/K/V, a single
  block's attention and MLP input) already hold equal scales, since each refresh
  derives them from that tensor's amax. Making them alias one buffer lets the
  compiler see identical casts and keep one; only the group's first linear writes
  the scale. Groups are found in one extra no-grad, eager forward before the
  first step. Without the Inductor cast the extension's quantize ops are opaque
  to the compiler and nothing is deduplicated.

  The GEMM backend is passed to Primus-Turbo as the default, so a shape that
  backend cannot run (FlyDSL needs K > 128) falls back to another one by itself.
  Auto-tune (``PRIMUS_TURBO_AUTO_TUNE``) and ``PRIMUS_TURBO_GEMM_BACKEND`` take
  precedence over a default, so with either set this setting has no effect.

Tensorwise granularity on gfx950 only: gfx942 runs the backward in a different
layout, and other granularities keep ``Float8Linear``.
"""
from __future__ import annotations

import collections
import contextlib
import functools
import logging
import weakref
from typing import NamedTuple, Optional

import torch

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusTurbo-FP8]"
_GEMM_BACKENDS = ("hipblaslt", "flydsl", "ck", "triton")
_SCALE_INV_FLOOR = 1e-30
_SCALE_BUFFERS = ("_fp8_scale_x", "_fp8_scale_w", "_fp8_scale_g")

# Bound by ``resolve`` when the swap runs, before anything is compiled, so the
# traced code reads them as constants.
_DELAYED_SCALE_STATE = {"step": 0, "refresh": True}
_AMAX_WARMUP_STEPS = 2
_AMAX_REFRESH_INTERVAL = 16
_INDUCTOR_CAST = False
_GEMM_BACKEND: Optional[int] = None
_QUANTIZE_EXTRA_ARGS: tuple = ()


class Settings(NamedTuple):
    delayed_scaling: bool
    amax_warmup_steps: int
    amax_refresh_interval: int
    inductor_cast: bool
    shared_input_scales: bool
    gemm_backend: str


def requested() -> bool:
    return options.flag("primus_turbo.fp8_linear_traceable")


def read_settings() -> Settings:
    """The settings, validated. Raises on a value or combination that cannot apply.

    A dependent option set without what it depends on raises rather than being
    ignored, for the reason ``fp8_linear.resolve_config`` raises on a typo: an
    A/B arm that silently measures a different configuration is worse than a
    failed launch.
    """
    traceable = requested()
    delayed = options.flag("primus_turbo.fp8_linear_delayed_scaling")
    warmup = options.integer("primus_turbo.fp8_linear_amax_warmup_steps", 2)
    interval = options.integer("primus_turbo.fp8_linear_amax_refresh_interval", 16)
    inductor_cast = options.flag("primus_turbo.fp8_linear_inductor_cast")
    shared = options.flag("primus_turbo.fp8_linear_shared_input_scales")
    backend = options.string("primus_turbo.fp8_linear_gemm_backend", "hipblaslt").lower()

    if backend not in _GEMM_BACKENDS:
        raise ValueError(
            f"primus_turbo.fp8_linear_gemm_backend={backend!r} invalid; expected one of {', '.join(_GEMM_BACKENDS)}"
        )
    if warmup < 0 or interval < 0:
        raise ValueError(
            "primus_turbo.fp8_linear_amax_warmup_steps and fp8_linear_amax_refresh_interval must be "
            f"non-negative, got {warmup} and {interval}"
        )
    if delayed and warmup == 0 and interval == 0:
        raise ValueError(
            "primus_turbo.fp8_linear_amax_warmup_steps and fp8_linear_amax_refresh_interval are both 0, "
            "so the delayed scales would never be measured"
        )

    def require(present: bool, needed: str, keys) -> None:
        for key, value in keys:
            if value and not present:
                raise ValueError(f"primus_turbo.{key} needs primus_turbo.{needed}: true")

    require(
        traceable,
        "fp8_linear_traceable",
        (
            ("fp8_linear_delayed_scaling", delayed),
            ("fp8_linear_inductor_cast", inductor_cast),
            ("fp8_linear_shared_input_scales", shared),
            ("fp8_linear_gemm_backend", backend != "hipblaslt"),
        ),
    )
    require(
        delayed,
        "fp8_linear_delayed_scaling",
        (("fp8_linear_inductor_cast", inductor_cast), ("fp8_linear_shared_input_scales", shared)),
    )
    return Settings(delayed, warmup, interval, inductor_cast, shared, backend)


def supported(config) -> tuple[bool, str]:
    """Whether the traceable path covers this quantization config on this device."""
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.core.utils import is_gfx942

    if config.granularity != ScalingGranularity.TENSORWISE:
        return False, f"granularity={config.granularity.name} (only TENSORWISE)"
    if is_gfx942():
        return False, "gfx942 runs the FP8 backward in the non-NT layout"
    return True, ""


def resolve(config) -> Optional[Settings]:
    """Validate and bind the settings for the swap that is about to run.

    Returns the settings when the traceable path applies, ``None`` when it was not
    requested or does not support ``config`` (logged; the swap keeps
    ``Float8Linear``). Raises on invalid settings either way.
    """
    global _AMAX_WARMUP_STEPS, _AMAX_REFRESH_INTERVAL, _INDUCTOR_CAST, _GEMM_BACKEND, _QUANTIZE_EXTRA_ARGS

    settings = read_settings()
    if not requested():
        return None
    ok, reason = supported(config)
    if not ok:
        logger.warning(
            "%s primus_turbo.fp8_linear_traceable ignored: %s. The swap uses Float8Linear and the "
            "delayed-scaling, cast and backend settings do not apply.",
            _LOG_PREFIX,
            reason,
        )
        return None

    from primus_turbo.pytorch.core.backend import BackendType

    _DELAYED_SCALE_STATE.update(step=0, refresh=True)
    _AMAX_WARMUP_STEPS = settings.amax_warmup_steps
    _AMAX_REFRESH_INTERVAL = settings.amax_refresh_interval
    _INDUCTOR_CAST = settings.inductor_cast
    _GEMM_BACKEND = BackendType[settings.gemm_backend.upper()].value
    _QUANTIZE_EXTRA_ARGS = _unpadded_quantize_args()

    logger.info(
        "%s TraceableFloat8Linear: FP8 GEMMs stay inside compiled graphs (default GEMM backend %s)",
        _LOG_PREFIX,
        settings.gemm_backend,
    )
    if settings.gemm_backend != "hipblaslt":
        _warn_if_backend_overridden(settings.gemm_backend)
    if settings.delayed_scaling:
        logger.info(
            "%s delayed tensorwise scaling: refresh the first %d forwards and every %d after",
            _LOG_PREFIX,
            settings.amax_warmup_steps,
            settings.amax_refresh_interval,
        )
    if settings.inductor_cast:
        logger.info("%s Inductor-fusible cast+transpose on scaled forwards", _LOG_PREFIX)
    if settings.shared_input_scales:
        logger.info(
            "%s shared input scales: groups are found in one eager forward before the first step", _LOG_PREFIX
        )
        if not settings.inductor_cast:
            logger.warning(
                "%s primus_turbo.fp8_linear_shared_input_scales has no effect without "
                "primus_turbo.fp8_linear_inductor_cast: the extension's quantize ops are opaque to the "
                "compiler, so the shared casts are not deduplicated.",
                _LOG_PREFIX,
            )
    return settings


def _warn_if_backend_overridden(backend: str) -> None:
    from primus_turbo.pytorch.core.backend import GlobalBackendManager, PrecisionType

    if GlobalBackendManager.auto_tune_enabled():
        logger.warning(
            "%s Primus-Turbo auto-tune is on and picks the FP8 GEMM backend per shape, so "
            "primus_turbo.fp8_linear_gemm_backend=%s only applies where auto-tune finds no backend.",
            _LOG_PREFIX,
            backend,
        )
    choice = GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)
    if choice is not None and choice.backend is not None:
        logger.warning(
            "%s PRIMUS_TURBO_GEMM_BACKEND pins the FP8 GEMMs to %s, which overrides "
            "primus_turbo.fp8_linear_gemm_backend=%s.",
            _LOG_PREFIX,
            choice.backend.name,
            backend,
        )


def _unpadded_quantize_args() -> tuple:
    """Extra arguments that keep ``quantize_fp8_tensorwise`` from padding its output.

    From Primus-Turbo 0.4 the op takes ``padding_align_size`` and
    ``pad_penultimate_align_size`` after the scale and pads K to a multiple of 128
    by default. The autograd functions here need the unpadded layout, which is
    what ``Float8Linear`` quantizes to: a padded operand gives FLUX's x_embedder
    (K = 64) a gradient of the wrong shape. Older versions take neither argument.
    """
    op = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise
    return (1, 1) if len(op.default._schema.arguments) >= 5 else ()


def fp8_stage_dtype(fmt, is_fwd_stage: bool):
    """The FP8 dtype a Primus-Turbo format uses for one GEMM stage."""
    from primus_turbo.pytorch.core.low_precision import Format, float8_e4m3, float8_e5m2

    if fmt == Format.E4M3:
        return float8_e4m3
    if fmt == Format.E5M2:
        return float8_e5m2
    return float8_e4m3 if is_fwd_stage else float8_e5m2


def _quantize(x, fp8_dtype, scale=None):
    """Tensorwise FP8 cast: with ``scale=None`` it measures the amax, as ``Float8Linear`` does."""
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
        x.contiguous(), fp8_dtype, scale, *_QUANTIZE_EXTRA_ARGS
    )


def _transpose(quantized):
    return torch.ops.primus_turbo_cpp_extension.transpose_2d(quantized, 0, 1).contiguous()


def _cast_transpose_torch(x, fp8_dtype, scale):
    """The extension's scaled cast as plain ops, which Inductor fuses into x's producer."""
    fp8_max = torch.finfo(fp8_dtype).max
    quantized = (x.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return quantized, quantized.t().contiguous(), torch.reciprocal(scale)


def cast_transpose_scaled(x, fp8_dtype, scale):
    """Quantize ``x`` with a known scale: data, its transpose, inverse scale."""
    if _INDUCTOR_CAST:
        return _cast_transpose_torch(x, fp8_dtype, scale)
    quantized, scale_inv = _quantize(x, fp8_dtype, scale)
    return quantized, _transpose(quantized), scale_inv


def _stash_scale(buffer, scale_inv) -> None:
    # An all-zero operand returns scale_inv = 0; its reciprocal would make every
    # later cast NaN, steps after the cause.
    buffer.copy_(scale_inv.clamp(min=_SCALE_INV_FLOOR).reciprocal())


def _gemm_kwargs(granularity):
    return {"granularity": granularity, "default_backend": _GEMM_BACKEND}


class _TraceableFP8GemmNT(torch.autograd.Function):
    """Tensorwise ``a @ b.T`` without Primus-Turbo tensor subclasses."""

    @staticmethod
    def forward(ctx, a, b, out_dtype, fwd_dtype, bwd_dtype, granularity):
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

        aq, a_scale_inv = _quantize(a, fwd_dtype)
        bq, b_scale_inv = _quantize(b, fwd_dtype)
        out = gemm_fp8_impl(
            aq, a_scale_inv, False, bq, b_scale_inv, True, out_dtype, False, **_gemm_kwargs(granularity)
        )
        ctx.save_for_backward(_transpose(aq), a_scale_inv, _transpose(bq), b_scale_inv)
        ctx.out_dtype = out_dtype
        ctx.bwd_dtype = bwd_dtype
        ctx.granularity = granularity
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

        aq_t, a_scale_inv, bq_t, b_scale_inv = ctx.saved_tensors
        gq, g_scale_inv = _quantize(grad_out, ctx.bwd_dtype)
        kwargs = _gemm_kwargs(ctx.granularity)
        grad_a = gemm_fp8_impl(
            gq, g_scale_inv, False, bq_t, b_scale_inv, True, ctx.out_dtype, False, **kwargs
        )
        grad_b = gemm_fp8_impl(
            aq_t, a_scale_inv, False, _transpose(gq), g_scale_inv, True, ctx.out_dtype, True, **kwargs
        )
        return grad_a, grad_b, None, None, None, None


class _TraceableFP8GemmNTDelayed(torch.autograd.Function):
    """``_TraceableFP8GemmNT`` whose operand scales are carried across steps."""

    @staticmethod
    def forward(
        ctx,
        a,
        b,
        scale_a,
        scale_b,
        scale_g,
        out_dtype,
        fwd_dtype,
        bwd_dtype,
        granularity,
        refresh,
        update_scale_a=True,
    ):
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

        if refresh:
            aq, a_scale_inv = _quantize(a, fwd_dtype)
            bq, b_scale_inv = _quantize(b, fwd_dtype)
            if update_scale_a:
                _stash_scale(scale_a, a_scale_inv)
            _stash_scale(scale_b, b_scale_inv)
            aq_t, bq_t = _transpose(aq), _transpose(bq)
        else:
            aq, aq_t, a_scale_inv = cast_transpose_scaled(a, fwd_dtype, scale_a)
            bq, bq_t, b_scale_inv = cast_transpose_scaled(b, fwd_dtype, scale_b)
        out = gemm_fp8_impl(
            aq, a_scale_inv, False, bq, b_scale_inv, True, out_dtype, False, **_gemm_kwargs(granularity)
        )
        ctx.save_for_backward(aq_t, a_scale_inv, bq_t, b_scale_inv, scale_g)
        ctx.out_dtype = out_dtype
        ctx.bwd_dtype = bwd_dtype
        ctx.granularity = granularity
        ctx.refresh = refresh
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

        aq_t, a_scale_inv, bq_t, b_scale_inv, scale_g = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if ctx.refresh:
            gq, g_scale_inv = _quantize(grad_out, ctx.bwd_dtype)
            _stash_scale(scale_g, g_scale_inv)
            gq_t = _transpose(gq)
        else:
            gq, gq_t, g_scale_inv = cast_transpose_scaled(grad_out, ctx.bwd_dtype, scale_g)
        kwargs = _gemm_kwargs(ctx.granularity)
        grad_a = gemm_fp8_impl(
            gq, g_scale_inv, False, bq_t, b_scale_inv, True, ctx.out_dtype, False, **kwargs
        )
        grad_b = gemm_fp8_impl(
            aq_t, a_scale_inv, False, gq_t, g_scale_inv, True, ctx.out_dtype, True, **kwargs
        )
        return grad_a, grad_b, None, None, None, None, None, None, None, None, None


def make_float8_linear(base_cls, *, delayed_scaling: bool):
    """A ``Float8Linear`` subclass whose FP8 GEMMs remain traceable."""

    class TraceableFloat8Linear(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fp8_stage_dtypes = (
                fp8_stage_dtype(self.config.format, True),
                fp8_stage_dtype(self.config.format, False),
            )
            if delayed_scaling:
                for name in _SCALE_BUFFERS:
                    self.register_buffer(
                        name,
                        torch.ones((), dtype=torch.float32, device=self.weight.device),
                        persistent=False,
                    )
                self._fp8_scale_x_owner = True

        def forward(self, x):
            output_shape = x.shape[:-1]
            activations = x.reshape(-1, self.in_features)
            fwd_dtype, bwd_dtype = self._fp8_stage_dtypes
            granularity = self.config.granularity.value
            if delayed_scaling:
                out = _TraceableFP8GemmNTDelayed.apply(
                    activations,
                    self.weight,
                    self._fp8_scale_x,
                    self._fp8_scale_w,
                    self._fp8_scale_g,
                    x.dtype,
                    fwd_dtype,
                    bwd_dtype,
                    granularity,
                    _DELAYED_SCALE_STATE["refresh"],
                    self._fp8_scale_x_owner,
                )
            else:
                out = _TraceableFP8GemmNT.apply(
                    activations, self.weight, x.dtype, fwd_dtype, bwd_dtype, granularity
                )
            if self.bias is not None:
                out = out + self.bias
            return out.view(*output_shape, self.out_features)

    return TraceableFloat8Linear


def _advance_delayed_scale_step() -> None:
    """Decide once per transformer forward whether this forward refreshes the scales."""
    state = _DELAYED_SCALE_STATE
    step = state["step"]
    interval = _AMAX_REFRESH_INTERVAL
    state["refresh"] = step < _AMAX_WARMUP_STEPS or (interval > 0 and step % interval == 0)
    state["step"] = step + 1
    logger.debug("%s delayed-scale forward %d: refresh=%s", _LOG_PREFIX, step, state["refresh"])


@contextlib.contextmanager
def discover_shared_inputs(root):
    """Make FP8 linears that consume the same tensor share one input-scale buffer.

    Run one eager forward inside this context. A tensor is matched by object
    identity through a weakref, so a freed tensor whose memory is reused is never
    mistaken for the original. A linear called more than once per forward is left
    out, because its scale reflects whichever input it saw last.
    """
    linears = [m for m in root.modules() if hasattr(m, "_fp8_scale_x")]
    seen = []
    owner_of = {}
    calls = collections.Counter()

    def hook(module, args):
        calls[module] += 1
        x = args[0] if args else None
        if not isinstance(x, torch.Tensor):
            return
        for ref, owner in seen:
            if ref() is x:
                owner_of[module] = owner
                return
        seen.append((weakref.ref(x), module))
        owner_of[module] = module

    handles = [m.register_forward_pre_hook(hook) for m in linears]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

    groups = collections.defaultdict(list)
    for module, owner in owner_of.items():
        groups[owner].append(module)
    shared = 0
    num_groups = 0
    for owner, members in groups.items():
        if len(members) < 2 or any(calls[m] != 1 for m in members):
            continue
        num_groups += 1
        for member in members:
            if member is owner:
                continue
            member._fp8_scale_x = owner._fp8_scale_x
            member._fp8_scale_x_owner = False
            shared += 1
    logger.info(
        "%s shared input scales: %d linears in %d groups now reuse their group's quantized input "
        "(%d FP8 linears seen)",
        _LOG_PREFIX,
        shared + num_groups,
        num_groups,
        len(linears),
    )


def install_delayed_scale_step_counter(module, *, discover_shared_input_scales: bool = False) -> None:
    """Advance the delayed-scaling schedule once per forward of ``module``.

    With ``discover_shared_input_scales`` the first call also runs the discovery
    forward. It needs real tensor identities, so it runs eagerly, and without
    grad, so it saves no activations and leaves peak memory unchanged.
    """
    if getattr(module, "_primus_fp8_step_counter", False):
        return
    original_forward = module.forward
    pending_discovery = [discover_shared_input_scales]

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        _advance_delayed_scale_step()
        if pending_discovery[0]:
            pending_discovery[0] = False
            with torch.no_grad(), torch.compiler.set_stance("force_eager"), discover_shared_inputs(module):
                original_forward(*args, **kwargs)
        return original_forward(*args, **kwargs)

    module.forward = forward
    module._primus_fp8_step_counter = True
