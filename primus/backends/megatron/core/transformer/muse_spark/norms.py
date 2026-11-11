###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark norms and the learned residual gate (eager torch; fp32 math).

The three norm kinds of ``modeling_muse_spark`` (transformers), plus the residual gate:

* ``MuseSparkRMSNorm``: plain RMSNorm, optional weight (``with_scale``).
* ``MuseSparkCenteredRMSNorm``: normalize in fp32, then ``* (1 + w)`` (w init 0).
* ``MuseSparkInputScaledRMSNorm``: ``x * (1 + w)`` in fp32, round to the input dtype,
  then normalize (no weight after the norm).
* ``MuseSparkGatedResidual``: ``alpha * residual + beta * branch`` with
  ``beta = sigmoid(g / T)``, ``alpha = sqrt(clamp(sigmoid(-g / T) * (1 + beta), 1e-3))``.

The module signatures follow Megatron's norm builders
(``config, hidden_size, eps``) so they can be dropped into layer specs.
"""

import torch
from torch import nn

from primus.backends.megatron.core.models.muse_spark.model_types import (
    config_flag,
    mark_muse_spark_param,
)
from primus.backends.megatron.core.transformer.muse_spark.centered_norm import (
    centered_rms_norm,
    weighted_rms_norm,
)
from primus.backends.megatron.core.transformer.muse_spark.post_expert_norm import (
    fused_rms_norm_available,
    post_expert_norm_probs,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rmsnorm import (
    eager_rms_norm,
    fused_rms_norm,
)

_HALF = (torch.bfloat16, torch.float16)


def casts_to_half(fn) -> bool:
    """Whether a ``Module._apply`` function casts fp32 tensors to bf16 / fp16.

    Probed on an empty fp32 CPU tensor (no copy of real data). If the probe fails,
    returns True (keep the safe snapshot path).
    """
    try:
        out = fn(torch.empty(0, dtype=torch.float32))
    except Exception:  # noqa: BLE001 - e.g. a device-only fn that rejects CPU tensors
        return True
    return isinstance(out, torch.Tensor) and out.dtype in _HALF


def apply_keeping_fp32(module, name, fn, super_apply, *args, **kwargs):
    """``super()._apply`` that keeps the fp32 parameter ``name`` fp32 through a half cast.

    Only a cast to bf16 / fp16 (Float16Module's blanket ``.bfloat16()``) snapshots the
    fp32 value first and restores it after (lossless). Pure device moves and fp32 / fp64
    casts go straight to ``super_apply`` with no clone; a param that is not fp32 yet
    (built in bf16) is promoted on the next half cast.
    """
    p = getattr(module, name, None)
    keep = p.detach().float().clone() if p is not None and casts_to_half(fn) else None
    out = super_apply(fn, *args, **kwargs)
    p = getattr(module, name, None)
    if keep is not None and p is not None and p.dtype != torch.float32:
        p.data = keep.to(device=p.device)
    return out


def _norm_size(config, hidden_size) -> int:
    """``hidden_size``, else ``config.hidden_size``, else 1 (weightless use / bare build)."""
    if hidden_size:
        return int(hidden_size)
    return int(getattr(config, "hidden_size", None) or 1)


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """The fp32 weightless RMSNorm of ``x``, rounded once to ``x.dtype``: Primus's shared
    eager RMSNorm (``eager_rms_norm``, the reference of the fused kernel ``fused_rms_norm``)."""
    return eager_rms_norm(x, None, eps=eps, mid_cast=False, out_dtype=x.dtype)


def weightless_rms(x: torch.Tensor, eps: float, fused: bool = True) -> torch.Tensor:
    """``rms_norm(x.float()).type_as(x)`` over the last dim, no weight.

    ``fused``: Primus's shared RMSNorm (``fused_rms_norm``: one Triton forward and one
    backward kernel on GPU, ``PRIMUS_RMSNORM_TRITON``; the eager reference elsewhere).
    Otherwise the eager reference (``eager_rms_norm``): the same fp32 math."""
    if fused:
        return fused_rms_norm(x, None, eps=eps, mid_cast=False, out_dtype=x.dtype)
    return _rms(x, eps)


class MuseSparkRMSNorm(nn.RMSNorm):
    """RMSNorm in fp32; ``with_scale=False`` gives the weightless norm.

    Built on ``torch.nn.RMSNorm`` (the norm behind Megatron's ``WrappedTorchNorm``); the
    forward is Primus's shared RMSNorm (``fused_rms_norm``) with the HF fp32 math: normalize
    and scale in fp32, round once to the input dtype. The weightless norm runs the eager fp32
    reference instead when ``muse_spark_fused_attn_norm`` is false.
    """

    def __init__(
        self, config=None, hidden_size: int = None, eps: float = 1e-5, with_scale: bool = True, **_kw
    ):
        super().__init__(_norm_size(config, hidden_size), eps=float(eps), elementwise_affine=with_scale)
        if with_scale and getattr(config, "sequence_parallel", False):
            setattr(self.weight, "sequence_parallel", True)
        # the weightless norms (attention post_norm, Q / K norm) run fused on GPU
        self.fused = config_flag(config, "muse_spark_fused_attn_norm", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return weightless_rms(x, self.eps, self.fused)
        # with_scale: fp32 norm * fp32 weight, rounded once to x.dtype (deterministic in
        # deterministic mode, see weighted_rms_norm)
        return weighted_rms_norm(x, self.weight.float(), self.eps)


class _ZeroInitRMSNorm(nn.RMSNorm):
    """The shared part of the Muse Spark ``(1 + w)`` norms: a zero-initialised weight of the
    config's width, marked sequence-parallel when the config is, and ``self.fused`` from the
    subclass's ``FUSED_FLAG`` (default on). A subclass sets ``FUSED_FLAG``, its default eps
    and ``forward``."""

    FUSED_FLAG: str = None

    def __init__(self, config=None, hidden_size: int = None, eps: float = 1e-5, **_kw):
        # nn.RMSNorm.__init__ calls reset_parameters (below), which zeroes the weight.
        super().__init__(_norm_size(config, hidden_size), eps=float(eps), elementwise_affine=True)
        if getattr(config, "sequence_parallel", False):
            setattr(self.weight, "sequence_parallel", True)
        # the fused kernel on GPU; the flag set to false runs the eager formula
        self.fused = config_flag(config, self.FUSED_FLAG, True)

    def reset_parameters(self) -> None:
        # elementwise_affine=True: nn.RMSNorm sets self.weight before it calls this
        torch.nn.init.zeros_(self.weight)


class MuseSparkCenteredRMSNorm(_ZeroInitRMSNorm):
    """``norm(x) * (1 + w)`` in fp32 (w init 0), rounded once to the input dtype.

    It runs Primus's shared fused RMSNorm with the weight ``1 + w`` (``centered_norm.py``):
    Triton forward and backward on GPU, the eager formula elsewhere
    (``muse_spark_fused_centered_norm: false``).
    """

    FUSED_FLAG = "muse_spark_fused_centered_norm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return centered_rms_norm(x, self.weight, self.eps, self.fused)


class MuseSparkInputScaledRMSNorm(_ZeroInitRMSNorm):
    """``norm(round(x * (1 + w)))``: scale first, round to the input dtype, then normalize.

    On GPU this runs as the fused post-expert-norm Triton kernel without routing probs
    (``probs=None``: the norm alone), one forward and one backward kernel instead of about six
    eager ones with fp32 copies of the activation. ``config.muse_spark_fused_post_expert_norm =
    False`` keeps the eager formula; CPU and widths above ``MAX_FUSED_HIDDEN`` always use it.
    """

    FUSED_FLAG = "muse_spark_fused_post_expert_norm"

    def __init__(self, config=None, hidden_size: int = None, eps: float = 1e-8, **kw):
        super().__init__(config, hidden_size, eps, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and x.dim() >= 1:
            rows = x.reshape(-1, x.shape[-1])
            if fused_rms_norm_available(rows):
                # probs=None: the norm alone (no ones tensor, no prob load, no dp in backward)
                return post_expert_norm_probs(rows, self.weight, None, self.eps).view(x.shape)
        out = (x.float() * (1.0 + self.weight.float())).type_as(x)
        return _rms(out, self.eps)  # out is in x.dtype; _rms normalizes in fp32 and rounds back


def residual_gate_coeffs(gate: torch.Tensor, temperature: float, eps: float = 1e-3):
    w = gate.float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=eps))
    return alpha, beta


class MuseSparkGatedResidual(nn.Module):
    """Replaces Megatron's bias-dropout-add: ``post_norm(branch)`` then the learned residual gate.

    Called like the bda: ``module(training, fused)((out, bias), residual, dropout_p)``.
    ``post_norm`` is the weightless RMSNorm (after attention) or the input-scaled
    RMSNorm (after the MoE), both with ``post_norm_eps``. ``gate`` is kept fp32.
    """

    # FSDP2: the fp32 ``gate`` is sharded as its own fp32 group (muse_spark/fsdp2_islands.py).
    fsdp2_dtype_island = True

    def __init__(self, config, kind: str = "attn"):
        super().__init__()
        h = config.hidden_size
        # Direct access (no getattr fallbacks that could drift from the config defaults).
        eps = float(config.post_norm_eps)
        if kind == "attn":
            self.post_norm = MuseSparkRMSNorm(config, h, eps=eps, with_scale=False)
        else:
            self.post_norm = MuseSparkInputScaledRMSNorm(config, h, eps=eps)
        # Initial gate value for a run from random weights (a checkpoint load overwrites it).
        # A constant: the gate is replicated over TP ranks and must start equal on all of them.
        self.init_value = float(config.residual_gate_init_value)
        self.gate = mark_muse_spark_param(
            nn.Parameter(torch.full((h,), self.init_value, dtype=torch.float32))
        )
        if getattr(config, "sequence_parallel", False):
            setattr(self.gate, "sequence_parallel", True)
        self.temperature = float(config.residual_gate_temperature)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.gate.fill_(self.init_value)

    def _apply(self, fn, *args, **kwargs):
        # Keep the gate fp32 through Float16Module's blanket cast (HF _keep_in_fp32_modules_strict);
        # only half casts snapshot / restore, device moves are untouched.
        return apply_keeping_fp32(self, "gate", fn, super()._apply, *args, **kwargs)

    def gated(self, branch: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        branch = self.post_norm(branch)
        alpha, beta = residual_gate_coeffs(self.gate, self.temperature)
        return (alpha * residual + beta * branch).to(residual.dtype)

    @staticmethod
    def _is_factory_call(args, kwargs) -> bool:
        return not kwargs and len(args) <= 2 and (not args or isinstance(args[0], bool))

    def _bda_factory(self, training=None, fused=None):
        training = self.training if training is None else bool(training)

        def _bda(x_with_bias, residual, prob):
            return self(x_with_bias, residual, prob, training)

        return _bda

    def __call__(self, *args, **kwargs):
        # Megatron calls the bda as ``module(training, fused)(x_with_bias, residual, prob)``.
        # The bool-only factory call returns the closure without nn.Module.__call__, so module
        # hooks (FSDP2's unshard / reshard of the fp32 ``gate`` island) run once per use: only
        # around the closure's tensor call, where the gating runs.
        if self._is_factory_call(args, kwargs):
            return self._bda_factory(*args)
        return super().__call__(*args, **kwargs)

    def forward(self, *args):
        if self._is_factory_call(args, {}):  # direct .forward(training, fused) use
            return self._bda_factory(*args)
        x_with_bias, residual, prob, training = args
        x, bias = x_with_bias
        if bias is not None:
            x = x + bias
        if training and prob:
            x = torch.nn.functional.dropout(x, p=prob, training=True)
        return self.gated(x, residual)
