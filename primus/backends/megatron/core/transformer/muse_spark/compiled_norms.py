###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``torch.compile`` regions for the Muse Spark norms (``muse_spark_compile_norms``).

The fused-kernel path (``muse_spark_fused_*``) runs each RMSNorm as one Triton kernel but
leaves the ops around it eager: the residual-gate coefficients and ``alpha * res + beta *
branch``, the sigmoid output gate, the ``1 + w`` of the centered norms. Each function below
is the eager formula of one whole region; compiled, Inductor emits one reduction + pointwise
kernel per region forward and one per backward. The math and the roundings are those of the
eager modules in ``norms.py`` / ``attention.py`` (fp32 math, rounded once where they round).

``gated_residual_norm_math`` also folds the next sublayer's centered norm into the gated
residual (``MuseSparkGatedResidual.gated`` hands the norm output to the next norm, see
``link_next_norm`` in ``norms.py``); ``swiglu_math`` is the expert activation.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "compiled",
    "weightless_rms_math",
    "centered_norm_math",
    "input_scaled_norm_math",
    "gated_residual_math",
    "gated_residual_norm_math",
    "o_norm_gate_math",
    "swiglu_math",
]


def _rms32(x32: torch.Tensor, eps: float) -> torch.Tensor:
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)


def weightless_rms_math(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``rms_norm(x.float()).type_as(x)`` (``norms._rms``)."""
    return _rms32(x.float(), eps).to(x.dtype)


def centered_norm_math(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``round(norm(x.float()) * (1 + w.float()))`` (``centered_rms_norm_reference``)."""
    return (_rms32(x.float(), eps) * (1.0 + weight.float())).to(x.dtype)


def input_scaled_norm_math(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``norm(round(x * (1 + w)))`` (``MuseSparkInputScaledRMSNorm`` eager path)."""
    u = (x.float() * (1.0 + weight.float())).to(x.dtype)
    return weightless_rms_math(u, eps)


def gated_residual_math(
    branch: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    post_weight,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    """``MuseSparkGatedResidual.gated``: post-norm of the branch (weightless when
    ``post_weight`` is None, input-scaled otherwise), then ``alpha * res + beta * branch``."""
    if post_weight is None:
        post = weightless_rms_math(branch, eps)
    else:
        post = input_scaled_norm_math(branch, post_weight, eps)
    w = gate.float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return (alpha * residual + beta * post).to(residual.dtype)


def gated_residual_norm_math(
    branch: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    post_weight,
    temperature: float,
    eps: float,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple:
    """The gated residual and the next sublayer's centered norm in one graph: returns the
    new residual ``h`` and ``centered_norm(h)``."""
    h = gated_residual_math(branch, residual, gate, post_weight, temperature, eps)
    return h, centered_norm_math(h, norm_weight, norm_eps)


def swiglu_math(x: torch.Tensor) -> torch.Tensor:
    """``silu(a) * b`` on the halves of ``x`` in its dtype, rounding after the silu and the
    product as the eager glu in ``PrimusGroupedMLP.bias_act_func`` does."""
    a, b = torch.chunk(x, 2, dim=-1)
    return F.silu(a) * b


def o_norm_gate_math(x: torch.Tensor, gate: torch.Tensor, head_dim: int, eps: float) -> torch.Tensor:
    """Per-head weightless ``o_norm`` then Megatron's ``x * sigmoid(gate.float())``."""
    shape = x.shape
    y = weightless_rms_math(x.view(*shape[:-1], -1, head_dim), eps).view(shape)
    return (y * torch.sigmoid(gate.contiguous().view(shape).float())).to(x.dtype)


_COMPILED = {}


def compiled(fn):
    """``fn`` compiled once per process (static shapes, no graph breaks). Inductor may keep
    an intermediate bf16 value in fp32 inside a fused kernel, so results can differ from
    the eager modules by one bf16 rounding step (``emulate_precision_casts`` would round
    there too, at ~1-2 % of the step on the 4-layer proxy)."""
    c = _COMPILED.get(fn)
    if c is None:
        c = _COMPILED[fn] = torch.compile(fn, dynamic=False, fullgraph=True)
    return c
