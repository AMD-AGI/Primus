###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark norms and the learned residual gate (eager torch; fp32 math).

All four norm kinds of ``modeling_muse_spark`` (transformers):

* ``MuseSparkRMSNorm``: plain RMSNorm, optional weight (``with_scale``).
* ``CenteredRMSNorm``: normalize in fp32, then ``* (1 + w)`` (w init 0).
* ``InputScaledRMSNorm``: ``x * (1 + w)`` in fp32, round to the input dtype,
  then normalize (no weight after the norm).
* ``ResidualGate``: ``alpha * residual + beta * branch`` with
  ``beta = sigmoid(g / T)``, ``alpha = sqrt(clamp(sigmoid(-g / T) * (1 + beta), 1e-3))``.

The module signatures follow Megatron's norm builders
(``config, hidden_size, eps``) so they can be dropped into layer specs.
"""

import torch
from torch import nn


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)


class MuseSparkRMSNorm(nn.Module):
    """RMSNorm in fp32; ``with_scale=False`` gives the weightless norm."""

    def __init__(
        self, config=None, hidden_size: int = None, eps: float = 1e-5, with_scale: bool = True, **_kw
    ):
        super().__init__()
        self.eps = float(eps)
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            if config is not None and getattr(config, "sequence_parallel", False):
                setattr(self.weight, "sequence_parallel", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _rms(x.float(), self.eps)
        if self.with_scale:
            out = out * self.weight.float()
        return out.type_as(x)


class MuseSparkCenteredRMSNorm(nn.Module):
    """``norm(x) * (1 + w)`` in fp32 (w init 0)."""

    def __init__(self, config=None, hidden_size: int = None, eps: float = 1e-5, **_kw):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        if config is not None and getattr(config, "sequence_parallel", False):
            setattr(self.weight, "sequence_parallel", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _rms(x.float(), self.eps) * (1.0 + self.weight).float()
        return out.type_as(x)


class MuseSparkInputScaledRMSNorm(nn.Module):
    """``norm(round(x * (1 + w)))``: scale first, round to the input dtype, then normalize."""

    def __init__(
        self, config=None, hidden_size: int = None, eps: float = 1e-8, sequence_parallel=None, **_kw
    ):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        sp = (
            sequence_parallel
            if sequence_parallel is not None
            else getattr(config, "sequence_parallel", False)
        )
        if sp:
            setattr(self.weight, "sequence_parallel", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x.float() * (1.0 + self.weight).float()).type_as(x)
        return _rms(out.float(), self.eps).type_as(x)


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

    def __init__(self, config, kind: str = "attn"):
        super().__init__()
        h = config.hidden_size
        eps = float(getattr(config, "post_norm_eps", 1e-8))
        if kind == "attn":
            self.post_norm = MuseSparkRMSNorm(config, h, eps=eps, with_scale=False)
        else:
            self.post_norm = MuseSparkInputScaledRMSNorm(config, h, eps=eps)
        self.gate = nn.Parameter(torch.zeros(h, dtype=torch.float32))
        if getattr(config, "sequence_parallel", False):
            setattr(self.gate, "sequence_parallel", True)
        self.temperature = float(getattr(config, "residual_gate_temperature", 0.3))
        self.kind = kind

    def _apply(self, fn, *args, **kwargs):
        # Keep the gate fp32 through Float16Module's blanket cast (HF _keep_in_fp32_modules_strict).
        keep = self.gate.detach().float().clone() if self.gate is not None else None
        module = super()._apply(fn, *args, **kwargs)
        if keep is not None and self.gate.dtype != torch.float32:
            self.gate.data = keep.to(device=self.gate.device)
        return module

    def gated(self, branch: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        branch = self.post_norm(branch)
        alpha, beta = residual_gate_coeffs(self.gate, self.temperature)
        return (alpha * residual + beta * branch).to(residual.dtype)

    def forward(self, *args):
        # Megatron calls the bda as ``module(training, fused)(x_with_bias, residual, prob)``.
        # The returned closure calls the module again with tensors, so the gating runs inside
        # a real module call: FSDP2 unshards ``gate`` (its own fp32 group) around that call.
        if len(args) <= 2 and (not args or isinstance(args[0], bool)):
            training = bool(args[0]) if args else self.training

            def _bda(x_with_bias, residual, prob):
                return self(x_with_bias, residual, prob, training)

            return _bda
        x_with_bias, residual, prob, training = args
        x, bias = x_with_bias
        if bias is not None:
            x = x + bias
        if training and prob:
            x = torch.nn.functional.dropout(x, p=prob, training=True)
        return self.gated(x, residual)
