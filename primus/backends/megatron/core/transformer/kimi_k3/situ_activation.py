###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``situ`` — Moonshot's soft-clamped SwiGLU, the Kimi K3 FFN activation.

Reference: ``class SituAndMul`` (``modeling_kimi_linear.py``), registered
as ``ACT2FN["situ"]`` and selected by ``hidden_act: "situ"``. Both
``KimiMLP`` (dense MLP + shared experts) and ``KimiBlockSparseMLP``
(routed experts) use it with the same two betas.

.. code-block:: python

    gate, up = x.chunk(2, dim=-1)                     # both to fp32
    situ_a = beta * tanh(gate / beta) * sigmoid(gate)  # beta = 4.0
    up     = linear_beta * tanh(up / linear_beta)      # linear_beta = 25.0
    return (situ_a * up).to(x.dtype)

``beta * tanh(g / beta)`` is a *smooth* clamp of ``g`` to
``[-beta, beta]``. Since ``silu(g) = g * sigmoid(g)``, ``situ_a`` is
exactly SiLU with its linear factor soft-clamped to ±4, and the ``up``
branch is soft-clamped to ±25. (The bound is open in exact arithmetic
and closed in floating point: ``tanh`` saturates to exactly ``1.0`` at
``|g / beta| >= 10`` or so.)

This is the direct analogue of DeepSeek-V4's clamped SwiGLU
(:mod:`primus.backends.megatron.core.transformer.clamped_swiglu`),
differing only in using ``tanh`` soft clamps rather than hard
``clamp``, and in using two different bounds. Both exist for the same
reason: keep FFN activations inside FP8 / FP4 range.

How this reaches Megatron
-------------------------
``situ`` is a **fused** GLU activation: it needs both halves of the
post-``fc1`` tensor, because the ``up`` half gets a soft clamp of its
own and the final multiply happens inside the fp32 region. Megatron
therefore cannot run it through ``config.activation_func``. With
``gated_linear_unit=True`` (which ``swiglu: true`` sets, and which K3's
``[gate | up]`` packing requires) ``MLP.forward`` builds

.. code-block:: python

    # mlp.py
    def glu(x):
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
        if (val := self.config.activation_func_clamp_value) is not None:
            x_glu = x_glu.clamp(min=None, max=val)
            x_linear = x_linear.clamp(min=-val, max=val)
        return self.config.activation_func(x_glu) * (x_linear + self.config.glu_linear_offset)

so ``config.activation_func`` only ever sees the **gate** half, and the
only transform available for the ``up`` half is a hard clamp sharing a
single bound with the gate. That cannot express ``situ``.

The one hook that hands a caller-supplied callable the **fused**
``[..., 2I]`` tensor while keeping ``gated_linear_unit=True`` — i.e.
keeping ``fc1``'s doubled width and ``fc2``'s un-doubled input — is the
``activation_func`` *module* slot on the MLP submodules (``mlp.py``, and
identically for grouped experts at ``experts.py``), gated by
``config.use_te_activation_func``. :class:`SituActivation` matches that
slot's builder protocol (``TEActivationFunctionBuilder``, ``mlp.py``), so
the wiring is

.. code-block:: python

    config.use_te_activation_func = True     # route to the fused module slot
    config.activation_func = F.silu          # kept only to satisfy the
                                             # whitelist in
                                             # transformer_config.py
    MLPSubmodules(..., activation_func=SituActivation)

``config.activation_func`` must stay one of ``{F.gelu, F.silu, F.relu}``
because ``TransformerConfig.__post_init__`` rejects anything else once
``use_te_activation_func`` is set; it is dead code on this path (the
module slot wins at ``mlp.py``). ``bias_activation_fusion`` must be
false — ``transformer_config.py`` rejects the combination, and
the fused SwiGLU kernels have no soft clamp anyway (which is why
``kimi_k3_base.yaml`` already sets ``bias_swiglu_fusion: false``).

For code that does have both halves in hand (unit tests, an eager
reference, a custom expert backend), :func:`situ_pre_mul` takes them
split and :func:`situ_pre_mul_fused` takes them concatenated.

Numerics
--------
The HF reference hardcodes ``.to(torch.float32)``. Casting *down* from
float64 would make ``torch.autograd.gradcheck`` meaningless, so the
compute dtype here is ``promote_types(x.dtype, float32)``: identical to
HF for every dtype a real model uses (bf16 / fp16 / fp32) and honest in
float64. The operation order is HF's exactly, so fp32 agreement is
bit-for-bit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "situ_pre_mul",
    "situ_pre_mul_fused",
    "situ_betas_from_config",
    "SituActivation",
]

# ``SituAndMul.__init__``'s defaults (modeling_kimi_linear.py): a beta of
# 1.0 and no soft clamp on the up branch. Kimi K3's config.json overrides
# both (4.0 / 25.0).
_DEFAULT_BETA = 1.0


def _soft_clamp(x: Tensor, bound: float) -> Tensor:
    """``bound * tanh(x / bound)`` — a smooth clamp of ``x`` to ``[-bound, bound]``."""
    return bound * torch.tanh(x / bound)


def situ_pre_mul(
    gate: Tensor,
    up: Tensor,
    *,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
) -> Tensor:
    """``situ`` on split inputs.

    Args:
        gate: ``[..., I]`` gate stream (HF: ``gate_proj``).
        up: ``[..., I]`` up stream (HF: ``up_proj``).
        beta: soft-clamp bound on the gate branch's linear factor; K3
            uses ``4.0``. ``0`` / ``None`` falls back to HF's default of
            ``1.0``, matching ``_get_situ_activation_params``'s
            ``beta or 1.0`` (``modeling_kimi_linear.py``) — note
            that this is *not* "clamping disabled".
        linear_beta: soft-clamp bound on the up branch; K3 uses ``25.0``.
            ``None`` leaves ``up`` untransformed, which is what
            ``SituAndMul`` does when ``linear_beta`` is unset.

    Returns:
        ``[..., I]`` activation output in ``gate``'s dtype.
    """
    if gate.shape != up.shape:
        raise ValueError(
            "situ_pre_mul expects matching gate / up shapes; "
            f"got {tuple(gate.shape)} vs {tuple(up.shape)}."
        )
    out_dtype = gate.dtype
    compute_dtype = torch.promote_types(out_dtype, torch.float32)
    beta = float(beta) if beta else _DEFAULT_BETA

    gate = gate.to(compute_dtype)
    up = up.to(compute_dtype)

    situ_a = _soft_clamp(gate, beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = _soft_clamp(up, float(linear_beta))
    return (situ_a * up).to(out_dtype)


def situ_pre_mul_fused(
    x: Tensor,
    *,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
) -> Tensor:
    """``situ`` on a fused ``[..., 2I]`` ``[gate | up]`` input.

    This is the Megatron GLU packing (``mlp.py``) and also what
    ``KimiMLP`` / ``KimiBlockSparseMLP`` hand ``SituAndMul`` — both
    ``cat([gate, up], -1)`` before calling the activation
    (``modeling_kimi_linear.py``).
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "situ_pre_mul_fused expects a [gate | up] last dim; "
            f"got shape {tuple(x.shape)} (last dim must be even)."
        )
    gate, up = x.chunk(2, dim=-1)
    return situ_pre_mul(gate, up, beta=beta, linear_beta=linear_beta)


def situ_betas_from_config(config) -> Tuple[float, Optional[float]]:
    """Read ``(beta, linear_beta)`` off a transformer config.

    The field names are HF's, which is also what
    :class:`KimiK3TransformerConfig` declares:
    ``activation_situ_beta`` / ``activation_situ_linear_beta``. The
    fallbacks mirror ``_get_situ_activation_params``
    (``modeling_kimi_linear.py``) — an unset or zero ``beta``
    becomes ``1.0``, an unset ``linear_beta`` stays ``None``.
    """
    beta = getattr(config, "activation_situ_beta", None)
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return (float(beta) if beta else _DEFAULT_BETA), (None if linear_beta is None else float(linear_beta))


class SituActivation(nn.Module):
    """``situ`` as an MLP ``activation_func`` submodule.

    Built by :class:`megatron.core.transformer.mlp.MLP` and
    :class:`megatron.core.transformer.moe.experts.TEGroupedMLP` from the
    ``activation_func`` slot of their submodules dataclass, which is
    invoked as ``submodules.activation_func(config=config)`` and then
    called on the **fused** post-``fc1`` tensor (``mlp.py``). The betas
    therefore come from the config rather than from constructor
    arguments, so a single spec works for every layer.

    Parameter-less, so it adds nothing to the state dict -- but it still has
    to *implement* :meth:`sharded_state_dict`; see that method.
    """

    def __init__(self, *, config=None, beta: Optional[float] = None, linear_beta: Optional[float] = None):
        super().__init__()
        if config is not None:
            cfg_beta, cfg_linear_beta = situ_betas_from_config(config)
            beta = cfg_beta if beta is None else beta
            linear_beta = cfg_linear_beta if linear_beta is None else linear_beta
        self.beta = float(beta) if beta else _DEFAULT_BETA
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``situ`` to a fused ``[..., 2I]`` ``[gate | up]`` tensor."""
        return situ_pre_mul_fused(x, beta=self.beta, linear_beta=self.linear_beta)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ):
        """Empty sharded state dict -- required, not optional.

        Without this method **no Kimi K3 checkpoint can be saved at all**.
        ``MLP.sharded_state_dict`` walks ``self._modules`` and calls
        ``module.sharded_state_dict(...)`` on every child *unconditionally*
        (``mlp.py``), unlike ``sharded_state_dict_default``, which
        guards the call with ``hasattr`` (``utils.py``). This class
        occupies the ``activation_func`` slot of the dense MLP on layer 0 and
        of the shared experts on every MoE layer, so a save raised
        ``AttributeError: 'SituActivation' object has no attribute
        'sharded_state_dict'`` before this existed. (The *routed* experts are
        unaffected: ``TEGroupedMLP.sharded_state_dict`` goes through
        ``sharded_state_dict_default`` and its guard, ``experts.py``.)

        The body is ``sharded_state_dict_default``'s else-branch verbatim
        rather than a bare ``return {}``: the module has no parameters today,
        and this way it would not silently drop one if it ever gained a buffer.
        """
        from megatron.core.transformer.utils import (
            ensure_metadata_has_dp_cp_group,
            make_sharded_tensors_for_checkpoint,
        )

        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {},
            sharded_offsets,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def extra_repr(self) -> str:
        return f"beta={self.beta}, linear_beta={self.linear_beta}"
