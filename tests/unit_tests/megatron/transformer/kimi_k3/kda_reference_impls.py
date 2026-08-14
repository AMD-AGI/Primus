###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Independent KDA references used to test the Primus implementation.

Everything in this file is written **without** reference to
:mod:`primus.backends.megatron.core.transformer.kimi_k3` so that a shared
bug cannot hide: the point is to have two transcriptions of the same
published math produced from different sources.

* :func:`naive_kda_loop` — maximally literal ``O(B·H·T)`` Python triple
  loop in float64. Loops batch and head explicitly and keeps the state as
  a plain ``[K, V]`` matrix, so ``S^T k`` is spelled ``k @ S`` and the
  rank-1 write is spelled ``torch.outer``. Slow and unambiguous.
* :class:`HFKdaReference` — self-contained transcription of the HF
  ``KimiDeltaAttention`` module (``modeling_kimi_linear.py:477-663``),
  with ``fla``'s ``ShortConvolution`` / ``FusedRMSNormGated`` /
  ``chunk_kda`` replaced by explicit PyTorch. Parameter names match the
  released checkpoint's.

The ``chunk_kda`` call the HF module makes
(``modeling_kimi_linear.py:610-627``) sets ``use_qk_l2norm_in_kernel``,
``use_gate_in_kernel`` and ``use_beta_sigmoid_in_kernel`` all True, so
the kernel receives **raw** ``g`` / ``beta`` / ``A_log`` / ``dt_bias``
and applies the L2 norms, the gate and ``sigmoid(beta)`` internally.
:class:`HFKdaReference` therefore performs all three explicitly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "naive_kda_loop",
    "hf_kda_gate",
    "hf_l2norm",
    "hf_causal_conv1d_silu",
    "hf_rms_norm_gated",
    "HFKdaReference",
    "scaled_max_err",
    "assert_close_scaled",
]


# ---------------------------------------------------------------------------
# Tolerance helper
# ---------------------------------------------------------------------------


def scaled_max_err(got: torch.Tensor, want: torch.Tensor) -> float:
    """``max|got - want| / max(max|want|, 1)`` — error relative to the tensor's scale.

    Elementwise ``rtol`` is the wrong criterion for these comparisons: a
    KDA output tensor spans several orders of magnitude, and an element
    that happens to be near zero will fail any ``rtol`` no matter how
    accurate the computation is, because its own error is set by the much
    larger elements it was accumulated alongside. Normalising by the
    tensor's own scale is the criterion that actually tracks whether the
    two algebraically-identical orderings agree to fp32 precision.

    The ``max(., 1)`` floor keeps the metric absolute for tensors that are
    themselves tiny, so a near-zero output cannot inflate the ratio.
    """
    got = got.detach().float()
    want = want.detach().float()
    return ((got - want).abs().max() / want.abs().max().clamp(min=1.0)).item()


def assert_close_scaled(got: torch.Tensor, want: torch.Tensor, tol: float, label: str) -> float:
    """Assert :func:`scaled_max_err` is within ``tol``, printing the achieved value."""
    assert got.shape == want.shape, f"{label}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    err = scaled_max_err(got, want)
    print(
        f"[{label}] scale-relative max err = {err:.3e}  (tol {tol:.1e}, "
        f"|want|max = {want.detach().float().abs().max().item():.3e})"
    )
    assert err <= tol, f"{label}: scale-relative max error {err:.3e} exceeds tolerance {tol:.1e}"
    return err


# ---------------------------------------------------------------------------
# The literal recurrence
# ---------------------------------------------------------------------------


def naive_kda_loop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``S_t = (I - b_t k_t k_t^T) Diag(exp(g_t)) S_{t-1} + b_t k_t v_t^T``.

    Args:
        q, k, g: ``[B, T, H, K]`` (``g`` in log space).
        v: ``[B, T, H, V]``.
        beta: ``[B, T, H]``, already sigmoid-activated.
        scale: applied to ``q``; defaults to ``K ** -0.5``.
        initial_state: ``[B, H, K, V]`` or ``None``.

    Returns:
        ``(o, final_state)`` in float64.
    """
    batch, seq_len, num_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    if scale is None:
        scale = k_dim**-0.5
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))

    o = torch.zeros(batch, seq_len, num_heads, v_dim, dtype=torch.float64, device=q.device)
    final_state = torch.zeros(batch, num_heads, k_dim, v_dim, dtype=torch.float64, device=q.device)
    for b in range(batch):
        for h in range(num_heads):
            state = torch.zeros(k_dim, v_dim, dtype=torch.float64, device=q.device)
            if initial_state is not None:
                state = state + initial_state[b, h].double()
            for t in range(seq_len):
                alpha = torch.exp(g[b, t, h])  # [K] per-channel retention
                k_t, v_t = k[b, t, h], v[b, t, h]
                # 1. decay: row d of the state is scaled by alpha[d]
                state = state * alpha.reshape(k_dim, 1)
                # 2. delta correction against the decayed state
                pred = k_t @ state  # S^T k_t -> [V]
                state = state + beta[b, t, h] * torch.outer(k_t, v_t - pred)
                # 3. read the POST-update state
                o[b, t, h] = (q[b, t, h] * scale) @ state
            final_state[b, h] = state
    return o, final_state


# ---------------------------------------------------------------------------
# The transforms `fla` folds into its kernels
# ---------------------------------------------------------------------------


def hf_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``fla.modules.l2norm`` — row-wise L2 normalisation over the last axis."""
    x32 = x.float()
    return (x32 / torch.sqrt(x32.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def hf_kda_gate(
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float],
) -> torch.Tensor:
    """``fla.ops.kda.gate.naive_kda_lowerbound_gate`` / ``naive_kda_gate``.

    ``z``: ``[..., H, K]``; ``A_log``: ``[H]``; ``dt_bias``: ``[H*K]``.
    """
    num_heads, head_dim = z.shape[-2:]
    z = z.float()
    if dt_bias is not None:
        z = z + dt_bias.view(num_heads, head_dim)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(A_log.view(num_heads, 1).exp() * z)
    return -A_log.view(num_heads, 1).exp() * F.softplus(z)


def hf_causal_conv1d_silu(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """``fla.modules.ShortConvolution(activation='silu')`` on a ``[B, T, D]`` tensor.

    Depthwise, left-padded (hence causal), then SiLU. ``weight`` is the
    ``nn.Conv1d`` weight of shape ``[D, 1, W]``.
    """
    kernel_size = weight.shape[-1]
    y = F.conv1d(
        F.pad(x.transpose(1, 2), (kernel_size - 1, 0)),
        weight,
        bias=bias,
        groups=x.shape[-1],
    )
    return F.silu(y.transpose(1, 2))


def hf_rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``fla.modules.FusedRMSNormGated(activation='sigmoid')``.

    The Triton kernel upcasts its tile on load, applies
    ``x_hat * weight * sigmoid(gate)`` in fp32, and casts only on store
    (``fla/modules/fused_norm_gate.py:84-99``).
    """
    out_dtype = x.dtype
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    y = y * weight.float()
    y = y * torch.sigmoid(gate.float())
    return y.to(out_dtype)


# ---------------------------------------------------------------------------
# The HF module
# ---------------------------------------------------------------------------


class HFKdaReference(nn.Module):
    """Self-contained transcription of HF ``KimiDeltaAttention``.

    Parameter names mirror ``modeling_kimi_linear.py:498-541`` so a state
    dict can be moved between this module and the Primus one by name.
    The recurrence is delegated to :func:`naive_kda_loop`, i.e. this class
    shares no code at all with the implementation under test.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_size: int = 4,
        rms_norm_eps: float = 1e-5,
        gate_lower_bound: Optional[float] = -5.0,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_size = conv_size
        self.rms_norm_eps = rms_norm_eps
        self.gate_lower_bound = gate_lower_bound
        projection_size = num_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)

        def _conv() -> nn.Conv1d:
            return nn.Conv1d(
                projection_size,
                projection_size,
                kernel_size=conv_size,
                groups=projection_size,
                bias=conv_bias,
                padding=conv_size - 1,
            )

        self.q_conv1d, self.k_conv1d, self.v_conv1d = _conv(), _conv(), _conv()

        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.o_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[B, T, hidden] -> [B, T, hidden]``."""
        batch, seq_len, _ = hidden_states.shape
        num_heads, head_dim = self.num_heads, self.head_dim

        q = hf_causal_conv1d_silu(self.q_proj(hidden_states), self.q_conv1d.weight, self.q_conv1d.bias)
        k = hf_causal_conv1d_silu(self.k_proj(hidden_states), self.k_conv1d.weight, self.k_conv1d.bias)
        v = hf_causal_conv1d_silu(self.v_proj(hidden_states), self.v_conv1d.weight, self.v_conv1d.bias)
        q = q.reshape(batch, seq_len, num_heads, head_dim)
        k = k.reshape(batch, seq_len, num_heads, head_dim)
        v = v.reshape(batch, seq_len, num_heads, head_dim)
        # use_qk_l2norm_in_kernel=True
        q, k = hf_l2norm(q), hf_l2norm(k)

        # use_gate_in_kernel=True
        z = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(batch, seq_len, num_heads, head_dim)
        g = hf_kda_gate(z, self.A_log, self.dt_bias, self.gate_lower_bound)
        # use_beta_sigmoid_in_kernel=True
        beta = torch.sigmoid(self.b_proj(hidden_states).float())

        o, _ = naive_kda_loop(q, k, v, g, beta)
        o = o.to(hidden_states.dtype)

        gate = self.g_proj(hidden_states).reshape(batch, seq_len, num_heads, head_dim)
        o = hf_rms_norm_gated(o, gate, self.o_norm_weight, self.rms_norm_eps)
        return self.o_proj(o.reshape(batch, seq_len, num_heads * head_dim))
