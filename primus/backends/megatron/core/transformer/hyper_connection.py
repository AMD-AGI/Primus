###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Manifold-Constrained Hyper-Connections (mHC) for DeepSeek-V4.

Reference: deepseek-v4/develop/techblog/01-deepseek-v4-architecture-deep-dive.md
section 2 ("mHC: Manifold-Constrained Hyper-Connections") and section 9.3.

Three modules live here:

* :class:`HyperMixer` — per-layer mixer used twice per block (once before /
  after the attention sub-block, once for the FFN sub-block). Produces
  ``(pre, post, comb)`` triplets and exposes ``collapse`` / ``expand``
  helpers for the surrounding block to drive the K parallel hidden streams.

* :class:`HyperHead` — final collapse used once at the end of the main trunk
  (and once *per MTP layer*, with its own copy). Sigmoid-weighted sum, no
  Sinkhorn.

* :func:`sinkhorn_normalize` — alternating row/column normalization, kept in
  fp32 for stability (per RedNote slide 9 and NeMo port pitfall #3).

Phase 4 contract:
* Plain ``nn.Linear`` for the projection ``fn``. TP-friendly variants
  (``ColumnParallelLinear``) come in Phase 6 once the rest of the V4 path
  is correct end-to-end on a single device.
* All HC parameters (``fn.weight``, ``scale``, ``base``) live in fp32.
  The block is responsible for passing in/out tensors in whatever activation
  dtype it uses; the module up-casts internally to fp32 around the Sinkhorn
  region and casts back at the end.

Plan-5 P29 (RESCOPED — see ``deepseek-v4/develop/plan-5/02-phase-details.md``)
adds a ``torch.compile`` fast path for :func:`sinkhorn_normalize`. The eager
loop issues 1 + 2*(n_iters - 1) separate fp32 ``aten::sum`` reductions per
call; at V4-Flash production widths each reduction runs at ~250x over the
memory-bound floor because HIP's default ``reduce_kernel<512, 1, ...>`` is
sized for huge reductions and our ``[1, 4096, 4, 4] -> [1, 4096, 4, 1]``
shape has only 4 elements per output. The compiled path collapses every
sum / divide / broadcast into one Inductor-fused Triton kernel
(``fullgraph=True, dynamic=False``). The compiled callable is cached
module-globally on ``(n_iters, eps, dtype)`` so each combination compiles
exactly once per process.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Plan-5 P29 fast path: torch.compile-fused Sinkhorn-Knopp.
#
# Cache keyed on ``(n_iters, eps, dtype)``.  Each freshly-decorated callable
# uses ``@torch.compile(fullgraph=True, dynamic=True)`` so a SINGLE compiled
# artefact handles every input shape variation Inductor generates code that
# accepts shape as a runtime argument.  The dynamic path is the right
# trade-off for V4 because:
#
# * In production each rank sees exactly one Sinkhorn input shape
#   (``[B, S, K, K] = [1, 4096, 4, 4]`` at V4-Flash) so ``dynamic=True``
#   pays no specialization cost vs ``dynamic=False``.
# * Multi-shape harnesses (unit tests, MTP heads with different leading
#   dims) all hit the same compiled artefact, so we never thrash Dynamo's
#   recompile_limit.  ``dynamic=False`` would force one Dynamo
#   specialization per shape AND ALL CLOSURES SHARE THE SAME CODE OBJECT
#   (closures from the same factory function inherit the factory's code
#   object), which means Dynamo's cache_size_limit (default 8) is shared
#   across every cached callable.  After ~8 distinct ``(shape, stride,
#   requires_grad)`` combinations the limit is hit and ``fullgraph=True``
#   raises ``FailOnRecompileLimitHit`` even though our cache is fine —
#   the cache key just is not what Dynamo's internal cache is keyed on.
#
# ``in_dtype`` still participates in the cache key because the trailing
# cast back to the input activation dtype changes the Inductor IR; running
# bf16 and fp32 callers through the same compiled artefact would force a
# miscompile or a redundant recompile.
# ---------------------------------------------------------------------------

_SinkhornCacheKey = tuple[int, float, torch.dtype]
_compiled_sinkhorn_cache: dict[_SinkhornCacheKey, Callable[[torch.Tensor], torch.Tensor]] = {}


def _build_compiled_sinkhorn(
    n_iters: int,
    eps: float,
    in_dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build (and torch.compile) one Sinkhorn-Knopp implementation specialised
    on ``(n_iters, eps, in_dtype)`` but generic over input shape.

    The body is the same algorithm as the eager :func:`sinkhorn_normalize`
    path but written so Inductor sees one straight-line graph of fp32
    sums / divides / broadcasts (Python ``for`` is unrolled at compile time
    because ``n_iters`` is a closure-captured Python int).
    """

    @torch.compile(fullgraph=True, dynamic=True)
    def _impl(logits: torch.Tensor) -> torch.Tensor:
        m = logits.float()
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        for _ in range(max(n_iters - 1, 0)):
            m = m / (m.sum(dim=-1, keepdim=True) + eps)
            m = m / (m.sum(dim=-2, keepdim=True) + eps)
        return m.to(in_dtype)

    return _impl


def _get_compiled_sinkhorn(
    n_iters: int,
    eps: float,
    in_dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Cache-or-build accessor; returns the compiled callable for the given
    Sinkhorn signature.  Shape is NOT part of the key — see module-level
    cache notes (we use ``dynamic=True`` so one artefact handles every
    shape).
    """
    key: _SinkhornCacheKey = (int(n_iters), float(eps), in_dtype)
    fn = _compiled_sinkhorn_cache.get(key)
    if fn is None:
        fn = _build_compiled_sinkhorn(n_iters, eps, in_dtype)
        _compiled_sinkhorn_cache[key] = fn
    return fn


def sinkhorn_normalize(
    logits: torch.Tensor,
    *,
    n_iters: int = 20,
    eps: float = 1e-6,
    use_compiled: bool = False,
) -> torch.Tensor:
    """Project a non-negative ``[..., K, K]`` matrix onto the doubly-stochastic
    manifold via the Sinkhorn-Knopp algorithm.

    The algorithm itself is the alternating row / column ``L1`` normalization:

    .. code-block::

        for _ in range(n_iters):
            M /= M.sum(dim=-1, keepdim=True)   # rows  -> 1
            M /= M.sum(dim=-2, keepdim=True)   # cols  -> 1

    To match the released V4 reference (cf. ``compute_weights`` in the
    techblog) we emit the **first** column-normalization once before the
    standard alternating loop, so that the final iteration ends on a column
    normalization — this matches the (pre, post, comb) projection convention
    that the surrounding block consumes.

    Stability:
    * the input must already be non-negative (typically ``softmax`` output
      plus an ``eps`` floor)
    * runs in fp32 regardless of the input dtype, then casts back

    Args:
        logits: ``[..., K, K]`` non-negative.
        n_iters: total Sinkhorn iterations (= 1 priming column step
            + ``n_iters - 1`` row/col cycles).
        eps: numerical floor to prevent divide-by-zero.
        use_compiled: when ``True``, dispatch to the cached
            ``torch.compile(fullgraph=True, dynamic=False)`` build of the
            same loop (plan-5 P29 RESCOPED).  The compiled callable
            collapses the 1 + 2*(n_iters - 1) fp32 ``aten::sum`` launches
            (and their divides / broadcasts) into one Inductor-fused Triton
            kernel; AOT autograd handles the BWD.  Numerical contract is
            unchanged (algorithm is identical; only the kernel boundary
            moves).  Default ``False`` until the V4 config flag
            ``use_v4_compiled_sinkhorn`` is flipped on.

    Returns:
        Approximately doubly-stochastic ``[..., K, K]`` (same dtype as input).
    """
    if use_compiled:
        return _get_compiled_sinkhorn(n_iters, eps, logits.dtype)(logits)
    in_dtype = logits.dtype
    m = logits.float()
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(n_iters - 1, 0)):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m.to(in_dtype)


class HyperMixer(nn.Module):
    """Per-layer mHC mixer.

    Maintains the three projection scales / biases (``pre``, ``post``,
    ``comb``) and a single packed ``Linear`` ``fn`` that produces
    ``[..., (2 + K) * K]`` from ``[..., K * D]``.

    Shapes (B and S are arbitrary; the mixer is agnostic to which is which):

    * ``compute_weights(x)`` → ``pre [..., K]``, ``post [..., K]``,
      ``comb [..., K, K]``
    * ``collapse(x, pre)`` → ``[..., D]``
    * ``expand(x, out, post, comb)`` → ``[..., K, D]``

    Args:
        hidden_size: per-stream feature dim ``D``.
        hc_mult: number of parallel streams ``K``.
        eps: floor for ``sigmoid(...) + eps`` and for Sinkhorn.
        sinkhorn_iters: Sinkhorn iteration count (``20`` matches V4 release).
        use_compiled_sinkhorn: plan-5 P29 (RESCOPED) fast path.  When
            ``True`` the call to :func:`sinkhorn_normalize` inside
            :meth:`compute_weights` dispatches to the
            ``torch.compile(fullgraph=True, dynamic=False)`` build of the
            same algorithm, kept in :data:`_compiled_sinkhorn_cache`.
            Defaults to ``False`` so existing callers (and existing
            checkpoints) are bit-equivalent.  The V4 block reads
            ``config.use_v4_compiled_sinkhorn`` and forwards it here.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        eps: float = 1e-6,
        sinkhorn_iters: int = 20,
        use_compiled_sinkhorn: bool = False,
    ) -> None:
        super().__init__()
        if hc_mult < 1:
            raise ValueError(f"hc_mult must be >= 1, got {hc_mult}")

        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps
        self.sinkhorn_iters = sinkhorn_iters
        self.use_compiled_sinkhorn = bool(use_compiled_sinkhorn)

        out_dim = (2 + hc_mult) * hc_mult
        # All HC params kept in fp32; see techblog §2.2 pitfall #3.
        self.fn = nn.Linear(hc_mult * hidden_size, out_dim, bias=False, dtype=torch.float32)
        # Three independent scale scalars: one each for pre / post / comb.
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        # Bias terms — same partition as the linear output.
        self.base = nn.Parameter(torch.zeros(out_dim, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NormalInitGain-style init keeps the post-rms-scale logits well-behaved.
        nn.init.normal_(self.fn.weight, std=1.0 / math.sqrt(self.hc_mult * self.hidden_size))
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    # ---- internals -------------------------------------------------------

    def _packed_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pack streams along the last dim, RMS-normalize, project to ``out_dim``.

        ``x``: ``[..., K, D]`` → ``flat`` ``[..., K*D]`` → ``logits`` ``[..., out_dim]``.
        Done in fp32 for stability (HC parameters are fp32 anyway).
        """
        flat = x.flatten(-2)
        flat32 = flat.float()
        rsqrt = torch.rsqrt(flat32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        logits = F.linear(flat32 * rsqrt, self.fn.weight.to(dtype=flat32.dtype))
        return logits

    # ---- public API ------------------------------------------------------

    def compute_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(pre, post, comb)`` for the K parallel streams ``x``.

        ``x``: ``[..., K, D]`` (typically ``[B, S, K, D]`` or ``[S, B, K, D]``).
        """
        K = self.hc_mult
        logits = self._packed_logits(x)  # [..., (2+K)*K], fp32

        pre_logit = logits[..., :K] * self.scale[0] + self.base[:K]
        post_logit = logits[..., K : 2 * K] * self.scale[1] + self.base[K : 2 * K]
        comb_logit = logits[..., 2 * K :].view(*logits.shape[:-1], K, K) * self.scale[2] + self.base[
            2 * K :
        ].view(K, K)

        pre = torch.sigmoid(pre_logit) + self.eps  # (eps, 1+eps]
        post = 2.0 * torch.sigmoid(post_logit)  # (0, 2)  no eps
        comb = torch.softmax(comb_logit, dim=-1) + self.eps
        comb = sinkhorn_normalize(
            comb,
            n_iters=self.sinkhorn_iters,
            eps=self.eps,
            use_compiled=self.use_compiled_sinkhorn,
        )

        # Cast back to the activation dtype to match downstream block compute.
        out_dtype = x.dtype
        return pre.to(out_dtype), post.to(out_dtype), comb.to(out_dtype)

    @staticmethod
    def collapse(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Collapse K streams into 1 via ``pre`` weights.

        ``x``: ``[..., K, D]``;  ``pre``: ``[..., K]``;
        returns ``[..., D]``.
        """
        return (pre.unsqueeze(-1) * x).sum(dim=-2)

    @staticmethod
    def expand(
        x: torch.Tensor,
        out: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Write the sub-block output ``out`` back to K streams.

        ``new_stream[h] = post[h] * out + Σ_k comb[h, k] * x[k]``

        Shapes:
        * ``x``: ``[..., K, D]`` — current K streams
        * ``out``: ``[..., D]`` — sub-block (attn or FFN) output
        * ``post``: ``[..., K]``
        * ``comb``: ``[..., K, K]``

        Returns ``[..., K, D]``.
        """
        # post[..., K] * out[..., D] -> [..., K, D]
        write = post.unsqueeze(-1) * out.unsqueeze(-2)
        # comb[..., K, K] @ x[..., K, D] -> [..., K, D]
        mix = torch.matmul(comb, x)
        return write + mix


class HyperHead(nn.Module):
    """Final collapse from K streams → 1 stream.

    Used once at the end of the main trunk, and once *per* MTP layer (each
    with its own copy — ``num_nextn_predict_layers`` separate heads). Plain
    sigmoid-weighted sum; no Sinkhorn.

    Args:
        hidden_size: per-stream feature dim ``D``.
        hc_mult: number of input streams ``K``.
        eps: floor for ``sigmoid(...) + eps``.
    """

    def __init__(self, *, hidden_size: int, hc_mult: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hc_mult < 1:
            raise ValueError(f"hc_mult must be >= 1, got {hc_mult}")
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps

        self.fn = nn.Linear(hc_mult * hidden_size, hc_mult, bias=False, dtype=torch.float32)
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(hc_mult, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fn.weight, std=1.0 / math.sqrt(self.hc_mult * self.hidden_size))
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[..., K, D]`` → ``[..., D]``."""
        flat = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        mixes = F.linear(flat * rsqrt, self.fn.weight.to(dtype=flat.dtype))  # [..., K]
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=-2).to(x.dtype)


__all__ = [
    "HyperMixer",
    "HyperHead",
    "sinkhorn_normalize",
]
