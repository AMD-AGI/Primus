###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Head-dimension-tuned Triton var-len flash attention for Ideogram-4 (hd=256).

WHY:
  Ideogram-4 attends at head_dim=256, and aiter's Triton MHA is configured
  head-dimension-BLIND: ``_get_config()`` takes no arguments and returns one block-size
  dict for every shape, tuned for the smaller head dims. At hd=256 those blocks overflow
  the register file (512 registers/lane, 212 spilled), so the shipped Triton path loses
  badly to CK-tile and Ideogram-4 runs CK by default. Measured on one MI355X at the
  production packing (40968 tokens, 18 heads, hd=256, 16 segments, non-causal):

      arm                     fwd       bwd      total
      CK-tile (default)      5.25 ms   39.35 ms  44.60 ms
      Triton, aiter config  22.23 ms   66.83 ms  89.07 ms
      Triton, tuned here     6.45 ms   14.90 ms  21.35 ms   <- 2.09x vs CK

  The entire 4.1x came from FIVE config values; the kernels themselves were never the
  problem. So this module owns the CONFIG, not the kernel: it calls aiter's own Triton
  forward and one-kernel backward and only supplies block sizes that suit hd=256.

WHY WE DRIVE AITER'S TWO HALVES OURSELVES (and not just ``aiter.flash_attn_varlen_func``):
  Both of aiter's halves accept a ``config``, but the autograd wrapper BETWEEN them drops
  it: ``_FlashAttnVarlenFunc.backward`` calls ``flash_attn_onekernel_backward(...)``
  without one, so the backward config is unreachable through any public aiter API. The
  backward is where 24 of the 27 ms live, which leaves exactly three ways to set it —
  edit aiter's installed ``gfx950-MHA-DEFAULT.json``, monkeypatch ``_get_config``, or
  drive the two halves directly. This module does the third: one custom op per half plus
  an autograd rule joining them, specialized to the one call Ideogram-4 makes (var-len,
  non-causal, no dropout / bias / alibi / sink / sliding-window / block-table). It needs no
  ``ENABLE_CK=0`` (process-global, and it also changes JIT flags for anything compiled at
  runtime), no writes to an installed site-packages file, and no patching.

TUNING:
  ``_TUNED_DELTAS`` is keyed on ``(arch, head_dim)`` and holds only the values that differ
  from aiter's shipped config, which stays the base — so an upstream retune of any other
  field is inherited rather than frozen. Individual fields stay env-overridable
  (``PRIMUS_IDEOGRAM_ATTN_{FWD,BWD}_*``) so a re-sweep needs no code change, mirroring the
  ``PRIMUS_V4_ATTN_*`` knobs on the Megatron V4 attention kernels.

  An unlisted ``(arch, head_dim)`` gets ``{}`` and therefore aiter's head-dim-blind config,
  and that fallthrough is NOT a harmless no-op: at hd=256 it is the 89.07 ms row above,
  2.0x SLOWER than the CK default. :func:`is_tuned` reports whether a shape was actually
  swept so a caller can pick CK instead of a silent regression — which is what the
  Ideogram-4 install hook does, since ``triton`` is now the preset's default.

TORCH.COMPILE:
  Clean, but only because the two halves are registered as ``torch.library`` custom ops
  (below) instead of being called from a ``torch.autograd.Function``. Dynamo cannot trace
  aiter's Triton launch path at all: it passes ``num_ctas`` straight to the kernel (rejected
  outright), calls into ``triton._C.libtriton``, and ``hasattr``s the kernel object. Traced
  directly, that produced 57 graph breaks inside a compiled transformer block and failed
  ``fullgraph``; wrapped as opaque ops it traces with fullgraph, ZERO breaks and ZERO
  recompiles -- the same bar the CK arm clears, which matters because a break inside the
  block is the FSDP2 collective-desync hazard ``attention.py`` documents. Ideogram-4 ships
  ``enable_compile: false`` today, so this is about not blocking that flip.

  Reproducer, both arms: scripts/check_compile_varlen_attn.py (tracelens_zero23 scratch
  tree). Note the compiled result there is BIT-IDENTICAL to eager for this arm while CK
  moves by ~1e-5 -- the atomics-free backward showing up as reproducibility.

CORRECTNESS:
  Output and all three gradients match CK element-wise at the production shape (max
  relative error 1.5e-3, i.e. the bf16 floor). Two traps are guarded, both silent:

    * dQ UNDER-COVERAGE. The backward grid is sized ``cdiv(seqlen, BLOCK_N1)`` while the
      dq pass steps by ``BLOCK_M2``, so any config with ``BLOCK_M2 < BLOCK_N1`` leaves the
      tail of Q at its zero init and returns a WRONG dq with correct dk/dv and no error
      raised. aiter asserts ``BLOCK_N1 % BLOCK_M1 == 0`` and ``BLOCK_M2 % BLOCK_N2 == 0``
      but not this, and the sweep hit it on four candidates (dq rel-err 0.37-0.62).
      :func:`_check_dq_coverage` rejects such a config before it can run.
    * SHARED-DICT MUTATION. aiter's ``_get_config`` is ``lru_cache``d and hands back the
      one dict it parsed from JSON. Overriding it in place would silently retune every
      other Triton MHA caller in the process, so the base is deep-copied first.

  The one-kernel backward computes dq/dk/dv in separate passes with no atomics, so it is
  run-to-run deterministic regardless of the ``deterministic`` flag, and it has none of
  the workspace that makes CK's *deterministic* hd=256 backward OOM at 2048px.
"""

from __future__ import annotations

import copy
import functools
import logging
import os
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Ideogram-4's attention head dim; used only to report the configs at install time, since
# the real value is read off the tensors at every call.
IDEOGRAM4_HEAD_DIM = 256

# Only the fields that differ from aiter's shipped gfx950 config; everything else (
# BLK_SLICE_FACTOR, matrix_instr_nonkdim, waves_per_eu, PRELOAD_V, ...) is inherited.
# fwd BLOCK_M is absent because the shipped value (128) already won the sweep.
#
# gfx950 / hd=256, non-causal bf16, from varlen_sweep/triton_hd256_sweep.json:
#   bwd  BLOCK_N1 128->64 + num_stages 1->2   66.83 ms -> 14.90 ms
#   fwd  BLOCK_N   64->16, warps 4->8, st 1->2  22.23 ms ->  6.45 ms
# The backward pair is not separable: BLOCK_N1=64 alone gives 29.4 ms and num_stages=2
# alone gives 64.5 ms. Pipelining only pays once the narrower block frees the registers
# it needs (512 regs/lane with 212 spilled -> 256 with 81).
#
# Re-swept over a range of packed sequence lengths (4k-20k tokens) rather than the single
# shape above. The warps/stages half of the fwd line turned out to be an artefact of having
# been tuned jointly with BLOCK_N: once BLOCK_N=16 is in force, aiter's original 4/1 wins
# again by 3.7-11.2%, widening with sequence length, with bit-identical out and grads. The
# pairing is what mattered -- 4 warps with 2 stages is 59% SLOWER, which is presumably how
# 8/2 was arrived at. BLOCK_N=16 is confirmed optimal: every larger N is worse, and halving
# BLOCK_M to 64 to free the accumulator registers for one costs 26%, so the fp32 accumulator
# footprint -- not the online-softmax rescale frequency -- is what bounds this kernel. The
# backward is already at its optimum; nothing in the swept grid beats the line below.
_TUNED_DELTAS: Dict[Tuple[str, int], Dict[str, Dict[str, int]]] = {
    ("gfx950", 256): {
        "fwd": {"BLOCK_N": 16, "num_warps": 4, "num_stages": 1},
        "bwd": {"BLOCK_N1": 64, "num_stages": 2},
    },
}


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    """Lower-cased GPU gfx arch (e.g. ``gfx950``); ``""`` if it cannot be determined."""
    try:
        name = torch.cuda.get_device_properties(0).gcnArchName  # e.g. "gfx950:sramecc+:xnack-"
        return name.split(":")[0].strip().lower()
    except Exception:  # pragma: no cover - no GPU / driver
        return ""


def _env_overrides(prefix: str, fields: Dict[str, str]) -> Dict[str, int]:
    """Read ``{prefix}{suffix}`` env knobs into kernel-config fields (unset -> absent)."""
    out: Dict[str, int] = {}
    for suffix, field in fields.items():
        raw = os.getenv(prefix + suffix)
        if raw is not None:
            out[field] = int(raw)
    return out


_FWD_ENV = {
    "BLOCK_M": "BLOCK_M",
    "BLOCK_N": "BLOCK_N",
    "WARPS": "num_warps",
    "STAGES": "num_stages",
}
_BWD_ENV = {
    "BLOCK_M1": "BLOCK_M1",
    "BLOCK_N1": "BLOCK_N1",
    "BLOCK_M2": "BLOCK_M2",
    "BLOCK_N2": "BLOCK_N2",
    "WARPS": "num_warps",
    "STAGES": "num_stages",
}


def _deltas(head_dim: int, half: str) -> Dict[str, int]:
    return dict(_TUNED_DELTAS.get((_arch(), head_dim), {}).get(half, {}))


def is_tuned(head_dim: int) -> bool:
    """Whether this ``(arch, head_dim)`` has block sizes somebody actually measured.

    Off the tuned pairs the kernels fall back to aiter's head-dim-blind config, which at
    hd=256 is 2.0x SLOWER than CK (module docstring), so this is the guard that keeps a
    ``triton`` default from turning into a silent regression on an unswept device: the
    caller routes to CK instead. Callers should ask once, at install time — the answer is
    fixed for the process, and asking per call would put a Python branch in the hot path.
    """
    return bool(_TUNED_DELTAS.get((_arch(), head_dim)))


def _check_dq_coverage(onekernel: Dict[str, int]) -> None:
    """Reject a backward config that would silently return a partially-written dq.

    See the module docstring: the grid covers ``cdiv(seqlen, BLOCK_N1)`` blocks and the dq
    pass advances ``BLOCK_M2`` per block, so ``BLOCK_M2 < BLOCK_N1`` never visits the tail
    of Q and leaves it zero. Fails loudly here instead of corrupting gradients.
    """
    block_m2 = int(onekernel["BLOCK_M2"])
    block_n1 = int(onekernel["BLOCK_N1"])
    if block_m2 < block_n1:
        raise ValueError(
            f"backward config has BLOCK_M2={block_m2} < BLOCK_N1={block_n1}: the dq pass "
            f"would cover only {block_m2}/{block_n1} of each grid block and return a "
            "silently wrong dq (dk/dv stay correct). Raise BLOCK_M2 or lower BLOCK_N1."
        )


def fwd_config(head_dim: int, dtype: torch.dtype) -> Dict[str, int]:
    """aiter's forward config for this dtype, with the hd/arch-tuned fields applied."""
    from aiter.ops.triton._triton_kernels.attention.mha import _get_config

    config = copy.deepcopy(_get_config(False, dtype, has_pe=False))
    config.update(_deltas(head_dim, "fwd"))
    config.update(_env_overrides("PRIMUS_IDEOGRAM_ATTN_FWD_", _FWD_ENV))
    return config


def bwd_config(head_dim: int) -> Dict[str, Dict[str, int]]:
    """aiter's one-kernel backward config, with the hd/arch-tuned fields applied."""
    from aiter.ops.triton.attention.mha_onekernel_bwd import _get_config

    config = copy.deepcopy(_get_config())
    onekernel = config["onekernel"]
    onekernel.update(_deltas(head_dim, "bwd"))
    onekernel.update(_env_overrides("PRIMUS_IDEOGRAM_ATTN_BWD_", _BWD_ENV))
    _check_dq_coverage(onekernel)
    return config


_OP_NS = "primus_ideogram4"

# Registered as torch.library custom ops rather than called from a torch.autograd.Function,
# so Dynamo sees ONE opaque node per half instead of trying to trace aiter's launch path --
# which it cannot: aiter passes num_ctas straight to the kernel, calls into
# triton._C.libtriton, and hasattrs the kernel object. As a plain autograd.Function this
# path produced 57 graph breaks inside a compiled transformer block (CK: zero), and a break
# there is the FSDP2 collective-desync hazard. Neither half mutates its inputs -- verified
# for cu_seqlens, which aiter's CK var-len op DOES write -- hence mutates_args=().


@torch.library.custom_op(f"{_OP_NS}::varlen_attn_fwd", mutates_args=())
def _varlen_attn_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    softmax_scale: float,
) -> Tuple[Tensor, Tensor]:
    """aiter's Triton var-len forward under our config; returns ``(out, softmax_lse)``."""
    from aiter.ops.triton.attention.mha import _flash_attn_forward

    out, softmax_lse, _, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        0.0,  # dropout_p
        softmax_scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        bias=None,
        alibi_slopes=None,
        return_lse=True,
        return_softmax=False,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        sink=None,
        config=fwd_config(q.shape[-1], q.dtype),
    )
    return out, softmax_lse


@_varlen_attn_fwd.register_fake
def _(q, k, v, cu_seqlens, max_seqlen, softmax_scale):
    # Shapes only, no data: the var-len LSE is (total_tokens, num_heads) in fp32.
    return (
        torch.empty_like(q),
        q.new_empty((q.shape[0], q.shape[1]), dtype=torch.float32),
    )


@torch.library.custom_op(f"{_OP_NS}::varlen_attn_bwd", mutates_args=())
def _varlen_attn_bwd(
    do: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    softmax_scale: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """aiter's Triton one-kernel backward under our config; returns ``(dq, dk, dv)``."""
    from aiter.ops.triton.attention.mha_onekernel_bwd import (
        flash_attn_onekernel_backward,
    )

    # dq is zero-initialised because the dq pass accumulates into it. The kernel indexes do
    # with its own strides and has no packed-layout fallback, so a non-contiguous grad (which
    # a compiled backward may hand us) is materialised rather than misread.
    dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
    flash_attn_onekernel_backward(
        do if do.is_contiguous() else do.contiguous(),
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        None,  # dbias
        softmax_scale,
        None,  # alibi_slopes
        False,  # causal
        cu_seqlens,
        cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        philox_seed=0,
        philox_offset=0,
        config=bwd_config(q.shape[-1]),
    )
    return dq, dk, dv


@_varlen_attn_bwd.register_fake
def _(do, q, k, v, out, softmax_lse, cu_seqlens, max_seqlen, softmax_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _fwd_setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, max_seqlen, softmax_scale = inputs
    out, softmax_lse = output
    ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
    ctx.max_seqlen = max_seqlen
    ctx.softmax_scale = softmax_scale


def _fwd_backward(ctx, grad_out, grad_lse):
    """Gradient of the forward op. ``grad_lse`` is unused: nothing consumes the LSE."""
    q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
    dq, dk, dv = torch.ops.primus_ideogram4.varlen_attn_bwd(
        grad_out,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens,
        ctx.max_seqlen,
        ctx.softmax_scale,
    )
    return dq, dk, dv, None, None, None


torch.library.register_autograd(f"{_OP_NS}::varlen_attn_fwd", _fwd_backward, setup_context=_fwd_setup_context)


def triton_varlen_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    *,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """Var-len bf16 flash attention on aiter's Triton kernels, tuned for this head dim.

    Signature-compatible with
    :func:`primus.backends.nemo_automodel.models.ideogram4.attention.varlen_flash_attention`:
    q/k/v are packed ``(total_tokens, H, D)``, ``cu_seqlens`` is the ``int32``
    ``(num_segments + 1,)`` prefix sum, and the result is ``(total_tokens, H, D)``.

    The head-dim check and the default scale are resolved HERE, outside the op: both read
    static shape metadata, so they cost a guard under ``torch.compile``, whereas raising
    from inside an opaque op would only surface at runtime.
    """
    head_dim = q.shape[-1]
    if head_dim % 8 != 0:
        # aiter pads to a multiple of 8 and slices the gradients back; hd=256 never needs
        # it, so the padding bookkeeping is left out rather than left untested.
        raise ValueError(f"head_dim must be a multiple of 8, got {head_dim}")
    scale = float(head_dim ** (-0.5)) if softmax_scale is None else float(softmax_scale)
    out, _ = torch.ops.primus_ideogram4.varlen_attn_fwd(q, k, v, cu_seqlens, int(max_seqlen), scale)
    return out


def describe_config(head_dim: int) -> str:
    """One-line summary of the configs in force, for the install log."""
    try:
        fwd = fwd_config(head_dim, torch.bfloat16)
        bwd = bwd_config(head_dim)["onekernel"]
    except Exception as e:  # pragma: no cover - aiter missing / unreadable config
        return f"config unavailable ({e})"
    tuned = "tuned" if is_tuned(head_dim) else "aiter default (unswept shape)"
    return (
        f"{_arch() or 'unknown-arch'} hd={head_dim} [{tuned}] "
        f"fwd BLOCK_M={fwd['BLOCK_M']} BLOCK_N={fwd['BLOCK_N']} "
        f"warps={fwd['num_warps']} stages={fwd['num_stages']}; "
        f"bwd BLOCK_M1={bwd['BLOCK_M1']} BLOCK_N1={bwd['BLOCK_N1']} "
        f"BLOCK_M2={bwd['BLOCK_M2']} BLOCK_N2={bwd['BLOCK_N2']} "
        f"warps={bwd['num_warps']} stages={bwd['num_stages']}"
    )
