###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_EMB_MAIN_GRAD=1 (default off): the word-embedding backward writes ``weight.main_grad`` directly.

Megatron's ``VocabParallelEmbedding`` at TP=1 runs ``F.embedding``; its backward (``embedding_dense_backward``,
sorted path for > 3072 indices) allocates a zero ``[vocab, hidden]`` gradient, scatters the per-row sums into it,
and the DDP hook then copies that ~1 GB tensor into ``main_grad`` (~0.28 ms on MI455X per step). Here the backward
reproduces the same sums straight into ``main_grad`` on the step's first write (beta 0 via the wgrad_beta0 policy):
``main_grad.zero_()`` + one Triton segment-sum kernel; later writes (shared / multi-microbatch) fall back to
``embedding_dense_backward`` + add, i.e. the DDP hook's add.

Bit-identical to PyTorch's CUDA kernels: indices are stably sorted (cub radix sort is stable), every vocabulary
row's gradient is the fp32 sum of its tokens' rows in ascending token order, taken in partial sums of
NROWS_PER_THREAD = 10 consecutive tokens (compute_grad_weight) that are then added in order (sum_and_scatter), and
rounded once to the parameter dtype. Only used for the plain TP=1 path (no masking, no reduce-scatter, not
deterministic_mode, no GTP remat); everything else keeps Megatron's forward.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get("PRIMUS_EMB_MAIN_GRAD", "0") == "1"
_NROWS_PER_THREAD = 10  # aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu


@triton.jit
def _emb_segment_sum_kernel(G, PERM, SORTED, OUT, N, D, stride_g, NROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    """One program per sorted position (x one D block); only a segment's first position works. No host sync: the
    segment end is found in-kernel, so the whole backward stays CUDA-graph capturable."""
    i = tl.program_id(0)
    pd = tl.program_id(1)
    row = tl.load(SORTED + i)
    prev = tl.load(SORTED + tl.maximum(i - 1, 0))
    if (i == 0) | (prev != row):
        end = i + 1
        while (end < N) & (tl.load(SORTED + tl.minimum(end, N - 1)) == row):
            end += 1
        d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
        dm = d < D
        total = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for c in range(i, end, NROWS):
            part = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for j in range(c, tl.minimum(c + NROWS, end)):
                r = tl.load(PERM + j).to(tl.int64)
                part += tl.load(G + r * stride_g + d, mask=dm, other=0.0).to(tl.float32)
            total += part
        tl.store(OUT + row.to(tl.int64) * D + d, total.to(OUT.dtype.element_ty), mask=dm)


@triton.jit
def _emb_zero_rows_kernel(PREV, CUR, OUT, D, BLOCK_D: tl.constexpr):
    """PRIMUS_GLUE_EMB_TOUCHED: zero the rows the previous step wrote (PREV[i]) and record this step's (CUR[i])."""
    i = tl.program_id(0)
    row = tl.load(PREV + i).to(tl.int64)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        tl.store(OUT + row * D + d, tl.zeros((BLOCK_D,), dtype=OUT.dtype.element_ty), mask=d < D)
    tl.store(PREV + i, tl.load(CUR + i))


_touched = {}  # main_grad data_ptr -> persistent int64 [N] buffer of the rows written by the last call


def _clear_for_write(out: torch.Tensor, idx: torch.Tensor) -> None:
    """Make ``out`` all-zero before the segment-sum writes this step's rows.

    Default: ``out.zero_()`` (1 GB at vocab 129280 x 4096 bf16, ~72 us). PRIMUS_GLUE_EMB_TOUCHED=1: only the rows the
    previous call of this function wrote are non-zero (every other row was zeroed then and nobody else writes this
    main_grad between two first writes: the embedding owns its slice under wgrad_beta0, and a clear / scaling keeps
    zeros zero), so zeroing those N rows (32 MB) restores the all-zero state. The first call per buffer, or a token
    count change, does the full clear. Bit-identical: the buffer is all zeros either way.
    """
    st = _touched.get(out.data_ptr())
    N = idx.numel()
    if os.environ.get("PRIMUS_GLUE_EMB_TOUCHED", "0") != "1":
        out.zero_()
        return
    if st is None or st.numel() != N or st.device != idx.device:
        out.zero_()
        _touched[out.data_ptr()] = idx.to(torch.int64).clone()
        return
    cur = idx if (idx.dtype == torch.int64 and idx.is_contiguous()) else idx.to(torch.int64).contiguous()
    _emb_zero_rows_kernel[(N,)](st, cur, out, out.shape[1], BLOCK_D=1024, num_warps=4)


@triton.jit
def _emb_sort_kernel(IDX, SORTED, PERM, N, SH: tl.constexpr, BLOCK: tl.constexpr):
    """PRIMUS_LAUNCH_EMB_SORT: torch.sort(idx, stable=True) of N <= BLOCK int64 keys in ONE program: the composite
    key (idx << SH) | position is unique, so its plain sort is the stable sort of idx (same values and perm)."""
    i = tl.arange(0, BLOCK)
    m = i < N
    k = tl.load(IDX + i, mask=m, other=0).to(tl.int64)
    c = tl.where(m, (k << SH) | i.to(tl.int64), 0x7FFFFFFFFFFFFFFF)
    c = tl.sort(c)
    tl.store(SORTED + i, c >> SH, mask=m)
    tl.store(PERM + i, c & ((1 << SH) - 1), mask=m)


def _launch_sort(idx: torch.Tensor, num_rows: int):
    """(sorted_idx, perm) == torch.sort(idx, stable=True) from one launch, or None when not applicable."""
    if os.environ.get("PRIMUS_LAUNCH_EMB_SORT", "0") != "1":
        return None
    N = idx.numel()
    if not (idx.is_cuda and idx.dtype == torch.int64 and idx.is_contiguous() and 0 < N <= 8192):
        return None
    sh = max(1, (N - 1).bit_length())
    if num_rows <= 0 or (num_rows - 1).bit_length() + sh > 62:
        return None
    sorted_idx = torch.empty_like(idx)
    perm = torch.empty_like(idx)
    _emb_sort_kernel[(1,)](idx, sorted_idx, perm, N, SH=sh, BLOCK=triton.next_power_of_2(N), num_warps=16)
    return sorted_idx, perm


def embedding_grad_into(grad: torch.Tensor, indices: torch.Tensor, out: torch.Tensor) -> None:
    """out[:] = embedding_dense_backward(grad, indices, out.shape[0], -1, False), bit for bit (sorted path)."""
    D = out.shape[1]
    g = grad.reshape(-1, D)
    if g.stride(1) != 1:
        g = g.contiguous()
    idx = indices.reshape(-1)
    N = idx.numel()
    r = _launch_sort(idx, out.shape[0])
    sorted_idx, perm = r if r is not None else torch.sort(idx, stable=True)
    _clear_for_write(out, idx)

    BLOCK_D = 1024
    _emb_segment_sum_kernel[(N, triton.cdiv(D, BLOCK_D))](
        g, perm, sorted_idx, out, N, D, g.stride(0), NROWS=_NROWS_PER_THREAD, BLOCK_D=BLOCK_D, num_warps=4
    )


class EmbeddingMainGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight):  # type: ignore[override]
        weight.grad_added_to_main_grad = True  # set in forward, like the other fused-wgrad writers
        weight._primus_fwd_claim = True
        ctx.save_for_backward(ids)
        ctx.weight = weight
        return torch.nn.functional.embedding(ids, weight)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

        (ids,) = ctx.saved_tensors
        w = ctx.weight
        ctx.weight = None
        mg = w.main_grad
        if main_grad_first_write(w) and ids.numel() > 3072:
            embedding_grad_into(grad, ids, mg)
        else:  # accumulate: exactly the DDP hook's add of the autograd gradient
            if os.environ.get("PRIMUS_GLUE_EMB_TOUCHED", "0") == "1":
                # the touched-rows clear only tracks the rows of the step's single first write
                raise RuntimeError("PRIMUS_GLUE_EMB_TOUCHED=1 needs exactly one embedding gradient write per step "
                                   "(one microbatch, untied, no MTP sharing); unset it for this configuration")
            mg.add_(torch.ops.aten.embedding_dense_backward(grad, ids, w.shape[0], -1, False))
        return None, _get_dummy_wgrad(list(w.shape), w.dtype)


def _usable(emb) -> bool:
    w = emb.weight
    mg = getattr(w, "main_grad", None)
    return (
        emb.tp_group.size() == 1
        and not emb.deterministic_mode
        and getattr(emb, "gtp_remat_size", 1) <= 1
        and not getattr(emb, "reduce_scatter_embeddings", False)
        and isinstance(mg, torch.Tensor)
        and mg.shape == w.shape
        and mg.dtype == w.dtype
        and mg.is_contiguous()
        and hasattr(w, "grad_added_to_main_grad")
        and not getattr(w, "shared", False)
        and not getattr(w, "shared_embedding", False)
        and torch.is_grad_enabled()
        and w.requires_grad
    )


def install() -> None:
    from megatron.core.tensor_parallel import layers

    cls = layers.VocabParallelEmbedding
    if getattr(cls, "_primus_emb_main_grad", False):
        return
    orig = cls.forward

    def forward(self, input_):
        if _usable(self):
            return EmbeddingMainGradFn.apply(input_, self.weight)
        return orig(self, input_)

    cls.forward = forward
    cls._primus_emb_main_grad = True
    try:
        from primus.core.utils.module_utils import log_rank_0

        log_rank_0("[Patch:emb_main_grad] TP=1 word-embedding backward writes main_grad directly (beta 0)")
    except Exception:  # noqa: BLE001
        pass


try:
    from primus.core.patches import PatchContext, register_patch

    @register_patch(
        "megatron.core.tensor_parallel.emb_main_grad",
        backend="megatron",
        phase="before_train",
        description="TP=1 word-embedding backward writes main_grad directly (PRIMUS_EMB_MAIN_GRAD=1).",
        condition=lambda ctx: _ENABLED,
        priority=72,
    )
    def patch_emb_main_grad(ctx: PatchContext) -> None:
        del ctx
        install()

except ImportError:
    pass
