###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Context-parallel support for the DeepSeek-V4 dense / SWA attention branch.

V4 attention materialises the query tensor at FULL head width on every tensor-parallel rank
(``deepseek_v4_attention.py`` sets ``num_attention_heads_per_partition = num_heads`` with no
``divide()``), so TP shards weights but not attention activations. Context parallelism is the
only lever that shrinks the per-rank sequence, and therefore the only way past the long-context
memory wall: measured on 8x MI308X, an unmodified 4-layer V4-Flash needs ~396 GB at 128k
against a 192 GiB card, while 32k fits in 145 GB.

For the dense (``compress_ratio == 0``) branch the CP contract is small, because that branch is
already index-driven -- causality and the sliding window live entirely in the index matrix the
sparse-MLA adapter builds, not in the kernel. A CP rank therefore needs exactly two things:

  1. the ``d_window`` post-RoPE KV rows immediately left of its shard, so a query near the shard
     start can still see its full window (:class:`LeftBoundaryExchange`), and
  2. its global row offset, so the adapter can validate the window against global positions
     while indexing into the local ``[boundary ++ local]`` buffer
     (``cp_dwindow`` / ``cp_global_start`` in ``v4_sparse_mla_adapter``).

Exchanging post-RoPE KV rather than pre-projection hidden states is deliberate: the neighbour
has already applied RoPE with the correct global positions, and it moves ``d_window`` rows
instead of a full hidden block.

Ported from NVIDIA/Megatron-LM PR #5087 (`csa_cp_utils.py`), whose CP path is THD-only; this
is the BSHD-shaped equivalent for Primus's V4 attention.
"""


import torch
import torch.distributed as dist


class LeftBoundaryExchange(torch.autograd.Function):
    """Receive the previous CP rank's trailing ``d_window`` rows; scatter grads back.

    Forward is one batched isend/irecv step around the CP ring. Backward returns the boundary
    gradient to the rank that actually owns those rows, so they accumulate where the parameters
    that produced them live.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, d_window: int, cp_group):
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        if tensor.shape[0] < d_window:
            raise RuntimeError(
                "DeepSeek-V4 CP boundary exchange needs local rows >= d_window: "
                f"local_rows={tensor.shape[0]}, d_window={d_window}. Reduce "
                "context_parallel_size or the sliding window."
            )
        boundary = tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))

        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(dist.irecv, boundary, dist.get_global_rank(cp_group, cp_rank - 1), cp_group)
            )
        if cp_rank + 1 < cp_size:
            send_tail = tensor[-d_window:].contiguous()
            ops.append(
                dist.P2POp(dist.isend, send_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group)
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        grad_input = grad_boundary.new_zeros(ctx.input_shape)

        ops = []
        recv_grad = None
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_boundary.contiguous(),
                    dist.get_global_rank(cp_group, cp_rank - 1),
                    cp_group,
                )
            )
        if cp_rank + 1 < cp_size:
            recv_grad = grad_boundary.new_empty(grad_boundary.shape)
            ops.append(
                dist.P2POp(dist.irecv, recv_grad, dist.get_global_rank(cp_group, cp_rank + 1), cp_group)
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if recv_grad is not None:
            grad_input[-ctx.d_window :] = recv_grad
        return grad_input, None, None


def get_cp_group():
    """The context-parallel process group, or None when CP is off / torch.distributed is not up."""
    if not dist.is_available() or not dist.is_initialized():
        return None
    try:
        from megatron.core import parallel_state
    except ImportError:
        return None
    try:
        group = parallel_state.get_context_parallel_group()
    except (AssertionError, RuntimeError):
        return None
    if group is None or group.size() <= 1:
        return None
    return group


def exchange_boundary_kv(kv_bshd: torch.Tensor, d_window: int, cp_group) -> torch.Tensor:
    """Boundary KV for a ``[B, S, 1, head_dim]`` post-RoPE latent.

    Returns ``[B, d_window, 1, head_dim]``. Rank 0 gets zeros, which the adapter's
    global-position validity mask then excludes -- no separate special case is needed.
    """
    B, S, G, Dh = kv_bshd.shape
    if B != 1:
        raise RuntimeError(f"DeepSeek-V4 CP currently assumes micro_batch_size=1, got B={B}.")
    flat = kv_bshd.reshape(S, G * Dh)
    boundary = LeftBoundaryExchange.apply(flat, int(d_window), cp_group)
    return boundary.reshape(1, int(d_window), G, Dh)


class _AllGatherPool(torch.autograd.Function):
    """All-gather the per-rank compressed pool into the global, sequence-ordered pool.

    Concatenating in rank order IS sequence order here: this path is BSHD with one
    sequence, every rank owns a contiguous block of `S_total / cp_size` rows, and that
    block length is a multiple of `ratio`, so compressed group boundaries never straddle
    a rank boundary. (Upstream's THD path needs a seq-major -> rank-major remap precisely
    because ragged packed sequences break that property; here it is free.)

    Backward is a reduce-scatter, NOT a plain slice. Rank r's pool rows are read by the
    queries of every rank at or after r, so each of those ranks holds a partial gradient
    for them; slicing this rank's block out of its OWN grad_out would keep only the
    contribution from its own queries and silently drop the rest. That error is invisible
    in the forward and compounds over training steps -- measured as loss drift growing
    2e-5 -> 4.5e-4 across three steps against a 5e-5 CP noise floor.
    """

    @staticmethod
    def forward(ctx, pool_local: torch.Tensor, cp_group):
        cp_size = cp_group.size()
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_group.rank()
        ctx.p_local = pool_local.shape[1]
        gathered = [torch.empty_like(pool_local) for _ in range(cp_size)]
        dist.all_gather(gathered, pool_local.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad_out):
        grad = grad_out.contiguous()
        dist.all_reduce(grad, group=ctx.cp_group)
        lo = ctx.cp_rank * ctx.p_local
        return grad[:, lo : lo + ctx.p_local].contiguous(), None


def build_global_pool(pool_local: torch.Tensor, cp_group) -> torch.Tensor:
    """`[B, P_local, D]` -> `[B, P_local * cp_size, D]` in sequence order."""
    return _AllGatherPool.apply(pool_local, cp_group)


def compressor_boundary_rows(compress_ratio: int, overlap: bool) -> int:
    """Hidden rows this rank must receive from its left neighbour before compressing.

    The compressor pools each window independently EXCEPT in overlap mode (V4 uses it for
    ratio 4 / CSA), where window i is stitched with the previous window's second channel
    half. At a CP boundary that previous window lives on the left neighbour, so `ratio`
    hidden rows have to come across. Non-overlap (ratio 128 / HCA) is purely local.
    """
    return int(compress_ratio) if overlap else 0


def compressed_causal_mask(
    s_local: int, p_global: int, global_start: int, ratio: int, *, device, dtype
) -> torch.Tensor:
    """`[S_local, P_global]` additive mask against GLOBAL query positions.

    Pool slot `s` covers raw tokens `[s*ratio, (s+1)*ratio)`, so a query at global token
    `t` may attend to it iff `(s+1)*ratio - 1 <= t`. Under CP the query's global position
    is `global_start + local_row`, which is the only change from the non-CP form.
    """
    t = torch.arange(s_local, device=device).unsqueeze(1) + int(global_start)
    s_end = (torch.arange(p_global, device=device).unsqueeze(0) + 1) * int(ratio) - 1
    return torch.where(s_end <= t, 0.0, float("-inf")).to(dtype)


__all__ = [
    "LeftBoundaryExchange",
    "get_cp_group",
    "exchange_boundary_kv",
    "build_global_pool",
    "compressor_boundary_rows",
    "compressed_causal_mask",
]
