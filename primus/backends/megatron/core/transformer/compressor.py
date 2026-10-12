###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 Compressor.

Reference: techblog §1.3 ("Compressor: the Long-Range Compression Branch") and
the diagrams in ``deepseek-v4/develop/techblog/diagrams/csa.png`` /
``hca.png``.

Two configurations:

* ``ratio == 4`` (CSA branch) — overlap mode, ``coff == 2``: each compressed
  token sees an effective window of ``2*ratio`` raw tokens (current window
  plus the previous window's "leftover-half" channels). This smooths
  boundary effects between adjacent compressed positions.
* ``ratio == 128`` (HCA branch) — non-overlap mode, ``coff == 1``: each
  compressed token covers exactly ``ratio`` raw tokens.

Compressor returns the pooled KV after a final ``kv_norm`` (RMSNorm). RoPE
at the compress branch theta is applied **outside** this module by the
caller (the dual-RoPE module produced in P4.3 + the CSA / HCA modules in
P4.4 will consume the output).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.keep_in_fp32 import (
    KeepInFp32Mixin,
    mark_keep_in_fp32,
)
from primus.backends.megatron.core.transformer.local_rmsnorm import LocalRMSNorm


class Compressor(KeepInFp32Mixin, nn.Module):
    """V4 Compressor block.

    Args:
        hidden_size: input feature dim ``D``.
        head_dim: output channel dim per compressed position.
        ratio: compression ratio ``m``. Must be a divisor of the runtime
            sequence length (``S % ratio == 0``).
        overlap: whether to use the overlap-stitched mode. If ``None``,
            defaults to ``ratio == 4`` (the V4 convention).
        rmsnorm_eps: RMSNorm stability eps.

    Shapes:
        Forward input  ``hidden``: ``[B, S, D]``.
        Forward output ``pooled``: ``[B, S // ratio, head_dim]``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        ratio: int,
        overlap: Optional[bool] = None,
        rmsnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.ratio = ratio
        self.overlap = bool(ratio == 4 if overlap is None else overlap)
        # coff is the projection multiplier — overlap mode needs 2x the
        # channels because half goes to the "current window" half and half to
        # the "previous window" half.
        self.coff = 2 if self.overlap else 1

        proj_out = self.coff * head_dim
        self._proj_out = proj_out
        # Plain ``nn.Linear``, deliberately: the compressor has no low-precision
        # path. It builds the KV entries every query reads and the pooling
        # weights that mix them, and the open-source reference keeps it out of
        # FP8 for that reason -- it wraps exactly this projection plus the
        # indexer in an fp8-disabled context. ``nn.Linear`` is also immune to the
        # enclosing TE / Turbo quantization context, so "high precision" here
        # needs no guard beyond not adding a quantized branch.
        #
        # Fuse the kv + gate projections into ONE [hidden -> 2*proj_out] GEMM
        # (default-on): ~1.5x on the projection and one launch instead of two.
        # PRIMUS_COMPRESS_FUSE_PROJ=0 restores the two separate linears.
        self._fuse_proj = os.environ.get("PRIMUS_COMPRESS_FUSE_PROJ", "1") != "0"
        if self._fuse_proj:
            self.wkv_gate = nn.Linear(hidden_size, 2 * proj_out, bias=False)
        else:
            self.wkv = nn.Linear(hidden_size, proj_out, bias=False)
            self.wgate = nn.Linear(hidden_size, proj_out, bias=False)

        # Learnable absolute position embedding (APE) added on top of the
        # softmax score. After overlap, the effective window length is
        # ``2*ratio`` slots of size ``head_dim``; in non-overlap mode it's
        # ``ratio`` slots of size ``head_dim``.
        # FP32 in the released checkpoint. The pooling path promotes with
        # ``score.float()`` before its softmax, so following the model dtype
        # costs stored resolution only; pinning it is opt-in via
        # ``PRIMUS_V4_KEEP_FP32`` (see ``keep_in_fp32``).
        ape_len = 2 * ratio if self.overlap else ratio
        self.ape = nn.Parameter(torch.zeros(ape_len, head_dim, dtype=torch.float32))
        nn.init.normal_(self.ape, std=0.02)
        mark_keep_in_fp32(self.ape)

        self.kv_norm = LocalRMSNorm(head_dim, eps=rmsnorm_eps)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Bridge checkpoints across the fused/unfused projection layouts.

        Old checkpoints store ``wkv.weight`` + ``wgate.weight``; the fused path
        wants ``wkv_gate.weight`` = ``cat([wkv, wgate])`` (and vice-versa). Remap
        in-place so either layout loads under either runtime setting.
        """
        wkv_k, wgate_k, fused_k = prefix + "wkv.weight", prefix + "wgate.weight", prefix + "wkv_gate.weight"
        if self._fuse_proj and wkv_k in state_dict and fused_k not in state_dict:
            state_dict[fused_k] = torch.cat([state_dict.pop(wkv_k), state_dict.pop(wgate_k)], dim=0)
        elif (not self._fuse_proj) and fused_k in state_dict and wkv_k not in state_dict:
            w = state_dict.pop(fused_k)
            state_dict[wkv_k], state_dict[wgate_k] = w[: self._proj_out], w[self._proj_out :]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    class _FusedCompactGather(torch.autograd.Function):
        """Autograd wrapper around dsv4_cp's compressor_input_compact kernels.

        The package ships forward and backward as two BARE functions -- nothing connects
        them to the graph, so using the kernel in training requires this. Without it the
        gather would silently detach: the compressed keys would still be computed, but no
        gradient would reach the hidden states that produced them.

        The forward is a pure gather (each output row is a copy of exactly one input row),
        so the backward is a scatter-add of the incoming gradient, which is what the
        package's ``_bwd`` computes -- returning gradients for the local rows and for the
        boundary rows separately. The boundary gradient then rides
        :class:`LeftBoundaryExchange` back to the neighbour that owns those rows.
        """

        @staticmethod
        def forward(
            ctx,
            hidden_local,
            boundary_hidden,
            cu_seqlens,
            backend,
            global_start,
            l_local,
            ratio,
            d_comp,
            d_window,
            compact_len,
            row_width,
        ):
            ctx.backend = backend
            ctx.args = (cu_seqlens, global_start, l_local, ratio, d_comp, d_window, compact_len, row_width)
            ctx.shapes = (hidden_local.shape, boundary_hidden.shape)
            out, comp_ids = backend.compressor_input_compact_fwd(
                hidden_local.contiguous(),
                boundary_hidden.contiguous(),
                cu_seqlens,
                global_start,
                l_local,
                ratio,
                d_comp,
                d_window,
                compact_len,
                row_width,
            )
            ctx.mark_non_differentiable(comp_ids)
            return out, comp_ids

        @staticmethod
        def backward(ctx, grad_out, _grad_comp_ids):
            cu, gs, l_local, ratio, d_comp, d_window, compact_len, row_width = ctx.args
            # The torch_native reference derives its backward by running autograd on a
            # throwaway graph (`out.backward(...)` inside the _bwd function). Doing that
            # while the engine is already unwinding OUR backward needs grad explicitly
            # enabled -- backward runs under no-grad by default, so the throwaway graph
            # would have no grad_fn and the inner call would raise. The FlyDSL backend
            # computes its scatter directly and is unaffected, but the wrapper has to work
            # for both since they are meant to be interchangeable.
            with torch.enable_grad():
                g_hidden, g_boundary = ctx.backend.compressor_input_compact_bwd(
                    grad_out.detach().contiguous(),
                    cu,
                    gs,
                    l_local,
                    ratio,
                    d_comp,
                    d_window,
                    compact_len,
                    row_width,
                )
            h_shape, b_shape = ctx.shapes

            # Either gradient can legitimately be None: on rank 0 no window reaches into
            # the boundary buffer, so those rows never participate. Autograd needs a
            # zero tensor rather than None there, because the boundary is a real graph
            # input (it comes from LeftBoundaryExchange) and dropping its gradient would
            # silently stop the neighbour's share of the update.
            def _or_zeros(g, shape):
                return (
                    torch.zeros(shape, dtype=grad_out.dtype, device=grad_out.device)
                    if g is None
                    else g.reshape(shape)
                )

            return (
                _or_zeros(g_hidden, h_shape),
                _or_zeros(g_boundary, b_shape),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    @staticmethod
    def _fused_compact_backend():
        """The dsv4_cp layout backend, or None when the package is not installed.

        Optional on purpose: the PyTorch window gather is the reference implementation and
        is what every test validates. The fused kernel is an alternative, selected with
        PRIMUS_THD_COMPACT_BACKEND, and a contract test pins the two bit-identical.
        """
        name = os.environ.get("PRIMUS_THD_COMPACT_BACKEND", "").strip()
        if not name or name == "torch":
            return None
        try:
            from dsv4_cp_layout.backends import get as _get

            return _get(name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"PRIMUS_THD_COMPACT_BACKEND={name} requested but the dsv4_cp package "
                f"could not provide it: {type(exc).__name__}: {exc}"
            ) from exc

    @property
    def thd_d_comp(self) -> int:
        """Left-neighbour lookahead, in rows, that this compressor needs under CP.

        A window belongs to the rank holding its LAST row, so its earlier rows may sit on
        the previous rank. The worst-case gap is exactly this many rows -- verified by
        exhaustive search over random ragged layouts, where the maximum gap equals d_comp
        and is attained whenever ``(global_start - seq_start - d_comp) % ratio == 0``.

        Overlap mode (ratio 4 / CSA) needs a whole extra window on top: the stitch pairs
        window i with window i-1, so the PREDECESSOR's rows must also be in this rank's
        compact buffer. Exhaustively: ratio=4 fails 131/500 layouts at d_comp=6, 583 at 5
        and 856 at 4, but 0 at 7 (= 2*ratio-1). Upstream rounds that up to 2*ratio.
        """
        return 2 * self.ratio if self.overlap else self.ratio

    def thd_capacity(self, l_local: int, group_alignment: int = 0) -> int:
        """Slots to reserve for this rank's compressed windows.

        Replaces the "every rank has the same number of windows" property that sequence
        alignment used to provide. Each window occupies ``ratio`` distinct rows drawn from
        ``[global_start - d_comp, global_end)``, so a rank can own at most
        ``(l_local + d_comp) / ratio`` of them -- never exceeded in exhaustive search.

        A FIXED capacity is what keeps the pool all-gather well-formed:
        :class:`_AllGatherPool` sizes its receive buffers with
        ``torch.empty_like(pool_local)``, so two ranks disagreeing on the pool width is a
        shape mismatch, not a recoverable ragged gather. (The count itself is never zero
        -- :meth:`thd_capacity` floors it at 1 -- and no rank ever skips the collective.)
        """
        cap = max(1, (int(l_local) + self.thd_d_comp) // self.ratio)
        if group_alignment <= 0:
            import math

            group_alignment = 32 // math.gcd(32, self.ratio)
        return ((cap + group_alignment - 1) // group_alignment) * group_alignment

    def thd_compact_plan(self, cu_seqlens: torch.Tensor, global_start: int, l_local: int):
        """Which windows this rank owns, in the fixed-capacity compact layout.

        Returns ``(row_idx [c_cap, ratio], comp_ids [c_cap], seq_ids [c_cap])`` where
        ``comp_ids[j]`` is the window's index WITHIN ITS SEQUENCE (also its compress-base
        RoPE phase) and ``-1`` marks an unused slot. ``row_idx`` holds GLOBAL row indices;
        the caller maps them into its local buffer or its boundary buffer.

        Ownership: window g of sequence s is this rank's iff its last row
        ``seq_start + (g+1)*ratio - 1`` lies in ``[global_start, global_end)``. That rule
        is what makes ownership computable without any cross-rank communication.
        """
        device = cu_seqlens.device
        cu = cu_seqlens.to(torch.int64)
        starts, ends = cu[:-1], cu[1:]
        n_full = (ends - starts) // self.ratio  # whole windows per sequence
        n_seq = starts.numel()
        c_cap = self.thd_capacity(l_local)
        g_end = int(global_start) + int(l_local)

        # Enumerate every window in the pack, then keep the ones this rank owns. The pack
        # is small next to the row axis (one entry per window, not per token).
        total = int(n_full.sum().item())
        if total == 0:
            empty = torch.full((c_cap,), -1, dtype=torch.int64, device=device)
            return None, empty, empty.clone()
        seq_of = torch.repeat_interleave(torch.arange(n_seq, device=device), n_full)
        cum = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), n_full.cumsum(0)])
        comp_of = torch.arange(total, device=device) - cum[:-1][seq_of]
        last_row = starts[seq_of] + (comp_of + 1) * self.ratio - 1
        mine = (last_row >= int(global_start)) & (last_row < g_end)

        sel_seq, sel_comp = seq_of[mine], comp_of[mine]
        n_mine = sel_seq.numel()
        if n_mine > c_cap:
            raise RuntimeError(
                f"compressed windows ({n_mine}) exceed capacity ({c_cap}) for "
                f"l_local={l_local}, ratio={self.ratio}; the capacity formula is wrong."
            )
        comp_ids = torch.full((c_cap,), -1, dtype=torch.int64, device=device)
        seq_ids = torch.full((c_cap,), -1, dtype=torch.int64, device=device)
        comp_ids[:n_mine], seq_ids[:n_mine] = sel_comp, sel_seq

        first_row = starts[sel_seq] + sel_comp * self.ratio
        # Unused slots must still hold a LEGAL global row, not 0: the caller subtracts
        # global_start to get a buffer offset, and 0 - global_start is negative on every
        # rank but the first. Their contents are never read (the visibility masks drop
        # comp_id == -1), so point them at this rank's own first row.
        row_idx = torch.full((c_cap, self.ratio), int(global_start), dtype=torch.int64, device=device)
        row_idx[:n_mine] = first_row.unsqueeze(1) + torch.arange(self.ratio, device=device)
        return row_idx, comp_ids, seq_ids

    def thd_window_plan(self, cu_seqlens: torch.Tensor):
        """Per-sequence window layout for packed (THD) input.

        Returns ``(row_idx [N, ratio], is_first [N], cu_pool [n_seq + 1])`` where window
        ``w`` pools rows ``row_idx[w]``, ``is_first[w]`` marks the first window OF ITS
        SEQUENCE, and ``cu_pool`` gives each sequence's slice of the pool.

        A sequence of length ``L`` contributes ``L // ratio`` whole windows; its trailing
        ``L % ratio`` rows contribute no compressed key. That is the same floor rule the
        contiguous path uses (``N = S // ratio``), just applied per sequence instead of
        once over the whole pack -- which is the entire point, because a window spanning a
        boundary would blend two different samples into one compressed key.
        """
        device = cu_seqlens.device
        cu = cu_seqlens.to(torch.int64)
        lens = cu[1:] - cu[:-1]
        n_per = lens // self.ratio  # whole windows per sequence
        cu_pool = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), n_per.cumsum(0)])
        N = int(cu_pool[-1].item())
        if N == 0:
            return None, None, cu_pool
        seq_of_win = torch.repeat_interleave(torch.arange(lens.numel(), device=device), n_per)  # [N]
        win_in_seq = torch.arange(N, device=device) - cu_pool[:-1][seq_of_win]
        start = cu[:-1][seq_of_win] + win_in_seq * self.ratio  # [N]
        row_idx = start.unsqueeze(1) + torch.arange(self.ratio, device=device).unsqueeze(0)
        return row_idx, win_in_seq == 0, cu_pool

    def _reshape_into_windows(self, t: torch.Tensor, row_idx=None) -> torch.Tensor:
        """``[B, S, coff*head_dim]`` → ``[B, N, ratio, coff*head_dim]``.

        Contiguous path: ``N = S // ratio`` by reshape. Packed path: ``row_idx`` from
        :meth:`thd_window_plan` gathers each window's rows, so windows never straddle a
        packed-sequence boundary and a sequence length that is not a whole number of
        windows is fine (its remainder is simply not pooled).
        """
        B, S, C = t.shape
        if row_idx is not None:
            return t[:, row_idx.reshape(-1)].reshape(B, row_idx.shape[0], self.ratio, C)
        assert S % self.ratio == 0, f"Compressor: sequence length {S} not divisible by ratio {self.ratio}"
        N = S // self.ratio
        return t.reshape(B, N, self.ratio, C)

    def _overlap_transform(self, t: torch.Tensor, is_first=None) -> torch.Tensor:
        """``[B, N, ratio, 2*head_dim]`` → ``[B, N, 2*ratio, head_dim]``.

        For window ``i``, the augmented sequence is
        ``[half_a[i], half_b[i-1]]`` concatenated along the per-window
        axis. Window 0's "previous half" is filled with zeros (causal
        padding).

        ``is_first`` (packed input) marks the first window of EACH sequence, not just
        window 0 of the pack. Without it the roll below stitches every sequence's first
        window onto the previous sequence's last one -- a boundary leak that survives
        even when the windows themselves are correctly binned, because it is the stitch
        rather than the pooling that crosses over.
        """
        # Split channels.
        half_a, half_b = torch.chunk(t, 2, dim=-1)  # each [B, N, ratio, head_dim]
        # Roll along the window dim so half_b[i] becomes "previous-window's b" of i+1.
        half_b_prev = torch.cat(
            [torch.zeros_like(half_b[:, :1]), half_b[:, :-1]],
            dim=1,
        )
        if is_first is not None:
            half_b_prev = half_b_prev.masked_fill(is_first.view(1, -1, 1, 1).to(half_b_prev.device), 0.0)
        # Concat along the per-window token axis.
        return torch.cat([half_a, half_b_prev], dim=2)  # [B, N, 2*ratio, head_dim]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        global_start: int = 0,
        boundary_hidden: torch.Tensor = None,
    ) -> torch.Tensor:
        """Pool ``hidden[B, S, D]`` to ``[B, P, head_dim]``.

        ``cu_seqlens`` switches to packed (THD) pooling: windows are laid out per sequence
        instead of over the whole row axis, and the result has the fixed capacity
        :meth:`thd_capacity` with unused slots left zero. ``None`` reproduces the original
        contiguous behaviour exactly.

        ``global_start`` / ``boundary_hidden`` are the context-parallel part. A window
        belongs to the rank holding its last row, so its leading rows can sit on the
        previous rank; ``boundary_hidden`` is that rank's trailing ``d_window`` rows, and
        row ``0`` of it corresponds to global row ``global_start - d_window``.

        With ``global_start > 0``, ``boundary_hidden`` is REQUIRED and must be at least
        :attr:`thd_d_comp` rows -- ``2 * ratio`` in overlap mode, because the stitch also
        needs the predecessor window. Too short and this raises rather than quietly
        pooling a truncated window.
        """
        row_idx = is_first = None
        if cu_seqlens is not None:
            # Fixed-capacity compact layout: this rank's windows in slots [0, n_mine) and
            # -1 in the rest. The capacity is constant across ranks because the pool
            # all-gather allocates its receive buffers from the LOCAL width, so unequal
            # widths would be a shape mismatch. Padding slots are excluded by the
            # visibility masks, which key on comp_ids == -1.
            l_local = hidden.shape[1]
            gs = int(global_start)
            row_idx_g, comp_ids, _ = self.thd_compact_plan(cu_seqlens, gs, l_local)
            c_cap = comp_ids.numel()
            if row_idx_g is None:
                # No sequence in this pack is long enough to fill even one window, so the
                # pool is genuinely empty -- routine for HCA (ratio=128) on short packed
                # samples, where a segment needs 128 tokens to contribute anything.
                #
                # Returning a bare `new_zeros` here would be a DETACHED constant: correct
                # in value, but it severs wkv_gate / ape / kv_norm from the graph, so they
                # receive no gradient at all. Megatron's DDP notices (its grad buffer
                # asserts every parameter fired a grad-ready hook) and the job dies on
                # iteration 2 with a message that names no parameter -- which is a long
                # way from "this batch had no 128-token sequence". Under a DDP that did
                # not check, it would instead be silent.
                #
                # So build the zeros THROUGH the projections: pool one dummy window and
                # multiply by zero. Same value, same shape, but the parameters stay in the
                # graph and take a well-defined zero gradient.
                B_ = hidden.shape[0]
                dummy = hidden[:, :1].expand(-1, self.ratio, -1)
                rows = torch.arange(self.ratio, device=hidden.device).reshape(1, self.ratio)
                pooled = self._pool_windows(
                    dummy, rows, torch.ones(1, dtype=torch.bool, device=hidden.device)
                )
                return (pooled * 0.0).expand(B_, c_cap, self.head_dim)
            # comp_ids doubles as the "first window of its sequence" flag: the overlap
            # stitch must not reach back past a sequence start.
            is_first = comp_ids <= 0
            # Global rows -> buffer rows. Under CP the leading rows of a window may live
            # on the previous rank, so prepend boundary_hidden and shift every index by
            # its length: the concatenated buffer is [boundary ++ local] and a global row
            # r maps to r - global_start + d_window. Rows before the boundary window
            # cannot occur -- d_comp is exactly the worst-case reach.
            d_window = 0 if boundary_hidden is None else boundary_hidden.shape[1]
            backend = self._fused_compact_backend()
            if backend is not None:
                # Fused path: the kernel gathers the hidden rows straight into the compact
                # buffer, so the projections below run on [c_cap*ratio, D] instead of on
                # the whole shard. Gather-then-project is numerically identical to
                # project-then-gather because the projections are per-row.
                B_, _, D_ = hidden.shape
                if B_ != 1:
                    raise RuntimeError(f"fused compact gather assumes B=1, got {B_}")
                bnd = (
                    boundary_hidden.reshape(d_window, D_)
                    if d_window
                    else hidden.new_zeros(self.thd_d_comp, D_)
                )
                compact, kernel_ids = Compressor._FusedCompactGather.apply(
                    hidden.reshape(l_local, D_),
                    bnd,
                    cu_seqlens.to(torch.int32),
                    backend,
                    gs,
                    l_local,
                    self.ratio,
                    self.thd_d_comp,
                    bnd.shape[0],
                    c_cap * self.ratio,
                    D_,
                )
                hidden = compact.reshape(1, c_cap * self.ratio, D_)
                # The kernel enumerates windows by upstream's reach-back rule, which is
                # NOT this class's strict last-row ownership: it also emits up to
                # d_comp // ratio leading windows whose last row precedes global_start.
                # Upstream can afford that because it addresses pool entries through a
                # seq_to_rank_row table that simply never points at the duplicates; Primus
                # has no such table -- it concatenates the ranks' pools and uses the
                # column number directly -- so a mismatch here does not merely reorder the
                # buffer, it silently pairs every row group with the wrong identity
                # (comp_id drives the RoPE phase, is_first drives the stitch). Refuse
                # rather than train on it. The two agree whenever no sequence starts
                # before global_start, which covers the non-CP case.
                if not torch.equal(kernel_ids.to(torch.int64).cpu(), comp_ids.cpu()):
                    raise RuntimeError(
                        "the fused compact gather enumerates different windows than "
                        "thd_compact_plan (kernel uses upstream's reach-back rule, Primus "
                        "uses strict last-row ownership), so the gathered rows would be "
                        "paired with the wrong comp_id / is_first. Unset "
                        "PRIMUS_THD_COMPACT_BACKEND to use the PyTorch gather."
                    )
                # Rows are already in window order, so the later reshape is a no-op view.
                row_idx = torch.arange(c_cap * self.ratio, device=hidden.device).reshape(c_cap, self.ratio)
                return self._pool_windows(hidden, row_idx, is_first)
            if d_window:
                hidden = torch.cat([boundary_hidden, hidden], dim=1)
            row_idx = row_idx_g - gs + d_window
            # Unused capacity slots must pool to ZERO, matching what the fused kernel
            # writes (its compact buffer starts zeroed). Their contents are never read --
            # the visibility masks drop comp_id == -1 -- but the two backends have to be
            # value-identical to be interchangeable, and zero is the safer convention: if
            # a mask ever missed one, it would contribute nothing rather than a copy of
            # some real row. Achieved by pointing them at a zero row appended to the
            # buffer. Keyed on the MASK, not on a prefix count: the owned slots are a
            # prefix today, but a prefix assumption breaks silently (it zeroes a real
            # window instead of a padding one) the moment that stops holding.
            unused = comp_ids < 0
            if bool(unused.any()):
                hidden = torch.cat([hidden, hidden.new_zeros(hidden.shape[0], 1, hidden.shape[2])], dim=1)
                row_idx = row_idx.clone()
                row_idx[unused] = hidden.shape[1] - 1
            # The overlap stitch pairs window k with window k-1, and _overlap_transform
            # takes the predecessor from the PRECEDING SLOT. On a rank whose first owned
            # window is not its sequence's window 0, that predecessor belongs to the left
            # neighbour and has no slot here, so the stitch would silently use the
            # hardcoded zeros meant for a sequence start -- an O(1) error on one
            # compressed key per rank per layer. Its rows are already available (that is
            # exactly why thd_d_comp is 2 * ratio for overlap), so pool it locally as a
            # leading SHADOW slot and drop it again before returning: it is the previous
            # rank's window and must not enter the all-gathered pool twice.
            shadow = self.overlap and c_cap > 0 and int(comp_ids[0].item()) > 0
            if shadow:
                # Predecessor of the first owned window occupies the ratio rows directly
                # before it, in the same sequence (guaranteed by comp_ids[0] > 0).
                row_idx = torch.cat([row_idx[:1] - self.ratio, row_idx], dim=0)
                # Slot 0 is the shadow; it gets zeros for ITS predecessor, which is fine
                # because it is discarded. The first real window is no longer slot 0, so
                # it now reads its true predecessor instead of zeros.
                is_first = torch.cat([torch.ones(1, dtype=torch.bool, device=is_first.device), is_first])
            if int(row_idx.min().item()) < 0:
                raise RuntimeError(
                    f"a compressed window reaches {-int(row_idx.min().item())} rows before "
                    f"the boundary buffer (d_window={d_window}, ratio={self.ratio}, "
                    f"d_comp={self.thd_d_comp}). The boundary exchange is too short."
                )
            pooled = self._pool_windows(hidden, row_idx, is_first)
            # Drop the shadow again: the emitted pool must stay exactly thd_capacity wide
            # on every rank, or the all-gather collective has mismatched shapes and the
            # column numbering that addresses it goes wrong.
            return pooled[:, 1:] if shadow else pooled

        return self._pool_windows(hidden, row_idx, is_first)

    def _pool_windows(self, hidden, row_idx, is_first):
        """Project, cut into windows, and softmax-pool -> ``[B, N, head_dim]``.

        Shared by the contiguous path, the PyTorch packed path and the fused packed path.
        The three differ only in how ``hidden`` and ``row_idx`` were produced -- the fused
        gather hands over a compact buffer whose rows are already in window order, so its
        ``row_idx`` is just ``arange`` -- and factoring the rest out is what keeps them
        numerically interchangeable rather than three drifting copies.
        """
        if self._fuse_proj:
            kv_proj, score_proj = self.wkv_gate(hidden).split(self._proj_out, dim=-1)
        else:
            kv_proj = self.wkv(hidden)  # [B, S, coff*head_dim]
            score_proj = self.wgate(hidden)  # [B, S, coff*head_dim]

        kv = self._reshape_into_windows(kv_proj, row_idx)  # [B, N, ratio, coff*head_dim]
        score = self._reshape_into_windows(score_proj, row_idx)  # [B, N, ratio, coff*head_dim]

        if self.overlap:
            kv = self._overlap_transform(kv, is_first)  # [B, N, 2*ratio, head_dim]
            score = self._overlap_transform(score, is_first)  # [B, N, 2*ratio, head_dim]
        # else: kv / score already at [B, N, ratio, head_dim]

        # Per-window-softmax pool: APE bias + softmax over the window axis (dim=2)
        # + weighted sum -- each compressed token is a softmax-weighted average of
        # its window members. The forward burst (add + cast + softmax + cast + mul
        # + reduce) is fused into one Triton launch on CUDA fp16/bf16/fp32 inputs;
        # PRIMUS_COMPRESS_POOL_TRITON=0 (or non-CUDA / unsupported dtype) falls back
        # to eager.
        if (
            os.environ.get("PRIMUS_COMPRESS_POOL_TRITON", "1") != "0"
            and kv.is_cuda
            and kv.dtype
            in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            )
        ):
            from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.compressor_pool import (
                fused_softmax_weighted_pool,
            )

            pooled = fused_softmax_weighted_pool(kv, score, self.ape)  # [B, N, head_dim]
        else:
            score = score + self.ape  # [B, N, win, head_dim]
            weights = F.softmax(score.float(), dim=2).to(kv.dtype)
            pooled = (kv * weights).sum(dim=2)  # [B, N, head_dim]

        pooled = self.kv_norm(pooled)
        return pooled


__all__ = ["Compressor"]
