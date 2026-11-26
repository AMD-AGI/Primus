# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""e5m2 dV/dK sink of the FlyDSL attention backward (Arbor R15 M-E1).

DEFAULT OFF: nothing calls this unless DLRM_HSTU_FP8_BWD_DKDV_E5M2=1 (the
ordinary-layer backward in triton_hstu_preprocess_and_attention builds a sink
and passes it as
`hstu_flydsl_mha.flydsl_hstu_attention_bwd(..., dkdv_fp8_sink=sink)`).

With the sink, the bwd_dkdv epilogue (compile_hstu_bwd_dkdv(e8emit="only"))
writes, for every dV/dK element, the e5m2 byte the shipped cast
`bwd_fp8.quantize_amax_colsum` writes for it -- the BF16 value the kernel would
store (today's v_cvt_pk_bf16_f32, RNE), back to f32, times the layer's delayed
inv_scale, clamped to +-57344, v_cvt_scalef32_pk_bf8_f32 with scale 1.0 (the
instruction Triton emits for `.to(tl.float8e5)` on gfx950) -- INSTEAD of the
BF16 dV/dK stores ("only": nothing may read BF16 dV/dK of an engaged call).
Payload layout is the dz8 [Mp, 4*H*D] u|v|q|k buffer: dV -> cols
[H*D, 2*H*D), dK -> cols [3*H*D, 4*H*D), row = token.  Rows >= M (padding)
are NOT written.

Partials (deterministic, no atomics), row index prow = pfx[b] + n_block with
pfx = exclusive prefix of ceil(L_b / 128) (the same rows as R14's dq partials):
  * pamax [R_bound, H*32]: max |bf16| per (prow, head, wave, 16-lane row);
  * pcol  [R_bound, 2*H*D]: fp32 column sums over the block's 128 key rows,
    cols t*H*D + h*D + d (t = 0 dV, 1 dK); in-kernel order: one bf16 MFMA
    (A = ones) per 32 rows, then the 4 waves summed in fixed order.
`finish()` reduces them (2-stage, fixed order) to (amax, colsum[2*H*D]).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_FIN_PROGS: int = 256
_BN: int = 128  # bwd_dkdv n block (BN) == rows per partial


@triton.jit
def _e8_pfx_kernel(so_ptr, pfx_ptr, B, BB: tl.constexpr):
    """pfx[0..B] = exclusive prefix of ceil(L_b / 128) (one program)."""
    i = tl.arange(0, BB)
    m = i < B
    a = tl.load(so_ptr + i, mask=m, other=0).to(tl.int64)
    b = tl.load(so_ptr + i + 1, mask=m, other=0).to(tl.int64)
    nb = tl.where(m, (b - a + 127) // 128, 0).to(tl.int32)
    c = tl.cumsum(nb, 0)
    tl.store(pfx_ptr + i, c - nb, mask=m)
    tl.store(pfx_ptr + 1 + i, c, mask=i == B - 1)


@triton.jit
def _e8_fin_kernel(pcol_ptr, pamax_ptr, pfx_ptr, B, ocol_ptr, oam_ptr,
                   RPP: tl.constexpr, NC: tl.constexpr, AK: tl.constexpr,
                   BLK: tl.constexpr):
    """Stage 1: program p sums rows [p*RPP, (p+1)*RPP) of pcol [R, NC] and
    maxes pamax [R, AK]; rows >= R = pfx[B] masked; fixed order."""
    pid = tl.program_id(0)
    R = tl.load(pfx_ptr + B).to(tl.int64)
    cols = tl.arange(0, NC)
    ak = tl.arange(0, AK)
    acc = tl.zeros((BLK, NC), tl.float32)
    am = tl.zeros((BLK, AK), tl.float32)
    r0 = pid.to(tl.int64) * RPP
    for r in range(0, RPP, BLK):
        rr = r0 + r + tl.arange(0, BLK)
        m = rr < R
        acc += tl.load(pcol_ptr + rr[:, None] * NC + cols[None, :],
                       mask=m[:, None], other=0.0)
        am = tl.maximum(am, tl.load(pamax_ptr + rr[:, None] * AK
                                    + ak[None, :], mask=m[:, None], other=0.0))
    tl.store(ocol_ptr + pid * NC + cols, tl.sum(acc, axis=0))
    tl.store(oam_ptr + pid, tl.max(tl.max(am, axis=1), axis=0))


def partial_rows_bound(n_rows: int, batch: int) -> int:
    """Host bound on sum_b ceil(L_b / 128) without a device sync."""
    return (int(n_rows) + 127 * int(batch)) // _BN + 1


class DkdvE5m2Sink:
    """The `dkdv_fp8_sink` of `flydsl_hstu_attention_bwd` (single use).

    dz8: [Mp, 4*H*D] contiguous float8_e5m2 (or uint8) buffer, Mp >= M; only
    the dV/dK columns of rows < M are written.  inv_scale: fp32 device tensor,
    element 0 = 1/scale.  The caller passes ONE immutable scale epoch per
    layer call: inv_scale and decode_scale are device snapshots of the
    layer's published (inv_scale, scale) taken together before the attention
    backward, so a later in-place refresh of the live state cannot change
    what this call casts or decodes with.
    """

    mode = "only"

    def __init__(self, dz8: torch.Tensor, inv_scale: torch.Tensor,
                 decode_scale: Optional[torch.Tensor] = None) -> None:
        self.dz8 = dz8
        self.inv_scale = inv_scale
        self.decode_scale = decode_scale
        self.partials = None
        self.filled = False
        self.declined: Optional[str] = None

    def release(self) -> None:
        """Drop dz8 and the partials (an exact fallback no longer needs
        them; frees the bytes before the shipped path allocates its own)."""
        self.dz8 = None
        self.partials = None

    def decline_reason(self, config: dict, heads: int, dim: int,
                       n_rows: int) -> Optional[str]:
        valu = set(str(config.get("valu", "")).split(","))
        mss = set(str(config.get("ms_sched", "")).split(","))
        if self.filled:
            return "sink already filled"
        if not config.get("dqsplit") or int(config.get("msplit", 0)) != 1:
            return "needs the dq-split msplit=1 kernel"
        if not {"nobr", "msat"} <= valu:
            return f"needs valu nobr,msat (valu={config.get('valu')!r})"
        if "mc" not in mss or {"b2e", "b2ew"} & mss:
            return f"needs ms_sched mc without b2e/b2ew ({config.get('ms_sched')!r})"
        if int(config.get("BN", 128)) != _BN:
            return "BN != 128"
        z = self.dz8
        if z.dim() != 2 or not z.is_contiguous() or z.shape[1] != 4 * heads * dim \
                or z.shape[0] < n_rows or z.element_size() != 1:
            return f"dz8 {tuple(z.shape)} {z.dtype} != [>= {n_rows}, {4 * heads * dim}] 1-byte"
        s = self.inv_scale
        if s.dtype is not torch.float32 or s.numel() < 1 or s.device != z.device:
            return "inv_scale must be an fp32 tensor on the dz8 device"
        return None

    def launch_kwargs(self, seq_offsets: torch.Tensor, n_rows: int,
                      heads: int, dim: int) -> dict:
        """Partial buffers (uninitialized: the finalize masks rows >= R) and
        the row prefix (one tiny kernel, no host sync)."""
        dev = self.dz8.device
        B = seq_offsets.numel() - 1
        pfx = torch.empty(B + 1, dtype=torch.int32, device=dev)
        _e8_pfx_kernel[(1,)](seq_offsets, pfx, B,
                             BB=triton.next_power_of_2(max(B, 2)), num_warps=4)
        rb = partial_rows_bound(n_rows, B)
        pamax = torch.empty((rb, heads * 32), dtype=torch.float32, device=dev)
        pcol = torch.empty((rb, 2 * heads * dim), dtype=torch.float32,
                           device=dev)
        self.partials = (pamax, pcol, pfx)
        self.filled = True
        inv = self.inv_scale.reshape(-1)
        if inv.numel() != 1:
            inv = inv[:1]
        return dict(dz8=self.dz8.view(torch.uint8), e8inv=inv.contiguous(),
                    e8pfx=pfx, e8pam=pamax, e8pcs=pcol)

    def finish(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(amax of |bf16 dV|dK|, colsum [2*H*D] = [dV cols | dK cols])."""
        assert self.filled and self.partials is not None
        pamax, pcol, pfx = self.partials
        rb, nc = pcol.shape
        ak = pamax.shape[1]
        B = pfx.numel() - 1
        rpp = triton.cdiv(rb, _FIN_PROGS)
        blk = 8 if ak <= 128 else (4 if ak <= 512 else 1)
        rpp = triton.cdiv(rpp, blk) * blk
        ocol = torch.empty((_FIN_PROGS, nc), dtype=torch.float32,
                           device=pcol.device)
        oam = torch.empty((_FIN_PROGS,), dtype=torch.float32, device=pcol.device)
        # waves_per_eu=4 (ROCm 10.2 port, G2): register allocation only, bitwise.
        # Triton 3.9 caps this kernel at 77 VGPRs and then reuses one load
        # destination per row block (load, s_waitcnt vmcnt(0), add, load ...),
        # so the BLK x NC tile loads run one at a time: 16.7 -> 25.7 us per
        # call at the production launch (RPP 80, NC 1024, AK 128, BLK 8).
        # With waves_per_eu=4 the loads stay in flight: 14.5 us on Triton 3.9,
        # 14.6 us on 3.6 (same node, replayed launch, triton-perf tune1).
        _e8_fin_kernel[(_FIN_PROGS,)](pcol, pamax, pfx, B, ocol, oam, RPP=rpp,
                                      NC=nc, AK=ak, BLK=blk, num_warps=4,
                                      waves_per_eu=4)
        return oam.amax(), ocol.sum(dim=0)
