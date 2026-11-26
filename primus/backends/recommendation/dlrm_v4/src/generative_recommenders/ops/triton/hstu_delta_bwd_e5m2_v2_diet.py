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

"""DLRM_HSTU_FP8_DELTA_BWD_V2_DIET: VALU-diet form of the VKP_V2 delta backward.

Arbor X4C19 item (48)(b') (iter40 F_v2port prototype, wired iter41). Bitwise
drop-in for `hstu_delta_bwd_e5m2._hstu_attn_delta_bwd_e5m2_v2` (gin item
(25)(b), DLRM_HSTU_FP8_BWD_VKP_V2) in the configuration it is written for:
the v2 module constants AMAX_ROWMAX=1, DQ_ELEM=1, LPT=2, NUM_STAGES=1, and
REDUCE_EACH=False, BLOCK_N=32, num_warps=4, D_Q = D_V = 128, bf16 q/dout/dQ,
e4m3 K/V, e5m2 payload, gfx950 (first verified on Triton 3.6.0; any Triton
build is accepted per specialization by the runtime self-check, see below).
`hstu_delta_bwd_e5m2.hstu_attention_delta_bwd_e5m2` dispatches here only for
exactly that configuration and a specialization whose self-check passed, and
keeps the v2 kernel for everything else (and behind
DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=0 / absent).

Outputs are byte-identical to the v2 kernel: payload incl. zeroed pad rows,
dQ, PARTIAL_COL, PARTIAL_AMAX and the returned amax / colsum. Arbor X4C19
iter40 F_v2port: production / in-model / dense pools, both Triton builds
(<= 2 GiB buffer-op and > 2 GiB 64-bit global-memory), 42 edge cases per
build (NaN / sNaN / inf / -0 in K, V, q, dout; product overflow and
subnormals; clamp-heavy and tiny scales; edge lengths; NaN meeting NaN in the
same column), with NaN payload and sign bits included.

v2 already has the row-max amax and the element-wise dq accumulator. The
remaining VALU work per [32, 128] tile is reduced, each change value-preserving:
 * bf16 round trips (k_real, v_real, silu, dqk, dK_b, dV_b): one
   `v_cvt_pk_bf16_f32 d, 0, x` per element (= f32(bf16_rne(x))) instead of
   the conversion plus an unpack.
 * fp32 products as explicit `v_pk_mul_f32` in the v2 build's operand order;
   the clamp is `v_med3_f32 x, -L, L`.
 * the qk / dqk per-thread partial sums in asm, reproducing the v2 build's
   exact fused/unfused sequence and operand order (read off its ISA/IR):
   LLVM would otherwise pick its own split once the surrounding code changes,
   which differs whenever a product overflows or is subnormal, and the
   operand order decides which NaN payload propagates. The cross-lane
   reduction is the stock tl.sum (with a bitcast -0.0 filler: a `-0.0`
   literal reaches LLVM as +0.0 and would turn a -0.0 partial into +0.0).
 * `1 + qk*(1-sig)` keeps its unfused multiply (asm), as v2 does.
 * the running row maxima of |silu| / |dqk| as one `v_max_f32 acc, acc, |x|`.
 * scalar row addressing; in the > 2 GiB (in-model) build the masked tail
   tile runs after the unmasked full tiles (v2 tile order), with the uniform
   multiplier pairs in SGPRs.
Program -> sequence mapping (LPT PERM) and the v2 epilogue (dq / colsum
reductions, bf16(max m_s * max|dout|), bf16((max m_d * max|q|) * |alpha|))
are v2 verbatim.
Bitwise equality depends on asm that mirrors the current Triton/LLVM's
choices for the v2 kernel. Two codegen profiles (constexpr RED):
 * RED_CHAIN (0), the Triton 3.6.0 v2 build: qk / dqk per-thread partial sums
   as one sequential chain (_QK_SPEC / _DQK_SPEC), `1 + qk*(1-sig)` unfused.
 * RED_TREE (1), the Triton 3.9.0 v2 build (ROCm 10.2 image; Triton 3.9 sums a
   thread's 16 elements as a tree over its 8 packed pairs): per pair group
   G_g = fma(a_{2g}, b_{2g}, fl(a_{2g+1} * b_{2g+1})) (pairs of adjacent
   elements, packed ops), (G0 + G1) + (G2 + G3), and the lo + hi add of the
   resulting pair is tl.sum's own last in-thread step (slots 0 and 1 kept,
   the rest -0.0); `1 + qk*(1-sig)` is one fma(qk, 1 - sig, 1). Read off the
   3.9 v2 ISA (v_pk_mul_f32 / v_pk_fma_f32 / v_pk_add_f32 operand order).
 * RED_TREE_DQX (2) = RED_TREE plus dqk's cross-lane butterfly in asm with the
   3.9 v2 build's operand order (its xor-2 step adds own + partner: LLVM packs
   it with qk's lo + hi into one v_pk_add_f32; the stock tl.sum adds partner +
   own). The two differ only in which NaN payload / sign survives when NaNs of
   different payload or sign meet across the threads of one dqk row. The
   Triton 3.9.0 default; verified bitwise against the 3.9 v2 build.
`hstu_delta_bwd_e5m2` arms this kernel only after a runtime bitwise
self-check against the v2 kernel on the same device, Triton build and
argument specialization (no version string): re-verified automatically after
any Triton or LLVM change; a mismatch keeps the v2 kernel.

Measured (Arbor X4C19 iter40 F_v2port, lab, in-model lengths, > 2 GiB build):
163 VGPR (3 waves/SIMD) vs v2's 179 (2 waves/SIMD); 0.767-0.770x the v2 time.
Triton 3.9.0 (ROCm 10.2 image, RED_TREE_DQX; production launch captured at
step 120, 2.4M rows, same node as the v26.3 image): > 2 GiB build 166 VGPR
(3 waves) vs v2 178 (2 waves), 881-896 us vs v2 1091-1098 us and vs the
Triton 3.6 diet 885-894 us; <= 2 GiB build 164 VGPR, 862-878 us vs v2
894-904 us and vs the Triton 3.6 diet 890-894 us.
"""
import torch
import triton
import triton.language as tl

from generative_recommenders.ops.triton.triton_hstu_attention import (
    fast_dividef,
    fast_expf,
)


def _asm_from_spec(spec):
    # Per-thread partial sum over its 16 elements. spec: ("mul0", j) acc = fl(a_j*b_j); ("fma", j)
    # acc = fma(a_j, b_j, acc); ("add", j, acc_first) acc = acc + fl(a_j*b_j) (acc_first: add operand order).
    # $0 acc, $1 scratch; $16+j b_j (k_j / v_j); $32+j a_j (q_j / dout_j).
    a = []
    for s in spec:
        kind, j = s[0], s[1]
        if kind == "mul0":
            a.append(f"v_mul_f32 $0, ${32 + j}, ${16 + j}")
        elif kind == "fma":
            a.append(f"v_fmac_f32 $0, ${32 + j}, ${16 + j}")
        elif kind == "add":
            a.append(f"v_mul_f32 $1, ${32 + j}, ${16 + j}")
            a.append("v_add_f32 $0, $0, $1" if s[2] else "v_add_f32 $0, $1, $0")
        else:
            raise ValueError(s)
    return "\n".join(a)


# The v2 build's sequences (both the <= 2 GiB and the > 2 GiB build, Triton 3.6.0 / gfx950): qk = 16 rounded
# products summed in order; dqk = fl(d1*v1), fma j = 0, 2..5, then rounded products 6..15. Every add takes the
# product as src0 except the first qk add.
_QK_SPEC = [("mul0", 0), ("add", 1, True)] + [("add", j, False) for j in range(2, 16)]
_DQK_SPEC = [("mul0", 1), ("fma", 0), ("fma", 2), ("fma", 3), ("fma", 4), ("fma", 5)] + [
    ("add", j, False) for j in range(6, 16)]
_ASM_QK = tl.constexpr(_asm_from_spec(_QK_SPEC))
_ASM_DQK = tl.constexpr(_asm_from_spec(_DQK_SPEC))
_RED_CONSTRAINTS = tl.constexpr(",".join(["=&v", "=&v"] + ["=v"] * 14 + ["v"] * 32))

# Codegen profiles (constexpr RED of the kernel; see the module docstring).
RED_CHAIN = 0
RED_TREE = 1
RED_TREE_DQX = 2
# The profile tried first per Triton version (the runtime self-check in hstu_delta_bwd_e5m2 decides; a version
# not listed tries every profile). Other versions: PROFILES order.
PROFILE_BY_TRITON = {"3.6.0": RED_CHAIN, "3.9.0": RED_TREE_DQX}
PROFILES = (RED_TREE_DQX, RED_TREE, RED_CHAIN)
PROFILE_NAMES = {RED_CHAIN: "chain", RED_TREE: "tree", RED_TREE_DQX: "tree+dqx"}


def profile_candidates(triton_version: str):
    first = PROFILE_BY_TRITON.get(triton_version)
    return ((first,) if first is not None else ()) + tuple(p for p in PROFILES if p != first)


def _asm_tree():
    # Triton 3.9.0 v2 build, per thread over its 8 packed pairs (pair p = elements 2p, 2p+1 of its 16):
    # G_g = fma(a_pair(2g), b_pair(2g), fl(a_pair(2g+1) * b_pair(2g+1))), g = 0..3; (G0 + G1) + (G2 + G3).
    # Every product / fma takes a (q / dout) as src0 and b (k / v) as src1, every add the left subtree as src0
    # (as v2's v_pk_mul_f32 / v_pk_fma_f32 / v_pk_add_f32). $0..$2 early-clobber pair temps (result in $0),
    # $3..$7 unused outputs; $8+p b pair p, $16+p a pair p.
    a = []
    for dst, g in ((0, 0), (1, 1)):
        a.append(f"v_pk_mul_f32 ${dst}, ${16 + 2 * g + 1}, ${8 + 2 * g + 1}")
        a.append(f"v_pk_fma_f32 ${dst}, ${16 + 2 * g}, ${8 + 2 * g}, ${dst}")
    a.append("v_pk_add_f32 $0, $0, $1")
    for dst, g in ((1, 2), (2, 3)):
        a.append(f"v_pk_mul_f32 ${dst}, ${16 + 2 * g + 1}, ${8 + 2 * g + 1}")
        a.append(f"v_pk_fma_f32 ${dst}, ${16 + 2 * g}, ${8 + 2 * g}, ${dst}")
    a.append("v_pk_add_f32 $1, $1, $2")
    a.append("v_pk_add_f32 $0, $0, $1")
    return "\n".join(a)


_ASM_TREE = tl.constexpr(_asm_tree())
_TREE_CONSTRAINTS = tl.constexpr(",".join(["=&v"] * 3 + ["=v"] * 5 + ["v"] * 16))

# RED_TREE_DQX: dqk's lo + hi add and its 8-lane butterfly (the 8 threads of a row are 8 consecutive lanes) as the
# 3.9 v2 build runs them: lo + hi (lo src0); xor 4 = row_half_mirror then quad_perm [3,2,1,0], partner as src0;
# xor 2 = quad_perm [2,3,0,1] moved, OWN as src0 (v2 packs this add with qk's lo + hi into one v_pk_add_f32);
# xor 1 = quad_perm [1,0,3,2], partner as src0. Every lane keeps its own result (NaN payloads may differ between
# the lanes of a row, as in v2); _keep_first then turns slot 0 into the row value without any cross-lane op.
# s_nop 1: two wait states between a VALU write and a DPP read of that VGPR. $0 result, $1 / $2 temps; $16 lo,
# $17 hi (slots 0 / 1 of the thread's 16).
_ASM_DQX = tl.constexpr("\n".join([
    "v_add_f32 $1, $16, $17",
    "s_nop 1",
    "v_mov_b32_dpp $2, $1 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "s_nop 1",
    "v_add_f32_dpp $1, $2, $1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "s_nop 1",
    "v_mov_b32_dpp $2, $1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "s_nop 1",
    "v_add_f32 $1, $1, $2",
    "s_nop 1",
    "v_add_f32_dpp $0, $1, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
]))
_DQX_CONSTRAINTS = tl.constexpr(",".join(["=&v"] * 3 + ["=v"] * 13 + ["v"] * 16))


@triton.jit
def _bf16_rt(x):
    return tl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, 0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul_nc(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _vmax_abs(acc, x):
    return tl.inline_asm_elementwise(
        "v_max_f32_e64 $0, $1, |$2|", "=v,v,v", [acc, x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _med3(x, lo, hi):
    return tl.inline_asm_elementwise(
        "v_med3_f32 $0, $1, $2, $3", "=v,v,v,v", [x, lo, hi], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _pair(x):
    x0, x1 = tl.split(tl.reshape(x, [x.shape[0], x.shape[1] // 2, 2]))
    return x0.to(tl.uint32, bitcast=True).to(tl.uint64) | (
        x1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )


@triton.jit
def _unpair(w):
    r0 = w.to(tl.uint32).to(tl.float32, bitcast=True)
    r1 = (w >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    j = tl.join(r0, r1)
    return tl.reshape(j, [j.shape[0], j.shape[1] * 2])


@triton.jit
def _pkmul(wa, wb, SP: tl.constexpr):
    if SP:
        # uniform first operand in an SGPR pair (as the v2 build's `v_pk_mul_f32 v, s[..], v`)
        return tl.inline_asm_elementwise(
            "v_pk_mul_f32 $0, $1, $2", "=v,s,v", [wa, wb], dtype=tl.uint64, is_pure=True, pack=1
        )
    else:
        return tl.inline_asm_elementwise(
            "v_pk_mul_f32 $0, $1, $2", "=v,v,v", [wa, wb], dtype=tl.uint64, is_pure=True, pack=1
        )


@triton.jit
def _pkmul_vv(wa, wb):
    return tl.inline_asm_elementwise(
        "v_pk_mul_f32 $0, $1, $2", "=v,v,v", [wa, wb], dtype=tl.uint64, is_pure=True, pack=1
    )


@triton.jit
def _red16(a, b, ASM: tl.constexpr, CONS: tl.constexpr, first):
    # one asm per thread over its 16 elements; slot 0 of each chunk = partial sum, other slots -> -0.0 so that
    # x + -0.0 = x folds away in tl.sum's in-thread chain (the cross-lane DPP reduction is the stock's).
    # The -0.0 is built by bitcast (0x80000000): a `-0.0` literal reaches LLVM as +0.0 and leaves one
    # `v_add_f32 x, 0, x` per chain, which turns a -0.0 partial sum into +0.0 (v2 keeps -0.0).
    r = tl.inline_asm_elementwise(ASM, CONS, [a, b], dtype=tl.float32, is_pure=True, pack=16)
    neg0 = tl.full(r.shape, -2147483648, tl.int32).to(tl.float32, bitcast=True)
    return tl.sum(tl.where(first, r, neg0), axis=1)


@triton.jit
def _keep_first(a, b):
    return a


@triton.jit
def _red_tree_dqx(b_p, a_p):
    # RED_TREE_DQX: the in-thread tree as _red_tree, then lo + hi and the cross-lane butterfly in asm (_ASM_DQX);
    # tl.reduce(_keep_first) picks slot 0 of each thread (no data movement).
    w = tl.inline_asm_elementwise(_ASM_TREE, _TREE_CONSTRAINTS, [b_p, a_p], dtype=tl.uint64, is_pure=True, pack=8)
    r = tl.inline_asm_elementwise(_ASM_DQX, _DQX_CONSTRAINTS, [_unpair(w)], dtype=tl.float32, is_pure=True,
                                  pack=16)
    return tl.reduce(r, 1, _keep_first)


@triton.jit
def _red_tree(b_p, a_p, first):
    # RED_TREE: one asm per thread over its 8 pairs -> (lane-0 sum, lane-1 sum) pair in slots 0 / 1; the other
    # slots -> -0.0 (bitcast, see _red16), so tl.sum's in-thread tree folds to its last step slot0 + slot1 (the
    # v2 build's lo + hi add, lo as src0) and its cross-lane part is the stock one.
    w = tl.inline_asm_elementwise(_ASM_TREE, _TREE_CONSTRAINTS, [b_p, a_p], dtype=tl.uint64, is_pure=True, pack=8)
    r = _unpair(w)
    neg0 = tl.full(r.shape, -2147483648, tl.int32).to(tl.float32, bitcast=True)
    return tl.sum(tl.where(first, r, neg0), axis=1)


@triton.jit
def _tile(
    K, V, DVK8, k_row, v_row, o_row, kcol, vcol, ocol_v, ocol_k, mask_n,
    q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
    dq_acc, col_k, col_v, m_d, m_s,
    BLOCK_N: tl.constexpr, BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.constexpr,
    MASKED: tl.constexpr, SP: tl.constexpr, SEL: tl.constexpr, RED: tl.constexpr,
):
    if MASKED:
        k = tl.load(K + k_row + kcol, mask=mask_n[:, None], other=0.0)
        v = tl.load(V + v_row + vcol, mask=mask_n[:, None], other=0.0)
    else:
        k = tl.load(K + k_row + kcol)
        v = tl.load(V + v_row + vcol)
    # SEL: apply the row-mask selects to silu/dqk (v2 arithmetic); False only for the PEEL=1 full tiles
    k_real = _bf16_rt(_unpair(_pkmul(ks_p, _pair(k.to(tl.float32)), SP)))
    v_real = _bf16_rt(_unpair(_pkmul(vs_p, _pair(v.to(tl.float32)), SP)))
    if RED == 0:
        qk = _red16(k_real, qb, _ASM_QK, _RED_CONSTRAINTS, first) * alpha
    else:
        qk = _red_tree(_pair(k_real), q_p, first) * alpha
    sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
    if SEL:
        silu = tl.where(mask_n, silu, 0.0)
    silu = _bf16_rt(silu)
    if RED == 0:
        dqk = _red16(v_real, dob, _ASM_DQK, _RED_CONSTRAINTS, first)
        dqk *= sig * (1.0 + _mul_nc(qk, 1.0 - sig)) * (1.0 / MAX_SEQ_LEN)
    else:
        if RED == 2:
            dqk = _red_tree_dqx(_pair(v_real), do_p)
        else:
            dqk = _red_tree(_pair(v_real), do_p, first)
        # 3.9 v2 build: `1 + qk * (1 - sig)` contracted to v_fma_f32 qk, 1 - sig, 1.0
        dqk *= sig * tl.fma(qk, 1.0 - sig, 1.0) * (1.0 / MAX_SEQ_LEN)
    if SEL:
        dqk = tl.where(mask_n, dqk, 0.0)
    dqk = _bf16_rt(dqk)
    dq_acc = tl.fma(tl.broadcast_to(dqk[:, None], [BLOCK_N, BLOCK_D_Q]), k_real, dq_acc)  # v2: fused
    # output path: same instructions / operand order as v2, minus the bf16 unpack
    dv = _unpair(_pkmul_vv(_pair(tl.broadcast_to(silu[:, None], [BLOCK_N, BLOCK_D_V])), do_p))
    dk = _unpair(
        _pkmul(al_p, _pkmul_vv(_pair(tl.broadcast_to(dqk[:, None], [BLOCK_N, BLOCK_D_Q])), q_p), SP)
    )
    dk_b = _bf16_rt(dk)
    dv_b = _bf16_rt(dv)
    col_k += dk_b
    col_v += dv_b
    m_d = _vmax_abs(m_d, dqk)
    m_s = _vmax_abs(m_s, silu)
    dk8 = _med3(_unpair(_pkmul(inv_p, _pair(dk_b), SP)), lo_c, hi_c)
    dv8 = _med3(_unpair(_pkmul(inv_p, _pair(dv_b), SP)), lo_c, hi_c)
    if MASKED:
        tl.store(DVK8 + o_row + ocol_v, dv8.to(DVK8.dtype.element_ty), mask=mask_n[:, None])
        tl.store(DVK8 + o_row + ocol_k, dk8.to(DVK8.dtype.element_ty), mask=mask_n[:, None])
    else:
        tl.store(DVK8 + o_row + ocol_v, dv8.to(DVK8.dtype.element_ty))
        tl.store(DVK8 + o_row + ocol_k, dk8.to(DVK8.dtype.element_ty))
    return dq_acc, col_k, col_v, m_d, m_s


@triton.jit
def _hstu_attn_delta_bwd_e5m2_v2_diet(
    Q,
    K,
    V,
    seq_offsets,
    DOut,
    DQ,
    DVK8,
    PARTIAL_COL,
    PARTIAL_AMAX,
    INV_SCALE,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dvk,
    alpha,
    K_SCALE,
    V_SCALE,
    H,
    MAX_SEQ_LEN,
    COL_K0,
    PERM,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LIMIT: tl.constexpr,
    USE_PERM: tl.constexpr,
    STAGES: tl.constexpr,
    PEEL: tl.constexpr,
    SP: tl.constexpr,
    RED: tl.constexpr = 0,
):
    off_hz = tl.program_id(0)
    if USE_PERM:
        # longest-first (LPT) sequence order; outputs are per-sequence, order-free (v2 verbatim)
        off_z = tl.load(PERM + off_hz // H)
    else:
        off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    off_hz = off_z * H + off_h
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    inv = tl.load(INV_SCALE)

    offs_q = tl.arange(0, BLOCK_D_Q)
    offs_v = tl.arange(0, BLOCK_D_V)
    q = tl.load(
        Q + off_z * stride_qm + off_h * stride_qh + offs_q,
        mask=offs_q < BLOCK_D_Q,
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DOut + off_z * stride_dom + off_h * stride_doh + offs_v,
        mask=offs_v < BLOCK_D_V,
        other=0.0,
    ).to(tl.float32)
    dq_acc = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    m_s = tl.zeros([BLOCK_N], dtype=tl.float32)
    m_d = tl.zeros([BLOCK_N], dtype=tl.float32)
    col_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    col_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    ks_p = _pair(tl.full([BLOCK_N, BLOCK_D_Q], K_SCALE, tl.float32))
    vs_p = _pair(tl.full([BLOCK_N, BLOCK_D_V], V_SCALE, tl.float32))
    al_p = _pair(tl.full([BLOCK_N, BLOCK_D_Q], alpha, tl.float32))
    inv_p = _pair(tl.full([BLOCK_N, BLOCK_D_Q], inv, tl.float32))
    q_p = _pair(tl.broadcast_to(q[None, :], [BLOCK_N, BLOCK_D_Q]))
    do_p = _pair(tl.broadcast_to(dout[None, :], [BLOCK_N, BLOCK_D_V]))
    dob = tl.broadcast_to(dout[None, :], [BLOCK_N, BLOCK_D_V])
    qb = tl.broadcast_to(q[None, :], [BLOCK_N, BLOCK_D_Q])
    # slots kept by the in-thread partial-sum reduction: RED_CHAIN slot 0, RED_TREE(_DQX) slots 0 and 1
    first = ((offs_q % 16) <= (0 if RED == 0 else 1))[None, :]
    lo_c = tl.full([BLOCK_N, BLOCK_D_Q], -LIMIT, tl.float32)
    hi_c = tl.full([BLOCK_N, BLOCK_D_Q], LIMIT, tl.float32)

    rows = tl.arange(0, BLOCK_N)
    k_row0 = (seq_start + rows[:, None]) * stride_kn
    v_row0 = (seq_start + rows[:, None]) * stride_vn
    o_row0 = (seq_start + rows[:, None]) * stride_dvk
    kcol = off_h * stride_kh + offs_q[None, :]
    vcol = off_h * stride_vh + offs_v[None, :]
    ocol_v = off_h * BLOCK_D_V + offs_v[None, :]
    ocol_k = COL_K0 + off_h * BLOCK_D_Q + offs_q[None, :]

    if PEEL >= 1:
        n_full = (seq_len // BLOCK_N) * BLOCK_N
        for start_n in tl.range(0, n_full, BLOCK_N, num_stages=STAGES):
            dq_acc, col_k, col_v, m_d, m_s = _tile(
                K, V, DVK8, k_row0 + start_n * stride_kn, v_row0 + start_n * stride_vn,
                o_row0 + start_n * stride_dvk, kcol, vcol, ocol_v, ocol_k, (start_n + rows) < seq_len,
                q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
                dq_acc, col_k, col_v, m_d, m_s,
                BLOCK_N, BLOCK_D_Q, BLOCK_D_V, False, SP, PEEL == 2, RED,
            )
        if n_full < seq_len:
            # the partial last tile, masked, after all full tiles (v2 tile order)
            dq_acc, col_k, col_v, m_d, m_s = _tile(
                K, V, DVK8, k_row0 + n_full * stride_kn, v_row0 + n_full * stride_vn,
                o_row0 + n_full * stride_dvk, kcol, vcol, ocol_v, ocol_k, (n_full + rows) < seq_len,
                q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
                dq_acc, col_k, col_v, m_d, m_s,
                BLOCK_N, BLOCK_D_Q, BLOCK_D_V, True, SP, True, RED,
            )
    else:
        for start_n in tl.range(0, seq_len, BLOCK_N, num_stages=STAGES):
            dq_acc, col_k, col_v, m_d, m_s = _tile(
                K, V, DVK8, k_row0 + start_n * stride_kn, v_row0 + start_n * stride_vn,
                o_row0 + start_n * stride_dvk, kcol, vcol, ocol_v, ocol_k, (start_n + rows) < seq_len,
                q, dout, dob, qb, first, q_p, do_p, ks_p, vs_p, al_p, inv_p, lo_c, hi_c, alpha, MAX_SEQ_LEN, inv,
                dq_acc, col_k, col_v, m_d, m_s,
                BLOCK_N, BLOCK_D_Q, BLOCK_D_V, True, SP, True, RED,
            )

    # ---- v2 epilogue (verbatim math)
    dq = tl.sum(dq_acc, axis=0)
    tl.store(
        DQ + off_z * stride_dqm + off_h * stride_dqh + offs_q,
        (dq * alpha).to(DQ.dtype.element_ty),
        mask=offs_q < BLOCK_D_Q,
    )
    col_row = PARTIAL_COL + off_z.to(tl.int64) * stride_dvk
    tl.store(col_row + off_h * BLOCK_D_V + offs_v, tl.sum(col_v, axis=0))
    tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, tl.sum(col_k, axis=0))
    av = (tl.max(m_s, axis=0) * tl.max(tl.abs(dout), axis=0)).to(tl.bfloat16).to(tl.float32)
    ak = ((tl.max(m_d, axis=0) * tl.max(tl.abs(q), axis=0)) * tl.abs(alpha)).to(tl.bfloat16).to(tl.float32)
    tl.store(PARTIAL_AMAX + off_hz, tl.maximum(ak, av))


_MAX_INT_32 = 2**31 - 1
# (stages, peel, sp) per Triton build, re-selected on SYNTHETIC lengths only (Arbor x4c19 iter67 ddiet kit, run 70705,
# PREREG sha256 3d52cba2bcdcf858): robust pick = min worst normalised time over the 7 gen_lengths.py
# regimes, 24 configs (STAGES 1-4 x PEEL 0-2 x SP 0-1), bitwise vs the incumbent, flushed paired timing:
# small (<= 2 GiB storages: buffer-op build): SAME, worst regret 0.90% (TOL 1.62 pp);
# big (> 2 GiB: 64-bit global-memory build, the in-model case): SAME, worst regret 0.71% (TOL 1.69 pp).
DEFAULT_SMALL = (2, 0, 0)
DEFAULT_BIG = (1, 1, 1)


def default_config(k: torch.Tensor, v: torch.Tensor, dvk_fp8: torch.Tensor):
    small = all(t.untyped_storage().size() <= _MAX_INT_32 for t in (k, v, dvk_fp8))
    return DEFAULT_SMALL if small else DEFAULT_BIG


def launch_kernel(dout, q, k, v, dq, dvk_fp8, seq_offsets, N, alpha, inv_scale, limit, k_scale, v_scale,
                  partial_col, partial_amax, perm, use_perm, profile=RED_CHAIN, grid_z=None):
    """The v2 launch of `hstu_attention_delta_bwd_e5m2` (same grid, same PERM / USE_PERM) with this kernel.
    Inputs as the v2 branch has them (contiguous innermost dims; outputs allocated by the caller). profile: RED
    (codegen profile). grid_z: number of sequences to launch (default all; the first grid_z entries of PERM).
    Returns the launch's compiled kernel."""
    Z = seq_offsets.numel() - 1 if grid_z is None else grid_z
    _, H, DimQ = q.shape
    DimV = v.shape[2]
    stages, peel, sp = default_config(k, v, dvk_fp8)
    return _hstu_attn_delta_bwd_e5m2_v2_diet[(Z * H,)](
        Q=q, K=k, V=v, seq_offsets=seq_offsets, DOut=dout, DQ=dq, DVK8=dvk_fp8,
        PARTIAL_COL=partial_col, PARTIAL_AMAX=partial_amax, INV_SCALE=inv_scale,
        stride_qm=q.stride(0), stride_qh=q.stride(1), stride_kn=k.stride(0), stride_kh=k.stride(1),
        stride_vn=v.stride(0), stride_vh=v.stride(1), stride_dom=dout.stride(0), stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0), stride_dqh=dq.stride(1), stride_dvk=dvk_fp8.stride(0),
        alpha=alpha, K_SCALE=k_scale, V_SCALE=v_scale, H=H, MAX_SEQ_LEN=N, COL_K0=H * DimV, PERM=perm,
        BLOCK_D_Q=DimQ, BLOCK_D_V=DimV, BLOCK_N=32, LIMIT=limit, USE_PERM=use_perm, STAGES=stages, PEEL=peel,
        SP=sp, RED=profile, num_warps=4,
    )
