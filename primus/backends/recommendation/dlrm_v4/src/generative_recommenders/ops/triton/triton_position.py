# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

#!/usr/bin/env python3


import logging
from typing import Dict, List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    torch.ops.load_library("//hammer/ops/cuda:cuda_ops")
except OSError:
    pass

import os

from generative_recommenders.common import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from generative_recommenders.ops.eval_fwd_hygiene import (
    backward_possible,
    eval_fwd_hygiene_enabled,
)
from generative_recommenders.ops.input_dropout_fold import (
    DROPOUT_HANDOFF_ATTR,
    dropout_handoff_backward,
    DropoutHandoff,
    pack_dropout_mask_bits,
)
from generative_recommenders.ops.zero_fold import enabled as _zero_fold_enabled
from torch.fx._symbolic_trace import is_fx_tracing

_logger: logging.Logger = logging.getLogger(__name__)


# DLRM_HSTU_FP8_POSENC_FAST geometry for the tiled backward (see
# _add_embeddings_bwd_tiled_kernel). Measured at the production shape.
_POSENC_BWD_BLOCK: int = int(os.environ.get("DLRM_HSTU_FP8_POSENC_BWD_BLOCK", "64"))
_POSENC_BWD_WARPS: int = int(os.environ.get("DLRM_HSTU_FP8_POSENC_BWD_WARPS", "4"))

def _env_true(name: str) -> bool:
    """Default-off switch: on only for an explicit 1/true/on/yes (the gin
    binding writes "1"); unset, empty, 0 or anything else is off."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


# DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE (default OFF in code; the gin binding
# apply_env_bootstrap.DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE turns it on and =0 is
# the kill switch; read once at import). Off = the separate position kernel +
# aten::native_dropout. The HSTU input dropout that follows
# the concat-folded position kernel is computed inside that kernel by an exact
# replica of aten's fused_dropout_kernel_vec<bf16, float, uint32, 1, 8, bool>
# Philox stream (same output bits, same bool mask, same CUDA generator offset
# afterwards), so x is never written to / re-read from HBM. Gate:
# posenc_dropout_fuse_supported(); see _posenc_concat_dropout_kernel.
_POSENC_DROPOUT_FUSE: bool = _env_true("DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE")


def _posenc_env_int(name: str, default: int, pow2: bool = False) -> int:
    """Integer launch knob of the fused posenc+dropout kernels. Unset, empty or
    whitespace-only means the default (an empty arm variable must not stop
    training at import); pow2 knobs must be powers of two >= 1."""
    raw = os.environ.get(name, "").strip()
    v = int(raw) if raw else default
    if pow2 and (v < 1 or v & (v - 1)):
        raise ValueError(f"{name}={raw!r}: must be a power of two >= 1")
    return v


# Fixed launch geometry of the fused kernel (numerics do not depend on it; a
# fixed config also avoids a cold-start autotune inside the timed region).
# Best of a 42-config BLOCK_D x BLOCK_N x num_warps sweep at the production shape on
# MI350X, lengths drawn from the real_lens_dense pool (r21 bench_blockd: 1.75 ms vs
# 2.03 ms for the concat kernel's BLOCK_D=64 tiling; disclosure: yambda_5b.gin (41)):
# 128-column panels coalesce the fp32 pos/ts table gathers into 512 B row segments
# per warp. These four DLRM_HSTU_FP8_POSENC_DROP_{BLOCK_D,BLOCK_N,WARPS,STAGES} knobs
# apply to design A (_posenc_concat_dropout_kernel) ONLY; the V2 layout below has its
# own DLRM_HSTU_FP8_POSENC_DROP_V2_* knobs.
_POSENC_DROP_BLOCK_D: int = _posenc_env_int("DLRM_HSTU_FP8_POSENC_DROP_BLOCK_D", 128)
_POSENC_DROP_BLOCK_N: int = _posenc_env_int("DLRM_HSTU_FP8_POSENC_DROP_BLOCK_N", 16)
_POSENC_DROP_WARPS: int = _posenc_env_int("DLRM_HSTU_FP8_POSENC_DROP_WARPS", 4)
_POSENC_DROP_STAGES: int = _posenc_env_int("DLRM_HSTU_FP8_POSENC_DROP_STAGES", 1)

# DLRM_HSTU_FP8_POSENC_DROPOUT_V2 (r21v2; default OFF in code, the gin binding
# turns it on, =0 = design A above). A refinement of
# DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE: inert while that switch is off. Same fused
# op and the same exact aten Philox replica, launched
# as _posenc_concat_dropout_v2_kernel: (1) the streamed seq rows are loaded with
# .cg and y / mask are stored non-temporal (.cs) (the main measured gain); (2)
# each thread's 8-column chunk draws its two Philox calls in its own registers
# (no LDS layout conversion of the random words; 54 -> 44 VGPRs); (3) each
# sequence's row tiles are counted from its END, so the concurrently resident
# tiles of all sequences (grid axis 0) gather the same fp32 pos-table rows (pos
# index = distance from the end); (4) 8-row x 128-column tiles, 2 warps, 1 stage.
# r21v2 paired kernel bench (MI350X, N ~2.3-2.4M x 512, lengths from the real_lens_dense
# pool; disclosure: yambda_5b.gin (41)): about 1.80 -> 1.65-1.69 ms; in-model estimate
# ~0.11 ms per call (0.08-0.15, review timing with values_right freshly written).
# Element -> Philox mapping, all per-element ops, outputs and generator state
# are unchanged (bitwise). Fail-closed: the first use per (device, D, ctx_len,
# kernel constexprs, p) runs _posenc_drop_v2_verified (V2 vs design A, and V2's
# mask vs aten::native_dropout; RNG-neutral); a mismatch, or a V2 compile /
# launch exception, logs a WARNING and launches design A instead. Geometry
# (numerics do not depend on it; tuned at the defaults only):
# DLRM_HSTU_FP8_POSENC_DROP_V2_{BLOCK_D,BLOCK_N,WARPS,STAGES}, defaults
# 128 / 8 / 2 / 1. With V2 on, the design-A DROP_* knobs above have no effect.
_POSENC_DROPOUT_V2: bool = _env_true("DLRM_HSTU_FP8_POSENC_DROPOUT_V2")
_POSENC_DROP_V2_BLOCK_D: int = _posenc_env_int(
    "DLRM_HSTU_FP8_POSENC_DROP_V2_BLOCK_D", 128, pow2=True
)
_POSENC_DROP_V2_BLOCK_N: int = _posenc_env_int(
    "DLRM_HSTU_FP8_POSENC_DROP_V2_BLOCK_N", 8, pow2=True
)
_POSENC_DROP_V2_WARPS: int = _posenc_env_int(
    "DLRM_HSTU_FP8_POSENC_DROP_V2_WARPS", 2, pow2=True
)
_POSENC_DROP_V2_STAGES: int = _posenc_env_int("DLRM_HSTU_FP8_POSENC_DROP_V2_STAGES", 1)

# DLRM_HSTU_FP8_DROPOUT_MASK_BITS (R-C7; default OFF in code: on only for an
# explicit 1/true/on/yes; the gin binding
# apply_env_bootstrap.DLRM_HSTU_FP8_DROPOUT_MASK_BITS turns it on and =0 is the
# kill switch = the bool mask exactly as before; read once at import). With it
# on, both fused posenc+dropout kernels (design A and V2) store
# the input-dropout mask as packed bits, uint8 [N, D / 8] with bit k of byte j
# of a row = column 8 * j + k (input_dropout_fold.pack_dropout_mask_bits),
# instead of bool [N, D]: at N ~2.38M, D = 512 the forward writes and the
# layer-0 LN-backward mask/scale fold (_weighted_layer_norm_bwd_dx_v2,
# MASK_BITS) reads ~1.05 GB less. Same keep decisions (each bit is the kernel's
# own r < KEEP_BELOW test), y, generator offset and gradients: bitwise. The
# forward packs only when that fold will consume the packed mask
# (_dropout_mask_bits_why_not: a handoff, the FP8 arm's fused residual, the LN
# predicate and the fold's first-use self-test); otherwise it keeps the bool
# mask, because unpacking for aten (3 elementwise passes + masked_scale, about
# +3.5 ms and +1.2 GB at the production shape) costs far more than packing
# saves. The remaining unpack (input_dropout_fold.dropout_handoff_backward when
# no consumer claimed the handoff) is exact, and logs a one-time WARNING.
_DROPOUT_MASK_BITS: bool = _env_true("DLRM_HSTU_FP8_DROPOUT_MASK_BITS")

# DLRM_HSTU_FP8_POSENC_DROP_HWEPI (R-C15; default OFF in code: on only for an
# explicit 1/true/on/yes; the gin binding
# apply_env_bootstrap.DLRM_HSTU_FP8_POSENC_DROP_HWEPI turns it on and =0 is the
# kill switch = the V2 kernel exactly as before; read once at import). A
# refinement of DLRM_HSTU_FP8_POSENC_DROPOUT_V2 (inert while V2 is off or has
# fallen back to design A), gfx950 only (gcnArchName gfx950*; any other arch
# keeps the V2 binary as before, with one WARNING). The V2 kernel is VALU-bound
# (Philox rounds + the software fp32 -> bf16 epilogue; VALUBusy ~94 %, iter70
# r21mem). Its HW_EPI compile (1) converts y with f.to(tl.bfloat16), which
# gfx950 lowers to v_cvt_pk_bf16_f32 (round to nearest even, NaN quieted with
# its payload: equal to _f32_to_bf16_bits_hip on all 2^32 fp32 inputs), and
# (2) computes each Philox round's hi ^ c ^ key as one v_bitop3_b32 (3-input
# XOR, round key an SGPR operand; _philox4x32_10_bitop3). Tiling, cache hints,
# grid, mask stores (bool or packed), the element -> Philox (counter, word)
# mapping and every other per-element op are V2's: y, mask, indices and the
# generator offset are bitwise. The bf16 add x keeps Triton's software
# rounding (a hardware cvt there would keep NaN payloads that rounding maps to
# 0x7FFF). r21mem kernel lab, paired: -0.20..-0.28 ms early (N 2.38M) and
# -0.27..-0.29 ms late (N 2.63M) per call. Fail-closed: the first use per V2
# key runs _posenc_drop_hwepi_verified (f.to(bf16) vs torch's cast on the
# reachable epilogue domain, then the V2 self-test with the HW_EPI binary vs
# design A and aten::native_dropout, plus all 65536 bf16 bit patterns;
# RNG-neutral); a mismatch, or a compile / launch exception, logs one WARNING
# and launches V2's software epilogue instead.
_POSENC_DROP_HWEPI: bool = _env_true("DLRM_HSTU_FP8_POSENC_DROP_HWEPI")


def _dropout_mask_bits_why_not(
    attach_handoff: bool, D: int, device: torch.device
) -> str:
    """DLRM_HSTU_FP8_DROPOUT_MASK_BITS (set) layout decision of one fused
    forward: "" = write the packed mask, else why the bool mask is kept. The
    mask must ride on a mask/scale handoff (DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD)
    and the layer-0 LN-backward fold must take the packed form at this D
    (triton_layer_norm.wln_bwd_mscale_bits_why_not: Triton v2 LN backward on,
    D % 8 == 0, the validated launch config). The handoff is claimed only by
    the layer-0 STU preprocess that returns the residual proxy (modules/stu.py
    fuse_residual: DLRM_HSTU_FP8_FUSE_RESIDUAL with the FP8 input LN,
    DLRM_HSTU_FP8_FUSE_LN; both come from the FP8 forward arm
    DLRM_HSTU_FP8_FWD=1). Without them nothing folds and a packed handoff would
    be unpacked for native_dropout_backward, so the bool mask is kept. Last,
    the packed fold must have passed its first-use self-test on this device
    (_mask_bits_fold_verified)."""
    if not attach_handoff:
        return (
            "no mask/scale handoff (DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD off): "
            "the dropout backward needs aten's bool mask"
        )
    from generative_recommenders.ops.triton.triton_fp8 import (
        hstu_fp8_fuse_ln_enabled,
        hstu_fp8_fuse_residual_enabled,
    )

    if not (hstu_fp8_fuse_residual_enabled() and hstu_fp8_fuse_ln_enabled()):
        return (
            "no layer-0 consumer claims the handoff without the fused residual "
            "(DLRM_HSTU_FP8_FUSE_RESIDUAL and DLRM_HSTU_FP8_FUSE_LN, FP8 forward "
            "arm DLRM_HSTU_FP8_FWD=1)"
        )
    from generative_recommenders.ops.triton.triton_layer_norm import (
        wln_bwd_mscale_bits_why_not,
    )

    why = wln_bwd_mscale_bits_why_not(D)
    if why:
        return why
    return _mask_bits_fold_verified(device, D)


# DLRM_HSTU_FP8_DROPOUT_MASK_BITS, fail closed on the consumer side. The packed
# LN-backward fold equals the bool fold bit for bit only while Triton gives
# both compiles of _weighted_layer_norm_bwd_dx_v2 (MASK_SCALE with and without
# MASK_BITS) the same tt.reduce layouts (validated on Triton 3.6 / gfx950). Its
# in-tree tripwire (posenc_concat_dropout_test.py
# test_mask_bits_ln_bwd_reduce_layouts, which also covers the production
# > 2 GiB specialization) is a GPU test, and no automatic gate runs GPU tests.
# So the first packing forward per (device, D, POW2) also checks it here: the
# production launch (triton_layer_norm._wln_bwd_dx_partials with the residual
# mscale config) runs with the bool and with the packed mask on synthetic
# batches of N = 4099 and 4096 rows (both Triton specializations of N). dx, the
# fp32 dweight / dbias partials (the finalize kernel's only inputs) and the
# tt.reduce operand layouts of the two TTGIRs must be identical. Any difference
# or exception keeps the bool mask (the forward's "set but NOT armed" WARNING
# names it). RNG-neutral (CPU generator; no CUDA generator use); no autotuned
# kernel is launched (mean / rstd in torch, no dweight / dbias finalize); ~20
# MB of temporaries; 4 extra compiles of _weighted_layer_norm_bwd_dx_v2 at
# first use (pre-RUN_START precompile list).
_MASK_BITS_FOLD_OK: Dict[Tuple, str] = {}
_MASK_BITS_FOLD_SELF_TEST_N: Tuple[int, ...] = (4099, 4096)


def _ttgir_reduce_layouts(ttgir: str) -> List[str]:
    """Axis and resolved operand layouts of every tt.reduce in a TTGIR (the
    parser of the tripwire test test_mask_bits_ln_bwd_reduce_layouts)."""
    import re

    alias = dict(re.findall(r"^(#\w+) = (.+)$", ttgir, flags=re.M))

    def resolve(s: str) -> str:
        for _ in range(4):
            s = re.sub(r"#\w+", lambda m: alias.get(m.group(0), m.group(0)), s)
        return s

    out, lines = [], ttgir.splitlines()
    for i, line in enumerate(lines):
        if '"tt.reduce"' not in line:
            continue
        axis = re.search(r"axis = (\d+)", line)
        for j in range(i, min(i + 100, len(lines))):
            m = re.search(r"\}\) : \((.*?)\) -> ", lines[j])
            if m:
                ax = axis.group(1) if axis else "?"
                out.append(f"axis={ax} {resolve(m.group(1))}")
                break
    return out


def _mask_bits_fold_self_test(device: torch.device, D: int) -> str:
    """"" if the packed and the bool LN-backward fold agree (raw bits and
    TTGIR reduce layouts) on synthetic batches at D, else the difference."""
    from generative_recommenders.ops.triton import triton_layer_norm as tln

    BLOCK_D = triton.next_power_of_2(D)  # compute_BLOCK_D (bf16, D <= 512)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    g = torch.Generator()  # CPU: the CUDA generators are not touched
    g.manual_seed(0x5C7)
    real = tln._weighted_layer_norm_bwd_dx_v2
    launches: List = []

    class _Spy:  # _wln_bwd_dx_partials looks the kernel up as a module global
        def __getitem__(self, grid):
            def launch(*a, **k):
                launches.append(real[grid](*a, **k))

            return launch

    def rows(N: int, scale: float) -> torch.Tensor:
        t = torch.randn(N, D, generator=g) * scale
        return t.to(torch.bfloat16).to(device)

    tln._weighted_layer_norm_bwd_dx_v2 = _Spy()
    try:
        for N in _MASK_BITS_FOLD_SELF_TEST_N:
            x, dy, res = rows(N, 0.7), rows(N, 0.01), rows(N, 0.01)
            w = (1.0 + 0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
            b = (0.1 * torch.randn(D, generator=g)).to(torch.bfloat16)
            w, b = w.to(device), b.to(device)
            xf = x.float()
            mean = xf.mean(1)
            rstd = torch.rsqrt(xf.var(1, correction=0) + 1e-6)
            m = (torch.rand(N, D, generator=g) >= 0.2).to(device)
            outs, irs = [], []
            for mm in (m, pack_dropout_mask_bits(m)):
                dx = torch.empty_like(x)
                launches.clear()
                _, dw, db, _ = tln._wln_bwd_dx_partials(
                    dx,
                    dy,
                    res,
                    x,
                    w,
                    b,
                    mean,
                    rstd,
                    1e-6,
                    is_swish=False,
                    add_residual=True,
                    sum_dx=False,
                    BLOCK_D=BLOCK_D,
                    sms=sms,
                    dropout_mask=mm,
                    dropout_scale=1.25,
                )
                if len(launches) != 1:
                    return f"N={N}: {len(launches)} dx_v2 launches, expected 1"
                irs.append(_ttgir_reduce_layouts(launches[0].asm["ttgir"]))
                outs.append(
                    (dx.view(torch.int16), dw.view(torch.int32), db.view(torch.int32))
                )
            if not irs[0]:
                return f"N={N}: no tt.reduce found in the bool fold's TTGIR"
            if irs[0] != irs[1]:
                return f"N={N}: the packed and bool folds' tt.reduce layouts differ"
            for nm, a, bb in zip(("dx", "dweight partials", "dbias partials"), *outs):
                if not torch.equal(a, bb):
                    return f"N={N}: the packed and bool folds' {nm} differ"
    finally:
        tln._weighted_layer_norm_bwd_dx_v2 = real
    return ""


def _mask_bits_fold_verified(device: torch.device, D: int) -> str:
    """"" once the packed LN-backward fold passed _mask_bits_fold_self_test
    for (device, D, POW2), else why the bool mask is kept. Cached; any
    exception fails closed."""
    from generative_recommenders.ops.triton.triton_layer_norm import (
        _wln_bwd_pow2_enabled,
    )

    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, D, _wln_bwd_pow2_enabled())
    hit = _MASK_BITS_FOLD_OK.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        return "stream capture: the packed LN-backward fold is not self-tested yet"
    try:
        with torch.cuda.device(index):
            why = _mask_bits_fold_self_test(torch.device("cuda", index), D)
    except Exception as e:
        why = f"raised {type(e).__name__}: {e}"
    if why:
        why = "the packed LN-backward fold failed its self-test: " + why
    _MASK_BITS_FOLD_OK[key] = why
    return why



# DLRM_HSTU_FP8_POSENC_SORT16 (default OFF in code; the gin binding turns it
# on, =0 is the kill switch: int32 sort keys and the torch.arange values of the
# hammer sort attempt, as before). Acts on the concat-folded position kernel
# (DLRM_HSTU_FP8_POSENC_CONCAT_FOLD), fused with the dropout or not. That
# kernel writes pos/ts indices as int16 when their static range fits (host-side
# proof from the table sizes), so the two torch.sort calls run a 16-bit radix
# sort (2 onesweep passes instead of 4). torch.sort's permutation (the int64
# values) is the same as for int32 keys: the cast is order-preserving and the
# sort is deterministic. The dead torch.arange is skipped when
# torch.ops.hammer.sort_kv_pairs does not exist (decided once).
_POSENC_SORT16: bool = _env_true("DLRM_HSTU_FP8_POSENC_SORT16")
_HAMMER_SORT_KV: Optional[bool] = None
_LOGGED: set = set()

# yambda_5b.gin item (49) (a)/(b): the two posenc forward sorts without torch.sort. Both default OFF in code (the
# gin bindings turn them on; =0 is each one's kill switch), both act only inside the POSENC_SORT16 branch of
# _posenc_sort_pairs (int16 keys, hammer sort absent) above torch's 4096-element small-sort cutoff, where
# torch.sort runs its stable radix sort, so the stable permutation is THE incumbent permutation.
# (a) DLRM_HSTU_FP8_POSENC_POS_ARITH: the pos keys are a function of the sequence lengths / targets only, so
# their stable sorted order (keys + int64 indices) is written by one Triton kernel from the jagged metadata
# (ops/triton/triton_pos_perm.py) instead of a 2-pass radix sort of pos_inds (11 launches).
# (b) DLRM_HSTU_FP8_POSENC_TS_ONESWEEP: the timestamp-bucket keys (<= ts table rows - 1) are sorted by
# ops/onesweep_sort.py: rocPRIM's onesweep device code over the 12 key bits in two 6-bit digit passes, a
# counting-iterator index, one fill (torch.sort: copy, iota, 5 fills, 2 histogram kernels, two 8-bit passes).
# Both are self-checked against torch.sort at their first _POSENC_ITEM34_CHECKS uses per process (raw bits of
# keys and indices); a mismatch, a raising call or an unavailable extension restores torch.sort for the
# process with one log line.
_POSENC_POS_ARITH: bool = _env_true("DLRM_HSTU_FP8_POSENC_POS_ARITH")
_POSENC_TS_ONESWEEP: bool = _env_true("DLRM_HSTU_FP8_POSENC_TS_ONESWEEP")
_POSENC_ITEM34_CHECKS: int = 2
_POSENC_ITEM34_STATE: Dict[str, int] = {"pos": _POSENC_ITEM34_CHECKS, "ts": _POSENC_ITEM34_CHECKS}

# DLRM_HSTU_FP8_POSENC_BWD_V2 (default OFF in code; the gin binding turns it
# on, =0 is the kill switch -> the two tiled backward kernels). The
# concat-folded backward's two
# sorted segment reductions (d_pos + d_left/d_right split, d_ts) run as
# row-loop kernels (_add_embeddings_bwd_rowloop_{split_,}kernel): one-warp
# programs walk SUB-row sub-tiles of the sorted order and carry a running fp32
# sum per key across NSUB sub-tiles, flushing one D-wide atomic per (program,
# key); the one-warp tiles keep the gathers in flight without the 64x512 tile's
# register footprint. Programs of SUB*NSUB rows: 64 for pos/split (NSUB=8:
# about the tiled kernel's atomic count; its gain is the sub-tile structure and
# the fine program granularity, ~37K programs at occupancy 3) and 1024 / 512
# for ts (NSUB 128 / 64: N=2.38M issues ~2.2M fp32 atomics vs ~20M tiled). For
# small N, NSUB drops further (programs under 64 rows) and V2 can issue more
# atomics than the tiled kernels.
# NUMERICS (declared): d_left/d_right are bitwise those of the tiled kernel
# (same per-element op); d_pos/d_ts are fp32-order -- only the fp32 summation
# order changes (the tiled kernel's atomic order is already run-to-run
# nondeterministic). A bundle with this switch on is "bitwise except
# d_pos/d_ts (fp32-order)". Geometry (MI350X, D=512; lengths from real_lens_{dense,prod},
# iter65r posencbwd; disclosure: yambda_5b.gin (41)): SUB=8, 1 warp, NSUB 8 for pos/split;
# NSUB for ts is 128 when d_out > 2 GiB and 64 when d_out <= 2 GiB, the same test as the
# Triton AMD backend's 'S' argument specialization (tt.pointer_range=32, buffer ops). The
# 'S' build of the ts kernel fits in 128 VGPRs (occupancy 4) and is fastest with NSUB=64;
# the non-'S' build needs 130 (occupancy 3) and is fastest with 128. Because NSUB only
# changes where every kernel here (the tiled ones included) already recompiles for 'S', V2
# adds no compile variant of its own at production N. NSUB shrinks further for small N so
# the grid keeps >= _POSENC_BWD_V2_MIN_PROGRAMS programs. At d_out <= 2 GiB
# the 'S' build also speeds up the tiled split kernel by ~9%; the row-loop
# split (NSUB=8) stays faster there (0.02-0.06 ms at the shapes measured). Any
# launch failure falls back to the tiled kernels (fail closed).
_POSENC_BWD_V2: bool = _env_true("DLRM_HSTU_FP8_POSENC_BWD_V2")
# Geometry of the row-loop kernels. Unset/empty = default (the launcher
# forwards every DLRM_HSTU_FP8_* name, so an exported empty value must not stop
# training at import); SUB/WARPS are pow2.
_POSENC_BWD_V2_SUB: int = _posenc_env_int("DLRM_HSTU_FP8_POSENC_BWD_V2_SUB", 8, pow2=True)
_POSENC_BWD_V2_NSUB_POS: int = _posenc_env_int("DLRM_HSTU_FP8_POSENC_BWD_V2_NSUB_POS", 8)
# ts kernel (d_ts) geometry of the d_out > 2 GiB build (Triton 3.9, ROCm 10.2 port; triton-perf replay of the
# production step on one node, d_out 2.48 GB): Triton 3.9 lowers the atomic's layout conversion through LDS
# (3.6: ds_bpermute) and the SUB=8 build needs 170 VGPRs (3.6: 130), occupancy 2: the 2369 programs of
# SUB 8 x NSUB 128 no longer fit one round (2048 wave slots) and the kernel takes 664 us (3.6: 446 us).
# SUB 4 x NSUB 32 (104 VGPRs, occupancy 4, 18951 programs): 449 us on 3.9. d_ts stays fp32-order (as before:
# atomics). The d_out <= 2 GiB ('S') build keeps SUB / NSUB_TS_2G (not measured on 3.9).
# Triton 3.6 (v26.3) keeps SUB x NSUB 128: there the SUB 8 build fits 130 VGPRs (occupancy 3) and SUB 4 x NSUB 32
# is 11-13% slower (494-502 vs 442-448 us at the same launch, verify-triton-perf / triton-perf tune3+tune4), so
# the v26.3 geometry, d_ts order and TSPEC keys stay as they were.
_POSENC_BWD_V2_TS_SUB4: bool = triton.__version__ != "3.6.0"
_POSENC_BWD_V2_SUB_TS: int = _posenc_env_int(
    "DLRM_HSTU_FP8_POSENC_BWD_V2_SUB_TS", 4 if _POSENC_BWD_V2_TS_SUB4 else _POSENC_BWD_V2_SUB, pow2=True
)
_POSENC_BWD_V2_NSUB_TS: int = _posenc_env_int(
    "DLRM_HSTU_FP8_POSENC_BWD_V2_NSUB_TS", 32 if _POSENC_BWD_V2_TS_SUB4 else 128
)
_POSENC_BWD_V2_NSUB_TS_2G: int = _posenc_env_int(
    "DLRM_HSTU_FP8_POSENC_BWD_V2_NSUB_TS_2G", 64
)
_POSENC_BWD_V2_WARPS: int = _posenc_env_int(
    "DLRM_HSTU_FP8_POSENC_BWD_V2_WARPS", 1, pow2=True
)
_POSENC_BWD_V2_MIN_PROGRAMS: int = 2048
_POSENC_BWD_V2_FAILED: bool = False


def _log_once(key: str, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _logger.info(msg)


def _warn_once(key: str, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _logger.warning(msg)


def _hammer_sort_kv_available() -> bool:
    global _HAMMER_SORT_KV
    if _HAMMER_SORT_KV is None:
        try:
            torch.ops.hammer.sort_kv_pairs  # noqa: B018
            _HAMMER_SORT_KV = True
        except (AttributeError, RuntimeError):
            _HAMMER_SORT_KV = False
    return _HAMMER_SORT_KV


def _posenc_index_dtype(
    max_pos_ind: int, max_contextual_seq_len: int, ts_emb_size: int
) -> torch.dtype:
    """int16 when DLRM_HSTU_FP8_POSENC_SORT16 is on, the hammer sort is absent
    and every value the kernel can write fits: pos_inds lie in
    [0, max(max_pos_ind, max_contextual_seq_len) - 1] and ts_inds in
    [0, ts_emb_size - 1] (both are clamped in the kernel)."""
    if (
        _POSENC_SORT16
        and not _hammer_sort_kv_available()
        and 1 <= max_pos_ind
        and max(max_pos_ind, max_contextual_seq_len) - 1 <= 32767
        and 1 <= ts_emb_size <= 32768
    ):
        _log_once(
            "sort16",
            "armed OK: posenc int16 sort keys ACTIVE "
            "(DLRM_HSTU_FP8_POSENC_SORT16)",
        )
        return torch.int16
    return torch.int32


def _item34_raw(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() == 2:
        return t.view(torch.int16)
    return t.view(torch.int64) if t.element_size() == 8 else t.view(torch.int32)


def _item34_checked(
    part: str, got: Tuple[torch.Tensor, torch.Tensor], ref_fn, what: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Self-check of item (49) part `part` against torch.sort for its first uses (raw bits, dtype, shape); a
    mismatch logs one ERROR, returns the incumbent result and turns the part off for the process."""
    global _POSENC_POS_ARITH, _POSENC_TS_ONESWEEP
    left = _POSENC_ITEM34_STATE[part]
    if left <= 0:
        return got
    ref = ref_fn()
    bad = [
        i
        for i, (g, r) in enumerate(zip(got, ref))
        if g.dtype != r.dtype or g.shape != r.shape or not torch.equal(_item34_raw(g), _item34_raw(r))
    ]
    if bad:
        if part == "pos":
            _POSENC_POS_ARITH = False
        else:
            _POSENC_TS_ONESWEEP = False
        _logger.error(
            "DLRMV4 posenc %s SELF-CHECK FAILED (N=%d): %s differ from torch.sort -- incumbent result used, "
            "torch.sort for the rest of the run",
            what,
            ref[0].numel(),
            "keys" if bad == [0] else "indices" if bad == [1] else "keys and indices",
        )
        return ref
    _POSENC_ITEM34_STATE[part] = left - 1
    return got


def _posenc_item34_off(part: str, why: str) -> None:
    global _POSENC_POS_ARITH, _POSENC_TS_ONESWEEP
    if part == "pos":
        _POSENC_POS_ARITH = False
    else:
        _POSENC_TS_ONESWEEP = False
    _warn_once(
        "item34_off:" + part,
        f"DLRMV4 posenc {part} sort FAILED CLOSED -> torch.sort for the rest of the run ({why})",
    )


def _posenc_pos_arith(
    pos_inds: torch.Tensor, N: int, pos_meta: Optional[Tuple]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Item (49)(a) DLRM_HSTU_FP8_POSENC_POS_ARITH: torch.sort(pos_inds) written from the jagged metadata, or
    None (off / out of scope -> torch.sort)."""
    if not _POSENC_POS_ARITH or pos_meta is None:
        return None
    seq_offsets, seq_lengths, num_targets, interleave_targets, C, M = pos_meta
    from generative_recommenders.ops.triton import triton_pos_perm as tpp

    B = seq_lengths.shape[0]
    if not (
        tpp.pos_perm_supported(N, B, int(C), int(M))
        and seq_offsets.dim() == 1
        and seq_offsets.numel() == B + 1
        and seq_offsets.is_contiguous()
        and seq_lengths.is_contiguous()
        and (num_targets is None or num_targets.is_contiguous())
    ):
        return None
    try:
        got = tpp.pos_sorted_pairs(
            seq_offsets, seq_lengths, num_targets, interleave_targets, int(C), int(M), N, pos_inds.dtype
        )
    except Exception as e:  # fail closed: torch.sort
        _posenc_item34_off("pos", f"{type(e).__name__}: {str(e)[:200]}")
        return None
    got = _item34_checked("pos", got, lambda: torch.sort(pos_inds), "pos permutation arithmetic")
    if _POSENC_POS_ARITH:
        _log_once(
            "item34_pos",
            "DLRMV4 armed OK: posenc pos permutation arithmetic -- stable pos order from the jagged lengths, no "
            f"radix sort (N={N}, B={B}, C={int(C)}, M={int(M)}; first {_POSENC_ITEM34_CHECKS} uses == torch.sort; "
            "kill switch DLRM_HSTU_FP8_POSENC_POS_ARITH=0)",
        )
    return got


def _posenc_ts_onesweep(
    ts_inds: torch.Tensor, N: int, ts_max_key: Optional[int]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Item (49)(b) DLRM_HSTU_FP8_POSENC_TS_ONESWEEP: torch.sort(ts_inds) on the key-bound onesweep extension, or
    None (off / out of scope / extension unavailable -> torch.sort)."""
    if not _POSENC_TS_ONESWEEP or ts_max_key is None:
        return None
    from generative_recommenders.ops import onesweep_sort as osort

    try:
        got = osort.sort16_pairs(ts_inds, int(ts_max_key))
    except Exception as e:  # fail closed: torch.sort
        _posenc_item34_off("ts", f"{type(e).__name__}: {str(e)[:200]}")
        return None
    if got is None:
        if osort._FAILED is not None:
            _posenc_item34_off("ts", "onesweep sort extension unavailable")
        return None
    got = _item34_checked("ts", got, lambda: torch.sort(ts_inds), "ts onesweep sort")
    if _POSENC_TS_ONESWEEP:
        _log_once(
            "item34_ts",
            "DLRMV4 armed OK: posenc ts onesweep sort -- key-bound bits, 6-bit digits, no iota/copy, one fill "
            f"(N={N}, max key {int(ts_max_key)}; first {_POSENC_ITEM34_CHECKS} uses == torch.sort; kill switch "
            "DLRM_HSTU_FP8_POSENC_TS_ONESWEEP=0)",
        )
    return got


def _posenc_sort_pairs(
    ts_inds: torch.Tensor,
    pos_inds: torch.Tensor,
    N: int,
    device: torch.device,
    pos_meta: Optional[Tuple] = None,
    ts_max_key: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sorted_ts_keys, sorted_ts_values, sorted_pos_keys, sorted_pos_values).
    pos_meta = (seq_offsets, seq_lengths, num_targets, interleave_targets, max_contextual_seq_len, max_pos_ind) of
    the position kernel that wrote pos_inds, ts_max_key = its largest ts key (item (49); None = torch.sort)."""
    if _POSENC_SORT16 and not _hammer_sort_kv_available():
        # The try below would always land in its torch.sort fallback; skip the
        # dead torch.arange (int16 keys only ever take this branch).
        ts_pair = _posenc_ts_onesweep(ts_inds, N, ts_max_key)
        if ts_pair is None:
            ts_pair = torch.sort(ts_inds)
        pos_pair = _posenc_pos_arith(pos_inds, N, pos_meta)
        if pos_pair is None:
            pos_pair = torch.sort(pos_inds)
        sorted_ts_key_inds, sorted_ts_value_inds = ts_pair
        sorted_pos_key_inds, sorted_pos_value_inds = pos_pair
        return (
            sorted_ts_key_inds,
            sorted_ts_value_inds,
            sorted_pos_key_inds,
            sorted_pos_value_inds,
        )
    try:
        values = torch.arange(0, N, dtype=torch.int32, device=device)
        sorted_ts_key_inds, sorted_ts_value_inds = torch.ops.hammer.sort_kv_pairs(
            ts_inds, values
        )
        sorted_pos_key_inds, sorted_pos_value_inds = torch.ops.hammer.sort_kv_pairs(
            pos_inds, values
        )
    except Exception:
        sorted_ts_key_inds, sorted_ts_value_inds = torch.sort(ts_inds)
        sorted_pos_key_inds, sorted_pos_value_inds = torch.sort(pos_inds)
    return (
        sorted_ts_key_inds,
        sorted_ts_value_inds,
        sorted_pos_key_inds,
        sorted_pos_value_inds,
    )


def _autotune_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [16, 32, 64]:
        for num_stages in [1, 2]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_N": BLOCK_N,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


@triton_autotune(
    configs=_autotune_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _add_timestamp_position_embeddings_kernel(
    SeqEmb,
    Offsets,
    Lengths,
    PosEmb,
    TsEmb,
    Out,
    TS,
    PosInds,
    TsInds,
    NumTargets,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    num_time_buckets,
    time_bucket_increments,
    time_bucket_scale,
    time_delta,
    max_contextual_seq_len,
    max_pos_ind,
    stride_sn,
    stride_pn,
    stride_tn,
    stride_on,
    alpha,
    TRAINING: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SeqEmb has shape (sum_B(N_i), D),
    PosEmb has shape (N_p, D),
    TsEmb has shape (N_t, D),
    Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_b)
    seq_end = tl.load(Offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    seq_emb_offsets = offs_n[:, None] * stride_sn + offs_d[None, :]
    SeqEmb += seq_start.to(tl.int64) * stride_sn
    mask_n = offs_n < seq_len
    # position encoding
    seq_len = tl.load(Lengths + off_b)
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + off_b)
        if INTERLEAVE_TARGETS:
            high_ind = seq_len - num_targets * 2
        else:
            high_ind = seq_len - num_targets
    else:
        high_ind = seq_len
    pos_inds = tl.where(offs_n < high_ind, offs_n, high_ind)
    pos_inds = high_ind - pos_inds + max_contextual_seq_len
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(offs_n < max_contextual_seq_len, offs_n, pos_inds)
    if TRAINING:
        tl.store(PosInds + seq_start + offs_n, pos_inds, mask=mask_n)
    pos_emb_offsets = pos_inds[:, None] * stride_pn + offs_d[None, :]
    # timestamp encoding
    ts = tl.load(TS + seq_start + offs_n, mask=mask_n)
    query_time = tl.load(TS + seq_end - 1)
    ts = query_time - ts + time_delta
    ts = tl.where(ts > 1e-6, ts, 1e-6) / time_bucket_increments
    if TIME_BUCKET_FN == "log":
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    if TRAINING:
        tl.store(TsInds + seq_start + offs_n, ts, mask=mask_n)
    ts_emb_offsets = ts[:, None] * stride_tn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_offsets = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < D)  # Triton 3.9: `and` on tensors is deprecated
        seq_emb = tl.load(SeqEmb + seq_emb_offsets, mask=mask)
        if HAS_ALPHA:
            # == the `seq_embeddings * alpha` the caller used to run as its own
            # pass: fp32 opmath product rounded once to the storage dtype.
            seq_emb = (seq_emb.to(tl.float32) * alpha).to(seq_emb.dtype)
        pos_emb = tl.load(PosEmb + pos_emb_offsets, mask=mask)
        ts_emb = tl.load(TsEmb + ts_emb_offsets, mask=mask)
        tl.store(out_offsets, seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype), mask=mask)
        seq_emb_offsets += BLOCK_D
        pos_emb_offsets += BLOCK_D
        ts_emb_offsets += BLOCK_D
        out_offsets += BLOCK_D
        offs_d += BLOCK_D


def bwd_pre_hook(nargs):
    nargs["Out"].zero_()


def _add_embeddings_bwd_configs() -> List[triton.Config]:
    configs = []
    for BLOCK in [32, 64, 128]:
        for num_stages in [2, 3, 4]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK": BLOCK,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                        pre_hook=bwd_pre_hook,
                    )
                )
    return configs


@triton_autotune(
    configs=_add_embeddings_bwd_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "AUTOTUNE_B", "D"],
)
@triton.jit
def _add_embeddings_bwd_kernel(
    In,
    KeyInds,
    ValueInds,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    AUTOTUNE_B,
    D,
    jagged_size,
    stride_in,
    stride_on,
    BLOCK_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_block = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    key_ind = -1
    key_ind = key_ind.to(KeyInds.dtype.element_ty)  # pyre-ignore[16]
    accumulator = tl.zeros((BLOCK_D,), dtype=In.dtype.element_ty)
    for off_i in range(0, BLOCK):
        off = off_block * BLOCK + off_i
        if off < jagged_size:
            value_ind = tl.load(ValueInds + off)
            in_offset = In + value_ind.to(tl.int64) * stride_in
            jagged_in = tl.load(in_offset + offs_d, mask=mask_d)
            key_ind_new = tl.load(KeyInds + off)
            if key_ind == key_ind_new:
                accumulator += jagged_in
            else:
                if key_ind >= 0:
                    out_offset = Out + key_ind.to(tl.int64) * stride_on
                    tl.atomic_add(
                        out_offset + offs_d,
                        accumulator.to(Out.dtype.element_ty),
                        mask=mask_d,
                        sem="relaxed",
                    )
                key_ind = key_ind_new
                accumulator = jagged_in
    if key_ind >= 0:
        out_offset = Out + key_ind.to(tl.int64) * stride_on
        tl.atomic_add(
            out_offset + offs_d,
            accumulator.to(Out.dtype.element_ty),
            mask=mask_d,
            sem="relaxed",
        )


@triton.jit
def _add_embeddings_bwd_tiled_kernel(
    In,
    KeyInds,
    ValueInds,
    Out,
    DSeq,
    N,
    D,
    stride_in,
    stride_on,
    stride_dn,
    alpha,
    WRITE_DSEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK: tl.constexpr,
    KEY_I16: tl.constexpr = False,
):
    """Tiled segmented reduction of sorted (key, value) rows (DLRM_HSTU_FP8_POSENC_FAST).

    The incumbent kernel walks its BLOCK rows one at a time; every row load
    depends on the index load before it, so it streams ~1 KB per dependent
    HBM round trip and runs at ~1.3 TB/s.  Here a program gathers its BLOCK
    rows as one [BLOCK, D] tile (BLOCK independent loads in flight), then
    folds each distinct key in the tile (keys are sorted, so usually 1-2) with
    tl.sum in fp32 and issues one fp32 atomic per key, like the incumbent.

    WRITE_DSEQ: the pos pass visits every row exactly once (sorted_value_inds
    is a permutation), so it also writes d_seq = (d_out * alpha) -> the
    MulBackward pass of the folded `seq_embeddings * alpha` disappears.

    KEY_I16 (DLRM_HSTU_FP8_POSENC_SORT16): int16 keys; the out-of-range
    sentinel is applied after widening (same int32 key values as before).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_r = offs < N
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    if KEY_I16:
        keys = tl.load(KeyInds + offs, mask=mask_r, other=0).to(tl.int32)
        keys = tl.where(mask_r, keys, 2147483647)
    else:
        keys = tl.load(KeyInds + offs, mask=mask_r, other=2147483647).to(tl.int32)
    vals = tl.load(ValueInds + offs, mask=mask_r, other=0).to(tl.int64)
    m2 = mask_r[:, None] & mask_d[None, :]
    x = tl.load(In + vals[:, None] * stride_in + offs_d[None, :], mask=m2, other=0.0)
    if WRITE_DSEQ:
        tl.store(
            DSeq + vals[:, None] * stride_dn + offs_d[None, :],
            (x.to(tl.float32) * alpha).to(DSeq.dtype.element_ty),
            mask=m2,
        )
    xf = x.to(tl.float32)
    k = tl.min(keys, axis=0)
    k_last = tl.max(tl.where(mask_r, keys, -1), axis=0)
    while k <= k_last:
        sel = keys == k
        acc = tl.sum(tl.where(sel[:, None], xf, 0.0), axis=0)
        tl.atomic_add(
            Out + k.to(tl.int64) * stride_on + offs_d,
            acc,
            mask=mask_d,
            sem="relaxed",
        )
        k = tl.min(tl.where(keys > k, keys, 2147483647), axis=0)


@triton_autotune(
    configs=_autotune_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _add_timestamp_position_embeddings_concat_kernel(
    Left,
    Right,
    Offsets,
    Lengths,
    PosEmb,
    TsEmb,
    Out,
    TS,
    PosInds,
    TsInds,
    SrcInds,
    NumTargets,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    num_time_buckets,
    time_bucket_increments,
    time_bucket_scale,
    time_delta,
    max_contextual_seq_len,
    max_pos_ind,
    ctx_len,
    stride_ln,
    stride_rn,
    stride_pn,
    stride_tn,
    stride_on,
    alpha,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """DLRM_HSTU_FP8_POSENC_CONCAT_FOLD: _add_timestamp_position_embeddings_kernel
    (TRAINING, HAS_ALPHA) with the preprocessor's contextual concat folded in.

    Row n of sequence b of the (virtual) concatenated input is
      Left[b * ctx_len + n]                          if n < ctx_len
      Right[seq_start - ctx_len * (b + 1) + n]       otherwise
    (Offsets are the concatenated offsets = right offsets + ctx_len * b), i.e.
    exactly what concat_2D_jagged(dense left, jagged right) would have copied
    there.  SrcInds[row] records the source (-(left_row + 1) or right_row) so
    the backward pos pass can write d_left / d_right without a split.
    The arithmetic per element is identical to the incumbent kernel.
    """
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_b)
    seq_end = tl.load(Offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < seq_len
    is_left = offs_n < ctx_len
    is_right = offs_n >= ctx_len
    left_rows = off_b.to(tl.int64) * ctx_len + offs_n
    right_rows = seq_start.to(tl.int64) - ctx_len * (off_b.to(tl.int64) + 1) + offs_n
    src = tl.where(is_left, -(left_rows + 1), right_rows)
    tl.store(SrcInds + seq_start + offs_n, src.to(tl.int32), mask=mask_n)
    # Two masked loads instead of a select between two pointer tensors: the
    # ROCm TritonAMDGPUCanonicalizePointers pass rejects the latter. Only the
    # first tile of a sequence (start_n < ctx_len) can contain left rows.
    has_left = start_n < ctx_len
    left_offs = left_rows[:, None] * stride_ln + offs_d[None, :]
    right_offs = right_rows[:, None] * stride_rn + offs_d[None, :]
    # position encoding
    seq_len = tl.load(Lengths + off_b)
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + off_b)
        if INTERLEAVE_TARGETS:
            high_ind = seq_len - num_targets * 2
        else:
            high_ind = seq_len - num_targets
    else:
        high_ind = seq_len
    pos_inds = tl.where(offs_n < high_ind, offs_n, high_ind)
    pos_inds = high_ind - pos_inds + max_contextual_seq_len
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(offs_n < max_contextual_seq_len, offs_n, pos_inds)
    tl.store(PosInds + seq_start + offs_n, pos_inds, mask=mask_n)
    pos_emb_offsets = pos_inds[:, None] * stride_pn + offs_d[None, :]
    # timestamp encoding
    ts = tl.load(TS + seq_start + offs_n, mask=mask_n)
    query_time = tl.load(TS + seq_end - 1)
    ts = query_time - ts + time_delta
    ts = tl.where(ts > 1e-6, ts, 1e-6) / time_bucket_increments
    if TIME_BUCKET_FN == "log":
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    tl.store(TsInds + seq_start + offs_n, ts, mask=mask_n)
    ts_emb_offsets = ts[:, None] * stride_tn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_offsets = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < D)  # Triton 3.9: `and` on tensors is deprecated
        if has_left:
            seq_l = tl.load(Left + left_offs, mask=mask & is_left[:, None])
            seq_r = tl.load(Right + right_offs, mask=mask & is_right[:, None])
            seq_emb = tl.where(is_left[:, None], seq_l, seq_r)
        else:
            seq_emb = tl.load(Right + right_offs, mask=mask)
        seq_emb = (seq_emb.to(tl.float32) * alpha).to(seq_emb.dtype)
        pos_emb = tl.load(PosEmb + pos_emb_offsets, mask=mask)
        ts_emb = tl.load(TsEmb + ts_emb_offsets, mask=mask)
        tl.store(out_offsets, seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype), mask=mask)
        left_offs += BLOCK_D
        right_offs += BLOCK_D
        pos_emb_offsets += BLOCK_D
        ts_emb_offsets += BLOCK_D
        out_offsets += BLOCK_D
        offs_d += BLOCK_D


@triton.jit
def _philox4x32_10_ctr_mad64(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 of the 128-bit counter (c0, c1, c2, c3) under the key
    (k0, k1): rocrand philox4x32_10_engine::ten_rounds (== Random123 and
    tl.philox) bit for bit. Each round takes the hi and lo halves of a 32x32
    product from one 64-bit product (v_mad_u64_u32 on CDNA). All uint32."""
    for _ in tl.static_range(10):
        prod_a = c0.to(tl.uint64) * 0xD2511F53
        prod_b = c2.to(tl.uint64) * 0xCD9E8D57
        c0 = (prod_b >> 32).to(tl.uint32) ^ c1 ^ k0
        c2 = (prod_a >> 32).to(tl.uint32) ^ c3 ^ k1
        c1 = prod_b.to(tl.uint32)
        c3 = prod_a.to(tl.uint32)
        k0 = tl.add(k0, 0x9E3779B9, sanitize_overflow=False)
        k1 = tl.add(k1, 0xBB67AE85, sanitize_overflow=False)
    return c0, c1, c2, c3


@triton.jit
def _f32_to_bf16_bits_hip(f):
    """uint32 holding the bf16 bits of c10::BFloat16(float) on ROCm >= 7.2
    (HIP __float2bfloat16): round-to-nearest-even of the fp32 bits (subnormals
    kept), NaN -> its top 16 bits with the quiet bit set. Equal to torch's
    fp32 -> bf16 cast for all 2^32 inputs (r21 validation)."""
    bits = f.to(tl.uint32, bitcast=True)
    rne = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where(f != f, (bits >> 16) | 0x40, rne)


@triton.jit
def _keep_nibble(w0, w1, w2, w3, KEEP_BELOW: tl.constexpr):
    """DLRM_HSTU_FP8_DROPOUT_MASK_BITS: keep bits of 4 consecutive columns
    from their Philox words (bit j <- word j, the kernels' r < KEEP_BELOW
    test), int32 in [0, 15]."""
    return (
        (w0 < KEEP_BELOW).to(tl.int32)
        | ((w1 < KEEP_BELOW).to(tl.int32) << 1)
        | ((w2 < KEEP_BELOW).to(tl.int32) << 2)
        | ((w3 < KEEP_BELOW).to(tl.int32) << 3)
    )


@triton.jit
def _bf16_cvt_probe_kernel(X, Out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    f = tl.load(X + offs, mask=m, other=0.0)
    tl.store(Out + offs, _f32_to_bf16_bits_hip(f).to(tl.int16), mask=m)


@triton.jit
def _xor3_sgpr(a, b, k):
    """DLRM_HSTU_FP8_POSENC_DROP_HWEPI (gfx950 only): a ^ b ^ k as one
    v_bitop3_b32 (truth table 0x96 = 3-input XOR); k is uniform (a Philox
    round key) and is passed as an SGPR operand."""
    return tl.inline_asm_elementwise(
        "v_bitop3_b32 $0, $1, $2, $3 bitop3:0x96",
        "=v,v,v,s",
        [a, b, k],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _philox4x32_10_bitop3(c0, c1, c2, c3, k0, k1):
    """_philox4x32_10_ctr_mad64 (same rounds, products, keys and output words)
    with each round's two XORs hi ^ c ^ key as one _xor3_sgpr (gfx950 only;
    DLRM_HSTU_FP8_POSENC_DROP_HWEPI)."""
    for _ in tl.static_range(10):
        prod_a = c0.to(tl.uint64) * 0xD2511F53
        prod_b = c2.to(tl.uint64) * 0xCD9E8D57
        c0 = _xor3_sgpr((prod_b >> 32).to(tl.uint32), c1, k0)
        c2 = _xor3_sgpr((prod_a >> 32).to(tl.uint32), c3, k1)
        c1 = prod_b.to(tl.uint32)
        c3 = prod_a.to(tl.uint32)
        k0 = tl.add(k0, 0x9E3779B9, sanitize_overflow=False)
        k1 = tl.add(k1, 0xBB67AE85, sanitize_overflow=False)
    return c0, c1, c2, c3


@triton.jit
def _bf16_hw_cvt_probe_kernel(X, Out, n, BLOCK: tl.constexpr):
    """HW_EPI's y epilogue, f.to(tl.bfloat16) (gfx950: v_cvt_pk_bf16_f32)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    f = tl.load(X + offs, mask=m, other=0.0)
    tl.store(Out + offs, f.to(tl.bfloat16).to(tl.int16, bitcast=True), mask=m)


@triton.jit(do_not_specialize=["seed_lo", "seed_hi", "ctr_lo", "ctr_hi"])
def _posenc_concat_dropout_kernel(
    Left,
    Right,
    Offsets,
    Lengths,
    PosEmb,
    TsEmb,
    Y,
    Mask,
    TS,
    PosInds,
    TsInds,
    SrcInds,
    NumTargets,
    D,
    num_time_buckets,
    time_bucket_increments,
    time_bucket_scale,
    time_delta,
    max_contextual_seq_len,
    max_pos_ind,
    ctx_len,
    stride_ln,
    stride_rn,
    stride_pn,
    stride_tn,
    stride_on,
    alpha,
    scale,
    seed_lo,
    seed_hi,
    ctr_lo,
    ctr_hi,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    T: tl.constexpr,
    LOG2_T: tl.constexpr,
    KEEP_BELOW: tl.constexpr,
    MASK_BITS: tl.constexpr = False,
):
    """DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE: _add_timestamp_position_embeddings_concat_kernel
    followed by torch.ops.aten.native_dropout(x, p, True) without writing x.

    x is computed with the concat kernel's exact ops. The dropout replicates
    aten's fused_dropout_kernel_vec<bf16, float, uint32, 1, VEC=8, bool>
    (block 256, grid G, T = 256 * G threads, curand_init(seed, thread,
    offset) -> rocrand Philox4x32-10): element e = row * D + col of the
    contiguous [N, D] output is handled by thread t = (e >> 3) % T in loop
    iteration it = (e >> 3) // T, whose curand_uniform4 call
    i = 2 * it + ((e >> 2) & 1) returns word j = e & 3 of
    Philox(counter = (offset / 4 + i) in (c0, c1), t in c2, 0 in c3;
    key = seed). keep <=> 2^-32 + float(r) * 2^-32 < (float)(1 - p)
    <=> r < KEEP_BELOW (the float expression is monotone in r; the integer
    threshold is checked over all 2^32 r). y = BFloat16(float(x) *
    (keep ? scale : 0)) with ROCm's c10 fp32 -> bf16 conversion
    (_f32_to_bf16_bits_hip; one multiply by scale or 0.0 == aten's
    (x * keep) * scale for every input incl. NaN/Inf/-0/subnormals), written
    as int16 bits; mask = keep (one byte), or with MASK_BITS
    (DLRM_HSTU_FP8_DROPOUT_MASK_BITS) bit k of byte j of a contiguous
    [N, D / 8] Mask row = keep of column 8 * j + k. T (power of two when
    LOG2_T >= 0), KEEP_BELOW and scale come from the host (_aten_dropout_rng,
    _dropout_keep_params). BLOCK_D: any power of two >= 8 (D % 8 == 0).
    """
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_b)
    seq_end = tl.load(Offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < seq_len
    is_left = offs_n < ctx_len
    is_right = offs_n >= ctx_len
    left_rows = off_b.to(tl.int64) * ctx_len + offs_n
    right_rows = seq_start.to(tl.int64) - ctx_len * (off_b.to(tl.int64) + 1) + offs_n
    src = tl.where(is_left, -(left_rows + 1), right_rows)
    tl.store(SrcInds + seq_start + offs_n, src.to(tl.int32), mask=mask_n)
    has_left = start_n < ctx_len
    left_offs = left_rows[:, None] * stride_ln + offs_d[None, :]
    right_offs = right_rows[:, None] * stride_rn + offs_d[None, :]
    # position encoding
    seq_len = tl.load(Lengths + off_b)
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + off_b)
        if INTERLEAVE_TARGETS:
            high_ind = seq_len - num_targets * 2
        else:
            high_ind = seq_len - num_targets
    else:
        high_ind = seq_len
    pos_inds = tl.where(offs_n < high_ind, offs_n, high_ind)
    pos_inds = high_ind - pos_inds + max_contextual_seq_len
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(offs_n < max_contextual_seq_len, offs_n, pos_inds)
    tl.store(PosInds + seq_start + offs_n, pos_inds, mask=mask_n)
    pos_emb_offsets = pos_inds[:, None] * stride_pn + offs_d[None, :]
    # timestamp encoding
    ts = tl.load(TS + seq_start + offs_n, mask=mask_n)
    query_time = tl.load(TS + seq_end - 1)
    ts = query_time - ts + time_delta
    ts = tl.where(ts > 1e-6, ts, 1e-6) / time_bucket_increments
    if TIME_BUCKET_FN == "log":
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    tl.store(TsInds + seq_start + offs_n, ts, mask=mask_n)
    ts_emb_offsets = ts[:, None] * stride_tn + offs_d[None, :]
    # dropout counters: one Philox call per 4 consecutive columns (a group)
    NG: tl.constexpr = BLOCK_D // 4
    offs_g = tl.arange(0, NG)
    g_v8 = offs_g >> 1
    g_call = (offs_g & 1).to(tl.uint32)
    row_v8 = (seq_start.to(tl.int32) + offs_n) * (D >> 3)
    k0 = seed_lo.to(tl.uint32, bitcast=True)
    k1 = seed_hi.to(tl.uint32, bitcast=True)
    base_lo = ctr_lo.to(tl.uint32, bitcast=True)
    base_hi = ctr_hi.to(tl.uint32, bitcast=True)
    zero_g = tl.zeros([BLOCK_N, NG], dtype=tl.uint32)
    Y += seq_start.to(tl.int64) * stride_on
    if MASK_BITS:
        Mask += seq_start.to(tl.int64) * (D >> 3)
    else:
        Mask += seq_start.to(tl.int64) * stride_on
    out_offsets = offs_n[:, None] * stride_on + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < D)  # Triton 3.9: `and` on tensors is deprecated
        if has_left:
            seq_l = tl.load(Left + left_offs, mask=mask & is_left[:, None])
            seq_r = tl.load(Right + right_offs, mask=mask & is_right[:, None])
            seq_emb = tl.where(is_left[:, None], seq_l, seq_r)
        else:
            seq_emb = tl.load(Right + right_offs, mask=mask)
        seq_emb = (seq_emb.to(tl.float32) * alpha).to(seq_emb.dtype)
        pos_emb = tl.load(PosEmb + pos_emb_offsets, mask=mask)
        ts_emb = tl.load(TsEmb + ts_emb_offsets, mask=mask)
        x = seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype)
        # aten's Philox words for this [BLOCK_N, BLOCK_D] tile
        v8 = row_v8[:, None] + ((_d >> 3) + g_v8)[None, :]
        if LOG2_T >= 0:
            t = (v8 & (T - 1)).to(tl.uint32)
            it = (v8 >> LOG2_T).to(tl.uint32)
        else:
            t = (v8 % T).to(tl.uint32)
            it = (v8 // T).to(tl.uint32)
        cnt = it * 2 + g_call[None, :]
        c0 = cnt + base_lo
        c1 = base_hi + (c0 < cnt).to(tl.uint32)
        r0, r1, r2, r3 = _philox4x32_10_ctr_mad64(c0, c1, t, zero_g, k0, k1)
        # column 4 * g + j <- word j of group g
        r = tl.reshape(
            tl.join(tl.join(r0, r2), tl.join(r1, r3)), (BLOCK_N, BLOCK_D)
        )
        keep = r < KEEP_BELOW
        f = x.to(tl.float32) * tl.where(keep, scale, 0.0)
        y16 = _f32_to_bf16_bits_hip(f)
        tl.store(Y + out_offsets, y16.to(tl.int16), mask=mask)
        if MASK_BITS:
            # mask byte (_d >> 3) + v of a row <- groups 2 v (bits 0-3) and
            # 2 v + 1 (bits 4-7)
            lo, hi = tl.split(
                tl.reshape(
                    _keep_nibble(r0, r1, r2, r3, KEEP_BELOW), (BLOCK_N, NG // 2, 2)
                )
            )
            offs_b = (_d >> 3) + tl.arange(0, NG // 2)
            tl.store(
                Mask + offs_n[:, None] * (D >> 3) + offs_b[None, :],
                (lo | (hi << 4)).to(tl.int8),
                mask=(offs_n[:, None] < seq_len) & (offs_b[None, :] < (D >> 3)),
            )
        else:
            tl.store(Mask + out_offsets, keep.to(tl.int8), mask=mask)
        left_offs += BLOCK_D
        right_offs += BLOCK_D
        pos_emb_offsets += BLOCK_D
        ts_emb_offsets += BLOCK_D
        out_offsets += BLOCK_D
        offs_d += BLOCK_D


@triton.jit(do_not_specialize=["seed_lo", "seed_hi", "ctr_lo", "ctr_hi"])
def _posenc_concat_dropout_v2_kernel(
    Left,
    Right,
    Offsets,
    Lengths,
    PosEmb,
    TsEmb,
    Y,
    Mask,
    TS,
    PosInds,
    TsInds,
    SrcInds,
    NumTargets,
    D,
    num_time_buckets,
    time_bucket_increments,
    time_bucket_scale,
    time_delta,
    max_contextual_seq_len,
    max_pos_ind,
    ctx_len,
    stride_ln,
    stride_rn,
    stride_pn,
    stride_tn,
    stride_on,
    alpha,
    scale,
    seed_lo,
    seed_hi,
    ctr_lo,
    ctr_hi,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    T: tl.constexpr,
    LOG2_T: tl.constexpr,
    KEEP_BELOW: tl.constexpr,
    MASK_BITS: tl.constexpr = False,
    HW_EPI: tl.constexpr = False,
):
    """DLRM_HSTU_FP8_POSENC_DROPOUT_V2: _posenc_concat_dropout_kernel (same
    arguments, same per-element ops, same element -> Philox mapping; see its
    docstring) in a bandwidth layout:
    - tile off_n of sequence off_b covers rows
      [seq_len - (off_n + 1) * BLOCK_N, seq_len - off_n * BLOCK_N) (counted from
      the end; the first, partial tile is masked at row < 0), so the tiles that
      run concurrently across sequences read the same pos-table rows;
    - thread chunk = 8 consecutive columns = Philox calls 2 * it and 2 * it + 1
      of one aten thread, both evaluated in the x layout ([BLOCK_N, BLOCK_D // 8]
      counters; words interleaved in registers, no LDS conversion);
    - streamed seq loads use .cg, y / mask stores are non-temporal (.cs).
    MASK_BITS: packed mask, as in _posenc_concat_dropout_kernel (its byte
    store keeps the default cache policy).
    HW_EPI (DLRM_HSTU_FP8_POSENC_DROP_HWEPI; gfx950 only, passed only when on):
    y = f.to(tl.bfloat16) (v_cvt_pk_bf16_f32) instead of
    _f32_to_bf16_bits_hip(f), and the Philox XOR pairs as v_bitop3_b32
    (_philox4x32_10_bitop3); same output bits."""
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_b)
    seq_end = tl.load(Offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    first_n = (seq_len - start_n - BLOCK_N).to(tl.int32)
    offs_n = first_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n >= 0
    is_left = offs_n < ctx_len
    is_right = offs_n >= ctx_len
    left_rows = off_b.to(tl.int64) * ctx_len + offs_n
    right_rows = seq_start.to(tl.int64) - ctx_len * (off_b.to(tl.int64) + 1) + offs_n
    src = tl.where(is_left, -(left_rows + 1), right_rows)
    tl.store(SrcInds + seq_start + offs_n, src.to(tl.int32), mask=mask_n)
    has_left = first_n < ctx_len
    left_offs = left_rows[:, None] * stride_ln + offs_d[None, :]
    right_offs = right_rows[:, None] * stride_rn + offs_d[None, :]
    # position encoding
    seq_len = tl.load(Lengths + off_b)
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + off_b)
        if INTERLEAVE_TARGETS:
            high_ind = seq_len - num_targets * 2
        else:
            high_ind = seq_len - num_targets
    else:
        high_ind = seq_len
    pos_inds = tl.where(offs_n < high_ind, offs_n, high_ind)
    pos_inds = high_ind - pos_inds + max_contextual_seq_len
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(offs_n < max_contextual_seq_len, offs_n, pos_inds)
    tl.store(PosInds + seq_start + offs_n, pos_inds, mask=mask_n)
    pos_emb_offsets = pos_inds[:, None] * stride_pn + offs_d[None, :]
    # timestamp encoding
    ts = tl.load(TS + seq_start + offs_n, mask=mask_n)
    query_time = tl.load(TS + seq_end - 1)
    ts = query_time - ts + time_delta
    ts = tl.where(ts > 1e-6, ts, 1e-6) / time_bucket_increments
    if TIME_BUCKET_FN == "log":
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    tl.store(TsInds + seq_start + offs_n, ts, mask=mask_n)
    ts_emb_offsets = ts[:, None] * stride_tn + offs_d[None, :]
    # dropout counters: per 8-column chunk v8, aten thread t = v8 % T makes the
    # calls 2 * it and 2 * it + 1 (it = v8 // T) for its columns 0-3 and 4-7
    NV: tl.constexpr = BLOCK_D // 8
    offs_v = tl.arange(0, NV)
    row_v8 = (seq_start.to(tl.int32) + offs_n) * (D >> 3)
    k0 = seed_lo.to(tl.uint32, bitcast=True)
    k1 = seed_hi.to(tl.uint32, bitcast=True)
    base_lo = ctr_lo.to(tl.uint32, bitcast=True)
    base_hi = ctr_hi.to(tl.uint32, bitcast=True)
    zero_v = tl.zeros([BLOCK_N, NV], dtype=tl.uint32)
    Y += seq_start.to(tl.int64) * stride_on
    if MASK_BITS:
        Mask += seq_start.to(tl.int64) * (D >> 3)
    else:
        Mask += seq_start.to(tl.int64) * stride_on
    out_offsets = offs_n[:, None] * stride_on + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] >= 0) & (offs_n[:, None] < seq_len) & (offs_d[None, :] < D)
        if has_left:
            seq_l = tl.load(Left + left_offs, mask=mask & is_left[:, None], cache_modifier=".cg")
            seq_r = tl.load(Right + right_offs, mask=mask & is_right[:, None], cache_modifier=".cg")
            seq_emb = tl.where(is_left[:, None], seq_l, seq_r)
        else:
            seq_emb = tl.load(Right + right_offs, mask=mask, cache_modifier=".cg")
        seq_emb = (seq_emb.to(tl.float32) * alpha).to(seq_emb.dtype)
        pos_emb = tl.load(PosEmb + pos_emb_offsets, mask=mask)
        ts_emb = tl.load(TsEmb + ts_emb_offsets, mask=mask)
        x = seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype)
        v8 = row_v8[:, None] + ((_d >> 3) + offs_v)[None, :]
        if LOG2_T >= 0:
            t = (v8 & (T - 1)).to(tl.uint32)
            it = (v8 >> LOG2_T).to(tl.uint32)
        else:
            t = (v8 % T).to(tl.uint32)
            it = (v8 // T).to(tl.uint32)
        cnt = it * 2
        c0 = cnt + base_lo
        c1 = base_hi + (c0 < cnt).to(tl.uint32)
        if HW_EPI:
            a0, a1, a2, a3 = _philox4x32_10_bitop3(c0, c1, t, zero_v, k0, k1)
        else:
            a0, a1, a2, a3 = _philox4x32_10_ctr_mad64(c0, c1, t, zero_v, k0, k1)
        cnt = cnt + 1
        c0 = cnt + base_lo
        c1 = base_hi + (c0 < cnt).to(tl.uint32)
        if HW_EPI:
            b0, b1, b2, b3 = _philox4x32_10_bitop3(c0, c1, t, zero_v, k0, k1)
        else:
            b0, b1, b2, b3 = _philox4x32_10_ctr_mad64(c0, c1, t, zero_v, k0, k1)
        # column 8 * v + 4 * h + j <- word j of call h of chunk v
        ra = tl.join(tl.join(a0, a2), tl.join(a1, a3))
        rb = tl.join(tl.join(b0, b2), tl.join(b1, b3))
        r = tl.reshape(
            tl.permute(tl.join(ra, rb), (0, 1, 4, 2, 3)), (BLOCK_N, BLOCK_D)
        )
        keep = r < KEEP_BELOW
        f = x.to(tl.float32) * tl.where(keep, scale, 0.0)
        if HW_EPI:
            # gfx950 v_cvt_pk_bf16_f32 == _f32_to_bf16_bits_hip on all 2^32 f32
            y16 = f.to(tl.bfloat16).to(tl.int16, bitcast=True)
        else:
            y16 = _f32_to_bf16_bits_hip(f)
        tl.store(Y + out_offsets, y16.to(tl.int16), mask=mask, cache_modifier=".cs")
        if MASK_BITS:
            # mask byte (_d >> 3) + v of a row <- chunk v: call 2 * it (bits
            # 0-3), call 2 * it + 1 (bits 4-7). Default cache policy: a row's
            # bytes arrive in BLOCK_D / 8-byte pieces, which the non-temporal
            # hint made slower (paired, MI350X: -0.045 vs -0.022 ms with .cs)
            offs_b = (_d >> 3) + offs_v
            tl.store(
                Mask + offs_n[:, None] * (D >> 3) + offs_b[None, :],
                (
                    _keep_nibble(a0, a1, a2, a3, KEEP_BELOW)
                    | (_keep_nibble(b0, b1, b2, b3, KEEP_BELOW) << 4)
                ).to(tl.int8),
                mask=(offs_n[:, None] >= 0)
                & (offs_n[:, None] < seq_len)
                & (offs_b[None, :] < (D >> 3)),
            )
        else:
            tl.store(Mask + out_offsets, keep.to(tl.int8), mask=mask, cache_modifier=".cs")
        left_offs += BLOCK_D
        right_offs += BLOCK_D
        pos_emb_offsets += BLOCK_D
        ts_emb_offsets += BLOCK_D
        out_offsets += BLOCK_D
        offs_d += BLOCK_D


@triton.jit
def _add_embeddings_bwd_tiled_split_kernel(
    In,
    KeyInds,
    ValueInds,
    Out,
    SrcInds,
    DLeft,
    DRight,
    N,
    D,
    stride_in,
    stride_on,
    stride_ln,
    stride_rn,
    alpha,
    BLOCK_D: tl.constexpr,
    BLOCK: tl.constexpr,
    KEY_I16: tl.constexpr = False,
):
    """_add_embeddings_bwd_tiled_kernel(WRITE_DSEQ=True) whose d_seq rows go
    straight to d_left / d_right (DLRM_HSTU_FP8_POSENC_CONCAT_FOLD): replaces
    the d_seq write + split_2D_jagged copy.  Same values, same reduction.
    KEY_I16: see _add_embeddings_bwd_tiled_kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_r = offs < N
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    if KEY_I16:
        keys = tl.load(KeyInds + offs, mask=mask_r, other=0).to(tl.int32)
        keys = tl.where(mask_r, keys, 2147483647)
    else:
        keys = tl.load(KeyInds + offs, mask=mask_r, other=2147483647).to(tl.int32)
    vals = tl.load(ValueInds + offs, mask=mask_r, other=0).to(tl.int64)
    m2 = mask_r[:, None] & mask_d[None, :]
    x = tl.load(In + vals[:, None] * stride_in + offs_d[None, :], mask=m2, other=0.0)
    src = tl.load(SrcInds + vals, mask=mask_r, other=0).to(tl.int64)
    is_left = src < 0
    is_right = src >= 0
    left_rows = -src - 1
    dx = (x.to(tl.float32) * alpha).to(DRight.dtype.element_ty)
    # two masked stores (no pointer select; see the forward kernel)
    tl.store(
        DLeft + left_rows[:, None] * stride_ln + offs_d[None, :],
        dx,
        mask=m2 & is_left[:, None],
    )
    tl.store(
        DRight + src[:, None] * stride_rn + offs_d[None, :],
        dx,
        mask=m2 & is_right[:, None],
    )
    xf = x.to(tl.float32)
    k = tl.min(keys, axis=0)
    k_last = tl.max(tl.where(mask_r, keys, -1), axis=0)
    while k <= k_last:
        sel = keys == k
        acc = tl.sum(tl.where(sel[:, None], xf, 0.0), axis=0)
        tl.atomic_add(
            Out + k.to(tl.int64) * stride_on + offs_d,
            acc,
            mask=mask_d,
            sem="relaxed",
        )
        k = tl.min(tl.where(keys > k, keys, 2147483647), axis=0)


@triton.jit
def _add_embeddings_bwd_rowloop_body(
    In,
    KeyInds,
    ValueInds,
    Out,
    SrcInds,
    DLeft,
    DRight,
    N,
    D,
    stride_in,
    stride_on,
    stride_ln,
    stride_rn,
    alpha,
    WRITE_SPLIT: tl.constexpr,
    SUB: tl.constexpr,
    NSUB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KEY_I16: tl.constexpr,
):
    """DLRM_HSTU_FP8_POSENC_BWD_V2: segmented reduction of the sorted (key,
    value) rows of In into Out (fp32, atomics) over NSUB sub-tiles of SUB rows.
    A running fp32 sum of the current key is carried across sub-tiles and
    flushed with one atomic per (program, key); keys are sorted, so the
    distinct keys of a sub-tile arrive in increasing order. WRITE_SPLIT: also
    writes the rows' (In * alpha) to DLeft / DRight (each row exactly once;
    same per-element op as _add_embeddings_bwd_tiled_split_kernel).
    KEY_I16: see _add_embeddings_bwd_tiled_kernel. NSUB stays a constexpr: a
    runtime loop bound costs the ts kernel 4 VGPRs (128 -> 132, occupancy
    4 -> 3 in its d_out <= 2 GiB build), +0.06-0.07 ms at NSUB=64."""
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    row0 = pid.to(tl.int64) * (SUB * NSUB)
    cur = tl.full([], -1, tl.int32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(NSUB):
        offs = row0 + i * SUB + tl.arange(0, SUB)
        mask_r = offs < N
        if KEY_I16:
            keys = tl.load(KeyInds + offs, mask=mask_r, other=0).to(tl.int32)
            keys = tl.where(mask_r, keys, 2147483647)
        else:
            keys = tl.load(KeyInds + offs, mask=mask_r, other=2147483647).to(tl.int32)
        vals = tl.load(ValueInds + offs, mask=mask_r, other=0).to(tl.int64)
        m2 = mask_r[:, None] & mask_d[None, :]
        x = tl.load(
            In + vals[:, None] * stride_in + offs_d[None, :], mask=m2, other=0.0
        )
        if WRITE_SPLIT:
            src = tl.load(SrcInds + vals, mask=mask_r, other=0).to(tl.int64)
            is_left = src < 0
            is_right = src >= 0
            left_rows = -src - 1
            dx = (x.to(tl.float32) * alpha).to(DRight.dtype.element_ty)
            tl.store(
                DLeft + left_rows[:, None] * stride_ln + offs_d[None, :],
                dx,
                mask=m2 & is_left[:, None],
            )
            tl.store(
                DRight + src[:, None] * stride_rn + offs_d[None, :],
                dx,
                mask=m2 & is_right[:, None],
            )
        xf = x.to(tl.float32)
        k = tl.min(keys, axis=0)
        k_last = tl.max(tl.where(mask_r, keys, -1), axis=0)
        if k == k_last:
            # one key in the sub-tile (the common case): where(sel, xf, 0) == xf
            part = tl.sum(xf, axis=0)
            if k == cur:
                acc += part
            else:
                if cur >= 0:
                    tl.atomic_add(
                        Out + cur.to(tl.int64) * stride_on + offs_d,
                        acc,
                        mask=mask_d,
                        sem="relaxed",
                    )
                cur = k
                acc = part
            k = k_last + 1
        while k <= k_last:
            sel = keys == k
            part = tl.sum(tl.where(sel[:, None], xf, 0.0), axis=0)
            if k == cur:
                acc += part
            else:
                if cur >= 0:
                    tl.atomic_add(
                        Out + cur.to(tl.int64) * stride_on + offs_d,
                        acc,
                        mask=mask_d,
                        sem="relaxed",
                    )
                cur = k
                acc = part
            k = tl.min(tl.where(keys > k, keys, 2147483647), axis=0)
    if cur >= 0:
        tl.atomic_add(
            Out + cur.to(tl.int64) * stride_on + offs_d,
            acc,
            mask=mask_d,
            sem="relaxed",
        )


@triton.jit
def _add_embeddings_bwd_rowloop_split_kernel(
    In,
    KeyInds,
    ValueInds,
    Out,
    SrcInds,
    DLeft,
    DRight,
    N,
    D,
    stride_in,
    stride_on,
    stride_ln,
    stride_rn,
    alpha,
    SUB: tl.constexpr,
    NSUB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KEY_I16: tl.constexpr,
):
    """DLRM_HSTU_FP8_POSENC_BWD_V2 replacement of
    _add_embeddings_bwd_tiled_split_kernel (d_pos + d_left/d_right)."""
    _add_embeddings_bwd_rowloop_body(
        In, KeyInds, ValueInds, Out, SrcInds, DLeft, DRight, N, D, stride_in,
        stride_on, stride_ln, stride_rn, alpha, True, SUB, NSUB, BLOCK_D, KEY_I16,
    )


@triton.jit
def _add_embeddings_bwd_rowloop_kernel(
    In,
    KeyInds,
    ValueInds,
    Out,
    N,
    D,
    stride_in,
    stride_on,
    SUB: tl.constexpr,
    NSUB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KEY_I16: tl.constexpr,
):
    """DLRM_HSTU_FP8_POSENC_BWD_V2 replacement of
    _add_embeddings_bwd_tiled_kernel(WRITE_DSEQ=False) (d_ts)."""
    _add_embeddings_bwd_rowloop_body(
        In, KeyInds, ValueInds, Out, In, In, In, N, D, stride_in,
        stride_on, 0, 0, 1.0, False, SUB, NSUB, BLOCK_D, KEY_I16,
    )


def _posenc_bwd_v2_within_2gb(t: torch.Tensor) -> bool:
    """The predicate of Triton's AMD backend (HIPBackend.is_within_2gb, with
    knobs.amd.use_buffer_ops): a tensor argument whose storage is <= 2 GiB
    gets the 'S' (tt.pointer_range=32) specialization, i.e. its own compile."""
    return t.untyped_storage().size() <= 2**31 - 1


def _posenc_bwd_v2_nsub(N: int, sub: int, nsub_max: int) -> int:
    """Largest power of two <= nsub_max keeping >= _POSENC_BWD_V2_MIN_PROGRAMS
    programs (1 for small N)."""
    nsub = 1
    while (
        nsub * 2 <= nsub_max
        and triton.cdiv(N, sub * nsub * 2) >= _POSENC_BWD_V2_MIN_PROGRAMS
    ):
        nsub *= 2
    return nsub


def _posenc_bwd_v2_launch(
    d_out: torch.Tensor,
    sorted_pos_key_inds: torch.Tensor,
    sorted_pos_value_inds: torch.Tensor,
    sorted_ts_key_inds: torch.Tensor,
    sorted_ts_value_inds: torch.Tensor,
    src_inds: torch.Tensor,
    d_pos_embeddings: torch.Tensor,
    d_ts_embeddings: torch.Tensor,
    d_left: torch.Tensor,
    d_right: torch.Tensor,
    D: int,
    BLOCK_D: int,
    alpha: float,
) -> None:
    N = d_out.shape[0]
    sub = _POSENC_BWD_V2_SUB
    nsub_pos = _posenc_bwd_v2_nsub(N, sub, _POSENC_BWD_V2_NSUB_POS)
    # The ts geometry follows d_out's 2 GiB 'S' specialization, so SUB / NSUB only
    # change where every kernel here (tiled ones included) recompiles anyway.
    if _posenc_bwd_v2_within_2gb(d_out):
        sub_ts, nsub_ts_max = sub, _POSENC_BWD_V2_NSUB_TS_2G
    else:
        sub_ts, nsub_ts_max = _POSENC_BWD_V2_SUB_TS, _POSENC_BWD_V2_NSUB_TS
    nsub_ts = _posenc_bwd_v2_nsub(N, sub_ts, nsub_ts_max)
    _add_embeddings_bwd_rowloop_split_kernel[(triton.cdiv(N, sub * nsub_pos),)](
        d_out,
        sorted_pos_key_inds,
        sorted_pos_value_inds,
        d_pos_embeddings,
        src_inds,
        d_left,
        d_right,
        N,
        D,
        d_out.stride(0),
        d_pos_embeddings.stride(0),
        d_left.stride(0),
        d_right.stride(0),
        alpha,
        SUB=sub,
        NSUB=nsub_pos,
        BLOCK_D=BLOCK_D,
        KEY_I16=sorted_pos_key_inds.dtype == torch.int16,
        num_warps=_POSENC_BWD_V2_WARPS,
    )
    _add_embeddings_bwd_rowloop_kernel[(triton.cdiv(N, sub_ts * nsub_ts),)](
        d_out,
        sorted_ts_key_inds,
        sorted_ts_value_inds,
        d_ts_embeddings,
        N,
        D,
        d_out.stride(0),
        d_ts_embeddings.stride(0),
        SUB=sub_ts,
        NSUB=nsub_ts,
        BLOCK_D=BLOCK_D,
        KEY_I16=sorted_ts_key_inds.dtype == torch.int16,
        num_warps=_POSENC_BWD_V2_WARPS,
    )


class _AddTimestampPositionEmbeddingsFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        pos_embeddings: torch.Tensor,
        ts_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        max_seq_len: int,
        max_contextual_seq_len: int,
        seq_lengths: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        interleave_targets: bool,
        time_bucket_fn: str,
        alpha: Optional[float] = None,
        need_bwd: bool = True,
    ):
        seq_embeddings = switch_to_contiguous_if_needed(seq_embeddings)
        pos_embeddings = switch_to_contiguous_if_needed(pos_embeddings)
        ts_embeddings = switch_to_contiguous_if_needed(ts_embeddings)

        max_pos_ind = pos_embeddings.shape[0]
        B = seq_lengths.shape[0]
        N, D = seq_embeddings.shape
        assert len(pos_embeddings.shape) == 2
        assert len(ts_embeddings.shape) == 2
        assert pos_embeddings.shape[1] == D, (
            "shape[1] of pos_embeddings much match seq_embeddings"
        )
        assert ts_embeddings.shape[1] == D, (
            "shape[1] of ts_embeddings much match seq_embeddings"
        )
        out = torch.empty_like(seq_embeddings)

        timestamps = switch_to_contiguous_if_needed(timestamps)
        ts_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)
        pos_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)
        ts_emb_size = ts_embeddings.shape[0]

        grid = lambda meta: (  # noqa E731
            B,
            triton.cdiv(max_seq_len, meta["BLOCK_N"]),
        )
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        _add_timestamp_position_embeddings_kernel[grid](
            SeqEmb=seq_embeddings,
            Offsets=seq_offsets,
            Lengths=seq_lengths,
            PosEmb=pos_embeddings,
            TsEmb=ts_embeddings,
            Out=out,
            TS=timestamps,
            PosInds=pos_inds,
            TsInds=ts_inds,
            NumTargets=num_targets,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            D=D,
            num_time_buckets=ts_emb_size - 1,
            time_bucket_increments=60.0,
            time_bucket_scale=1.0,
            time_delta=0,
            max_contextual_seq_len=max_contextual_seq_len,
            max_pos_ind=max_pos_ind,
            stride_sn=seq_embeddings.stride(0),
            stride_pn=pos_embeddings.stride(0),
            stride_tn=ts_embeddings.stride(0),
            stride_on=out.stride(0),
            alpha=1.0 if alpha is None else float(alpha),
            TRAINING=True,
            HAS_MULTIPLE_TARGETS=num_targets is not None,
            INTERLEAVE_TARGETS=interleave_targets,
            TIME_BUCKET_FN=time_bucket_fn,
            BLOCK_D=BLOCK_D,
            HAS_ALPHA=alpha is not None,
        )
        if not need_bwd:
            # DLRM_HSTU_FP8_EVAL_FWD_HYGIENE (decided by the wrapper): no
            # backward can run, and the arange + sorts below feed only its
            # table-gradient reduction. Nothing is saved.
            return out
        try:
            values = torch.arange(0, N, dtype=torch.int32, device=timestamps.device)
            sorted_ts_key_inds, sorted_ts_value_inds = torch.ops.hammer.sort_kv_pairs(
                ts_inds, values
            )
            sorted_pos_key_inds, sorted_pos_value_inds = torch.ops.hammer.sort_kv_pairs(
                pos_inds, values
            )
        except Exception:
            sorted_ts_key_inds, sorted_ts_value_inds = torch.sort(ts_inds)
            sorted_pos_key_inds, sorted_pos_value_inds = torch.sort(pos_inds)
        ctx.save_for_backward(
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
        )
        ctx.B = B
        ctx.D = D
        ctx.max_seq_len = max_seq_len
        ctx.pos_emb_size = pos_embeddings.shape[0]
        ctx.ts_emb_size = ts_emb_size
        ctx.pos_dtype = pos_embeddings.dtype
        ctx.ts_dtype = ts_embeddings.dtype
        ctx.alpha = alpha
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        None,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        if ctx.alpha is not None:
            return _AddTimestampPositionEmbeddingsFunction._backward_tiled(
                ctx, d_out
            )
        (
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
        ) = ctx.saved_tensors
        d_pos_embeddings = torch.empty(
            (ctx.pos_emb_size, ctx.D), device=d_out.device, dtype=torch.float32
        )
        d_ts_embeddings = torch.empty(
            (ctx.ts_emb_size, ctx.D), device=d_out.device, dtype=torch.float32
        )
        grid = lambda meta: (triton.cdiv(d_out.shape[0], meta["BLOCK"]),)  # noqa E731
        AUTOTUNE_B = prev_power_of_2(ctx.B)
        _add_embeddings_bwd_kernel[grid](
            In=d_out,
            KeyInds=sorted_pos_key_inds,
            ValueInds=sorted_pos_value_inds,
            Out=d_pos_embeddings,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            AUTOTUNE_B=AUTOTUNE_B,
            D=ctx.D,
            jagged_size=d_out.shape[0],
            stride_in=d_out.stride(0),
            stride_on=d_pos_embeddings.stride(0),
            BLOCK_D=triton.next_power_of_2(ctx.D),
        )
        _add_embeddings_bwd_kernel[grid](
            In=d_out,
            KeyInds=sorted_ts_key_inds,
            ValueInds=sorted_ts_value_inds,
            Out=d_ts_embeddings,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            AUTOTUNE_B=AUTOTUNE_B,
            D=ctx.D,
            jagged_size=d_out.shape[0],
            stride_in=d_out.stride(0),
            stride_on=d_ts_embeddings.stride(0),
            BLOCK_D=triton.next_power_of_2(ctx.D),
        )
        return (
            d_out,
            None,
            d_pos_embeddings.to(ctx.pos_dtype),
            d_ts_embeddings.to(ctx.ts_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # need_bwd
        )

    @staticmethod
    def _backward_tiled(ctx, d_out: torch.Tensor):
        (
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
        ) = ctx.saved_tensors
        if d_out.stride(-1) != 1:
            d_out = d_out.contiguous()
        N = d_out.shape[0]
        D = ctx.D
        d_pos_embeddings = torch.zeros(
            (ctx.pos_emb_size, D), device=d_out.device, dtype=torch.float32
        )
        d_ts_embeddings = torch.zeros(
            (ctx.ts_emb_size, D), device=d_out.device, dtype=torch.float32
        )
        d_seq = torch.empty((N, D), device=d_out.device, dtype=d_out.dtype)
        BLOCK = _POSENC_BWD_BLOCK
        grid = (triton.cdiv(N, BLOCK),)
        for keys, vals, out, write in (
            (sorted_pos_key_inds, sorted_pos_value_inds, d_pos_embeddings, True),
            (sorted_ts_key_inds, sorted_ts_value_inds, d_ts_embeddings, False),
        ):
            _add_embeddings_bwd_tiled_kernel[grid](
                d_out,
                keys,
                vals,
                out,
                d_seq,
                N,
                D,
                d_out.stride(0),
                out.stride(0),
                d_seq.stride(0),
                float(ctx.alpha),
                WRITE_DSEQ=write,
                BLOCK_D=triton.next_power_of_2(D),
                BLOCK=BLOCK,
                num_warps=_POSENC_BWD_WARPS,
            )
        return (
            d_seq,
            None,
            d_pos_embeddings.to(ctx.pos_dtype),
            d_ts_embeddings.to(ctx.ts_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # need_bwd
        )


def _posenc_concat_backward(
    ctx,  # pyre-ignore[2]
    d_out: torch.Tensor,
    sorted_pos_key_inds: torch.Tensor,
    sorted_pos_value_inds: torch.Tensor,
    sorted_ts_key_inds: torch.Tensor,
    sorted_ts_value_inds: torch.Tensor,
    src_inds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of the concat-folded position kernel (shared by
    _AddTimestampPositionEmbeddingsConcatFunction and the posenc+dropout
    fusion): (d_left, d_right, d_pos_embeddings, d_ts_embeddings)."""
    if d_out.stride(-1) != 1:
        d_out = d_out.contiguous()
    N = d_out.shape[0]
    D = ctx.D
    if _zero_fold_enabled():
        # zero-fills (ops/zero_fold.py): both fp32 accumulators from ONE zero fill
        # (rows [0, P) pos, [P, P + T) ts; each view contiguous with row stride D,
        # as before) -- one hipMemsetAsync on the compute stream instead of two.
        _dz = torch.zeros(
            (ctx.pos_emb_size + ctx.ts_emb_size, D),
            device=d_out.device,
            dtype=torch.float32,
        )
        d_pos_embeddings = _dz[: ctx.pos_emb_size]
        d_ts_embeddings = _dz[ctx.pos_emb_size :]
    else:
        d_pos_embeddings = torch.zeros(
            (ctx.pos_emb_size, D), device=d_out.device, dtype=torch.float32
        )
        d_ts_embeddings = torch.zeros(
            (ctx.ts_emb_size, D), device=d_out.device, dtype=torch.float32
        )
    d_left = torch.empty((ctx.N_left, D), device=d_out.device, dtype=d_out.dtype)
    d_right = torch.empty((ctx.N_right, D), device=d_out.device, dtype=d_out.dtype)
    BLOCK = _POSENC_BWD_BLOCK
    grid = (triton.cdiv(N, BLOCK),)
    BLOCK_D = triton.next_power_of_2(D)
    global _POSENC_BWD_V2_FAILED
    if _POSENC_BWD_V2 and not _POSENC_BWD_V2_FAILED and N > 0 and BLOCK_D <= 512:
        try:
            _posenc_bwd_v2_launch(
                d_out,
                sorted_pos_key_inds,
                sorted_pos_value_inds,
                sorted_ts_key_inds,
                sorted_ts_value_inds,
                src_inds,
                d_pos_embeddings,
                d_ts_embeddings,
                d_left,
                d_right,
                D,
                BLOCK_D,
                ctx.alpha,
            )
            _log_once(
                "bwd_v2",
                "armed OK: posenc row-loop backward ACTIVE "
                "(DLRM_HSTU_FP8_POSENC_BWD_V2)",
            )
            return (
                d_left,
                d_right,
                d_pos_embeddings.to(ctx.pos_dtype),
                d_ts_embeddings.to(ctx.ts_dtype),
            )
        except Exception as e:  # fail closed: the tiled kernels below
            _POSENC_BWD_V2_FAILED = True
            _logger.warning(
                "DLRM_HSTU_FP8_POSENC_BWD_V2 launch failed (%r); "
                "falling back to the tiled posenc backward",
                e,
            )
            d_pos_embeddings.zero_()
            d_ts_embeddings.zero_()
    _add_embeddings_bwd_tiled_split_kernel[grid](
        d_out,
        sorted_pos_key_inds,
        sorted_pos_value_inds,
        d_pos_embeddings,
        src_inds,
        d_left,
        d_right,
        N,
        D,
        d_out.stride(0),
        d_pos_embeddings.stride(0),
        d_left.stride(0),
        d_right.stride(0),
        ctx.alpha,
        BLOCK_D=BLOCK_D,
        BLOCK=BLOCK,
        KEY_I16=sorted_pos_key_inds.dtype == torch.int16,
        num_warps=_POSENC_BWD_WARPS,
    )
    _add_embeddings_bwd_tiled_kernel[grid](
        d_out,
        sorted_ts_key_inds,
        sorted_ts_value_inds,
        d_ts_embeddings,
        d_left,  # unused (WRITE_DSEQ=False)
        N,
        D,
        d_out.stride(0),
        d_ts_embeddings.stride(0),
        d_left.stride(0),
        ctx.alpha,
        WRITE_DSEQ=False,
        BLOCK_D=BLOCK_D,
        BLOCK=BLOCK,
        KEY_I16=sorted_ts_key_inds.dtype == torch.int16,
        num_warps=_POSENC_BWD_WARPS,
    )
    return (
        d_left,
        d_right,
        d_pos_embeddings.to(ctx.pos_dtype),
        d_ts_embeddings.to(ctx.ts_dtype),
    )


class _AddTimestampPositionEmbeddingsConcatFunction(torch.autograd.Function):
    """DLRM_HSTU_FP8_POSENC_CONCAT_FOLD: concat_2D_jagged(left dense, right
    jagged) followed by _AddTimestampPositionEmbeddingsFunction(alpha=...) as
    one autograd node; forward/backward values are bitwise those of the pair."""

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values_left: torch.Tensor,
        values_right: torch.Tensor,
        seq_offsets: torch.Tensor,
        pos_embeddings: torch.Tensor,
        ts_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        max_seq_len: int,
        max_contextual_seq_len: int,
        seq_lengths: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        interleave_targets: bool,
        time_bucket_fn: str,
        alpha: float,
        need_bwd: bool = True,
    ):
        values_left = switch_to_contiguous_if_needed(values_left)
        values_right = switch_to_contiguous_if_needed(values_right)
        pos_embeddings = switch_to_contiguous_if_needed(pos_embeddings)
        ts_embeddings = switch_to_contiguous_if_needed(ts_embeddings)
        timestamps = switch_to_contiguous_if_needed(timestamps)

        max_pos_ind = pos_embeddings.shape[0]
        B = seq_lengths.shape[0]
        N_left, D = values_left.shape
        N_right = values_right.shape[0]
        N = N_left + N_right
        ctx_len = N_left // B
        assert ctx_len * B == N_left
        assert values_right.shape[1] == D
        assert pos_embeddings.shape[1] == D and ts_embeddings.shape[1] == D
        out = torch.empty((N, D), device=values_right.device, dtype=values_right.dtype)
        ts_emb_size = ts_embeddings.shape[0]
        # int32 unless DLRM_HSTU_FP8_POSENC_SORT16 proves the int16 range
        idx_dtype = _posenc_index_dtype(
            max_pos_ind, max_contextual_seq_len, ts_emb_size
        )
        ts_inds = torch.empty((N,), device=out.device, dtype=idx_dtype)
        pos_inds = torch.empty((N,), device=out.device, dtype=idx_dtype)
        src_inds = torch.empty((N,), device=out.device, dtype=torch.int32)

        grid = lambda meta: (  # noqa E731
            B,
            triton.cdiv(max_seq_len, meta["BLOCK_N"]),
        )
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        _add_timestamp_position_embeddings_concat_kernel[grid](
            Left=values_left,
            Right=values_right,
            Offsets=seq_offsets,
            Lengths=seq_lengths,
            PosEmb=pos_embeddings,
            TsEmb=ts_embeddings,
            Out=out,
            TS=timestamps,
            PosInds=pos_inds,
            TsInds=ts_inds,
            SrcInds=src_inds,
            NumTargets=num_targets,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            D=D,
            num_time_buckets=ts_emb_size - 1,
            time_bucket_increments=60.0,
            time_bucket_scale=1.0,
            time_delta=0,
            max_contextual_seq_len=max_contextual_seq_len,
            max_pos_ind=max_pos_ind,
            ctx_len=ctx_len,
            stride_ln=values_left.stride(0),
            stride_rn=values_right.stride(0),
            stride_pn=pos_embeddings.stride(0),
            stride_tn=ts_embeddings.stride(0),
            stride_on=out.stride(0),
            alpha=float(alpha),
            HAS_MULTIPLE_TARGETS=num_targets is not None,
            INTERLEAVE_TARGETS=interleave_targets,
            TIME_BUCKET_FN=time_bucket_fn,
            BLOCK_D=BLOCK_D,
        )
        if not need_bwd:
            # DLRM_HSTU_FP8_EVAL_FWD_HYGIENE (decided by the wrapper): no
            # backward can run, and the index sorts below (_posenc_sort_pairs)
            # feed only its table-gradient reduction. Nothing is saved.
            return out
        (
            sorted_ts_key_inds,
            sorted_ts_value_inds,
            sorted_pos_key_inds,
            sorted_pos_value_inds,
        ) = _posenc_sort_pairs(
            ts_inds,
            pos_inds,
            N,
            timestamps.device,
            pos_meta=(
                seq_offsets,
                seq_lengths,
                num_targets,
                interleave_targets,
                max_contextual_seq_len,
                max_pos_ind,
            ),
            ts_max_key=ts_emb_size - 1,
        )
        ctx.save_for_backward(
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
            src_inds,
        )
        ctx.D = D
        ctx.N_left = N_left
        ctx.N_right = N_right
        ctx.pos_emb_size = pos_embeddings.shape[0]
        ctx.ts_emb_size = ts_emb_size
        ctx.pos_dtype = pos_embeddings.dtype
        ctx.ts_dtype = ts_embeddings.dtype
        ctx.alpha = float(alpha)
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, d_out: torch.Tensor):
        d_left, d_right, d_pos_embeddings, d_ts_embeddings = _posenc_concat_backward(
            ctx, d_out, *ctx.saved_tensors
        )
        return (
            d_left,
            d_right,
            None,
            d_pos_embeddings,
            d_ts_embeddings,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # need_bwd
        )


_DROPOUT_MAX_BLOCKS: Dict[int, int] = {}
_DROPOUT_KEEP_PARAMS: Dict[float, Tuple[int, float]] = {}


def _aten_dropout_rng(nelem: int) -> Tuple[int, int, int]:
    """Replay of aten dropout_cuda's launch geometry and
    gen->philox_cuda_state(counter_offset) for an nelem-element input on the
    current device (default CUDA generator, no graph capture): returns
    (seed, offset, T = grid * 256) and advances the generator exactly as that
    call does (offset += counter_offset rounded up to a multiple of 4)."""
    dev = torch.cuda.current_device()
    max_blocks = _DROPOUT_MAX_BLOCKS.get(dev)
    if max_blocks is None:
        props = torch.cuda.get_device_properties(dev)
        max_blocks = props.multi_processor_count * (
            props.max_threads_per_multi_processor // 256
        )
        _DROPOUT_MAX_BLOCKS[dev] = max_blocks
    grid = min(max_blocks, (nelem + 255) // 256)
    counter_offset = ((nelem - 1) // (256 * grid * 4) + 1) * 4
    gen = torch.cuda.default_generators[dev]
    seed = gen.initial_seed()
    offset = gen.get_offset()
    gen.set_offset(offset + (counter_offset + 3) // 4 * 4)
    return seed, offset, grid * 256


def _dropout_keep_params(p: float) -> Tuple[int, float]:
    """(KEEP_BELOW, scale) for aten's dropout_cuda(self, p1m = 1 - p).

    rocrand's curand_uniform4 element is u = 2^-32 + float(r) * 2^-32 (u32 ->
    f32 round-to-nearest-even; the product is exact) and the element is kept
    iff u < (float)p1m. u is monotone non-decreasing in r, so keep <=>
    r < KEEP_BELOW with KEEP_BELOW the smallest dropped r, found by bisection on
    the same fp32 expression (exhaustively re-checked over all 2^32 r on the
    GPU by the r21 validation). scale = (float)(1.0 / (double)(float)p1m)."""
    hit = _DROPOUT_KEEP_PARAMS.get(p)
    if hit is not None:
        return hit
    import numpy as np

    p1m = np.float32(1.0 - p)
    inv = np.float32(2.0**-32)

    def kept(r: int) -> bool:
        return bool(inv + np.float32(r) * inv < p1m)

    if not kept(0):
        thr = 0
    else:
        lo, hi = 0, 2**32  # kept(lo); hi dropped (2**32: sentinel)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if kept(mid):
                lo = mid
            else:
                hi = mid
        thr = hi
    assert thr < 2**32
    scale = float(np.float32(1.0 / float(p1m)))
    _DROPOUT_KEEP_PARAMS[p] = (thr, scale)
    return thr, scale


_CVT_SELF_TEST: Dict[Tuple[int, float], bool] = {}


def _bf16_cvt_self_test(device: torch.device, scale: float) -> bool:
    """One-time (per device and scale) check that the fused kernel's fp32 ->
    bf16 epilogue (_f32_to_bf16_bits_hip) equals this build's own cast
    (c10::BFloat16(float), which aten's dropout kernel uses) on the whole
    reachable domain: float(x) * scale, float(x) * 0.0 and float(x) for all
    65536 bf16 bit patterns x (NaN payloads, +-Inf, -0, subnormals, ties).
    A mismatch (another arch / HIP bf16 implementation) disables the fusion."""
    key = (device.index if device.index is not None else torch.cuda.current_device(), scale)
    hit = _CVT_SELF_TEST.get(key)
    if hit is None:
        x = (
            torch.arange(-32768, 32768, dtype=torch.int32, device=device)
            .to(torch.int16)
            .view(torch.bfloat16)
            .float()
        )
        f = torch.cat([x * scale, x * 0.0, x])
        ref = f.to(torch.bfloat16).view(torch.int16)
        out = torch.empty_like(ref)
        _bf16_cvt_probe_kernel[(triton.cdiv(f.numel(), 1024),)](
            f, out, f.numel(), BLOCK=1024
        )
        hit = bool(torch.equal(out, ref))
        _CVT_SELF_TEST[key] = hit
        if not hit:
            _logger.warning(
                "posenc+input-dropout fusion DISABLED: this build's fp32 -> bf16 "
                "cast differs from the fused epilogue (DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE)"
            )
    return hit


_HW_CVT_SELF_TEST: Dict[Tuple[int, float], str] = {}


def _bf16_hw_cvt_self_test(device: torch.device, scale: float) -> str:
    """DLRM_HSTU_FP8_POSENC_DROP_HWEPI: one-time (per device and scale) check
    that HW_EPI's y epilogue, f.to(tl.bfloat16) in a Triton kernel, equals this
    build's own cast (c10::BFloat16(float), aten's) on the reachable domain,
    as _bf16_cvt_self_test does for the software epilogue: float(x) * scale,
    float(x) * 0.0 and float(x) for all 65536 bf16 bit patterns x. Returns ""
    or the mismatch."""
    key = (device.index if device.index is not None else torch.cuda.current_device(), scale)
    hit = _HW_CVT_SELF_TEST.get(key)
    if hit is None:
        x = (
            torch.arange(-32768, 32768, dtype=torch.int32, device=device)
            .to(torch.int16)
            .view(torch.bfloat16)
            .float()
        )
        f = torch.cat([x * scale, x * 0.0, x])
        ref = f.to(torch.bfloat16).view(torch.int16)
        out = torch.empty_like(ref)
        _bf16_hw_cvt_probe_kernel[(triton.cdiv(f.numel(), 1024),)](
            f, out, f.numel(), BLOCK=1024
        )
        bad = int((out != ref).sum().item())
        hit = (
            ""
            if bad == 0
            else f"f.to(bf16) differs from torch's cast in {bad} of {f.numel()} "
            "epilogue values"
        )
        _HW_CVT_SELF_TEST[key] = hit
    return hit


def _as_i32(v: int) -> int:
    """uint32 bit pattern as a Python int in the int32 range (stable i32
    kernel-argument type)."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _posenc_concat_dropout_launch(
    v2: bool,
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    out: torch.Tensor,
    mask: torch.Tensor,
    pos_inds: torch.Tensor,
    ts_inds: torch.Tensor,
    src_inds: torch.Tensor,
    seed: int,
    ctr: int,
    T: int,
    dropout_p: float,
    hw_epi: bool = False,
) -> None:
    """One launch of the fused posenc concat + input-dropout kernel: design A
    (_posenc_concat_dropout_kernel) or the V2 layout, with an explicit Philox
    key (seed) and 64-bit counter base (ctr = generator offset // 4, counter
    words c0/c1) and aten thread count T. Contiguous inputs (as after
    switch_to_contiguous_if_needed). mask: bool [N, D], or uint8 [N, D / 8]
    for packed bits (DLRM_HSTU_FP8_DROPOUT_MASK_BITS). hw_epi: V2's gfx950
    HW_EPI compile (DLRM_HSTU_FP8_POSENC_DROP_HWEPI; the caller checks the
    arch and the self-test)."""
    B = seq_lengths.shape[0]
    N_left, D = values_left.shape
    keep_below, scale = _dropout_keep_params(dropout_p)
    log2_t = T.bit_length() - 1 if T & (T - 1) == 0 else -1
    # MASK_BITS is passed only for the packed mask (the bool launch is unchanged)
    bits_kwargs = {}
    if mask.dtype == torch.uint8:
        assert mask.shape == (out.shape[0], D // 8) and mask.is_contiguous()
        bits_kwargs["MASK_BITS"] = True
    # HW_EPI likewise only when on: without it V2 compiles exactly as before
    if hw_epi:
        assert v2, "HW_EPI is a V2 compile"
        bits_kwargs["HW_EPI"] = True
    if v2:
        kernel = _posenc_concat_dropout_v2_kernel
        BLOCK_N = _POSENC_DROP_V2_BLOCK_N
        num_warps = _POSENC_DROP_V2_WARPS
        num_stages = _POSENC_DROP_V2_STAGES
        block_d = _POSENC_DROP_V2_BLOCK_D
    else:
        kernel = _posenc_concat_dropout_kernel
        BLOCK_N = _POSENC_DROP_BLOCK_N
        num_warps = _POSENC_DROP_WARPS
        num_stages = _POSENC_DROP_STAGES
        block_d = _POSENC_DROP_BLOCK_D
    grid = (B, triton.cdiv(max_seq_len, BLOCK_N))
    # any power of two >= 8 is exact (columns are processed in 8-aligned
    # chunks; tiling does not change any per-element op)
    BLOCK_D = min(
        triton.next_power_of_2(D),
        triton.next_power_of_2(max(8, block_d)),
    )
    kernel[grid](
        Left=values_left,
        Right=values_right,
        Offsets=seq_offsets,
        Lengths=seq_lengths,
        PosEmb=pos_embeddings,
        TsEmb=ts_embeddings,
        Y=out.view(torch.int16),
        Mask=mask.view(torch.int8),
        TS=timestamps,
        PosInds=pos_inds,
        TsInds=ts_inds,
        SrcInds=src_inds,
        NumTargets=num_targets,
        D=D,
        num_time_buckets=ts_embeddings.shape[0] - 1,
        time_bucket_increments=60.0,
        time_bucket_scale=1.0,
        time_delta=0,
        max_contextual_seq_len=max_contextual_seq_len,
        max_pos_ind=pos_embeddings.shape[0],
        ctx_len=N_left // B,
        stride_ln=values_left.stride(0),
        stride_rn=values_right.stride(0),
        stride_pn=pos_embeddings.stride(0),
        stride_tn=ts_embeddings.stride(0),
        stride_on=out.stride(0),
        alpha=float(alpha),
        scale=scale,
        seed_lo=_as_i32(seed),
        seed_hi=_as_i32(seed >> 32),
        ctr_lo=_as_i32(ctr),
        ctr_hi=_as_i32(ctr >> 32),
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        INTERLEAVE_TARGETS=interleave_targets,
        TIME_BUCKET_FN=time_bucket_fn,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        T=T,
        LOG2_T=log2_t,
        KEEP_BELOW=keep_below,
        num_warps=num_warps,
        num_stages=num_stages,
        **bits_kwargs,
    )


# DLRM_HSTU_FP8_POSENC_DROPOUT_V2 fail-closed state: key -> V2 usable
_POSENC_DROP_V2_OK: Dict[Tuple, bool] = {}


def _posenc_drop_v2_key(
    values_right: torch.Tensor,
    ctx_len: int,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    dropout_p: float,
    *,
    mask_bits: bool,
) -> Tuple:
    # mask_bits: the forward's effective mask layout (packed or bool)
    return (
        torch.cuda.current_device(),
        values_right.shape[1],
        ctx_len,
        pos_embeddings.dtype,
        ts_embeddings.dtype,
        timestamps.dtype,
        num_targets is not None,
        bool(interleave_targets),
        time_bucket_fn,
        dropout_p,
        bool(mask_bits),
    )


def _posenc_drop_v2_disable(key: Tuple, why: str) -> None:
    _POSENC_DROP_V2_OK[key] = False
    _logger.warning(
        "posenc+input-dropout fusion V2 layout DISABLED for %s, falling back to "
        "design A (_posenc_concat_dropout_kernel; DLRM_HSTU_FP8_POSENC_DROPOUT_V2): %s",
        key,
        why[:800],
    )


def _raw_bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(
        {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            t.element_size()
        ]
    )


def _posenc_drop_v2_self_test(
    D: int,
    ctx_len: int,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: float,
    max_contextual_seq_len: int,
    dropout_p: float,
    *,
    mask_bits: bool,
    hw_epi: bool = False,
) -> str:
    """Bitwise self-test of the V2 layout on a small synthetic batch built with
    the caller's D, ctx_len, tables, dtypes and kernel constexprs (N * D > 2^19
    saturates T as at the production shape, so all constexprs match; when the
    production Right / Y storage exceeds 2 GiB, Triton's AMD buffer-op pointer
    specialization of those two arguments differs, the Philox word mapping IR
    does not, and a compile / launch error there is caught by the forward's
    launch fallback). Right lengths 0..47 cover every tail residue mod 8 / 16.
    Checks, raw bits: V2 == design A for y, mask, pos/ts/src indices at the
    current generator state and at a counter base whose low word carries
    (ctr_lo = 2^32 - 1, seed >= 2^63); V2's mask == aten::native_dropout's
    mask at the same generator state; _aten_dropout_rng's offset advance ==
    aten's; no V2 poison left. mask_bits: the mask layout under test (the
    forward's effective one): the packed layout of
    DLRM_HSTU_FP8_DROPOUT_MASK_BITS (both kernels write it; aten's mask is
    packed with pack_dropout_mask_bits; packed bytes take every value, so the
    poison check is a second V2 run with the complementary poison, which must
    give the same bytes) or the bool layout. Returns "" when all hold, else
    what failed. hw_epi (DLRM_HSTU_FP8_POSENC_DROP_HWEPI): the V2 binary under
    test is the HW_EPI compile, and the synthetic seq rows start with all
    65536 bf16 bit patterns (NaN payloads, +-Inf, -0, subnormals) so its
    hardware epilogue is compared with design A's on every special value.
    Consumes the default CUDA generator: the caller restores its offset.
    Synthetic data come from a private generator."""
    dev = pos_embeddings.device
    g = torch.Generator(device=dev)
    g.manual_seed(0x5EEDF00D)
    right = list(range(48))
    n = sum(right) + ctx_len * len(right)
    need = -(-(2**19 + 4096) // D)
    if n < need:
        right.append(need - n)
    B = len(right)
    lengths = [r + ctx_len for r in right]
    N = sum(lengths)
    values_left = (
        torch.randn((B * ctx_len, D), device=dev, generator=g) * 0.05
    ).to(torch.bfloat16)
    values_right = (
        torch.randn((N - B * ctx_len, D), device=dev, generator=g) * 0.05
    ).to(torch.bfloat16)
    if hw_epi:
        flat = values_right.view(torch.int16).view(-1)
        k = min(flat.numel(), 65536)
        flat[:k] = torch.arange(-32768, -32768 + k, device=dev, dtype=torch.int32).to(
            torch.int16
        )
    lens_t = torch.tensor(lengths, device=dev, dtype=seq_lengths.dtype)
    offs_t = torch.zeros((B + 1,), device=dev, dtype=seq_offsets.dtype)
    offs_t[1:] = torch.cumsum(lens_t, 0)
    ts_t = torch.cumsum(
        torch.randint(0, 200000, (N,), device=dev, generator=g), 0
    ).to(timestamps.dtype)
    nt_t = None
    if num_targets is not None:
        nt_t = torch.tensor(
            [min(r, 1 + i % 3) for i, r in enumerate(right)],
            device=dev,
            dtype=num_targets.dtype,
        )
    idx_dtype = _posenc_index_dtype(
        pos_embeddings.shape[0], max_contextual_seq_len, ts_embeddings.shape[0]
    )
    bits = bool(mask_bits)

    def run(
        v2: bool, seed: int, ctr: int, T: int, poison: int = 0x5A, ypoison: int = 0x7FC1
    ) -> List[torch.Tensor]:
        y = torch.full((N, D), ypoison, device=dev, dtype=torch.int16)
        m = torch.full(
            (N, D // 8 if bits else D), poison, device=dev, dtype=torch.uint8
        )
        p = torch.full((N,), -7, device=dev, dtype=idx_dtype)
        t = torch.full((N,), -7, device=dev, dtype=idx_dtype)
        s = torch.full((N,), -7, device=dev, dtype=torch.int32)
        _posenc_concat_dropout_launch(
            v2,
            values_left,
            values_right,
            offs_t,
            lens_t,
            pos_embeddings,
            ts_embeddings,
            ts_t,
            nt_t,
            interleave_targets,
            time_bucket_fn,
            alpha,
            max(lengths),
            max_contextual_seq_len,
            y.view(torch.bfloat16),
            m if bits else m.view(torch.bool),
            p,
            t,
            s,
            seed,
            ctr,
            T,
            dropout_p,
            hw_epi=v2 and hw_epi,
        )
        return [y, m, p, t, s]

    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    off0 = gen.get_offset()
    _, m_ref = torch.ops.aten.native_dropout(
        torch.zeros((N, D), device=dev, dtype=torch.bfloat16), dropout_p, True
    )
    off_aten = gen.get_offset()
    gen.set_offset(off0)
    seed, offset, T = _aten_dropout_rng(N * D)
    if offset != off0 or gen.get_offset() != off_aten:
        return (
            f"generator replica {offset}->{gen.get_offset()} vs aten "
            f"{off0}->{off_aten}"
        )
    names = []
    counts = []
    v2n = "V2 hw" if hw_epi else "V2"
    for tag, sd, ctr in (
        ("gen", seed, offset // 4),
        ("carry", 0x9E3779B97F4A7C15, (5 << 32) | 0xFFFFFFFF),
    ):
        outs_a = run(False, sd, ctr, T)
        outs_b = run(True, sd, ctr, T)
        for nm, a, b in zip(("y", "mask", "pos", "ts", "src"), outs_a, outs_b):
            names.append(f"{tag}:{nm} {v2n} vs A")
            counts.append((a != b).sum())
        if hw_epi:
            # y may hold any bit pattern here: a second run with other y and
            # mask poisons must give the same y and mask
            outs_c = run(True, sd, ctr, T, poison=0xA5, ypoison=0x1357)
            names.append(f"{tag}:{v2n} y poison 0x7FC1 vs 0x1357")
            counts.append((outs_c[0] != outs_b[0]).sum())
            names.append(f"{tag}:{v2n} mask poison 0x5A vs 0xA5")
            counts.append((outs_c[1] != outs_b[1]).sum())
        else:
            names.append(f"{tag}:V2 y poison")
            counts.append((outs_b[0] == 0x7FC1).sum())
            if bits:
                names.append(f"{tag}:V2 mask poison 0x5A vs 0xA5")
                counts.append(
                    (run(True, sd, ctr, T, poison=0xA5)[1] != outs_b[1]).sum()
                )
        if not bits:
            names.append(f"{tag}:{v2n} mask not 0/1")
            counts.append((outs_b[1] > 1).sum())
        if tag == "gen":
            names.append(f"gen:mask {v2n} vs aten::native_dropout")
            ref = pack_dropout_mask_bits(m_ref) if bits else _raw_bits(m_ref)
            counts.append((outs_b[1] != ref).sum())
    bad = [
        f"{nm}: {c}" for nm, c in zip(names, torch.stack(counts).tolist()) if c != 0
    ]
    return "; ".join(bad)


def _posenc_drop_v2_verified(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: float,
    max_contextual_seq_len: int,
    dropout_p: float,
    *,
    mask_bits: bool,
) -> bool:
    """DLRM_HSTU_FP8_POSENC_DROPOUT_V2 fail-closed gate: runs
    _posenc_drop_v2_self_test once per key (device, D, ctx_len, dtypes,
    kernel constexprs, p, mask layout) and caches the verdict. A failed check
    or any exception (e.g. a V2 compile error) disables V2 for that key with a
    WARNING (design A is launched instead). The default CUDA generator's offset
    is restored afterwards, so the self-test is invisible to the RNG stream.
    mask_bits: the forward's effective mask layout (self-tested as such)."""
    ctx_len = values_left.shape[0] // seq_lengths.shape[0]
    key = _posenc_drop_v2_key(
        values_right,
        ctx_len,
        pos_embeddings,
        ts_embeddings,
        timestamps,
        num_targets,
        interleave_targets,
        time_bucket_fn,
        dropout_p,
        mask_bits=mask_bits,
    )
    hit = _POSENC_DROP_V2_OK.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        return False
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    off0 = gen.get_offset()
    try:
        why = _posenc_drop_v2_self_test(
            values_right.shape[1],
            ctx_len,
            seq_offsets,
            seq_lengths,
            pos_embeddings,
            ts_embeddings,
            timestamps,
            num_targets,
            interleave_targets,
            time_bucket_fn,
            alpha,
            max_contextual_seq_len,
            dropout_p,
            mask_bits=mask_bits,
        )
    except Exception as e:
        why = f"self-test raised {type(e).__name__}: {e}"
    finally:
        gen.set_offset(off0)
    if why:
        _posenc_drop_v2_disable(key, "self-test failed: " + why)
        return False
    _POSENC_DROP_V2_OK[key] = True
    return True


# DLRM_HSTU_FP8_POSENC_DROP_HWEPI fail-closed state: V2 key -> HW_EPI usable
_POSENC_DROP_HWEPI_OK: Dict[Tuple, bool] = {}
_GCN_ARCH: Dict[int, str] = {}


def _gcn_arch(index: int) -> str:
    """gcnArchName of a CUDA device ("gfx950:sramecc+:xnack-"; "" when the
    property is absent), cached."""
    hit = _GCN_ARCH.get(index)
    if hit is None:
        props = torch.cuda.get_device_properties(index)
        hit = str(getattr(props, "gcnArchName", "") or "")
        _GCN_ARCH[index] = hit
    return hit


def _posenc_drop_hwepi_disable(key: Tuple, why: str) -> None:
    _POSENC_DROP_HWEPI_OK[key] = False
    _logger.warning(
        "posenc dropout hw epilogue DISABLED for %s, falling back to the V2 "
        "software epilogue (_posenc_concat_dropout_v2_kernel without HW_EPI; "
        "DLRM_HSTU_FP8_POSENC_DROP_HWEPI): %s",
        key,
        why[:800],
    )


def _posenc_drop_hwepi_verified(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: float,
    max_contextual_seq_len: int,
    dropout_p: float,
    *,
    mask_bits: bool,
) -> bool:
    """DLRM_HSTU_FP8_POSENC_DROP_HWEPI fail-closed gate, asked only for a V2
    launch whose own self-test passed (same key as _posenc_drop_v2_verified).
    False unless the knob is on and the device is gfx950 (another arch keeps
    the V2 binary, one WARNING). Then once per key: _bf16_hw_cvt_self_test,
    and _posenc_drop_v2_self_test with the HW_EPI binary (vs design A and
    aten::native_dropout, both RNG states, all 65536 bf16 bit patterns); the
    verdict is cached. A failed check or any exception (e.g. a compile error)
    disables HW_EPI for that key with one WARNING (V2's software epilogue is
    launched instead). The default CUDA generator's offset is restored."""
    if not _POSENC_DROP_HWEPI:
        return False
    dev = values_right.device
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    arch = _gcn_arch(index)
    if not arch.startswith("gfx950"):
        _warn_once(
            "posenc_drop_hwepi_arch:" + arch,
            "DLRM_HSTU_FP8_POSENC_DROP_HWEPI set but NOT armed: arch "
            f"{arch or 'unknown'} (v_cvt_pk_bf16_f32 / v_bitop3_b32 are "
            "gfx950-only); the V2 kernel keeps its software epilogue",
        )
        return False
    ctx_len = values_left.shape[0] // seq_lengths.shape[0]
    key = _posenc_drop_v2_key(
        values_right,
        ctx_len,
        pos_embeddings,
        ts_embeddings,
        timestamps,
        num_targets,
        interleave_targets,
        time_bucket_fn,
        dropout_p,
        mask_bits=mask_bits,
    )
    hit = _POSENC_DROP_HWEPI_OK.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        return False
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    off0 = gen.get_offset()
    try:
        why = _bf16_hw_cvt_self_test(dev, _dropout_keep_params(dropout_p)[1])
        if not why:
            why = _posenc_drop_v2_self_test(
                values_right.shape[1],
                ctx_len,
                seq_offsets,
                seq_lengths,
                pos_embeddings,
                ts_embeddings,
                timestamps,
                num_targets,
                interleave_targets,
                time_bucket_fn,
                alpha,
                max_contextual_seq_len,
                dropout_p,
                mask_bits=mask_bits,
                hw_epi=True,
            )
    except Exception as e:
        why = f"self-test raised {type(e).__name__}: {e}"
    finally:
        gen.set_offset(off0)
    if why:
        _posenc_drop_hwepi_disable(key, "self-test failed: " + why)
        return False
    _POSENC_DROP_HWEPI_OK[key] = True
    return True


def posenc_dropout_fuse_supported(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    dropout_p: float,
    training: bool,
) -> bool:
    """Gate of DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE (caller: the concat-fold path).
    Anything else keeps the concat kernel + the separate input dropout."""
    if not (_POSENC_DROPOUT_FUSE and training and 0.0 < dropout_p < 1.0):
        return False
    if torch.jit.is_scripting() or is_fx_tracing() or not torch.is_grad_enabled():
        return False
    if values_left.dim() != 2 or values_right.dim() != 2:
        return False
    if not (values_left.is_cuda and values_right.is_cuda):
        return False
    if values_left.dtype != torch.bfloat16 or values_right.dtype != torch.bfloat16:
        return False
    D = values_right.shape[1]
    N = values_left.shape[0] + values_right.shape[0]
    if D % 8 != 0 or N == 0 or N * D > 2**31 - 1:
        return False
    if not (
        values_left.requires_grad
        or values_right.requires_grad
        or pos_embeddings.requires_grad
        or ts_embeddings.requires_grad
    ):
        return False
    if values_right.device.index != torch.cuda.current_device():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return _bf16_cvt_self_test(values_right.device, _dropout_keep_params(dropout_p)[1])


class _AddTimestampPositionEmbeddingsConcatDropoutFunction(torch.autograd.Function):
    """DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE: _AddTimestampPositionEmbeddingsConcatFunction
    followed by the HSTU input dropout (aten::native_dropout(x, p, True) via
    F.dropout, or _InputDropoutHandoffFunction when the mask/scale fold is
    armed) as one autograd node. Output bits, bool mask (bit-packed under
    DLRM_HSTU_FP8_DROPOUT_MASK_BITS where the layer-0 LN fold consumes it, see
    _dropout_mask_bits_why_not), CUDA generator offset and gradients are those
    of the pair (see _posenc_concat_dropout_kernel)."""

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values_left: torch.Tensor,
        values_right: torch.Tensor,
        seq_offsets: torch.Tensor,
        pos_embeddings: torch.Tensor,
        ts_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        max_seq_len: int,
        max_contextual_seq_len: int,
        seq_lengths: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        interleave_targets: bool,
        time_bucket_fn: str,
        alpha: float,
        dropout_p: float,
        attach_handoff: bool,
    ):
        values_left = switch_to_contiguous_if_needed(values_left)
        values_right = switch_to_contiguous_if_needed(values_right)
        pos_embeddings = switch_to_contiguous_if_needed(pos_embeddings)
        ts_embeddings = switch_to_contiguous_if_needed(ts_embeddings)
        timestamps = switch_to_contiguous_if_needed(timestamps)

        max_pos_ind = pos_embeddings.shape[0]
        B = seq_lengths.shape[0]
        N_left, D = values_left.shape
        N_right = values_right.shape[0]
        N = N_left + N_right
        ctx_len = N_left // B
        assert ctx_len * B == N_left
        assert values_right.shape[1] == D and D % 8 == 0
        assert pos_embeddings.shape[1] == D and ts_embeddings.shape[1] == D
        assert values_right.dtype == torch.bfloat16
        assert N * D <= 2**31 - 1
        # the dropout output (aten: empty_like(x)) and its bool mask, or the
        # mask as packed bits (DLRM_HSTU_FP8_DROPOUT_MASK_BITS, only where the
        # layer-0 LN-backward fold will consume them)
        out = torch.empty((N, D), device=values_right.device, dtype=values_right.dtype)
        bits = False
        if _DROPOUT_MASK_BITS:
            why_not = _dropout_mask_bits_why_not(
                attach_handoff, D, values_right.device
            )
            bits = not why_not
            if bits:
                _log_once(
                    "mask_bits",
                    "DLRMV4 armed OK: input-dropout mask bit-packed, uint8 "
                    "[N, D / 8] (DLRM_HSTU_FP8_DROPOUT_MASK_BITS; packed LN-backward "
                    "fold self-test passed; kill switch =0)",
                )
            else:
                _warn_once(
                    "mask_bits_off:" + why_not,
                    "DLRM_HSTU_FP8_DROPOUT_MASK_BITS set but NOT armed, bool "
                    "[N, D] mask kept: " + why_not,
                )
        if bits:
            mask = torch.empty(
                (N, D // 8), device=values_right.device, dtype=torch.uint8
            )
        else:
            mask = torch.empty((N, D), device=values_right.device, dtype=torch.bool)
        ts_emb_size = ts_embeddings.shape[0]
        idx_dtype = _posenc_index_dtype(
            max_pos_ind, max_contextual_seq_len, ts_emb_size
        )
        ts_inds = torch.empty((N,), device=out.device, dtype=idx_dtype)
        pos_inds = torch.empty((N,), device=out.device, dtype=idx_dtype)
        src_inds = torch.empty((N,), device=out.device, dtype=torch.int32)

        # V2 only after its one-time (cached) RNG-neutral self-test passed
        v2 = _POSENC_DROPOUT_V2 and _posenc_drop_v2_verified(
            values_left,
            values_right,
            seq_offsets,
            seq_lengths,
            pos_embeddings,
            ts_embeddings,
            timestamps,
            num_targets,
            interleave_targets,
            time_bucket_fn,
            alpha,
            max_contextual_seq_len,
            float(dropout_p),
            mask_bits=bits,
        )
        # V2's gfx950 hardware epilogue (DLRM_HSTU_FP8_POSENC_DROP_HWEPI), only
        # after its own one-time (cached) RNG-neutral self-test passed
        hw = (
            v2
            and _POSENC_DROP_HWEPI
            and _posenc_drop_hwepi_verified(
                values_left,
                values_right,
                seq_offsets,
                seq_lengths,
                pos_embeddings,
                ts_embeddings,
                timestamps,
                num_targets,
                interleave_targets,
                time_bucket_fn,
                alpha,
                max_contextual_seq_len,
                float(dropout_p),
                mask_bits=bits,
            )
        )
        seed, offset, T = _aten_dropout_rng(N * D)
        launch_args = (
            values_left,
            values_right,
            seq_offsets,
            seq_lengths,
            pos_embeddings,
            ts_embeddings,
            timestamps,
            num_targets,
            interleave_targets,
            time_bucket_fn,
            alpha,
            max_seq_len,
            max_contextual_seq_len,
            out,
            mask,
            pos_inds,
            ts_inds,
            src_inds,
            seed,
            offset // 4,
            T,
            float(dropout_p),
        )
        if hw:
            try:
                _posenc_concat_dropout_launch(True, *launch_args, hw_epi=True)
            except Exception as e:  # fail closed: V2's software epilogue
                _posenc_drop_hwepi_disable(
                    _posenc_drop_v2_key(
                        values_right,
                        ctx_len,
                        pos_embeddings,
                        ts_embeddings,
                        timestamps,
                        num_targets,
                        interleave_targets,
                        time_bucket_fn,
                        float(dropout_p),
                        mask_bits=bits,
                    ),
                    f"launch raised {type(e).__name__}: {e}",
                )
                hw = False
        if v2 and not hw:
            try:
                _posenc_concat_dropout_launch(True, *launch_args)
            except Exception as e:  # fail closed: design A, same RNG reservation
                _posenc_drop_v2_disable(
                    _posenc_drop_v2_key(
                        values_right,
                        ctx_len,
                        pos_embeddings,
                        ts_embeddings,
                        timestamps,
                        num_targets,
                        interleave_targets,
                        time_bucket_fn,
                        float(dropout_p),
                        mask_bits=bits,
                    ),
                    f"launch raised {type(e).__name__}: {e}",
                )
                v2 = False
        if v2:
            _log_once(
                "posenc_drop_v2",
                "armed OK: posenc+input-dropout fusion V2 layout ACTIVE "
                "(_posenc_concat_dropout_v2_kernel, DLRM_HSTU_FP8_POSENC_DROPOUT_V2; "
                "first-use self-test vs design A and aten::native_dropout passed)",
            )
            if hw:
                _log_once(
                    "posenc_drop_hwepi",
                    "DLRMV4 armed OK: posenc dropout hw epilogue "
                    "(v_cvt_pk_bf16_f32 + v_bitop3 philox) ACTIVE",
                )
        else:
            _posenc_concat_dropout_launch(False, *launch_args)
        (
            sorted_ts_key_inds,
            sorted_ts_value_inds,
            sorted_pos_key_inds,
            sorted_pos_value_inds,
        ) = _posenc_sort_pairs(
            ts_inds,
            pos_inds,
            N,
            timestamps.device,
            pos_meta=(
                seq_offsets,
                seq_lengths,
                num_targets,
                interleave_targets,
                max_contextual_seq_len,
                max_pos_ind,
            ),
            ts_max_key=ts_emb_size - 1,
        )
        saved = [
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
            src_inds,
        ]
        if attach_handoff:
            # == _InputDropoutHandoffFunction.forward's handoff
            handoff = DropoutHandoff(mask, 1.0 / (1.0 - dropout_p), (N, D))
            ctx.handoff = handoff
            setattr(out, DROPOUT_HANDOFF_ATTR, handoff)
        else:
            ctx.handoff = None
            saved.append(mask)
        ctx.save_for_backward(*saved)
        # native_dropout's autograd scale (double)
        ctx.dropout_scale = 1.0 / (1.0 - dropout_p)
        ctx.D = D
        ctx.N_left = N_left
        ctx.N_right = N_right
        ctx.pos_emb_size = pos_embeddings.shape[0]
        ctx.ts_emb_size = ts_emb_size
        ctx.pos_dtype = pos_embeddings.dtype
        ctx.ts_dtype = ts_embeddings.dtype
        ctx.alpha = float(alpha)
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, d_out: torch.Tensor):
        saved = ctx.saved_tensors
        if ctx.handoff is not None:
            d_x = dropout_handoff_backward(ctx.handoff, d_out)
        else:
            d_x = torch.ops.aten.native_dropout_backward(
                d_out, saved[5], ctx.dropout_scale
            )
        d_left, d_right, d_pos_embeddings, d_ts_embeddings = _posenc_concat_backward(
            ctx, d_x, *saved[:5]
        )
        return (
            d_left,
            d_right,
            None,
            d_pos_embeddings,
            d_ts_embeddings,
            None,
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


@torch.jit.unused
@torch.fx.wrap
def triton_concat_add_timestamp_positional_embeddings_dropout(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: float,
    dropout_p: float,
    attach_dropout_handoff: bool,
) -> torch.Tensor:
    return _AddTimestampPositionEmbeddingsConcatDropoutFunction.apply(
        values_left,
        values_right,
        seq_offsets,
        pos_embeddings,
        ts_embeddings,
        timestamps,
        max_seq_len,
        max_contextual_seq_len,
        seq_lengths,
        num_targets,
        interleave_targets,
        time_bucket_fn,
        alpha,
        dropout_p,
        attach_dropout_handoff,
    )


@torch.jit.unused
@torch.fx.wrap
def triton_concat_add_timestamp_positional_embeddings(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: float,
) -> torch.Tensor:
    # DLRM_HSTU_FP8_EVAL_FWD_HYGIENE: decided here, not in the Function's
    # forward (grad mode is off in there). Training forwards never read it.
    need_bwd = backward_possible(
        values_left,
        values_right,
        seq_offsets,
        pos_embeddings,
        ts_embeddings,
        timestamps,
        seq_lengths,
        num_targets,
    ) or not eval_fwd_hygiene_enabled()
    return _AddTimestampPositionEmbeddingsConcatFunction.apply(
        values_left,
        values_right,
        seq_offsets,
        pos_embeddings,
        ts_embeddings,
        timestamps,
        max_seq_len,
        max_contextual_seq_len,
        seq_lengths,
        num_targets,
        interleave_targets,
        time_bucket_fn,
        alpha,
        need_bwd,
    )


@torch.jit.unused
@torch.fx.wrap
def triton_add_timestamp_positional_embeddings(
    seq_embeddings: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    # DLRM_HSTU_FP8_EVAL_FWD_HYGIENE: see triton_concat_add_timestamp_positional_embeddings.
    need_bwd = backward_possible(
        seq_embeddings,
        seq_offsets,
        pos_embeddings,
        ts_embeddings,
        timestamps,
        seq_lengths,
        num_targets,
    ) or not eval_fwd_hygiene_enabled()
    return _AddTimestampPositionEmbeddingsFunction.apply(
        seq_embeddings,
        seq_offsets,
        pos_embeddings,
        ts_embeddings,
        timestamps,
        max_seq_len,
        max_contextual_seq_len,
        seq_lengths,
        num_targets,
        interleave_targets,
        time_bucket_fn,
        alpha,
        need_bwd,
    )
