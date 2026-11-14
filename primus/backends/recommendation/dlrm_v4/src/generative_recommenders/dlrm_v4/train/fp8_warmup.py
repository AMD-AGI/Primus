# pyre-strict
"""FP8 first-use warmup before RUN_START ($FP8_WARMUP, default 0 = off).

WHY. With the FP8 arm on, the first step after the scale freeze (the "FP8
switch", ~ts18 in the streaming loop) is the first time the e4m3 kernels run:
the FlyDSL e4m3 attention forward and full-FP8 backward (JIT + compile), the
Triton targets-only delta attention with FP8_INPUT=True, the FP8 LayerNorm
variants, the dq-split / block-dO Triton kernels and the bwd_fp8 casts. Each
pays a one-time compile inside the timed region. This module pays it during
initialization, by calling the REAL production wrappers once on synthetic,
zero-filled tensors of the production shape class (B, H, D, row strides, N
class, int dtypes), so every env-dependent config (FAST_INTERIOR, dq-split,
PIPE, EPILOGUE, QK_MFMA, FULL, gorder, SHIPPED dicts) is resolved by production
code rather than mirrored here.

RULES. MLPerf training_rules.adoc L545-549: model/library initialization
(including JIT/compilation on synthetic inputs) before RUN_START is untimed, and
the clock starts before touching the dataset; this touches no dataset sample.
L290-291: the state at RUN_START must equal a fresh initialization; no module
forward, autograd Function, parameter value, buffer, optimizer, FP8 scale or
calibration state is read or written, the RNG is checked (and restored), and
every once-latch / counter / workspace the wrappers mutate is snapshotted and
restored in `finally` (design: probes/iter12/fp8_warmup_design.md section 3).

NEVER CALLED: try_flydsl_* (counters, _max_num_targets memo), any autograd
Function, bwd_fp8.quantize_gradient / output_projection_dgrad (site state),
calibrate_scale_buffer (all-reduce), the *_preq scaled-GEMM wrappers (weight /
scale caches), triton_fp8._get_weight_shadow (the weight-shadow cache),
install_for_tokens, seed_max_num_targets, module forwards, the M-E1 helpers of
triton_hstu_preprocess_and_attention (_dkdv_e5m2_arm / _engaged / _prequant /
_projection_backward: E1 counters, log sets, latch) and
bwd_fp8.producer_fp8t_inv_scale / producer_inv_scale (scale epochs).

Knobs (all read here, at call time):
  FP8_WARMUP=0|1|2          0 off, 1 FP8-switch tier, 2 also the bf16
                            calibration-window tier (gin env_str).
  FP8_WARMUP_SKIP=a,b       skip items by tag (F1, T2, W1, ...) or name.
  FP8_WARMUP_STORES=small|both
                            default small. "both" also runs the Triton items on
                            views carved from a >2 GiB arena (the >2 GiB storage
                            class, see STORAGE CLASS below).
  FP8_WARMUP_VERIFY=1       sha256 of the model's dense (non-embedding,
                            numel < 64M) params/buffers before vs after, plus a
                            digest of the FP8 modules' globals before vs after.
  FP8_WARMUP_QUIET=1        silence fd 2 (Triton UserWarnings) during items.
  FP8_WARMUP_N11=0          do not warm the c=11 N class (N=2000) in F2 (default on).
  DLRM_HSTU_FP8_WARMUP_F2E=0
                            skip item F2e (absent or "1" = run it; gin binds
                            it through apply_env_bootstrap).
  DLRM_HSTU_FP8_WARMUP_FIN_RPP=0
                            skip item F2r (absent or "1" = run it; gin binds
                            it through apply_env_bootstrap).

MEMORY. Every tensor the items allocate is released before this returns (torch
reserved memory is back at its pre-warmup value after empty_cache()). The items
also leave about 6.2 GiB of NON-torch device memory behind (every rank, 1n and
2n). empty_cache() and 10 s of polling do not return it, but about 6.0 GiB of
it is free again by the end of the allocator pre-grow's first pass (trigger not
isolated: its hipMemCreate/Map calls or its synchronize). That pass sizes the
main pool from mem_get_info() before the release, so it would map a pool about
6 GiB smaller than in a run without the warmup; alloc_pregrow.py's converge pass
maps what came back (no-op when nothing did), so the pool matches a run without
the warmup. The "[fp8-warmup] memory" line logs the torch / non-torch split.

SHIP-11 PORT (ship-12 gin item (35)). The 595b51e version predates ship-11
items (24)-(30), and the ship-11 FP8 switch first-uses more kernels. Covered:
  * through the existing items, because the production wrappers they call now
    dispatch to the ship-11 kernels: store_t_v2 (L3t), vkp_v2 and its
    longest-first permutation (T2e), colsum_v2 (Q3), LN*U*dropout v3 / maskgen
    (L4), the FlyDSL FWD_PIPE forward config (F1), the item (27) bwd_dkdv /
    dq-split tokens (VALU nobr, ws store l128t, load opts rsk; F2), the BM16
    delta forward (T1, T1b);
  * new items: W1 the fused weight-shadow refresh (triton_fp8_wshadow kernels
    on synthetic weights of the production shapes and layouts, one at a time
    and batched), H1 the fused backward helpers (fp8_helpers_fused site and
    weight-cast kernels on a throwaway _SiteScale, for the output-projection,
    uvqk and vk sites), F2q the dq_e5m2 sink of the full-FP8 FlyDSL backward
    (hstu_dq_e5m2 kernels, then the u|v|k-only cast), U1 the FWD_U_DP
    hipBLASLt extension at a production-class row count, and, tier 2, L4b the
    bf16-output LN*U*dropout forward of the calibration steps.
STORAGE CLASS. The Triton AMD backend specializes every pointer argument on
its storage size (amd compiler.py get_tensor_specialization adds "S" for
storages <= 2^31-1 bytes while AMDGCN_USE_BUFFER_OPS is on, the default), so a
kernel warmed on small tensors compiles again when production hands it a
>2 GiB storage. Production rows/rank are ~2M+, so some operands are ALWAYS
>2 GiB (the [rows, 2048] duvqk, the [rows, 1536] e4m3 vqk, the dq-split
workspace, the [rows, 1536] LN*U*dropout output) and others straddle 2 GiB
from batch to batch (the [rows, 512] bf16 activations). The items with
always-large operands now use that class by default: F2/F2q/M2 run on views of
a >2 GiB arena with a >2 GiB dq-split workspace, and L4/L4b's large-row sizes
give the output its >2 GiB class (x/u stay below 2 GiB; with STORES=both a
second size puts them above it too). Straddling operands keep the 595b51e
rule: small by default, both classes with FP8_WARMUP_STORES=both.
STATE. Besides the 595b51e set: every plain scalar global (bool/int/float/str/
None) of the FP8 modules is snapshotted and restored (the lazy env flags and
once-latches of ship-11: LN*U*dropout v3/maskgen, FWD_PIPE, the weighted-LN
lazy switches, ...), the ARMED/once sets of bwd_fp8 and the new caches are put
back to their pre-warmup contents (triton_fp8_wshadow._plans,
fp8_helpers_fused._dummy, the u_dp workspace / logged / quiet-heuristic state,
the BM16 and targets-only latches), and triton_fp8._weight_cache is never
touched (W1 calls the shadow kernels, not _get_weight_shadow). With
FP8_WARMUP_VERIFY=1 every plain global and container of the FP8 modules is
digested before and after; any difference outside the loaded-library handles
is a guard failure ("module globals"). Every item runs with Python stdout
captured, so arming lines the production wrappers print are dropped here and
print at the real first use. Exception: ship17 row (100)'s two lines
(_STDOUT_PASS) print here, once, and their latches are kept (_KEEP_GLOBALS).
Ship17 item (120) LUT-SiLU fwd likewise: F1 is the first FP8 FlyDSL forward,
so it compiles the lut build before RUN_START, prints the armed line once
(_STDOUT_PASS) and uploads the warm-up scale's table; the latch, the table
cache and the side streams are kept (_KEEP_GLOBALS).
Exception, ship17 item (105) SPF: F1 prints the SPF armed line after the
item, outside the capture, and its latch is kept (_KEEP_GLOBALS), so that
line prints once, before RUN_START.
Exception, ship17 item (106) BWD_SPF: the ARMED line F2/F2q/F2e latch (mha
logger, muted by _Guards) is logged after the restore, and its latch is kept
(_KEEP_GLOBALS), so that line prints once, before RUN_START.
SHIP-13 RUNG (R15 item F2e, gin item (47) M-E1). With
DLRM_HSTU_FP8_BWD_DKDV_E5M2=1 every engaged ordinary-layer backward hands the
FlyDSL backward a DkdvE5m2Sink, which compiles a second bwd_dkdv build (F2's
compile key plus e8emit="only") at the first engaged call, after the per-layer
scale warm-up, inside the timed region. F2e makes F2's call with the sinks of
an engaged call (the E1 sink and, where F2q runs, the dq sink adopting the same
dz8), then the cast and finishes the engaged call runs after it (on the ship-12
joint path: no cast, the dq sink's finish_q and the E1 finish). It runs only
when the knob is "1" at warmup time (tpa._dkdv_e5m2_requested, after the env
bootstrap), under F2's preconditions, for F2's N list. Of tpa only its env
predicates are read; both sinks are local objects. Its ok line prints on every
rank, with the synchronized backward time (the e8 compile on a cold FlyDSL
cache) and the number of new compile_hstu_bwd_dkdv builds.
F2r (gin item (86)). The two sinks' finalizers (_e8_fin_kernel of
DkdvE5m2Sink.finish, _dq8_fin_kernel of _dq8_finalize) take RPP, the partial
rows per program, as a constexpr that grows with the call's row count M; F2e
and F2q finish at T = 2B (RPP 8) only. F2r calls both finalizers once per RPP
bucket of M = 0 .. B * _N_C on synthetic partials whose all-zero row prefix
masks every row (_fin_rpp_prebuild), under F2e's (e8) and F2q's (dq8)
preconditions, so no engaged step compiles one inside the timed region.
L4n (r16 NIB, gin item (94), DLRM_HSTU_FP8_LNMD_MASK_NIB=1 only). The two
NIB keep-mask writers are reached in training only through the NIB gate,
which needs an HSTUComputeOutputFunction FP8 call with a dO8 site on its
attention output; L4's synthetic calls have none. L4n (_nib_items) launches
both writers directly (triton_hstu_linear.lnmd_mask_nib_warmup_launch: the
maskgen branch's production launch arguments, no gate state touched) at L4's
large-row sizes plus the > 2 GiB x / u class (_ROWS_XL, whatever
FP8_WARMUP_STORES says: no TSPEC rows exist for them yet) and L4's seeds.
On L4's rows (not _ROWS_XL) it is also the writers' startup byte check: the
int8 kernel each writer replaces runs on the same inputs and every output
byte must match (y / mean / rstd, NIB mask == pack_ref(int8)); a mismatch
latches NIB for the process (one DISABLED line, printed after the item;
_LNMD_MASK_NIB_BROKEN is in _KEEP_GLOBALS so the restore keeps it).
With the knob off it runs, logs and counts nothing.
Ship17 item (111) _pk_nib V3 (DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3): L4n's first
pk launch is the pk writer selector's first pick, so the (111) line (ARMED,
or OFF / NOT armed) latches inside that item's stdout capture. _pk_v3_reprint
prints it right after the item, outside the capture, on every rank, and the
two line marks are in _KEEP_GLOBALS (not restored), so it prints once per
process, before RUN_START, and no training launch prints it.
"""

import collections
import contextlib
import hashlib
import io
import logging
import os
import re
import sys
import time
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gin
import torch

from generative_recommenders.dlrm_v4.train.blas_warmup import (
    _quiet_stderr,
    _rng_snapshot,
)

logger: logging.Logger = logging.getLogger(__name__)

_TAG = "[fp8-warmup]"

# Set once this rank started the warmup items (they may leave non-torch device
# memory that comes back during the pre-grow). dlrm_v4/alloc_pregrow.py runs
# its converge pass only when this is True, so FP8_WARMUP=0 keeps the ship-11
# pre-grow exactly.
WARMUP_RAN: bool = False

# N classes (synthetic; each <= gin max_seq_len 4096). N reaches a compile key
# in three ways only; F1 (FlyDSL e4m3 forward) and M2 (bf16 FlyDSL backward)
# take N as a launch argument, so any N serves them:
#  (a) MAX_SEQ_LEN = N, a non-constexpr int of the Triton delta forward and
#      backward and main forward kernels (T1/T2/T2e, tier 2 T1b/T2b/M1).
#      Triton specializes such ints as == 1, % 16 == 0 or neither;
#      AUTOTUNE_MAX_SEQ_LEN is the static max_seq_len for every N <= 4096.
#      Those loops run (_N_A, _N_B): N % 16 == 0 and N % 16 != 0.
#  (b) U_TILES = cdiv(N, BMQ), a non-constexpr int of the dq-split kernels
#      (hstu_dq_split, hstu_dq_e5m2._dq8_sched_kernel; BMQ 128 by default,
#      or 64), same rule. F2/F2q/F2e run (_N_A, _N_C): U_TILES 24 (47 at
#      BMQ 64), not % 16, and 32 (64), % 16 == 0.
#  (c) c = ceil(log2 N) sets the full-FP8 FlyDSL backward's mu2_ssc/mu2_dsc,
#      so F2/F2q/F2e compile one build per c. c = 12 is the class of every
#      N in 2049..4096, the top class under the configured max_seq_len
#      (4096 = 2^12, the gin default MAX_SEQ_LEN; an override changes the top
#      class); c = 11 (_N_D) too, unless FP8_WARMUP_N11=0.
# _N_A = 2992 is the only value in both loop lists: N % 16 == 0 for (a) and
# U_TILES % 16 != 0 for (b), with c = 12. Every multiple of 16 in 2064..3008
# or 3088..3968 selects the same classes; 2992 is one of them.
# _N_B = 3000: N % 16 != 0, c = 12. _N_C = 4096 = max_seq_len: U_TILES % 16
# == 0, c = 12, and F2r's row bound B * _N_C. _N_D = 2000: c = 11 (a full run
# reaches it at window 50). Not warmed (they compile at first use, as without
# this warmup): N or U_TILES == 1, c <= 10, and c = 11 with FP8_WARMUP_N11=0.
# Synthetic scales (no calibration state is read). Runtime fp32 arguments,
# with one exception: under DLRM_HSTU_FP8_BWD_VF3P / _VF4 (gin True) the
# dq-split backward adds its vf3p/vf4 VALU tokens, a compile-key change, only
# when 0 < qkv_scale <= hstu_flydsl_mha._VF3P_QKV_SCALE_MAX (0.25). 2^-6 is a
# power of two inside that guard, so F2/F2q/F2e build the vf3p/vf4 variant; a
# run whose calibrated scale exceeds 0.25 builds the other one at first use.
# _LN_INV_SCALE = 2^6 is a finite positive fp32 value: where the MASKGEN_PK
# path applies (gin), it passes _use_lnmd_fwd_maskgen_pk's 0 < Y_INV_SCALE <
# inf check, so the maskgen_pk kernel is built as in production; otherwise it
# only reaches fp32 kernel arguments.
_N_A = 2992
_N_B = 3000
_N_C = 4096
_N_D = 2000

# Runtime floats (never a compile/autotune key).
_QKV_SCALE = 2.0 ** -6
_LN_INV_SCALE = 64.0
_LN_EPS = 1e-6  # stu.py passes norm_eps=1e-6 to every LN wrapper.
# Explicit dropout seeds: no RNG draw. Production draws randint(0, 2**62) (i64),
# so the i64 class and both %16 classes occur.
_SEEDS = (2 ** 40, 2 ** 40 + 1)

_ARENA_BYTES = 2 ** 31 + 2 ** 28
_ALIGN = 256

# LN*U*dropout large-row sizes (rows >= 2^20 run the v2/v3/maskgen forward,
# triton_hstu_linear._use_lnmd_fwd_v2). Chosen from the 2^31 - 1 byte storage
# rule (STORAGE CLASS) at H*Dv = 512, not from batch statistics; both lie in
# the configured row range 2B .. B * max_seq_len.
# _ROWS_L = 1572864 is in 1398102 .. 2097151, where the [rows, 3*512] output
# is > 2 GiB in e4m3 (and bf16) and x/u [rows, 512] bf16 stay <= 2 GiB.
# _ROWS_XL = 2162688 (FP8_WARMUP_STORES=both) is above 2097151: x/u > 2 GiB
# too. Each also runs at + 1, the % 16 != 0 class of the row-count argument.
# Not warmed: rows 2^20 .. 1398101 (e4m3 output <= 2 GiB on this kernel).
_ROWS_L = 3 << 19
_ROWS_XL = 33 << 16
# U1 (FWD_U_DP): synthetic large token counts for the U-slice GEMM (_ROWS_L
# and the odd _ROWS_L + 1), so the code object hipBLASLt's heuristic picks for
# such an M loads before RUN_START. Not a class bound: an M for which the
# heuristic picks another solution loads that one at first use.
_U_DP_ROWS = (_ROWS_L, _ROWS_L + 1)

_VERIFY_MAX_NUMEL = 64 * 1024 * 1024
_EMBEDDING_NAME = re.compile(
    r"(embedding_collection|embedding_bag|embedding_module|embeddings_by|"
    r"_embedding_tables|(^|\.)ebc(\.|$)|(^|\.)ec(\.|$)|tbe|fused_optimizer|"
    r"_lookups|emb_module)",
    re.IGNORECASE,
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _tunableop_tuning_or_recording() -> bool:
    """Same check as blas_warmup.warmup() (inline there, not a helper). No GEMM
    is issued here, but a tuning/recording session is not a run to warm."""
    try:
        import torch.cuda.tunable as tunable

        return bool(tunable.is_enabled() and (
            tunable.tuning_is_enabled()
            or getattr(tunable, "record_untuned_is_enabled", lambda: False)()
        ))
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Triton bookkeeping: JIT cache sizes (jit_new) and autotune cache keys.
# ----------------------------------------------------------------------------

_JIT_KERNELS: Tuple[Tuple[str, str], ...] = (
    ("generative_recommenders.ops.triton.triton_hstu_attention", "_hstu_attn_fwd"),
    ("generative_recommenders.ops.triton.triton_hstu_attention", "_hstu_attn_delta_bwd"),
    ("generative_recommenders.ops.triton.triton_layer_norm", "_weighted_layer_norm_fwd"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_rng"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_generate_random_mask"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_generate_random_mask16"),
    ("generative_recommenders.ops.triton.bwd_fp8", "_quantize_amax_kernel"),
    ("generative_recommenders.ops.flydsl.hstu_do_fp8", "_blk_kernel"),
    ("generative_recommenders.ops.flydsl.hstu_dq_split", "_dq_from_ds_kernel"),
    # ship8/ship10 first uses at the FP8 switch (DQ_SPLIT_SCHED, STORE_T,
    # VK_PRODUCER, OUT_WGRAD dual cast, uvqk/vk colsum cast, LNMD v2).
    ("generative_recommenders.ops.flydsl.hstu_dq_split", "_dq_from_ds_sched_kernel"),
    ("generative_recommenders.ops.triton.hstu_delta_bwd_e5m2", "_hstu_attn_delta_bwd_e5m2"),
    ("generative_recommenders.ops.triton.triton_layer_norm",
     "_weighted_layer_norm_fwd_fp8_store_t"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_rng_v2"),
    ("generative_recommenders.ops.triton.bwd_fp8", "_quantize_amax_dual_kernel"),
    ("generative_recommenders.ops.triton.bwd_fp8", "_quantize_amax_colsum_kernel"),
)

# Ship-11 first uses (items (24)-(30)). Counted when the module and kernel exist;
# an absent one (a pruned item) is skipped instead of making the census unknown.
_JIT_KERNELS_S11: Tuple[Tuple[str, str], ...] = (
    ("generative_recommenders.ops.triton.triton_layer_norm",
     "_weighted_layer_norm_fwd_fp8_store_t_v2"),
    ("generative_recommenders.ops.triton.hstu_delta_bwd_e5m2", "_hstu_attn_delta_bwd_e5m2_v2"),
    # X4C19 DLRM_HSTU_FP8_ATTN_INTGLUE (default off): F2/F2q compile it when armed.
    ("generative_recommenders.ops.flydsl.hstu_dq_split", "_intglue_kernel"),
    ("generative_recommenders.ops.triton.hstu_delta_bwd_e5m2",
     "_hstu_attn_delta_bwd_e5m2_lpt_perm"),
    ("generative_recommenders.ops.triton.bwd_fp8", "_quantize_amax_colsum_v2_kernel"),
    ("generative_recommenders.ops.triton.triton_fp8_wshadow", "_wshadow_amax_kernel"),
    ("generative_recommenders.ops.triton.triton_fp8_wshadow", "_wshadow_quant_kernel"),
    ("generative_recommenders.ops.triton.fp8_helpers_fused", "_fp8h_amax_tiles_kernel"),
    ("generative_recommenders.ops.triton.fp8_helpers_fused", "_fp8h_weight_quant_kernel"),
    ("generative_recommenders.ops.triton.hstu_dq_e5m2", "_dq8_sched_kernel"),
    ("generative_recommenders.ops.triton.hstu_dq_e5m2", "_dq8_pfx_kernel"),
    ("generative_recommenders.ops.triton.hstu_dq_e5m2", "_dq8_fin_kernel"),
    ("generative_recommenders.ops.triton.hstu_dq_e5m2", "_quantize_amax_colsum_skip_kernel"),
    ("generative_recommenders.ops.flydsl.hstu_dq_split", "_dq_bc_bounds_kernel"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_rng_v3"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_maskgen"),
    # DLRM_HSTU_FP8_MASKGEN_PK (L4 reaches it through the production wrapper)
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_maskgen_pk"),
    # DLRM_HSTU_FP8_LNMD_MASK_NIB (r16 NIB, gin item (94); item L4n compiles
    # both writers; never compiled with the knob off, so they add 0 there)
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_maskgen_nib"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_maskgen_pk_nib"),
    # ship17 (111) pk writer V3 killed (DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3 not
    # "1"): item L4n compiles the 544e691 body under this name; never
    # compiled with (111) on (the default), so it adds 0 there
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_maskgen_pk_nib_r1"),
)

# Autotuned kernels this warmup can reach. Any key it ADDS is popped afterwards
# (counted as autotune_dropped), so no key is fixed by tiny-T timings; pinned
# keys (autotune_pin_table.py) are already present and are left alone.
_AUTOTUNED: Tuple[Tuple[str, str], ...] = (
    ("generative_recommenders.ops.triton.triton_hstu_attention", "_hstu_attn_fwd"),
    ("generative_recommenders.ops.triton.triton_layer_norm", "_weighted_layer_norm_fwd"),
    ("generative_recommenders.ops.triton.triton_hstu_linear", "_ln_mul_dropout_fwd_rng"),
)

# Ship-13 rung first uses (item F2e, gin item (47) M-E1): the E1 sink's prefix
# and finalize kernels and the casts of the E1 paths off the ship-12 joint
# path. Counted only when the module is already loaded (sys.modules, no import
# for the census), so a warmup with E1 off counts exactly as before.
_JIT_KERNELS_S13: Tuple[Tuple[str, str], ...] = (
    ("generative_recommenders.ops.flydsl.hstu_dkdv_e5m2", "_e8_pfx_kernel"),
    ("generative_recommenders.ops.flydsl.hstu_dkdv_e5m2", "_e8_fin_kernel"),
    ("generative_recommenders.ops.triton.hstu_dq_e5m2",
     "_quantize_amax_colsum_skipn_kernel"),
    ("generative_recommenders.ops.triton.bwd_fp8", "_quantize_amax_colsum_sel_kernel"),
)


def _get_kernel(mod_name: str, attr: str) -> Any:
    import importlib

    return getattr(importlib.import_module(mod_name), attr)


def _unwrap_jit(obj: Any) -> Any:
    try:
        from triton.runtime.jit import JITFunction
    except Exception:
        return None
    for _ in range(5):
        if isinstance(obj, JITFunction):
            return obj
        obj = getattr(obj, "fn", None)
        if obj is None:
            return None
    return None


def _jit_cache_size(jit_fn: Any) -> int:
    """Number of compiled variants across devices, or -1 if unknown."""
    caches = getattr(jit_fn, "device_caches", None)
    if caches is not None:
        total = 0
        for entry in list(caches.values()):
            first = entry[0] if isinstance(entry, (tuple, list)) else entry
            total += len(first)
        return total
    cache = getattr(jit_fn, "cache", None)
    if isinstance(cache, dict):
        return sum(len(v) for v in cache.values() if isinstance(v, dict))
    return -1


def _jit_total() -> int:
    total = 0
    for mod_name, attr in _JIT_KERNELS:
        try:
            jit_fn = _unwrap_jit(_get_kernel(mod_name, attr))
            if jit_fn is None:
                return -1
            n = _jit_cache_size(jit_fn)
            if n < 0:
                return -1
            total += n
        except Exception:
            return -1
    for mod_name, attr in _JIT_KERNELS_S11:
        try:
            jit_fn = _unwrap_jit(_get_kernel(mod_name, attr))
        except Exception:
            continue
        n = _jit_cache_size(jit_fn) if jit_fn is not None else -1
        if n > 0:
            total += n
    for mod_name, attr in _JIT_KERNELS_S13:
        mod = sys.modules.get(mod_name)
        jit_fn = _unwrap_jit(getattr(mod, attr, None)) if mod is not None else None
        n = _jit_cache_size(jit_fn) if jit_fn is not None else -1
        if n > 0:
            total += n
    return total


def _autotuners() -> List[Any]:
    out = []
    for mod_name, attr in _AUTOTUNED:
        try:
            at = _get_kernel(mod_name, attr)
            # Same unwrap as autotune_pin_table.apply_autotune_pins.
            for _ in range(4):
                if hasattr(at, "cache") and hasattr(at, "configs"):
                    break
                at = at.fn
            if isinstance(getattr(at, "cache", None), dict):
                out.append(at)
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------------
# Synthetic buffers.
# ----------------------------------------------------------------------------


class _Buffers:
    """Zero-filled tensors. small: a fresh allocation per tensor. big: a view
    carved (256-byte aligned, bump-allocated, reset per item) from one arena
    whose storage is > 2 GiB, for the Triton AMD pointer storage-class
    specialization (STORAGE CLASS in the module docstring). The arena is
    allocated at the first big request."""

    def __init__(self, device: torch.device, want_big: bool) -> None:
        self.device = device
        self.arena: Optional[torch.Tensor] = (
            torch.zeros(_ARENA_BYTES, dtype=torch.uint8, device=device)
            if want_big else None
        )
        self.offset = 0

    def reset(self) -> None:
        if self.arena is not None and self.offset:
            # Outputs may have been written into the arena; keep inputs zero.
            self.arena[: self.offset].zero_()
        self.offset = 0

    def act(self, shape: Sequence[int], dtype: torch.dtype,
            big: bool = False) -> torch.Tensor:
        numel = 1
        for s in shape:
            numel *= int(s)
        elsize = torch.empty((), dtype=dtype).element_size()
        nbytes = numel * elsize
        if big:
            if self.arena is None:
                self.arena = torch.zeros(_ARENA_BYTES, dtype=torch.uint8,
                                         device=self.device)
            start = (self.offset + _ALIGN - 1) // _ALIGN * _ALIGN
            if start + nbytes > self.arena.numel():
                raise RuntimeError("fp8-warmup arena exhausted")
            self.offset = start + nbytes
            raw = self.arena[start:start + nbytes]
        else:
            raw = torch.zeros(nbytes, dtype=torch.uint8, device=self.device)
        return raw.view(dtype).view(tuple(int(s) for s in shape))

    def release(self) -> None:
        self.arena = None


@contextlib.contextmanager
def _big_dq_split_workspace(device: torch.device):
    """Make the FlyDSL dq-split workspace a view of a > 2 GiB storage, as in
    production (the persistent workspace is several GiB there), by seeding
    hstu_dq_split's persistent cache for this device; the previous entry is
    put back afterwards (and _Guards.restore puts the whole cache back)."""
    from generative_recommenders.ops.flydsl import hstu_dq_split as dqs

    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, index)
    had = key in dqs._WS_CACHE
    prev = dqs._WS_CACHE.get(key)
    dqs._WS_CACHE[key] = torch.empty(_ARENA_BYTES, dtype=torch.uint8, device=device)
    try:
        yield
    finally:
        dqs._WS_CACHE.pop(key, None)
        if had:
            dqs._WS_CACHE[key] = prev


# ----------------------------------------------------------------------------
# State guards (design section 3).
# ----------------------------------------------------------------------------

_MHA_SETS = (
    "_logged_layouts", "_logged_fp8_fwd_layouts", "_logged_fallbacks",
    "_logged_fwd_layouts", "_logged_fwd_fallbacks", "_DQ_SPLIT_ORDER_LOGGED",
    # ship8 dq-split arm lines (gin documents them as the arm proof).
    "_DQ_SPLIT_SCHED_LOGGED", "_DQ_SPLIT_OPT_LOGGED",
    # ship-12 item (48)(d) FWD_MASK_ARITH once-latch (F1 reaches it).
    "_fp8_fwd_mask_arith_logged",
    # X4C19 DLRM_HSTU_FP8_FWD_PV armed-line / declined once-latch
    # (default off; F1 reaches fp8_fwd_pv_config through _fp8_fwd_config).
    "_fp8_fwd_pv_logged",
    # NOT listed (ship17 row (100) RB1): _fp8_fwd_v8swz_logged and
    # _fp8_fwd_rider_logged, the LAB_FWD_V8SWZ / LAB_FWD_RIDER armed / declined
    # once-latches. F1 reaches both through _fp8_fwd_config, and the line it
    # prints there is the arming record (_STDOUT_PASS lets it out of the item's
    # stdout capture). The latch is not restored, so the timed region prints
    # nothing (_KEEP_GLOBALS keeps it out of the VERIFY digest).
    # X4C19 DLRM_HSTU_FP8_ATTN_INTGLUE armed / declined once-latch (default
    # off; F2/F2q reach it through the dq-split backward).
    "_ATTN_INTGLUE_LOGGED",
)
_MHA_DICTS = ("_e1_bwd_census",)
_MHA_SCALARS = ("_backward_calls", "_forward_calls", "_fp8_forward_calls",
                "_e1_bwd_calls",
                # ship-11 item (26)(a) FWD_PIPE once-latch.
                "_fp8_fwd_pipe_logged")
_MHA_LIST_CELLS = ("_full_fp8_stats_calls", "_bwd_dump_calls",
                   "_hybrid_force_off", "_hybrid_inner")
_TF8_CONTAINERS = ("_scale_value_cache", "_inverse_scale_cache", "_weight_cache",
                   "_packed_weight_cache", "_capability_sites",
                   "_folded_input_scale_cache", "_scaled_output_bias_cache",
                   "_residual_workspaces", "_capability_contracts",
                   # ship-11 (25)(e) SHADOW_FUSED latch (never reached here).
                   "_shadow_fused_logged")

_P_TRITON = "generative_recommenders.ops.triton."
_P_FLYDSL = "generative_recommenders.ops.flydsl"

# Modules whose plain scalar globals (bool/int/float/str/None) are snapshotted
# and restored: lazy env flags and once-latches (the ship-11 LN*U*dropout v3 /
# maskgen flags, FWD_PIPE, the weighted-LN lazy switches, BM16 is a list cell
# below, ...). The FlyDSL compile modules (hstu_fwd, hstu_bwd_dkdv, hstu_do_fp8)
# hold no mutable scalar and are digested only.
_SCALAR_MODULES: Tuple[str, ...] = (
    _P_TRITON + "bwd_fp8",
    _P_TRITON + "triton_fp8",
    _P_TRITON + "triton_fp8_wshadow",
    _P_TRITON + "fp8_helpers_fused",
    _P_TRITON + "hstu_dq_e5m2",
    _P_TRITON + "hstu_delta_bwd_e5m2",
    _P_TRITON + "triton_layer_norm",
    _P_TRITON + "triton_hstu_linear",
    _P_TRITON + "triton_hstu_attention",
    _P_TRITON + "triton_hstu_preprocess_and_attention",
    _P_FLYDSL + ".hstu_flydsl_mha",
    _P_FLYDSL + ".hstu_dq_split",
    _P_FLYDSL,
)
# Loaded-library handles and an opened /dev/null fd: kept as the warmup left
# them (the point of the warmup), never restored, never a VERIFY difference.
_KEEP_GLOBALS = frozenset({
    (_P_TRITON + "triton_fp8", "_devnull_fd"),
    (_P_TRITON + "triton_fp8", "_wshadow_module_ref"),
    (_P_TRITON + "bwd_fp8", "_uvqk_tm_ext"),
    (_P_FLYDSL + ".hstu_flydsl_mha", "_kernel"),
    (_P_FLYDSL + ".hstu_flydsl_mha", "_kernel_error"),
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fwd_kernel"),
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fwd_kernel_error"),
    # Ship17 item (111) _pk_nib V3, item L4n: the pk writer selector's two
    # once-per-process line marks (triton_hstu_linear._lnmd_mask_nib_pk_kernel;
    # each holds the line it printed). L4n's first pk launch sets one inside
    # the item's stdout capture and _pk_v3_reprint prints that line after the
    # item; kept, not restored, so it prints once, before RUN_START, and the
    # timed region prints none.
    (_P_TRITON + "triton_hstu_linear", "_LNMD_MASK_NIB_PK_V3_ARMED_LOGGED"),
    (_P_TRITON + "triton_hstu_linear", "_LNMD_MASK_NIB_PK_V3_OFF_LOGGED"),
    # r16 NIB (gin item (94)) item L4n: the NIB latch its startup byte check may set
    # (triton_hstu_linear.lnmd_mask_nib_latch) must outlive the restore; it is
    # None unless that check found a writer whose bytes differ.
    (_P_TRITON + "triton_hstu_linear", "_LNMD_MASK_NIB_BROKEN"),
    # Ship17 row (100) RB1: the LAB_FWD_V8SWZ / LAB_FWD_RIDER armed / declined
    # once-latches F1 sets (not in _MHA_SETS) are kept, so each line prints
    # once per process, before RUN_START, and never in the timed region.
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fp8_fwd_v8swz_logged"),
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fp8_fwd_rider_logged"),
    # Ship17 item (105) SPF fold fwd: the armed-line once-latch (mode -> line)
    # F1 sets outlives the restore, deliberately not in _MHA_SETS: F1's line,
    # printed after the item outside its stdout capture, is the process's one
    # SPF arming line, so none prints in the timed region.
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fp8_fwd_spf_logged"),
    # Ship17 item (106) SPF fold bwd: the BWD_SPF ARMED / stray-tsc-strip
    # once-latch (key -> line) F2/F2q/F2e set outlives the restore,
    # deliberately not in _MHA_SETS: its line, logged after the restore
    # (_spf_bwd_armed_reprint), is the process's one BWD_SPF ARMED line, so
    # none prints in the timed region.
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fp8_bwd_spf_logged"),
    # Ship17 item (120) LUT-SiLU fwd: F1's armed / declined once-latch (not in
    # _MHA_SETS: the line it prints there is the arming record), the per-scale
    # device table cache (entries are never freed: a launch may still read
    # them) and the per-device upload side streams. Kept, not restored.
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fp8_fwd_lut_logged"),
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fwd_lut_tables"),
    (_P_FLYDSL + ".hstu_flydsl_mha", "_fwd_lut_streams"),
    # gin item (48)(b') VALU-diet delta backward: the per-specialization verdicts of its runtime bitwise self-check
    # against the v2 kernel (hstu_delta_bwd_e5m2._v2_diet_verdict). T2e reaches the check before RUN_START; the
    # verdicts are kept so the timed region runs none (its armed / declined latches are restored and print at the
    # real first use).
    (_P_TRITON + "hstu_delta_bwd_e5m2", "_V2_DIET_VERDICT"),
})
# Ship-11 containers put back to their pre-warmup contents (same objects:
# clear + update / slice assignment): ARMED/once sets, the new per-process
# caches (dropped entries are rebuilt at first use) and list-cell latches.
_RESTORE_CONTAINERS: Tuple[Tuple[str, str], ...] = (
    (_P_TRITON + "bwd_fp8", "_uvqk_armed_logged"),
    (_P_TRITON + "bwd_fp8", "_uvqk_once_logged"),
    (_P_TRITON + "triton_fp8", "_u_dp_workspaces"),
    (_P_TRITON + "triton_fp8", "_u_dp_logged"),
    (_P_TRITON + "triton_fp8", "_quiet_seen"),
    (_P_TRITON + "triton_fp8_wshadow", "_plans"),
    (_P_TRITON + "fp8_helpers_fused", "_dummy"),
    (_P_TRITON + "triton_layer_norm", "_WLN_FWD_V2_MODE"),
    (_P_TRITON + "triton_layer_norm", "_D256_V3_REPORTED"),
    (_P_TRITON + "triton_layer_norm", "_WLN_BWD_FLYDSL_LOGGED"),
    (_P_TRITON + "triton_hstu_attention", "_DELTA_FWD_BM16_LOGGED"),
    (_P_TRITON + "triton_hstu_preprocess_and_attention", "_TGT_SPARSE_LOGGED"),
    # Ship-12 item (48): (a) the work-list dO quantizer's once-latch (F2/F2q
    # reach it through the C8 backward) and (b') the VALU-diet delta backward's
    # armed / declined latches and its arch cache (T2e reaches them).
    (_P_FLYDSL + ".hstu_do_fp8", "_WL_LOGGED"),
    (_P_TRITON + "hstu_delta_bwd_e5m2", "_V2_DIET_LOGGED"),
    (_P_TRITON + "hstu_delta_bwd_e5m2", "_V2_DIET_DECLINE_LOGGED"),
    (_P_TRITON + "hstu_delta_bwd_e5m2", "_V2_DIET_ARCH"),
)
# Digested under FP8_WARMUP_VERIFY=1 (globals and containers, by value or
# identity): the scalar modules plus the FlyDSL compile modules.
_DIGEST_MODULES: Tuple[str, ...] = _SCALAR_MODULES + (
    _P_FLYDSL + ".hstu_fwd",
    _P_FLYDSL + ".hstu_bwd_dkdv",
    _P_FLYDSL + ".hstu_do_fp8",
    "generative_recommenders.modules.stu",
    "generative_recommenders.modules.hstu_transducer",
)
# Imported before the snapshot (so their pre-warmup state is recorded): the
# ship-11 modules production imports lazily at first use. Definitions only at
# import (no launch, no print, no environment read beyond constants).
_PREIMPORT: Tuple[str, ...] = (
    _P_TRITON + "triton_fp8_wshadow",
    _P_TRITON + "fp8_helpers_fused",
    _P_TRITON + "hstu_dq_e5m2",
    _P_TRITON + "hstu_delta_bwd_e5m2",
    # ship-12 item (48)(a): imported lazily by the C8 backward; pre-imported so
    # its _WL_LOGGED cell exists when the containers are snapshotted.
    _P_FLYDSL + ".hstu_do_fp8",
)

_SCALAR_TYPES = (bool, int, float, str, type(None))


def _plain_scalar(value: Any) -> bool:
    return type(value) in _SCALAR_TYPES


def _module(name: str) -> Optional[types.ModuleType]:
    mod = sys.modules.get(name)
    return mod if isinstance(mod, types.ModuleType) else None


def _short(value: Any) -> Any:
    """Hashable, comparable stand-in for a container member."""
    if _plain_scalar(value) or isinstance(value, (torch.dtype, torch.device)):
        return ("v", value)
    if isinstance(value, tuple) and len(value) <= 16 and all(
            _plain_scalar(x) or isinstance(x, (torch.dtype, torch.device, tuple))
            for x in value):
        return ("v", repr(value))
    return ("i", id(value))


def _digest_value(value: Any) -> Any:
    if _plain_scalar(value) or isinstance(value, (torch.dtype, torch.device)):
        return ("v", value)
    if isinstance(value, (set, frozenset)):
        return ("s", len(value), frozenset(_short(x) for x in value))
    if isinstance(value, dict):
        return ("d", len(value),
                frozenset((_short(k), id(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        if len(value) <= 64:
            return ("l", tuple(_short(x) for x in value))
        return ("l", len(value), id(value))
    return ("o", id(value))


def _globals_digest(names: Sequence[str]) -> Dict[Tuple[str, str], Any]:
    """Plain globals and containers of the loaded modules in `names` (modules,
    classes, callables and loggers are skipped)."""
    out: Dict[Tuple[str, str], Any] = {}
    for mod_name in names:
        mod = _module(mod_name)
        if mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            if attr.startswith("__") or (mod_name, attr) in _KEEP_GLOBALS:
                continue
            if isinstance(value, (types.ModuleType, type, logging.Logger)):
                continue
            if callable(value):
                continue
            try:
                out[(mod_name, attr)] = _digest_value(value)
            except Exception:
                out[(mod_name, attr)] = ("x", id(value))
    return out


def _digest_diff(before: Dict[Tuple[str, str], Any],
                 after: Dict[Tuple[str, str], Any]) -> List[str]:
    changed = []
    for key, value in after.items():
        if key not in before:
            # A module first loaded during the warmup is not compared.
            if any(k[0] == key[0] for k in before):
                changed.append(f"{key[0].rsplit('.', 1)[-1]}.{key[1]}(new)")
            continue
        if before[key] != value:
            changed.append(f"{key[0].rsplit('.', 1)[-1]}.{key[1]}")
    for key in before:
        if key not in after and _module(key[0]) is not None:
            changed.append(f"{key[0].rsplit('.', 1)[-1]}.{key[1]}(deleted)")
    return sorted(changed)


class _Guards:
    def __init__(self) -> None:
        self.changed: List[str] = []
        self.snap: Dict[str, Any] = {}
        self.scalars: Dict[Tuple[str, str], Any] = {}
        self.containers: Dict[Tuple[str, str], Any] = {}
        self.autotune_snap: List[Tuple[Any, set]] = []
        self.autotune_dropped = 0
        self.logger_levels: List[Tuple[logging.Logger, int]] = []
        self.bwd_sites = 0
        self.tf8_unchanged = True
        self.restored_scalars = 0
        self.digest_before: Optional[Dict[Tuple[str, str], Any]] = None
        self.globals_changed: List[str] = []

    def take(self, verify: bool = False) -> None:
        import importlib

        from generative_recommenders.ops.flydsl import hstu_dq_split
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        from generative_recommenders.ops.triton import bwd_fp8
        from generative_recommenders.ops.triton import triton_fp8 as tf8
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        for name in _PREIMPORT:
            try:
                importlib.import_module(name)
            except Exception:
                pass

        s = self.snap
        for name in _MHA_SETS + _MHA_DICTS:
            obj = getattr(mha, name, None)
            if obj is not None:
                s["mha." + name] = obj.copy()
        for name in _MHA_SCALARS:
            if hasattr(mha, name):
                s["mha." + name] = getattr(mha, name)
        for name in _MHA_LIST_CELLS:
            obj = getattr(mha, name, None)
            if isinstance(obj, list):
                s["mha." + name] = list(obj)
        s["dqs._WS_CACHE"] = dict(hstu_dq_split._WS_CACHE)
        # Once-latches whose lines are logged while the loggers below are
        # muted: restored so "[flydsl] launch stream: ...", "[hstu-attn]
        # backward kernel: ..." and "[hstu-attn] forward kernel: ..." print
        # at the real first use (the arm/selection record), not inside the
        # warmup. Re-arming them only repeats the cheap arch/kernel check.
        import generative_recommenders.ops.flydsl as flydsl_pkg

        s["flydsl._stream_logged"] = set(flydsl_pkg._stream_logged)
        for name in ("_host_ok", "_fwd_host_ok"):
            if hasattr(mha, name):
                s["mha." + name] = bool(getattr(mha, name))
        s["thl._MASK16_REPORTED"] = getattr(thl, "_MASK16_REPORTED", None)
        s["bwd._sites"] = dict(bwd_fp8._sites)
        s["bwd._layer_index"] = dict(bwd_fp8._layer_index)
        s["bwd._engaged_sites"] = set(bwd_fp8._engaged_sites)
        s["bwd._logged_resolved"] = bwd_fp8._logged_resolved
        for name in _TF8_CONTAINERS:
            obj = getattr(tf8, name, None)
            if obj is not None:
                s["tf8." + name] = len(obj)
        s["tf8._resolved_logged"] = getattr(tf8, "_resolved_logged", None)
        # Ship-11: every plain scalar global of the FP8 modules (lazy env
        # flags, once-latches, counters) and the ship-11 containers.
        for mod_name in _SCALAR_MODULES:
            mod = _module(mod_name)
            if mod is None:
                continue
            for attr, value in list(vars(mod).items()):
                if attr.startswith("__") or (mod_name, attr) in _KEEP_GLOBALS:
                    continue
                if _plain_scalar(value):
                    self.scalars[(mod_name, attr)] = value
        for mod_name, attr in _RESTORE_CONTAINERS:
            mod = _module(mod_name)
            obj = getattr(mod, attr, None) if mod is not None else None
            if isinstance(obj, collections.OrderedDict):
                self.containers[(mod_name, attr)] = collections.OrderedDict(obj)
            elif isinstance(obj, dict):
                self.containers[(mod_name, attr)] = dict(obj)
            elif isinstance(obj, set):
                self.containers[(mod_name, attr)] = set(obj)
            elif isinstance(obj, list):
                self.containers[(mod_name, attr)] = list(obj)
        for at in _autotuners():
            self.autotune_snap.append((at, set(at.cache.keys())))
        # Ship-12 merge (r14 B11, DLRM_HSTU_FP8_TM_NONEXP): the store_t / dual-cast
        # wrappers the items call allocate their token-major operand from
        # fp8_tm_alloc's private pool when it is active. Before the pre-grow that
        # pool is not prewarmed yet: an item would map it lazily, and its once-per-
        # device ARMED line would latch inside the warmup (stdout captured) and
        # never print at the real first use. Hold the pool off for the warmup
        # (plain torch.empty, the same path as with the knob off) and restore it.
        try:
            from generative_recommenders.ops.triton import fp8_tm_alloc

            self.snap["tm._broken"] = fp8_tm_alloc._broken
            fp8_tm_alloc._broken = "fp8-warmup (plain allocations until restore)"
        except Exception:
            pass
        if verify:
            self.digest_before = _globals_digest(_DIGEST_MODULES)
        # Quiet the FlyDSL layout / ARMED / PERSIST-grow lines: they are
        # restored latches and must print at the real first use instead.
        for name in ("generative_recommenders.ops.flydsl",
                     getattr(mha.logger, "name", ""),
                     "generative_recommenders.ops.flydsl.hstu_dq_split"):
            if name:
                lg = logging.getLogger(name)
                self.logger_levels.append((lg, lg.level))
                lg.setLevel(logging.ERROR)

    def restore(self) -> None:
        """Every step individually guarded; never raises."""
        s = self.snap

        def step(label: str, fn: Callable[[], None]) -> None:
            try:
                fn()
            except Exception as error:
                self.changed.append(f"{label}(restore error: {error})")

        def mha_restore() -> None:
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

            for name in _MHA_SETS + _MHA_DICTS:
                key = "mha." + name
                if key in s:
                    obj = getattr(mha, name)
                    obj.clear()
                    obj.update(s[key])
            for name in _MHA_SCALARS:
                key = "mha." + name
                if key in s:
                    setattr(mha, name, s[key])
            for name in _MHA_LIST_CELLS:
                key = "mha." + name
                if key in s:
                    getattr(mha, name)[:] = s[key]
            for name in ("_host_ok", "_fwd_host_ok"):
                key = "mha." + name
                if key in s:
                    setattr(mha, name, s[key])

        def stream_logged_restore() -> None:
            import generative_recommenders.ops.flydsl as flydsl_pkg

            flydsl_pkg._stream_logged.clear()
            flydsl_pkg._stream_logged.update(s["flydsl._stream_logged"])

        def dqs_restore() -> None:
            from generative_recommenders.ops.flydsl import hstu_dq_split

            # Drops the warmup's workspace so empty_cache() can release it
            # (torch reserved memory is back at its pre-warmup value).
            hstu_dq_split._WS_CACHE.clear()
            hstu_dq_split._WS_CACHE.update(s["dqs._WS_CACHE"])

        def thl_restore() -> None:
            from generative_recommenders.ops.triton import triton_hstu_linear as thl

            if s.get("thl._MASK16_REPORTED") is not None:
                thl._MASK16_REPORTED = s["thl._MASK16_REPORTED"]

        def bwd_check() -> None:
            from generative_recommenders.ops.triton import bwd_fp8

            moved = (
                dict(bwd_fp8._sites) != s["bwd._sites"]
                or dict(bwd_fp8._layer_index) != s["bwd._layer_index"]
                or set(bwd_fp8._engaged_sites) != s["bwd._engaged_sites"]
                or bwd_fp8._logged_resolved != s["bwd._logged_resolved"]
            )
            self.bwd_sites = len(bwd_fp8._sites) - len(s["bwd._sites"])
            if moved:
                logger.error("%s bwd_fp8 site state changed; restoring it", _TAG)
                self.changed.append("bwd_fp8")
                bwd_fp8._sites.clear()
                bwd_fp8._sites.update(s["bwd._sites"])
                bwd_fp8._layer_index.clear()
                bwd_fp8._layer_index.update(s["bwd._layer_index"])
                bwd_fp8._engaged_sites.clear()
                bwd_fp8._engaged_sites.update(s["bwd._engaged_sites"])
                bwd_fp8._logged_resolved = s["bwd._logged_resolved"]

        def tf8_check() -> None:
            from generative_recommenders.ops.triton import triton_fp8 as tf8

            for name in _TF8_CONTAINERS:
                key = "tf8." + name
                if key in s and len(getattr(tf8, name)) != s[key]:
                    self.tf8_unchanged = False
                    logger.error("%s triton_fp8.%s changed size %d -> %d",
                                 _TAG, name, s[key], len(getattr(tf8, name)))
            if getattr(tf8, "_resolved_logged", None) != s["tf8._resolved_logged"]:
                self.tf8_unchanged = False
                logger.error("%s triton_fp8._resolved_logged changed", _TAG)
            if not self.tf8_unchanged:
                self.changed.append("triton_fp8_caches")

        def containers_restore() -> None:
            for (mod_name, attr), saved in self.containers.items():
                mod = _module(mod_name)
                obj = getattr(mod, attr, None) if mod is not None else None
                if obj is None:
                    continue
                if isinstance(obj, list):
                    obj[:] = saved
                else:
                    obj.clear()
                    obj.update(saved)

        def scalars_restore() -> None:
            for (mod_name, attr), saved in self.scalars.items():
                mod = _module(mod_name)
                if mod is None:
                    continue
                current = getattr(mod, attr, saved)
                if type(current) is not type(saved) or current != saved:
                    setattr(mod, attr, saved)
                    self.restored_scalars += 1

        def autotune_restore() -> None:
            for at, keys in self.autotune_snap:
                for key in [k for k in list(at.cache.keys()) if k not in keys]:
                    at.cache.pop(key, None)
                    self.autotune_dropped += 1

        def loggers_restore() -> None:
            for lg, level in reversed(self.logger_levels):
                lg.setLevel(level)

        step("mha", mha_restore)
        step("flydsl._stream_logged", stream_logged_restore)
        step("hstu_dq_split._WS_CACHE", dqs_restore)
        step("thl._MASK16_REPORTED", thl_restore)
        step("bwd_fp8", bwd_check)
        step("triton_fp8", tf8_check)
        step("ship11 containers", containers_restore)
        step("module scalars", scalars_restore)
        step("autotune", autotune_restore)
        step("loggers", loggers_restore)

        def tm_restore() -> None:
            if "tm._broken" in s:
                from generative_recommenders.ops.triton import fp8_tm_alloc

                fp8_tm_alloc._broken = s["tm._broken"]

        step("fp8_tm_alloc", tm_restore)

    def verify_globals(self) -> None:
        """FP8_WARMUP_VERIFY=1: every digested global equals its pre-warmup
        value after restore(); a difference is a guard failure."""
        if self.digest_before is None:
            return
        self.globals_changed = _digest_diff(self.digest_before,
                                            _globals_digest(_DIGEST_MODULES))
        if self.globals_changed:
            logger.error("%s module globals differ after restore: %s", _TAG,
                         ", ".join(self.globals_changed[:24]))
            self.changed.append("module_globals")


def _dense_state_sha(model: torch.nn.Module) -> Tuple[str, int]:
    """sha256 over the model's dense params/buffers. Skips embedding tables /
    TBE state (by name), non-plain tensors (sharded/DTensor) and anything with
    numel >= 64M: too large to hash at init, and nothing here can reach them."""
    h = hashlib.sha256()
    count = 0
    items: List[Tuple[str, torch.Tensor]] = []
    seen = set()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if name in seen:
            continue
        seen.add(name)
        items.append((name, t))
    for name, t in sorted(items, key=lambda x: x[0]):
        if type(t) not in (torch.Tensor, torch.nn.Parameter):
            continue
        if _EMBEDDING_NAME.search(name) or t.numel() >= _VERIFY_MAX_NUMEL:
            continue
        if t.device.type == "meta":
            continue
        raw = t.detach().reshape(-1).contiguous().view(torch.uint8)
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(raw.cpu().numpy().tobytes())
        count += 1
    return h.hexdigest(), count


# ----------------------------------------------------------------------------
# Item runner.
# ----------------------------------------------------------------------------


def _mem_state(device: torch.device) -> Tuple[int, int]:
    """(free bytes per mem_get_info, torch caching-allocator reserved bytes);
    (-1, -1) if unavailable."""
    try:
        return (int(torch.cuda.mem_get_info(device)[0]),
                int(torch.cuda.memory_reserved(device)))
    except Exception:
        return -1, -1


# Ship17 row (100) RB1: item stdout lines that are printed, not dropped. Their
# once-latches are not restored (see _MHA_SETS), so the line F1 prints is the
# only one: the arming record, before RUN_START, once per process.
_STDOUT_PASS: Tuple[str, ...] = (
    "[hstu-fwd] DLRM_HSTU_FP8_LAB_FWD_V8SWZ=",
    "[hstu-fwd] DLRM_HSTU_FP8_LAB_FWD_RIDER=",
    # Item (120): the armed / declined line (not "... table built:").
    "[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD=",
)


class _Runner:
    def __init__(self, device: torch.device, bufs: _Buffers, rank0: bool,
                 skip: set, quiet: bool) -> None:
        self.device = device
        self.bufs = bufs
        self.rank0 = rank0
        self.skipset = skip
        self.quiet = quiet
        self.ok = 0
        self.skipped = 0
        self.failed = 0
        self.jit_new = 0
        self.jit_known = True
        self.tags: Dict[str, int] = {}

    def skip(self, tag: str, name: str, reason: str) -> None:
        self.skipped += 1
        if self.rank0:
            logger.info("%s item=%s %s skip(reason=%s)", _TAG, tag, name, reason)

    def run(self, tag: str, name: str, detail: str,
            fn: Callable[[], Optional[str]], all_ranks: bool = False) -> None:
        """Run one item. A str returned by `fn` is appended to its ok line;
        `all_ranks` prints that line on every rank (tagged rank=R) instead of
        rank 0 only (FAIL lines always print on every rank)."""
        if tag.lower() in self.skipset or name.lower() in self.skipset:
            self.skip(tag, name, "FP8_WARMUP_SKIP")
            return
        jit0 = _jit_total()
        t0 = time.perf_counter()
        captured = io.StringIO()
        extra: Optional[str] = None
        try:
            self.bufs.reset()
            if self.quiet:
                logger.info("%s item=%s %s %s (stderr silenced)", _TAG, tag,
                            name, detail)
            # Production wrappers print arming lines to stdout at their first
            # call (ARMED store_t_v2/vkp_v2/u_dp, FWD_PIPE, BM16, maskgen, ...);
            # their latches are restored, so they print at the real first use.
            # The _STDOUT_PASS lines (latch kept) print in `finally` below.
            with _quiet_stderr(self.quiet), contextlib.redirect_stdout(captured):
                extra = fn()
                # Attribute an async device fault to the item that caused it.
                torch.cuda.synchronize(self.device)
        except Exception as error:
            self.failed += 1
            logger.warning("%s item=%s %s %s FAIL(%s: %s) t=%.2fs", _TAG, tag,
                           name, detail, type(error).__name__,
                           str(error).splitlines()[0][:300] if str(error) else "",
                           time.perf_counter() - t0)
            logger.debug("%s item=%s traceback (stdout tail %r)", _TAG, tag,
                         captured.getvalue()[-300:], exc_info=True)
            return
        finally:
            for line in captured.getvalue().splitlines():
                if line.startswith(_STDOUT_PASS):
                    print(line, flush=True)
        jit1 = _jit_total()
        new = jit1 - jit0 if (jit0 >= 0 and jit1 >= 0) else -1
        if new < 0:
            self.jit_known = False
        else:
            self.jit_new += new
        self.ok += 1
        self.tags[tag] = self.tags.get(tag, 0) + 1
        if self.rank0 or all_ranks:
            logger.info("%s item=%s %s %s ok t=%.2fs jit_new=%d%s%s", _TAG, tag,
                        name, detail, time.perf_counter() - t0, new,
                        f" {extra}" if isinstance(extra, str) and extra else "",
                        f" rank={_rank()}" if all_ranks else "")


# ----------------------------------------------------------------------------
# The items.
# ----------------------------------------------------------------------------


def _find_layers(model: torch.nn.Module) -> Tuple[Any, Any, List[Any]]:
    from generative_recommenders.modules.hstu_transducer import HSTUTransducer
    from generative_recommenders.modules.stu import STULayer, STUStack

    transducer = next(
        (m for m in model.modules() if isinstance(m, HSTUTransducer)), None)
    stack = next((m for m in model.modules() if isinstance(m, STUStack)), None)
    if stack is not None:
        layers = [m for m in stack._stu_layers if isinstance(m, STULayer)]
    else:
        layers = [m for m in model.modules() if isinstance(m, STULayer)]
    return transducer, stack, layers


def _shadow_specs(main_layers: List[Any], last: Any, targets_only: bool,
                  fuse_qkv: bool, H: int, Dv: int, Dq: int
                  ) -> List[Tuple[Tuple[int, int], torch.dtype,
                                  Tuple[Tuple[int, int], ...]]]:
    """(weight shape, dtype, column layout) of every FP8 forward weight shadow
    production refreshes (metadata only): per ordinary layer the uvqk weight
    as the U slice + the VQK slice (QKV-fused forward, tpa) or whole, and the
    output-projection weight (triton_hstu_linear); the targets-only layer's
    uvqk weight as the V|K layout. Distinct entries only."""
    out = []
    seen = set()

    def add(w: Any, layout: Tuple[Tuple[int, int], ...]) -> None:
        shape = (int(w.shape[0]), int(w.shape[1]))
        key = (shape, w.dtype, layout)
        if w.dtype in (torch.float32, torch.bfloat16) and key not in seen:
            seen.add(key)
            out.append(key)

    u, v, q = H * Dv, H * Dv, H * Dq
    for m in main_layers:
        uw = m._uvqk_weight
        n = int(uw.shape[1])
        if fuse_qkv:
            add(uw, ((0, u),))
            add(uw, ((u, n),))
        else:
            add(uw, ((0, n),))
        ow = m._output_weight
        add(ow, ((0, int(ow.shape[1])),))
    if targets_only:
        uw = last._uvqk_weight
        n = int(uw.shape[1])
        add(uw, ((u, u + v), (u + v + q, n)))
    return out


def _fin_rpp_buckets(B: int, n_max: int, ak: int) -> List[Tuple[int, int]]:
    """[(RPP, M)] for every RPP a sink finalizer whose amax partials have `ak`
    columns takes for a layer call of M = 0 .. n_max rows at batch B, with M
    the largest such row count <= n_max. The wrappers' rule (hstu_dkdv_e5m2
    partial_rows_bound + DkdvE5m2Sink.finish, hstu_dq_e5m2 _dq8_partials +
    _dq8_finalize; 256 = their _FIN_PROGS): rb = (M + 127 B) // 128 + 1
    partial rows, RPP = blk * ceil(ceil(rb / 256) / blk) with blk = 8 if
    ak <= 128 else 4 if ak <= 512 else 1."""
    blk = 8 if ak <= 128 else (4 if ak <= 512 else 1)
    rb_max = (n_max + 127 * B) // 128 + 1
    top: Dict[int, int] = {}
    for j in range(1, -(-rb_max // 256) + 1):  # j = ceil(rb / 256)
        m = min(128 * 256 * j - 1 - 127 * B, n_max)  # largest M with that j
        if m >= 0:
            top[blk * -(-j // blk)] = m
    return sorted(top.items())


def _fin_rpp_prebuild(dev: torch.device, B: int, H: int, D: int, n_max: int,
                      e8: bool, dq8: bool) -> str:
    """Item F2r: the engaged call's finalizers, through its production
    wrappers, once per RPP bucket of M = 0 .. n_max rows (_fin_rpp_buckets).
    The partials are allocated by the wrappers for the bucket's largest M and
    left uninitialized; the row prefix is built from all-zero seq_offsets, so
    R = pfx[B] = 0 masks every partial row (nothing is read) and the outputs
    are local and dropped. Same L1 key as the engaged call: same wrappers,
    dtypes and constexprs, fresh < 2 GiB allocations (pointer class DS), B the
    production batch size (its % 16 class)."""
    from generative_recommenders.ops.flydsl.hstu_dkdv_e5m2 import DkdvE5m2Sink
    from generative_recommenders.ops.triton import hstu_dq_e5m2 as dqe

    # int64 [B + 1] as production's seq_offsets: the prefix kernels reuse
    # F2e's / F2q's compile.
    so0 = torch.zeros(B + 1, dtype=torch.int64, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    z8 = torch.zeros(1, dtype=torch.uint8, device=dev)
    done: List[str] = []
    # Largest bucket first: the smaller partials then reuse its cached block.
    if e8:
        rows = _fin_rpp_buckets(B, n_max, H * 32)  # pamax [rb, H * 32]
        for _rpp, m in reversed(rows):
            sink = DkdvE5m2Sink(z8, one)
            sink.launch_kwargs(so0, m, H, D)  # _e8_pfx_kernel + partials
            sink.finish()  # _e8_fin_kernel at this RPP
        done.append(f"e8_rpp={rows[0][0]}..{rows[-1][0]}/{len(rows)}")
    if dq8:
        rows = _fin_rpp_buckets(B, n_max, H)  # pamax [rb * H]
        for _rpp, m in reversed(rows):
            pfx, pcol, pamax = dqe._dq8_partials(so0, m, H, D)
            dqe._dq8_finalize(pamax, pcol, pfx)  # _dq8_fin_kernel
        done.append(f"dq8_rpp={rows[0][0]}..{rows[-1][0]}/{len(rows)}")
    return " ".join(done)


# Ship17 item (111) _pk_nib V3: the pk writer selector's line marks
# (triton_hstu_linear; each holds the line it printed, "" = none yet).
_PK_V3_MARKS: Tuple[str, ...] = ("_LNMD_MASK_NIB_PK_V3_ARMED_LOGGED",
                                 "_LNMD_MASK_NIB_PK_V3_OFF_LOGGED")


def _pk_v3_marks(thl: Any) -> Dict[str, Any]:
    """The (111) line marks now: the `before` of _pk_v3_reprint."""
    return {name: getattr(thl, name, "") for name in _PK_V3_MARKS}


def _pk_v3_reprint(thl: Any, before: Dict[str, Any]) -> int:
    """Ship17 item (111) _pk_nib V3: print every line the pk writer selector
    (triton_hstu_linear._lnmd_mask_nib_pk_kernel) latched since `before`, i.e.
    inside one L4n item, whose stdout capture dropped it: the ARMED line (knob
    "1") or the OFF / NOT armed line (any other value: a gin False binding,
    the env kill switch "0"). Called after the item, outside the capture,
    through the module's own arm_print_all_ranks (every rank, tagged [rank N]:
    the text the selector printed). The marks are kept (_KEEP_GLOBALS), so
    this is the process's one (111) line, before RUN_START, and the first
    training launch prints none. Returns the count."""
    n = 0
    for name in _PK_V3_MARKS:
        line = getattr(thl, name, "")
        if isinstance(line, str) and line and line != before.get(name):
            thl.arm_print_all_ranks(line)
            n += 1
    return n


def _nib_items(r: _Runner, thl: Any, ol: Any,
               ln_wb: Callable[[int, torch.dtype], Tuple[torch.Tensor, torch.Tensor]],
               Do: int, ratio: float, training: bool, fuse: bool, inv: float,
               big_rows: List[int]) -> None:
    """Item L4n (r16 NIB, gin item (94), DLRM_HSTU_FP8_LNMD_MASK_NIB=1
    only; knob absent or not "1": returns at once, nothing runs, logs or
    counts). The two NIB keep-mask writers, _ln_mul_dropout_fwd_maskgen_nib
    (layer 0) and _ln_mul_dropout_fwd_maskgen_pk_nib (MGPK's layers), compile
    here, before RUN_START. L4 cannot reach them through the production
    wrapper: the NIB gate (triton_hstu_linear._lnmd_mask_nib_why_not) arms
    only for an HSTUComputeOutputFunction FP8 call whose x carries a dO8 site,
    and L4's synthetic x has neither that call's out-proj context nor a site.
    So lnmd_mask_nib_warmup_launch launches both kernels with item (28)(e)'s
    production launch arguments (no gate, no NIB / MGPK / maskgen state
    touched) at L4's large-row sizes plus _ROWS_XL and _ROWS_XL + 1 whatever
    FP8_WARMUP_STORES says, with L4's two seeds: every specialization class a
    training step can hit (N % 16, seed % 16, and x / u below and above
    2 GiB; the [rows, 1536] e4m3 y is above 2 GiB at every size), because the
    checked-in TSPEC table has no rows for these kernels yet (the maskgen
    kernels get their > 2 GiB x / u class from it). Transient memory at
    _ROWS_XL ~8.5 GiB per item, freed after it. Skipped (one skip line) where
    item (28)(e) cannot arm: no dropout, no fused silu, no large-row kernel,
    maskgen or mask16 off, not gfx950.
    STARTUP BYTE CHECK (review R1-F5): on L4's rows (not _ROWS_XL, ~9 GiB
    transient at _ROWS_L) the launch helper also runs the int8 kernel each
    writer replaces (the shipped maskgen / MGPK's kernel; the specializations
    L4 just compiled) on the same inputs and compares every output byte. A
    mismatch latches NIB for the process: triton_hstu_linear.
    lnmd_mask_nib_latch, called here after the item, outside its stdout
    capture, so its DISABLED line prints; fp8_warmup keeps the latch out of
    the scalar restore (_KEEP_GLOBALS). L4n then stops (NIB cannot run).
    Ship17 item (111): the first item's pk launch latches the pk writer line
    inside the item's capture; _pk_v3_reprint prints it after that item."""
    try:
        from generative_recommenders.ops import lnmd_mask_nib as nibm

        if not nibm.requested():
            return
    except Exception:
        return
    warm = getattr(thl, "lnmd_mask_nib_warmup_launch", None) if thl is not None else None
    why: Optional[str] = None
    if warm is None:
        why = "no NIB writers in triton_hstu_linear"
    elif not (training and fuse and big_rows):
        why = ("item (28)(e) maskgen cannot arm (needs dropout > 0, the fused "
               "silu and the large-row kernel)")
    elif (os.environ.get("DLRM_HSTU_FP8_LNMD_FWD_MASKGEN", "0") == "0"
          or os.environ.get("DLRM_HSTU_FP8_MASK16", "0") != "1"):
        why = "item (28)(e) maskgen or mask16 off"
    else:
        dev = r.device
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        arch_of = getattr(thl, "_gcn_arch_name", None)
        arch = str(arch_of(index)) if arch_of is not None else ""
        if not arch.startswith("gfx950"):
            why = f"arch {arch or '<unknown>'} is not gfx950 (maskgen does not arm)"
    if why is not None:
        r.skip("L4n", "lnmd_mask_nib_writers", why)
        return
    bf16 = torch.bfloat16
    onw_dtype = ol._output_norm_weight.dtype
    rows = list(big_rows) + [t for t in (_ROWS_XL, _ROWS_XL + 1) if t not in big_rows]
    latch = getattr(thl, "lnmd_mask_nib_latch", None)
    for Tp in rows:
        # the startup byte check on L4's rows only (not the > 2 GiB x / u class)
        cmp = latch is not None and Tp not in (_ROWS_XL, _ROWS_XL + 1)
        for seed in _SEEDS:
            verdict: Dict[str, str] = {}

            def l4n(Tp: int = Tp, seed: int = seed, cmp: bool = cmp,
                    verdict: Dict[str, str] = verdict) -> Optional[str]:
                w, b = ln_wb(Do, onw_dtype)
                kw: Dict[str, Any] = dict(compare=True, verdict=verdict) if cmp else {}
                return warm(
                    x=r.bufs.act((Tp, Do), bf16), u=r.bufs.act((Tp, Do), bf16),
                    weight=w, bias=b, eps=_LN_EPS, dropout_ratio=ratio, seed=seed,
                    inv_scale=inv, silu_u=fuse, **kw)

            pk_v3 = _pk_v3_marks(thl)
            r.run("L4n", "lnmd_mask_nib_writers",
                  f"T={Tp} seed={seed} store=rows" + (" cmp=int8" if cmp else ""),
                  l4n)
            # ship17 item (111): the pk writer line this item latched (its
            # stdout capture dropped it) prints here, before RUN_START, once
            # (the marks are kept: _KEEP_GLOBALS)
            _pk_v3_reprint(thl, pk_v3)
            if verdict.get("mismatch"):
                # outside the item's stdout capture: the DISABLED line prints;
                # the latch outlives the restore (_KEEP_GLOBALS)
                latch(verdict["mismatch"])
                return


def _spf_armed_reprint(mha: Any, before: Dict[str, str]) -> int:
    """Ship17 item (105) SPF: print every armed line that
    hstu_flydsl_mha.fp8_fwd_spf_config latched (mode -> line) since `before`.
    Called after F1, outside the item's stdout capture, where the line was
    dropped; the latch is kept (_KEEP_GLOBALS), so this is the process's one
    SPF arming line and none prints in the timed region. Returns the count."""
    n = 0
    for mode, line in mha._fp8_fwd_spf_logged.items():
        if mode not in before:
            print(line, flush=True)
            n += 1
    return n


def _spf_bwd_armed_reprint(mha: Any, before: Dict[Any, str]) -> int:
    """Ship17 item (106) BWD_SPF: log every line that hstu_flydsl_mha's
    backward SPF hook latched (key -> line: the ARMED line, a stray-tsc strip
    line) since `before`, as the hook does (mha logger, WARNING). Called after
    _Guards.restore(): F2/F2q/F2e reach the hook while _Guards mutes that
    logger, so the line was dropped there; the latch is kept (_KEEP_GLOBALS),
    so this is the process's one BWD_SPF ARMED line and none prints in the
    timed region. Returns the count."""
    n = 0
    for key, line in mha._fp8_bwd_spf_logged.items():
        if key not in before:
            mha.logger.warning("[hstu-attn] %s", line)
            n += 1
    return n


def _warm_items(r: _Runner, model: torch.nn.Module, B: int, tier: int,
                stores: Tuple[bool, ...]) -> Dict[str, Any]:
    from generative_recommenders.modules import hstu_transducer as htr
    from generative_recommenders.modules import stu as stu_mod
    from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
    from generative_recommenders.ops.triton import bwd_fp8
    from generative_recommenders.ops.triton import triton_fp8 as tf8
    from generative_recommenders.ops.triton import (
        triton_hstu_preprocess_and_attention as tpa,
    )
    from generative_recommenders.ops.triton.triton_hstu_attention import (
        _should_enable_tma,
        triton_cached_hstu_mha,
        triton_hstu_attention_delta_bwd,
        triton_hstu_attention_fwd,
    )
    from generative_recommenders.ops.triton.triton_hstu_linear import (
        triton_layer_norm_mul_dropout_fwd,
    )
    from generative_recommenders.ops.triton.triton_layer_norm import (
        triton_weighted_layer_norm_dual_fwd,
        triton_weighted_layer_norm_fp8_save_bf16_fwd,
        triton_weighted_layer_norm_fwd,
    )

    dev = r.device
    bufs = r.bufs
    bf16 = torch.bfloat16
    f32 = torch.float32
    e4m3 = tf8.fp8_e4m3_dtype()
    both = True in stores

    transducer, stack, layers = _find_layers(model)
    if not layers:
        raise RuntimeError("no STULayer in the model")
    layer = layers[0]
    # Metadata only (no parameter value is read).
    H = int(layer._num_heads)
    Dq = int(layer._attention_dim)
    Dv = int(layer._hidden_dim)
    Dm = int(layer._embedding_dim)
    alpha = float(layer._attn_alpha)  # FlyDSL compile key: must be bit-exact.
    listwise = bool(getattr(transducer, "_listwise_training", False))

    # Targets-only last layer (hstu_transducer._resolve_targets_only): knob on
    # and the configuration supports it (raises in production otherwise).
    targets_only = False
    to_reason = "TRITON_HSTU_LAST_LAYER_TARGETS_ONLY off"
    if getattr(htr, "_HSTU_LAST_LAYER_TARGETS_ONLY", False):
        try:
            to_reason = (
                transducer._targets_only_unsupported_reason()
                if transducer is not None
                else stack.targets_only_unsupported_reason()
            )
        except Exception as error:
            to_reason = f"reason check failed: {error}"
        targets_only = to_reason is None
    main_layers = layers[:-1] if targets_only else list(layers)
    last = layers[-1]

    T = 2 * B
    # seq_offsets int64 cumsum (production carries int64; FlyDSL casts to
    # int32 itself, the Triton dq-split / delta kernels take the raw pointer).
    so = torch.arange(0, T + 1, 2, dtype=torch.int64, device=dev)
    # One target per sequence (HAS_MULTIPLE_TARGETS=True). int64 is an
    # assumption from the KJT lengths (yambda.py builds them dtype=long).
    nt = torch.ones(B, dtype=torch.int64, device=dev)
    nt_main = None if listwise else nt
    s = _QKV_SCALE
    inv = _LN_INV_SCALE
    store_names = {False: "small", True: "big"}

    fp8_fwd_on = os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1"

    # ship10 (23) FP8 uvqk/vk backward: which FP8-step kernels replace the
    # ship7 ones (getattr: absent on older trees = the ship7 path).
    def _call(obj: Any, name: str, default: Any) -> Any:
        fn = getattr(obj, name, None)
        try:
            return fn() if fn is not None else default
        except Exception:
            return default

    vk_producer = bool(_call(bwd_fp8, "vk_producer_enabled", False)) and (
        (H * Dv + H * Dq) % 256 == 0)
    uvqk_fp8t = bool(_call(bwd_fp8, "uvqk_fp8t_requested", False))
    store_t = bool(_call(bwd_fp8, "uvqk_store_fp8t", False))
    tm_pad = int(_call(bwd_fp8, "uvqk_tm_pad", 128) or 128)
    grad_dtype, grad_limit = _call(bwd_fp8, "grad_dtype_and_limit", (None, 0.0))
    if grad_dtype is None:
        vk_producer = False
    if vk_producer:
        from generative_recommenders.ops.triton.hstu_delta_bwd_e5m2 import (
            hstu_attention_delta_bwd_e5m2,
        )

    def vqk8(big: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Production fused vqk: scaled_mm_fp8_out_preq(...).contiguous() [T,
        # 3*H*D] e4m3, split v, q, k (tpa.py forward), row stride 1536.
        buf = bufs.act((T, H * Dv + 2 * H * Dq), e4m3, big)
        v, q, k = buf.split([H * Dv, H * Dq, H * Dq], dim=1)
        return q.view(T, H, Dq), k.view(T, H, Dq), v.view(T, H, Dv)

    # Production's per-launch shape predicates (try_flydsl_attention_fwd_fp8
    # raises, try_flydsl_attention_bwd falls back to Triton), evaluated on
    # meta tensors so nothing is allocated or launched for a shape production
    # would never send to FlyDSL. num_targets is not known here (None).
    attn_l = main_layers[0] if main_layers else layer
    mal = int(attn_l._max_attn_len)
    csl = int(attn_l._contextual_seq_len)

    def meta(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="meta")

    def fly_shape_reason(qk_dtype: torch.dtype, bwd: bool) -> Optional[str]:
        try:
            q = meta((T, H, Dq), qk_dtype)
            k = meta((T, H, Dq), qk_dtype)
            v = meta((T, H, Dv), qk_dtype)
            if bwd:
                return mha.unsupported_shape_reason(
                    q, k, v, meta((T, H, Dv), bf16), None, mal, csl)
            return mha.unsupported_fp8_fwd_shape_reason(q, k, v, None, mal, csl)
        except Exception as error:
            return f"shape check failed: {error}"

    # ---- F1: FlyDSL e4m3 attention forward -------------------------------
    # Ship-11 (26)(a): with DLRM_HSTU_FP8_FWD_PIPE=1 the wrapper compiles the
    # BM128/waves4 pipelined config (read per call); its latch is restored.
    f1_reason = fly_shape_reason(e4m3, bwd=False)
    if not fp8_fwd_on:
        r.skip("F1", "flydsl_fwd_fp8", "DLRM_HSTU_FP8_FWD off")
    elif not (tf8.hstu_fp8_fuse_qkv_enabled() and tf8.hstu_fp8_attention_enabled()):
        r.skip("F1", "flydsl_fwd_fp8", "FUSE_QKV/ATTN off")
    elif not mha.flydsl_fp8_fwd_selected():
        r.skip("F1", "flydsl_fwd_fp8", "HSTU_ATTN_FWD!=flydsl")
    elif f1_reason is not None:
        r.skip("F1", "flydsl_fwd_fp8", f"unsupported shape: {f1_reason}")
    else:
        def f1() -> None:
            mha.require_flydsl_fwd()
            q8, k8, v8 = vqk8()
            # flydsl_hstu_attention_fwd_fp8(q, k, v, seq_offsets, N, alpha,
            # qkv_scale): compile key is (heads, dim, alpha, rs, cfg); N and
            # the scale are launch args, so one call covers every N (and,
            # under item (105) SPF, every grid exponent j). With SPF on, the
            # forward accepts only a scale on the SPF grid, so F1 passes the
            # snapped synthetic scale (the same 2**-6 with the knob off).
            mha.flydsl_hstu_attention_fwd_fp8(
                q8, k8, v8, so, _N_A, alpha,
                mha.fp8_fwd_spf_warmup_scale(s, alpha))

        spf_before = dict(mha._fp8_fwd_spf_logged)
        r.run("F1", "flydsl_fwd_fp8", f"N={_N_A} store=small", f1)
        # Item (105) SPF: the armed line F1 latched went into the item's
        # stdout capture; print it here, outside the capture, on every rank.
        _spf_armed_reprint(mha, spf_before)

    # ---- F2: FlyDSL full-FP8 attention backward ---------------------------
    # Ship-11: the item (27) VALU / MSPLIT_SCHED / DQ_SPLIT_OPT tokens and the
    # ws store layout are read per call by the wrapper (their ARMED latches
    # live in _logged_layouts / _DQ_SPLIT_*_LOGGED, restored). Production's
    # dq-split operands (workspace, k, dq) are > 2 GiB storages, so F2 runs on
    # the >2 GiB class (arena views + a >2 GiB persistent workspace).
    # F2q: (25)(d) DQ_E5M2, the ordinary layer hands the backward a DqE5m2Sink
    # (tpa backward) and the dq-split runs _dq8_sched_kernel; then the sink's
    # u|v|k-only cast and partial reduction (finish).
    n_list = [_N_A, _N_C] + ([_N_D] if os.environ.get("FP8_WARMUP_N11", "1").strip().lower()
                             not in ("0", "false", "off", "no") else [])
    W_uvqk = 2 * H * Dv + 2 * H * Dq
    q_lo = 2 * H * Dv
    dq_sink_reason: Optional[str] = None
    if not bool(_call(bwd_fp8, "dq_producer_enabled", False)):
        dq_sink_reason = "DLRM_HSTU_FP8_BWD_DQ_E5M2 off (or no fp8t wgrad)"
    elif grad_dtype is None:
        dq_sink_reason = "no FP8 gradient dtype"
    elif not (W_uvqk % 256 == 0 and q_lo % 256 == 0 and (H * Dq) % 256 == 0):
        dq_sink_reason = "duvqk layout not whole 256-column blocks"
    elif bool(_call(tpa, "_dq_sink_diagnostics_on", False)):
        dq_sink_reason = "FlyDSL backward diagnostics on"
    if not mha.flydsl_selected():
        r.skip("F2", "flydsl_bwd_full_fp8", "HSTU_ATTN_BWD!=flydsl")
    elif not mha.bwd_full_fp8_enabled():
        r.skip("F2", "flydsl_bwd_full_fp8", "DLRM_HSTU_FP8_BWD_FULL off")
    elif not (fp8_fwd_on and tf8.hstu_fp8_fuse_qkv_enabled()
              and tf8.hstu_fp8_attention_enabled()):
        r.skip("F2", "flydsl_bwd_full_fp8", "no e4m3 q/k/v (FP8 fwd arm off)")
    elif not main_layers:
        r.skip("F2", "flydsl_bwd_full_fp8", "no main HSTU layer")
    elif fly_shape_reason(e4m3, bwd=True) is not None:
        r.skip("F2", "flydsl_bwd_full_fp8",
               "unsupported shape: %s" % fly_shape_reason(e4m3, bwd=True))
    else:
        if dq_sink_reason is not None:
            r.skip("F2q", "flydsl_bwd_dq_e5m2", dq_sink_reason)
        for N in n_list:
            def f2(N: int = N) -> None:
                mha.require_flydsl()
                with _big_dq_split_workspace(dev):
                    q8, k8, v8 = vqk8(True)
                    # Production: duvqk [T, uvqk_weight.shape[1]] bf16 split
                    # du, dv, dq, dk (rs_out 2048); out= is what keeps rs_out.
                    duvqk = bufs.act((T, W_uvqk), bf16, True)
                    _du, dv, dq, dk = duvqk.split([H * Dv, H * Dv, H * Dq, H * Dq], 1)
                    dout = bufs.act((T, H, Dv), bf16)
                    # Under the BWD_SPF lab knob (default off) the backward
                    # accepts only a scale on the SPF grid, so F2/F2q/F2e pass
                    # the snapped synthetic scale (the same 2**-6 when off).
                    mha.flydsl_hstu_attention_bwd(
                        dout, q8, k8, v8, so, N, alpha,
                        out=(dq.view(T, H, Dq), dk.view(T, H, Dq), dv.view(T, H, Dv)),
                        qkv_scale=mha.fp8_bwd_spf_warmup_scale(s, alpha),
                    )

            r.run("F2", "flydsl_bwd_full_fp8", f"N={N} store=big", f2)
            if dq_sink_reason is not None:
                continue

            def f2q(N: int = N) -> None:
                from generative_recommenders.ops.triton import hstu_dq_e5m2 as dqe

                mha.require_flydsl()
                with _big_dq_split_workspace(dev):
                    q8, k8, v8 = vqk8(True)
                    duvqk = bufs.act((T, W_uvqk), bf16, True)
                    _du, dv, dq, dk = duvqk.split([H * Dv, H * Dv, H * Dq, H * Dq], 1)
                    dout = bufs.act((T, H, Dv), bf16)
                    # Same constructor arguments as the tpa backward (per-layer
                    # scale pair -> fresh [1] f32 tensors here).
                    sink = dqe.DqE5m2Sink(
                        rows=T, width=W_uvqk, q_cols=(q_lo, q_lo + H * Dq),
                        inv_scale=torch.ones(1, dtype=f32, device=dev),
                        cast_scale=torch.ones(1, dtype=f32, device=dev),
                        pad=tm_pad, grad_dtype=grad_dtype, limit=grad_limit,
                        device=dev)
                    # Production's dz8 [rows_out, 2048] is > 2 GiB as well.
                    sink.dz8 = bufs.act((sink.rows_out, W_uvqk), grad_dtype, True)
                    mha.flydsl_hstu_attention_bwd(
                        dout, q8, k8, v8, so, N, alpha,
                        out=(dq.view(T, H, Dq), dk.view(T, H, Dq), dv.view(T, H, Dv)),
                        qkv_scale=mha.fp8_bwd_spf_warmup_scale(s, alpha),
                        dq_fp8_sink=sink,
                    )
                    if not sink.filled:
                        raise RuntimeError(
                            f"dq_e5m2 sink not used (declined={sink.declined})")
                    sink.finish(duvqk)
                # The u|v|k-only cast at the odd row class (row counts vary).
                rows = T + 1
                rows_out = (rows + tm_pad - 1) // tm_pad * tm_pad
                dqe.quantize_amax_colsum_skip(
                    bufs.act((rows, W_uvqk), bf16, True),
                    torch.ones(1, dtype=f32, device=dev),
                    bufs.act((rows_out, W_uvqk), grad_dtype, True),
                    (q_lo, q_lo + H * Dq))

            r.run("F2q", "flydsl_bwd_dq_e5m2", f"N={N} store=big", f2q)

        # ---- F2e: R15 M-E1 e8 build of the same backward (ship-13 rung) -----
        # DLRM_HSTU_FP8_BWD_DKDV_E5M2=1 (gin item (47); the predicate tpa
        # reads, after the env bootstrap): every engaged ordinary-layer call
        # hands this backward a DkdvE5m2Sink and its config gains
        # e8emit="only", a SECOND compile_hstu_bwd_dkdv build (lru_cached on
        # its arguments; c = ceil(log2 N) enters through mu2_ssc/mu2_dsc as in
        # F2, so N=2992 and 4096 share one build and N11 adds the c=11 one).
        # Without this item it compiles at the first engaged call, after the
        # per-layer scale warm-up, inside the timed region. F2e is F2's call
        # with the sinks of an engaged call (tpa backward): the E1 sink over a
        # dz8 [roundup(T, uvqk_tm_pad), 4*H*D] (tpa._dkdv_e5m2_arm; the >2 GiB
        # class, as F2q's) and, where F2q runs, the dq sink adopting that dz8
        # (pad_zeroed=True). Then what tpa._dkdv_e5m2_prequant runs: the cast
        # of the columns no producer wrote (ship-12 joint path: none, u comes
        # from B1a, q from the dq sink, v|k from E1; dq sink without B1a: u;
        # B1a without the dq sink: q; neither: u|q; a cast also runs at the odd
        # row class T+1, as F2q's), the dq sink's finish_q (_dq8_fin) and the
        # E1 finish (_e8_fin); _e8_pfx runs in the launch.
        # Synthetic zeros and a ones(1) scale; both sinks are local objects
        # and no tpa helper runs (only its env predicates are read), so its
        # E1 counters, log/latch sets and the per-layer scale epochs are
        # untouched (production prints its E1 ARMED/engaged lines at the
        # first engaged call); the mha latches and the dq-split workspace
        # are put back by _Guards as for F2. A sink
        # decline raises (FAIL line; the run continues). The ok line prints
        # on every rank: bwd= the synchronized attention backward (the e8
        # compile when the FlyDSL cache is cold), fin= the cast and finishes,
        # e8_builds_new= compile_hstu_bwd_dkdv cache misses during the call.
        tag_e, name_e = "F2e", "flydsl_bwd_full_fp8_e8"
        b1a = bool(_call(bwd_fp8, "du_producer_enabled", False))
        joint = dq_sink_reason is None

        def e1_layout_ok() -> bool:
            # tpa._dkdv_e5m2_arm's layout checks, on metadata only.
            try:
                uw = attn_l._uvqk_weight
                bn = int(getattr(bwd_fp8, "_COLSUM_BLOCK_N", 256))
                return (Dq == Dv and W_uvqk == 4 * H * Dv and (H * Dv) % bn == 0
                        and uw.dim() == 2 and int(uw.shape[1]) == W_uvqk)
            except Exception:
                return False

        if os.environ.get("DLRM_HSTU_FP8_WARMUP_F2E", "1").strip() == "0":
            # Kill switch of the gin binding apply_env_bootstrap.
            # DLRM_HSTU_FP8_WARMUP_F2E (absent or "1" = run, the code default).
            r.skip(tag_e, name_e, "DLRM_HSTU_FP8_WARMUP_F2E=0")
        elif not bool(_call(tpa, "_dkdv_e5m2_requested", False)):
            r.skip(tag_e, name_e, "DLRM_HSTU_FP8_BWD_DKDV_E5M2 off")
        elif grad_dtype is None or not (
                uvqk_fp8t and _call(bwd_fp8, "uvqk_fp8_backward_enabled", False)):
            r.skip(tag_e, name_e, "no FP8 uvqk backward with the fp8t wgrad")
        elif not e1_layout_ok():
            r.skip(tag_e, name_e, "duvqk layout (the E1 arm declines it)")
        elif bool(_call(tpa, "_dq_sink_diagnostics_on", False)):
            r.skip(tag_e, name_e, "FlyDSL backward diagnostics on")
        else:
            path = ("b1a_" if b1a else "") + ("joint" if joint else "solo")

            def dkdv_builds() -> int:
                # compile_hstu_bwd_dkdv's lru_cache misses (read only).
                try:
                    build = getattr(getattr(mha, "_kernel", None),
                                    "compile_hstu_bwd_dkdv", None)
                    return int(build.cache_info().misses)
                except Exception:
                    return -1

            for N in n_list:
                # so: bound at definition (this function deletes it at the end).
                def f2e(N: int = N, so: torch.Tensor = so) -> str:
                    from generative_recommenders.ops.flydsl.hstu_dkdv_e5m2 import (
                        DkdvE5m2Sink,
                    )
                    from generative_recommenders.ops.triton import hstu_dq_e5m2 as dqe

                    hd = H * Dv
                    mha.require_flydsl()
                    with _big_dq_split_workspace(dev):
                        q8, k8, v8 = vqk8(True)
                        duvqk = bufs.act((T, W_uvqk), bf16, True)
                        _du, dv, dq, dk = duvqk.split([hd, hd, H * Dq, H * Dq], 1)
                        dout = bufs.act((T, H, Dv), bf16)
                        rows_out = (T + tm_pad - 1) // tm_pad * tm_pad
                        dz8 = bufs.act((rows_out, W_uvqk), grad_dtype, True)
                        if rows_out > T:
                            dz8[T:].zero_()  # tpa: the pad rows zeroed once
                        one = torch.ones(1, dtype=f32, device=dev)
                        e8s = DkdvE5m2Sink(dz8, one, decode_scale=one)
                        sinks: Dict[str, Any] = {"dkdv_fp8_sink": e8s}
                        dqs = None
                        if joint:
                            dqs = dqe.DqE5m2Sink(
                                rows=T, width=W_uvqk, q_cols=(q_lo, q_lo + H * Dq),
                                inv_scale=one, cast_scale=one, pad=tm_pad,
                                grad_dtype=grad_dtype, limit=grad_limit,
                                device=dev, dz8=dz8, pad_zeroed=True)
                            sinks["dq_fp8_sink"] = dqs
                        # BWD_SPF lab knob (default off): F2's snapped scale.
                        s_bwd = mha.fp8_bwd_spf_warmup_scale(s, alpha)
                        n0 = dkdv_builds()
                        torch.cuda.synchronize(dev)
                        t0 = time.perf_counter()
                        mha.flydsl_hstu_attention_bwd(
                            dout, q8, k8, v8, so, N, alpha,
                            out=(dq.view(T, H, Dq), dk.view(T, H, Dq), dv.view(T, H, Dv)),
                            qkv_scale=s_bwd, **sinks,
                        )
                        torch.cuda.synchronize(dev)
                        t_bwd = time.perf_counter() - t0
                        n1 = dkdv_builds()
                        if e8s.declined is not None or not e8s.filled:
                            raise RuntimeError(
                                f"E1 sink not engaged (declined={e8s.declined})")
                        dq_note = ""
                        if dqs is not None and not dqs.filled:
                            # tpa: the engaged call continues without it.
                            dq_note = f" dq_sink=unused({dqs.declined})"
                            dqs = None
                        # The cast tpa._dkdv_e5m2_prequant runs before the
                        # finishes: none with the dq sink AND B1a (ship-12
                        # joint path); u with the dq sink only; q with B1a
                        # only; u|q (keep-list kernel) with neither.
                        has_cast = dqs is None or not b1a

                        def cast(value: torch.Tensor, out: torch.Tensor) -> None:
                            if dqs is not None:
                                dqe.quantize_amax_colsum_skip(
                                    value, one, out, (hd, 4 * hd))
                            elif b1a:
                                dqe.quantize_amax_colsum_skip(
                                    value, one, out, [(0, 2 * hd), (3 * hd, 4 * hd)])
                            else:
                                bwd_fp8.quantize_amax_colsum_select(
                                    value, one, out, group_cols=hd)

                        t0 = time.perf_counter()
                        if has_cast:
                            cast(duvqk, dz8)
                        if dqs is not None:
                            dqs.finish_q()
                        e8s.finish()
                        torch.cuda.synchronize(dev)
                        t_fin = time.perf_counter() - t0
                    if has_cast:
                        # The same cast at the odd row class (F2q's rule: Triton
                        # specializes its row-count arguments on % 16; T+1 is
                        # odd, the class most production row counts are in,
                        # T = 2B the % 16 class when B % 8 == 0).
                        rows = T + 1
                        rows_o = (rows + tm_pad - 1) // tm_pad * tm_pad
                        cast(bufs.act((rows, W_uvqk), bf16, True),
                             bufs.act((rows_o, W_uvqk), grad_dtype, True))
                    return (f"bwd={t_bwd:.2f}s fin={t_fin:.2f}s e8_builds_new="
                            f"{n1 - n0 if min(n0, n1) >= 0 else -1}{dq_note}")

                r.run(tag_e, name_e, f"N={N} store=big path={path}", f2e,
                      all_ranks=True)

        # ---- F2r: every RPP bucket of the sink finalizers (item (86)) ----
        # gin item (86). F2e/F2q finish at T = 2B, i.e. RPP 8 only; RPP follows
        # the live row count M, so an engaged step whose M reaches another RPP
        # would compile _e8_fin_kernel / _dq8_fin_kernel inside the timed
        # region (seen in profiled 1n benchmark steps). F2r covers every
        # bucket from config alone: both finalizers once per RPP of M = 0 ..
        # B * _N_C (any batch of B sequences of at most max_seq_len tokens):
        # _fin_rpp_prebuild. e8 under F2e's preconditions (not its kill
        # switch), dq8 under F2q's. Kill switch:
        # DLRM_HSTU_FP8_WARMUP_FIN_RPP=0 (or FP8_WARMUP_SKIP=F2r).
        e8_fin = (bool(_call(tpa, "_dkdv_e5m2_requested", False))
                  and grad_dtype is not None and uvqk_fp8t
                  and bool(_call(bwd_fp8, "uvqk_fp8_backward_enabled", False))
                  and e1_layout_ok()
                  and not bool(_call(tpa, "_dq_sink_diagnostics_on", False)))
        dq8_fin = dq_sink_reason is None
        if os.environ.get("DLRM_HSTU_FP8_WARMUP_FIN_RPP", "1").strip() == "0":
            # Kill switch of the gin binding (absent or "1" = run it).
            r.skip("F2r", "sink_fin_rpp", "DLRM_HSTU_FP8_WARMUP_FIN_RPP=0")
        elif not (e8_fin or dq8_fin):
            r.skip("F2r", "sink_fin_rpp", "neither the E1 nor the dq e5m2 sink")
        else:
            def f2r() -> str:
                return _fin_rpp_prebuild(dev, B, H, Dq, B * _N_C, e8_fin, dq8_fin)

            r.run("F2r", "sink_fin_rpp", f"B={B} M<={B * _N_C}", f2r,
                  all_ranks=True)

    # ---- T1/T2: targets-only delta attention, e4m3 V/K --------------------
    # Ship-11: (26)(c) DELTA_FWD_BM16 (T1, read per call, latch restored) and
    # (25)(b) VKP_V2 (T2e: _hstu_attn_delta_bwd_e5m2_v2 + its longest-first
    # permutation kernel) are reached through the same wrappers.
    def delta_items(tag_f: str, tag_b: str, suffix: str, kv_dtype: torch.dtype,
                    k_scale: float) -> None:
        for N in (_N_A, _N_B):
            for big in stores:
                detail = f"N={N} store={store_names[big]}"

                def fwd(N: int = N, big: bool = big) -> None:
                    # Forward: vk [T, 2*H*D] contiguous split v, k (rs 1024);
                    # uq [B, 2*H*D] contiguous, q = uq[:, H*Dv:] (rs 1024).
                    vk = bufs.act((T, H * Dv + H * Dq), kv_dtype, big)
                    v, k = vk.split([H * Dv, H * Dq], dim=1)
                    uq = bufs.act((B, H * Dv + H * Dq), bf16)
                    q = uq.split([H * Dv, H * Dq], dim=1)[1].view(B, H, Dq)
                    triton_cached_hstu_mha(
                        N=N, alpha=alpha, delta_q=q,
                        k=k.view(T, H, Dq), v=v.view(T, H, Dv),
                        seq_offsets=so, num_targets=nt, max_attn_len=0,
                        contextual_seq_len=0, enable_tma=_should_enable_tma(),
                        q_scale=1.0, k_scale=k_scale, v_scale=k_scale,
                    )

                def bwd(N: int = N, big: bool = big) -> None:
                    vk = bufs.act((T, H * Dv + H * Dq), kv_dtype, big)
                    v, k = vk.split([H * Dv, H * Dq], dim=1)
                    uq = bufs.act((B, H * Dv + H * Dq), bf16)
                    q = uq.split([H * Dv, H * Dq], dim=1)[1].view(B, H, Dq)
                    # Backward: dq = empty_like(q); dV/dK written straight into
                    # a packed dvk [T, 2*H*D] buffer (tpa.py targets bwd).
                    dq = torch.empty_like(q)
                    dvk = bufs.act((T, H * Dv + H * Dq), bf16, big)
                    dv = dvk[:, : H * Dv].view(T, H, Dv)
                    dk = dvk[:, H * Dv:].view(T, H, Dq)
                    dout = bufs.act((B, H, Dv), bf16)
                    triton_hstu_attention_delta_bwd(
                        dout=dout, q=q, k=k.view(T, H, Dq), v=v.view(T, H, Dv),
                        dq=dq, dk=dk, dv=dv, seq_offsets=so, N=N, alpha=alpha,
                        k_scale=k_scale, v_scale=k_scale,
                    )

                r.run(tag_f, "triton_delta_fwd_" + suffix, detail, fwd)
                if tag_b == "T2" and vk_producer:
                    # ship10 VK_PRODUCER: once the vk_projection site is
                    # published (during calibration) the FP8-step backward is
                    # hstu_attention_delta_bwd_e5m2, writing the E5M2 dV|dK
                    # pack padded to uvqk_tm_pad() rows (tpa targets bwd).
                    # Pure function: no bwd_fp8 _sites entry. This kernel takes
                    # no row-count argument, so rows T and T+1 build the same
                    # variant (kept for parity with the other items).
                    for rows in (T, T + 1):
                        def bwd8(N: int = N, big: bool = big, rows: int = rows) -> None:
                            vk = bufs.act((rows, H * Dv + H * Dq), kv_dtype, big)
                            v, k = vk.split([H * Dv, H * Dq], dim=1)
                            uq = bufs.act((B, H * Dv + H * Dq), bf16)
                            q = uq.split([H * Dv, H * Dq], dim=1)[1].view(B, H, Dq)
                            dq = torch.empty_like(q)
                            mp = (rows + tm_pad - 1) // tm_pad * tm_pad
                            dvk8 = bufs.act((mp, H * Dv + H * Dq), grad_dtype, big)
                            dout = bufs.act((B, H, Dv), bf16)
                            hstu_attention_delta_bwd_e5m2(
                                dout=dout, q=q, k=k.view(rows, H, Dq),
                                v=v.view(rows, H, Dv), dq=dq, dvk_fp8=dvk8,
                                rows=rows, seq_offsets=so, N=N, alpha=alpha,
                                inv_scale=torch.ones(1, dtype=torch.float32,
                                                     device=dev),
                                limit=grad_limit, k_scale=k_scale,
                                v_scale=k_scale)

                        r.run("T2e", "triton_delta_bwd_e5m2",
                              f"{detail} rows={rows}", bwd8)
                    continue
                r.run(tag_b, "triton_delta_bwd_" + suffix, detail, bwd)

    if not targets_only:
        r.skip("T1", "triton_delta_fwd_fp8", f"targets-only: {to_reason}")
        r.skip("T2", "triton_delta_bwd_fp8", f"targets-only: {to_reason}")
    elif not (fp8_fwd_on and tf8.hstu_fp8_fuse_qkv_enabled()
              and tf8.hstu_fp8_attention_enabled()):
        r.skip("T1", "triton_delta_fwd_fp8", "FUSE_QKV/ATTN off")
        r.skip("T2", "triton_delta_bwd_fp8", "FUSE_QKV/ATTN off")
    else:
        delta_items("T1", "T2", "fp8", e4m3, s)

    # ---- Q1/Q2: bwd_fp8 casts ---------------------------------------------
    if not bwd_fp8.scale_controller_enabled():
        r.skip("Q1", "bwd_quant_grad_e5m2", "DLRM_HSTU_FP8_BWD off")
        r.skip("Q2", "bwd_quant_weight_e4m3", "DLRM_HSTU_FP8_BWD off")
    else:
        for big in stores:
            def q1(big: bool = big) -> None:
                # Same call as quantize_gradient's published path: default
                # _GRAD_DTYPE/_GRAD_MAX, inv_scale an f32 [1] device tensor
                # (_SiteScale.inv_scale). Pure function: no _sites entry.
                bwd_fp8.quantize_row_major(
                    bufs.act((T, Dm), bf16, big),
                    torch.ones(1, dtype=torch.float32, device=dev),
                )

            r.run("Q1", "bwd_quant_grad_e5m2", f"store={store_names[big]}", q1)

        def q2() -> None:
            w = layer._output_weight  # metadata only: shape and dtype
            bwd_fp8.quantize_weight(
                torch.ones(tuple(w.shape), dtype=w.dtype, device=dev))

        r.run("Q2", "bwd_quant_weight_e4m3", "store=small", q2)

        # Q1d (ship10 (22) FP8 out-proj wgrad): FP8 steps cast the output-
        # projection dgrad with quantize_row_major_dual (row-major E5M2 + a
        # token-major e4m3 copy at inv_scale * 2^-shift), not the plain cast
        # Q1 warms (bwd_fp8.quantize_gradient token_major branch; the plain
        # e5m2 cast never runs in calibration either: the 4th call publishes).
        # T_INV_MUL is a constexpr: SHIFT=auto starts at the floor (5) and may
        # move up to 7 mid-run, one compile per value, so warm floor..7.
        # Pure function: no _sites entry. Rows T and T+1.
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        t_e4m3 = bool(getattr(bwd_fp8, "_OUT_WGRAD_T_E4M3", False))
        if not getattr(thl, "_FP8_OUT_WGRAD", False) or not hasattr(
                bwd_fp8, "quantize_row_major_dual"):
            r.skip("Q1d", "bwd_quant_grad_dual", "DLRM_HSTU_FP8_BWD_OUT_WGRAD off")
        else:
            if t_e4m3:
                s0 = int(getattr(bwd_fp8, "_OUT_WGRAD_DFMT_SHIFT", 7))
                if getattr(bwd_fp8, "_OUT_WGRAD_DFMT_SHIFT_AUTO", False):
                    s0 = min(s0, int(getattr(bwd_fp8, "_OUT_WGRAD_DFMT_SHIFT_FLOOR", s0)))
                shifts: List[Optional[int]] = list(range(s0, 8))
            else:
                shifts = [None]
            for shift in shifts:
                for Tp in (T, T + 1):
                    def q1d(Tp: int = Tp, shift: Optional[int] = shift) -> None:
                        bwd_fp8.quantize_row_major_dual(
                            bufs.act((Tp, Dm), bf16),
                            torch.ones(1, dtype=torch.float32, device=dev),
                            t_dtype=e4m3 if shift is not None else None,
                            t_inv_mul=2.0 ** -shift if shift is not None else 1.0,
                            t_limit=448.0 if shift is not None else None,
                        )

                    r.run("Q1d", "bwd_quant_grad_dual", f"T={Tp} shift={shift}", q1d)

        # Q3 (ship10 (23) uvqk/vk dgrad): quantize_amax_colsum at pad_to =
        # uvqk_tm_pad() when the token-major wgrad runs (calibration uses
        # pad_to=1, a different specialization). Widths: uvqk 2*H*(Dv+Dq),
        # vk H*(Dv+Dq). Pure function. Ship-11 (25)(c): COLSUM_V2 launches
        # the 1-D XCD-grouped kernel through the same wrapper.
        if not uvqk_fp8t or not hasattr(bwd_fp8, "quantize_amax_colsum"):
            r.skip("Q3", "bwd_quant_colsum_tm", "uvqk fp8t wgrad off")
        else:
            # With VK_PRODUCER the vk site gets a prequantized pack, so only
            # the uvqk width runs this cast; vk's is the producer fallback.
            widths = [2 * H * (Dv + Dq)] + ([] if vk_producer else [H * (Dv + Dq)])
            for width in widths:
                for Tp in (T, T + 1):
                    def q3(Tp: int = Tp, width: int = width) -> None:
                        bwd_fp8.quantize_amax_colsum(
                            bufs.act((Tp, width), bf16),
                            torch.ones(1, dtype=torch.float32, device=dev),
                            pad_to=tm_pad)

                    r.run("Q3", "bwd_quant_colsum_tm", f"T={Tp} N={width}", q3)

    # ---- H1: ship-11 (25)(f) BWD_HELPERS_FUSED ----------------------------
    # A published site's scale bookkeeping runs as program 0 of the weight-amax
    # kernel (fp8_helpers_fused._fp8h_amax_tiles_kernel, HAS_SITE/HAS_T/
    # HAS_SNAP/HAS_SNAP_T constexprs) and quantize_weight becomes that kernel
    # plus _fp8h_weight_quant_kernel. Called on a THROWAWAY _SiteScale (fresh
    # tensors, never registered in bwd_fp8._sites) through the same bwd_fp8
    # helpers the sites use, with synthetic weights of the production
    # shape/dtype. Sites and their compile classes:
    #   out:  output_projection_dgrad[_with_operand] (dual-cast partials at T,
    #         T+1; HAS_SNAP/HAS_SNAP_T as production sets them);
    #   uvqk: the M2 dq-sink prequant path (one-element amax), refresh and
    #         non-refresh calls (HAS_SNAP 1/0, the refresh comes at call
    #         DLRM_HSTU_FP8_BWD_REFRESH_CALLS);
    #   uvqk_bf16: the same site when the sink declines (colsum partials);
    #   vk:   the VK-producer prequant path (one-element amax), packed V|K
    #         weight in the activation dtype.
    helpers_on = (bool(getattr(bwd_fp8, "_BWD_HELPERS_FUSED", False))
                  and hasattr(bwd_fp8, "_fused_site_launch")
                  and hasattr(bwd_fp8, "_fused_weight_quant")
                  and hasattr(bwd_fp8, "_SiteScale"))
    if not bwd_fp8.scale_controller_enabled():
        r.skip("H1", "bwd_helpers_fused", "DLRM_HSTU_FP8_BWD off")
    elif not helpers_on:
        r.skip("H1", "bwd_helpers_fused", "DLRM_HSTU_FP8_BWD_HELPERS_FUSED off")
    else:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        def site_call(site: str, partial: torch.Tensor, w: torch.Tensor,
                      snapshot: bool, t_note: bool, t_shift: int,
                      snapshot_t: bool, cast: bool = True) -> None:
            if not bwd_fp8._helpers_fused_ok(w):
                raise RuntimeError(f"weight {tuple(w.shape)}:{w.dtype} not eligible")
            st = bwd_fp8._SiteScale(site, dev)
            w_partial, _snap, _snap_t = bwd_fp8._fused_site_launch(
                st, partial, w, snapshot=snapshot, t_note=t_note,
                t_shift=t_shift, snapshot_t=snapshot_t)
            if cast:
                bwd_fp8._fused_weight_quant(w, w_partial)

        ow = layer._output_weight  # metadata only
        uw = layer._uvqk_weight  # metadata only
        # Output projection: with the FP8 out-proj wgrad (ship10 (22)) the
        # forward saves e4m3 y and the backward is
        # output_projection_dgrad_with_operand (snapshot_decode_scale=True,
        # token-major dout for the "nn"/"producer" wgrads); otherwise the
        # plain output_projection_dgrad.
        with_operand = bool(getattr(thl, "_FP8_OUT_WGRAD", False))
        token_major = with_operand and str(
            getattr(thl, "_FP8_OUT_WGRAD_IMPL", "nn")) in ("nn", "producer")
        t_e4m3_o = (token_major
                    and bool(getattr(bwd_fp8, "_OUT_WGRAD_T_E4M3", False))
                    and bwd_fp8._GRAD_MAX == bwd_fp8._E5M2_MAX)
        probe = bwd_fp8._SiteScale("output_projection.L0", dev)
        t_shift_o = int(probe.t_shift)
        t_note_o = bool(t_e4m3_o and (
            t_shift_o != 7 or getattr(bwd_fp8, "_OUT_WGRAD_DFMT_SHIFT_AUTO", False))
            and probe.t_clipped_windows is not None)
        del probe
        for Tp in (T, T + 1):
            def h_out(Tp: int = Tp) -> None:
                g = bufs.act((Tp, Dm), bf16)
                one = torch.ones(1, dtype=f32, device=dev)
                if token_major:
                    _q, _qt, partial = bwd_fp8.quantize_row_major_dual(
                        g, one,
                        t_dtype=e4m3 if t_e4m3_o else None,
                        t_inv_mul=2.0 ** -t_shift_o if t_e4m3_o else 1.0,
                        t_limit=bwd_fp8._E4M3_MAX if t_e4m3_o else None,
                        return_partial=True)
                else:
                    _q, partial = bwd_fp8.quantize_row_major(
                        g, one, return_partial=True)
                w = torch.zeros(tuple(ow.shape), dtype=ow.dtype, device=dev)
                site_call("output_projection.L0", partial, w,
                          snapshot=with_operand, t_note=t_note_o,
                          t_shift=t_shift_o,
                          snapshot_t=token_major and t_e4m3_o and with_operand,
                          cast=g.shape[-1] % 16 == 0)

            r.run("H1", "bwd_helpers_fused_out",
                  f"T={Tp} token_major={int(token_major)} snap={int(with_operand)}",
                  h_out)
        dq_prod = bool(_call(bwd_fp8, "dq_producer_enabled", False))
        uvqk_bwd = bool(_call(bwd_fp8, "uvqk_fp8_backward_enabled", False))
        for snap in (False, True):
            if dq_prod and main_layers:
                def h_uvqk(snap: bool = snap) -> None:
                    w = torch.zeros(tuple(uw.shape), dtype=uw.dtype, device=dev)
                    site_call("uvqk_projection.L0",
                              torch.zeros((), dtype=f32, device=dev), w,
                              snapshot=snap, t_note=False, t_shift=0,
                              snapshot_t=False)

                r.run("H1", "bwd_helpers_fused_uvqk", f"snap={int(snap)}", h_uvqk)
            if uvqk_bwd and main_layers:
                for Tp in (T, T + 1):
                    def h_uvqk_bf16(snap: bool = snap, Tp: int = Tp) -> None:
                        _o, partial_amax, _c = bwd_fp8.quantize_amax_colsum(
                            bufs.act((Tp, W_uvqk), bf16),
                            torch.ones(1, dtype=f32, device=dev),
                            pad_to=tm_pad if uvqk_fp8t else 1,
                            return_partial_amax=True)
                        w = torch.zeros(tuple(uw.shape), dtype=uw.dtype, device=dev)
                        site_call("uvqk_projection.L0", partial_amax, w,
                                  snapshot=snap, t_note=False, t_shift=0,
                                  snapshot_t=False)

                    r.run("H1", "bwd_helpers_fused_uvqk_bf16",
                          f"T={Tp} snap={int(snap)}", h_uvqk_bf16)
            if vk_producer and targets_only:
                # w_vk = packed_weight_for_layout(uvqk_weight, V|K, dtype=x.dtype)
                # (tpa targets backward): [Dm, H*(Dv+Dq)] in the activation
                # dtype (bf16 under the transducer's bf16 training dtype).
                act_dt = getattr(last, "_training_dtype", None)
                for dt in ((act_dt,) if act_dt in (bf16, f32) else (bf16, f32)):
                    def h_vk(snap: bool = snap, dt: torch.dtype = dt) -> None:
                        w = torch.zeros((int(uw.shape[0]), H * (Dv + Dq)),
                                        dtype=dt, device=dev)
                        site_call("vk_projection",
                                  torch.zeros((), dtype=f32, device=dev), w,
                                  snapshot=snap, t_note=False, t_shift=0,
                                  snapshot_t=False)

                    r.run("H1", "bwd_helpers_fused_vk", f"snap={int(snap)} w={dt}",
                          h_vk)

    # ---- W1: ship-11 (24)(e) WSHADOW_FUSED forward weight-shadow refresh ---
    # The first FP8 forward refreshes every shadow one at a time (an empty
    # cache has nothing else stale), later steps refresh all stale shadows in
    # one batch; the compile key is the source dtype and BLOCK_P (the largest
    # per-shadow tile count of the launch). triton_fp8_wshadow.refresh_shadows
    # directly, on synthetic weights: triton_fp8._weight_cache is not touched;
    # the geometry plans it caches (_plans) are put back afterwards.
    if not (fp8_fwd_on and tf8.hstu_fp8_gemm_enabled()):
        r.skip("W1", "wshadow_fused", "FP8 forward GEMM off")
    elif not bool(getattr(tf8, "_HSTU_FP8_WSHADOW_FUSED", False)):
        r.skip("W1", "wshadow_fused", "DLRM_HSTU_FP8_WSHADOW_FUSED off")
    else:
        specs = _shadow_specs(main_layers, last, targets_only,
                              bool(tf8.hstu_fp8_fuse_qkv_enabled()), H, Dv, Dq)

        def w1(batch: List[Tuple[Tuple[int, int], torch.dtype,
                                 Tuple[Tuple[int, int], ...]]]) -> None:
            from generative_recommenders.ops.triton import triton_fp8_wshadow as twm

            ws = [torch.zeros(shape, dtype=dt, device=dev) for shape, dt, _ in batch]
            for w in ws:
                if not twm.fused_shadow_supported(w):
                    raise RuntimeError(f"weight {tuple(w.shape)} not supported")
            twm.refresh_shadows(ws, [lay for _, _, lay in batch], margin=1.0)

        for spec in specs:
            r.run("W1", "wshadow_fused",
                  f"w={spec[0]}:{str(spec[1]).replace('torch.', '')} layout={spec[2]}",
                  lambda spec=spec: w1([spec]))
        for dt in sorted({sp[1] for sp in specs}, key=str):
            group = [sp for sp in specs if sp[1] == dt]
            if len(group) > 1:
                r.run("W1", "wshadow_fused",
                      f"batch={len(group)} w={str(dt).replace('torch.', '')}",
                      lambda group=group: w1(group))

    # ---- U1: ship-11 (25)(g) FWD_U_DP hipBLASLt extension ------------------
    # The (18)(c) BLAS warmup loads the extension and its handle at M=16; this
    # runs triton_fp8._u_dp_scaled_mm_bias (the production call: private
    # workspace, heuristic top-1, quiet-heuristic bookkeeping) at production-
    # class token counts so the data-parallel solution's code object loads
    # here. Its workspace / logged / quiet state is put back afterwards.
    u_dp_fn = getattr(tf8, "_u_dp_extension", None)
    if not (fp8_fwd_on and getattr(tf8, "_HSTU_FP8_FWD_U_DP", False)):
        r.skip("U1", "fwd_u_dp", "DLRM_HSTU_FP8_FWD_U_DP off")
    elif not getattr(tf8, "_HSTU_FP8_FUSE_QKV", False):
        r.skip("U1", "fwd_u_dp", "QKV-fused FP8 forward off")
    elif u_dp_fn is None or not hasattr(u_dp_fn, "cache_info") \
            or u_dp_fn.cache_info().currsize <= 0:
        r.skip("U1", "fwd_u_dp", "extension not prebuilt (DLRM_HSTU_FP8_PREBUILD)")
    else:
        u_cols = H * Dv
        for M in _U_DP_ROWS:
            def u1(M: int = M) -> None:
                act = bufs.act((M, Dm), e4m3)
                w = bufs.act((u_cols, Dm), e4m3).t()  # column-major shadow
                bias = torch.zeros((u_cols,), dtype=bf16, device=dev)
                one = torch.ones((1, 1), dtype=f32, device=dev)
                reason = tf8._u_dp_ineligible_reason(
                    act, w, bias, None, one, one, bf16)
                if reason is not None:
                    raise RuntimeError(f"u_dp ineligible: {reason}")
                tf8._u_dp_scaled_mm_bias(act, w, bias, one, one)

            r.run("U1", "fwd_u_dp", f"M={M} K={Dm} N={u_cols}", u1)

    # ---- L1..L5: FP8 LayerNorm variants -----------------------------------
    nw_dtype = layer._input_norm_weight.dtype

    def ln_wb(width: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mirrors norm_weight=self._input_norm_weight.to(x.dtype) (stu.py).
        return (torch.ones(width, dtype=dtype, device=dev).to(bf16),
                torch.zeros(width, dtype=dtype, device=dev).to(bf16))

    def ln_loop(tag: str, name: str, fn: Callable[[int, bool], None]) -> None:
        for Tp in (T, T + 1):
            for big in stores:
                r.run(tag, name, f"T={Tp} store={store_names[big]}",
                      lambda Tp=Tp, big=big: fn(Tp, big))

    fuse_ln = tf8.hstu_fp8_fuse_ln_enabled()
    save_uvqk = bool(getattr(tpa, "_SAVE_FP8_UVQK", False))
    early_release = bool(getattr(tpa, "_EARLY_RELEASE_FP8_UVQK", False))
    save_normed = bool(getattr(tpa, "_SAVE_BF16_NORMED_X", False))
    save_targets_vk = bool(getattr(stu_mod, "_SAVE_TARGETS_VK", False))

    def allow_saved(lay: Any) -> bool:
        try:
            return bool(lay._allow_saved_fp8_uvqk())
        except Exception:
            return False

    fp8_ln_layers = [m for m in layers if m._recompute_normed_x]
    if not fp8_fwd_on or not fuse_ln:
        for tag, name in (("L1", "ln_fp8_fwd"), ("L2", "ln_fp8_recompute"),
                          ("L3", "ln_fp8_save_bf16"), ("L5", "ln_dual_recompute")):
            r.skip(tag, name, "FUSE_LN off")
    elif not fp8_ln_layers:
        for tag, name in (("L1", "ln_fp8_fwd"), ("L2", "ln_fp8_recompute"),
                          ("L3", "ln_fp8_save_bf16"), ("L5", "ln_dual_recompute")):
            r.skip(tag, name, "recompute_normed_x off")
    else:
        # L1: plain e4m3 forward LN (triton_weighted_layer_norm_fwd(output_fp8=
        # True)). DESIGN CORRECTION: the design says main layers use it in
        # training because DLRM_HSTU_FP8_SAVE_UVQK is unset, but the FP8 arm
        # preset (_env_bootstrap._FP8_FORWARD_ARM) exports SAVE_UVQK=1, so
        # with SAVE_NORMED_X=1 TRAINING uses the save-bf16 variant (L3). L1
        # is still the eval forward (needs_input_grad False) and the path of
        # any layer outside SAVE_UVQK_LAYERS, so it stays (cheap).
        def l1(Tp: int, big: bool) -> None:
            w, b = ln_wb(Dm, nw_dtype)
            triton_weighted_layer_norm_fwd(
                x=bufs.act((Tp, Dm), bf16, big), weight=w, bias=b, eps=_LN_EPS,
                output_fp8=True, inv_scale=inv)

        ln_loop("L1", "ln_fp8_fwd", l1)

        # L2: e4m3 recompute from saved mean/rstd (COMPUTE_MEAN_AND_RSTD=
        # False). DESIGN CORRECTION: main layers never take it (their
        # recompute is the dual kernel, L5); only the targets-only backward
        # does, when it has no saved e4m3 V/K pack.
        targets_saved_vk = (save_uvqk and early_release and save_targets_vk
                            and allow_saved(last))
        if not (targets_only and fp8_fwd_on and not targets_saved_vk):
            r.skip("L2", "ln_fp8_recompute",
                   "targets-only V/K pack is saved" if targets_only
                   else "targets-only off")
        else:
            def l2(Tp: int, big: bool) -> None:
                w, b = ln_wb(Dm, nw_dtype)
                triton_weighted_layer_norm_fwd(
                    x=bufs.act((Tp, Dm), bf16, big), weight=w, bias=b,
                    eps=_LN_EPS,
                    mean=torch.zeros(Tp, dtype=torch.float32, device=dev),
                    rstd=torch.ones(Tp, dtype=torch.float32, device=dev),
                    output_fp8=True, inv_scale=inv)

            ln_loop("L2", "ln_fp8_recompute", l2)

        # L3: save-bf16 forward (OUTPUT_BOTH, COMPUTE_MEAN_AND_RSTD=True).
        # Used by the targets-only layer AND (SAVE_UVQK=1) every main layer in
        # SAVE_UVQK_LAYERS during training.
        main_saving = save_uvqk and any(
            allow_saved(m) and m._recompute_uvqk for m in main_layers)
        if not save_normed:
            r.skip("L3", "ln_fp8_save_bf16", "DLRM_HSTU_FP8_SAVE_NORMED_X off")
        elif not (targets_only or main_saving):
            r.skip("L3", "ln_fp8_save_bf16", "no layer saves the e4m3 pack")
        else:
            def l3(Tp: int, big: bool) -> None:
                w, b = ln_wb(Dm, nw_dtype)
                triton_weighted_layer_norm_fp8_save_bf16_fwd(
                    x=bufs.act((Tp, Dm), bf16, big), weight=w, bias=b,
                    eps=_LN_EPS, inv_scale=inv)

            ln_loop("L3", "ln_fp8_save_bf16", l3)

        # L3t (ship10 STORE_T): once the uvqk/vk sites are published (during
        # calibration) the training forward LN of the main and targets-only
        # layers is triton_weighted_layer_norm_fwd_fp8_store_t (row-major e4m3
        # + token-major e4m3 padded to uvqk_tm_pad()), tpa forward. Pure.
        # Ship-11 (25)(a): STORE_T_V2 dispatches the XCD-grouped v2 kernel
        # inside the same wrapper (its ARMED latch is restored).
        if not store_t:
            r.skip("L3t", "ln_fp8_store_t", "DLRM_HSTU_FP8_BWD_UVQK_STORE_T off")
        else:
            from generative_recommenders.ops.triton.triton_layer_norm import (
                triton_weighted_layer_norm_fwd_fp8_store_t,
            )

            def l3t(Tp: int, big: bool) -> None:
                w, b = ln_wb(Dm, nw_dtype)
                triton_weighted_layer_norm_fwd_fp8_store_t(
                    x=bufs.act((Tp, Dm), bf16, big), weight=w, bias=b,
                    eps=_LN_EPS, inv_scale=inv, pad_to=tm_pad)

            ln_loop("L3t", "ln_fp8_store_t", l3t)

        # L5 (added; not in the design): main-layer backward recompute of a
        # layer WITHOUT a saved e4m3 pack is triton_weighted_layer_norm_dual_fwd
        # (OUTPUT_BOTH, TRAINING=False, COMPUTE_MEAN_AND_RSTD=False), tpa.py
        # backward `recompute_fp8_qkv` branch.
        recompute_layers = [m for m in main_layers if m._recompute_uvqk
                            and not (save_uvqk and allow_saved(m))]
        if not recompute_layers:
            r.skip("L5", "ln_dual_recompute", "every main layer saves its pack")
        else:
            def l5(Tp: int, big: bool) -> None:
                w, b = ln_wb(Dm, nw_dtype)
                triton_weighted_layer_norm_dual_fwd(
                    x=bufs.act((Tp, Dm), bf16, big), weight=w, bias=b,
                    eps=_LN_EPS,
                    mean=torch.zeros(Tp, dtype=torch.float32, device=dev),
                    rstd=torch.ones(Tp, dtype=torch.float32, device=dev),
                    inv_scale=inv)

            ln_loop("L5", "ln_dual_recompute", l5)

    # ---- L4: main-layer output LN * u * dropout, e4m3 out ------------------
    outln_layers = [m for m in main_layers
                    if not m._use_group_norm and m._recompute_y]
    ol = outln_layers[0] if outln_layers else None
    fuse = bool(ol is not None and tf8.hstu_fp8_fuse_silu_enabled()
                and not ol._use_group_norm)
    ratio = float(ol._output_dropout_ratio) if ol is not None else 0.0
    training = ratio != 0.0
    Do = H * Dv
    seeds = _SEEDS if training else (None,)
    # Row sizes: T, T+1 (the autotuned small-row kernel; x small or, with
    # STORES=both, from the arena) and, where the large-row kernel exists
    # (_use_lnmd_fwd_v2: ship8 (20) v2, ship-11 (28) v3 / maskgen), _ROWS_L
    # and +1 (output > 2 GiB, x/u <= 2 GiB: the window at _ROWS_L) plus,
    # with STORES=both, _ROWS_XL and +1 (x/u > 2 GiB too). Large rows use
    # plain allocations (the arena cannot hold them). Transient memory at
    # _ROWS_XL is ~9 GiB, freed after each item.
    big_rows: List[int] = []
    try:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl2

        v2 = getattr(thl2, "_use_lnmd_fwd_v2", None)
        if v2 is not None and v2(1 << 20, True):
            big_rows = [_ROWS_L, _ROWS_L + 1] + (
                [_ROWS_XL, _ROWS_XL + 1] if both else [])
    except Exception:
        pass

    def lnmd_loop(tag: str, name: str, output_fp8: bool, rows: List[Tuple[int, bool]]
                  ) -> None:
        assert ol is not None
        onw_dtype = ol._output_norm_weight.dtype
        for Tp, big in rows:
            for seed in seeds:
                def l4(Tp: int = Tp, seed: Optional[int] = seed,
                       big: bool = big) -> None:
                    w, b = ln_wb(Do, onw_dtype)
                    # x is the attention output [T, H*Dv]; production
                    # passes seed=None (randint draw) -> explicit seed.
                    # stu.py: fuse_silu = FUSE_SILU and not group_norm;
                    # hstu_compute_output passes silu_u=(mul_u_activation_type
                    # =="silu"), concat_u=concat_x=True, rng_row_indices=None
                    # (FP8 output requires it None), and
                    # HSTUComputeOutputFunction sets training=False when
                    # dropout_ratio==0.
                    triton_layer_norm_mul_dropout_fwd(
                        x=bufs.act((Tp, Do), bf16, big),
                        u=bufs.act((Tp, Do), bf16),
                        weight=w, bias=b, eps=_LN_EPS,
                        dropout_ratio=ratio, training=training,
                        silu_u=fuse, concat_u=True, concat_x=True,
                        mul_u_activation_type="silu" if fuse else "none",
                        seed=seed if training else 0,
                        rng_row_indices=None,
                        output_fp8=output_fp8,
                        inv_scale=inv if output_fp8 else 1.0)

                r.run(tag, name,
                      f"T={Tp} seed={seed} store={store_names[big] if Tp <= T + 1 else 'rows'}",
                      l4)

    if not (fp8_fwd_on and tf8.hstu_fp8_fuse_outln_enabled()):
        r.skip("L4", "ln_mul_dropout_fp8", "FUSE_OUTLN off")
    elif ol is None:
        r.skip("L4", "ln_mul_dropout_fp8", "no main layer with recompute_y")
    else:
        # MGPK layer list (DLRM_HSTU_FP8_MASKGEN_PK_LAYERS): L4's calls must be
        # untagged (-1) so that, with a list, they run both maskgen kernels and
        # compile both here. A tag left by an earlier STU forward would narrow
        # that to one kernel; the scalar snapshot above restores the tag.
        _thl = sys.modules.get(_P_TRITON + "triton_hstu_linear")
        if _thl is not None and hasattr(_thl, "set_lnmd_layer_tag"):
            _thl.set_lnmd_layer_tag(-1)
        lnmd_loop("L4", "ln_mul_dropout_fp8", True,
                  [(Tp, big) for Tp in (T, T + 1) for big in stores]
                  + [(Tp, False) for Tp in big_rows])
        _nib_items(r, _thl, ol, ln_wb, Do, ratio, training, fuse, inv, big_rows)

    # ---- Tier 2: bf16 calibration window ----------------------------------
    if tier >= 2:
        if not targets_only:
            r.skip("T1b", "triton_delta_fwd_bf16", f"targets-only: {to_reason}")
            r.skip("T2b", "triton_delta_bwd_bf16", f"targets-only: {to_reason}")
        else:
            delta_items("T1b", "T2b", "bf16", bf16, 1.0)

        # M1: bf16 Triton main-layer forward. The bf16 FlyDSL forward returns
        # None while DLRM_HSTU_FP8_FWD=1 (try_flydsl_attention_fwd), so during
        # calibration the main layers run triton_hstu_attention_fwd.
        if not main_layers:
            r.skip("M1", "triton_main_fwd_bf16", "no main HSTU layer")
        elif not fp8_fwd_on and mha.flydsl_fwd_selected():
            r.skip("M1", "triton_main_fwd_bf16", "bf16 FlyDSL forward serves it")
        else:
            ml = main_layers[0]
            nt_ml = nt_main if ml._target_aware else None
            for N in (_N_A, _N_B):
                for big in stores:
                    def m1(N: int = N, big: bool = big) -> None:
                        uvqk = bufs.act((T, 2 * H * Dv + 2 * H * Dq), bf16, big)
                        _u, v, q, k = uvqk.split([H * Dv, H * Dv, H * Dq, H * Dq], 1)
                        sort_idx = None
                        if ml._sort_by_length:
                            lengths = so[1:] - so[:-1]
                            _, sort_idx = torch.sort(lengths, descending=True,
                                                     stable=False)
                        triton_hstu_attention_fwd(
                            N, alpha, q.view(T, H, Dq), k.view(T, H, Dq),
                            v.view(T, H, Dv), so, nt_ml,
                            int(ml._max_attn_len), int(ml._contextual_seq_len),
                            sort_idx, _should_enable_tma(), 0)

                    r.run("M1", "triton_main_fwd_bf16",
                          f"N={N} store={store_names[big]}", m1)

        # M2: bf16 FlyDSL backward at rs 2048 (calibration recomputes bf16
        # uvqk; q/k/v and dq/dk/dv are column slices of [T, 2048] buffers).
        # Production's [rows, 2048] uvqk/duvqk and any dq-split workspace are
        # > 2 GiB storages: same class here (arena views, >2 GiB workspace).
        if not mha.flydsl_selected():
            r.skip("M2", "flydsl_bwd_bf16", "HSTU_ATTN_BWD!=flydsl")
        elif not main_layers:
            r.skip("M2", "flydsl_bwd_bf16", "no main HSTU layer")
        elif fly_shape_reason(bf16, bwd=True) is not None:
            r.skip("M2", "flydsl_bwd_bf16",
                   "unsupported shape: %s" % fly_shape_reason(bf16, bwd=True))
        else:
            for N in (_N_A, _N_C):
                def m2(N: int = N) -> None:
                    mha.require_flydsl()
                    with _big_dq_split_workspace(dev):
                        uvqk = bufs.act((T, W_uvqk), bf16, True)
                        _u, v, q, k = uvqk.split([H * Dv, H * Dv, H * Dq, H * Dq], 1)
                        duvqk = bufs.act((T, W_uvqk), bf16, True)
                        _du, dv, dq, dk = duvqk.split([H * Dv, H * Dv, H * Dq, H * Dq], 1)
                        dout = bufs.act((T, H, Dv), bf16)
                        mha.flydsl_hstu_attention_bwd(
                            dout, q.view(T, H, Dq), k.view(T, H, Dq), v.view(T, H, Dv),
                            so, N, alpha,
                            out=(dq.view(T, H, Dq), dk.view(T, H, Dq), dv.view(T, H, Dv)),
                            qkv_scale=1.0,
                        )

                r.run("M2", "flydsl_bwd_bf16", f"N={N} store=big", m2)

        # L4b (ship-11 (28)(c)): the calibration steps' training LN*U*dropout
        # forward has a bf16 output; at production row counts it runs the
        # 16-bit mask generator + _ln_mul_dropout_fwd_rng_v3 (maskgen needs an
        # e4m3 output). Large-row sizes only (the small-row bf16 kernel is the
        # pinned autotune key the first step already uses).
        if ol is None:
            r.skip("L4b", "ln_mul_dropout_bf16", "no main layer with recompute_y")
        elif not big_rows:
            r.skip("L4b", "ln_mul_dropout_bf16", "no large-row LN*U*dropout kernel")
        else:
            lnmd_loop("L4b", "ln_mul_dropout_bf16", False,
                      [(Tp, False) for Tp in big_rows])

    del so, nt
    return {"T": T, "N": sorted(set(n_list + [_N_B]))}


# ----------------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------------


@gin.configurable
def run_fp8_first_use_warmup(device: torch.device, model: torch.nn.Module,
                             batch_size: Any, prestarted: bool,
                             enabled: Any = 0) -> None:
    """train_ranker hook (gin run_fp8_first_use_warmup.enabled = $FP8_WARMUP),
    called after the DataLoader pre-start and before run_startup_preload /
    pregrow / RUN_START. Never raises."""
    # gin binds enabled = @fp8w/env_str() (raw string), so a malformed
    # $FP8_WARMUP is caught here instead of raising at gin resolution.
    try:
        tier = int(str(enabled).strip() or "0")
    except Exception:
        logger.warning("%s skipped: FP8_WARMUP=%r is not an integer", _TAG,
                       enabled)
        tier = 0
    if tier <= 0:
        return
    try:
        _run(device, model, batch_size, bool(prestarted), tier)
    except Exception as error:  # never abort a run over a warmup
        logger.error("%s skipped: unexpected error (%s: %s)", _TAG,
                     type(error).__name__, error, exc_info=True)


def _run(device: torch.device, model: torch.nn.Module, batch_size: Any,
         prestarted: bool, tier: int) -> None:
    rank = _rank()
    rank0 = rank == 0
    if not prestarted:
        # Same rule as run_startup_preload: FlyDSL must load only after every
        # worker pool is forked.
        logger.info("%s skipped: needs DL_PRESTART=1 (pools must be forked "
                    "before the FlyDSL import)", _TAG)
        return
    if not isinstance(device, torch.device) or device.type != "cuda" \
            or not torch.cuda.is_available():
        logger.info("%s skipped: no CUDA device (%s)", _TAG, device)
        return
    try:
        B = int(batch_size)
    except Exception:
        B = 0
    if B <= 0:
        logger.warning("%s skipped: batch_size=%r", _TAG, batch_size)
        return
    if _tunableop_tuning_or_recording():
        logger.warning("%s skipped: TunableOp tuning or record-untuned is ON", _TAG)
        return
    if os.environ.get("DLRM_HSTU_FP8_FWD", "0") != "1":
        logger.info("%s skipped: DLRM_HSTU_FP8_FWD!=1", _TAG)
        return
    try:
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

        with torch.cuda.device(device):
            reason = mha.unsupported_hardware_reason()
    except Exception as error:
        logger.warning("%s skipped: import error (%s)", _TAG, error)
        return
    if reason is not None:
        logger.info("%s skipped: %s", _TAG, reason)
        return

    stores_env = os.environ.get("FP8_WARMUP_STORES", "small").strip().lower()
    stores: Tuple[bool, ...] = (False, True) if stores_env == "both" else (False,)
    skipset = {x.strip().lower() for x in
               os.environ.get("FP8_WARMUP_SKIP", "").split(",") if x.strip()}
    quiet = _env_flag("FP8_WARMUP_QUIET")
    verify = _env_flag("FP8_WARMUP_VERIFY")

    t0 = time.perf_counter()
    mem_before = _mem_state(device)
    mem_after: Tuple[int, int] = (-1, -1)
    try:
        rng_before = _rng_snapshot(device)
    except Exception as error:
        logger.warning("%s skipped: RNG snapshot failed: %s", _TAG, error)
        return
    sha_before: Optional[Tuple[str, int]] = None
    if verify:
        try:
            sha_before = _dense_state_sha(model)
        except Exception as error:
            logger.warning("%s verify: pre-hash failed: %s", _TAG, error)

    guards = _Guards()
    bufs: Optional[_Buffers] = None
    runner: Optional[_Runner] = None
    info: Dict[str, Any] = {"T": 2 * B, "N": []}
    device_fault: Optional[BaseException] = None
    rng_equal = True
    global WARMUP_RAN
    WARMUP_RAN = True
    bwd_spf_before = dict(mha._fp8_bwd_spf_logged)
    try:
        guards.take(verify=verify)
        with torch.cuda.device(device), torch.autocast("cuda", enabled=False), \
                torch.no_grad():
            bufs = _Buffers(device, want_big=True in stores)
            runner = _Runner(device, bufs, rank0, skipset, quiet)
            try:
                info = _warm_items(runner, model, B, tier, stores)
            except Exception as error:
                runner.failed += 1
                logger.warning("%s could not build the item list: %s: %s",
                               _TAG, type(error).__name__, error)
    finally:
        guards.restore()
        # Item (106) BWD_SPF: the ARMED line the backward items latched was
        # dropped by the muted mha logger; log it here, after the restore,
        # outside every item, on every rank.
        _spf_bwd_armed_reprint(mha, bwd_spf_before)
        try:
            if bufs is not None:
                bufs.release()
            bufs = None
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            mem_after = _mem_state(device)
        except Exception as error:
            device_fault = error
        try:
            rng_after = _rng_snapshot(device)
            if not torch.equal(rng_before[0], rng_after[0]):
                rng_equal = False
                logger.error("%s CPU RNG state moved; restoring it", _TAG)
                torch.random.set_rng_state(rng_before[0])
            if rng_before[1] is not None and rng_after[1] is not None and not (
                torch.equal(rng_before[1], rng_after[1])
            ):
                rng_equal = False
                logger.error("%s CUDA RNG state moved; restoring it", _TAG)
                torch.cuda.set_rng_state(rng_before[1], device)
        except Exception as error:
            rng_equal = False
            logger.error("%s RNG state check failed: %s", _TAG, error)

    if device_fault is not None:
        logger.error("%s device fault during warmup (FP8_WARMUP=0 to isolate): %s",
                     _TAG, device_fault)
        return
    if not rng_equal:
        guards.changed.append("rng")

    if verify:
        state_equal = -1
        n_hashed = 0
        try:
            sha_after = _dense_state_sha(model)
            n_hashed = sha_after[1]
            if sha_before is not None:
                state_equal = int(sha_before[0] == sha_after[0])
                if not state_equal:
                    guards.changed.append("model_state")
        except Exception as error:
            logger.warning("%s verify: post-hash failed: %s", _TAG, error)
        try:
            guards.verify_globals()
        except Exception as error:
            logger.warning("%s verify: globals digest failed: %s", _TAG, error)
        logger.info(
            "%s verify state_sha equal=%d rng_equal=%d bwd_sites=%d "
            "tf8_caches_unchanged=%d dense_tensors=%d globals_equal=%d "
            "globals_digested=%d scalars_restored=%d",
            _TAG, state_equal, int(rng_equal), guards.bwd_sites,
            int(guards.tf8_unchanged), n_hashed,
            int(not guards.globals_changed),
            len(guards.digest_before or {}), guards.restored_scalars,
        )

    if rank0 and min(mem_before + mem_after) >= 0:
        # Diagnostic only (init, rank 0): free memory the warmup did not hand
        # back, split into torch reserved vs non-torch. Most of the non-torch
        # part is free again by the end of the pre-grow's first pass, and its
        # converge pass maps it (see MEMORY in the module docstring).
        free_short = mem_before[0] - mem_after[0]
        torch_delta = mem_after[1] - mem_before[1]
        logger.info("%s memory: free %.2f GiB below the pre-warmup value; torch "
                    "reserved %+.2f GiB, non-torch %+.2f GiB (rank %d)", _TAG,
                    free_short / 2 ** 30, torch_delta / 2 ** 30,
                    (free_short - torch_delta) / 2 ** 30, rank)

    assert runner is not None
    seconds = time.perf_counter() - t0
    if guards.changed:
        for name in guards.changed:
            logger.error("%s NOT armed: guard %s changed", _TAG, name)
        return
    if runner.ok == 0:
        logger.warning("%s requested but no item ran (skip=%d fail=%d, %.1fs)",
                       _TAG, runner.skipped, runner.failed, seconds)
        return
    # Ship-12: every rank prints its arming line (rank-0-only lines get lost in
    # the merged log pipe; the per-rank count is the arming proof).
    if True:
        T = int(info.get("T", 2 * B))
        logger.info(
            "armed OK: %s tier=%d ok=%d skip=%d fail=%d in %.1fs before "
            "run_start (B=%d T=%d/%d N=%s stores=%s triton_jit_new=%d "
            "autotune_dropped=%d guards=ok items=%s rank=%d)",
            _TAG, tier, runner.ok, runner.skipped, runner.failed, seconds, B, T,
            T + 1, ",".join(str(n) for n in info.get("N", [])),
            "both" if True in stores else "small",
            runner.jit_new if runner.jit_known else -1,
            guards.autotune_dropped,
            ",".join(f"{k}:{v}" for k, v in sorted(runner.tags.items())), rank,
        )
