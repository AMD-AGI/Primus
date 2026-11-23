# pyre-strict
"""Indexed-X L1: the preprocessor L1 forward GEMM and its weight gradient read their input rows THROUGH the int64 EC
index from the bf16 unique-row table U, so the EC expand and the saved X are never materialized.
(rccl R-C3 / actckpt E-X2; productionized from the rccl iteration-82 lab kernels gidx_kern.l1_gemm GATHER=1 and
gidxw_kern.l1_wgrad GATHER=1 + wgrad_reduce; iteration-83 integration draft.)

DEFAULT OFF. Gate: DLRM_HSTU_FP8_EC_INDEXED_L1=1 (1/true/on/yes), read once per process at first use (forwarded by
launch_slurm.sh's DLRM_HSTU_FP8_ prefix); unset / "0" = off. ROCm gfx950 only. This module holds the kernels, their
fail-closed launch wrappers, the shipped-numerics references and the pre-RUN_START prebuild. The EC handle
(U_bf16, idx) and the L1 autograd node belong to the integration (ec_bf16_output / preproc_fp8_l2 /
fused_linear_norm); nothing here imports them.

WHAT. ContextualPreprocessor's content MLP L1 (W1_c [256, 512]) and additional MLP L1 (W1_a [256, 1024]) read
X = expand(U_bf16, idx) today (X_item [M, 512]; X_cat [2M, 512] viewed [M, 1024] = artist | album). Here:
  member  NH  idx (int64, contiguous)              A row m (never written to memory)
  item     1  [M]   (merged uih+candidate index)   A[m, 0:512]    = U[idx[m]]
  cat      2  [2M]  interleaved [art_0, alb_0, ..] A[m, 0:512]    = U[idx[2m]]    (artist, K 0..511)
                                                   A[m, 512:1024] = U[idx[2m+1]]  (album,  K 512..1023)
  _ecix_fwd           h[m, n] = bf16_rne( fp32( sum_k A[m, k] W1[n, k] ) + fp32(b1[n]) )
                      = torch.addmm(lin_b_c, x_c, lin_w_c.t()) of ship13 (_LinearSwishLNEmitFunction.forward,
                      _LinearNormFunction.forward) under autocast(bf16): x_c bf16, lin_w_c bf16 [256, K], lin_b_c
                      bf16 [256], h bf16 [M, 256], fp32 accumulate, bias added in fp32 at the end (hipBLASLt's
                      Bias epilogue order), RNE to bf16.
  _ecix_wgrad         split-K stage 1: P[s, n, k] = fp32 sum over the tokens m of split s, ascending, of
                      dh[m, n] * A[m, k]. Every token, no atomics, one [256, 256] tile per program.
  _ecix_wgrad_reduce  stage 2: dW1[n, k] = out_dtype( bf16_rne( fp32 sum_{s = 0, 1, ..., S-1} P[s, n, k] ) )
                      = _match(dh.t() @ x, param dtype) of ship13 (bf16 GEMM output, then .to(fp32)).
The dgrad (dX = dh @ W1, or actckpt E-X2b's fused dgrad + segment reduce) and db1 (the SwishLN backward's
dh_colsum) are not touched by this module.

MLPerf closed division: no computation reduction. Every token's L1 product is computed from that token's own
gathered row: the forward computes every output row from its own A row; the wgrad multiplies every token's dh row
with its own A row and sums over ALL M tokens. Nothing is pre-summed over tokens that share a unique row, nothing is
reused across tokens, samples or steps. The index only says WHERE a row is loaded from (a fused gather).

TILES (frozen from iteration 82; GIDX.md section 3, GIDXW.md section 2; both members; make_ids_runs ids: gin (78)):
  _ecix_fwd     t128x256x64w8s2k32: BM 128, BN 256 (= all outputs, so each A row is gathered once), BK 64,
                8 warps, 2 stages, matrix_instr_nonkdim 32 (v_mfma_f32_32x32x16_bf16); grid cdiv(M, 128).
  _ecix_wgrad   a256x256x64w8s2k16p256x8i: BN 256 x BKO 256 output tile, BM 64 tokens per step, 8 warps,
                2 stages, matrix_instr_nonkdim 16 (v_mfma_f32_16x16x32_bf16); dh^T loaded directly as [BN, BM]
                (ALOAD); the next block's indices loaded after the current block's dot (item (83) S_L); XCD remap 8.
                Item (83) gate DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT (gin True; read once, wgrad_idx_after_dot());
                "0" / absent launches _ecix_wgrad_idxpf instead, ship14's IDXPF order (indices one iteration
                ahead), bitwise identical P and dW1.
  TRITON 3.9 (ROCm 10.2 port, ec-kernels lane): same tiles, same instruction stream order of values, so the same
                bits as on 3.6 (checked bitwise against the v26.3 stack). Two code-level changes keep 3.6's register
                budget on 3.9's LLVM:
                (a) the forward's K loop has an opaque trip count (D // BK + (M < 0), = D // BK since M >= 0):
                    with a constant 7-iteration pipelined loop LLVM unrolls by 2 with an exit between the copies and
                    the 64-VGPR accumulator lives in two register sets (186 / 188 VGPRs, occupancy 2);
                (b) each kernel pins LLVM's AMDGPU machine-scheduler strategy (llvm_sched_kw: HIPOptions.llvm_fn_attrs
                    "amdgpu-sched-strategy", Triton >= 3.9 only; {} elsewhere): 3.9's default strategy issues all
                    of a step's LDS operand reads / global prefetches ahead of the MFMAs. _ecix_fwd iterative-maxocc
                    (122 / 124 VGPRs, occupancy 4; 3.6: 128), _ecix_wgrad[_idxpf] iterative-minreg (246 VGPRs, 0
                    spills; default: 256 + 104 / 100 B spills in the loop, 2x slower; 3.6: 230). Measured (M 2.5M,
                    U 150k, same node as the v26.3 stack, interleaved): fwd -1..+2 %, wgrad -2..0 % vs 3.6.
  split rule    NT = 2 (item) / 4 (cat) output tiles; S0 = 256 // NT (one program per CU, 256 CUs);
                CHUNK = cdiv(cdiv(M, S0), 64) * 64; S = cdiv(M, CHUNK) <= S0; grid NT * S.
  workspace     fp32 [S, 256, NH*512]: <= 64 MiB per call for every M (S <= S0); caching allocator, per call.
  reduce        BLOCK 1024, 4 warps; grid 256*NH*512 / 1024.

NUMERICS. Declared fp32 reduction-order class. Forward: raw-bit equal to the dense same-tile kernel on the
materialized X and to hipBLASLt's addmm except hipBLASLt's stream-K tail tiles (iter82: 0 of 669M differ at late
M; 688 / 1,631 of 606M at early M, max 16 / 56 bf16 ulp at near-zero outputs). Wgrad: raw-bit equal to the dense
same-tile kernel, fp32 partials included; vs the shipped dh.t() @ x it differs in ~1e-3 of the elements (an fp32
sum straddling a bf16 rounding boundary), same SNR vs fp64 (55.6 dB). Deterministic run to run (no atomics, fixed
split rule, fixed-order reduce). Masked tail rows contribute exact +0.0 products (dh and the gathered row are both
masked to 0 in the tail; the iteration-82 lab kernel masked only dh and gathered U[0], so a non-finite U[0] would
have reached the products). The two tails gave identical bits on every input tested (iter83 R2 tB); they can
differ only in the sign of an exactly-zero partial sum.

INDEX ARITHMETIC (int32 where it is cheap, asserted on the host): gathered row offsets idx * 512 are int32
(U * 512 < 2^31); token index arithmetic (row * NH + h) is int32 (M * NH < 2^31); dh, output and partial offsets
are int64. idx values must lie in [0, U): a structural invariant of the EC dedup (reverse indices into the unique
rows), NOT checked on the device (check_index_range() is the optional one-host-sync check).

SPECIALIZATION (Triton 3.6.0 on AMD; rccl iter83 R2 lab tasks tA / tB): an int argument is classed (== 1 ->
constexpr, % 16 == 0 -> 'D', else '') unless do_not_specialize; a pointer by 16-byte alignment ('D', unless
do_not_specialize_on_alignment) and by storage within 2 GiB ('S' = tt.pointer_range 32 = buffer ops;
HIPBackend.is_within_2gb: untyped_storage().size() <= 2^31 - 1). Here: M, CHUNK and S are do_not_specialize (they
change every step); IDX is do_not_specialize_on_alignment (an int64 view at an odd element offset is only 8-byte
aligned; the instruction stream is identical to the alignment-specialized iteration-82 binaries). Every other
pointer must be 16-byte aligned and every pointer's storage within 2 GiB: the wrappers check both and decline
otherwise (IndexedL1Unsupported), so a production call can only hit the ONE class prebuild() compiles per kernel
and member (dh / h [M, 256] bf16 leave the 'S' class at M >= 2^22 rows, U at >= 2^21 rows; benchmark traces show
M ~2.4-2.6M and U ~54k-155k per rank; a call outside the class is declined, never mis-computed).

FAILURE MODEL. The wrappers raise IndexedL1Unsupported (a ValueError) BEFORE launching anything when an input is
outside the validated class; the caller takes the shipped path (reference_fwd / reference_wgrad reproduce it bit for
bit from (U, idx)). Any other exception (compile, launch) should latch the feature off for the process
(latch_off) and also take the shipped path. prebuild() raises on any mismatch with its exact reference.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

try:  # CPU-only test environments have no Triton; only the kernels need it.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

logger: logging.Logger = logging.getLogger(__name__)

ENV_VAR: str = "DLRM_HSTU_FP8_EC_INDEXED_L1"
TAG: str = "[ec-indexed-l1]"
ARMED_PREFIX: str = "armed OK: " + TAG
HID: int = 256  # L1 out features (hstu_preprocessor_hidden_dim) = dh width
WIDTH: int = 512  # EC row width (HSTU_EMBEDDING_DIM) = K per member half

# Frozen iteration-82 tiles (both members).


def llvm_sched_kw(strategy: str) -> Dict[str, Any]:
    """Launch option pinning LLVM's AMDGPU machine-scheduler strategy for one kernel (TILES, TRITON 3.9 (b)):
    {"llvm_fn_attrs": "amdgpu-sched-strategy=<strategy>"} when this Triton's HIPOptions has llvm_fn_attrs (3.9),
    else {} (Triton 3.6 has no such option and would reject the keyword; its default schedule is the tuned one).
    Scheduling only: the same instructions on the same values, so the same bits."""
    try:
        from triton.backends.amd.compiler import HIPOptions
    except Exception:  # no Triton / no AMD backend (CPU tests)
        return {}
    if "llvm_fn_attrs" not in getattr(HIPOptions, "__dataclass_fields__", {}):
        return {}
    return {"llvm_fn_attrs": f"amdgpu-sched-strategy={strategy}"}


FWD_BM, FWD_BN, FWD_BK = 128, 256, 64  # gidx t128x256x64w8s2k32
FWD_KW: Dict[str, Any] = dict(num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32,
                              **llvm_sched_kw("iterative-maxocc"))
WG_BN, WG_BKO, WG_BM = 256, 256, 64  # gidxw a256x256x64w8s2k16p256x8i
WG_NUM_XCD: int = 8
WG_PROGS: int = 256  # target stage-1 programs (NT x S) = one per CU
WG_KW: Dict[str, Any] = dict(num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=16,
                             **llvm_sched_kw("iterative-minreg"))
RED_BLOCK: int = 1024
RED_KW: Dict[str, int] = dict(num_warps=4)
# Tail rows of a split: mask the gathered row too (exact +0.0 products whatever U holds).
TAIL_ZERO_B: bool = True

# HIPBackend.is_within_2gb (Triton 3.6 backends/amd/compiler.py:164-172): storage bytes <= this -> 'S' class.
MAX_PTR_RANGE: int = 2**31 - 1
MAX_ROWS: int = MAX_PTR_RANGE // (HID * 2)  # 4,194,303: dh / h [M, 256] bf16 stay in the 'S' class
MAX_TABLE_ROWS: int = MAX_PTR_RANGE // (WIDTH * 2)  # 2,097,151: U [U, 512] bf16 stays in the 'S' class
assert WIDTH % FWD_BK == 0 and WIDTH % WG_BKO == 0 and HID % FWD_BN == 0 and HID % WG_BN == 0
assert (HID * WIDTH) % RED_BLOCK == 0

_ENABLED: Optional[bool] = None
# Item (83) S_L: the wgrad loop order. "1" (the gin default): _ecix_wgrad, the next block's indices loaded after the
# current block's dot; absent / "0" (kill switch): _ecix_wgrad_idxpf, ship14's IDXPF order. Bitwise identical P / dW1.
WGRAD_IDX_AFTER_DOT_ENV: str = "DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT"
_WGRAD_IDX_AFTER_DOT: Optional[bool] = None
_BROKEN: Optional[str] = None
_ARCH_OK: Dict[int, Optional[str]] = {}
# name -> Triton CompiledKernel of the last launch (tests / diagnostics: n_regs, n_spills, asm).
LAST_KERNELS: Dict[str, Any] = {}


class IndexedL1Unsupported(ValueError):
    """The inputs are outside the class these kernels were validated and prebuilt for. Raised BEFORE any launch;
    the caller takes the shipped path (reference_fwd / reference_wgrad, or the expand + addmm it replaces)."""


def _say(msg: str) -> None:
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(msg)
    except Exception:  # standalone use (lab, tests)
        logger.warning("%s", msg)


def enabled() -> bool:
    """Gate on, a ROCm build with the kernels defined, and not latched off (env read once per process)."""
    global _ENABLED
    if _ENABLED is None:
        on = os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
        why = None
        if on and torch.version.hip is None:
            why = "not a ROCm/HIP build (validated on gfx950 only)"
        elif on and _KERNELS_ERROR is not None:
            why = f"Triton kernels unavailable ({_KERNELS_ERROR})"
        if why is not None:
            _say(f"{TAG} {ENV_VAR}=1 IGNORED (NOT armed): {why}; the shipped expand + L1 GEMMs run")
            on = False
        _ENABLED = on
    return bool(_ENABLED) and _BROKEN is None


def wgrad_idx_after_dot() -> bool:
    """Item (83): which _ecix_wgrad loop order ecix_wgrad launches (env read once per process, at the first wgrad
    or prebuild; absent = off in code = ship14's IDXPF kernel; the gin file binds it on)."""
    global _WGRAD_IDX_AFTER_DOT
    if _WGRAD_IDX_AFTER_DOT is None:
        _WGRAD_IDX_AFTER_DOT = os.environ.get(WGRAD_IDX_AFTER_DOT_ENV, "").strip().lower() in ("1", "true", "on", "yes")
        logger.info("%s wgrad loop order: %s (%s=%r)", TAG,
                    "_ecix_wgrad, index load after the dot (item (83))" if _WGRAD_IDX_AFTER_DOT
                    else "_ecix_wgrad_idxpf, ship14 IDXPF", WGRAD_IDX_AFTER_DOT_ENV,
                    os.environ.get(WGRAD_IDX_AFTER_DOT_ENV))
    return _WGRAD_IDX_AFTER_DOT


def latch_off(where: str, exc: BaseException) -> None:
    """Indexed-X off for the rest of the process (every later call: the shipped path)."""
    global _BROKEN
    if _BROKEN is None:
        _BROKEN = f"{where}: {type(exc).__name__}: {str(exc)[:300]}"
        _say(f"{TAG} FAILED CLOSED (DISABLED for the rest of the run) -> shipped EC expand + L1 GEMMs ({_BROKEN})")


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ------------------------------------------------------------------------------------------------------------ kernels
_KERNELS_ERROR: Optional[str] = "Triton is not importable" if triton is None else None
if triton is not None:
    try:

        @triton.jit
        def _ecix_remap_xcd(pid, GRID, NUM_XCD: tl.constexpr):
            # Workgroup pid is dispatched to XCD pid % NUM_XCD; give each XCD a contiguous range of logical ids
            # (a bijection on [0, GRID); iter82 gidxw VERIFY2 checked GRID % 8 == 0 and != 0).
            per = (GRID + NUM_XCD - 1) // NUM_XCD
            tall = GRID % NUM_XCD
            tall = tl.where(tall == 0, NUM_XCD, tall)
            xcd = pid % NUM_XCD
            loc = pid // NUM_XCD
            return tl.where(xcd < tall, xcd * per + loc, tall * per + (xcd - tall) * (per - 1) + loc)

        @triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["IDX"])
        def _ecix_fwd(U, IDX, W, BIAS, C, M,
                      NH: tl.constexpr, D: tl.constexpr, NOUT: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
            """C[m, :] = bf16_rne(fp32 sum_k A[m, k] W[:, k] + fp32 BIAS); A row m = U[IDX[m*NH + h]] for K half h.
            iter82 gidx l1_gemm with GATHER=1, ONE_LOOP=0, NUM_XCD=1, SCALE=0 (same statements, same tl.dot
            order: h = 0..NH-1, k = 0..D-1 step BK). Rows >= M of the last block read the block's first row
            (valid) and are not stored."""
            KT: tl.constexpr = NH * D
            NPN: tl.constexpr = NOUT // BN
            pid = tl.program_id(0)
            pid_m = pid // NPN  # the NPN N-tiles of one row block are adjacent ids (NPN = 1 at BN = NOUT)
            pid_n = pid % NPN
            row0 = pid_m * BM
            offs_r = tl.arange(0, BM)
            rmask = offs_r < M - row0
            offs_rc = tl.where(rmask, offs_r, 0)
            offs_k = tl.arange(0, BK)
            offs_n = pid_n * BN + tl.arange(0, BN)
            # K steps per half: D // BK, written so LLVM cannot see a constant trip count (M >= 0 always; the
            # wrapper never launches M = 0): no unroll with a mid-loop exit (TILES, TRITON 3.9 (a)). Same values,
            # same order.
            nk = D // BK + (M < 0).to(tl.int32)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for h in tl.static_range(NH):
                ih = tl.load(IDX + (row0 + offs_rc) * NH + h)
                a_ptrs = U + (ih.to(tl.int32) * D)[:, None] + offs_k[None, :]
                b_ptrs = W + h * D + offs_n[None, :] * KT + offs_k[:, None]
                for k in range(0, nk):
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    acc = tl.dot(a, b, acc)
                    a_ptrs += BK
                    b_ptrs += BK
            bias = tl.load(BIAS + offs_n).to(tl.float32)
            acc = acc + bias[None, :]
            c_ptrs = C + (row0 + offs_r).to(tl.int64)[:, None] * NOUT + offs_n[None, :]
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=rmask[:, None])

        @triton.jit(do_not_specialize=["M", "CHUNK"], do_not_specialize_on_alignment=["IDX"])
        def _ecix_wgrad(DH, U, IDX, P, M, CHUNK,
                        NH: tl.constexpr, D: tl.constexpr, NOUT: tl.constexpr,
                        BN: tl.constexpr, BKO: tl.constexpr, BM: tl.constexpr, NUM_XCD: tl.constexpr,
                        TAIL_ZERO_B: tl.constexpr):
            """P[s] = fp32 [BN, BKO] tile of dh[m0:m1]^T @ A[m0:m1] over split s's tokens [s*CHUNK, min(M, (s+1)*CHUNK)),
            in token order: n_full unmasked BM blocks, then one masked tail block. iter82 gidxw l1_wgrad with
            GATHER=1, ALOAD=1, DH_EVICT=0 (same tl.dot order), plus TAIL_ZERO_B; the next block's indices are loaded
            after the dot (item (83) S_L, bitwise identical to the iteration-82 IDXPF=1 statement order)."""
            KT: tl.constexpr = NH * D
            NT_K: tl.constexpr = KT // BKO
            NT: tl.constexpr = (NOUT // BN) * NT_K
            pid = tl.program_id(0)
            if NUM_XCD > 1:
                pid = _ecix_remap_xcd(pid, tl.num_programs(0), NUM_XCD)
            s = pid // NT  # the NT output tiles of one split are adjacent logical ids (same XCD: shared dh rows)
            t = pid % NT
            tn = t // NT_K
            tk = t % NT_K
            m0 = s * CHUNK
            m1 = tl.minimum(M, m0 + CHUNK)
            nrow = m1 - m0
            n_full = nrow // BM
            offs_bm = tl.arange(0, BM)
            offs_n = tn * BN + tl.arange(0, BN)
            offs_o = tk * BKO + tl.arange(0, BKO)
            rows = (m0 + offs_bm).to(tl.int64)
            dh_ptrs = DH + offs_n[:, None] + rows[None, :] * NOUT  # dh^T tile [BN, BM] (MN-major A operand)
            h = (tk * BKO) // D  # which member half this K tile reads (0 for item); BKO divides D
            col = offs_o - h * D
            ip = IDX + rows * NH + h
            acc = tl.zeros((BN, BKO), dtype=tl.float32)
            ii = tl.load(ip, mask=offs_bm < nrow, other=0)  # block 0's indices (the chunk may be one partial block)
            for it in range(0, n_full):
                a = tl.load(dh_ptrs)
                b = tl.load(U + (ii.to(tl.int32) * D)[:, None] + col[None, :])
                acc = tl.dot(a, b, acc)
                # the next block's indices, loaded AFTER this block's dot (item (83) S_L, rccl iter91 kit1 lab ADOPT:
                # same values, same tl.dot sequence, bitwise identical P and dW1; the software pipeliner then
                # schedules the next block's gathered rows earlier in the loop body)
                ii = tl.load(ip + BM * NH, mask=(it + 1) * BM + offs_bm < nrow, other=0)
                ip += BM * NH
                dh_ptrs += BM * NOUT
            rem = nrow - n_full * BM
            if rem > 0:
                rmask = offs_bm < rem
                a = tl.load(dh_ptrs, mask=rmask[None, :], other=0.0)
                if TAIL_ZERO_B:
                    b = tl.load(U + (ii.to(tl.int32) * D)[:, None] + col[None, :], mask=rmask[:, None], other=0.0)
                else:  # the iteration-82 lab tail: masked lanes gather U[0]
                    b = tl.load(U + (ii.to(tl.int32) * D)[:, None] + col[None, :])
                acc = tl.dot(a, b, acc)
            p_ptrs = P + s.to(tl.int64) * (NOUT * KT) + offs_n[:, None] * KT + offs_o[None, :]
            tl.store(p_ptrs, acc)

        @triton.jit(do_not_specialize=["M", "CHUNK"], do_not_specialize_on_alignment=["IDX"])
        def _ecix_wgrad_idxpf(DH, U, IDX, P, M, CHUNK,
                              NH: tl.constexpr, D: tl.constexpr, NOUT: tl.constexpr,
                              BN: tl.constexpr, BKO: tl.constexpr, BM: tl.constexpr, NUM_XCD: tl.constexpr,
                              TAIL_ZERO_B: tl.constexpr):
            """Item (83) OFF (DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT=0): ship14's (c68395d) _ecix_wgrad body verbatim,
            the next block's indices prefetched one iteration ahead (IDXPF=1). Same parameters, same values, same
            tl.dot sequence as _ecix_wgrad, so P and dW1 are bitwise identical; only the loop statement order (and
            with it the software-pipelined schedule) differs."""
            KT: tl.constexpr = NH * D
            NT_K: tl.constexpr = KT // BKO
            NT: tl.constexpr = (NOUT // BN) * NT_K
            pid = tl.program_id(0)
            if NUM_XCD > 1:
                pid = _ecix_remap_xcd(pid, tl.num_programs(0), NUM_XCD)
            s = pid // NT  # the NT output tiles of one split are adjacent logical ids (same XCD: shared dh rows)
            t = pid % NT
            tn = t // NT_K
            tk = t % NT_K
            m0 = s * CHUNK
            m1 = tl.minimum(M, m0 + CHUNK)
            nrow = m1 - m0
            n_full = nrow // BM
            offs_bm = tl.arange(0, BM)
            offs_n = tn * BN + tl.arange(0, BN)
            offs_o = tk * BKO + tl.arange(0, BKO)
            rows = (m0 + offs_bm).to(tl.int64)
            dh_ptrs = DH + offs_n[:, None] + rows[None, :] * NOUT  # dh^T tile [BN, BM] (MN-major A operand)
            h = (tk * BKO) // D  # which member half this K tile reads (0 for item); BKO divides D
            col = offs_o - h * D
            ip = IDX + rows * NH + h
            acc = tl.zeros((BN, BKO), dtype=tl.float32)
            ii = tl.load(ip, mask=offs_bm < nrow, other=0)  # block 0's indices (the chunk may be one partial block)
            for it in range(0, n_full):
                a = tl.load(dh_ptrs)
                # the next block's indices one iteration ahead (breaks the idx -> row load chain)
                ii_n = tl.load(ip + BM * NH, mask=(it + 1) * BM + offs_bm < nrow, other=0)
                b = tl.load(U + (ii.to(tl.int32) * D)[:, None] + col[None, :])
                ii = ii_n
                ip += BM * NH
                acc = tl.dot(a, b, acc)
                dh_ptrs += BM * NOUT
            rem = nrow - n_full * BM
            if rem > 0:
                rmask = offs_bm < rem
                a = tl.load(dh_ptrs, mask=rmask[None, :], other=0.0)
                if TAIL_ZERO_B:
                    b = tl.load(U + (ii.to(tl.int32) * D)[:, None] + col[None, :], mask=rmask[:, None], other=0.0)
                else:  # the iteration-82 lab tail: masked lanes gather U[0]
                    b = tl.load(U + (ii.to(tl.int32) * D)[:, None] + col[None, :])
                acc = tl.dot(a, b, acc)
            p_ptrs = P + s.to(tl.int64) * (NOUT * KT) + offs_n[:, None] * KT + offs_o[None, :]
            tl.store(p_ptrs, acc)

        @triton.jit(do_not_specialize=["S"])
        def _ecix_wgrad_reduce(P, OUT, S, NEL: tl.constexpr, BLOCK: tl.constexpr):
            """OUT = OUT.dtype(bf16_rne(sum_{s = 0..S-1} P[s])), summed in fixed order s = 0, 1, ... (deterministic).
            The bf16 rounding is the dtype path of the shipped dh.t() @ x (bf16 GEMM output); the store into an
            fp32 OUT is then exact (= .to(float32)), into a bf16 OUT a no-op."""
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            acc = tl.load(P + offs)
            for s in range(1, S):
                acc += tl.load(P + s * NEL + offs)
            acc = acc.to(tl.bfloat16).to(tl.float32)
            tl.store(OUT + offs, acc.to(OUT.dtype.element_ty))

    except Exception as exc:  # pragma: no cover
        _KERNELS_ERROR = f"{type(exc).__name__}: {str(exc)[:200]}"


# ------------------------------------------------------------------------------------------------------------ checks
def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise IndexedL1Unsupported(msg)


def _strides_ok(t: torch.Tensor, strides: Tuple[int, ...]) -> bool:
    # dims of size 1 do not address memory; their stride is free
    return all(t.shape[i] <= 1 or t.stride(i) == strides[i] for i in range(t.dim()))


def _ptr_class_ok(name: str, t: torch.Tensor, align: bool = True) -> None:
    """The one production specialization class: 16-byte aligned ('D') unless align=False, storage within 2 GiB."""
    if t.numel() == 0:
        return
    if align:
        _require(t.data_ptr() % 16 == 0, f"{name} data_ptr not 16-byte aligned (new Triton 'D' class)")
    _require(t.untyped_storage().size() <= MAX_PTR_RANGE,
             f"{name} storage {t.untyped_storage().size()} B > 2 GiB (new Triton 'S' / buffer-ops class)")


def _check_device(dev: torch.device) -> None:
    _require(_KERNELS_ERROR is None, f"Triton kernels unavailable ({_KERNELS_ERROR})")
    _require(torch.version.hip is not None, "not a ROCm/HIP build")
    _require(dev.type == "cuda" and dev.index is not None, f"device {dev} is not a GPU")
    _require(dev.index == torch.cuda.current_device(),
             f"tensors on {dev} but the current device is {torch.cuda.current_device()} (Triton launches there)")
    why = _ARCH_OK.get(dev.index, "")
    if why == "":
        arch = str(torch.cuda.get_device_properties(dev).gcnArchName)
        why = None if arch.startswith("gfx950") else f"arch {arch} (tiles validated on gfx950 only)"
        _ARCH_OK[dev.index] = why
    _require(why is None, str(why))


def _check_table_idx(table: torch.Tensor, idx: torch.Tensor, nh: int) -> int:
    """-> M. table = U_bf16 [U, 512] (the EC's one-time bf16 cast); idx int64 [M * nh]."""
    _require(isinstance(nh, int) and not isinstance(nh, bool) and nh in (1, 2), f"nh={nh!r} (1 item, 2 cat)")
    _require(isinstance(table, torch.Tensor) and isinstance(idx, torch.Tensor), "table / idx must be tensors")
    dev = table.device
    _check_device(dev)
    _require(table.dtype == torch.bfloat16, f"table dtype {table.dtype} != bfloat16")
    _require(table.dim() == 2 and table.shape[1] == WIDTH and table.shape[0] >= 1,
             f"table shape {tuple(table.shape)} != (U >= 1, {WIDTH})")
    _require(_strides_ok(table, (WIDTH, 1)), f"table strides {tuple(table.stride())} != ({WIDTH}, 1)")
    n_rows = int(table.shape[0])
    _require(n_rows * WIDTH < 2**31, f"U={n_rows}: idx * {WIDTH} overflows the int32 row offsets")
    _ptr_class_ok("table", table)
    _require(idx.device == dev, f"idx on {idx.device}, table on {dev}")
    _require(idx.dtype == torch.int64, f"idx dtype {idx.dtype} != int64")
    _require(idx.dim() == 1 and (idx.numel() <= 1 or idx.stride(0) == 1), "idx must be 1-D contiguous")
    _require(idx.numel() % nh == 0, f"idx numel {idx.numel()} not a multiple of nh={nh}")
    m = idx.numel() // nh
    _require(m <= MAX_ROWS, f"M={m} > {MAX_ROWS}: dh / h [M, {HID}] bf16 would leave the 2 GiB 'S' class")
    _require(m * nh < 2**31, f"M*nh={m * nh} overflows int32 token arithmetic")
    _ptr_class_ok("idx", idx, align=False)  # IDX is do_not_specialize_on_alignment
    return m


def check_fwd(table: torch.Tensor, idx: torch.Tensor, nh: int, w1: torch.Tensor, b1: torch.Tensor) -> int:
    """Raise IndexedL1Unsupported unless (table, idx, nh, w1, b1) is the validated forward class. -> M."""
    m = _check_table_idx(table, idx, nh)
    dev, kt = table.device, nh * WIDTH
    for name, t, shape, strides in (("w1", w1, (HID, kt), (kt, 1)), ("b1", b1, (HID,), (1,))):
        _require(isinstance(t, torch.Tensor) and t.device == dev, f"{name} must be a tensor on {dev}")
        _require(t.dtype == torch.bfloat16,
                 f"{name} dtype {t.dtype} != bfloat16 (pass the autocast casts lin_w_c / lin_b_c)")
        _require(tuple(t.shape) == shape and _strides_ok(t, strides),
                 f"{name} shape {tuple(t.shape)} strides {tuple(t.stride())} != {shape} contiguous")
        _ptr_class_ok(name, t)
    return m


def check_wgrad(dh: torch.Tensor, table: torch.Tensor, idx: torch.Tensor, nh: int,
                out_dtype: torch.dtype = torch.float32) -> int:
    """Raise IndexedL1Unsupported unless (dh, table, idx, nh, out_dtype) is the validated wgrad class. -> M."""
    m = _check_table_idx(table, idx, nh)
    dev = table.device
    _require(out_dtype in (torch.float32, torch.bfloat16), f"out_dtype {out_dtype} (fp32 or bf16 parameters only)")
    _require(isinstance(dh, torch.Tensor) and dh.device == dev, f"dh must be a tensor on {dev}")
    _require(dh.dtype == torch.bfloat16, f"dh dtype {dh.dtype} != bfloat16")
    _require(dh.dim() == 2 and tuple(dh.shape) == (m, HID) and _strides_ok(dh, (HID, 1)),
             f"dh shape {tuple(dh.shape)} strides {tuple(dh.stride())} != ({m}, {HID}) contiguous")
    _ptr_class_ok("dh", dh)
    return m


def check_index_range(idx: torch.Tensor, num_rows: int) -> Optional[str]:
    """Optional debug check (ONE host sync): None, or why idx is not inside [0, num_rows). The kernels read
    U[idx] unchecked; the EC dedup guarantees the range structurally."""
    if idx.numel() == 0:
        return None
    lo, hi = (int(v) for v in torch.stack([idx.min(), idx.max()]).tolist())
    if lo < 0 or hi >= num_rows:
        return f"idx range [{lo}, {hi}] outside [0, {num_rows})"
    return None


# ------------------------------------------------------------------------------------------------------------ wrappers
def nsplit(M: int, nh: int) -> Tuple[int, int, int]:
    """(NT, S, CHUNK) of the wgrad split-K (iter82 gidxw nsplit at PROGS 256). Depends on (M, nh) only, so a given
    M always sums in the same order. M = 0 -> (NT, 0, 0) (nothing is launched)."""
    kt = nh * WIDTH
    nt = (HID // WG_BN) * (kt // WG_BKO)
    if M <= 0:
        return nt, 0, 0
    s0 = max(1, WG_PROGS // nt)
    chunk = _cdiv(_cdiv(M, s0), WG_BM) * WG_BM
    return nt, _cdiv(M, chunk), chunk


def workspace_numel(M: int, nh: int) -> int:
    """fp32 elements of the split-K workspace: S * 256 * nh * 512 (<= 16,777,216 = 64 MiB for every M)."""
    return nsplit(M, nh)[1] * HID * nh * WIDTH


def ecix_fwd(table: torch.Tensor, idx: torch.Tensor, nh: int, w1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """h [M, 256] bf16 = addmm(b1, gather(table, idx).view(M, nh*512), w1.t()) without materializing the gather.
    table: U_bf16 [U, 512] contiguous; idx: int64 [M*nh] (cat: interleaved artist/album); w1: bf16 [256, nh*512]
    contiguous (lin_w_c); b1: bf16 [256] (lin_b_c). Raises IndexedL1Unsupported before launching (fail closed)."""
    m = check_fwd(table, idx, nh, w1, b1)
    out = torch.empty((m, HID), dtype=torch.bfloat16, device=table.device)
    if m == 0:
        return out
    grid = (_cdiv(m, FWD_BM) * (HID // FWD_BN),)
    LAST_KERNELS["fwd%d" % nh] = _ecix_fwd[grid](
        table, idx, w1, b1, out, m,
        NH=nh, D=WIDTH, NOUT=HID, BM=FWD_BM, BN=FWD_BN, BK=FWD_BK, **FWD_KW)
    return out


def ecix_wgrad(dh: torch.Tensor, table: torch.Tensor, idx: torch.Tensor, nh: int,
               out_dtype: torch.dtype = torch.float32,
               workspace: Optional[torch.Tensor] = None) -> torch.Tensor:
    """dW1 [256, nh*512] = out_dtype(bf16(dh.t() @ gather(table, idx).view(M, nh*512))) over ALL M tokens, split-K
    with a fixed-order reduce (= _match(dh.t() @ x, param dtype) of ship13 up to fp32 summation order).
    dh: bf16 [M, 256] contiguous. out_dtype: the parameter dtype (fp32 in production). workspace: optional fp32
    contiguous buffer of >= workspace_numel(M, nh) elements (default: a per-call caching-allocator buffer).
    Raises IndexedL1Unsupported before launching (fail closed)."""
    m = check_wgrad(dh, table, idx, nh, out_dtype)
    kt = nh * WIDTH
    dev = table.device
    out = torch.empty((HID, kt), dtype=out_dtype, device=dev)
    if m == 0:  # torch: dh.t() @ x over zero tokens is all +0.0
        return out.zero_()
    nt, s, chunk = nsplit(m, nh)
    need = s * HID * kt
    _require(need < 2**31, "split-K workspace offsets overflow int32")  # S * NEL <= 2^24 by the split rule
    if workspace is None:
        ws = torch.empty((need,), dtype=torch.float32, device=dev)
    else:
        _require(workspace.device == dev and workspace.dtype == torch.float32 and workspace.is_contiguous()
                 and workspace.numel() >= need, f"workspace must be contiguous fp32 on {dev} with >= {need} elements")
        _ptr_class_ok("workspace", workspace)
        ws = workspace
    wgrad_kernel = _ecix_wgrad if wgrad_idx_after_dot() else _ecix_wgrad_idxpf  # item (83)
    LAST_KERNELS["wgrad%d" % nh] = wgrad_kernel[(nt * s,)](
        dh, table, idx, ws, m, chunk,
        NH=nh, D=WIDTH, NOUT=HID, BN=WG_BN, BKO=WG_BKO, BM=WG_BM, NUM_XCD=WG_NUM_XCD,
        TAIL_ZERO_B=TAIL_ZERO_B, **WG_KW)
    LAST_KERNELS["reduce%d" % nh] = _ecix_wgrad_reduce[(HID * kt // RED_BLOCK,)](
        ws, out, s, NEL=HID * kt, BLOCK=RED_BLOCK, **RED_KW)
    return out


# ------------------------------------------------------------------------------------------------------------ references
def materialize_x(table: torch.Tensor, idx: torch.Tensor, nh: int) -> torch.Tensor:
    """The bytes the shipped EC expand writes (ec_expand_gather.expand_rows / torch.gather): [M, nh*512] bf16."""
    m = idx.numel() // nh
    return table.index_select(0, idx).view(m, nh * WIDTH)


def reference_fwd(table: torch.Tensor, idx: torch.Tensor, nh: int, w1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """The shipped forward bit for bit: torch.addmm(lin_b_c, x_c, lin_w_c.t()) on the materialized X (preproc_fp8_l2
    _LinearSwishLNEmitFunction.forward / fused_linear_norm _LinearNormFunction.forward). Fallback and tests."""
    return torch.addmm(b1, materialize_x(table, idx, nh), w1.t())


def reference_wgrad(dh: torch.Tensor, table: torch.Tensor, idx: torch.Tensor, nh: int,
                    out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The shipped wgrad bit for bit: _match(dh.t() @ x, param dtype). Fallback (re-materializes X) and tests."""
    return (dh.t() @ materialize_x(table, idx, nh)).to(out_dtype)


# ------------------------------------------------------------------------------------------------------------ prebuild
def _tern(r: int, c: int, a: int, b: int) -> torch.Tensor:
    i = torch.arange(r, dtype=torch.int64)[:, None]
    j = torch.arange(c, dtype=torch.int64)[None, :]
    return ((i * a + j * b) % 3 - 1).to(torch.float64)


def _prebuild_inputs(nh: int, m: int, n_rows: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor]:
    """CPU fp64 integer-valued {-1, 0, 1} table / w1 / b1 / dh and an int64 idx with a hot row spanning many
    blocks, the last row (U-1) and an artist / album split of the rows (cat). No RNG."""
    t = torch.arange(m, dtype=torch.int64)
    if nh == 1:
        idx = torch.where(t < 3000, torch.zeros_like(t), 1 + (t * 7) % (n_rows - 1))
    else:
        half = n_rows // 2
        art = torch.where(t < 2500, torch.full_like(t, 1), (t * 5) % half)
        alb = half + (t * 11) % (n_rows - half)
        idx = torch.stack([art, alb], dim=1).reshape(-1)
    idx[min(5, idx.numel() - 1)] = n_rows - 1
    kt = nh * WIDTH
    return (_tern(n_rows, WIDTH, 5, 3), idx, _tern(HID, kt, 7, 2 + nh), _tern(1, HID, 3, 1).view(HID),
            _tern(m, HID, 13, 7))


def _prebuild_reference(table: torch.Tensor, idx: torch.Tensor, nh: int, w1: torch.Tensor, b1: torch.Tensor,
                        dh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact (integer) fp64 reference on the CPU, rounded once like the kernels: (h bf16 [M, 256], dW fp32)."""
    m = idx.numel() // nh
    h = b1.expand(m, HID).clone()
    dw = torch.zeros((HID, nh * WIDTH), dtype=torch.float64)
    for q in range(nh):
        iq = idx[q::nh]
        h += (table @ w1[:, q * WIDTH:(q + 1) * WIDTH].t())[iq]  # row u's product, per token: exact integers
        g = torch.zeros((table.shape[0], HID), dtype=torch.float64).index_add_(0, iq, dh)
        dw[:, q * WIDTH:(q + 1) * WIDTH] = g.t() @ table  # exact integers (reference only)
    return h.float().to(torch.bfloat16), dw.float().to(torch.bfloat16).float()


def prebuild(device: Any, m: int = 20100, n_rows: int = 96) -> str:
    """Before RUN_START (e.g. from blas_warmup, like E-X2b's prebuild): compile the production specialization of
    _ecix_fwd / _ecix_wgrad (or _ecix_wgrad_idxpf, item (83) off) / _ecix_wgrad_reduce for both members (fresh
    16-byte-aligned tensors within 2 GiB, M unspecialized, fp32 dW1: the class every production call is checked
    into) and check every output EXACTLY against an fp64 CPU reference on integer-valued inputs (no device GEMM,
    no RNG). M = 20,100 exercises a masked forward block (4 rows), S > 1 splits and a last split with full blocks
    plus a masked tail (item S 105 / CHUNK 192 / last 132 rows; cat S 63 / CHUNK 320 / last 260 rows). Raises on
    any mismatch (the caller latches Indexed-X off)."""
    dev = torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    bad = []
    for nh in (1, 2):
        table, idx, w1, b1, dh = _prebuild_inputs(nh, m, n_rows)
        h_ref, dw_ref = _prebuild_reference(table, idx, nh, w1, b1, dh)

        def bf(x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.bfloat16).to(dev)

        h = ecix_fwd(bf(table), idx.to(dev), nh, bf(w1), bf(b1))
        dw = ecix_wgrad(bf(dh), bf(table), idx.to(dev), nh, out_dtype=torch.float32)
        torch.cuda.synchronize(dev)
        nb_h = int((h.cpu().view(torch.int16) != h_ref.view(torch.int16)).sum())
        nb_w = int((dw.cpu().view(torch.int32) != dw_ref.view(torch.int32)).sum())
        if nb_h or nb_w:
            bad.append(f"nh={nh}: h {nb_h} / dW {nb_w} elements differ")
    if bad:
        raise RuntimeError(f"{TAG} prebuild differs from the exact reference: " + "; ".join(bad))
    wg = "_ecix_wgrad" if wgrad_idx_after_dot() else "_ecix_wgrad_idxpf"  # item (83): the kernel just checked
    msg = (f"{TAG} pre-run_start prebuild OK (_ecix_fwd, {wg}, _ecix_wgrad_reduce x item+cat; exact check "
           f"M={m} U={n_rows} bitwise, device={dev})")
    _say(msg)
    return msg
