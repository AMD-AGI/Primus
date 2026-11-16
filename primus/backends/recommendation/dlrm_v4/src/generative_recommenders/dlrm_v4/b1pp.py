"""B1'' sparse-chain removals (comm iter 46): fused KJT recat, TBE rebag, Triton bounds check.

Three independent parts for the 2n/4n critical sparse chain (the forward
sequence all-to-all's last arriver), yambda_5b.gin ship14 item (64).
Each part only REMOVES kernels:
every output value, gradient, optimizer state and bounds-check repair is
bit-identical to the stock path. (The fused seq-a2a K1-K4 of the same bundle
live in ``seq_a2a_fused.py`` / ``seq_a2a_rowperm.py``; only their pre-RUN_START
warmup is here, in ``run_first_use_warmup``.)

(B) Fused KJT recat -- ``DLRM_HSTU_FP8_KJT_RECAT_FUSED``.
    ``host_sync_free.p_kjtt_wait`` (equal-stride branch) runs
    ``fbgemm.permute_2D_sparse_data(recat, lengths.view(R, B), values)`` on the
    KJT lengths/values all-to-all output (7 kernels with its scans and fill),
    and the TBE prologue later recomputes the KJT offsets with
    ``asynchronous_complete_cumsum`` (3 more). Two Triton kernels do both:
      ``_b1pp_recat_rows_kernel``   grid (R,): permuted lengths of output row r
                                    = lengths of input row recat[r]; row sum.
      ``_b1pp_recat_values_kernel`` grid (R, P): each output row's values are
                                    ONE contiguous block of the input (rows are
                                    permuted whole, sample order kept), copied
                                    in BLOCK chunks; program column 0 also
                                    writes the KJT offsets (complete cumsum of
                                    the permuted lengths, int64, R*B+1).
    No in-kernel cross-program scan, no atomics, no spin-wait: the stream order
    of the two launches is the only global barrier and every K_B program
    recomputes its exclusive prefixes from all R row sums (in BLOCK_R-row
    chunks), so correctness does not depend on co-residency (comm iter 43
    residency stress: 20/20 bit-identical up to R=14336, grid 112x residency).
    The KJT is then built with ``offsets=`` in addition to ``lengths=``;
    ``features.offsets()`` in the TBE call returns it (no cumsum) when the
    lookup takes the whole KJT (one sequence TBE group: 2n/4n row-wise);
    ``KeyedJaggedTensor.split`` children (1n, several TBE groups) recompute
    theirs from the sliced lengths as stock (torchrec jagged_tensor.py split:
    ``offsets=None``), and the parent's offsets are then simply unused.
    Engages only when ``recat`` is an int32 permutation of the R lengths rows
    (torchrec ``_get_recat`` without variable batch always is: equal strides,
    numel == R; stock ``permute_2D_sparse_data`` reads the recat as int32),
    weights are None, lengths/values are contiguous int64 on the GPU.

(C) TBE rebag "Rkeep" -- ``DLRM_HSTU_FP8_TBE_REBAG``.
    For the pooling-NONE (sequence) TBE the bag structure enters the math in one
    place only: the ROCm backward launch predicate ``total_L/total_B > 1``
    (int32 / int64 integer division; fbgemm 10b7757
    embedding_backward_split_template.cu:719,1083,1086,1219). Every output row,
    the WARNING repair and counter, linearized ids, the radix sort, run lengths
    and every gradient segment sum are bag-invariant (comm iter 43 REBAG.md
    section 1.4). So the forward hands fbgemm a re-cut bag partition of the SAME
    indices: each feature t's index range [S_t, S_t + n_t) is cut into Bp
    even bags, ``out[t*Bp + j] = S_t + min(j * ceil(n_t/Bp), n_t)``,
    ``out[T*Bp] = L``, with Bp = max(1, ceil((L//2 + 1)/T)) -- the fewest bags
    that keep the predicate FALSE (L // (T*Bp) <= 1). S_t and n_t are read on
    the device (``_b1pp_rebag_kernel``, one launch, no host sync). The per-bag
    kernels (bounds check v2, nobag forward, linearize) then launch ~6x fewer
    bags. Not used when the original predicate is TRUE, when L <= 0 or
    L >= 2^31, or when it would not shrink the bag count. The bit-identity
    assumes well-formed KJT offsets (torchrec builds them as a cumsum of the
    lengths). With malformed offsets and (D) off, fbgemm's WARNING repair acts
    on the temporary rebagged copy instead of the KJT offsets, so the counter
    and the KJT offsets would differ from stock (never the case in production).

(D) Triton bounds check -- ``DLRM_HSTU_FP8_TBE_BOUNDS_TRITON``.
    fbgemm ``bounds_check_indices`` (WARNING, v2, non-VBE) is replaced by two
    Triton kernels on the ORIGINAL KJT offsets, then fbgemm runs with
    ``bounds_check_mode`` NONE for that call (restored in ``finally``):
      ``_b1pp_bc_validate_kernel`` per 1024-bag tile: bad-bag count (and the
                                   last-offset check), no atomics; zeroes the
                                   warning counter (fbgemm: warning.zero_()).
      ``_b1pp_bc_check_kernel``    no bad bag and offsets[total_B] == N: a flat,
                                   coalesced index check, table of index i from
                                   the T table-end offsets; otherwise a port of
                                   the v2 kernel (same clamp, same +1 per bad
                                   bag / bad index / last element).
    Well-formed offsets (offsets[0] >= 0, non-decreasing, offsets[total_B] ==
    N; every torchrec KJT): the same repairs (invalid id -> 0 in place, -1
    pruned ids skipped) and the same counter as v2 (comm iter 43 lab: S1
    bit-identical). Malformed offsets (not producible by a cumsum) take the
    slow path: its outcome is one of the interleavings v2 itself can produce,
    but not necessarily the one v2 shows on gfx950 (iter-43 lab, S2 "last
    offset high": v2 counted 1202 on every run, this kernel 1201, repaired
    tensors equal; racy cases can also differ in the repaired tensors).
    fbgemm's device printf on a bad id or offset is NOT reproduced: the repairs
    and the counter (``emb.bounds_check_warning``) are the only effects. It
    runs BEFORE the rebag (it may repair the offsets in place).

Hooks: (B) ``host_sync_free.p_kjtt_wait`` calls ``kjt_recat``; (C)+(D)
``install()`` (from ``dlrm_v4/utils.py`` MetricsLogger.__init__, next to
``seq_a2a_rowperm.install()``) patches torchrec
``BatchedFusedEmbedding.forward`` with ``_b1pp_tbe_forward`` -- only when C or
D is on; otherwise nothing is patched. The warmup ``run_first_use_warmup``
(train_ranker.py, before the allocator pre-grow and RUN_START) compiles the
new kernels on synthetic tensors, and the fused seq-a2a K1-K4 with the shapes
of the model's sequence all-to-alls (``_warm_seq_a2a``).

Switches (env; read once at first use; "1" = on, anything else or absent =
off in code; the yambda_5b.gin ``apply_env_bootstrap`` bindings are True, which
``train/_env_bootstrap.py`` turns into "1" only when the job spans more than
one node (item (64)'s multi-node default; "0" at one node); a pre-set env var
wins ("1" forces a part on, "0" is its kill switch); ``launch_slurm.sh``
forwards the ``DLRM_HSTU_FP8_`` prefix):
  DLRM_HSTU_FP8_KJT_RECAT_FUSED=1   (B)
  DLRM_HSTU_FP8_TBE_REBAG=1         (C)
  DLRM_HSTU_FP8_TBE_BOUNDS_TRITON=1 (D)
  DLRM_HSTU_FP8_B1PP_VERIFY=N       first N engaged calls of each part (per
                                    rank) are compared with the stock path
                                    (syncs); a mismatch raises RuntimeError.
                                    Bounds: exact for well-formed offsets; on
                                    malformed offsets (fbgemm is racy there)
                                    the outcome class only, with a WARNING.
Arming (INFO, once per process and kind, every rank, with the shape and rank):
  "DLRMV4 armed OK: b1pp kjt recat fused (R=.. B=.. N=.. offsets=1) rank=r"
  "DLRMV4 armed OK: b1pp tbe rebag Rkeep (T=.. B=.. Bp=.. L=..) rank=r"
  "DLRMV4 armed OK: b1pp tbe bounds triton (T=.. B=.. N=.. mode=WARNING) rank=r"
  "DLRMV4 b1pp <part> FALLBACK: <reason> ..." once per (part, reason).
  "DLRMV4 armed OK: [b1pp-warmup] kernels=.. triton_jit_new=.. in ..s rank=r"
  "DLRMV4 armed OK: [b1pp-warmup] seq_a2a K1-K4 variants=.. triton_jit_new=..
  in ..s rank=r (<kernel> D=.. ...)" (or "... seq_a2a K1-K4 not warmed (<why>)")
  "DLRMV4 b1pp jit after run_start: b1pp +<n> seq_a2a +<m> (<per-kernel>) ..."
  once, after the 64th engaged TBE call (or recat call when only (B) is on);
  <n> counts the five B1'' kernels only, <m> the fused seq-a2a K1-K4 (both
  should read +0 after the warmup).
Hardening: a kernel exception logs one WARNING, that call runs stock and the
part stays off for the rest of the run (as ``seq_a2a_rowperm._fused_failed``).
The warmup treats a failed launch the same way, but a failed synchronize after
its launches (a sticky device fault: the HIP context is unusable) raises
``WarmupDeviceFault`` before RUN_START, as ``blas_warmup.warmup`` does.
``install()`` refuses to patch (WARNING, stock) unless torchrec's
``BatchedFusedEmbedding.forward`` is the ``BaseBatchedEmbedding.forward`` body
that ``_b1pp_tbe_forward`` re-implements (same arguments, same referenced
names), so a torchrec change cannot be bypassed silently on engaged calls.

Specialization (no Triton recompile inside the timed region): every per-call
integer kernel argument is ``do_not_specialize`` (ints < 2^31 always bind as
i32); every block constexpr is ONE fixed value for every shape (recat
BLOCK_B = BLOCK = 1024, BLOCK_R = 512; bounds BB = BLOCK = 1024, BB2 = 16,
CH = 64, MAX_T = 16, BLOCK_F = 512; rebag BLOCK = 1024): R > 512 rows and
total_B > 512*1024 bags are summed in further chunks by a loop that runs zero
times for production shapes; the bounds kernel's ``indices_ptr`` is also
``do_not_specialize_on_alignment`` (a KJT split child's values are a view
whose 16-byte alignment changes per step; the gfx950 code is the same load/
store mix either way). Sizes are capped at 2^31 - 2^17 so the kernels' int32
index arithmetic (loop steps up to 64*1024, tails up to 1024 lanes) never
wraps; larger calls run stock.
"""

import logging
import os
import sys
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple

import torch

try:  # Triton is optional at import time; the parts check _HAVE_TRITON.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAVE_TRITON = False

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Switches, constants, state
# ---------------------------------------------------------------------------
_RECAT_ENV = "DLRM_HSTU_FP8_KJT_RECAT_FUSED"
_REBAG_ENV = "DLRM_HSTU_FP8_TBE_REBAG"
_BOUNDS_ENV = "DLRM_HSTU_FP8_TBE_BOUNDS_TRITON"
_VERIFY_ENV = "DLRM_HSTU_FP8_B1PP_VERIFY"
_ENVS: Dict[str, str] = {"recat": _RECAT_ENV, "rebag": _REBAG_ENV, "bounds": _BOUNDS_ENV}
_PARTS: Tuple[str, ...] = ("recat", "rebag", "bounds")

# Fixed block constexprs: one compiled variant per kernel for EVERY shape (no
# shape-dependent constexpr; production: recat R = keys x workers <= 448, bounds
# total_B <= 458,752, T <= 14 -- the R > BLOCK_R / n_bad > BLOCK_F chunk loops
# then run zero times).
RECAT_BLOCK_B = 1024
RECAT_BLOCK = 1024
RECAT_BLOCK_R = 512
RECAT_MAX_R = 16384  # largest R of the iter-43 residency stress
RECAT_NUM_WARPS = 4
BC_BB = 1024
BC_BLOCK = 1024
BC_BB2 = 16
BC_CH = 64
BC_MAX_T = 16
BC_BLOCK_F = 512
BC_NUM_WARPS = 4
REBAG_BLOCK = 1024
REBAG_NUM_WARPS = 4
JIT_REPORT_AFTER = 64  # engaged TBE calls before the "jit after run_start" line

_I32_MAX = 2**31 - 1
# Largest N, R*B+1 and offsets.numel() the recat / bounds kernels take: their
# index arithmetic is int32 (do_not_specialize ints bind i32), the recat values
# loop steps by up to P*BLOCK = 64*1024 and every tail block has <= 1024 lanes,
# so 2^17 of headroom below 2^31 keeps every index and loop counter from
# wrapping. Production sizes are ~1e5-3e6. (The rebag kernel indexes in int64;
# rebag_plan keeps its own L < 2^31 rule, fbgemm's int32 total_L.)
_SIZE_MAX = 2**31 - 2**17
_SIZE_REASON = "size > 2^31 - 2^17 (int32 index headroom)"
_T_REASON = "T > %d" % BC_MAX_T
# torchrec BaseBatchedEmbedding.forward as _b1pp_tbe_forward re-implements it
# (production overlay torchrec 1.7.0a0+bf55480, batched_embedding_kernel.py:1880-1902,
# CPython 3.12): its argument count and the global/attribute names its code
# references. install() refuses to patch when the live function differs.
_BASE_FORWARD_ARGCOUNT = 2
_BASE_FORWARD_NAMES = frozenset(
    {
        "_embedding_table_index_type",
        "_embedding_table_offset_type",
        "_get_hash_zch_identities_and_metadata",
        "emb_module",
        "len",
        "offsets",
        "to",
        "values",
    }
)
_WARNING = 1  # fbgemm BoundsCheckMode.WARNING
_NONE = 3  # fbgemm BoundsCheckMode.NONE

KERNEL_NAMES: Tuple[str, ...] = (
    "_b1pp_recat_rows_kernel",
    "_b1pp_recat_values_kernel",
    "_b1pp_bc_validate_kernel",
    "_b1pp_bc_check_kernel",
    "_b1pp_rebag_kernel",
)
_PART_KERNELS: Dict[str, Tuple[str, ...]] = {
    "recat": ("_b1pp_recat_rows_kernel", "_b1pp_recat_values_kernel"),
    "bounds": ("_b1pp_bc_validate_kernel", "_b1pp_bc_check_kernel"),
    "rebag": ("_b1pp_rebag_kernel",),
}
# The fused seq-a2a K1-K4 (seq_a2a_fused.py) are reported by jit_cache_sizes()
# too; run_first_use_warmup compiles them from the model's sequence all-to-alls
# (_warm_seq_a2a) when that path is on.
SEQ_A2A_KERNEL_NAMES: Tuple[str, ...] = (
    "_seq_a2a_send_fwd_kernel",
    "_seq_a2a_send_bwd_kernel",
    "_seq_a2a_recv_fwd_kernel",
    "_seq_a2a_recv_bwd_kernel",
)


class VerifyMismatch(RuntimeError):
    """DLRM_HSTU_FP8_B1PP_VERIFY found a difference from the stock path."""


def _new_state() -> Dict[str, Any]:
    return {
        "switch": {},  # part -> bool, read once
        "verify_left": None,  # part -> int, read once
        "disabled": set(),  # parts turned off after a kernel exception
        "armed": set(),
        "fallback_logged": set(),  # (part, reason)
        "fail_logged": set(),
        "engaged": {"recat": 0, "tbe": 0},
        "jit_base": None,  # kernel -> count (warmup end, or first use per part)
        "jit_base_src": None,  # "warmup" | "first-use"
        "jit_base_parts": set(),
        "jit_logged": False,
        "warmup_ran": False,
        "recat_requested_logged": False,
    }


_state: Dict[str, Any] = _new_state()
_installed = False
_install_refused: Optional[str] = None  # why install() did not patch (torchrec forward differs)
_patched_cls: Any = None
_had_own_forward = False
_orig_forward: Any = None  # the forward BatchedFusedEmbedding resolved before the patch
_base_forward: Any = None  # BaseBatchedEmbedding.forward (kept as a reference)
_static_cache: "weakref.WeakKeyDictionary[Any, Tuple[Optional[str], int]]" = weakref.WeakKeyDictionary()


def _switch(part: str) -> bool:
    on = _state["switch"].get(part)
    if on is None:
        on = os.environ.get(_ENVS[part]) == "1"
        _state["switch"][part] = on
    return on


def recat_on() -> bool:
    """(B) DLRM_HSTU_FP8_KJT_RECAT_FUSED, read once."""
    return _switch("recat")


def rebag_on() -> bool:
    """(C) DLRM_HSTU_FP8_TBE_REBAG, read once."""
    return _switch("rebag")


def bounds_on() -> bool:
    """(D) DLRM_HSTU_FP8_TBE_BOUNDS_TRITON, read once."""
    return _switch("bounds")


def any_on() -> bool:
    return recat_on() or rebag_on() or bounds_on()


def _active(part: str) -> bool:
    return _switch(part) and part not in _state["disabled"]


def _verify_left() -> Dict[str, int]:
    left = _state["verify_left"]
    if left is None:
        try:
            n = int(os.environ.get(_VERIFY_ENV, "0") or 0)
        except ValueError:
            n = 0
        left = _state["verify_left"] = {p: max(0, n) for p in _PARTS}
    return left


def _verify_peek(part: str) -> bool:
    """VERIFY budget left for `part` (nothing spent)."""
    return _verify_left().get(part, 0) > 0


def _verify_take(part: str) -> bool:
    """Spend one VERIFY call of `part` (only for an engaged call)."""
    left = _verify_left()
    if left.get(part, 0) > 0:
        left[part] -= 1
        return True
    return False


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(os.environ.get("RANK", "0") or 0)
    except ValueError:
        return 0


def _fallback(part: str, reason: str, detail: str = "", *args: Any) -> None:
    """Once per (part, reason). ``detail % args`` is formatted only for that first
    line, so a per-call caller (e.g. a predicate-true TBE group every step) pays
    a set lookup, not a string format."""
    key = (part, reason)
    if key in _state["fallback_logged"]:
        return
    _state["fallback_logged"].add(key)
    if args:
        detail = detail % args
    logger.info(
        "DLRMV4 b1pp %s FALLBACK: %s%s rank=%d",
        part,
        reason,
        f" ({detail})" if detail else "",
        _rank(),
    )


def log_fallback(part: str, reason: str, detail: str = "", *args: Any) -> None:
    """Public form of the once-per-(part, reason) FALLBACK line (host_sync_free uses it)."""
    _fallback(part, reason, detail, *args)


def _failed(part: str, where: str, e: BaseException) -> None:
    """A kernel raised: stock for this call and, for this part, the rest of the run."""
    _state["disabled"].add(part)
    if part not in _state["fail_logged"]:
        _state["fail_logged"].add(part)
        logger.warning(
            "DLRMV4 b1pp %s: %s raised %s: %s; %s path DISABLED for the rest of the run "
            "(stock) rank=%d",
            part,
            where,
            type(e).__name__,
            e,
            part,
            _rank(),
        )


def _arm(part: str, msg: str) -> None:
    if part not in _state["armed"]:
        _state["armed"].add(part)
        logger.info("%s rank=%d", msg, _rank())


def interpret_mode() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0").strip() == "1"


def _dev_ok(t: torch.Tensor) -> bool:
    """Compiled kernels need a GPU tensor; CPU tensors only under TRITON_INTERPRET=1."""
    return t.is_cuda or (t.device.type == "cpu" and interpret_mode())


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _DeviceGuard:
    """Launch on the tensors' device (Triton launches on the current device)."""

    def __init__(self, t: torch.Tensor) -> None:
        self._ctx = (
            torch.cuda.device(t.device)
            if t.is_cuda and t.device.index != torch.cuda.current_device()
            else None
        )

    def __enter__(self) -> None:
        if self._ctx is not None:
            self._ctx.__enter__()

    def __exit__(self, *exc: Any) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*exc)


# ---------------------------------------------------------------------------
# Kernels (ported from the GPU-verified comm iter 43 prototypes:
# probes/iter43/lab/comm43_kernels.py and probes/iter43/rebag/rebag_lab.py)
# ---------------------------------------------------------------------------
if _HAVE_TRITON:

    @triton.jit(do_not_specialize=["B"])
    def _b1pp_recat_rows_kernel(
        recat_ptr, lengths_ptr, out_lengths_ptr, row_sum_ptr, B, BLOCK_B: tl.constexpr
    ):
        # K_A, one program per output row r: permuted lengths + the row sum.
        r = tl.program_id(0)
        r_in = tl.load(recat_ptr + r).to(tl.int64)
        acc = tl.zeros((BLOCK_B,), dtype=tl.int64)
        for b0 in range(0, B, BLOCK_B):
            o = b0 + tl.arange(0, BLOCK_B)
            m = o < B
            x = tl.load(lengths_ptr + r_in * B + o, mask=m, other=0)
            tl.store(out_lengths_ptr + r.to(tl.int64) * B + o, x, mask=m)
            acc += x.to(tl.int64)
        tl.store(row_sum_ptr + r, tl.sum(acc, axis=0))

    @triton.jit(do_not_specialize=["R", "P", "B"])
    def _b1pp_recat_values_kernel(
        recat_ptr,
        row_sum_ptr,
        values_ptr,
        out_values_ptr,
        R,
        P,
        out_lengths_ptr,
        out_offsets_ptr,
        B,
        BLOCK_R: tl.constexpr,
        BLOCK: tl.constexpr,
        WITH_OFFSETS: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        # K_B, grid (R, P): copy output row r's values block; column 0 writes offsets.
        r = tl.program_id(0)
        c = tl.program_id(1)
        ro = tl.arange(0, BLOCK_R)
        mr = ro < R
        rs = tl.load(row_sum_ptr + ro, mask=mr, other=0)
        rc = tl.load(recat_ptr + ro, mask=mr, other=0)
        r_in = tl.load(recat_ptr + r)
        # output rows before r, and input rows before recat[r] (recat is a permutation of [0, R))
        out_base = tl.sum(tl.where(ro < r, rs, 0), axis=0)
        in_base = tl.sum(tl.where(mr & (rc < r_in), rs, 0), axis=0)
        # R > BLOCK_R: the same two sums over the further BLOCK_R-row chunks, so BLOCK_R is one fixed
        # constexpr for every R (no shape-dependent variant). Zero iterations for R <= BLOCK_R (production).
        for r0 in range(BLOCK_R, R, BLOCK_R):
            ro2 = r0 + ro
            mr2 = ro2 < R
            rs2 = tl.load(row_sum_ptr + ro2, mask=mr2, other=0)
            rc2 = tl.load(recat_ptr + ro2, mask=mr2, other=0)
            out_base += tl.sum(tl.where(ro2 < r, rs2, 0), axis=0)
            in_base += tl.sum(tl.where(mr2 & (rc2 < r_in), rs2, 0), axis=0)
        L = tl.load(row_sum_ptr + r).to(tl.int32)
        for k in range(c * BLOCK, L, P * BLOCK):
            o = k + tl.arange(0, BLOCK)
            m = o < L
            v = tl.load(values_ptr + in_base + o, mask=m)
            tl.store(out_values_ptr + out_base + o, v, mask=m)
        if WITH_OFFSETS:
            # KJT offsets = complete cumsum of the permuted lengths (what features.offsets() computes):
            # offsets[0] = 0, offsets[r*B + b + 1] = out_base(r) + inclusive_cumsum(permuted_lengths[r, :])[b]
            if c == 0:
                run = out_base
                for b0 in range(0, B, BLOCK_B):
                    ob = b0 + tl.arange(0, BLOCK_B)
                    mb = ob < B
                    x = tl.load(
                        out_lengths_ptr + r.to(tl.int64) * B + ob, mask=mb, other=0
                    ).to(tl.int64)
                    cs = tl.cumsum(x, axis=0) + run
                    tl.store(out_offsets_ptr + r.to(tl.int64) * B + ob + 1, cs, mask=mb)
                    run += tl.sum(x, axis=0)
                if r == 0:
                    tl.store(out_offsets_ptr, (out_base * 0).to(tl.int64))

    @triton.jit(do_not_specialize=["total_B", "N"])
    def _b1pp_bc_validate_kernel(offsets_ptr, bad_ptr, warning_ptr, total_B, N, BB: tl.constexpr):
        # K1: per-tile bad-bag count (+ the last-offset check in program 0); zeroes the counter.
        pid = tl.program_id(0)
        bt = pid * BB + tl.arange(0, BB)
        m = bt < total_B
        s = tl.load(offsets_ptr + bt, mask=m, other=0)
        e = tl.load(offsets_ptr + bt + 1, mask=m, other=0)
        bad = m & ((s < 0) | (s > e) | (e > N))
        nb = tl.sum(bad.to(tl.int32), axis=0)
        last = tl.load(offsets_ptr + total_B)
        nb = nb + ((last != N) & (pid == 0)).to(tl.int32)
        tl.store(bad_ptr + pid, nb)
        if pid == 0:
            tl.store(warning_ptr, (pid * 0).to(tl.int64))

    @triton.jit(
        do_not_specialize=["N", "total_B", "B", "T", "n_bad"],
        do_not_specialize_on_alignment=["indices_ptr"],
    )
    def _b1pp_bc_check_kernel(
        rows_ptr,
        indices_ptr,
        offsets_ptr,
        bad_ptr,
        warning_ptr,
        N,
        total_B,
        B,
        T,
        n_bad,
        BLOCK: tl.constexpr,
        MAX_T: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BB2: tl.constexpr,
        CH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        f = tl.arange(0, BLOCK_F)
        nbad = tl.sum(tl.load(bad_ptr + f, mask=f < n_bad, other=0), axis=0)
        # n_bad > BLOCK_F tiles (total_B > BLOCK_F * BB): add the further chunks, so BLOCK_F is one fixed
        # constexpr for every shape (no shape-dependent variant). Zero iterations in production.
        for f0 in range(BLOCK_F, n_bad, BLOCK_F):
            nbad += tl.sum(tl.load(bad_ptr + f0 + f, mask=f0 + f < n_bad, other=0), axis=0)
        if nbad == 0:
            # ---- fast path: offsets valid and offsets[total_B] == N; bags of table t cover
            # [offsets[t*B], offsets[(t+1)*B]) contiguously, so the per-index table id is the number of
            # table end offsets <= i.  Indices outside [offsets[0], N) belong to no bag (fbgemm: unchecked).
            i = pid * BLOCK + tl.arange(0, BLOCK)
            start0 = tl.load(offsets_ptr)
            inr = (i >= start0) & (i < N)
            t_of_i = tl.zeros((BLOCK,), dtype=tl.int32)
            for t in tl.static_range(MAX_T):
                # entries t >= T read offsets[T*B] == N (> any in-range i), so they never count
                e_t = tl.load(offsets_ptr + tl.minimum(t + 1, T).to(tl.int64) * B)
                t_of_i += (i >= e_t).to(tl.int32)
            nr = tl.load(rows_ptr + t_of_i, mask=inr, other=0)
            idx = tl.load(indices_ptr + i, mask=inr, other=0)
            badi = inr & (idx != -1) & ((idx < 0) | (idx >= nr))
            tl.store(indices_ptr + i, tl.zeros_like(idx), mask=badi)
            cnt = tl.sum(badi.to(tl.int64), axis=0)
            if cnt > 0:
                tl.atomic_add(warning_ptr, cnt)
        else:
            # ---- slow path: port of bounds_check_indices_kernel_v2 (WARNING, non-VBE), grid-stride over bags.
            # Like fbgemm, reads/writes of shared offsets by neighbouring bags race; the result is one of the
            # interleavings fbgemm itself can produce, not necessarily the one fbgemm shows: here program 0
            # repairs offsets[total_B] before its chunk loop, so the last bag usually reads the repaired value
            # (v2 on gfx950 usually reads the stale one and counts that bag too: +1 in the counter).
            if pid == 0:
                last = tl.load(offsets_ptr + total_B)
                if last != N:
                    tl.store(offsets_ptr + total_B, (pid * 0 + N).to(tl.int64))
                    tl.atomic_add(warning_ptr, (pid * 0 + 1).to(tl.int64))
            n_chunks = tl.cdiv(total_B, BB2)
            for chunk in range(pid, n_chunks, nprog):
                bt = chunk * BB2 + tl.arange(0, BB2)
                mb = bt < total_B
                s = tl.load(offsets_ptr + bt, mask=mb, other=0)
                e = tl.load(offsets_ptr + bt + 1, mask=mb, other=0)
                badb = mb & ((s < 0) | (s > e) | (e > N))
                s2 = tl.minimum(tl.maximum(s, 0), N)
                e2 = tl.maximum(s2, tl.minimum(e, N))
                tl.store(offsets_ptr + bt, s2, mask=badb)
                tl.store(offsets_ptr + bt + 1, e2, mask=badb)
                s = tl.where(badb, s2, s)
                e = tl.where(badb, e2, e)
                nr = tl.load(rows_ptr + bt // B, mask=mb, other=0)
                L = tl.where(mb, e - s, 0).to(tl.int32)
                maxL = tl.max(L, axis=0)
                cnt = tl.sum(badb.to(tl.int64), axis=0)
                for j in range(0, maxL, CH):
                    jj = j + tl.arange(0, CH)
                    m2 = mb[:, None] & (jj[None, :] < L[:, None])
                    pos = s[:, None] + jj[None, :]
                    idx = tl.load(indices_ptr + pos, mask=m2, other=-1)
                    badi = m2 & (idx != -1) & ((idx < 0) | (idx >= nr[:, None]))
                    tl.store(indices_ptr + pos, tl.zeros_like(idx), mask=badi)
                    cnt += tl.sum(tl.sum(badi.to(tl.int64), axis=1), axis=0)
                if cnt > 0:
                    tl.atomic_add(warning_ptr, cnt)

    @triton.jit(do_not_specialize=["B", "Bp", "T", "total", "last_pos"])
    def _b1pp_rebag_kernel(OFF, OUT, B, Bp, T, total, last_pos, BLOCK: tl.constexpr):
        # Rkeep (per-feature even split): out[t*Bp + j] = S_t + min(j * ceil(n_t/Bp), n_t),
        # out[T*Bp] = OFF[T*B] (= L); S_t = OFF[t*B], n_t = OFF[(t+1)*B] - S_t read on the device.
        pid = tl.program_id(0)
        i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = i <= total
        t = i // Bp
        j = i - t * Bp
        tc = tl.minimum(t, T - 1)
        s0 = tl.load(OFF + tc * B, mask=mask, other=0)
        s1 = tl.load(OFF + (tc + 1) * B, mask=mask, other=0)
        n = s1 - s0
        ct = (n + Bp - 1) // Bp
        pos = tl.minimum(j * ct, n)
        last = tl.load(OFF + last_pos)
        val = tl.where(i == total, last, s0 + pos)
        tl.store(OUT + i, val, mask=mask)


# ---------------------------------------------------------------------------
# (B) fused recat: pure wrapper
# ---------------------------------------------------------------------------
def recat_reason(
    recat: Optional[torch.Tensor],
    lengths2d: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """Why the fused recat cannot serve this call (None = eligible). Host-side reads only.

    A recat whose numel equals the number of lengths rows is a permutation of
    [0, R): torchrec ``_get_recat`` without variable batch builds
    ``i + j*local_split`` over all i < local_split and j < num_splits, as an
    int32 tensor (the only recat dtype stock ``permute_2D_sparse_data`` reads,
    fbgemm sparse_permute_2d.cu ``permute.data_ptr<int32_t>()``). The
    permutation property itself is not checked per call (a device read);
    DLRM_HSTU_FP8_B1PP_VERIFY compares the first calls with the stock op."""
    if not _HAVE_TRITON:
        return "triton unavailable"
    if weights is not None:
        return "weights present"
    if recat is None or lengths2d.dim() != 2 or recat.dim() != 1:
        return "recat/lengths shape"
    R, B = lengths2d.shape
    if recat.numel() != R:
        return "recat is not a permutation of the lengths rows (numel != R)"
    if R == 0 or B == 0:
        return "empty (R or B is 0)"
    if R > RECAT_MAX_R:
        return "R > %d" % RECAT_MAX_R
    if recat.dtype != torch.int32:
        return "recat dtype is not int32 (stock permute_2D_sparse_data reads int32)"
    if lengths2d.dtype != torch.int64 or values.dtype != torch.int64:
        return "lengths/values not int64"
    if not (recat.is_contiguous() and lengths2d.is_contiguous() and values.is_contiguous()):
        return "not contiguous"
    if not (_dev_ok(recat) and _dev_ok(lengths2d) and _dev_ok(values)) or not (
        recat.device == lengths2d.device == values.device
    ):
        return "not on one CUDA device"
    if values.numel() > _SIZE_MAX or R * B + 1 > _SIZE_MAX:
        return _SIZE_REASON
    return None


def _recat_launch(
    recat: torch.Tensor,
    lengths2d: torch.Tensor,
    values: torch.Tensor,
    *,
    block: int = RECAT_BLOCK,
    block_b: int = RECAT_BLOCK_B,
    block_r: int = RECAT_BLOCK_R,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R, B = lengths2d.shape
    N = values.numel()
    dev = lengths2d.device
    out_l = torch.empty((R, B), dtype=lengths2d.dtype, device=dev)
    out_v = torch.empty((N,), dtype=values.dtype, device=dev)
    out_o = torch.empty((R * B + 1,), dtype=torch.int64, device=dev)
    rs = torch.empty((R,), dtype=torch.int64, device=dev)
    # programs per row: ~2 per average row's worth of BLOCK chunks (census N 0.1-0.2M -> P=2)
    P = max(1, min(64, 2 * _cdiv(N, R * block)))
    with _DeviceGuard(lengths2d):
        _b1pp_recat_rows_kernel[(R,)](
            recat, lengths2d, out_l, rs, B, BLOCK_B=block_b, num_warps=RECAT_NUM_WARPS
        )
        _b1pp_recat_values_kernel[(R, P)](
            recat,
            rs,
            values,
            out_v,
            R,
            P,
            out_l,
            out_o,
            B,
            BLOCK_R=block_r,
            BLOCK=block,
            WITH_OFFSETS=True,
            BLOCK_B=block_b,
            num_warps=RECAT_NUM_WARPS,
        )
    return out_l, out_v, out_o


def recat_fused(
    recat: torch.Tensor, lengths2d: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(permuted lengths [R, B], permuted values [N], KJT offsets int64 [R*B+1]).

    Bit-identical to ``fbgemm.permute_2D_sparse_data(recat, lengths2d, values,
    None, N)[:2]`` plus ``asynchronous_complete_cumsum`` of the permuted
    lengths, for an int32 recat that is a permutation of the R rows. Launches on the
    current stream; no host sync. Raises ValueError when not eligible
    (``recat_reason``; e.g. R == 0 or B == 0: the caller runs stock). N == 0 is
    served (offsets all zero, empty values)."""
    why = recat_reason(recat, lengths2d, values)
    if why is not None:
        raise ValueError("b1pp.recat_fused: not eligible: " + why)
    return _recat_launch(recat, lengths2d, values)


# ---------------------------------------------------------------------------
# (D) Triton bounds check: pure wrapper
# ---------------------------------------------------------------------------
def _bounds_launch(
    rows_per_table: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    warning: torch.Tensor,
    *,
    BB: int = BC_BB,
    BLOCK: int = BC_BLOCK,
    BB2: int = BC_BB2,
    CH: int = BC_CH,
    MAX_T: int = BC_MAX_T,
    BLOCK_F: int = BC_BLOCK_F,
) -> None:
    T = rows_per_table.numel()
    total_B = offsets.numel() - 1
    if total_B <= 0 or T == 0:
        return  # fbgemm returns before warning.zero_() too
    B = total_B // T
    if offsets.numel() != B * T + 1:
        raise ValueError(
            "b1pp.bounds_check_warning: offsets size %d is not B (%d) * T (%d) + 1"
            % (offsets.numel(), B, T)
        )
    if T > MAX_T:
        raise ValueError("b1pp.bounds_check_warning: T=%d > MAX_T=%d" % (T, MAX_T))
    if (
        indices.dtype != torch.int64
        or offsets.dtype != torch.int64
        or rows_per_table.dtype != torch.int64
        or warning.dtype != torch.int64
        or warning.numel() < 1
    ):
        raise ValueError("b1pp.bounds_check_warning: int64 indices/offsets/rows/warning needed")
    if not (indices.is_contiguous() and offsets.is_contiguous() and rows_per_table.is_contiguous()):
        raise ValueError("b1pp.bounds_check_warning: contiguous tensors needed")
    N = indices.numel()
    if N > _SIZE_MAX or offsets.numel() > _SIZE_MAX:
        raise ValueError("b1pp.bounds_check_warning: " + _SIZE_REASON)
    n_bad = _cdiv(total_B, BB)
    bad = torch.empty((n_bad,), dtype=torch.int32, device=indices.device)
    grid = (max(1, _cdiv(N, BLOCK)),)
    with _DeviceGuard(indices):
        _b1pp_bc_validate_kernel[(n_bad,)](
            offsets, bad, warning, total_B, N, BB=BB, num_warps=BC_NUM_WARPS
        )
        _b1pp_bc_check_kernel[grid](
            rows_per_table,
            indices,
            offsets,
            bad,
            warning,
            N,
            total_B,
            B,
            T,
            n_bad,
            BLOCK=BLOCK,
            MAX_T=MAX_T,
            BLOCK_F=BLOCK_F,
            BB2=BB2,
            CH=CH,
            num_warps=BC_NUM_WARPS,
        )


def bounds_check_warning(
    rows_per_table: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    warning: torch.Tensor,
) -> None:
    """fbgemm ``bounds_check_indices(rows_per_table, indices, offsets, WARNING,
    warning)`` (v2, non-VBE, 10b7757) semantics, in place.

    ``rows_per_table`` is per FEATURE (fbgemm's buffer, T entries). The warning
    counter is zeroed first (fbgemm: warning.zero_()), then +1 per bad bag, per
    invalid index (set to 0; -1 pruned ids skipped) and for a wrong last offset.
    Well-formed offsets (a cumsum): repairs and counter identical to v2.
    Malformed offsets are repaired in place with fbgemm's clamp, racy between
    neighbouring bags as in fbgemm: the result is a legal v2 interleaving, not
    necessarily the one v2 lands on (a high last offset: v2 usually counts the
    last bag too, this kernel usually not). fbgemm's device printf is not
    reproduced. total_B == 0 or T == 0: no-op. Launches on the current stream;
    no host sync."""
    if not _HAVE_TRITON:
        raise RuntimeError("b1pp: triton is not available")
    _bounds_launch(rows_per_table, indices, offsets, warning)


# ---------------------------------------------------------------------------
# (C) Rkeep rebag: plan + pure wrapper
# ---------------------------------------------------------------------------
def rocm_bwd_pred(L: int, total_B: int) -> bool:
    """fbgemm's ROCm nobag-backward predicate ``total_L/total_B > 1`` exactly as the
    host code evaluates it (int32_t total_L = indices.numel(); int64 total_B;
    integer division truncating toward zero). Copied from rebag_lab.rocm_bwd_pred."""
    tl32 = ((int(L) + 2**31) % 2**32) - 2**31
    tb = int(total_B)
    q = abs(tl32) // abs(tb)
    if (tl32 < 0) != (tb < 0):
        q = -q
    return q > 1


def rebag_plan(L: int, T: int, B: int) -> Tuple[Optional[int], Optional[str]]:
    """(Bp, None) for the Rkeep layout, or (None, reason). Host ints only."""
    L, T, B = int(L), int(T), int(B)
    if L <= 0:
        return None, "L <= 0"
    if T <= 0 or B <= 0:
        return None, "T*B <= 0"
    if L > _I32_MAX:
        return None, "L >= 2^31"
    if T * B + 1 > _I32_MAX:
        return None, "T*B+1 >= 2^31"
    if rocm_bwd_pred(L, T * B):
        return None, "original predicate total_L/total_B > 1"
    Bp = max(1, _cdiv(L // 2 + 1, T))
    if T * Bp >= T * B:
        return None, "no gain (T*Bp >= T*B)"
    if rocm_bwd_pred(L, T * Bp):  # T*Bp >= L//2 + 1 > L/2: never by construction
        return None, "rebag would flip the ROCm backward predicate"
    return Bp, None


def _rebag_launch(
    offsets: torch.Tensor, T: int, B: int, Bp: int, *, block: int = REBAG_BLOCK
) -> torch.Tensor:
    total = T * Bp
    out = torch.empty((total + 1,), dtype=offsets.dtype, device=offsets.device)
    with _DeviceGuard(offsets):
        _b1pp_rebag_kernel[(_cdiv(total + 1, block),)](
            offsets, out, B, Bp, T, total, T * B, BLOCK=block, num_warps=REBAG_NUM_WARPS
        )
    return out


def rebag_rkeep(offsets: torch.Tensor, T: int, B: int, L: int) -> Optional[torch.Tensor]:
    """Rebagged int64 offsets [T*Bp + 1] of the same indices, or None (not
    eligible: L <= 0, T*B <= 0, L >= 2^31, original predicate L // (T*B) > 1,
    or T*Bp >= T*B). One launch on the current stream, no host sync."""
    Bp, _ = rebag_plan(L, T, B)
    if Bp is None:
        return None
    if not _HAVE_TRITON:
        raise RuntimeError("b1pp: triton is not available")
    if (
        offsets.dtype != torch.int64
        or offsets.dim() != 1
        or offsets.numel() != T * B + 1
        or not offsets.is_contiguous()
    ):
        raise ValueError("b1pp.rebag_rkeep: contiguous int64 offsets [T*B+1] needed")
    return _rebag_launch(offsets, int(T), int(B), Bp)


# ---------------------------------------------------------------------------
# JIT bookkeeping
# ---------------------------------------------------------------------------
def _jit_cache_size(jit_fn: Any) -> int:
    """Compiled variants across devices, or -1 if unknown (same reading as
    fp8_warmup._jit_cache_size; an interpreted function has no cache)."""
    if jit_fn is None:
        return -1
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


def jit_cache_sizes() -> Dict[str, int]:
    """kernel name -> compiled variants (-1 = unknown): the five B1'' kernels and
    the fused seq-a2a K1-K4 (seq_a2a_fused.py)."""
    out: Dict[str, int] = {}
    g = globals()
    for name in KERNEL_NAMES:
        out[name] = _jit_cache_size(g.get(name))
    try:
        from generative_recommenders.dlrm_v4 import seq_a2a_fused as _saf

        for name in SEQ_A2A_KERNEL_NAMES:
            out[name] = _jit_cache_size(getattr(_saf, name, None))
    except Exception:  # noqa: BLE001
        for name in SEQ_A2A_KERNEL_NAMES:
            out[name] = -1
    return out


def _jit_first_use(part: str) -> None:
    """No warmup ran: the baseline of a part's kernels is taken at the end of its
    first engaged call (K1-K4 at the first engaged call of any part)."""
    if _state["jit_base_src"] == "warmup" or part in _state["jit_base_parts"]:
        return
    _state["jit_base_parts"].add(part)
    cur = jit_cache_sizes()
    base = _state["jit_base"]
    if base is None:
        base = _state["jit_base"] = {k: cur[k] for k in SEQ_A2A_KERNEL_NAMES}
        _state["jit_base_src"] = "first-use"
    for k in _PART_KERNELS[part]:
        base[k] = cur[k]


def jit_delta() -> Tuple[int, Dict[str, Optional[int]]]:
    """(new variants of the five B1'' kernels, per-kernel delta of those and of the
    seq-a2a K1-K4) since the warmup / first use. K1-K4 are left out of the total
    (the jit line reports them as "seq_a2a +<m>"); the per-kernel dict shows
    them."""
    cur = jit_cache_sizes()
    base = _state["jit_base"] or {}
    per: Dict[str, Optional[int]] = {}
    total = 0
    for k, v in cur.items():
        b = base.get(k)
        if v < 0 or b is None or b < 0:
            per[k] = None
            continue
        d = v - b
        per[k] = d
        if k in KERNEL_NAMES:
            total += max(0, d)
    return total, per


def _note_engaged(kind: str) -> None:
    n = _state["engaged"][kind] = _state["engaged"][kind] + 1
    if _state["jit_logged"] or n != JIT_REPORT_AFTER:
        return
    if kind == "recat" and (rebag_on() or bounds_on()):
        return  # the TBE count decides when C or D is on
    _state["jit_logged"] = True
    total, per = jit_delta()
    seq = sum(max(0, d) for k, d in per.items() if k in SEQ_A2A_KERNEL_NAMES and d is not None)
    logger.info(
        "DLRMV4 b1pp jit after run_start: b1pp +%d seq_a2a +%d (%s) base=%s rank=%d",
        total,
        seq,
        ", ".join("%s:%s" % (k, "?" if d is None else "%+d" % d) for k, d in per.items()),
        _state["jit_base_src"] or "none",
        _rank(),
    )


# ---------------------------------------------------------------------------
# (B) hook body: host_sync_free.p_kjtt_wait equal-stride branch
# ---------------------------------------------------------------------------
def _bits_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype
        and tuple(a.shape) == tuple(b.shape)
        and bool(torch.equal(a.contiguous(), b.contiguous()))
    )


def _verify_recat(
    recat: torch.Tensor,
    l2: torch.Tensor,
    values: torch.Tensor,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    fl, fv, _ = torch.ops.fbgemm.permute_2D_sparse_data(recat, l2, values, None, values.numel())
    fo = torch.ops.fbgemm.asynchronous_complete_cumsum(fl.view(-1))
    R, B = l2.shape
    bad = [
        n
        for n, a, b in (("lengths", out[0], fl), ("values", out[1], fv), ("offsets", out[2], fo))
        if not _bits_eq(a, b)
    ]
    if bad:
        raise VerifyMismatch(
            "DLRMV4 b1pp VERIFY recat MISMATCH in %s (R=%d B=%d N=%d) rank=%d"
            % (",".join(bad), R, B, values.numel(), _rank())
        )
    logger.info(
        "DLRMV4 b1pp VERIFY recat OK (R=%d B=%d N=%d) rank=%d", R, B, values.numel(), _rank()
    )


def kjt_recat(
    recat: Optional[torch.Tensor],
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    stride: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(permuted lengths [R, stride], values, KJT offsets) for the equal-stride
    branch of host_sync_free.p_kjtt_wait, or None: the caller runs stock."""
    if not _active("recat"):
        return None
    if stride <= 0 or lengths.dim() != 1 or lengths.numel() % stride:
        _fallback("recat", "lengths not [R * stride]")
        return None
    l2 = lengths.view(-1, stride)
    why = recat_reason(recat, l2, values, weights)
    if why is not None:
        _fallback("recat", why, "R=%d B=%d N=%d", l2.shape[0], stride, values.numel())
        return None
    try:
        out = _recat_launch(recat, l2, values)
    except Exception as e:  # noqa: BLE001 (Triton compile/launch failure)
        _failed("recat", "fused recat kernels", e)
        return None
    if _verify_take("recat"):
        _verify_recat(recat, l2, values, out)
    R = l2.shape[0]
    _arm(
        "recat",
        "DLRMV4 armed OK: b1pp kjt recat fused (R=%d B=%d N=%d offsets=1)"
        % (R, stride, values.numel()),
    )
    _jit_first_use("recat")
    _note_engaged("recat")
    return out


# ---------------------------------------------------------------------------
# (C)+(D) hook body: the patched BatchedFusedEmbedding.forward
# ---------------------------------------------------------------------------
def _tbe_static_reason(mod: Any) -> Tuple[Optional[str], int]:
    """Per-module static eligibility (cached): (reason or None, T)."""
    emb = getattr(mod, "_emb_module", None)
    if emb is None:
        return "no _emb_module", 0
    try:
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            SplitTableBatchedEmbeddingBagsCodegen,
        )
        from fbgemm_gpu.tbe.config import EmbeddingLocation, PoolingMode
    except Exception as e:  # noqa: BLE001
        return "fbgemm import failed (%s)" % type(e).__name__, 0
    if type(emb) is not SplitTableBatchedEmbeddingBagsCodegen:
        return "emb_module is %s" % type(emb).__name__, 0
    if emb.pooling_mode != PoolingMode.NONE:
        return "pooled TBE", 0
    try:
        if any(spec[2] != EmbeddingLocation.DEVICE for spec in emb.embedding_specs):
            return "non-DEVICE table placement", 0
    except Exception:  # noqa: BLE001
        return "embedding_specs unreadable", 0
    if (
        getattr(mod, "_embedding_table_index_type", None) != torch.int64
        or getattr(mod, "_embedding_table_offset_type", None) != torch.int64
        or getattr(emb, "embedding_table_index_type", None) != torch.int64
        or getattr(emb, "embedding_table_offset_type", None) != torch.int64
    ):
        return "index/offset type not int64", 0
    T = len(emb.feature_table_map)
    rpt = getattr(emb, "rows_per_table", None)
    if T <= 0 or rpt is None or rpt.numel() != T or rpt.dtype != torch.int64:
        return "rows_per_table is not int64 [T]", 0
    return None, T


def _tbe_static(mod: Any) -> Tuple[Optional[str], int]:
    try:
        hit = _static_cache.get(mod)
    except TypeError:
        hit = None
    if hit is None:
        hit = _tbe_static_reason(mod)
        try:
            _static_cache[mod] = hit
        except TypeError:
            pass
    return hit


def _orig(mod: Any, features: Any) -> torch.Tensor:
    fn = _orig_forward
    if fn is None:  # not installed (unit tests call _b1pp_tbe_forward directly)
        from torchrec.distributed.batched_embedding_kernel import BaseBatchedEmbedding

        fn = BaseBatchedEmbedding.forward
    return fn(mod, features)


def _fbgemm_bounds_check(emb: Any, ic: torch.Tensor, oc: torch.Tensor, wc: torch.Tensor) -> None:
    """The stock check the TBE would have run (WARNING, v2, non-VBE), in place on clones."""
    torch.ops.fbgemm.bounds_check_indices(
        emb.rows_per_table,
        ic,
        oc,
        _WARNING,
        wc,
        None,
        B_offsets=None,
        max_B=-1,
        b_t_map=None,
        info_B_num_bits=emb.info_B_num_bits,
        info_B_mask=emb.info_B_mask,
        bounds_check_version=2,
        prefetch_pipeline=False,
    )


def _offsets_well_formed(off: torch.Tensor, N: int) -> bool:
    """offsets[0] >= 0, non-decreasing, offsets[-1] == N: no bad bag and a right
    last offset (every cumsum; the Triton fast path). Reads the device (VERIFY only)."""
    if off.numel() == 0:
        return True
    return bool(int(off[0]) >= 0 and int(off[-1]) == N and bool((off[1:] >= off[:-1]).all()))


def _verify_bounds(
    emb: Any,
    pre: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    offsets: torch.Tensor,
    T: int,
    B: int,
) -> None:
    """Well-formed offsets (production): indices, offsets and the counter must be
    bit-identical to stock. Malformed offsets: fbgemm v2 itself is racy (its
    outcome is one of several legal interleavings, and not always the one the
    Triton slow path lands on), so only the outcome class is compared -- both
    repair every offset into [0, N] and count > 0 -- and a WARNING names the
    malformed input (fbgemm would have printed one from the device)."""
    ic, oc, wc = pre
    N = ic.numel()
    well_formed = _offsets_well_formed(oc, N)  # the ORIGINAL offsets (clone taken before the Triton call)
    first, last = (int(oc[0]), int(oc[-1])) if oc.numel() else (0, 0)
    _fbgemm_bounds_check(emb, ic, oc, wc)
    w_t, w_f = int(emb.bounds_check_warning[0]), int(wc[0])
    if well_formed:
        bad = [
            n
            for n, a, b in (
                ("indices", indices, ic),
                ("offsets", offsets, oc),
                ("warning", emb.bounds_check_warning, wc),
            )
            if not _bits_eq(a, b)
        ]
        if bad:
            raise VerifyMismatch(
                "DLRMV4 b1pp VERIFY bounds MISMATCH in %s (T=%d B=%d N=%d warning triton=%d fbgemm=%d) rank=%d"
                % (",".join(bad), T, B, N, w_t, w_f, _rank())
            )
        logger.info(
            "DLRMV4 b1pp VERIFY bounds OK (T=%d B=%d N=%d warning=%d) rank=%d", T, B, N, w_f, _rank()
        )
        return
    in_range = [
        n
        for n, o in (("triton", offsets), ("fbgemm", oc))
        if o.numel() and not bool(((o >= 0) & (o <= N)).all())
    ]
    no_count = [n for n, w in (("triton", w_t), ("fbgemm", w_f)) if w <= 0]
    if in_range or no_count:
        raise VerifyMismatch(
            "DLRMV4 b1pp VERIFY bounds MISMATCH (outcome class) on malformed offsets: offsets outside [0, N] "
            "after the repair: %s; counter not > 0: %s (T=%d B=%d N=%d) rank=%d"
            % (in_range or "none", no_count or "none", T, B, N, _rank())
        )
    logger.warning(
        "DLRMV4 b1pp VERIFY bounds: MALFORMED offsets in the input (first=%d last=%d N=%d T=%d B=%d): "
        "fbgemm v2 is racy on such input, so only the outcome class was compared (every offset repaired "
        "into [0, N], counter > 0): same; warning triton=%d fbgemm=%d; repaired indices equal=%s offsets "
        "equal=%s rank=%d",
        first,
        last,
        N,
        T,
        B,
        w_t,
        w_f,
        _bits_eq(indices, ic),
        _bits_eq(offsets, oc),
        _rank(),
    )


def _feature_of_pos(off: torch.Tensor, bags_per_feature: int, L: int) -> torch.Tensor:
    pos = torch.arange(L, device=off.device, dtype=off.dtype)
    bag = torch.searchsorted(off, pos, right=True) - 1
    return torch.div(bag, bags_per_feature, rounding_mode="floor")


def _verify_rebag(
    offsets: torch.Tensor, new: torch.Tensor, T: int, B: int, Bp: int, L: int
) -> None:
    """Structural check (fbgemm's fused optimizer updates in backward, so an
    in-model forward/backward comparison is impossible; the lab and unit tests
    cover the numerics)."""
    probs: List[str] = []
    if new.numel() != T * Bp + 1 or new.dtype != offsets.dtype:
        probs.append("shape/dtype")
    else:
        if not bool((new[1:] >= new[:-1]).all()):
            probs.append("not monotone")
        if not torch.equal(new[::Bp][: T + 1], offsets[::B][: T + 1]):
            probs.append("feature starts differ")
        if int(new[-1]) != L or int(offsets[-1]) != L:
            probs.append("last offset != L")
        if L > 0 and not torch.equal(_feature_of_pos(offsets, B, L), _feature_of_pos(new, Bp, L)):
            probs.append("feature of an index position changed")
    if rocm_bwd_pred(L, T * Bp) != rocm_bwd_pred(L, T * B):
        probs.append("ROCm backward predicate flipped")
    if probs:
        raise VerifyMismatch(
            "DLRMV4 b1pp VERIFY rebag MISMATCH: %s (T=%d B=%d Bp=%d L=%d) rank=%d"
            % ("; ".join(probs), T, B, Bp, L, _rank())
        )
    logger.info(
        "DLRMV4 b1pp VERIFY rebag OK (T=%d B=%d Bp=%d L=%d) rank=%d", T, B, Bp, L, _rank()
    )


def _b1pp_tbe_forward(self: Any, features: Any) -> torch.Tensor:
    """BatchedFusedEmbedding.forward with (D) the Triton bounds check and (C) the
    Rkeep rebag for the pooling-NONE TBE; the original forward otherwise."""
    c_on = _active("rebag")
    d_on = _active("bounds")
    if not (c_on or d_on):
        return _orig(self, features)
    parts = [p for p, on in (("rebag", c_on), ("bounds", d_on)) if on]
    why, T = _tbe_static(self)
    if why is None:
        # == _get_hash_zch_identities_and_metadata(features) is None: it returns
        # None at once without a raw-id tracker (torchrec embedding_kernel.py
        # BaseEmbedding); read without calling it so a fallback calls it once.
        if getattr(self, "_raw_id_tracker_wrapper", None) is not None:
            why = "hash-zch raw-id tracker"
        elif features.weights_or_none() is not None:
            why = "per-sample weights"
        elif features.variable_stride_per_key():
            why = "variable stride per key"
    if why is not None:
        for p in parts:
            _fallback(p, why)
        return _orig(self, features)
    indices = features.values()
    if (
        indices.dtype != torch.int64
        or not indices.is_contiguous()
        or not _dev_ok(indices)
        or indices.numel() > _SIZE_MAX
    ):
        for p in parts:
            _fallback(p, "values not contiguous int64 on CUDA with numel <= 2^31 - 2^17")
        return _orig(self, features)
    B = int(features.stride())
    offsets = features.offsets()  # computed by torchrec if absent (as the stock forward does)
    if (
        offsets.dtype != torch.int64
        or offsets.dim() != 1
        or offsets.numel() != T * B + 1
        or offsets.numel() > _SIZE_MAX
        or not offsets.is_contiguous()
        or offsets.device != indices.device
    ):
        for p in parts:
            _fallback(p, "offsets not contiguous int64 [T*B+1] (<= 2^31 - 2^17)", "T=%d B=%d", T, B)
        return _orig(self, features)
    emb = self._emb_module
    L = indices.numel()
    engaged = False
    fbgemm_bc_off = False
    if d_on:
        if emb.bounds_check_mode_int != _WARNING:
            _fallback("bounds", "bounds_check_mode is not WARNING", "mode=%d", emb.bounds_check_mode_int)
        elif getattr(emb, "bounds_check_version", None) != 2:
            _fallback("bounds", "bounds_check_version is not 2")
        elif T > BC_MAX_T:
            _fallback("bounds", _T_REASON, "T=%d", T)
        else:
            # clones only while VERIFY budget is left; the budget itself is spent
            # only for a call that engaged (like the recat and rebag VERIFY)
            pre = None
            if _verify_peek("bounds"):
                pre = (indices.clone(), offsets.clone(), emb.bounds_check_warning.clone())
            try:
                bounds_check_warning(emb.rows_per_table, indices, offsets, emb.bounds_check_warning)
                fbgemm_bc_off = True
            except Exception as e:  # noqa: BLE001
                _failed("bounds", "Triton bounds check", e)
            if fbgemm_bc_off:
                if pre is not None and _verify_take("bounds"):
                    _verify_bounds(emb, pre, indices, offsets, T, B)
                _arm(
                    "bounds",
                    "DLRMV4 armed OK: b1pp tbe bounds triton (T=%d B=%d N=%d mode=WARNING)"
                    % (T, B, L),
                )
                _jit_first_use("bounds")
                engaged = True
    new: Optional[torch.Tensor] = None
    if c_on:
        Bp, why = rebag_plan(L, T, B)
        if Bp is None:
            _fallback("rebag", str(why), "T=%d B=%d L=%d", T, B, L)
        else:
            try:
                new = _rebag_launch(offsets, T, B, Bp)
            except Exception as e:  # noqa: BLE001
                _failed("rebag", "rebag kernel", e)
                new = None
            if new is not None:
                if _verify_take("rebag"):
                    _verify_rebag(offsets, new, T, B, Bp, L)
                _arm(
                    "rebag",
                    "DLRMV4 armed OK: b1pp tbe rebag Rkeep (T=%d B=%d Bp=%d L=%d)" % (T, B, Bp, L),
                )
                _jit_first_use("rebag")
                engaged = True
    emb_offsets = new if new is not None else offsets
    if fbgemm_bc_off:
        prev = emb.bounds_check_mode_int
        emb.bounds_check_mode_int = _NONE
        try:
            out = self._emb_module(indices=indices, offsets=emb_offsets)
        finally:
            emb.bounds_check_mode_int = prev
    else:
        out = self._emb_module(indices=indices, offsets=emb_offsets)
    if engaged:
        _note_engaged("tbe")
    return out


def _forward_mismatch(fn: Any) -> Optional[str]:
    """None if `fn` is the torchrec BaseBatchedEmbedding.forward body that
    _b1pp_tbe_forward re-implements (argument count and referenced names as in
    the production overlay); else why not. A new forward argument, keyword or
    helper (e.g. more hash-ZCH plumbing) changes the referenced names."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return "no __code__ (%s)" % type(fn).__name__
    if code.co_argcount != _BASE_FORWARD_ARGCOUNT:
        return "argcount %d (want %d)" % (code.co_argcount, _BASE_FORWARD_ARGCOUNT)
    names = frozenset(code.co_names)
    if names != _BASE_FORWARD_NAMES:
        return "referenced names differ: +%s -%s" % (
            sorted(names - _BASE_FORWARD_NAMES),
            sorted(_BASE_FORWARD_NAMES - names),
        )
    return None


def install() -> None:
    """Patch torchrec BatchedFusedEmbedding.forward (class attribute; idempotent;
    original kept). A no-op unless DLRM_HSTU_FP8_TBE_REBAG or
    DLRM_HSTU_FP8_TBE_BOUNDS_TRITON is 1. (B) needs no patch: host_sync_free's
    p_kjtt_wait asks kjt_recat() directly. Refuses (one WARNING; C and D then
    stay off, stock) when BatchedFusedEmbedding.forward is not the
    BaseBatchedEmbedding.forward body that _b1pp_tbe_forward re-implements."""
    global _installed, _install_refused, _patched_cls, _had_own_forward, _orig_forward, _base_forward
    if _installed or _install_refused is not None:
        return
    if recat_on() and "recat" not in _state["armed"] and not _state.get("recat_requested_logged"):
        _state["recat_requested_logged"] = True
        logger.info(
            "DLRMV4 b1pp kjt recat fused requested; arms at the first eligible "
            "dist_init (host-sync-free path) rank=%d",
            _rank(),
        )
    if not (rebag_on() or bounds_on()):
        return
    if not _HAVE_TRITON:
        logger.warning("DLRMV4 b1pp: triton unavailable; tbe rebag/bounds off (stock) rank=%d", _rank())
        return
    try:
        from torchrec.distributed.batched_embedding_kernel import (
            BaseBatchedEmbedding,
            BatchedFusedEmbedding,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "DLRMV4 b1pp: torchrec import failed (%s: %s); tbe rebag/bounds off rank=%d",
            type(e).__name__,
            e,
            _rank(),
        )
        return
    if BatchedFusedEmbedding.__dict__.get("forward") is _b1pp_tbe_forward:
        _installed = True
        return
    resolved = BatchedFusedEmbedding.forward
    if resolved is not BaseBatchedEmbedding.forward:
        why: Optional[str] = "BatchedFusedEmbedding resolves its own forward (%s)" % getattr(
            resolved, "__qualname__", type(resolved).__name__
        )
    else:
        why = _forward_mismatch(resolved)
    if why is not None:
        _install_refused = why
        _state["disabled"].update(("rebag", "bounds"))  # nothing to warm or arm
        logger.warning(
            "DLRMV4 b1pp: torchrec BatchedFusedEmbedding.forward is not the BaseBatchedEmbedding.forward "
            "body b1pp re-implements (%s); tbe rebag/bounds NOT installed (stock) rank=%d",
            why,
            _rank(),
        )
        return
    _base_forward = BaseBatchedEmbedding.forward
    _had_own_forward = "forward" in BatchedFusedEmbedding.__dict__
    _orig_forward = BatchedFusedEmbedding.forward
    BatchedFusedEmbedding.forward = _b1pp_tbe_forward
    _patched_cls = BatchedFusedEmbedding
    _installed = True
    logger.info(
        "DLRMV4 b1pp requested (rebag=%d bounds=%d); BatchedFusedEmbedding.forward patched "
        "(original %s; arms at the first eligible TBE call) rank=%d",
        int(rebag_on()),
        int(bounds_on()),
        "BaseBatchedEmbedding.forward" if _orig_forward is _base_forward else "own forward",
        _rank(),
    )


def _uninstall_for_tests() -> None:
    """Restore the stock BatchedFusedEmbedding.forward (unit tests only)."""
    global _installed, _install_refused, _patched_cls, _orig_forward
    if _patched_cls is not None:
        if _had_own_forward:
            _patched_cls.forward = _orig_forward
        elif "forward" in _patched_cls.__dict__:
            delattr(_patched_cls, "forward")
    _patched_cls = None
    _orig_forward = None
    _installed = False
    _install_refused = None


def _reset_for_tests() -> None:
    """Forget the read-once switches, budgets, latches and caches (unit tests only)."""
    _state.clear()
    _state.update(_new_state())
    _static_cache.clear()


# ---------------------------------------------------------------------------
# Warmup (pre-RUN_START; synthetic tensors only)
# ---------------------------------------------------------------------------
def _warm_recat(dev: torch.device) -> int:
    # 2 keys x 8 workers (torchrec _get_recat order, int32), B = 64; every block constexpr is the run's
    # (fixed for all shapes), so this one variant per kernel is the only one any call needs.
    K, W, B = 2, 8, 64
    recat = torch.tensor([i + j * K for i in range(K) for j in range(W)], dtype=torch.int32, device=dev)
    l2_host = (torch.arange(K * W * B, dtype=torch.int64) % 3).view(K * W, B)
    l2 = l2_host.to(dev)
    values = torch.arange(int(l2_host.sum()), dtype=torch.int64, device=dev)
    out = recat_fused(recat, l2, values)
    del recat, l2, values, out
    return 2


def _warm_offsets(T: int, B: int, dev: torch.device) -> Tuple[torch.Tensor, int]:
    lengths = torch.arange(T * B, dtype=torch.int64, device=dev) % 2
    offsets = torch.zeros(T * B + 1, dtype=torch.int64, device=dev)
    torch.cumsum(lengths, 0, out=offsets[1:])
    return offsets, (T * B) // 2


def _warm_bounds(dev: torch.device) -> int:
    T, B = 3, 64
    offsets, L = _warm_offsets(T, B, dev)
    rows = torch.full((T,), 1000, dtype=torch.int64, device=dev)
    indices = torch.arange(L, dtype=torch.int64, device=dev) % 1000
    warning = torch.zeros(1, dtype=torch.int64, device=dev)
    bounds_check_warning(rows, indices, offsets, warning)
    del offsets, rows, indices, warning
    return 2


def _warm_rebag(dev: torch.device) -> int:
    T, B = 3, 64
    offsets, L = _warm_offsets(T, B, dev)
    out = rebag_rkeep(offsets, T, B, L)
    if out is None:  # cannot happen for this shape (L = T*B/2 keeps the predicate false)
        raise RuntimeError("b1pp warmup: synthetic rebag not eligible")
    del offsets, out
    return 1


class WarmupDeviceFault(RuntimeError):
    """A synchronize after the warmup's launches failed: a device fault (the HIP
    context is unusable, so step 1 would fail too, later and inside the timed
    region). run_first_use_warmup re-raises it, as blas_warmup.warmup does with
    any failed synchronize, so the run stops before RUN_START."""


_SARP_MOD = "generative_recommenders.dlrm_v4.seq_a2a_rowperm"
_UBG_MOD = "generative_recommenders.dlrm_v4.unbucketize_gather"
# torchrec SequenceEmbeddingsAllToAll attributes the K1-K4 warmup reads
_A2A_ATTRS = ("_local_split", "_num_splits", "_forward_recat_tensor", "_backward_recat_tensor", "_codecs")


def _seq_a2a_sites(model: Any) -> List[Tuple[str, int, Any]]:
    """(sharding type, D, SequenceEmbeddingsAllToAll) of every sequence all-to-all
    of the sharded EmbeddingCollections in ``model``. torchrec
    ShardedEmbeddingCollection keeps ``_output_dists`` in the order of
    ``_sharding_type_to_sharding`` (``_create_output_dist`` appends in that order,
    ``compute_and_output_dist`` zips the two); the a2a is ``odist._dist``, and D is
    ``_embedding_dim_for_sharding_type`` (the ``local_embs.shape[1]`` the output
    dist is called with). Read only; anything that does not match is skipped."""
    sites: List[Tuple[str, int, Any]] = []
    if model is None or not hasattr(model, "modules"):
        return sites
    seen = set()
    for m in model.modules():
        st2s = getattr(m, "_sharding_type_to_sharding", None)
        odists = getattr(m, "_output_dists", None)
        dim_of = getattr(m, "_embedding_dim_for_sharding_type", None)
        if not isinstance(st2s, dict) or odists is None or not callable(dim_of):
            continue
        for st, od in zip(list(st2s.keys()), list(odists)):
            a2a = getattr(od, "_dist", None)
            if a2a is None or id(a2a) in seen or not all(hasattr(a2a, k) for k in _A2A_ATTRS):
                continue
            seen.add(id(a2a))
            sites.append((str(st), int(dim_of(st)), a2a))
    return sites


def _seq_a2a_wanted() -> bool:
    """The fused seq-a2a send or receive path is installed and switched on
    (seq_a2a_rowperm is imported and installed by MetricsLogger.__init__ before
    the warmup; its switches are read once there, so nothing is imported or
    read here otherwise)."""
    sarp = sys.modules.get(_SARP_MOD)
    if sarp is None or not getattr(sarp, "_installed", False):
        return False
    fwd, bwd = sarp.fused_config()
    return bool(fwd or bwd or sarp.recv_config())


def _warm_seq_a2a(dev: torch.device, model: Any, batch_sizes: Any) -> Tuple[List[str], Optional[str]]:
    """Compile the fused seq-a2a K1-K4 variants of this rank's sequence
    all-to-alls (seq_a2a_fused.py) before RUN_START.

    seq_a2a_fused's own launchers run on synthetic zero tensors, so every
    constexpr is derived as at run time: D, BLOCK_N from n_blocks = local
    features x world size, BLOCK_B from the per-rank batch, BLOCK_R / BLOCK_D
    from D, HAS_PERM, APPLY_SCALE (torchrec GRADIENT_DIVISION). The dtypes are
    the run's (fp32 embeddings, int64 lengths viewed [local_T, W * B], the
    module's own int32 recat buffers, int64 permutation, fp16 wire) and every
    pointer is a fresh allocation or a registered buffer, i.e. 16-byte
    aligned, with a storage below 2 GiB (Triton's AMD buffer-ops pointer
    class), as the run's are (TBE output, a2a buffers, fresh KJT lengths,
    fbgemm's unbucketize permutation; every per-call int is do_not_specialize).
    Eligibility is seq_a2a_rowperm's: its switches, ``_static_reason`` per
    codecs / D / world size, MAX_BLOCKS, recat layout, and the row-wise
    unbucketize permutation only where its hand-off is installed. A failed
    launch disables that path as its first call would. Returns (one label per
    launched kernel variant, None) or ([], why not)."""
    sarp = sys.modules.get(_SARP_MOD)
    if sarp is None:
        return [], "seq_a2a_rowperm not imported"
    from generative_recommenders.dlrm_v4 import seq_a2a_fused as saf

    fwd, bwd = sarp.fused_config()
    fwd = bool(fwd) and not sarp._state["fused_disabled"]
    bwd = bool(bwd) and not sarp._state["fused_disabled"]
    recv = bool(sarp.recv_config() and sarp._recv_installed) and not sarp._state["recv_disabled"]
    if not (fwd or bwd or recv):
        return [], "fused seq-a2a path not installed or switched off"
    sizes = sorted({int(b) for b in (batch_sizes or ()) if b is not None and int(b) > 0})
    if (fwd or bwd) and not sizes:
        return [], "no per-rank batch size given"
    sites = _seq_a2a_sites(model)
    if not sites:
        return [], "no sharded sequence all-to-all in the model"
    try:
        from torchrec.distributed import comm_ops

        grad_div = bool(comm_ops.GRADIENT_DIVISION)
    except Exception:  # noqa: BLE001
        grad_div = True
    ubg = sys.modules.get(_UBG_MOD)
    hand_off = bool(
        ubg is not None and ubg._installed and ubg._RECV_HOOK is not None and not ubg._state["disabled"]
    )
    labels: List[str] = []
    seen = set()
    for st, D, a2a in sites:
        local_T, W = int(a2a._local_split), int(a2a._num_splits)
        if W <= 1:
            continue  # one rank: torchrec returns a NoWait, no fused kernel runs
        n_blocks = local_T * W
        codecs = a2a._codecs
        recat, bwd_recat = a2a._forward_recat_tensor, a2a._backward_recat_tensor
        recats_ok = all(
            r.dtype in (torch.int32, torch.int64) and r.numel() == n_blocks and r.is_contiguous()
            and r.device == dev
            for r in (recat, bwd_recat)
        )
        # send (K1/K2) needs local rows (rowperm _eligible: local_T > 0); the
        # receive side (K3'/K4) runs on every rank of the a2a
        if (
            (fwd or bwd)
            and local_T > 0
            and recats_ok
            and 1 <= n_blocks <= saf.MAX_BLOCKS
            and sarp._static_reason("send", codecs, D, W, dev) is None
        ):
            for b in sizes:
                k1 = ("K1", D, n_blocks, b, recat.dtype)
                k2 = ("K2", D, n_blocks, recat.dtype, bwd_recat.dtype, grad_div)
                if not ((fwd and k1 not in seen) or (bwd and k2 not in seen)):
                    continue
                try:
                    tok = torch.zeros((n_blocks, b), dtype=torch.int64, device=dev)
                    tok[:, 0] = 1  # one row per block
                    tok = tok.view(local_T, W * b)  # as lengths_after_input_dist
                    blk = sarp.block_rows(tok, n_blocks).view(-1)
                    wire = None
                    if fwd and k1 not in seen:
                        emb = torch.zeros((n_blocks, D), dtype=torch.float32, device=dev)
                        _, wire = saf.send_fwd(emb, blk, recat, tok, n_blocks, D)
                        seen.add(k1)
                        labels.append("K1 D=%d blocks=%d B=%d" % (D, n_blocks, b))
                    if bwd and k2 not in seen:
                        if wire is None:
                            wire = torch.zeros(n_blocks * D, dtype=torch.float16, device=dev)
                        saf.send_bwd(wire, blk, recat, bwd_recat, D, W if grad_div else None)
                        seen.add(k2)
                        labels.append("K2 D=%d blocks=%d div=%d" % (D, n_blocks, int(grad_div)))
                except Exception as e:  # noqa: BLE001 -- the first fused call would fail too
                    sarp._fused_failed("[b1pp-warmup] send (K1/K2)", e)
                    fwd = bwd = False
        has_perm = st == "row_wise" and hand_off
        k3 = ("K3", D, has_perm)
        if recv and k3 not in seen and sarp._static_reason("recv", codecs, D, 0, dev) is None:
            try:
                U = 2
                wire16 = torch.zeros(U * D, dtype=torch.float16, device=dev)
                perm = torch.arange(U, dtype=torch.int64, device=dev) if has_perm else None
                out = saf.recv_fwd(wire16, perm, D)
                labels.append("K3' D=%d perm=%d" % (D, int(has_perm)))
                saf.recv_bwd(out, perm, D)
                labels.append("K4 D=%d perm=%d" % (D, int(has_perm)))
                seen.add(k3)
            except Exception as e:  # noqa: BLE001 -- the first fused call would fail too
                sarp._recv_failed("[b1pp-warmup] recv (K3'/K4)", e)
                recv = False
    if not labels:
        return [], "no eligible sequence all-to-all (%d found)" % len(sites)
    return labels, None


def run_first_use_warmup(device: Any = None, model: Any = None, batch_sizes: Any = ()) -> None:
    """Compile, before RUN_START, the B1'' kernels of the parts that are on and the
    fused seq-a2a K1-K4 of this rank's sequence all-to-alls.

    B1'' parts: synthetic tensors of the production dtypes (torch.arange, no
    RNG), the same constexprs as the run (fixed blocks above), each kernel
    launched once. K1-K4 (only when DLRM_HSTU_FP8_SEQ_A2A_FUSED is on and the
    rowperm patch is installed): ``_warm_seq_a2a`` with ``model`` (the sharded
    model: D, local features and world size of every sequence all-to-all) and
    ``batch_sizes`` (the per-rank batches the loops run). One synchronize after
    each part (K1-K4 included), everything freed. No data, no RNG, no collective;
    no model, optimizer, dataset or b1pp latch (arming, VERIFY budget,
    engaged-call counters) is written. The JIT counts after it are the
    baseline of the "jit after run_start" line. A no-op when nothing is on,
    without CUDA, and once it has run (train_ranker calls it before the startup
    preload's gc.freeze in the streaming mode and once more after that block
    for the other modes).

    Never raises, except ``WarmupDeviceFault``: an ordinary failure (a compile
    or launch error) is logged and that part runs stock, as it would after a
    failure at its first call; ANY failed synchronize (each runs outside the
    part's try) is a device fault (the HIP context is unusable and step 1 would
    fail inside the timed region), so it is raised, before RUN_START and after
    leaving the device context, as blas_warmup.warmup does."""
    try:
        _run_warmup(device, model, batch_sizes)
    except WarmupDeviceFault:
        raise
    except Exception as e:  # noqa: BLE001 -- never abort a run over an ordinary warmup failure
        logger.warning(
            "DLRMV4 b1pp [b1pp-warmup] skipped: unexpected error (%s: %s) rank=%d",
            type(e).__name__,
            e,
            _rank(),
        )


def _run_warmup(device: Any, model: Any = None, batch_sizes: Any = ()) -> None:
    if _state["warmup_ran"]:
        return
    parts = [p for p in _PARTS if _active(p)]
    seq = _seq_a2a_wanted()
    if not parts and not seq:
        return
    rank = _rank()
    if not _HAVE_TRITON:
        logger.info("DLRMV4 b1pp [b1pp-warmup] skipped: triton unavailable rank=%d", rank)
        return
    if not torch.cuda.is_available():
        logger.info("DLRMV4 b1pp [b1pp-warmup] skipped: no CUDA rank=%d", rank)
        return
    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    if dev.type != "cuda":
        logger.info("DLRMV4 b1pp [b1pp-warmup] skipped: device %s is not CUDA rank=%d", dev, rank)
        return
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.perf_counter()
    before = jit_cache_sizes()
    n = 0
    first_err: Optional[str] = None
    seq_labels: List[str] = []
    seq_why: Optional[str] = None
    t1 = t0
    # (part, error) of the first failed synchronize: a device fault, raised below.
    fault: Optional[Tuple[str, BaseException]] = None
    try:
        with torch.cuda.device(dev):
            for part, fn in (("recat", _warm_recat), ("bounds", _warm_bounds), ("rebag", _warm_rebag)):
                if part not in parts:
                    continue
                try:
                    n += fn(dev)
                except Exception as e:  # noqa: BLE001 -- the part would fail at first use too
                    _failed(part, "[b1pp-warmup] kernels", e)
                    first_err = first_err or "%s: %s: %s" % (part, type(e).__name__, e)
                # Outside the part's try: attributes an asynchronous fault to this
                # part and never downgrades it to that part's fallback.
                err = _sync_error(dev)
                if err is not None:
                    fault = (part, err)
                    break
            t1 = time.perf_counter()
            if seq and fault is None:
                try:
                    seq_labels, seq_why = _warm_seq_a2a(dev, model, batch_sizes)
                except Exception as e:  # noqa: BLE001 -- the K1-K4 then compile at first use
                    seq_labels, seq_why = [], "%s: %s" % (type(e).__name__, e)
                    first_err = first_err or "seq_a2a K1-K4: " + seq_why
                err = _sync_error(dev)
                if err is not None:
                    fault = ("seq_a2a K1-K4", err)
    except Exception:  # noqa: BLE001 -- re-raised unless a device fault is already recorded
        if fault is None:
            raise
        # e.g. the device context's exit failed after the fault: the fault is the news
    if fault is not None:
        # Not swallowed, as in blas_warmup.warmup(): ANY failed synchronize is a
        # device fault. The HIP context is unusable and step 1 would fail anyway
        # -- later, inside the timed region, with a misleading stack. Fail now,
        # before RUN_START, outside the device context (nothing can replace it).
        where, err = fault
        raise WarmupDeviceFault(
            "DLRMV4 b1pp [b1pp-warmup] device fault during the warmup (synchronize after %s: %s: %s; "
            "first failure: %s); the HIP context is unusable. Rerun with DLRM_HSTU_FP8_KJT_RECAT_FUSED=0 "
            "DLRM_HSTU_FP8_TBE_REBAG=0 DLRM_HSTU_FP8_TBE_BOUNDS_TRITON=0 "
            "DLRM_HSTU_FP8_SEQ_A2A_FUSED=0 to isolate. rank=%d"
            % (where, type(err).__name__, err, first_err or "none", rank)
        ) from err
    after = jit_cache_sizes()

    def _new(names: Tuple[str, ...]) -> int:
        return sum(
            max(0, after[k] - before[k])
            for k in names
            if after.get(k, -1) >= 0 and before.get(k, -1) >= 0
        )

    _state["jit_base"] = after
    _state["jit_base_src"] = "warmup"
    _state["warmup_ran"] = True
    if parts:
        logger.info(
            "DLRMV4 armed OK: [b1pp-warmup] kernels=%d triton_jit_new=%d in %.2fs rank=%d (parts=%s)",
            n,
            _new(KERNEL_NAMES),
            t1 - t0,
            rank,
            ",".join(p for p in parts if p not in _state["disabled"]),
        )
    if seq:
        if seq_why is None:
            logger.info(
                "DLRMV4 armed OK: [b1pp-warmup] seq_a2a K1-K4 variants=%d triton_jit_new=%d in %.2fs "
                "rank=%d (%s)",
                len(seq_labels),
                _new(SEQ_A2A_KERNEL_NAMES),
                time.perf_counter() - t1,
                rank,
                "; ".join(seq_labels),
            )
        else:
            logger.info(
                "DLRMV4 b1pp [b1pp-warmup] seq_a2a K1-K4 not warmed (%s); they compile at first use "
                "rank=%d",
                seq_why,
                rank,
            )


def _sync_error(dev: torch.device) -> Optional[BaseException]:
    """torch.cuda.synchronize(dev): None, or the error it raised (a device fault;
    _run_warmup raises it as WarmupDeviceFault once outside the device context)."""
    try:
        torch.cuda.synchronize(dev)
    except Exception as e:  # noqa: BLE001 -- returned, then raised as WarmupDeviceFault
        return e
    return None
