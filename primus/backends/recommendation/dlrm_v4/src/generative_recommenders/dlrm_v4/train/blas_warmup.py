# pyre-strict
"""BLAS first-use warmup before RUN_START ($BLAS_WARMUP; gin yambda_5b.gin item (18)).

WHY. The first train step pays a one-time, per-process cost the first time
each GEMM *family* is used: hipBLASLt handle creation, the TunableOp operator
object for each (op, dtype, layout) template (its constructor enumerates every
hipBLASLt solution for that type), the results-file load, and the Tensile code
objects of the kernels it picks. A stack sample of the first forward attributes
seconds of it to BLAS first use, mostly torch.addmm in ops/fused_linear_norm.py,
torch.baddbmm in modules/preprocessors.py and the first torch._scaled_mm. None
of that depends on the data or on M. This module pays it during initialization by
running each family the first forward AND backward use once, on synthetic
tensors, before INIT_STOP/RUN_START. MLPerf training rules allow library
initialization / JIT on synthetic inputs before RUN_START (model init time,
training_rules L545-549); NVIDIA's submissions do the same with comm/compile
warmups.

WHAT IT DOES NOT TOUCH. No module, parameter, buffer, optimizer, FP8 scale or
weight cache (ops/triton/triton_fp8.py _weight_cache/_scale_value_cache/...,
_residual_workspaces), FP8 calibration counter (bwd_fp8 _sites), Triton
autotune table, or BLAS pin table (ops/blas_solution_pins.py
_seen/_state/_active). One intentional, state-equivalent exception: the "fp8
output-scale-mode probe" case runs triton_fp8._fp8_output_scale_mode(), which
caches the per-device mode string in _fp8_output_scale_mode_by_device and adds
its probe key to _quiet_seen -- the identical, data-free values the first
fp8_supported() call of step 1 would cache. A second, only under
DLRM_HSTU_FP8_BWD_OUT_WGRAD=1 (default off): the FP8 out-proj wgrad NN case
creates ops/hipblaslt_fp8_nn's per-device handle and workspace (the objects
its first armed call would create) and caches one heuristic result keyed by
M=16 (valid for exactly that problem, so it cannot shadow a real batch's
algorithm). A third, only under DLRM_HSTU_FP8_FWD_U_DP=1 (default off): the
U-slice data-parallel case creates hipblaslt_scaled_mm_bias's per-device
hipBLASLt handle (the object its first armed call would create); it keeps no
algorithm cache and gets a private workspace. Every operand is a fresh
tensor built from torch.full/torch.ones (no RNG draw: the CPU and CUDA default
generators are checked before/after and restored if anything moved). FP8 calls
go straight to torch._scaled_mm with private scale tensors, and the
residual-fused extension is called with a private workspace.

WHY A TINY M CANNOT SHADOW A PIN. M (rows) is the token count on the HSTU
projections (total jagged tokens per rank, ~2.0-2.4 M at yambda-5b), the
pre-contextual token count on the preprocessor MLPs, and the per-rank sample
count (BATCH_SIZE=1024) on the item MLP / multitask head / contextual baddbmm.
The warmup uses M=16 (a multiple of 16 for _scaled_mm). Even if some tail batch
ever produced M=16, nothing here can change what that batch resolves to: with
PYTORCH_TUNABLEOP_TUNING=0, TunableOp::operator() looks the key up and, on a
miss, runs "Default" WITHOUT adding the key to the results table
(ATen/cuda/tunable/TunableOp.h: mgr.Add only under IsTuningEnabled()), so a
later blas_solution_pins.install_for_tokens(16) still merges its pin and wins.
The pins are installed lazily per M by install_for_tokens (hooks in
modules/dlrm_hstu.py, ops/hstu_compute.py,
ops/triton/triton_hstu_preprocess_and_attention.py); this module never calls
it, so the pin state is left exactly as the first real batch expects it.
TunableOp itself is configured from os.environ by apply_env_bootstrap (Phase 1
of train_ranker), long before this runs.

E-M3a HOOK (_warmup_preproc_fp8_l2, only with DLRM_HSTU_FP8_PREPROC_L2=1, default
OFF; gate unset or "0": it returns before importing anything). It runs
preproc_fp8_l2.warmup_kernels on private torch.full tensors: no _Site, module,
parameter or RNG state. What it leaves behind: Triton compiled kernels, the
finalize's autotune best_config under the gin pins, ops/hipblaslt_fp8_nn heuristic
entries keyed by the exact warm-up token counts (removed again unless that
module's cache mode is "exact"), and triton_fp8._quiet_seen shape keys. It skips
like warmup() when TunableOp tuning or record-untuned is on, and releases its
~2.5 GB transient (torch.cuda.empty_cache) before alloc_pregrow runs. It imports
no FlyDSL: run_blas_warmup runs before the DataLoader pre-start forks its pools.
E-M3a's stacked LayerNorm(512) backward kernels (FlyDSL) are JIT-built after the
forks by run_startup_preload (_prebuild_flydsl_ln_bwd, " grp3f8=..."), next to
PR #7's and E-M2's; without DL_PRESTART they compile at first use.

E-X2b HOOK (_prebuild_ec_dgrad_segreduce, only with
DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE=1, default OFF; unset or "0": returns before
importing anything). triton_ec_dgrad_segreduce.prebuild compiles the four
Triton kernels on a ~1 MB synthetic input built without RNG and checks the
result bitwise against fp64. It leaves Triton compiled kernels and that
module's gate cache behind; a failure latches only E-X2b off.

X2H HOOK (_prebuild_ec_dgrad_x2h, inside the E-X2b hook's RNG / sync bracket,
only with DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=1 (default OFF) while E-X2b is
still armed; unset or "0": nothing is imported). ops/hip/ec_dgrad_x2h.prebuild
builds x2h_pass.cu with hipcc (cached per source / flags / toolchain), checks it
exactly on E-X2b's synthetic input and bitwise against the Triton passes on a
deterministic rounding-heavy input (no RNG state; sentinel-filled outputs,
counters checked after every call), then arms the device. It leaves the loaded
library, the zero row and the tile counters behind and releases its transient
(~0.8 GB at 256 CUs; torch.cuda.empty_cache) before alloc_pregrow. A failure
latches only X2H off (E-X2b keeps its Triton passes). With the knob on but E-X2b
off, not armed or latched off by its prebuild, the hook prints the knob's
IGNORED line instead.

Indexed-X HOOK (_prebuild_ec_indexed_l1, only with DLRM_HSTU_FP8_EC_INDEXED_L1=1, default OFF; unset or "0":
returns before importing anything). triton_ec_indexed_l1.prebuild compiles the six Triton kernels (_ecix_fwd,
_ecix_wgrad, _ecix_wgrad_reduce x item + cat) on a synthetic integer-valued input built without RNG and checks
them bitwise against fp64; it releases its ~80 MiB transient (torch.cuda.empty_cache) before alloc_pregrow.
A failure latches only Indexed-X off.

IX-FP8F HOOK (_prebuild_ec_indexed_l1_fp8, inside the Indexed-X hook's RNG / sync / empty_cache bracket, only with
DLRM_HSTU_FP8_EC_INDEXED_L1_FP8=1 (default OFF) while Indexed-X is armed; unset or "0": nothing is imported).
triton_ec_indexed_l1_fp8.prebuild compiles every FP8F kernel at the production class and checks it against its exact
reference; the hook prints "[ec-indexed-l1-fp8] pre-run_start prebuild OK (...)". A failure latches only IX-FP8F
off (Indexed-X stays on). With the knob on but Indexed-X off, the hook prints the knob's IGNORED line instead.

X2L-B HOOKS (gin item (93), ops/hip/preproc_l2_x2lb.py, only with DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1, default OFF;
unset or "0": nothing is imported, nothing is logged and the result strings are unchanged). Two phases:
  phase 1 (_prebuild_preproc_l2_x2lb_build, in run_blas_warmup after the E-M3a hook, before the DataLoader forks; no
    FlyDSL import): hipcc builds x2lb_pass.cu (cached per source / flags / toolchain), then the resource census, the
    ISA gate (x2lb_isa/isa_check.py C1-C18, a child process) and the ABI check. With E-M3a or PR #7 off it prints
    the knob's IGNORED line instead.
  phase 2 (_prebuild_preproc_l2_x2lb_check, in _prebuild_flydsl_ln_bwd after E-M3a's grp3f8 block, inside its
    synchronize bracket): the self-check against the shipped chain (triton_fp8._scaled_mm + PR #7's ln_bwd) on
    sentinel-filled synthetic data at three production M (no RNG state; checked and restored like the other
    hooks), then the device is armed; " x2lb=1|off|error" joins the startup-preload line and the transient
    (<= 5 x M x 512 B, ~6.4 GB) is released (torch.cuda.empty_cache) before alloc_pregrow. When the preload does not
    reach it (not prestarted, FlyDSL preload or PR #7 off, not gfx950) the reason is noted and the first publish
    prints "[preproc-l2-x2lb] DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 IGNORED (NOT armed): phase-2 self-check not run".
A failure in either phase latches only X2L-B off (one FAILED CLOSED line); E-M3a and PR #7 are not touched.
"""

import contextlib
import logging
import os
import sys
import time
from typing import Callable, Iterator, List, Optional, Tuple

import gin
import torch
import torch.nn.functional as F

logger: logging.Logger = logging.getLogger(__name__)

# Rows of every synthetic GEMM. See the module docstring for why no real batch
# depends on what this resolves to. Multiple of 16 for torch._scaled_mm.
WARMUP_M: int = 16

# Production shapes (yambda_5b.gin + dlrm_v4/configs.py): embedding/transducer
# dim D=512, 4 heads x (linear 128, qk 128) -> uvqk N=2048, attention output
# 3*4*128=1536, targets-only V/K pack N=1024, preprocessor hidden 256,
# additional-embedding MLP in 2*512=1024, action MLP in 8*3=24, timestamp
# postprocessor in 512+2*2=516, item MLP in 3*512=1536, 1 task, 8 contextual
# slots (uid + 7 crosses).
_D = 512
_UVQK_N = 2048
_ATTN_OUT = 1536
_VK_N = 1024
_U_N = 512
_VQK_N = 1536
_PRE_HIDDEN = 256
_ADDL_IN = 1024
_ACTION_IN = 24
_POST_IN = 516
_ITEM_IN = 1536
_NUM_TASKS = 1
_CONTEXTUAL = 8

_BF16 = torch.bfloat16
_F32 = torch.float32


def _full(shape: Tuple[int, ...], value: float, dtype: torch.dtype,
          device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.full(shape, value, dtype=dtype, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ----------------------------------------------------------------------------
# BF16 / FP32 families. Each case returns the number of BLAS calls it issued.
# Every case reproduces the call form (function, operand layout/strides, bias
# kind, dtype) of a production call site; see the file list in run_cases().
# ----------------------------------------------------------------------------


def _addmm_bias_nn(device: torch.device, k: int, n: int, m: int) -> int:
    """addmm(bias1d, x[M,K], W[K,N]) -- HSTU uvqk / targets V/K, U/Q projection
    (GemmAndBias BFloat16 NN), plus the matching triton_addmm_bwd pair:
    wgrad mm(x.t(), dz) (NT) and dgrad mm(dz, W.t()) (TN)."""
    with torch.no_grad():
        x = _full((m, k), 0.0625, _BF16, device)
        w = _full((k, n), 0.0625, _BF16, device)
        b = _full((n,), 0.0625, _BF16, device)
        z = torch.addmm(b, x, w)
        dz = torch.ones_like(z)
        torch.mm(x.t(), dz)
        torch.mm(dz, w.t())
    return 3


def _addmm_residual_nn(device: torch.device, k: int, n: int, m: int) -> int:
    """addmm(residual2d[M,N], y[M,K], W[K,N]) -- bf16 HSTU output projection
    (calibration steps / bf16 arm), plus its backward
    dgrad mm(dout, W.t()) and wgrad mm(y.t(), dout) (triton_hstu_linear)."""
    with torch.no_grad():
        y = _full((m, k), 0.0625, _BF16, device)
        w = _full((k, n), 0.0625, _BF16, device)
        r = _full((m, n), 0.0625, _BF16, device)
        out = torch.addmm(r, y, w)
        dout = torch.ones_like(out)
        torch.mm(dout, w.t())
        torch.mm(y.t(), dout)
    return 3


def _fused_linear_norm(device: torch.device, k: int, n: int, m: int) -> int:
    """ops/fused_linear_norm.py _LinearNormFunction (preprocessor MLPs under
    autocast bf16): fwd addmm(b, x, W.t()) with W[N,K] (GemmAndBias TN), bwd
    dh @ W (NN) and dh.t() @ x (NT)."""
    with torch.no_grad():
        x = _full((m, k), 0.0625, _BF16, device)
        w = _full((n, k), 0.0625, _BF16, device)
        b = _full((n,), 0.0625, _BF16, device)
        h = torch.addmm(b, x, w.t())
        dh = torch.ones_like(h)
        dh @ w
        dh.t() @ x
    return 3


def _linear_autograd(device: torch.device, k: int, n: int, m: int,
                     dtype: torch.dtype) -> int:
    """nn.Linear forward + autograd backward on private leaves (F.linear ->
    addmm(b, x, W.t()), AddmmBackward0 dgrad/wgrad). bf16: timestamp
    postprocessor under the transducer autocast. fp32: item-embedding MLP and
    multitask head, which run outside any autocast."""
    with torch.enable_grad():
        x = _full((m, k), 0.0625, dtype, device, requires_grad=True)
        w = _full((n, k), 0.0625, dtype, device, requires_grad=True)
        b = _full((n,), 0.0625, dtype, device, requires_grad=True)
        y = F.linear(x, w, b)
        y.backward(torch.ones_like(y))
    return 3


def _contextual_baddbmm(device: torch.device, m: int) -> int:
    """modules/preprocessors.py contextual projection: baddbmm(bias[C,1,D],
    x.view(-1,C,D).transpose(0,1), W[C,D,D]) in bf16 (strided-batched GEMM,
    A with ld=C*D), plus its autograd backward (two bmm)."""
    with torch.enable_grad():
        bias = _full((_CONTEXTUAL, _D), 0.0, _BF16, device, requires_grad=True)
        w = _full((_CONTEXTUAL, _D, _D), 0.0625, _BF16, device,
                  requires_grad=True)
        x = _full((m * _CONTEXTUAL, _D), 0.0625, _BF16, device,
                  requires_grad=True)
        out = torch.baddbmm(
            bias.view(-1, 1, _D),
            x.view(-1, _CONTEXTUAL, _D).transpose(0, 1),
            w,
        ).transpose(0, 1)
        out.backward(torch.ones_like(out))
    return 3


# ----------------------------------------------------------------------------
# FP8 families (only when the FP8 arm is on). Lowest-level torch._scaled_mm
# with private scales; no triton_fp8 wrapper (they populate weight/scale/probe
# caches), no bwd_fp8 entry point (it advances per-site counters).
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet_stderr(enabled: bool) -> Iterator[None]:
    """Keep hipBLASLt's first-FP8-shape heuristic chatter (~170 lines per new
    shape, see triton_fp8._scaled_mm_call) off fd 2. Exceptions still
    propagate and are logged by the caller after fd 2 is restored."""
    if not enabled:
        yield
        return
    try:
        sys.stderr.flush()
    except Exception:
        pass
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _fp8_cases(device: torch.device, m: int
               ) -> List[Tuple[str, Callable[[], int]]]:
    if torch.version.hip is None or not hasattr(torch, "_scaled_mm"):
        return []
    from generative_recommenders.ops.triton import triton_fp8 as tf8

    e4m3 = tf8.fp8_e4m3_dtype()
    from generative_recommenders.ops.triton import bwd_fp8

    grad_dtype = getattr(bwd_fp8, "_GRAD_DTYPE", None)
    dgrad_cases: List[Tuple[str, Callable[[], int]]] = []
    if getattr(bwd_fp8, "_BWD_ENABLED", False) and grad_dtype is not None:
        # bwd_fp8.output_projection_dgrad (gated by DLRM_HSTU_FP8_BWD alone):
        # e5m2 grad x e4m3 W^T.
        def dgrad() -> int:
            g = torch.full((m, _D), 0.5, dtype=_BF16, device=device).to(grad_dtype)
            w = torch.full((_ATTN_OUT, _D), 0.5, dtype=_BF16,
                           device=device).to(e4m3)
            torch._scaled_mm(
                g, w.t(),
                scale_a=torch.ones((1,), dtype=_F32, device=device),
                scale_b=torch.ones((1,), dtype=_F32, device=device),
                out_dtype=_BF16,
            )
            return 1

        dgrad_cases.append(("fp8 e5m2 x e4m3 -> bf16 out_proj dgrad", dgrad))
        if getattr(bwd_fp8, "_OUT_WGRAD_ARMED", False) and os.environ.get(
            "DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL", "nn"
        ) == "nn":
            # FP8 output-projection wgrad, NN sub-mode (DLRM_HSTU_FP8_BWD_OUT_WGRAD=1
            # only): ops/hipblaslt_fp8_nn.fp8_wgrad_nn, e4m3 y [M, 1536] x
            # token-major dout [512, M] in the format the arm resolved. Warms
            # the library load, its own hipBLASLt handle and workspace (the
            # same per-device objects the first armed call would create) and
            # one heuristic query. The algo cache is keyed by the exact token
            # count, so the M=16 entry is valid only for an M=16 problem.
            t_dtype = (
                e4m3 if getattr(bwd_fp8, "_OUT_WGRAD_T_E4M3", False) else grad_dtype
            )

            def out_wgrad_nn() -> int:
                from generative_recommenders.ops.hipblaslt_fp8_nn import (
                    fp8_wgrad_nn,
                )

                y = torch.full((m, _ATTN_OUT), 0.5, dtype=_BF16,
                               device=device).to(e4m3)
                dout_t = torch.full((_D, m), 0.5, dtype=_BF16,
                                    device=device).to(t_dtype)
                one = torch.ones((1,), dtype=_F32, device=device)
                out = fp8_wgrad_nn(y, dout_t, one, one, out_dtype=_F32)
                if out is None:
                    raise RuntimeError("no hipBLASLt NN algorithm")
                return 1

            dgrad_cases.append(("fp8 e4m3 x fp8 -> f32 out_proj wgrad (nn)",
                                out_wgrad_nn))
    if not tf8.hstu_fp8_gemm_enabled():
        return dgrad_cases
    fast_accum = bool(tf8._HSTU_FP8_FAST_ACCUM)
    one11 = lambda: torch.ones((1, 1), dtype=_F32, device=device)  # noqa: E731

    def act(k: int, dtype: torch.dtype = e4m3) -> torch.Tensor:
        return torch.full((m, k), 0.5, dtype=_BF16, device=device).to(dtype)

    def wcol(k: int, n: int) -> torch.Tensor:
        # Column-major [K,N] E4M3, the layout _get_weight_shadow produces.
        return torch.full((n, k), 0.5, dtype=_BF16, device=device).to(e4m3).t()

    def scaled(k: int, n: int, out_dtype: torch.dtype, bias: bool,
               scale_result: bool) -> int:
        kwargs = {
            "scale_a": one11(), "scale_b": one11(), "out_dtype": out_dtype,
            "use_fast_accum": fast_accum,
        }
        if bias:
            kwargs["bias"] = torch.zeros((n,), dtype=_BF16, device=device)
        if scale_result:
            kwargs["scale_result"] = one11()
        a, b = act(k), wcol(k, n)
        try:
            torch._scaled_mm(a, b, **kwargs)
        except TypeError:
            # Same API-compat retry as triton_fp8._scaled_mm.
            kwargs.pop("use_fast_accum", None)
            torch._scaled_mm(a, b, **kwargs)
        return 1

    def scaled_out_fp8(k: int, n: int) -> int:
        # scaled_mm_fp8_out_preq: E4M3 output with bf16 bias and scale_result
        # (multiply/divide backends) or without it (input_scale backend). Try
        # both forms so whichever the backend uses is warm.
        count = 0
        last_error: Optional[Exception] = None
        for with_result in (True, False):
            try:
                count += scaled(k, n, e4m3, bias=True, scale_result=with_result)
            except Exception as error:
                last_error = error
        if count == 0 and last_error is not None:
            raise last_error
        return count

    def output_scale_probe() -> int:
        # triton_fp8.fp8_supported()'s first-use backend probe (16x16 e4m3
        # _scaled_mm + .item()): data-free; caches the same per-device mode
        # string the first training step would cache.
        tf8._fp8_output_scale_mode(device)
        return 1

    cases: List[Tuple[str, Callable[[], int]]] = [
        ("fp8 output-scale-mode probe", output_scale_probe),
        # scaled_addmm_fp8_preq, bf16 out + bf16 bias: QKV-fused U slice,
        # full uvqk (non-QKV path), targets-only V/K bf16 output.
        ("fp8 e4m3->bf16 +bias uvqk_u", lambda: scaled(_D, _U_N, _BF16, True, False)),
        ("fp8 e4m3->bf16 +bias uvqk", lambda: scaled(_D, _UVQK_N, _BF16, True, False)),
        ("fp8 e4m3->bf16 +bias targets_vk", lambda: scaled(_D, _VK_N, _BF16, True, False)),
        # scaled_mm_fp8_out_preq, e4m3 out: QKV-fused V/Q/K and targets V/K.
        ("fp8 e4m3->e4m3 +bias vqk", lambda: scaled_out_fp8(_D, _VQK_N)),
        ("fp8 e4m3->e4m3 +bias targets_vk", lambda: scaled_out_fp8(_D, _VK_N)),
        # Non-fused residual output projection fallback (_scaled_mm, no bias).
        ("fp8 e4m3->bf16 out_proj", lambda: scaled(_ATTN_OUT, _D, _BF16, False, False)),
    ]

    # Output projection with the residual fused in (hipblaslt_scaled_addmm.cpp).
    # Only if prebuild already loaded it: never trigger the (slow) build here.
    ext_fn = getattr(tf8, "_residual_fused_extension", None)
    if (
        tf8._HSTU_FP8_FUSE_RESIDUAL
        and ext_fn is not None
        and ext_fn.cache_info().currsize > 0
    ):
        def residual_fused() -> int:
            ext = ext_fn()
            workspace = torch.empty(
                tf8._HSTU_FP8_FUSE_RESIDUAL_WORKSPACE_BYTES,
                dtype=torch.uint8, device=device,
            )
            a = act(_ATTN_OUT).contiguous()
            w = wcol(_ATTN_OUT, _D)
            r = torch.zeros((m, _D), dtype=_BF16, device=device)
            algo = int(tf8._HSTU_FP8_FUSE_RESIDUAL_ALGORITHM)
            try:
                ext.scaled_addmm(a, w, r, one11(), one11(), workspace, algo)
            except Exception:
                # The heuristic may return fewer candidates at tiny M; the
                # handle/library init is what we want, index 0 still gives it.
                if algo == 0:
                    raise
                ext.scaled_addmm(a, w, r, one11(), one11(), workspace, 0)
            return 1

        cases.append(("fp8 residual-fused out_proj (ext)", residual_fused))

    # U slice on the data-parallel hipBLASLt path (hipblaslt_scaled_mm_bias.cpp,
    # DLRM_HSTU_FP8_FWD_U_DP=1 only): loads the code, creates the extension's
    # per-device hipBLASLt handle and runs one heuristic query + GEMM with a
    # private workspace of the production size. Only if prebuild already
    # loaded it: never trigger the (slow) build here.
    u_dp_fn = getattr(tf8, "_u_dp_extension", None)
    if (
        getattr(tf8, "_HSTU_FP8_FWD_U_DP", False)
        and u_dp_fn is not None
        and u_dp_fn.cache_info().currsize > 0
    ):
        def u_dp() -> int:
            ext = u_dp_fn()
            workspace = torch.empty(
                tf8._HSTU_FP8_FWD_U_DP_WS_BYTES,
                dtype=torch.uint8, device=device,
            )
            ext.scaled_mm_bias(
                act(_D).contiguous(), wcol(_D, _U_N),
                torch.zeros((_U_N,), dtype=_BF16, device=device),
                one11(), one11(), workspace,
            )
            return 1

        cases.append(("fp8 e4m3->bf16 +bias uvqk_u data-parallel (ext)", u_dp))

    return cases + dgrad_cases


def run_cases(device: torch.device, m: int = WARMUP_M
              ) -> List[Tuple[str, Callable[[], int]]]:
    """Every distinct (op, dtype, layout, epilogue) family of the first
    forward+backward, at production N/K. Call sites:

      bf16 addmm+bias1d NN (+NT wgrad, TN dgrad)
          ops/hstu_compute.py:126, triton_hstu_preprocess_and_attention.py
          (maybe_triton_addmm_fwd -> triton_addmm.py:1656; bwd triton_addmm.py:1637-1638)
      bf16 addmm+residual2d NN (+TN dgrad, NT wgrad)
          triton_hstu_linear.py:2726 fwd, :2812 / :2878 bwd
      bf16 addmm(b, x, W.t()) TN (+NN dgrad, NT wgrad)
          ops/fused_linear_norm.py:170, :235-236 (preprocessor MLPs)
      bf16 F.linear + AddmmBackward0
          modules/postprocessors.py _time_feature_combiner (autocast)
      fp32 F.linear + AddmmBackward0
          modules/dlrm_hstu.py _item_embedding_mlp, multitask prediction head
      bf16 baddbmm strided-batched (+2 bmm)
          modules/preprocessors.py:325
      fp8 (see _fp8_cases)
          triton_fp8.py scaled_addmm_fp8_preq / scaled_mm_fp8_out_preq,
          hipblaslt_scaled_addmm.cpp, bwd_fp8.py:453
    """
    cases: List[Tuple[str, Callable[[], int]]] = [
        # HSTU projections.
        ("bf16 addmm+bias uvqk", lambda: _addmm_bias_nn(device, _D, _UVQK_N, m)),
        ("bf16 addmm+bias targets_vk/uq", lambda: _addmm_bias_nn(device, _D, _VK_N, m)),
        ("bf16 addmm+residual out_proj", lambda: _addmm_residual_nn(device, _ATTN_OUT, _D, m)),
        # Preprocessor MLPs (FusedLinearNormMLP).
        ("bf16 linear_norm content.0", lambda: _fused_linear_norm(device, _D, _PRE_HIDDEN, m)),
        ("bf16 linear_norm *.2", lambda: _fused_linear_norm(device, _PRE_HIDDEN, _D, m)),
        ("bf16 linear_norm additional.0", lambda: _fused_linear_norm(device, _ADDL_IN, _PRE_HIDDEN, m)),
        ("bf16 linear_norm action.0", lambda: _fused_linear_norm(device, _ACTION_IN, _PRE_HIDDEN, m)),
        # Contextual projection.
        ("bf16 baddbmm contextual", lambda: _contextual_baddbmm(device, m)),
        # Postprocessor (autocast bf16 nn.Linear).
        ("bf16 linear postprocessor", lambda: _linear_autograd(device, _POST_IN, _D, m, _BF16)),
        # Item MLP + multitask head (fp32, outside autocast).
        ("fp32 linear item_mlp.0", lambda: _linear_autograd(device, _ITEM_IN, _D, m, _F32)),
        ("fp32 linear item_mlp.2/head.0", lambda: _linear_autograd(device, _D, _D, m, _F32)),
        ("fp32 linear head.2", lambda: _linear_autograd(device, _D, _NUM_TASKS, m, _F32)),
    ]
    cases += _fp8_cases(device, m)
    return cases


def _rng_snapshot(device: torch.device
                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cpu = torch.random.get_rng_state()
    cuda = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    return cpu, cuda


def warmup(device: torch.device, m: int = WARMUP_M) -> Tuple[int, int, float]:
    """Run every GEMM family once on synthetic tensors. Per-case failures are
    logged and ignored; only a sticky device fault (failed synchronize) raises,
    so the run dies before RUN_START instead of at step 1.

    Returns (gemms_ok, families_failed, seconds)."""
    t0 = time.perf_counter()
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.info("[blas-warmup] skipped: no CUDA device (%s)", device)
        return 0, 0, 0.0
    if m <= 0 or m % 16 != 0:
        logger.warning("[blas-warmup] skipped: BLAS_WARMUP_M=%d must be a "
                       "positive multiple of 16", m)
        return 0, 0, 0.0
    try:
        import torch.cuda.tunable as tunable

        if tunable.is_enabled() and (
            tunable.tuning_is_enabled()
            or getattr(tunable, "record_untuned_is_enabled", lambda: False)()
        ):
            # Tuning/recording would tune or log tiny-M keys (minutes, and
            # junk lines in the results/untuned files): never warm then.
            logger.warning("[blas-warmup] skipped: TunableOp tuning or "
                           "record-untuned is ON")
            return 0, 0, 0.0
    except Exception:
        pass
    try:
        rng_before = _rng_snapshot(device)
    except Exception as error:
        logger.warning("[blas-warmup] skipped: RNG snapshot failed: %s", error)
        return 0, 0, 0.0
    ok = 0
    failed = 0
    try:
        quiet = True
        try:
            from generative_recommenders.ops.triton import triton_fp8 as tf8

            quiet = bool(tf8._HSTU_FP8_QUIET_HEURISTIC)
        except Exception:
            pass
        with torch.cuda.device(device), torch.autocast("cuda", enabled=False):
            try:
                cases = run_cases(device, m)
            except Exception:
                logger.warning("[blas-warmup] could not build case list",
                               exc_info=True)
                cases = []
            for name, fn in cases:
                silenced = quiet and name.startswith("fp8")
                if silenced:
                    # fd 2 is /dev/null inside; leave a breadcrumb first so a
                    # hard GPU abort in this window stays attributable.
                    logger.info("[blas-warmup] %s (stderr silenced)", name)
                try:
                    with _quiet_stderr(silenced):
                        ok += fn()
                except Exception as error:  # never abort a run over a warmup
                    failed += 1
                    logger.warning("[blas-warmup] %s failed (ignored): %s",
                                   name, error)
            # Not swallowed: an asynchronous device fault here is sticky, the
            # HIP context is unusable and step 1 would fail anyway -- later,
            # inside the timed region, with a misleading stack. Fail now,
            # before RUN_START.
            try:
                torch.cuda.synchronize(device)
            except Exception as error:
                raise RuntimeError(
                    "[blas-warmup] device fault during warmup; the HIP context "
                    "is unusable. Rerun with BLAS_WARMUP=0 to isolate."
                ) from error
    finally:
        try:
            rng_after = _rng_snapshot(device)
            if not torch.equal(rng_before[0], rng_after[0]):
                logger.error("[blas-warmup] CPU RNG state moved; restoring it")
                torch.random.set_rng_state(rng_before[0])
            if rng_before[1] is not None and rng_after[1] is not None and not (
                torch.equal(rng_before[1], rng_after[1])
            ):
                logger.error("[blas-warmup] CUDA RNG state moved; restoring it")
                torch.cuda.set_rng_state(rng_before[1], device)
        except Exception as error:
            logger.error("[blas-warmup] RNG state check failed: %s", error)
    return ok, failed, time.perf_counter() - t0


def _prebuild_action_dedup(device: torch.device) -> None:
    """(119) ACTION_DEDUP (ops/action_dedup.py, DLRM_HSTU_FP8_ACTION_DEDUP=1,
    code default OFF): action_dedup.prebuild checks the per-code gradient sum
    (bitwise vs fp64 on dyadic data) and the gather-add, warms the 18-row table
    MLP forward kernels and its GEMMs and E-M1's NUM_IN = 2 forward, then prints
    the ARMED line on every rank. No RNG draw (checked and restored like the
    other hooks), no model state; it releases its transient (empty_cache)
    before alloc_pregrow. Unset: returns before importing anything; "0": one
    "[action-dedup] OFF" line. A failure latches only (119) off."""
    raw = os.environ.get("DLRM_HSTU_FP8_ACTION_DEDUP")
    if raw is None:
        return
    if raw.strip().lower() not in ("1", "true", "on", "yes"):
        _x2lb_say(f"[action-dedup] OFF (DLRM_HSTU_FP8_ACTION_DEDUP={raw.strip()}): "
                  "the action branch runs per token")
        return
    try:
        import torch.cuda.tunable as tunable

        if tunable.is_enabled() and (
            tunable.tuning_is_enabled()
            or getattr(tunable, "record_untuned_is_enabled", lambda: False)()
        ):
            logger.warning("[blas-warmup] action-dedup prebuild skipped: TunableOp "
                           "tuning or record-untuned is ON")
            return
    except Exception:
        pass
    rng_before = _rng_snapshot(device)
    try:
        from generative_recommenders.ops import action_dedup

        action_dedup.prebuild(device)
    finally:
        rng_after = _rng_snapshot(device)
        if not torch.equal(rng_before[0], rng_after[0]):
            logger.error("[blas-warmup] CPU RNG state moved; restoring it")
            torch.random.set_rng_state(rng_before[0])
        if rng_before[1] is not None and rng_after[1] is not None and not (
            torch.equal(rng_before[1], rng_after[1])
        ):
            logger.error("[blas-warmup] CUDA RNG state moved; restoring it")
            torch.cuda.set_rng_state(rng_before[1], device)


# ----------------------------------------------------------------------------
# Per-stack BLAS first-use replay (ttt-fix). The M=16 pass above pays every
# M-INDEPENDENT first-use cost. The ROCm 10.2 hipBLASLt (100500) adds an
# M-DEPENDENT one: its gfx950 Tensile solutions live in lazily loaded library
# files (..._Contraction_l_Ailk_Bljk_Cijk_Dijk_CU256_ID75a0_gfx950.co 475.6 MB /
# 826 kernels, ..._Ailk_Bjlk_..._CU256_ID75a0 227 MB, ..._ID75a3-75a2 401 + 235
# MB, ...), and the heuristic reaches into one the first time a GEMM's size
# enters its region. On the yambda-5b full run that is the PREPROC_DEDUP unique
# chain's FusedLinearNorm backward (ops/fused_linear_norm.py:291/292 at R =
# 2 * M_pad rows) at rank-specific steps of windows 43-56 and on every rank at
# window 63: 0.35-1.15 s of hipModuleLoad per file inside the timed region, the
# other ranks waiting at the next collective (6-10 s per run). The v26.3
# hipBLASLt (100202) keeps those solutions in files it loads at init, so the
# cost and the table are per stack.
#
# The table (blas_replay_table_<stack>.jsonl, tools/recommendation/dlrm_v4/gen_blas_replay_table.py)
# lists the (K, N, op, M) calls whose FIRST use was slow in a dense scan of every
# row count the dedup chain can produce (training R = 512 * [1, 8192], eval R =
# 256 * [1, 8192]: preproc_dedup's structural bounds, not data). Replaying them
# once here, on synthetic constants with the production call forms
# (_fused_linear_norm), leaves every library a reachable row count needs
# resident. Rows are keyed only by sizes, so like the M=16 pass nothing here can
# change which solution a real batch resolves to (TunableOp tuning is off; a
# miss runs "Default" without caching). The table applies only when the
# TunableOp validators (torch, HIP, hipBLASLt, rocBLAS, arch) and the CU count
# match the stack it was recorded on; anything else (v26.3 included) is a no-op.
# Kill switch DLRM_BLAS_REPLAY=0.
# ----------------------------------------------------------------------------
_BLAS_REPLAY_ENV: str = "DLRM_BLAS_REPLAY"
_BLAS_REPLAY_TABLES: Tuple[str, ...] = ("blas_replay_table_rocm10.2.jsonl",)
_BLAS_REPLAY_KEYS: Tuple[str, ...] = (
    "PT_VERSION", "HIP_VERSION", "HIPBLASLT_VERSION", "ROCBLAS_VERSION", "GCN_ARCH_NAME")
_BLAS_REPLAY_SLOW_MS: float = 40.0


def _replay_view(big: torch.Tensor, width: int, m: int) -> torch.Tensor:
    """Contiguous [m, width] (ld = width) prefix of big's flat storage."""
    flat = big.view(-1)
    return flat.narrow(0, 0, m * width).view(m, width)


def replay_blas_table(device: torch.device, path: Optional[str] = None) -> dict:
    """Replay the per-stack BLAS first-use table (see the block comment above).
    Returns a small report dict; never raises (a failure leaves today's lazy loads)."""
    import json

    raw = os.environ.get(_BLAS_REPLAY_ENV, "1").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return {"state": "off"}
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"state": "no_cuda"}
    t0 = time.perf_counter()
    try:
        import torch.cuda.tunable as tunable

        live_v = dict(tunable.get_validators())
        props = torch.cuda.get_device_properties(device)
        live_cu = int(props.multi_processor_count)
    except Exception as error:
        return {"state": "error", "why": f"validators: {error}"}
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [path] if path else [os.path.join(here, t) for t in _BLAS_REPLAY_TABLES]
    mismatched = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            hdr, rows = lines[0], lines[1:]
            st = hdr.get("stack", {})
            want = st.get("validators", {})
            bad = [k for k in _BLAS_REPLAY_KEYS if live_v.get(k) != want.get(k)]
            if st.get("cu") != live_cu:
                bad.append(f"cu {st.get('cu')} != {live_cu}")
            if bad:
                mismatched.append(f"{os.path.basename(p)}: {', '.join(bad)}")
                continue
            if not rows:
                return {"state": "ok", "table": os.path.basename(p), "rows": 0, "slow": 0,
                        "seconds": round(time.perf_counter() - t0, 2)}
            kmax = max(int(r["K"]) for r in rows)
            nmax = max(int(r["N"]) for r in rows)
            mmax = max(int(r["M"]) for r in rows)
            slow = 0
            load_ms = 0.0
            with torch.cuda.device(device), torch.no_grad(), torch.autocast("cuda", enabled=False):
                x_big = torch.full((mmax * kmax,), 0.0625, dtype=_BF16, device=device)
                dh_big = torch.full((mmax * nmax,), 0.0625, dtype=_BF16, device=device)
                wb = {}
                for r in rows:
                    k, n, op, m = int(r["K"]), int(r["N"]), str(r["op"]), int(r["M"])
                    if (k, n) not in wb:
                        wb[(k, n)] = (_full((n, k), 0.0625, _BF16, device), _full((n,), 0.0625, _BF16, device))
                    w, b = wb[(k, n)]
                    x = _replay_view(x_big, k, m)
                    dh = _replay_view(dh_big, n, m)
                    torch.cuda.synchronize(device)
                    a = time.perf_counter()
                    if op == "fwd":
                        y = torch.addmm(b, x, w.t())
                    elif op == "dgrad":
                        y = dh @ w
                    else:
                        y = dh.t() @ x
                    torch.cuda.synchronize(device)
                    ms = (time.perf_counter() - a) * 1e3
                    del y
                    if ms > _BLAS_REPLAY_SLOW_MS:
                        slow += 1
                        load_ms += ms
                del x_big, dh_big, wb
            torch.cuda.empty_cache()
            return {"state": "ok", "table": os.path.basename(p), "rows": len(rows), "slow": slow,
                    "load_s": round(load_ms / 1e3, 2), "seconds": round(time.perf_counter() - t0, 2),
                    "m_max": mmax}
        except Exception as error:  # never abort a run over a warmup
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return {"state": "error", "why": f"{os.path.basename(p)}: {error}"}
    return {"state": "no_table", "why": "; ".join(mismatched) or "no table file"}


def _run_blas_replay(device: torch.device) -> None:
    try:
        import torch.cuda.tunable as tunable

        if tunable.is_enabled() and (
            tunable.tuning_is_enabled()
            or getattr(tunable, "record_untuned_is_enabled", lambda: False)()
        ):
            logger.warning("[blas-replay] skipped: TunableOp tuning or record-untuned is ON")
            return
    except Exception:
        pass
    rng_before = _rng_snapshot(device)
    try:
        rep = replay_blas_table(device)
    finally:
        rng_after = _rng_snapshot(device)
        if not torch.equal(rng_before[0], rng_after[0]):
            torch.random.set_rng_state(rng_before[0])
        if rng_before[1] is not None and rng_after[1] is not None and not torch.equal(rng_before[1], rng_after[1]):
            torch.cuda.set_rng_state(rng_before[1], device)
    st = rep.get("state")
    if st == "ok":
        logger.info(
            "armed OK: [blas-replay] table=%s rows=%d first-use loads=%d (%.2fs) in %.2fs before run_start "
            "(M<=%s, device=%s; kill switch %s=0)", rep.get("table"), rep.get("rows", 0), rep.get("slow", 0),
            rep.get("load_s", 0.0), rep.get("seconds", 0.0), rep.get("m_max"), device, _BLAS_REPLAY_ENV)
    elif st == "no_table":
        logger.info("[blas-replay] no table for this stack (%s): lazy-loading hipBLASLt files are not "
                    "pre-loaded", rep.get("why"))
    elif st == "off":
        logger.info("[blas-replay] OFF (%s=0)", _BLAS_REPLAY_ENV)
    else:
        logger.warning("[blas-replay] FAILED (%s); training continues with lazy loads", rep.get("why", st))


@gin.configurable
def run_blas_warmup(device: torch.device, enabled: int = 0,
                    m: int = WARMUP_M) -> None:
    """train_ranker hook (gin run_blas_warmup.enabled = $BLAS_WARMUP)."""
    if not int(enabled):
        return
    _prebuild_action_dedup(device)
    ok, failed, seconds = warmup(device, m=m)
    _run_blas_replay(device)
    if ok:
        logger.info(
            "armed OK: [blas-warmup] %d gemms in %.1fs before run_start "
            "(M=%d, failed_families=%d, device=%s)",
            ok, seconds, m, failed, device,
        )
    else:
        logger.warning(
            "[blas-warmup] requested but no GEMM ran (failed_families=%d, %.1fs)",
            failed, seconds,
        )
    _warmup_preproc_fp8_l2(device)
    _prebuild_preproc_l2_x2lb_build(device)
    _prebuild_ec_dgrad_segreduce(device)
    _prebuild_ec_indexed_l1(device)
    _prebuild_preproc_dedup(device)


# (93) X2L-B: a literal copy of ops/hip/preproc_l2_x2lb.ENV_VAR (read without importing).
_X2LB_ENV: str = "DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"
_X2LB_TAG: str = "[preproc-l2-x2lb]"


def _x2lb_requested() -> bool:
    return os.environ.get(_X2LB_ENV, "").strip().lower() in ("1", "true", "on", "yes")


def _x2lb_say(msg: str) -> None:
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(msg)
    except Exception:  # never abort a run over a warmup line
        logger.warning("%s", msg)


def _prebuild_preproc_l2_x2lb_build(device: torch.device) -> None:
    """X2L-B phase 1 (ops/hip/preproc_l2_x2lb.py, DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1, default OFF): build or hit
    the cache, census, ISA gate, ABI ("[preproc-l2-x2lb] pre-run_start build OK (...)", every rank). A failure
    latches only X2L-B off (one FAILED CLOSED line). Knob unset or "0": returns before importing anything."""
    if not _x2lb_requested():
        return
    try:
        from generative_recommenders.ops.hip import preproc_l2_x2lb as x2lb
    except Exception as error:  # nothing can publish without the module: say it now
        _x2lb_say(f"{_X2LB_TAG} FAILED CLOSED at prebuild (import): {type(error).__name__}: {str(error)[:300]} "
                  "-> shipped dgrad + SwishLN bwd for the rest of the run")
        try:  # the train forward then never retries the import (no second line)
            from generative_recommenders.ops import preproc_fp8_l2

            preproc_fp8_l2._x2lb_off()
        except Exception:
            pass
        return
    try:
        x2lb.prebuild_build(device)  # never raises
    except Exception as error:  # belt and braces
        try:
            x2lb.latch_off("phase 1", error)
        except Exception:
            pass


def _x2lb_phase2_not_run(why: str) -> None:
    """X2L-B: the startup preload does not reach phase 2 for `why` (noted; the first publish prints it in the knob's
    IGNORED line). Knob unset or "0": returns at once (no import, no log)."""
    if not _x2lb_requested():
        return
    try:
        from generative_recommenders.ops.hip import preproc_l2_x2lb as x2lb

        x2lb.phase2_not_run(why)
    except Exception as error:  # never abort a run over this note
        logger.warning("[startup-preload] x2lb note failed (ignored): %s", error)


def _prebuild_preproc_l2_x2lb_check(device: torch.device) -> str:
    """X2L-B phase 2, from _prebuild_flydsl_ln_bwd after E-M3a's grp3f8 block (PR #7's FlyDSL is loaded; the
    DataLoader pools are forked): the self-check against the shipped chain on synthetic data, then the device is
    armed ("[preproc-l2-x2lb] pre-run_start self-check OK (...)", every rank). Returns " x2lb=1|off|error" for the
    startup-preload line, "" with the knob unset or "0" (nothing imported). No RNG draw (checked and restored);
    the transient is released (torch.cuda.empty_cache) before alloc_pregrow. A failure latches only X2L-B off."""
    if not _x2lb_requested():
        return ""
    try:
        from generative_recommenders.ops.hip import preproc_l2_x2lb as x2lb

        rng_before = _rng_snapshot(device)
    except Exception as error:  # never abort a run over a prebuild
        logger.warning("[startup-preload] x2lb phase 2 skipped: %s", error)
        return " x2lb=error"
    try:
        token = x2lb.prebuild_check(device)  # arms the device or latches X2L-B off; never raises
    except Exception as error:  # belt and braces
        try:
            x2lb.latch_off("phase-2 self-check", error)
        except Exception:
            pass
        token = "x2lb=error"
    try:
        rng_after = _rng_snapshot(device)
        if not torch.equal(rng_before[0], rng_after[0]) or (
            rng_before[1] is not None and not torch.equal(rng_before[1], rng_after[1])
        ):
            logger.error("[startup-preload] RNG state moved in the x2lb phase 2; restoring it")
            torch.random.set_rng_state(rng_before[0])
            if rng_before[1] is not None:
                torch.cuda.set_rng_state(rng_before[1], device)
    except Exception as error:
        logger.error("[startup-preload] RNG state check failed: %s", error)
    try:  # release the self-check's transient (<= 5 U) before alloc_pregrow
        torch.cuda.empty_cache()
    except Exception as error:
        logger.warning("[startup-preload] empty_cache after the x2lb phase 2 failed: %s", error)
    return " " + token if token else ""


def _prebuild_ec_dgrad_segreduce(device: torch.device) -> None:
    """E-X2b (ops/triton/triton_ec_dgrad_segreduce.py,
    DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE=1, default OFF): JIT-build its four
    Triton kernels before RUN_START on a synthetic exact-arithmetic input
    (no RNG, no module/parameter state; ~1 MB) and check the result bitwise
    against an fp64 reference. A failure latches E-X2b off for the process
    (one FAILED CLOSED line) and startup goes on; a sticky device fault fails
    the run here, as in _warmup_preproc_fp8_l2. Gate unset or "0": returns
    before importing anything. X2H (_prebuild_ec_dgrad_x2h) runs inside this
    bracket, after E-X2b's prebuild."""
    if os.environ.get("DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE", "").strip().lower() not in (
        "1", "true", "on", "yes",
    ):
        _x2h_ignored_without_ex2b("E-X2b (DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE) is off")
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs

        if not xs.enabled():
            _x2h_ignored_without_ex2b("E-X2b is not armed (see its IGNORED / FAILED CLOSED line)")
            return
        rng_before = _rng_snapshot(device)
    except Exception as error:  # never abort a run over a warmup
        logger.warning("[blas-warmup] ec-dgrad-segreduce prebuild skipped: %s", error)
        return
    try:
        xs.prebuild(device)
    except Exception as error:
        xs.latch_off("prebuild", error)
    x2h_ran = _prebuild_ec_dgrad_x2h(device, xs)  # X2H: its own knob, only while E-X2b is armed
    try:
        rng_after = _rng_snapshot(device)
        if not torch.equal(rng_before[0], rng_after[0]) or (
            rng_before[1] is not None and not torch.equal(rng_before[1], rng_after[1])
        ):
            logger.error("[blas-warmup] RNG state moved in the ec-dgrad-segreduce "
                         "prebuild; restoring it")
            torch.random.set_rng_state(rng_before[0])
            if rng_before[1] is not None:
                torch.cuda.set_rng_state(rng_before[1], device)
    except Exception as error:
        logger.error("[blas-warmup] RNG state check failed: %s", error)
    try:
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError(
            "[blas-warmup] device fault during the ec-dgrad-segreduce prebuild; "
            "the HIP context is unusable. Rerun with "
            "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=0 (X2H) or "
            "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE=0 to isolate."
        ) from error
    if x2h_ran:  # release X2H's ~0.8 GB cross-check transient before alloc_pregrow
        try:
            torch.cuda.empty_cache()
        except Exception as error:
            logger.warning("[blas-warmup] empty_cache after the ec-dgrad-x2h prebuild failed: %s", error)


# (87) X2H: literal copies of ops/hip/ec_dgrad_x2h.ENV_VAR / TAG (read without importing).
_X2H_ENV: str = "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP"
_X2H_TAG: str = "[ec-dgrad-x2h]"


def _x2h_requested() -> bool:
    return os.environ.get(_X2H_ENV, "").strip().lower() in ("1", "true", "on", "yes")


def _x2h_ignored_without_ex2b(why: str) -> None:
    """X2H knob on while E-X2b is off, not armed or latched off: its IGNORED line, once per process (this
    hook runs once), every rank. Imports only ops/utils; knob off: returns at once."""
    if not _x2h_requested():
        return
    msg = f"{_X2H_TAG} {_X2H_ENV}=1 IGNORED (NOT armed): {why}; x2h_pass runs only as E-X2b's pass"
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(msg)
    except Exception:  # never abort a run over a warmup line
        logger.warning("%s", msg)


def _prebuild_ec_dgrad_x2h(device: torch.device, xs) -> bool:  # pyre-ignore[2]
    """X2H (ops/hip/ec_dgrad_x2h.py, DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP=1, default OFF): called inside
    _prebuild_ec_dgrad_segreduce's RNG / sync bracket, after E-X2b's prebuild, and acts only while E-X2b
    is still armed. ec_dgrad_x2h.prebuild builds and loads x2h_pass, checks it exactly and bitwise
    against the Triton passes and arms the device ("[ec-dgrad-x2h] pre-run_start prebuild OK (...)",
    every rank). A failure latches only X2H off (one FAILED CLOSED line; E-X2b stays on). True when it
    ran (the caller then releases its transient). Knob unset or "0": returns before importing."""
    if not _x2h_requested():
        return False
    if not xs.enabled():
        _x2h_ignored_without_ex2b("E-X2b failed its pre-run_start prebuild (see its FAILED CLOSED line)")
        return False
    try:
        from generative_recommenders.ops.hip import ec_dgrad_x2h as x2h
    except Exception as error:  # dgrad_segreduce then declines X2H the same way: say it now
        xs._say(f"{_X2H_TAG} FAILED CLOSED at prebuild (import): {type(error).__name__}: "
                f"{str(error)[:300]} -> Triton _ecx_dgrad_pass for the rest of the run")
        return False
    try:
        x2h.prebuild(device)  # arms the device or latches X2H off; never raises
    except Exception as error:  # belt and braces
        try:
            x2h.latch_off("prebuild", error)
        except Exception:
            pass
    return True


def _prebuild_ec_indexed_l1(device: torch.device) -> None:
    """Indexed-X L1 (ops/triton/triton_ec_indexed_l1.py, DLRM_HSTU_FP8_EC_INDEXED_L1=1, default OFF): JIT-build
    its six Triton kernels before RUN_START on a synthetic exact-arithmetic input (no RNG, no module/parameter
    state) and check the results bitwise against fp64. A failure latches Indexed-X off (one FAILED CLOSED line)
    and startup goes on; a sticky device fault fails the run here. Gate unset or "0": returns before importing."""
    if os.environ.get("DLRM_HSTU_FP8_EC_INDEXED_L1", "").strip().lower() not in ("1", "true", "on", "yes"):
        _fp8f_ignored_without_ix()
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        from generative_recommenders.ops.triton import triton_ec_indexed_l1 as ixk

        if not ixk.enabled():
            return
        rng_before = _rng_snapshot(device)
    except Exception as error:  # never abort a run over a warmup
        logger.warning("[blas-warmup] ec-indexed-l1 prebuild skipped: %s", error)
        return
    try:
        ixk.prebuild(device)
    except Exception as error:
        ixk.latch_off("prebuild", error)
    _prebuild_ec_indexed_l1_fp8(device, ixk)  # IX-FP8F: its own knob, only while Indexed-X is still armed
    try:
        rng_after = _rng_snapshot(device)
        if not torch.equal(rng_before[0], rng_after[0]) or (
            rng_before[1] is not None and not torch.equal(rng_before[1], rng_after[1])
        ):
            logger.error("[blas-warmup] RNG state moved in the ec-indexed-l1 prebuild; restoring it")
            torch.random.set_rng_state(rng_before[0])
            if rng_before[1] is not None:
                torch.cuda.set_rng_state(rng_before[1], device)
    except Exception as error:
        logger.error("[blas-warmup] RNG state check failed: %s", error)
    try:
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError(
            "[blas-warmup] device fault during the ec-indexed-l1 prebuild; the HIP context is unusable. "
            "Rerun with DLRM_HSTU_FP8_EC_INDEXED_L1=0 to isolate."
        ) from error
    # Release the ~80 MiB prebuild transient before alloc_pregrow maps the free HBM (as E-M3a's hook does).
    try:
        torch.cuda.empty_cache()
    except Exception as error:
        logger.warning("[blas-warmup] empty_cache after the ec-indexed-l1 prebuild failed: %s", error)


# IX-FP8F (gin item (79)): literal copies of ops/triton/triton_ec_indexed_l1_fp8.ENV_VAR / TAG (read without importing).
_FP8F_ENV: str = "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8"
_FP8F_TAG: str = "[ec-indexed-l1-fp8]"


def _fp8f_requested() -> bool:
    return os.environ.get(_FP8F_ENV, "").strip().lower() in ("1", "true", "on", "yes")


def _fp8f_ignored_without_ix() -> None:
    """IX-FP8F knob on with Indexed-X off: its IGNORED line, once per process (this hook runs once), every rank.
    Imports only ops/utils; knob off: returns at once."""
    if not _fp8f_requested():
        return
    msg = (f"{_FP8F_TAG} {_FP8F_ENV}=1 IGNORED (NOT armed): Indexed-X (DLRM_HSTU_FP8_EC_INDEXED_L1) is off; the FP8 "
           "forward runs only on the Indexed-X path")
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(msg)
    except Exception:  # never abort a run over a warmup line
        logger.warning("%s", msg)


def _prebuild_ec_indexed_l1_fp8(device: torch.device, ixk) -> None:  # pyre-ignore[2]
    """IX-FP8F (ops/triton/triton_ec_indexed_l1_fp8.py, DLRM_HSTU_FP8_EC_INDEXED_L1_FP8=1, default OFF): called
    inside _prebuild_ec_indexed_l1's RNG / sync / empty_cache bracket, after the Indexed-X prebuild, and acts only
    while Indexed-X is still armed. F8.prebuild JIT-builds every FP8F kernel at the production specialization class
    and checks it against its exact reference; its summary becomes the "[ec-indexed-l1-fp8] pre-run_start prebuild
    OK (...)" line (every rank). Same failure policy as the Indexed-X prebuild: a failure latches only IX-FP8F off
    (F8.latch_off, one FAILED CLOSED line; Indexed-X stays on) and startup goes on. Knob unset or "0": returns
    before importing anything."""
    if not _fp8f_requested() or not ixk.enabled():
        return
    try:
        from generative_recommenders.ops.triton import triton_ec_indexed_l1_fp8 as f8
    except Exception as error:  # the first FP8 forward fails the same way (ec_indexed_l1 fails closed): say it now
        ixk._say(f"{_FP8F_TAG} FAILED CLOSED at prebuild (import): {type(error).__name__}: {str(error)[:300]} -> "
                 "bf16 Indexed-X for the rest of the run")
        return
    try:
        if not f8.fp8f_enabled():
            return
        summary = str(f8.prebuild(device))
    except Exception as error:
        try:
            f8.latch_off("prebuild", error)
        except Exception:
            ixk._say(f"{_FP8F_TAG} FAILED CLOSED at prebuild: {type(error).__name__}: {str(error)[:300]} -> "
                     "bf16 Indexed-X for the rest of the run")
        return
    ok = f"{_FP8F_TAG} pre-run_start prebuild OK"
    ixk._say(summary if summary.startswith(ok) else f"{ok} ({summary})")


# Item (118) PREPROC_DEDUP: a literal copy of ops/preproc_dedup.ENV_VAR (read without importing; only "1" arms it).
_PPDD_ENV: str = "DLRM_HSTU_FP8_PREPROC_DEDUP"


def _prebuild_preproc_dedup(device: torch.device) -> None:
    """Item (118) PREPROC_DEDUP (ops/preproc_dedup.py, DLRM_HSTU_FP8_PREPROC_DEDUP=1, default OFF): preproc_dedup.
    prebuild = the exact self-check (construction vs a host reference, gather + segment sums, hi / lo) and the
    warm-up of every Triton launch class the unique-row chain, the token gather and the x combine reach (forward
    calls + the backward's segment reduces, synthetic integer data, no RNG; no LN backward, so no FlyDSL import
    before the DataLoader pools fork: its variants are prebuilt after the fork), then ONE "armed OK: [preproc-dedup]
    ARMED preproc_dedup (118) (...)" line per rank, or its IGNORED / NOT armed line. A failure leaves only (118) off
    (today's statements) and startup goes on; a sticky device fault fails the run here. The transient (~8 GB at the
    largest class) is released before alloc_pregrow. Knob not "1": returns before importing anything."""
    if os.environ.get(_PPDD_ENV, "").strip() != "1":
        return
    try:
        from generative_recommenders.ops import preproc_dedup as ppdd

        rng_before = _rng_snapshot(device)
    except Exception as error:  # never abort a run over a warmup (the first EC call then prints NOT armed)
        logger.warning("[blas-warmup] preproc-dedup prebuild skipped: %s", error)
        return
    ppdd.prebuild(device)  # never raises
    try:
        rng_after = _rng_snapshot(device)
        if not torch.equal(rng_before[0], rng_after[0]) or (
            rng_before[1] is not None and not torch.equal(rng_before[1], rng_after[1])
        ):
            logger.error("[blas-warmup] RNG state moved in the preproc-dedup prebuild; restoring it")
            torch.random.set_rng_state(rng_before[0])
            if rng_before[1] is not None:
                torch.cuda.set_rng_state(rng_before[1], device)
    except Exception as error:
        logger.error("[blas-warmup] RNG state check failed: %s", error)
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError(
            "[blas-warmup] device fault during the preproc-dedup prebuild; the HIP context is unusable. "
            "Rerun with DLRM_HSTU_FP8_PREPROC_DEDUP=0 to isolate."
        ) from error
    try:
        torch.cuda.empty_cache()
    except Exception as error:
        logger.warning("[blas-warmup] empty_cache after the preproc-dedup prebuild failed: %s", error)


def _warmup_preproc_fp8_l2(device: torch.device) -> None:
    """E-M3a (ops/preproc_fp8_l2.py, DLRM_HSTU_FP8_PREPROC_L2=1, default OFF):
    JIT-compile and first-use-load its Triton kernels and FP8 GEMMs on
    synthetic tensors before RUN_START (preproc_fp8_l2.warmup_kernels: no
    _Site, module, parameter or RNG state touched), instead of in the first
    FP8 training step (seconds with a cold Triton cache). Logs one INFO line;
    E-M3a's engagement is still proven only by its "armed OK: ... first FP8
    call" line in step WARMUP + 1. Gate unset or "0": returns before importing
    anything (nothing changes). Gate on but declined (preproc_fp8_l2.enabled()
    False), or TunableOp tuning / record-untuned on: no warm-up. Called from
    run_blas_warmup, so BLAS_WARMUP=0 skips it too. Imports no FlyDSL (this
    runs before the DataLoader pre-start forks its pools): E-M3a's stacked
    LayerNorm(512) backward is prebuilt by run_startup_preload after the
    forks (_prebuild_flydsl_ln_bwd -> preproc_fp8_l2.prebuild_grouped)."""
    # The gate's name, read without importing the module (unarmed: no import).
    if os.environ.get("DLRM_HSTU_FP8_PREPROC_L2", "0") != "1":
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        import torch.cuda.tunable as tunable

        if tunable.is_enabled() and (
            tunable.tuning_is_enabled()
            or getattr(tunable, "record_untuned_is_enabled", lambda: False)()
        ):
            # As warmup(): tuning/recording would tune or log warm-up-size keys.
            logger.warning("[blas-warmup] preproc-fp8-l2 warm-up skipped: "
                           "TunableOp tuning or record-untuned is ON")
            return
    except Exception:
        pass
    try:
        from generative_recommenders.ops import preproc_fp8_l2 as l2

        if not l2.enabled():
            return
        rng_before = _rng_snapshot(device)
    except Exception as error:  # never abort a run over a warmup
        logger.warning("[blas-warmup] preproc-fp8-l2 warm-up skipped: %s", error)
        return
    failure: Optional[BaseException] = None
    calls, seconds = 0, 0.0
    try:
        quiet = True
        try:
            from generative_recommenders.ops.triton import triton_fp8 as tf8

            quiet = bool(tf8._HSTU_FP8_QUIET_HEURISTIC)
        except Exception:
            pass
        calls, seconds = l2.warmup_kernels(
            device, quiet_stderr=lambda: _quiet_stderr(quiet)
        )
    except Exception as error:  # judged below, after the device check
        failure = error
    finally:
        try:
            rng_after = _rng_snapshot(device)
            if not torch.equal(rng_before[0], rng_after[0]):
                logger.error("[blas-warmup] CPU RNG state moved; restoring it")
                torch.random.set_rng_state(rng_before[0])
            if rng_before[1] is not None and rng_after[1] is not None and not (
                torch.equal(rng_before[1], rng_after[1])
            ):
                logger.error("[blas-warmup] CUDA RNG state moved; restoring it")
                torch.cuda.set_rng_state(rng_before[1], device)
        except Exception as error:
            logger.error("[blas-warmup] RNG state check failed: %s", error)
    # Not swallowed, as in warmup(): a sticky device fault fails the run here,
    # before RUN_START.
    try:
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError(
            "[blas-warmup] device fault during the preproc-fp8-l2 warm-up "
            f"({failure!r}); the HIP context is unusable. Rerun with "
            "DLRM_HSTU_FP8_PREPROC_L2=0 or BLAS_WARMUP=0 to isolate."
        ) from error
    if failure is not None:
        logger.warning("[blas-warmup] preproc-fp8-l2 warm-up failed (ignored; "
                       "the first FP8 step compiles instead): %s", failure)
    else:
        logger.info(
            "[preproc-fp8-l2] pre-run_start warm-up OK: %d kernel/GEMM calls in "
            "%.2fs (rows=%s, emu=%s, device=%s)",
            calls, seconds,
            ",".join(str(r) for r in l2._WARM_ROWS),
            ",".join(str(e) for e in l2._warm_emus()), device,
        )
    # Release the warm-up's transient (~2.5 GB of cached blocks, incl. one
    # 2 GiB buffer) before alloc_pregrow maps the free HBM.
    try:
        torch.cuda.empty_cache()
    except Exception as error:
        logger.warning("[blas-warmup] empty_cache after the preproc-fp8-l2 "
                       "warm-up failed: %s", error)


def _preload_flydsl() -> str:
    """Import the FlyDSL fwd/bwd kernel modules (dlopens the ~235 MB
    libFlyPythonCAPI once, a noticeable fraction of a second) -- the import the
    first attention call would otherwise pay inside the timed region. Uses the
    same latched loaders the first call uses, so the cached module objects are
    exactly what production then reads. No kernel is compiled or launched."""
    try:
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as _mha

        if _mha.unsupported_hardware_reason() is not None:
            return "skip(unsupported)"
        bwd = _mha._load_kernel() is not None
        fwd = _mha._load_fwd_kernel() is not None
        return f"bwd={int(bwd)} fwd={int(fwd)}"
    except Exception as error:  # never abort a run over a preload
        logger.warning("[startup-preload] flydsl import failed (ignored): %s", error)
        return "error"


# (wln_bwd_flydsl, DLRM_HSTU_FP8_WLN_BWD_FLYDSL) variants the step runs, as
# (D, is_swish, residual, sum_dx, weight dtype): the STU norms pass weights
# cast to x.dtype (modules/stu.py -> bf16), the preprocessor MLP norms pass the
# fp32 Parameter uncast (ops/fused_linear_norm.py). Exactly the production
# specializations: 4 mains + 3 finalizes (the two bf16 D=512 variants share
# one finalize), plus the residual variant's mscale main with
# DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE on (and its packed-mask main when the
# forward writes the bit-packed mask, _mscale_mask_bits_armed); any other
# dtype only costs a first-step compile, it cannot break the run.
_LN_BWD_WARM_VARIANTS = ((512, False, True, False, torch.bfloat16),
                         # targets-only last STU layer under item (20e)
                         # TGT_SPARSE_RESID: residual=None (stu.py
                         # residual_rows_only)
                         (512, False, False, False, torch.bfloat16),
                         (512, False, False, True, torch.float32),
                         (256, True, False, True, torch.float32))


def _mscale_mask_bits_armed(D: int) -> bool:
    """Whether the fused posenc+dropout forward writes the input-dropout mask
    bit-packed (DLRM_HSTU_FP8_DROPOUT_MASK_BITS, R-C7) at this D: its static
    switches and the forward's own layout decision
    (triton_position._dropout_mask_bits_why_not with the mask/scale handoff
    switch, on this device: the FP8 arm's fused residual + LN and rccl's
    first-use packed-fold self-test, whose cached result the forward then
    reuses). Then the layer-0 mscale call gets the uint8 [N, D / 8] mask.
    A prebuild hint only: if the check itself fails (e.g. a renamed R-C7
    name), it says so and returns False -- the packed-mask kernel then
    compiles at its first call -- instead of latching FlyDSL off."""
    try:
        from generative_recommenders.ops.input_dropout_fold import input_dropout_fold_enabled
        from generative_recommenders.ops.triton import triton_position as _tp

        return bool(
            _tp._DROPOUT_MASK_BITS
            and _tp._POSENC_DROPOUT_FUSE
            and not _tp._dropout_mask_bits_why_not(
                input_dropout_fold_enabled(),
                D,
                torch.device("cuda", torch.cuda.current_device()),
            )
        )
    except Exception as error:
        logger.warning("[startup-preload] packed-mask (R-C7) arming check failed, "
                       "not prebuilt (ignored): %s", error)
        return False


def _prebuild_flydsl_ln_bwd() -> str:
    """JIT-compile, load and launch once the WLN_BWD_FLYDSL FlyDSL
    weighted-LayerNorm backward kernels (ops/flydsl/ln_bwd.py: build_ln_bwd
    main + build_finalize, one flyc.jit specialization each per (variant,
    weight dtype)) on synthetic zero/one tensors, so the first training
    backward does not compile them after RUN_START. The jit cache key holds no
    tensor shape or runtime scalar (flydsl jit_function.py _make_cache_key:
    tensors key by type, M/ITERS/G are fx.Int32 runtime args), so a small M
    builds exactly the production specialization. Compile/JIT on synthetic
    inputs only: no data, no model state, no RNG draw (zeros/ones only);
    outputs are discarded.

    A FlyDSL import, compile or launch error latches WLN_BWD_FLYDSL off
    (triton_layer_norm._wln_bwd_flydsl_disable: one loud "[ln-bwd] flydsl
    DISABLED" line, every call then takes Triton) and returns "ln_bwd=error".

    Then the same for E-M2's FlyDSL grouped LayerNorm backward, when that
    path can run and WLN_BWD_FLYDSL is not latched off (" grp3=1|error"), and,
    only with E-M3a's gate DLRM_HSTU_FP8_PREPROC_L2=1, for E-M3a's stacked
    LayerNorm(512) backward (preproc_fp8_l2.prebuild_grouped:
    " grp3f8=1|error|off"; "off" when that path cannot run, e.g. its kill
    switch DLRM_HSTU_FP8_PREPROC_L2_GROUPED is "0" or WLN_BWD_FLYDSL is latched
    off). Each latches only its own path off on an error and never aborts the
    run. Neither runs when WLN_BWD_FLYDSL is off or the device is not gfx950
    (the function returns before them). Then, only with
    DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1, X2L-B's phase 2 (gin item (93);
    _prebuild_preproc_l2_x2lb_check: " x2lb=1|off|error"; its own latch). A
    failed synchronize is a sticky device fault: it raises here, before
    RUN_START, as warmup() does."""
    from generative_recommenders.ops.triton import triton_layer_norm as _tln

    if not _tln._wln_bwd_flydsl_enabled():
        _x2lb_phase2_not_run("PR #7 (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) is off")
        return "ln_bwd=off"
    device = torch.device("cuda", torch.cuda.current_device())
    result = "ln_bwd=error"
    try:
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
        if not arch.startswith("gfx950"):
            _x2lb_phase2_not_run(f"device arch {arch or 'unknown'} is not gfx950")
            return f"ln_bwd=skip(arch {arch or 'unknown'})"
        from generative_recommenders.ops.flydsl.ln_bwd import ln_bwd

        m = 1024
        built = 0
        mscale = ""
        for D, is_swish, residual, sum_dx, w_dtype in _LN_BWD_WARM_VARIANTS:
            x = torch.zeros((m, D), dtype=torch.bfloat16, device=device)
            dy = torch.zeros_like(x)
            w = torch.ones((D,), dtype=w_dtype, device=device)
            b = torch.zeros((D,), dtype=w_dtype, device=device)
            mean = torch.zeros((m,), dtype=torch.float32, device=device)
            rstd = torch.ones((m,), dtype=torch.float32, device=device)
            res = torch.zeros_like(x) if residual else None
            ln_bwd(dy, x, w, b, mean, rstd, res, is_swish=is_swish, sum_dx=sum_dx)
            if residual and _tln._wln_bwd_flydsl_mscale_enabled():
                # the same call with item (20)(d)'s dropout mask*scale
                # epilogue (DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE; the layer-0
                # input-LayerNorm backward): one more main kernel, this
                # variant's finalize; the scale is a runtime argument
                mask = torch.ones((m, D), dtype=torch.bool, device=device)
                ln_bwd(dy, x, w, b, mean, rstd, res, dropout_mask=mask, dropout_scale=1.0)
                mscale = " mscale=1"
                if _mscale_mask_bits_armed(D):
                    # the packed-mask main (..._mscaleb_...) the R-C7 forward's
                    # uint8 [N, D / 8] mask selects; same finalize
                    bits = torch.full((m, D // 8), 0xFF, dtype=torch.uint8, device=device)
                    ln_bwd(dy, x, w, b, mean, rstd, res, dropout_mask=bits, dropout_scale=1.0)
                    mscale += " bits=1"
            built += 1
        result = f"ln_bwd={built}{mscale}"
    except Exception as error:  # the run continues on the Triton path
        _tln._wln_bwd_flydsl_disable("startup prebuild", error)
    # E-M2's FlyDSL grouped LayerNorm backward (ops/preproc_grouped_tail.py,
    # ops/flydsl/ln_bwd_grouped.py), only when that path can run: its own
    # latch on error, PR #7's per-call path is not affected (" grp3=1|error").
    if _tln._WLN_BWD_FLYDSL_BROKEN is None:
        try:
            from generative_recommenders.ops import preproc_grouped_tail as _pgt

            if _pgt.flydsl_grouped_bwd_wanted():
                result += " " + _pgt.prebuild_flydsl_grouped(device)
        except Exception as error:  # never abort a run over this prebuild
            logger.warning("[startup-preload] grouped LN bwd prebuild skipped: %s", error)
    # E-M3a's stacked LayerNorm(512) backward (ops/preproc_fp8_l2.py STACKED
    # BACKWARD, ops/flydsl/ln_bwd_grouped_fp8.py), only with E-M3a's gate on
    # (unset or "0": this block imports nothing and appends nothing; the E-M2
    # block above already imported preproc_grouped_tail and, through it,
    # preproc_fp8_l2 -- no FlyDSL, no build), here after the
    # DataLoader forks: " grp3f8=1|error|off". prebuild_grouped checks that the
    # path can run (kill switch on, PR #7 not latched off, ...) and latches only
    # the stacked path off on an error (the P3 path runs).
    if os.environ.get("DLRM_HSTU_FP8_PREPROC_L2", "0") == "1":
        try:
            from generative_recommenders.ops import preproc_fp8_l2 as _l2

            result += " " + _l2.prebuild_grouped(device)
        except Exception as error:  # never abort a run over this prebuild
            logger.warning("[startup-preload] stacked LN bwd prebuild skipped: %s", error)
            result += " grp3f8=error"
    # (93) X2L-B phase 2: the self-check against the shipped chain, then the
    # device is armed (" x2lb=1|off|error"); knob unset or "0": "" (nothing
    # imported). Inside this function's synchronize bracket below.
    result += _prebuild_preproc_l2_x2lb_check(device)
    # r14 E-RM (DLRM_HSTU_FP8_BWD_ERM, yambda_5b.gin item (97), default off;
    # ops/triton/dy_erm.py): the E-RM variant of the residual main kernel
    # (ln_bwd_flydsl_d512_res_erm_x32_c3), compiled and launched once on the
    # same synthetic zeros (inv_scale 1, t_shift 5; M, ITERS, G and 2^-t are
    # runtime arguments, so this one specialization serves every call), only
    # with the knob on and the shipped path not latched; appended LAST, so
    # every earlier token of the line keeps its neighbours: " erm=1|error|inert".
    # Its own latch on error (dy_erm.set_broken); never _wln_bwd_flydsl_disable.
    # Knob off: nothing is imported, built or appended.
    if (
        os.environ.get("DLRM_HSTU_FP8_BWD_ERM", "0") == "1"
        and _tln._WLN_BWD_FLYDSL_BROKEN is None
    ):
        try:
            from generative_recommenders.ops.triton import dy_erm as _erm

            erm = _erm.prebuild(device)
            if erm:
                result += " " + erm
        except Exception as error:  # never abort a run over this prebuild
            logger.warning("[startup-preload] erm prebuild skipped: %s", error)
            result += " erm=error"
    # Not swallowed, as in warmup(): an asynchronous device fault here is
    # sticky, the HIP context is unusable and step 1 would fail anyway --
    # later, inside the timed region, with a misleading stack. Fail now,
    # before RUN_START.
    try:
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError(
            "[startup-preload] device fault during the FlyDSL LayerNorm-backward "
            "prebuild; the HIP context is unusable. Rerun with "
            "DLRM_HSTU_FP8_WLN_BWD_FLYDSL=0 to isolate."
            # r14 E-RM: its block (" erm=" in the result) launched last
            + (
                " The E-RM variant ran in it: DLRM_HSTU_FP8_BWD_ERM=0 is the "
                "narrower switch."
                if " erm=" in result
                else ""
            )
        ) from error
    return result


def _gc_collect_and_freeze() -> int:
    """One full Python GC + gc.freeze() before RUN_START, so the in-loop
    utils._collect_and_freeze_gc (kept as is) only walks objects created
    after it. Honors its kill switch $DLRM_HSTU_FP8_GC_FREEZE."""
    import gc

    knob = os.environ.get("DLRM_HSTU_FP8_GC_FREEZE", "1").strip().lower()
    if knob in ("0", "false", "off", "no"):
        return -1
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()


@gin.configurable
def run_startup_preload(prestarted: bool, enabled: int = 0,
                        flydsl: int = 1, gc_freeze: int = 1,
                        ln_bwd: int = 1) -> None:
    """train_ranker hook, called AFTER the DataLoader pre-start (gin
    run_startup_preload.enabled = $BLAS_WARMUP). Runs only when every worker
    pool was already forked by the pre-start (``prestarted``), so no pool forks
    with the FlyDSL library loaded or with a different GC state than the legacy
    order would give it. Library import, the JIT build of the WLN_BWD_FLYDSL
    FlyDSL LayerNorm-backward kernels (``ln_bwd``: compiled and launched once
    on synthetic zero/one tensors, outputs discarded; with them E-M2's grouped
    and, under DLRM_HSTU_FP8_PREPROC_L2=1, E-M3a's stacked LayerNorm(512)
    backward kernels when those paths can run, see _prebuild_flydsl_ln_bwd)
    and Python memory management only: no data, no RNG, no training state
    (MLPerf training_rules L545-549: library initialization and JIT on
    synthetic inputs). Not prestarted: none of these FlyDSL kernels is built
    here; each compiles at its first use."""
    if not int(enabled):
        return
    if not prestarted:
        logger.info("[startup-preload] skipped: needs DL_PRESTART=1 (pools must "
                    "be forked before the FlyDSL import)")
        _x2lb_phase2_not_run("the DataLoader was not pre-started (DL_PRESTART, gin item (18))")
        return
    t0 = time.perf_counter()
    fly = _preload_flydsl() if int(flydsl) else "off"
    ln_time = ""
    if int(flydsl) and int(ln_bwd) and fly.startswith("bwd="):
        # after the pools fork (prestarted) and before gc.freeze, so the
        # trace-time garbage is collected by the freeze below. A FlyDSL error
        # latches WLN_BWD_FLYDSL off inside; only a device fault raises.
        t_ln = time.perf_counter()
        fly += " " + _prebuild_flydsl_ln_bwd()
        ln_time = f" (ln_bwd prebuild {time.perf_counter() - t_ln:.2f}s)"
    else:
        _x2lb_phase2_not_run(f"the FlyDSL preload / ln_bwd prebuild did not run (flydsl={flydsl} ln_bwd={ln_bwd} "
                             f"result {fly})")
    t1 = time.perf_counter()
    try:
        frozen = _gc_collect_and_freeze() if int(gc_freeze) else -2
    except Exception as error:
        logger.warning("[startup-preload] gc failed (ignored): %s", error)
        frozen = -3
    t2 = time.perf_counter()
    logger.info(
        "armed OK: [startup-preload] flydsl %s in %.2fs%s, gc frozen=%d in %.2fs "
        "before run_start",
        fly, t1 - t0, ln_time, frozen, t2 - t1,
    )
