# pyre-unsafe
"""Bitwise single-launch replacement for torch.optim.Adam's foreach step (R-C9).

``FusedDenseAdam`` is a ``torch.optim.Adam`` subclass. For a param group that
torch 2.12 would run on its default foreach path (``_multi_tensor_adam``: no
amsgrad / maximize / capturable / differentiable / fused, weight_decay 0, float
hyper-parameters, CUDA fp32 contiguous tensors on one device) it replaces the
foreach op chain -- eleven ``multi_tensor_apply_kernel`` launches for the
DLRMv4 dense parameter set -- with ONE Triton launch per distinct step count
(normally one per step). Every element goes through torch's own fp32 rounding
sequence, as torch's ROCm kernels actually execute it (each contraction below
was decided empirically on gfx950 against the torch foreach kernels, not taken
from the source):

    g = g * c                       _foreach_mul_(grads, clip coef tensor)  [clip fold only]
    m = fma(w, g - m, m)            _foreach_lerp_(exp_avgs, grads, w), |w| <  0.5
      | fma(-(g - m), 1 - w, g)                                          |w| >= 0.5
    v = v * b2                      _foreach_mul_(exp_avg_sqs, beta2)
    v = fma(1 - b2, g * g, v)       _foreach_addcmul_ (g * g rounded first)
    d = sqrt(v)                     _foreach_sqrt (correctly rounded incl. denormals)
    d = d / bc2_sqrt                _foreach_div_ ScalarList (IEEE division)
    d = d + eps                     _foreach_add_
    p = fma(step_size, m / d, p)    _foreach_addcdiv_ ScalarList (IEEE division)

with every scalar converted to fp32 the way c10::Scalar::to<float>() does
(round to nearest even) from the Python doubles torch computes on the host:
w = 1 - beta1, b2 = beta2, 1 - b2, eps, step_size = (lr / (1 - beta1 ** step))
* -1 and bc2_sqrt = (1 - beta2 ** step) ** 0.5, with step the CPU fp32
``state['step']`` after torch's own ``_foreach_add_``. Optimizer state
(``state[p]['step'|'exp_avg'|'exp_avg_sq']``, param_groups, state_dict) is
created by ``Adam._init_group`` and is therefore identical to torch's.

Triton notes (gfx950, Triton 3.6): kernels are compiled with
``enable_fp_fusion=False`` and the contractions above are explicit ``tl.fma``;
``tl.math.sqrt_rn`` is NOT correctly rounded for denormal / small inputs, the
HIP libdevice (ocml) ``sqrt`` is, and is what this module uses.

Clip fold: ``clip_grad_norm_fold_`` is a twin of
``torch.nn.utils.clip_grad_norm_`` that computes the same total norm and
clamped coefficient with the same torch ops, multiplies every grad NOT owned by
an armed ``FusedDenseAdam`` exactly as torch does, and hands the coefficient
tensor to the fused optimizer(s); their next ``step()`` multiplies the grads in
the kernel (fp32 product rounded first, written back to ``.grad``, exactly what
``_foreach_mul_`` leaves behind) before using them. A group that falls back to
torch applies the pending coefficient with ``torch._foreach_mul_`` first.

Anything ineligible falls back, per group, to the torch functional ``adam()``
call ``Adam.step`` makes (same arguments, so the same numerics).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import _use_grad_for_differentiable

try:  # HIP libdevice (ocml): correctly rounded fp32 sqrt incl. denormals
    from triton.language.extra.hip import libdevice as _libdevice
except Exception:  # pragma: no cover - non-HIP Triton
    _libdevice = None

logger: logging.Logger = logging.getLogger(__name__)

ENV_KNOB = "DLRM_HSTU_FP8_DENSE_ADAM_FUSED"
_BLOCK = int(os.environ.get("DLRM_HSTU_FP8_DENSE_ADAM_FUSED_BLOCK", "1024"))
_NUM_WARPS = int(os.environ.get("DLRM_HSTU_FP8_DENSE_ADAM_FUSED_WARPS", "4"))


def env_enabled() -> bool:
    """DLRM_HSTU_FP8_DENSE_ADAM_FUSED: only "0" (or absent) disarms."""
    return os.environ.get(ENV_KNOB, "0") != "0"


def _f32(x: float) -> float:
    # c10::Scalar(double).to<float>() == checked_convert + static_cast<float>: round to nearest even, and a
    # finite double outside the float range raises (torch's foreach ops raise this exact error) -- REVIEW FIX
    with np.errstate(over="ignore"):
        f = np.float32(x)
    if np.isinf(f) and np.isfinite(x):
        raise RuntimeError("value cannot be converted to type float without overflow")
    return float(f)


@triton.jit
def _sqrt_cr(x):
    # ocml __ocml_sqrt_f32: correctly rounded incl. denormal inputs (tl.math.sqrt_rn is not, on gfx950)
    return tl.extra.hip.libdevice.sqrt(x)


@triton.jit
def _adam_elem(p, g, m, v, lerp_w, beta2, omb2, eps, step_size, bc2_sqrt, LERP_SMALL: tl.constexpr):
    # REVIEW FIX: broadcast scalars with `* 1.0` (exact for every value, keeps -0.0) instead of `+ 0.0`
    # (which turns a -0.0 step_size, i.e. lr == 0, into +0.0); negate with a sign-bit flip (torch's fneg)
    # instead of Triton's unary minus, which is `0 - x` (+0 for x = +0, NaN sign kept).
    one = tl.zeros_like(p) + 1.0
    d = g - m
    if LERP_SMALL:
        m = tl.fma(lerp_w * one, d, m)
    else:
        # torch's HIP lerp compiles `end - diff * (1 - w)` to fma(diff, -(1 - w), end): negate the weight
        # factor (exact sign-bit flip), not diff -- this also reproduces torch's NaN signs
        omw = (1.0 - lerp_w) * one
        nomw = (omw.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)
        m = tl.fma(d, nomw, g)
    v = v * beta2
    v = tl.fma(omb2 * one, g * g, v)
    den = _sqrt_cr(v)
    den = tl.math.div_rn(den, bc2_sqrt * one)
    den = den + eps
    p = tl.fma(step_size * one, tl.math.div_rn(m, den), p)
    return p, m, v


@triton.jit
def _dense_adam_fused_kernel(
    ptr_tab,  # int64 [4 * T]: param | grad | exp_avg | exp_avg_sq addresses
    chunk_tid,  # int32 [C]: tensor index of each chunk
    chunk_off,  # int64 [C]: element offset of each chunk in its tensor (multiple of BLOCK)
    numel_tab,  # int64 [T]
    coef_ptr,  # fp32 [1]: clip coefficient (read only when HAS_COEF)
    T,
    lerp_w,
    beta2,
    omb2,
    eps,
    step_size,
    bc2_sqrt,
    HAS_COEF: tl.constexpr,
    LERP_SMALL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.load(chunk_tid + pid)
    off = tl.load(chunk_off + pid)
    n = tl.load(numel_tab + t)
    pa = tl.load(ptr_tab + t)
    ga = tl.load(ptr_tab + T + t)
    ma = tl.load(ptr_tab + 2 * T + t)
    va = tl.load(ptr_tab + 3 * T + t)
    # Addresses come from the table, so Triton cannot prove 16-byte alignment and
    # issues dword loads/stores. On gfx950 the kernel still streams near the HBM
    # rate, and grads that are DDP bucket views (only 4-byte aligned) cost the
    # same as whole allocations.
    p_ptr = pa.to(tl.pointer_type(tl.float32))
    g_ptr = ga.to(tl.pointer_type(tl.float32))
    m_ptr = ma.to(tl.pointer_type(tl.float32))
    v_ptr = va.to(tl.pointer_type(tl.float32))
    idx = off + tl.arange(0, BLOCK)
    if off + BLOCK <= n:
        p = tl.load(p_ptr + idx)
        g = tl.load(g_ptr + idx)
        m = tl.load(m_ptr + idx)
        v = tl.load(v_ptr + idx)
        if HAS_COEF:
            g = g * tl.load(coef_ptr)
            tl.store(g_ptr + idx, g)
        p, m, v = _adam_elem(p, g, m, v, lerp_w, beta2, omb2, eps, step_size, bc2_sqrt, LERP_SMALL)
        tl.store(p_ptr + idx, p)
        tl.store(m_ptr + idx, m)
        tl.store(v_ptr + idx, v)
    else:
        mask = idx < n
        p = tl.load(p_ptr + idx, mask=mask, other=0.0)
        g = tl.load(g_ptr + idx, mask=mask, other=0.0)
        m = tl.load(m_ptr + idx, mask=mask, other=0.0)
        v = tl.load(v_ptr + idx, mask=mask, other=0.0)
        if HAS_COEF:
            g = g * tl.load(coef_ptr)
            tl.store(g_ptr + idx, g, mask=mask)
        p, m, v = _adam_elem(p, g, m, v, lerp_w, beta2, omb2, eps, step_size, bc2_sqrt, LERP_SMALL)
        tl.store(p_ptr + idx, p, mask=mask)
        tl.store(m_ptr + idx, m, mask=mask)
        tl.store(v_ptr + idx, v, mask=mask)


class _Plan:
    """Chunk table for one list of tensor sizes plus a cached address table."""

    def __init__(self, numels: Tuple[int, ...], device: torch.device, block: int, num_warps: int) -> None:
        self.block = block
        self.num_warps = num_warps
        tids: List[int] = []
        offs: List[int] = []
        for t, n in enumerate(numels):
            for o in range(0, n, block):
                tids.append(t)
                offs.append(o)
        self.T = len(numels)
        self.C = len(tids)
        self.numel = sum(numels)
        self.chunk_tid = torch.tensor(tids, dtype=torch.int32).to(device)
        self.chunk_off = torch.tensor(offs, dtype=torch.int64).to(device)
        self.numel_tab = torch.tensor(list(numels), dtype=torch.int64).to(device)
        # fp32 stand-in for coef_ptr when there is no clip coefficient: keeps the
        # kernel signature identical across the two HAS_COEF variants
        self.coef_dummy = torch.ones(1, dtype=torch.float32, device=device)
        self.addr_key: Optional[Tuple[int, ...]] = None
        self.ptr_tab: Optional[torch.Tensor] = None
        self.warmed = False
        self.ptr_stream: Optional[torch.cuda.Stream] = None

    def ptrs(self, addrs: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        if addrs != self.addr_key:
            host = torch.tensor(addrs, dtype=torch.int64).pin_memory()
            self.ptr_tab = host.to(device, non_blocking=True)
            self.addr_key = addrs
            self.ptr_stream = torch.cuda.current_stream(device)
        else:
            cur = torch.cuda.current_stream(device)
            if cur != self.ptr_stream:
                self.ptr_tab.record_stream(cur)  # pyre-ignore
        return self.ptr_tab  # pyre-ignore


def fused_adam_launch(
    plan: _Plan,
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    eps: float,
    step_size: float,
    bc2_sqrt: float,
    coef: Optional[torch.Tensor] = None,
) -> None:
    """One launch over the plan's tensors; host scalars are Python doubles."""
    dev = params[0].device
    addrs = tuple(
        [x.data_ptr() for x in params]
        + [x.data_ptr() for x in grads]
        + [x.data_ptr() for x in exp_avgs]
        + [x.data_ptr() for x in exp_avg_sqs]
    )
    ptr_tab = plan.ptrs(addrs, dev)
    w = 1 - beta1  # torch: cast(float, 1 - device_beta1)
    coef_t = coef if coef is not None else plan.coef_dummy  # not read when HAS_COEF is False
    args = (
        ptr_tab,
        plan.chunk_tid,
        plan.chunk_off,
        plan.numel_tab,
        coef_t,
        plan.T,
        _f32(w),
        _f32(beta2),
        _f32(1 - beta2),
        _f32(eps),
        _f32(step_size),
        _f32(bc2_sqrt),
    )
    lerp_small = abs(_f32(w)) < 0.5  # at::native::is_lerp_weight_small on the fp32 weight
    _dense_adam_fused_kernel[(plan.C,)](
        *args,
        HAS_COEF=coef is not None,
        LERP_SMALL=lerp_small,
        BLOCK=plan.block,
        num_warps=plan.num_warps,
        enable_fp_fusion=False,
    )
    if not plan.warmed:
        # Compile (never launch) the other clip-fold variant now: the first step
        # is torchrec init_state's dummy step (no clip, before RUN_START), so the
        # training steps' clip-fold variant would otherwise JIT inside the run.
        # Also load it on this GPU (CompiledKernel._init_handles, as the TSPEC
        # replay primes its rows): warmup() only compiles, and the first launch
        # would load its code object inside training step 1.
        plan.warmed = True
        try:
            k = _dense_adam_fused_kernel.warmup(
                *args,
                grid=(1,),
                HAS_COEF=coef is None,
                LERP_SMALL=lerp_small,
                BLOCK=plan.block,
                num_warps=plan.num_warps,
                enable_fp_fusion=False,
            )
            getattr(k, "_init_handles", lambda: None)()
        except Exception as e:  # warmup only; the variant then compiles at first use
            logger.warning("[dense-adam-fused] precompile of the other variant skipped: %s", e)


def _foreach_supported(params: List[torch.Tensor]) -> bool:
    from torch.optim.optimizer import (
        _foreach_supported_types,
        _get_foreach_kernels_supported_devices,
    )

    devs = _get_foreach_kernels_supported_devices()
    return all(type(p) in _foreach_supported_types and p.device.type in devs for p in params)


class FusedDenseAdam(torch.optim.Adam):
    """torch.optim.Adam whose eligible groups step in one bitwise Triton launch."""

    def __init__(self, params: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(params, *args, **kwargs)
        self._rc9_plans: Dict[Tuple[int, ...], _Plan] = {}
        # (clamped clip coefficient, ids of the params whose grads it still has to scale)
        self._rc9_pending: Optional[Tuple[torch.Tensor, frozenset]] = None
        self._rc9_armed_logged = False
        self._rc9_fold_logged = False
        self._rc9_fallback_logged: set = set()
        self._rc9_disabled = _libdevice is None

    # -- clip fold ---------------------------------------------------------
    def rc9_owned_params(self) -> List[torch.Tensor]:
        """Params whose clip multiply this optimizer takes over (eligible groups)."""
        out: List[torch.Tensor] = []
        if self._rc9_disabled:
            return out
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            if ps and self._rc9_group_static_reason(group) is None:
                out.extend(ps)
        return out

    def rc9_set_pending_grad_scale(self, coef: torch.Tensor, owned: Optional[frozenset] = None) -> None:
        """Defer `grad *= coef` for the params in `owned` (ids; default: every
        param this optimizer currently owns) into the next step()."""
        self.rc9_flush_pending()
        if owned is None:
            owned = frozenset(id(p) for p in self.rc9_owned_params())
        self._rc9_pending = (coef, owned)

    def rc9_flush_pending(self) -> None:
        """Apply a deferred clip multiply now, with torch (what clip_grad_norm_
        would have done), e.g. before the grads are read for another purpose."""
        pending = self._rc9_pending
        self._rc9_pending = None
        if pending is None:
            return
        coef, owned = pending
        _torch_clip_mul_(
            [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None and id(p) in owned], coef
        )

    # -- eligibility -------------------------------------------------------
    @staticmethod
    def _rc9_group_static_reason(group: Dict[str, Any]) -> Optional[str]:
        if group["amsgrad"]:
            return "amsgrad"
        if group["maximize"]:
            return "maximize"
        if group["capturable"]:
            return "capturable"
        if group["differentiable"]:
            return "differentiable"
        if group["fused"]:
            return "fused=True"
        if group["foreach"] is False:
            return "foreach=False"
        if group["weight_decay"] != 0:
            return "weight_decay!=0"
        lr = group["lr"]
        b1, b2 = group["betas"]
        if any(isinstance(x, torch.Tensor) for x in (lr, b1, b2, group["eps"])):
            return "tensor hyper-parameter"
        return None

    def _rc9_reason(
        self,
        group: Dict[str, Any],
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
        has_complex: bool,
    ) -> Optional[str]:
        if self._rc9_disabled:
            return "no HIP libdevice sqrt in this Triton"
        r = self._rc9_group_static_reason(group)
        if r is not None:
            return r
        if has_complex:
            return "complex params"
        if torch.compiler.is_compiling() or torch.jit.is_scripting():
            return "compiling"
        if not _foreach_supported(params):
            return "torch would not take the foreach path"
        dev = params[0].device
        if dev.type != "cuda":
            return "params not on CUDA"
        for lst in (params, grads, exp_avgs, exp_avg_sqs):
            for x in lst:
                if x.device != dev or x.dtype != torch.float32 or x.is_sparse or not x.is_contiguous():
                    return "tensor not fp32 contiguous dense on the params' device"
        for p, g, m, v in zip(params, grads, exp_avgs, exp_avg_sqs):
            if g.shape != p.shape or m.shape != p.shape or v.shape != p.shape:
                return "shape mismatch"
        for s in state_steps:
            if not s.is_cpu or s.dtype != torch.float32 or s.numel() != 1:
                return "state step not a CPU fp32 scalar tensor"
        return None

    def _rc9_torch_group(self, group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, has_complex):
        from torch.optim.adam import adam as _torch_adam

        beta1, beta2 = group["betas"]
        _torch_adam(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=group["amsgrad"],
            has_complex=has_complex,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            decoupled_weight_decay=group["decoupled_weight_decay"],
        )

    def _rc9_plan(self, numels: Tuple[int, ...], device: torch.device) -> _Plan:
        plan = self._rc9_plans.get(numels)
        if plan is None:
            plan = _Plan(numels, device, _BLOCK, _NUM_WARPS)
            self._rc9_plans[numels] = plan
        return plan

    def _rc9_fused_group(self, group, params, grads, exp_avgs, exp_avg_sqs, state_steps, coef) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        # step += 1 exactly as _multi_tensor_adam does for CPU steps
        torch._foreach_add_(state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0)
        vals = [s.item() for s in state_steps]
        buckets: Dict[float, List[int]] = {}
        for i, sv in enumerate(vals):
            buckets.setdefault(sv, []).append(i)
        for sv, idx in buckets.items():
            # host scalars exactly as _multi_tensor_adam (non-capturable) computes them
            bias_correction1 = 1 - beta1**sv
            bias_correction2 = 1 - beta2**sv
            step_size = (lr / bias_correction1) * -1
            bias_correction2_sqrt = bias_correction2**0.5
            if len(idx) == len(params):
                sel = (params, grads, exp_avgs, exp_avg_sqs)
            else:
                sel = tuple([lst[i] for i in idx] for lst in (params, grads, exp_avgs, exp_avg_sqs))
            plan = self._rc9_plan(tuple(x.numel() for x in sel[0]), sel[0][0].device)
            fused_adam_launch(
                plan,
                *sel,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                step_size=step_size,
                bc2_sqrt=bias_correction2_sqrt,
                coef=coef,
            )
            # The kernel writes through raw pointers, so autograd's version counters do not move; bump them exactly
            # as torch's foreach Adam chain does. Caches keyed on weight._version (the FP8 weight shadows of item
            # (24)(e) / triton_fp8._get_weight_shadow, the packed-weight cache, glue_fold3 bf16 shadows) depend on it:
            # without the bump they keep step-1 weights for the whole run.
            for _t in (*sel[0], *sel[2], *sel[3]):
                torch.autograd.graph.increment_version(_t)
            if coef is not None:
                for _t in sel[1]:
                    torch.autograd.graph.increment_version(_t)
            if not self._rc9_armed_logged or (coef is not None and not self._rc9_fold_logged):
                first = not self._rc9_armed_logged
                self._rc9_armed_logged = True
                self._rc9_fold_logged |= coef is not None
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                msg = (
                    "armed OK: dense Adam fused (%s): rank=%d tensors=%d elements=%d chunks=%d "
                    "BLOCK=%d warps=%d clip_fold=%s step_buckets=%d -- one Triton launch "
                    "replaces torch's foreach Adam chain%s, bitwise"
                    % (ENV_KNOB, rank, plan.T, plan.numel, plan.C, plan.block, plan.num_warps,
                       coef is not None, len(buckets),
                       " and the clip_grad_norm_ multiply" if coef is not None else "")
                    + ("" if first else " (first step with the clip fold)")
                )
                logger.info(msg)
                print(msg, flush=True)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault("_rc9_plans", {})
        self.__dict__.setdefault("_rc9_pending", None)
        self.__dict__.setdefault("_rc9_armed_logged", False)
        self.__dict__.setdefault("_rc9_fold_logged", False)
        self.__dict__.setdefault("_rc9_fallback_logged", set())
        self.__dict__.setdefault("_rc9_disabled", _libdevice is None)

    def zero_grad(self, set_to_none: bool = True) -> None:
        # a coefficient that was never consumed belongs to grads being discarded
        self._rc9_pending = None
        super().zero_grad(set_to_none)

    @_use_grad_for_differentiable
    def step(self, closure=None):  # noqa: C901
        self._accelerator_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        pending = self._rc9_pending
        self._rc9_pending = None
        coef, owned = pending if pending is not None else (None, frozenset())
        for group in self.param_groups:
            params: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            exp_avgs: List[torch.Tensor] = []
            exp_avg_sqs: List[torch.Tensor] = []
            max_exp_avg_sqs: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []
            has_complex = self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)
            if not params:
                continue
            reason = self._rc9_reason(group, params, grads, exp_avgs, exp_avg_sqs, state_steps, has_complex)
            if reason is None:
                coef_d = None
                if coef is not None:
                    if all(id(p) in owned for p in params):
                        coef_d = coef.to(params[0].device)
                    else:  # not expected: some grads were already scaled by the caller
                        _torch_clip_mul_([g for p, g in zip(params, grads) if id(p) in owned], coef)
                self._rc9_fused_group(group, params, grads, exp_avgs, exp_avg_sqs, state_steps, coef_d)
                continue
            if reason not in self._rc9_fallback_logged:
                self._rc9_fallback_logged.add(reason)
                logger.warning("[dense-adam-fused] group falls back to torch Adam: %s", reason)
            if coef is not None:
                _torch_clip_mul_([g for p, g in zip(params, grads) if id(p) in owned], coef)
            self._rc9_torch_group(group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, has_complex)
        return loss


def _torch_clip_mul_(grads: List[torch.Tensor], clip_coef_clamped: torch.Tensor) -> None:
    """The multiply half of torch.nn.utils.clip_grads_with_norm_ (2.12), verbatim."""
    from torch.utils._foreach_utils import (
        _device_has_foreach_support,
        _group_tensors_by_device_and_dtype,
        _has_foreach_support,
    )

    if len(grads) == 0:
        return
    grouped = _group_tensors_by_device_and_dtype([grads])
    for (device, _), ([device_grads], _) in grouped.items():
        if _has_foreach_support(device_grads, device):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        else:
            c = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(c)


def find_fused_dense_adams(optimizer: Any) -> List[FusedDenseAdam]:
    """FusedDenseAdam instances inside a torchrec CombinedOptimizer / KeyedOptimizerWrapper tree."""
    out: List[FusedDenseAdam] = []
    seen = set()

    def visit(o: Any) -> None:
        if o is None or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, FusedDenseAdam):
            out.append(o)
            return
        for _, sub in getattr(o, "_optims", []) or []:  # torchrec CombinedOptimizer
            visit(sub)
        visit(getattr(o, "_optimizer", None))  # KeyedOptimizerWrapper / OptimizerWrapper

    visit(optimizer)
    return out


def clip_grad_norm_fold_(
    parameters: Any,
    max_norm: float,
    optimizer: Any,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    """torch.nn.utils.clip_grad_norm_ (2.12) with the grad multiply of the params an
    armed FusedDenseAdam owns deferred into that optimizer's next step() (same
    fp32 products, written back to .grad in the kernel). Same total norm, same
    clamped coefficient (same torch ops). Without a FusedDenseAdam in
    `optimizer`, or with an explicit `foreach`, it IS torch's clip_grad_norm_.
    Between this call and step() the owned .grad tensors still hold the
    unscaled values; nothing in the training loop reads them there."""
    fused = find_fused_dense_adams(optimizer)
    for opt in fused:
        opt.rc9_flush_pending()  # a still-deferred multiply belongs to these grads
    if not fused or foreach is not None:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    with torch.no_grad():
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, None)
        if len(grads) == 0:
            return total_norm
        max_norm = float(max_norm)
        # the same ops as torch.nn.utils.clip_grads_with_norm_ (2.12)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        in_params = {id(p) for p in parameters if p.grad is not None}
        owned_by = [
            (opt, frozenset(id(p) for p in opt.rc9_owned_params() if id(p) in in_params)) for opt in fused
        ]
        owned = set().union(*[o for _, o in owned_by])
        _torch_clip_mul_(
            [p.grad for p in parameters if p.grad is not None and id(p) not in owned],
            clip_coef_clamped,
        )
        for opt, o in owned_by:
            if o:
                opt.rc9_set_pending_grad_scale(clip_coef_clamped, o)
    return total_norm
